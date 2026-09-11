#!/usr/bin/env python3
import csv,hashlib,itertools,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.e2_dataset import E2Dataset
from src.e2_metrics import family_metrics,response_vectors,scale_consistency,xy_subspace
from src.feature_io import arguments,configuration,config_hash,digest,read_jsonl,records_hash,write_json
from src.metrics import EPS,summary

LOCAL={'TC_e','TC_o','scale_e','scale_o','S_e','S_o','cos_signed','cos_abs','S_r','S_c','M_r','M_c','magnitude_log_ratio','C_resp','C_norm','D_cause_legacy'}

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('empty\n');return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def feature_meta(key):
    layer,phase,rep=key.split('__');return {'layer':int(layer[1:]),'phase':phase,'representation':rep}

def select_array(array,rep,stored_ids,core_ids):
    if rep not in ('patch','dense'):return array,np.arange(len(array),dtype=np.int64),'entity' if rep!='register' else 'register'
    index={int(p):i for i,p in enumerate(stored_ids)}
    try:ix=[index[int(p)] for p in core_ids]
    except KeyError as exc:raise ValueError(f'Feature shard lacks panel core point {exc.args[0]}')
    return array[ix],np.asarray(core_ids,dtype=np.int64),'physical_point'

def unit_records(values,unit_ids):
    a=np.asarray(values)
    if a.ndim==0:a=a[None]
    return zip(unit_ids,a)

def aggregate_units(rows,group_keys,value_keys):
    buckets=defaultdict(list)
    for row in rows:buckets[tuple(row[k] for k in group_keys)].append(row)
    result=[]
    for key,values in buckets.items():
        out={**dict(zip(group_keys,key)),'unit_count':len(values)}
        for metric in value_keys:
            out.update({f'{metric}_{stat}':value for stat,value in summary([r[metric] for r in values]).items()})
        result.append(out)
    return result

def rgb_control(d,a,b):
    fa=d.frames_for(a)[-1];fb=d.frames_for(b)[-1]
    ia=np.asarray(Image.open(d.root/fa['rgb']).convert('RGB'),np.float32)/255
    ib=np.asarray(Image.open(d.root/fb['rgb']).convert('RGB'),np.float32)/255
    ma=np.asarray(Image.open(d.root/fa['target_mask']).convert('L'))>0
    mb=np.asarray(Image.open(d.root/fb['target_mask']).convert('L'))>0;inter=ma&mb
    diff=np.abs(ia-ib)
    return float(diff.mean()),float(diff[inter].mean()) if inter.any() else float('nan'),int(inter.sum())

def dataset_diagnostics(d,seqs,core_name):
    projection=[];rgb=[];equivalent=[];core=d.cores[core_name];groups=defaultdict(list)
    for s in seqs:groups[s['group_id']].append(s)
    for group,rr in groups.items():
        pc=rr[0]['physical_context_id'];ids=core[pc]['point_ids'];static=next(s for s in rr if (s['ego_level'],s['object_level'])==(0,0));uv0=d.point_uv(static,ids)
        for s in rr:
            if ((s['ego_level']==0)^(s['object_level']==0)) and max(abs(s['ego_level']),abs(s['object_level'])) in (1,2):
                disp=np.linalg.norm(d.point_uv(s,ids)-uv0,axis=1)
                projection.append({'physical_context_id':pc,'group_id':group,'motion_family_id':s['motion_family_id'],
                    'axis':'x' if s['motion_axis_world'][0] else 'y','delta_m':s['delta_m'],'cause':'ego' if s['object_level']==0 else 'object',
                    'level':s['ego_level'] or s['object_level'],'median_px':float(np.median(disp)),'q25_px':float(np.quantile(disp,.25)),
                    'q75_px':float(np.quantile(disp,.75)),'min_px':float(disp.min()),'max_px':float(disp.max()),'median_div_patch16':float(np.median(disp)/16)})
        for a,b in itertools.combinations(sorted(rr,key=lambda s:(s['ego_level'],s['object_level'])),2):
            if a['relative_level'] and a['relative_level']==b['relative_level']:
                full,target,n=rgb_control(d,a,b);rgb.append({'physical_context_id':pc,'motion_family_id':a['motion_family_id'],
                    'r':a['relative_level'],'c_a':(a['ego_level']+a['object_level'])/2,'c_b':(b['ego_level']+b['object_level'])/2,
                    'delta_c':abs(a['ego_level']+a['object_level']-b['ego_level']-b['object_level'])/2,
                    'e_a':a['ego_level'],'o_a':a['object_level'],'e_b':b['ego_level'],'o_b':b['object_level'],
                    'full_rgb_mean_abs_01':full,'target_intersection_rgb_mean_abs_01':target,'intersection_pixels':n})
    # Same physical endpoint amplitudes across scale families, audited independently for X and Y.
    by=defaultdict(list)
    for s in seqs:by[s['physical_context_id'],tuple(s['motion_axis_world']),round(s['ego_amplitude_m'],8),round(s['object_amplitude_m'],8)].append(s)
    for (pc,axis,ea,oa),rr in by.items():
        for a,b in itertools.combinations(rr,2):
            if a['motion_family_id']==b['motion_family_id']:continue
            ta,tb=d.track(a),d.track(b);fa=d.frames_for(a);fb=d.frames_for(b)
            matrix_error=max(float(np.max(np.abs(np.asarray(x['camera_to_world'])-np.asarray(y['camera_to_world'])))) for x,y in zip(fa,fb))
            object_error=max(float(np.max(np.abs(np.asarray(x['object_to_world'])-np.asarray(y['object_to_world'])))) for x,y in zip(fa,fb))
            amplitude_error=max(max(abs(x[k]-y[k]) for k in ('actual_ego_amplitude_m','actual_object_amplitude_m','actual_relative_amplitude_m')) for x,y in zip(fa,fb))
            intrinsics_error=max(float(np.max(np.abs(np.asarray(x['K'])-np.asarray(y['K'])))) for x,y in zip(fa,fb))
            uv_error=float(np.max(np.abs(ta['projected_uv']-tb['projected_uv'])))
            rgb_equal=all(digest(d.root/x['rgb'])==digest(d.root/y['rgb']) for x,y in zip(fa,fb))
            equivalent.append({'physical_context_id':pc,'axis':list(axis),'ego_amplitude_m':ea,'object_amplitude_m':oa,
                'family_a':a['motion_family_id'],'family_b':b['motion_family_id'],'camera_transform_max_abs':matrix_error,
                'object_transform_max_abs':object_error,'physical_amplitude_max_abs_m':amplitude_error,
                'intrinsics_max_abs':intrinsics_error,'projected_uv_max_abs_px':uv_error,'rgb_bit_identical':rgb_equal})
    return projection,rgb,equivalent

def main():
    p=arguments('Compute E2 panel metrics from validated shared feature cache');p.add_argument('--panel',required=True)
    a=p.parse_args();c=configuration(a);out=Path(c['output_root']);gate=out/'audit'/f'extraction_{a.panel}_full_validation.json'
    if float(c['epsilon'])!=EPS:raise ValueError(f'E2 epsilon must remain E1-compatible at {EPS}')
    checked=json.loads(gate.read_text()) if gate.exists() else {}
    if not checked.get('passed') or checked.get('config_hash')!=config_hash(c):raise ValueError(f'Compatible full extraction validation missing for {a.panel}')
    d=E2Dataset(c);da=json.loads((out/'audit/dataset_audit.json').read_text());d.cores=da['cores'];d.sequence_point_ids=da['sequence_point_ids']
    if a.panel not in c['panels']:raise ValueError('Analysis requires one explicit panel; stage aliases are extraction/planning only')
    panels=[a.panel];seqs=d.panel_sequences(a.panel);allowed={s['sequence_id'] for s in seqs}
    manifest=out/'features/feature_manifest.jsonl';rows=[r for r in read_jsonl(manifest) if r['sequence_id'] in allowed]
    if records_hash(rows)!=checked.get('feature_selection_sha256'):raise ValueError('Panel feature selection changed since extraction validation')
    by=defaultdict(list)
    for r in rows:by[r['group_id'],r['model'],r['regime']].append(r)
    family_rows=[];local_rows=[];common_rows=[];vectors={}
    for (group,model,regime),rr in by.items():
        if len(rr)!=25:raise ValueError(f'{group}/{model}/{regime}: expected 25 conditions, got {len(rr)}')
        s0=next(s for s in seqs if s['group_id']==group);pc=s0['physical_context_id'];family=s0['motion_family_id']
        applicable=[c['panels'][p]['core'] for p in panels if family in c['panels'][p]['families'] and pc in d.cores[c['panels'][p]['core']]]
        core_name=min(applicable,key=lambda name:len(d.cores[name][pc]['point_ids']));core_ids=d.cores[core_name][pc]['point_ids']
        conditions={};stored_ids=None
        for r in rr:
            if digest(out/r['path'])!=r['sha256']:raise ValueError(f'Feature changed: {r["path"]}')
            with np.load(out/r['path']) as z:
                ids=z['point_ids'];stored_ids=ids if stored_ids is None else stored_ids
                if not np.array_equal(ids,stored_ids):raise ValueError(f'{group}: inconsistent shard point IDs')
                conditions[r['ego_level'],r['object_level']]={k:z[k].astype(np.float32) for k in z.files if k not in ('point_ids','metadata_json')}
        for feature in conditions[0,0]:
            fm=feature_meta(feature);arrays={k:select_array(v[feature],fm['representation'],stored_ids,core_ids)[0] for k,v in conditions.items()}
            unit_ids=select_array(conditions[0,0][feature],fm['representation'],stored_ids,core_ids)[1];unit_type=select_array(conditions[0,0][feature],fm['representation'],stored_ids,core_ids)[2]
            metrics,pairs=family_metrics(arrays,s0['delta_m']);meta={'physical_context_id':pc,'context_id':s0['context_id'],'target_id':s0['target_id'],
                'background_id':s0['background_id'],'group_id':group,'motion_family_id':family,'axis':'x' if s0['motion_axis_world'][0] else 'y',
                'delta_m':s0['delta_m'],'model':model,'regime':regime,**fm,'core_set':core_name}
            for metric,values in metrics.items():
                row={**meta,'metric':metric,**summary(values)};family_rows.append(row)
                if metric in LOCAL:local_rows.append(row)
            for pair in pairs:
                for uid,dv,cc,na,nb in zip(unit_ids,pair['D_c_given_r'],pair['cause_cos'],pair['response_norm_a'],pair['response_norm_b']):
                    common_rows.append({**meta,'unit_id':int(uid),'unit_type':unit_type,'r':pair['r'],'c_a':pair['c_a'],'c_b':pair['c_b'],
                        'delta_c':pair['delta_c'],'e_a':pair['e_a'],'o_a':pair['o_a'],'e_b':pair['e_b'],'o_b':pair['o_b'],
                        'D_c_given_r':float(dv),'cause_cos':float(cc),'response_norm_a':float(na),'response_norm_b':float(nb)})
            ve,vo=response_vectors(arrays,s0['delta_m']);vectors[pc,family,model,regime,feature]=(unit_ids,unit_type,ve,vo,meta)
    xy=[]
    if any(p in panels for p in ('xy_d004','representation_specialization')):
        for key,x in list(vectors.items()):
            pc,family,model,regime,feature=key
            if family!='tx_d004' or (pc,'ty_d004',model,regime,feature) not in vectors:continue
            y=vectors[pc,'ty_d004',model,regime,feature]
            if not np.array_equal(x[0],y[0]):raise ValueError(f'{pc}/{feature}: X/Y units differ')
            for i,uid in enumerate(x[0]):
                value=xy_subspace(x[2][i],y[2][i],x[3][i],y[3][i],c['rank_relative_tolerance'],c['well_conditioned_rho'])
                xy.append({**x[4],'unit_id':int(uid),'unit_type':x[1],**value,'principal_angles_deg':json.dumps(value['principal_angles_deg'])})
    scale=[]
    if 'scale_locality' in panels:
        buckets=defaultdict(dict)
        for (pc,family,model,regime,feature),v in vectors.items():
            axis='x' if family[1]=='x' else 'y';buckets[pc,axis,model,regime,feature][v[4]['delta_m']]=v
        for (pc,axis,model,regime,feature),ds in buckets.items():
            if set(ds)!={.02,.04,.06}:raise ValueError(f'{pc}/{axis}/{feature}: incomplete scales {sorted(ds)}')
            reference=ds[.04]
            if any(not np.array_equal(reference[0],value[0]) for value in ds.values()):
                raise ValueError(f'{pc}/{axis}/{feature}: cross-scale physical units differ')
            for cause,index in [('ego',2),('object',3)]:
                values={delta:v[index] for delta,v in ds.items()}
                for pair in scale_consistency(values):
                    for uid,cosv,ratio in zip(reference[0],pair['cosine'],pair['norm_ratio']):
                        scale.append({'physical_context_id':pc,'axis':axis,'model':model,'regime':regime,**feature_meta(feature),
                            'unit_id':int(uid),'unit_type':reference[1],'cause':cause,'delta_a':pair['delta_a'],'delta_b':pair['delta_b'],
                            'cross_scale_cosine':float(cosv),'derivative_norm_ratio':float(ratio)})
    core_name=c['panels'][panels[-1]]['core'];projection,rgb,equivalent=dataset_diagnostics(d,seqs,core_name)
    common_aggregates=[];aggregate_groups=defaultdict(list)
    for row in common_rows:
        key=tuple(row[k] for k in ('physical_context_id','motion_family_id','axis','delta_m','model','regime','layer','phase','representation','r','delta_c'))
        aggregate_groups[key].append(row)
    keys=('physical_context_id','motion_family_id','axis','delta_m','model','regime','layer','phase','representation','r','delta_c')
    for key,values in aggregate_groups.items():
        common_aggregates.append({**dict(zip(keys,key)),'unit_count':len(values),
            **{f'{metric}_{stat}':value for metric in ('D_c_given_r','cause_cos','response_norm_a','response_norm_b')
               for stat,value in summary([r[metric] for r in values]).items()}})
    xy_context=aggregate_units(xy,
        ('physical_context_id','model','regime','layer','phase','representation','unit_type'),
        ('sigma_e_1','sigma_e_2','sigma_o_1','sigma_o_2','rho_e','rho_o','cos_ego_xy','cos_object_xy','subspace_overlap','relative_energy','common_energy')) if xy else []
    scale_context=aggregate_units(scale,
        ('physical_context_id','axis','model','regime','layer','phase','representation','unit_type','cause','delta_a','delta_b'),
        ('cross_scale_cosine','derivative_norm_ratio')) if scale else []
    projection_context=aggregate_units(projection,
        ('physical_context_id','motion_family_id','axis','delta_m','cause'),('median_px','median_div_patch16')) if projection else []
    folder=out/'metrics'/a.panel
    for name,data in [('family_metrics',family_rows),('local_geometry',local_rows),('matched_common_mode_pairs',common_rows),
                      ('xy_subspace_metrics',xy),('xy_subspace_context_metrics',xy_context),
                      ('scale_consistency_metrics',scale),('scale_consistency_context_metrics',scale_context),
                      ('projection_diagnostics',projection),('projection_context_summary',projection_context),
                      ('matched_common_mode_aggregates',common_aggregates),('input_rgb_controls',rgb),
                      ('equivalent_trajectory_audit',equivalent)]:write_csv(folder/(name+'.csv'),data)
    provenance={'experiment':'E2','panel':a.panel,'panels_included':panels,'config_hash':config_hash(c),'feature_manifest_sha256':digest(manifest),
        'feature_selection_sha256':records_hash(rows),
        'feature_rows_used':len(rows),'physical_context_is_reporting_unit':True,'p_values':False,'epsilon':c['epsilon'],
        'D_c_given_r_interpretation':'representation variation along common/world coordinate while target-camera relative translation r is fixed',
        'C_norm_interpretation':'compensated/common-mode sensitivity at r=0; not a special case of D_c_given_r'}
    write_json(folder/'analysis_provenance.json',provenance);print('E2 analysis complete',a.panel,{n:len(v) for n,v in [('family',family_rows),('common_pairs',common_rows),('xy',xy),('scale',scale)]})
if __name__=='__main__':main()
