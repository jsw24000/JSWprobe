#!/usr/bin/env python3
import sys,json,csv
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.feature_io import arguments,configuration,config_hash,read_jsonl,write_json,digest
from src.metrics import compute_metrics,summary
from src.dataset import PilotDataset
from src.model_registry import shards_per_sequence


def write_csv(path,rows):
    if not rows:
        Path(path).write_text('group,model,regime,layer,phase,representation,unit_id,unit_type,metric,value\n');return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    c=configuration(arguments('E1 group-wise intervention geometry').parse_args());out=Path(c['output_root'])
    PilotDataset(c)  # Reject heterogeneous axes/scales before using cached audit delta.
    gate=json.loads((out/'audit/extraction_full_validation.json').read_text());assert gate['passed'] and gate['config_hash']==config_hash(c)
    da=json.loads((out/'audit/dataset_audit.json').read_text());rows=read_jsonl(out/'features/feature_manifest.jsonl');assert len(rows)==da['sequences']*shards_per_sequence(c)
    grouped=defaultdict(list)
    for r in rows:grouped[r['group'],r['model'],r['regime']].append(r)
    points=[];groups=[];matched=[];registers=[];dense=[]
    for (g,model,regime),rr in grouped.items():
        conditions={};pointids=None
        for r in rr:
            with np.load(out/r['path']) as z:
                assert digest(out/r['path'])==r['sha256']
                ids=z['point_ids']
                if pointids is None:pointids=ids
                assert np.array_equal(ids,pointids)
                conditions[r['e'],r['o']]={k:z[k].astype(np.float32) for k in z.files if k not in ['point_ids','metadata_json']}
        assert len(conditions)==25
        for feature in conditions[0,0]:
            layer,phase,rep=feature.split('__');layer=int(layer[1:]);meta={'group':g,'model':model,'regime':regime,'layer':layer,'phase':phase,'representation':rep}
            m,pairs=compute_metrics({k:v[feature] for k,v in conditions.items()},da['delta_m'])
            if rep=='dense':m={k:v for k,v in m.items() if k in ['D_cause','cause_cos','S_e','S_o','C_resp','C_norm'] or k.startswith(('C_resp_','C_norm_'))}
            # Long group table shares schema across all representations.
            for metric,values in m.items():
                s=summary(values);groups.append({**meta,'metric':metric,**s})
                for u,value in enumerate(values):
                    row={**meta,'unit_id':int(pointids[u]) if rep in ['patch','dense'] else u,'unit_type':'physical_point' if rep in ['patch','dense'] else ('register' if rep=='register' else 'entity'),'metric':metric,'value':float(value)}
                    if rep=='register':registers.append(row)
                    elif rep=='dense':dense.append(row)
                    else:points.append(row)
            for p in pairs:
                for metric in ['D_cause','cause_cos','delta_norm_a','delta_norm_b']:
                    matched.append({**meta,'r':p['r'],'e_a':p['a'][0],'o_a':p['a'][1],'e_b':p['b'][0],'o_b':p['b'][1],'metric':metric,**summary(p[metric])})
    metrics=out/'metrics';metrics.mkdir(exist_ok=True)
    for name,data in [('point_metrics',points),('group_metrics',groups),('matched_relative_metrics',matched),('register_metrics',registers),('dense_aux_metrics',dense)]:write_csv(metrics/(name+'.csv'),data)
    layergroup=defaultdict(list)
    for r in groups:layergroup[tuple(r[k] for k in ['model','regime','layer','phase','representation','metric'])].append(r['median'])
    layerrows=[]
    for key,values in layergroup.items():layerrows.append({**dict(zip(['model','regime','layer','phase','representation','metric'],key)),**summary(values)})
    write_csv(metrics/'layer_metrics.csv',layerrows)
    write_json(metrics/'analysis_provenance.json',{'config_hash':config_hash(c),'feature_manifest_sha256':digest(out/'features/feature_manifest.jsonl'),'groups':da['groups'],'scenes':da['scenes'],'pairs_per_group':20,'D_cause_aggregation':'mean over 20 nonzero-r condition pairs per unit, then median over physical points or registers per group; equal group median across groups','epsilon':1e-8,'p_values':False,'undefined_cosines':sum(r['undefined'] for r in groups if 'cos' in r['metric'] or r['metric'].startswith('TC'))})
    print('analysis complete:',len(groups),'group metric rows;',len(matched),'pair summary rows')
if __name__=='__main__':main()
