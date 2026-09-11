#!/usr/bin/env python3
import atexit,gc,hashlib,json,os,sys,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.e2_dataset import E2Dataset,validate_cache_metadata
from src.feature_io import arguments,configuration,config_hash,digest,save_features,write_json,exclusive_file_lock
from src.model_registry import MODELS,adapter_for,configured_models,regimes_for,transform_for

def json_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def main():
    p=arguments('Extract resumable E2 endpoint features into one shared cache');p.add_argument('--panel',required=True)
    p.add_argument('--stage',choices=['smoke','full'],required=True);p.add_argument('--model',choices=['all',*MODELS],default='all')
    a=p.parse_args();c=configuration(a);out=Path(c['output_root']);da=json.loads((out/'audit/dataset_audit.json').read_text());ma=json.loads((out/'audit/model_audit.json').read_text())
    if ma['config_hash']!=config_hash(c):raise ValueError('E2 config changed since audit')
    if a.stage=='full':
        gate=out/'audit'/f'extraction_{a.panel}_smoke_validation.json'
        smoke=json.loads(gate.read_text()) if gate.exists() else {}
        if not smoke.get('passed') or smoke.get('config_hash')!=config_hash(c):raise ValueError(f'Passed compatible smoke validation required before full extraction: {gate}')
    d=E2Dataset(c);d.cores=da['cores'];d.sequence_point_ids=da['sequence_point_ids'];seqs=d.sequences_for_request(a.panel,a.stage=='smoke')
    for rel,sha in da['input_manifest_sha256'].items():
        if digest(d.root/rel)!=sha:raise ValueError(f'Dataset manifest changed since audit: {rel}')
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    features=out/'features';features.mkdir(parents=True,exist_ok=True)
    lock=exclusive_file_lock(features/'.writer.lock');lock.__enter__();atexit.register(lock.__exit__,None,None,None)
    for model in configured_models(c):
        if a.model not in ('all',model):continue
        if digest(c[model]['checkpoint'])!=ma[model]['checkpoint_sha256']:raise ValueError(f'{model} checkpoint changed')
        for rel,sha in ma[model]['source_sha256'].items():
            if digest(Path(c[model]['repo'])/rel)!=sha:raise ValueError(f'{model} source changed: {rel}')
        adapter=adapter_for(model,c);write_json(out/'audit'/f'{model}_runtime.json',adapter.audit);transform=transform_for(model)
        source_hash=json_digest(ma[model]['source_sha256'])
        for i,s in enumerate(seqs):
            track_sha=da['input_manifest_sha256'].get(s['tracks_path'])
            if track_sha and digest(d.root/s['tracks_path'])!=track_sha:raise ValueError(f'Dataset track changed: {s["tracks_path"]}')
            point_ids=d.sequence_point_ids[s['sequence_id']];uv=d.point_uv(s,point_ids);frames=d.frames_for(s);mask=transform.mask(d.root/frames[-1]['target_mask'])
            for regime in regimes_for(model):
                indices={'Single':[7],'Pair':[0,7],'Full':list(range(8))}[regime];rgb_hashes=[digest(d.root/frames[j]['rgb']) for j in indices]
                expected={'experiment':'E2','config_hash':config_hash(c),'sequence_id':s['sequence_id'],'group_id':s['group_id'],
                    'physical_context_id':s['physical_context_id'],'context_id':s['context_id'],'target_id':s['target_id'],
                    'background_id':s['background_id'],'motion_family_id':s['motion_family_id'],'motion_axis_world':s['motion_axis_world'],
                    'delta_m':s['delta_m'],'ego_level':s['ego_level'],'object_level':s['object_level'],'relative_level':s['relative_level'],
                    'model':model,'regime':regime,'target_frame_index':7,'selected_frame_indices':indices,'point_ids':point_ids,
                    'checkpoint_sha256':ma[model]['checkpoint_sha256'],'model_source_hashes_sha256':source_hash,
                    'input_rgb_sha256':rgb_hashes,'preprocessing_transform':transform.metadata()}
                path=features/model/regime/(s['sequence_id']+'.npz')
                if path.exists():
                    with np.load(path) as z:metadata=json.loads(str(z['metadata_json']))
                    validate_cache_metadata(metadata,expected);continue
                start=time.time();torch.cuda.reset_peak_memory_stats();arrays,debug=adapter.extract([d.root/frames[j]['rgb'] for j in indices],uv,mask,transform)
                arrays={k:v.astype(c['storage_dtype']) for k,v in arrays.items()};arrays['point_ids']=np.asarray(point_ids,np.int32)
                metadata={**expected,'debug':debug,'seconds':time.time()-start,'peak_cuda_gb':torch.cuda.max_memory_allocated()/1e9,
                          'feature_names':sorted(arrays),'shapes':{k:list(v.shape) for k,v in arrays.items()}}
                save_features(path,arrays,metadata);print(f'{a.panel}/{a.stage} {model} {i+1}/{len(seqs)} {regime}',flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()
    rows=[]
    for path in sorted(features.rglob('*.npz')):
        with np.load(path) as z:meta=json.loads(str(z['metadata_json']))
        rows.append({**meta,'path':str(path.relative_to(out)),'sha256':digest(path)})
    tmp=features/'feature_manifest.jsonl.tmp';tmp.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));os.replace(tmp,features/'feature_manifest.jsonl')
    print('E2 feature manifest entries:',len(rows))
if __name__=='__main__':main()
