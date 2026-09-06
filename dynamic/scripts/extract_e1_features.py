#!/usr/bin/env python3
import sys,os,json,time,gc
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.feature_io import arguments,configuration,config_hash,save_features,write_json,digest
from src.dataset import PilotDataset
from src.model_registry import configured_models,adapter_for,regimes_for,transform_for,MODELS

SMOKE={(0,0),(1,0),(-1,0),(2,0),(-2,0),(0,1),(0,-1),(0,2),(0,-2),(1,1)}

def main():
    p=arguments('Extract compact endpoint features with audited physical correspondence')
    p.add_argument('--stage',choices=['smoke','full'],required=True)
    p.add_argument('--model',choices=['all',*MODELS],default='all')
    a=p.parse_args();cfg=configuration(a);out=Path(cfg['output_root']);audit=out/'audit'
    ma=json.loads((audit/'model_audit.json').read_text()); assert ma['config_hash']==config_hash(cfg)
    if a.stage=='full':
        gate=json.loads((audit/'extraction_smoke_validation.json').read_text());assert gate['passed'] and gate['config_hash']==config_hash(cfg)
    d=PilotDataset(cfg);da=json.loads((audit/'dataset_audit.json').read_text());d.core=da['core']
    for rel,sha in da['input_sha256'].items():
        if digest(d.root/rel)!=sha:raise ValueError(f'Dataset input changed since audit: {rel}')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    seqs=d.sequences
    if a.stage=='smoke':seqs=[s for s in seqs if s['group_id']==sorted(d.groups)[0] and (s['ego_level'],s['object_level']) in SMOKE]
    features=out/'features';features.mkdir(exist_ok=True)
    for name in configured_models(cfg):
        if a.model not in ['all',name]:continue
        assert digest(cfg[name]['checkpoint'])==ma[name]['checkpoint_sha256'],'Checkpoint changed since audit'
        for rel,sha in ma[name]['source_sha256'].items():
            assert digest(Path(cfg[name]['repo'])/rel)==sha,f'Model source changed: {rel}'
        adapter=adapter_for(name,cfg);write_json(audit/f'{name}_runtime.json',adapter.audit)
        regimes=regimes_for(name);transform=transform_for(name)
        for si,s in enumerate(seqs):
            frames=d.frames_for(s);uv=d.point_uv(s);mask=transform.mask(d.root/frames[-1]['target_mask'])
            for regime in regimes:
                path=features/name/regime/(s['sequence_id']+'.npz')
                if path.exists():
                    with np.load(path) as z: meta=json.loads(str(z['metadata_json']))
                    assert meta['config_hash']==config_hash(cfg) and meta['checkpoint_sha256']==ma[name]['checkpoint_sha256']
                    assert meta['point_ids']==d.core[s['group_id']]['point_ids']
                    continue
                indices={'Single':[7],'Pair':[0,7],'Full':list(range(8))}[regime]
                t=time.time();torch.cuda.reset_peak_memory_stats()
                arrays,debug=adapter.extract([d.root/frames[i]['rgb'] for i in indices],uv,mask,transform)
                arrays={k:v.astype(cfg['storage_dtype']) for k,v in arrays.items()}
                arrays['point_ids']=np.array(d.core[s['group_id']]['point_ids'],dtype=np.int32)
                meta={'config_hash':config_hash(cfg),'sequence_id':s['sequence_id'],'group':s['group_id'],'e':s['ego_level'],'o':s['object_level'],'r':s['relative_level'],'model':name,'regime':regime,'frame_indices':indices,'target_observation':7,'point_ids':arrays['point_ids'].tolist(),'transform':transform.metadata(),'checkpoint_sha256':ma[name]['checkpoint_sha256'],'input_rgb_sha256':[da['input_sha256'][frames[i]['rgb']] for i in indices],'debug':debug,'seconds':time.time()-t,'peak_cuda_gb':torch.cuda.max_memory_allocated()/1e9,'shapes':{k:list(v.shape) for k,v in arrays.items()}}
                save_features(path,arrays,meta)
                print(f'{a.stage} {name} {si+1}/{len(seqs)} {regime} {s["sequence_id"]} {meta["seconds"]:.2f}s peak={meta["peak_cuda_gb"]:.2f}GB',flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()
    # Rebuild manifest from complete atomic shards, making interrupted runs resumable.
    rows=[]
    for path in sorted(features.rglob('*.npz')):
        with np.load(path) as z:meta=json.loads(str(z['metadata_json']))
        rows.append({**meta,'path':str(path.relative_to(out)),'sha256':digest(path)})
    tmp=features/'feature_manifest.jsonl.tmp';tmp.write_text(''.join(json.dumps(m,sort_keys=True)+'\n' for m in rows));os.replace(tmp,features/'feature_manifest.jsonl')
    print('manifest entries',len(rows),flush=True)
if __name__=='__main__':main()
