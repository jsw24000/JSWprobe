#!/usr/bin/env python3
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.feature_io import arguments,configuration,config_hash,read_jsonl,write_json,digest
from extract_e1_features import SMOKE

def main():
    p=arguments('Validate exact extraction coverage and correctness before analysis');p.add_argument('--stage',choices=['smoke','full'],required=True)
    a=p.parse_args();c=configuration(a);out=Path(c['output_root']);da=json.loads((out/'audit/dataset_audit.json').read_text());ma=json.loads((out/'audit/model_audit.json').read_text())
    frames={f['frame_id']:f for f in read_jsonl(Path(c['dataset_root'])/'manifests/frames.jsonl')}
    seqs=read_jsonl(Path(c['dataset_root'])/'manifests/sequences.jsonl')
    if a.stage=='smoke':seqs=[s for s in seqs if s['group_id']==sorted(da['core'])[0] and (s['ego_level'],s['object_level']) in SMOKE]
    rows=read_jsonl(out/'features/feature_manifest.jsonl');by={(r['sequence_id'],r['model'],r['regime']):r for r in rows};assert len(by)==len(rows)
    checked=[];maxerror=0.;nonfinite=0
    for s in seqs:
        for model,regimes in [('dinov3',['Single']),('vggt_omega',['Single','Pair','Full'])]:
            for regime in regimes:
                r=by[s['sequence_id'],model,regime];assert r['config_hash']==config_hash(c)
                assert (r['e'],r['o'],r['r'],r['group'])==(s['ego_level'],s['object_level'],s['relative_level'],s['group_id'])
                assert r['target_observation']==7 and r['checkpoint_sha256']==ma[model]['checkpoint_sha256']
                assert r['frame_indices']=={'Single':[7],'Pair':[0,7],'Full':list(range(8))}[regime]
                assert r['input_rgb_sha256']==[da['input_sha256'][frames[s['frame_ids'][i]]['rgb']] for i in r['frame_indices']]
                path=out/r['path'];assert digest(path)==r['sha256']
                with np.load(path) as z:
                    meta=json.loads(str(z['metadata_json']));assert all(r[k]==value for k,value in meta.items())
                    n=da['core'][s['group_id']]['count'];assert np.array_equal(z['point_ids'],da['core'][s['group_id']]['point_ids'])
                    expected={'point_ids','metadata_json'}
                    for l in c[model]['layers']:
                        for phase in (['norm'] if model=='dinov3' else ['pre','post']):
                            for rep,num in ([('patch',n),('pool',1)] if model=='dinov3' else [('patch',n),('pool',1),('camera',1),('register',16)]):
                                key=f'L{l}__{phase}__{rep}';expected.add(key);assert z[key].shape==(num,1024)
                    if model=='vggt_omega':expected.add('L23__fused__dense');assert z['L23__fused__dense'].shape==(n,256)
                    assert set(z.files)==expected
                    for key in expected-{'metadata_json'}:nonfinite+=int((~np.isfinite(z[key])).sum())
                if model=='vggt_omega':
                    errs=r['debug']['register_only_patch_max_errors'];assert set(errs)==set(map(str,c[model]['register_only_layers']))
                    maxerror=max(maxerror,max(errs.values()));assert r['debug']['dense_shape']==[1,256,128,128]
                assert r['debug']['clean_patches']>0
                checked.append(r['path'])
    assert nonfinite==0 and maxerror<=1e-6
    result={'passed':True,'stage':a.stage,'config_hash':config_hash(c),'shards_checked':len(checked),'expected_shards':len(seqs)*4,'nonfinite':nonfinite,'register_only_patch_max_error':maxerror,'checked':checked}
    write_json(out/f'audit/extraction_{a.stage}_validation.json',result);print({k:v for k,v in result.items() if k!='checked'})
if __name__=='__main__':main()
