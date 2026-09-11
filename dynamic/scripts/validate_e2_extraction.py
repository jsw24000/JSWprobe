#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.e2_dataset import E2Dataset,validate_cache_metadata
from src.feature_io import arguments,configuration,config_hash,digest,json_safe,read_jsonl,records_hash,write_json
from src.model_registry import MODELS,configured_models,expected_shapes,regimes_for,transform_for

def main():
    p=arguments('Validate E2 cache coverage, identities and provenance');p.add_argument('--panel',required=True);p.add_argument('--stage',choices=['smoke','full'],required=True)
    a=p.parse_args();c=configuration(a);out=Path(c['output_root']);da=json.loads((out/'audit/dataset_audit.json').read_text());ma=json.loads((out/'audit/model_audit.json').read_text())
    d=E2Dataset(c);d.cores=da['cores'];d.sequence_point_ids=da['sequence_point_ids'];seqs=d.sequences_for_request(a.panel,a.stage=='smoke')
    rows=read_jsonl(out/'features/feature_manifest.jsonl');by={(r['sequence_id'],r['model'],r['regime']):r for r in rows}
    checked=[];nonfinite=0;max_register_error=0.
    for s in seqs:
        ids=d.sequence_point_ids[s['sequence_id']]
        for model in configured_models(c):
            for regime in regimes_for(model):
                key=(s['sequence_id'],model,regime)
                if key not in by:raise ValueError(f'Missing E2 feature shard: {key}')
                r=by[key];expected={'experiment':'E2','config_hash':config_hash(c),'sequence_id':s['sequence_id'],'group_id':s['group_id'],
                    'physical_context_id':s['physical_context_id'],'motion_family_id':s['motion_family_id'],'point_ids':ids,'model':model,'regime':regime}
                validate_cache_metadata(r,expected);path=out/r['path']
                if digest(path)!=r['sha256']:raise ValueError(f'Feature digest changed: {path}')
                with np.load(path) as z:
                    metadata=json.loads(str(z['metadata_json']))
                    for key,value in metadata.items():
                        if r.get(key)!=value:raise ValueError(f'Manifest/shard metadata mismatch for {key}: {path}')
                    if not np.array_equal(z['point_ids'],ids):raise ValueError(f'Point IDs changed: {path}')
                    expected_keys={'point_ids','metadata_json'}
                    for name,shape in expected_shapes(model,c,len(ids)).items():expected_keys.add(name);assert z[name].shape==shape,(name,z[name].shape,shape)
                    if set(z.files)!=expected_keys:raise ValueError(f'Unexpected arrays in {path}')
                    nonfinite+=sum(int((~np.isfinite(z[k])).sum()) for k in expected_keys-{'metadata_json'})
                if MODELS[model]['family']=='geometry':max_register_error=max(max_register_error,max(r['debug']['register_only_patch_max_errors'].values(),default=0.))
                if r['preprocessing_transform']!=json_safe(transform_for(model).metadata()):raise ValueError(f'Preprocessing drift: {model}')
                checked.append(r['path'])
    if nonfinite or max_register_error>1e-6:raise ValueError(f'Invalid features: nonfinite={nonfinite}, register_error={max_register_error}')
    result={'passed':True,'experiment':'E2','panel':a.panel,'stage':a.stage,'config_hash':config_hash(c),'shards_checked':len(checked),
            'expected_shards':sum(len(regimes_for(m)) for m in configured_models(c))*len(seqs),'nonfinite':nonfinite,
            'register_only_patch_max_error':max_register_error,'feature_manifest_sha256':digest(out/'features/feature_manifest.jsonl'),
            'feature_selection_sha256':records_hash([by[(s['sequence_id'],m,r)] for s in seqs for m in configured_models(c) for r in regimes_for(m)]),
            'checked':checked}
    write_json(out/'audit'/f'extraction_{a.panel}_{a.stage}_validation.json',result);print({k:v for k,v in result.items() if k!='checked'})
if __name__=='__main__':main()
