#!/usr/bin/env python3
import sys,subprocess,platform,inspect,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.feature_io import arguments,configuration,write_json,digest,config_hash
from src.dataset import PilotDataset
from src.spatial_sampling import sanity_checks
from src.model_registry import configured_models,transform_for,MODELS

def main():
    a=arguments('Read-only source audit; writes only dynamic output audit').parse_args();cfg=configuration(a);out=Path(cfg['output_root'])/'audit'
    dataset=PilotDataset(cfg);data=dataset.audit();print('dataset audit:',{k:data[k] for k in ['scenes','groups','sequences','frames']},flush=True)
    spatial=sanity_checks()
    import transformers.models.dinov3_vit.modeling_dinov3_vit as dm
    model={'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,'device':torch.cuda.get_device_name(),'config_hash':config_hash(cfg),'config':cfg,'amp':'bfloat16; dense head float32','storage':cfg['storage_dtype'],'metric_dtype':'float32','available_conda_envs':[p.name for p in Path(sys.prefix).parent.iterdir() if (p/'bin/python').exists()]}
    for name in configured_models(cfg):
        c=cfg[name];repo=Path(c['repo']);source={str(p.relative_to(repo)):digest(p) for p in sorted(repo.rglob('*.py')) if '.git' not in p.parts}
        git=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],text=True,capture_output=True)
        status=subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True) if git.returncode==0 else 'No .git metadata; source file hashes recorded'
        model[name]={**c,'checkpoint_sha256':digest(c['checkpoint']),'checkpoint_bytes':Path(c['checkpoint']).stat().st_size,'git_sha':git.stdout.strip() if git.returncode==0 else None,'git_status':status,'source_sha256':source}
    if 'dinov3' in model:
        model['dinov3']['runtime_source']=inspect.getfile(dm);model['dinov3']['runtime_source_sha256']=digest(inspect.getfile(dm))
    # Reject incompatible reuse before replacing any existing provenance.
    for filename,new in [('dataset_audit.json',data),('model_audit.json',model),('spatial_sampling_audit.json',spatial),('effective_config.json',cfg)]:
        path=out/filename
        if path.exists():
            from src.feature_io import json_safe
            if json.loads(path.read_text())!=json_safe(new):
                raise ValueError(f'Audit changed: {filename}. Use a fresh --output-root; existing provenance preserved.')
    for filename,new in [('dataset_audit.json',data),('model_audit.json',model),('spatial_sampling_audit.json',spatial),('effective_config.json',cfg)]:
        if not (out/filename).exists():write_json(out/filename,new)
    # Same physical IDs must satisfy the original 8 px rule in every model's input space.
    from src.spatial_sampling import boundary_distances,occupancy
    model_spatial={}
    for name in configured_models(cfg):
        transform=transform_for(name);distances=[];counts=[]
        for s in dataset.sequences:
            uv=transform.to_input(dataset.point_uv(s));mask=transform.mask(dataset.root/dataset.frames_for(s)[-1]['target_mask'])
            distances.extend(boundary_distances(mask,uv).tolist())
            counts.append(int((occupancy(mask,MODELS[name]['patch'])>=cfg['clean_patch_occupancy']).sum()))
        assert min(distances)>=cfg['min_mask_boundary_distance_px'] and min(counts)>0
        model_spatial[name]={'transform':transform.metadata(),'patch_size':MODELS[name]['patch'],'min_core_boundary_distance_px':min(distances),'clean_patch_count_range':[min(counts),max(counts)],'shared_core_ids':True}
    write_json(out/'model_spatial_audit.json',model_spatial)
    print('audit passed; core counts:',{k:v['count'] for k,v in data['core'].items()},flush=True)
if __name__=='__main__':main()
