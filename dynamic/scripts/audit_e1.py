#!/usr/bin/env python3
import sys,subprocess,platform,inspect,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.feature_io import arguments,configuration,write_json,digest,config_hash
from src.dataset import PilotDataset
from src.spatial_sampling import sanity_checks

def main():
    a=arguments('Read-only source audit; writes only dynamic output audit').parse_args();cfg=configuration(a);out=Path(cfg['output_root'])/'audit'
    data=PilotDataset(cfg).audit();print('dataset audit:',{k:data[k] for k in ['scenes','groups','sequences','frames']},flush=True)
    spatial=sanity_checks()
    import transformers.models.dinov3_vit.modeling_dinov3_vit as dm
    model={'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,'device':torch.cuda.get_device_name(),'config_hash':config_hash(cfg),'config':cfg,'amp':'bfloat16; dense head float32','storage':cfg['storage_dtype'],'metric_dtype':'float32','available_conda_envs':[p.name for p in Path(sys.prefix).parent.iterdir() if (p/'bin/python').exists()]}
    for name in ['dinov3','vggt_omega']:
        c=cfg[name];repo=Path(c['repo']);source={str(p.relative_to(repo)):digest(p) for p in sorted(repo.rglob('*.py')) if '.git' not in p.parts}
        model[name]={**c,'checkpoint_sha256':digest(c['checkpoint']),'checkpoint_bytes':Path(c['checkpoint']).stat().st_size,'git_sha':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'git_status':subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True),'source_sha256':source}
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
    print('audit passed; core counts:',{k:v['count'] for k,v in data['core'].items()},flush=True)
if __name__=='__main__':main()
