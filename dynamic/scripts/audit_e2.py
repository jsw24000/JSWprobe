#!/usr/bin/env python3
import inspect,json,platform,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.e2_dataset import E2Dataset
from src.feature_io import arguments,configuration,config_hash,digest,json_safe,write_json
from src.model_registry import configured_models
from src.spatial_sampling import sanity_checks

def main():
    c=configuration(arguments('Audit E2 dataset/model provenance without model forward').parse_args());out=Path(c['output_root'])/'audit'
    data=E2Dataset(c).audit(hash_tracks=True)
    model={'experiment':'E2','python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,
           'config_hash':config_hash(c),'config':c,'device_requested':c['device']}
    for name in configured_models(c):
        spec=c[name];repo=Path(spec['repo']);source={str(p.relative_to(repo)):digest(p) for p in sorted(repo.rglob('*.py')) if '.git' not in p.parts}
        git=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],text=True,capture_output=True)
        model[name]={**spec,'checkpoint_sha256':digest(spec['checkpoint']),'checkpoint_bytes':Path(spec['checkpoint']).stat().st_size,
            'git_sha':git.stdout.strip() if git.returncode==0 else None,'source_sha256':source}
    outputs={'dataset_audit.json':data,'model_audit.json':model,'spatial_sampling_audit.json':sanity_checks(),'effective_config.json':c}
    for name,value in outputs.items():
        path=out/name
        if path.exists() and json.loads(path.read_text())!=json_safe(value):raise ValueError(f'E2 audit changed: {name}; use a fresh output root')
    for name,value in outputs.items():
        if not (out/name).exists():write_json(out/name,value)
    print('E2 audit passed:',{k:data[k] for k in ('physical_contexts','groups','sequences','frames')})
if __name__=='__main__':main()
