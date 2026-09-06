"""Forward fidelity checks for any configured model subset."""
import gc,json
from pathlib import Path
import numpy as np
import torch
from .dataset import PilotDataset
from .model_registry import configured_models,adapter_for,transform_for,regimes_for
from .spatial_sampling import sample_lattice
from .feature_io import write_json

@torch.inference_mode()
def check_models(cfg):
    out=Path(cfg['output_root']);d=PilotDataset(cfg)
    d.core=json.loads((out/'audit/dataset_audit.json').read_text())['core']
    s=next(s for s in d.sequences if s['group_id']==sorted(d.groups)[0] and s['is_static'])
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results={}
    for name in configured_models(cfg):
        adapter=adapter_for(name,cfg);transform=transform_for(name);regime=regimes_for(name)[-1]
        paths=[d.root/f['rgb'] for f in d.frames_for(s)]
        if regime=='Single':paths=paths[-1:]
        uv=d.point_uv(s);mask=transform.mask(d.root/d.frames_for(s)[-1]['target_mask'])
        arrays,debug=adapter.extract(paths,uv,mask,transform)
        with np.load(out/'features'/name/regime/(s['sequence_id']+'.npz')) as z:
            repeat=max(float(np.max(np.abs(z[k]-v))) for k,v in arrays.items())
        assert repeat==0,(name,repeat)
        result={'passed':True,'sequence_id':s['sequence_id'],'regime':regime,'saved_shard_repeat_max_abs':repeat}
        if name in ['vggt','vggt_omega']:
            if name=='vggt':images=transform.images(paths).to(cfg['device'])[None]
            else:images=adapter.load_images(paths,image_resolution=512).to(cfg['device'])[None]
            with torch.autocast('cuda',dtype=torch.bfloat16):cached,start=adapter.model.aggregator(images)
            with torch.autocast('cuda',enabled=False):
                if name=='vggt':pred,conf=adapter.model.depth_head(cached,images,start)
                else:pred,conf=adapter.model.dense_head(cached,images,start)
            official=sample_lattice(adapter.fused[-1],uv,transform).cpu().numpy()
            compact=arrays['L23__fused__dense'];err=float(np.max(np.abs(official-compact)))
            assert np.allclose(official,compact,atol=2e-4,rtol=1e-4),(name,err)
            assert torch.isfinite(pred).all() and torch.isfinite(conf).all()
            result.update(official_all_frame_dense_shape=list(adapter.fused.shape),endpoint_only_vs_full_dense_max_abs=err,tolerance={'atol':2e-4,'rtol':1e-4})
            del cached,images,pred,conf
        results[name]=result;print('fidelity',name,result,flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()
    write_json(out/'audit/forward_fidelity.json',{'passed':True,'models':results})
