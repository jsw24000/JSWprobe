#!/usr/bin/env python3
"""One audited example: official forward equivalence and deterministic repeat."""
import os,sys,json,gc
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from src.feature_io import arguments,configuration,write_json
from src.dataset import PilotDataset
from src.spatial_sampling import sample_lattice
from src.vggt_omega_adapter import VGGTOmegaAdapter

@torch.inference_mode()
def main():
    c=configuration(arguments('Validate wrapper versus official forward').parse_args())
    if 'models' in c:
        from src.fidelity import check_models
        check_models(c);return
    out=Path(c['output_root']);d=PilotDataset(c);d.core=json.loads((out/'audit/dataset_audit.json').read_text())['core'];torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    s=next(s for s in d.sequences if s['group_id']==sorted(d.groups)[0] and s['is_static']);paths=[d.root/f['rgb'] for f in d.frames_for(s)];uv=d.point_uv(s)
    adapter=VGGTOmegaAdapter(c);arrays,debug=adapter.extract(paths,uv,d.mask(s),d.transform)
    images=adapter.load_images(paths,image_resolution=512).to(c['device'])
    prediction=adapter.model(images)
    assert adapter.fused.shape==(8,256,128,128)
    official=sample_lattice(adapter.fused[-1],uv,d.transform).cpu().numpy();compact=arrays['L23__fused__dense']
    err=float(np.max(np.abs(official-compact)));rms=float(np.sqrt(np.mean((official-compact)**2)))
    assert np.allclose(official,compact,atol=2e-4,rtol=1e-4)
    for k in ['depth','depth_conf','pose_enc']:assert torch.isfinite(prediction[k]).all(),k
    path=out/'features/vggt_omega/Full'/(s['sequence_id']+'.npz')
    with np.load(path) as z:
        repeat=max(float(np.max(np.abs(z[k]-v))) for k,v in arrays.items())
    assert repeat==0
    result={'passed':True,'sequence_id':s['sequence_id'],'regime':'Full','official_full_dense_shape':list(adapter.fused.shape),'endpoint_only_vs_all_frames_dense_max_abs':err,'endpoint_only_vs_all_frames_dense_rmse':rms,'tolerance':{'atol':2e-4,'rtol':1e-4},'saved_shard_deterministic_repeat_max_abs':repeat,'official_depth_camera_outputs_finite':True}
    write_json(out/'audit/forward_fidelity.json',result);print(result)
if __name__=='__main__':main()
