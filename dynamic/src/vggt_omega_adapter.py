import sys
import inspect
import numpy as np
import torch
from .spatial_sampling import patch_readout,sample_lattice


class VGGTOmegaAdapter:
    def __init__(self,cfg):
        self.cfg=cfg;sys.path.insert(0,cfg['vggt_omega']['repo'])
        from vggt_omega.models.vggt_omega import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        self.load_images=load_and_preprocess_images
        self.model=VGGTOmega().eval()
        weights=torch.load(cfg['vggt_omega']['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        info=self.model.load_state_dict(weights,strict=True);del weights
        self.model.to(cfg['device']);self.layers=cfg['vggt_omega']['layers']
        agg=self.model.aggregator;agg.cached_layer_indices=set(self.layers)
        assert (agg.depth,agg.patch_size,agg.patch_token_start)==(24,16,17)
        assert [i for i,t in enumerate(agg.inter_frame_attention_types) if t=='register']==cfg['vggt_omega']['register_only_layers']
        self.fused=None
        self.handle=self.model.dense_head.proj.register_forward_pre_hook(self.capture_dense)
        self.audit={'depth':agg.depth,'embed_dim':agg.camera_token.shape[-1],'patch_size':agg.patch_size,'register_tokens':16,'layers':self.layers,'backend_source':inspect.getfile(VGGTOmega),'strict_checkpoint_load':str(info),'patch_start':17,'cached_shape':'[1,S,1041,2048]','pre_post_split':1024,'dense_hook':'dense_head.proj forward_pre_hook, after final positional embedding, before depth projection','resolution':[512,512],'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225],'location':'aggregator.forward; input is RGB [0,1]'},'register_only_layers':cfg['vggt_omega']['register_only_layers']}

    def capture_dense(self,module,args): self.fused=args[0]

    @torch.inference_mode()
    def extract(self,paths,uv,mask,transform):
        images=self.load_images(paths,image_resolution=512).to(self.cfg['device'])[None]
        assert images.shape==(1,len(paths),3,512,512)
        with torch.autocast('cuda',dtype=torch.bfloat16): cached,start=self.model.aggregator(images)
        arrays={};checks={};shapes={}
        for l in self.layers:
            t=cached[l];assert t.shape==(1,len(paths),1041,2048)
            pre,post=t[0,-1].split(1024,dim=-1);shapes[str(l)]=list(t.shape)
            if l in self.cfg['vggt_omega']['register_only_layers']:
                error=float((pre[start:].float()-post[start:].float()).abs().max())
                checks[str(l)]=error
                if error>1e-6:raise ValueError(f'Register-only patch mismatch at {l}: {error}')
            for phase,z in [('pre',pre),('post',post)]:
                points,pool,n=patch_readout(z[start:],uv,mask,transform,threshold=self.cfg['clean_patch_occupancy'])
                for name,v in [('patch',points),('pool',pool),('camera',z[0:1]),('register',z[1:17])]:
                    arrays[f'L{l}__{phase}__{name}']=v.float().cpu().numpy()
        # DenseHead has no frame mixing: slice endpoint tokens after full-context aggregation.
        target=[None if v is None else v[:,-1:] for v in cached]
        with torch.autocast('cuda',enabled=False):
            self.model.dense_head(target,images[:,-1:],start)
        assert self.fused.shape==(1,256,128,128)
        arrays['L23__fused__dense']=sample_lattice(self.fused[0],uv,transform).cpu().numpy()
        meta={'token_shapes':shapes,'dense_shape':list(self.fused.shape),'clean_patches':n,'register_only_patch_max_errors':checks,'nonfinite':0}
        self.fused=None
        return arrays,meta
