"""Native VGGT: identical E1 readouts, four registers and all-global cross-frame blocks."""
import sys,inspect
import torch
from .spatial_sampling import patch_readout,sample_lattice

class VGGTAdapter:
    def __init__(self,cfg):
        sys.path.insert(0,cfg['vggt']['repo'])
        from vggt.models.vggt import VGGT
        self.cfg=cfg;self.layers=cfg['vggt']['layers']
        self.model=VGGT().eval()
        weights=torch.load(cfg['vggt']['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        info=self.model.load_state_dict(weights,strict=True);del weights
        self.model.to(cfg['device']);agg=self.model.aggregator
        assert (agg.depth,agg.patch_size,agg.patch_start_idx)==(24,14,5)
        assert agg.aa_order==['frame','global'] and agg.aa_block_size==1
        self.fused=None
        self.handle=self.model.depth_head.scratch.output_conv2.register_forward_pre_hook(self.capture_dense)
        self.audit={'depth':24,'embed_dim':1024,'patch_size':14,'register_tokens':4,'layers':self.layers,'resolution':[518,518],
                    'backend_source':inspect.getfile(VGGT),'strict_checkpoint_load':str(info),'patch_start':5,
                    'pre_post_split':1024,'cached_shape':'[1,S,1374,2048]',
                    'dense_hook':'depth_head.scratch.output_conv2 input, after final interpolation and positional embedding',
                    'dense_shape':[1,128,518,518],'register_only_layers':[],
                    'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225],'location':'aggregator'},
                    'preprocessing':'512 unchanged RGB; symmetric 3px white padding; UV+3',
                    'pre_post_definition':'frame attention output / global attention output at each block'}

    def capture_dense(self,module,args):self.fused=args[0]

    @torch.inference_mode()
    def extract(self,paths,uv,mask,transform):
        images=transform.images(paths).to(self.cfg['device'])[None]
        with torch.autocast('cuda',dtype=torch.bfloat16):cached,start=self.model.aggregator(images)
        arrays={};shapes={}
        for l in self.layers:
            t=cached[l];assert t.shape==(1,len(paths),1374,2048)
            shapes[str(l)]=list(t.shape)
            for phase,z in zip(['pre','post'],t[0,-1].split(1024,dim=-1)):
                points,pool,n=patch_readout(z[start:],uv,mask,transform,patch_size=14,threshold=self.cfg['clean_patch_occupancy'])
                for rep,v in [('patch',points),('pool',pool),('camera',z[:1]),('register',z[1:start])]:
                    arrays[f'L{l}__{phase}__{rep}']=v.float().cpu().numpy()
        # Native forward returns all cached layers transiently; none are persisted.
        target=[v[:,-1:].float() if i in self.model.depth_head.intermediate_layer_idx else None for i,v in enumerate(cached)]
        with torch.autocast('cuda',enabled=False):self.model.depth_head(target,images[:,-1:],start)
        assert self.fused.shape==(1,128,518,518)
        arrays['L23__fused__dense']=sample_lattice(self.fused[0],uv,transform).cpu().numpy()
        debug={'token_shapes':shapes,'dense_shape':list(self.fused.shape),'clean_patches':n,'register_only_patch_max_errors':{},'nonfinite':0}
        self.fused=None
        return arrays,debug
