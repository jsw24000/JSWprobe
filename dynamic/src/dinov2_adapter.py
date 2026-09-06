"""Native local DINOv2 ViT-L/14-reg4; no downloads or artificial frame mixing."""
import sys,os,inspect
import numpy as np
import torch
from .spatial_sampling import patch_readout

class DINOv2Adapter:
    def __init__(self,cfg):
        os.environ['XFORMERS_DISABLED']='1'
        sys.path.insert(0,cfg['dinov2']['repo'])
        from dinov2.hub.backbones import dinov2_vitl14_reg
        self.cfg=cfg;self.layers=cfg['dinov2']['layers']
        self.model=dinov2_vitl14_reg(pretrained=False)
        weights=torch.load(cfg['dinov2']['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        info=self.model.load_state_dict(weights,strict=True);del weights
        self.model.eval().to(cfg['device'])
        assert len(self.model.blocks)==24 and self.model.embed_dim==1024 and self.model.num_register_tokens==4
        self.audit={'depth':24,'embed_dim':1024,'patch_size':14,'register_tokens':4,'layers':self.layers,'resolution':[518,518],
                    'backend_source':inspect.getfile(type(self.model)),'strict_checkpoint_load':str(info),
                    'feature_definition':'get_intermediate_layers(norm=True); CLS index 0, registers 1:5, patches 5:',
                    'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},'preprocessing':'512 unchanged RGB; symmetric 3px white padding; UV+3'}

    @torch.inference_mode()
    def extract(self,paths,uv,mask,transform):
        assert len(paths)==1
        images=transform.images(paths).to(self.cfg['device'])
        x=(images-images.new_tensor([.485,.456,.406])[None,:,None,None])/images.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=self.model.get_intermediate_layers(x,n=self.layers,norm=True)
        arrays={};shapes={}
        for l,t in zip(self.layers,outputs):
            assert t.shape==(1,1369,1024)
            points,pool,n=patch_readout(t[0],uv,mask,transform,patch_size=14,threshold=self.cfg['clean_patch_occupancy'])
            for rep,v in [('patch',points),('pool',pool)]:arrays[f'L{l}__norm__{rep}']=v.float().cpu().numpy()
            shapes[str(l)]=list(t.shape)
        return arrays,{'token_shapes':shapes,'clean_patches':n,'nonfinite':0}
