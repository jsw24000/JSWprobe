import inspect
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import DINOv3ViTModel
from .spatial_sampling import patch_readout


class DINOv3Adapter:
    def __init__(self,cfg):
        self.cfg=cfg; self.layers=cfg['dinov3']['layers']; self.device=cfg['device']
        if Path(cfg['dinov3']['checkpoint']).name != 'model.safetensors':
            raise ValueError('This offline HF adapter requires the exact model.safetensors file beside config.json; arbitrary filenames are not silently substituted.')
        self.model,info=DINOv3ViTModel.from_pretrained(str(Path(cfg['dinov3']['checkpoint']).parent),local_files_only=True,output_loading_info=True,attn_implementation='sdpa')
        assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'],info
        self.model=self.model.eval().to(self.device)
        c=self.model.config;assert (c.num_hidden_layers,c.hidden_size,c.patch_size,c.num_register_tokens)==(24,1024,16,4)
        assert self.layers==[round(c.num_hidden_layers*f)-1 for f in [.25,.5,.75,1]]
        self.captured={}; self.handles=[]
        for l in self.layers:
            self.handles.append(self.model.layer[l].register_forward_hook(self.hook(l)))
        self.audit={'depth':c.num_hidden_layers,'embed_dim':c.hidden_size,'patch_size':c.patch_size,'register_tokens':c.num_register_tokens,'layers':self.layers,'backend_source':inspect.getfile(DINOv3ViTModel),'loading_info':info,'feature_definition':'selected block output followed by shared model.norm, patch tokens [5:]','resolution':[512,512],'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},'resolution_support':'Runtime RoPE grid uses input H/16,W/16; no resize/crop; model config training default is 224'}

    def hook(self,l):
        def capture(module,args,out): self.captured[l]=self.model.norm(out)
        return capture

    @torch.inference_mode()
    def extract(self,paths,uv,mask,transform):
        assert len(paths)==1
        a=np.asarray(Image.open(paths[0]).convert('RGB')).copy()
        assert a.shape==(512,512,3)
        x=torch.from_numpy(a).permute(2,0,1).float()[None].to(self.device)/255
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        self.captured={}
        with torch.autocast('cuda',dtype=torch.bfloat16): self.model(pixel_values=(x-mean)/std)
        arrays={};shapes={}
        for l,t in self.captured.items():
            assert t.shape==(1,1029,1024)
            points,pool,n=patch_readout(t[0,5:],uv,mask,transform,threshold=self.cfg['clean_patch_occupancy'])
            for name,v in [('patch',points),('pool',pool)]:arrays[f'L{l}__norm__{name}']=v.float().cpu().numpy()
            shapes[str(l)]=list(t.shape)
        self.captured={}
        return arrays,{'token_shapes':shapes,'clean_patches':n,'nonfinite':0}
