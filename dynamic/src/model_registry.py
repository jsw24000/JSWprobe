"""Model capabilities shared by audit, extraction, validation and reporting."""
from importlib import import_module
from .spatial_sampling import SpatialTransform

MODELS = {
    'dinov3': dict(adapter='dinov3_adapter.DINOv3Adapter', family='image', patch=16, registers=4, dim=1024, depth=24, resolution=512),
    'vggt_omega': dict(adapter='vggt_omega_adapter.VGGTOmegaAdapter', family='geometry', patch=16, registers=16, dim=1024, depth=24, resolution=512, dense_dim=256, dense_hw=128),
    'dinov2': dict(adapter='dinov2_adapter.DINOv2Adapter', family='image', patch=14, registers=4, dim=1024, depth=24, resolution=518),
    'vggt': dict(adapter='vggt_adapter.VGGTAdapter', family='geometry', patch=14, registers=4, dim=1024, depth=24, resolution=518, dense_dim=128, dense_hw=518),
}


def configured_models(cfg):
    names=cfg.get('models',['dinov3','vggt_omega'])
    if not names or len(names)!=len(set(names)):raise ValueError('models must be nonempty and unique')
    for name in names:
        if name not in MODELS or name not in cfg:raise ValueError(f'Unknown/unconfigured model {name}')
        layers=cfg[name]['layers']
        if not layers or layers!=sorted(set(layers)) or any(l<0 or l>=MODELS[name]['depth'] for l in layers):
            raise ValueError(f'Invalid layer selection for {name}: {layers}')
        if cfg[name].get('regimes',regimes_for(name))!=regimes_for(name):
            raise ValueError(f'E1 requires the fixed regimes {regimes_for(name)} for {name}')
    return names


def regimes_for(name):
    return ['Single'] if MODELS[name]['family']=='image' else ['Single','Pair','Full']


def shards_per_sequence(cfg):return sum(len(regimes_for(n)) for n in configured_models(cfg))


def transform_for(name):
    if MODELS[name]['resolution']==512:return SpatialTransform()
    return SpatialTransform(input_hw=(518,518),pad_xy=(3.,3.))


def adapter_for(name,cfg):
    module,cls=MODELS[name]['adapter'].split('.')
    return getattr(import_module('.'+module,__package__),cls)(cfg)


def expected_shapes(name,cfg,n):
    s=MODELS[name]; shapes={}
    for l in cfg[name]['layers']:
        if s['family']=='image': phases=['norm'];reps=[('patch',n),('pool',1)]
        else:phases=['pre','post'];reps=[('patch',n),('pool',1),('camera',1),('register',s['registers'])]
        for phase in phases:
            for rep,count in reps:shapes[f'L{l}__{phase}__{rep}']=(count,s['dim'])
    if s['family']=='geometry':shapes[f'L{s["depth"]-1}__fused__dense']=(n,s['dense_dim'])
    return shapes
