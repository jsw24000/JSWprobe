"""Explicit edge-coordinate transforms. Render UV origin is top-left image edge.
Blender K has principal point (W/2,H/2); pixel centers are (x+.5,y+.5).
"""
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialTransform:
    original_hw: tuple = (512, 512)
    input_hw: tuple = (512, 512)
    crop_xy: tuple = (0., 0.)
    crop_hw: tuple = (512, 512)
    pad_xy: tuple = (0., 0.)

    def to_input(self, uv):
        scale = np.array(self.input_hw[::-1]) / np.array(self.crop_hw[::-1])
        return (np.asarray(uv) - self.crop_xy) * scale + self.pad_xy

    def grid(self, uv, grid_hw):
        return self.to_input(uv) * (np.array(grid_hw[::-1]) / np.array(self.input_hw[::-1])) - .5

    def metadata(self): return asdict(self)

    def mask(self, path):
        im = Image.open(path).convert('L')
        if im.size != tuple(self.original_hw[::-1]): raise ValueError('Original resolution mismatch')
        x,y = self.crop_xy; h,w = self.crop_hw
        im = im.crop((int(x),int(y),int(x+w),int(y+h)))
        if any(self.pad_xy): raise NotImplementedError('Padding requires explicit padded canvas')
        return np.asarray(im.resize(tuple(self.input_hw[::-1]), Image.Resampling.NEAREST)) > 127


def boundary_distances(mask, input_uv):
    # EDT is a center-to-center distance; subtract 0.5 conservatively for pixel boundary.
    field = np.maximum(distance_transform_edt(np.pad(mask, 1))[1:-1,1:-1] - .5, 0)
    return map_coordinates(field, [input_uv[:,1]-.5, input_uv[:,0]-.5], order=1, mode='constant', cval=0)


def occupancy(mask, patch_size):
    h,w = mask.shape
    assert h % patch_size == 0 and w % patch_size == 0
    return mask.reshape(h//patch_size,patch_size,w//patch_size,patch_size).mean((1,3))


def sample_lattice(feature, uv, transform):
    """CHW -> PD, bilinear latent lattice sampling (not pixel-level features)."""
    assert feature.ndim == 3
    c,h,w = feature.shape
    coords = transform.grid(uv, (h,w))
    grid = torch.as_tensor((coords+.5)/np.array([w,h])*2-1, device=feature.device, dtype=torch.float32)
    return F.grid_sample(feature.float()[None], grid[None,None], mode='bilinear', padding_mode='border', align_corners=False)[0,:,0].T


def patch_readout(tokens, uv, mask, transform, patch_size=16, threshold=.9):
    h,w = transform.input_hw; gh,gw = h//patch_size,w//patch_size
    assert tokens.shape[0] == gh*gw
    lattice = tokens.reshape(gh,gw,-1).permute(2,0,1)
    points = sample_lattice(lattice, uv, transform)
    clean = occupancy(mask,patch_size).reshape(-1) >= threshold
    if not clean.any(): raise ValueError('No clean interior patches; auxiliary pool unavailable')
    pool = tokens[torch.as_tensor(clean,device=tokens.device)].float().mean(0,keepdim=True)
    return points, pool, int(clean.sum())


def farthest_points(ids, xyz, limit):
    order = np.argsort(ids); ids,xyz = ids[order],xyz[order]
    if len(ids)<=limit: return ids
    chosen = [0]; dist = np.full(len(ids), np.inf)
    for _ in range(1,limit):
        dist = np.minimum(dist, ((xyz-xyz[chosen[-1]])**2).sum(1))
        chosen.append(int(np.argmax(dist)))
    return np.sort(ids[chosen])


def sanity_checks():
    t=SpatialTransform(); centers=np.array([[8.,8.],[24.,8.],[504.,504.]])
    assert np.array_equal(t.grid(centers,(32,32)),[[0,0],[1,0],[31,31]])
    assert np.array_equal(t.grid(np.array([[0,0],[512,512]]),(32,32)),[[-.5,-.5],[31.5,31.5]])
    lattice=torch.arange(32*32).reshape(1,32,32).float()
    assert torch.allclose(sample_lattice(lattice,centers,t)[:,0],torch.tensor([0.,1.,1023.]))
    assert torch.allclose(sample_lattice(lattice,np.array([[16.,16.]]),t)[0,0],torch.tensor(16.5))
    resize=SpatialTransform(input_hw=(256,256)); assert np.array_equal(resize.to_input(centers),centers/2)
    return {'passed':True,'patch_centers':centers.tolist(),'patch_grid':t.grid(centers,(32,32)).tolist(),
            'corners_grid':[[-.5,-.5],[31.5,31.5]],'row_major_reshape':True,'bilinear_ramp_check':True,
            'uv_convention':'top-left edge coordinates; pixel center x+0.5; patch center 16*j+8',
            'transform':t.metadata(),'boundary_distance':'bilinear EDT minus 0.5 px (conservative)',
            'interpolation_limit':'Latent feature lattice interpolation, not true pixel features'}
