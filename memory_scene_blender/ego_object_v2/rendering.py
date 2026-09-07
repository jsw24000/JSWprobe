"""V2 Eevee depth convention: retain native axial Z, expose ray-range NPY."""
from pathlib import Path
import os
import shutil
import numpy as np
from memory_scene_blender.ego_object_factorial.rendering import render_or_reuse_physical_frame as legacy_render

DEPTH_CONVENTION = dict(depth_exr='native Eevee axial camera Z in metres',
                        native_depth_npy='native Eevee axial camera Z in metres',
                        depth_npy='Euclidean camera-ray range in metres; axial Z times pixel-ray norm')


def axial_to_range(depth, intrinsic_k):
    """Pixel centers are (u+0.5,v+0.5); K uses the image edge coordinate convention."""
    depth=np.asarray(depth)
    y,x=np.indices(depth.shape,dtype=np.float64)
    rays=np.stack([x+.5,y+.5,np.ones_like(x)],axis=-1)@np.linalg.inv(np.asarray(intrinsic_k)).T
    return (depth*np.linalg.norm(rays,axis=-1)).astype(np.float32)


def render_physical_frame(frame_dir,target_asset,camera_payload,samples,resolution,resume,cached_frame_dir):
    import bpy
    if bpy.app.version[:2] != (5, 2) or bpy.context.scene.render.engine != 'BLENDER_EEVEE':
        raise RuntimeError('V2 native axial-depth calibration currently supports Blender 5.2 / BLENDER_EEVEE only')
    result=legacy_render(frame_dir,target_asset,camera_payload,samples,resolution,resume,cached_frame_dir)
    frame_dir=Path(frame_dir)
    native=frame_dir/'native_depth.npy'
    if cached_frame_dir is None:
        if resume:raise ValueError('V2 does not support implicit resume')
        axial=np.load(frame_dir/'depth.npy',allow_pickle=False)
        np.save(native,axial)
        np.save(frame_dir/'depth.npy',axial_to_range(axial,camera_payload['intrinsics']['K']))
    else:
        source=Path(cached_frame_dir)/'native_depth.npy'
        try:os.link(source,native)
        except OSError:shutil.copy2(source,native)
    return result
