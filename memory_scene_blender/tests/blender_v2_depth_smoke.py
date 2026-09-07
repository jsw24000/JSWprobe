"""Render one V2 endpoint with the corrected Eevee depth helper, then test cache reuse."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from memory_scene_blender.ego_object_v2.geometry import build_scene_bundle_explicit
from memory_scene_blender.ego_object_v2.rendering import render_physical_frame,axial_to_range
from memory_scene_blender.ego_object_factorial.geometry import configure_factorial_camera,geometric_track_observation
from memory_scene_blender.ego_object_factorial.rendering import refine_observation_from_render
from memory_scene_blender.object_translation.scene_builder import set_target_position


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);r=a.dataset_root;out=a.output_root
    if out.exists():raise FileExistsError(out)
    cfg=json.loads((r/'config_used.yaml').read_text());base=json.loads((ROOT/cfg['base_config']).read_text())
    c=json.loads((r/'manifests/contexts.jsonl').read_text().splitlines()[0]);g=json.loads((r/'manifests/groups.jsonl').read_text().splitlines()[0])
    b=build_scene_bundle_explicit(base,cfg['effective_mode_settings'],**{k:c[k] for k in ('context_id','room_layout_index','static_layout_id','target_category','target_variant','context_seed')})
    frames=[json.loads(x) for x in (r/'manifests/frames.jsonl').read_text().splitlines()]
    f=next(f for f in frames if f['ego_level']==2 and f['object_level']==-2 and f['frame_index']==7)
    set_target_position(b.target_asset,np.array(f['object_to_world'])[:2,3])
    cam=dict(g['base_camera'],**{k:f[k] for k in ('blender_camera_to_world','opencv_world_to_camera','opencv_camera_to_world')})
    cam['position']=f['camera_world_position'];configure_factorial_camera(cam)
    with np.load(r/c['canonical_points_path']) as z:points=z['xyz_object_local']
    observation=geometric_track_observation(points,np.array(f['object_to_world']),cam,[512,512],b.target_asset.parts,0,.025)
    result=render_physical_frame(out/'fresh',b.target_asset,cam,4,[512,512],False,None)
    updated=refine_observation_from_render(observation,out/'fresh',cfg['tracks'])
    assert updated['visible'].mean()>=.1
    assert np.array_equal(np.load(out/'fresh/depth.npy'),axial_to_range(np.load(out/'fresh/native_depth.npy'),cam['intrinsics']['K']))
    # Metadata needed by the existing physical cache contract.
    (out/'fresh/frame_metadata.json').write_text(json.dumps(dict(f,target_mask_pixel_count=result['mask_stats']['mask_pixel_count'],
        target_mask_area_ratio=result['mask_stats']['mask_area_ratio'])))
    render_physical_frame(out/'cached',b.target_asset,cam,4,[512,512],False,out/'fresh')
    for name in ('rgb.png','depth.npy','native_depth.npy','depth.exr'):
        assert (out/'fresh'/name).read_bytes()==(out/'cached'/name).read_bytes()
    report=dict(ok=True,actual_rendered_frames=1,cache_copies=1,visible_points=int(updated['visible'].sum()),point_count=len(points))
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
