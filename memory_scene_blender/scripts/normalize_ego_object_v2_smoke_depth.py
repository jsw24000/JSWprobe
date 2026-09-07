#!/usr/bin/env python3
"""Preserve an initial V2 smoke; normalize native Eevee depth in a fresh copy.

No images are rendered. Only V2 smoke from the initial native-Z-as-range schema
is accepted. RGB/EXR bytes stay unchanged; range NPY and track visibility are
recomputed, with explicit migration provenance. V1 and full roots are rejected.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from memory_scene_blender.ego_object_v2.rendering import axial_to_range, DEPTH_CONVENTION
from memory_scene_blender.ego_object_factorial.geometry import depth_buffer_track_visibility, observation_summary
from memory_scene_blender.ego_object_factorial.rendering import save_npz
from memory_scene_blender.object_translation.manifest_utils import read_json,read_jsonl,write_json,write_jsonl
from memory_scene_blender.scripts.validate_ego_object_x_factorial_dataset import _load_mask
from memory_scene_blender.ego_object_v2.reports import write_reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();src=a.source_root.resolve();out=a.output_root.resolve()
    summary=read_json(src/'dataset_summary.json')
    if summary.get('dataset_name')!='ego_object_factorial_v2' or summary.get('mode')!='smoke' or summary.get('dry_run') or summary['counts']['frames']>400:
        raise ValueError('Only an initial rendered V2 smoke is accepted')
    if out.exists() or out in src.parents or src in out.parents:raise ValueError('Choose a fresh, separate output root')
    ff=read_jsonl(src/'manifests/frames.jsonl')
    if any(f['render_convention'].get('depth_exr_and_npy')!='Blender Z-pass camera-ray range in metres' for f in ff):
        raise ValueError('Input is not the initial V2 depth schema')
    shutil.copytree(src,out)
    (out/'validation_summary.json').unlink(missing_ok=True)
    if (out/'previews').exists():shutil.rmtree(out/'previews')
    cfg=read_json(out/'config_used.yaml');tc=cfg['tracks']
    by_id={f['frame_id']:f for f in ff}
    for f in ff:
        native_rel=str(Path(f['frame_dir'])/'native_depth.npy')
        native=np.load(src/f['depth_npy'],allow_pickle=False)
        np.save(out/native_rel,native)
        np.save(out/f['depth_npy'],axial_to_range(native,f['K']))
        f['native_depth_npy']=native_rel;f['frame_paths']['native_depth_npy']=native_rel
        f['render_convention'].pop('depth_exr_and_npy')
        f['render_convention'].update(DEPTH_CONVENTION)
    sequences=read_jsonl(out/'manifests/sequences.jsonl')
    for s in sequences:
        with np.load(src/s['tracks_path'],allow_pickle=False) as z:tr={k:z[k] for k in z.files}
        for i,fid in enumerate(s['frame_ids']):
            f=by_id[fid]
            obs={k:tr[k][i] for k in ['xyz_world','xyz_camera','projected_uv','depth_camera_z_m','range_to_camera_m','in_front_of_camera','in_image','visible']}
            obs=depth_buffer_track_visibility(obs,np.load(out/f['depth_npy']),_load_mask(out/f['target_mask']),
                tc['visibility_depth_abs_tolerance_m'],tc['visibility_depth_rel_tolerance'],tc['visibility_pixel_radius'])
            for k in ['visible','depth_buffer_range_m','depth_buffer_error_m']:tr[k][i]=obs[k]
            f['track_observation']=observation_summary(obs)
            write_json(out/f['frame_metadata'],f)
        save_npz(out/s['tracks_path'],tr)
    write_jsonl(out/'manifests/frames.jsonl',ff)
    manifests={p.stem:read_jsonl(p) for p in (out/'manifests').glob('*.jsonl')}
    write_reports(out,read_json(out/'context_plan.json'),manifests)
    provenance=read_json(out/'provenance.json')
    provenance['depth_normalization']=dict(source_dataset_root=str(src),source_provenance_sha256=hashlib.sha256((src/'provenance.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),conventions=DEPTH_CONVENTION,images_rerendered=False,
        original_generator_source_hashes_retained=True)
    write_json(out/'provenance.json',provenance)
    print(json.dumps(dict(output_root=str(out),counts=summary['counts'],images_rerendered=False),indent=2))


if __name__=='__main__':main()
