#!/usr/bin/env python3
"""Generate V2 novel contexts. Real full rendering requires --allow-full."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from memory_scene_blender.ego_object_v2.protocol import (
    make_plan, LEVELS, FULL_COUNTS, group_id, sequence_id, matched_relative_groups)
from memory_scene_blender.ego_object_v2.geometry import build_scene_bundle_explicit, select_physical_context
from memory_scene_blender.ego_object_factorial.geometry import (
    configure_factorial_camera, geometric_track_observation, observation_summary, sample_target_surface_points)
from memory_scene_blender.ego_object_factorial.protocol import linear_alphas, motion_conditions, project_opencv, translated_camera_payload
from memory_scene_blender.ego_object_factorial.rendering import (
    save_npz, stack_track_observations, refine_observation_from_render)
from memory_scene_blender.ego_object_v2.rendering import render_physical_frame, DEPTH_CONVENTION
from memory_scene_blender.object_translation.assets import asset_world_bbox
from memory_scene_blender.object_translation.scene_builder import set_target_position
from memory_scene_blender.object_translation.manifest_utils import ensure_dir, load_config, write_json, write_jsonl, np_to_list, vector_to_list


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'memory_scene_blender/configs/ego_object_factorial_v2.yaml')
    p.add_argument('--mode', choices=['smoke','full'], default='smoke')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--allow-full', action='store_true')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--smoke-all-families', action='store_true', help='Geometry-only smoke for all six matched families')
    p.add_argument('--context-id', action='append', dest='context_ids',
                   help='Generate only this planned context (repeatable; requires --mode full)')
    p.add_argument('--camera-distance-scale', type=float,
                   help='Move candidate cameras toward look-at by this factor; intended for scoped repair')
    p.add_argument('--context-camera-distance-scale', action='append', default=[], metavar='CONTEXT_ID=SCALE',
                   help='Override camera distance scale for one requested context')
    return p.parse_args(argv)


def relative_path(path, root):
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def target_centers(target_asset: Any, camera: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    bbox = asset_world_bbox(target_asset)
    array = np.asarray(bbox, dtype=np.float64)
    center_world = 0.5 * (array.min(axis=0) + array.max(axis=0))
    center_camera, _uv = project_opencv(
        center_world[None, :], camera["opencv_world_to_camera"], camera["intrinsics"]["K"]
    )
    return center_world, center_camera[0], [vector_to_list(corner) for corner in bbox]


def frame_paths(frame_dir: Path, root: Path) -> Dict[str, str]:
    names = {
        "rgb": "rgb.png",
        "depth": "depth.exr",
        "depth_exr": "depth.exr",
        "depth_npy": "depth.npy",
        "native_depth_npy": "native_depth.npy",
        "normal": "normal.png",
        "albedo": "albedo.png",
        "target_mask": "target_mask.png",
        "instance": "instance.png",
        "semantic": "semantic.png",
        "object_id": "object_id.png",
        "frame_metadata": "frame_metadata.json",
    }
    return {key: relative_path(frame_dir / filename, root) for key, filename in names.items()}



def generate_dataset(args):
    if args.mode == 'full' and not args.dry_run and not args.allow_full:
        raise ValueError('Real full rendering requires explicit --allow-full')
    if args.smoke_all_families and (args.mode != 'smoke' or not args.dry_run):
        raise ValueError('--smoke-all-families requires smoke --dry-run')
    if args.context_ids and args.mode != 'full':
        raise ValueError('--context-id requires --mode full')
    if args.camera_distance_scale is not None and not args.context_ids:
        raise ValueError('--camera-distance-scale requires --context-id')
    scale_overrides = {}
    for item in args.context_camera_distance_scale:
        context_id, separator, value = item.partition('=')
        if not separator:
            raise ValueError('--context-camera-distance-scale must be CONTEXT_ID=SCALE')
        scale_overrides[context_id] = float(value)
    if scale_overrides and not args.context_ids:
        raise ValueError('--context-camera-distance-scale requires --context-id')
    import bpy
    config = load_config(args.config)
    plan = make_plan(config)
    if config['motion'] != dict(levels=LEVELS, temporal_profile='linear') or config['expected_full_counts'] != FULL_COUNTS:
        raise ValueError('V2 confirmation protocol/count oracle changed')
    settings = config['modes'][args.mode]
    if settings['num_frames'] != 8 or settings['resolution'] != [512,512] or (args.mode=='full' and settings['samples'] != 8):
        raise ValueError('V2 requires 8 frames, 512x512 and full samples=8')
    output_root = (args.output_root or PROJECT_ROOT/config['output_root']/('_dry_runs/smoke' if args.dry_run and args.mode=='smoke' else '_dry_runs/full' if args.dry_run else args.mode)).resolve()
    protected=[PROJECT_ROOT/'memory_scene_blender/outputs'/name for name in ('object_translation_v1','ego_object_x_factorial_v1')]
    if any(output_root==p.resolve() or p.resolve() in output_root.parents or output_root in p.resolve().parents for p in protected):
        raise ValueError('V2 output must not overlap a V1 output tree')
    if args.mode=='smoke' and (settings.get('context_count') != 1 or not set(settings['motion_family_ids']).issubset({'tx_d004','ty_d004'}) or 'tx_d004' not in settings['motion_family_ids']):
        raise ValueError('Rendered smoke is restricted to one context and tx_d004, optionally ty_d004')
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f'Refusing nonempty output: {output_root}; choose a fresh --output-root')
    ensure_dir(output_root)
    write_json(output_root/'.ego_object_factorial_v2_dataset',dict(dataset_name=config['dataset_name']))
    base_path = PROJECT_ROOT/config['base_config']
    base = load_config(base_path)
    contexts = plan['contexts'] if args.mode=='full' else [next(c for c in plan['contexts'] if c['extension'])]
    if args.context_ids:
        requested = list(dict.fromkeys(args.context_ids))
        known = {c['context_id'] for c in plan['contexts']}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(f'Unknown planned context IDs: {unknown}')
        contexts = [c for c in plan['contexts'] if c['context_id'] in requested]
        unused_overrides = sorted(set(scale_overrides) - set(requested))
        if unused_overrides:
            raise ValueError(f'Camera scale overrides are outside the requested contexts: {unused_overrides}')
    realized = []
    for context in contexts:
        families = context['motion_family_ids'] if args.mode=='full' or args.smoke_all_families else settings['motion_family_ids']
        realized.append(dict(context, generated_motion_family_ids=families))
    plan['realized_contexts'] = realized
    write_json(output_root/'context_plan.json', plan)
    write_json(output_root/'config_used.yaml',dict(config, mode_used=args.mode, dry_run=args.dry_run,effective_mode_settings=settings,smoke_all_families=args.smoke_all_families))
    source_paths = sorted(set([Path(__file__).resolve(), * (PROJECT_ROOT/'memory_scene_blender/ego_object_v2').glob('*.py'), *(PROJECT_ROOT/'memory_scene_blender/ego_object_factorial').glob('*.py'), *(PROJECT_ROOT/'memory_scene_blender/object_translation').glob('*.py')]))
    write_json(output_root/'provenance.json',dict(scene_semantics='novel explicit target/background scenes; no source-output reconstruction',
        config_sha256=sha256_file(args.config),base_config_path=config['base_config'],base_config_sha256=sha256_file(base_path),
        generator_source_sha256={str(p.relative_to(PROJECT_ROOT)):sha256_file(p) for p in source_paths},
        seed=config['seed'],blender_version=bpy.app.version_string,
        git_commit=subprocess.check_output(['git','-C',str(PROJECT_ROOT),'rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','-C',str(PROJECT_ROOT),'status','--short'],text=True)))
    scene_rows=[]; anchor_rows=[]; base_camera_rows=[]; group_rows=[]; sequence_rows=[]; frame_rows=[]; track_rows=[]
    selection_reports=[]
    width,image_height=settings['resolution']; alphas=linear_alphas(8)
    family_by_id={f['motion_family_id']:f for f in plan['motion_families']}
    for context in realized:
        scene_id_value=context['context_id']
        print('Selecting',scene_id_value,flush=True)
        report=dict(context_id=scene_id_value,candidates=[])
        selection_reports.append(report)
        try:
            bundle=build_scene_bundle_explicit(base,settings,**{k:context[k] for k in ('context_id','room_layout_index','static_layout_id','target_category','target_variant','context_seed')})
        except Exception as exc:
            report['error']=f'Explicit scene construction failed: {exc}'
            write_json(output_root/'selection_report.json',dict(ok=False,contexts=selection_reports))
            raise
        set_target_position(bundle.target_asset,[0,0])
        canonical=sample_target_surface_points(bundle.target_asset,config['tracks']['num_surface_points'],context['canonical_sampling_seed'])
        canonical_path=output_root/context['canonical_points_path']
        if canonical_path.exists():
            with np.load(canonical_path) as previous:
                if any(not np.array_equal(previous[k],v) for k,v in canonical.items()):
                    raise RuntimeError('Canonical target identity drift across backgrounds')
        else: save_npz(canonical_path,canonical,dict(target_id=context['target_id'],seed=context['canonical_sampling_seed']))
        # Even single-family rendered smoke selects a context valid for all extension families.
        selection_families=[family_by_id[f] for f in context['motion_family_ids']]
        try:
            context_scale = scale_overrides.get(scene_id_value, args.camera_distance_scale)
            selected,base_camera,selection=select_physical_context(
                bundle,canonical['xyz_object_local'],selection_families,settings,
                config['selection'],report,context_scale,base['camera'])
        except Exception:
            write_json(output_root/'selection_report.json',dict(ok=False,contexts=selection_reports))
            raise
        write_json(output_root/'selection_report.json',dict(ok=True,contexts=selection_reports))
        anchor_id=selected['anchor_id']; base_camera_id=base_camera['camera_id']; base_xy=np.array(selected['xy'])
        physical_context_id=f'{scene_id_value}__{anchor_id}__{base_camera_id}'
        identity={k:context[k] for k in ('context_id','background_id','target_id')}
        identity.update(physical_context_id=physical_context_id)
        split='unassigned'  # Context crossing does not imply a scientific train/test split.
        target_object_id=context['target_id']; target_category=context['target_category'];target_variant=context['target_variant']
        scene_row=dict(context,room_dimensions=bundle.layout['dimensions'],static_objects=bundle.static_metadata,
            target_asset_signature=bundle.target_asset.signature,split=split,target_object_id=target_object_id,target_object_numeric_id=20)
        scene_rows.append(scene_row)
        anchor_rows.append(dict(**identity,scene_id=scene_id_value,object_anchor_id=anchor_id,anchor_id=anchor_id,
            anchor_root_xy=base_xy.tolist(),anchor_object_to_world=np.array(bundle.target_asset.root.matrix_world).tolist(),sweeps=selected['sweeps']))
        base_camera_rows.append(dict(base_camera,**identity,base_camera_id=base_camera_id,camera_selection=selection))
        physical_render_cache={}
        for family_id in context['generated_motion_family_ids']:
            family=family_by_id[family_id];axis_world=np.array(family['axis_world']);delta_m=family['delta_m']
            conditions=motion_conditions(LEVELS,delta_m)
            current_group_id=group_id(physical_context_id,family_id)
            panel_membership=['core_confirmation'] if family_id=='tx_d004' else []
            if context['extension']: panel_membership.append('motion_extension')
            group_identity=dict(identity,motion_family_id=family_id,panel_membership=panel_membership)
            group_dir=ensure_dir(output_root/'contexts'/scene_id_value/family_id)
            group_rows.append(dict(**group_identity,group_id=current_group_id,scene_id=scene_id_value,
                object_anchor_id=anchor_id,base_camera_id=base_camera_id,canonical_points_path=context['canonical_points_path'],
                anchor_root_xy=base_xy.tolist(),base_camera=base_camera,delta_m=delta_m,motion_axis_world=axis_world.tolist(),
                levels=LEVELS,num_frames=8,condition_count=25,anchor_sweep=selected['sweeps'][family_id],camera_selection=selection))
            for condition in conditions:
                ego_level = int(condition["ego_level"])
                object_level = int(condition["object_level"])
                current_sequence_id = sequence_id(current_group_id, ego_level, object_level)
                sequence_dir = ensure_dir(group_dir / "sequences" / current_sequence_id)
                tracks_path = sequence_dir / "tracks.npz"
                sequence_metadata_path = sequence_dir / "sequence_metadata.json"
                sequence_row: Dict[str, Any] = {
                    **group_identity,
                    "sequence_id": current_sequence_id,
                    "group_id": current_group_id,
                    "scene_id": scene_id_value,
                    "object_anchor_id": anchor_id,
                    "anchor_id": anchor_id,
                    "base_camera_id": base_camera_id,
                    "ego_level": ego_level,
                    "object_level": object_level,
                    "ego_amplitude_m": float(condition["ego_amplitude_m"]),
                    "object_amplitude_m": float(condition["object_amplitude_m"]),
                    "relative_level": int(condition["relative_level"]),
                    "relative_amplitude_m": float(condition["relative_amplitude_m"]),
                    "num_frames": int(settings["num_frames"]),
                    "delta_m": delta_m,
                    "motion_axis_world": vector_to_list(axis_world),
                    "temporal_profile": "alpha_t=t/(T-1)",
                    "split": split,
                    "target_object_id": target_object_id,
                    "target_object_numeric_id": 20,
                    "target_category": target_category,
                    "target_variant": target_variant,
                    "target_asset_signature": bundle.target_asset.signature,
                    "is_compensated_motion": bool(condition["is_compensated_motion"]),
                    "is_static": bool(condition["is_static"]),
                    "sequence_dir": relative_path(sequence_dir, output_root),
                    "sequence_metadata_path": relative_path(sequence_metadata_path, output_root),
                    "tracks_path": relative_path(tracks_path, output_root),
                    "canonical_points_path": relative_path(canonical_path, output_root),
                    "rendered": not args.dry_run,
                }
                observations: List[Dict[str, np.ndarray]] = []
                sequence_frame_ids: List[str] = []
                for frame_index, alpha_t in enumerate(alphas):
                    actual_ego_amplitude = float(condition["ego_amplitude_m"] * alpha_t)
                    actual_object_amplitude = float(condition["object_amplitude_m"] * alpha_t)
                    actual_relative_amplitude = float(actual_object_amplitude - actual_ego_amplitude)
                    target_xy = base_xy + axis_world[:2] * actual_object_amplitude
                    set_target_position(bundle.target_asset, target_xy)
                    camera = translated_camera_payload(base_camera, actual_ego_amplitude, axis_world)
                    configure_factorial_camera(camera)
                    object_to_world = np.asarray(bundle.target_asset.root.matrix_world, dtype=np.float64)
                    observation = geometric_track_observation(
                        canonical["xyz_object_local"],
                        object_to_world,
                        camera,
                        settings["resolution"],
                        bundle.target_asset.parts,
                        edge_margin_px=0,
                        ray_tolerance_m=float(config["selection"]["ray_depth_tolerance_m"]),
                    )
                    frame_id = f"{current_sequence_id}__frame_{frame_index:03d}"
                    frame_dir = ensure_dir(sequence_dir / "frames" / f"frame_{frame_index:03d}")

                    if args.dry_run:
                        render_result: Dict[str, Any] = {
                            "render_source": "dry_run_geometry_only",
                            "cache_method": None,
                            "mask_stats": {
                                "mask_pixel_count": 0,
                                "mask_area_ratio": 0.0,
                                "bbox": None,
                                "truncated": False,
                                "edge_touch_ratio": 0.0,
                                "image_size": [width, image_height],
                            },
                            "visible_fraction": float(observation["visible"].mean()),
                            "depth_range_m": None,
                        }
                    else:
                        physical_key = tuple(np.round(np.r_[axis_world * actual_ego_amplitude, axis_world * actual_object_amplitude], 12))
                        cached_dir = physical_render_cache.get(physical_key)
                        render_result = render_physical_frame(
                            frame_dir,
                            bundle.target_asset,
                            camera,
                            int(settings["samples"]),
                            settings["resolution"],
                            False,
                            cached_dir,
                        )
                        observation = refine_observation_from_render(
                            observation, frame_dir, config["tracks"]
                        )
                        physical_render_cache.setdefault(physical_key, frame_dir)

                    center_world, center_camera, bbox_corners = target_centers(bundle.target_asset, camera)
                    paths = frame_paths(frame_dir, output_root)
                    frame_row: Dict[str, Any] = {
                        **group_identity,
                        "frame_id": frame_id,
                        "sequence_id": current_sequence_id,
                        "group_id": current_group_id,
                        "scene_id": scene_id_value,
                        "object_anchor_id": anchor_id,
                        "base_camera_id": base_camera_id,
                        "ego_level": ego_level,
                        "object_level": object_level,
                        "relative_level": int(condition["relative_level"]),
                        "frame_index": int(frame_index),
                        "alpha_t": float(alpha_t),
                        "actual_ego_amplitude_m": actual_ego_amplitude,
                        "actual_ego_translation_world_m": (axis_world * actual_ego_amplitude).tolist(),
                        "actual_object_amplitude_m": actual_object_amplitude,
                        "actual_object_translation_world_m": (axis_world * actual_object_amplitude).tolist(),
                        "actual_relative_amplitude_m": actual_relative_amplitude,
                        "actual_relative_translation_world_m": (axis_world * actual_relative_amplitude).tolist(),
                        "camera_to_world": camera["opencv_camera_to_world"],
                        "world_to_camera": camera["opencv_world_to_camera"],
                        "opencv_camera_to_world": camera["opencv_camera_to_world"],
                        "opencv_world_to_camera": camera["opencv_world_to_camera"],
                        "blender_camera_to_world": camera["blender_camera_to_world"],
                        "blender_world_to_camera": camera["blender_world_to_camera"],
                        "camera_rotation": camera["rotation_matrix"],
                        "camera_world_position": camera["position"],
                        "intrinsics_K": camera["intrinsics"]["K"],
                        "K": camera["intrinsics"]["K"],
                        "intrinsics": camera["intrinsics"],
                        "object_to_world": np_to_list(object_to_world),
                        "target_object_to_world": np_to_list(object_to_world),
                        "target_world_center": vector_to_list(center_world),
                        "target_camera_coordinate_center": vector_to_list(center_camera),
                        "target_center_world": vector_to_list(center_world),
                        "target_center_camera": vector_to_list(center_camera),
                        "target_bbox_corners_world": bbox_corners,
                        "motion_axis_world": vector_to_list(axis_world),
                        "image_size": [width, image_height],
                        "coordinate_convention": {
                            "world": "Blender world, metres, +Z up",
                            "camera": "OpenCV, +X right, +Y down, +Z forward",
                            "pixel": "top-left origin, u right, v down",
                        },
                        "render_convention": {
                            "camera_type": "perspective",
                            "sensor_fit": "HORIZONTAL",
                            "pixel_aspect": [1.0, 1.0],
                            "principal_point_shift": [0.0, 0.0],
                            **DEPTH_CONVENTION,
                            "instance_and_object_id": "identical color-coded PNG plus object_id_palette.json",
                            "semantic": "color-coded PNG plus semantic_palette.json",
                            "normal": "world normal mapped by n*0.5+0.5 into color-managed 8-bit PNG",
                            "albedo": "first material diffuse color rendered as emission",
                        },
                        "frame_dir": relative_path(frame_dir, output_root),
                        "frame_paths": paths,
                        **paths,
                        "rendered": not args.dry_run,
                        "render_source": render_result["render_source"],
                        "cache_method": render_result.get("cache_method"),
                        "target_mask_pixel_count": int(render_result["mask_stats"]["mask_pixel_count"]),
                        "target_mask_area_ratio": float(render_result["mask_stats"]["mask_area_ratio"]),
                        "target_bbox_2d": render_result["mask_stats"].get("bbox"),
                        "target_truncated": bool(render_result["mask_stats"].get("truncated", False)),
                        "target_edge_touch_ratio": float(
                            render_result["mask_stats"].get("edge_touch_ratio", 0.0)
                        ),
                        "render_estimated_visible_fraction": float(render_result["visible_fraction"]),
                        "depth_range_m": render_result.get("depth_range_m"),
                        "track_observation": observation_summary(observation),
                        "look_at_recomputed": False,
                    }
                    write_json(frame_dir / "frame_metadata.json", frame_row)
                    frame_rows.append(frame_row)
                    sequence_frame_ids.append(frame_id)
                    observations.append(observation)

                tracks_payload = stack_track_observations(canonical, observations)
                track_metadata = {
                    "track_schema_version": 1,
                    "sequence_id": current_sequence_id,
                    "group_id": current_group_id,
                    "point_identity": "stable target-root-local physical surface samples",
                    "depth_camera_z_m": "OpenCV axial +Z depth",
                    "range_to_camera_m": "Euclidean camera-ray range",
                    "depth_buffer_range_m": "Blender Z-pass ray range when rendered",
                    "visibility": (
                        "rendered target mask plus Z-pass range consistency"
                        if not args.dry_run
                        else "exact-point Blender ray cast (geometry-only preflight)"
                    ),
                }
                save_npz(tracks_path, tracks_payload, track_metadata)
                sequence_row["frame_ids"] = sequence_frame_ids
                sequence_row["track_point_count"] = int(len(canonical["point_id"]))
                sequence_row["track_visibility_source"] = (
                    "rendered_depth_and_target_mask" if not args.dry_run else "geometry_raycast"
                )
                write_json(sequence_metadata_path, sequence_row)
                sequence_rows.append(sequence_row)
                track_rows.append(
                    {
                        "track_set_id": f"{current_sequence_id}__tracks",
                        "sequence_id": current_sequence_id,
                        "group_id": current_group_id,
                        "scene_id": scene_id_value,
                        "object_anchor_id": anchor_id,
                        "base_camera_id": base_camera_id,
                        "tracks_path": relative_path(tracks_path, output_root),
                        "canonical_points_path": relative_path(canonical_path, output_root),
                        "point_count": int(len(canonical["point_id"])),
                        "num_frames": int(settings["num_frames"]),
                        "visibility_source": sequence_row["track_visibility_source"],
                        "coordinate_convention": track_metadata,
                    }
                )

    manifests=dict(scenes=scene_rows,contexts=scene_rows,targets=[t for t in plan['targets'] if any(c['target_id']==t['target_id'] for c in realized)],
        backgrounds=[b for b in plan['backgrounds'] if any(c['background_id']==b['background_id'] for c in realized)],
        motion_families=plan['motion_families'],anchors=anchor_rows,base_cameras=base_camera_rows,groups=group_rows,
        sequences=sequence_rows,frames=frame_rows,track_sets=track_rows,matched_relative_groups=matched_relative_groups(sequence_rows))
    for name,rows in manifests.items(): write_jsonl(output_root/'manifests'/f'{name}.jsonl',rows)
    write_json(output_root/'manifests/splits.json',dict(policy='unassigned; define task-specific splits before fitting',unassigned_scenes=[c['context_id'] for c in realized]))
    counts=dict(contexts=len(scene_rows),groups=len(group_rows),sequences=len(sequence_rows),frames=len(frame_rows))
    expected=(dict(contexts=len(realized),groups=sum(len(c['generated_motion_family_ids']) for c in realized),
                   sequences=25*sum(len(c['generated_motion_family_ids']) for c in realized),
                   frames=200*sum(len(c['generated_motion_family_ids']) for c in realized))
              if args.context_ids else FULL_COUNTS if args.mode=='full' else
              dict(contexts=1,groups=len(realized[0]['generated_motion_family_ids']),sequences=25*len(realized[0]['generated_motion_family_ids']),frames=200*len(realized[0]['generated_motion_family_ids'])))
    if counts != expected: raise RuntimeError(f'Incomplete plan: {counts} != {expected}')
    summary=dict(dataset_name=config['dataset_name'],mode=args.mode,dry_run=args.dry_run,counts=counts,complete=True,
                 full_dataset_was_started=args.mode=='full' and not args.dry_run,
                 subset_context_ids=args.context_ids or [],camera_distance_scale=args.camera_distance_scale,
                 context_camera_distance_scales=scale_overrides)
    write_json(output_root/'dataset_summary.json',summary)
    from memory_scene_blender.ego_object_v2.reports import write_reports
    write_reports(output_root,plan,manifests)
    (output_root/'README.md').write_text('V2 novel procedural contexts. tx_d004 is the direct E1 replication panel.\nSee context_plan.json, provenance.json and reports/. Auxiliary X/Y scales must be analyzed separately.\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    generate_dataset(parse_args(argv))
