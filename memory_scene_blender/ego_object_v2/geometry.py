"""Explicit novel scenes and source-free, jointly matched family selection."""
import numpy as np
from memory_scene_blender.object_translation import scene_builder as sb
from memory_scene_blender.object_translation.camera_builder import camera_payload_from_matrix
from memory_scene_blender.ego_object_factorial.geometry import (
    geometric_track_observation, configure_factorial_camera, validate_anchor_sweep,
    _projected_bbox_margin)
from memory_scene_blender.ego_object_factorial.protocol import translated_camera_payload, project_opencv
from .protocol import LEVELS


def build_scene_bundle_explicit(config, mode_settings, *, context_id, room_layout_index,
                                static_layout_id, target_category, target_variant, context_seed):
    import bpy
    layout = dict(config['scene']['room_layouts'][room_layout_index], static_layout_id=static_layout_id)
    sb.reset_scene()
    sb.configure_render(config['render'], mode_settings)
    materials = sb.create_materials()
    room = sb.create_room_shell(layout, materials)
    static = sb.create_static_reference_assets(static_layout_id, layout['dimensions'], materials)
    target = sb.create_target_asset(target_category, target_variant, materials, object_id=sb.TARGET_OBJECT_ID)
    sb.setup_lighting(config['scene']['lighting_presets'][layout['lighting_preset']])
    bpy.context.view_layer.update()
    metadata = [sb.logical_asset_metadata(a, is_room_shell=True) for a in room]
    metadata += [sb.logical_asset_metadata(a, is_room_shell=False) for a in static]
    positions = sb.generate_positions(config, mode_settings, layout, target, metadata,
                                     np.random.default_rng(context_seed))
    raw = sb.generate_cameras(context_id, config['camera'], mode_settings)
    # The legacy camera bank shadows image height with camera height. Correct only V2 payloads.
    cameras = [dict(camera_payload_from_matrix(c['camera_id'], np.array(c['blender_camera_to_world']),
        *mode_settings['resolution'], config['camera'], c['look_at'], c['is_primary']), scene_id=context_id)
        for c in raw]
    return sb.SceneBundle(context_id, context_seed, layout, materials, room, static, target,
                          cameras, positions, metadata)


def evaluate_camera(bundle, canonical, xy, camera, families, settings, cfg):
    """Check every distinct temporal physical state, including intermediate occlusion."""
    seen = set()
    minimum = dict(in_image_fraction=1., visible_surface_fraction=1., bbox_margin_px=1e12)
    sb.set_target_position(bundle.target_asset, xy)
    _, uv = project_opencv(np.array(sb.asset_world_bbox(bundle.target_asset)),
                          camera['opencv_world_to_camera'], camera['intrinsics']['K'])
    if not np.isfinite(uv).all():
        return dict(valid=False, **minimum, tested_physical_states=0,
                    initial_projected_bbox_area_ratio=None, rejection_reason='initial_bbox_behind_camera')
    initial_ratio = float(np.prod(np.ptp(uv, axis=0))/np.prod(settings['resolution']))
    for family in families:
        axis = np.array(family['axis_world'])
        # Paired offsets must share alpha, unlike their unconstrained cross-product.
        for alpha in np.linspace(0,1,8):
            for e in LEVELS:
                for o in LEVELS:
                    ev, ov = axis*(e*family['delta_m']*alpha), axis*(o*family['delta_m']*alpha)
                    key = tuple(np.round(np.r_[ev,ov],12))
                    if key in seen: continue
                    seen.add(key)
                    sb.set_target_position(bundle.target_asset, np.array(xy)+ov[:2])
                    moved = translated_camera_payload(camera, e*family['delta_m']*alpha, axis)
                    configure_factorial_camera(moved)
                    obs = geometric_track_observation(canonical, np.array(bundle.target_asset.root.matrix_world),
                        moved, settings['resolution'], bundle.target_asset.parts, cfg['edge_margin_px'], cfg['ray_depth_tolerance_m'])
                    minimum['in_image_fraction'] = min(minimum['in_image_fraction'], float(obs['in_image'].mean()))
                    minimum['visible_surface_fraction'] = min(minimum['visible_surface_fraction'], float(obs['visible'].mean()))
                    minimum['bbox_margin_px'] = min(minimum['bbox_margin_px'], _projected_bbox_margin(
                        sb.asset_world_bbox(bundle.target_asset), moved, settings['resolution']))
                    if minimum['in_image_fraction'] < cfg['min_projected_point_in_image_fraction'] or minimum['visible_surface_fraction'] < cfg['min_visible_surface_fraction'] or minimum['bbox_margin_px'] < cfg['edge_margin_px']:
                        return dict(valid=False, **minimum, tested_physical_states=len(seen), initial_projected_bbox_area_ratio=initial_ratio)
    valid = cfg['min_initial_mask_area_ratio'] <= initial_ratio <= cfg['max_initial_mask_area_ratio']
    return dict(valid=valid, **minimum, tested_physical_states=len(seen), initial_projected_bbox_area_ratio=initial_ratio)


def scaled_camera_distance(camera, scale, camera_config, resolution):
    """Move a base camera toward its look-at point while preserving its rotation."""
    if not 0.0 < scale < 1.0:
        raise ValueError('camera distance scale must be strictly between 0 and 1')
    matrix = np.asarray(camera['blender_camera_to_world'], dtype=np.float64).copy()
    position = matrix[:3, 3]
    look_at = np.asarray(camera['look_at'], dtype=np.float64)
    matrix[:3, 3] = look_at + scale * (position - look_at)
    camera_id = f"{camera['camera_id']}__near_{int(round(scale * 1000)):03d}"
    result = camera_payload_from_matrix(
        camera_id, matrix, *resolution, camera_config, look_at.tolist(), camera['is_primary'])
    return dict(result, scene_id=camera['scene_id'], distance_scale=float(scale),
                source_camera_id=camera['camera_id'])


def select_physical_context(bundle, canonical, families, settings, cfg, report,
                            camera_distance_scale=None, camera_config=None):
    """First valid anchor/camera in stable ID order; never replace a planned context."""
    for pos in bundle.positions:
        sb.set_target_position(bundle.target_asset, pos['xy'])
        state = dict(object_transform_world=np.array(bundle.target_asset.root.matrix_world).tolist())
        sweeps = {f['motion_family_id']:validate_anchor_sweep(bundle.target_asset, state,
            bundle.layout['dimensions'], bundle.static_metadata, LEVELS, f['delta_m'], f['axis_world'],
            cfg['wall_margin_m'], cfg['static_clearance_m']) for f in families}
        candidate = dict(anchor_id=pos['state_id'].replace('state','anchor'), xy=pos['xy'], sweeps=sweeps, cameras=[])
        report['candidates'].append(candidate)
        if not all(s['valid'] for s in sweeps.values()): continue
        cameras = bundle.cameras
        if camera_distance_scale is not None:
            if camera_config is None:
                raise ValueError('camera_config is required for distance-scaled selection')
            cameras = [scaled_camera_distance(c, camera_distance_scale, camera_config,
                                              settings['resolution']) for c in bundle.cameras]
        for camera in cameras:
            result = evaluate_camera(bundle, canonical, pos['xy'], camera, families, settings, cfg)
            candidate['cameras'].append(dict(camera_id=camera['camera_id'], **result))
            if result['valid']:
                sb.set_target_position(bundle.target_asset, pos['xy'])
                report['selected'] = dict(anchor_id=candidate['anchor_id'], base_camera_id=camera['camera_id'])
                return candidate, camera, result
    report['error'] = 'No anchor/base-camera passes all required families; planned context was not replaced.'
    raise RuntimeError(report['error'])
