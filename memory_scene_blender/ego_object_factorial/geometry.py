"""Blender-side geometry, surface sampling, and deterministic selection.

The functions in this module never call ``look_at``.  Candidate cameras come
from the source spatial camera bank, and temporal camera poses are produced by
pure world-space translation in :mod:`protocol`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from memory_scene_blender.object_translation.assets import asset_world_bbox
from memory_scene_blender.object_translation.label_utils import (
    aabb_corners,
    corners_inside_room,
    validate_no_collision,
)
from memory_scene_blender.object_translation.manifest_utils import np_to_list, vector_to_list
from memory_scene_blender.object_translation.scene_builder import (
    configure_camera as configure_source_camera,
    set_target_position,
)

from .protocol import motion_conditions, project_opencv, transform_points, translated_camera_payload


def configure_factorial_camera(camera_payload: Mapping[str, Any]) -> None:
    """Assign pose and explicitly pin the physical camera/intrinsic controls."""
    import bpy  # type: ignore

    configure_source_camera(camera_payload)
    camera = bpy.context.scene.camera
    if camera is None:
        raise RuntimeError("Scene has no camera")
    intrinsics = camera_payload["intrinsics"]
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = float(intrinsics.get("sensor_width_mm", 36.0))
    camera.data.shift_x = 0.0
    camera.data.shift_y = 0.0
    camera.data.dof.use_dof = False
    camera.data.angle = np.deg2rad(float(camera_payload["fov_degrees"]))
    camera.data.clip_start = float(camera_payload["clip_start"])
    camera.data.clip_end = float(camera_payload["clip_end"])
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = 1.0
    bpy.context.view_layer.update()


def _triangles_in_target_local(target_asset: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return triangle vertices, part indices, and polygon indices in root-local coordinates."""
    from mathutils import Vector  # type: ignore

    root_inverse = target_asset.root.matrix_world.inverted()
    triangles: List[np.ndarray] = []
    part_indices: List[int] = []
    polygon_indices: List[int] = []
    for part_index, obj in enumerate(target_asset.parts):
        if obj.type != "MESH":
            continue
        to_root = root_inverse @ obj.matrix_world
        for polygon in obj.data.polygons:
            vertex_indices = list(polygon.vertices)
            if len(vertex_indices) < 3:
                continue
            first = to_root @ Vector(obj.data.vertices[vertex_indices[0]].co)
            for offset in range(1, len(vertex_indices) - 1):
                second = to_root @ Vector(obj.data.vertices[vertex_indices[offset]].co)
                third = to_root @ Vector(obj.data.vertices[vertex_indices[offset + 1]].co)
                triangle = np.asarray([first[:], second[:], third[:]], dtype=np.float64)
                area = 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))
                if area <= 1.0e-12:
                    continue
                triangles.append(triangle)
                part_indices.append(part_index)
                polygon_indices.append(int(polygon.index))
    if not triangles:
        raise RuntimeError("Target asset has no non-degenerate mesh triangles")
    return (
        np.stack(triangles, axis=0),
        np.asarray(part_indices, dtype=np.int32),
        np.asarray(polygon_indices, dtype=np.int32),
    )


def sample_target_surface_points(target_asset: Any, count: int, seed: int) -> Dict[str, np.ndarray]:
    """Area-sample stable canonical points on the target mesh surface."""
    if int(count) <= 0:
        raise ValueError("surface point count must be positive")
    triangles, triangle_parts, polygon_indices = _triangles_in_target_local(target_asset)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    probabilities = areas / areas.sum()
    rng = np.random.default_rng(int(seed))
    choices = rng.choice(len(triangles), size=int(count), replace=True, p=probabilities)

    # sqrt transform gives a uniform density over each selected triangle.
    r1 = np.sqrt(rng.random(int(count)))
    r2 = rng.random(int(count))
    selected = triangles[choices]
    points = (
        (1.0 - r1)[:, None] * selected[:, 0]
        + (r1 * (1.0 - r2))[:, None] * selected[:, 1]
        + (r1 * r2)[:, None] * selected[:, 2]
    )
    normals = cross[choices]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return {
        "point_id": np.arange(int(count), dtype=np.int32),
        "xyz_object_local": points.astype(np.float64),
        "normal_object_local": normals.astype(np.float64),
        "part_index": triangle_parts[choices].astype(np.int32),
        "polygon_index": polygon_indices[choices].astype(np.int32),
        "sampling_triangle_index": choices.astype(np.int32),
    }


def anchor_root_xy(source_state: Mapping[str, Any]) -> np.ndarray:
    transform = np.asarray(source_state["object_transform_world"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{source_state.get('state_id')}: object transform is not 4x4")
    return transform[:2, 3].copy()


def anchor_id_from_state_id(state_id: str) -> str:
    suffix = str(state_id).split("_", 1)[-1]
    return f"anchor_{suffix}"


def validate_anchor_sweep(
    target_asset: Any,
    source_state: Mapping[str, Any],
    room_dimensions: Sequence[float],
    static_metadata: Sequence[Mapping[str, Any]],
    levels: Sequence[int],
    delta_m: float,
    axis_world: Sequence[float],
    wall_margin_m: float,
    static_clearance_m: float,
) -> Dict[str, Any]:
    """Check every object amplitude, including the complete +/-2delta sweep."""
    base_xy = anchor_root_xy(source_state)
    axis = np.asarray(axis_world, dtype=np.float64)
    checks: List[Dict[str, Any]] = []
    all_sweep_corners: List[List[float]] = []
    all_inside = True
    all_collision_free = True
    for level in [int(value) for value in levels]:
        offset = axis * float(level * delta_m)
        xy = base_xy + offset[:2]
        set_target_position(target_asset, xy)
        corners = asset_world_bbox(target_asset)
        all_sweep_corners.extend([vector_to_list(corner) for corner in corners])
        inside = corners_inside_room(corners, room_dimensions, float(wall_margin_m))
        collision_free = validate_no_collision(corners, static_metadata, float(static_clearance_m))
        checks.append(
            {
                "object_level": level,
                "object_dx_m": float(level * delta_m),
                "root_xy": vector_to_list(xy),
                "bbox_corners_world": [vector_to_list(corner) for corner in corners],
                "inside_room": bool(inside),
                "collision_free": bool(collision_free),
            }
        )
        all_inside = all_inside and bool(inside)
        all_collision_free = all_collision_free and bool(collision_free)
    sweep_array = np.asarray(all_sweep_corners, dtype=np.float64)
    swept_bbox = aabb_corners(sweep_array.min(axis=0), sweep_array.max(axis=0))
    swept_collision_free = validate_no_collision(
        swept_bbox, static_metadata, float(static_clearance_m)
    )
    swept_inside_room = corners_inside_room(
        swept_bbox, room_dimensions, float(wall_margin_m)
    )
    set_target_position(target_asset, base_xy)
    return {
        "valid": bool(
            all_inside
            and all_collision_free
            and swept_inside_room
            and swept_collision_free
        ),
        "inside_room_for_full_sweep": bool(all_inside and swept_inside_room),
        "collision_free_for_discrete_levels": bool(all_collision_free),
        "collision_free_for_full_swept_volume": bool(swept_collision_free),
        "swept_bbox_corners_world": swept_bbox,
        "checks": checks,
    }


def _ray_visible_points(
    points_world: np.ndarray,
    in_image: np.ndarray,
    camera_position: Sequence[float],
    target_parts: Sequence[Any],
    tolerance_m: float,
) -> np.ndarray:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_names = {obj.name for obj in target_parts}
    origin = Vector(tuple(float(value) for value in camera_position))
    visible = np.zeros(points_world.shape[0], dtype=np.bool_)
    for index in np.flatnonzero(in_image):
        point = Vector(tuple(float(value) for value in points_world[index]))
        direction = point - origin
        distance = float(direction.length)
        if distance <= 1.0e-8:
            continue
        direction.normalize()
        hit, location, _normal, _face_index, hit_object, _matrix = scene.ray_cast(
            depsgraph,
            origin,
            direction,
            distance=distance + float(tolerance_m),
        )
        if not hit or hit_object is None or hit_object.name not in target_names:
            continue
        hit_distance = float((location - origin).length)
        visible[index] = abs(hit_distance - distance) <= float(tolerance_m)
    return visible


def geometric_track_observation(
    canonical_points: np.ndarray,
    object_to_world: Sequence[Sequence[float]],
    camera_payload: Mapping[str, Any],
    image_size: Sequence[int],
    target_parts: Sequence[Any],
    edge_margin_px: int,
    ray_tolerance_m: float,
) -> Dict[str, np.ndarray]:
    """Project canonical points and estimate exact-point visibility by ray casting."""
    width, height = [int(value) for value in image_size]
    points_world = transform_points(object_to_world, canonical_points)
    points_camera, uv = project_opencv(
        points_world,
        camera_payload["opencv_world_to_camera"],
        camera_payload["intrinsics"]["K"],
    )
    in_front = points_camera[:, 2] > 0.0
    margin = int(edge_margin_px)
    in_image = (
        in_front
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= margin)
        & (uv[:, 0] < width - margin)
        & (uv[:, 1] >= margin)
        & (uv[:, 1] < height - margin)
    )
    visible = _ray_visible_points(
        points_world,
        in_image,
        camera_payload["position"],
        target_parts,
        float(ray_tolerance_m),
    )
    return {
        "xyz_world": points_world,
        "xyz_camera": points_camera,
        "projected_uv": uv,
        "depth_camera_z_m": points_camera[:, 2],
        "range_to_camera_m": np.linalg.norm(points_camera, axis=1),
        "in_front_of_camera": in_front,
        "in_image": in_image,
        "visible": visible,
    }


def _projected_bbox_margin(
    bbox_corners_world: Sequence[Sequence[float]],
    camera_payload: Mapping[str, Any],
    image_size: Sequence[int],
) -> float:
    width, height = [int(value) for value in image_size]
    camera_points, uv = project_opencv(
        np.asarray(bbox_corners_world, dtype=np.float64),
        camera_payload["opencv_world_to_camera"],
        camera_payload["intrinsics"]["K"],
    )
    if np.any(camera_points[:, 2] <= 0.0) or not np.isfinite(uv).all():
        # Keep persisted selection reports strict JSON (no -Infinity literal).
        return -1.0e12
    distances = np.concatenate(
        [uv[:, 0], (width - 1) - uv[:, 0], uv[:, 1], (height - 1) - uv[:, 1]]
    )
    return float(distances.min())


def evaluate_base_camera(
    target_asset: Any,
    canonical_points: np.ndarray,
    source_state: Mapping[str, Any],
    base_camera: Mapping[str, Any],
    source_frame: Optional[Mapping[str, Any]],
    levels: Sequence[int],
    delta_m: float,
    axis_world: Sequence[float],
    image_size: Sequence[int],
    selection_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate a bank camera over all 25 terminal intervention conditions."""
    base_xy = anchor_root_xy(source_state)
    condition_rows = motion_conditions(levels, delta_m)
    endpoint_stats: List[Dict[str, Any]] = []
    min_in_image = 1.0
    min_visible = 1.0
    min_bbox_margin = float("inf")
    observations_by_condition: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

    for condition in condition_rows:
        object_dx = float(condition["object_amplitude_m"])
        ego_dx = float(condition["ego_amplitude_m"])
        object_xy = base_xy + np.asarray(axis_world, dtype=np.float64)[:2] * object_dx
        set_target_position(target_asset, object_xy)
        camera = translated_camera_payload(base_camera, ego_dx, axis_world)
        configure_factorial_camera(camera)
        observation = geometric_track_observation(
            canonical_points,
            np.asarray(target_asset.root.matrix_world, dtype=np.float64),
            camera,
            image_size,
            target_asset.parts,
            int(selection_cfg["edge_margin_px"]),
            float(selection_cfg["ray_depth_tolerance_m"]),
        )
        observations_by_condition[
            (int(condition["ego_level"]), int(condition["object_level"]))
        ] = observation
        in_image_fraction = float(observation["in_image"].mean())
        visible_fraction = float(observation["visible"].mean())
        bbox_margin = _projected_bbox_margin(asset_world_bbox(target_asset), camera, image_size)
        endpoint_stats.append(
            {
                "ego_level": int(condition["ego_level"]),
                "object_level": int(condition["object_level"]),
                "in_image_fraction": in_image_fraction,
                "visible_surface_fraction": visible_fraction,
                "projected_bbox_margin_px": bbox_margin,
            }
        )
        min_in_image = min(min_in_image, in_image_fraction)
        min_visible = min(min_visible, visible_fraction)
        min_bbox_margin = min(min_bbox_margin, bbox_margin)

    set_target_position(target_asset, base_xy)
    configure_factorial_camera(base_camera)

    reasons: List[str] = []
    if source_frame is None:
        reasons.append("missing_source_frame_metadata")
        initial_mask_ratio = None
        initial_truncated = None
        initial_edge_touch_ratio = None
    else:
        initial_mask_ratio = float(source_frame.get("target_mask_area_ratio", 0.0))
        initial_truncated = bool(source_frame.get("target_truncated", False))
        initial_edge_touch_ratio = float(source_frame.get("target_edge_touch_ratio", 1.0))
        if initial_mask_ratio < float(selection_cfg["min_initial_mask_area_ratio"]):
            reasons.append("initial_mask_too_small")
        if initial_mask_ratio > float(selection_cfg["max_initial_mask_area_ratio"]):
            reasons.append("initial_mask_too_large_for_static_background")
        if bool(selection_cfg.get("reject_initial_truncation", True)) and initial_truncated:
            reasons.append("initial_target_truncated")
        if initial_edge_touch_ratio > float(selection_cfg["max_initial_edge_touch_ratio"]):
            reasons.append("initial_target_near_edge")
    if min_in_image < float(selection_cfg["min_projected_point_in_image_fraction"]):
        reasons.append("surface_points_leave_image")
    if min_visible < float(selection_cfg["min_visible_surface_fraction"]):
        reasons.append("insufficient_surface_visibility")
    if min_bbox_margin < float(selection_cfg["edge_margin_px"]):
        reasons.append("projected_bbox_violates_edge_margin")

    # Higher tuple values are better.  Camera ID is applied as a final stable
    # ascending tie-break by the caller, not encoded in this numerical score.
    score = [
        float(min_visible),
        float(min_in_image),
        float(min_bbox_margin),
        float(initial_mask_ratio or 0.0),
    ]

    static_observation = observations_by_condition[(0, 0)]
    local_displacement_checks: List[Dict[str, Any]] = []
    for ego_level, object_level in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        moved = observations_by_condition[(ego_level, object_level)]
        common = np.asarray(static_observation["visible"]) & np.asarray(moved["visible"])
        distances = np.linalg.norm(
            np.asarray(moved["projected_uv"])[common]
            - np.asarray(static_observation["projected_uv"])[common],
            axis=1,
        )
        local_displacement_checks.append(
            {
                "ego_level": ego_level,
                "object_level": object_level,
                "relative_level": object_level - ego_level,
                "common_visible_point_count": int(len(distances)),
                "mean_px": float(np.mean(distances)) if len(distances) else None,
                "median_px": float(np.median(distances)) if len(distances) else None,
                "p95_px": float(np.percentile(distances, 95)) if len(distances) else None,
                "max_px": float(np.max(distances)) if len(distances) else None,
            }
        )
    return {
        "camera_id": str(base_camera["camera_id"]),
        "valid": not reasons,
        "rejection_reasons": reasons,
        "initial_mask_area_ratio": initial_mask_ratio,
        "initial_target_truncated": initial_truncated,
        "initial_edge_touch_ratio": initial_edge_touch_ratio,
        "min_in_image_fraction": min_in_image,
        "min_visible_surface_fraction": min_visible,
        "min_projected_bbox_margin_px": min_bbox_margin,
        "score": score,
        "local_plus_minus_one_projected_displacement": local_displacement_checks,
        "endpoint_stats": endpoint_stats,
    }


def choose_anchors_and_cameras(
    target_asset: Any,
    canonical_points: np.ndarray,
    scene_id: str,
    source_states: Sequence[Mapping[str, Any]],
    base_cameras: Sequence[Mapping[str, Any]],
    source_frames: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    room_dimensions: Sequence[float],
    static_metadata: Sequence[Mapping[str, Any]],
    levels: Sequence[int],
    delta_m: float,
    axis_world: Sequence[float],
    image_size: Sequence[int],
    selection_cfg: Mapping[str, Any],
    anchors_needed: int,
    cameras_per_anchor: int,
    forced_anchor_id: Optional[str] = None,
    forced_camera_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select valid source anchors and cameras reproducibly, with full audit output."""
    candidates: List[Dict[str, Any]] = []
    selected_groups: List[Dict[str, Any]] = []
    camera_by_id = {str(camera["camera_id"]): camera for camera in base_cameras}
    if forced_camera_id is not None and forced_camera_id not in camera_by_id:
        raise ValueError(f"Unknown forced base camera {forced_camera_id!r} in {scene_id}")

    for source_state in sorted(source_states, key=lambda row: str(row["state_id"])):
        anchor_id = anchor_id_from_state_id(str(source_state["state_id"]))
        if forced_anchor_id is not None and anchor_id != forced_anchor_id and str(source_state["state_id"]) != forced_anchor_id:
            continue
        sweep = validate_anchor_sweep(
            target_asset,
            source_state,
            room_dimensions,
            static_metadata,
            levels,
            delta_m,
            axis_world,
            float(selection_cfg["wall_margin_m"]),
            float(selection_cfg["static_clearance_m"]),
        )
        camera_results: List[Dict[str, Any]] = []
        if sweep["valid"]:
            for camera in sorted(base_cameras, key=lambda row: str(row["camera_id"])):
                camera_id = str(camera["camera_id"])
                if forced_camera_id is not None and camera_id != forced_camera_id:
                    continue
                if bool(selection_cfg.get("source_primary_cameras_only", False)) and not camera.get("is_primary", False):
                    continue
                source_frame = source_frames.get((scene_id, str(source_state["state_id"]), camera_id))
                result = evaluate_base_camera(
                    target_asset,
                    canonical_points,
                    source_state,
                    camera,
                    source_frame,
                    levels,
                    delta_m,
                    axis_world,
                    image_size,
                    selection_cfg,
                )
                camera_results.append(result)

        valid_cameras = [row for row in camera_results if row["valid"]]
        valid_cameras.sort(
            key=lambda row: (
                -float(row["score"][0]),
                -float(row["score"][1]),
                -float(row["score"][2]),
                -float(row["score"][3]),
                str(row["camera_id"]),
            )
        )
        anchor_score = valid_cameras[0]["score"] if valid_cameras else [-1.0e12] * 4
        candidates.append(
            {
                "scene_id": scene_id,
                "anchor_id": anchor_id,
                "source_state_id": str(source_state["state_id"]),
                "source_position_index": list(source_state["position_index"]),
                "anchor_root_xy": vector_to_list(anchor_root_xy(source_state)),
                "anchor_sweep": sweep,
                "camera_candidates": camera_results,
                "valid_camera_count": len(valid_cameras),
                "valid": bool(sweep["valid"] and len(valid_cameras) >= int(cameras_per_anchor)),
                "score": anchor_score,
            }
        )

    valid_anchors = [row for row in candidates if row["valid"]]
    valid_anchors.sort(
        key=lambda row: (
            -float(row["score"][0]),
            -float(row["score"][1]),
            -float(row["score"][2]),
            -float(row["score"][3]),
            str(row["anchor_id"]),
        )
    )
    if len(valid_anchors) < int(anchors_needed):
        message = (
            f"{scene_id}: only {len(valid_anchors)} anchors passed the full sweep/camera checks; "
            f"need {anchors_needed}. Inspect selection_report.json and adjust the global protocol explicitly."
        )
        report = {
            "scene_id": scene_id,
            "selection_is_deterministic": True,
            "selection_ok": False,
            "error": message,
            "ranking": "min visible fraction, min in-image fraction, min bbox margin, initial mask area, then IDs",
            "candidate_anchor_count": len(candidates),
            "valid_anchor_count": len(valid_anchors),
            "requested_anchor_count": int(anchors_needed),
            "requested_base_cameras_per_anchor": int(cameras_per_anchor),
            "selected_groups": [],
            "anchor_candidates": candidates,
        }
        return [], report

    for anchor in valid_anchors[: int(anchors_needed)]:
        valid_cameras = [row for row in anchor["camera_candidates"] if row["valid"]]
        valid_cameras.sort(
            key=lambda row: (
                -float(row["score"][0]),
                -float(row["score"][1]),
                -float(row["score"][2]),
                -float(row["score"][3]),
                str(row["camera_id"]),
            )
        )
        for camera_result in valid_cameras[: int(cameras_per_anchor)]:
            selected_groups.append(
                {
                    "scene_id": scene_id,
                    "anchor_id": anchor["anchor_id"],
                    "source_state_id": anchor["source_state_id"],
                    "source_position_index": anchor["source_position_index"],
                    "anchor_root_xy": anchor["anchor_root_xy"],
                    "base_camera_id": camera_result["camera_id"],
                    "anchor_sweep": anchor["anchor_sweep"],
                    "camera_selection": camera_result,
                }
            )

    report = {
        "scene_id": scene_id,
        "selection_is_deterministic": True,
        "selection_ok": True,
        "error": None,
        "ranking": "min visible fraction, min in-image fraction, min bbox margin, initial mask area, then IDs",
        "candidate_anchor_count": len(candidates),
        "valid_anchor_count": len(valid_anchors),
        "requested_anchor_count": int(anchors_needed),
        "requested_base_cameras_per_anchor": int(cameras_per_anchor),
        "selected_groups": selected_groups,
        "anchor_candidates": candidates,
    }
    return selected_groups, report


def load_mask_blender(path: Any) -> np.ndarray:
    import bpy  # type: ignore

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = [int(value) for value in image.size]
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, image.channels)
        return np.flipud(pixels[:, :, 0]) > 0.5
    finally:
        bpy.data.images.remove(image)


def depth_buffer_track_visibility(
    observation: Mapping[str, np.ndarray],
    depth_buffer: np.ndarray,
    target_mask: np.ndarray,
    abs_tolerance_m: float,
    rel_tolerance: float,
    pixel_radius: int,
) -> Dict[str, np.ndarray]:
    """Refine point visibility using the rendered target mask and Z buffer.

    Blender's Z pass stores distance along the camera ray.  Therefore the
    comparison uses ``range_to_camera_m``, while ``depth_camera_z_m`` remains
    available separately as the OpenCV axial depth requested by downstream
    geometry code.
    """
    depth = np.asarray(depth_buffer, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=np.bool_)
    if depth.shape != mask.shape:
        raise ValueError(f"depth/mask shape mismatch: {depth.shape} versus {mask.shape}")
    uv = np.asarray(observation["projected_uv"], dtype=np.float64)
    in_image = np.asarray(observation["in_image"], dtype=np.bool_)
    ranges = np.asarray(observation["range_to_camera_m"], dtype=np.float64)
    visible = np.zeros(len(uv), dtype=np.bool_)
    sampled_depth = np.full(len(uv), np.nan, dtype=np.float64)
    depth_error = np.full(len(uv), np.nan, dtype=np.float64)
    height, width = depth.shape
    radius = int(pixel_radius)
    for index in np.flatnonzero(in_image):
        center_x = int(round(float(uv[index, 0])))
        center_y = int(round(float(uv[index, 1])))
        best_error = float("inf")
        best_depth = float("nan")
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                value = float(depth[y, x])
                if not mask[y, x] or not np.isfinite(value) or value <= 0.0:
                    continue
                error = abs(value - float(ranges[index]))
                if error < best_error:
                    best_error = error
                    best_depth = value
        if np.isfinite(best_depth):
            tolerance = max(float(abs_tolerance_m), float(rel_tolerance) * float(ranges[index]))
            sampled_depth[index] = best_depth
            depth_error[index] = best_error
            visible[index] = best_error <= tolerance
    result = {key: np.asarray(value).copy() for key, value in observation.items()}
    result["depth_buffer_range_m"] = sampled_depth
    result["depth_buffer_error_m"] = depth_error
    result["visible"] = visible
    return result


def observation_summary(observation: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    count = int(len(observation["visible"]))
    return {
        "point_count": count,
        "in_front_count": int(np.asarray(observation["in_front_of_camera"]).sum()),
        "in_image_count": int(np.asarray(observation["in_image"]).sum()),
        "visible_count": int(np.asarray(observation["visible"]).sum()),
        "in_image_fraction": float(np.asarray(observation["in_image"]).mean()) if count else 0.0,
        "visible_fraction": float(np.asarray(observation["visible"]).mean()) if count else 0.0,
    }
