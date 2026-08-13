"""Ground-truth label, visibility, and validation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .manifest_utils import np_to_list, transform_point, vector_to_list


def aabb_corners(min_xyz: Sequence[float], max_xyz: Sequence[float]) -> List[List[float]]:
    xmin, ymin, zmin = [float(v) for v in min_xyz]
    xmax, ymax, zmax = [float(v) for v in max_xyz]
    return [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmin, ymax, zmin],
        [xmax, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmin, ymax, zmax],
        [xmax, ymax, zmax],
    ]


def bounds_from_corners(corners: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(corners, dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)


def xy_aabb_from_corners(corners: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    arr = np.asarray(corners, dtype=np.float64)
    return float(arr[:, 0].min()), float(arr[:, 0].max()), float(arr[:, 1].min()), float(arr[:, 1].max())


def xy_aabb_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return not (
        ax1 + margin <= bx0
        or bx1 + margin <= ax0
        or ay1 + margin <= by0
        or by1 + margin <= ay0
    )


def room_bounds_xy(room_dimensions: Sequence[float], wall_margin: float) -> Tuple[float, float, float, float]:
    width, depth, _height = [float(v) for v in room_dimensions]
    return (
        -width * 0.5 + wall_margin,
        width * 0.5 - wall_margin,
        -depth * 0.5 + wall_margin,
        depth * 0.5 - wall_margin,
    )


def corners_inside_room(corners: Sequence[Sequence[float]], room_dimensions: Sequence[float], wall_margin: float) -> bool:
    xmin, xmax, ymin, ymax = room_bounds_xy(room_dimensions, wall_margin)
    arr = np.asarray(corners, dtype=np.float64)
    return bool(
        np.all(arr[:, 0] >= xmin)
        and np.all(arr[:, 0] <= xmax)
        and np.all(arr[:, 1] >= ymin)
        and np.all(arr[:, 1] <= ymax)
    )


def validate_no_collision(
    target_corners: Sequence[Sequence[float]],
    static_objects: Sequence[Mapping[str, Any]],
    clearance: float,
) -> bool:
    target_aabb = xy_aabb_from_corners(target_corners)
    for obj in static_objects:
        if obj.get("is_room_shell"):
            continue
        static_corners = obj.get("bbox_corners_world")
        if not static_corners:
            continue
        if xy_aabb_overlap(target_aabb, xy_aabb_from_corners(static_corners), margin=clearance):
            return False
    return True


def mask_stats_from_array(mask: np.ndarray, edge_margin_px: int = 0) -> Dict[str, Any]:
    mask = np.asarray(mask).astype(bool)
    height, width = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())
    if area == 0:
        return {
            "mask_pixel_count": 0,
            "mask_area_ratio": 0.0,
            "bbox": None,
            "truncated": False,
            "edge_touch_ratio": 0.0,
            "center_2d": None,
            "image_size": [int(width), int(height)],
        }

    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    touches = (
        (xs <= edge_margin_px).sum()
        + (xs >= width - 1 - edge_margin_px).sum()
        + (ys <= edge_margin_px).sum()
        + (ys >= height - 1 - edge_margin_px).sum()
    )
    edge_touch_ratio = float(touches / max(1, area))
    return {
        "mask_pixel_count": area,
        "mask_area_ratio": float(area / max(1, width * height)),
        "bbox": bbox,
        "truncated": bool(
            bbox[0] <= edge_margin_px
            or bbox[1] <= edge_margin_px
            or bbox[2] >= width - 1 - edge_margin_px
            or bbox[3] >= height - 1 - edge_margin_px
        ),
        "edge_touch_ratio": edge_touch_ratio,
        "center_2d": [float(xs.mean()), float(ys.mean())],
        "image_size": [int(width), int(height)],
    }


def load_mask_with_pillow(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 127


def mask_stats_from_png(path: Path, edge_margin_px: int = 0) -> Dict[str, Any]:
    return mask_stats_from_array(load_mask_with_pillow(path), edge_margin_px=edge_margin_px)


def image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def assert_valid_matrix(matrix: Sequence[Sequence[float]], name: str) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise AssertionError(f"{name} must be 4x4, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise AssertionError(f"{name} contains NaN or Inf")
    if abs(float(arr[3, 3]) - 1.0) > 1e-7:
        raise AssertionError(f"{name} has invalid homogeneous bottom-right value")


def state_label_payload(
    scene_id: str,
    target_object_id: str,
    target_category: str,
    state_id: str,
    object_transform_world: Sequence[Sequence[float]],
    bbox_corners_world: Sequence[Sequence[float]],
    reference_camera: Mapping[str, Any],
    scene_seed: int,
    position_index: Sequence[int],
    target_asset_signature: Mapping[str, Any],
) -> Dict[str, Any]:
    min_xyz, max_xyz = bounds_from_corners(bbox_corners_world)
    center_world = (min_xyz + max_xyz) * 0.5
    bottom_center_world = np.array([center_world[0], center_world[1], min_xyz[2]], dtype=np.float64)
    center_ref = transform_point(reference_camera["opencv_world_to_camera"], center_world)
    transform = np.asarray(object_transform_world, dtype=np.float64)
    return {
        "scene_id": scene_id,
        "target_object_id": target_object_id,
        "target_category": target_category,
        "state_id": state_id,
        "object_center_world": vector_to_list(center_world),
        "object_bottom_center_world": vector_to_list(bottom_center_world),
        "object_center_ref_camera": vector_to_list(center_ref),
        "object_transform_world": np_to_list(transform),
        "bbox_corners_world": [vector_to_list(corner) for corner in bbox_corners_world],
        "object_local_scale": vector_to_list([transform[0, 0], transform[1, 1], transform[2, 2]]),
        "object_orientation_euler_xyz": vector_to_list([0.0, 0.0, 0.0]),
        "orientation_fixed": True,
        "scene_seed": int(scene_seed),
        "position_index": [int(position_index[0]), int(position_index[1])],
        "target_asset_signature": dict(target_asset_signature),
        "ground_contact_z": float(min_xyz[2]),
    }


def frame_label_payload(
    scene_id: str,
    target_object_id: str,
    target_category: str,
    state: Mapping[str, Any],
    camera: Mapping[str, Any],
    frame_dir: Path,
    mask_stats: Mapping[str, Any],
    visible_fraction: float,
    depth_range_m: Optional[Sequence[float]],
    min_mask_area_ratio: float,
) -> Dict[str, Any]:
    center_current = transform_point(camera["opencv_world_to_camera"], state["object_center_world"])
    return {
        "frame_id": f"{scene_id}_{state['state_id']}_{camera['camera_id']}",
        "scene_id": scene_id,
        "target_object_id": target_object_id,
        "target_category": target_category,
        "state_id": state["state_id"],
        "camera_id": camera["camera_id"],
        "frame_dir": str(frame_dir),
        "rgb": str(frame_dir / "rgb.png"),
        "depth": str(frame_dir / "depth.exr"),
        "depth_npy": str(frame_dir / "depth.npy"),
        "normal": str(frame_dir / "normal.png"),
        "albedo": str(frame_dir / "albedo.png"),
        "target_mask": str(frame_dir / "target_mask.png"),
        "instance": str(frame_dir / "instance.png"),
        "semantic": str(frame_dir / "semantic.png"),
        "object_id": str(frame_dir / "object_id.png"),
        "frame_metadata": str(frame_dir / "frame_metadata.json"),
        "object_center_world": state["object_center_world"],
        "object_center_ref_camera": state["object_center_ref_camera"],
        "object_center_current_camera": vector_to_list(center_current),
        "target_mask_pixel_count": int(mask_stats["mask_pixel_count"]),
        "target_mask_area_ratio": float(mask_stats["mask_area_ratio"]),
        "target_bbox_2d": mask_stats["bbox"],
        "target_truncated": bool(mask_stats["truncated"]),
        "target_edge_touch_ratio": float(mask_stats["edge_touch_ratio"]),
        "visible_fraction": float(visible_fraction),
        "image_size": mask_stats.get("image_size"),
        "visibility_ok": bool(mask_stats["mask_area_ratio"] >= min_mask_area_ratio and mask_stats["mask_pixel_count"] > 0),
        "depth_range_m": list(depth_range_m) if depth_range_m is not None else None,
    }


def blender_image_mask_stats(path: Path, edge_margin_px: int = 0) -> Dict[str, Any]:
    """Load a PNG through Blender and return mask stats.

    Blender's Python bundle in this environment does not include Pillow.  This
    helper keeps generation self-contained; validation uses Pillow afterwards.
    """
    import bpy  # type: ignore

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = [int(value) for value in image.size]
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, image.channels)
        # Blender image buffers are bottom-up for file images.  Flip so bboxes
        # use conventional top-left image coordinates.
        pixels = np.flipud(pixels)
        mask = pixels[:, :, 0] > 0.5
        return mask_stats_from_array(mask, edge_margin_px=edge_margin_px)
    finally:
        bpy.data.images.remove(image)


def blender_depth_range_from_exr(path: Path) -> Optional[List[float]]:
    import bpy  # type: ignore

    if not path.exists():
        return None
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = np.asarray(image.pixels[:], dtype=np.float32)
        if pixels.size == 0:
            return None
        finite = pixels[np.isfinite(pixels)]
        finite = finite[(finite > 0.0) & (finite < 1.0e6)]
        if finite.size == 0:
            return None
        return [float(finite.min()), float(finite.max())]
    finally:
        bpy.data.images.remove(image)


def blender_target_visibility_fraction(target_parts: Sequence[Any]) -> float:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
    from bpy_extras.object_utils import world_to_camera_view  # type: ignore

    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        return 0.0

    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: List[Any] = []
    for obj in target_parts:
        if obj.type != "MESH":
            continue
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        center = sum(bbox, Vector((0.0, 0.0, 0.0))) / len(bbox)
        points.extend([center, *bbox])
        vertices = list(obj.data.vertices)
        if vertices:
            step = max(1, len(vertices) // 32)
            points.extend(obj.matrix_world @ vertex.co for vertex in vertices[::step])

    inside = 0
    visible = 0
    for point in points:
        ndc = world_to_camera_view(scene, camera, point)
        if not (0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0 and ndc.z > 0.0):
            continue
        inside += 1
        origin = camera.location
        direction = point - origin
        distance = direction.length
        if distance <= 1e-7:
            continue
        direction.normalize()
        hit, _loc, _normal, _face_index, hit_obj, _matrix = scene.ray_cast(
            depsgraph, origin, direction, distance=distance + 0.02
        )
        if hit and hit_obj in target_parts:
            visible += 1
    return float(visible / inside) if inside else 0.0


def json_has_no_nan(payload: Any) -> bool:
    if isinstance(payload, float):
        return math.isfinite(payload)
    if isinstance(payload, dict):
        return all(json_has_no_nan(value) for value in payload.values())
    if isinstance(payload, list):
        return all(json_has_no_nan(value) for value in payload)
    return True
