"""Pure protocol math for the ego/object world-X factorial experiment.

This module intentionally has no Blender dependency.  Both the generator and
the post-render validator use the same definitions, while the validator also
recomputes the important equalities independently from saved matrices.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from memory_scene_blender.object_translation.camera_builder import (
    BLENDER_CAMERA_TO_OPENCV,
    camera_payload_from_matrix,
)
from memory_scene_blender.object_translation.manifest_utils import np_to_list, vector_to_list


MOTION_AXIS_WORLD = np.array([1.0, 0.0, 0.0], dtype=np.float64)


def camera_from_source_extrinsics(
    source_camera: Mapping[str, Any],
    camera_config: Mapping[str, Any],
    resolution: Sequence[int],
) -> Dict[str, Any]:
    """Reuse a bank extrinsic while recomputing K for the real image size.

    This explicit reconstruction is required because historical
    ``object_translation_v1`` camera rows contain an invalid image height/K.
    """
    width, image_height = [int(value) for value in resolution]
    if width <= 0 or image_height <= 0:
        raise ValueError("resolution must contain positive width and height")
    payload = camera_payload_from_matrix(
        str(source_camera["camera_id"]),
        np.asarray(source_camera["blender_camera_to_world"], dtype=np.float64),
        width,
        image_height,
        camera_config,
        source_camera.get("look_at", camera_config["look_at"]),
        bool(source_camera.get("is_primary", False)),
    )
    payload["scene_id"] = str(source_camera["scene_id"])
    payload["azimuth_degrees"] = float(source_camera.get("azimuth_degrees", 0.0))
    payload["source_camera_manifest_intrinsics"] = source_camera.get("intrinsics")
    payload["source_intrinsics_reused"] = False
    payload["intrinsics_recomputed_for_resolution"] = [width, image_height]
    return payload


def linear_alphas(num_frames: int) -> np.ndarray:
    """Return alpha_t=t/(T-1), including exactly zero and one."""
    if int(num_frames) < 2:
        raise ValueError("num_frames must be at least 2")
    return np.linspace(0.0, 1.0, int(num_frames), dtype=np.float64)


def motion_conditions(levels: Sequence[int], delta_m: float) -> List[Dict[str, Any]]:
    """Return the Cartesian ego/object intervention grid in stable order."""
    unique_levels = [int(level) for level in levels]
    if len(set(unique_levels)) != len(unique_levels):
        raise ValueError("motion levels must be unique")
    if 0 not in unique_levels:
        raise ValueError("motion levels must include zero")
    if float(delta_m) <= 0.0:
        raise ValueError("delta_m must be positive")

    rows: List[Dict[str, Any]] = []
    for ego_level in unique_levels:
        for object_level in unique_levels:
            relative_level = object_level - ego_level
            rows.append(
                {
                    "ego_level": ego_level,
                    "object_level": object_level,
                    "ego_amplitude_m": float(ego_level * delta_m),
                    "object_amplitude_m": float(object_level * delta_m),
                    "relative_level": int(relative_level),
                    "relative_amplitude_m": float(relative_level * delta_m),
                    "is_compensated_motion": bool(ego_level == object_level and ego_level != 0),
                    "is_static": bool(ego_level == 0 and object_level == 0),
                }
            )
    return rows


def level_token(level: int) -> str:
    value = int(level)
    if value < 0:
        return f"m{abs(value)}"
    if value > 0:
        return f"p{value}"
    return "z0"


def condition_key(ego_level: int, object_level: int) -> str:
    return f"e_{level_token(ego_level)}__o_{level_token(object_level)}"


def sequence_id(
    scene_id: str,
    anchor_id: str,
    base_camera_id: str,
    ego_level: int,
    object_level: int,
) -> str:
    return (
        f"{scene_id}__{anchor_id}__{base_camera_id}__"
        f"{condition_key(ego_level, object_level)}"
    )


def group_id(scene_id: str, anchor_id: str, base_camera_id: str) -> str:
    return f"{scene_id}__{anchor_id}__{base_camera_id}"


def build_matched_relative_groups(sequences: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Group sequences by physical context and observable relative level."""
    grouped: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        key = (
            str(sequence["scene_id"]),
            str(sequence["object_anchor_id"]),
            str(sequence["base_camera_id"]),
            int(sequence["relative_level"]),
        )
        grouped[key].append(
            {
                "sequence_id": str(sequence["sequence_id"]),
                "ego_level": int(sequence["ego_level"]),
                "object_level": int(sequence["object_level"]),
                "ego_amplitude_m": float(sequence["ego_amplitude_m"]),
                "object_amplitude_m": float(sequence["object_amplitude_m"]),
            }
        )

    rows: List[Dict[str, Any]] = []
    for (scene_id, anchor_id, camera_id, relative_level), members in sorted(grouped.items()):
        members.sort(key=lambda row: (row["ego_level"], row["object_level"]))
        rows.append(
            {
                "matched_relative_group_id": (
                    f"{group_id(scene_id, anchor_id, camera_id)}__r_{level_token(relative_level)}"
                ),
                "scene_id": scene_id,
                "anchor_id": anchor_id,
                "object_anchor_id": anchor_id,
                "base_camera_id": camera_id,
                "relative_level": int(relative_level),
                "members": members,
                "member_count": len(members),
                "has_multiple_causal_decompositions": len(members) > 1,
            }
        )
    return rows


def translated_camera_payload(
    base_camera: Mapping[str, Any],
    displacement_m: float,
    axis_world: Sequence[float] = MOTION_AXIS_WORLD,
) -> Dict[str, Any]:
    """Translate a camera in world coordinates without changing its rotation.

    No look-at operation occurs here.  The OpenCV transforms are derived from
    the translated Blender camera-to-world matrix using the declared fixed
    coordinate conversion.
    """
    axis = np.asarray(axis_world, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("axis_world must be a finite length-3 vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("axis_world must be nonzero")
    axis = axis / norm

    base_c2w = np.asarray(base_camera["blender_camera_to_world"], dtype=np.float64)
    if base_c2w.shape != (4, 4):
        raise ValueError("base camera transform must be 4x4")
    c2w = base_c2w.copy()
    c2w[:3, 3] = base_c2w[:3, 3] + axis * float(displacement_m)
    blender_w2c = np.linalg.inv(c2w)
    opencv_w2c = BLENDER_CAMERA_TO_OPENCV @ blender_w2c
    opencv_c2w = np.linalg.inv(opencv_w2c)

    payload = dict(base_camera)
    payload.update(
        {
            "blender_camera_to_world": np_to_list(c2w),
            "blender_world_to_camera": np_to_list(blender_w2c),
            "opencv_camera_to_world": np_to_list(opencv_c2w),
            "opencv_world_to_camera": np_to_list(opencv_w2c),
            "position": vector_to_list(c2w[:3, 3]),
            "rotation_matrix": np_to_list(c2w[:3, :3]),
            "base_camera_id": str(base_camera["camera_id"]),
            "camera_translation_world_m": vector_to_list(axis * float(displacement_m)),
            "look_at_recomputed": False,
        }
    )
    if "look_at" in payload:
        payload["initial_look_at"] = payload.pop("look_at")
    return payload


def transform_points(matrix: Sequence[Sequence[float]], points: np.ndarray) -> np.ndarray:
    matrix_array = np.asarray(matrix, dtype=np.float64)
    points_array = np.asarray(points, dtype=np.float64)
    if matrix_array.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4, 4), got {matrix_array.shape}")
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points_array.shape}")
    homogeneous = np.concatenate(
        [points_array, np.ones((points_array.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (matrix_array @ homogeneous.T).T[:, :3]


def project_opencv(
    points_world: np.ndarray,
    opencv_world_to_camera: Sequence[Sequence[float]],
    intrinsic_k: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points to top-left-origin pixel coordinates."""
    camera_points = transform_points(opencv_world_to_camera, points_world)
    k = np.asarray(intrinsic_k, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {k.shape}")
    uv = np.full((camera_points.shape[0], 2), np.nan, dtype=np.float64)
    positive = camera_points[:, 2] > 0.0
    if positive.any():
        projected = (k @ camera_points[positive].T).T
        uv[positive] = projected[:, :2] / projected[:, 2:3]
    return camera_points, uv
