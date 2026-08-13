"""Camera placement and coordinate convention helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .manifest_utils import np_to_list, vector_to_list


BLENDER_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def look_at_cam2world(location: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """Build a Blender camera-to-world matrix that points local -Z at target."""
    loc = np.asarray(location, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    forward = tgt - loc
    norm = np.linalg.norm(forward)
    if norm <= 1e-9:
        raise ValueError("Camera location and look-at point must differ")
    forward /= norm

    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up_guess)
    if np.linalg.norm(right) <= 1e-9:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up_guess)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward
    mat[:3, 3] = loc
    return mat


def camera_intrinsics_from_fov(width: int, height: int, fov_degrees: float) -> np.ndarray:
    fov = math.radians(float(fov_degrees))
    fx = width / (2.0 * math.tan(fov * 0.5))
    fy = fx
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> List[float]:
    """Convert a 3x3 rotation matrix to a Blender-style WXYZ quaternion."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return vector_to_list(quat)


def camera_payload_from_matrix(
    camera_id: str,
    cam2world_blender: np.ndarray,
    width: int,
    height: int,
    camera_cfg: Mapping[str, Any],
    look_at: Sequence[float],
    is_primary: bool,
) -> Dict[str, Any]:
    blender_world_to_camera = np.linalg.inv(cam2world_blender)
    opencv_world_to_camera = BLENDER_CAMERA_TO_OPENCV @ blender_world_to_camera
    opencv_camera_to_world = np.linalg.inv(opencv_world_to_camera)
    intrinsic = camera_intrinsics_from_fov(width, height, float(camera_cfg["fov_degrees"]))

    return {
        "camera_id": camera_id,
        "intrinsics": {
            "width": int(width),
            "height": int(height),
            "fov_degrees": float(camera_cfg["fov_degrees"]),
            "sensor_width_mm": float(camera_cfg.get("sensor_width_mm", 36.0)),
            "K": np_to_list(intrinsic),
            "principal_point": [float(width) * 0.5, float(height) * 0.5],
        },
        "blender_camera_to_world": np_to_list(cam2world_blender),
        "blender_world_to_camera": np_to_list(blender_world_to_camera),
        "opencv_camera_to_world": np_to_list(opencv_camera_to_world),
        "opencv_world_to_camera": np_to_list(opencv_world_to_camera),
        "blender_to_opencv_camera": np_to_list(BLENDER_CAMERA_TO_OPENCV),
        "opencv_to_blender_camera": np_to_list(BLENDER_CAMERA_TO_OPENCV),
        "position": vector_to_list(cam2world_blender[:3, 3]),
        "rotation_matrix": np_to_list(cam2world_blender[:3, :3]),
        "quaternion_wxyz": rotation_matrix_to_quaternion_wxyz(cam2world_blender[:3, :3]),
        "look_at": vector_to_list(look_at),
        "fov_degrees": float(camera_cfg["fov_degrees"]),
        "clip_start": float(camera_cfg["clip_start"]),
        "clip_end": float(camera_cfg["clip_end"]),
        "is_primary": bool(is_primary),
        "is_held_out": bool(not is_primary),
    }


def generate_cameras(
    scene_id: str,
    camera_cfg: Mapping[str, Any],
    mode_settings: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    count = int(mode_settings["camera_count"])
    primary_count = int(mode_settings["primary_camera_count"])
    width, height = [int(v) for v in mode_settings["resolution"]]
    start_deg, end_deg = [float(value) for value in camera_cfg["arc_degrees"]]
    radius = float(camera_cfg["radius_m"])
    heights = [float(value) for value in camera_cfg["height_levels_m"]]
    look_at = [float(value) for value in camera_cfg["look_at"]]

    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        t = idx / max(1, count - 1)
        angle = math.radians(start_deg + (end_deg - start_deg) * t)
        height = heights[idx % len(heights)]
        # A shallow ellipse keeps cameras around the open side of the room while
        # preserving different azimuths and depths.
        location = [
            radius * math.cos(angle),
            radius * math.sin(angle),
            height,
        ]
        pose = look_at_cam2world(location, look_at)
        payload = camera_payload_from_matrix(
            f"camera_{idx:03d}",
            pose,
            width,
            height,
            camera_cfg,
            look_at,
            idx < primary_count,
        )
        payload["scene_id"] = scene_id
        payload["azimuth_degrees"] = float(round(math.degrees(angle), 6))
        rows.append(payload)
    return rows
