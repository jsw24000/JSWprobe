from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_camera_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def intrinsic_matrix(camera_data: dict[str, Any]) -> np.ndarray:
    return np.asarray(camera_data["intrinsic"]["K"], dtype=np.float32)


def gt_cam2worlds(camera_data: dict[str, Any]) -> np.ndarray:
    frames = sorted(camera_data["frames"], key=lambda x: x["frame"])
    return np.asarray([frame["cam2world"] for frame in frames], dtype=np.float32)


def gt_world2cams(camera_data: dict[str, Any]) -> np.ndarray:
    frames = sorted(camera_data["frames"], key=lambda x: x["frame"])
    return np.asarray([frame["world2cam"] for frame in frames], dtype=np.float32)


def blender_cam2world_to_opencv_cam2world(cam2world: np.ndarray) -> np.ndarray:
    """Convert Blender/OpenGL camera-to-world to OpenCV camera-to-world.

    Blender camera local axes are x-right, y-up, z-backward. LingBot's pose
    utilities and the local depth unprojection use OpenCV camera axes:
    x-right, y-down, z-forward. The world frame remains the Blender world.
    """
    cam2world = np.asarray(cam2world, dtype=np.float32)
    opencv_to_blender = np.eye(4, dtype=np.float32)
    opencv_to_blender[1, 1] = -1.0
    opencv_to_blender[2, 2] = -1.0
    return cam2world @ opencv_to_blender


def gt_cam2worlds_opencv(camera_data: dict[str, Any]) -> np.ndarray:
    return blender_cam2world_to_opencv_cam2world(gt_cam2worlds(camera_data))


def pose_translation(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w)[..., :3, 3]


def to_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.shape[-2:] == (4, 4):
        return pose
    if pose.shape[-2:] != (3, 4):
        raise ValueError(f"Expected pose ending in 3x4 or 4x4, got {pose.shape}")
    out = np.zeros((*pose.shape[:-2], 4, 4), dtype=pose.dtype)
    out[..., :3, :4] = pose
    out[..., 3, 3] = 1
    return out


def invert_se3(pose_4x4: np.ndarray) -> np.ndarray:
    pose_4x4 = np.asarray(pose_4x4)
    out = np.zeros_like(pose_4x4)
    r = pose_4x4[..., :3, :3]
    t = pose_4x4[..., :3, 3]
    rt = np.swapaxes(r, -1, -2)
    out[..., :3, :3] = rt
    out[..., :3, 3] = -(rt @ t[..., None]).squeeze(-1)
    out[..., 3, 3] = 1
    return out
