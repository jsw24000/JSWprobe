"""Pose loading and lightweight geometry utilities for ScanNet manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_pose_matrix(path: str | Path) -> np.ndarray:
    """Load a ScanNet pose text file as a 4x4 float matrix."""

    pose_path = Path(path)
    matrix = np.loadtxt(pose_path, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix, got {matrix.shape}: {pose_path}")
    return matrix


def is_valid_pose(path: str | Path) -> bool:
    """Return True when a pose file exists, is 4x4, and contains finite values."""

    try:
        matrix = load_pose_matrix(path)
    except Exception:
        return False
    return bool(np.isfinite(matrix).all())


def pose_translation(pose: Any) -> np.ndarray:
    """Return the 3D translation vector from a 4x4 pose-like object."""

    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose, got {matrix.shape}")
    return matrix[:3, 3]


def translation_distance(pose_a: Any, pose_b: Any) -> float:
    """Compute Euclidean distance between two camera translations."""

    return float(np.linalg.norm(pose_translation(pose_a) - pose_translation(pose_b)))


def rotation_distance_deg(pose_a: Any, pose_b: Any) -> float:
    """Compute angular distance in degrees between two 3x3 pose rotations."""

    rot_a = np.asarray(pose_a, dtype=float)[:3, :3]
    rot_b = np.asarray(pose_b, dtype=float)[:3, :3]
    relative = rot_a.T @ rot_b
    trace = float(np.trace(relative))
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def safe_pose_distances(path_a: str | Path | None, path_b: str | Path | None) -> dict[str, float | None]:
    """Return translation and rotation distances, using None if loading fails."""

    if not path_a or not path_b:
        return {"translation": None, "rotation_deg": None}
    try:
        pose_a = load_pose_matrix(path_a)
        pose_b = load_pose_matrix(path_b)
    except Exception:
        return {"translation": None, "rotation_deg": None}
    return {
        "translation": translation_distance(pose_a, pose_b),
        "rotation_deg": rotation_distance_deg(pose_a, pose_b),
    }
