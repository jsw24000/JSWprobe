from __future__ import annotations

import math
from typing import Any

import numpy as np


def rotation_orthonormal_error(R: np.ndarray) -> float:
    return float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))


def rotation_det_error(R: np.ndarray) -> float:
    return float(abs(np.linalg.det(R) - 1.0))


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = R_a.T @ R_b
    trace = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(math.degrees(math.acos(cos_theta)))


def view_direction_c2w(pose_c2w: np.ndarray) -> np.ndarray:
    direction = pose_c2w[:3, 2].astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    return direction / norm


def pose_validity(poses_c2w: np.ndarray) -> dict[str, Any]:
    rotations = poses_c2w[:, :3, :3]
    bottom = poses_c2w[:, 3, :]
    orth_errors = np.array([rotation_orthonormal_error(R) for R in rotations])
    det_errors = np.array([rotation_det_error(R) for R in rotations])
    bottom_errors = np.linalg.norm(bottom - np.array([0.0, 0.0, 0.0, 1.0]), axis=1)
    return {
        "finite": bool(np.isfinite(poses_c2w).all()),
        "max_rotation_orthonormal_error": float(orth_errors.max()) if len(orth_errors) else None,
        "mean_rotation_orthonormal_error": float(orth_errors.mean()) if len(orth_errors) else None,
        "max_rotation_det_error": float(det_errors.max()) if len(det_errors) else None,
        "mean_rotation_det_error": float(det_errors.mean()) if len(det_errors) else None,
        "max_bottom_row_error": float(bottom_errors.max()) if len(bottom_errors) else None,
    }


def trajectory_continuity(poses_c2w: np.ndarray) -> dict[str, Any]:
    centers = poses_c2w[:, :3, 3]
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    rot_step = np.array([
        rotation_angle_deg(poses_c2w[i, :3, :3], poses_c2w[i + 1, :3, :3])
        for i in range(len(poses_c2w) - 1)
    ])
    duplicate_steps = int(np.count_nonzero(step < 1e-9))
    return {
        "translation_step_m": {
            "min": float(step.min()) if len(step) else None,
            "median": float(np.median(step)) if len(step) else None,
            "mean": float(step.mean()) if len(step) else None,
            "p95": float(np.percentile(step, 95)) if len(step) else None,
            "max": float(step.max()) if len(step) else None,
            "num_zero_or_duplicate_steps": duplicate_steps,
        },
        "rotation_step_deg": {
            "min": float(rot_step.min()) if len(rot_step) else None,
            "median": float(np.median(rot_step)) if len(rot_step) else None,
            "mean": float(rot_step.mean()) if len(rot_step) else None,
            "p95": float(np.percentile(rot_step, 95)) if len(rot_step) else None,
            "max": float(rot_step.max()) if len(rot_step) else None,
        },
    }


def loop_pair_metrics(poses_c2w: np.ndarray, i: int, j: int) -> dict[str, Any]:
    pose_i = poses_c2w[i]
    pose_j = poses_c2w[j]
    center_i = pose_i[:3, 3]
    center_j = pose_j[:3, 3]
    rel_t = center_j - center_i
    direction_i = view_direction_c2w(pose_i)
    direction_j = view_direction_c2w(pose_j)
    view_cos = float(np.dot(direction_i, direction_j))
    rot_deg = rotation_angle_deg(pose_i[:3, :3], pose_j[:3, :3])
    dist = float(np.linalg.norm(rel_t))
    return {
        "frame_i": int(i),
        "frame_j": int(j),
        "camera_center_i": center_i,
        "camera_center_j": center_j,
        "relative_translation_j_minus_i": rel_t,
        "center_distance_m": dist,
        "relative_rotation_deg": rot_deg,
        "view_direction_i": direction_i,
        "view_direction_j": direction_j,
        "view_direction_cosine": view_cos,
        "pose_based_view_overlap_possible": bool(dist < 2.0 and view_cos > -0.25),
        "strict_same_direction_possible": bool(dist < 2.0 and view_cos > 0.5),
    }
