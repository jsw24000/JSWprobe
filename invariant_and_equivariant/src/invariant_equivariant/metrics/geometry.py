from __future__ import annotations

from typing import Dict

import numpy as np


def orthonormal_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    if matrix.shape[0] <= matrix.shape[1]:
        # Row-space basis for W_abs [3, D] lives in D dimensions.
        return vt[:rank].T
    return u[:, :rank]


def effective_rank(singular_values: np.ndarray, threshold: float = 0.05) -> int:
    if singular_values.size == 0 or singular_values[0] <= 0:
        return 0
    return int((singular_values / singular_values[0] >= threshold).sum())


def subspace_metrics(q_a: np.ndarray, q_b: np.ndarray) -> Dict[str, float]:
    if q_a.size == 0 or q_b.size == 0:
        return {"projection_overlap": float("nan"), "grassmann_distance": float("nan"), "max_principal_angle_deg": float("nan"), "min_principal_angle_deg": float("nan")}
    m = q_a.T @ q_b
    s = np.linalg.svd(m, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.arccos(s)
    k = min(q_a.shape[1], q_b.shape[1])
    return {
        "projection_overlap": float(np.sum(s**2) / max(1, k)),
        "grassmann_distance": float(np.linalg.norm(np.sin(angles))),
        "max_principal_angle_deg": float(np.degrees(np.max(angles))),
        "min_principal_angle_deg": float(np.degrees(np.min(angles))),
        "mean_principal_angle_deg": float(np.degrees(np.mean(angles))),
    }

