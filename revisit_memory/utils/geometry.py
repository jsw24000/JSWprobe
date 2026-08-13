from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def bbox_bounds(center: np.ndarray, dimensions: np.ndarray, expansion: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float32)
    dimensions = np.asarray(dimensions, dtype=np.float32) * (1.0 + float(expansion))
    return center - dimensions / 2.0, center + dimensions / 2.0


def points_in_bbox(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    finite = np.isfinite(points).all(axis=-1)
    inside = np.logical_and(points >= bbox_min, points <= bbox_max).all(axis=-1)
    return finite & inside


def occupancy_metrics(
    points: np.ndarray,
    confidence: np.ndarray | None,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    points_flat = points.reshape(-1, 3)
    valid = np.isfinite(points_flat).all(axis=1)
    if confidence is not None:
        conf_flat = confidence.reshape(-1)
        valid &= np.isfinite(conf_flat)
        if confidence_threshold is not None:
            valid &= conf_flat >= confidence_threshold
    else:
        conf_flat = np.ones(points_flat.shape[0], dtype=np.float32)

    if not valid.any():
        return {
            "total_points": 0,
            "bbox_points": 0,
            "bbox_confidence_weighted_points": 0.0,
            "bbox_point_fraction": 0.0,
        }

    inside = points_in_bbox(points_flat, bbox_min, bbox_max) & valid
    total = int(valid.sum())
    bbox_count = int(inside.sum())
    weighted = float(conf_flat[inside].sum())
    return {
        "total_points": total,
        "bbox_points": bbox_count,
        "bbox_confidence_weighted_points": weighted,
        "bbox_point_fraction": float(bbox_count / max(total, 1)),
    }


def depth_to_camera_points(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    z = depth
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    hom = np.concatenate([pts, ones], axis=1)
    out = hom @ transform_4x4.T
    return out[:, :3].reshape(points.shape)


def depth_to_world_points(depth: np.ndarray, intrinsic: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    return transform_points(depth_to_camera_points(depth, intrinsic), cam2world)


def sample_points(points: np.ndarray, confidence: np.ndarray | None, max_points: int, seed: int = 0):
    pts = points.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    conf = None
    if confidence is not None:
        conf = confidence.reshape(-1)
        valid &= np.isfinite(conf)
    idx = np.flatnonzero(valid)
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    sampled_points = pts[idx]
    sampled_conf = conf[idx] if conf is not None else None
    return sampled_points, sampled_conf


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3)[valid]
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    with path.open("w", encoding="utf-8") as f:
        if colors is None:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def sim3_align(source: np.ndarray, target: np.ndarray, allow_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama alignment: returns scale, rotation, translation mapping source to target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be [N, 3]")
    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)
    src_c = source - mu_src
    tgt_c = target - mu_tgt
    cov = (tgt_c.T @ src_c) / max(len(source), 1)
    u, s, vh = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vh
    if allow_scale:
        var_src = np.mean(np.sum(src_c * src_c, axis=1))
        scale = float(np.sum(s * d) / max(var_src, 1e-12))
    else:
        scale = 1.0
    trans = mu_tgt - scale * (rot @ mu_src)
    return scale, rot.astype(np.float64), trans.astype(np.float64)


def apply_sim3_to_poses(c2w: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[..., :3, :3] = rot @ out[..., :3, :3]
    out[..., :3, 3] = scale * (rot @ out[..., :3, 3][..., None]).squeeze(-1) + trans
    return out


def rotation_geodesic_deg(r_pred: np.ndarray, r_gt: np.ndarray) -> np.ndarray:
    rel = np.matmul(np.swapaxes(r_pred, -1, -2), r_gt)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

