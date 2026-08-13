"""Geometry for ScanNet cross-view correspondences.

Coordinate convention:
- ScanNet pose files are interpreted as camera-to-world transforms.
- Depth values are in meters after applying cfg.geometry.depth_scale.
- Source resized RGB token centers are mapped to source color coordinates, then
  to source depth coordinates by image-size scaling.
- Backprojection uses the depth intrinsic matrix.
- The 3D point is transformed source camera -> world -> target camera, projected
  with the target depth intrinsic, visibility checked against target depth, and
  mapped back to resized target RGB coordinates for pixel/token metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scannet_io import FrameRecord, SceneInfo, load_depth_m


@dataclass
class GridSpec:
    input_hw: tuple[int, int]
    grid_hw: tuple[int, int]
    patch_size: int


def token_centers(grid: GridSpec) -> np.ndarray:
    h, w = grid.input_hw
    ht, wt = grid.grid_hw
    xs = (np.arange(wt, dtype=np.float64) + 0.5) * (w / wt)
    ys = (np.arange(ht, dtype=np.float64) + 0.5) * (h / ht)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def token_xy_from_index(idx: np.ndarray, grid_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    wt = grid_hw[1]
    idx = np.asarray(idx)
    return idx % wt, idx // wt


def token_index_from_xy(x: np.ndarray, y: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    return np.asarray(y) * grid_hw[1] + np.asarray(x)


def resized_to_color(uv: np.ndarray, frame: FrameRecord, input_hw: tuple[int, int]) -> np.ndarray:
    h, w = input_hw
    color_w, color_h = frame.color_size
    out = uv.copy().astype(np.float64)
    out[:, 0] *= color_w / w
    out[:, 1] *= color_h / h
    return out


def color_to_resized(uv: np.ndarray, frame: FrameRecord, input_hw: tuple[int, int]) -> np.ndarray:
    h, w = input_hw
    color_w, color_h = frame.color_size
    out = uv.copy().astype(np.float64)
    out[:, 0] *= w / color_w
    out[:, 1] *= h / color_h
    return out


def color_to_depth(uv: np.ndarray, frame: FrameRecord) -> np.ndarray:
    color_w, color_h = frame.color_size
    depth_w, depth_h = frame.depth_size
    out = uv.copy().astype(np.float64)
    out[:, 0] *= depth_w / color_w
    out[:, 1] *= depth_h / color_h
    return out


def depth_to_color(uv: np.ndarray, frame: FrameRecord) -> np.ndarray:
    color_w, color_h = frame.color_size
    depth_w, depth_h = frame.depth_size
    out = uv.copy().astype(np.float64)
    out[:, 0] *= color_w / depth_w
    out[:, 1] *= color_h / depth_h
    return out


def sample_depth_nearest(depth_m: np.ndarray, uv_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    x = np.rint(uv_depth[:, 0]).astype(np.int64)
    y = np.rint(uv_depth[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    d = np.zeros(len(uv_depth), dtype=np.float64)
    valid_idx = np.where(inside)[0]
    d[valid_idx] = depth_m[y[valid_idx], x[valid_idx]]
    valid = inside & np.isfinite(d) & (d > 0)
    return d, valid


def backproject(uv_depth: np.ndarray, depth: np.ndarray, k_depth: np.ndarray) -> np.ndarray:
    fx, fy = k_depth[0, 0], k_depth[1, 1]
    cx, cy = k_depth[0, 2], k_depth[1, 2]
    x = (uv_depth[:, 0] - cx) * depth / fx
    y = (uv_depth[:, 1] - cy) * depth / fy
    return np.stack([x, y, depth], axis=1)


def transform_points(transform: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    homog = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=xyz.dtype)], axis=1)
    out = (transform @ homog.T).T
    return out[:, :3]


def project_depth(xyz_cam: np.ndarray, k_depth: np.ndarray) -> np.ndarray:
    z = xyz_cam[:, 2]
    u = k_depth[0, 0] * xyz_cam[:, 0] / z + k_depth[0, 2]
    v = k_depth[1, 1] * xyz_cam[:, 1] / z + k_depth[1, 2]
    return np.stack([u, v], axis=1)


def relative_pose_stats(src: FrameRecord, tgt: FrameRecord) -> tuple[float, float]:
    c_src = src.pose_c2w[:3, 3]
    c_tgt = tgt.pose_c2w[:3, 3]
    trans = float(np.linalg.norm(c_tgt - c_src))
    r_rel = tgt.pose_c2w[:3, :3].T @ src.pose_c2w[:3, :3]
    trace = float(np.trace(r_rel))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
    return trans, angle


def valid_token_correspondences(
    scene: SceneInfo,
    src: FrameRecord,
    tgt: FrameRecord,
    grid: GridSpec,
    geom_cfg: dict,
) -> pd.DataFrame:
    depth_scale = float(geom_cfg.get("depth_scale", 1000.0))
    abs_tol = float(geom_cfg.get("depth_abs_tol_m", 0.05))
    rel_tol = float(geom_cfg.get("depth_rel_tol", 0.05))
    src_depth = load_depth_m(src.depth_path, depth_scale)
    tgt_depth = load_depth_m(tgt.depth_path, depth_scale)

    centers = token_centers(grid)
    src_token_idx = np.arange(len(centers), dtype=np.int64)
    src_x, src_y = token_xy_from_index(src_token_idx, grid.grid_hw)
    src_color = resized_to_color(centers, src, grid.input_hw)
    src_depth_uv = color_to_depth(src_color, src)
    src_d, src_valid = sample_depth_nearest(src_depth, src_depth_uv)

    rows = []
    if not np.any(src_valid):
        return pd.DataFrame(rows)

    idx = np.where(src_valid)[0]
    xyz_src = backproject(src_depth_uv[idx], src_d[idx], scene.intrinsic_depth)
    xyz_world = transform_points(src.pose_c2w, xyz_src)
    tgt_w2c = np.linalg.inv(tgt.pose_c2w)
    xyz_tgt = transform_points(tgt_w2c, xyz_world)
    front = xyz_tgt[:, 2] > 1e-6
    tgt_uv_depth = project_depth(xyz_tgt, scene.intrinsic_depth)
    tgt_d, tgt_inside_depth = sample_depth_nearest(tgt_depth, tgt_uv_depth)
    depth_tol = np.maximum(abs_tol, rel_tol * xyz_tgt[:, 2])
    visible = front & tgt_inside_depth & (np.abs(tgt_d - xyz_tgt[:, 2]) < depth_tol)
    if not np.any(visible):
        return pd.DataFrame(rows)

    good_local = np.where(visible)[0]
    good_idx = idx[good_local]
    tgt_color = depth_to_color(tgt_uv_depth[good_local], tgt)
    tgt_resized = color_to_resized(tgt_color, tgt, grid.input_hw)
    h, w = grid.input_hw
    inside_resized = (
        (tgt_resized[:, 0] >= 0)
        & (tgt_resized[:, 0] < w)
        & (tgt_resized[:, 1] >= 0)
        & (tgt_resized[:, 1] < h)
    )
    if not np.any(inside_resized):
        return pd.DataFrame(rows)

    good_idx = good_idx[inside_resized]
    tgt_resized = tgt_resized[inside_resized]
    tgt_token_x = np.floor(tgt_resized[:, 0] / grid.patch_size).astype(np.int64)
    tgt_token_y = np.floor(tgt_resized[:, 1] / grid.patch_size).astype(np.int64)
    ht, wt = grid.grid_hw
    in_grid = (tgt_token_x >= 0) & (tgt_token_x < wt) & (tgt_token_y >= 0) & (tgt_token_y < ht)
    good_idx = good_idx[in_grid]
    tgt_resized = tgt_resized[in_grid]
    tgt_token_x = tgt_token_x[in_grid]
    tgt_token_y = tgt_token_y[in_grid]
    tgt_token_idx = token_index_from_xy(tgt_token_x, tgt_token_y, grid.grid_hw)

    for n, src_idx in enumerate(good_idx):
        rows.append(
            {
                "src_token_idx": int(src_idx),
                "src_token_x": int(src_x[src_idx]),
                "src_token_y": int(src_y[src_idx]),
                "src_u": float(centers[src_idx, 0]),
                "src_v": float(centers[src_idx, 1]),
                "gt_tgt_u": float(tgt_resized[n, 0]),
                "gt_tgt_v": float(tgt_resized[n, 1]),
                "gt_tgt_token_idx": int(tgt_token_idx[n]),
                "gt_tgt_token_x": int(tgt_token_x[n]),
                "gt_tgt_token_y": int(tgt_token_y[n]),
            }
        )
    return pd.DataFrame(rows)


def estimate_overlap(
    scene: SceneInfo,
    src: FrameRecord,
    tgt: FrameRecord,
    geom_cfg: dict,
    *,
    max_samples: int = 4096,
    seed: int = 0,
) -> float:
    depth_scale = float(geom_cfg.get("depth_scale", 1000.0))
    abs_tol = float(geom_cfg.get("depth_abs_tol_m", 0.05))
    rel_tol = float(geom_cfg.get("depth_rel_tol", 0.05))
    src_depth = load_depth_m(src.depth_path, depth_scale)
    tgt_depth = load_depth_m(tgt.depth_path, depth_scale)
    ys, xs = np.where(src_depth > 0)
    if len(xs) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    take = min(max_samples, len(xs))
    chosen = rng.choice(len(xs), size=take, replace=False)
    uv = np.stack([xs[chosen].astype(np.float64), ys[chosen].astype(np.float64)], axis=1)
    d = src_depth[ys[chosen], xs[chosen]].astype(np.float64)
    xyz_src = backproject(uv, d, scene.intrinsic_depth)
    xyz_world = transform_points(src.pose_c2w, xyz_src)
    xyz_tgt = transform_points(np.linalg.inv(tgt.pose_c2w), xyz_world)
    front = xyz_tgt[:, 2] > 1e-6
    tgt_uv = project_depth(xyz_tgt, scene.intrinsic_depth)
    tgt_d, tgt_valid = sample_depth_nearest(tgt_depth, tgt_uv)
    depth_tol = np.maximum(abs_tol, rel_tol * xyz_tgt[:, 2])
    visible = front & tgt_valid & (np.abs(tgt_d - xyz_tgt[:, 2]) < depth_tol)
    return float(np.mean(visible)) if len(visible) else 0.0


def target_valid_token_mask(frame: FrameRecord, grid: GridSpec, geom_cfg: dict) -> np.ndarray:
    depth = load_depth_m(frame.depth_path, float(geom_cfg.get("depth_scale", 1000.0)))
    centers = token_centers(grid)
    uv_color = resized_to_color(centers, frame, grid.input_hw)
    uv_depth = color_to_depth(uv_color, frame)
    _, valid = sample_depth_nearest(depth, uv_depth)
    return valid

