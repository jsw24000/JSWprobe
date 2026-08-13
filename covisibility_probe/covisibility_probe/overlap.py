from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scannet_io import FrameRecord, SceneInfo, load_depth_m, rotation_angle_deg, translation_distance


@dataclass(frozen=True)
class OverlapResult:
    frame_ids: np.ndarray
    overlap_cos: np.ndarray
    overlap_iou: np.ndarray
    coverage_i_to_j: np.ndarray
    counts: np.ndarray


def _unique_rows(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.reshape(0, 3).astype(np.int64)
    return np.unique(np.ascontiguousarray(values.astype(np.int64)), axis=0)


def _row_view(values: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(values.astype(np.int64))
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).reshape(-1)


def intersection_count(a: np.ndarray, b: np.ndarray) -> int:
    if len(a) == 0 or len(b) == 0:
        return 0
    return int(np.intersect1d(_row_view(a), _row_view(b), assume_unique=False).shape[0])


def visible_voxels_for_frame(
    frame: FrameRecord,
    scene: SceneInfo,
    *,
    depth_scale: float = 1000.0,
    pixel_stride: int = 4,
    voxel_size: float = 0.05,
) -> np.ndarray:
    if not frame.valid_pose or not frame.valid_depth:
        return np.empty((0, 3), dtype=np.int64)

    depth = load_depth_m(frame.depth_path, depth_scale=depth_scale)
    stride = max(1, int(pixel_stride))
    ys = np.arange(0, depth.shape[0], stride)
    xs = np.arange(0, depth.shape[1], stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    z = depth[yy, xx].reshape(-1).astype(np.float64)
    x_pix = xx.reshape(-1).astype(np.float64)
    y_pix = yy.reshape(-1).astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.int64)

    z = z[valid]
    x_pix = x_pix[valid]
    y_pix = y_pix[valid]
    k = scene.intrinsic_depth
    x_cam = (x_pix - k[0, 2]) * z / k[0, 0]
    y_cam = (y_pix - k[1, 2]) * z / k[1, 1]
    points_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
    points_world = (frame.pose_c2w @ points_cam.T).T[:, :3]
    quantized = np.floor(points_world / float(voxel_size)).astype(np.int64)
    return _unique_rows(quantized)


def pair_overlap_from_voxels(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, int]:
    na = len(a)
    nb = len(b)
    inter = intersection_count(a, b)
    if na == 0 or nb == 0:
        return 0.0, 0.0, 0.0, 0.0, inter
    union = na + nb - inter
    overlap_cos = inter / math.sqrt(na * nb)
    overlap_iou = inter / union if union > 0 else 0.0
    coverage_a_to_b = inter / nb
    coverage_b_to_a = inter / na
    return float(overlap_cos), float(overlap_iou), float(coverage_a_to_b), float(coverage_b_to_a), inter


def build_overlap_matrices(frame_ids: list[int], voxels: list[np.ndarray]) -> OverlapResult:
    n = len(frame_ids)
    overlap_cos = np.zeros((n, n), dtype=np.float32)
    overlap_iou = np.zeros((n, n), dtype=np.float32)
    coverage = np.zeros((n, n), dtype=np.float32)
    counts = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i, n):
            oc, oi, cov_i_to_j, cov_j_to_i, inter = pair_overlap_from_voxels(voxels[i], voxels[j])
            overlap_cos[i, j] = overlap_cos[j, i] = oc
            overlap_iou[i, j] = overlap_iou[j, i] = oi
            coverage[i, j] = cov_i_to_j
            coverage[j, i] = cov_j_to_i
            counts[i, j] = counts[j, i] = inter
    return OverlapResult(
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        overlap_cos=overlap_cos,
        overlap_iou=overlap_iou,
        coverage_i_to_j=coverage,
        counts=counts,
    )


def save_visible_voxels(voxels: list[np.ndarray], frame_ids: list[int], out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fid, arr in zip(frame_ids, voxels):
        p = out / f"{int(fid):06d}.npz"
        np.savez_compressed(p, voxels=arr.astype(np.int64))
        paths.append(str(p))
    return paths


def save_overlap_npz(result: OverlapResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        frame_ids=result.frame_ids,
        overlap_cos=result.overlap_cos,
        overlap_iou=result.overlap_iou,
        coverage_i_to_j=result.coverage_i_to_j,
        counts=result.counts,
    )


def load_overlap_npz(path: str | Path) -> OverlapResult:
    data = np.load(path)
    return OverlapResult(
        frame_ids=data["frame_ids"],
        overlap_cos=data["overlap_cos"],
        overlap_iou=data["overlap_iou"],
        coverage_i_to_j=data["coverage_i_to_j"],
        counts=data["counts"],
    )


def pose_pair_stats(frames: list[FrameRecord], i: int, j: int) -> tuple[float, float]:
    return translation_distance(frames[i].pose_c2w, frames[j].pose_c2w), rotation_angle_deg(frames[i].pose_c2w, frames[j].pose_c2w)


def project_depth(xyz_cam: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = xyz_cam[:, 2]
    u = k[0, 0] * xyz_cam[:, 0] / z + k[0, 2]
    v = k[1, 1] * xyz_cam[:, 1] / z + k[1, 2]
    return np.stack([u, v], axis=1)


def backproject_depth_pixels(uv: np.ndarray, depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    x = (uv[:, 0] - k[0, 2]) * depth / k[0, 0]
    y = (uv[:, 1] - k[1, 2]) * depth / k[1, 1]
    return np.stack([x, y, depth], axis=1)


def reprojection_overlap_sample(
    scene: SceneInfo,
    src: FrameRecord,
    tgt: FrameRecord,
    *,
    depth_scale: float,
    max_samples: int = 4096,
    seed: int = 0,
    abs_tol_m: float = 0.05,
    rel_tol: float = 0.05,
) -> float:
    src_depth = load_depth_m(src.depth_path, depth_scale)
    tgt_depth = load_depth_m(tgt.depth_path, depth_scale)
    ys, xs = np.where(src_depth > 0)
    if len(xs) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    take = min(int(max_samples), len(xs))
    chosen = rng.choice(len(xs), size=take, replace=False)
    uv = np.stack([xs[chosen].astype(np.float64), ys[chosen].astype(np.float64)], axis=1)
    z = src_depth[ys[chosen], xs[chosen]].astype(np.float64)
    src_xyz = backproject_depth_pixels(uv, z, scene.intrinsic_depth)
    src_h = np.concatenate([src_xyz, np.ones((len(src_xyz), 1), dtype=np.float64)], axis=1)
    world = (src.pose_c2w @ src_h.T).T[:, :3]
    tgt_w2c = np.linalg.inv(tgt.pose_c2w)
    tgt_h = np.concatenate([world, np.ones((len(world), 1), dtype=np.float64)], axis=1)
    tgt_xyz = (tgt_w2c @ tgt_h.T).T[:, :3]
    front = tgt_xyz[:, 2] > 1e-6
    tgt_uv = project_depth(tgt_xyz, scene.intrinsic_depth)
    x = np.rint(tgt_uv[:, 0]).astype(np.int64)
    y = np.rint(tgt_uv[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < tgt_depth.shape[1]) & (y >= 0) & (y < tgt_depth.shape[0])
    sampled = np.zeros(len(x), dtype=np.float64)
    valid_idx = np.where(inside)[0]
    sampled[valid_idx] = tgt_depth[y[valid_idx], x[valid_idx]]
    depth_tol = np.maximum(abs_tol_m, rel_tol * tgt_xyz[:, 2])
    visible = front & inside & (sampled > 0) & (np.abs(sampled - tgt_xyz[:, 2]) < depth_tol)
    return float(np.mean(visible)) if len(visible) else 0.0

