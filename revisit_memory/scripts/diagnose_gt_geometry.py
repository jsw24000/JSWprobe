#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.camera import blender_cam2world_to_opencv_cam2world, intrinsic_matrix, load_camera_file  # noqa: E402
from revisit_memory.utils.depth_scale import estimate_depth_scale, resize_bool_stack, resize_float, resize_float_stack  # noqa: E402
from revisit_memory.utils.geometry import depth_to_world_points, points_in_bbox, write_ply  # noqa: E402
from revisit_memory.utils.io import (  # noqa: E402
    camera_json_path,
    condition_dir,
    depth_path,
    load_config,
    load_depth_exr,
    load_rgb,
    metadata_path,
    output_root,
    read_json,
    target_object,
    write_json,
)


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)
    area_a = max(0.0, a[2] - a[0] + 1.0) * max(0.0, a[3] - a[1] + 1.0)
    area_b = max(0.0, b[2] - b[0] + 1.0) * max(0.0, b[3] - b[1] + 1.0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def project_points(points: np.ndarray, world2cam: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    cam = (world2cam @ hom.T).T[:, :3]
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    front = cam[:, 2] > 1.0e-6
    if front.any():
        proj = (intrinsic @ cam[front].T).T
        uv[front] = proj[:, :2] / proj[:, 2:3]
    return uv, cam[:, 2]


def target_corners(target: dict[str, Any]) -> np.ndarray:
    center = np.asarray(target["location"], dtype=np.float64)
    dims = np.asarray(target["dimensions"], dtype=np.float64)
    bmin = center - dims / 2.0
    bmax = center + dims / 2.0
    return np.asarray(
        [[x, y, z] for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])],
        dtype=np.float64,
    )


def projection_checks(cfg: dict[str, Any], condition: str, frames: list[int]) -> list[dict[str, Any]]:
    cam_data = load_camera_file(camera_json_path(cfg, condition))
    meta = read_json(metadata_path(cfg, condition))
    target = next(obj for obj in meta["objects"] if obj.get("is_target"))
    corners = target_corners(target)
    center = np.asarray(target["location"], dtype=np.float64)[None]
    intrinsic = intrinsic_matrix(cam_data).astype(np.float64)
    rows = []
    for fid in frames:
        object_map_path = condition_dir(cfg, condition) / "masks" / "object_id" / f"frame_{fid:04d}.npy"
        target_mask = np.load(object_map_path) == int(target["object_id"]) if object_map_path.exists() else np.zeros((1, 1), dtype=bool)
        observed_bbox = mask_bbox(target_mask)
        raw_c2w = np.asarray(cam_data["frames"][fid]["cam2world"], dtype=np.float64)
        cv_c2w = blender_cam2world_to_opencv_cam2world(raw_c2w).astype(np.float64)
        for name, c2w in (("raw_blender_as_opencv", raw_c2w), ("converted_opencv", cv_c2w)):
            uv_center, z_center = project_points(center, np.linalg.inv(c2w), intrinsic)
            uv_corners, _ = project_points(corners, np.linalg.inv(c2w), intrinsic)
            finite = np.isfinite(uv_corners).all(axis=1)
            projected_bbox = None
            if finite.any():
                projected_bbox = [
                    float(np.nanmin(uv_corners[finite, 0])),
                    float(np.nanmin(uv_corners[finite, 1])),
                    float(np.nanmax(uv_corners[finite, 0])),
                    float(np.nanmax(uv_corners[finite, 1])),
                ]
            rows.append(
                {
                    "frame": int(fid),
                    "pose_variant": name,
                    "target_mask_pixels": int(target_mask.sum()),
                    "observed_mask_bbox_xyxy": observed_bbox,
                    "target_center_camera_z": float(z_center[0]),
                    "target_center_uv": [float(x) if np.isfinite(x) else None for x in uv_center[0]],
                    "projected_bbox_xyxy": projected_bbox,
                    "bbox_iou": bbox_iou(observed_bbox, projected_bbox),
                }
            )
    return rows


def geometry_stats_for_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    cam2world: np.ndarray,
    object_map: np.ndarray,
    target_id: int,
    target_min: np.ndarray,
    target_max: np.ndarray,
) -> dict[str, Any]:
    points = depth_to_world_points(depth.astype(np.float32), intrinsic.astype(np.float32), cam2world.astype(np.float32))
    floor_mask = object_map == 1
    target_mask = object_map == target_id
    row: dict[str, Any] = {
        "floor_pixels": int(floor_mask.sum()),
        "target_pixels": int(target_mask.sum()),
    }
    if floor_mask.any():
        z = points[floor_mask][:, 2]
        row.update(
            {
                "floor_world_z_median": float(np.nanmedian(z)),
                "floor_world_z_mean_abs_to_zero": float(np.nanmean(np.abs(z))),
                "floor_world_z_p05": float(np.nanpercentile(z, 5)),
                "floor_world_z_p95": float(np.nanpercentile(z, 95)),
            }
        )
    if target_mask.any():
        target_points = points[target_mask]
        inside = points_in_bbox(target_points, target_min, target_max)
        row.update(
            {
                "target_inside_bbox_fraction": float(np.mean(inside)),
                "target_world_median_xyz": np.nanmedian(target_points, axis=0).astype(float).tolist(),
            }
        )
    return row


def gt_depth_checks(cfg: dict[str, Any], condition: str, frames: list[int]) -> list[dict[str, Any]]:
    cam_data = load_camera_file(camera_json_path(cfg, condition))
    intrinsic = intrinsic_matrix(cam_data)
    target = target_object(cfg)
    target_id = int(target["object_id"])
    center = np.asarray(target["location"], dtype=np.float32)
    dims = np.asarray(target["dimensions"], dtype=np.float32)
    target_min = center - dims / 2.0 - np.asarray([0.06, 0.06, 0.08], dtype=np.float32)
    target_max = center + dims / 2.0 + np.asarray([0.06, 0.06, 0.08], dtype=np.float32)
    rows = []
    for fid in frames:
        object_map = np.load(condition_dir(cfg, condition) / "masks" / "object_id" / f"frame_{fid:04d}.npy")
        raw_c2w = np.asarray(cam_data["frames"][fid]["cam2world"], dtype=np.float32)
        cv_c2w = blender_cam2world_to_opencv_cam2world(raw_c2w)
        row = {
            "frame": int(fid),
            "depth_source": "gt_depth",
            "camera_source": "converted_gt_camera",
            **geometry_stats_for_depth(load_depth_exr(depth_path(cfg, condition, fid)), intrinsic, cv_c2w, object_map, target_id, target_min, target_max),
        }
        rows.append(row)
    return rows


def prediction_depth_checks(cfg: dict[str, Any], condition: str, frames: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    recon_dir = output_root(cfg) / "reconstruction" / condition
    pred_path = recon_dir / "predictions.npz"
    inputs_path = recon_dir / "inputs.npz"
    if not pred_path.exists() or not inputs_path.exists():
        return [], None

    with np.load(pred_path, allow_pickle=True) as pred_npz:
        pred = {key: pred_npz[key] for key in pred_npz.files}
    with np.load(inputs_path, allow_pickle=True) as inp_npz:
        inputs = {key: inp_npz[key] for key in inp_npz.files}
    pred_depth = pred["depth"][..., 0] if pred["depth"].ndim == 4 else pred["depth"].squeeze(-1)
    if "gt_depth" not in inputs:
        return [], {"scale_policy": "unavailable_no_gt_depth_in_inputs"}
    gt_depth_pre = resize_float_stack(inputs["gt_depth"].astype(np.float32), pred_depth.shape[-2:])
    mask_pre = resize_bool_stack(inputs["counterfactual_target_mask"], pred_depth.shape[-2:]) if "counterfactual_target_mask" in inputs else None
    scale, scale_info = estimate_depth_scale(pred_depth, gt_depth_pre, mask_pre, seed=int(cfg["project"]["random_seed"]))

    target = target_object(cfg)
    target_id = int(target["object_id"])
    center = np.asarray(target["location"], dtype=np.float32)
    dims = np.asarray(target["dimensions"], dtype=np.float32)
    target_min = center - dims / 2.0 - np.asarray([0.06, 0.06, 0.08], dtype=np.float32)
    target_max = center + dims / 2.0 + np.asarray([0.06, 0.06, 0.08], dtype=np.float32)
    frame_ids = inputs["frame_ids"].astype(int).tolist()
    intrinsic = inputs["gt_intrinsic_preprocessed"].astype(np.float32)
    rows = []
    for fid in frames:
        if fid not in frame_ids:
            continue
        idx = frame_ids.index(fid)
        object_map = np.load(condition_dir(cfg, condition) / "masks" / "object_id" / f"frame_{fid:04d}.npy")
        object_map_pre = np.asarray(
            Image.fromarray(object_map.astype(np.uint16)).resize((pred_depth.shape[-1], pred_depth.shape[-2]), resample=Image.Resampling.NEAREST)
        )
        gt_pre = gt_depth_pre[idx]
        valid = np.isfinite(gt_pre) & np.isfinite(pred_depth[idx]) & (gt_pre > 1.0e-6) & (pred_depth[idx] > 1.0e-6)
        for name, depth in (("pred_depth_raw", pred_depth[idx]), ("pred_depth_metric_scaled", pred_depth[idx] * scale)):
            row = {
                "frame": int(fid),
                "depth_source": name,
                "camera_source": "converted_gt_camera",
                "pred_over_gt_depth_median": float(np.nanmedian(pred_depth[idx][valid] / gt_pre[valid])) if valid.any() else None,
                **geometry_stats_for_depth(depth, intrinsic, inputs["gt_cam2world"][idx], object_map_pre, target_id, target_min, target_max),
            }
            rows.append(row)
    return rows, scale_info


def resize_rgb(path: Path, hw: tuple[int, int]) -> np.ndarray:
    rgb = Image.fromarray(load_rgb(path))
    return np.asarray(rgb.resize((hw[1], hw[0]), resample=Image.Resampling.BICUBIC), dtype=np.uint8)


def sample_colored(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pts = points.reshape(-1, 3)
    cols = colors.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    idx = np.flatnonzero(valid)
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    return pts[idx], cols[idx]


def write_diagnostic_plys(cfg: dict[str, Any], condition: str, frames: list[int], out_dir: Path, max_points: int) -> list[dict[str, Any]]:
    cam_data = load_camera_file(camera_json_path(cfg, condition))
    intrinsic = intrinsic_matrix(cam_data)
    entries = []

    gt_pts_parts = []
    gt_cols_parts = []
    per_frame = max(1, max_points // max(len(frames), 1))
    for offset, fid in enumerate(frames):
        depth = load_depth_exr(depth_path(cfg, condition, fid))
        c2w = blender_cam2world_to_opencv_cam2world(np.asarray(cam_data["frames"][fid]["cam2world"], dtype=np.float32))
        pts = depth_to_world_points(depth, intrinsic, c2w)
        cols = load_rgb(condition_dir(cfg, condition) / "rgb" / f"frame_{fid:04d}.png")
        spts, scols = sample_colored(pts, cols, per_frame, seed=int(cfg["project"]["random_seed"]) + offset)
        gt_pts_parts.append(spts)
        gt_cols_parts.append(scols)
    gt_path = out_dir / "ply" / "gt_depth_gt_camera.ply"
    write_ply(gt_path, np.concatenate(gt_pts_parts, axis=0), np.concatenate(gt_cols_parts, axis=0))
    entries.append({"name": "gt_depth_gt_camera", "path": str(gt_path), "frames": frames})

    recon_dir = output_root(cfg) / "reconstruction" / condition
    pred_path = recon_dir / "predictions.npz"
    inputs_path = recon_dir / "inputs.npz"
    if not pred_path.exists() or not inputs_path.exists():
        return entries

    with np.load(pred_path, allow_pickle=True) as pred_npz:
        pred = {key: pred_npz[key] for key in pred_npz.files}
    with np.load(inputs_path, allow_pickle=True) as inp_npz:
        inputs = {key: inp_npz[key] for key in inp_npz.files}
    if "gt_depth" not in inputs:
        return entries

    pred_depth = pred["depth"][..., 0] if pred["depth"].ndim == 4 else pred["depth"].squeeze(-1)
    gt_depth_pre = resize_float_stack(inputs["gt_depth"].astype(np.float32), pred_depth.shape[-2:])
    mask_pre = resize_bool_stack(inputs["counterfactual_target_mask"], pred_depth.shape[-2:]) if "counterfactual_target_mask" in inputs else None
    scale, _ = estimate_depth_scale(pred_depth, gt_depth_pre, mask_pre, seed=int(cfg["project"]["random_seed"]))
    frame_ids = inputs["frame_ids"].astype(int).tolist()
    pred_frames = [fid for fid in frames if fid in frame_ids]
    if not pred_frames:
        return entries

    for name, multiplier in (("pred_depth_gt_camera_raw", 1.0), ("pred_depth_gt_camera_metric_scaled", scale)):
        pts_parts = []
        cols_parts = []
        per_frame = max(1, max_points // max(len(pred_frames), 1))
        for offset, fid in enumerate(pred_frames):
            idx = frame_ids.index(fid)
            depth = pred_depth[idx].astype(np.float32) * np.float32(multiplier)
            pts = depth_to_world_points(depth, inputs["gt_intrinsic_preprocessed"].astype(np.float32), inputs["gt_cam2world"][idx].astype(np.float32))
            cols = resize_rgb(condition_dir(cfg, condition) / "rgb" / f"frame_{fid:04d}.png", depth.shape)
            spts, scols = sample_colored(pts, cols, per_frame, seed=int(cfg["project"]["random_seed"]) + offset + len(name))
            pts_parts.append(spts)
            cols_parts.append(scols)
        ply_path = out_dir / "ply" / f"{name}.ply"
        write_ply(ply_path, np.concatenate(pts_parts, axis=0), np.concatenate(cols_parts, axis=0))
        entries.append({"name": name, "path": str(ply_path), "frames": pred_frames, "depth_metric_scale": float(multiplier)})
    return entries


def run(cfg: dict[str, Any], condition: str, frames: list[int], write_ply_outputs: bool, max_points: int) -> None:
    out_dir = output_root(cfg) / "diagnostics" / "gt_geometry" / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    projection = projection_checks(cfg, condition, frames)
    gt_geometry = gt_depth_checks(cfg, condition, frames)
    pred_geometry, scale_info = prediction_depth_checks(cfg, condition, frames)
    ply_outputs = write_diagnostic_plys(cfg, condition, frames, out_dir, max_points) if write_ply_outputs else []
    summary = {
        "condition": condition,
        "frames": frames,
        "conclusion_hint": (
            "If converted_opencv projection matches masks and gt_depth floor z is near 0, camera GT/data capture is self-consistent. "
            "If pred_depth_raw floor z is far from 0 but pred_depth_metric_scaled is near 0, the qualitative PLY issue is model depth scale."
        ),
        "projection_checks": projection,
        "gt_depth_geometry_checks": gt_geometry,
        "prediction_depth_geometry_checks": pred_geometry,
        "prediction_depth_scale": scale_info,
        "ply_outputs": ply_outputs,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'summary.json'}")
    if scale_info:
        print(f"Pred depth metric scale: {scale_info.get('depth_metric_scale')}")
    if ply_outputs:
        print("PLY outputs:")
        for entry in ply_outputs:
            print(f"  {entry['name']}: {entry['path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose GT camera/depth geometry and reconstruction PLY scale.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    parser.add_argument("--condition", default="always_present")
    parser.add_argument("--run-id", default=None, help="Use an isolated revisit_memory/outputs/runs/<run-id> reconstruction.")
    parser.add_argument("--frames", nargs="+", type=int, default=[12, 23, 28, 39, 56, 67, 69, 74, 83])
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--max-points", type=int, default=60000)
    args = parser.parse_args()
    if args.run_id:
        os.environ["REVISIT_MEMORY_RUN_ID"] = args.run_id
    run(load_config(args.config), args.condition, args.frames, args.write_ply, args.max_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
