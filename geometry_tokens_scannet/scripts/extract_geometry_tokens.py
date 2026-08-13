#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import (
    build_patch_features,
    deep_get,
    deep_set,
    discover_paths,
    fit_ridge_model,
    frame_output_name,
    infer_patch_grid,
    load_config,
    load_manifest_or_metadata,
    overlay_heatmap,
    overlay_mask,
    predict_ridge,
    read_intrinsic,
    robust_normalize,
    save_json,
    save_npy,
    save_rgb,
    select_frames,
    top_percent_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract depth-residual Geometry Tokens on ScanNet frames.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--scene-id", default=None, help="Override controlled.scene_id.")
    parser.add_argument("--setting-name", default=None, help="Override controlled.setting_name.")
    parser.add_argument("--condition-id", default=None, help="Override controlled.condition_id.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for debugging.")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.scene_id:
        deep_set(cfg, "controlled.scene_id", args.scene_id)
    if args.setting_name:
        deep_set(cfg, "controlled.setting_name", args.setting_name)
    if args.condition_id:
        deep_set(cfg, "controlled.condition_id", args.condition_id)
    if args.max_frames is not None:
        deep_set(cfg, "frames.max_frames", args.max_frames)


def residual_score(depth: np.ndarray, pred: np.ndarray, valid: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    mode = str(deep_get(cfg, "depth.residual", "absolute")).lower()
    if mode in {"square", "squared", "l2"}:
        residual = (depth - pred) ** 2
    else:
        residual = np.abs(depth - pred)
    residual = np.where(valid, residual, np.nan).astype(np.float32)
    norm_mode = str(deep_get(cfg, "depth.per_frame_normalization", "percentile")).lower()
    if norm_mode == "none":
        score = np.nan_to_num(residual, nan=0.0)
    else:
        score = robust_normalize(
            residual,
            valid,
            lo=float(deep_get(cfg, "depth.percentile_low", 5.0)),
            hi=float(deep_get(cfg, "depth.percentile_high", 95.0)),
        )
    score = np.where(valid, score, 0.0).astype(np.float32)
    return residual.astype(np.float32), score


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    paths = discover_paths(cfg)
    manifest = load_manifest_or_metadata(paths, cfg)
    frames = select_frames(manifest.get("frames", []), deep_get(cfg, "frames.max_frames"))
    if not frames:
        raise RuntimeError("No frames found in manifest or fallback metadata.")

    grid = infer_patch_grid(cfg, paths, frames)
    out_dir = paths["scene_output"] / "geometry"
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    all_X = []
    all_y = []
    feature_names: list[str] | None = None
    skipped = []
    for frame in frames:
        rgb_path = Path(frame.get("rgb_path", ""))
        depth_path = Path(frame.get("depth_path", ""))
        if not rgb_path.exists() or not depth_path.exists():
            skipped.append(
                {
                    "t": frame.get("t"),
                    "source_frame_id": frame.get("source_frame_id"),
                    "rgb_exists": rgb_path.exists(),
                    "depth_exists": depth_path.exists(),
                }
            )
            continue
        rec = build_patch_features(frame, grid, cfg)
        valid_flat = rec["valid_patch"].reshape(-1)
        if np.any(valid_flat):
            all_X.append(rec["feature"][valid_flat])
            all_y.append(rec["depth_patch"].reshape(-1)[valid_flat])
        if feature_names is None:
            feature_names = list(rec["feature_names"])
        rec["frame"] = frame
        records.append(rec)

    if not records:
        raise RuntimeError("No usable RGB/depth frames were found.")
    if not all_X:
        raise RuntimeError("No valid depth patches were found for Ridge training.")

    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    sequence_model = fit_ridge_model(X_train, y_train, cfg)
    train_scope = str(deep_get(cfg, "ridge.train_scope", "sequence")).lower()

    top_percentiles = [int(x) for x in deep_get(cfg, "geometry_tokens.top_percentiles", [5, 10, 20])]
    default_top = int(deep_get(cfg, "geometry_tokens.default_top_percentile", 10))
    cmap_name = str(deep_get(cfg, "visualization.heatmap_cmap", "magma"))

    score_stack = []
    residual_stack = []
    depth_stack = []
    pred_stack = []
    valid_stack = []
    mask_stacks = {top: [] for top in top_percentiles}
    frame_index_map = []

    for rec in records:
        frame = rec["frame"]
        h, w = grid
        valid = rec["valid_patch"]
        model = sequence_model
        if train_scope == "per_frame" and np.count_nonzero(valid) > rec["feature"].shape[1] + 2:
            valid_flat = valid.reshape(-1)
            model = fit_ridge_model(rec["feature"][valid_flat], rec["depth_patch"].reshape(-1)[valid_flat], cfg)
        pred = predict_ridge(model, rec["feature"]).reshape(h, w).astype(np.float32)
        residual, score = residual_score(rec["depth_patch"], pred, valid, cfg)
        masks = {top: top_percent_mask(score, valid, top) for top in top_percentiles}

        frame_name = frame_output_name(frame)
        fdir = frames_dir / frame_name
        save_npy(fdir / "geometry_score.npy", score)
        save_npy(fdir / "depth_residual.npy", residual)
        save_npy(fdir / "depth_gt_patch.npy", rec["depth_patch"])
        save_npy(fdir / "depth_pred_patch.npy", pred)
        save_npy(fdir / "valid_depth_mask.npy", valid.astype(bool))
        for top, mask in masks.items():
            save_npy(fdir / f"geometry_mask_top{top}.npy", mask.astype(bool))

        rgb = rec["rgb"]
        save_rgb(fdir / "rgb.png", rgb)
        save_rgb(fdir / "depth_residual_heatmap.png", overlay_heatmap(np.ones_like(rgb), score, alpha=1.0, cmap_name=cmap_name))
        save_rgb(fdir / "geometry_score_overlay.png", overlay_heatmap(rgb, score, alpha=0.45, cmap_name=cmap_name))
        if default_top in masks:
            save_rgb(
                fdir / f"geometry_mask_top{default_top}_overlay.png",
                overlay_mask(rgb, masks[default_top], deep_get(cfg, "visualization.geometry_color", [255, 64, 64])),
            )

        score_stack.append(score)
        residual_stack.append(residual)
        depth_stack.append(rec["depth_patch"].astype(np.float32))
        pred_stack.append(pred)
        valid_stack.append(valid.astype(bool))
        for top in top_percentiles:
            mask_stacks[top].append(masks[top].astype(bool))
        frame_index_map.append(
            {
                "stack_index": len(score_stack) - 1,
                "t": int(frame.get("t", len(score_stack) - 1)),
                "source_frame_id": str(frame.get("source_frame_id", "")),
                "rgb_path": str(frame.get("rgb_path", "")),
                "depth_path": str(frame.get("depth_path", "")),
                "frame_output": str(fdir),
            }
        )

    score_arr = np.stack(score_stack, axis=0).astype(np.float32)
    save_npy(out_dir / "geometry_score.npy", score_arr)
    save_npy(out_dir / "depth_residual.npy", np.stack(residual_stack, axis=0).astype(np.float32))
    save_npy(out_dir / "depth_gt_patch.npy", np.stack(depth_stack, axis=0).astype(np.float32))
    save_npy(out_dir / "depth_pred_patch.npy", np.stack(pred_stack, axis=0).astype(np.float32))
    save_npy(out_dir / "valid_depth_mask.npy", np.stack(valid_stack, axis=0).astype(bool))
    for top, masks in mask_stacks.items():
        save_npy(out_dir / f"geometry_mask_top{top}.npy", np.stack(masks, axis=0).astype(bool))

    first_intrinsic = None
    for frame in frames:
        first_intrinsic = read_intrinsic(frame.get("intrinsic_path"))
        if first_intrinsic is not None:
            break

    metadata = {
        "status": "ok",
        "scene_id": manifest.get("scene_id", deep_get(cfg, "controlled.scene_id")),
        "setting_name": manifest.get("setting_name", deep_get(cfg, "controlled.setting_name")),
        "condition_id": manifest.get("condition_id", deep_get(cfg, "controlled.condition_id")),
        "manifest": str(paths["manifest"]),
        "token_dir": str(paths["token_dir"]),
        "patch_grid": list(grid),
        "frame_count": len(records),
        "skipped_frames": skipped,
        "feature_names": feature_names or [],
        "ridge": {
            "backend": sequence_model.get("backend"),
            "alpha": sequence_model.get("alpha"),
            "cv_mse": sequence_model.get("cv_mse"),
            "train_scope": train_scope,
            "train_patch_count": int(len(y_train)),
        },
        "depth": {
            "residual": deep_get(cfg, "depth.residual", "absolute"),
            "per_frame_normalization": deep_get(cfg, "depth.per_frame_normalization", "percentile"),
            "depth_scale": deep_get(cfg, "depth.depth_scale", 1000.0),
        },
        "intrinsic_first_available": None if first_intrinsic is None else first_intrinsic.tolist(),
        "outputs": {
            "geometry_score": str(out_dir / "geometry_score.npy"),
            "frame_index_map": str(out_dir / "frame_index_map.json"),
            "frames_dir": str(frames_dir),
        },
    }
    save_json(frame_index_map, out_dir / "frame_index_map.json")
    save_json(metadata, out_dir / "metadata.json")
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "frame_count": len(records), "patch_grid": list(grid)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"extract_geometry_tokens failed: {exc}", file=sys.stderr)
        raise
