#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.depth_scale import (  # noqa: E402
    collect_depth_ratios,
    estimate_pred_to_reference_scale,
    resize_bool_stack,
    resize_float_stack,
    squeeze_depth_stack,
)
from revisit_memory.utils.io import load_config, output_root, write_csv, write_json  # noqa: E402
from revisit_memory.utils.visualization import overlay_mask_outline, save_heatmap  # noqa: E402


def inclusive(start_end: list[int]) -> list[int]:
    return list(range(int(start_end[0]), int(start_end[1]) + 1))


def depth_scale_config(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "policy": "shared_anchor_background",
        "shared_anchor_background_frames": [0, 11],
        "exclude_target_mask": True,
        "min_unmasked_pixels": 1024,
        "max_samples": 5_000_000,
        "pair_background_alignment": True,
        "pair_background_reference_condition": "always_absent",
        "pair_background_source_condition": "seen_then_removed",
    }
    return {**defaults, **cfg.get("analysis", {}).get("depth_scale", {})}


def load_condition_depth_data(cfg: dict[str, Any], condition: str) -> dict[str, Any]:
    recon_dir = output_root(cfg) / "reconstruction" / condition
    with np.load(recon_dir / "predictions.npz", allow_pickle=True) as pred_npz:
        pred = {k: pred_npz[k] for k in pred_npz.files}
    with np.load(recon_dir / "inputs.npz", allow_pickle=True) as inp_npz:
        inputs = {k: inp_npz[k] for k in inp_npz.files}

    if "gt_depth" not in inputs:
        raise RuntimeError(f"{condition} inputs.npz does not contain gt_depth; rerun without --skip-gt-depth")

    frame_ids = inputs["frame_ids"].astype(int).tolist()
    pred_depth = squeeze_depth_stack(pred["depth"])
    h, w = pred_depth.shape[-2:]
    return {
        "condition": condition,
        "pred": pred,
        "inputs": inputs,
        "frame_ids": frame_ids,
        "frame_to_idx": {fid: idx for idx, fid in enumerate(frame_ids)},
        "pred_depth_raw": pred_depth.astype(np.float32),
        "gt_depth_resized": resize_float_stack(inputs["gt_depth"].astype(np.float32), (h, w)),
        "target_mask_resized": resize_bool_stack(inputs["counterfactual_target_mask"], (h, w)),
        "hw": (h, w),
    }


def frame_indices_for(data: dict[str, Any], frames: list[int]) -> list[int]:
    return [data["frame_to_idx"][fid] for fid in frames if fid in data["frame_to_idx"]]


def estimate_shared_anchor_scale(cfg: dict[str, Any], data_by_condition: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scale_cfg = depth_scale_config(cfg)
    selected_frames = inclusive(scale_cfg["shared_anchor_background_frames"])
    max_samples = int(scale_cfg["max_samples"])
    seed = int(cfg["project"]["random_seed"])
    ratio_parts = []
    per_condition = []

    for condition, data in data_by_condition.items():
        idx = frame_indices_for(data, selected_frames)
        if not idx:
            per_condition.append({"condition": condition, "frames": [], "valid_ratio_count": 0})
            continue
        exclude_mask = data["target_mask_resized"][idx] if bool(scale_cfg["exclude_target_mask"]) else None
        ratios, info = collect_depth_ratios(
            data["pred_depth_raw"][idx],
            data["gt_depth_resized"][idx],
            exclude_mask,
            eps=float(cfg["analysis"]["eps"]),
            min_unmasked_pixels=int(scale_cfg["min_unmasked_pixels"]),
            max_samples=None,
            seed=seed + len(condition),
        )
        if ratios.size:
            ratio_parts.append(ratios)
        per_condition.append(
            {
                "condition": condition,
                "frames": [data["frame_ids"][i] for i in idx],
                "condition_anchor_scale": float(np.nanmedian(ratios)) if ratios.size else 1.0,
                "ratio_p05": float(np.nanpercentile(ratios, 5)) if ratios.size else None,
                "ratio_p50": float(np.nanmedian(ratios)) if ratios.size else None,
                "ratio_p95": float(np.nanpercentile(ratios, 95)) if ratios.size else None,
                **info,
            }
        )

    pooled = np.concatenate(ratio_parts, axis=0) if ratio_parts else np.asarray([], dtype=np.float32)
    pooled_count_before_sampling = int(pooled.size)
    sampled = False
    if pooled.size > max_samples:
        rng = np.random.default_rng(seed)
        pooled = pooled[rng.choice(pooled.size, size=max_samples, replace=False)]
        sampled = True

    if pooled.size == 0:
        scale = 1.0
        scale_policy = "fallback_no_valid_shared_anchor_ratios"
        percentiles = {"ratio_p05": None, "ratio_p50": None, "ratio_p95": None}
    else:
        scale = float(np.nanmedian(pooled))
        scale_policy = "shared_anchor_background_median_gt_depth_over_pred_depth"
        percentiles = {
            "ratio_p05": float(np.nanpercentile(pooled, 5)),
            "ratio_p50": scale,
            "ratio_p95": float(np.nanpercentile(pooled, 95)),
        }

    return {
        "depth_metric_scale": scale,
        "scale_policy": scale_policy,
        "shared_anchor_background_frames": selected_frames,
        "conditions_pooled": list(data_by_condition),
        "exclude_target_mask": bool(scale_cfg["exclude_target_mask"]),
        "valid_ratio_count_before_sampling": pooled_count_before_sampling,
        "valid_ratio_count": int(pooled.size),
        "sampled": sampled,
        "max_samples": max_samples,
        "per_condition_anchor_diagnostics": per_condition,
        **percentiles,
    }


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(arr)[mask]
    return float(np.mean(vals)) if vals.size else float("nan")


def masked_percentile(arr: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    vals = np.asarray(arr)[mask]
    return float(np.percentile(vals, percentile)) if vals.size else float("nan")


def condition_metrics(
    cfg: dict[str, Any],
    condition: str,
    data: dict[str, Any],
    frames: list[int],
    depth_metric_scale: float,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], dict[int, np.ndarray]]:
    pred = data["pred"]
    frame_to_idx = data["frame_to_idx"]
    pred_depth = data["pred_depth_raw"] * np.float32(depth_metric_scale)
    rows: list[dict[str, Any]] = []
    residual_maps: dict[int, np.ndarray] = {}
    scaled_pred_maps: dict[int, np.ndarray] = {}
    eps = float(cfg["analysis"]["eps"])
    dmin = float(cfg["analysis"]["depth_valid_min"])
    dmax = float(cfg["analysis"]["depth_valid_max"])
    save_figures = bool(cfg["analysis"].get("save_depth_figures", True))

    fig_dir = output_root(cfg) / "analysis" / "depth" / condition
    if save_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    for fid in frames:
        idx = frame_to_idx[fid]
        gt = data["gt_depth_resized"][idx]
        mask = data["target_mask_resized"][idx]
        pred_d = pred_depth[idx].astype(np.float32)
        raw_pred_d = data["pred_depth_raw"][idx].astype(np.float32)
        valid = np.isfinite(gt) & np.isfinite(pred_d) & (gt > dmin) & (gt < dmax) & (pred_d > dmin)
        mask_valid = mask & valid
        outside_valid = (~mask) & valid
        abs_err = np.abs(pred_d - gt)
        signed = gt - pred_d
        residual_maps[fid] = signed
        scaled_pred_maps[fid] = pred_d
        row = {
            "condition": condition,
            "frame": fid,
            "depth_scale_applied": float(depth_metric_scale),
            "mask_pixel_count": int(mask_valid.sum()),
            "valid_pixel_count": int(valid.sum()),
            "depth_mae_in_mask": float(abs_err[mask_valid].mean()) if mask_valid.any() else np.nan,
            "depth_mae_outside_mask": float(abs_err[outside_valid].mean()) if outside_valid.any() else np.nan,
            "signed_depth_residual": float(signed[mask_valid].mean()) if mask_valid.any() else np.nan,
            "signed_depth_residual_outside_mask": float(signed[outside_valid].mean()) if outside_valid.any() else np.nan,
            "pred_depth_scaled_mean_in_mask": masked_mean(pred_d, mask_valid),
            "pred_depth_scaled_mean_outside_mask": masked_mean(pred_d, outside_valid),
            "pred_depth_raw_mean_in_mask": masked_mean(raw_pred_d, mask_valid),
            "gt_depth_mean_in_mask": masked_mean(gt, mask_valid),
        }
        rows.append(row)

        if save_figures:
            save_heatmap(fig_dir / f"frame_{fid:04d}_pred_depth.png", pred_d)
            save_heatmap(fig_dir / f"frame_{fid:04d}_gt_depth.png", gt)
            save_heatmap(fig_dir / f"frame_{fid:04d}_abs_error.png", abs_err)
            save_heatmap(fig_dir / f"frame_{fid:04d}_signed_residual.png", signed)
            if "images" in pred:
                rgb = np.transpose(pred["images"][idx], (1, 2, 0))
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                overlay_mask_outline(rgb, mask).save(fig_dir / f"frame_{fid:04d}_mask_overlay.png")

    return rows, residual_maps, scaled_pred_maps


def weighted_summary(rows: list[dict[str, Any]], key: str) -> float:
    vals = np.asarray([row[key] for row in rows], dtype=np.float64)
    weights = np.asarray([row["mask_pixel_count"] for row in rows], dtype=np.float64)
    valid = np.isfinite(vals) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.sum(vals[valid] * weights[valid]) / max(np.sum(weights[valid]), 1.0))


def available_eval_frames(cfg: dict[str, Any]) -> list[int]:
    start, end = cfg["frames"]["second_loop_eval"]
    requested = set(range(int(start), int(end) + 1))
    available_sets = []
    for condition in cfg["conditions"]:
        recon_dir = output_root(cfg) / "reconstruction" / condition
        with np.load(recon_dir / "inputs.npz", allow_pickle=True) as inp_npz:
            available_sets.append(set(inp_npz["frame_ids"].astype(int).tolist()))
    frames = sorted(set.intersection(*available_sets).intersection(requested))
    if not frames:
        raise RuntimeError("No common evaluated frames are present in this run.")
    return frames


def valid_comparison_mask(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray | None,
    cfg: dict[str, Any],
    *,
    positive_depth: bool = True,
) -> np.ndarray:
    dmin = float(cfg["analysis"]["depth_valid_min"])
    dmax = float(cfg["analysis"]["depth_valid_max"])
    valid = np.isfinite(a) & np.isfinite(b)
    if positive_depth:
        valid &= (a > dmin) & (a < dmax) & (b > dmin) & (b < dmax)
    if mask is not None:
        valid &= mask
    return valid


def depth_delta_stats(
    a: np.ndarray,
    b: np.ndarray,
    target_mask: np.ndarray,
    cfg: dict[str, Any],
    *,
    positive_depth: bool = True,
) -> dict[str, Any]:
    valid = valid_comparison_mask(a, b, None, cfg, positive_depth=positive_depth)
    inside = target_mask & valid
    outside = (~target_mask) & valid
    delta = a.astype(np.float32) - b.astype(np.float32)
    return {
        "mean_delta_all": masked_mean(delta, valid),
        "mean_abs_delta_all": masked_mean(np.abs(delta), valid),
        "p95_abs_delta_all": masked_percentile(np.abs(delta), valid, 95),
        "mean_delta_in_mask": masked_mean(delta, inside),
        "mean_abs_delta_in_mask": masked_mean(np.abs(delta), inside),
        "p95_abs_delta_in_mask": masked_percentile(np.abs(delta), inside, 95),
        "mean_delta_outside_mask": masked_mean(delta, outside),
        "mean_abs_delta_outside_mask": masked_mean(np.abs(delta), outside),
        "p95_abs_delta_outside_mask": masked_percentile(np.abs(delta), outside, 95),
        "mask_pixel_count": int(inside.sum()),
        "outside_pixel_count": int(outside.sum()),
        "valid_pixel_count": int(valid.sum()),
    }


def pair_background_alignment_row(
    cfg: dict[str, Any],
    frame: int,
    source_depth: np.ndarray,
    reference_depth: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    include_background = ~target_mask
    scale, info = estimate_pred_to_reference_scale(
        source_depth,
        reference_depth,
        include_background,
        eps=float(cfg["analysis"]["eps"]),
        min_valid_pixels=int(depth_scale_config(cfg)["min_unmasked_pixels"]),
        seed=int(cfg["project"]["random_seed"]) + frame,
    )
    aligned = source_depth * np.float32(scale)
    stats = depth_delta_stats(aligned, reference_depth, target_mask, cfg)
    return {
        "frame": frame,
        "background_alignment_scale_source_to_reference": float(scale),
        **{f"background_alignment_{k}": v for k, v in stats.items()},
        **{f"background_alignment_info_{k}": v for k, v in info.items()},
    }


def condition_difference_rows(
    cfg: dict[str, Any],
    data_by_condition: dict[str, dict[str, Any]],
    scaled_pred_maps: dict[str, dict[int, np.ndarray]],
    residuals: dict[str, dict[int, np.ndarray]],
    frames: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_condition = depth_scale_config(cfg)["pair_background_source_condition"]
    reference_condition = depth_scale_config(cfg)["pair_background_reference_condition"]
    source_data = data_by_condition[source_condition]
    reference_data = data_by_condition[reference_condition]
    for fid in frames:
        if fid not in scaled_pred_maps[source_condition] or fid not in scaled_pred_maps[reference_condition]:
            continue
        mask = source_data["target_mask_resized"][source_data["frame_to_idx"][fid]]
        source = scaled_pred_maps[source_condition][fid]
        reference = scaled_pred_maps[reference_condition][fid]
        pred_stats = depth_delta_stats(source, reference, mask, cfg)
        residual_stats = depth_delta_stats(
            residuals[source_condition][fid],
            residuals[reference_condition][fid],
            mask,
            cfg,
            positive_depth=False,
        )
        row: dict[str, Any] = {
            "comparison": f"{source_condition}_minus_{reference_condition}",
            "frame": fid,
            "source_condition": source_condition,
            "reference_condition": reference_condition,
            **{f"pred_depth_{k}": v for k, v in pred_stats.items()},
            **{f"signed_residual_{k}": v for k, v in residual_stats.items()},
        }
        if bool(depth_scale_config(cfg)["pair_background_alignment"]):
            row.update(pair_background_alignment_row(cfg, fid, source, reference, mask))
        rows.append(row)
    return rows


def equivalence_rows(
    cfg: dict[str, Any],
    data_by_condition: dict[str, dict[str, Any]],
    depth_metric_scale: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks = [
        {
            "check": "shared_anchor_background_scale_diagnostic",
            "condition_a": "seen_then_removed",
            "condition_b": "always_absent",
            "frames": inclusive(depth_scale_config(cfg)["shared_anchor_background_frames"]),
            "status": "diagnostic",
        },
        {
            "check": "first_loop_identical_prefix_depth",
            "condition_a": "always_present",
            "condition_b": "seen_then_removed",
            "frames": list(range(0, int(cfg["frames"]["last_first_loop_visible"]) + 1)),
            "status": "expected_close",
        },
        {
            "check": "second_loop_same_current_input_memory_probe_depth",
            "condition_a": "seen_then_removed",
            "condition_b": "always_absent",
            "frames": inclusive(cfg["frames"]["second_loop_eval"]),
            "status": "memory_probe_not_asserted_equal",
        },
    ]
    for check in checks:
        a_data = data_by_condition[check["condition_a"]]
        b_data = data_by_condition[check["condition_b"]]
        common_frames = [fid for fid in check["frames"] if fid in a_data["frame_to_idx"] and fid in b_data["frame_to_idx"]]
        if not common_frames:
            continue
        per_frame = []
        for fid in common_frames:
            a_idx = a_data["frame_to_idx"][fid]
            b_idx = b_data["frame_to_idx"][fid]
            mask = a_data["target_mask_resized"][a_idx]
            a_raw = a_data["pred_depth_raw"][a_idx]
            b_raw = b_data["pred_depth_raw"][b_idx]
            raw_stats = depth_delta_stats(a_raw, b_raw, mask, cfg)
            scaled_stats = depth_delta_stats(a_raw * np.float32(depth_metric_scale), b_raw * np.float32(depth_metric_scale), mask, cfg)
            per_frame.append((raw_stats, scaled_stats))
        raw_abs = [stats["mean_abs_delta_all"] for stats, _ in per_frame if np.isfinite(stats["mean_abs_delta_all"])]
        scaled_abs = [stats["mean_abs_delta_all"] for _, stats in per_frame if np.isfinite(stats["mean_abs_delta_all"])]
        raw_in = [stats["mean_abs_delta_in_mask"] for stats, _ in per_frame if np.isfinite(stats["mean_abs_delta_in_mask"])]
        scaled_in = [stats["mean_abs_delta_in_mask"] for _, stats in per_frame if np.isfinite(stats["mean_abs_delta_in_mask"])]
        rows.append(
            {
                "check": check["check"],
                "comparison": f"{check['condition_a']}_minus_{check['condition_b']}",
                "frame_scope": f"{min(common_frames)}-{max(common_frames)}",
                "num_frames": len(common_frames),
                "status": check["status"],
                "depth_metric_scale_applied": float(depth_metric_scale),
                "mean_abs_delta_raw_all": float(np.mean(raw_abs)) if raw_abs else float("nan"),
                "max_mean_abs_delta_raw_all": float(np.max(raw_abs)) if raw_abs else float("nan"),
                "mean_abs_delta_scaled_all": float(np.mean(scaled_abs)) if scaled_abs else float("nan"),
                "max_mean_abs_delta_scaled_all": float(np.max(scaled_abs)) if scaled_abs else float("nan"),
                "mean_abs_delta_raw_in_mask": float(np.mean(raw_in)) if raw_in else float("nan"),
                "mean_abs_delta_scaled_in_mask": float(np.mean(scaled_in)) if scaled_in else float("nan"),
            }
        )
    return rows


def run(cfg: dict[str, Any]) -> None:
    frames = available_eval_frames(cfg)
    data_by_condition = {condition: load_condition_depth_data(cfg, condition) for condition in cfg["conditions"]}
    shared_scale_info = estimate_shared_anchor_scale(cfg, data_by_condition)
    depth_metric_scale = float(shared_scale_info["depth_metric_scale"])
    all_rows: list[dict[str, Any]] = []
    residuals: dict[str, dict[int, np.ndarray]] = {}
    scaled_preds: dict[str, dict[int, np.ndarray]] = {}
    for condition in cfg["conditions"]:
        rows, maps, pred_maps = condition_metrics(cfg, condition, data_by_condition[condition], frames, depth_metric_scale)
        all_rows.extend(rows)
        residuals[condition] = maps
        scaled_preds[condition] = pred_maps

    diff_rows = condition_difference_rows(cfg, data_by_condition, scaled_preds, residuals, frames)
    diff_dir = output_root(cfg) / "analysis" / "depth" / "condition_differences"
    save_figures = bool(cfg["analysis"].get("save_depth_figures", True))
    if save_figures:
        diff_dir.mkdir(parents=True, exist_ok=True)
        for fid in frames:
            if fid in residuals["seen_then_removed"] and fid in residuals["always_absent"]:
                save_heatmap(
                    diff_dir / f"frame_{fid:04d}_seen_removed_minus_absent_signed_depth.png",
                    residuals["seen_then_removed"][fid] - residuals["always_absent"][fid],
                )
            save_heatmap(
                diff_dir / f"frame_{fid:04d}_seen_removed_minus_absent_scaled_pred_depth.png",
                scaled_preds["seen_then_removed"][fid] - scaled_preds["always_absent"][fid],
            )

    metrics_dir = output_root(cfg) / "metrics"
    write_csv(metrics_dir / "depth_metrics.csv", all_rows)
    write_csv(metrics_dir / "depth_condition_differences.csv", diff_rows)
    write_csv(metrics_dir / "depth_scale_equivalence.csv", equivalence_rows(cfg, data_by_condition, depth_metric_scale))
    summary = {}
    for condition in cfg["conditions"]:
        rows = [row for row in all_rows if row["condition"] == condition]
        summary[condition] = {
            "weighted_depth_mae_in_mask": weighted_summary(rows, "depth_mae_in_mask"),
            "weighted_depth_mae_outside_mask": weighted_summary(rows, "depth_mae_outside_mask"),
            "weighted_signed_depth_residual": weighted_summary(rows, "signed_depth_residual"),
            "total_mask_pixels": int(sum(row["mask_pixel_count"] for row in rows)),
        }
    deltas = [row["pred_depth_mean_delta_in_mask"] for row in diff_rows if np.isfinite(row["pred_depth_mean_delta_in_mask"])]
    aligned_deltas = [
        row["background_alignment_mean_delta_in_mask"]
        for row in diff_rows
        if "background_alignment_mean_delta_in_mask" in row and np.isfinite(row["background_alignment_mean_delta_in_mask"])
    ]
    summary["seen_removed_minus_absent"] = {
        "mean_scaled_pred_depth_delta_in_mask": float(np.mean(deltas)) if deltas else float("nan"),
        "mean_background_aligned_pred_depth_delta_in_mask": float(np.mean(aligned_deltas)) if aligned_deltas else float("nan"),
        "delta_sign_note": "positive means seen_then_removed predicts larger/farther depth than always_absent in the target mask.",
    }
    write_json(output_root(cfg) / "analysis" / "depth" / "shared_depth_scale.json", shared_scale_info)
    write_json(
        output_root(cfg) / "analysis" / "depth" / "summary.json",
        {
            "depth_scale": shared_scale_info,
            "evaluated_frames": frames,
            "condition_summaries": summary,
            "condition_difference_metrics": str(metrics_dir / "depth_condition_differences.csv"),
            "scale_equivalence_metrics": str(metrics_dir / "depth_scale_equivalence.csv"),
        },
    )
    print(f"Wrote {metrics_dir / 'depth_metrics.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Depth residual analysis for revisit-memory.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
