#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.io import load_config, output_root, write_csv, write_json  # noqa: E402


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("sanity_checks.py requires torch to load saved feature tensors.") from exc


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_payload(cfg: dict[str, Any], condition: str):
    torch = require_torch()
    path = output_root(cfg) / "features" / condition / "features.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file for sanity checks: {path}")
    return torch.load(path, map_location="cpu")


def load_memory_payload(cfg: dict[str, Any], condition: str):
    torch = require_torch()
    path = output_root(cfg) / "memory_tokens" / condition / "special_tokens.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing memory-token file for sanity checks: {path}")
    return torch.load(path, map_location="cpu")


def inclusive(start_end: list[int]) -> set[int]:
    return set(range(int(start_end[0]), int(start_end[1]) + 1))


def diff_stats(a, b) -> dict[str, float]:
    torch = require_torch()
    a = a.float()
    b = b.float()
    d = a - b
    l2 = torch.linalg.norm(d, dim=-1)
    ref_norm = torch.maximum(torch.linalg.norm(a, dim=-1), torch.linalg.norm(b, dim=-1)).clamp_min(1e-12)
    relative_l2 = l2 / ref_norm
    return {
        "mean_abs": float(torch.mean(torch.abs(d)).item()),
        "max_abs": float(torch.max(torch.abs(d)).item()),
        "mean_l2": float(torch.mean(l2).item()),
        "max_l2": float(torch.max(l2).item()),
        "mean_relative_l2": float(torch.mean(relative_l2).item()),
        "max_relative_l2": float(torch.max(relative_l2).item()),
        "num_values": int(d.numel()),
    }


def status_for(stats: dict[str, float], cfg: dict[str, Any]) -> str:
    thresholds = cfg["validation"]
    if stats["mean_relative_l2"] > float(thresholds["feature_close_mean_relative_l2_threshold"]):
        return "fail"
    if stats["max_relative_l2"] > float(thresholds["feature_close_max_relative_l2_threshold"]):
        if bool(thresholds.get("feature_equivalence_hard_fail_on_max_relative_l2", False)):
            return "fail"
        return "caveat"
    return "pass"


def feature_equivalence_rows(cfg: dict[str, Any], payloads: dict[str, Any]) -> list[dict[str, Any]]:
    torch = require_torch()
    first_loop = inclusive(cfg["frames"]["first_loop_visible"])
    thresholds = cfg["validation"]
    a = payloads["always_present"]
    b = payloads["seen_then_removed"]
    common_frames = sorted(set(a["frame_ids"]).intersection(b["frame_ids"]).intersection(first_loop))
    rows: list[dict[str, Any]] = []
    if not common_frames:
        return rows
    idx_a = [list(a["frame_ids"]).index(fid) for fid in common_frames]
    idx_b = [list(b["frame_ids"]).index(fid) for fid in common_frames]
    layers = [int(x) for x in a["selected_layers"]]
    for group in ["frame_block", "global_block", "frame_special_tokens", "global_special_tokens"]:
        for layer in layers:
            left = a[group][str(layer)][idx_a]
            right = b[group][str(layer)][idx_b]
            stats = diff_stats(left, right)
            rows.append(
                {
                    "check": "first_loop_feature_equivalence",
                    "comparison": "always_present_vs_seen_then_removed",
                    "group": group,
                    "layer": layer,
                    "frame_scope": f"{min(common_frames)}-{max(common_frames)}",
                    "num_frames": len(common_frames),
                    "threshold_mean_abs": float(thresholds["feature_equal_mean_abs_threshold"]),
                    "threshold_max_abs": float(thresholds["feature_equal_max_abs_threshold"]),
                    "threshold_mean_relative_l2": float(thresholds["feature_close_mean_relative_l2_threshold"]),
                    "threshold_max_relative_l2": float(thresholds["feature_close_max_relative_l2_threshold"]),
                    "status": status_for(stats, cfg),
                    **stats,
                }
            )
    _ = torch
    return rows


def memory_equivalence_rows(cfg: dict[str, Any], payloads: dict[str, Any]) -> list[dict[str, Any]]:
    first_prefix = set(range(0, int(cfg["frames"]["last_first_loop_visible"]) + 1))
    thresholds = cfg["validation"]
    a = payloads["always_present"]
    b = payloads["seen_then_removed"]
    index_a = {int(fid): idx for idx, fid in enumerate(a["source_frame_ids"])}
    index_b = {int(fid): idx for idx, fid in enumerate(b["source_frame_ids"])}
    common_sources = sorted(set(index_a).intersection(index_b).intersection(first_prefix))
    rows: list[dict[str, Any]] = []
    if not common_sources:
        return rows
    idx_a = [index_a[fid] for fid in common_sources]
    idx_b = [index_b[fid] for fid in common_sources]
    layers = [int(x) for x in a["selected_layers"]]
    for group in ["frame_special_tokens", "global_special_tokens"]:
        for layer in layers:
            left = a[group][str(layer)][idx_a]
            right = b[group][str(layer)][idx_b]
            stats = diff_stats(left, right)
            rows.append(
                {
                    "check": "first_prefix_memory_token_equivalence",
                    "comparison": "always_present_vs_seen_then_removed",
                    "group": group,
                    "layer": layer,
                    "frame_scope": f"{min(common_sources)}-{max(common_sources)}",
                    "num_frames": len(common_sources),
                    "threshold_mean_abs": float(thresholds["feature_equal_mean_abs_threshold"]),
                    "threshold_max_abs": float(thresholds["feature_equal_max_abs_threshold"]),
                    "threshold_mean_relative_l2": float(thresholds["feature_close_mean_relative_l2_threshold"]),
                    "threshold_max_relative_l2": float(thresholds["feature_close_max_relative_l2_threshold"]),
                    "status": status_for(stats, cfg),
                    **stats,
                }
            )
    return rows


def second_loop_info_rows(cfg: dict[str, Any], payloads: dict[str, Any]) -> list[dict[str, Any]]:
    second_loop = inclusive(cfg["frames"]["second_loop_eval"])
    a = payloads["seen_then_removed"]
    b = payloads["always_absent"]
    common_frames = sorted(set(a["frame_ids"]).intersection(b["frame_ids"]).intersection(second_loop))
    rows: list[dict[str, Any]] = []
    if not common_frames:
        return rows
    idx_a = [list(a["frame_ids"]).index(fid) for fid in common_frames]
    idx_b = [list(b["frame_ids"]).index(fid) for fid in common_frames]
    layers = [int(x) for x in a["selected_layers"]]
    for group in ["frame_block", "global_block"]:
        for layer in layers:
            left = a[group][str(layer)][idx_a]
            right = b[group][str(layer)][idx_b]
            rows.append(
                {
                    "check": "second_loop_feature_difference_info",
                    "comparison": "seen_then_removed_vs_always_absent",
                    "group": group,
                    "layer": layer,
                    "frame_scope": f"{min(common_frames)}-{max(common_frames)}",
                    "num_frames": len(common_frames),
                    "threshold_mean_abs": None,
                    "threshold_max_abs": None,
                    "threshold_mean_relative_l2": None,
                    "threshold_max_relative_l2": None,
                    "status": "info",
                    **diff_stats(left, right),
                }
            )
    return rows


def summarize_input_validation(root: Path) -> dict[str, Any]:
    validation = load_json_if_exists(root / "validation" / "input_validation.json")
    if validation is None:
        return {"status": "not_run", "path": str(root / "validation" / "input_validation.json")}

    def max_field(rows: list[dict[str, Any]], field: str) -> float | None:
        vals = [row[field] for row in rows if field in row]
        return float(max(vals)) if vals else None

    pair_checks = validation.get("pair_checks", {})
    first_rows = pair_checks.get("always_present_vs_seen_then_removed_frames_0_to_39", [])
    second_rows = pair_checks.get("seen_then_removed_vs_always_absent_frames_56_to_83", [])
    camera_rows = validation.get("camera_checks", {}).get("rows", [])
    return {
        "status": validation.get("status"),
        "errors": validation.get("errors", []),
        "warnings": validation.get("warnings", []),
        "first_loop_identical_input_max_mean_abs_rgb": max_field(first_rows, "mean_abs_rgb_difference"),
        "first_loop_identical_input_max_depth_abs": max_field(first_rows, "max_abs_gt_depth_difference"),
        "first_loop_target_mask_max_different_pixels": max_field(first_rows, "target_different_pixels"),
        "second_loop_identical_input_max_mean_abs_rgb": max_field(second_rows, "mean_abs_rgb_difference"),
        "second_loop_identical_input_max_depth_abs": max_field(second_rows, "max_abs_gt_depth_difference"),
        "camera_intrinsic_max_abs_delta": max_field(camera_rows, "intrinsic_max_abs_delta"),
        "camera_pose_max_abs_delta": max_field(camera_rows, "pose_max_abs_delta"),
        "local_window_check": validation.get("local_window_check", {}),
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    root = output_root(cfg)
    feature_payloads = {condition: load_feature_payload(cfg, condition) for condition in cfg["conditions"]}
    memory_payloads = {condition: load_memory_payload(cfg, condition) for condition in cfg["conditions"]}

    rows: list[dict[str, Any]] = []
    rows.extend(feature_equivalence_rows(cfg, feature_payloads))
    rows.extend(memory_equivalence_rows(cfg, memory_payloads))
    rows.extend(second_loop_info_rows(cfg, feature_payloads))

    errors = [
        f"{row['check']} {row['group']} layer {row['layer']} failed: "
        f"mean_relative_l2={row['mean_relative_l2']}, max_relative_l2={row['max_relative_l2']}"
        for row in rows
        if row["status"] == "fail"
    ]
    caveats = [
        f"{row['check']} {row['group']} layer {row['layer']} caveat: "
        f"mean_relative_l2={row['mean_relative_l2']}, max_relative_l2={row['max_relative_l2']}"
        for row in rows
        if row["status"] == "caveat"
    ]
    if not any(row["check"] == "first_loop_feature_equivalence" for row in rows):
        errors.append("No saved first-loop feature frames were available for feature equivalence sanity checks.")
    if not any(row["check"] == "first_prefix_memory_token_equivalence" for row in rows):
        errors.append("No first-prefix memory tokens were available for memory-token equivalence sanity checks.")

    input_summary = summarize_input_validation(root)
    if input_summary.get("status") not in {"pass", "not_run"}:
        errors.append("Input validation did not pass; inspect validation/input_validation.json.")

    metrics_path = root / "metrics" / "sanity_feature_equivalence.csv"
    write_csv(metrics_path, rows)
    status = "pass" if not errors and not caveats else "pass_with_caveats" if not errors else "fail"
    summary = {
        "status": status,
        "errors": errors,
        "caveats": caveats,
        "input_validation": input_summary,
        "asserted_feature_checks": [
            "always_present vs seen_then_removed feature tensors for saved first-loop frames",
            "always_present vs seen_then_removed special/context tokens through frame 39",
        ],
        "informational_checks": [
            "seen_then_removed vs always_absent second-loop feature differences are reported but not asserted equal",
        ],
        "metrics_csv": str(metrics_path),
        "num_rows": len(rows),
        "caveat_policy": (
            "Feature equivalence uses mean_relative_l2 as the hard gate. "
            "max_relative_l2 outliers are recorded as caveats unless "
            "validation.feature_equivalence_hard_fail_on_max_relative_l2 is true."
        ),
    }
    write_json(root / "validation" / "sanity_checks.json", summary)
    if errors:
        raise RuntimeError(f"Sanity checks failed with {len(errors)} issue(s). See {root / 'validation' / 'sanity_checks.json'}")
    print(f"Wrote {root / 'validation' / 'sanity_checks.json'}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run input/feature sanity checks for revisit-memory outputs.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
