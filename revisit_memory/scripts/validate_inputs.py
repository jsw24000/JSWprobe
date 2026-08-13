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

from revisit_memory.utils.io import (  # noqa: E402
    camera_json_path,
    condition_dir,
    depth_path,
    find_counterfactual_mask,
    find_target_presence_mask,
    list_frame_ids,
    load_any_target_mask,
    load_config,
    load_depth_exr,
    load_rgb,
    metadata_path,
    output_root,
    read_json,
    rgb_path,
    sha256_file,
    target_object,
    write_json,
)


def inclusive(start_end: list[int]) -> range:
    return range(int(start_end[0]), int(start_end[1]) + 1)


def max_abs_matrix_delta(a: Any, b: Any) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def compare_rgb(cfg: dict[str, Any], cond_a: str, cond_b: str, frame_id: int) -> dict[str, Any]:
    a = load_rgb(rgb_path(cfg, cond_a, frame_id)).astype(np.int16)
    b = load_rgb(rgb_path(cfg, cond_b, frame_id)).astype(np.int16)
    diff = np.abs(a - b)
    return {
        "frame": frame_id,
        "sha256_a": sha256_file(rgb_path(cfg, cond_a, frame_id)),
        "sha256_b": sha256_file(rgb_path(cfg, cond_b, frame_id)),
        "hash_equal": sha256_file(rgb_path(cfg, cond_a, frame_id)) == sha256_file(rgb_path(cfg, cond_b, frame_id)),
        "mean_abs_rgb_difference": float(diff.mean()),
        "max_abs_rgb_difference": float(diff.max()),
    }


def compare_depth(cfg: dict[str, Any], cond_a: str, cond_b: str, frame_id: int) -> dict[str, Any]:
    a = load_depth_exr(depth_path(cfg, cond_a, frame_id))
    b = load_depth_exr(depth_path(cfg, cond_b, frame_id))
    diff = np.abs(a - b)
    return {
        "frame": frame_id,
        "mean_abs_gt_depth_difference": float(np.nanmean(diff)),
        "max_abs_gt_depth_difference": float(np.nanmax(diff)),
    }


def compare_mask(cfg: dict[str, Any], cond_a: str, cond_b: str, frame_id: int, target_id: int) -> dict[str, Any]:
    path_a = find_target_presence_mask(cfg, cond_a, frame_id, target_id) or find_counterfactual_mask(cfg, cond_a, frame_id)
    path_b = find_target_presence_mask(cfg, cond_b, frame_id, target_id) or find_counterfactual_mask(cfg, cond_b, frame_id)
    if path_a is None or path_b is None:
        return {"frame": frame_id, "mask_checked": False, "reason": "missing mask"}
    a = load_any_target_mask(path_a, target_id)
    b = load_any_target_mask(path_b, target_id)
    return {
        "frame": frame_id,
        "mask_checked": True,
        "mask_a": str(path_a),
        "mask_b": str(path_b),
        "pixel_count_a": int(a.sum()),
        "pixel_count_b": int(b.sum()),
        "different_pixels": int(np.count_nonzero(a != b)),
    }


def camera_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    conditions = cfg["conditions"]
    cameras = {cond: read_json(camera_json_path(cfg, cond)) for cond in conditions}
    ref = cameras[conditions[0]]
    rows = []
    for cond in conditions[1:]:
        cur = cameras[cond]
        intrinsic_delta = max_abs_matrix_delta(ref["intrinsic"]["K"], cur["intrinsic"]["K"])
        pose_delta = 0.0
        for a, b in zip(ref["frames"], cur["frames"]):
            pose_delta = max(pose_delta, max_abs_matrix_delta(a["cam2world"], b["cam2world"]))
            pose_delta = max(pose_delta, max_abs_matrix_delta(a["world2cam"], b["world2cam"]))
        rows.append({"condition": cond, "intrinsic_max_abs_delta": intrinsic_delta, "pose_max_abs_delta": pose_delta})
    return {"comparisons_to": conditions[0], "rows": rows}


def mask_pixel_count(cfg: dict[str, Any], condition: str, frame_id: int, target_id: int) -> int:
    path = find_target_presence_mask(cfg, condition, frame_id, target_id) or find_counterfactual_mask(cfg, condition, frame_id)
    if path is None:
        return 0
    return int(load_any_target_mask(path, target_id).sum())


def validate(cfg: dict[str, Any], skip_depth: bool = False) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    target = target_object(cfg)
    target_id = int(target["object_id"])
    thresholds = cfg["validation"]
    frame_cfg = cfg["frames"]
    model_cfg = cfg["model"]

    result: dict[str, Any] = {
        "data_root": str(condition_dir(cfg, cfg["conditions"][0]).parent),
        "target_object": target,
        "conditions": {},
        "pair_checks": {},
        "camera_checks": camera_checks(cfg),
        "local_window_check": {},
    }

    expected_frames = list(range(int(frame_cfg["total_frames"])))
    for cond in cfg["conditions"]:
        frames = list_frame_ids(cfg, cond)
        result["conditions"][cond] = {
            "condition_dir": str(condition_dir(cfg, cond)),
            "metadata": str(metadata_path(cfg, cond)),
            "frame_count": len(frames),
            "first_frame": frames[0] if frames else None,
            "last_frame": frames[-1] if frames else None,
        }
        if frames != expected_frames:
            errors.append(f"{cond}: expected frames {expected_frames[0]}..{expected_frames[-1]}, got {frames[:3]}..{frames[-3:]}")

    first_loop_rows = []
    for fid in range(0, int(frame_cfg["last_first_loop_visible"]) + 1):
        row = compare_rgb(cfg, "always_present", "seen_then_removed", fid)
        if row["mean_abs_rgb_difference"] > thresholds["rgb_mean_abs_threshold"] or row["max_abs_rgb_difference"] > thresholds["rgb_max_abs_threshold"]:
            errors.append(f"first-loop RGB mismatch at frame {fid}: {row}")
        if not skip_depth:
            try:
                depth_row = compare_depth(cfg, "always_present", "seen_then_removed", fid)
                row.update(depth_row)
                if depth_row["mean_abs_gt_depth_difference"] > thresholds["depth_mean_abs_threshold"] or depth_row["max_abs_gt_depth_difference"] > thresholds["depth_max_abs_threshold"]:
                    errors.append(f"first-loop depth mismatch at frame {fid}: {depth_row}")
            except Exception as exc:
                errors.append(f"could not check first-loop depth at frame {fid}: {exc}")
        mask_row = compare_mask(cfg, "always_present", "seen_then_removed", fid, target_id)
        row.update({f"target_{k}": v for k, v in mask_row.items() if k != "frame"})
        if mask_row.get("mask_checked") and mask_row.get("different_pixels", 0) != 0:
            errors.append(f"first-loop target mask mismatch at frame {fid}: {mask_row}")
        first_loop_rows.append(row)

    second_loop_rows = []
    for fid in inclusive(frame_cfg["second_loop_eval"]):
        row = compare_rgb(cfg, "seen_then_removed", "always_absent", fid)
        if row["mean_abs_rgb_difference"] > thresholds["rgb_mean_abs_threshold"] or row["max_abs_rgb_difference"] > thresholds["rgb_max_abs_threshold"]:
            errors.append(f"second-loop RGB mismatch at frame {fid}: {row}")
        if not skip_depth:
            try:
                depth_row = compare_depth(cfg, "seen_then_removed", "always_absent", fid)
                row.update(depth_row)
                if depth_row["mean_abs_gt_depth_difference"] > thresholds["depth_mean_abs_threshold"] or depth_row["max_abs_gt_depth_difference"] > thresholds["depth_max_abs_threshold"]:
                    errors.append(f"second-loop depth mismatch at frame {fid}: {depth_row}")
            except Exception as exc:
                errors.append(f"could not check second-loop depth at frame {fid}: {exc}")
        second_loop_rows.append(row)

    result["pair_checks"]["always_present_vs_seen_then_removed_frames_0_to_39"] = first_loop_rows
    result["pair_checks"]["seen_then_removed_vs_always_absent_frames_56_to_83"] = second_loop_rows

    for row in result["camera_checks"]["rows"]:
        if row["intrinsic_max_abs_delta"] > thresholds["intrinsic_abs_threshold"]:
            errors.append(f"camera intrinsic mismatch: {row}")
        if row["pose_max_abs_delta"] > thresholds["pose_abs_threshold"]:
            errors.append(f"camera pose mismatch: {row}")

    anchor_start, anchor_end = frame_cfg["anchor_frames"]
    anchor_counts = {
        str(fid): mask_pixel_count(cfg, "always_present", fid, target_id)
        for fid in range(int(anchor_start), int(anchor_end) + 1)
    }
    local_window_size = int(model_cfg["local_window_size"])
    revisit = int(frame_cfg["expected_first_revisit_frame"])
    last_visible = int(frame_cfg["last_first_loop_visible"])
    history_window = list(range(max(0, revisit - local_window_size), revisit))
    result["local_window_check"] = {
        "assertion": f"{revisit} - {last_visible} = {revisit - last_visible} > local_window_size = {local_window_size}",
        "passes_gap_check": revisit - last_visible > local_window_size,
        "frame_56_history_window_source_frame_ids": history_window,
        "target_visible_frames": list(inclusive(frame_cfg["first_loop_visible"])),
        "target_visible_frames_in_history_window": sorted(set(history_window).intersection(inclusive(frame_cfg["first_loop_visible"]))),
        "anchor_mask_pixel_counts": anchor_counts,
        "anchor_contains_target": any(count > 0 for count in anchor_counts.values()),
    }
    if not result["local_window_check"]["passes_gap_check"]:
        errors.append("local-window gap check failed")
    if result["local_window_check"]["anchor_contains_target"]:
        errors.append(f"anchor frames contain target pixels: {anchor_counts}")

    if skip_depth:
        warnings.append("Depth EXR comparison skipped by --skip-depth.")

    result["errors"] = errors
    result["warnings"] = warnings
    result["status"] = "pass" if not errors else "fail"
    return result, errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paired revisit-memory inputs.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    parser.add_argument("--skip-depth", action="store_true", help="Skip EXR comparisons when no EXR reader is installed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result, errors, warnings = validate(cfg, skip_depth=args.skip_depth)
    out_path = output_root(cfg) / "validation" / "input_validation.json"
    write_json(out_path, result)

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAILED input validation with {len(errors)} error(s). See {out_path}")
        for error in errors[:10]:
            print(f"- {error}")
        return 1
    print(f"PASS input validation. Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

