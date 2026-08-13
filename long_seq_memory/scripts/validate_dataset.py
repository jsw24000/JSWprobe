#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.dataset import first_image_size, load_scene
from long_seq_memory.io_utils import ensure_dir, load_config, write_json
from long_seq_memory.plotting import (
    plot_trajectory_3d,
    plot_trajectory_xy,
    save_frame_jpg,
    save_frame_pair_comparison,
)
from long_seq_memory.trajectory_metrics import (
    loop_pair_metrics,
    pose_validity,
    trajectory_continuity,
)


def check_image_sequence(scene, expected_frames: int) -> dict:
    names = [p.name for p in scene.image_paths]
    expected_names = [f"{i:06d}.png" for i in range(len(names))]
    mismatches = [
        {"index": i, "found": found, "expected": expected}
        for i, (found, expected) in enumerate(zip(names, expected_names))
        if found != expected
    ]
    return {
        "count": len(scene.image_paths),
        "expected_count": expected_frames,
        "count_ok": len(scene.image_paths) == expected_frames,
        "sequential_zero_padded_png_names": len(mismatches) == 0,
        "first_mismatches": mismatches[:10],
        "first_image_size_wh": first_image_size(scene.image_paths),
    }


def check_intrinsics(scene) -> dict:
    K = scene.intrinsics
    image_size = first_image_size(scene.image_paths)
    principal_inside = 0 <= K.cx <= K.width and 0 <= K.cy <= K.height
    first_image_matches = image_size == (K.width, K.height)
    return {
        "fx": K.fx,
        "fy": K.fy,
        "cx": K.cx,
        "cy": K.cy,
        "width": K.width,
        "height": K.height,
        "positive_focal_lengths": K.fx > 0 and K.fy > 0,
        "principal_point_inside_image": principal_inside,
        "first_image_size_matches_intrinsics": first_image_matches,
        "first_image_size_wh": image_size,
    }


def check_symlinks(scene_root: Path) -> dict:
    groups = ["calibration", "gt_trajectory", "lidar_tls", "timestamp_sync"]
    entries = []
    for group in groups:
        group_dir = scene_root / group
        if not group_dir.exists():
            entries.append({"path": str(group_dir), "exists": False, "is_symlink": False, "target_exists": False})
            continue
        for path in sorted(group_dir.iterdir()):
            if path.name.startswith("."):
                continue
            entries.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "is_symlink": path.is_symlink(),
                    "target": os.readlink(path) if path.is_symlink() else None,
                    "target_exists": path.exists(),
                }
            )
    return {
        "entries": entries,
        "all_expected_links_exist": all(item["target_exists"] for item in entries),
    }


def check_timestamp_sync(scene, expected_frames: int) -> dict:
    rows = scene.frame_sync
    frame_ids = [row.get("frame_idx") for row in rows]
    image_timestamps = np.array([row.get("image_timestamp", np.nan) for row in rows], dtype=np.float64) if rows else np.array([])
    gt_diffs = np.array([abs(row.get("gt_time_diff_s", np.nan)) for row in rows], dtype=np.float64) if rows else np.array([])
    slam_diffs = np.array([abs(row.get("slam_time_diff_s", np.nan)) for row in rows], dtype=np.float64) if rows else np.array([])
    lidar_diffs = np.array([abs(row.get("lidar_time_diff_s", np.nan)) for row in rows], dtype=np.float64) if rows else np.array([])
    return {
        "row_count": len(rows),
        "expected_count": expected_frames,
        "row_count_ok": len(rows) == expected_frames,
        "frame_idx_sequential": frame_ids == list(range(len(rows))),
        "image_timestamp_strictly_increasing": bool(np.all(np.diff(image_timestamps) > 0)) if len(image_timestamps) > 1 else None,
        "max_abs_gt_time_diff_s": float(np.nanmax(gt_diffs)) if len(gt_diffs) else None,
        "max_abs_slam_time_diff_s": float(np.nanmax(slam_diffs)) if len(slam_diffs) else None,
        "max_abs_lidar_time_diff_s": float(np.nanmax(lidar_diffs)) if len(lidar_diffs) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "configs" / "dataset_keble02.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    scene = load_scene(cfg["dataset_root"])
    expected_frames = int(cfg["sequence"]["expected_frames"])
    out_dir = ensure_dir(cfg["validation"]["output_dir"])

    image_check = check_image_sequence(scene, expected_frames)
    pose_check = {
        "count": int(scene.poses_c2w.shape[0]),
        "expected_count": expected_frames,
        "count_ok": int(scene.poses_c2w.shape[0]) == expected_frames,
        **pose_validity(scene.poses_c2w),
    }
    intrinsics_check = check_intrinsics(scene)
    continuity = trajectory_continuity(scene.poses_c2w)
    timestamp_check = check_timestamp_sync(scene, expected_frames)
    symlink_check = check_symlinks(scene.root)

    history_frame = int(cfg["loop_event"]["history_frame"])
    current_frame = int(cfg["loop_event"]["current_frame"])
    loop_metrics = loop_pair_metrics(scene.poses_c2w, history_frame, current_frame)
    ts_i = scene.timestamp(history_frame)
    ts_j = scene.timestamp(current_frame)
    loop_metrics.update(
        {
            "index_base": 0,
            "image_path_i": str(scene.image_path(history_frame)),
            "image_path_j": str(scene.image_path(current_frame)),
            "timestamp_i": ts_i,
            "timestamp_j": ts_j,
            "time_gap_s": None if ts_i is None or ts_j is None else float(ts_j - ts_i),
            "frame_gap": int(current_frame - history_frame),
            "expected_center_distance_m": cfg["loop_event"]["expected_center_distance_m"],
        }
    )

    tolerance = float(cfg["validation"]["center_distance_tolerance_m"])
    loop_metrics["center_distance_matches_expected"] = (
        abs(loop_metrics["center_distance_m"] - float(cfg["loop_event"]["expected_center_distance_m"])) <= tolerance
    )

    plot_trajectory_xy(
        scene.poses_c2w,
        out_dir / "trajectory_gt_xy.png",
        {history_frame: "history", current_frame: "current"},
    )
    plot_trajectory_3d(
        scene.poses_c2w,
        out_dir / "trajectory_gt_3d.png",
        {history_frame: "history", current_frame: "current"},
    )
    save_frame_jpg(scene.image_path(history_frame), out_dir / f"frame_{history_frame}.jpg")
    save_frame_jpg(scene.image_path(current_frame), out_dir / f"frame_{current_frame}.jpg")
    save_frame_pair_comparison(
        scene.image_path(history_frame),
        scene.image_path(current_frame),
        out_dir / "frame_pair_comparison.png",
        f"frame {history_frame}",
        f"frame {current_frame}",
    )

    max_orth = float(cfg["validation"]["max_rotation_orthonormal_error"])
    max_det = float(cfg["validation"]["max_rotation_det_error"])
    validation = {
        "config": cfg["_config_path"],
        "dataset_root": str(scene.root),
        "index_base": 0,
        "images": image_check,
        "poses": pose_check,
        "intrinsics": intrinsics_check,
        "trajectory_continuity": continuity,
        "timestamp_sync": timestamp_check,
        "symlinks": symlink_check,
        "loop_pair": loop_metrics,
    }
    validation["passed"] = bool(
        image_check["count_ok"]
        and image_check["sequential_zero_padded_png_names"]
        and pose_check["count_ok"]
        and pose_check["finite"]
        and pose_check["max_rotation_orthonormal_error"] <= max_orth
        and pose_check["max_rotation_det_error"] <= max_det
        and intrinsics_check["positive_focal_lengths"]
        and intrinsics_check["principal_point_inside_image"]
        and timestamp_check["row_count_ok"]
        and timestamp_check["frame_idx_sequential"]
        and symlink_check["all_expected_links_exist"]
        and loop_metrics["center_distance_matches_expected"]
    )

    write_json(out_dir / "validation.json", validation)
    write_json(out_dir / "loop_3403_4414.json", loop_metrics)

    print(f"Wrote {out_dir / 'validation.json'}")
    print(f"Wrote {out_dir / 'loop_3403_4414.json'}")
    print(f"validation_passed={validation['passed']}")
    print(
        "loop distance/time/rotation/view_cos = "
        f"{loop_metrics['center_distance_m']:.6f} m / "
        f"{loop_metrics['time_gap_s']:.6f} s / "
        f"{loop_metrics['relative_rotation_deg']:.3f} deg / "
        f"{loop_metrics['view_direction_cosine']:.3f}"
    )


if __name__ == "__main__":
    main()
