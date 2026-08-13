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

from revisit_memory.utils.depth_scale import estimate_depth_scale, resize_bool_stack, resize_float_stack  # noqa: E402
from revisit_memory.utils.geometry import bbox_bounds, depth_to_world_points, occupancy_metrics, sample_points, write_ply  # noqa: E402
from revisit_memory.utils.io import load_config, output_root, target_bbox, write_csv, write_json  # noqa: E402
from revisit_memory.utils.visualization import save_scatter_comparison, save_scatter_view  # noqa: E402


def configured_limits(cfg: dict[str, Any], key: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    limits = cfg["analysis"].get("pointcloud_plot_limits", {}).get(key)
    if limits is None:
        return None
    return (float(limits[0][0]), float(limits[0][1])), (float(limits[1][0]), float(limits[1][1]))


def bbox_zoom_limits(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    axes: tuple[int, int],
    margin: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(bbox_min[axes[0]] - margin), float(bbox_max[axes[0]] + margin)),
        (float(bbox_min[axes[1]] - margin), float(bbox_max[axes[1]] + margin)),
    )


def load_condition(cfg: dict[str, Any], condition: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, str, dict[str, Any]]:
    recon_dir = output_root(cfg) / "reconstruction" / condition
    with np.load(recon_dir / "predictions.npz", allow_pickle=True) as pred_npz:
        pred = {key: pred_npz[key] for key in pred_npz.files}
    with np.load(recon_dir / "inputs.npz", allow_pickle=True) as inp_npz:
        inputs = {key: inp_npz[key] for key in inp_npz.files}
    depth = pred["depth"][..., 0] if pred["depth"].ndim == 4 else pred["depth"].squeeze(-1)
    scale_info: dict[str, Any] = {"depth_metric_scale": 1.0, "scale_policy": "raw_pred_depth"}
    if bool(cfg["analysis"].get("pointcloud_metric_depth_scale", True)) and "gt_depth" in inputs:
        gt_depth_pre = resize_float_stack(inputs["gt_depth"].astype(np.float32), depth.shape[-2:])
        exclude_mask = None
        if bool(cfg["analysis"].get("pointcloud_depth_scale_exclude_target_mask", True)) and "counterfactual_target_mask" in inputs:
            exclude_mask = resize_bool_stack(inputs["counterfactual_target_mask"], depth.shape[-2:])
        scale, scale_info = estimate_depth_scale(
            depth,
            gt_depth_pre,
            exclude_mask,
            seed=int(cfg["project"]["random_seed"]),
        )
        depth = (depth.astype(np.float32) * np.float32(scale)).astype(np.float32)
    intrinsic = inputs["gt_intrinsic_preprocessed"].astype(np.float32)
    poses = inputs["gt_cam2world"].astype(np.float32)
    points = np.stack(
        [depth_to_world_points(depth[idx].astype(np.float32), intrinsic, poses[idx]) for idx in range(depth.shape[0])],
        axis=0,
    )
    conf = pred["depth_conf"].astype(np.float32) if "depth_conf" in pred else None
    source = "pred_depth_metric_scaled_reprojected_with_gt_pose" if scale_info["depth_metric_scale"] != 1.0 else "pred_depth_reprojected_with_gt_pose"
    frame_ids = inputs["frame_ids"].astype(int)
    return points, conf, frame_ids, source, scale_info


def frame_indices(frame_ids: np.ndarray, start_end: list[int]) -> np.ndarray:
    start, end = int(start_end[0]), int(start_end[1])
    return np.flatnonzero((frame_ids >= start) & (frame_ids <= end))


def segment_points(points: np.ndarray, conf: np.ndarray | None, indices: np.ndarray, cfg: dict[str, Any]):
    pts = points[indices]
    cnf = conf[indices] if conf is not None else None
    per_frame_limit = int(cfg["analysis"]["pointcloud_max_points_per_frame"]) * max(len(indices), 1)
    segment_limit = int(cfg["analysis"].get("pointcloud_max_points_per_segment", per_frame_limit))
    return sample_points(
        pts,
        cnf,
        max_points=min(per_frame_limit, segment_limit),
        seed=int(cfg["project"]["random_seed"]),
    )


def run(cfg: dict[str, Any]) -> None:
    bbox = target_bbox(cfg)
    center = np.asarray(bbox["center"], dtype=np.float32)
    dims = np.asarray(bbox["dimensions"], dtype=np.float32)
    bbox_min, bbox_max = bbox_bounds(center, dims, expansion=0.0)
    conf_threshold = float(cfg["analysis"]["confidence_threshold"])
    save_ply = bool(cfg["analysis"].get("pointcloud_save_ply", False))
    save_figures = bool(cfg["analysis"].get("pointcloud_save_figures", True))
    xy_limits = configured_limits(cfg, "xy")
    xz_limits = configured_limits(cfg, "xz")
    zoom_margin = float(cfg["analysis"].get("pointcloud_target_zoom_margin", 0.7))
    out_dir = output_root(cfg) / "analysis" / "pointcloud"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    condition_segment_points: dict[str, np.ndarray] = {}
    depth_scale_by_condition: dict[str, dict[str, Any]] = {}

    for condition in cfg["conditions"]:
        points, conf, frame_ids, point_source, scale_info = load_condition(cfg, condition)
        depth_scale_by_condition[condition] = scale_info
        segments = {
            "all_sequence_cumulative_map_reference_only": np.arange(len(frame_ids)),
            "first_loop_12_39": frame_indices(frame_ids, cfg["frames"]["first_loop_visible"]),
            "second_loop_56_83": frame_indices(frame_ids, cfg["frames"]["second_loop_eval"]),
        }
        for segment_name, idx in segments.items():
            if idx.size == 0:
                continue
            pts_segment = points[idx]
            conf_segment = conf[idx] if conf is not None else None
            for expansion in cfg["analysis"]["bbox_expansions"]:
                bmin, bmax = bbox_bounds(center, dims, expansion=float(expansion))
                metrics = occupancy_metrics(pts_segment, conf_segment, bmin, bmax, confidence_threshold=conf_threshold)
                rows.append(
                    {
                        "condition": condition,
                        "segment": segment_name,
                        "point_source": point_source,
                        "bbox_expansion": float(expansion),
                        **metrics,
                    }
                )

            sampled_pts, sampled_conf = segment_points(points, conf, idx, cfg)
            if save_ply:
                write_ply(out_dir / condition / f"{segment_name}.ply", sampled_pts)
            if save_figures:
                save_scatter_view(
                    out_dir / condition / f"{segment_name}_top_xy.png",
                    sampled_pts,
                    axes=(0, 1),
                    title=f"{condition} {segment_name}",
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    limits=xy_limits,
                )
                save_scatter_view(
                    out_dir / condition / f"{segment_name}_side_xz.png",
                    sampled_pts,
                    axes=(0, 2),
                    title=f"{condition} {segment_name}",
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    limits=xz_limits,
                )
                if segment_name == "second_loop_56_83":
                    save_scatter_view(
                        out_dir / condition / f"{segment_name}_target_zoom_top_xy.png",
                        sampled_pts,
                        axes=(0, 1),
                        title=f"{condition} {segment_name} target zoom",
                        bbox_min=bbox_min,
                        bbox_max=bbox_max,
                        limits=bbox_zoom_limits(bbox_min, bbox_max, (0, 1), zoom_margin),
                    )
                    save_scatter_view(
                        out_dir / condition / f"{segment_name}_target_zoom_side_xz.png",
                        sampled_pts,
                        axes=(0, 2),
                        title=f"{condition} {segment_name} target zoom",
                        bbox_min=bbox_min,
                        bbox_max=bbox_max,
                        limits=bbox_zoom_limits(bbox_min, bbox_max, (0, 2), zoom_margin),
                    )
            if segment_name == "second_loop_56_83":
                condition_segment_points[condition] = sampled_pts

    if save_figures and condition_segment_points:
        ordered = {condition: condition_segment_points[condition] for condition in cfg["conditions"] if condition in condition_segment_points}
        save_scatter_comparison(
            out_dir / "comparison_second_loop_56_83_top_xy.png",
            ordered,
            axes=(0, 1),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            limits=xy_limits,
            title="second_loop_56_83 top xy",
        )
        save_scatter_comparison(
            out_dir / "comparison_second_loop_56_83_side_xz.png",
            ordered,
            axes=(0, 2),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            limits=xz_limits,
            title="second_loop_56_83 side xz",
        )
        save_scatter_comparison(
            out_dir / "comparison_second_loop_56_83_target_zoom_top_xy.png",
            ordered,
            axes=(0, 1),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            limits=bbox_zoom_limits(bbox_min, bbox_max, (0, 1), zoom_margin),
            title="second_loop_56_83 target zoom top xy",
        )
        save_scatter_comparison(
            out_dir / "comparison_second_loop_56_83_target_zoom_side_xz.png",
            ordered,
            axes=(0, 2),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            limits=bbox_zoom_limits(bbox_min, bbox_max, (0, 2), zoom_margin),
            title="second_loop_56_83 target zoom side xz",
        )

    row_by = {(row["condition"], row["segment"], row["bbox_expansion"]): row for row in rows}
    deltas = []
    for expansion in cfg["analysis"]["bbox_expansions"]:
        key_removed = ("seen_then_removed", "second_loop_56_83", float(expansion))
        key_absent = ("always_absent", "second_loop_56_83", float(expansion))
        key_present = ("always_present", "second_loop_56_83", float(expansion))
        if key_removed in row_by and key_absent in row_by:
            removed = row_by[key_removed]
            absent = row_by[key_absent]
            present = row_by.get(key_present, {})
            deltas.append(
                {
                    "segment": "second_loop_56_83",
                    "bbox_expansion": float(expansion),
                    "ghost_occupancy_delta_points": removed["bbox_points"] - absent["bbox_points"],
                    "ghost_occupancy_delta_weighted": removed["bbox_confidence_weighted_points"]
                    - absent["bbox_confidence_weighted_points"],
                    "seen_then_removed_bbox_points": removed["bbox_points"],
                    "always_absent_bbox_points": absent["bbox_points"],
                    "always_present_bbox_points_reference": present.get("bbox_points"),
                }
            )

    write_csv(output_root(cfg) / "metrics" / "pointcloud_metrics.csv", rows)
    write_csv(output_root(cfg) / "metrics" / "pointcloud_ghost_occupancy_delta.csv", deltas)
    write_json(
        out_dir / "summary.json",
        {
            "pointcloud_source_note": (
                "Point clouds in this experiment are derived from LingBot-Map depth-head predictions "
                "reprojected with GT camera poses. When GT depth is available, one global median GT/pred depth "
                "scale is applied before comparing to metric GT bboxes. The reconstruction runner does not build "
                "or save a pointmap head."
            ),
            "point_source_policy": "pred_depth_metric_scaled_reprojected_with_gt_pose",
            "depth_scale_by_condition": depth_scale_by_condition,
            "target_bbox": bbox,
            "confidence_threshold": conf_threshold,
            "pointcloud_max_points_per_segment": int(cfg["analysis"].get("pointcloud_max_points_per_segment", 0)),
            "ply_outputs_enabled": save_ply,
            "scatter_figures_enabled": save_figures,
            "scatter_bbox_overlay": True,
            "scatter_fixed_limits": {"xy": xy_limits, "xz": xz_limits},
            "ghost_occupancy_delta": deltas,
            "ply_outputs": str(out_dir),
        },
    )
    print(f"Wrote {output_root(cfg) / 'metrics' / 'pointcloud_metrics.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Point cloud occupancy analysis for revisit-memory.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
