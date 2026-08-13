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

from revisit_memory.utils.geometry import apply_sim3_to_poses, rotation_geodesic_deg, sim3_align  # noqa: E402
from revisit_memory.utils.io import load_config, output_root, write_csv, write_json  # noqa: E402
from revisit_memory.utils.visualization import save_scatter_view  # noqa: E402


def load_condition(cfg: dict[str, Any], condition: str) -> dict[str, np.ndarray]:
    recon_dir = output_root(cfg) / "reconstruction" / condition
    with np.load(recon_dir / "predictions.npz", allow_pickle=True) as pred_npz:
        pred = {k: pred_npz[k] for k in pred_npz.files}
    with np.load(recon_dir / "inputs.npz", allow_pickle=True) as inp_npz:
        inputs = {k: inp_npz[k] for k in inp_npz.files}
    return {
        "frame_ids": inputs["frame_ids"].astype(int),
        "pred_cam2world": pred["pred_cam2world"].astype(np.float64),
        "gt_cam2world": inputs["gt_cam2world"].astype(np.float64),
    }


def metrics_for_condition(cfg: dict[str, Any], condition: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = load_condition(cfg, condition)
    pred = data["pred_cam2world"]
    gt = data["gt_cam2world"]
    frame_ids = data["frame_ids"]

    scale, rot, trans = sim3_align(pred[:, :3, 3], gt[:, :3, 3], allow_scale=True)
    aligned = apply_sim3_to_poses(pred, scale, rot, trans)

    raw_t = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    aligned_t = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    raw_r = rotation_geodesic_deg(pred[:, :3, :3], gt[:, :3, :3])
    aligned_r = rotation_geodesic_deg(aligned[:, :3, :3], gt[:, :3, :3])

    rows = []
    for idx, fid in enumerate(frame_ids.tolist()):
        rows.append(
            {
                "condition": condition,
                "frame": int(fid),
                "raw_translation_error": float(raw_t[idx]),
                "aligned_translation_error": float(aligned_t[idx]),
                "raw_rotation_error_deg": float(raw_r[idx]),
                "aligned_rotation_error_deg": float(aligned_r[idx]),
            }
        )

    eval_start, eval_end = cfg["frames"]["second_loop_eval"]
    eval_mask = (frame_ids >= int(eval_start)) & (frame_ids <= int(eval_end))
    summary = {
        "condition": condition,
        "alignment": {
            "method": "full-sequence Umeyama Sim(3), applied once per condition",
            "scale": scale,
            "rotation": rot.tolist(),
            "translation": trans.tolist(),
        },
        "ate_rmse_aligned": float(np.sqrt(np.mean(aligned_t**2))),
        "mean_translation_error_aligned_all": float(np.mean(aligned_t)),
        "mean_rotation_error_deg_aligned_all": float(np.mean(aligned_r)),
        "mean_translation_error_aligned_eval_56_83": float(np.mean(aligned_t[eval_mask])),
        "mean_rotation_error_deg_aligned_eval_56_83": float(np.mean(aligned_r[eval_mask])),
    }

    out_dir = output_root(cfg) / "analysis" / "camera" / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    save_scatter_view(out_dir / "gt_xy.png", gt[:, :3, 3], axes=(0, 1), title=f"{condition} GT xy")
    save_scatter_view(out_dir / "pred_aligned_xy.png", aligned[:, :3, 3], axes=(0, 1), title=f"{condition} pred aligned xy")
    return rows, summary


def run(cfg: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"conditions": {}}
    for condition in cfg["conditions"]:
        cond_rows, cond_summary = metrics_for_condition(cfg, condition)
        rows.extend(cond_rows)
        summary["conditions"][condition] = cond_summary

    data_seen = load_condition(cfg, "seen_then_removed")
    data_abs = load_condition(cfg, "always_absent")
    shared = np.intersect1d(data_seen["frame_ids"], data_abs["frame_ids"])
    seen_idx = {fid: i for i, fid in enumerate(data_seen["frame_ids"])}
    abs_idx = {fid: i for i, fid in enumerate(data_abs["frame_ids"])}
    diffs = []
    for fid in shared.tolist():
        ps = data_seen["pred_cam2world"][seen_idx[fid]]
        pa = data_abs["pred_cam2world"][abs_idx[fid]]
        diffs.append(
            {
                "frame": int(fid),
                "seen_removed_vs_absent_pred_translation_delta": float(np.linalg.norm(ps[:3, 3] - pa[:3, 3])),
                "seen_removed_vs_absent_pred_rotation_delta_deg": float(
                    rotation_geodesic_deg(ps[None, :3, :3], pa[None, :3, :3])[0]
                ),
            }
        )
    write_csv(output_root(cfg) / "metrics" / "camera_pose_condition_differences.csv", diffs)
    write_csv(output_root(cfg) / "metrics" / "camera_metrics.csv", rows)
    summary["pose_difference_between_conditions"] = {
        "seen_then_removed_vs_always_absent": {
            "mean_translation_delta_all": float(np.mean([d["seen_removed_vs_absent_pred_translation_delta"] for d in diffs])),
            "mean_rotation_delta_deg_all": float(np.mean([d["seen_removed_vs_absent_pred_rotation_delta_deg"] for d in diffs])),
        }
    }
    write_json(output_root(cfg) / "analysis" / "camera" / "summary.json", summary)
    print(f"Wrote {output_root(cfg) / 'metrics' / 'camera_metrics.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Camera pose analysis for revisit-memory.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

