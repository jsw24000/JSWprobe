#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from noise_slot_utils import (
    append_run_log,
    heatmap_entropy,
    positional_r2,
    resolve_config,
    top1_top2_gap,
    write_csv,
)


PER_QUERY_FIELDS = [
    "scene",
    "stage",
    "k",
    "src_frame",
    "tgt_frame",
    "query_id",
    "qh",
    "qw",
    "raw_argmax_h",
    "raw_argmax_w",
    "debiased_argmax_h",
    "debiased_argmax_w",
    "raw_dist_to_same_pos",
    "debiased_dist_to_same_pos",
    "raw_positional_r2",
    "debiased_positional_r2",
    "raw_peakiness",
    "debiased_peakiness",
    "raw_top1_top2_gap",
    "debiased_top1_top2_gap",
    "raw_entropy",
    "debiased_entropy",
    "argmax_shift",
]


SUMMARY_FIELDS = [
    "scene",
    "stage",
    "k",
    "mean_raw_dist_to_same_pos",
    "mean_debiased_dist_to_same_pos",
    "mean_raw_positional_r2",
    "mean_debiased_positional_r2",
    "mean_raw_peakiness",
    "mean_debiased_peakiness",
    "mean_raw_entropy",
    "mean_debiased_entropy",
    "mean_argmax_shift",
    "median_argmax_shift",
    "num_queries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate no-GT diagnostics for debiased correspondence.")
    parser.add_argument("--config", default="configs/noise_slot_svd_debias.yaml")
    return parser.parse_args()


def dist(a_h: int, a_w: int, b_h: int, b_w: int) -> float:
    return float(math.sqrt((a_h - b_h) ** 2 + (a_w - b_w) ** 2))


def metrics_for_heatmap(sim: np.ndarray, temp: float) -> dict:
    return {
        "positional_r2": positional_r2(sim),
        "peakiness": float(sim.max() - sim.mean()),
        "top1_top2_gap": top1_top2_gap(sim),
        "entropy": heatmap_entropy(sim, temperature=temp),
    }


def mean(rows, key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config)
    output_root = Path(cfg["output_root"])
    metrics_dir = output_root / "05_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    temp = float(cfg.get("metrics", {}).get("softmax_temperature", 0.07))
    append_run_log(cfg, "Starting evaluate_noise_debiased_correspondence.py")

    rows = []
    for stage in cfg["stages_resolved"]:
        for k in cfg.get("k_list", [1, 2, 4, 8, 16]):
            k = int(k)
            corr_dir = output_root / "04_debiased_correspondence" / stage / f"k{k:02d}"
            metadata_path = corr_dir / "query_metadata.json"
            if not metadata_path.is_file():
                append_run_log(cfg, f"Warning: missing query metadata for {stage} k={k}")
                continue
            metadata = json.loads(metadata_path.read_text())
            for query in metadata.get("queries", []):
                qid = int(query["query_id"])
                raw_path = corr_dir / "heatmaps_raw" / f"query_{qid:02d}.npy"
                deb_path = corr_dir / "heatmaps_debiased" / f"query_{qid:02d}.npy"
                if not raw_path.is_file() or not deb_path.is_file():
                    append_run_log(cfg, f"Warning: missing heatmaps for {stage} k={k} query={qid}")
                    continue
                raw = np.load(raw_path)
                deb = np.load(deb_path)
                raw_m = metrics_for_heatmap(raw, temp)
                deb_m = metrics_for_heatmap(deb, temp)
                qh, qw = int(query["qh"]), int(query["qw"])
                raw_h, raw_w = int(query["raw_argmax_h"]), int(query["raw_argmax_w"])
                deb_h, deb_w = int(query["debiased_argmax_h"]), int(query["debiased_argmax_w"])
                rows.append(
                    {
                        "scene": cfg["scene"],
                        "stage": stage,
                        "k": k,
                        "src_frame": int(metadata["src_frame"]),
                        "tgt_frame": int(metadata["tgt_frame"]),
                        "query_id": qid,
                        "qh": qh,
                        "qw": qw,
                        "raw_argmax_h": raw_h,
                        "raw_argmax_w": raw_w,
                        "debiased_argmax_h": deb_h,
                        "debiased_argmax_w": deb_w,
                        "raw_dist_to_same_pos": dist(raw_h, raw_w, qh, qw),
                        "debiased_dist_to_same_pos": dist(deb_h, deb_w, qh, qw),
                        "raw_positional_r2": raw_m["positional_r2"],
                        "debiased_positional_r2": deb_m["positional_r2"],
                        "raw_peakiness": raw_m["peakiness"],
                        "debiased_peakiness": deb_m["peakiness"],
                        "raw_top1_top2_gap": raw_m["top1_top2_gap"],
                        "debiased_top1_top2_gap": deb_m["top1_top2_gap"],
                        "raw_entropy": raw_m["entropy"],
                        "debiased_entropy": deb_m["entropy"],
                        "argmax_shift": dist(raw_h, raw_w, deb_h, deb_w),
                    }
                )

    write_csv(metrics_dir / "metrics_per_query.csv", rows, PER_QUERY_FIELDS)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["stage"], int(row["k"]))].append(row)
    summary_rows = []
    for (scene, stage, k), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2])):
        shifts = [float(row["argmax_shift"]) for row in group]
        summary_rows.append(
            {
                "scene": scene,
                "stage": stage,
                "k": k,
                "mean_raw_dist_to_same_pos": mean(group, "raw_dist_to_same_pos"),
                "mean_debiased_dist_to_same_pos": mean(group, "debiased_dist_to_same_pos"),
                "mean_raw_positional_r2": mean(group, "raw_positional_r2"),
                "mean_debiased_positional_r2": mean(group, "debiased_positional_r2"),
                "mean_raw_peakiness": mean(group, "raw_peakiness"),
                "mean_debiased_peakiness": mean(group, "debiased_peakiness"),
                "mean_raw_entropy": mean(group, "raw_entropy"),
                "mean_debiased_entropy": mean(group, "debiased_entropy"),
                "mean_argmax_shift": float(np.mean(shifts)) if shifts else float("nan"),
                "median_argmax_shift": float(np.median(shifts)) if shifts else float("nan"),
                "num_queries": len(group),
            }
        )
    write_csv(metrics_dir / "metrics_summary_by_stage_k.csv", summary_rows, SUMMARY_FIELDS)

    lines = [
        "# Noise-Slot SVD Debias Metrics Summary",
        "",
        f"Scene: `{cfg['scene']}`",
        "",
        "| stage | K | raw R2 | debiased R2 | delta R2 | raw dist | debiased dist | argmax shift | raw entropy | debiased entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        delta = float(row["mean_raw_positional_r2"]) - float(row["mean_debiased_positional_r2"])
        lines.append(
            f"| {row['stage']} | {row['k']} | {float(row['mean_raw_positional_r2']):.4f} | "
            f"{float(row['mean_debiased_positional_r2']):.4f} | {delta:.4f} | "
            f"{float(row['mean_raw_dist_to_same_pos']):.3f} | {float(row['mean_debiased_dist_to_same_pos']):.3f} | "
            f"{float(row['mean_argmax_shift']):.3f} | {float(row['mean_raw_entropy']):.3f} | "
            f"{float(row['mean_debiased_entropy']):.3f} |"
        )
    if summary_rows:
        best = max(summary_rows, key=lambda row: float(row["mean_raw_positional_r2"]) - float(row["mean_debiased_positional_r2"]))
        best_delta = float(best["mean_raw_positional_r2"]) - float(best["mean_debiased_positional_r2"])
        lines.extend(
            [
                "",
                f"Largest mean positional R2 drop: `{best['stage']}` K={best['k']} (delta={best_delta:.4f}).",
            ]
        )
    else:
        lines.append("")
        lines.append("No metric rows were generated.")
    (metrics_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n")
    append_run_log(cfg, f"Wrote metrics: {len(rows)} query rows, {len(summary_rows)} grouped rows")
    append_run_log(cfg, "Finished evaluate_noise_debiased_correspondence.py")


if __name__ == "__main__":
    main()
