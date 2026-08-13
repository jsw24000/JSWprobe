#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from common import (
    compare_overlay,
    deep_get,
    deep_set,
    discover_paths,
    load_config,
    load_manifest_or_metadata,
    make_panel,
    nearest_patch_distance,
    overlay_mask,
    pearsonr_np,
    read_rgb,
    resize_grid_to_shape,
    save_json,
    save_rgb,
    score_to_heatmap,
    select_frames,
    spearmanr_np,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Geometry Tokens with PCA visualization noise patches.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--scene-id", default=None, help="Override controlled.scene_id.")
    parser.add_argument("--setting-name", default=None, help="Override controlled.setting_name.")
    parser.add_argument("--condition-id", default=None, help="Override controlled.condition_id.")
    parser.add_argument("--top", type=int, default=None, help="Top percentile used for comparison visual panels.")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.scene_id:
        deep_set(cfg, "controlled.scene_id", args.scene_id)
    if args.setting_name:
        deep_set(cfg, "controlled.setting_name", args.setting_name)
    if args.condition_id:
        deep_set(cfg, "controlled.condition_id", args.condition_id)


def load_frame_map(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["t"]): row for row in data}


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else math.nan


def finite_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if len(arr) else math.nan


def metric_row(t: int, top: int, geom_mask: np.ndarray, noise_mask: np.ndarray, geom_score: np.ndarray, noise_score: np.ndarray) -> dict:
    geom_mask = geom_mask.astype(bool)
    noise_mask = noise_mask.astype(bool)
    intersection = int(np.count_nonzero(geom_mask & noise_mask))
    geom_count = int(np.count_nonzero(geom_mask))
    noise_count = int(np.count_nonzero(noise_mask))
    union = int(np.count_nonzero(geom_mask | noise_mask))
    min_count = min(geom_count, noise_count)
    return {
        "t": int(t),
        "top_percent": int(top),
        "geometry_count": geom_count,
        "pca_noise_count": noise_count,
        "intersection_count": intersection,
        "overlap_ratio": float(intersection / min_count) if min_count else math.nan,
        "iou": float(intersection / union) if union else math.nan,
        "precision_at_geometry_topK": float(intersection / geom_count) if geom_count else math.nan,
        "recall_at_geometry_topK": float(intersection / noise_count) if noise_count else math.nan,
        "noise_to_nearest_geometry_mean_patch_distance": nearest_patch_distance(noise_mask, geom_mask),
        "pearson_score": pearsonr_np(geom_score, noise_score),
        "spearman_score": spearmanr_np(geom_score, noise_score),
    }


def make_visual_panel(
    out_path: Path,
    rgb_path: str,
    pca_path: str,
    geom_score: np.ndarray,
    geom_mask: np.ndarray,
    noise_mask: np.ndarray,
    cfg: dict,
) -> None:
    rgb = read_rgb(rgb_path)
    pca = read_rgb(pca_path)
    residual_heat = resize_grid_to_shape(score_to_heatmap(geom_score, deep_get(cfg, "visualization.heatmap_cmap", "magma")), rgb.shape[:2])
    geom_color = deep_get(cfg, "visualization.geometry_color", [255, 64, 64])
    noise_color = deep_get(cfg, "visualization.pca_noise_color", [0, 190, 255])
    overlap_color = deep_get(cfg, "visualization.overlap_color", [255, 220, 40])
    geom_mask_vis = overlay_mask(np.ones_like(rgb), geom_mask, geom_color, alpha=0.85)
    pca_noise_vis = overlay_mask(pca, noise_mask, noise_color, alpha=0.65)
    compare = compare_overlay(rgb, geom_mask, noise_mask, geom_color, noise_color, overlap_color)
    make_panel(
        [
            ("RGB", rgb),
            ("Depth residual", residual_heat),
            ("Geometry mask", geom_mask_vis),
            ("PCA visualization", pca),
            ("PCA noise mask", pca_noise_vis),
            ("Geometry/PCA overlay", compare),
        ],
        out_path,
        max_width=int(deep_get(cfg, "visualization.max_panel_width", 360)),
        columns=3,
    )


def save_summary_plots(rows: list[dict], geom_scores: list[np.ndarray], noise_scores: list[np.ndarray], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tops = sorted({int(row["top_percent"]) for row in rows})
    metrics = ["iou", "precision_at_geometry_topK", "recall_at_geometry_topK", "overlap_ratio"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(tops))
    width = 0.18
    for i, metric in enumerate(metrics):
        values = [finite_mean([row[metric] for row in rows if int(row["top_percent"]) == top]) for top in tops]
        ax.bar(x + (i - 1.5) * width, values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels([f"top{top}" for top in tops])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_overlap_metrics.png", dpi=180)
    plt.close(fig)

    dist_values = [
        row["noise_to_nearest_geometry_mean_patch_distance"]
        for row in rows
        if int(row["top_percent"]) == tops[min(1, len(tops) - 1)]
        and np.isfinite(row["noise_to_nearest_geometry_mean_patch_distance"])
    ]
    if dist_values:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(dist_values, bins=16, color="#4c78a8")
        ax.set_xlabel("mean patch distance")
        ax.set_ylabel("frames")
        fig.tight_layout()
        fig.savefig(out_dir / "nearest_geometry_distance_hist.png", dpi=180)
        plt.close(fig)

    if geom_scores and noise_scores:
        g = np.concatenate([x.reshape(-1) for x in geom_scores])
        n = np.concatenate([x.reshape(-1) for x in noise_scores])
        finite = np.isfinite(g) & np.isfinite(n)
        g = g[finite]
        n = n[finite]
        if len(g):
            rng = np.random.default_rng(2026)
            max_points = min(len(g), 50000)
            idx = rng.choice(len(g), size=max_points, replace=False) if len(g) > max_points else np.arange(len(g))
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(g[idx], n[idx], s=2, alpha=0.12, color="#222222")
            ax.set_xlabel("geometry residual score")
            ax.set_ylabel("PCA noise score")
            fig.tight_layout()
            fig.savefig(out_dir / "score_correlation_scatter.png", dpi=180)
            plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    paths = discover_paths(cfg)
    manifest = load_manifest_or_metadata(paths, cfg)
    frames = select_frames(manifest.get("frames", []), deep_get(cfg, "frames.max_frames"))

    scene_out = paths["scene_output"]
    geom_dir = scene_out / "geometry"
    pca_dir = scene_out / "pca_noise"
    compare_dir = scene_out / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)

    geom_score = np.load(geom_dir / "geometry_score.npy")
    pca_score = np.load(pca_dir / "pca_noise_score.npy")
    geom_map = load_frame_map(geom_dir / "frame_index_map.json")
    pca_map = load_frame_map(pca_dir / "frame_index_map.json")

    common_t = sorted(set(geom_map) & set(pca_map))
    if not common_t:
        raise RuntimeError("No common frame indices between geometry outputs and PCA noise outputs.")

    top_percentiles = [int(x) for x in deep_get(cfg, "geometry_tokens.top_percentiles", [5, 10, 20])]
    panel_top = args.top or int(deep_get(cfg, "geometry_tokens.default_top_percentile", 10))
    rows = []
    corr_geom_scores = []
    corr_noise_scores = []
    for top in top_percentiles:
        geom_masks = np.load(geom_dir / f"geometry_mask_top{top}.npy")
        pca_masks = np.load(pca_dir / "pca_noise_mask.npy")
        for t in common_t:
            gi = int(geom_map[t]["stack_index"])
            pi = int(pca_map[t]["stack_index"])
            g_score = geom_score[gi]
            n_score = pca_score[pi]
            if n_score.shape != g_score.shape:
                n_score = resize_grid_to_shape(n_score, g_score.shape, nearest=False)
            g_mask = geom_masks[gi]
            n_mask = pca_masks[pi]
            if n_mask.shape != g_mask.shape:
                n_mask = resize_grid_to_shape(n_mask.astype(np.float32), g_mask.shape, nearest=True) > 0.5
            rows.append(metric_row(t, top, g_mask, n_mask, g_score, n_score))
            if top == panel_top:
                corr_geom_scores.append(g_score)
                corr_noise_scores.append(n_score)
                panel_path = compare_dir / f"t{t:03d}_frame_{geom_map[t].get('source_frame_id', '')}_compare_top{top}.png"
                make_visual_panel(
                    panel_path,
                    geom_map[t]["rgb_path"],
                    pca_map[t]["pca_path"],
                    g_score,
                    g_mask,
                    n_mask,
                    cfg,
                )

    aggregate = {}
    for top in top_percentiles:
        top_rows = [row for row in rows if int(row["top_percent"]) == top]
        aggregate[f"top{top}"] = {
            "frame_count": len(top_rows),
            "geometry_count_mean": finite_mean([row["geometry_count"] for row in top_rows]),
            "pca_noise_count_mean": finite_mean([row["pca_noise_count"] for row in top_rows]),
            "overlap_ratio_mean": finite_mean([row["overlap_ratio"] for row in top_rows]),
            "overlap_ratio_std": finite_std([row["overlap_ratio"] for row in top_rows]),
            "iou_mean": finite_mean([row["iou"] for row in top_rows]),
            "iou_std": finite_std([row["iou"] for row in top_rows]),
            "precision_at_geometry_topK_mean": finite_mean([row["precision_at_geometry_topK"] for row in top_rows]),
            "precision_at_geometry_topK_std": finite_std([row["precision_at_geometry_topK"] for row in top_rows]),
            "recall_at_geometry_topK_mean": finite_mean([row["recall_at_geometry_topK"] for row in top_rows]),
            "recall_at_geometry_topK_std": finite_std([row["recall_at_geometry_topK"] for row in top_rows]),
            "noise_to_nearest_geometry_mean_patch_distance_mean": finite_mean(
                [row["noise_to_nearest_geometry_mean_patch_distance"] for row in top_rows]
            ),
            "pearson_score_mean": finite_mean([row["pearson_score"] for row in top_rows]),
            "spearman_score_mean": finite_mean([row["spearman_score"] for row in top_rows]),
        }

    write_csv(rows, compare_dir / "metrics_per_frame.csv")
    save_summary_plots(rows, corr_geom_scores, corr_noise_scores, compare_dir)

    metrics = {
        "status": "ok",
        "scene_id": manifest.get("scene_id", deep_get(cfg, "controlled.scene_id")),
        "setting_name": manifest.get("setting_name", deep_get(cfg, "controlled.setting_name")),
        "condition_id": manifest.get("condition_id", deep_get(cfg, "controlled.condition_id")),
        "common_frame_count": len(common_t),
        "common_frame_indices": common_t,
        "patch_grid": list(geom_score.shape[1:3]),
        "definitions": {
            "overlap_ratio": "intersection / min(num_geometry_tokens, num_pca_noise_patches)",
            "precision_at_geometry_topK": "intersection / num_geometry_tokens",
            "recall_at_geometry_topK": "intersection / num_pca_noise_patches",
            "noise_to_nearest_geometry_mean_patch_distance": "Mean Euclidean distance in patch-grid units from each PCA noise patch to the closest geometry token.",
        },
        "aggregate": aggregate,
        "per_frame_csv": str(compare_dir / "metrics_per_frame.csv"),
        "visualization_dir": str(compare_dir),
    }
    save_json(metrics, compare_dir / "metrics.json")
    save_json(metrics, scene_out / "metrics.json")
    print(json.dumps({"status": "ok", "metrics": str(scene_out / "metrics.json"), "common_frame_count": len(common_t)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"compare_geometry_pca_noise failed: {exc}", file=sys.stderr)
        raise
