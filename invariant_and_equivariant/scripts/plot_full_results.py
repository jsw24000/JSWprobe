#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


MODEL_LABELS = {
    "dinov2_official_vitl14_reg4": "DINOv2 ViT-L/14 reg4",
    "vggt_1b_aggregator": "VGGT aggregator",
}

MODEL_LAYERS = {
    "dinov2_official_vitl14_reg4": ["block_05", "block_11", "block_17", "block_last"],
    "vggt_1b_aggregator": ["aggregator_block_04", "aggregator_block_11", "aggregator_block_17", "aggregator_block_last"],
}
STAGE_LABELS = ["early", "middle", "deep", "last"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_value(rows: Sequence[Mapping[str, str]], analysis: str, metric: str, model: str, layer: str, space: str = "raw_standardized") -> float:
    vals = [
        row
        for row in rows
        if row.get("analysis") == analysis
        and row.get("metric_name") == metric
        and row.get("model_name") == model
        and row.get("layer_name") == layer
        and row.get("split_name") == "test"
        and row.get("coordinate_system") == "ref"
        and row.get("feature_space") == space
    ]
    if not vals:
        return float("nan")
    return float(vals[0]["metric_value"])


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    return plt


def plot_layerwise_main(rows: Sequence[Mapping[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    panels = [
        ("absolute_position", "r2", "Absolute position R2", "higher"),
        ("delta_decode", "r2", "Delta decode R2", "higher"),
        ("forward_equivariance", "forward_r2", "Forward map R2", "higher"),
        ("composition", "composite_forward_r2", "Composition R2", "higher"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    for ax, (analysis, metric, title, _direction) in zip(axes.ravel(), panels):
        for model, layers in MODEL_LAYERS.items():
            values = [metric_value(rows, analysis, metric, model, layer) for layer in layers]
            ax.plot(range(len(layers)), values, marker="o", linewidth=2, label=MODEL_LABELS[model])
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(STAGE_LABELS, rotation=0, ha="center")
        ax.set_title(title)
        ax.set_ylabel(metric)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Layer-wise linear probe and equivariance metrics")
    fig.tight_layout()
    fig.savefig(out / "layerwise_main_r2.png")
    plt.close(fig)


def plot_layerwise_aux(rows: Sequence[Mapping[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    panels = [
        ("absolute_position", "vector_mae", "Absolute vector MAE (m)", "lower"),
        ("forward_equivariance", "normalized_forward_error", "Normalized forward error", "lower"),
        ("forward_equivariance", "state_retrieval_top3", "Forward state retrieval Top-3", "higher"),
        ("composition", "triplet_top3", "Composition triplet Top-3", "higher"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    for ax, (analysis, metric, title, _direction) in zip(axes.ravel(), panels):
        for model, layers in MODEL_LAYERS.items():
            values = [metric_value(rows, analysis, metric, model, layer) for layer in layers]
            ax.plot(range(len(layers)), values, marker="o", linewidth=2, label=MODEL_LABELS[model])
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(STAGE_LABELS, rotation=0, ha="center")
        ax.set_title(title)
        ax.set_ylabel(metric)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Layer-wise errors and retrieval metrics")
    fig.tight_layout()
    fig.savefig(out / "layerwise_errors_retrieval.png")
    plt.close(fig)


def plot_baseline_comparison(rows: Sequence[Mapping[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    abs_baseline = [
        row
        for row in rows
        if row.get("analysis") == "absolute_position"
        and row.get("model_name") == "2d_mask_bbox_baseline"
        and row.get("metric_name") == "r2"
        and row.get("split_name") == "test"
    ][0]
    delta_baseline = [
        row
        for row in rows
        if row.get("analysis") == "delta_decode"
        and row.get("model_name") == "2d_mask_bbox_delta_baseline"
        and row.get("metric_name") == "r2"
        and row.get("split_name") == "test"
    ][0]
    bars = [
        ("Abs 2D", float(abs_baseline["metric_value"])),
        ("Abs DINO best", max(metric_value(rows, "absolute_position", "r2", "dinov2_official_vitl14_reg4", layer) for layer in MODEL_LAYERS["dinov2_official_vitl14_reg4"])),
        ("Abs VGGT best", max(metric_value(rows, "absolute_position", "r2", "vggt_1b_aggregator", layer) for layer in MODEL_LAYERS["vggt_1b_aggregator"])),
        ("Delta 2D", float(delta_baseline["metric_value"])),
        ("Delta DINO best", max(metric_value(rows, "delta_decode", "r2", "dinov2_official_vitl14_reg4", layer) for layer in MODEL_LAYERS["dinov2_official_vitl14_reg4"])),
        ("Delta VGGT best", max(metric_value(rows, "delta_decode", "r2", "vggt_1b_aggregator", layer) for layer in MODEL_LAYERS["vggt_1b_aggregator"])),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = ["#8c8c8c", "#5975a4", "#5f9e6e", "#8c8c8c", "#5975a4", "#5f9e6e"]
    ax.bar(range(len(bars)), [v for _, v in bars], color=colors)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([k for k, _ in bars], rotation=25, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("2D baseline versus best feature probes")
    fig.tight_layout()
    fig.savefig(out / "baseline_vs_feature_r2.png")
    plt.close(fig)


def plot_absolute_scatter(predictions: Sequence[Mapping[str, str]], out: Path, model: str, filename: str) -> None:
    plt = setup_matplotlib()
    layers = MODEL_LAYERS[model]
    axes_names = ["x", "y", "z"]
    fig, axes = plt.subplots(len(layers), 3, figsize=(10, 10), sharex=False, sharey=False)
    for row_idx, layer in enumerate(layers):
        rows = [
            row
            for row in predictions
            if row["model_name"] == model and row["layer_name"] == layer and row["coordinate_system"] == "ref" and row["split"] == "test"
        ]
        for col_idx, axis_name in enumerate(axes_names):
            ax = axes[row_idx, col_idx]
            gt = np.asarray([float(row[f"gt_{axis_name}"]) for row in rows], dtype=float)
            pred = np.asarray([float(row[f"pred_{axis_name}"]) for row in rows], dtype=float)
            ax.scatter(gt, pred, s=10, alpha=0.65)
            lo = min(gt.min(), pred.min())
            hi = max(gt.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
            if row_idx == 0:
                ax.set_title(f"{axis_name} axis")
            if col_idx == 0:
                ax.set_ylabel(layer)
            if row_idx == len(layers) - 1:
                ax.set_xlabel("ground truth")
            if col_idx == 2:
                ax.text(1.02, 0.5, "predicted", transform=ax.transAxes, rotation=90, va="center")
    fig.suptitle(f"{MODEL_LABELS[model]} absolute position probe: predicted vs ground truth")
    fig.tight_layout()
    fig.savefig(out / filename)
    plt.close(fig)


def plot_subspace_metrics(rows: Sequence[Mapping[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    metrics = [
        ("projection_overlap", "Projection overlap (higher)"),
        ("min_principal_angle_deg", "Min principal angle (deg)"),
        ("mean_principal_angle_deg", "Mean principal angle (deg)"),
        ("grassmann_distance", "Grassmann distance"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for model, layers in MODEL_LAYERS.items():
            values = [metric_value(rows, "subspaces", metric, model, layer) for layer in layers]
            ax.plot(range(len(layers)), values, marker="o", linewidth=2, label=MODEL_LABELS[model])
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(STAGE_LABELS, rotation=0, ha="center")
        ax.set_title(title)
        ax.set_ylabel(metric)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Read subspace versus motion subspace")
    fig.tight_layout()
    fig.savefig(out / "subspace_overlap_angles.png")
    plt.close(fig)


def read_spectrum(path: Path) -> np.ndarray:
    rows = read_csv(path)
    return np.asarray([float(row["singular_value"]) for row in rows], dtype=float)


def plot_singular_spectra(run_dir: Path, out: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, kind in zip(axes, ["read", "move"]):
        for model, layers in MODEL_LAYERS.items():
            for layer in layers:
                path = run_dir / "subspaces" / f"{model}_{layer}_singular_values_{kind}.csv"
                if not path.exists():
                    continue
                s = read_spectrum(path)
                if s.size and s[0] != 0:
                    s = s / s[0]
                ax.plot(range(1, len(s) + 1), s, marker="o", linewidth=1.5, alpha=0.8, label=f"{MODEL_LABELS[model]} {layer}")
        ax.set_title(f"Normalized singular values: S_{kind}")
        ax.set_xlabel("component")
        ax.set_ylabel("singular value / first")
        ax.set_xticks([1, 2, 3])
    axes[1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "subspace_singular_spectra.png")
    plt.close(fig)


def plot_intervention(rows: Sequence[Mapping[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    spaces = ["raw", "keep_read_subspace", "remove_read_subspace", "keep_move_subspace", "remove_move_subspace"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for ax, metric in zip(axes, ["r2", "same_scene_cross_position_cosine"]):
        labels = []
        x = np.arange(len(spaces))
        width = 0.1
        offset = -0.35
        for model, layers in MODEL_LAYERS.items():
            for layer in layers:
                vals = []
                for space in spaces:
                    found = [
                        row
                        for row in rows
                        if row.get("model_name") == model
                        and row.get("layer_name") == layer
                        and row.get("metric_name") == metric
                        and row.get("split_name") == "test"
                        and row.get("feature_space") == space
                    ]
                    vals.append(float(found[0]["metric_value"]) if found else float("nan"))
                ax.plot(x, vals, marker="o", linewidth=1.3, alpha=0.82, label=f"{MODEL_LABELS[model]} {layer}")
        ax.set_xticks(x)
        ax.set_xticklabels(spaces, rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Intervention metric: {metric}")
    axes[0].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Subspace intervention: keep/remove read and move subspaces")
    fig.tight_layout()
    fig.savefig(out / "intervention_keep_remove.png")
    plt.close(fig)


def plot_forward_vs_composition(rows: Sequence[Mapping[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for model, layers in MODEL_LAYERS.items():
        xs = [metric_value(rows, "forward_equivariance", "forward_r2", model, layer) for layer in layers]
        ys = [metric_value(rows, "composition", "composite_forward_r2", model, layer) for layer in layers]
        ax.scatter(xs, ys, s=55, label=MODEL_LABELS[model])
        for x, y, layer in zip(xs, ys, layers):
            ax.text(x, y, layer.replace("aggregator_", ""), fontsize=8, ha="left", va="bottom")
    lim_hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim_hi], [0, lim_hi], color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("single-pair forward R2")
    ax.set_ylabel("composite forward R2")
    ax.set_title("E7 depends on the one-step forward map")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "forward_vs_composition_r2.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = run_dir / "figures" / "extra"
    out.mkdir(parents=True, exist_ok=True)

    rows = read_csv(run_dir / "tables" / "main_smoke_results.csv")
    predictions = read_csv(run_dir / "absolute_position" / "predictions.csv")

    plot_layerwise_main(rows, out)
    plot_layerwise_aux(rows, out)
    plot_baseline_comparison(rows, out)
    plot_absolute_scatter(predictions, out, "dinov2_official_vitl14_reg4", "dinov2_absolute_probe_pred_vs_gt.png")
    plot_absolute_scatter(predictions, out, "vggt_1b_aggregator", "vggt_absolute_probe_pred_vs_gt.png")
    plot_subspace_metrics(rows, out)
    plot_singular_spectra(run_dir, out)
    intervention_rows = read_csv(run_dir / "intervention" / "metrics.csv")
    plot_intervention(intervention_rows, out)
    plot_forward_vs_composition(rows, out)

    for path in sorted(out.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
