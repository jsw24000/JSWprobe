#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ORDER = ["state5", "presence", "position4"]
TASK_LABELS = {
    "state5": "State5",
    "presence": "Presence",
    "position4": "Position4",
}
LAYER_ORDER = ["block04", "block11", "block17", "block23"]
LAYER_LABELS = {"block04": "B04", "block11": "B11", "block17": "B17", "block23": "B23"}
TASK_COLORS = {
    "state5": "#4C78A8",
    "presence": "#2A9D8F",
    "position4": "#E76F51",
}
MODEL_MARKERS = {"linear": "o", "mlp": "^"}
CHANCE = {"state5": 0.20, "presence": 0.50, "position4": 0.25}
SLOT_LABELS = ["slot0 camera", "slot1 reg0", "slot2 reg1", "slot3 reg2", "slot4 reg3", "slot5 scale"]


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    for row in rows:
        row["val_score_f"] = parse_float(row.get("val_score"))
        row["test_score_f"] = parse_float(row.get("test_score"))
        row["test_accuracy_f"] = parse_float(row.get("test_accuracy"))
        row["test_macro_f1_f"] = parse_float(row.get("test_macro_f1"))
        row["best_epoch_i"] = int(float(row["best_epoch"])) if row.get("best_epoch") else None
        row["readout"] = readout_name(row)
        row["readout_family"] = readout_family(row)
    return [row for row in rows if row["test_score_f"] is not None and row["val_score_f"] is not None]


def readout_name(row: Dict[str, Any]) -> str:
    mode = row.get("feature_mode", "")
    if mode == "all6_mean":
        return "all6_mean"
    if mode == "all6_flatten":
        pca_dim = row.get("pca_dim") or "none"
        return f"all6_flatten_pca{pca_dim}" if pca_dim != "none" else "all6_flatten"
    if mode == "single_slot":
        return f"slot{row.get('slot_index')}"
    if mode == "slot_group":
        return f"group_{row.get('slot_group')}"
    return mode or "unknown"


def readout_family(row: Dict[str, Any]) -> str:
    mode = row.get("feature_mode", "")
    if mode == "single_slot":
        return "single_slot"
    if mode == "slot_group":
        return "slot_group"
    return readout_name(row)


def grouped_values(rows: Iterable[Dict[str, Any]], keys: Sequence[str], value_key: str) -> Dict[Tuple[Any, ...], List[float]]:
    groups: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is not None:
            groups[tuple(row.get(key) for key in keys)].append(float(value))
    return groups


def matrix_from_groups(
    groups: Dict[Tuple[Any, ...], List[float]],
    row_values: Sequence[str],
    col_values: Sequence[str],
    reducer: str,
) -> np.ndarray:
    mat = np.full((len(row_values), len(col_values)), np.nan, dtype=float)
    for i, row_key in enumerate(row_values):
        for j, col_key in enumerate(col_values):
            values = groups.get((row_key, col_key), [])
            if not values:
                continue
            mat[i, j] = max(values) if reducer == "max" else mean(values)
    return mat


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFBFD",
            "axes.edgecolor": "#D6D9E0",
            "axes.labelcolor": "#202632",
            "axes.titlecolor": "#202632",
            "xtick.color": "#4C5565",
            "ytick.color": "#4C5565",
            "grid.color": "#E7E9EF",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> List[str]:
    paths = []
    for suffix in ["png", "pdf"]:
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=220)
        paths.append(str(path))
    plt.close(fig)
    return paths


def annotate_heatmap(ax: plt.Axes, mat: np.ndarray, fmt: str = ".2f") -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value >= 0.78 else "#202632"
            ax.text(j, i, format(value, fmt), ha="center", va="center", color=color, fontsize=9, fontweight="bold")


def draw_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    title: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    vmin: float = 0.2,
    vmax: float = 1.0,
) -> None:
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    annotate_heatmap(ax, mat)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_layer_trends(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    groups = grouped_values(rows, ["task", "layer"], "test_score_f")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for reducer, ax, title in [("mean", axes[0], "Mean test score across readouts"), ("max", axes[1], "Best test score across readouts")]:
        for task in TASK_ORDER:
            ys = []
            for layer in LAYER_ORDER:
                vals = groups.get((task, layer), [])
                ys.append(max(vals) if reducer == "max" and vals else mean(vals) if vals else np.nan)
            ax.plot(
                range(len(LAYER_ORDER)),
                ys,
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=TASK_COLORS[task],
                label=TASK_LABELS[task],
            )
            ax.axhline(CHANCE[task], color=TASK_COLORS[task], linestyle=":", linewidth=1.0, alpha=0.35)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(LAYER_ORDER)))
        ax.set_xticklabels([LAYER_LABELS[layer] for layer in LAYER_ORDER])
        ax.set_xlabel("Memory layer")
        ax.grid(axis="y")
        ax.set_ylim(0.15, 1.02)
    axes[0].set_ylabel("Unified score: acc, presence uses balanced acc")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Layer trends by probe task", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    return save_figure(fig, out_dir, "01_layer_trends")


def plot_task_layer_heatmaps(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    groups = grouped_values(rows, ["task", "layer"], "test_score_f")
    mean_mat = matrix_from_groups(groups, TASK_ORDER, LAYER_ORDER, "mean")
    max_mat = matrix_from_groups(groups, TASK_ORDER, LAYER_ORDER, "max")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    row_labels = [TASK_LABELS[t] for t in TASK_ORDER]
    col_labels = [LAYER_LABELS[layer] for layer in LAYER_ORDER]
    im0 = draw_heatmap(axes[0], mean_mat, "Mean test score", row_labels, col_labels)
    im1 = draw_heatmap(axes[1], max_mat, "Best test score", row_labels, col_labels)
    cbar = fig.colorbar(im1, ax=axes, shrink=0.82, pad=0.03)
    cbar.set_label("Score")
    return save_figure(fig, out_dir, "02_task_layer_heatmaps")


def plot_readout_family_boxes(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    families = ["all6_mean", "all6_flatten_pca64", "single_slot", "slot_group"]
    labels = ["All6 mean", "All6 flat PCA64", "Single slot", "Slot group"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, task in zip(axes, TASK_ORDER):
        data = []
        for family in families:
            vals = [row["test_score_f"] for row in rows if row["task"] == task and row["readout_family"] == family]
            data.append(vals)
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.6, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(TASK_COLORS[task])
            patch.set_alpha(0.22)
            patch.set_edgecolor(TASK_COLORS[task])
        for key in ["whiskers", "caps", "medians"]:
            for artist in bp[key]:
                artist.set_color(TASK_COLORS[task])
                artist.set_linewidth(1.4)
        for idx, vals in enumerate(data, start=1):
            if not vals:
                continue
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(
                np.full(len(vals), idx) + jitter,
                vals,
                s=26,
                color=TASK_COLORS[task],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.scatter(idx, mean(vals), s=70, marker="D", color="#202632", edgecolor="white", linewidth=0.7, zorder=4)
        ax.axhline(CHANCE[task], color="#6B7280", linestyle=":", linewidth=1.0)
        ax.set_title(TASK_LABELS[task], fontweight="bold")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y")
        ax.set_ylim(0.15, 1.02)
    axes[0].set_ylabel("Test score")
    fig.suptitle("Readout family distributions", fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    return save_figure(fig, out_dir, "03_readout_family_distributions")


def plot_slot_heatmaps(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    single_slot_rows = [row for row in rows if row["feature_mode"] == "single_slot" and row["model"] == "linear"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), constrained_layout=True)
    for ax, task in zip(axes, TASK_ORDER):
        mat = np.full((6, len(LAYER_ORDER)), np.nan)
        for row in single_slot_rows:
            if row["task"] != task:
                continue
            slot = int(row["slot_index"])
            layer_idx = LAYER_ORDER.index(row["layer"])
            mat[slot, layer_idx] = row["test_score_f"]
        draw_heatmap(
            ax,
            mat,
            TASK_LABELS[task],
            SLOT_LABELS,
            [LAYER_LABELS[layer] for layer in LAYER_ORDER],
            vmin=0.2,
            vmax=1.0,
        )
    fig.suptitle("Single-slot linear probes: test score by layer", fontsize=14, fontweight="bold", y=1.04)
    return save_figure(fig, out_dir, "04_single_slot_heatmaps")


def clean_run_label(run_name: str, task: str) -> str:
    label = run_name
    prefix = f"{task}_"
    if label.startswith(prefix):
        label = label[len(prefix) :]
    return label.replace("_", " ")


def plot_top_runs(rows: List[Dict[str, Any]], out_dir: Path, top_k: int) -> List[str]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), sharex=True)
    for ax, task in zip(axes, TASK_ORDER):
        task_rows = sorted([row for row in rows if row["task"] == task], key=lambda row: row["test_score_f"], reverse=True)[:top_k]
        task_rows = list(reversed(task_rows))
        y = range(len(task_rows))
        scores = [row["test_score_f"] for row in task_rows]
        labels = [clean_run_label(row["run_name"], task) for row in task_rows]
        ax.barh(y, scores, color=TASK_COLORS[task], alpha=0.84)
        ax.axvline(CHANCE[task], color="#333333", linestyle=":", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(TASK_LABELS[task], fontweight="bold")
        ax.set_xlim(0.15, 1.02)
        ax.grid(axis="x")
        for yy, score in zip(y, scores):
            ax.text(score + 0.01, yy, f"{score:.2f}", va="center", fontsize=8)
    axes[0].set_xlabel("Test score")
    axes[1].set_xlabel("Test score")
    axes[2].set_xlabel("Test score")
    fig.suptitle(f"Top {top_k} runs by test score", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_figure(fig, out_dir, "05_top_runs_by_task")


def plot_val_test_scatter(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    for task in TASK_ORDER:
        for model, marker in MODEL_MARKERS.items():
            subset = [row for row in rows if row["task"] == task and row["model"] == model]
            if not subset:
                continue
            ax.scatter(
                [row["val_score_f"] for row in subset],
                [row["test_score_f"] for row in subset],
                s=50 if model == "linear" else 66,
                marker=marker,
                color=TASK_COLORS[task],
                alpha=0.74,
                edgecolor="white",
                linewidth=0.6,
                label=f"{TASK_LABELS[task]} {model}",
            )
    ax.plot([0.15, 1.0], [0.15, 1.0], color="#343A46", linestyle="--", linewidth=1.1, alpha=0.65)
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(0.15, 1.02)
    ax.set_xlabel("Validation score")
    ax.set_ylabel("Test score")
    ax.set_title("Validation vs test generalization", fontweight="bold")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "06_val_vs_test_scatter")


def plot_mlp_linear_deltas(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    readouts = ["all6_mean", "all6_flatten_pca64"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, readout in zip(axes, readouts):
        mat = np.full((len(TASK_ORDER), len(LAYER_ORDER)), np.nan)
        for i, task in enumerate(TASK_ORDER):
            for j, layer in enumerate(LAYER_ORDER):
                linear = [
                    row["test_score_f"]
                    for row in rows
                    if row["task"] == task and row["layer"] == layer and row["model"] == "linear" and row["readout"] == readout
                ]
                mlp = [
                    row["test_score_f"]
                    for row in rows
                    if row["task"] == task and row["layer"] == layer and row["model"] == "mlp" and row["readout"] == readout
                ]
                if linear and mlp:
                    mat[i, j] = mlp[0] - linear[0]
        im = ax.imshow(mat, cmap="coolwarm", vmin=-0.25, vmax=0.25, aspect="auto")
        ax.set_title(readout.replace("_", " "), fontweight="bold")
        ax.set_xticks(range(len(LAYER_ORDER)))
        ax.set_xticklabels([LAYER_LABELS[layer] for layer in LAYER_ORDER])
        ax.set_yticks(range(len(TASK_ORDER)))
        ax.set_yticklabels([TASK_LABELS[task] for task in TASK_ORDER])
        ax.tick_params(length=0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.03)
    cbar.set_label("MLP test score - linear test score")
    fig.suptitle("Capacity control deltas", fontsize=14, fontweight="bold", y=1.04)
    return save_figure(fig, out_dir, "07_mlp_vs_linear_deltas")


def plot_epoch_bars(rows: List[Dict[str, Any]], out_dir: Path) -> List[str]:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    positions = []
    labels = []
    values = []
    colors = []
    pos = 0
    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task and row["best_epoch_i"] is not None]
        positions.append(pos)
        labels.append(TASK_LABELS[task])
        values.append(median([row["best_epoch_i"] for row in task_rows]))
        colors.append(TASK_COLORS[task])
        pos += 1
    ax.bar(positions, values, color=colors, alpha=0.84)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median best epoch")
    ax.set_title("Early stopping behavior by task", fontweight="bold")
    ax.grid(axis="y")
    for x, value in zip(positions, values):
        ax.text(x, value + 0.7, f"{value:.0f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, out_dir, "08_best_epoch_by_task")


def best_row(rows: List[Dict[str, Any]], task: str, key: str) -> Dict[str, Any]:
    return max([row for row in rows if row["task"] == task], key=lambda row: row[key])


def write_index(rows: List[Dict[str, Any]], out_dir: Path, generated: Dict[str, List[str]]) -> None:
    summary: Dict[str, Any] = {
        "num_runs": len(rows),
        "figures": generated,
        "tasks": {},
    }
    lines = ["# Probe Figures", ""]
    lines.append(f"Runs plotted: {len(rows)}")
    lines.append("")
    lines.append("Main score: test_score. Presence uses balanced accuracy; state5 and position4 use accuracy.")
    lines.append("")
    lines.append("## Figures")
    for name, paths in generated.items():
        lines.append(f"- `{Path(paths[0]).name}`")
    lines.append("")
    lines.append("## Task Summary")
    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task]
        scores = [row["test_score_f"] for row in task_rows]
        best_test = best_row(rows, task, "test_score_f")
        best_val = best_row(rows, task, "val_score_f")
        payload = {
            "mean_test_score": mean(scores),
            "median_test_score": median(scores),
            "min_test_score": min(scores),
            "max_test_score": max(scores),
            "best_test_run": best_test["run_name"],
            "best_test_score": best_test["test_score_f"],
            "best_val_run": best_val["run_name"],
            "best_val_score": best_val["val_score_f"],
            "best_val_run_test_score": best_val["test_score_f"],
        }
        summary["tasks"][task] = payload
        lines.append(
            f"- {TASK_LABELS[task]}: mean={payload['mean_test_score']:.3f}, "
            f"median={payload['median_test_score']:.3f}, best_test={payload['best_test_score']:.3f} "
            f"({payload['best_test_run']}), best_val={payload['best_val_score']:.3f} "
            f"({payload['best_val_run']}, test={payload['best_val_run_test_score']:.3f})"
        )
    (out_dir / "figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "figure_index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot probe sweep summary figures.")
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "probe_summaries" / "broad_probe_sweep" / "probe_summary.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "probe_summaries" / "broad_probe_sweep" / "figures",
    )
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.summary_csv)
    generated: Dict[str, List[str]] = {}
    generated["layer_trends"] = plot_layer_trends(rows, args.output_dir)
    generated["task_layer_heatmaps"] = plot_task_layer_heatmaps(rows, args.output_dir)
    generated["readout_family_distributions"] = plot_readout_family_boxes(rows, args.output_dir)
    generated["single_slot_heatmaps"] = plot_slot_heatmaps(rows, args.output_dir)
    generated["top_runs_by_task"] = plot_top_runs(rows, args.output_dir, args.top_k)
    generated["val_vs_test_scatter"] = plot_val_test_scatter(rows, args.output_dir)
    generated["mlp_vs_linear_deltas"] = plot_mlp_linear_deltas(rows, args.output_dir)
    generated["best_epoch_by_task"] = plot_epoch_bars(rows, args.output_dir)
    write_index(rows, args.output_dir, generated)
    print(f"Plotted {len(rows)} runs")
    print(f"Wrote figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
