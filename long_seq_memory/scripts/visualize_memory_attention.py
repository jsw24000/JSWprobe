#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, load_config, write_json  # noqa: E402


SPECIAL_QUERY_TYPES = ["camera_query", "register_queries", "scale_query"]
SELECTED_LAYERS = [4, 11, 17, 23]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mass_ratio(query_payload: dict[str, Any]) -> float:
    old = np.asarray(query_payload["old_memory_total_attention_mass_by_head"], dtype=np.float64)
    non_old = np.asarray(query_payload["non_old_context_total_attention_mass_by_head"], dtype=np.float64)
    return float(old.sum() / max(float((old + non_old).sum()), 1e-12))


def combined_special_mass_ratio(aggregation: dict[str, Any]) -> float:
    old_total = 0.0
    denom_total = 0.0
    for query_type in SPECIAL_QUERY_TYPES:
        payload = aggregation[query_type]
        old = np.asarray(payload["old_memory_total_attention_mass_by_head"], dtype=np.float64)
        non_old = np.asarray(payload["non_old_context_total_attention_mass_by_head"], dtype=np.float64)
        old_total += float(old.sum())
        denom_total += float((old + non_old).sum())
    return old_total / max(denom_total, 1e-12)


def stream_global_summary(global_path: Path) -> list[dict[str, Any]]:
    rows = []
    with global_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            aggregation = record["aggregation"]
            old_tokens = int(record["old_memory_token_count"])
            total_tokens = int(record["k_shape"][1])
            non_old_tokens = max(total_tokens - old_tokens, 0)
            rows.append(
                {
                    "frame_id": int(record["frame_id"]),
                    "layer_id": int(record["layer_id"]),
                    "old_memory_tokens": old_tokens,
                    "non_old_visible_tokens": non_old_tokens,
                    "visible_tokens": total_tokens,
                    "old_token_fraction": old_tokens / max(total_tokens, 1),
                    "image_old_mass": mass_ratio(aggregation["image_queries"]),
                    "special_old_mass": combined_special_mass_ratio(aggregation),
                    "camera_old_mass": mass_ratio(aggregation["camera_query"]),
                    "register_old_mass": mass_ratio(aggregation["register_queries"]),
                    "scale_old_mass": mass_ratio(aggregation["scale_query"]),
                    "all_old_mass": mass_ratio(aggregation["all_queries"]),
                }
            )
    return rows


def rows_to_grid(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, list[int], list[int]]:
    frames = sorted({row["frame_id"] for row in rows})
    layers = sorted({row["layer_id"] for row in rows})
    frame_index = {frame: idx for idx, frame in enumerate(frames)}
    layer_index = {layer: idx for idx, layer in enumerate(layers)}
    grid = np.full((len(layers), len(frames)), np.nan, dtype=np.float64)
    for row in rows:
        grid[layer_index[row["layer_id"]], frame_index[row["frame_id"]]] = float(row[key])
    return grid, frames, layers


def group_by_frame(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["frame_id"]].append(row)
    out = []
    for frame, items in sorted(grouped.items()):
        out.append(
            {
                "frame_id": frame,
                "old_memory_tokens": float(np.mean([item["old_memory_tokens"] for item in items])),
                "non_old_visible_tokens": float(np.mean([item["non_old_visible_tokens"] for item in items])),
                "visible_tokens": float(np.mean([item["visible_tokens"] for item in items])),
                "old_token_fraction": float(np.mean([item["old_token_fraction"] for item in items])),
                "image_old_mass": float(np.mean([item["image_old_mass"] for item in items])),
                "special_old_mass": float(np.mean([item["special_old_mass"] for item in items])),
                "all_old_mass": float(np.mean([item["all_old_mass"] for item in items])),
            }
        )
    return out


def group_by_layer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["layer_id"]].append(row)
    out = []
    for layer, items in sorted(grouped.items()):
        out.append(
            {
                "layer_id": layer,
                "image_old_mass_mean": float(np.mean([item["image_old_mass"] for item in items])),
                "image_old_mass_p10": float(np.percentile([item["image_old_mass"] for item in items], 10)),
                "image_old_mass_p90": float(np.percentile([item["image_old_mass"] for item in items], 90)),
                "special_old_mass_mean": float(np.mean([item["special_old_mass"] for item in items])),
                "special_old_mass_p10": float(np.percentile([item["special_old_mass"] for item in items], 10)),
                "special_old_mass_p90": float(np.percentile([item["special_old_mass"] for item in items], 90)),
            }
        )
    return out


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    return plt


def plot_layer_heatmaps(rows: list[dict[str, Any]], figures_dir: Path) -> str:
    plt = setup_matplotlib()
    image_grid, frames, layers = rows_to_grid(rows, "image_old_mass")
    special_grid, _, _ = rows_to_grid(rows, "special_old_mass")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for ax, grid, title in [
        (axes[0], image_grid, "Image Patch Queries: Old-Memory Mass"),
        (axes[1], special_grid, "Special Queries: Old-Memory Mass"),
    ]:
        im = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("Layer")
        ax.set_yticks(np.arange(len(layers))[:: max(1, len(layers) // 8)])
        ax.set_yticklabels([str(v) for v in layers[:: max(1, len(layers) // 8)]])
        fig.colorbar(im, ax=ax, label="old memory mass")
    tick_count = min(10, len(frames))
    ticks = np.linspace(0, len(frames) - 1, tick_count, dtype=int)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([str(frames[i]) for i in ticks], rotation=35, ha="right")
    axes[-1].set_xlabel("Current frame")
    out = figures_dir / "old_memory_mass_layer_heatmaps.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_selected_layer_timeseries(rows: list[dict[str, Any]], figures_dir: Path, selected_layers: list[int]) -> str:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for layer in selected_layers:
        layer_rows = sorted([row for row in rows if row["layer_id"] == layer], key=lambda r: r["frame_id"])
        if not layer_rows:
            continue
        frames = [row["frame_id"] for row in layer_rows]
        axes[0].plot(frames, [row["image_old_mass"] for row in layer_rows], label=f"L{layer}", linewidth=1.5)
        axes[1].plot(frames, [row["special_old_mass"] for row in layer_rows], label=f"L{layer}", linewidth=1.5)
    axes[0].set_title("Image Patch Queries")
    axes[1].set_title("Special Queries")
    for ax in axes:
        ax.set_ylabel("old memory mass")
        ax.legend(ncol=len(selected_layers), fontsize=9)
    axes[-1].set_xlabel("Current frame")
    out = figures_dir / "old_memory_mass_selected_layers_timeseries.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_layer_profile(layer_rows: list[dict[str, Any]], figures_dir: Path) -> str:
    plt = setup_matplotlib()
    layers = np.asarray([row["layer_id"] for row in layer_rows], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for prefix, label, color in [
        ("image_old_mass", "image patch queries", "tab:blue"),
        ("special_old_mass", "special queries", "tab:orange"),
    ]:
        mean = np.asarray([row[f"{prefix}_mean"] for row in layer_rows], dtype=np.float64)
        p10 = np.asarray([row[f"{prefix}_p10"] for row in layer_rows], dtype=np.float64)
        p90 = np.asarray([row[f"{prefix}_p90"] for row in layer_rows], dtype=np.float64)
        ax.plot(layers, mean, label=label, color=color, linewidth=2)
        ax.fill_between(layers, p10, p90, color=color, alpha=0.15, linewidth=0)
    ax.set_title("Old-Memory Mass By Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("old memory mass")
    ax.legend()
    out = figures_dir / "old_memory_mass_by_layer_profile.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_token_counts_and_mass(frame_rows: list[dict[str, Any]], figures_dir: Path) -> str:
    plt = setup_matplotlib()
    frames = np.asarray([row["frame_id"] for row in frame_rows], dtype=np.int64)
    old_tokens = np.asarray([row["old_memory_tokens"] for row in frame_rows], dtype=np.float64)
    non_old_tokens = np.asarray([row["non_old_visible_tokens"] for row in frame_rows], dtype=np.float64)
    old_fraction = np.asarray([row["old_token_fraction"] for row in frame_rows], dtype=np.float64)
    image_mass = np.asarray([row["image_old_mass"] for row in frame_rows], dtype=np.float64)
    special_mass = np.asarray([row["special_old_mass"] for row in frame_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    axes[0].plot(frames, old_tokens, label="old trajectory-memory tokens", color="tab:red", linewidth=2)
    axes[0].plot(frames, non_old_tokens, label="non-old visible tokens", color="tab:gray", linewidth=2)
    axes[0].set_ylabel("token count")
    axes[0].set_title("Visible Token Counts Over Time")
    axes[0].legend()

    axes[1].plot(frames, old_fraction, label="old tokens / visible tokens", color="tab:red", linewidth=2)
    axes[1].plot(frames, image_mass, label="image query old-memory mass", color="tab:blue", linewidth=1.8)
    axes[1].plot(frames, special_mass, label="special query old-memory mass", color="tab:orange", linewidth=1.8)
    axes[1].set_ylabel("ratio")
    axes[1].set_xlabel("Current frame")
    axes[1].set_title("Token Fraction vs Attention Mass")
    axes[1].legend()
    out = figures_dir / "token_counts_vs_old_memory_mass_timeseries.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_token_fraction_scatter(frame_rows: list[dict[str, Any]], figures_dir: Path) -> str:
    plt = setup_matplotlib()
    frames = np.asarray([row["frame_id"] for row in frame_rows], dtype=np.float64)
    x = np.asarray([row["old_token_fraction"] for row in frame_rows], dtype=np.float64)
    image = np.asarray([row["image_old_mass"] for row in frame_rows], dtype=np.float64)
    special = np.asarray([row["special_old_mass"] for row in frame_rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, y, title in [
        (axes[0], image, "Image Patch Queries"),
        (axes[1], special, "Special Queries"),
    ]:
        sc = ax.scatter(x, y, c=frames, s=18, cmap="viridis", alpha=0.8)
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
        ax.set_title(f"{title} | r={corr:.3f}")
        ax.set_xlabel("old tokens / visible tokens")
        ax.set_ylabel("old memory mass")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="current frame")
    out = figures_dir / "old_token_fraction_vs_old_memory_mass_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def load_patch_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def plot_patch_metric_bars(patch_summary: dict[str, Any], figures_dir: Path) -> str:
    plt = setup_matplotlib()
    records = sorted(patch_summary["records"], key=lambda r: (r["frame_id"], r["layer_id"]))
    labels = [f"{r['frame_id']}\nL{r['layer_id']}" for r in records]
    full_cv = [r.get("old_memory_full_context_mass", {}).get("cv", np.nan) for r in records]
    source_cos = [r["source_distribution_cosine_to_frame_mean"]["mean"] for r in records]
    output_cos = [r["memory_output_cosine_to_frame_mean"]["mean"] for r in records]
    x = np.arange(len(records))
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].bar(x, full_cv, color="tab:red")
    axes[0].set_title("Patch Variation In Full-Context Old-Memory Mass")
    axes[0].set_ylabel("CV")
    axes[1].bar(x, source_cos, color="tab:blue")
    axes[1].set_title("Old-Memory Source Distribution Similarity Across Patches")
    axes[1].set_ylabel("cosine to frame mean")
    axes[1].set_ylim(0.9, 1.005)
    axes[2].bar(x, output_cos, color="tab:green")
    axes[2].set_title("Memory-Only Output Similarity Across Patches")
    axes[2].set_ylabel("cosine to frame mean")
    axes[2].set_ylim(0.9, 1.005)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=0, fontsize=8)
    out = figures_dir / "patch_specific_metric_bars.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def render_patch_grid_montage(
    patch_dir: Path,
    patch_summary: dict[str, Any],
    figures_dir: Path,
    frame_filter: list[int],
    layer_filter: list[int],
    name: str,
) -> str:
    plt = setup_matplotlib()
    metrics = {(r["frame_id"], r["layer_id"]): r for r in patch_summary["records"]}
    rows = []
    for frame in frame_filter:
        for layer in layer_filter:
            path = patch_dir / "maps" / f"patch_memory_frame{frame:06d}_layer{layer:02d}.npz"
            if path.exists():
                rows.append((frame, layer, path))
    cols = [
        ("old_memory_full_context_mass", "full old mass"),
        ("loop_history_conditional_mass", "loop mass"),
        ("source_cosine_to_frame_mean", "source cos"),
        ("memory_output_cosine_to_frame_mean", "output cos"),
    ]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.1 * len(cols), 2.2 * max(1, len(rows))), squeeze=False)
    for row_idx, (frame, layer, path) in enumerate(rows):
        z = np.load(path)
        metric = metrics[(frame, layer)]
        for col_idx, (key, title) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            if key not in z.files:
                ax.axis("off")
                continue
            im = ax.imshow(z[key], aspect="auto")
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(f"F{frame} L{layer}\n{metric['classification']}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    out = figures_dir / name
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def build_summary(rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([row["old_token_fraction"] for row in frame_rows], dtype=np.float64)
    image = np.asarray([row["image_old_mass"] for row in frame_rows], dtype=np.float64)
    special = np.asarray([row["special_old_mass"] for row in frame_rows], dtype=np.float64)
    return {
        "num_records": len(rows),
        "frame_range": [int(min(row["frame_id"] for row in rows)), int(max(row["frame_id"] for row in rows))],
        "layers": sorted({int(row["layer_id"]) for row in rows}),
        "selected_layers": SELECTED_LAYERS,
        "mean_image_old_mass": float(np.mean([row["image_old_mass"] for row in rows])),
        "mean_special_old_mass": float(np.mean([row["special_old_mass"] for row in rows])),
        "mean_old_token_fraction": float(np.mean([row["old_token_fraction"] for row in rows])),
        "corr_old_token_fraction_image_mass_by_frame": float(np.corrcoef(x, image)[0, 1]),
        "corr_old_token_fraction_special_mass_by_frame": float(np.corrcoef(x, special)[0, 1]),
        "layer_profile": layer_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--interaction-dir", default=None)
    parser.add_argument("--patch-specific-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    interaction_dir = Path(args.interaction_dir or cfg["interaction_output_dir"]).expanduser()
    default_output = Path(cfg.get("analysis_output_dir", interaction_dir.parent / "analysis")).expanduser().parent / "visualizations"
    output_dir = ensure_dir(args.output_dir or default_output)
    figures_dir = ensure_dir(output_dir / "figures")
    global_path = interaction_dir / "global_summary" / "global_summary.jsonl"
    patch_dir = Path(
        args.patch_specific_dir
        or interaction_dir.parent / "analysis" / "patch_memory_specificity_full_context"
    ).expanduser()

    rows = stream_global_summary(global_path)
    frame_rows = group_by_frame(rows)
    layer_rows = group_by_layer(rows)
    figures = [
        plot_layer_heatmaps(rows, figures_dir),
        plot_selected_layer_timeseries(rows, figures_dir, SELECTED_LAYERS),
        plot_layer_profile(layer_rows, figures_dir),
        plot_token_counts_and_mass(frame_rows, figures_dir),
        plot_token_fraction_scatter(frame_rows, figures_dir),
    ]
    patch_summary = load_patch_summary(patch_dir / "summary.json")
    if patch_summary is not None:
        figures.append(plot_patch_metric_bars(patch_summary, figures_dir))
        figures.append(
            render_patch_grid_montage(
                patch_dir,
                patch_summary,
                figures_dir,
                frame_filter=[2755, 4400, 4414, 4430],
                layer_filter=[11],
                name="patch_specific_layer11_montage.png",
            )
        )
        figures.append(
            render_patch_grid_montage(
                patch_dir,
                patch_summary,
                figures_dir,
                frame_filter=[4414],
                layer_filter=SELECTED_LAYERS,
                name="patch_specific_frame4414_layers_montage.png",
            )
        )

    summary = build_summary(rows, frame_rows, layer_rows)
    summary["figures"] = figures
    summary["patch_specific_summary"] = str(patch_dir / "summary.json") if patch_summary is not None else None
    write_json(output_dir / "memory_attention_visualization_summary.json", summary)
    print(f"Wrote visualization summary: {output_dir / 'memory_attention_visualization_summary.json'}")
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
