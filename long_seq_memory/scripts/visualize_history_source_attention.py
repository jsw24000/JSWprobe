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


SELECTED_LAYERS = [4, 11, 17, 23]
QUERY_SPECS = {
    "all_queries": [("all_queries", None)],
    "image_queries": [("image_queries", None)],
    "special_queries": [("camera_query", 1), ("register_queries", 4), ("scale_query", 1)],
}


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
        }
    )
    return plt


def resolve_loop_dense(interaction_dir: Path) -> Path:
    index_path = interaction_dir / "interaction_index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        path = index.get("files", {}).get("loop_dense")
        if path and Path(path).exists():
            return Path(path)
    fallback = interaction_dir / "loop_dense" / "loop_dense.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot find loop_dense.jsonl below {interaction_dir}")


def by_source(summary: dict[str, Any], query_type: str) -> dict[str, Any] | None:
    query = summary.get("aggregation", {}).get(query_type)
    if not query:
        return None
    return query.get("by_source_frame")


def q_count(summary: dict[str, Any], query_name: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    q_tokens = int(summary.get("q_shape", [0, 0])[1])
    if query_name == "image_queries":
        return max(q_tokens - 6, 1)
    if query_name == "all_queries":
        return max(q_tokens, 1)
    raise ValueError(query_name)


def source_values(summary: dict[str, Any], spec_name: str) -> tuple[list[int], np.ndarray] | None:
    arrays = []
    source_frames: list[int] | None = None
    total_q = 0
    heads = None
    for query_name, explicit_count in QUERY_SPECS[spec_name]:
        source = by_source(summary, query_name)
        if not source:
            return None
        frames = [int(v) for v in source["source_frames"]]
        if source_frames is None:
            source_frames = frames
        elif source_frames != frames:
            raise ValueError("Source frame grids differ across query groups.")
        arr = np.asarray(source["attention_mass_by_head_frame"], dtype=np.float64)
        if heads is None:
            heads = arr.shape[0]
        arrays.append(arr.sum(axis=0))
        total_q += q_count(summary, query_name, explicit_count)
    if source_frames is None or heads is None:
        return None
    values = np.sum(np.stack(arrays, axis=0), axis=0) / max(float(heads * total_q), 1e-12)
    return source_frames, values


def add_record(
    accum: dict[tuple[str, str, str], dict[tuple[int, int], list[float]]],
    summary: dict[str, Any],
    selected_layers: set[int],
    positive_range: tuple[int, int],
    target_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
) -> bool:
    frame_id = int(summary["frame_id"])
    layer_id = int(summary["layer_id"])
    if by_source(summary, "all_queries") is None:
        return False
    groups = ["all_layers"]
    if layer_id in selected_layers:
        groups.append(f"layer{layer_id:02d}")
    for spec_name in QUERY_SPECS:
        result = source_values(summary, spec_name)
        if result is None:
            return False
        source_frames, absolute = result
        conditional = absolute / max(float(absolute.sum()), 1e-12)
        target_mask = np.asarray(
            [positive_range[0] <= source <= positive_range[1] for source in source_frames],
            dtype=bool,
        )
        target_rows.append(
            {
                "current_frame": frame_id,
                "layer_id": layer_id,
                "query_spec": spec_name,
                "target_history_absolute_mass": float(absolute[target_mask].sum()),
                "target_history_conditional_mass": float(conditional[target_mask].sum()),
            }
        )
        top_idx = np.argsort(-conditional)[:5]
        for rank, idx in enumerate(top_idx, start=1):
            top_rows.append(
                {
                    "current_frame": frame_id,
                    "layer_id": layer_id,
                    "query_spec": spec_name,
                    "rank": rank,
                    "source_frame": int(source_frames[idx]),
                    "conditional_probability": float(conditional[idx]),
                    "absolute_mass": float(absolute[idx]),
                }
            )
        for group in groups:
            for metric_name, values in [("absolute", absolute), ("conditional", conditional)]:
                store = accum[(spec_name, group, metric_name)]
                for source_frame, value in zip(source_frames, values):
                    cell = store.setdefault((frame_id, int(source_frame)), [0.0, 0.0])
                    cell[0] += float(value)
                    cell[1] += 1.0
    return True


def matrix_from_cells(cells: dict[tuple[int, int], list[float]]) -> tuple[np.ndarray, list[int], list[int]]:
    current_frames = sorted({key[0] for key in cells})
    source_frames = sorted({key[1] for key in cells})
    cur_idx = {frame: idx for idx, frame in enumerate(current_frames)}
    src_idx = {frame: idx for idx, frame in enumerate(source_frames)}
    matrix = np.full((len(current_frames), len(source_frames)), np.nan, dtype=np.float64)
    for (current_frame, source_frame), pair in cells.items():
        matrix[cur_idx[current_frame], src_idx[source_frame]] = pair[0] / max(pair[1], 1.0)
    return matrix, current_frames, source_frames


def draw_heatmap(ax: Any, matrix: np.ndarray, current_frames: list[int], source_frames: list[int], title: str) -> Any:
    data = np.nan_to_num(matrix, nan=0.0)
    im = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_ylabel("current frame")
    tick_count = min(6, len(current_frames))
    if tick_count:
        ticks = np.linspace(0, len(current_frames) - 1, tick_count, dtype=int)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(current_frames[i]) for i in ticks])
    tick_count = min(8, len(source_frames))
    if tick_count:
        ticks = np.linspace(0, len(source_frames) - 1, tick_count, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(source_frames[i]) for i in ticks], rotation=35, ha="right")
    ax.set_xlabel("historical source frame")
    return im


def plot_query_heatmaps(
    accum: dict[tuple[str, str, str], dict[tuple[int, int], list[float]]],
    figures_dir: Path,
    metric: str,
) -> str:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    for ax, spec_name in zip(axes, ["all_queries", "image_queries", "special_queries"]):
        matrix, current_frames, source_frames = matrix_from_cells(accum[(spec_name, "all_layers", metric)])
        im = draw_heatmap(ax, matrix, current_frames, source_frames, f"{spec_name} | all selected loop frames | {metric}")
        fig.colorbar(im, ax=ax, label=metric)
    out = figures_dir / f"history_source_attention_{metric}_all_layers.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_selected_layer_heatmaps(
    accum: dict[tuple[str, str, str], dict[tuple[int, int], list[float]]],
    figures_dir: Path,
    metric: str,
    spec_name: str,
    selected_layers: list[int],
) -> str:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(len(selected_layers), 1, figsize=(13, 10), constrained_layout=True)
    for ax, layer in zip(axes, selected_layers):
        group = f"layer{layer:02d}"
        matrix, current_frames, source_frames = matrix_from_cells(accum[(spec_name, group, metric)])
        im = draw_heatmap(ax, matrix, current_frames, source_frames, f"{spec_name} | layer {layer} | {metric}")
        fig.colorbar(im, ax=ax, label=metric)
    out = figures_dir / f"history_source_attention_{spec_name}_{metric}_selected_layers.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def nearest_available_frames(targets: list[int], available: list[int]) -> list[int]:
    out = []
    for target in targets:
        nearest = min(available, key=lambda frame: (abs(frame - target), frame))
        if nearest not in out:
            out.append(nearest)
    return out


def plot_source_profiles(
    accum: dict[tuple[str, str, str], dict[tuple[int, int], list[float]]],
    figures_dir: Path,
    requested_frames: list[int],
) -> str:
    plt = setup_matplotlib()
    matrix_by_spec = {}
    available_frames: list[int] | None = None
    source_frames: list[int] | None = None
    for spec_name in ["all_queries", "image_queries", "special_queries"]:
        matrix, current_frames, src = matrix_from_cells(accum[(spec_name, "all_layers", "conditional")])
        matrix_by_spec[spec_name] = matrix
        available_frames = current_frames
        source_frames = src
    assert available_frames is not None and source_frames is not None
    selected = nearest_available_frames(requested_frames, available_frames)
    frame_to_row = {frame: idx for idx, frame in enumerate(available_frames)}
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for ax, spec_name in zip(axes, ["all_queries", "image_queries", "special_queries"]):
        matrix = matrix_by_spec[spec_name]
        for frame in selected:
            ax.plot(source_frames, matrix[frame_to_row[frame]], label=f"current {frame}", linewidth=1.5)
        ax.set_title(f"{spec_name}: historical source profile")
        ax.set_ylabel("old-memory conditional probability")
        ax.legend(ncol=min(5, len(selected)), fontsize=8)
    axes[-1].set_xlabel("historical source frame")
    out = figures_dir / "history_source_profiles_selected_current_frames.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def average_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["current_frame"], row["query_spec"])].append(float(row[key]))
    out = []
    for (frame, spec), values in sorted(grouped.items()):
        out.append({"current_frame": frame, "query_spec": spec, key: float(np.mean(values))})
    return out


def plot_target_history_mass(target_rows: list[dict[str, Any]], figures_dir: Path) -> str:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for ax, key, title in [
        (axes[0], "target_history_absolute_mass", "Target History Absolute Mass"),
        (axes[1], "target_history_conditional_mass", "Target History Conditional Mass"),
    ]:
        averaged = average_rows(target_rows, key)
        for spec_name in ["all_queries", "image_queries", "special_queries"]:
            rows = [row for row in averaged if row["query_spec"] == spec_name]
            ax.plot([r["current_frame"] for r in rows], [r[key] for r in rows], label=spec_name, linewidth=1.7)
        ax.set_title(title)
        ax.set_ylabel(key)
        ax.legend()
    axes[-1].set_xlabel("current frame")
    out = figures_dir / "target_history_attention_over_current.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_top_source_scatter(top_rows: list[dict[str, Any]], figures_dir: Path) -> str:
    plt = setup_matplotlib()
    averaged: dict[tuple[int, str, int, int], list[float]] = defaultdict(list)
    for row in top_rows:
        averaged[(row["current_frame"], row["query_spec"], row["rank"], row["source_frame"])].append(
            row["conditional_probability"]
        )
    rows = [
        {
            "current_frame": key[0],
            "query_spec": key[1],
            "rank": key[2],
            "source_frame": key[3],
            "conditional_probability": float(np.mean(values)),
        }
        for key, values in averaged.items()
        if key[2] <= 3
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for ax, spec_name in zip(axes, ["all_queries", "image_queries", "special_queries"]):
        spec_rows = [row for row in rows if row["query_spec"] == spec_name]
        x = [row["current_frame"] for row in spec_rows]
        y = [row["source_frame"] for row in spec_rows]
        c = [row["conditional_probability"] for row in spec_rows]
        sizes = [70 / row["rank"] for row in spec_rows]
        sc = ax.scatter(x, y, c=c, s=sizes, cmap="magma", alpha=0.85)
        ax.set_title(f"{spec_name}: top historical source frames")
        ax.set_ylabel("source frame")
        fig.colorbar(sc, ax=ax, label="conditional probability")
    axes[-1].set_xlabel("current frame")
    out = figures_dir / "history_source_top_frames_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--interaction-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--selected-layers", default=",".join(str(v) for v in SELECTED_LAYERS))
    parser.add_argument("--profile-frames", default="4381,4400,4414,4430,4445")
    args = parser.parse_args()

    cfg = load_config(args.config)
    interaction_dir = Path(args.interaction_dir or cfg["interaction_output_dir"]).expanduser()
    output_dir = ensure_dir(
        args.output_dir
        or Path(cfg.get("analysis_output_dir", interaction_dir.parent / "analysis")).expanduser().parent
        / "history_source_attention"
    )
    figures_dir = ensure_dir(output_dir / "figures")
    loop_dense_path = resolve_loop_dense(interaction_dir)
    selected_layers = [int(v) for v in args.selected_layers.split(",") if v.strip()]
    profile_targets = [int(v) for v in args.profile_frames.split(",") if v.strip()]
    positive_range = tuple(int(v) for v in cfg.get("extraction", {}).get("target_history_range", [3370, 3440]))

    accum: dict[tuple[str, str, str], dict[tuple[int, int], list[float]]] = defaultdict(dict)
    target_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    total = 0
    used = 0
    skipped: dict[int, list[int]] = defaultdict(list)
    with loop_dense_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            summary = json.loads(line)
            ok = add_record(
                accum=accum,
                summary=summary,
                selected_layers=set(selected_layers),
                positive_range=positive_range,
                target_rows=target_rows,
                top_rows=top_rows,
            )
            if ok:
                used += 1
            else:
                skipped[int(summary["frame_id"])].append(int(summary["layer_id"]))
            if total % 100 == 0:
                print(f"processed {total} records, used {used}", flush=True)

    figures = [
        plot_query_heatmaps(accum, figures_dir, "conditional"),
        plot_query_heatmaps(accum, figures_dir, "absolute"),
        plot_selected_layer_heatmaps(accum, figures_dir, "conditional", "all_queries", selected_layers),
        plot_selected_layer_heatmaps(accum, figures_dir, "conditional", "image_queries", selected_layers),
        plot_selected_layer_heatmaps(accum, figures_dir, "conditional", "special_queries", selected_layers),
        plot_source_profiles(accum, figures_dir, profile_targets),
        plot_target_history_mass(target_rows, figures_dir),
        plot_top_source_scatter(top_rows, figures_dir),
    ]
    summary = {
        "status": "complete",
        "loop_dense_path": str(loop_dense_path),
        "total_records": total,
        "used_records_with_by_source": used,
        "skipped_records_without_by_source": total - used,
        "skipped_frames": [
            {"current_frame": int(frame), "layers": [int(layer) for layer in sorted(layers)]}
            for frame, layers in sorted(skipped.items())
        ],
        "selected_layers": selected_layers,
        "profile_requested_frames": profile_targets,
        "target_history_range": list(positive_range),
        "figures": figures,
        "top_rows_preview": top_rows[:100],
        "target_rows_preview": target_rows[:100],
    }
    write_json(output_dir / "history_source_attention_summary.json", summary)
    print(f"Wrote history source attention summary: {output_dir / 'history_source_attention_summary.json'}")
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
