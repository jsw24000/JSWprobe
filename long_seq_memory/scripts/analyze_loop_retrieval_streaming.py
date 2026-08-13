#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_loop_retrieval import (  # noqa: E402
    MEMORY_TYPES,
    METRICS,
    QUERY_TYPES,
    age_matched_negative_indices,
    by_source_frame,
    metric_array,
)
from long_seq_memory.io_utils import ensure_dir, load_config, write_json  # noqa: E402


HEATMAP_SPECS = [
    ("all_queries", "all_memory", "attention"),
    ("all_queries", "all_memory", "contribution"),
    ("image_queries", "all_memory", "attention"),
    ("camera_query", "all_memory", "attention"),
    ("all_queries", "camera", "attention"),
    ("all_queries", "register", "attention"),
    ("all_queries", "scale", "attention"),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    reconstruction_dir = Path(args.reconstruction_dir or cfg["reconstruction_output_dir"]).expanduser()
    interaction_dir = Path(args.interaction_dir or cfg["interaction_output_dir"]).expanduser()
    output_dir = Path(args.output_dir or cfg["analysis_output_dir"]).expanduser()
    return {
        "reconstruction_dir": reconstruction_dir,
        "interaction_dir": interaction_dir,
        "output_dir": output_dir,
    }


def resolve_loop_dense(interaction_dir: Path) -> Path:
    index_path = interaction_dir / "interaction_index.json"
    if index_path.exists():
        index = load_json(index_path)
        loop_dense = index.get("files", {}).get("loop_dense")
        if loop_dense and Path(loop_dense).exists():
            return Path(loop_dense)
    fallback = interaction_dir / "loop_dense" / "loop_dense.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot find loop_dense.jsonl under {interaction_dir}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def ratio_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10.0)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def summarize_grouped_values(grouped: dict[tuple[Any, ...], list[float]], key_names: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        row = dict(zip(key_names, key))
        row.update(ratio_summary(values))
        out.append(row)
    return out


def render_heatmaps(
    accum: dict[tuple[str, str, str], dict[tuple[int, int], list[float]]],
    current_frames: set[int],
    source_frames: set[int],
    figures_dir: Path,
    positive_range: tuple[int, int],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    ensure_dir(figures_dir)
    if not current_frames or not source_frames:
        return []
    cur_list = sorted(current_frames)
    src_list = sorted(source_frames)
    cur_idx = {frame: idx for idx, frame in enumerate(cur_list)}
    src_idx = {frame: idx for idx, frame in enumerate(src_list)}
    outputs: list[str] = []

    for spec, cells in accum.items():
        if not cells:
            continue
        query_type, memory_type, metric = spec
        matrix = np.zeros((len(cur_list), len(src_list)), dtype=np.float64)
        counts = np.zeros_like(matrix)
        for (current_frame, source_frame), pair in cells.items():
            row = cur_idx[current_frame]
            col = src_idx[source_frame]
            matrix[row, col] += pair[0]
            counts[row, col] += pair[1]
        matrix = np.divide(matrix, np.maximum(counts, 1.0))

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_title(f"{query_type} / {memory_type} / {metric}")
        ax.set_xlabel("source frame")
        ax.set_ylabel("current frame")
        tick_count = min(8, len(src_list))
        if tick_count:
            ticks = np.linspace(0, len(src_list) - 1, tick_count, dtype=int)
            ax.set_xticks(ticks, [str(src_list[i]) for i in ticks], rotation=45, ha="right")
        tick_count = min(8, len(cur_list))
        if tick_count:
            ticks = np.linspace(0, len(cur_list) - 1, tick_count, dtype=int)
            ax.set_yticks(ticks, [str(cur_list[i]) for i in ticks])
        for bound in positive_range:
            if bound in src_idx:
                ax.axvline(src_idx[bound], color="white", linewidth=1.0, linestyle="--")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        filename = f"heatmap_{query_type}_{memory_type}_{metric}.png"
        out = figures_dir / filename
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(str(out))
    return outputs


def write_report(path: Path, summary: dict[str, Any]) -> None:
    old = summary.get("old_memory_attention_mass", {})
    top = summary.get("top_frame_recall", {})
    lines = [
        f"# Loop Retrieval Analysis: {summary.get('run_name')}",
        "",
        f"- Status: `{summary['status']}`",
        f"- Loop dense records: {summary.get('num_loop_dense_records', 0)}",
        f"- Records with by-source support: {summary.get('num_records_with_by_source', 0)}",
        f"- Frames without by-source support: {[row['current_frame'] for row in summary.get('frames_without_by_source', [])]}",
        f"- Positive history range: {summary.get('positive_history_range')}",
        "",
        "## Current Answer",
        "",
        summary.get("current_answer", "No complete analysis is available yet."),
        "",
        "## Key Numbers",
        "",
    ]
    if "all_queries" in old:
        lines.append(
            "- Old-memory attention mass, all queries: "
            f"mean={old['all_queries']['mean']:.6f}, "
            f"median={old['all_queries']['median']:.6f}"
        )
    if "all_queries" in top:
        lines.append(
            "- Top-32 target hit rate, all queries: "
            f"{top['all_queries']['top32_any_positive_fraction']:.6f}; "
            f"top-1 target hit rate={top['all_queries']['top1_positive_fraction']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    limitations = summary.get("limitations", [])
    if limitations:
        lines.extend([f"- {item}" for item in limitations])
    else:
        lines.append("- None recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_cell(
    cells: dict[tuple[int, int], list[float]],
    current_frame: int,
    source_frames: list[int],
    values: np.ndarray,
) -> None:
    for source_frame, value in zip(source_frames, values):
        key = (current_frame, int(source_frame))
        pair = cells.setdefault(key, [0.0, 0.0])
        pair[0] += float(value)
        pair[1] += 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reconstruction-dir", default=None)
    parser.add_argument("--interaction-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args)
    out_dir = ensure_dir(paths["output_dir"])
    figures_dir = ensure_dir(out_dir / "figures")
    loop_dense_path = resolve_loop_dense(paths["interaction_dir"])

    frame_ids = np.load(paths["reconstruction_dir"] / "frame_ids.npy")
    gt_poses = np.load(paths["reconstruction_dir"] / "gt_poses_c2w.npy")
    id_to_offset = {int(frame_id): idx for idx, frame_id in enumerate(frame_ids.tolist())}

    extraction = cfg.get("extraction", {})
    loop = cfg.get("loop_event", {})
    positive_range = tuple(int(v) for v in loop.get("history_segment", extraction.get("target_history_range", [3370, 3440])))
    history_frame = int(loop.get("history_frame", sum(positive_range) // 2))
    history_center = gt_poses[id_to_offset[history_frame], :3, 3] if history_frame in id_to_offset else None

    old_mass_ratios: dict[str, list[float]] = {q: [] for q in QUERY_TYPES}
    old_mass_by_layer: dict[tuple[int, str], list[float]] = defaultdict(list)
    entropy_values: dict[str, list[float]] = {q: [] for q in QUERY_TYPES}
    token_fraction_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    top_hit_counts: dict[str, Counter[str]] = {q: Counter() for q in QUERY_TYPES}
    top_target_frames: dict[str, Counter[int]] = {q: Counter() for q in QUERY_TYPES}
    enrichment_values: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    heatmap_cells: dict[tuple[str, str, str], dict[tuple[int, int], list[float]]] = {
        spec: {} for spec in HEATMAP_SPECS
    }
    heatmap_current_frames: set[int] = set()
    heatmap_source_frames: set[int] = set()
    per_current: dict[int, list[dict[str, float]]] = defaultdict(list)

    total_records = 0
    records_with_by_source = 0
    skipped_no_by_source = 0
    missing_by_source_layers: dict[int, list[int]] = defaultdict(list)

    for _, summary in iter_jsonl(loop_dense_path):
        total_records += 1
        frame_id = int(summary["frame_id"])
        layer_id = int(summary["layer_id"])
        aggregation = summary.get("aggregation", {})

        any_by_source = by_source_frame(summary, "all_queries", "all_memory") is not None
        if any_by_source:
            records_with_by_source += 1
        else:
            skipped_no_by_source += 1
            missing_by_source_layers[frame_id].append(layer_id)

        for query_type in QUERY_TYPES:
            query = aggregation.get(query_type)
            if not query:
                continue
            old = np.asarray(query.get("old_memory_total_attention_mass_by_head", []), dtype=np.float64)
            non_old = np.asarray(query.get("non_old_context_total_attention_mass_by_head", []), dtype=np.float64)
            if old.size and non_old.size:
                ratio = float(old.sum() / max(float(old.sum() + non_old.sum()), 1e-12))
                old_mass_ratios[query_type].append(ratio)
                old_mass_by_layer[(layer_id, query_type)].append(ratio)
            ent = query.get("normalized_source_entropy_by_head")
            if ent:
                entropy_values[query_type].append(float(np.mean(np.asarray(ent, dtype=np.float64))))

            top_frames = query.get("top_frames_by_mean_attention", [])
            target_ranks = [
                rank
                for rank, row in enumerate(top_frames, start=1)
                if positive_range[0] <= int(row.get("source_frame_id", -1)) <= positive_range[1]
            ]
            top_hit_counts[query_type]["records"] += 1
            top_hit_counts[query_type]["positive_top32_total"] += len(target_ranks)
            if target_ranks:
                top_hit_counts[query_type]["records_with_positive_top32"] += 1
                top_hit_counts[query_type]["best_rank_sum"] += min(target_ranks)
                top_hit_counts[query_type]["best_rank_count"] += 1
            if top_frames and positive_range[0] <= int(top_frames[0].get("source_frame_id", -1)) <= positive_range[1]:
                top_hit_counts[query_type]["records_with_positive_top1"] += 1
            for row in top_frames:
                src = int(row.get("source_frame_id", -1))
                if positive_range[0] <= src <= positive_range[1]:
                    top_target_frames[query_type][src] += 1

            token_totals = {}
            for token_type in ["camera", "register", "scale"]:
                token = query.get(token_type, {})
                values = token.get("total_attention_mass_by_head")
                if values:
                    token_totals[token_type] = float(np.asarray(values, dtype=np.float64).sum())
            denom = max(sum(token_totals.values()), 1e-12)
            for token_type, value in token_totals.items():
                token_fraction_values[(query_type, token_type)].append(value / denom)

        for query_type in QUERY_TYPES:
            for memory_type in MEMORY_TYPES:
                source = by_source_frame(summary, query_type, memory_type)
                if not source:
                    continue
                source_frames = [int(v) for v in source["source_frames"]]
                positive_mask = np.asarray(
                    [positive_range[0] <= frame <= positive_range[1] for frame in source_frames],
                    dtype=bool,
                )
                if not positive_mask.any():
                    continue
                negative_idx = age_matched_negative_indices(source_frames, frame_id, positive_mask)
                if negative_idx.size == 0:
                    continue
                for metric in METRICS:
                    values = metric_array(source, metric)
                    for head_id in range(values.shape[0]):
                        pos = values[head_id, positive_mask]
                        neg = values[head_id, negative_idx]
                        ratio = float(pos.mean() / max(float(neg.mean()), 1e-12))
                        enrichment_values[(layer_id, query_type, memory_type, metric)].append(ratio)

        all_source = by_source_frame(summary, "all_queries", "all_memory")
        if all_source:
            source_frames = [int(v) for v in all_source["source_frames"]]
            positive_mask = np.asarray([positive_range[0] <= frame <= positive_range[1] for frame in source_frames], dtype=bool)
            if positive_mask.any():
                attn = metric_array(all_source, "attention").mean(axis=0)
                contrib = metric_array(all_source, "contribution").mean(axis=0)
                row = {
                    "target_attention": float(attn[positive_mask].sum()),
                    "target_contribution": float(contrib[positive_mask].sum()),
                    "old_memory_absolute_mass": float(attn.sum()),
                }
                if history_center is not None and frame_id in id_to_offset:
                    center = gt_poses[id_to_offset[frame_id], :3, 3]
                    row["gt_distance_to_history_m"] = float(np.linalg.norm(center - history_center))
                per_current[frame_id].append(row)

        if not args.no_figures:
            for spec in HEATMAP_SPECS:
                query_type, memory_type, metric = spec
                source = by_source_frame(summary, query_type, memory_type)
                if not source:
                    continue
                source_frames = [int(v) for v in source["source_frames"]]
                values = metric_array(source, metric).mean(axis=0)
                add_cell(heatmap_cells[spec], frame_id, source_frames, values)
                heatmap_current_frames.add(frame_id)
                heatmap_source_frames.update(source_frames)

    enrichment_summary = summarize_grouped_values(
        enrichment_values,
        ["layer_id", "query_type", "memory_type", "metric"],
    )
    top_enrichment = sorted(enrichment_summary, key=lambda row: row.get("mean", 0.0), reverse=True)[:20]
    old_mass_summary = {q: ratio_summary(values) for q, values in old_mass_ratios.items()}
    old_mass_layer_summary = summarize_grouped_values(old_mass_by_layer, ["layer_id", "query_type"])
    entropy_summary = {q: ratio_summary(values) for q, values in entropy_values.items()}
    token_fraction_summary = {
        f"{query_type}/{token_type}": ratio_summary(values)
        for (query_type, token_type), values in sorted(token_fraction_values.items())
    }

    top_recall_summary: dict[str, Any] = {}
    for query_type, counts in top_hit_counts.items():
        records = max(int(counts["records"]), 1)
        hit_count = max(int(counts["best_rank_count"]), 1)
        top_recall_summary[query_type] = {
            "records": int(counts["records"]),
            "top32_any_positive_fraction": float(counts["records_with_positive_top32"] / records),
            "top1_positive_fraction": float(counts["records_with_positive_top1"] / records),
            "mean_positive_frames_in_top32": float(counts["positive_top32_total"] / records),
            "mean_best_positive_rank_when_present": float(counts["best_rank_sum"] / hit_count)
            if counts["best_rank_count"]
            else None,
            "most_common_positive_top_frames": [
                {"source_frame_id": int(frame), "count": int(count)}
                for frame, count in top_target_frames[query_type].most_common(20)
            ],
        }

    time_process = []
    for frame_id, rows in sorted(per_current.items()):
        out = {
            "current_frame": int(frame_id),
            "target_attention_mean_over_layers": float(np.mean([r["target_attention"] for r in rows])),
            "target_contribution_mean_over_layers": float(np.mean([r["target_contribution"] for r in rows])),
            "old_memory_absolute_mass_mean_over_layers": float(np.mean([r["old_memory_absolute_mass"] for r in rows])),
            "num_layers_observed": int(len(rows)),
        }
        if "gt_distance_to_history_m" in rows[0]:
            out["gt_distance_to_history_m"] = float(np.mean([r["gt_distance_to_history_m"] for r in rows]))
        time_process.append(out)

    figures = []
    if not args.no_figures:
        figures = render_heatmaps(heatmap_cells, heatmap_current_frames, heatmap_source_frames, figures_dir, positive_range)

    status = "complete" if enrichment_summary else "incomplete"
    missing_by_source_frames = [
        {"current_frame": int(frame), "layers": [int(layer) for layer in sorted(layers)]}
        for frame, layers in sorted(missing_by_source_layers.items())
    ]
    all_old = old_mass_summary.get("all_queries", {})
    all_top = top_recall_summary.get("all_queries", {})
    current_answer = (
        "Loop-window interaction summaries were analyzed with streaming by-source support. "
        f"Mean old-memory attention mass for all queries is {all_old.get('mean', float('nan')):.6f}; "
        f"the positive loop history appears in top-32 source frames for "
        f"{all_top.get('top32_any_positive_fraction', float('nan')):.3%} of all-query records."
    )
    if status != "complete":
        current_answer = "Loop dense files existed, but by-source support was insufficient for enrichment analysis."

    summary = {
        "status": status,
        "run_name": cfg.get("run_name"),
        "loop_dense_path": str(loop_dense_path),
        "num_loop_dense_records": total_records,
        "num_records_with_by_source": records_with_by_source,
        "num_records_without_by_source": skipped_no_by_source,
        "frames_without_by_source": missing_by_source_frames,
        "positive_history_range": list(positive_range),
        "history_frame": history_frame,
        "old_memory_attention_mass": old_mass_summary,
        "old_memory_attention_mass_by_layer": old_mass_layer_summary,
        "normalized_source_entropy": entropy_summary,
        "token_type_attention_fraction": token_fraction_summary,
        "top_frame_recall": top_recall_summary,
        "top_enrichment": top_enrichment,
        "num_enrichment_groups": len(enrichment_summary),
        "time_process": time_process,
        "figures": figures,
        "current_answer": current_answer,
        "limitations": [
            "Frames listed in frames_without_by_source have compact loop summaries without by_source_frame, so enrichment and heatmaps exclude those current-frame/layer records.",
            "Positive loop history is the pose-validated 3370-3440 segment; TLS overlap is not part of this analysis yet.",
            "Contribution is a value-norm weighted proxy, not a causal intervention.",
        ],
    }

    write_json(out_dir / "enrichment_summary.json", enrichment_summary)
    write_json(out_dir / "old_memory_attention_by_layer.json", old_mass_layer_summary)
    write_json(out_dir / "top_frame_recall_summary.json", top_recall_summary)
    write_json(out_dir / "time_process.json", time_process)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)
    print(f"Wrote streaming analysis summary: {out_dir / 'summary.json'}")
    print(f"Status: {status}")
    print(current_answer)


if __name__ == "__main__":
    main()
