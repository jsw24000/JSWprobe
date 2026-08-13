#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, load_config, read_table, write_json


QUERY_TYPES = ["all_queries", "image_queries", "camera_query", "register_queries", "scale_query"]
MEMORY_TYPES = ["all_memory", "camera", "register", "scale"]
METRICS = ["attention", "contribution"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def default_paths(cfg: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Path | None]:
    if cfg is None:
        return {
            "reconstruction_dir": Path(args.reconstruction_dir).expanduser() if args.reconstruction_dir else None,
            "interaction_dir": Path(args.interaction_dir).expanduser() if args.interaction_dir else None,
            "schedule": Path(args.schedule).expanduser() if args.schedule else None,
            "output_dir": Path(args.output_dir).expanduser() if args.output_dir else None,
        }
    run_name = cfg.get("run_name", "run")
    output_dir = Path(cfg["output_dir"]).expanduser()
    return {
        "reconstruction_dir": Path(args.reconstruction_dir or cfg.get("reconstruction_output_dir", "")).expanduser(),
        "interaction_dir": Path(args.interaction_dir or cfg.get("interaction_output_dir", "")).expanduser(),
        "schedule": Path(args.schedule).expanduser()
        if args.schedule
        else output_dir / "planning" / run_name / "keyframe_schedule.parquet",
        "output_dir": Path(args.output_dir or cfg.get("analysis_output_dir", "")).expanduser(),
    }


def existing_required(paths: dict[str, Path | None]) -> tuple[list[str], dict[str, Path]]:
    required = {
        "reconstruction_dir": paths["reconstruction_dir"],
        "interaction_dir": paths["interaction_dir"],
        "schedule": paths["schedule"],
        "output_dir": paths["output_dir"],
    }
    missing = []
    existing: dict[str, Path] = {}
    for name, path in required.items():
        if path is None or str(path) == ".":
            missing.append(name)
        elif name == "output_dir":
            existing[name] = path
        elif not path.exists():
            missing.append(str(path))
        else:
            existing[name] = path
    if "reconstruction_dir" in existing:
        for filename in ["run_manifest.json", "memory_state_summary.jsonl", "gt_poses_c2w.npy", "frame_ids.npy"]:
            path = existing["reconstruction_dir"] / filename
            if not path.exists():
                missing.append(str(path))
    return missing, existing


def resolve_summary_paths(interaction_dir: Path) -> list[Path]:
    index_path = interaction_dir / "interaction_index.json"
    if index_path.exists():
        index = load_json(index_path)
        files = index.get("files", {})
        if files.get("loop_dense") and Path(files["loop_dense"]).exists():
            return [Path(files["loop_dense"])]
        return [Path(p) for p in index.get("summaries", []) if Path(p).exists()]
    return sorted(interaction_dir.glob("*_interaction_summary.json"))


def load_interaction_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        summaries.append(json.loads(line))
        else:
            summaries.append(load_json(path))
    return summaries


def by_source_frame(summary: dict[str, Any], query_type: str, memory_type: str) -> dict[str, Any] | None:
    query = summary.get("aggregation", {}).get(query_type)
    if not query:
        return None
    if memory_type == "all_memory":
        return query.get("by_source_frame")
    return query.get(memory_type, {}).get("by_source_frame")


def metric_array(source: dict[str, Any], metric: str) -> np.ndarray:
    key = "attention_mass_by_head_frame" if metric == "attention" else "contribution_proxy_by_head_frame"
    return np.asarray(source[key], dtype=np.float64)


def age_matched_negative_indices(source_frames: list[int], current_frame: int, positive_mask: np.ndarray) -> np.ndarray:
    source = np.asarray(source_frames, dtype=np.int64)
    positive_ages = current_frame - source[positive_mask]
    candidates = np.where(~positive_mask)[0]
    if candidates.size == 0 or positive_ages.size == 0:
        return np.array([], dtype=np.int64)
    target_age = float(np.median(positive_ages))
    order = sorted(candidates.tolist(), key=lambda idx: (abs((current_frame - int(source[idx])) - target_age), int(source[idx])))
    return np.asarray(order[: positive_ages.size], dtype=np.int64)


def bootstrap_ratio_ci(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator, samples: int = 500) -> list[float] | None:
    if pos.size == 0 or neg.size == 0:
        return None
    ratios = []
    for _ in range(samples):
        pos_sample = rng.choice(pos, size=pos.size, replace=True)
        neg_sample = rng.choice(neg, size=neg.size, replace=True)
        ratios.append(float(pos_sample.mean() / max(float(neg_sample.mean()), 1e-12)))
    return [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]


def compute_enrichment(summaries: list[dict[str, Any]], positive_range: tuple[int, int]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    records = []
    for summary in summaries:
        frame_id = int(summary["frame_id"])
        layer_id = int(summary["layer_id"])
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
                        records.append(
                            {
                                "current_frame": frame_id,
                                "layer_id": layer_id,
                                "head_id": head_id,
                                "query_type": query_type,
                                "memory_type": memory_type,
                                "metric": metric,
                                "positive_mean": float(pos.mean()),
                                "negative_mean": float(neg.mean()),
                                "enrichment_ratio": ratio,
                                "bootstrap_ratio_ci95": bootstrap_ratio_ci(pos, neg, rng),
                                "num_positive_frames": int(pos.size),
                                "num_negative_frames": int(neg.size),
                            }
                        )
    return records


def summarize_enrichment(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in records:
        key = (row["layer_id"], row["query_type"], row["memory_type"], row["metric"])
        grouped.setdefault(key, []).append(float(row["enrichment_ratio"]))
    out = []
    for (layer_id, query_type, memory_type, metric), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        out.append(
            {
                "layer_id": int(layer_id),
                "query_type": query_type,
                "memory_type": memory_type,
                "metric": metric,
                "mean_enrichment_ratio": float(arr.mean()),
                "median_enrichment_ratio": float(np.median(arr)),
                "num_head_frame_samples": int(arr.size),
            }
        )
    return out


def compute_time_process(
    summaries: list[dict[str, Any]],
    gt_poses: np.ndarray,
    frame_ids: np.ndarray,
    history_frame: int,
    positive_range: tuple[int, int],
) -> list[dict[str, Any]]:
    id_to_offset = {int(frame_id): idx for idx, frame_id in enumerate(frame_ids.tolist())}
    if history_frame not in id_to_offset:
        return []
    history_center = gt_poses[id_to_offset[history_frame], :3, 3]
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        source = by_source_frame(summary, "all_queries", "all_memory")
        if not source:
            continue
        source_frames = [int(v) for v in source["source_frames"]]
        positive_mask = np.asarray([positive_range[0] <= frame <= positive_range[1] for frame in source_frames], dtype=bool)
        if not positive_mask.any():
            continue
        attn = metric_array(source, "attention").mean(axis=0)
        contrib = metric_array(source, "contribution").mean(axis=0)
        frame_id = int(summary["frame_id"])
        by_frame.setdefault(frame_id, []).append(
            {
                "target_attention": float(attn[positive_mask].sum()),
                "target_contribution": float(contrib[positive_mask].sum()),
                "old_memory_absolute_mass": float(attn.sum()),
            }
        )
    rows = []
    for frame_id, values in sorted(by_frame.items()):
        if frame_id not in id_to_offset:
            continue
        center = gt_poses[id_to_offset[frame_id], :3, 3]
        rows.append(
            {
                "current_frame": frame_id,
                "gt_distance_to_history_m": float(np.linalg.norm(center - history_center)),
                "target_attention_mean_over_layers": float(np.mean([v["target_attention"] for v in values])),
                "target_contribution_mean_over_layers": float(np.mean([v["target_contribution"] for v in values])),
                "old_memory_absolute_mass_mean_over_layers": float(np.mean([v["old_memory_absolute_mass"] for v in values])),
                "num_layers_observed": len(values),
            }
        )
    return rows


def render_heatmaps(summaries: list[dict[str, Any]], figures_dir: Path, positive_range: tuple[int, int]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    ensure_dir(figures_dir)
    outputs = []
    specs = [
        ("all_queries", "all_memory", "attention"),
        ("all_queries", "all_memory", "contribution"),
        ("image_queries", "all_memory", "attention"),
        ("camera_query", "all_memory", "attention"),
        ("all_queries", "camera", "attention"),
        ("all_queries", "register", "attention"),
        ("all_queries", "scale", "attention"),
    ]
    for query_type, memory_type, metric in specs:
        current_frames = sorted({int(s["frame_id"]) for s in summaries})
        source_frames = sorted(
            {
                int(frame)
                for summary in summaries
                for source in [by_source_frame(summary, query_type, memory_type)]
                if source
                for frame in source["source_frames"]
            }
        )
        if not current_frames or not source_frames:
            continue
        cur_idx = {frame: idx for idx, frame in enumerate(current_frames)}
        src_idx = {frame: idx for idx, frame in enumerate(source_frames)}
        matrix = np.zeros((len(current_frames), len(source_frames)), dtype=np.float64)
        counts = np.zeros_like(matrix)
        for summary in summaries:
            source = by_source_frame(summary, query_type, memory_type)
            if not source:
                continue
            values = metric_array(source, metric).mean(axis=0)
            row = cur_idx[int(summary["frame_id"])]
            for frame, value in zip(source["source_frames"], values):
                col = src_idx[int(frame)]
                matrix[row, col] += float(value)
                counts[row, col] += 1.0
        matrix = np.divide(matrix, np.maximum(counts, 1.0))
        filename = f"heatmap_{query_type}_{memory_type}_{metric}.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_title(f"{query_type} / {memory_type} / {metric}")
        ax.set_xlabel("source frame")
        ax.set_ylabel("current frame")
        tick_count = min(8, len(source_frames))
        if tick_count:
            ticks = np.linspace(0, len(source_frames) - 1, tick_count, dtype=int)
            ax.set_xticks(ticks, [str(source_frames[i]) for i in ticks], rotation=45, ha="right")
        tick_count = min(8, len(current_frames))
        if tick_count:
            ticks = np.linspace(0, len(current_frames) - 1, tick_count, dtype=int)
            ax.set_yticks(ticks, [str(current_frames[i]) for i in ticks])
        for bound in positive_range:
            if bound in src_idx:
                ax.axvline(src_idx[bound], color="white", linewidth=1.0, linestyle="--")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out = figures_dir / filename
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(str(out))
    return outputs


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Loop Retrieval Analysis: {summary.get('run_name')}",
        "",
        f"- Status: `{summary['status']}`",
        f"- Missing inputs: {summary.get('missing_inputs', [])}",
        f"- Interaction summaries used: {summary.get('num_interaction_summaries', 0)}",
        f"- Enrichment rows: {summary.get('num_enrichment_rows', 0)}",
        "",
        "## Current Answer",
        "",
        summary.get("current_answer", "No complete analysis is available yet."),
        "",
        "## Limitations",
        "",
    ]
    limitations = summary.get("limitations", [])
    if limitations:
        lines.extend([f"- {item}" for item in limitations])
    else:
        lines.append("- None recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--reconstruction-dir", default=None)
    parser.add_argument("--interaction-dir", default=None)
    parser.add_argument("--schedule", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    paths = default_paths(cfg, args)
    missing, existing = existing_required(paths)
    out_dir = ensure_dir(existing.get("output_dir") or Path(args.output_dir or "analysis/loop_3403_4414"))
    figures_dir = ensure_dir(out_dir / "figures")
    run_name = cfg.get("run_name") if cfg else None
    loop = cfg.get("loop_event", {}) if cfg else {}
    history_frame = int(loop.get("history_frame", 3403))
    positive_range = tuple(loop.get("history_segment", [3370, 3440]))

    if missing:
        summary = {
            "status": "incomplete",
            "run_name": run_name,
            "missing_inputs": missing,
            "current_answer": "Required reconstruction, schedule, or interaction inputs are missing; no loop retrieval conclusion was computed.",
            "limitations": ["No fake analysis was generated."],
        }
        write_json(out_dir / "summary.json", summary)
        write_report(out_dir / "report.md", summary)
        if not args.allow_incomplete:
            raise SystemExit(f"Missing inputs: {missing}")
        return

    interaction_paths = resolve_summary_paths(existing["interaction_dir"])
    if not interaction_paths:
        summary = {
            "status": "incomplete",
            "run_name": run_name,
            "missing_inputs": [str(existing["interaction_dir"] / "interaction_index.json")],
            "current_answer": "No real interaction summaries were found, so retrieval heatmaps and enrichment were not computed.",
            "limitations": ["Run extract_memory_interactions.py or online extraction first."],
        }
        write_json(out_dir / "summary.json", summary)
        write_report(out_dir / "report.md", summary)
        if not args.allow_incomplete:
            raise SystemExit("No real interaction summaries found.")
        return

    summaries = load_interaction_summaries(interaction_paths)
    frame_ids = np.load(existing["reconstruction_dir"] / "frame_ids.npy")
    gt_poses = np.load(existing["reconstruction_dir"] / "gt_poses_c2w.npy")
    memory_rows = load_jsonl(existing["reconstruction_dir"] / "memory_state_summary.jsonl")
    schedule_rows = read_table(existing["schedule"])
    enrichment = compute_enrichment(summaries, positive_range)
    enrichment_summary = summarize_enrichment(enrichment)
    time_process = compute_time_process(summaries, gt_poses, frame_ids, history_frame, positive_range)
    figures = render_heatmaps(summaries, figures_dir, positive_range)

    best = sorted(enrichment_summary, key=lambda row: row["mean_enrichment_ratio"], reverse=True)[:10]
    current_answer = (
        "Real interaction summaries were analyzed. "
        "Use the enrichment tables and generated heatmaps to determine whether the loop segment receives above-baseline retrieval."
    )
    if not enrichment:
        current_answer = "Interaction summaries existed, but none contained positive/negative source-frame support for enrichment."

    summary = {
        "status": "complete" if enrichment else "incomplete",
        "run_name": run_name,
        "num_interaction_summaries": len(summaries),
        "num_memory_rows": len(memory_rows),
        "num_schedule_rows": len(schedule_rows),
        "positive_history_range": list(positive_range),
        "num_enrichment_rows": len(enrichment),
        "top_enrichment": best,
        "time_process": time_process,
        "figures": figures,
        "current_answer": current_answer,
        "limitations": [
            "Patch-specific query specificity requires Level 3 patch summaries; aggregate QKV summaries alone cannot prove patch-level diversity.",
            "TLS overlap is not yet used here; positive history defaults to the validated pose loop segment.",
            "Results depend on available real interaction summaries and do not infer missing layers or frames.",
        ],
    }
    write_json(out_dir / "enrichment_records.json", enrichment)
    write_json(out_dir / "enrichment_summary.json", enrichment_summary)
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)
    print(f"Wrote analysis summary: {out_dir / 'summary.json'}")
    print(f"Status: {summary['status']}")

    if summary["status"] != "complete" and not args.allow_incomplete:
        raise SystemExit("Analysis is incomplete; inspect summary.json for missing support.")


if __name__ == "__main__":
    main()
