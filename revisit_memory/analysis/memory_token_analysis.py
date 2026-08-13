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

from revisit_memory.utils.io import load_config, output_root, write_csv, write_json  # noqa: E402


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("memory_token_analysis.py requires torch to load memory token .pt files.") from exc


def load_memory(cfg: dict[str, Any], condition: str):
    torch = require_torch()
    path = output_root(cfg) / "memory_tokens" / condition / "special_tokens.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing memory token file: {path}")
    return torch.load(path, map_location="cpu")


def token_l2(a, b):
    torch = require_torch()
    return torch.linalg.norm(a.float() - b.float(), dim=-1)


def source_band(cfg: dict[str, Any], source_frame: int) -> str:
    anchor_start, anchor_end = [int(x) for x in cfg["frames"]["anchor_frames"]]
    first_start, first_end = [int(x) for x in cfg["frames"]["first_loop_visible"]]
    second_start, second_end = [int(x) for x in cfg["frames"]["second_loop_eval"]]
    revisit = int(cfg["frames"]["expected_first_revisit_frame"])
    if anchor_start <= source_frame <= anchor_end:
        return "anchor_background"
    if first_start <= source_frame <= first_end:
        return "first_loop_memory_target_interval"
    if source_frame < revisit:
        return "post_memory_gap_before_revisit"
    if second_start <= source_frame <= second_end:
        return "second_loop_eval"
    return "outside_eval"


def band_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["memory_group"], row["layer"], row["token_type"], row["source_band"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (memory_group, layer, token_type, band), group_rows in sorted(grouped.items()):
        removed_absent = np.asarray([r["seen_removed_vs_absent_l2"] for r in group_rows], dtype=np.float64)
        present_absent = np.asarray([r["always_present_vs_absent_l2"] for r in group_rows], dtype=np.float64)
        removed_present = np.asarray([r["seen_removed_vs_present_l2"] for r in group_rows], dtype=np.float64)
        summary.append(
            {
                "memory_group": memory_group,
                "layer": layer,
                "token_type": token_type,
                "source_band": band,
                "num_source_frames": len(group_rows),
                "seen_removed_vs_absent_l2_mean": float(np.mean(removed_absent)),
                "seen_removed_vs_absent_l2_p95": float(np.percentile(removed_absent, 95)),
                "always_present_vs_absent_l2_mean": float(np.mean(present_absent)),
                "seen_removed_vs_present_l2_mean": float(np.mean(removed_present)),
            }
        )
    return summary


def run(cfg: dict[str, Any]) -> None:
    torch = require_torch()
    payloads = {condition: load_memory(cfg, condition) for condition in cfg["conditions"]}
    layers = [int(x) for x in payloads["seen_then_removed"]["selected_layers"]]
    token_names = payloads["seen_then_removed"]["token_type_names"]
    source_sets = {cond: set(payloads[cond]["source_frame_ids"]) for cond in cfg["conditions"]}
    common_sources = sorted(set.intersection(*source_sets.values()))
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for cond, sources in source_sets.items():
        missing = sorted(set.union(*source_sets.values()) - sources)
        for fid in missing:
            missing_rows.append({"condition": cond, "source_frame": int(fid), "status": "missing"})

    index_maps = {
        cond: {int(fid): idx for idx, fid in enumerate(payloads[cond]["source_frame_ids"])}
        for cond in cfg["conditions"]
    }

    for layer in layers:
        for memory_group in ["frame_special_tokens", "global_special_tokens"]:
            removed_tokens = payloads["seen_then_removed"][memory_group][str(layer)]
            absent_tokens = payloads["always_absent"][memory_group][str(layer)]
            present_tokens = payloads["always_present"][memory_group][str(layer)]
            for source_frame in common_sources:
                ridx = index_maps["seen_then_removed"][source_frame]
                aidx = index_maps["always_absent"][source_frame]
                pidx = index_maps["always_present"][source_frame]
                l2_removed_absent = token_l2(removed_tokens[ridx], absent_tokens[aidx])
                l2_present_absent = token_l2(present_tokens[pidx], absent_tokens[aidx])
                l2_removed_present = token_l2(removed_tokens[ridx], present_tokens[pidx])
                norms = torch.linalg.norm(removed_tokens[ridx].float(), dim=-1)
                for token_idx, token_name in enumerate(token_names):
                    rows.append(
                        {
                            "memory_group": memory_group,
                            "layer": layer,
                            "source_frame": int(source_frame),
                            "source_band": source_band(cfg, int(source_frame)),
                            "token_index": token_idx,
                            "token_type": token_name,
                            "seen_removed_vs_absent_l2": float(l2_removed_absent[token_idx]),
                            "always_present_vs_absent_l2": float(l2_present_absent[token_idx]),
                            "seen_removed_vs_present_l2": float(l2_removed_present[token_idx]),
                            "seen_removed_token_norm": float(norms[token_idx]),
                            "is_first_loop_target_visible_frame": bool(
                                cfg["frames"]["first_loop_visible"][0]
                                <= source_frame
                                <= cfg["frames"]["first_loop_visible"][1]
                            ),
                        }
                    )

    current_rows = []
    max_frame = max(common_sources) if common_sources else -1
    for layer in layers:
        l2_by_source = {
            row["source_frame"]: row["seen_removed_vs_absent_l2"]
            for row in rows
            if row["memory_group"] == "global_special_tokens" and row["layer"] == layer and row["token_type"] == "camera"
        }
        for current_time in range(max_frame + 1):
            vals = [v for source, v in l2_by_source.items() if source <= current_time]
            current_rows.append(
                {
                    "layer": layer,
                    "current_time": current_time,
                    "mean_memory_state_l2_camera_token": float(np.mean(vals)) if vals else float("nan"),
                    "num_aligned_source_frames": len(vals),
                }
            )

    write_csv(output_root(cfg) / "metrics" / "memory_token_metrics.csv", rows)
    write_csv(output_root(cfg) / "metrics" / "memory_token_state_metrics.csv", current_rows)
    write_csv(output_root(cfg) / "metrics" / "memory_token_missing_entries.csv", missing_rows)
    band_rows = band_summary_rows(rows)
    write_csv(output_root(cfg) / "metrics" / "memory_token_band_summary.csv", band_rows)
    sample_frames = set(int(x) for x in cfg["analysis"].get("memory_token_sample_frames", []))
    sampled_rows = [row for row in rows if int(row["source_frame"]) in sample_frames]
    write_csv(output_root(cfg) / "metrics" / "memory_token_sampled_metrics.csv", sampled_rows)

    first_start, first_end = cfg["frames"]["first_loop_visible"]
    first_rows = [r for r in rows if first_start <= r["source_frame"] <= first_end]
    other_rows = [r for r in rows if not (first_start <= r["source_frame"] <= first_end)]
    summary = {
        "aligned_source_frames": common_sources,
        "missing_entries": missing_rows,
        "first_loop_target_visible_frame_mean_l2": float(np.mean([r["seen_removed_vs_absent_l2"] for r in first_rows]))
        if first_rows
        else float("nan"),
        "other_frame_mean_l2": float(np.mean([r["seen_removed_vs_absent_l2"] for r in other_rows])) if other_rows else float("nan"),
        "sampled_source_frames": sorted(sample_frames),
        "band_summary_csv": str(output_root(cfg) / "metrics" / "memory_token_band_summary.csv"),
        "sampled_metrics_csv": str(output_root(cfg) / "metrics" / "memory_token_sampled_metrics.csv"),
        "note": "This analysis uses saved special/context tokens from frame/global block outputs, not raw K/V pages.",
    }
    write_json(output_root(cfg) / "analysis" / "memory_token_simple" / "summary.json", summary)
    torch.save(
        {
            "rows_lite": rows,
            "state_rows": current_rows,
            "token_type_names": token_names,
            "layers": layers,
        },
        output_root(cfg) / "analysis" / "memory_token_simple" / "raw_memory_metrics.pt",
    )
    print(f"Wrote {output_root(cfg) / 'metrics' / 'memory_token_metrics.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple special memory-token analysis.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
