#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, load_config, write_json


QKV_RE = re.compile(r"qkv_frame(?P<frame>\d+)_layer(?P<layer>\d+)_")
NUM_SPECIAL = 6
OLD_TOKEN_TYPES = ["camera", "register", "register", "register", "register", "scale"]


def load_memory_rows(run_dir: Path) -> dict[int, dict]:
    rows = {}
    path = run_dir / "memory_state_summary.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[int(row["current_frame"])] = row
    return rows


def token_groups_for_old_memory(row: dict) -> list[dict]:
    old_frames = row.get("old_memory_frames")
    if old_frames is None:
        frame_range = row.get("old_memory_frame_range")
        if not frame_range:
            return []
        start, end = frame_range
        old_frames = list(range(int(start), int(end) + 1))
    groups = []
    slot = 0
    for frame_id in old_frames:
        for local_idx, token_type in enumerate(OLD_TOKEN_TYPES):
            groups.append(
                {
                    "slot": slot,
                    "source_frame_id": int(frame_id),
                    "local_index": local_idx,
                    "token_type": token_type,
                    "token_type_index": local_idx - 1 if token_type == "register" else 0,
                }
            )
            slot += 1
    return groups


def parse_qkv_name(path: Path) -> tuple[int, int]:
    match = QKV_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse frame/layer from {path.name}")
    return int(match.group("frame")), int(match.group("layer"))


def compute_attention_summary(q: np.ndarray, k: np.ndarray, v: np.ndarray, old_token_count: int, chunk_size: int = 64):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_t = torch.from_numpy(q[0]).to(device=device, dtype=torch.float32)  # [H, Q, D]
    k_t = torch.from_numpy(k[0]).to(device=device, dtype=torch.float32)  # [H, K, D]
    # Only old-memory V is needed for the contribution proxy.
    v_old_norm = torch.linalg.norm(torch.from_numpy(v[0, :, :old_token_count]).to(device=device, dtype=torch.float32), dim=-1)
    heads, q_tokens, dim = q_t.shape
    scale = 1.0 / math.sqrt(dim)

    old_mass = torch.zeros((heads, q_tokens, old_token_count), device=device, dtype=torch.float32)
    full_old_mass_by_query = torch.zeros((heads, q_tokens), device=device, dtype=torch.float32)
    for h in range(heads):
        k_h = k_t[h].T.contiguous()
        for start in range(0, q_tokens, chunk_size):
            end = min(start + chunk_size, q_tokens)
            logits = (q_t[h, start:end] @ k_h) * scale
            attn = torch.softmax(logits, dim=-1)
            old_mass[h, start:end] = attn[:, :old_token_count]
            full_old_mass_by_query[h, start:end] = attn[:, :old_token_count].sum(dim=-1)
            del logits, attn

    contribution_proxy = old_mass * v_old_norm[:, None, :]
    return (
        old_mass.cpu().numpy(),
        contribution_proxy.cpu().numpy(),
        {"full_old_mass_by_head_query": full_old_mass_by_query.cpu().numpy()},
    )


def aggregate_by_frame_and_type(mass: np.ndarray, contribution: np.ndarray, groups: list[dict]) -> dict:
    # mass/contribution: [H, Q, old_tokens]
    heads, q_tokens, _ = mass.shape
    query_sets = {
        "camera_query": np.array([0]),
        "register_queries": np.arange(1, 5),
        "scale_query": np.array([5]),
        "image_queries": np.arange(6, q_tokens),
        "all_queries": np.arange(q_tokens),
    }
    source_frames = sorted({g["source_frame_id"] for g in groups})
    token_types = ["camera", "register", "scale"]

    out: dict[str, dict] = {}
    for query_name, q_idx in query_sets.items():
        if len(q_idx) == 0:
            continue
        out[query_name] = {}
        for token_type in token_types:
            token_indices = np.array([g["slot"] for g in groups if g["token_type"] == token_type], dtype=np.int64)
            if len(token_indices) == 0:
                continue
            type_mass = mass[:, q_idx][:, :, token_indices]
            type_contrib = contribution[:, q_idx][:, :, token_indices]
            out[query_name][token_type] = {
                "total_attention_mass_by_head": type_mass.sum(axis=(1, 2)).tolist(),
                "mean_attention_mass_by_head": type_mass.mean(axis=(1, 2)).tolist(),
                "total_contribution_proxy_by_head": type_contrib.sum(axis=(1, 2)).tolist(),
                "mean_contribution_proxy_by_head": type_contrib.mean(axis=(1, 2)).tolist(),
            }

            type_frame_mass = []
            type_frame_contrib = []
            for frame_id in source_frames:
                slots = np.array(
                    [g["slot"] for g in groups if g["source_frame_id"] == frame_id and g["token_type"] == token_type],
                    dtype=np.int64,
                )
                type_frame_mass.append(mass[:, q_idx][:, :, slots].sum(axis=(1, 2)))
                type_frame_contrib.append(contribution[:, q_idx][:, :, slots].sum(axis=(1, 2)))
            type_frame_mass_arr = np.stack(type_frame_mass, axis=1)
            type_frame_contrib_arr = np.stack(type_frame_contrib, axis=1)
            out[query_name][token_type]["by_source_frame"] = {
                "source_frames": source_frames,
                "attention_mass_by_head_frame": type_frame_mass_arr.tolist(),
                "contribution_proxy_by_head_frame": type_frame_contrib_arr.tolist(),
            }

        frame_mass = []
        frame_contrib = []
        for frame_id in source_frames:
            slots = np.array([g["slot"] for g in groups if g["source_frame_id"] == frame_id], dtype=np.int64)
            frame_mass.append(mass[:, q_idx][:, :, slots].sum(axis=(1, 2)))
            frame_contrib.append(contribution[:, q_idx][:, :, slots].sum(axis=(1, 2)))
        frame_mass_arr = np.stack(frame_mass, axis=1)
        frame_contrib_arr = np.stack(frame_contrib, axis=1)
        frame_prob = frame_mass_arr / np.maximum(frame_mass_arr.sum(axis=1, keepdims=True), 1e-12)
        frame_entropy = -(frame_prob * np.log(np.maximum(frame_prob, 1e-12))).sum(axis=1)
        normalized_entropy = frame_entropy / max(math.log(max(len(source_frames), 2)), 1e-12)
        top_idx = np.argsort(-frame_mass_arr.mean(axis=0))[:10]
        out[query_name]["by_source_frame"] = {
            "source_frames": source_frames,
            "attention_mass_by_head_frame": frame_mass_arr.tolist(),
            "contribution_proxy_by_head_frame": frame_contrib_arr.tolist(),
            "old_memory_conditional_probability_by_head_frame": frame_prob.tolist(),
            "source_entropy_by_head": frame_entropy.tolist(),
            "normalized_source_entropy_by_head": normalized_entropy.tolist(),
            "top_frames_by_mean_attention": [
                {
                    "source_frame_id": int(source_frames[i]),
                    "mean_attention_mass": float(frame_mass_arr[:, i].mean()),
                    "mean_old_conditional_probability": float(frame_prob[:, i].mean()),
                }
                for i in top_idx
            ],
        }
        out[query_name]["normalization"] = {
            "attention_mass": "full_context_softmax_denominator",
            "old_memory_conditional_probability": "renormalized_within_old_memory_attention_mass",
        }
    return out


def assign_summary_levels(summary: dict, cfg: dict | None) -> list[str]:
    if cfg is None:
        return []
    extraction = cfg.get("extraction", {})
    frame_id = int(summary["frame_id"])
    levels = []
    stride = int(extraction.get("global_summary_stride", 10))
    if stride > 0 and frame_id % stride == 0:
        levels.append("global_summary")
    loop_range = extraction.get("loop_current_range", cfg.get("loop_event", {}).get("current_segment", [4380, 4445]))
    if int(loop_range[0]) <= frame_id <= int(loop_range[1]):
        levels.append("loop_dense")
    selected = set(int(v) for v in extraction.get("selected_current_frames", []))
    raw = set(int(v) for v in extraction.get("raw_qkv_frames", []))
    if frame_id in selected or frame_id in raw:
        levels.append("patch_specific")
    if frame_id in raw:
        levels.append("raw_qkv_debug")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    qkv_dir = run_dir / "qkv"
    if not qkv_dir.exists():
        if args.allow_incomplete:
            out_dir = ensure_dir(args.output_dir or (run_dir / "interactions"))
            write_json(
                out_dir / "interaction_index.json",
                {
                    "status": "incomplete",
                    "missing_inputs": [str(qkv_dir)],
                    "summaries": [],
                },
            )
            return
        raise SystemExit(f"No Q/K/V directory found: {qkv_dir}")
    rows = load_memory_rows(run_dir)
    out_dir = ensure_dir(args.output_dir or (run_dir / "interactions"))
    cfg = load_config(args.config) if args.config else None

    outputs = []
    level_index: dict[str, list[str]] = {
        "global_summary": [],
        "loop_dense": [],
        "patch_specific": [],
        "raw_qkv_debug": [],
    }
    for path in sorted(qkv_dir.glob("*.npz")):
        frame_id, layer_id = parse_qkv_name(path)
        if frame_id not in rows:
            print(f"Skipping {path.name}: no memory row for frame {frame_id}")
            continue
        row = rows[frame_id]
        groups = token_groups_for_old_memory(row)
        if not groups:
            print(f"Skipping {path.name}: no old memory at frame {frame_id}")
            continue
        old_token_count = len(groups)
        data = np.load(path)
        capture_kind = str(data["capture_kind"]) if "capture_kind" in data else "unknown"
        if capture_kind != "visible_memory_qkv":
            print(f"Skipping {path.name}: capture_kind={capture_kind}")
            continue
        q = data["q"]
        k = data["k"]
        v = data["v"]
        mass, contribution, dense_stats = compute_attention_summary(q, k, v, old_token_count, chunk_size=args.chunk_size)
        summary = {
            "status": "complete",
            "frame_id": frame_id,
            "layer_id": layer_id,
            "capture_file": str(path),
            "old_memory_frame_range": row["old_memory_frame_range"],
            "old_memory_frames": row.get("old_memory_frames"),
            "old_memory_token_count": old_token_count,
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "index_base": 0,
            "normalization_note": {
                "full_context_mass": "Softmax denominator is every visible KV token.",
                "old_memory_conditional_distribution": "Attention is renormalized after slicing old trajectory-memory tokens.",
            },
            "full_old_memory_mass_by_head_mean_query": dense_stats["full_old_mass_by_head_query"].mean(axis=1).tolist(),
            "aggregation": aggregate_by_frame_and_type(mass, contribution, groups),
        }
        out_path = out_dir / f"frame{frame_id:06d}_layer{layer_id:02d}_interaction_summary.json"
        write_json(out_path, summary)
        outputs.append(str(out_path))
        for level in assign_summary_levels(summary, cfg):
            level_index.setdefault(level, []).append(str(out_path))
        print(f"Wrote {out_path}")
    write_json(
        out_dir / "interaction_index.json",
        {
            "status": "complete" if outputs else "incomplete",
            "summaries": outputs,
            "levels": level_index,
        },
    )


if __name__ == "__main__":
    main()
