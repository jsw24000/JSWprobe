from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from long_seq_memory.extraction_plan import build_extraction_plan
from long_seq_memory.io_utils import ensure_dir, json_ready, write_json
from long_seq_memory.interaction_hooks import QKVRecord


QUERY_SETS = {
    "camera_query": [0],
    "register_queries": [1, 2, 3, 4],
    "scale_query": [5],
    "image_queries": None,
    "all_queries": None,
}
TOKEN_TYPE_SLOTS = {
    "camera": [0],
    "register": [1, 2, 3, 4],
    "scale": [5],
}


@dataclass(frozen=True)
class InteractionSelection:
    capture_frames: set[int]
    capture_layers: set[int]
    raw_frames: set[int]
    raw_layers: set[int]


def _as_torch(tensor: Any):
    import torch

    if torch.is_tensor(tensor):
        return tensor
    return torch.from_numpy(tensor)


def _query_indices(name: str, q_tokens: int, device) -> Any:
    import torch

    if name == "image_queries":
        return torch.arange(6, q_tokens, device=device, dtype=torch.long)
    if name == "all_queries":
        return torch.arange(0, q_tokens, device=device, dtype=torch.long)
    return torch.tensor(QUERY_SETS[name], device=device, dtype=torch.long)


def token_metadata_for_old_memory(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = []
    old_frames = row.get("old_memory_frames") or []
    slot = 0
    for frame_id in old_frames:
        for local_index, token_type in enumerate(["camera", "register", "register", "register", "register", "scale"]):
            metadata.append(
                {
                    "global_k_index": slot,
                    "old_memory_slot": slot,
                    "source_frame_id": int(frame_id),
                    "source_local_index": local_index,
                    "token_type": token_type,
                    "token_type_index": local_index - 1 if token_type == "register" else 0,
                    "stream": "old_trajectory_memory",
                }
            )
            slot += 1
    return metadata


def build_interaction_selection(cfg: dict[str, Any]) -> InteractionSelection:
    plan = build_extraction_plan(cfg)
    levels = plan["levels"]
    capture_frames = set(levels["global_summary"]["frames"])
    capture_frames.update(levels["loop_dense"]["current_frames"])
    capture_frames.update(levels["raw_qkv_debug"]["frames"])
    capture_layers = set(levels["global_summary"]["layers"])
    capture_layers.update(levels["loop_dense"]["layers"])
    capture_layers.update(levels["raw_qkv_debug"]["layers"])
    return InteractionSelection(
        capture_frames={int(v) for v in capture_frames},
        capture_layers={int(v) for v in capture_layers},
        raw_frames={int(v) for v in levels["raw_qkv_debug"]["frames"]},
        raw_layers={int(v) for v in levels["raw_qkv_debug"]["layers"]},
    )


class OnlineInteractionWriter:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.plan = build_extraction_plan(cfg)
        self.output_dir = ensure_dir(cfg.get("interaction_output_dir") or (Path(cfg["output_dir"]) / "interaction"))
        self.global_dir = ensure_dir(self.output_dir / "global_summary")
        self.loop_dir = ensure_dir(self.output_dir / "loop_dense")
        self.patch_dir = ensure_dir(self.output_dir / "patch_specific")
        self.raw_dir = ensure_dir(self.output_dir / "raw_qkv_debug")
        self.current_memory_row: dict[str, Any] | None = None
        self.counts = {
            "global_summary": 0,
            "loop_dense": 0,
            "patch_specific": 0,
            "raw_qkv_debug": 0,
        }
        self.files = {
            "global_summary": str(self.global_dir / "global_summary.jsonl"),
            "loop_dense": str(self.loop_dir / "loop_dense.jsonl"),
            "patch_specific": str(self.patch_dir / "patch_specific.jsonl"),
        }
        self._open_files: dict[str, Any] = {}
        for level, path in self.files.items():
            self._open_files[level] = Path(path).open("w", encoding="utf-8")
        write_json(self.output_dir / "feature_extraction_plan.json", self.plan)

    @property
    def selection(self) -> InteractionSelection:
        return build_interaction_selection(self.cfg)

    def set_current_memory_row(self, row: dict[str, Any] | None) -> None:
        self.current_memory_row = row

    def should_capture(self, frame_id: int, layer_id: int) -> bool:
        levels = self._levels_for(frame_id, layer_id)
        return bool(levels)

    def should_save_raw(self, frame_id: int, layer_id: int) -> bool:
        raw = self.plan["levels"]["raw_qkv_debug"]
        return int(frame_id) in set(raw["frames"]) and int(layer_id) in set(raw["layers"])

    def handle_record(self, record: QKVRecord) -> None:
        if record.capture_kind != "visible_memory_qkv":
            return
        if self.current_memory_row is None:
            return
        row = self.current_memory_row
        old_frames = row.get("old_memory_frames") or []
        if not old_frames:
            return
        levels = self._levels_for(record.frame_id, record.layer_id)
        if not levels:
            return
        summary_full = any(level in levels for level in ("loop_dense", "patch_specific"))
        summary = compute_interaction_summary(record, row, dense_by_source=summary_full)
        summary["levels"] = levels
        for level in levels:
            if level == "raw_qkv_debug":
                self._write_raw_metadata(record, row)
            elif level in self._open_files:
                payload = summary if level != "global_summary" else compact_summary(summary)
                self._write_jsonl(level, payload)
            if level in self.counts:
                self.counts[level] += 1

    def close(self) -> None:
        for f in self._open_files.values():
            f.close()
        index = {
            "status": "complete",
            "run_name": self.cfg.get("run_name"),
            "output_dir": str(self.output_dir),
            "files": self.files,
            "counts": self.counts,
            "raw_qkv_dir": str(self.raw_dir / "qkv"),
            "raw_qkv_metadata_dir": str(self.raw_dir / "token_metadata"),
            "normalization_note": {
                "full_context_mass": "Softmax denominator is every visible KV token.",
                "old_memory_conditional_distribution": "Attention is renormalized after slicing old trajectory-memory tokens.",
            },
        }
        write_json(self.output_dir / "interaction_index.json", index)

    def _write_jsonl(self, level: str, payload: dict[str, Any]) -> None:
        self._open_files[level].write(json.dumps(json_ready(payload), sort_keys=True) + "\n")
        self._open_files[level].flush()

    def _write_raw_metadata(self, record: QKVRecord, row: dict[str, Any]) -> None:
        meta_dir = ensure_dir(self.raw_dir / "token_metadata")
        metadata = {
            "status": "complete",
            "frame_id": int(record.frame_id),
            "layer_id": int(record.layer_id),
            "old_memory_token_count": len(row.get("old_memory_frames") or []) * 6,
            "old_memory_frames": row.get("old_memory_frames") or [],
            "old_memory_frame_range": row.get("old_memory_frame_range"),
            "patch_memory_frames": row.get("patch_memory_frames"),
            "token_metadata": token_metadata_for_old_memory(row),
            "visible_k_layout": [
                {
                    "name": "old_trajectory_memory",
                    "start": 0,
                    "end_exclusive": len(row.get("old_memory_frames") or []) * 6,
                    "token_metadata": "token_metadata contains per-token source frame/type for this span.",
                },
                {
                    "name": "retained_full_patch_and_current_context",
                    "start": len(row.get("old_memory_frames") or []) * 6,
                    "end_exclusive": None,
                    "token_metadata": "Not expanded here; use memory_state_summary patch_memory_frames plus LingBot token layout.",
                },
            ],
        }
        write_json(meta_dir / f"frame{record.frame_id:06d}_layer{record.layer_id:02d}_token_metadata.json", metadata)

    def _levels_for(self, frame_id: int, layer_id: int) -> list[str]:
        levels = self.plan["levels"]
        out = []
        if int(frame_id) in set(levels["global_summary"]["frames"]) and int(layer_id) in set(levels["global_summary"]["layers"]):
            out.append("global_summary")
        if int(frame_id) in set(levels["loop_dense"]["current_frames"]) and int(layer_id) in set(levels["loop_dense"]["layers"]):
            out.append("loop_dense")
        patch_frames = set(levels["patch_specific"]["selected_current_frames"]) | set(levels["patch_specific"]["auto_control_frames"])
        if int(frame_id) in patch_frames and int(layer_id) in set(levels["patch_specific"]["layers"]):
            out.append("patch_specific")
        if self.should_save_raw(frame_id, layer_id):
            out.append("raw_qkv_debug")
        return out


def compute_interaction_summary(record: QKVRecord, row: dict[str, Any], dense_by_source: bool) -> dict[str, Any]:
    import torch

    q = _as_torch(record.q).detach()
    k = _as_torch(record.k).detach()
    v = _as_torch(record.v).detach()
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected q/k/v [B,H,T,D], got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    q = q[0].to(device=k.device, dtype=torch.float32)
    k = k[0].to(dtype=torch.float32)
    v = v[0].to(dtype=torch.float32)
    heads, q_tokens, dim = q.shape
    old_frames = [int(v) for v in row.get("old_memory_frames") or []]
    old_frame_count = len(old_frames)
    old_token_count = old_frame_count * 6
    if old_token_count <= 0:
        raise ValueError("Cannot summarize interaction without old memory tokens.")
    k = k[:, : k.shape[1]]
    v_old_norm = torch.linalg.norm(v[:, :old_token_count], dim=-1)
    source_frames = old_frames

    query_names = list(QUERY_SETS.keys())
    token_names = list(TOKEN_TYPE_SLOTS.keys())
    frame_mass = {name: torch.zeros((heads, old_frame_count), device=q.device) for name in query_names}
    frame_contrib = {name: torch.zeros((heads, old_frame_count), device=q.device) for name in query_names}
    type_frame_mass = {
        qname: {tname: torch.zeros((heads, old_frame_count), device=q.device) for tname in token_names}
        for qname in query_names
    }
    type_frame_contrib = {
        qname: {tname: torch.zeros((heads, old_frame_count), device=q.device) for tname in token_names}
        for qname in query_names
    }
    old_mass_sum_by_query = {name: torch.zeros(heads, device=q.device) for name in query_names}
    old_mass_mean_by_query = {name: torch.zeros(heads, device=q.device) for name in query_names}
    old_contrib_sum_by_query = {name: torch.zeros(heads, device=q.device) for name in query_names}
    query_counts = {name: len(_query_indices(name, q_tokens, q.device)) for name in query_names}

    scale = 1.0 / math.sqrt(dim)
    chunk_size = int(record.__dict__.get("chunk_size", 64) or 64)
    for head_id in range(heads):
        k_h = k[head_id].T.contiguous()
        for start in range(0, q_tokens, chunk_size):
            end = min(start + chunk_size, q_tokens)
            attn = torch.softmax((q[head_id, start:end] @ k_h) * scale, dim=-1)
            old_attn = attn[:, :old_token_count]
            old_contrib = old_attn * v_old_norm[head_id, None, :]
            old_attn_by_frame_token = old_attn.reshape(end - start, old_frame_count, 6)
            old_contrib_by_frame_token = old_contrib.reshape(end - start, old_frame_count, 6)
            for query_name in query_names:
                idx = _query_indices(query_name, q_tokens, q.device)
                mask = (idx >= start) & (idx < end)
                if not bool(mask.any()):
                    continue
                local_idx = idx[mask] - start
                selected_attn = old_attn_by_frame_token[local_idx]
                selected_contrib = old_contrib_by_frame_token[local_idx]
                summed_attn = selected_attn.sum(dim=(0, 2))
                summed_contrib = selected_contrib.sum(dim=(0, 2))
                frame_mass[query_name][head_id] += summed_attn
                frame_contrib[query_name][head_id] += summed_contrib
                old_mass_sum_by_query[query_name][head_id] += summed_attn.sum()
                old_contrib_sum_by_query[query_name][head_id] += summed_contrib.sum()
                for token_name, slots in TOKEN_TYPE_SLOTS.items():
                    slot_tensor = torch.tensor(slots, device=q.device, dtype=torch.long)
                    type_frame_mass[query_name][token_name][head_id] += selected_attn[:, :, slot_tensor].sum(dim=(0, 2))
                    type_frame_contrib[query_name][token_name][head_id] += selected_contrib[:, :, slot_tensor].sum(dim=(0, 2))
            del attn, old_attn, old_contrib

    aggregation: dict[str, Any] = {}
    age_bins = int(32)
    current_frame = int(record.frame_id)
    for query_name in query_names:
        q_count = max(int(query_counts[query_name]), 1)
        old_mass_mean_by_query[query_name] = old_mass_sum_by_query[query_name] / q_count
        total_possible_mass = float(q_count)
        mass_np = frame_mass[query_name].detach().cpu().numpy()
        contrib_np = frame_contrib[query_name].detach().cpu().numpy()
        mass_prob = mass_np / np.maximum(mass_np.sum(axis=1, keepdims=True), 1e-12)
        entropy = -(mass_prob * np.log(np.maximum(mass_prob, 1e-12))).sum(axis=1)
        normalized_entropy = entropy / max(math.log(max(old_frame_count, 2)), 1e-12)
        mean_mass = mass_np.mean(axis=0)
        top_idx = np.argsort(-mean_mass)[:32]
        query_payload: dict[str, Any] = {
            "normalization": {
                "attention_mass": "full_context_softmax_denominator",
                "old_memory_conditional_probability": "renormalized_within_old_memory_attention_mass",
            },
            "old_memory_total_attention_mass_by_head": old_mass_sum_by_query[query_name].detach().cpu().tolist(),
            "old_memory_mean_attention_mass_by_head_query": old_mass_mean_by_query[query_name].detach().cpu().tolist(),
            "non_old_context_total_attention_mass_by_head": (
                torch.full_like(old_mass_sum_by_query[query_name], total_possible_mass) - old_mass_sum_by_query[query_name]
            ).detach().cpu().tolist(),
            "old_memory_total_contribution_proxy_by_head": old_contrib_sum_by_query[query_name].detach().cpu().tolist(),
            "source_entropy_by_head": entropy.tolist(),
            "normalized_source_entropy_by_head": normalized_entropy.tolist(),
            "top_frames_by_mean_attention": [
                {
                    "source_frame_id": int(source_frames[i]),
                    "mean_attention_mass": float(mean_mass[i]),
                    "mean_old_conditional_probability": float(mass_prob[:, i].mean()),
                }
                for i in top_idx
            ],
            "age_histogram": build_age_histogram(source_frames, mass_np, current_frame, age_bins),
        }
        for token_name in token_names:
            tm = type_frame_mass[query_name][token_name]
            tc = type_frame_contrib[query_name][token_name]
            query_payload[token_name] = {
                "total_attention_mass_by_head": tm.sum(dim=1).detach().cpu().tolist(),
                "mean_attention_mass_by_head": tm.mean(dim=1).detach().cpu().tolist(),
                "total_contribution_proxy_by_head": tc.sum(dim=1).detach().cpu().tolist(),
                "mean_contribution_proxy_by_head": tc.mean(dim=1).detach().cpu().tolist(),
            }
            if dense_by_source:
                query_payload[token_name]["by_source_frame"] = {
                    "source_frames": source_frames,
                    "attention_mass_by_head_frame": tm.detach().cpu().tolist(),
                    "contribution_proxy_by_head_frame": tc.detach().cpu().tolist(),
                }
        if dense_by_source:
            query_payload["by_source_frame"] = {
                "source_frames": source_frames,
                "attention_mass_by_head_frame": mass_np.tolist(),
                "contribution_proxy_by_head_frame": contrib_np.tolist(),
                "old_memory_conditional_probability_by_head_frame": mass_prob.tolist(),
            }
        aggregation[query_name] = query_payload

    return {
        "status": "complete",
        "frame_id": int(record.frame_id),
        "layer_id": int(record.layer_id),
        "capture_kind": record.capture_kind,
        "old_memory_frame_range": row.get("old_memory_frame_range"),
        "old_memory_frames": source_frames if dense_by_source else None,
        "old_memory_token_count": old_token_count,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "index_base": 0,
        "dense_by_source": dense_by_source,
        "aggregation": aggregation,
    }


def build_age_histogram(source_frames: list[int], mass_by_head_frame: np.ndarray, current_frame: int, bins: int) -> dict[str, Any]:
    ages = np.asarray([current_frame - int(frame) for frame in source_frames], dtype=np.float64)
    if ages.size == 0:
        return {"bin_edges": [], "attention_mass_by_head_bin": []}
    lo = float(max(0.0, ages.min()))
    hi = float(max(lo + 1.0, ages.max()))
    edges = np.linspace(lo, hi, bins + 1)
    out = np.zeros((mass_by_head_frame.shape[0], bins), dtype=np.float64)
    assignments = np.clip(np.digitize(ages, edges, right=False) - 1, 0, bins - 1)
    for idx, bin_id in enumerate(assignments):
        out[:, bin_id] += mass_by_head_frame[:, idx]
    return {"bin_edges": edges.tolist(), "attention_mass_by_head_bin": out.tolist()}


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    import copy

    compact = copy.deepcopy(summary)
    compact["old_memory_frames"] = None
    compact["dense_by_source"] = False
    for query_payload in compact.get("aggregation", {}).values():
        query_payload.pop("by_source_frame", None)
        for token_name in TOKEN_TYPE_SLOTS:
            if token_name in query_payload:
                query_payload[token_name].pop("by_source_frame", None)
    return compact
