from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch


SPECIAL_TOKEN_NAMES = ["camera", "register_0", "register_1", "register_2", "register_3", "scale_or_anchor"]


def clone_nested(value: Any, *, detach: bool = True) -> Any:
    if torch.is_tensor(value):
        out = value.detach().clone() if detach else value.clone()
        return out
    if isinstance(value, dict):
        return {key: clone_nested(val, detach=detach) for key, val in value.items()}
    if isinstance(value, list):
        return [clone_nested(val, detach=detach) for val in value]
    if isinstance(value, tuple):
        return tuple(clone_nested(val, detach=detach) for val in value)
    return deepcopy(value)


@dataclass
class ModelStreamState:
    aggregator_kv_cache: dict[str, Any]
    aggregator_total_frames_processed: int
    aggregator_cached_pos3d: Any
    camera_kv_cache: Any
    camera_frame_idx: int | None
    camera_pos_cache: Any


def clone_stream_state(state: ModelStreamState) -> ModelStreamState:
    return ModelStreamState(
        aggregator_kv_cache=clone_nested(state.aggregator_kv_cache),
        aggregator_total_frames_processed=int(state.aggregator_total_frames_processed),
        aggregator_cached_pos3d=clone_nested(state.aggregator_cached_pos3d),
        camera_kv_cache=clone_nested(state.camera_kv_cache),
        camera_frame_idx=None if state.camera_frame_idx is None else int(state.camera_frame_idx),
        camera_pos_cache=clone_nested(state.camera_pos_cache),
    )


def clone_model_stream_state(model: Any) -> ModelStreamState:
    aggregator = model.aggregator
    camera_head = getattr(model, "camera_head", None)
    return ModelStreamState(
        aggregator_kv_cache=clone_nested(getattr(aggregator, "kv_cache", {})),
        aggregator_total_frames_processed=int(getattr(aggregator, "total_frames_processed", 0)),
        aggregator_cached_pos3d=clone_nested(getattr(aggregator, "_cached_pos3d", None)),
        camera_kv_cache=clone_nested(getattr(camera_head, "kv_cache", None)) if camera_head is not None else None,
        camera_frame_idx=int(getattr(camera_head, "frame_idx", 0)) if camera_head is not None else None,
        camera_pos_cache=clone_nested(getattr(camera_head, "pos_cache", None)) if camera_head is not None else None,
    )


def restore_model_stream_state(model: Any, state: ModelStreamState) -> None:
    aggregator = model.aggregator
    aggregator.kv_cache = clone_nested(state.aggregator_kv_cache)
    aggregator.total_frames_processed = int(state.aggregator_total_frames_processed)
    aggregator._cached_pos3d = clone_nested(state.aggregator_cached_pos3d)

    camera_head = getattr(model, "camera_head", None)
    if camera_head is not None:
        camera_head.kv_cache = clone_nested(state.camera_kv_cache)
        if state.camera_frame_idx is not None:
            camera_head.frame_idx = int(state.camera_frame_idx)
        camera_head.pos_cache = clone_nested(state.camera_pos_cache)


def global_block_indices(state: ModelStreamState) -> list[int]:
    indices = []
    for key in state.aggregator_kv_cache:
        if key.startswith("k_") and not key.endswith("_special"):
            try:
                indices.append(int(key.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(set(indices))


def special_frame_ids_before_target(target_frame: int, scale_frames: int, sliding_window: int) -> list[int]:
    end_exclusive = int(target_frame) - int(sliding_window)
    if end_exclusive <= int(scale_frames):
        return []
    return list(range(int(scale_frames), end_exclusive))


def local_cache_frame_ids_before_target(target_frame: int, scale_frames: int, sliding_window: int) -> list[int]:
    target_frame = int(target_frame)
    scale_frames = int(scale_frames)
    sliding_window = int(sliding_window)
    scale = list(range(scale_frames))
    live_start = max(scale_frames, target_frame - sliding_window)
    live = list(range(live_start, target_frame))
    return scale + live


def replace_all_trajectory_special(dst: ModelStreamState, src: ModelStreamState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx in global_block_indices(dst):
        for prefix in ["k", "v"]:
            key = f"{prefix}_{idx}_special"
            dst_tensor = dst.aggregator_kv_cache.get(key)
            src_tensor = src.aggregator_kv_cache.get(key)
            if dst_tensor is None or src_tensor is None:
                events.append(
                    {
                        "global_block": idx,
                        "cache_key": key,
                        "replaced": False,
                        "reason": "missing_special_cache",
                        "dst_shape": list(dst_tensor.shape) if torch.is_tensor(dst_tensor) else None,
                        "src_shape": list(src_tensor.shape) if torch.is_tensor(src_tensor) else None,
                    }
                )
                continue
            if tuple(dst_tensor.shape) != tuple(src_tensor.shape):
                raise ValueError(f"Shape mismatch for {key}: dst={tuple(dst_tensor.shape)} src={tuple(src_tensor.shape)}")
            dst.aggregator_kv_cache[key] = src_tensor.detach().clone()
            events.append(
                {
                    "global_block": idx,
                    "cache_key": key,
                    "replaced": True,
                    "record_scope": "all_trajectory_records",
                    "slot_axis": 2,
                    "token_axis": 3,
                    "token_slots": list(range(int(src_tensor.shape[3]))),
                    "shape": list(src_tensor.shape),
                }
            )
    return events


def replace_selected_trajectory_records(
    dst: ModelStreamState,
    src: ModelStreamState,
    *,
    selected_source_frames: list[int],
    special_frame_ids: list[int],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    slot_by_frame = {int(frame_id): slot for slot, frame_id in enumerate(special_frame_ids)}
    selected_slots = [slot_by_frame[int(frame_id)] for frame_id in selected_source_frames if int(frame_id) in slot_by_frame]
    missing = [int(frame_id) for frame_id in selected_source_frames if int(frame_id) not in slot_by_frame]
    replaced_frames = [int(frame_id) for frame_id in selected_source_frames if int(frame_id) in slot_by_frame]
    events: list[dict[str, Any]] = []

    for idx in global_block_indices(dst):
        for prefix in ["k", "v"]:
            key = f"{prefix}_{idx}_special"
            dst_tensor = dst.aggregator_kv_cache.get(key)
            src_tensor = src.aggregator_kv_cache.get(key)
            if dst_tensor is None or src_tensor is None:
                events.append(
                    {
                        "global_block": idx,
                        "cache_key": key,
                        "replaced": False,
                        "reason": "missing_special_cache",
                        "requested_source_frames": selected_source_frames,
                        "missing_source_frames": missing,
                    }
                )
                continue
            if tuple(dst_tensor.shape) != tuple(src_tensor.shape):
                raise ValueError(f"Shape mismatch for {key}: dst={tuple(dst_tensor.shape)} src={tuple(src_tensor.shape)}")
            out = dst_tensor.detach().clone()
            if selected_slots:
                out[:, :, selected_slots, :, :] = src_tensor[:, :, selected_slots, :, :]
            dst.aggregator_kv_cache[key] = out
            events.append(
                {
                    "global_block": idx,
                    "cache_key": key,
                    "replaced": bool(selected_slots),
                    "record_scope": "selected_trajectory_records",
                    "requested_source_frames": [int(x) for x in selected_source_frames],
                    "replaced_source_frames": replaced_frames,
                    "missing_source_frames": missing,
                    "memory_slot_indices": [int(x) for x in selected_slots],
                    "token_slots": list(range(int(src_tensor.shape[3]))),
                    "token_names": SPECIAL_TOKEN_NAMES[: int(src_tensor.shape[3])],
                    "shape": list(src_tensor.shape),
                }
            )
    return events, replaced_frames, missing


def state_layout_summary(
    state: ModelStreamState,
    *,
    source_sequence: str,
    target_frame: int,
    scale_frames: int,
    sliding_window: int,
) -> dict[str, Any]:
    special_frames = special_frame_ids_before_target(target_frame, scale_frames, sliding_window)
    local_frames = local_cache_frame_ids_before_target(target_frame, scale_frames, sliding_window)
    blocks: list[dict[str, Any]] = []
    for idx in global_block_indices(state):
        k_key = f"k_{idx}"
        k_special_key = f"k_{idx}_special"
        k = state.aggregator_kv_cache.get(k_key)
        ks = state.aggregator_kv_cache.get(k_special_key)
        blocks.append(
            {
                "layer_or_global_block_index": idx,
                "source_sequence": source_sequence,
                "local_anchor_cache": {
                    "k_key": k_key,
                    "v_key": f"v_{idx}",
                    "shape": list(k.shape) if torch.is_tensor(k) else None,
                    "frame_axis": 2,
                    "token_axis": 3,
                    "source_frame_ids": local_frames,
                    "classification": ["anchor_context"] * int(scale_frames)
                    + ["active_local_window"] * max(0, len(local_frames) - int(scale_frames)),
                },
                "trajectory_special_cache": {
                    "k_key": k_special_key,
                    "v_key": f"v_{idx}_special",
                    "shape": list(ks.shape) if torch.is_tensor(ks) else None,
                    "memory_slot_axis": 2,
                    "token_axis": 3,
                    "source_frame_ids": special_frames,
                    "token_names": SPECIAL_TOKEN_NAMES[: int(ks.shape[3])] if torch.is_tensor(ks) else SPECIAL_TOKEN_NAMES,
                    "classification": "long_term_trajectory_memory",
                    "video_rope": "RoPE is already baked into cached K tensors; V tensors carry no separate RoPE field.",
                },
            }
        )
    return {
        "source_sequence": source_sequence,
        "target_frame": int(target_frame),
        "scale_frames": int(scale_frames),
        "sliding_window": int(sliding_window),
        "aggregator_total_frames_processed_before_target": int(state.aggregator_total_frames_processed),
        "special_source_frame_ids_before_target": special_frames,
        "local_anchor_source_frame_ids_before_target": local_frames,
        "global_blocks": blocks,
        "camera_head_cache_note": (
            "Camera head has its own pose-token KV cache. It is cloned/restored for replay identity, "
            "but it is not modified by trajectory-memory transplants."
        ),
    }


def tensor_stats(value: Any) -> dict[str, Any] | None:
    if not torch.is_tensor(value):
        return None
    data = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "mean": float(data.mean().item()) if data.numel() else 0.0,
        "std": float(data.std().item()) if data.numel() > 1 else 0.0,
    }
