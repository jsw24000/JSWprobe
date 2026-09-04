from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from long_seq_memory.dataset import frame_ids, load_scene
from long_seq_memory.io_utils import collect_environment, ensure_dir, sha256_file, write_json, write_table
from long_seq_memory.memory_index import (
    memory_state_summary,
    preprocessed_patch_grid,
    write_memory_state_jsonl,
)
from long_seq_memory.online_interactions import OnlineInteractionWriter, build_interaction_selection
from long_seq_memory.schedule import build_keyframe_schedule, schedule_config_from_dict, written_frame_ids


def _add_lingbot_to_path(lingbot_root: str | Path) -> None:
    import sys

    root = str(Path(lingbot_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _select_dtype(torch, device, requested: str):
    if requested and requested != "auto":
        return getattr(torch, requested)
    if device.type == "cuda":
        major = torch.cuda.get_device_capability(device)[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _model_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(cfg.get("model_defaults", {}))
    defaults.update(cfg.get("inference", {}))
    return defaults


def _selected_frame_ids(cfg: dict[str, Any]) -> list[int]:
    dataset = cfg.get("dataset", {})
    start = int(dataset.get("start_frame", cfg.get("start_frame", 0)))
    end = int(dataset.get("end_frame", cfg.get("end_frame", cfg.get("sequence", {}).get("expected_frames", 0))))
    stride = int(dataset.get("input_stride", cfg.get("stride", 1)))
    return frame_ids(start, end, stride)


def _reconstruction_output_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("reconstruction_output_dir", cfg["output_dir"])).expanduser()


def load_lingbot_model(cfg: dict[str, Any], device):
    import torch

    _add_lingbot_to_path(cfg["lingbot_root"])
    from lingbot_map.models.gct_stream import GCTStream

    defaults = _model_defaults(cfg)
    model = GCTStream(
        img_size=int(defaults.get("image_size", 518)),
        patch_size=int(defaults.get("patch_size", 14)),
        enable_3d_rope=bool(defaults.get("enable_3d_rope", True)),
        max_frame_num=int(defaults.get("max_frame_num", 1024)),
        kv_cache_sliding_window=int(defaults.get("kv_cache_sliding_window", 64)),
        kv_cache_scale_frames=int(defaults.get("kv_cache_scale_frames", 8)),
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=bool(defaults.get("use_sdpa", False)),
        camera_num_iterations=int(defaults.get("camera_num_iterations", 4)),
    )

    checkpoint = cfg.get("checkpoint")
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def load_preprocessed_images(cfg: dict[str, Any], selected_frame_ids: list[int]):
    _add_lingbot_to_path(cfg["lingbot_root"])
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    scene = load_scene(cfg["dataset_root"])
    image_paths = [str(scene.image_path(i)) for i in selected_frame_ids]
    defaults = _model_defaults(cfg)
    images = load_and_preprocess_images(
        image_paths,
        mode="crop",
        image_size=int(defaults.get("image_size", 518)),
        patch_size=int(defaults.get("patch_size", 14)),
    )
    return images, image_paths


def decode_pose_enc_to_c2w(predictions: dict[str, Any], image_hw: tuple[int, int]):
    import torch

    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    c2w = closed_form_inverse_se3_general(extrinsic_4x4)
    return c2w, intrinsic


def run_streaming_reconstruction(cfg: dict[str, Any]) -> dict[str, Any]:
    import random

    import torch

    from long_seq_memory.interaction_hooks import (
        QKVHookStore,
        RepresentationHookStore,
        causal_intervention_from_config,
        install_flashinfer_attn_pre_hooks,
        install_flashinfer_memory_hooks,
        install_block_output_hooks,
        install_sdpa_causal_intervention_patch,
        install_sdpa_memory_hooks,
        install_sdpa_skip_append_patch,
        representation_layers_from_config,
    )

    out_dir = ensure_dir(_reconstruction_output_dir(cfg))
    defaults = _model_defaults(cfg)
    run_frames = _selected_frame_ids(cfg)
    scene = load_scene(cfg["dataset_root"])
    scfg = schedule_config_from_dict(cfg)
    schedule_rows = build_keyframe_schedule(scfg)
    schedule_by_frame = {int(row["source_frame_id"]): row for row in schedule_rows}

    seed = int(cfg.get("seed", defaults.get("seed", 0)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = cfg.get("device", defaults.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    dtype = _select_dtype(torch, device, cfg.get("dtype", defaults.get("dtype", "auto")))

    images, image_paths = load_preprocessed_images(cfg, run_frames)
    model = load_lingbot_model(cfg, device)

    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    extraction = cfg.get("extraction", {})
    interaction_cfg = cfg.get("interaction", {})
    interaction_writer = OnlineInteractionWriter(cfg) if interaction_cfg.get("online_aggregation", False) else None
    selected_frames = set(cfg.get("selected_frames", []))
    selected_layers = set(cfg.get("selected_layers", []))
    raw_frames = set(extraction.get("raw_qkv_frames", selected_frames))
    raw_layers = set(extraction.get("representative_layers", selected_layers))
    if interaction_writer is not None:
        selection = build_interaction_selection(cfg)
        selected_frames.update(selection.capture_frames)
        selected_layers.update(selection.capture_layers)
        raw_frames = selection.raw_frames
        raw_layers = selection.raw_layers
    hook_store = QKVHookStore(
        selected_frames=selected_frames,
        selected_layers=selected_layers,
        raw_selected_frames=raw_frames if raw_frames else None,
        raw_selected_layers=raw_layers if raw_layers else None,
        storage_dtype=str(extraction.get("raw_qkv_storage_dtype", "float16")),
        aggregate_callback=interaction_writer.handle_record if interaction_writer is not None else None,
    )
    removers = []
    sdpa_skip_patch_installed = False
    causal_intervention = causal_intervention_from_config(cfg)
    causal_intervention_installed = False
    enforce_skip_patch = bool(defaults.get("enforce_keyframe_skip_append", cfg.get("enforce_keyframe_skip_append", False)))
    if bool(defaults.get("use_sdpa", False)) and causal_intervention is not None:
        _add_lingbot_to_path(cfg["lingbot_root"])
        from lingbot_map.layers import attention

        removers.append(install_sdpa_causal_intervention_patch(attention, causal_intervention))
        sdpa_skip_patch_installed = True
        causal_intervention_installed = True
    elif bool(defaults.get("use_sdpa", False)) and (enforce_skip_patch or int(defaults.get("keyframe_interval", 1)) > 1):
        _add_lingbot_to_path(cfg["lingbot_root"])
        from lingbot_map.layers import attention

        removers.append(install_sdpa_skip_append_patch(attention))
        sdpa_skip_patch_installed = True
    capture_interactions = cfg.get("save_qkv_for_selected_frames", False) or interaction_writer is not None
    if capture_interactions:
        if bool(defaults.get("use_sdpa", False)):
            _add_lingbot_to_path(cfg["lingbot_root"])
            from lingbot_map.layers import attention

            removers.append(install_sdpa_memory_hooks(attention, hook_store))
        else:
            removers = install_flashinfer_attn_pre_hooks(model, hook_store)
            _add_lingbot_to_path(cfg["lingbot_root"])
            from lingbot_map.layers import flashinfer_cache

            removers.append(install_flashinfer_memory_hooks(flashinfer_cache, hook_store))

    representation_cfg = cfg.get("representation_capture", {}) or {}
    representation_store = None
    if representation_cfg.get("enabled", False):
        representation_frames = representation_cfg.get("frames")
        if representation_frames is None:
            loop_event = cfg.get("loop_event", {})
            representation_frames = list(extraction.get("selected_current_frames", []))
            representation_frames += list(extraction.get("raw_qkv_frames", []))
            representation_frames += list(loop_event.get("selected_current_frames", []))
            if "history_frame" in loop_event:
                representation_frames.append(loop_event["history_frame"])
            if "current_frame" in loop_event:
                representation_frames.append(loop_event["current_frame"])
        representation_frames_set = {int(frame_id) for frame_id in representation_frames}
        num_layers = len(getattr(getattr(model, "aggregator", None), "global_blocks", []))
        representation_layers = representation_layers_from_config(representation_cfg.get("layers", "all"), num_layers)
        representation_store = RepresentationHookStore(
            output_dir=Path(representation_cfg.get("output_dir", out_dir / "representations")).expanduser(),
            selected_frames=representation_frames_set,
            selected_layers=representation_layers,
            storage_dtype=str(representation_cfg.get("storage_dtype", extraction.get("raw_qkv_storage_dtype", "float16"))),
            num_special_tokens=int(cfg.get("memory_policy", {}).get("patch_start_idx", 6)),
        )
        removers.extend(install_block_output_hooks(model, representation_store))

    images = images.to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    scale_frames = min(int(defaults.get("num_scale_frames", 8)), images.shape[0])
    keyframe_interval = int(defaults.get("keyframe_interval", 1))
    sliding_window = int(defaults.get("kv_cache_sliding_window", 64))
    num_register_tokens = int(cfg.get("memory_policy", {}).get("num_register_tokens", 4))

    per_frame_time = []
    memory_rows = []
    causal_intervention_rows = []
    pose_chunks = []
    dense_chunks: dict[str, list[Any]] = {}
    dense_frame_ids: dict[str, list[int]] = {}
    written_frames: list[int] = []

    outputs_cfg = cfg.get("outputs", {})
    save_depth_frames = {int(v) for v in outputs_cfg.get("save_depth_frames", [])}
    save_world_points_frames = {int(v) for v in outputs_cfg.get("save_world_points_frames", [])}

    def should_save_dense_key(key: str, frame_id: int) -> bool:
        if cfg.get("save_dense_outputs", False):
            return True
        if key.startswith("depth"):
            if frame_id in save_depth_frames:
                return True
            if outputs_cfg.get("save_depth_all_frames", False):
                return True
            stride = int(outputs_cfg.get("save_depth_stride", 0) or 0)
            return stride > 0 and frame_id % stride == 0
        if key.startswith("world_points"):
            if frame_id in save_world_points_frames:
                return True
            if outputs_cfg.get("save_world_points_all_frames", False):
                return True
            stride = int(outputs_cfg.get("save_world_points_stride", 0) or 0)
            return stride > 0 and frame_id % stride == 0
        return False

    def stash_dense_outputs(output: dict[str, Any], output_frame_ids: list[int]) -> None:
        for key, value in output.items():
            if key == "pose_enc" or not hasattr(value, "detach"):
                continue
            for idx, frame_id in enumerate(output_frame_ids):
                if should_save_dense_key(key, frame_id):
                    dense_chunks.setdefault(key, []).append(value[:, idx : idx + 1].detach().cpu())
                    dense_frame_ids.setdefault(key, []).append(frame_id)

    def collect_causal_intervention_stats(output_frame_ids: list[int]) -> None:
        if causal_intervention is None:
            return
        for layer_id, block in enumerate(getattr(getattr(model, "aggregator", None), "global_blocks", [])):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            stats = getattr(attn, "_long_seq_last_intervention_stats", None)
            if stats is None:
                continue
            if stats.get("active_frame") and (stats.get("active_layer") or stats.get("applied")):
                item = dict(stats)
                item["frame_ids"] = [int(frame_id) for frame_id in output_frame_ids]
                item["source_frame_id"] = int(output_frame_ids[-1]) if output_frame_ids else None
                causal_intervention_rows.append(item)
            delattr(attn, "_long_seq_last_intervention_stats")

    model.clean_kv_cache()
    try:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            t0 = time.perf_counter()
            hook_store.current_frame_id = run_frames[0] if run_frames else None
            if causal_intervention is not None:
                causal_intervention.current_frame_id = None
            if representation_store is not None:
                representation_store.set_current_frame_ids(run_frames[:scale_frames])
            if interaction_writer is not None:
                interaction_writer.set_current_memory_row(None)
            scale_output = model.forward(
                images[None, :scale_frames],
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
            collect_causal_intervention_stats(run_frames[:scale_frames])
            per_frame_time.extend([None] * scale_frames)
            pose_chunks.append(scale_output["pose_enc"].detach().cpu())
            stash_dense_outputs(scale_output, run_frames[:scale_frames])
            for write_index, frame_id in enumerate(run_frames[:scale_frames]):
                written_frames.append(frame_id)
                schedule_row = dict(schedule_by_frame.get(frame_id, {}))
                row = memory_state_summary(
                    current_frame=frame_id,
                    processed_frames=written_frames[: write_index + 1],
                    target_frame=int(cfg.get("loop_event", {}).get("history_frame", 3403)),
                    scale_frames=scale_frames,
                    sliding_window=sliding_window,
                    num_register_tokens=num_register_tokens,
                )
                row.update(
                    {
                        "input_index": schedule_row.get("input_index", write_index),
                        "source_frame_id": frame_id,
                        "is_scale_frame": True,
                        "is_keyframe": bool(schedule_row.get("is_keyframe", False)),
                        "expected_memory_write": True,
                        "keyframe_index": schedule_row.get("keyframe_index", write_index),
                        "skip_append_requested": False,
                        "long_memory_written_frames": written_frames[: write_index + 1],
                    }
                )
                memory_rows.append(row)
            scale_elapsed = time.perf_counter() - t0
            if scale_frames:
                per_frame_time[:scale_frames] = [scale_elapsed / scale_frames] * scale_frames

            for local_idx in range(scale_frames, images.shape[0]):
                frame_id = run_frames[local_idx]
                schedule_row = dict(schedule_by_frame.get(frame_id, {}))
                is_keyframe = bool(schedule_row.get("expected_memory_write", keyframe_interval <= 1))
                planned_written_frames = written_frames + ([frame_id] if is_keyframe else [])
                row = memory_state_summary(
                    current_frame=frame_id,
                    processed_frames=planned_written_frames,
                    target_frame=int(cfg.get("loop_event", {}).get("history_frame", 3403)),
                    scale_frames=scale_frames,
                    sliding_window=sliding_window,
                    num_register_tokens=num_register_tokens,
                )
                row.update(
                    {
                        "input_index": schedule_row.get("input_index", local_idx),
                        "source_frame_id": frame_id,
                        "is_scale_frame": bool(schedule_row.get("is_scale_frame", False)),
                        "is_keyframe": bool(schedule_row.get("is_keyframe", False)),
                        "expected_memory_write": bool(is_keyframe),
                        "keyframe_index": schedule_row.get("keyframe_index") if is_keyframe else None,
                        "skip_append_requested": not is_keyframe,
                        "long_memory_written_frames": planned_written_frames[:],
                    }
                )
                if not is_keyframe:
                    model._set_skip_append(True)
                hook_store.current_frame_id = frame_id
                if causal_intervention is not None:
                    causal_intervention.current_frame_id = frame_id
                if representation_store is not None:
                    representation_store.set_current_frame_ids([frame_id])
                if interaction_writer is not None:
                    interaction_writer.set_current_memory_row(row)
                start = time.perf_counter()
                try:
                    frame_output = model.forward(
                        images[None, local_idx : local_idx + 1],
                        num_frame_for_scale=scale_frames,
                        num_frame_per_block=1,
                        causal_inference=True,
                    )
                finally:
                    if not is_keyframe:
                        model._set_skip_append(False)
                per_frame_time.append(time.perf_counter() - start)
                collect_causal_intervention_stats([frame_id])
                pose_chunks.append(frame_output["pose_enc"].detach().cpu())
                stash_dense_outputs(frame_output, [frame_id])
                if is_keyframe:
                    written_frames.append(frame_id)
                memory_rows.append(row)
    finally:
        for remove in reversed(removers):
            remove()

    pose_enc = torch.cat(pose_chunks, dim=1)
    predictions = {"pose_enc": pose_enc}
    pred_c2w, pred_intrinsic = decode_pose_enc_to_c2w(predictions, images.shape[-2:])

    np.save(out_dir / "frame_ids.npy", np.array(run_frames, dtype=np.int64))
    np.save(out_dir / "gt_poses_c2w.npy", scene.poses_c2w[run_frames])
    np.save(out_dir / "pred_poses_c2w_demo_convention.npy", pred_c2w.detach().cpu().numpy())
    np.save(out_dir / "pred_intrinsics.npy", pred_intrinsic.detach().cpu().numpy())
    np.save(out_dir / "per_frame_time_s.npy", np.array(per_frame_time, dtype=np.float64))
    write_memory_state_jsonl(out_dir / "memory_state_summary.jsonl", memory_rows)
    run_frame_set = set(run_frames)
    write_table(
        out_dir / "keyframe_schedule.parquet",
        [row for row in schedule_rows if int(row["source_frame_id"]) in run_frame_set],
        metadata={"run_name": cfg.get("run_name"), "scope": "reconstruction_frames"},
    )
    if hook_store.records:
        qkv_dir = (
            Path(interaction_writer.raw_dir) / "qkv"
            if interaction_writer is not None
            else out_dir / "qkv"
        )
        hook_store.save_npz(qkv_dir)
    if interaction_writer is not None:
        interaction_writer.close()

    causal_intervention_stats_path = None
    if causal_intervention_rows:
        causal_intervention_stats_path = out_dir / "causal_intervention_stats.jsonl"
        write_memory_state_jsonl(causal_intervention_stats_path, causal_intervention_rows)

    representation_index_path = None
    if representation_store is not None:
        representation_index_path = out_dir / "representation_capture_index.json"
        write_json(
            representation_index_path,
            {
                "status": "complete",
                "output_dir": str(representation_store.output_dir),
                "selected_frames": sorted(representation_store.selected_frames),
                "selected_layers": (
                    "all"
                    if representation_store.selected_layers is None
                    else sorted(representation_store.selected_layers)
                ),
                "storage_dtype": representation_store.storage_dtype,
                "num_records": len(representation_store.saved),
                "records": representation_store.saved,
            },
        )

    if dense_chunks:
        dense_dir = ensure_dir(out_dir / "dense")
        for key, chunks in dense_chunks.items():
            torch.save(torch.cat(chunks, dim=1), dense_dir / f"{key}.pt")
            np.save(dense_dir / f"{key}_frame_ids.npy", np.array(dense_frame_ids.get(key, []), dtype=np.int64))

    checkpoint = Path(cfg.get("checkpoint", ""))
    first_size = scene.image_paths and scene.image_path(run_frames[0])
    patch_grid = preprocessed_patch_grid(
        original_width=scene.intrinsics.width,
        original_height=scene.intrinsics.height,
        image_size=int(defaults.get("image_size", 518)),
        patch_size=int(defaults.get("patch_size", 14)),
    )
    manifest = {
        "run_name": cfg.get("run_name"),
        "config": cfg.get("_config_path"),
        "frame_ids": run_frames,
        "num_frames": len(run_frames),
        "first_image_path": str(first_size) if first_size else None,
        "preprocessed_shape_chw": list(images.shape[-3:]),
        "patch_grid_wh": patch_grid,
        "tokens_per_frame_estimate": patch_grid[0] * patch_grid[1] + 6,
        "device": str(device),
        "dtype": str(dtype),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
        "environment": collect_environment(cfg.get("lingbot_root")),
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None,
        "state_reset_detected": False,
        "keyframe_interval": keyframe_interval,
        "total_memory_writes": len(written_frame_ids(schedule_rows, run_frames[-1] if run_frames else None)),
        "interaction_online_aggregation_enabled": interaction_writer is not None,
        "interaction_output_dir": cfg.get("interaction_output_dir") if interaction_writer is not None else None,
        "sdpa_skip_append_patch_installed": sdpa_skip_patch_installed,
        "aggregator_keyframe_skip_append_effective": bool(sdpa_skip_patch_installed and keyframe_interval > 1),
        "causal_intervention": cfg.get("causal_intervention", {"enabled": False}),
        "causal_intervention_patch_installed": bool(causal_intervention_installed),
        "causal_intervention_stats_path": str(causal_intervention_stats_path) if causal_intervention_stats_path else None,
        "representation_capture": cfg.get("representation_capture", {"enabled": False}),
        "representation_capture_index": str(representation_index_path) if representation_index_path else None,
        "resume_policy": cfg.get("resume", {"enabled": False, "reason": "KV-state serialization is not implemented."}),
        "notes": [
            "Predicted poses use the official demo postprocess convention.",
            "Non-keyframes still produce predictions; only scale/keyframes are expected to persist in long-term KV.",
        ],
    }
    write_json(out_dir / "run_manifest.json", manifest)
    model.clean_kv_cache()
    return manifest


def config_to_namespace(cfg: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**cfg)
