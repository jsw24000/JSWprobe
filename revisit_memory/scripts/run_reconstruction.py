#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.hooks.feature_hooks import AggregateFeatureCapture, token_type_names  # noqa: E402
from revisit_memory.utils.camera import gt_cam2worlds, gt_cam2worlds_opencv, intrinsic_matrix  # noqa: E402
from revisit_memory.utils.depth_scale import estimate_depth_scale, resize_bool_stack, resize_float_stack  # noqa: E402
from revisit_memory.utils.geometry import depth_to_world_points, write_ply  # noqa: E402
from revisit_memory.utils.io import (  # noqa: E402
    camera_json_path,
    condition_dir,
    depth_path,
    find_counterfactual_mask,
    list_frame_ids,
    load_any_target_mask,
    load_config,
    load_depth_exr,
    output_root,
    project_path,
    read_json,
    rgb_path,
    target_bbox,
    target_object,
    write_json,
)


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "run_reconstruction.py must be run in the LingBot-Map Python environment with torch installed."
        ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_lingbot(cfg: dict[str, Any]) -> None:
    lingbot_root = project_path(cfg["paths"]["lingbot_root"])
    if str(lingbot_root) not in sys.path:
        sys.path.insert(0, str(lingbot_root))


def build_model(cfg: dict[str, Any], device):
    torch = require_torch()
    import_lingbot(cfg)
    from lingbot_map.models.gct_stream import GCTStream

    model_cfg = cfg["model"]
    ckpt_path = project_path(cfg["paths"]["model_path"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    checkpoint_has_point_head = any(key.startswith("point_head.") for key in state_dict)
    model = GCTStream(
        img_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        pretrained_path="",
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_local_point=False,
        enable_3d_rope=bool(model_cfg["enable_3d_rope"]),
        max_frame_num=int(model_cfg["max_frame_num"]),
        kv_cache_sliding_window=int(model_cfg["local_window_size"]),
        kv_cache_scale_frames=int(model_cfg["anchor_frame_count"]),
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=bool(model_cfg.get("use_sdpa", False)),
        camera_num_iterations=int(model_cfg["camera_num_iterations"]),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, {
        "checkpoint": str(ckpt_path),
        "checkpoint_has_point_head": checkpoint_has_point_head,
        "dinov2_preinit_loaded_before_checkpoint": False,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "output_heads_enabled": {
            "camera_head": model.camera_head is not None,
            "depth_head": model.depth_head is not None,
            "point_head": model.point_head is not None,
            "local_point_head": model.local_point_head is not None,
        },
        "point_head_note": "Point/local point heads are disabled in this experiment; reconstruction uses LingBot-Map camera and depth heads only.",
        "missing_key_prefix_counts": {
            "aggregator": sum(1 for key in missing if key.startswith("aggregator.")),
            "camera_head": sum(1 for key in missing if key.startswith("camera_head.")),
            "depth_head": sum(1 for key in missing if key.startswith("depth_head.")),
        },
        "unexpected_key_prefix_counts": {
            "point_head": sum(1 for key in unexpected if key.startswith("point_head.")),
            "local_point_head": sum(1 for key in unexpected if key.startswith("local_point_head.")),
        },
    }


def choose_dtype(cfg: dict[str, Any], device):
    torch = require_torch()
    precision = cfg["model"].get("precision", "auto")
    if device.type != "cuda":
        return torch.float32
    if precision == "float32":
        return torch.float32
    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_images_tensor(cfg: dict[str, Any], condition: str, frame_ids: list[int]):
    import_lingbot(cfg)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    paths = [str(rgb_path(cfg, condition, fid)) for fid in frame_ids]
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=int(cfg["model"]["image_size"]),
        patch_size=int(cfg["model"]["patch_size"]),
    )
    return images, paths


def preprocessed_intrinsic(original_k: np.ndarray, original_hw: tuple[int, int], preprocessed_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = original_hw
    dst_h, dst_w = preprocessed_hw
    k = original_k.astype(np.float32).copy()
    k[0, :] *= dst_w / src_w
    k[1, :] *= dst_h / src_h
    return k


def append_predictions(store: dict[str, list[np.ndarray]], output: dict[str, Any], source_frame_ids: list[int]) -> None:
    for key in ["pose_enc", "depth", "depth_conf", "images"]:
        if key not in output:
            continue
        arr = output[key].detach().cpu().float().numpy()[0]
        for idx, _ in enumerate(source_frame_ids):
            store.setdefault(key, []).append(arr[idx])
    if "pose_enc_list" in output:
        stacked = [x.detach().cpu().float().numpy()[0] for x in output["pose_enc_list"]]
        arr = np.stack(stacked, axis=0)
        for idx, _ in enumerate(source_frame_ids):
            store.setdefault("pose_enc_iterations", []).append(arr[:, idx])


def consume_features(
    capture: AggregateFeatureCapture,
    source_frame_ids: list[int],
    feature_store: dict[str, Any],
    memory_store: dict[str, Any],
    save_frame_ids: set[int],
) -> None:
    torch = require_torch()
    batch = capture.last
    if batch is None:
        return

    feature_seen = feature_store.setdefault("_seen", set())
    for local_idx, frame_id in enumerate(source_frame_ids):
        memory_store.setdefault("source_frame_ids", []).append(frame_id)
        for layer_id in batch.selected_layers:
            memory_store.setdefault("frame_special_tokens", {}).setdefault(str(layer_id), []).append(
                batch.frame_special_tokens[layer_id][0, local_idx].clone()
            )
            memory_store.setdefault("global_special_tokens", {}).setdefault(str(layer_id), []).append(
                batch.global_special_tokens[layer_id][0, local_idx].clone()
            )

        if frame_id not in save_frame_ids:
            continue
        if frame_id not in feature_seen:
            feature_store.setdefault("frame_ids", []).append(frame_id)
            feature_seen.add(frame_id)
        for layer_id in batch.selected_layers:
            feature_store.setdefault("frame_block", {}).setdefault(str(layer_id), []).append(
                batch.frame_image_tokens[layer_id][0, local_idx].clone()
            )
            feature_store.setdefault("global_block", {}).setdefault(str(layer_id), []).append(
                batch.global_image_tokens[layer_id][0, local_idx].clone()
            )
            feature_store.setdefault("frame_special_tokens", {}).setdefault(str(layer_id), []).append(
                batch.frame_special_tokens[layer_id][0, local_idx].clone()
            )
            feature_store.setdefault("global_special_tokens", {}).setdefault(str(layer_id), []).append(
                batch.global_special_tokens[layer_id][0, local_idx].clone()
            )

    feature_store["patch_start_idx"] = batch.patch_start_idx
    feature_store["token_grid_hw"] = batch.token_grid_hw
    feature_store["raw_shapes"] = {str(k): v for k, v in batch.raw_shapes.items()}
    feature_store["selected_layers"] = list(batch.selected_layers)
    memory_store["patch_start_idx"] = batch.patch_start_idx
    memory_store["selected_layers"] = list(batch.selected_layers)
    memory_store["token_grid_hw"] = batch.token_grid_hw
    _ = torch


def configured_feature_frame_ids(cfg: dict[str, Any], available_frame_ids: list[int]) -> set[int]:
    frame_cfg = cfg["frames"]
    stride = max(1, int(frame_cfg.get("default_feature_stride", 1)))
    save_frame_ids: set[int] = set()
    for start, end in frame_cfg["default_feature_ranges"]:
        save_frame_ids.update(range(int(start), int(end) + 1, stride))
    save_frame_ids.update(int(fid) for fid in frame_cfg.get("default_feature_extra_frames", []))
    available = set(int(fid) for fid in available_frame_ids)
    return save_frame_ids.intersection(available)


def cache_record(cfg: dict[str, Any], current_frame: int, phase: str, is_keyframe: bool, model) -> dict[str, Any]:
    local_window = int(cfg["model"]["local_window_size"])
    anchor_count = int(cfg["model"]["anchor_frame_count"])
    target_frames = set(range(cfg["frames"]["first_loop_visible"][0], cfg["frames"]["first_loop_visible"][1] + 1))
    scale_ids = list(range(0, min(anchor_count - 1, current_frame) + 1))
    history_start = max(anchor_count, current_frame - local_window)
    history_before_current = list(range(history_start, current_frame))
    live_start = max(anchor_count, current_frame - local_window + 1)
    live_after = list(range(live_start, current_frame + 1)) if current_frame >= anchor_count else []
    special_ids = list(range(0, current_frame + 1))
    manager_info: dict[str, Any] = {}
    manager = getattr(getattr(model, "aggregator", None), "kv_cache_manager", None)
    if manager is not None:
        manager_info = {
            "backend": "flashinfer",
            "num_frames_block0": int(manager.num_frames),
            "scale_patch_page_count_block0": len(manager.scale_patch_pages[0]),
            "live_window_patch_page_count_block0": len(manager.live_window_patch_pages[0]),
            "special_token_count_block0": int(manager.special_token_count[0]),
            "special_page_count_block0": len(manager.all_special_pages[0]),
        }
    elif getattr(getattr(model, "aggregator", None), "kv_cache", None) is not None:
        kv = model.aggregator.kv_cache
        k0 = kv.get("k_0") if isinstance(kv, dict) else None
        manager_info = {
            "backend": "sdpa",
            "k0_cached_shape": list(k0.shape) if k0 is not None else None,
            "sdpa_warning": "SDPA cache metadata is shape-based; FlashInfer is preferred for the strict experiment.",
        }
    return {
        "current_frame": current_frame,
        "phase": phase,
        "is_keyframe": is_keyframe,
        "anchor_scale_source_frame_ids": scale_ids,
        "local_window_history_source_frame_ids_before_current": history_before_current,
        "local_patch_cache_source_frame_ids_after_step_excluding_anchor": live_after,
        "trajectory_special_token_source_frame_ids_after_step": special_ids,
        "target_visible_frames_in_local_history_before_current": sorted(target_frames.intersection(history_before_current)),
        "target_visible_frames_in_live_patch_cache_after_step": sorted(target_frames.intersection(live_after)),
        "target_visible_frames_as_special_tokens_after_step": sorted(target_frames.intersection(special_ids)),
        "frame_12_39_left_local_window": current_frame >= 56 and not bool(target_frames.intersection(history_before_current)),
        "runtime_cache_info": manager_info,
    }


def save_feature_store(path: Path, feature_store: dict[str, Any]) -> None:
    torch = require_torch()
    payload: dict[str, Any] = {
        "frame_ids": feature_store.get("frame_ids", []),
        "selected_layers": feature_store.get("selected_layers", []),
        "patch_start_idx": feature_store.get("patch_start_idx"),
        "token_grid_hw": feature_store.get("token_grid_hw"),
        "raw_shapes": feature_store.get("raw_shapes", {}),
        "frame_block": {},
        "global_block": {},
        "frame_special_tokens": {},
        "global_special_tokens": {},
    }
    for group in ["frame_block", "global_block", "frame_special_tokens", "global_special_tokens"]:
        for layer, tensors in feature_store.get(group, {}).items():
            payload[group][layer] = torch.stack(tensors, dim=0) if tensors else torch.empty(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_memory_store(path: Path, memory_store: dict[str, Any]) -> None:
    torch = require_torch()
    payload: dict[str, Any] = {
        "source_frame_ids": memory_store.get("source_frame_ids", []),
        "selected_layers": memory_store.get("selected_layers", []),
        "patch_start_idx": memory_store.get("patch_start_idx"),
        "token_grid_hw": memory_store.get("token_grid_hw"),
        "token_type_names": token_type_names(),
        "frame_special_tokens": {},
        "global_special_tokens": {},
    }
    for group in ["frame_special_tokens", "global_special_tokens"]:
        for layer, tensors in memory_store.get(group, {}).items():
            payload[group][layer] = torch.stack(tensors, dim=0) if tensors else torch.empty(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def postprocess_pose_arrays(cfg: dict[str, Any], pred: dict[str, np.ndarray], image_hw: tuple[int, int]) -> None:
    torch = require_torch()
    import_lingbot(cfg)
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    pose = torch.from_numpy(pred["pose_enc"]).unsqueeze(0).float()
    extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(pose, image_hw)
    extrinsic_4x4 = torch.zeros((*extrinsic_w2c.shape[:-2], 4, 4), dtype=extrinsic_w2c.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic_w2c
    extrinsic_4x4[..., 3, 3] = 1.0
    pred_c2w = closed_form_inverse_se3_general(extrinsic_4x4)[0].cpu().numpy()
    pred["pred_cam2world"] = pred_c2w
    pred["pred_intrinsic"] = intrinsic[0].cpu().numpy()


def reconstruction_ply_segments(
    cfg: dict[str, Any],
    frame_ids: list[int],
    selected_segments: list[str] | None = None,
) -> dict[str, list[int]]:
    first_start, first_end = [int(x) for x in cfg["frames"]["first_loop_visible"]]
    second_start, second_end = [int(x) for x in cfg["frames"]["second_loop_eval"]]
    revisit = int(cfg["frames"]["expected_first_revisit_frame"])
    selected = selected_segments or ["second_loop_eval"]
    available: dict[str, tuple[str, list[int]]] = {
        "all_processed": ("all_processed_overview", list(frame_ids)),
        "first_loop": (
            f"first_loop_{first_start:04d}_{first_end:04d}_overview",
            [fid for fid in frame_ids if first_start <= fid <= first_end],
        ),
        "second_loop_eval": (
            f"second_loop_{second_start:04d}_{second_end:04d}_overview",
            [fid for fid in frame_ids if second_start <= fid <= second_end],
        ),
        "revisit_frame": (f"revisit_frame_{revisit:04d}", [fid for fid in frame_ids if fid == revisit]),
    }
    segments: dict[str, list[int]] = {}
    for key in selected:
        if key not in available:
            raise ValueError(f"Unknown reconstruction pointcloud segment {key!r}. Expected one of {sorted(available)}.")
        name, ids = available[key]
        if ids:
            segments[name] = ids
    return segments


def sample_colored_points(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray | None,
    max_points: int,
    seed: int,
    confidence_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    pts = points.reshape(-1, 3)
    cols = colors.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    if confidence is not None:
        conf = confidence.reshape(-1)
        valid &= np.isfinite(conf)
        if confidence_threshold is not None:
            valid &= conf >= confidence_threshold
    idx = np.flatnonzero(valid)
    if idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    return pts[idx], cols[idx]


def collect_segment_pointcloud(
    depth: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray | None,
    frame_ids: list[int],
    selected_frame_ids: list[int],
    intrinsics: np.ndarray,
    cam2worlds: np.ndarray,
    max_points: int,
    seed: int,
    confidence_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    index_by_frame = {int(fid): idx for idx, fid in enumerate(frame_ids)}
    per_frame_limit = max(1, int(np.ceil(max_points / max(len(selected_frame_ids), 1))))
    pts_parts = []
    color_parts = []
    for offset, frame_id in enumerate(selected_frame_ids):
        idx = index_by_frame[int(frame_id)]
        intrinsic = intrinsics[idx] if intrinsics.ndim == 3 else intrinsics
        pts = depth_to_world_points(depth[idx].astype(np.float32), intrinsic.astype(np.float32), cam2worlds[idx].astype(np.float32))
        sampled_pts, sampled_colors = sample_colored_points(
            pts,
            colors[idx],
            confidence[idx] if confidence is not None else None,
            max_points=per_frame_limit,
            seed=seed + int(frame_id) + offset,
            confidence_threshold=confidence_threshold,
        )
        pts_parts.append(sampled_pts)
        color_parts.append(sampled_colors)

    if not pts_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    pts_all = np.concatenate(pts_parts, axis=0)
    colors_all = np.concatenate(color_parts, axis=0)
    if pts_all.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(pts_all.shape[0]), size=max_points, replace=False)
        pts_all = pts_all[idx]
        colors_all = colors_all[idx]
    return pts_all, colors_all


def export_reconstruction_pointclouds(
    cfg: dict[str, Any],
    recon_dir: Path,
    pred: dict[str, np.ndarray],
    inputs: dict[str, Any],
    frame_ids: list[int],
    images_cpu,
) -> list[dict[str, Any]]:
    export_cfg = cfg.get("exports", {}).get("reconstruction_pointclouds", {})
    if not export_cfg.get("enabled", True):
        return []
    pred_depth = pred["depth"][..., 0] if pred["depth"].ndim == 4 else pred["depth"].squeeze(-1)
    confidence = pred["depth_conf"].astype(np.float32) if "depth_conf" in pred else None
    colors = np.clip(images_cpu.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0, 0, 255).astype(np.uint8)
    max_points = int(export_cfg.get("max_points_per_file", 120000))
    confidence_threshold = export_cfg.get("confidence_threshold")
    confidence_threshold = float(confidence_threshold) if confidence_threshold is not None else None
    camera_sources = set(export_cfg.get("camera_sources", ["gt_camera"]))
    depth_sources = list(export_cfg.get("depth_sources", ["pred_depth_metric_scaled"]))

    depth_variants: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    depth_variants["pred_depth"] = (pred_depth.astype(np.float32), {"depth_metric_scale": 1.0})

    needs_metric_depth = any(source in {"pred_depth_metric_scaled", "gt_depth"} for source in depth_sources)
    gt_depth_pre = None
    depth_scale_info: dict[str, Any] | None = None
    if needs_metric_depth and "gt_depth" in inputs:
        gt_depth_pre = resize_float_stack(inputs["gt_depth"].astype(np.float32), pred_depth.shape[-2:])
        target_mask_pre = None
        if "counterfactual_target_mask" in inputs:
            target_mask_pre = resize_bool_stack(inputs["counterfactual_target_mask"], pred_depth.shape[-2:])
        depth_scale, depth_scale_info = estimate_depth_scale(
            pred_depth,
            gt_depth_pre,
            target_mask_pre,
            seed=int(cfg["project"]["random_seed"]),
        )
        depth_variants["pred_depth_metric_scaled"] = (
            (pred_depth.astype(np.float32) * np.float32(depth_scale)).astype(np.float32),
            depth_scale_info,
        )
        depth_variants["gt_depth"] = (gt_depth_pre.astype(np.float32), {"depth_metric_scale": 1.0})
    elif needs_metric_depth:
        raise RuntimeError(
            "reconstruction_pointclouds requested gt_depth or pred_depth_metric_scaled, "
            "but inputs.npz does not contain gt_depth. Rerun without --skip-gt-depth or use depth_sources: [pred_depth]."
        )

    source_params: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "pred_camera" in camera_sources:
        source_params["pred_camera"] = (pred["pred_intrinsic"].astype(np.float32), pred["pred_cam2world"].astype(np.float32))
    if "gt_camera" in camera_sources:
        source_params["gt_camera"] = (
            inputs["gt_intrinsic_preprocessed"].astype(np.float32),
            inputs["gt_cam2world"].astype(np.float32),
        )

    manifest: list[dict[str, Any]] = []
    segments = reconstruction_ply_segments(cfg, frame_ids, export_cfg.get("segments"))
    for depth_source in depth_sources:
        if depth_source not in depth_variants:
            raise ValueError(f"Unknown reconstruction pointcloud depth_source {depth_source!r}. Expected one of {sorted(depth_variants)}.")
        depth, depth_info = depth_variants[depth_source]
        for camera_source, (intrinsics, cam2worlds) in source_params.items():
            for segment_name, segment_frame_ids in segments.items():
                pts, cols = collect_segment_pointcloud(
                    depth,
                    colors,
                    confidence,
                    frame_ids,
                    segment_frame_ids,
                    intrinsics,
                    cam2worlds,
                    max_points=max_points,
                    seed=int(cfg["project"]["random_seed"]) + len(segment_name) + len(depth_source),
                    confidence_threshold=confidence_threshold,
                )
                ply_path = recon_dir / "pointclouds" / depth_source / camera_source / f"{segment_name}.ply"
                write_ply(ply_path, pts, cols)
                manifest.append(
                    {
                        "depth_source": depth_source,
                        "camera_source": camera_source,
                        "segment": segment_name,
                        "path": str(ply_path),
                        "source_frame_ids": segment_frame_ids,
                        "points_written": int(pts.shape[0]),
                        "max_points_per_file": max_points,
                        "confidence_threshold": confidence_threshold,
                        **depth_info,
                    }
                )
    if depth_scale_info is not None:
        write_json(recon_dir / "pointclouds" / "depth_metric_scale.json", depth_scale_info)
    write_json(recon_dir / "pointclouds" / "manifest.json", manifest)
    return manifest


def run_condition(cfg: dict[str, Any], condition: str, smoke: bool = False, skip_gt_depth: bool = False) -> Path:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch = require_torch()
    set_seed(int(cfg["project"]["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(cfg, device)

    all_frame_ids = list_frame_ids(cfg, condition)
    if smoke:
        end = int(cfg["frames"]["smoke_prefix_end"])
        frame_ids = [fid for fid in all_frame_ids if fid <= end]
        save_frame_ids = set(int(x) for x in cfg["frames"]["smoke_save_frames"])
    else:
        frame_ids = all_frame_ids
        save_frame_ids = configured_feature_frame_ids(cfg, frame_ids)

    images_cpu, image_paths = load_images_tensor(cfg, condition, frame_ids)
    pre_h, pre_w = int(images_cpu.shape[-2]), int(images_cpu.shape[-1])
    images = images_cpu.to(device)

    model, load_info = build_model(cfg, device)
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    pred_store: dict[str, list[np.ndarray]] = {}
    feature_store: dict[str, Any] = {}
    memory_store: dict[str, Any] = {}
    cache_log: list[dict[str, Any]] = []

    selected_layers = [int(x) for x in cfg["features"]["selected_layers"]]
    expected_layers = [4, 11, 17, 23]
    if selected_layers != expected_layers:
        raise ValueError(
            f"GCTStream._aggregate_features returns hard-coded selected_idx={expected_layers}; "
            f"got config features.selected_layers={selected_layers}."
        )
    save_dtype = torch.float16 if cfg["features"].get("save_dtype") == "float16" else torch.float32
    scale_frames = int(cfg["model"]["anchor_frame_count"])
    keyframe_interval = int(cfg["model"]["keyframe_interval"])
    output_device = torch.device("cpu") if cfg["model"].get("offload_to_cpu", True) else None

    model.clean_kv_cache()
    with AggregateFeatureCapture(model, selected_layers, save_dtype=save_dtype) as capture:
        autocast_enabled = device.type == "cuda" and dtype != torch.float32
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            scale_images = images[:scale_frames].unsqueeze(0)
            scale_output = model.forward(
                scale_images,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
            if output_device is not None:
                for key, value in list(scale_output.items()):
                    if torch.is_tensor(value):
                        scale_output[key] = value.to(output_device)
                    elif isinstance(value, list):
                        scale_output[key] = [v.to(output_device) if torch.is_tensor(v) else v for v in value]
            append_predictions(pred_store, scale_output, frame_ids[:scale_frames])
            consume_features(capture, frame_ids[:scale_frames], feature_store, memory_store, save_frame_ids)
            for fid in frame_ids[:scale_frames]:
                cache_log.append(cache_record(cfg, fid, "scale_batch", True, model))
            del scale_output

            for local_idx, frame_id in enumerate(frame_ids[scale_frames:], start=scale_frames):
                is_keyframe = (keyframe_interval <= 1) or ((local_idx - scale_frames) % keyframe_interval == 0)
                if not is_keyframe:
                    model._set_skip_append(True)
                frame_output = model.forward(
                    images[local_idx : local_idx + 1].unsqueeze(0),
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
                if not is_keyframe:
                    model._set_skip_append(False)
                if output_device is not None:
                    for key, value in list(frame_output.items()):
                        if torch.is_tensor(value):
                            frame_output[key] = value.to(output_device)
                        elif isinstance(value, list):
                            frame_output[key] = [v.to(output_device) if torch.is_tensor(v) else v for v in value]
                append_predictions(pred_store, frame_output, [frame_id])
                consume_features(capture, [frame_id], feature_store, memory_store, save_frame_ids)
                cache_log.append(cache_record(cfg, frame_id, "streaming", is_keyframe, model))
                del frame_output

    pred = {key: np.stack(values, axis=0) for key, values in pred_store.items()}
    postprocess_pose_arrays(cfg, pred, (pre_h, pre_w))
    if not bool(cfg.get("exports", {}).get("save_model_input_images", False)):
        pred.pop("images", None)

    cam_data = read_json(camera_json_path(cfg, condition))
    original_k = intrinsic_matrix(cam_data)
    original_h = int(cam_data["intrinsic"]["height"])
    original_w = int(cam_data["intrinsic"]["width"])
    gt_c2w_blender = gt_cam2worlds(cam_data)[frame_ids]
    gt_c2w = gt_cam2worlds_opencv(cam_data)[frame_ids]
    gt_k_pre = preprocessed_intrinsic(original_k, (original_h, original_w), (pre_h, pre_w))

    masks = []
    mask_counts = []
    target_id = int(target_object(cfg)["object_id"])
    for fid in frame_ids:
        mask_path = find_counterfactual_mask(cfg, condition, fid)
        if mask_path is None:
            masks.append(np.zeros((original_h, original_w), dtype=bool))
        else:
            masks.append(load_any_target_mask(mask_path, target_id))
        mask_counts.append(int(masks[-1].sum()))

    inputs: dict[str, Any] = {
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "gt_cam2world": gt_c2w.astype(np.float32),
        "gt_cam2world_blender": gt_c2w_blender.astype(np.float32),
        "gt_intrinsic_original": original_k.astype(np.float32),
        "gt_intrinsic_preprocessed": gt_k_pre.astype(np.float32),
        "counterfactual_target_mask": np.stack(masks, axis=0).astype(bool),
        "counterfactual_mask_pixel_count": np.asarray(mask_counts, dtype=np.int32),
    }
    if not skip_gt_depth:
        try:
            inputs["gt_depth"] = np.stack([load_depth_exr(depth_path(cfg, condition, fid)) for fid in frame_ids], axis=0)
        except Exception as exc:
            inputs["gt_depth_error"] = np.asarray([str(exc)])

    out_root = output_root(cfg)
    recon_dir = out_root / "reconstruction" / condition
    feature_dir = out_root / "features" / condition
    memory_dir = out_root / "memory_tokens" / condition
    recon_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(recon_dir / "predictions.npz", **pred)
    np.savez_compressed(recon_dir / "inputs.npz", **inputs)
    pointcloud_manifest = export_reconstruction_pointclouds(cfg, recon_dir, pred, inputs, frame_ids, images_cpu)
    save_feature_store(feature_dir / "features.pt", feature_store)
    save_memory_store(memory_dir / "special_tokens.pt", memory_store)

    run_metadata = {
        "condition": condition,
        "smoke": smoke,
        "frame_ids": frame_ids,
        "saved_feature_frame_ids": sorted(save_frame_ids.intersection(frame_ids)),
        "image_paths": image_paths,
        "input_image_shape_chw": list(images_cpu.shape[1:]),
        "preprocessed_hw": [pre_h, pre_w],
        "original_hw": [original_h, original_w],
        "gt_intrinsic_preprocessed": gt_k_pre.tolist(),
        "model_load": load_info,
        "config": {
            "anchor_frame_count": scale_frames,
            "local_window_size": cfg["model"]["local_window_size"],
            "keyframe_interval": keyframe_interval,
            "enable_3d_rope": cfg["model"]["enable_3d_rope"],
            "use_sdpa": cfg["model"]["use_sdpa"],
            "load_dinov2_preinit": False,
            "dtype": str(dtype),
        },
        "output_head_note": (
            "This runner builds LingBot-Map with camera and depth heads enabled and point/local point heads disabled. "
            "predictions.npz stores pose_enc, depth/depth_conf, and camera postprocess outputs. "
            "Model input images are omitted unless exports.save_model_input_images=true."
        ),
        "feature_capture_note": (
            "Features are captured inside the same reconstruction forward pass from GCTStream._aggregate_features. "
            "LingBot-Map selects block groups [4, 11, 17, 23] and returns concat(frame_block, global_block) along channels; "
            "the experiment splits that last dimension into frame_block and global_block tensors."
        ),
        "gt_pose_note": (
            "inputs.npz stores gt_cam2world in OpenCV camera convention for LingBot/depth unprojection. "
            "The original Blender/OpenGL camera matrices are stored as gt_cam2world_blender."
        ),
        "pointcloud_exports": pointcloud_manifest,
        "target_bbox": target_bbox(cfg),
        "cache_log_path": str(recon_dir / "cache_log.json"),
    }
    write_json(recon_dir / "metadata.json", run_metadata)
    write_json(recon_dir / "cache_log.json", cache_log)
    print(f"Wrote reconstruction for {condition}: {recon_dir}")
    return recon_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LingBot-Map streaming reconstruction for one revisit-memory condition.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    parser.add_argument("--condition", required=True, choices=["always_present", "always_absent", "seen_then_removed"])
    parser.add_argument("--smoke", action="store_true", help="Run prefix through frame 56 and save only smoke feature frames.")
    parser.add_argument("--skip-gt-depth", action="store_true", help="Do not load EXR GT depth into outputs.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_condition(cfg, args.condition, smoke=args.smoke, skip_gt_depth=args.skip_gt_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
