from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import torch


def add_lingbot_to_path(repo: str | Path) -> None:
    root = Path(repo).expanduser().resolve()
    if not (root / "lingbot_map").is_dir():
        raise FileNotFoundError(f"Not a LingBot-Map repo: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def build_lingbot_model(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    repo = Path(cfg["paths"]["lingbot_repo"]).expanduser().resolve()
    ckpt_path = Path(cfg["paths"]["lingbot_checkpoint"]).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LingBot checkpoint not found: {ckpt_path}")
    add_lingbot_to_path(repo)
    from lingbot_map.models.gct_stream import GCTStream

    model_cfg = cfg.get("model", {})
    features_cfg = cfg.get("features", {})
    model = GCTStream(
        img_size=int(model_cfg.get("image_size", 518)),
        patch_size=int(model_cfg.get("patch_size", 14)),
        enable_camera=bool(model_cfg.get("save_predicted_pose", False)),
        enable_point=False,
        enable_depth=False,
        enable_local_point=False,
        enable_track=False,
        enable_3d_rope=bool(model_cfg.get("video_rope", True)),
        max_frame_num=int(model_cfg.get("max_frame_num", 1024)),
        kv_cache_sliding_window=int(model_cfg.get("local_window", 32)),
        kv_cache_scale_frames=int(model_cfg.get("num_scale_frames", 8)),
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        kv_cache_camera_only=False,
        use_sdpa=bool(model_cfg.get("use_sdpa", True)),
        camera_num_iterations=int(model_cfg.get("camera_num_iterations", 4)),
        use_gradient_checkpoint=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    meta = {
        "repo_path": str(repo),
        "checkpoint_path": str(ckpt_path),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "layers": list(features_cfg.get("layers", [4, 11, 17, 23])),
        "token_layout": {
            "camera_token_idx": 0,
            "register_token_slice": [1, 5],
            "scale_token_idx": 5,
            "patch_start_idx": 6,
            "layer_indexing": "0-based block group indices",
        },
        "feature_source": (
            "LingBot aggregator output_list entries produced inside AggregatorBase.forward "
            "by torch.cat([frame_intermediate, global_intermediate], dim=-1). These are the "
            "same concat tokens consumed by the prediction heads."
        ),
        "instrumentation": "none; no LingBot core source edits, forward hooks, or monkey patches are used",
    }
    return model, meta


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def _preferred_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major = torch.cuda.get_device_capability(device)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def _extract_backbone_pool(model: Any, images: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    pools: list[torch.Tensor] = []
    mean = model.aggregator._resnet_mean.to(device)
    std = model.aggregator._resnet_std.to(device)
    with torch.inference_mode():
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + int(batch_size)].unsqueeze(0).to(device)
            normalized = ((batch - mean) / std).view(-1, batch.shape[2], batch.shape[3], batch.shape[4])
            patch = model.aggregator.patch_embed(normalized)
            if isinstance(patch, dict):
                patch = patch["x_norm_patchtokens"]
            pools.append(patch.detach().float().cpu().mean(dim=1))
    return torch.cat(pools, dim=0)


_HEAD_FEATURE_KEYS = (
    "register_head_concat",
    "register_head_frame",
    "register_head_global",
    "camera_head_concat",
    "camera_head_frame",
    "camera_head_global",
)


def split_aggregator_head_outputs(
    outputs: list[torch.Tensor],
    layers: list[int],
    *,
    patch_start_idx: int,
    dtype: torch.dtype = torch.float16,
) -> dict[str, dict[int, torch.Tensor]]:
    """Split LingBot head-input concat tokens into register/camera views.

    `outputs` are the selected `AggregatorBase.forward` outputs, each with shape
    [B, S, P, 2C]. The first C channels are the completed frame-block output and
    the second C channels are the completed global-block output. This is the exact
    representation passed to LingBot heads.
    """
    if len(outputs) != len(layers):
        raise ValueError(f"Expected {len(layers)} selected outputs, got {len(outputs)}")
    if int(patch_start_idx) < 6:
        raise ValueError(f"Expected at least 6 special tokens before patches, got patch_start_idx={patch_start_idx}")

    split: dict[str, dict[int, torch.Tensor]] = {key: {} for key in _HEAD_FEATURE_KEYS}
    for layer, tensor in zip(layers, outputs):
        if tensor.ndim != 4:
            raise ValueError(f"Layer {layer} output must be [B, S, P, 2C], got {tuple(tensor.shape)}")
        if tensor.shape[0] != 1:
            raise ValueError(f"Only B=1 scene extraction is supported, got B={tensor.shape[0]} for layer {layer}")
        if tensor.shape[2] < int(patch_start_idx):
            raise ValueError(
                f"Layer {layer} has only {tensor.shape[2]} tokens, smaller than patch_start_idx={patch_start_idx}"
            )
        if tensor.shape[-1] % 2 != 0:
            raise ValueError(f"Layer {layer} concat dim must be even, got {tensor.shape[-1]}")

        cpu = tensor.detach().to(device="cpu", dtype=dtype)
        half_dim = cpu.shape[-1] // 2
        register_concat = cpu[0, :, 1:5, :].clone()
        camera_concat = cpu[0, :, 0, :].clone()
        if register_concat.shape[1] != 4:
            raise ValueError(f"Layer {layer} register slice must contain 4 tokens, got {register_concat.shape}")

        split["register_head_concat"][int(layer)] = register_concat
        split["register_head_frame"][int(layer)] = register_concat[..., :half_dim].clone()
        split["register_head_global"][int(layer)] = register_concat[..., half_dim:].clone()
        split["camera_head_concat"][int(layer)] = camera_concat
        split["camera_head_frame"][int(layer)] = camera_concat[..., :half_dim].clone()
        split["camera_head_global"][int(layer)] = camera_concat[..., half_dim:].clone()
    return split


def _new_head_feature_store(layers: list[int]) -> dict[str, dict[int, list[torch.Tensor]]]:
    return {key: {int(layer): [] for layer in layers} for key in _HEAD_FEATURE_KEYS}


def _append_head_features(
    store: dict[str, dict[int, list[torch.Tensor]]],
    chunk: dict[str, dict[int, torch.Tensor]],
) -> None:
    for key in _HEAD_FEATURE_KEYS:
        for layer, tensor in chunk[key].items():
            store[key][int(layer)].append(tensor)


def _finalize_head_features(store: dict[str, dict[int, list[torch.Tensor]]]) -> dict[str, dict[int, torch.Tensor]]:
    finalized: dict[str, dict[int, torch.Tensor]] = {}
    for key in _HEAD_FEATURE_KEYS:
        finalized[key] = {}
        for layer, chunks in store[key].items():
            if not chunks:
                raise RuntimeError(f"No head-output chunks captured for {key} layer {layer}")
            finalized[key][int(layer)] = torch.cat(chunks, dim=0)
    return finalized


def extract_lingbot_features_for_scene(
    cfg: dict[str, Any],
    image_paths: list[str | Path],
    *,
    frame_ids: list[int],
    input_positions: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    add_lingbot_to_path(cfg["paths"]["lingbot_repo"])
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    device_name = cfg.get("run", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and str(device_name).startswith("cuda") else "cpu")
    dtype = _preferred_dtype(device)
    model, model_meta = build_lingbot_model(cfg)
    model.to(device).eval().requires_grad_(False)

    model_cfg = cfg.get("model", {})
    features_cfg = cfg.get("features", {})
    layers = [int(x) for x in features_cfg.get("layers", [4, 11, 17, 23])]
    image_size = int(model_cfg.get("image_size", 518))
    patch_size = int(model_cfg.get("patch_size", 14))
    images = load_and_preprocess_images(
        [str(p) for p in image_paths],
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    input_hw = [int(images.shape[-2]), int(images.shape[-1])]
    grid_hw = [input_hw[0] // patch_size, input_hw[1] // patch_size]

    batch_size = int(features_cfg.get("batch_size", 4))
    backbone_pool = _extract_backbone_pool(model, images, batch_size=batch_size, device=device)

    if hasattr(model.aggregator, "clean_kv_cache"):
        model.aggregator.clean_kv_cache()
    elif hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()

    num_scale = min(int(model_cfg.get("num_scale_frames", 8)), len(images))
    local_window = int(model_cfg.get("local_window", 32))
    keyframe_interval = int(model_cfg.get("keyframe_interval", 1))
    if keyframe_interval != 1:
        raise ValueError("First probe version requires keyframe_interval=1 so every frame has a register record.")

    selected_layers = list(layers)
    head_store = _new_head_feature_store(selected_layers)
    patch_start_idx_seen: int | None = None
    predicted_pose_chunks: list[torch.Tensor] = []
    with torch.inference_mode(), _autocast_context(device, dtype):
        scale_images = images[:num_scale].unsqueeze(0).to(device)
        scale_tokens, patch_start_idx = model.aggregator(
            scale_images,
            selected_idx=selected_layers,
            num_frame_for_scale=num_scale,
            sliding_window_size=local_window,
            num_frame_per_block=num_scale,
        )
        patch_start_idx_seen = int(patch_start_idx)
        _append_head_features(
            head_store,
            split_aggregator_head_outputs(
                scale_tokens,
                selected_layers,
                patch_start_idx=patch_start_idx_seen,
                dtype=torch.float16,
            ),
        )
        if model.camera_head is not None:
            if selected_layers != [4, 11, 17, 23]:
                raise ValueError("save_predicted_pose requires LingBot head layers [4, 11, 17, 23].")
            pred = model._predict_camera(
                scale_tokens,
                causal_inference=True,
                num_frame_for_scale=num_scale,
                sliding_window_size=local_window,
                num_frame_per_block=num_scale,
            )
            predicted_pose_chunks.append(pred["pose_enc"].detach().float().cpu())
        del scale_tokens, scale_images

        for idx in range(num_scale, len(images)):
            one = images[idx : idx + 1].unsqueeze(0).to(device)
            tokens, patch_start_idx = model.aggregator(
                one,
                selected_idx=selected_layers,
                num_frame_for_scale=num_scale,
                sliding_window_size=local_window,
                num_frame_per_block=1,
            )
            if int(patch_start_idx) != patch_start_idx_seen:
                raise RuntimeError(f"patch_start_idx changed from {patch_start_idx_seen} to {patch_start_idx}")
            _append_head_features(
                head_store,
                split_aggregator_head_outputs(
                    tokens,
                    selected_layers,
                    patch_start_idx=int(patch_start_idx),
                    dtype=torch.float16,
                ),
            )
            if model.camera_head is not None:
                pred = model._predict_camera(
                    tokens,
                    causal_inference=True,
                    num_frame_for_scale=num_scale,
                    sliding_window_size=local_window,
                    num_frame_per_block=1,
                )
                predicted_pose_chunks.append(pred["pose_enc"].detach().float().cpu())
            del tokens, one
    head_features = _finalize_head_features(head_store)

    for layer in selected_layers:
        reg = head_features["register_head_concat"][int(layer)]
        if reg.shape[:2] != (len(images), 4):
            raise RuntimeError(f"Layer {layer} register head concat shape mismatch: {reg.shape}")
        cam = head_features["camera_head_concat"][int(layer)]
        if cam.shape[0] != len(images) or cam.ndim != 2:
            raise RuntimeError(f"Layer {layer} camera head concat shape mismatch: {cam.shape}")

    predicted_pose = None
    if predicted_pose_chunks:
        predicted_pose = torch.cat(predicted_pose_chunks, dim=1).squeeze(0)

    features: dict[str, Any] = {
        "frame_ids": torch.tensor(frame_ids, dtype=torch.long),
        "input_positions": torch.tensor(input_positions, dtype=torch.long),
        "lingbot_backbone_pool": backbone_pool.to(dtype=torch.float16),
        "register_head_concat": head_features["register_head_concat"],
        "register_head_frame": head_features["register_head_frame"],
        "register_head_global": head_features["register_head_global"],
        "camera_head_concat": head_features["camera_head_concat"],
        "camera_head_frame": head_features["camera_head_frame"],
        "camera_head_global": head_features["camera_head_global"],
        "predicted_pose_enc": predicted_pose.to(dtype=torch.float16) if predicted_pose is not None else None,
    }
    meta = dict(model_meta)
    meta.update(
        {
            "input_hw": input_hw,
            "grid_hw": grid_hw,
            "patch_start_idx_runtime": int(patch_start_idx_seen if patch_start_idx_seen is not None else -1),
            "num_frames": len(images),
            "num_scale_frames": int(num_scale),
            "local_window": int(local_window),
            "video_rope": bool(model_cfg.get("video_rope", True)),
            "use_sdpa": bool(model_cfg.get("use_sdpa", True)),
            "kv_cache_cross_frame_special": True,
            "kv_cache_include_scale_frames": True,
            "head_feature_shape": {
                "register_head_concat": "[T, 4, 2C]",
                "register_head_frame": "[T, 4, C]",
                "register_head_global": "[T, 4, C]",
                "camera_head_concat": "[T, 2C]",
                "camera_head_frame": "[T, C]",
                "camera_head_global": "[T, C]",
            },
            "head_feature_semantics": {
                "frame_half": "completed frame block output immediately before concat",
                "global_half": "completed global block output immediately before concat, including global attention residual and global FFN",
                "concat": "torch.cat([frame_half, global_half], dim=-1), exactly the token vector consumed by LingBot prediction heads",
            },
            "trajectory_memory": "evicted dense patch K/V are removed; camera/register/scale special K/V are retained for older frames",
            "dtype": str(dtype),
            "save_predicted_pose": bool(model_cfg.get("save_predicted_pose", False)),
        }
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features, meta
