#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from probe_utils import (
    find_scene,
    infer_token_hw,
    log,
    prepare_selected_frames,
    save_metadata,
    write_summary_md,
)


def parse_selected_idx(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("selected idx list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Lingbot-map image/patch tokens for a ScanNet scene."
    )
    parser.add_argument("--lingbot-root", default="../lingbot-map", help="Path to Lingbot-map repo.")
    parser.add_argument("--data-root", default="/disk1/3dsm/S2VGGT", help="ScanNet data root.")
    parser.add_argument("--scene", default="scene0000_00", help="Scene id, e.g. scene0000_00.")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of frames to sample.")
    parser.add_argument("--frame-stride", type=int, default=10, help="Sampling stride if enough frames exist.")
    parser.add_argument("--output", required=True, help="Output directory for frames/tokens/metadata.")
    parser.add_argument("--model-path", default=None, help="Checkpoint path. Defaults to lingbot-root/checkpoints/lingbot-map.pt.")
    parser.add_argument("--image-size", type=int, default=518, help="Lingbot-map input image width/size.")
    parser.add_argument("--patch-size", type=int, default=14, help="Patch size.")
    parser.add_argument("--selected-idx", type=parse_selected_idx, default=parse_selected_idx("4,11,17,23"))
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    parser.add_argument(
        "--use-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Lingbot-map SDPA backend. Disable to try FlashInfer.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    return parser.parse_args()


def add_lingbot_to_path(lingbot_root: Path) -> None:
    if not (lingbot_root / "lingbot_map").is_dir():
        raise FileNotFoundError(f"Not a Lingbot-map repo: {lingbot_root}")
    sys.path.insert(0, str(lingbot_root))


def load_model(args: argparse.Namespace, lingbot_root: Path, device: torch.device):
    from lingbot_map.models.gct_stream import GCTStream

    model_path = Path(args.model_path) if args.model_path else lingbot_root / "checkpoints" / "lingbot-map.pt"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}. Pass --model-path explicitly."
        )

    log("Building Lingbot-map GCTStream aggregator-only model.")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_local_point=False,
        enable_track=False,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        use_gradient_checkpoint=False,
    )

    log(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log(f"Checkpoint loaded. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    model.requires_grad_(False)
    return model, model_path


def extract_backbone_patch_tokens(
    model,
    images: torch.Tensor,
    model_input_hw: list[int],
    patch_size: int,
) -> tuple[torch.Tensor, tuple[int, int], list[int]]:
    """Return DINO patch tokens before Lingbot-map aggregator blocks."""
    aggregator = model.aggregator
    if not hasattr(aggregator, "patch_embed"):
        raise RuntimeError("Aggregator has no patch_embed module to extract backbone tokens from.")

    bsz, seq_len, channels, height, width = images.shape
    if bsz != 1:
        raise RuntimeError(f"Backbone extraction expects batch size 1, got {bsz}")

    normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
    normalized = normalized.reshape(bsz * seq_len, channels, height, width)
    patch_tokens = aggregator.patch_embed(normalized)
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]
    if patch_tokens.ndim != 3:
        raise RuntimeError(f"Unsupported backbone token shape: {list(patch_tokens.shape)}")

    raw_shape = list(patch_tokens.reshape(bsz, seq_len, patch_tokens.shape[1], patch_tokens.shape[2]).shape)
    token_hw = infer_token_hw(
        int(patch_tokens.shape[1]),
        image_hw=model_input_hw,
        patch_size=patch_size,
    )
    tokens = patch_tokens.reshape(seq_len, token_hw[0], token_hw[1], patch_tokens.shape[-1])
    return tokens.detach().float().cpu(), token_hw, raw_shape


def main() -> None:
    args = parse_args()
    t0 = time.time()

    output_dir = Path(args.output)
    frames_dir = output_dir / "frames"
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    lingbot_root = Path(args.lingbot_root).expanduser().resolve()
    add_lingbot_to_path(lingbot_root)

    from lingbot_map.utils.load_fn import load_and_preprocess_images

    log(f"Lingbot-map root: {lingbot_root}")
    log(f"Data root request: {args.data_root}")
    scene = find_scene(args.data_root, args.scene)
    log(f"Resolved scene: {scene.scene_dir}")
    if scene.rgb_dir:
        log(f"Using RGB directory: {scene.rgb_dir}")
    elif scene.sens_path:
        log(f"No extracted RGB directory; extracting sampled RGB frames from: {scene.sens_path}")

    frame_indices, source_paths, frame_paths, original_sizes, rgb_source_type = prepare_selected_frames(
        scene=scene,
        num_frames=args.num_frames,
        stride=args.frame_stride,
        output_frame_dir=frames_dir,
    )
    log(f"Selected frame indices: {frame_indices}")
    log(f"Prepared {len(frame_paths)} RGB frames under {frames_dir}")

    log("Loading and preprocessing images with Lingbot-map utility.")
    images = load_and_preprocess_images(
        frame_paths,
        mode="crop",
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    model_input_hw = [int(images.shape[-2]), int(images.shape[-1])]
    log(f"Model input tensor: {tuple(images.shape)}")

    device = torch.device(args.device)
    model, model_path = load_model(args, lingbot_root, device)
    if torch.cuda.is_available() and device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(device)}")

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    if dtype != torch.float32:
        log(f"Casting aggregator to {dtype}")
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.unsqueeze(0).to(device)
    if hasattr(model.aggregator, "clean_kv_cache"):
        model.aggregator.clean_kv_cache()

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    log("Extracting DINO backbone patch tokens before aggregator blocks.")
    with torch.no_grad(), autocast_ctx:
        backbone_tokens, backbone_hw, backbone_raw_shape = extract_backbone_patch_tokens(
            model,
            images,
            model_input_hw=model_input_hw,
            patch_size=args.patch_size,
        )

    if hasattr(model.aggregator, "clean_kv_cache"):
        model.aggregator.clean_kv_cache()

    log("Running aggregator forward pass and collecting selected token stages.")
    with torch.no_grad(), autocast_ctx:
        aggregated_tokens_list, patch_start_idx = model.aggregator(
            images,
            selected_idx=args.selected_idx,
            num_frame_for_scale=min(args.num_scale_frames, images.shape[1]),
            num_frame_per_block=images.shape[1],
        )

    if not aggregated_tokens_list:
        raise RuntimeError("Lingbot-map aggregator returned no token stages.")

    backbone_payload = {
        "tokens": backbone_tokens,
        "token_hw": list(backbone_hw),
        "frame_indices": frame_indices,
        "image_paths": frame_paths,
        "source_image_paths": source_paths,
        "stage_name": "backbone",
        "raw_shape": backbone_raw_shape,
        "patch_start_idx": 0,
        "model_input_hw": model_input_hw,
        "source": "aggregator.patch_embed.x_norm_patchtokens_before_aggregator",
    }
    backbone_file = tokens_dir / "backbone_tokens.pt"
    torch.save(backbone_payload, backbone_file)
    token_stages = [
        {
            "stage_name": "backbone",
            "file": str(backbone_file.relative_to(output_dir)),
            "raw_shape": backbone_raw_shape,
            "tokens_shape": list(backbone_tokens.shape),
            "token_hw": list(backbone_hw),
            "patch_start_idx": 0,
            "source": "aggregator.patch_embed.x_norm_patchtokens_before_aggregator",
        }
    ]
    log(f"Saved backbone: tokens shape {tuple(backbone_tokens.shape)} -> {backbone_file}")

    for stage_id, raw_tokens in enumerate(aggregated_tokens_list):
        stage_name = f"stage_{stage_id:02d}"
        raw_shape = list(raw_tokens.shape)
        if raw_tokens.ndim != 4:
            raise RuntimeError(f"{stage_name} has unsupported raw token shape: {raw_shape}")
        if raw_tokens.shape[0] != 1:
            raise RuntimeError(f"{stage_name} expected batch size 1, got raw shape {raw_shape}")

        spatial_tokens = raw_tokens[0, :, patch_start_idx:, :].detach().float().cpu()
        token_hw = infer_token_hw(
            int(spatial_tokens.shape[1]),
            image_hw=model_input_hw,
            patch_size=args.patch_size,
        )
        tokens = spatial_tokens.reshape(spatial_tokens.shape[0], token_hw[0], token_hw[1], spatial_tokens.shape[-1])

        token_payload = {
            "tokens": tokens,
            "token_hw": list(token_hw),
            "frame_indices": frame_indices,
            "image_paths": frame_paths,
            "source_image_paths": source_paths,
            "stage_name": stage_name,
            "raw_shape": raw_shape,
            "patch_start_idx": int(patch_start_idx),
            "model_input_hw": model_input_hw,
            "source_block_group_idx": int(args.selected_idx[stage_id])
            if stage_id < len(args.selected_idx)
            else None,
        }
        out_file = tokens_dir / f"{stage_name}_tokens.pt"
        torch.save(token_payload, out_file)
        token_stages.append(
            {
                "stage_name": stage_name,
                "file": str(out_file.relative_to(output_dir)),
                "raw_shape": raw_shape,
                "tokens_shape": list(tokens.shape),
                "token_hw": list(token_hw),
                "patch_start_idx": int(patch_start_idx),
                "source_block_group_idx": int(args.selected_idx[stage_id])
                if stage_id < len(args.selected_idx)
                else None,
            }
        )
        log(f"Saved {stage_name}: tokens shape {tuple(tokens.shape)} -> {out_file}")

    if hasattr(model.aggregator, "clean_kv_cache"):
        model.aggregator.clean_kv_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = {
        "scene": args.scene,
        "lingbot_root": str(lingbot_root),
        "data_root_request": args.data_root,
        "resolved_scene_dir": scene.scene_dir,
        "rgb_dir": scene.rgb_dir,
        "sens_path": scene.sens_path,
        "rgb_source_type": rgb_source_type,
        "frame_indices": frame_indices,
        "source_image_paths": source_paths,
        "frame_paths": frame_paths,
        "frame_output_names": [Path(p).name for p in frame_paths],
        "original_sizes": original_sizes,
        "model_input_hw": model_input_hw,
        "model_path": str(model_path),
        "model_config": {
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "selected_idx": args.selected_idx,
            "num_scale_frames": args.num_scale_frames,
            "use_sdpa": args.use_sdpa,
            "dtype": str(dtype),
            "device": str(device),
        },
        "patch_start_idx": int(patch_start_idx),
        "token_stages": token_stages,
        "pca": {},
        "controls": {},
        "correspondence": {},
        "elapsed_sec_extract": round(time.time() - t0, 3),
    }
    save_metadata(output_dir, metadata)
    write_summary_md(output_dir)
    log(f"Wrote metadata and summary under {output_dir}")


if __name__ == "__main__":
    main()
