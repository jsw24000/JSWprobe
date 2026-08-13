#!/usr/bin/env python3
"""Extract Lingbot-map intermediate aggregator tokens on the official demo path."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_DIR = PROJECT_DIR.parent
DEFAULT_INPUT = (
    PROJECT_DIR
    / "data"
    / "controlled_sequences"
    / "scene0000_00_lingbot_stream512_stride4"
    / "images"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "outputs"
    / "tokens"
    / "scene0000_00_lingbot_stream512_stride4"
    / "lingbot_official"
)


def parse_layers(value: str) -> list[int] | None:
    value = value.strip()
    if value.lower() in {"all", "*"}:
        return None
    layers = [int(part) for part in re.split(r"[, ]+", value) if part]
    if not layers:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers or 'all'")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Lingbot-map intermediate tokens with streaming state."
    )
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lingbot-repo", type=Path, default=MAIN_DIR / "lingbot-map")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MAIN_DIR / "lingbot-map" / "checkpoints" / "lingbot-map.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=parse_layers("4,11,17,23"),
        help="Aggregator block-group indices to save, e.g. '4,11,17,23' or 'all'.",
    )
    parser.add_argument(
        "--token-slice",
        choices=["patch", "special", "all", "camera", "register", "scale"],
        default="patch",
        help="Which token subset to save from each layer.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="On-disk token dtype.",
    )
    parser.add_argument(
        "--use-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SDPA attention. Recommended on this machine.",
    )
    parser.add_argument(
        "--save-prediction-summary",
        action="store_true",
        help="Save small prediction shape metadata after inference.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_images(image_folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        [p for p in image_folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=natural_key,
    )


def tensor_shape(value: Any) -> list[int] | None:
    if hasattr(value, "shape"):
        return [int(x) for x in value.shape]
    return None


class TokenMemmapWriter:
    def __init__(
        self,
        output_dir: Path,
        frame_count: int,
        save_layers: list[int] | None,
        token_slice: str,
        dtype: str,
    ) -> None:
        self.output_dir = output_dir
        self.frame_count = frame_count
        self.save_layers = save_layers
        self.token_slice = token_slice
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.memmaps: dict[str, np.memmap] = {}
        self.cursor = 0
        self.patch_start_idx: int | None = None
        self.layer_labels: list[str] = []
        self.initialized = False

    def _slice_tokens(self, array: np.ndarray, patch_start_idx: int) -> np.ndarray:
        if self.token_slice == "patch":
            return array[:, :, patch_start_idx:, :]
        if self.token_slice == "special":
            return array[:, :, :patch_start_idx, :]
        if self.token_slice == "camera":
            return array[:, :, :1, :]
        if self.token_slice == "register":
            return array[:, :, 1 : max(1, patch_start_idx - 1), :]
        if self.token_slice == "scale":
            return array[:, :, patch_start_idx - 1 : patch_start_idx, :]
        return array

    def _init_memmaps(self, token_list: list[Any], patch_start_idx: int) -> None:
        self.patch_start_idx = int(patch_start_idx)
        if self.save_layers is None:
            self.layer_labels = [f"layer_{i:02d}" for i in range(len(token_list))]
        else:
            self.layer_labels = [f"layer_{idx:02d}" for idx in self.save_layers]
        if len(self.layer_labels) != len(token_list):
            raise ValueError(
                f"layer label count {len(self.layer_labels)} != token count {len(token_list)}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for label, tensor in zip(self.layer_labels, token_list):
            arr = tensor.detach().cpu().numpy()
            if arr.shape[0] != 1:
                raise ValueError(f"Only batch size 1 is supported for token export, got {arr.shape}")
            sliced = self._slice_tokens(arr, patch_start_idx)
            _, _, token_count, dim = sliced.shape
            path = self.output_dir / f"{label}_{self.token_slice}_tokens.npy"
            self.memmaps[label] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=self.dtype,
                shape=(self.frame_count, token_count, dim),
            )
        self.initialized = True

    def append(self, token_list: list[Any], patch_start_idx: int) -> None:
        if not self.initialized:
            self._init_memmaps(token_list, patch_start_idx)

        frame_chunk = int(token_list[0].shape[1])
        start = self.cursor
        end = start + frame_chunk
        if end > self.frame_count:
            raise ValueError(f"token writer overflow: {end} > {self.frame_count}")

        for label, tensor in zip(self.layer_labels, token_list):
            arr = tensor.detach().cpu().numpy()
            sliced = self._slice_tokens(arr, patch_start_idx)[0]
            self.memmaps[label][start:end] = sliced.astype(self.dtype, copy=False)
        self.cursor = end

    def close(self) -> None:
        for mmap in self.memmaps.values():
            mmap.flush()

    def output_paths(self) -> dict[str, str]:
        return {
            label: str(self.output_dir / f"{label}_{self.token_slice}_tokens.npy")
            for label in self.layer_labels
        }


def main() -> int:
    args = parse_args()
    args.image_folder = args.image_folder.resolve()
    args.output_dir = args.output_dir.resolve()
    args.lingbot_repo = args.lingbot_repo.resolve()
    args.model_path = args.model_path.resolve()

    if not args.image_folder.is_dir():
        raise FileNotFoundError(f"image folder not found: {args.image_folder}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {args.model_path}")

    frame_paths = list_images(args.image_folder)
    if not frame_paths:
        raise FileNotFoundError(f"no images found under {args.image_folder}")

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(args.lingbot_repo))

    import torch
    from demo import load_images, load_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    demo_args = argparse.Namespace(
        image_folder=str(args.image_folder),
        video_path=None,
        fps=10,
        first_k=None,
        stride=1,
        model_path=str(args.model_path),
        image_size=args.image_size,
        patch_size=args.patch_size,
        mode="streaming",
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        camera_num_iterations=args.camera_num_iterations,
        use_sdpa=args.use_sdpa,
        offload_to_cpu=True,
        window_size=64,
        overlap_size=16,
    )

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    images, loaded_paths, _ = load_images(
        image_folder=demo_args.image_folder,
        video_path=None,
        fps=demo_args.fps,
        first_k=None,
        stride=1,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    frame_count = int(images.shape[0])

    model = load_model(demo_args, device)
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    writer = TokenMemmapWriter(
        args.output_dir,
        frame_count=frame_count,
        save_layers=args.layers,
        token_slice=args.token_slice,
        dtype=args.dtype,
    )

    original_aggregate = model._aggregate_features
    head_layers = [4, 11, 17, 23]

    def capture_aggregate(self: Any, images: Any, *a: Any, **kw: Any) -> tuple:
        if args.layers is None:
            aggregator_layers = None
        else:
            aggregator_layers = sorted(set(args.layers + head_layers))

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            selected_idx=aggregator_layers,
            num_frame_for_scale=kw.get("num_frame_for_scale"),
            sliding_window_size=kw.get("sliding_window_size"),
            num_frame_per_block=kw.get("num_frame_per_block", 1),
        )

        if aggregator_layers is None:
            token_by_layer = {idx: tensor for idx, tensor in enumerate(aggregated_tokens_list)}
            save_layers = list(range(len(aggregated_tokens_list))) if args.layers is None else args.layers
        else:
            token_by_layer = {
                idx: tensor for idx, tensor in zip(aggregator_layers, aggregated_tokens_list)
            }
            save_layers = args.layers

        save_tokens = [token_by_layer[idx] for idx in save_layers]
        head_tokens = [token_by_layer[idx] for idx in head_layers]
        writer.append(save_tokens, patch_start_idx)
        return head_tokens, patch_start_idx

    model._aggregate_features = MethodType(capture_aggregate, model)

    images = images.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t_infer = time.time()
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=torch.device("cpu"),
        )
    inference_s = time.time() - t_infer
    writer.close()
    if writer.cursor != frame_count:
        raise RuntimeError(f"token writer saved {writer.cursor} frames, expected {frame_count}")

    # Restore for cleanliness in case this is imported.
    model._aggregate_features = original_aggregate

    metadata = {
        "status": "ok",
        "model": "lingbot-map",
        "source": "official_demo_path_with_aggregate_capture",
        "image_folder": str(args.image_folder),
        "frame_count": frame_count,
        "frame_paths": [str(Path(p).resolve()) for p in loaded_paths],
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "inference_dtype": str(dtype),
        "token_dtype": args.dtype,
        "inference_seconds": round(inference_s, 3),
        "total_seconds": round(time.time() - t0, 3),
        "params": {
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "num_scale_frames": args.num_scale_frames,
            "camera_num_iterations": args.camera_num_iterations,
            "keyframe_interval": args.keyframe_interval,
            "kv_cache_sliding_window": args.kv_cache_sliding_window,
            "max_frame_num": args.max_frame_num,
            "use_sdpa": args.use_sdpa,
            "layers": "all" if args.layers is None else args.layers,
            "head_layers": head_layers,
            "token_slice": args.token_slice,
        },
        "token_layout": {
            "patch_start_idx": writer.patch_start_idx,
            "layer_labels": writer.layer_labels,
            "saved_frame_count": writer.cursor,
            "shape_per_file": {
                label: list(mmap.shape) for label, mmap in writer.memmaps.items()
            },
        },
        "outputs": {
            "metadata": str(args.output_dir / "metadata.json"),
            "tokens": writer.output_paths(),
        },
    }
    if args.save_prediction_summary:
        metadata["prediction_shapes"] = {
            key: tensor_shape(value) for key, value in predictions.items()
        }
    if device.type == "cuda":
        metadata["gpu_peak_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
