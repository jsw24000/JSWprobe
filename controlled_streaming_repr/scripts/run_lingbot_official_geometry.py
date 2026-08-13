#!/usr/bin/env python3
"""Run Lingbot-map via the official demo path and save geometry artifacts.

This script mirrors the upstream demo inference/postprocess flow, but writes
stable files for downstream analysis instead of only opening an interactive
viewer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
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
    / "reconstruction_official"
    / "scene0000_00_lingbot_stream512_stride4"
    / "geometry"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save Lingbot-map official-demo geometry artifacts."
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
        "--use-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SDPA attention. Recommended on this machine; FlashInfer JIT fails here.",
    )
    parser.add_argument(
        "--save-dense-world-points",
        action="store_true",
        help="Also save dense world_points_from_depth.npy. This is very large.",
    )
    parser.add_argument(
        "--float16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store dense depth/conf arrays as float16 to reduce disk size.",
    )
    parser.add_argument("--ply-max-points", type=int, default=1_000_000)
    parser.add_argument("--ply-conf-percentile", type=float, default=70.0)
    parser.add_argument("--ply-seed", type=int, default=42)
    return parser.parse_args()


def natural_key(path: Path) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_images(image_folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        [p for p in image_folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=natural_key,
    )


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def sample_point_cloud(
    world_points: np.ndarray,
    images: np.ndarray,
    confidence: np.ndarray,
    max_points: int,
    conf_percentile: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points = world_points.reshape(-1, 3)
    colors = np.transpose(images, (0, 2, 3, 1)).reshape(-1, 3)
    colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    conf = confidence.reshape(-1)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf)
    finite_points = int(valid.sum())
    threshold = float(np.percentile(conf[valid], conf_percentile)) if finite_points else 0.0
    valid &= conf >= threshold
    kept = int(valid.sum())

    points = points[valid]
    colors = colors[valid]
    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        choice = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[choice]
        colors = colors[choice]

    return points.astype(np.float32, copy=False), colors, {
        "finite_points": finite_points,
        "kept_after_confidence": kept,
        "saved_points": int(points.shape[0]),
        "confidence_percentile": float(conf_percentile),
        "confidence_threshold": threshold,
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

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(args.lingbot_repo))

    import torch
    from demo import load_images, load_model, postprocess, prepare_for_visualization
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

    random.seed(args.ply_seed)
    np.random.seed(args.ply_seed)
    torch.manual_seed(args.ply_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = list_images(args.image_folder)
    if not frame_paths:
        raise FileNotFoundError(f"no images found under {args.image_folder}")

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
    images, paths, _ = load_images(
        image_folder=demo_args.image_folder,
        video_path=None,
        fps=demo_args.fps,
        first_k=None,
        stride=1,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    model = load_model(demo_args, device)

    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

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

    del images
    if device.type == "cuda":
        torch.cuda.empty_cache()

    predictions, images_cpu = postprocess(predictions, predictions["images"])
    vis_predictions = prepare_for_visualization(predictions, images_cpu)

    depth = vis_predictions["depth"]
    depth_conf = vis_predictions["depth_conf"]
    extrinsic = vis_predictions["extrinsic"]
    intrinsic = vis_predictions["intrinsic"]
    pose_enc = vis_predictions["pose_enc"]
    images_np = vis_predictions["images"]

    world_points_from_depth = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    ply_points, ply_colors, ply_stats = sample_point_cloud(
        world_points_from_depth,
        images_np,
        depth_conf,
        max_points=args.ply_max_points,
        conf_percentile=args.ply_conf_percentile,
        seed=args.ply_seed,
    )
    ply_path = args.output_dir / "points_depth_camera_sample.ply"
    write_ascii_ply(ply_path, ply_points, ply_colors)

    dense_dtype = np.float16 if args.float16 else np.float32
    np.save(args.output_dir / "extrinsic.npy", extrinsic.astype(np.float32))
    np.save(args.output_dir / "intrinsic.npy", intrinsic.astype(np.float32))
    np.save(args.output_dir / "pose_enc.npy", pose_enc.astype(np.float32))
    np.save(args.output_dir / "depth.npy", depth.astype(dense_dtype))
    np.save(args.output_dir / "depth_conf.npy", depth_conf.astype(dense_dtype))
    if args.save_dense_world_points:
        np.save(args.output_dir / "world_points_from_depth.npy", world_points_from_depth.astype(dense_dtype))

    metadata = {
        "status": "ok",
        "model": "lingbot-map",
        "source": "official_demo_path",
        "image_folder": str(args.image_folder),
        "frame_count": len(paths),
        "frame_paths": [str(Path(p).resolve()) for p in paths],
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "dtype": str(dtype),
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
            "float16": args.float16,
        },
        "shapes": {
            "images": list(images_np.shape),
            "pose_enc": list(pose_enc.shape),
            "extrinsic": list(extrinsic.shape),
            "intrinsic": list(intrinsic.shape),
            "depth": list(depth.shape),
            "depth_conf": list(depth_conf.shape),
            "world_points_from_depth": list(world_points_from_depth.shape),
        },
        "outputs": {
            "metadata": str(args.output_dir / "metadata.json"),
            "extrinsic": str(args.output_dir / "extrinsic.npy"),
            "intrinsic": str(args.output_dir / "intrinsic.npy"),
            "pose_enc": str(args.output_dir / "pose_enc.npy"),
            "depth": str(args.output_dir / "depth.npy"),
            "depth_conf": str(args.output_dir / "depth_conf.npy"),
            "points_ply": str(ply_path),
        },
        "ply_stats": ply_stats,
    }
    if args.save_dense_world_points:
        metadata["outputs"]["world_points_from_depth"] = str(args.output_dir / "world_points_from_depth.npy")
    if device.type == "cuda":
        metadata["gpu_peak_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
