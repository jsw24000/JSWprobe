#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
SRC = EXP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from invariant_equivariant.data.manifests import filter_frames, load_manifests
from invariant_equivariant.features.cache import feature_path, save_feature_npz
from invariant_equivariant.features.indexing import read_feature_index, write_feature_index, index_path
from invariant_equivariant.features.pooling import mask_2d_features, pool_patch_tokens
from invariant_equivariant.models.dinov2_extractor import DINOv2Extractor
from invariant_equivariant.models.vggt_extractor import VGGTExtractor
from invariant_equivariant.utils import ensure_dir, load_config, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=["dinov2", "vggt"], required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def metadata_row(
    extractor: Any,
    layer_name: str,
    frame: Mapping[str, Any],
    path: Path,
    pooled: Mapping[str, Any],
    feature_dim: int,
) -> Dict[str, Any]:
    bbox = frame.get("target_bbox_2d") or [0, 0, 0, 0]
    image_size = frame.get("image_size") or [512, 512]
    bbox_w = (float(bbox[2]) - float(bbox[0]) + 1.0) / float(image_size[0]) if bbox else 0.0
    bbox_h = (float(bbox[3]) - float(bbox[1]) + 1.0) / float(image_size[1]) if bbox else 0.0
    centroid_x = (float(bbox[0]) + float(bbox[2])) * 0.5 / float(image_size[0]) if bbox else 0.0
    centroid_y = (float(bbox[1]) + float(bbox[3])) * 0.5 / float(image_size[1]) if bbox else 0.0
    valid = (
        bool(frame.get("visibility_ok", False))
        and float(frame.get("target_mask_area_ratio", 0.0)) >= float(extractor.config["pooling"]["min_mask_area_ratio"])
        and bool(pooled["valid_by_patch"])
    )
    grid = frame.get("state_position_index", [0, 0])
    return {
        "model_name": extractor.model_name,
        "model_variant": extractor.model_variant,
        "checkpoint_id": extractor.checkpoint_id,
        "layer_name": layer_name,
        "scene_id": frame["scene_id"],
        "state_id": frame["state_id"],
        "camera_id": frame["camera_id"],
        "split": frame["split"],
        "target_object_id": frame["target_object_id"],
        "target_category": frame["target_category"],
        "feature_path": str(path),
        "feature_dim": int(feature_dim),
        "mask_area_ratio": float(frame.get("target_mask_area_ratio", 0.0)),
        "patch_coverage_sum": float(pooled["patch_coverage_sum"]),
        "effective_patch_count": float(pooled["effective_patch_count"]),
        "valid": bool(valid),
        "position_world_x": float(frame["object_center_world"][0]),
        "position_world_y": float(frame["object_center_world"][1]),
        "position_world_z": float(frame["object_center_world"][2]),
        "position_ref_x": float(frame["object_center_ref_camera"][0]),
        "position_ref_y": float(frame["object_center_ref_camera"][1]),
        "position_ref_z": float(frame["object_center_ref_camera"][2]),
        "position_current_x": float(frame["object_center_current_camera"][0]),
        "position_current_y": float(frame["object_center_current_camera"][1]),
        "position_current_z": float(frame["object_center_current_camera"][2]),
        "grid_row": int(grid[0]),
        "grid_col": int(grid[1]),
        "mask_centroid_x": centroid_x,
        "mask_centroid_y": centroid_y,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "camera_numeric": int(str(frame["camera_id"]).split("_")[-1]),
    }


def save_one_feature(
    extractor: Any,
    output_dir: Path,
    layer_name: str,
    frame: Mapping[str, Any],
    patch_tokens: torch.Tensor,
    force: bool,
) -> Dict[str, Any]:
    path = feature_path(output_dir, extractor.model_name, layer_name, frame["scene_id"], frame["state_id"], frame["camera_id"])
    pooled = pool_patch_tokens(patch_tokens, frame["target_mask"], extractor.config["pooling"])
    if force or not path.exists():
        tokens_np = None
        if extractor.config["features"].get("save_patch_tokens", True):
            dtype = np.float16 if extractor.config["features"].get("dtype") == "float16" else np.float32
            tokens_np = patch_tokens.detach().cpu().numpy().astype(dtype)
        meta = {
            "patch_h": int(patch_tokens.shape[0]),
            "patch_w": int(patch_tokens.shape[1]),
            "feature_dim": int(patch_tokens.shape[2]),
            "patch_coverage_sum": pooled["patch_coverage_sum"],
            "effective_patch_count": pooled["effective_patch_count"],
        }
        save_feature_npz(path, tokens_np, pooled["pooled_raw"], pooled["pooled_l2"], meta)
    return metadata_row(extractor, layer_name, frame, path, pooled, int(patch_tokens.shape[-1]))


def extract_dinov2(config: Mapping[str, Any], frames: Sequence[Mapping[str, Any]], force: bool) -> List[Dict[str, Any]]:
    extractor = DINOv2Extractor(config)
    output_dir = Path(config["_output_dir"])
    rows: List[Dict[str, Any]] = []
    batch_size = int(config["models"]["dinov2"]["batch_size"])
    for start in range(0, len(frames), batch_size):
        batch = list(frames[start : start + batch_size])
        outputs = extractor.extract_batch([frame["rgb"] for frame in batch])
        for layer_name, tensor in outputs.items():
            for idx, frame in enumerate(batch):
                rows.append(save_one_feature(extractor, output_dir, layer_name, frame, tensor[idx], force))
    return rows


def camera_sort_key(camera_id: str) -> Tuple[int, str]:
    try:
        return int(str(camera_id).split("_")[-1]), str(camera_id)
    except ValueError:
        return 10**9, str(camera_id)


def resolve_vggt_camera_order(config: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> List[str]:
    cameras_cfg = config["selection"]["cameras"]
    if cameras_cfg == "all":
        camera_order = sorted({str(frame["camera_id"]) for frame in frames}, key=camera_sort_key)
    else:
        camera_order = [str(camera_id) for camera_id in cameras_cfg]

    reference_camera = str(config.get("reference_camera_id", "camera_000"))
    if reference_camera in camera_order:
        camera_order = [reference_camera] + [camera_id for camera_id in camera_order if camera_id != reference_camera]

    views_per_state = int(config["models"]["vggt"].get("views_per_state", len(camera_order)))
    return camera_order[:views_per_state]


def extract_vggt(config: Mapping[str, Any], frames: Sequence[Mapping[str, Any]], force: bool) -> List[Dict[str, Any]]:
    extractor = VGGTExtractor(config)
    output_dir = Path(config["_output_dir"])
    camera_order = resolve_vggt_camera_order(config, frames)
    if not camera_order:
        raise RuntimeError("VGGT camera order is empty; check selection.cameras and filtered frames.")
    rows: List[Dict[str, Any]] = []
    by_state: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for frame in frames:
        by_state[(frame["scene_id"], frame["state_id"])][frame["camera_id"]] = frame
    skipped_missing_camera = 0
    for key in sorted(by_state):
        frame_by_camera = by_state[key]
        if any(cam not in frame_by_camera for cam in camera_order):
            skipped_missing_camera += 1
            continue
        ordered = [frame_by_camera[cam] for cam in camera_order]
        outputs = extractor.extract_state_views([frame["rgb"] for frame in ordered])
        for layer_name, tensor in outputs.items():
            for idx, frame in enumerate(ordered):
                rows.append(save_one_feature(extractor, output_dir, layer_name, frame, tensor[idx], force))
    if not rows:
        available = sorted({str(frame["camera_id"]) for frame in frames}, key=camera_sort_key)
        raise RuntimeError(
            "VGGT extracted zero rows. "
            f"Resolved camera_order={camera_order}, available_cameras={available}, "
            f"state_count={len(by_state)}, skipped_missing_camera={skipped_missing_camera}."
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    output_dir = ensure_dir(Path(config["_output_dir"]))
    manifests = load_manifests(config)
    frames = filter_frames(config, manifests)
    force = bool(args.force or config["features"].get("force", False))
    existing: List[Dict[str, Any]] = []
    if index_path(output_dir).exists():
        existing = [dict(row) for row in read_feature_index(output_dir) if not str(row["model_name"]).startswith(config["models"].get(args.model, {}).get("model_name", f"__never__"))]
        # Simpler robust filtering by model family after extractor construction below.

    if args.model == "dinov2":
        new_rows = extract_dinov2(config, frames, force)
        model_name = new_rows[0]["model_name"] if new_rows else "dinov2"
    else:
        new_rows = extract_vggt(config, frames, force)
        model_name = new_rows[0]["model_name"] if new_rows else "vggt"

    if index_path(output_dir).exists():
        kept = [dict(row) for row in read_feature_index(output_dir) if row["model_name"] != model_name]
    else:
        kept = []
    all_rows = kept + new_rows
    write_feature_index(output_dir, all_rows)
    summary = {
        "model": args.model,
        "model_name": model_name,
        "feature_rows_written": len(new_rows),
        "valid_rows": sum(1 for row in new_rows if bool(row["valid"])),
    }
    ensure_dir(output_dir / "features")
    write_json(output_dir / "features" / f"{args.model}_extract_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
