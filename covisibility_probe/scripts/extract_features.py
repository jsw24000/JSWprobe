#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import torch

from covisibility_probe.config import load_config
from covisibility_probe.dinov2_adapter import extract_official_dino_pool
from covisibility_probe.lingbot_adapter import extract_lingbot_features_for_scene
from covisibility_probe.paths import resolve_run_dir
from covisibility_probe.scannet_io import load_scene
from covisibility_probe.utils import ensure_dir, read_parquet, save_json, setup_logging, sha256_file


def extract_features(config_path: str | Path, *, run_id: str | None = None, overwrite: bool = False) -> Path:
    cfg = load_config(config_path)
    run_dir = resolve_run_dir(cfg, run_id=run_id)
    manifest_path = run_dir / "manifests" / "frames.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")
    manifest = read_parquet(manifest_path)
    ensure_dir(run_dir / "features")
    ensure_dir(run_dir / "metadata")
    model_metadata: dict[str, object] = {
        "lingbot_checkpoint": cfg["paths"].get("lingbot_checkpoint"),
        "lingbot_checkpoint_sha256_first_64mb": sha256_file(cfg["paths"]["lingbot_checkpoint"], max_bytes=64 * 1024 * 1024),
        "dinov2_checkpoint": cfg["paths"].get("dinov2_checkpoint"),
        "dinov2_checkpoint_sha256_first_64mb": sha256_file(cfg["paths"]["dinov2_checkpoint"], max_bytes=64 * 1024 * 1024),
        "scenes": {},
    }

    for scene_id in manifest["scene_id"].drop_duplicates().tolist():
        out_path = run_dir / "features" / f"{scene_id}.pt"
        if out_path.is_file() and not overwrite:
            print(f"Features exist, skipping: {out_path}")
            continue
        df = manifest[manifest["scene_id"] == scene_id].sort_values("input_position")
        image_paths = df["color_path"].tolist()
        frame_ids = df["frame_id"].astype(int).tolist()
        input_positions = df["input_position"].astype(int).tolist()
        features, meta = extract_lingbot_features_for_scene(
            cfg,
            image_paths,
            frame_ids=frame_ids,
            input_positions=input_positions,
        )
        if cfg.get("features", {}).get("extract_official_dino", True):
            dino_pool, dino_meta = extract_official_dino_pool(
                image_paths,
                dinov2_repo=cfg["paths"]["dinov2_repo"],
                checkpoint_path=cfg["paths"]["dinov2_checkpoint"],
                lingbot_repo=cfg["paths"]["lingbot_repo"],
                image_size=int(cfg["model"].get("image_size", 518)),
                patch_size=int(cfg["model"].get("patch_size", 14)),
                batch_size=int(cfg["features"].get("batch_size", 4)),
                device=cfg["run"].get("device", "cuda"),
            )
            features["official_dino_pool"] = dino_pool.to(dtype=torch.float16)
            meta["official_dino"] = dino_meta
        else:
            features["official_dino_pool"] = None

        scene = load_scene(cfg["paths"]["scannet_root"], scene_id, pose_convention=cfg["overlap"].get("pose_convention", "camera_to_world"))
        pose_by_id = {f.frame_id: f.pose_c2w for f in scene.frames}
        gt_pose = np.stack([pose_by_id[fid] for fid in frame_ids], axis=0).astype(np.float32)
        features["gt_pose_c2w"] = torch.from_numpy(gt_pose)

        torch.save(features, out_path)
        model_metadata["scenes"][scene_id] = meta
        print(f"Features written: {out_path}")

    save_json(model_metadata, run_dir / "metadata" / "model_metadata.json")
    print(f"Model metadata written: {run_dir / 'metadata' / 'model_metadata.json'}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging()
    extract_features(args.config, run_id=args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

