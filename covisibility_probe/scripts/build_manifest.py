#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd

from covisibility_probe.config import load_config, save_config
from covisibility_probe.frame_sampling import choose_scenes, sample_scene_frames
from covisibility_probe.paths import resolve_run_dir
from covisibility_probe.scannet_io import discover_scene_ids, frame_table_rows, load_scene, scene_health
from covisibility_probe.utils import ensure_dir, git_commit, save_json, setup_logging, write_parquet


def build_manifest(config_path: str | Path, *, run_id: str | None = None, overwrite: bool = False) -> Path:
    cfg = load_config(config_path)
    run_dir = resolve_run_dir(cfg, run_id=run_id)
    manifest_path = run_dir / "manifests" / "frames.parquet"
    split_path = run_dir / "metadata" / "scene_split.json"
    if manifest_path.is_file() and split_path.is_file() and not overwrite:
        print(f"Manifest exists, skipping: {manifest_path}")
        return run_dir

    ensure_dir(run_dir / "metadata")
    ensure_dir(run_dir / "manifests")
    save_config(cfg, run_dir / "metadata" / "resolved_config.yaml")

    scannet_root = cfg["paths"].get("scannet_root")
    if not scannet_root:
        raise FileNotFoundError("No ScanNet root found. Run scripts/inspect_environment.py first.")
    pose_convention = cfg.get("overlap", {}).get("pose_convention", "camera_to_world")
    scene_ids = discover_scene_ids(scannet_root)
    scenes = [load_scene(scannet_root, sid, pose_convention=pose_convention) for sid in scene_ids]
    ds = cfg["dataset"]
    selection = choose_scenes(
        scenes,
        num_scenes=int(ds["num_scenes"]),
        frames_per_scene=int(ds["frames_per_scene"]),
        raw_frame_stride=int(ds["raw_frame_stride"]),
        preferred_debug_scenes=list(ds.get("preferred_debug_scenes") or []),
    )

    rows = []
    scene_frame_counts = {}
    for scene in scenes:
        if scene.scene_id not in selection.selected_scene_ids:
            continue
        sampled = sample_scene_frames(
            scene,
            frames_per_scene=int(ds["frames_per_scene"]),
            raw_frame_stride=int(ds["raw_frame_stride"]),
        )
        scene_frame_counts[scene.scene_id] = len(sampled)
        rows.extend(frame_table_rows(sampled))

    if not rows:
        raise RuntimeError("No valid sampled frames found.")
    df = pd.DataFrame(rows)
    write_parquet(df, manifest_path)

    split = {
        "selection_rule": "scene ids sorted after filtering valid RGB/depth/pose frames; debug preferred scenes are tried first before sorted fallback",
        "requested_num_scenes": int(ds["num_scenes"]),
        "requested_frames_per_scene": int(ds["frames_per_scene"]),
        "raw_frame_stride": int(ds["raw_frame_stride"]),
        "selected_scene_ids": selection.selected_scene_ids,
        "scene_frame_counts": scene_frame_counts,
        "skipped": selection.skipped,
        "available_scene_health": [scene_health(s) for s in scenes],
        "warnings": [],
    }
    if len(selection.selected_scene_ids) < int(ds["num_scenes"]):
        split["warnings"].append(
            f"Only {len(selection.selected_scene_ids)} scenes available/valid; requested {int(ds['num_scenes'])}."
        )
    for sid, count in scene_frame_counts.items():
        if count < int(ds["frames_per_scene"]):
            split["warnings"].append(f"{sid} has only {count} sampled frames; requested {int(ds['frames_per_scene'])}.")
    save_json(split, split_path)
    save_json(
        {
            "lingbot_map": git_commit(cfg["paths"]["lingbot_repo"]),
            "dinov2": git_commit(cfg["paths"]["dinov2_repo"]),
            "covisibility_probe": git_commit(HERE),
        },
        run_dir / "metadata" / "git_versions.json",
    )
    print(f"Manifest written: {manifest_path}")
    print(f"Scene split written: {split_path}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging()
    build_manifest(args.config, run_id=args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

