#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

from covisibility_probe.config import load_config
from covisibility_probe.frame_sampling import sample_scene_frames
from covisibility_probe.metrics import spearmanr
from covisibility_probe.overlap import (
    build_overlap_matrices,
    reprojection_overlap_sample,
    save_overlap_npz,
    save_visible_voxels,
    visible_voxels_for_frame,
)
from covisibility_probe.paths import resolve_run_dir
from covisibility_probe.scannet_io import load_scene
from covisibility_probe.utils import ensure_dir, read_parquet, save_json, setup_logging


def _audit_scene(cfg: dict, scene, frames, result, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = len(frames)
    upper = np.triu_indices(n, k=1)
    range_ok = bool(np.nanmin(result.overlap_cos) >= -1e-6 and np.nanmax(result.overlap_cos) <= 1.0 + 1e-6)
    diag = np.diag(result.overlap_cos)
    self_ok = bool(np.allclose(diag[result.counts.diagonal() > 0], 1.0, atol=1e-4)) if n else True
    symmetry_ok = bool(np.allclose(result.overlap_cos, result.overlap_cos.T, atol=1e-6))
    adjacent = []
    far = []
    for i in range(n - 1):
        adjacent.append(float(result.overlap_cos[i, i + 1]))
    if n > 4:
        for _ in range(min(200, n * 4)):
            i, j = rng.choice(n, size=2, replace=False)
            if abs(int(i) - int(j)) > max(4, n // 4):
                far.append(float(result.overlap_cos[min(i, j), max(i, j)]))

    num_pairs = min(int(cfg["overlap"].get("reprojection_audit_pairs", 100)), len(upper[0]))
    voxel_vals = []
    reproj_vals = []
    if num_pairs > 0:
        choices = rng.choice(len(upper[0]), size=num_pairs, replace=False)
        for idx in choices:
            i = int(upper[0][idx])
            j = int(upper[1][idx])
            voxel_vals.append(float(result.coverage_i_to_j[i, j]))
            reproj_vals.append(
                reprojection_overlap_sample(
                    scene,
                    frames[i],
                    frames[j],
                    depth_scale=float(cfg["overlap"]["depth_scale"]),
                    seed=seed + idx,
                    abs_tol_m=float(cfg["overlap"].get("depth_abs_tol_m", 0.05)),
                    rel_tol=float(cfg["overlap"].get("depth_rel_tol", 0.05)),
                )
            )
    return {
        "self_overlap_diag_close_to_one": self_ok,
        "range_0_1": range_ok,
        "symmetric_overlap": symmetry_ok,
        "adjacent_mean_overlap": float(np.mean(adjacent)) if adjacent else None,
        "far_random_mean_overlap": float(np.mean(far)) if far else None,
        "adjacent_greater_than_far": bool(np.mean(adjacent) > np.mean(far)) if adjacent and far else None,
        "reprojection_audit_pairs": int(num_pairs),
        "voxel_vs_reprojection_spearman": spearmanr(np.asarray(voxel_vals), np.asarray(reproj_vals)) if voxel_vals else None,
    }


def build_overlap(config_path: str | Path, *, run_id: str | None = None, overwrite: bool = False) -> Path:
    cfg = load_config(config_path)
    run_dir = resolve_run_dir(cfg, run_id=run_id)
    manifest_path = run_dir / "manifests" / "frames.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")
    manifest = read_parquet(manifest_path)
    ensure_dir(run_dir / "overlap" / "visible_voxels")
    ensure_dir(run_dir / "metadata")

    audits = {}
    for scene_id in manifest["scene_id"].drop_duplicates().tolist():
        out_path = run_dir / "overlap" / f"{scene_id}_overlap.npz"
        if out_path.is_file() and not overwrite:
            print(f"Overlap exists, skipping: {out_path}")
            continue
        scene = load_scene(cfg["paths"]["scannet_root"], scene_id, pose_convention=cfg["overlap"].get("pose_convention", "camera_to_world"))
        frame_ids = manifest[manifest["scene_id"] == scene_id]["frame_id"].astype(int).tolist()
        by_id = {f.frame_id: f for f in scene.frames}
        frames = [by_id[fid] for fid in frame_ids]
        voxels = [
            visible_voxels_for_frame(
                fr,
                scene,
                depth_scale=float(cfg["overlap"]["depth_scale"]),
                pixel_stride=int(cfg["overlap"]["pixel_stride"]),
                voxel_size=float(cfg["overlap"]["voxel_size"]),
            )
            for fr in frames
        ]
        save_visible_voxels(voxels, frame_ids, run_dir / "overlap" / "visible_voxels" / scene_id)
        result = build_overlap_matrices(frame_ids, voxels)
        save_overlap_npz(result, out_path)

        sensitivity = {}
        for voxel_size in cfg["overlap"].get("sensitivity_voxel_sizes", []):
            vs = float(voxel_size)
            if abs(vs - float(cfg["overlap"]["voxel_size"])) < 1e-9:
                continue
            vox_s = [
                visible_voxels_for_frame(
                    fr,
                    scene,
                    depth_scale=float(cfg["overlap"]["depth_scale"]),
                    pixel_stride=int(cfg["overlap"]["pixel_stride"]),
                    voxel_size=vs,
                )
                for fr in frames
            ]
            res_s = build_overlap_matrices(frame_ids, vox_s)
            sens_path = run_dir / "overlap" / f"{scene_id}_overlap_voxel_{vs:.2f}.npz"
            save_overlap_npz(res_s, sens_path)
            sensitivity[str(vs)] = str(sens_path)

        audits[scene_id] = _audit_scene(cfg, scene, frames, result, seed=int(cfg["run"].get("seed", 42)))
        audits[scene_id]["sensitivity_overlap_files"] = sensitivity
        print(f"Overlap written: {out_path}")

    save_json(audits, run_dir / "metadata" / "overlap_audit.json")
    print(f"Overlap audit written: {run_dir / 'metadata' / 'overlap_audit.json'}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging()
    build_overlap(args.config, run_id=args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

