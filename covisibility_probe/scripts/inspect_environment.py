#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from covisibility_probe.paths import candidate_scannet_roots, default_paths, project_root, workspace_root
from covisibility_probe.scannet_io import discover_scene_ids
from covisibility_probe.utils import git_commit, save_json, setup_logging, sha256_file


def module_status(names: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def find_experiments(root: Path) -> list[dict[str, object]]:
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name in {"lingbot-map", "dinov2", "vggt", "weights", "SSH", "SSH2"}:
            continue
        has_scripts = (p / "scripts").is_dir()
        has_configs = (p / "configs").is_dir()
        has_outputs = (p / "outputs").is_dir()
        if has_scripts or has_configs or has_outputs:
            out.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "has_scripts": has_scripts,
                    "has_configs": has_configs,
                    "has_outputs": has_outputs,
                }
            )
    return out


def _numeric_files(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []
    allowed = {s.lower() for s in suffixes}
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed and p.stem.isdigit()]
    return sorted(files, key=lambda p: int(p.stem))


def light_scene_report(scannet_root: Path, scene_id: str) -> dict[str, object]:
    scene_dir = scannet_root / scene_id
    color = _numeric_files(scene_dir / "color", (".jpg", ".jpeg", ".png"))
    depth = _numeric_files(scene_dir / "depth", (".png", ".tif", ".tiff"))
    pose = _numeric_files(scene_dir / "pose", (".txt",))
    intr = scene_dir / "intrinsic"
    color_size = None
    depth_size = None
    sample_pose_valid = None
    if color:
        color_size = list(Image.open(color[0]).size)
    if depth:
        depth_size = list(Image.open(depth[0]).size)
    if pose:
        try:
            mat = np.loadtxt(pose[0])
            sample_pose_valid = bool(mat.shape == (4, 4) and np.isfinite(mat).all())
        except Exception:
            sample_pose_valid = False
    common = sorted({int(p.stem) for p in color} & {int(p.stem) for p in depth} & {int(p.stem) for p in pose})
    return {
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "num_color_files": len(color),
        "num_depth_files": len(depth),
        "num_pose_files": len(pose),
        "num_common_rgb_depth_pose_ids": len(common),
        "has_rgb": bool(color),
        "has_depth": bool(depth),
        "has_pose": bool(pose),
        "has_intrinsic_color": (intr / "intrinsic_color.txt").is_file(),
        "has_intrinsic_depth": (intr / "intrinsic_depth.txt").is_file(),
        "has_extrinsic_color": (intr / "extrinsic_color.txt").is_file(),
        "has_extrinsic_depth": (intr / "extrinsic_depth.txt").is_file(),
        "color_sizes": [color_size] if color_size else [],
        "depth_sizes": [depth_size] if depth_size else [],
        "sample_pose_valid": sample_pose_valid,
        "first_frame_id": common[0] if common else None,
        "last_frame_id": common[-1] if common else None,
        "num_valid_frames": len(common),
        "num_frames_total": max(len(color), len(depth), len(pose)),
    }


def build_report(max_scenes: int = 50) -> dict[str, object]:
    root = workspace_root()
    defaults = default_paths()
    scannet_candidates = [
        {"path": str(p), "exists": p.is_dir(), "scene_count": len(discover_scene_ids(p)) if p.is_dir() else 0}
        for p in candidate_scannet_roots()
    ]
    scannet_root = defaults["scannet_root"]
    scene_reports = []
    if scannet_root:
        for sid in discover_scene_ids(scannet_root)[:max_scenes]:
            try:
                scene_reports.append(light_scene_report(Path(scannet_root), sid))
            except Exception as exc:
                scene_reports.append({"scene_id": sid, "status": "error", "error": str(exc)})

    paths = {
        "workspace_root": str(root),
        "project_root": str(project_root()),
        "lingbot_repo": str(defaults["lingbot_repo"]),
        "lingbot_checkpoint": str(defaults["lingbot_checkpoint"]),
        "dinov2_repo": str(defaults["dinov2_repo"]),
        "dinov2_checkpoint": str(defaults["dinov2_checkpoint"]),
        "vggt_repo": str(defaults["vggt_repo"]),
        "vggt_checkpoint": str(defaults["vggt_checkpoint"]),
        "scannet_root": str(scannet_root) if scannet_root else None,
    }
    weights = {}
    for key in ("lingbot_checkpoint", "dinov2_checkpoint", "vggt_checkpoint"):
        p = Path(paths[key])
        weights[key] = {
            "path": str(p),
            "exists": p.is_file(),
            "size_bytes": p.stat().st_size if p.is_file() else None,
            "sha256_first_64mb": sha256_file(p, max_bytes=64 * 1024 * 1024) if p.is_file() else None,
        }

    report = {
        "cwd": os.getcwd(),
        "paths": paths,
        "repositories": {
            "lingbot_map": {"exists": Path(paths["lingbot_repo"]).is_dir(), "git_commit": git_commit(paths["lingbot_repo"])},
            "dinov2": {"exists": Path(paths["dinov2_repo"]).is_dir(), "git_commit": git_commit(paths["dinov2_repo"])},
            "vggt": {"exists": Path(paths["vggt_repo"]).is_dir(), "git_commit": git_commit(paths["vggt_repo"])},
        },
        "weights": weights,
        "python": {"executable": sys.executable, "version": sys.version},
        "modules": module_status(["torch", "torchvision", "numpy", "pandas", "pyarrow", "scipy", "sklearn", "PIL", "matplotlib", "yaml", "tqdm"]),
        "scannet_candidates": scannet_candidates,
        "scannet_scenes": scene_reports,
        "existing_experiments": find_experiments(root),
        "notes": [
            "No source data, checkpoints, or existing experiments are modified by this script.",
            "Checkpoint hashes are first-64MB SHA256 fingerprints to keep inspection lightweight.",
        ],
    }
    return report


def print_report(report: dict[str, object]) -> None:
    paths = report["paths"]
    print("=== Covisibility Probe Environment ===")
    print(f"Workspace: {paths['workspace_root']}")
    print(f"LingBot-Map: {paths['lingbot_repo']}")
    print(f"DINOv2: {paths['dinov2_repo']}")
    print(f"VGGT: {paths['vggt_repo']}")
    print(f"ScanNet root: {paths['scannet_root']}")
    print("")
    print("Weights:")
    for name, item in report["weights"].items():
        print(f"  {name}: exists={item['exists']} size={item['size_bytes']} path={item['path']}")
    print("")
    print("Python modules:")
    for name, ok in report["modules"].items():
        print(f"  {name}: {'ok' if ok else 'missing'}")
    print("")
    scenes = report["scannet_scenes"]
    print(f"ScanNet scenes inspected: {len(scenes)}")
    for scene in scenes[:12]:
        print(
            "  {scene_id}: valid={num_valid_frames} total={num_frames_total} "
            "rgb={has_rgb} depth={has_depth} pose={has_pose}".format(**scene)
        )
    if len(scenes) > 12:
        print(f"  ... {len(scenes) - 12} more")
    print("")
    print(f"Existing experiment-like directories: {', '.join(x['name'] for x in report['existing_experiments'])}")
    print("Report written to covisibility_probe/outputs/environment_report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenes", type=int, default=50)
    args = parser.parse_args()
    setup_logging()
    report = build_report(max_scenes=args.max_scenes)
    out = project_root() / "outputs" / "environment_report.json"
    save_json(report, out)
    print_report(report)


if __name__ == "__main__":
    main()
