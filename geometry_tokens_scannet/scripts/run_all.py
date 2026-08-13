#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geometry-token extraction, PCA-noise extraction, and comparison.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--scene-id", default=None, help="Override controlled.scene_id.")
    parser.add_argument("--setting-name", default=None, help="Override controlled.setting_name.")
    parser.add_argument("--condition-id", default=None, help="Override controlled.condition_id.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for debugging.")
    parser.add_argument("--skip-geometry", action="store_true", help="Do not run extract_geometry_tokens.py.")
    parser.add_argument("--skip-pca-noise", action="store_true", help="Do not run extract_pca_noise.py.")
    parser.add_argument("--skip-compare", action="store_true", help="Do not run compare_geometry_pca_noise.py.")
    return parser.parse_args()


def add_common_args(cmd: list[str], args: argparse.Namespace, include_max_frames: bool) -> list[str]:
    cmd.extend(["--config", args.config])
    if args.scene_id:
        cmd.extend(["--scene-id", args.scene_id])
    if args.setting_name:
        cmd.extend(["--setting-name", args.setting_name])
    if args.condition_id:
        cmd.extend(["--condition-id", args.condition_id])
    if include_max_frames and args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    return cmd


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    if not args.skip_geometry:
        run(add_common_args([python, "scripts/extract_geometry_tokens.py"], args, include_max_frames=True), project_root)
    if not args.skip_pca_noise:
        run(add_common_args([python, "scripts/extract_pca_noise.py"], args, include_max_frames=True), project_root)
    if not args.skip_compare:
        run(add_common_args([python, "scripts/compare_geometry_pca_noise.py"], args, include_max_frames=False), project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
