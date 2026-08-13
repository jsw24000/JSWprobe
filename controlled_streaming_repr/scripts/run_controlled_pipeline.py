#!/usr/bin/env python3
"""Convenience wrapper for manifest build, extraction, and analysis steps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from controlled_common import PROJECT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled ScanNet pipeline.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "controlled_scannet_v1.yaml")
    parser.add_argument("--analysis-config", type=Path, default=PROJECT_DIR / "configs" / "analysis_controlled_v1.yaml")
    parser.add_argument("--manifest-index", type=Path, default=PROJECT_DIR / "outputs" / "manifests" / "controlled_scannet_v1" / "index.json")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--setting", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--overwrite-manifests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_build:
        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "build_sequence_manifests.py"),
            "--config",
            str(args.config),
        ]
        if args.limit_scenes is not None:
            command.extend(["--limit-scenes", str(args.limit_scenes)])
        if args.seq_len is not None:
            command.extend(["--seq-lens", str(args.seq_len)])
        if args.dry_run:
            command.append("--dry-run")
        if args.overwrite_manifests:
            command.append("--overwrite")
        run(command)

    if not args.skip_extraction:
        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "run_controlled_extraction.py"),
            "--config",
            str(args.config),
            "--manifest-index",
            str(args.manifest_index),
        ]
        add_filters(command, args)
        if args.models:
            command.extend(["--models", *args.models])
        if args.dry_run:
            command.append("--dry-run")
        run(command)

    if not args.skip_analysis and not args.dry_run:
        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "run_controlled_analysis.py"),
            "--experiment-config",
            str(args.config),
            "--analysis-config",
            str(args.analysis_config),
            "--manifest-index",
            str(args.manifest_index),
        ]
        add_filters(command, args)
        if args.models:
            command.extend(["--models", *args.models])
        if args.methods:
            command.extend(["--methods", *args.methods])
        run(command)
    return 0


def add_filters(command: list[str], args: argparse.Namespace) -> None:
    if args.setting:
        command.extend(["--setting", args.setting])
    if args.scene:
        command.extend(["--scene", args.scene])
    if args.condition_id:
        command.extend(["--condition-id", args.condition_id])
    if args.seq_len is not None:
        command.extend(["--seq-len", str(args.seq_len)])
    if args.max_runs is not None:
        command.extend(["--max-runs", str(args.max_runs)])


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
