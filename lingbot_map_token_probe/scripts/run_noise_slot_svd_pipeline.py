#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from noise_slot_utils import resolve_config, run_subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full noise-slot SVD debias pipeline.")
    parser.add_argument("--config", default="configs/noise_slot_svd_debias.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args.config)
    project_root = Path(cfg["project_root"])
    scripts = [
        "generate_noise_slot_samples.py",
        "estimate_noise_slot_subspace.py",
        "visualize_noise_slot_basis.py",
        "visualize_correspondence_noise_debiased.py",
        "evaluate_noise_debiased_correspondence.py",
        "make_noise_slot_debias_summary.py",
    ]
    for script in scripts:
        run_subprocess([sys.executable, f"scripts/{script}", "--config", str(Path(args.config))], cwd=project_root)


if __name__ == "__main__":
    main()
