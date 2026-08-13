#!/usr/bin/env python
"""CLI entry point for the cross-view consistency probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from cross_view_consistency.run import run_probe

    parser = argparse.ArgumentParser(description="Run cross-view feature consistency probe.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--tag", default=None, help="Override run.tag.")
    parser.add_argument("--num-frames", type=int, default=None, help="Override frame_sampling.num_frames.")
    parser.add_argument("--pairs-per-bucket", type=int, default=None, help="Override pairs.pairs_per_bucket.")
    parser.add_argument("--queries-per-pair", type=int, default=None, help="Override queries.queries_per_pair.")
    parser.add_argument("--backends", default=None, help="Comma-separated backend list.")
    args = parser.parse_args()

    run_dir = run_probe(
        args.config,
        tag=args.tag,
        num_frames=args.num_frames,
        pairs_per_bucket=args.pairs_per_bucket,
        queries_per_pair=args.queries_per_pair,
        backends=args.backends,
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()

