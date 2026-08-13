#!/usr/bin/env python
"""Print a compact summary for a completed run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory under outputs.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        print("status:", summary.get("status"))
        print("run_dir:", summary.get("run_dir"))
        for backend in summary.get("backend_results", []):
            print(f"- {backend['backend']}: {backend['status']} {backend.get('feature_shape') or backend.get('failure_reason')}")
    metrics_path = run_dir / "metrics_by_backend.csv"
    if metrics_path.is_file():
        df = pd.read_csv(metrics_path)
        cols = [c for c in ["bucket", "backend", "method", "num_queries", "recall_10px", "median_px_error"] if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

