#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", required=True)
    args = parser.parse_args()
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.exists():
        raise SystemExit(f"Analysis directory does not exist: {analysis_dir}")
    raise SystemExit("Report generation waits for completed validation, reconstruction, overlap, and retrieval outputs.")


if __name__ == "__main__":
    main()
