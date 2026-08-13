#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.extraction_plan import write_extraction_plan
from long_seq_memory.io_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    status = write_extraction_plan(cfg)
    print(f"Wrote feature extraction plan: {status['plan']}")


if __name__ == "__main__":
    main()
