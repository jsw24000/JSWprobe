#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import load_config, write_json
from long_seq_memory.lingbot_adapter import run_streaming_reconstruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = run_streaming_reconstruction(cfg)
    write_json(Path(cfg["output_dir"]) / "resolved_config.json", cfg)
    print(f"Wrote run manifest: {Path(cfg['output_dir']) / 'run_manifest.json'}")
    print(f"Frames: {manifest['num_frames']}")


if __name__ == "__main__":
    main()
