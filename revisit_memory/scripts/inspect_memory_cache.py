#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.io import load_config, output_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect generated memory-transplant cache layout metadata.")
    parser.add_argument("--config", default="revisit_memory/configs/memory_transplant_v1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = output_root(cfg)
    layout_path = root / "metadata" / "memory_layout.json"
    if not layout_path.exists():
        raise SystemExit(f"Missing {layout_path}. Run scripts/run_memory_transplant.py first.")
    with layout_path.open("r", encoding="utf-8") as f:
        layout = json.load(f)
    example = layout["layout_example"]["A"]
    print(f"Run root: {root}")
    print(f"Target frame: {example['target_frame']}")
    print(f"Local/anchor frames: {example['local_anchor_source_frame_ids_before_target']}")
    print(f"Trajectory special frames: {example['special_source_frame_ids_before_target']}")
    first_block = example["global_blocks"][0]
    print(f"Block 0 local cache shape: {first_block['local_anchor_cache']['shape']}")
    print(f"Block 0 special cache shape: {first_block['trajectory_special_cache']['shape']}")
    print(f"Special token order: {layout['token_names']}")
    print(f"Target-visible source frames: {layout['source_target_visible_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
