#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.dataset import load_scene
from long_seq_memory.io_utils import ensure_dir, load_config, write_json
from long_seq_memory.overlap import pose_candidate_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "configs" / "dataset_keble02.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    scene = load_scene(cfg["dataset_root"])
    i = int(cfg["loop_event"]["history_frame"])
    j = int(cfg["loop_event"]["current_frame"])
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "overlap")
    candidate = pose_candidate_score(scene.poses_c2w, i, j)
    write_json(
        out_dir / "loop_events.json",
        {
            "index_base": 0,
            "status": "pose_candidate_only",
            "geometric_overlap": None,
            "geometric_overlap_note": "TLS/LiDAR z-buffer overlap is not implemented yet.",
            "candidate": candidate.__dict__,
        },
    )
    print(f"Wrote {out_dir / 'loop_events.json'}")


if __name__ == "__main__":
    main()
