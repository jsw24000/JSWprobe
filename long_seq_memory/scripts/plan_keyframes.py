#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import load_config
from long_seq_memory.schedule import (
    build_keyframe_schedule,
    schedule_config_from_dict,
    schedule_report,
    write_schedule_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    scfg = schedule_config_from_dict(cfg)
    rows = build_keyframe_schedule(scfg)
    loop = cfg.get("loop_event", {})
    report = schedule_report(
        rows,
        scfg,
        history_frame=int(loop.get("history_frame", 3403)),
        current_frame=int(loop.get("current_frame", 4414)),
        history_segment=tuple(loop.get("history_segment", [3370, 3440])),
    )
    status = write_schedule_outputs(cfg["output_dir"], cfg.get("run_name", Path(args.config).stem), rows, report)
    print(f"Wrote planning outputs under {status['output_dir']}")
    print(f"history_frame_written={report['history_frame_written']}")
    print(f"current_frame_written={report['current_frame_written']}")
    print(f"total_memory_writes={report['total_memory_writes']}")
    print(f"nearest_retained_history={report['nearest_retained_history_to_history_frame_at_current']}")
    print(f"history_segment_writes={report['written_keyframes_in_history_segment']}")


if __name__ == "__main__":
    main()
