#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import load_config
from long_seq_memory.extraction_plan import write_extraction_plan
from long_seq_memory.resource_estimator import (
    estimate_resources,
    prepare_planning_outputs,
    write_resource_estimate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-large-output", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    schedule_status = prepare_planning_outputs(cfg)
    extraction_status = write_extraction_plan(cfg)
    estimate = estimate_resources(cfg)
    status = write_resource_estimate(cfg, estimate, allow_large_output=args.allow_large_output)

    report = estimate["keyframes"]["schedule_report"]
    print(f"Wrote schedule: {schedule_status['output_dir']}")
    print(f"Wrote feature extraction plan: {extraction_status['plan']}")
    print(f"Wrote resource estimate: {status['json']}")
    print(f"total_memory_writes={estimate['keyframes']['total_memory_writes']}")
    print(f"frame_{report['history_frame']}_written={report['history_frame_written']}")
    print(f"frame_{report['current_frame']}_written={report['current_frame_written']}")
    print(f"estimated_runtime_hours={estimate['runtime_estimate']['estimated_hours']:.2f}")
    print(f"estimated_kv_cache_gb={estimate['gpu_memory_estimate']['kv_cache_gb']:.2f}")
    print(f"estimated_total_output_gb={estimate['disk_estimate']['total_output_gb']:.2f}")

    if status["blocked_by_output_budget"]:
        raise SystemExit(
            "Estimated output exceeds the hard budget. "
            "Inspect resource_estimate.md or pass --allow-large-output if this is intentional."
        )


if __name__ == "__main__":
    main()
