#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import load_config, write_json
from long_seq_memory.extraction_plan import write_extraction_plan
from long_seq_memory.lingbot_adapter import run_streaming_reconstruction
from long_seq_memory.resource_estimator import (
    estimate_resources,
    prepare_planning_outputs,
    write_resource_estimate,
)
from long_seq_memory.schedule import build_keyframe_schedule, schedule_config_from_dict


def apply_frame_overrides(cfg: dict, start_frame: int | None, end_frame: int | None) -> dict:
    cfg = dict(cfg)
    dataset = dict(cfg.get("dataset", {}))
    original_start = int(dataset.get("start_frame", cfg.get("start_frame", 0)))
    original_end = int(dataset.get("end_frame", cfg.get("end_frame", cfg.get("sequence", {}).get("expected_frames", 0))))
    if start_frame is not None:
        dataset["start_frame"] = int(start_frame)
        cfg["start_frame"] = int(start_frame)
    if end_frame is not None:
        dataset["end_frame"] = int(end_frame)
        cfg["end_frame"] = int(end_frame)
    cfg["dataset"] = dataset
    if start_frame is not None or end_frame is not None:
        start = int(dataset.get("start_frame", original_start))
        end = int(dataset.get("end_frame", original_end))
        recon_dir = Path(cfg.get("reconstruction_output_dir", cfg["output_dir"])).expanduser()
        cfg["reconstruction_output_dir"] = str(recon_dir.parent / f"{recon_dir.name}_{start}_{end}")
    return cfg


def num_config_frames(cfg: dict) -> int:
    rows = build_keyframe_schedule(schedule_config_from_dict(cfg))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--resource-profile-only", action="store_true")
    parser.add_argument("--allow-large-output", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    args = parser.parse_args()

    cfg = apply_frame_overrides(load_config(args.config), args.start_frame, args.end_frame)
    if args.resume:
        raise SystemExit(
            "Resume is disabled for this experiment until a real KV checkpoint writer/restorer exists."
        )

    has_frame_override = args.start_frame is not None or args.end_frame is not None
    if not has_frame_override:
        prepare_planning_outputs(cfg)
        write_extraction_plan(cfg)
    estimate = estimate_resources(cfg)
    prefix = "resource_estimate"
    if has_frame_override:
        start = cfg.get("dataset", {}).get("start_frame", cfg.get("start_frame", 0))
        end = cfg.get("dataset", {}).get("end_frame", cfg.get("end_frame"))
        prefix = f"resource_profile_{start}_{end}"
    estimate_status = write_resource_estimate(
        cfg,
        estimate,
        allow_large_output=args.allow_large_output,
        filename_prefix=prefix,
    )
    if estimate_status["blocked_by_output_budget"]:
        raise SystemExit(
            "Resource estimate exceeds the hard output budget. "
            "Inspect resource_estimate.md or pass --allow-large-output."
        )

    if args.resource_profile_only:
        print(f"Wrote resource profile: {estimate_status['json']}")
        print(f"frames={num_config_frames(cfg)}")
        print(f"estimated_runtime_hours={estimate['runtime_estimate']['estimated_hours']:.2f}")
        return

    frames = num_config_frames(cfg)
    if frames > 256 and not args.confirm_full_run:
        raise SystemExit(
            f"Refusing to start {frames} frames without --confirm-full-run. "
            "Planning and resource estimates have been written."
        )

    manifest = run_streaming_reconstruction(cfg)
    recon_dir = Path(cfg.get("reconstruction_output_dir", cfg["output_dir"])).expanduser()
    write_json(recon_dir / "resolved_config.json", cfg)
    print(f"Wrote run manifest: {recon_dir / 'run_manifest.json'}")
    print(f"Frames: {manifest['num_frames']}")


if __name__ == "__main__":
    main()
