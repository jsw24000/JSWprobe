#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_step(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def sanitize_run_id(run_id: str) -> str:
    keep = []
    for ch in run_id.strip():
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    cleaned = "".join(keep).strip("._-")
    if not cleaned:
        raise ValueError("run_id cannot be empty")
    return cleaned


def default_run_id(smoke: bool) -> str:
    suffix = "smoke" if smoke else "full"
    return datetime.now().strftime(f"%Y%m%d_%H%M%S_{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full revisit-memory experiment.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run prefix-through-frame-56 smoke reconstruction.")
    parser.add_argument("--analysis-only", action="store_true", help="Skip validation/reconstruction and rerun analyses.")
    parser.add_argument("--reconstruct-only", action="store_true", help="Run validation/reconstruction only.")
    parser.add_argument("--skip-depth-validation", action="store_true", help="Skip EXR depth input comparison.")
    parser.add_argument("--skip-gt-depth", action="store_true", help="Do not store GT EXR depth during reconstruction.")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Write/read one isolated run under revisit_memory/outputs/runs/<run-id>. Defaults to a timestamp for new runs.",
    )
    args = parser.parse_args()

    if args.analysis_only and not args.run_id:
        raise SystemExit("--analysis-only requires --run-id so analyses read the intended run directory.")

    run_id = sanitize_run_id(args.run_id or default_run_id(args.smoke))
    env = os.environ.copy()
    env["REVISIT_MEMORY_RUN_ID"] = run_id
    run_root = ROOT / "revisit_memory" / "outputs" / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "config": args.config,
        "smoke": args.smoke,
        "analysis_only": args.analysis_only,
        "reconstruct_only": args.reconstruct_only,
        "skip_depth_validation": args.skip_depth_validation,
        "skip_gt_depth": args.skip_gt_depth,
        "command": sys.argv,
        "python": sys.executable,
        "run_root": str(run_root),
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Run ID: {run_id}")
    print(f"Output root: {run_root}")

    py = sys.executable
    if not args.analysis_only:
        validate = [py, "revisit_memory/scripts/validate_inputs.py", "--config", args.config]
        if args.skip_depth_validation:
            validate.append("--skip-depth")
        run_step(validate, env)
        run_step([py, "revisit_memory/scripts/inspect_lingbot_modules.py", "--config", args.config], env)
        for condition in ["always_present", "always_absent", "seen_then_removed"]:
            cmd = [
                py,
                "revisit_memory/scripts/run_reconstruction.py",
                "--config",
                args.config,
                "--condition",
                condition,
            ]
            if args.smoke:
                cmd.append("--smoke")
            if args.skip_gt_depth:
                cmd.append("--skip-gt-depth")
            run_step(cmd, env)
        run_step([py, "revisit_memory/scripts/extract_features.py", "--config", args.config], env)
        run_step([py, "revisit_memory/scripts/sanity_checks.py", "--config", args.config], env)
    else:
        run_step([py, "revisit_memory/scripts/sanity_checks.py", "--config", args.config], env)

    if not args.reconstruct_only:
        if not args.skip_gt_depth:
            run_step([py, "revisit_memory/analysis/depth_analysis.py", "--config", args.config], env)
        run_step([py, "revisit_memory/analysis/pointcloud_analysis.py", "--config", args.config], env)
        run_step([py, "revisit_memory/analysis/camera_analysis.py", "--config", args.config], env)
        run_step([py, "revisit_memory/analysis/image_token_analysis.py", "--config", args.config], env)
        run_step([py, "revisit_memory/analysis/image_token_pca_vis.py", "--config", args.config], env)
        run_step([py, "revisit_memory/analysis/memory_token_analysis.py", "--config", args.config], env)
        run_step([py, "revisit_memory/analysis/make_report.py", "--config", args.config], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
