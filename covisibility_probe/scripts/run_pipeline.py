#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from covisibility_probe.config import load_config, save_config
from covisibility_probe.paths import project_root, resolve_run_dir
from covisibility_probe.utils import ensure_dir, setup_logging


STAGES = [
    ("inspect_environment", "scripts/inspect_environment.py"),
    ("build_manifest", "scripts/build_manifest.py"),
    ("build_overlap_gt", "scripts/build_overlap_gt.py"),
    ("extract_features", "scripts/extract_features.py"),
    ("analyze_training_free", "scripts/analyze_training_free.py"),
    ("make_visualizations", "scripts/make_visualizations.py"),
]


def _stage_range(start: str | None, end: str | None) -> list[tuple[str, str]]:
    names = [s[0] for s in STAGES]
    start_idx = names.index(start) if start else 0
    end_idx = names.index(end) if end else len(STAGES) - 1
    if start_idx > end_idx:
        raise ValueError(f"start-stage {start} is after end-stage {end}")
    return STAGES[start_idx : end_idx + 1]


def run_pipeline(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    start_stage: str | None = None,
    end_stage: str | None = None,
    overwrite: bool = False,
) -> Path:
    cfg = load_config(config_path)
    run_dir = resolve_run_dir(cfg, run_id=run_id)
    ensure_dir(run_dir / "metadata")
    save_config(cfg, run_dir / "metadata" / "resolved_config.yaml")
    actual_run_id = cfg["run"]["run_id"]

    for stage_name, script_rel in _stage_range(start_stage, end_stage):
        script = project_root() / script_rel
        cmd = [sys.executable, str(script)]
        if stage_name != "inspect_environment":
            cmd.extend(["--config", str(config_path), "--run-id", actual_run_id])
            if overwrite:
                cmd.append("--overwrite")
        print(f"=== Running stage: {stage_name} ===")
        subprocess.run(cmd, check=True, cwd=str(project_root()))
        if stage_name == "inspect_environment":
            src = project_root() / "outputs" / "environment_report.json"
            if src.is_file():
                shutil.copy2(src, run_dir / "metadata" / "environment_report.json")

    print(f"Pipeline run directory: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--start-stage", choices=[x[0] for x in STAGES], default=None)
    parser.add_argument("--end-stage", choices=[x[0] for x in STAGES], default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging()
    run_pipeline(
        args.config,
        run_id=args.run_id,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

