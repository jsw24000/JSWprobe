#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_objectness_probe.utils import ensure_dir, normalize_layer_name, write_json


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    mode: str
    pca_dim: str = "none"
    slot_index: Optional[int] = None
    slot_group: Optional[str] = None

    def args(self) -> List[str]:
        out = ["--feature_mode", self.mode, "--pca_dim", self.pca_dim]
        if self.slot_index is not None:
            out += ["--slot_index", str(self.slot_index)]
        if self.slot_group is not None:
            out += ["--slot_group", self.slot_group]
        return out


def default_linear_feature_specs() -> List[FeatureSpec]:
    specs = [
        FeatureSpec("all6_mean", "all6_mean", "none"),
        FeatureSpec("all6_flatten_pca64", "all6_flatten", "64"),
    ]
    specs.extend(FeatureSpec(f"slot{i}", "single_slot", "none", slot_index=i) for i in range(6))
    specs.extend(
        [
            FeatureSpec("group_camera", "slot_group", "none", slot_group="camera"),
            FeatureSpec("group_register", "slot_group", "none", slot_group="register"),
            FeatureSpec("group_scale", "slot_group", "none", slot_group="scale"),
        ]
    )
    return specs


def default_mlp_feature_specs() -> List[FeatureSpec]:
    return [
        FeatureSpec("all6_mean", "all6_mean", "none"),
        FeatureSpec("all6_flatten_pca64", "all6_flatten", "64"),
    ]


def parse_csv_arg(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def run_name(task: str, layer: str, spec: FeatureSpec, model: str) -> str:
    return f"{task}_{normalize_layer_name(layer)}_{spec.name}_{model}"


def build_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    layers = parse_csv_arg(args.layers)
    tasks = parse_csv_arg(args.tasks)
    jobs: List[Dict[str, Any]] = []
    linear_specs = default_linear_feature_specs()
    mlp_specs = default_mlp_feature_specs() if args.include_mlp_controls else []

    for task in tasks:
        for layer in layers:
            for spec in linear_specs:
                jobs.append({"task": task, "layer": layer, "spec": spec, "model": "linear"})
            for spec in mlp_specs:
                jobs.append({"task": task, "layer": layer, "spec": spec, "model": "mlp"})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a broad probe sweep with resume support.")
    parser.add_argument("--feature_manifest", type=Path, default=PROJECT_ROOT / "outputs" / "features" / "feature_manifest.jsonl")
    parser.add_argument("--split_file", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "scene_split_seed0.json")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "outputs" / "probe_runs")
    parser.add_argument("--summary_dir", type=Path, default=PROJECT_ROOT / "outputs" / "probe_summaries")
    parser.add_argument("--layers", default="block04,block11,block17,block23")
    parser.add_argument("--tasks", default="state5,presence,position4")
    parser.add_argument("--include_mlp_controls", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_jobs", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.output_root)
    ensure_dir(args.summary_dir)
    logs_dir = ensure_dir(args.summary_dir / "logs")
    jobs = build_jobs(args)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    plan_rows: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        spec: FeatureSpec = job["spec"]
        name = run_name(job["task"], job["layer"], spec, job["model"])
        out_dir = args.output_root / name
        plan_rows.append(
            {
                "index": idx,
                "run_name": name,
                "task": job["task"],
                "layer": normalize_layer_name(job["layer"]),
                "model": job["model"],
                "feature_spec": spec.name,
                "feature_mode": spec.mode,
                "pca_dim": spec.pca_dim,
                "slot_index": spec.slot_index,
                "slot_group": spec.slot_group,
                "output_dir": str(out_dir),
            }
        )
    write_json(args.summary_dir / "sweep_plan.json", plan_rows)
    with (args.summary_dir / "sweep_plan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(plan_rows[0].keys()) if plan_rows else ["index"])
        writer.writeheader()
        writer.writerows(plan_rows)

    if args.dry_run:
        print(f"Planned {len(jobs)} jobs")
        print(f"Wrote: {args.summary_dir / 'sweep_plan.csv'}")
        return

    results: List[Dict[str, Any]] = []
    started = time.time()
    for idx, job in enumerate(jobs, start=1):
        spec: FeatureSpec = job["spec"]
        name = run_name(job["task"], job["layer"], spec, job["model"])
        out_dir = args.output_root / name
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(jobs)} skip] {name}")
            status = "skipped"
            returncode = 0
        else:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_probe.py"),
                "--feature_manifest",
                str(args.feature_manifest),
                "--split_file",
                str(args.split_file),
                "--task",
                job["task"],
                "--layer",
                str(job["layer"]),
                "--model",
                job["model"],
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight_decay",
                str(args.weight_decay),
                "--max_epochs",
                str(args.max_epochs),
                "--early_stop_patience",
                str(args.early_stop_patience),
                "--hidden_dim",
                str(args.hidden_dim),
                "--dropout",
                str(args.dropout),
                "--seed",
                str(args.seed),
                "--output_dir",
                str(out_dir),
                *spec.args(),
            ]
            if args.device:
                cmd += ["--device", args.device]
            print(f"[{idx}/{len(jobs)} run] {name}")
            t0 = time.time()
            proc = subprocess.run(cmd, text=True, capture_output=True)
            elapsed = time.time() - t0
            (logs_dir / f"{name}.stdout.log").write_text(proc.stdout, encoding="utf-8")
            (logs_dir / f"{name}.stderr.log").write_text(proc.stderr, encoding="utf-8")
            status = "success" if proc.returncode == 0 else "failed"
            returncode = proc.returncode
            print(f"  -> {status} in {elapsed:.1f}s")
            if proc.returncode != 0:
                print(proc.stderr[-2000:])

        results.append(
            {
                "run_name": name,
                "task": job["task"],
                "layer": normalize_layer_name(job["layer"]),
                "model": job["model"],
                "feature_spec": spec.name,
                "feature_mode": spec.mode,
                "pca_dim": spec.pca_dim,
                "slot_index": spec.slot_index,
                "slot_group": spec.slot_group,
                "output_dir": str(out_dir),
                "metrics_path": str(metrics_path),
                "status": status,
                "returncode": returncode,
            }
        )
        write_json(args.summary_dir / "sweep_status.json", results)

    write_json(
        args.summary_dir / "sweep_status.json",
        {
            "elapsed_sec": time.time() - started,
            "num_jobs": len(jobs),
            "num_failed": sum(1 for r in results if r["status"] == "failed"),
            "runs": results,
        },
    )
    print(f"Finished {len(jobs)} jobs in {time.time() - started:.1f}s")
    print(f"Wrote: {args.summary_dir / 'sweep_status.json'}")


if __name__ == "__main__":
    main()
