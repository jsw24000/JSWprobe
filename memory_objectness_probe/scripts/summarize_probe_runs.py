#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_objectness_probe.utils import ensure_dir, write_json


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_key(task: str, metrics: Dict[str, Any], split: str) -> Optional[float]:
    data = metrics.get(split, {})
    if task == "presence":
        return data.get("balanced_accuracy")
    return data.get("accuracy")


def row_from_metrics(path: Path) -> Dict[str, Any]:
    metrics = read_json(path)
    run_dir = path.parent
    val = metrics.get("val", {})
    test = metrics.get("test", {})
    train = metrics.get("train", {})
    task = metrics.get("task")
    args = {}
    # metrics.json stores the resolved values we need, so no checkpoint load is required.
    return {
        "run_name": run_dir.name,
        "task": task,
        "layer": metrics.get("layer"),
        "model": metrics.get("model"),
        "feature_mode": metrics.get("feature_mode"),
        "slot_index": metrics.get("slot_index"),
        "slot_group": metrics.get("slot_group"),
        "pca_dim": metrics.get("pca_dim"),
        "input_dim": metrics.get("input_dim"),
        "num_train": metrics.get("num_train"),
        "num_val": metrics.get("num_val"),
        "num_test": metrics.get("num_test"),
        "best_epoch": metrics.get("best_epoch"),
        "best_monitor_score": metrics.get("best_monitor_score"),
        "train_accuracy": train.get("accuracy"),
        "train_macro_f1": train.get("macro_f1"),
        "train_balanced_accuracy": train.get("balanced_accuracy"),
        "val_accuracy": val.get("accuracy"),
        "val_macro_f1": val.get("macro_f1"),
        "val_balanced_accuracy": val.get("balanced_accuracy"),
        "val_auroc": val.get("auroc"),
        "val_auprc": val.get("auprc"),
        "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "test_auroc": test.get("auroc"),
        "test_auprc": test.get("auprc"),
        "val_score": score_key(task, metrics, "val"),
        "test_score": score_key(task, metrics, "test"),
        "metrics_path": str(path),
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize probe run metrics.")
    parser.add_argument("--probe_root", type=Path, default=PROJECT_ROOT / "outputs" / "probe_runs")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "probe_summaries")
    parser.add_argument(
        "--plan_file",
        type=Path,
        default=None,
        help="Optional sweep_plan.json/csv; when provided, summarize only planned run names.",
    )
    parser.add_argument("--include_smoke", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    allowed_run_names = None
    if args.plan_file is not None:
        if args.plan_file.suffix == ".csv":
            with args.plan_file.open("r", encoding="utf-8", newline="") as f:
                allowed_run_names = {row["run_name"] for row in csv.DictReader(f)}
        else:
            plan = read_json(args.plan_file)
            allowed_run_names = {str(row["run_name"]) for row in plan}

    rows: List[Dict[str, Any]] = []
    for path in sorted(args.probe_root.glob("*/metrics.json")):
        if not args.include_smoke and path.parent.name.startswith("smoke_"):
            continue
        if allowed_run_names is not None and path.parent.name not in allowed_run_names:
            continue
        rows.append(row_from_metrics(path))

    rows.sort(key=lambda r: (str(r["task"]), -(r["val_score"] or -1), -(r["test_score"] or -1), str(r["run_name"])))
    write_json(args.output_dir / "probe_summary.json", rows)
    if rows:
        with (args.output_dir / "probe_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    best_by_task: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task = str(row["task"])
        if task not in best_by_task:
            best_by_task[task] = row
    write_json(args.output_dir / "best_by_task.json", best_by_task)

    lines = ["# Probe Sweep Summary", ""]
    lines.append(f"Total runs summarized: {len(rows)}")
    lines.append("")
    for task in sorted(best_by_task):
        lines.append(f"## {task}")
        task_rows = [r for r in rows if r["task"] == task]
        lines.append("")
        lines.append("| rank | run | val score | test score | val acc | test acc | macro-F1 test |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for rank, row in enumerate(task_rows[:10], start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        str(row["run_name"]),
                        fmt(row["val_score"]),
                        fmt(row["test_score"]),
                        fmt(row["val_accuracy"]),
                        fmt(row["test_accuracy"]),
                        fmt(row["test_macro_f1"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    (args.output_dir / "probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Summarized {len(rows)} runs")
    print(f"Wrote: {args.output_dir / 'probe_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'probe_summary.md'}")


if __name__ == "__main__":
    main()
