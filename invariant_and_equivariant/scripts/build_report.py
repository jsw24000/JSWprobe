#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ANALYSIS_METRIC_FILES = [
    ("absolute_position", "metrics.csv"),
    ("delta_decode", "metrics.csv"),
    ("forward_equivariance", "metrics.csv"),
    ("homogeneity", "metrics.csv"),
    ("subspaces", "subspace_metrics.csv"),
    ("intervention", "metrics.csv"),
    ("composition", "metrics.csv"),
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], limit: int = 40) -> str:
    if not rows:
        return "_No rows._\n"
    shown = list(rows[:limit])
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in shown:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > limit:
        lines.append(f"\n_Showing {limit} of {len(rows)} rows._")
    return "\n".join(lines) + "\n"


def metric_float(row: Mapping[str, str]) -> float:
    try:
        return float(row.get("metric_value", "nan"))
    except ValueError:
        return float("nan")


def compact_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    interesting = {
        "r2",
        "vector_mae",
        "direction_cosine",
        "forward_r2",
        "normalized_forward_error",
        "homogeneity_ratio",
        "projection_overlap",
        "mean_principal_angle_deg",
        "same_scene_cross_position_cosine",
        "composite_forward_r2",
        "composite_normalized_forward_error",
        "triplet_top1",
        "triplet_top3",
        "triplet_grid_distance",
    }
    compact: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("split_name") != "test":
            continue
        if row.get("coordinate_system") not in {"ref", ""}:
            continue
        if row.get("metric_name") not in interesting:
            continue
        compact.append(
            {
                "analysis": row.get("analysis", ""),
                "model": row.get("model_name", ""),
                "layer": row.get("layer_name", ""),
                "space": row.get("feature_space", ""),
                "metric": row.get("metric_name", ""),
                "value": metric_float(row),
                "n_test": row.get("num_test", ""),
            }
        )
    compact.sort(key=lambda r: (str(r["analysis"]), str(r["model"]), str(r["layer"]), str(r["space"]), str(r["metric"])))
    return compact


def feature_counts(run_dir: Path) -> Dict[str, Any]:
    rows = read_csv(run_dir / "features" / "feature_index.csv")
    by_model_layer = Counter()
    valid_by_model_layer = Counter()
    split_counts = Counter()
    scenes = defaultdict(set)
    cameras = set()
    states = defaultdict(set)
    for row in rows:
        key = (row["model_name"], row["layer_name"])
        by_model_layer[key] += 1
        if str(row.get("valid", "")).lower() in {"true", "1", "yes"}:
            valid_by_model_layer[key] += 1
        split_counts[row.get("split", "")] += 1
        scenes[row.get("split", "")].add(row.get("scene_id", ""))
        cameras.add(row.get("camera_id", ""))
        states[row.get("scene_id", "")].add(row.get("state_id", ""))
    return {
        "feature_index_rows": len(rows),
        "feature_rows_by_model_layer": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_model_layer.items())},
        "valid_feature_rows_by_model_layer": {f"{k[0]}::{k[1]}": v for k, v in sorted(valid_by_model_layer.items())},
        "feature_rows_by_split": dict(split_counts),
        "scenes_by_split": {split: sorted(v) for split, v in scenes.items()},
        "cameras": sorted(cameras),
        "states_per_scene": {scene: len(vals) for scene, vals in sorted(states.items())},
    }


def collect_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for analysis, rel in ANALYSIS_METRIC_FILES:
        for row in read_csv(run_dir / analysis / rel):
            item = dict(row)
            item["analysis"] = analysis
            rows.append(item)
    return rows


def plot_summary_figures(rows: Sequence[Mapping[str, Any]], run_dir: Path) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    selected_names = {
        "absolute_position": "r2",
        "delta_decode": "r2",
        "forward_equivariance": "forward_r2",
        "subspaces": "projection_overlap",
        "composition": "composite_forward_r2",
    }
    selected = []
    for row in rows:
        if row.get("split_name") != "test":
            continue
        if row.get("coordinate_system") not in {"ref", ""}:
            continue
        if row.get("feature_space") not in {"raw_standardized", ""}:
            continue
        if selected_names.get(row.get("analysis")) != row.get("metric_name"):
            continue
        if "baseline" in row.get("model_name", ""):
            continue
        selected.append(row)
    if not selected:
        return []
    labels = [f"{r['analysis']}\n{r['model_name'].replace('dinov2_official_', 'dino_').replace('vggt_1b_', 'vggt_')}\n{r['layer_name']}" for r in selected]
    values = [metric_float(r) for r in selected]
    fig_w = max(10, min(24, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(range(len(values)), values, color="#4f8db3")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("metric value")
    ax.set_title("Smoke main metrics")
    fig.tight_layout()
    path = figures / "main_smoke_metrics.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return [str(path)]


def build_report(run_dir: Path) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(run_dir)
    write_csv(tables_dir / "main_smoke_results.csv", metrics)
    compact = compact_rows(metrics)
    figure_paths = plot_summary_figures(metrics, run_dir)
    (tables_dir / "main_smoke_results.md").write_text(
        markdown_table(compact, ["analysis", "model", "layer", "space", "metric", "value", "n_test"], limit=80),
        encoding="utf-8",
    )

    audit = read_json(run_dir / "audit" / "audit_summary.json")
    inventory = read_json(run_dir / "environment" / "model_inventory.json")
    counts = feature_counts(run_dir)
    completed = {
        name: (run_dir / name).exists()
        for name, _ in ANALYSIS_METRIC_FILES
    }
    summary = {
        "run_dir": str(run_dir),
        "audit": audit,
        "feature_counts": counts,
        "completed_outputs": completed,
        "model_inventory": inventory,
        "main_results_csv": str(tables_dir / "main_smoke_results.csv"),
        "main_results_md": str(tables_dir / "main_smoke_results.md"),
        "figures": figure_paths,
        "metric_rows": len(metrics),
    }
    write_json(run_dir / "run_summary.json", summary)

    model_lines = []
    for model in inventory.get("models", []):
        model_lines.append(
            f"- `{model.get('model_name')}`: checkpoint `{model.get('checkpoint_path')}`, "
            f"patch size {model.get('patch_size')}, dim {model.get('embedding_dim')}, "
            f"layers {', '.join(model.get('layers', []))}"
        )
    if not model_lines:
        model_lines = ["- Model inventory was not available."]

    report = [
        "# Smoke Report",
        "",
        "## Data and Split",
        "",
        f"- Scenes: {audit.get('scene_count')} selected; train/validation/test scenes are listed in `run_summary.json`.",
        f"- Frames: {audit.get('frame_count')}; translation pairs: {audit.get('pair_count')}; composition triplets: {audit.get('triplet_count')}.",
        f"- Absolute-position rank in reference-camera coordinates: {audit.get('motion_rank_ref_camera')}; singular values: {audit.get('motion_singular_values_ref_camera')}.",
        f"- Pair-displacement rank: world={audit.get('displacement_rank_world')}, reference-camera={audit.get('displacement_rank_ref_camera')}.",
        f"- Cameras: {', '.join(counts.get('cameras', []))}.",
        "",
        "## Coordinate Definitions",
        "",
        "The main coordinate system is the fixed reference camera coordinate frame from the dataset manifests. Current-camera and Blender world coordinates are saved and probed as auxiliary targets.",
        "",
        "## Models and Checkpoints",
        "",
        *model_lines,
        "",
        "## Feature Extraction and Pooling",
        "",
        "Object features are area-weighted averages of patch tokens using the target binary mask resized to the patch grid. Invalid samples are excluded from probe fitting.",
        f"- Feature index rows: {counts.get('feature_index_rows')}.",
        f"- Valid rows by model/layer: `{counts.get('valid_feature_rows_by_model_layer')}`.",
        "",
        "## Main Results",
        "",
        "The full machine-readable table is `tables/main_smoke_results.csv`; a compact view is below.",
        f"Summary figure: `figures/main_smoke_metrics.png`.",
        "",
        (tables_dir / "main_smoke_results.md").read_text(encoding="utf-8"),
        "",
        "## Interpretation Boundary",
        "",
        "These smoke numbers validate the engineering loop. A successful probe means the information is linearly readable; delta decoding means displacement is present in feature differences; forward mapping means physical displacement explains some real feature change; subspace overlap is preliminary evidence that readout and motion may share directions; composition success would support primitive-to-composite generalization. This smoke run is not a proof of a strict group representation.",
        "",
        "## Full Experiment Command",
        "",
        "```bash",
        "bash scripts/run_full_pipeline.sh",
        "```",
        "",
        "## Adapter Extension",
        "",
        "Adapter training is intentionally not implemented in v1. The reserved interface is documented in `src/invariant_equivariant/adapters/README.md`.",
    ]
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
