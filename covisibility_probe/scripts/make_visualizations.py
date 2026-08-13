#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd

from covisibility_probe.config import load_config
from covisibility_probe.paths import resolve_run_dir
from covisibility_probe.utils import read_parquet, save_json, setup_logging
from covisibility_probe.visualization import (
    save_layer_trends,
    save_retrieval_cases,
    save_similarity_hexbin,
    save_time_overlap_heatmap,
)


def make_visualizations(config_path: str | Path, *, run_id: str | None = None, overwrite: bool = False) -> Path:
    cfg = load_config(config_path)
    run_dir = resolve_run_dir(cfg, run_id=run_id)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.get("visualization", {}).get("enabled", True):
        print("Visualization disabled in config.")
        return run_dir

    summary = pd.read_csv(run_dir / "metrics" / "summary_metrics.csv")
    pairs = read_parquet(run_dir / "pairs" / "all_pairs.parquet")
    manifest = read_parquet(run_dir / "manifests" / "frames.parquet")
    grid = pd.read_csv(run_dir / "metrics" / "time_overlap_grid.csv")
    rep_layer = int(cfg.get("analysis", {}).get("representative_layer", 17))
    rep_col = f"register_head_concat_slot_similarity_l{rep_layer:02d}"

    outputs = []
    outputs.extend(str(p) for p in save_layer_trends(summary, fig_dir))
    outputs.append(str(save_time_overlap_heatmap(grid, fig_dir / f"time_overlap_heatmap_l{rep_layer:02d}.png")))
    if rep_col in pairs:
        outputs.append(str(save_similarity_hexbin(pairs, rep_col, fig_dir / f"similarity_overlap_hexbin_l{rep_layer:02d}.png")))
        retrieval = save_retrieval_cases(
            pairs,
            manifest,
            similarity_column=rep_col,
            out_dir=fig_dir / "retrieval_cases",
            max_cases=int(cfg.get("visualization", {}).get("max_retrieval_cases", 6)),
        )
        outputs.extend(str(p) for p in retrieval)
    save_json({"figures": outputs, "representative_layer": rep_layer, "representative_column": rep_col}, fig_dir / "figure_manifest.json")
    print(f"Figures written under: {fig_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging()
    make_visualizations(args.config, run_id=args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
