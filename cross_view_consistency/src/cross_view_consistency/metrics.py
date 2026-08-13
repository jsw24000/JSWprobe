"""Metric aggregation for cross-view matching results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _metric_row(df: pd.DataFrame, cfg: dict) -> dict:
    row = {"num_queries": int(len(df))}
    if len(df) == 0:
        return row
    for th in cfg["metrics"].get("pixel_thresholds", [4, 8, 10, 16, 32, 64]):
        row[f"recall_{int(th)}px"] = float((df["px_error"] <= float(th)).mean())
    for th in cfg["metrics"].get("token_thresholds", [1, 2, 4]):
        row[f"recall_{int(th)}tok"] = float((df["token_error"] <= float(th)).mean())
    row["mean_px_error"] = float(df["px_error"].mean())
    row["median_px_error"] = float(df["px_error"].median())
    row["mean_token_error"] = float(df["token_error"].mean())
    row["median_token_error"] = float(df["token_error"].median())
    ranks = pd.to_numeric(df["gt_rank"], errors="coerce").dropna()
    if len(ranks):
        row["mrr"] = float((1.0 / ranks).mean())
        row["median_rank"] = float(ranks.median())
        for k in cfg["metrics"].get("topk", [1, 5, 10]):
            row[f"recall_top{int(k)}"] = float((ranks <= int(k)).mean())
    else:
        row["mrr"] = np.nan
        row["median_rank"] = np.nan
        for k in cfg["metrics"].get("topk", [1, 5, 10]):
            row[f"recall_top{int(k)}"] = np.nan
    return row


def metrics_by_pair(matches: pd.DataFrame, pairs_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    if len(matches) == 0:
        return pd.DataFrame(rows)
    for keys, group in matches.groupby(["pair_id", "backend", "layer_or_stage", "method"], dropna=False):
        pair_id, backend, stage, method = keys
        pair = pairs_df[pairs_df["pair_id"] == pair_id].iloc[0]
        row = {
            "pair_id": pair_id,
            "scene_id": pair["scene_id"],
            "bucket": pair["bucket"],
            "src_frame": int(pair["src_frame"]),
            "tgt_frame": int(pair["tgt_frame"]),
            "backend": backend,
            "layer_or_stage": stage,
            "method": method,
        }
        row.update(_metric_row(group, cfg))
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_backend(matches: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    if len(matches) == 0:
        return pd.DataFrame(rows)
    group_cols = ["bucket", "backend", "layer_or_stage", "method"]
    for keys, group in matches.groupby(group_cols, dropna=False):
        bucket, backend, stage, method = keys
        row = {
            "scene_id": cfg["paths"].get("scene_id", ""),
            "bucket": bucket,
            "backend": backend,
            "layer_or_stage": stage,
            "method": method,
            "num_pairs": int(group["pair_id"].nunique()),
        }
        row.update(_metric_row(group, cfg))
        rows.append(row)
    return pd.DataFrame(rows)

