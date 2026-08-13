"""Cosine nearest-neighbor matching and simple baselines."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .geometry import GridSpec, target_valid_token_mask, token_centers, token_index_from_xy, token_xy_from_index


def _center_for_token(x: np.ndarray, y: np.ndarray, grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    u = (np.asarray(x, dtype=np.float64) + 0.5) * grid.patch_size
    v = (np.asarray(y, dtype=np.float64) + 0.5) * grid.patch_size
    return u, v


def _rank_of_gt(sim: torch.Tensor, gt_idx: torch.Tensor) -> torch.Tensor:
    gt_sim = sim.gather(1, gt_idx[:, None])
    return (sim > gt_sim).sum(dim=1) + 1


def match_feature_result(
    feature_result,
    pairs_df: pd.DataFrame,
    query_sets: dict[str, pd.DataFrame],
    frames,
    grid: GridSpec,
    cfg: dict,
) -> pd.DataFrame:
    if feature_result.features is None:
        return pd.DataFrame()
    features = feature_result.features.float()
    if cfg.get("matching", {}).get("l2_normalize", True):
        features = torch.nn.functional.normalize(features, dim=-1)
    frame_to_local = {fid: i for i, fid in enumerate(feature_result.frame_ids)}
    rows = []
    methods = cfg.get("matching", {}).get("methods", ["random", "same_coord", "feature_nn"])
    topk = sorted(set(int(k) for k in cfg.get("metrics", {}).get("topk", [1, 5, 10])))
    max_topk = max(topk)
    rng = np.random.default_rng(int(cfg.get("run", {}).get("seed", 42)))
    match_only_valid = bool(cfg.get("matching", {}).get("match_only_valid_target_depth", True))

    for pair in pairs_df.itertuples(index=False):
        qdf = query_sets[pair.pair_id]
        if len(qdf) == 0:
            continue
        src_local = frame_to_local[int(pair.src_frame)]
        tgt_local = frame_to_local[int(pair.tgt_frame)]
        src_feat_map = features[src_local].reshape(-1, features.shape[-1])
        tgt_feat_map = features[tgt_local].reshape(-1, features.shape[-1])
        q_src_idx = torch.tensor(qdf["src_token_idx"].to_numpy(), dtype=torch.long)
        gt_idx = torch.tensor(qdf["gt_tgt_token_idx"].to_numpy(), dtype=torch.long)
        sim = src_feat_map[q_src_idx] @ tgt_feat_map.T

        valid_mask_np = np.ones(tgt_feat_map.shape[0], dtype=bool)
        if match_only_valid:
            valid_mask_np = target_valid_token_mask(frames[int(pair.tgt_index)], grid, cfg["geometry"])
            invalid = torch.tensor(~valid_mask_np, dtype=torch.bool)
            sim[:, invalid] = -torch.inf
        valid_indices = np.where(valid_mask_np)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(tgt_feat_map.shape[0])

        feature_top = torch.topk(sim, k=min(max_topk, sim.shape[1]), dim=1)
        feature_pred = feature_top.indices[:, 0].cpu().numpy()
        feature_sim = feature_top.values[:, 0].cpu().numpy()
        feature_rank = _rank_of_gt(sim, gt_idx).cpu().numpy()
        method_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if "feature_nn" in methods:
            method_preds["feature_nn"] = (feature_pred, feature_sim, feature_rank)
        if "random" in methods:
            random_pred = rng.choice(valid_indices, size=len(qdf), replace=True)
            method_preds["random"] = (random_pred, np.full(len(qdf), np.nan), np.full(len(qdf), np.nan))
        if "same_coord" in methods:
            sx = qdf["src_token_x"].to_numpy(dtype=np.int64)
            sy = qdf["src_token_y"].to_numpy(dtype=np.int64)
            same_pred = token_index_from_xy(sx, sy, grid.grid_hw).astype(np.int64)
            method_preds["same_coord"] = (same_pred, np.full(len(qdf), np.nan), np.full(len(qdf), np.nan))

        for method, (pred_idx, pred_sim, rank) in method_preds.items():
            pred_x, pred_y = token_xy_from_index(pred_idx, grid.grid_hw)
            pred_u, pred_v = _center_for_token(pred_x, pred_y, grid)
            gt_u = qdf["gt_tgt_u"].to_numpy(dtype=np.float64)
            gt_v = qdf["gt_tgt_v"].to_numpy(dtype=np.float64)
            gt_x = qdf["gt_tgt_token_x"].to_numpy(dtype=np.float64)
            gt_y = qdf["gt_tgt_token_y"].to_numpy(dtype=np.float64)
            px_error = np.sqrt((pred_u - gt_u) ** 2 + (pred_v - gt_v) ** 2)
            tok_error = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            for i, qrow in enumerate(qdf.itertuples(index=False)):
                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "backend": feature_result.backend_name,
                        "layer_or_stage": feature_result.layer_or_stage,
                        "method": method,
                        "query_id": qrow.query_id,
                        "src_frame": int(pair.src_frame),
                        "tgt_frame": int(pair.tgt_frame),
                        "bucket": pair.bucket,
                        "src_token_x": int(qrow.src_token_x),
                        "src_token_y": int(qrow.src_token_y),
                        "src_u": float(qrow.src_u),
                        "src_v": float(qrow.src_v),
                        "gt_tgt_u": float(qrow.gt_tgt_u),
                        "gt_tgt_v": float(qrow.gt_tgt_v),
                        "gt_tgt_token_x": int(qrow.gt_tgt_token_x),
                        "gt_tgt_token_y": int(qrow.gt_tgt_token_y),
                        "pred_tgt_u": float(pred_u[i]),
                        "pred_tgt_v": float(pred_v[i]),
                        "pred_tgt_token_x": int(pred_x[i]),
                        "pred_tgt_token_y": int(pred_y[i]),
                        "px_error": float(px_error[i]),
                        "token_error": float(tok_error[i]),
                        "similarity": float(pred_sim[i]) if not math.isnan(float(pred_sim[i])) else np.nan,
                        "gt_rank": float(rank[i]) if not math.isnan(float(rank[i])) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def similarity_map_for_query(feature_result, src_frame: int, tgt_frame: int, src_token_idx: int) -> np.ndarray:
    features = torch.nn.functional.normalize(feature_result.features.float(), dim=-1)
    frame_to_local = {fid: i for i, fid in enumerate(feature_result.frame_ids)}
    src = features[frame_to_local[src_frame]].reshape(-1, features.shape[-1])
    tgt = features[frame_to_local[tgt_frame]].reshape(-1, features.shape[-1])
    sim = src[int(src_token_idx)] @ tgt.T
    return sim.reshape(feature_result.grid_hw).detach().cpu().numpy()

