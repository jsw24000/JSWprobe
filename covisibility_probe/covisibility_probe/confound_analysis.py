from __future__ import annotations

import numpy as np

from .metrics import pearsonr, rankdata, standardize


def _ridge_residual(y: np.ndarray, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    resid = np.full_like(y, np.nan, dtype=np.float64)
    if mask.sum() < x.shape[1] + 3:
        return resid
    xm = x[mask]
    ym = y[mask]
    xm = np.concatenate([np.ones((len(xm), 1), dtype=np.float64), xm], axis=1)
    penalty = np.eye(xm.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xm.T @ xm + penalty, xm.T @ ym)
    resid[mask] = ym - xm @ coef
    return resid


def residualized_rank_correlation(
    similarity: np.ndarray,
    overlap: np.ndarray,
    covariates: dict[str, np.ndarray],
    *,
    alpha: float = 1.0,
) -> float:
    sim = np.asarray(similarity, dtype=np.float64)
    ov = np.asarray(overlap, dtype=np.float64)
    cov_list = [np.asarray(v, dtype=np.float64) for v in covariates.values()]
    mask = np.isfinite(sim) & np.isfinite(ov)
    for cov in cov_list:
        mask &= np.isfinite(cov)
    if mask.sum() < max(6, len(cov_list) + 3):
        return float("nan")

    sim_rank = np.full_like(sim, np.nan, dtype=np.float64)
    ov_rank = np.full_like(ov, np.nan, dtype=np.float64)
    sim_rank[mask] = standardize(rankdata(sim[mask]))
    ov_rank[mask] = standardize(rankdata(ov[mask]))

    x_cols = []
    for cov in cov_list:
        ranked = np.full_like(cov, np.nan, dtype=np.float64)
        ranked[mask] = standardize(rankdata(cov[mask]))
        x_cols.append(ranked)
    x = np.stack(x_cols, axis=1) if x_cols else np.empty((len(sim), 0), dtype=np.float64)
    sim_resid = _ridge_residual(sim_rank, x, alpha=alpha)
    ov_resid = _ridge_residual(ov_rank, x, alpha=alpha)
    return pearsonr(sim_resid, ov_resid)

