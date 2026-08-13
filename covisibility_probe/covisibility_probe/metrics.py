from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    sorted_vals = arr[order]
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 0:
        return float("nan")
    return float((x @ y) / denom)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return pearsonr(rankdata(x[mask]), rankdata(y[mask]))


def standardize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mean) / std


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 5) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    mask = np.isfinite(scores) & np.isfinite(relevance)
    scores = scores[mask]
    relevance = relevance[mask]
    if len(scores) == 0:
        return float("nan")
    k = min(int(k), len(scores))
    order = np.argsort(-scores)[:k]
    ideal = np.argsort(-relevance)[:k]
    gains = relevance[order]
    ideal_gains = relevance[ideal]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))
    idcg = float(np.sum(ideal_gains * discounts))
    if idcg <= 0:
        return float("nan")
    return dcg / idcg


def centered_kernel_alignment(kx: np.ndarray, ky: np.ndarray) -> float:
    x = np.asarray(kx, dtype=np.float64)
    y = np.asarray(ky, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 2:
        return float("nan")
    hx = x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + x.mean()
    hy = y - y.mean(axis=0, keepdims=True) - y.mean(axis=1, keepdims=True) + y.mean()
    denom = np.linalg.norm(hx, "fro") * np.linalg.norm(hy, "fro")
    if denom <= 0:
        return float("nan")
    return float(np.sum(hx * hy) / denom)


def bootstrap_ci(values: Iterable[float], *, iterations: int = 1000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.empty(int(iterations), dtype=np.float64)
    for i in range(int(iterations)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pose_similarity(translations: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    zt = standardize(np.asarray(translations, dtype=np.float64))
    zr = standardize(np.asarray(rotations, dtype=np.float64))
    return -(zt + zr)


def matrix_from_descriptor(desc: np.ndarray) -> np.ndarray:
    arr = np.asarray(desc, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    z = arr / (norm + 1e-8)
    return (z @ z.T).astype(np.float32)

