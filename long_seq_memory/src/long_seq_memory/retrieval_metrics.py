from __future__ import annotations

import numpy as np


def normalize_distribution(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    total = np.sum(x, axis=-1, keepdims=True)
    return x / np.maximum(total, eps)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(normalize_distribution(p), eps, 1.0)
    q = np.clip(normalize_distribution(q), eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = normalize_distribution(p, eps)
    q = normalize_distribution(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def source_entropy(distribution: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(normalize_distribution(distribution, eps), eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def enrichment(positive: np.ndarray, negative: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(positive) / (np.mean(negative) + eps))
