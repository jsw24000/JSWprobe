"""Gram and affinity matrix utilities."""

from __future__ import annotations

import numpy as np
from typing import Any


def _to_numpy(x: Any) -> np.ndarray:
    """Convert numpy arrays or torch-like tensors to numpy."""

    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _as_2d(x: Any) -> np.ndarray:
    arr = _to_numpy(x).astype(float, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {arr.shape}.")
    return arr


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def compute_gram_matrix(x: Any, normalize: bool = True) -> np.ndarray:
    """Compute X X^T, optionally after row-wise L2 normalization."""

    features = _as_2d(x)
    if normalize:
        features = _l2_normalize(features)
    return features @ features.T


def compute_cosine_affinity(x: Any, eps: float = 1e-8) -> np.ndarray:
    """Compute pairwise cosine affinity between rows of a feature matrix."""

    features = _as_2d(x)
    features = _l2_normalize(features, eps=eps)
    return features @ features.T

