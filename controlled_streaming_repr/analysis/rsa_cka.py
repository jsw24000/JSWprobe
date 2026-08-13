"""RSA and CKA helper functions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
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
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {arr.shape}.")
    return arr


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def cosine_similarity_matrix(features: Any) -> np.ndarray:
    """Compute pairwise cosine similarity between feature rows."""

    x = _l2_normalize(_as_2d(features))
    return x @ x.T


def compute_rdm(features: Any, metric: str = "cosine") -> np.ndarray:
    """Compute a representational dissimilarity matrix."""

    x = _as_2d(features)

    if metric == "cosine":
        return 1.0 - cosine_similarity_matrix(x)

    if metric == "correlation":
        centered = x - x.mean(axis=1, keepdims=True)
        return 1.0 - cosine_similarity_matrix(centered)

    if metric == "euclidean":
        diff = x[:, None, :] - x[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    raise ValueError(f"Unsupported RDM metric: {metric}")


def _center_gram(gram: np.ndarray) -> np.ndarray:
    return gram - gram.mean(axis=0, keepdims=True) - gram.mean(axis=1, keepdims=True) + gram.mean()


def linear_cka(x: Any, y: Any) -> float:
    """Compute linear CKA between two feature matrices with matched rows."""

    x_arr = _as_2d(x)
    y_arr = _as_2d(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"CKA requires matched sample counts, got {x_arr.shape[0]} and {y_arr.shape[0]}."
        )

    k = _center_gram(x_arr @ x_arr.T)
    l = _center_gram(y_arr @ y_arr.T)
    numerator = np.sum(k * l)
    denominator = np.sqrt(np.sum(k * k) * np.sum(l * l))
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)

