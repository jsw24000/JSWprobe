"""PCA helpers for controlled token visualization."""

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


def _as_feature_matrix(features: Any) -> np.ndarray:
    if isinstance(features, (list, tuple)):
        arrays = [_to_numpy(item).reshape(-1, _to_numpy(item).shape[-1]) for item in features]
        return np.concatenate(arrays, axis=0)

    arr = _to_numpy(features)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr.astype(float, copy=False)


def fit_shared_pca(features: Any, n_components: int = 3) -> Any:
    """Fit a shared PCA basis on a controlled feature set.

    In later experiments, fit this shared PCA basis across the same controlled
    conditions instead of fitting a separate PCA for each image or sequence.
    """

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "fit_shared_pca requires scikit-learn. Install it before running PCA visualization."
        ) from exc

    matrix = _as_feature_matrix(features)
    pca = PCA(n_components=n_components)
    return pca.fit(matrix)


def project_pca(features: Any, pca_model: Any) -> np.ndarray:
    """Project features using an already fitted PCA model."""

    matrix = _as_feature_matrix(features)
    return pca_model.transform(matrix)


def normalize_to_01(x: Any) -> np.ndarray:
    """Normalize values to [0, 1]."""

    arr = _to_numpy(x).astype(float, copy=False)
    min_value = np.nanmin(arr)
    max_value = np.nanmax(arr)
    denom = max_value - min_value
    if denom <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_value) / denom

