"""Token displacement helpers for comparing controlled conditions."""

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


def compute_token_displacement(tokens_a: Any, tokens_b: Any, metric: str = "cosine") -> np.ndarray:
    """Compare corresponding tokens from two conditions.

    The last dimension is treated as the feature dimension. The returned array
    has the same leading dimensions as the token inputs.
    """

    a = _to_numpy(tokens_a).astype(float, copy=False)
    b = _to_numpy(tokens_b).astype(float, copy=False)
    if a.shape != b.shape:
        raise ValueError(f"Token shapes must match, got {a.shape} and {b.shape}.")

    if metric == "cosine":
        a_norm = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
        b_norm = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
        return 1.0 - np.sum(a_norm * b_norm, axis=-1)

    if metric in {"l2", "euclidean"}:
        return np.linalg.norm(a - b, axis=-1)

    raise ValueError(f"Unsupported displacement metric: {metric}")


def summarize_displacement(displacement: Any) -> dict[str, float | int]:
    """Summarize a displacement map with basic statistics."""

    values = _to_numpy(displacement).astype(float, copy=False)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }

