from __future__ import annotations

import math
from typing import Literal

import numpy as np


def attention_logits(q: np.ndarray, k: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Compute QK^T logits.

    Accepted shapes:
    - q: [heads, q_tokens, dim], k: [heads, kv_tokens, dim]
    - q: [q_tokens, heads, dim], k: [kv_tokens, heads, dim]
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must be 3D arrays")
    if q.shape[0] == k.shape[0]:
        q_hqd = q
        k_hkd = k
    elif q.shape[1] == k.shape[1]:
        q_hqd = np.transpose(q, (1, 0, 2))
        k_hkd = np.transpose(k, (1, 0, 2))
    else:
        raise ValueError(f"Cannot infer head axis from q={q.shape}, k={k.shape}")
    if scale is None:
        scale = 1.0 / math.sqrt(q_hqd.shape[-1])
    return np.einsum("hqd,hkd->hqk", q_hqd, k_hkd) * scale


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    y = np.exp(x - x_max)
    return y / np.sum(y, axis=axis, keepdims=True)


def attention_mass(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    return softmax(attention_logits(q, k), axis=-1)


def value_contribution_proxy(attn: np.ndarray, v: np.ndarray, layout: Literal["hkd", "khd"] = "khd") -> np.ndarray:
    if layout == "khd":
        v_hkd = np.transpose(v, (1, 0, 2))
    else:
        v_hkd = v
    weighted = attn[..., :, None] * v_hkd[:, None, :, :]
    return np.linalg.norm(weighted, axis=-1)


def entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)
