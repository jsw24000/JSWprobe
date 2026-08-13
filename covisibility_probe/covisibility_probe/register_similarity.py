from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / (denom + eps)


def cosine_similarity_matrix(descriptors: np.ndarray) -> np.ndarray:
    z = l2_normalize(descriptors, axis=-1)
    return (z @ z.T).astype(np.float32)


def cosine_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    za = l2_normalize(a, axis=-1)
    zb = l2_normalize(b, axis=-1)
    return np.sum(za * zb, axis=-1).astype(np.float32)


def register_slot_similarity_matrix(registers: np.ndarray) -> np.ndarray:
    """Fixed-slot average cosine for registers with shape [T, 4, C]."""
    arr = np.asarray(registers, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 4:
        raise ValueError(f"Expected register tensor [T, 4, C], got {arr.shape}")
    z = l2_normalize(arr, axis=-1)
    sim = np.einsum("ikc,jkc->ij", z, z) / 4.0
    return sim.astype(np.float32)


def register_mean_descriptors(registers: np.ndarray) -> np.ndarray:
    arr = np.asarray(registers, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 4:
        raise ValueError(f"Expected register tensor [T, 4, C], got {arr.shape}")
    return l2_normalize(arr.mean(axis=1), axis=-1).astype(np.float32)


def register_mean_similarity_matrix(registers: np.ndarray) -> np.ndarray:
    return cosine_similarity_matrix(register_mean_descriptors(registers))


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(matrix.shape[0], k=1)
    return matrix[i, j]

