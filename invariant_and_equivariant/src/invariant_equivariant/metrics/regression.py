from __future__ import annotations

from typing import Any, Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = np.mean(err**2, axis=0)
    rmse_axis = np.sqrt(mse)
    mae_axis = np.mean(np.abs(err), axis=0)
    denom = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    numer = np.sum(err**2, axis=0)
    r2_axis = 1.0 - numer / np.maximum(denom, 1e-12)
    vector_err = np.linalg.norm(err, axis=1)
    out: Dict[str, float] = {
        f"{prefix}mae": float(np.mean(np.abs(err))),
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}vector_mae": float(np.mean(vector_err)),
        f"{prefix}vector_rmse": float(np.sqrt(np.mean(vector_err**2))),
        f"{prefix}r2": float(1.0 - np.sum(err**2) / max(1e-12, np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))),
    }
    for idx, axis in enumerate(["x", "y", "z"]):
        out[f"{prefix}mae_{axis}"] = float(mae_axis[idx])
        out[f"{prefix}rmse_{axis}"] = float(rmse_axis[idx])
        out[f"{prefix}r2_{axis}"] = float(r2_axis[idx])
    return out


def direction_metrics(delta_true: np.ndarray, delta_pred: np.ndarray) -> Dict[str, float]:
    dot = np.sum(delta_true * delta_pred, axis=1)
    norm = np.linalg.norm(delta_true, axis=1) * np.linalg.norm(delta_pred, axis=1)
    cos = dot / np.maximum(norm, 1e-12)
    length_err = np.abs(np.linalg.norm(delta_pred, axis=1) - np.linalg.norm(delta_true, axis=1))
    return {
        "direction_cosine": float(np.mean(cos)),
        "length_mae": float(np.mean(length_err)),
        "length_rmse": float(np.sqrt(np.mean(length_err**2))),
    }


def feature_forward_metrics(delta_true: np.ndarray, delta_pred: np.ndarray) -> Dict[str, float]:
    err = delta_true - delta_pred
    denom = np.sum((delta_true - delta_true.mean(axis=0, keepdims=True)) ** 2)
    r2 = 1.0 - float(np.sum(err**2) / max(1e-12, denom))
    normed = np.sum(err**2, axis=1) / np.maximum(np.sum(delta_true**2, axis=1), 1e-12)
    dot = np.sum(delta_true * delta_pred, axis=1)
    cos = dot / np.maximum(np.linalg.norm(delta_true, axis=1) * np.linalg.norm(delta_pred, axis=1), 1e-12)
    return {
        "forward_r2": float(r2),
        "normalized_forward_error": float(np.mean(normed)),
        "forward_cosine": float(np.mean(cos)),
    }

