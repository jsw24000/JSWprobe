from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass
class RidgeModel:
    weights: np.ndarray
    bias: np.ndarray
    alpha: float
    x_standardizer: Standardizer | None

    def predict(self, x: np.ndarray) -> np.ndarray:
        xt = self.x_standardizer.transform(x) if self.x_standardizer is not None else x
        return xt @ self.weights + self.bias


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, standardize_x: bool = True, fit_intercept: bool = True) -> RidgeModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    standardizer = Standardizer.fit(x) if standardize_x else None
    xs = standardizer.transform(x) if standardizer is not None else x.copy()
    if fit_intercept:
        y_mean = y.mean(axis=0, keepdims=True)
        yc = y - y_mean
    else:
        y_mean = np.zeros((1, y.shape[1]), dtype=np.float64)
        yc = y
    dim = xs.shape[1]
    n_samples = xs.shape[0]
    if n_samples < dim:
        gram = xs @ xs.T + float(alpha) * np.eye(n_samples, dtype=np.float64)
        dual = np.linalg.solve(gram, yc)
        weights = xs.T @ dual
    else:
        gram = xs.T @ xs + float(alpha) * np.eye(dim, dtype=np.float64)
        weights = np.linalg.solve(gram, xs.T @ yc)
    bias = y_mean.reshape(-1)
    return RidgeModel(weights=weights, bias=bias, alpha=float(alpha), x_standardizer=standardizer)


def choose_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Sequence[float],
    standardize_x: bool = True,
    fit_intercept: bool = True,
) -> RidgeModel:
    best_model = None
    best_score = float("inf")
    for alpha in alphas:
        model = fit_ridge(x_train, y_train, alpha=float(alpha), standardize_x=standardize_x, fit_intercept=fit_intercept)
        pred = model.predict(x_val)
        score = float(np.mean((pred - y_val) ** 2))
        if score < best_score:
            best_score = score
            best_model = model
    assert best_model is not None
    return best_model


def model_to_npz(path, model: RidgeModel, extra: Mapping[str, Any] | None = None) -> None:
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": model.weights,
        "bias": model.bias,
        "alpha": np.asarray([model.alpha], dtype=np.float64),
        "metadata_json": np.asarray([json.dumps(dict(extra or {}), sort_keys=True)]),
    }
    if model.x_standardizer is not None:
        payload["x_mean"] = model.x_standardizer.mean
        payload["x_std"] = model.x_standardizer.std
    np.savez(path, **payload)
