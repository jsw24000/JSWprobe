#!/usr/bin/env python3
"""Minimal analysis template using random features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from analysis.gram_matrix import compute_gram_matrix
from analysis.io_utils import save_json
from analysis.pca_vis import fit_shared_pca, project_pca


def main() -> int:
    rng = np.random.default_rng(seed=42)
    features = rng.normal(size=(8, 16))

    gram = compute_gram_matrix(features, normalize=True)
    try:
        pca = fit_shared_pca(features, n_components=3)
        projection = project_pca(features, pca)
    except ImportError as exc:
        print(f"PCA 依赖缺失，未生成模板分析 JSON: {exc}")
        return 1

    metrics = {
        "status": "template",
        "feature_shape": list(features.shape),
        "gram_shape": list(gram.shape),
        "pca_projection_shape": list(projection.shape),
        "gram_matrix": np.round(gram, 6).tolist(),
        "pca_projection": np.round(projection, 6).tolist(),
        "explained_variance_ratio": np.round(pca.explained_variance_ratio_, 6).tolist(),
    }

    output_path = PROJECT_DIR / "outputs" / "metrics" / "template_metrics.json"
    save_json(output_path, metrics)
    print(f"已保存模板分析 JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
