"""Frame-level Gram matrix analysis for controlled token bundles."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_analysis import AnalysisRun, BaseAnalysisMethod, cosine_matrix, frame_embeddings, iter_layer_token_refs
from .registry import register_analysis_method


@register_analysis_method("frame_gram")
class FrameGramAnalysis(BaseAnalysisMethod):
    """Compute frame-level cosine Gram matrices from pooled token embeddings."""

    def run(self, run: AnalysisRun) -> dict[str, Any]:
        token_types = self.config.get("token_types")
        refs = iter_layer_token_refs(run.bundle, token_types=token_types)
        if not refs:
            return self.skipped(run, "no requested token types found")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics: dict[str, Any] = {
            "method": self.method_name,
            "status": "ok",
            "model_name": run.bundle.model_name,
            "scene_id": run.manifest.get("scene_id"),
            "setting_name": run.manifest.get("setting_name"),
            "condition_id": run.manifest.get("condition_id"),
            "layers": {},
            "warnings": [],
        }
        seq_len = int(run.manifest.get("seq_len") or run.bundle.seq_len or 0) or None
        base_dir = run.bundle_path.parent

        for layer_name, token_type, ref, _layer_meta in refs:
            key = f"{layer_name}/{token_type}"
            try:
                embeddings = frame_embeddings(ref, seq_len=seq_len, base_dir=base_dir)
                gram = cosine_matrix(embeddings)
            except Exception as exc:
                metrics["warnings"].append({"layer_token": key, "warning": str(exc)})
                continue

            layer_metrics = summarize_gram(gram, run.manifest)
            layer_metrics["matrix"] = gram.tolist()
            metrics["layers"][key] = layer_metrics

            fig_path = run.figures_dir / f"{self.method_name}_{safe_name(layer_name)}_{token_type}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(5.0, 4.5))
            plt.imshow(gram, vmin=-1.0, vmax=1.0, cmap="viridis")
            plt.colorbar(label="cosine similarity")
            plt.title(f"{layer_name} {token_type}")
            plt.xlabel("frame t")
            plt.ylabel("frame t")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=int(self.figure_config.get("dpi", 180)))
            plt.close()

        if not metrics["layers"]:
            return self.skipped(run, "all token loads failed", extra={"warnings": metrics["warnings"]})
        self.save_json(run.metrics_dir / f"{self.method_name}.json", metrics)
        return metrics


def summarize_gram(gram: np.ndarray, manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute simple setting-agnostic Gram summary statistics."""

    n = gram.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = gram[mask] if n > 1 else np.array([], dtype=float)
    neighbor = np.array([gram[i, i + 1] for i in range(n - 1)], dtype=float) if n > 1 else np.array([])

    frame_ids = [frame.get("source_frame_id") for frame in manifest.get("frames", [])[:n]]
    repeated_values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if frame_ids[i] == frame_ids[j] and abs(i - j) > 1:
                repeated_values.append(float(gram[i, j]))

    return {
        "shape": list(gram.shape),
        "mean_similarity": float(np.mean(off_diag)) if off_diag.size else None,
        "min_similarity": float(np.min(off_diag)) if off_diag.size else None,
        "max_similarity": float(np.max(off_diag)) if off_diag.size else None,
        "neighbor_similarity_mean": float(np.mean(neighbor)) if neighbor.size else None,
        "far_repeated_frame_similarity_mean": float(np.mean(repeated_values)) if repeated_values else None,
    }


def safe_name(value: str) -> str:
    """Make a string safe for filenames."""

    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
