"""Attention summary analysis for streaming memory/GCA outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_analysis import AnalysisRun, BaseAnalysisMethod, to_numpy
from .registry import register_analysis_method


@register_analysis_method("attention_mass")
class AttentionMassAnalysis(BaseAnalysisMethod):
    """Analyze saved attention summaries without requiring raw attention dumps."""

    def run(self, run: AnalysisRun) -> dict[str, Any]:
        attention = run.bundle.attention or run.bundle.metadata.get("attention") or {}
        if not attention:
            return self.skipped(run, "attention summary is missing")

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

        summary = load_attention_summary(attention, run.bundle_path.parent)
        if not summary:
            return self.skipped(run, "attention object exists but no numeric summary was found")

        for layer_name, layer_summary in sorted(summary.items()):
            metrics["layers"][layer_name] = summarize_attention(layer_summary)
            plot_attention_curves(
                plt,
                layer_summary,
                run.figures_dir / f"{self.method_name}_{safe_name(layer_name)}.png",
                dpi=int(self.figure_config.get("dpi", 180)),
            )

        self.save_json(run.metrics_dir / f"{self.method_name}.json", metrics)
        return metrics


def load_attention_summary(attention: dict[str, Any], base_dir: Any) -> dict[str, Any]:
    """Load attention summary dictionaries or arrays referenced by path."""

    if "summary" in attention and isinstance(attention["summary"], dict):
        return attention["summary"]
    if "layers" in attention and isinstance(attention["layers"], dict):
        return attention["layers"]
    result: dict[str, Any] = {}
    for key, value in attention.items():
        if isinstance(value, dict):
            result[key] = value
        elif isinstance(value, (str, bytes)):
            try:
                result[key] = to_numpy(value, base_dir=base_dir)
            except Exception:
                continue
    return result


def summarize_attention(layer_summary: Any) -> dict[str, Any]:
    """Summarize known attention category curves or source-frame masses."""

    if isinstance(layer_summary, np.ndarray):
        return summarize_array(layer_summary)
    if not isinstance(layer_summary, dict):
        return {"available": False, "reason": f"unsupported type {type(layer_summary)!r}"}

    output: dict[str, Any] = {"available": True, "fields": {}}
    for key, value in sorted(layer_summary.items()):
        if key in {"topk_source_frames", "topk"}:
            output[key] = value
            continue
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            continue
        if arr.size == 0:
            continue
        output["fields"][key] = summarize_array(arr)
    return output


def summarize_array(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=float)
    return {
        "shape": list(arr.shape),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def plot_attention_curves(plt: Any, layer_summary: Any, path: Any, *, dpi: int) -> None:
    """Plot attention category curves when numeric per-frame fields exist."""

    if not isinstance(layer_summary, dict):
        return
    numeric_fields: dict[str, np.ndarray] = {}
    for key, value in layer_summary.items():
        if key.startswith("mass_") or key in {"attention_entropy", "entropy"}:
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
            except Exception:
                continue
            numeric_fields[key] = arr
    if not numeric_fields:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4, 3.4))
    for key, arr in numeric_fields.items():
        plt.plot(arr, marker="o", linewidth=1.2, label=key)
    plt.xlabel("frame t")
    plt.ylabel("attention summary")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
