"""Geometry-output stability analysis for controlled sequences."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_analysis import AnalysisRun, BaseAnalysisMethod, to_numpy
from .registry import register_analysis_method


@register_analysis_method("geometry_stability")
class GeometryStabilityAnalysis(BaseAnalysisMethod):
    """Compute lightweight metrics for predicted camera/depth/pointmap outputs."""

    def run(self, run: AnalysisRun) -> dict[str, Any]:
        geometry = run.bundle.geometry_outputs or run.bundle.output_predictions or {}
        if not geometry:
            return self.skipped(run, "geometry outputs are missing")

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
            "geometry": {},
            "warnings": [],
        }
        base_dir = run.bundle_path.parent

        camera_ref = first_present(geometry, ["predicted_camera", "predicted_cameras", "camera", "extrinsic"])
        if camera_ref is not None:
            try:
                camera = to_numpy(camera_ref, base_dir=base_dir)
                translations = camera_translations(camera)
                metrics["geometry"]["predicted_camera"] = {
                    "shape": list(camera.shape),
                    "translation_curve": translations.tolist(),
                    "translation_step_norm": np.linalg.norm(np.diff(translations, axis=0), axis=1).tolist()
                    if translations.shape[0] > 1
                    else [],
                }
                plot_translation_curve(
                    plt,
                    translations,
                    run.figures_dir / "camera_translation_curve.png",
                    dpi=int(self.figure_config.get("dpi", 180)),
                )
            except Exception as exc:
                metrics["warnings"].append({"field": "predicted_camera", "warning": str(exc)})

        depth_ref = first_present(geometry, ["predicted_depth", "depth", "depth_map"])
        if depth_ref is not None:
            try:
                depth = to_numpy(depth_ref, base_dir=base_dir).astype(float, copy=False)
                variance = np.nanvar(depth, axis=0) if depth.ndim >= 3 else np.asarray([])
                metrics["geometry"]["predicted_depth"] = {
                    "shape": list(depth.shape),
                    "variance_mean": float(np.nanmean(variance)) if variance.size else None,
                    "variance_max": float(np.nanmax(variance)) if variance.size else None,
                }
                if variance.size:
                    save_heatmap(
                        plt,
                        variance.squeeze(),
                        run.figures_dir / "depth_variance_map.png",
                        title="Depth variance",
                        dpi=int(self.figure_config.get("dpi", 180)),
                    )
            except Exception as exc:
                metrics["warnings"].append({"field": "predicted_depth", "warning": str(exc)})

        pointmap_ref = first_present(geometry, ["predicted_pointmap", "pointmap", "points"])
        if pointmap_ref is not None:
            try:
                pointmap = to_numpy(pointmap_ref, base_dir=base_dir).astype(float, copy=False)
                variance = np.nanvar(pointmap, axis=0) if pointmap.ndim >= 3 else np.asarray([])
                metrics["geometry"]["predicted_pointmap"] = {
                    "shape": list(pointmap.shape),
                    "variance_mean": float(np.nanmean(variance)) if variance.size else None,
                    "variance_max": float(np.nanmax(variance)) if variance.size else None,
                }
            except Exception as exc:
                metrics["warnings"].append({"field": "predicted_pointmap", "warning": str(exc)})

        if not metrics["geometry"]:
            return self.skipped(run, "no loadable geometry fields found", extra={"warnings": metrics["warnings"]})
        self.save_json(run.metrics_dir / "geometry_metrics.json", metrics)
        return metrics


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def camera_translations(camera: np.ndarray) -> np.ndarray:
    """Extract translations from [T,4,4], [T,3,4], or already [T,3] arrays."""

    arr = np.asarray(camera, dtype=float)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr
    if arr.ndim >= 3 and arr.shape[-2:] == (4, 4):
        return arr[..., :3, 3].reshape(-1, 3)
    if arr.ndim >= 3 and arr.shape[-2:] == (3, 4):
        return arr[..., :3, 3].reshape(-1, 3)
    if arr.ndim >= 2 and arr.shape[-1] >= 3:
        return arr.reshape(-1, arr.shape[-1])[:, :3]
    raise ValueError(f"Cannot extract camera translations from shape {arr.shape}")


def plot_translation_curve(plt: Any, translations: np.ndarray, path: Any, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 3.4))
    for axis, label in enumerate(["x", "y", "z"]):
        plt.plot(translations[:, axis], marker="o", linewidth=1.2, label=label)
    plt.xlabel("frame t")
    plt.ylabel("translation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_heatmap(plt: Any, image: np.ndarray, path: Any, *, title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.8, 4.0))
    plt.imshow(image, cmap="magma")
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
