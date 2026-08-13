"""Token drift analysis over controlled frame sequences."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_analysis import AnalysisRun, BaseAnalysisMethod, cosine_distance, frame_embeddings, iter_layer_token_refs
from .registry import register_analysis_method


@register_analysis_method("token_drift")
class TokenDriftAnalysis(BaseAnalysisMethod):
    """Compute drift-to-first and drift-to-previous curves for token embeddings."""

    supports_group_run = True

    def run_group(self, runs: list[AnalysisRun]) -> list[dict[str, Any]]:
        results = [self.run(run) for run in runs]
        results.extend(self._run_order_cross_condition(runs))
        return results

    def run(self, run: AnalysisRun) -> dict[str, Any]:
        refs = iter_layer_token_refs(run.bundle, token_types=self.config.get("token_types"))
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
        roles = [frame.get("role") for frame in run.manifest.get("frames", [])]
        base_dir = run.bundle_path.parent

        for layer_name, token_type, ref, _layer_meta in refs:
            key = f"{layer_name}/{token_type}"
            try:
                embeddings = frame_embeddings(ref, seq_len=seq_len, base_dir=base_dir)
            except Exception as exc:
                metrics["warnings"].append({"layer_token": key, "warning": str(exc)})
                continue
            if embeddings.shape[0] < 2:
                metrics["warnings"].append({"layer_token": key, "warning": "less than two frame embeddings"})
                continue

            drift_first = cosine_distance(embeddings, np.repeat(embeddings[:1], embeddings.shape[0], axis=0))
            drift_previous = np.concatenate(
                [np.array([0.0]), cosine_distance(embeddings[1:], embeddings[:-1])]
            )
            role_stats = role_similarity_stats(embeddings, roles[: embeddings.shape[0]])
            metrics["layers"][key] = {
                "drift_to_first": drift_first.tolist(),
                "drift_to_previous": drift_previous.tolist(),
                "drift_to_first_mean": float(np.mean(drift_first)),
                "drift_to_previous_mean": float(np.mean(drift_previous[1:])),
                "role_similarity": role_stats,
            }

            fig_path = run.figures_dir / f"{self.method_name}_{safe_name(layer_name)}_{token_type}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6.0, 3.2))
            plt.plot(drift_first, marker="o", linewidth=1.4, label="to first")
            plt.plot(drift_previous, marker="s", linewidth=1.4, label="to previous")
            plt.xlabel("frame t")
            plt.ylabel("1 - cosine")
            plt.title(f"{layer_name} {token_type}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=int(self.figure_config.get("dpi", 180)))
            plt.close()

        if not metrics["layers"]:
            return self.skipped(run, "all token loads failed", extra={"warnings": metrics["warnings"]})
        self.save_json(run.metrics_dir / f"{self.method_name}.json", metrics)
        return metrics

    def _run_order_cross_condition(self, runs: list[AnalysisRun]) -> list[dict[str, Any]]:
        """Compare order variants by matching the same source frame id."""

        order_runs = [run for run in runs if run.manifest.get("setting_name") == "order_perturbation"]
        groups: dict[tuple[str, str, int, str], list[AnalysisRun]] = {}
        for run in order_runs:
            metadata = run.manifest.get("metadata", {})
            segment_key = segment_id(run.manifest.get("condition_id", ""))
            key = (
                run.bundle.model_name,
                str(run.manifest.get("scene_id")),
                int(run.manifest.get("seq_len") or 0),
                segment_key,
            )
            groups.setdefault(key, []).append(run)

        results: list[dict[str, Any]] = []
        for group_runs in groups.values():
            normal = next((run for run in group_runs if run.manifest.get("metadata", {}).get("order_variant") == "normal"), None)
            if normal is None:
                continue
            normal_refs = {
                f"{layer}/{token_type}": (ref, layer_meta)
                for layer, token_type, ref, layer_meta in iter_layer_token_refs(normal.bundle, token_types=self.config.get("token_types"))
            }
            normal_frame_ids = [frame.get("source_frame_id") for frame in normal.manifest.get("frames", [])]
            for variant in group_runs:
                variant_name = variant.manifest.get("metadata", {}).get("order_variant")
                if variant is normal or variant_name == "normal":
                    continue
                result: dict[str, Any] = {
                    "method": f"{self.method_name}_order_cross_condition",
                    "status": "ok",
                    "model_name": variant.bundle.model_name,
                    "scene_id": variant.manifest.get("scene_id"),
                    "setting_name": variant.manifest.get("setting_name"),
                    "condition_id": variant.manifest.get("condition_id"),
                    "baseline_condition_id": normal.manifest.get("condition_id"),
                    "order_variant": variant_name,
                    "layers": {},
                    "warnings": [],
                }
                variant_refs = {
                    f"{layer}/{token_type}": ref
                    for layer, token_type, ref, _layer_meta in iter_layer_token_refs(variant.bundle, token_types=self.config.get("token_types"))
                }
                variant_frame_ids = [frame.get("source_frame_id") for frame in variant.manifest.get("frames", [])]
                for key, (normal_ref, _layer_meta) in normal_refs.items():
                    if key not in variant_refs:
                        continue
                    try:
                        normal_embeddings = frame_embeddings(
                            normal_ref,
                            seq_len=int(normal.manifest.get("seq_len")),
                            base_dir=normal.bundle_path.parent,
                        )
                        variant_embeddings = frame_embeddings(
                            variant_refs[key],
                            seq_len=int(variant.manifest.get("seq_len")),
                            base_dir=variant.bundle_path.parent,
                        )
                    except Exception as exc:
                        result["warnings"].append({"layer_token": key, "warning": str(exc)})
                        continue
                    distances = matched_source_frame_distances(
                        normal_embeddings,
                        normal_frame_ids,
                        variant_embeddings,
                        variant_frame_ids,
                    )
                    if not distances:
                        continue
                    values = [item["distance"] for item in distances]
                    result["layers"][key] = {
                        "matched_frame_count": len(distances),
                        "distance_mean": float(np.mean(values)),
                        "distance_std": float(np.std(values)),
                        "distance_by_source_frame": distances,
                    }
                if result["layers"]:
                    self.save_json(variant.metrics_dir / f"{self.method_name}_order_cross_condition.json", result)
                    results.append(result)
        return results


def role_similarity_stats(embeddings: np.ndarray, roles: list[str | None]) -> dict[str, Any]:
    """Summarize same-role and cross-role cosine similarities."""

    if not roles or len(roles) != embeddings.shape[0]:
        return {"available": False, "reason": "role metadata unavailable"}
    norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-8)
    similarity = norm @ norm.T
    same: list[float] = []
    cross: list[float] = []
    by_role: dict[str, list[float]] = {}
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            if roles[i] == roles[j]:
                same.append(float(similarity[i, j]))
                by_role.setdefault(str(roles[i]), []).append(float(similarity[i, j]))
            else:
                cross.append(float(similarity[i, j]))
    return {
        "available": True,
        "same_role_similarity_mean": float(np.mean(same)) if same else None,
        "cross_role_similarity_mean": float(np.mean(cross)) if cross else None,
        "same_role_similarity_by_role": {
            role: float(np.mean(values)) for role, values in sorted(by_role.items()) if values
        },
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def segment_id(condition_id: str) -> str:
    parts = condition_id.split("_")
    return parts[-1] if parts else condition_id


def matched_source_frame_distances(
    normal_embeddings: np.ndarray,
    normal_frame_ids: list[str | None],
    variant_embeddings: np.ndarray,
    variant_frame_ids: list[str | None],
) -> list[dict[str, Any]]:
    """Compute corresponding-frame distances by source frame id."""

    normal_positions = {frame_id: idx for idx, frame_id in enumerate(normal_frame_ids) if frame_id is not None}
    output: list[dict[str, Any]] = []
    for variant_index, frame_id in enumerate(variant_frame_ids):
        if frame_id not in normal_positions:
            continue
        normal_index = normal_positions[frame_id]
        if normal_index >= normal_embeddings.shape[0] or variant_index >= variant_embeddings.shape[0]:
            continue
        distance = cosine_distance(
            normal_embeddings[normal_index : normal_index + 1],
            variant_embeddings[variant_index : variant_index + 1],
        )[0]
        output.append(
            {
                "source_frame_id": frame_id,
                "normal_t": int(normal_index),
                "variant_t": int(variant_index),
                "distance": float(distance),
            }
        )
    return output
