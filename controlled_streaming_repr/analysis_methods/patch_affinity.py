"""Patch-to-patch affinity analysis for selected frame pairs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_analysis import AnalysisRun, BaseAnalysisMethod, iter_layer_token_refs, normalize_rows, patch_tokens_by_frame
from .registry import register_analysis_method


@register_analysis_method("patch_affinity")
class PatchAffinityAnalysis(BaseAnalysisMethod):
    """Compute cosine affinity matrices for a few informative frame pairs."""

    supports_group_run = True

    def run_group(self, runs: list[AnalysisRun]) -> list[dict[str, Any]]:
        results = [self.run(run) for run in runs]
        results.extend(self._run_order_cross_condition(runs))
        return results

    def run(self, run: AnalysisRun) -> dict[str, Any]:
        patch_refs = [
            ref for ref in iter_layer_token_refs(run.bundle, token_types=["patch_tokens", "d1_stage0_patch_tokens"])
            if ref[1] in {"patch_tokens", "d1_stage0_patch_tokens"}
        ]
        if not patch_refs:
            return self.skipped(run, "no patch token type found")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        seq_len = int(run.manifest.get("seq_len") or run.bundle.seq_len or 0) or None
        pairs = select_pairs(run.manifest)
        if not pairs:
            return self.skipped(run, "no valid frame pairs selected")

        metrics: dict[str, Any] = {
            "method": self.method_name,
            "status": "ok",
            "model_name": run.bundle.model_name,
            "scene_id": run.manifest.get("scene_id"),
            "setting_name": run.manifest.get("setting_name"),
            "condition_id": run.manifest.get("condition_id"),
            "pairs": {},
            "warnings": [],
        }
        base_dir = run.bundle_path.parent
        max_patches = int(self.config.get("max_patches_per_frame", 1024))

        for layer_name, token_type, ref, _layer_meta in patch_refs[:2]:
            try:
                tokens = patch_tokens_by_frame(ref, seq_len=seq_len, base_dir=base_dir)
            except Exception as exc:
                metrics["warnings"].append({"layer_token": f"{layer_name}/{token_type}", "warning": str(exc)})
                continue
            for i, j in pairs:
                if i >= tokens.shape[0] or j >= tokens.shape[0]:
                    continue
                a = downsample_patches(tokens[i], max_patches)
                b = downsample_patches(tokens[j], max_patches)
                affinity = normalize_rows(a) @ normalize_rows(b).T
                pair_key = f"{layer_name}/{token_type}/t{i}_t{j}"
                metrics["pairs"][pair_key] = affinity_summary(affinity)
                fig_path = run.figures_dir / f"{self.method_name}_{safe_name(layer_name)}_{token_type}_t{i}_t{j}.png"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(4.8, 4.2))
                plt.imshow(affinity, vmin=-1.0, vmax=1.0, cmap="viridis", aspect="auto")
                plt.colorbar(label="cosine affinity")
                plt.title(f"{layer_name} {token_type}: t{i} vs t{j}")
                plt.xlabel(f"t{j} patches")
                plt.ylabel(f"t{i} patches")
                plt.tight_layout()
                plt.savefig(fig_path, dpi=int(self.figure_config.get("dpi", 180)))
                plt.close()

        if not metrics["pairs"]:
            return self.skipped(run, "all patch affinity computations failed", extra={"warnings": metrics["warnings"]})
        self.save_json(run.metrics_dir / f"{self.method_name}.json", metrics)
        return metrics

    def _run_order_cross_condition(self, runs: list[AnalysisRun]) -> list[dict[str, Any]]:
        order_runs = [run for run in runs if run.manifest.get("setting_name") == "order_perturbation"]
        groups: dict[tuple[str, str, int, str], list[AnalysisRun]] = {}
        for run in order_runs:
            key = (
                run.bundle.model_name,
                str(run.manifest.get("scene_id")),
                int(run.manifest.get("seq_len") or 0),
                segment_id(run.manifest.get("condition_id", "")),
            )
            groups.setdefault(key, []).append(run)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results: list[dict[str, Any]] = []
        for group_runs in groups.values():
            normal = next((run for run in group_runs if run.manifest.get("metadata", {}).get("order_variant") == "normal"), None)
            if normal is None:
                continue
            normal_refs = iter_layer_token_refs(normal.bundle, token_types=["patch_tokens", "d1_stage0_patch_tokens"])
            if not normal_refs:
                continue
            normal_layer, normal_token_type, normal_ref, _normal_meta = normal_refs[0]
            try:
                normal_tokens = patch_tokens_by_frame(
                    normal_ref,
                    seq_len=int(normal.manifest.get("seq_len")),
                    base_dir=normal.bundle_path.parent,
                )
            except Exception:
                continue
            normal_frame_ids = [frame.get("source_frame_id") for frame in normal.manifest.get("frames", [])]
            normal_positions = {frame_id: idx for idx, frame_id in enumerate(normal_frame_ids)}

            for variant in group_runs:
                variant_name = variant.manifest.get("metadata", {}).get("order_variant")
                if variant is normal or variant_name == "normal":
                    continue
                variant_refs = [
                    ref for ref in iter_layer_token_refs(variant.bundle, token_types=[normal_token_type])
                    if ref[0] == normal_layer and ref[1] == normal_token_type
                ]
                if not variant_refs:
                    continue
                try:
                    variant_tokens = patch_tokens_by_frame(
                        variant_refs[0][2],
                        seq_len=int(variant.manifest.get("seq_len")),
                        base_dir=variant.bundle_path.parent,
                    )
                except Exception as exc:
                    result = {
                        "method": f"{self.method_name}_order_cross_condition",
                        "status": "skipped",
                        "reason": str(exc),
                        "condition_id": variant.manifest.get("condition_id"),
                    }
                    results.append(result)
                    continue
                variant_frame_ids = [frame.get("source_frame_id") for frame in variant.manifest.get("frames", [])]
                matches = [
                    (normal_positions[frame_id], variant_index, frame_id)
                    for variant_index, frame_id in enumerate(variant_frame_ids)
                    if frame_id in normal_positions
                ][: int(self.config.get("max_order_affinity_pairs", 2))]
                result: dict[str, Any] = {
                    "method": f"{self.method_name}_order_cross_condition",
                    "status": "ok",
                    "model_name": variant.bundle.model_name,
                    "scene_id": variant.manifest.get("scene_id"),
                    "setting_name": variant.manifest.get("setting_name"),
                    "condition_id": variant.manifest.get("condition_id"),
                    "baseline_condition_id": normal.manifest.get("condition_id"),
                    "layer_token": f"{normal_layer}/{normal_token_type}",
                    "pairs": {},
                }
                max_patches = int(self.config.get("max_patches_per_frame", 1024))
                for normal_index, variant_index, frame_id in matches:
                    affinity = (
                        normalize_rows(downsample_patches(normal_tokens[normal_index], max_patches))
                        @ normalize_rows(downsample_patches(variant_tokens[variant_index], max_patches)).T
                    )
                    pair_key = f"source_{frame_id}_normal_t{normal_index}_variant_t{variant_index}"
                    result["pairs"][pair_key] = affinity_summary(affinity)
                    fig_path = (
                        variant.figures_dir
                        / f"{self.method_name}_order_{safe_name(normal_layer)}_{normal_token_type}_{pair_key}.png"
                    )
                    fig_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.figure(figsize=(4.8, 4.2))
                    plt.imshow(affinity, vmin=-1.0, vmax=1.0, cmap="viridis", aspect="auto")
                    plt.colorbar(label="cosine affinity")
                    plt.title(f"normal t{normal_index} vs {variant_name} t{variant_index}")
                    plt.tight_layout()
                    plt.savefig(fig_path, dpi=int(self.figure_config.get("dpi", 180)))
                    plt.close()
                if result["pairs"]:
                    self.save_json(variant.metrics_dir / f"{self.method_name}_order_cross_condition.json", result)
                    results.append(result)
        return results


def select_pairs(manifest: dict[str, Any]) -> list[tuple[int, int]]:
    """Select a small set of frame pairs without hard-coding expected outcomes."""

    frames = manifest.get("frames", [])
    n = len(frames)
    if n < 2:
        return []
    setting = manifest.get("setting_name")
    if setting == "repeated_frame_stability":
        return [(0, n - 1)]
    if setting == "two_state_alternation":
        roles = [frame.get("role") for frame in frames]
        pairs: list[tuple[int, int]] = [(0, 1)]
        for role in sorted(set(roles)):
            positions = [idx for idx, value in enumerate(roles) if value == role]
            if len(positions) >= 2:
                pairs.append((positions[0], positions[-1]))
        return list(dict.fromkeys(pairs))[:4]
    return [(0, n - 1), (0, min(n - 1, n // 2))]


def downsample_patches(tokens: np.ndarray, max_patches: int) -> np.ndarray:
    """Uniformly downsample patches to keep affinity matrices manageable."""

    if tokens.shape[0] <= max_patches:
        return tokens
    indices = np.linspace(0, tokens.shape[0] - 1, max_patches).round().astype(int)
    return tokens[indices]


def affinity_summary(affinity: np.ndarray) -> dict[str, Any]:
    """Summarize an affinity matrix with top-k and entropy statistics."""

    row_max = affinity.max(axis=1)
    k = min(5, affinity.shape[1])
    topk = np.partition(affinity, -k, axis=1)[:, -k:]
    probs = np.exp(affinity - affinity.max(axis=1, keepdims=True))
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    return {
        "shape": list(affinity.shape),
        "max_affinity_mean": float(np.mean(row_max)),
        "max_affinity_max": float(np.max(row_max)),
        "mean_topk_affinity": float(np.mean(topk)),
        "entropy_mean": float(np.mean(entropy)),
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def segment_id(condition_id: str) -> str:
    parts = condition_id.split("_")
    return parts[-1] if parts else condition_id
