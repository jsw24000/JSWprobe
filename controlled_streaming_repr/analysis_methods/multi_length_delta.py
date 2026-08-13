"""B1/X final-frame PCA and delta analysis for controlled_multi_length_exp_v5."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .base_analysis import AnalysisRun, BaseAnalysisMethod
from .prefix_delta import FinalTokens, load_final_tokens, save_scalar_heatmap, token_ref
from .registry import register_analysis_method
from .staircase_delta import (
    fit_shared_pca,
    project_pca,
    save_contact_sheet,
    save_rgb_png,
    scores_to_rgb,
    shared_rgb_range,
    resize_rgb,
)


EPS = 1e-8


@register_analysis_method("multi_length_delta")
class MultiLengthDeltaAnalysis(BaseAnalysisMethod):
    """Analyze B1/X 128-frame segment pairs at their final target frame."""

    supports_group_run = True

    def run(self, run: AnalysisRun) -> dict[str, Any]:
        return self.run_group([run])[0]

    def run_group(self, runs: list[AnalysisRun]) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, str], list[AnalysisRun]] = defaultdict(list)
        for run in runs:
            groups[
                (
                    run.bundle.model_name,
                    str(run.manifest.get("scene_id")),
                    str(run.manifest.get("setting_name")),
                )
            ].append(run)
        return [self._run_one_group(group_runs) for group_runs in groups.values()]

    def _run_one_group(self, runs: list[AnalysisRun]) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        first = runs[0]
        metrics_dir = first.metrics_dir.parent / self.method_name
        figures_dir = first.figures_dir.parent / self.method_name
        arrays_dir = metrics_dir / "arrays"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        arrays_dir.mkdir(parents=True, exist_ok=True)

        by_segment = group_runs_by_segment(runs)
        token_type = str(self.config.get("token_type", "patch_tokens"))
        layers = [int(value) for value in self.config.get("analysis_layers", [4, 11, 17, 23])]
        positive_role = str(self.config.get("sequence_roles", {}).get("positive", "B1"))
        baseline_role = str(self.config.get("sequence_roles", {}).get("baseline", "X"))

        result: dict[str, Any] = {
            "method": self.method_name,
            "status": "ok",
            "model_name": first.bundle.model_name,
            "scene_id": first.manifest.get("scene_id"),
            "setting_name": first.manifest.get("setting_name"),
            "analysis_layers": layers,
            "token_type": token_type,
            "positive_role": positive_role,
            "baseline_role": baseline_role,
            "figures_dir": str(figures_dir),
            "metrics_dir": str(metrics_dir),
            "arrays_dir": str(arrays_dir),
            "segments": {},
            "warnings": [],
        }

        rows: list[dict[str, Any]] = []
        for segment_id, role_runs in sorted(by_segment.items()):
            if positive_role not in role_runs or baseline_role not in role_runs:
                result["warnings"].append(
                    {
                        "segment_id": segment_id,
                        "warning": "missing required B1/X pair",
                        "available_roles": sorted(role_runs),
                    }
                )
                continue
            segment_summary = {
                "positive_condition": role_runs[positive_role].manifest.get("condition_id"),
                "baseline_condition": role_runs[baseline_role].manifest.get("condition_id"),
                "layers": {},
            }
            for layer in layers:
                layer_label = f"layer_{layer:02d}"
                try:
                    layer_result = analyze_segment_layer(
                        plt,
                        positive_run=role_runs[positive_role],
                        baseline_run=role_runs[baseline_role],
                        segment_id=segment_id,
                        layer=layer,
                        layer_label=layer_label,
                        token_type=token_type,
                        config=self.config,
                        figure_config=self.figure_config,
                        figures_dir=figures_dir,
                        metrics_dir=metrics_dir,
                        arrays_dir=arrays_dir,
                        method_name=self.method_name,
                    )
                except Exception as exc:
                    result["warnings"].append(
                        {"segment_id": segment_id, "layer": layer_label, "warning": str(exc)}
                    )
                    continue
                segment_summary["layers"][layer_label] = layer_result
                rows.append(flatten_layer_result(segment_id, layer_label, layer_result))
            if segment_summary["layers"]:
                result["segments"][segment_id] = segment_summary

        if not rows:
            skipped = self.skipped(first, "no multi-length B1/X metrics could be computed", extra=result)
            self.save_json(metrics_dir / f"{self.method_name}.json", skipped)
            return skipped

        metrics_csv = metrics_dir / f"{self.method_name}_metrics.csv"
        write_csv(metrics_csv, rows)
        summary_path = metrics_dir / f"{self.method_name}_summary.md"
        summary_path.write_text(build_markdown_summary(result, rows), encoding="utf-8")
        result["metrics_csv"] = str(metrics_csv)
        result["summary_markdown"] = str(summary_path)
        self.save_json(metrics_dir / f"{self.method_name}.json", result)
        return result


def group_runs_by_segment(runs: list[AnalysisRun]) -> dict[str, dict[str, AnalysisRun]]:
    grouped: dict[str, dict[str, AnalysisRun]] = defaultdict(dict)
    for run in runs:
        metadata = run.manifest.get("metadata", {})
        segment_id = str(metadata.get("segment_id") or str(run.manifest.get("condition_id", "")).split("_")[0])
        role = str(metadata.get("sequence_role") or str(run.manifest.get("condition_id", "")).split("_")[-1])
        grouped[segment_id][role] = run
    return grouped


def load_final_layer_tokens(run: AnalysisRun, layer_label: str, token_type: str) -> FinalTokens:
    ref, layer_meta = token_ref(run, layer_label, token_type)
    return load_final_tokens(ref, run=run, layer_meta=layer_meta)


def analyze_segment_layer(
    plt: Any,
    *,
    positive_run: AnalysisRun,
    baseline_run: AnalysisRun,
    segment_id: str,
    layer: int,
    layer_label: str,
    token_type: str,
    config: dict[str, Any],
    figure_config: dict[str, Any],
    figures_dir: Path,
    metrics_dir: Path,
    arrays_dir: Path,
    method_name: str,
) -> dict[str, Any]:
    positive = load_final_layer_tokens(positive_run, layer_label, token_type)
    baseline = load_final_layer_tokens(baseline_run, layer_label, token_type)
    if positive.tokens.shape != baseline.tokens.shape:
        raise ValueError(f"token shape mismatch: {positive.tokens.shape} vs {baseline.tokens.shape}")
    grid = positive.grid or baseline.grid
    if grid is None:
        raise ValueError(f"could not infer patch grid for {positive.tokens.shape[0]} patches")

    b1 = positive.tokens.astype(np.float32, copy=False)
    x = baseline.tokens.astype(np.float32, copy=False)
    delta = b1 - x
    delta_norm = np.linalg.norm(delta, axis=-1)

    n_pair_components = int(config.get("fixed_pca_components", 3))
    n_delta_components = int(config.get("pca_components", 3))
    sign_alignment = str(config.get("sign_alignment", "max_abs_loading_positive"))
    rgb_low = float(config.get("rgb_percentile_low", 1.0))
    rgb_high = float(config.get("rgb_percentile_high", 99.0))
    patch_size = int(figure_config.get("pca_rgb_patch_size", 1))
    upsample_mode = str(figure_config.get("pca_rgb_upsample_mode", "nearest"))
    dpi = int(figure_config.get("dpi", 180))
    colormap = str(figure_config.get("colormap", "viridis"))
    save_scores = bool(config.get("save_scores", True))
    save_rgb_arrays = bool(config.get("save_rgb_arrays", True))
    save_delta_arrays = bool(config.get("save_delta_arrays", False))

    prefix = f"{method_name}_{layer_label}_{segment_id}"

    pair_pca = fit_shared_pca([b1, x], n_components=n_pair_components, sign_alignment=sign_alignment)
    b1_scores = project_pca(b1, pair_pca)
    x_scores = project_pca(x, pair_pca)
    pair_rgb_range = shared_rgb_range(
        [b1_scores, x_scores],
        low_percentile=rgb_low,
        high_percentile=rgb_high,
    )
    b1_rgb = scores_to_rgb(b1_scores, grid, pair_rgb_range)
    x_rgb = scores_to_rgb(x_scores, grid, pair_rgb_range)
    pair_basis_path = metrics_dir / f"{prefix}_B1_X_pair_pca_basis.npz"
    np.savez_compressed(
        pair_basis_path,
        mean=pair_pca["mean"],
        components=pair_pca["components"],
        explained_variance_ratio=pair_pca["explained_variance_ratio"],
        sign_flips=pair_pca["sign_flips"],
        rgb_low=pair_rgb_range["low"],
        rgb_high=pair_rgb_range["high"],
    )

    b1_png = figures_dir / f"{prefix}_B1_pca_rgb.png"
    x_png = figures_dir / f"{prefix}_X_pca_rgb.png"
    save_rgb_png(b1_rgb, b1_png, dpi=dpi, patch_size=patch_size, upsample_mode=upsample_mode)
    save_rgb_png(x_rgb, x_png, dpi=dpi, patch_size=patch_size, upsample_mode=upsample_mode)

    delta_pca = fit_shared_pca([delta], n_components=n_delta_components, sign_alignment=sign_alignment)
    delta_scores = project_pca(delta, delta_pca)
    delta_rgb_range = shared_rgb_range(
        [delta_scores],
        low_percentile=rgb_low,
        high_percentile=rgb_high,
    )
    delta_rgb = scores_to_rgb(delta_scores, grid, delta_rgb_range)
    delta_basis_path = metrics_dir / f"{prefix}_B1_minus_X_delta_pca_basis.npz"
    np.savez_compressed(
        delta_basis_path,
        mean=delta_pca["mean"],
        components=delta_pca["components"],
        explained_variance_ratio=delta_pca["explained_variance_ratio"],
        sign_flips=delta_pca["sign_flips"],
        rgb_low=delta_rgb_range["low"],
        rgb_high=delta_rgb_range["high"],
    )

    delta_png = figures_dir / f"{prefix}_B1_minus_X_pca_rgb.png"
    save_rgb_png(delta_rgb, delta_png, dpi=dpi, patch_size=patch_size, upsample_mode=upsample_mode)
    norm_path = figures_dir / f"{prefix}_B1_minus_X_norm_heatmap.png"
    save_scalar_heatmap(
        plt,
        delta_norm,
        grid,
        norm_path,
        title=f"{layer_label} {segment_id} B1-X delta norm",
        cmap=colormap,
        dpi=dpi,
    )

    sheet_path = figures_dir / f"{prefix}_contact_sheet.png"
    save_contact_sheet(
        plt,
        [
            ("X pair PCA", resize_rgb(x_rgb, patch_size=patch_size, mode=upsample_mode)),
            ("B1 pair PCA", resize_rgb(b1_rgb, patch_size=patch_size, mode=upsample_mode)),
            ("B1-X delta PCA", resize_rgb(delta_rgb, patch_size=patch_size, mode=upsample_mode)),
        ],
        sheet_path,
        dpi=dpi,
        cols=int(figure_config.get("contact_sheet_cols", 3)),
    )

    if save_scores:
        np.save(arrays_dir / f"{prefix}_B1_pair_scores.npy", b1_scores.astype(np.float32, copy=False))
        np.save(arrays_dir / f"{prefix}_X_pair_scores.npy", x_scores.astype(np.float32, copy=False))
        np.save(arrays_dir / f"{prefix}_B1_minus_X_delta_scores.npy", delta_scores.astype(np.float32, copy=False))
    if save_rgb_arrays:
        np.save(arrays_dir / f"{prefix}_B1_pair_rgb.npy", b1_rgb.astype(np.float32, copy=False))
        np.save(arrays_dir / f"{prefix}_X_pair_rgb.npy", x_rgb.astype(np.float32, copy=False))
        np.save(arrays_dir / f"{prefix}_B1_minus_X_delta_rgb.npy", delta_rgb.astype(np.float32, copy=False))
    if save_delta_arrays:
        np.save(arrays_dir / f"{prefix}_B1_minus_X_delta.npy", delta.astype(np.float32, copy=False))

    b1_norm = np.linalg.norm(b1, axis=-1)
    x_norm = np.linalg.norm(x, axis=-1)
    relative_delta = delta_norm / np.maximum(0.5 * (b1_norm + x_norm), EPS)
    cosine = np.sum(b1 * x, axis=-1) / np.maximum(b1_norm * x_norm, EPS)
    frame_index = int(positive_run.manifest.get("metadata", {}).get("target_t", positive_run.manifest.get("seq_len", 1) - 1))

    return {
        "segment_id": segment_id,
        "layer": layer_label,
        "frame_index": frame_index,
        "target_frame_id": positive_run.manifest.get("metadata", {}).get("target_frame_id"),
        "positive_sequence": positive_run.manifest.get("condition_id"),
        "baseline_sequence": baseline_run.manifest.get("condition_id"),
        "positive_original_token_shape": positive.original_shape,
        "baseline_original_token_shape": baseline.original_shape,
        "token_shape": positive.final_shape,
        "patch_grid": list(grid),
        "mean_delta_norm": float(np.mean(delta_norm)),
        "std_delta_norm": float(np.std(delta_norm)),
        "max_delta_norm": float(np.max(delta_norm)),
        "p95_delta_norm": float(np.percentile(delta_norm, 95)),
        "relative_delta_mean": float(np.mean(relative_delta)),
        "relative_delta_p95": float(np.percentile(relative_delta, 95)),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_p05": float(np.percentile(cosine, 5)),
        "pair_pca_explained_variance_ratio": pair_pca["explained_variance_ratio"].tolist(),
        "pair_pca_explained_variance_ratio_pc1_to_pc3": float(
            np.sum(pair_pca["explained_variance_ratio"][:3])
        ),
        "delta_pca_explained_variance_ratio": delta_pca["explained_variance_ratio"].tolist(),
        "delta_pca_explained_variance_ratio_pc1_to_pc3": float(
            np.sum(delta_pca["explained_variance_ratio"][:3])
        ),
        "figures": {
            "B1_pca_rgb": str(b1_png),
            "X_pca_rgb": str(x_png),
            "B1_minus_X_pca_rgb": str(delta_png),
            "B1_minus_X_norm_heatmap": str(norm_path),
            "contact_sheet": str(sheet_path),
        },
        "basis": {
            "pair_pca": str(pair_basis_path),
            "delta_pca": str(delta_basis_path),
        },
        "pca_rgb_patch_size": int(patch_size),
        "pca_rgb_upsample_mode": upsample_mode,
    }


def flatten_layer_result(segment_id: str, layer_label: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "layer": layer_label,
        "frame_index": values.get("frame_index"),
        "target_frame_id": values.get("target_frame_id"),
        "positive_sequence": values.get("positive_sequence"),
        "baseline_sequence": values.get("baseline_sequence"),
        "mean_delta_norm": values.get("mean_delta_norm"),
        "std_delta_norm": values.get("std_delta_norm"),
        "max_delta_norm": values.get("max_delta_norm"),
        "p95_delta_norm": values.get("p95_delta_norm"),
        "relative_delta_mean": values.get("relative_delta_mean"),
        "relative_delta_p95": values.get("relative_delta_p95"),
        "cosine_mean": values.get("cosine_mean"),
        "cosine_p05": values.get("cosine_p05"),
        "pair_pca_explained_variance_ratio": values.get("pair_pca_explained_variance_ratio"),
        "pair_pca_explained_variance_ratio_pc1_to_pc3": values.get(
            "pair_pca_explained_variance_ratio_pc1_to_pc3"
        ),
        "delta_pca_explained_variance_ratio": values.get("delta_pca_explained_variance_ratio"),
        "delta_pca_explained_variance_ratio_pc1_to_pc3": values.get(
            "delta_pca_explained_variance_ratio_pc1_to_pc3"
        ),
        "token_shape": values.get("token_shape"),
        "patch_grid": values.get("patch_grid"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_markdown_summary(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# controlled_multi_length_exp_v5 Multi-Length Delta Summary",
        "",
        f"- model: {result.get('model_name')}",
        f"- scene: {result.get('scene_id')}",
        f"- setting: {result.get('setting_name')}",
        "",
        "| segment | layer | target | mean_delta_norm | p95_delta_norm | relative_delta_mean | cosine_mean | delta PC1-3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('segment_id')} | {row.get('layer')} | {row.get('target_frame_id')} | "
            f"{format_float(row.get('mean_delta_norm'))} | "
            f"{format_float(row.get('p95_delta_norm'))} | "
            f"{format_float(row.get('relative_delta_mean'))} | "
            f"{format_float(row.get('cosine_mean'))} | "
            f"{format_float(row.get('delta_pca_explained_variance_ratio_pc1_to_pc3'))} |"
        )
    return "\n".join(lines) + "\n"


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"
