"""Prefix-induced final-frame token delta analysis."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .base_analysis import AnalysisRun, BaseAnalysisMethod, infer_patch_grid, to_numpy
from .registry import register_analysis_method


@register_analysis_method("prefix_delta")
class PrefixDeltaAnalysis(BaseAnalysisMethod):
    """Compare final-frame patch tokens across fixed-X prefix conditions."""

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
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        by_condition = {run.manifest.get("condition_id"): run for run in runs}
        comparisons = list(self.config.get("comparisons", []))
        required = sorted(
            {
                str(item.get("positive_sequence"))
                for item in comparisons
                if item.get("positive_sequence") and not item.get("optional", False)
            }
            | {
                str(item.get("baseline_sequence"))
                for item in comparisons
                if item.get("baseline_sequence") and not item.get("optional", False)
            }
        )
        missing = [name for name in required if name not in by_condition]
        if missing:
            result = self.skipped(
                first,
                "missing required prefix-delta conditions",
                extra={"missing_conditions": missing, "available_conditions": sorted(by_condition)},
            )
            self.save_json(metrics_dir / f"{self.method_name}.json", result)
            return result

        token_type = str(self.config.get("token_type", "patch_tokens"))
        layers = [int(value) for value in self.config.get("analysis_layers", [4, 11, 17, 23])]
        main_layer = int(self.config.get("main_layer", 17))
        pca_components = int(self.config.get("pca_components", 5))
        fixed_components = int(self.config.get("fixed_pca_components", 3))
        pca_rgb_patch_size = int(self.figure_config.get("pca_rgb_patch_size", 1))
        pca_rgb_upsample_mode = str(self.figure_config.get("pca_rgb_upsample_mode", "nearest"))

        result: dict[str, Any] = {
            "method": self.method_name,
            "status": "ok",
            "model_name": first.bundle.model_name,
            "scene_id": first.manifest.get("scene_id"),
            "setting_name": first.manifest.get("setting_name"),
            "analysis_layers": layers,
            "main_layer": main_layer,
            "token_type": token_type,
            "comparisons": {},
            "layers": {},
            "figures_dir": str(figures_dir),
            "metrics_dir": str(metrics_dir),
            "warnings": [],
        }
        manifest_summary = build_manifest_summary(runs)
        self.save_json(metrics_dir / f"{self.method_name}_manifest_summary.json", manifest_summary)

        final_tokens: dict[int, dict[str, FinalTokens]] = {}
        for layer in layers:
            layer_label = f"layer_{layer:02d}"
            final_tokens[layer] = {}
            for condition_id, run in sorted(by_condition.items()):
                try:
                    ref, layer_meta = token_ref(run, layer_label, token_type)
                    final_tokens[layer][str(condition_id)] = load_final_tokens(
                        ref,
                        run=run,
                        layer_meta=layer_meta,
                    )
                except Exception as exc:
                    result["warnings"].append(
                        {
                            "condition_id": condition_id,
                            "layer": layer_label,
                            "warning": str(exc),
                        }
                    )

        csv_rows: list[dict[str, Any]] = []
        for layer in layers:
            layer_label = f"layer_{layer:02d}"
            layer_tokens = final_tokens.get(layer, {})
            result["layers"].setdefault(layer_label, {})
            fixed_pca = None
            if len(layer_tokens) >= 2:
                try:
                    fixed_pca = fit_pca(
                        np.concatenate([item.tokens for item in layer_tokens.values()], axis=0),
                        n_components=fixed_components,
                    )
                    basis_path = metrics_dir / f"{self.method_name}_{layer_label}_fixed_pca_basis.npz"
                    np.savez_compressed(
                        basis_path,
                        mean=fixed_pca["mean"],
                        components=fixed_pca["components"],
                        explained_variance_ratio=fixed_pca["explained_variance_ratio"],
                    )
                    result["layers"][layer_label]["fixed_pca_basis"] = str(basis_path)
                    save_fixed_pca_maps(
                        plt,
                        layer=layer,
                        layer_tokens=layer_tokens,
                        fixed_pca=fixed_pca,
                        figures_dir=figures_dir,
                        method_name=self.method_name,
                        dpi=int(self.figure_config.get("dpi", 180)),
                        patch_size=pca_rgb_patch_size,
                        upsample_mode=pca_rgb_upsample_mode,
                    )
                except Exception as exc:
                    result["warnings"].append({"layer": layer_label, "warning": f"fixed PCA failed: {exc}"})

            for comparison in comparisons:
                comp_name = str(comparison.get("name"))
                positive = str(comparison.get("positive_sequence"))
                baseline = str(comparison.get("baseline_sequence"))
                optional = bool(comparison.get("optional", False))
                if positive not in layer_tokens or baseline not in layer_tokens:
                    if not optional:
                        result["warnings"].append(
                            {
                                "comparison": comp_name,
                                "layer": layer_label,
                                "warning": "missing tokens for comparison",
                                "positive_sequence": positive,
                                "baseline_sequence": baseline,
                            }
                        )
                    continue
                try:
                    layer_result = analyze_delta(
                        plt,
                        positive_tokens=layer_tokens[positive],
                        baseline_tokens=layer_tokens[baseline],
                        comparison_name=comp_name,
                        positive_sequence=positive,
                        baseline_sequence=baseline,
                        layer=layer,
                        figures_dir=figures_dir,
                        method_name=self.method_name,
                        pca_components=pca_components,
                        dpi=int(self.figure_config.get("dpi", 180)),
                        colormap=str(self.figure_config.get("colormap", "viridis")),
                        pca_rgb_patch_size=pca_rgb_patch_size,
                        pca_rgb_upsample_mode=pca_rgb_upsample_mode,
                    )
                except Exception as exc:
                    result["warnings"].append({"comparison": comp_name, "layer": layer_label, "warning": str(exc)})
                    continue
                result["comparisons"].setdefault(comp_name, {})[layer_label] = layer_result
                csv_rows.append(flatten_metric_row(comp_name, layer_label, layer_result))

        if not csv_rows:
            skipped = self.skipped(first, "no prefix delta metrics could be computed", extra=result)
            self.save_json(metrics_dir / f"{self.method_name}.json", skipped)
            return skipped

        write_metrics_csv(metrics_dir / f"{self.method_name}_metrics.csv", csv_rows)
        if main_layer in final_tokens:
            save_main_layer_panel(
                plt,
                runs_by_condition=by_condition,
                layer=main_layer,
                layer_tokens=final_tokens[main_layer],
                metrics=result,
                figures_dir=figures_dir,
                method_name=self.method_name,
                dpi=int(self.figure_config.get("dpi", 180)),
                patch_size=pca_rgb_patch_size,
                upsample_mode=pca_rgb_upsample_mode,
            )
        summary_path = metrics_dir / f"{self.method_name}_summary.md"
        summary_path.write_text(build_markdown_summary(result, manifest_summary), encoding="utf-8")
        result["manifest_summary"] = str(metrics_dir / f"{self.method_name}_manifest_summary.json")
        result["metrics_csv"] = str(metrics_dir / f"{self.method_name}_metrics.csv")
        result["summary_markdown"] = str(summary_path)
        result["overlap_analysis"] = {
            "status": "todo",
            "reason": "Depth/pose overlap validation is not required for the first prefix-delta pass.",
        }
        self.save_json(metrics_dir / f"{self.method_name}.json", result)
        return result


class FinalTokens:
    def __init__(
        self,
        *,
        tokens: np.ndarray,
        original_shape: list[int],
        final_shape: list[int],
        grid: tuple[int, int] | None,
        token_path: str | None,
    ) -> None:
        self.tokens = tokens
        self.original_shape = original_shape
        self.final_shape = final_shape
        self.grid = grid
        self.token_path = token_path


def token_ref(run: AnalysisRun, layer_label: str, token_type: str) -> tuple[Any, dict[str, Any]]:
    layer_values = run.bundle.layer_tokens.get(layer_label)
    if not isinstance(layer_values, dict):
        raise KeyError(f"{layer_label} not found in token bundle")
    ref = layer_values.get(token_type)
    if ref is None and token_type == "patch_tokens":
        ref = next((value for key, value in layer_values.items() if key.endswith("patch_tokens")), None)
    if ref is None:
        raise KeyError(f"{layer_label}/{token_type} not found in token bundle")
    return ref, layer_values


def load_final_tokens(ref: Any, *, run: AnalysisRun, layer_meta: dict[str, Any]) -> FinalTokens:
    arr = to_numpy(ref, base_dir=run.bundle_path.parent)
    token_path = str(ref) if isinstance(ref, (str, Path)) else None
    original_shape = [int(value) for value in arr.shape]
    if arr.ndim == 2:
        final = arr
    elif arr.ndim >= 3:
        seq_len = int(run.manifest.get("seq_len") or run.bundle.seq_len or 0) or None
        if seq_len is not None and arr.shape[0] != seq_len:
            frame_axis = next((axis for axis, size in enumerate(arr.shape[:-1]) if size == seq_len), 0)
            arr = np.moveaxis(arr, frame_axis, 0)
        final = arr[-1]
        if final.ndim > 2:
            final = final.reshape(-1, final.shape[-1])
    else:
        raise ValueError(f"Expected [T, P, C] or [P, C] tokens, got {arr.shape}")
    final = np.asarray(final, dtype=np.float32)
    grid = infer_patch_grid(final.shape[0], layer_meta)
    return FinalTokens(
        tokens=final,
        original_shape=original_shape,
        final_shape=[int(value) for value in final.shape],
        grid=grid,
        token_path=token_path,
    )


def analyze_delta(
    plt: Any,
    *,
    positive_tokens: FinalTokens,
    baseline_tokens: FinalTokens,
    comparison_name: str,
    positive_sequence: str,
    baseline_sequence: str,
    layer: int,
    figures_dir: Path,
    method_name: str,
    pca_components: int,
    dpi: int,
    colormap: str,
    pca_rgb_patch_size: int,
    pca_rgb_upsample_mode: str,
) -> dict[str, Any]:
    if positive_tokens.tokens.shape != baseline_tokens.tokens.shape:
        raise ValueError(
            f"token shape mismatch: {positive_tokens.tokens.shape} vs {baseline_tokens.tokens.shape}"
        )
    delta = positive_tokens.tokens - baseline_tokens.tokens
    delta_norm = np.linalg.norm(delta, axis=-1)
    grid = positive_tokens.grid or baseline_tokens.grid
    if grid is None:
        raise ValueError(f"could not infer patch grid for {delta.shape[0]} patches")

    prefix = f"{method_name}_layer_{layer:02d}_{safe_name(comparison_name)}"
    norm_path = figures_dir / f"{prefix}_norm_heatmap.png"
    save_scalar_heatmap(
        plt,
        delta_norm,
        grid,
        norm_path,
        title=f"layer {layer:02d} {comparison_name} delta norm",
        cmap=colormap,
        dpi=dpi,
    )

    pca = fit_pca(delta, n_components=pca_components)
    projection = project_pca(delta, pca)
    pc_paths: list[str] = []
    for pc_idx in range(min(3, projection.shape[1])):
        path = figures_dir / f"{prefix}_pc{pc_idx + 1}_heatmap.png"
        save_scalar_heatmap(
            plt,
            projection[:, pc_idx],
            grid,
            path,
            title=f"layer {layer:02d} {comparison_name} Delta-PC{pc_idx + 1}",
            cmap="coolwarm",
            dpi=dpi,
            symmetric=True,
        )
        pc_paths.append(str(path))

    rgb = pca_rgb_image(projection, grid)
    rgb_path = figures_dir / f"{prefix}_pca_rgb.png"
    save_rgb_image(
        plt,
        rgb,
        rgb_path,
        dpi=dpi,
        patch_size=pca_rgb_patch_size,
        upsample_mode=pca_rgb_upsample_mode,
    )

    explained = pca["explained_variance_ratio"].tolist()
    return {
        "positive_sequence": positive_sequence,
        "baseline_sequence": baseline_sequence,
        "delta_shape": [int(value) for value in delta.shape],
        "positive_token_shape": positive_tokens.final_shape,
        "baseline_token_shape": baseline_tokens.final_shape,
        "positive_original_token_shape": positive_tokens.original_shape,
        "baseline_original_token_shape": baseline_tokens.original_shape,
        "token_shape": positive_tokens.final_shape,
        "patch_grid": list(grid),
        "mean_delta_norm": float(np.mean(delta_norm)),
        "std_delta_norm": float(np.std(delta_norm)),
        "max_delta_norm": float(np.max(delta_norm)),
        "p95_delta_norm": float(np.percentile(delta_norm, 95)),
        "explained_variance_ratio": explained,
        "explained_variance_ratio_pc1": float(explained[0]) if explained else None,
        "explained_variance_ratio_pc1_to_pc3": float(np.sum(explained[:3])) if explained else None,
        "explained_variance_ratio_pc1_to_pc5": float(np.sum(explained[:5])) if explained else None,
        "figures": {
            "delta_norm_heatmap": str(norm_path),
            "delta_pc_heatmaps": pc_paths,
            "delta_pca_rgb": str(rgb_path),
        },
        "pca_rgb_patch_size": int(pca_rgb_patch_size),
        "pca_rgb_upsample_mode": pca_rgb_upsample_mode,
    }


def fit_pca(matrix: np.ndarray, *, n_components: int) -> dict[str, np.ndarray]:
    data = np.asarray(matrix, dtype=np.float32)
    if data.ndim != 2:
        data = data.reshape(-1, data.shape[-1])
    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n = max(1, min(int(n_components), vt.shape[0]))
    components = vt[:n]
    variance = singular_values**2 / max(data.shape[0] - 1, 1)
    total_variance = max(float(np.sum(variance)), 1e-12)
    explained = variance[:n] / total_variance
    return {
        "mean": mean.reshape(-1),
        "components": components,
        "explained_variance_ratio": explained,
    }


def project_pca(tokens: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    data = np.asarray(tokens, dtype=np.float32).reshape(-1, tokens.shape[-1])
    projected = (data - pca["mean"].reshape(1, -1)) @ pca["components"].T
    return projected


def save_fixed_pca_maps(
    plt: Any,
    *,
    layer: int,
    layer_tokens: dict[str, FinalTokens],
    fixed_pca: dict[str, np.ndarray],
    figures_dir: Path,
    method_name: str,
    dpi: int,
    patch_size: int,
    upsample_mode: str,
) -> None:
    for condition_id, tokens in sorted(layer_tokens.items()):
        if tokens.grid is None:
            continue
        projection = project_pca(tokens.tokens, fixed_pca)
        image = pca_rgb_image(projection, tokens.grid)
        path = figures_dir / f"{method_name}_layer_{layer:02d}_fixed_pca_{safe_name(condition_id)}_rgb.png"
        save_rgb_image(plt, image, path, dpi=dpi, patch_size=patch_size, upsample_mode=upsample_mode)


def pca_rgb_image(projected: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    height, width = grid
    values = np.zeros((projected.shape[0], 3), dtype=np.float32)
    values[:, : min(3, projected.shape[1])] = projected[:, : min(3, projected.shape[1])]
    image = values.reshape(height, width, 3)
    low = np.nanpercentile(image, 1, axis=(0, 1), keepdims=True)
    high = np.nanpercentile(image, 99, axis=(0, 1), keepdims=True)
    return np.clip((image - low) / np.maximum(high - low, 1e-8), 0.0, 1.0)


def scalar_grid(values: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    height, width = grid
    return np.asarray(values, dtype=np.float32).reshape(height, width)


def save_scalar_heatmap(
    plt: Any,
    values: np.ndarray,
    grid: tuple[int, int],
    path: Path,
    *,
    title: str,
    cmap: str,
    dpi: int,
    symmetric: bool = False,
) -> None:
    image = scalar_grid(values, grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.2, 3.8))
    kwargs: dict[str, Any] = {"cmap": cmap}
    if symmetric:
        vmax = float(np.nanmax(np.abs(image))) if image.size else 1.0
        kwargs.update({"vmin": -vmax, "vmax": vmax})
    plt.imshow(image, **kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_rgb_image(
    plt: Any,
    image: np.ndarray,
    path: Path,
    *,
    dpi: int,
    patch_size: int = 1,
    upsample_mode: str = "nearest",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    arr = (np.clip(np.nan_to_num(image, nan=0.0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pil_image = Image.fromarray(arr)
    if patch_size > 1:
        resample = Image.Resampling.NEAREST if upsample_mode == "nearest" else Image.Resampling.BILINEAR
        width, height = pil_image.size
        pil_image = pil_image.resize((width * patch_size, height * patch_size), resample=resample)
    pil_image.save(path, dpi=(dpi, dpi))


def save_main_layer_panel(
    plt: Any,
    *,
    runs_by_condition: dict[Any, AnalysisRun],
    layer: int,
    layer_tokens: dict[str, FinalTokens],
    metrics: dict[str, Any],
    figures_dir: Path,
    method_name: str,
    dpi: int,
    patch_size: int,
    upsample_mode: str,
) -> None:
    required = ["A_low_overlap", "B1_local_relevant", "B2_long_relevant"]
    if any(name not in layer_tokens for name in required):
        return
    grid = layer_tokens[required[0]].grid
    if grid is None:
        return
    fixed_pca = fit_pca(np.concatenate([layer_tokens[name].tokens for name in required], axis=0), n_components=3)
    pca_images = {
        name: pca_rgb_image(project_pca(layer_tokens[name].tokens, fixed_pca), grid)
        for name in required
    }
    b1_delta = layer_tokens["B1_local_relevant"].tokens - layer_tokens["A_low_overlap"].tokens
    b2_delta = layer_tokens["B2_long_relevant"].tokens - layer_tokens["A_low_overlap"].tokens
    b1_norm = scalar_grid(np.linalg.norm(b1_delta, axis=-1), grid)
    b2_norm = scalar_grid(np.linalg.norm(b2_delta, axis=-1), grid)
    b1_pca = pca_rgb_image(project_pca(b1_delta, fit_pca(b1_delta, n_components=3)), grid)
    rgb_x = load_target_rgb(runs_by_condition.get("A_low_overlap"))

    images: list[tuple[str, np.ndarray, str | None]] = [
        ("RGB X", rgb_x, None),
        ("A fixed PCA", pca_images["A_low_overlap"], None),
        ("B1 fixed PCA", pca_images["B1_local_relevant"], None),
        ("B2 fixed PCA", pca_images["B2_long_relevant"], None),
        ("B1-A norm", b1_norm, "viridis"),
        ("B2-A norm", b2_norm, "viridis"),
        ("B1-A delta PCA", b1_pca, None),
    ]
    path = figures_dir / f"{method_name}_layer_{layer:02d}_main_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(17.0, 3.0))
    for idx, (title, image, cmap) in enumerate(images, start=1):
        ax = plt.subplot(1, len(images), idx)
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    metrics.setdefault("layers", {}).setdefault(f"layer_{layer:02d}", {})["main_panel"] = str(path)


def load_target_rgb(run: AnalysisRun | None) -> np.ndarray:
    if run is None:
        return np.zeros((16, 16, 3), dtype=np.float32)
    frames = run.manifest.get("frames", [])
    path_value = frames[-1].get("rgb_path") if frames else None
    if not path_value:
        return np.zeros((16, 16, 3), dtype=np.float32)
    path = Path(path_value)
    if not path.is_absolute():
        path = (run.bundle_path.parent / path).resolve()
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def build_manifest_summary(runs: list[AnalysisRun]) -> dict[str, Any]:
    sequences: dict[str, Any] = {}
    for run in sorted(runs, key=lambda item: str(item.manifest.get("condition_id"))):
        condition_id = str(run.manifest.get("condition_id"))
        metadata = run.manifest.get("metadata", {})
        frames = run.manifest.get("frames", [])
        sequences[condition_id] = {
            "experiment_name": metadata.get("experiment_name"),
            "scene_id": run.manifest.get("scene_id"),
            "sequence_name": metadata.get("sequence_name", condition_id),
            "sequence_role": metadata.get("sequence_role"),
            "target_frame_id": metadata.get("target_frame_id"),
            "prefix_frame_ids": metadata.get("prefix_frame_ids"),
            "full_frame_ids": metadata.get("full_frame_ids"),
            "actual_image_paths": [frame.get("rgb_path") for frame in frames],
            "actual_depth_paths": [frame.get("depth_path") for frame in frames],
            "actual_pose_paths": [frame.get("pose_path") for frame in frames],
            "token_output_path": str(run.bundle_path.parent),
            "token_bundle": str(run.bundle_path),
            "config_path": metadata.get("experiment_config_path"),
            "git_commit": metadata.get("git_commit"),
            "built_at_utc": metadata.get("built_at_utc"),
            "all_frames_found": metadata.get("all_frames_found", True),
        }
    return {
        "experiment_name": runs[0].manifest.get("metadata", {}).get("experiment_name"),
        "scene_id": runs[0].manifest.get("scene_id"),
        "setting_name": runs[0].manifest.get("setting_name"),
        "sequences": sequences,
    }


def flatten_metric_row(comparison_name: str, layer_label: str, layer_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison": comparison_name,
        "layer": layer_label,
        "mean_delta_norm": layer_result.get("mean_delta_norm"),
        "std_delta_norm": layer_result.get("std_delta_norm"),
        "max_delta_norm": layer_result.get("max_delta_norm"),
        "p95_delta_norm": layer_result.get("p95_delta_norm"),
        "explained_variance_ratio_pc1": layer_result.get("explained_variance_ratio_pc1"),
        "explained_variance_ratio_pc1_to_pc3": layer_result.get("explained_variance_ratio_pc1_to_pc3"),
        "explained_variance_ratio_pc1_to_pc5": layer_result.get("explained_variance_ratio_pc1_to_pc5"),
        "token_shape": layer_result.get("token_shape"),
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_summary(metrics: dict[str, Any], manifest_summary: dict[str, Any]) -> str:
    lines = [
        f"# {manifest_summary.get('experiment_name') or 'controlled_16_exp_v3'} Prefix Delta Summary",
        "",
        "## Sequence Definitions",
    ]
    for name, sequence in manifest_summary.get("sequences", {}).items():
        lines.append(
            f"- {name}: role={sequence.get('sequence_role')}, "
            f"prefix={sequence.get('prefix_frame_ids')}, X={sequence.get('target_frame_id')}"
        )
    lines.extend(
        [
            "",
            "## Delta Definition",
            "- B1_minus_A = Z_X_B1 - Z_X_A",
            "- B2_minus_A = Z_X_B2 - Z_X_A",
            "- B1_minus_B2 = Z_X_B1 - Z_X_B2",
            "",
            "## Layer-Wise Metrics",
            "| comparison | layer | mean_delta_norm | p95_delta_norm | pc1 | pc1_to_pc3 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison, by_layer in sorted(metrics.get("comparisons", {}).items()):
        for layer, values in sorted(by_layer.items()):
            lines.append(
                f"| {comparison} | {layer} | "
                f"{format_float(values.get('mean_delta_norm'))} | "
                f"{format_float(values.get('p95_delta_norm'))} | "
                f"{format_float(values.get('explained_variance_ratio_pc1'))} | "
                f"{format_float(values.get('explained_variance_ratio_pc1_to_pc3'))} |"
            )
    lines.extend(["", "## Layer17 Emphasis", layer17_interpretation(metrics), ""])
    return "\n".join(lines)


def layer17_interpretation(metrics: dict[str, Any]) -> str:
    statements: list[str] = []
    for comparison, by_layer in sorted(metrics.get("comparisons", {}).items()):
        layer17 = by_layer.get("layer_17")
        if not layer17:
            continue
        other_means = [
            values.get("mean_delta_norm")
            for layer, values in by_layer.items()
            if layer != "layer_17" and values.get("mean_delta_norm") is not None
        ]
        mean17 = layer17.get("mean_delta_norm")
        pc17 = layer17.get("explained_variance_ratio_pc1_to_pc3")
        if other_means and mean17 is not None:
            rank = 1 + sum(1 for value in other_means if value > mean17)
            statements.append(
                f"- {comparison}: layer17 mean delta norm ranks {rank} among "
                f"{len(other_means) + 1} analyzed layers; PC1-3 explains {format_float(pc17)}."
            )
        elif mean17 is not None:
            statements.append(
                f"- {comparison}: layer17 mean delta norm is {format_float(mean17)}; "
                f"PC1-3 explains {format_float(pc17)}."
            )
    if not statements:
        return "- Layer17 metrics were not available in this run."
    return "\n".join(statements)


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
