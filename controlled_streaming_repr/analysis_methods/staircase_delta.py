"""Staircase final-frame token delta analysis for controlled_16_exp_v4."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from adapters.token_schema import load_token_bundle

from .base_analysis import AnalysisRun, BaseAnalysisMethod
from .prefix_delta import FinalTokens, load_final_tokens, token_ref
from .registry import register_analysis_method


EPS = 1e-8


@register_analysis_method("staircase_delta")
class StaircaseDeltaAnalysis(BaseAnalysisMethod):
    """Analyze final-frame patch-token changes across staircase_k00..k15."""

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

        by_condition = {str(run.manifest.get("condition_id")): run for run in runs}
        expected_conditions = list(
            self.config.get("staircase_conditions")
            or [f"staircase_k{k:02d}" for k in range(16)]
        )
        k_by_condition = {condition: parse_k(condition) for condition in expected_conditions}
        missing = [condition for condition in expected_conditions if condition not in by_condition]
        if missing and not bool(self.config.get("skip_missing", False)):
            result = self.skipped(
                first,
                "missing staircase token bundles",
                extra={"missing_conditions": missing, "available_conditions": sorted(by_condition)},
            )
            self.save_json(metrics_dir / f"{self.method_name}.json", result)
            return result

        layers = [int(value) for value in self.config.get("analysis_layers", [4, 11, 17, 23])]
        token_type = str(self.config.get("token_type", "patch_tokens"))
        baseline_runs = self._load_baseline_runs(first)
        if "X" not in baseline_runs or "B1" not in baseline_runs:
            result = self.skipped(
                first,
                "missing required v3 X/B1 baseline token bundles",
                extra={"baseline_runs": sorted(baseline_runs)},
            )
            self.save_json(metrics_dir / f"{self.method_name}.json", result)
            return result

        result: dict[str, Any] = {
            "method": self.method_name,
            "status": "ok",
            "model_name": first.bundle.model_name,
            "scene_id": first.manifest.get("scene_id"),
            "setting_name": first.manifest.get("setting_name"),
            "analysis_layers": layers,
            "token_type": token_type,
            "baseline_token_bundles": {
                role: str(run.bundle_path) for role, run in sorted(baseline_runs.items())
            },
            "figures_dir": str(figures_dir),
            "metrics_dir": str(metrics_dir),
            "arrays_dir": str(arrays_dir),
            "layers": {},
            "warnings": [],
        }

        global_rows: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        pca_rows: list[dict[str, Any]] = []
        object_rows: list[dict[str, Any]] = []
        object_saturation_rows: list[dict[str, Any]] = []

        patch_mask_cache: dict[tuple[int, int], np.ndarray | None] = {}
        for layer in layers:
            layer_label = f"layer_{layer:02d}"
            try:
                layer_output = self._analyze_layer(
                    plt,
                    layer=layer,
                    layer_label=layer_label,
                    token_type=token_type,
                    by_condition=by_condition,
                    expected_conditions=expected_conditions,
                    k_by_condition=k_by_condition,
                    baseline_runs=baseline_runs,
                    figures_dir=figures_dir,
                    arrays_dir=arrays_dir,
                    metrics_dir=metrics_dir,
                    patch_mask_cache=patch_mask_cache,
                )
            except Exception as exc:
                result["warnings"].append({"layer": layer_label, "warning": str(exc)})
                continue
            result["layers"][layer_label] = layer_output["summary"]
            global_rows.extend(layer_output["global_rows"])
            step_rows.extend(layer_output["step_rows"])
            pca_rows.extend(layer_output["pca_rows"])
            object_rows.extend(layer_output["object_rows"])
            object_saturation_rows.extend(layer_output["object_saturation_rows"])

        if not result["layers"]:
            skipped = self.skipped(first, "staircase_delta could not process any layer", extra=result)
            self.save_json(metrics_dir / f"{self.method_name}.json", skipped)
            return skipped

        paths = {
            "global_metrics_csv": str(write_csv(metrics_dir / f"{self.method_name}_global_metrics.csv", global_rows)),
            "step_metrics_csv": str(write_csv(metrics_dir / f"{self.method_name}_step_metrics.csv", step_rows)),
            "pca_metrics_csv": str(write_csv(metrics_dir / f"{self.method_name}_pca_metrics.csv", pca_rows)),
        }
        if object_rows:
            paths["object_metrics_csv"] = str(write_csv(metrics_dir / f"{self.method_name}_object_metrics.csv", object_rows))
            paths["object_saturation_csv"] = str(
                write_csv(metrics_dir / f"{self.method_name}_object_saturation.csv", object_saturation_rows)
            )
        summary_path = metrics_dir / f"{self.method_name}_summary.md"
        summary_path.write_text(
            build_markdown_summary(result, global_rows, step_rows, object_saturation_rows),
            encoding="utf-8",
        )
        paths["summary_markdown"] = str(summary_path)
        result.update(paths)
        self.save_json(metrics_dir / f"{self.method_name}.json", result)
        return result

    def _load_baseline_runs(self, first: AnalysisRun) -> dict[str, AnalysisRun]:
        baseline_cfg = dict(self.config.get("baseline_conditions", {}))
        baseline_project = str(self.config.get("baseline_project", "controlled_16_exp_v3"))
        baseline_setting = str(self.config.get("baseline_setting", "prefix_induced_modulation"))
        output_root = configured_output_root(first.config)
        model_name = first.bundle.model_name
        scene_id = str(first.manifest.get("scene_id"))
        baselines: dict[str, AnalysisRun] = {}
        for role, condition_id in baseline_cfg.items():
            condition = str(condition_id)
            bundle_path = (
                output_root
                / "tokens"
                / baseline_project
                / model_name
                / scene_id
                / baseline_setting
                / condition
                / "token_bundle.json"
            )
            if not bundle_path.is_file():
                continue
            manifest_path = (
                output_root
                / "manifests"
                / baseline_project
                / scene_id
                / baseline_setting
                / f"{condition}.json"
            )
            manifest = load_json(manifest_path) if manifest_path.is_file() else {"condition_id": condition}
            bundle = load_token_bundle(bundle_path)
            baselines[str(role)] = AnalysisRun(
                bundle=bundle,
                manifest=manifest,
                bundle_path=bundle_path,
                metrics_dir=first.metrics_dir,
                figures_dir=first.figures_dir,
                config=first.config,
            )
        return baselines

    def _analyze_layer(
        self,
        plt: Any,
        *,
        layer: int,
        layer_label: str,
        token_type: str,
        by_condition: dict[str, AnalysisRun],
        expected_conditions: list[str],
        k_by_condition: dict[str, int],
        baseline_runs: dict[str, AnalysisRun],
        figures_dir: Path,
        arrays_dir: Path,
        metrics_dir: Path,
        patch_mask_cache: dict[tuple[int, int], np.ndarray | None],
    ) -> dict[str, Any]:
        condition_tokens: dict[int, FinalTokens] = {}
        for condition in expected_conditions:
            run = by_condition.get(condition)
            if run is None:
                continue
            condition_tokens[k_by_condition[condition]] = load_final_layer_tokens(run, layer_label, token_type)
        if sorted(condition_tokens) != list(range(16)):
            raise ValueError(f"{layer_label}: expected k=0..15 tokens, got {sorted(condition_tokens)}")

        x_tokens = load_final_layer_tokens(baseline_runs["X"], layer_label, token_type)
        b1_tokens = load_final_layer_tokens(baseline_runs["B1"], layer_label, token_type)
        c_tokens = load_final_layer_tokens(baseline_runs["C"], layer_label, token_type) if "C" in baseline_runs else None
        grid = infer_common_grid([*condition_tokens.values(), x_tokens, b1_tokens, *( [c_tokens] if c_tokens else [] )])
        if grid is None:
            raise ValueError(f"{layer_label}: could not infer patch grid")
        shapes = {tuple(item.tokens.shape) for item in [*condition_tokens.values(), x_tokens, b1_tokens]}
        if len(shapes) != 1:
            raise ValueError(f"{layer_label}: token shape mismatch: {sorted(shapes)}")

        h = {k: item.tokens.astype(np.float32, copy=False) for k, item in sorted(condition_tokens.items())}
        x = x_tokens.tokens.astype(np.float32, copy=False)
        b1 = b1_tokens.tokens.astype(np.float32, copy=False)
        c = c_tokens.tokens.astype(np.float32, copy=False) if c_tokens else None
        full_delta = b1 - x
        dist_full = float(np.linalg.norm(full_delta.reshape(-1)))

        unrelated_reference = (
            by_condition["staircase_k00"]
            .manifest.get("metadata", {})
            .get("source_reference_sequences", {})
            .get("unrelated_sequence_name", "C")
        )
        baseline_tokens_by_name = {
            "X": x,
            "B1_local_relevant": b1,
        }
        if c is not None:
            baseline_tokens_by_name["C"] = c
        endpoint_checks = {
            "staircase_k15_vs_v3_B1": diff_summary(h[15], b1),
        }
        endpoint_key = f"staircase_k00_vs_v3_{unrelated_reference}"
        endpoint_checks[endpoint_key] = (
            diff_summary(h[0], baseline_tokens_by_name[unrelated_reference])
            if unrelated_reference in baseline_tokens_by_name
            else {"status": "missing_unrelated_baseline", "unrelated_reference": unrelated_reference}
        )

        global_rows = []
        step_rows = []
        for k in range(16):
            dx = h[k] - x
            db1 = h[k] - b1
            global_rows.append(
                {
                    **norm_metric_row(layer_label, "delta_X", k, dx),
                    **alignment_metrics(dx, db1, full_delta, dist_full, prefix="global"),
                    **patch_alignment_aggregates(dx, db1, full_delta, dist_full),
                }
            )
            global_rows.append(
                {
                    **norm_metric_row(layer_label, "delta_B1", k, db1),
                    "global_distance_to_X": float(np.linalg.norm(dx.reshape(-1))),
                    "global_distance_to_B1": float(np.linalg.norm(db1.reshape(-1))),
                    "global_normalized_progress": float(1.0 - np.linalg.norm(db1.reshape(-1)) / (dist_full + EPS)),
                }
            )
        for k in range(1, 16):
            step = h[k] - h[k - 1]
            step_rows.append(step_metric_row(layer_label, k, step, full_delta, grid))

        families = {
            "delta_X": {k: h[k] - x for k in range(16)},
            "delta_B1": {k: h[k] - b1 for k in range(16)},
            "delta_step": {k: h[k] - h[k - 1] for k in range(1, 16)},
        }
        pca_rows: list[dict[str, Any]] = []
        pca_summaries: dict[str, Any] = {}
        pca_projections_by_family: dict[str, dict[int, np.ndarray]] = {}
        rgb_images_by_family: dict[str, dict[int, np.ndarray]] = {}
        for family_name, deltas in families.items():
            pca_output = self._shared_pca_family(
                plt,
                layer=layer,
                layer_label=layer_label,
                family_name=family_name,
                deltas=deltas,
                grid=grid,
                figures_dir=figures_dir,
                arrays_dir=arrays_dir,
                metrics_dir=metrics_dir,
            )
            pca_rows.extend(pca_output["rows"])
            pca_summaries[family_name] = pca_output["summary"]
            pca_projections_by_family[family_name] = pca_output["projections"]
            rgb_images_by_family[family_name] = pca_output["rgb_images"]

        patch_mask = get_patch_mask_for_grid(self.config, baseline_runs["B1"], grid, patch_mask_cache)
        object_rows: list[dict[str, Any]] = []
        object_saturation_rows: list[dict[str, Any]] = []
        object_outputs: dict[str, Any] = {"status": "skipped", "reason": "object mask unavailable"}
        if patch_mask is not None:
            object_outputs = self._object_level_analysis(
                plt,
                layer_label=layer_label,
                h=h,
                x=x,
                b1=b1,
                full_delta=full_delta,
                dist_full=dist_full,
                grid=grid,
                patch_mask=patch_mask,
                pca_projections_by_family=pca_projections_by_family,
                rgb_images_by_family=rgb_images_by_family,
                figures_dir=figures_dir,
            )
            object_rows.extend(object_outputs["rows"])
            object_saturation_rows.extend(object_outputs["saturation_rows"])

        return {
            "summary": {
                "token_shape": list(next(iter(shapes))),
                "patch_grid": list(grid),
                "distance_X_to_B1": dist_full,
                "endpoint_checks": endpoint_checks,
                "pca": pca_summaries,
                "object_level": object_outputs.get("summary", object_outputs),
            },
            "global_rows": global_rows,
            "step_rows": step_rows,
            "pca_rows": pca_rows,
            "object_rows": object_rows,
            "object_saturation_rows": object_saturation_rows,
        }

    def _shared_pca_family(
        self,
        plt: Any,
        *,
        layer: int,
        layer_label: str,
        family_name: str,
        deltas: dict[int, np.ndarray],
        grid: tuple[int, int],
        figures_dir: Path,
        arrays_dir: Path,
        metrics_dir: Path,
    ) -> dict[str, Any]:
        n_components = int(self.config.get("pca_components", 3))
        exclude_zero_b1 = bool(self.config.get("exclude_zero_b1_endpoint_from_fit", True))
        fit_keys = sorted(deltas)
        if family_name == "delta_B1" and exclude_zero_b1 and len(fit_keys) > 1:
            fit_keys = [k for k in fit_keys if k != max(fit_keys)]
        pca = fit_shared_pca(
            [deltas[k] for k in fit_keys],
            n_components=n_components,
            sign_alignment=str(self.config.get("sign_alignment", "max_abs_loading_positive")),
        )
        projections = {k: project_pca(deltas[k], pca) for k in sorted(deltas)}
        rgb_range = shared_rgb_range(
            [projection for projection in projections.values()],
            low_percentile=float(self.config.get("rgb_percentile_low", 1.0)),
            high_percentile=float(self.config.get("rgb_percentile_high", 99.0)),
        )
        basis_path = metrics_dir / f"{self.method_name}_{layer_label}_{family_name}_shared_pca_basis.npz"
        np.savez_compressed(
            basis_path,
            mean=pca["mean"],
            components=pca["components"],
            explained_variance_ratio=pca["explained_variance_ratio"],
            sign_flips=pca["sign_flips"],
            rgb_low=rgb_range["low"],
            rgb_high=rgb_range["high"],
            fit_k_values=np.asarray(fit_keys, dtype=np.int32),
        )

        patch_size = int(self.figure_config.get("pca_rgb_patch_size", 1))
        upsample_mode = str(self.figure_config.get("pca_rgb_upsample_mode", "nearest"))
        dpi = int(self.figure_config.get("dpi", 180))
        save_scores = bool(self.config.get("save_scores", True))
        save_rgb_arrays = bool(self.config.get("save_rgb_arrays", True))
        save_delta_arrays = bool(self.config.get("save_delta_arrays", False))
        save_individual_pngs = bool(
            self.figure_config.get(
                "save_individual_pca_pngs",
                self.config.get("save_individual_pca_pngs", True),
            )
        )

        rows: list[dict[str, Any]] = []
        contact_images: list[tuple[str, np.ndarray]] = []
        rgb_images: dict[int, np.ndarray] = {}
        for k, projection in sorted(projections.items()):
            rgb = scores_to_rgb(projection, grid, rgb_range)
            rgb_images[k] = rgb
            prefix = f"{self.method_name}_{layer_label}_{family_name}_k{k:02d}"
            if save_individual_pngs:
                png_path = figures_dir / f"{prefix}_pca_rgb.png"
                save_rgb_png(rgb, png_path, dpi=dpi, patch_size=patch_size, upsample_mode=upsample_mode)
            contact_images.append((f"k={k:02d}", resize_rgb(rgb, patch_size=patch_size, mode=upsample_mode)))
            if save_scores:
                np.save(arrays_dir / f"{prefix}_scores.npy", projection.astype(np.float32, copy=False))
            if save_rgb_arrays:
                np.save(arrays_dir / f"{prefix}_rgb.npy", rgb.astype(np.float32, copy=False))
            if save_delta_arrays:
                np.save(arrays_dir / f"{prefix}_delta.npy", deltas[k].astype(np.float32, copy=False))
            rows.append(pca_metric_row(layer_label, family_name, k, projection, projections))

        sheet_path = figures_dir / f"{self.method_name}_{layer_label}_{family_name}_contact_sheet.png"
        save_contact_sheet(
            plt,
            contact_images,
            sheet_path,
            dpi=dpi,
            cols=int(self.figure_config.get("contact_sheet_cols", 4)),
        )
        gif_path = None
        if bool(self.figure_config.get("save_gif", self.config.get("save_gif", False))):
            gif_path = figures_dir / f"{self.method_name}_{layer_label}_{family_name}.gif"
            save_gif([image for _label, image in contact_images], gif_path)

        norm_sheet_path = figures_dir / f"{self.method_name}_{layer_label}_{family_name}_norm_contact_sheet.png"
        norm_images = [
            (f"k={k:02d}", scalar_to_rgb(np.linalg.norm(deltas[k], axis=-1).reshape(grid), cmap="viridis"))
            for k in sorted(deltas)
        ]
        save_contact_sheet(
            plt,
            norm_images,
            norm_sheet_path,
            dpi=dpi,
            cols=int(self.figure_config.get("contact_sheet_cols", 4)),
        )

        summary = {
            "basis_path": str(basis_path),
            "fit_k_values": fit_keys,
            "shared_basis_across_configs": True,
            "rgb_low": pca_array_to_list(rgb_range["low"]),
            "rgb_high": pca_array_to_list(rgb_range["high"]),
            "sign_alignment": pca["sign_alignment"],
            "sign_flips": pca["sign_flips"].astype(int).tolist(),
            "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
            "contact_sheet": str(sheet_path),
            "norm_contact_sheet": str(norm_sheet_path),
            "gif": str(gif_path) if gif_path else None,
        }
        return {
            "summary": summary,
            "rows": rows,
            "projections": projections,
            "rgb_images": rgb_images,
        }

    def _object_level_analysis(
        self,
        plt: Any,
        *,
        layer_label: str,
        h: dict[int, np.ndarray],
        x: np.ndarray,
        b1: np.ndarray,
        full_delta: np.ndarray,
        dist_full: float,
        grid: tuple[int, int],
        patch_mask: np.ndarray,
        pca_projections_by_family: dict[str, dict[int, np.ndarray]],
        rgb_images_by_family: dict[str, dict[int, np.ndarray]],
        figures_dir: Path,
    ) -> dict[str, Any]:
        mask_flat = patch_mask.reshape(-1)
        object_cfg = dict(self.config.get("object_mask", {}))
        include_background = bool(object_cfg.get("include_background", True))
        min_patch_count = int(object_cfg.get("min_patch_count", 3))
        thresholds = [float(value) for value in object_cfg.get("saturation_thresholds", [0.5, 0.8, 0.9])]
        object_ids = []
        for object_id in sorted(int(value) for value in np.unique(mask_flat)):
            if object_id == 0 and not include_background:
                continue
            count = int(np.sum(mask_flat == object_id))
            if count >= min_patch_count:
                object_ids.append(object_id)

        rows: list[dict[str, Any]] = []
        saturation_rows: list[dict[str, Any]] = []
        progress_by_object: dict[int, list[float]] = {}
        for object_id in object_ids:
            selector = mask_flat == object_id
            label = "background" if object_id == 0 else f"instance_{object_id}"
            full_obj = full_delta[selector]
            dist_full_obj = float(np.linalg.norm(full_obj.reshape(-1)))
            progress_curve: list[float] = []
            for k in range(16):
                dx = h[k] - x
                db1 = h[k] - b1
                dx_obj = dx[selector]
                db1_obj = db1[selector]
                alpha = dot_flat(dx_obj, full_obj) / (dot_flat(full_obj, full_obj) + EPS)
                progress = 1.0 - float(np.linalg.norm(db1_obj.reshape(-1))) / (dist_full_obj + EPS)
                progress_curve.append(progress)
                pc_values = {}
                for family_name in ("delta_X", "delta_B1"):
                    projection = pca_projections_by_family.get(family_name, {}).get(k)
                    if projection is None:
                        continue
                    means = projection[selector, :3].mean(axis=0)
                    for idx in range(3):
                        pc_values[f"{family_name}_pc{idx + 1}_mean"] = float(means[idx])
                rows.append(
                    {
                        "layer": layer_label,
                        "object_id": object_id,
                        "object_label": label,
                        "patch_count": int(np.sum(selector)),
                        "k": k,
                        "delta_X_mean_norm": float(np.linalg.norm(dx_obj, axis=-1).mean()),
                        "delta_B1_mean_norm": float(np.linalg.norm(db1_obj, axis=-1).mean()),
                        "alpha_to_B1_direction": float(alpha),
                        "distance_to_B1": float(np.linalg.norm(db1_obj.reshape(-1))),
                        "normalized_progress": float(progress),
                        **pc_values,
                    }
                )
            progress_by_object[object_id] = progress_curve
            for threshold in thresholds:
                reached = [k for k, value in enumerate(progress_curve) if value >= threshold]
                saturation_rows.append(
                    {
                        "layer": layer_label,
                        "object_id": object_id,
                        "object_label": label,
                        "patch_count": int(np.sum(selector)),
                        "threshold": threshold,
                        "saturation_k": min(reached) if reached else None,
                    }
                )

        max_objects = int(object_cfg.get("max_objects_for_plots", 16))
        ranked_objects = sorted(object_ids, key=lambda obj: int(np.sum(mask_flat == obj)), reverse=True)[:max_objects]
        heatmap_path = figures_dir / f"{self.method_name}_{layer_label}_object_progress_heatmap.png"
        save_object_heatmap(plt, progress_by_object, ranked_objects, heatmap_path, dpi=int(self.figure_config.get("dpi", 180)))
        curves_path = figures_dir / f"{self.method_name}_{layer_label}_object_progress_curves.png"
        save_object_curves(plt, progress_by_object, ranked_objects[: min(8, len(ranked_objects))], curves_path, dpi=int(self.figure_config.get("dpi", 180)))

        overlay_paths: dict[str, str] = {}
        for family_name, rgb_by_k in sorted(rgb_images_by_family.items()):
            overlay_images = [
                (f"k={k:02d}", overlay_boundaries(rgb, patch_mask))
                for k, rgb in sorted(rgb_by_k.items())
            ]
            path = figures_dir / f"{self.method_name}_{layer_label}_{family_name}_object_boundaries_contact_sheet.png"
            save_contact_sheet(
                plt,
                overlay_images,
                path,
                dpi=int(self.figure_config.get("dpi", 180)),
                cols=int(self.figure_config.get("contact_sheet_cols", 4)),
            )
            overlay_paths[family_name] = str(path)

        return {
            "summary": {
                "status": "ok",
                "object_count": len(object_ids),
                "top_object_ids": ranked_objects,
                "heatmap": str(heatmap_path),
                "curves": str(curves_path),
                "boundary_contact_sheets": overlay_paths,
            },
            "rows": rows,
            "saturation_rows": saturation_rows,
        }


def load_final_layer_tokens(run: AnalysisRun, layer_label: str, token_type: str) -> FinalTokens:
    ref, layer_meta = token_ref(run, layer_label, token_type)
    return load_final_tokens(ref, run=run, layer_meta=layer_meta)


def infer_common_grid(tokens: list[FinalTokens]) -> tuple[int, int] | None:
    for item in tokens:
        if item.grid is not None:
            return item.grid
    n_patches = tokens[0].tokens.shape[0] if tokens else 0
    side = int(round(math.sqrt(n_patches)))
    for height in range(side, 0, -1):
        if n_patches % height == 0:
            return height, n_patches // height
    return None


def configured_output_root(config: dict[str, Any]) -> Path:
    project_dir = Path(config.get("_project_dir", Path(__file__).resolve().parents[1]))
    root = Path(config.get("project", {}).get("output_root", "outputs")).expanduser()
    if not root.is_absolute():
        root = project_dir / root
    return root.resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_k(condition_id: str) -> int:
    marker = "k"
    idx = condition_id.rfind(marker)
    if idx < 0:
        raise ValueError(f"Could not parse staircase k from condition_id={condition_id!r}")
    return int(condition_id[idx + 1 :])


def norm_metric_row(layer: str, family: str, k: int, delta: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(delta, axis=-1)
    return {
        "layer": layer,
        "family": family,
        "k": k,
        "mean_norm": float(np.mean(norms)),
        "median_norm": float(np.median(norms)),
        "std_norm": float(np.std(norms)),
        "max_norm": float(np.max(norms)),
        "q25_norm": float(np.percentile(norms, 25)),
        "q75_norm": float(np.percentile(norms, 75)),
    }


def alignment_metrics(
    delta_x: np.ndarray,
    delta_b1: np.ndarray,
    full_delta: np.ndarray,
    dist_full: float,
    *,
    prefix: str,
) -> dict[str, Any]:
    alpha = dot_flat(delta_x, full_delta) / (dot_flat(full_delta, full_delta) + EPS)
    cosine = cosine_flat(delta_x, full_delta)
    orthogonal = delta_x - alpha * full_delta
    dist_x = float(np.linalg.norm(delta_x.reshape(-1)))
    dist_b1 = float(np.linalg.norm(delta_b1.reshape(-1)))
    return {
        f"{prefix}_alpha_to_B1_direction": float(alpha),
        f"{prefix}_cosine_to_B1_direction": float(cosine),
        f"{prefix}_orthogonal_residual_ratio": float(np.linalg.norm(orthogonal.reshape(-1)) / (dist_x + EPS)),
        f"{prefix}_distance_to_X": dist_x,
        f"{prefix}_distance_to_B1": dist_b1,
        f"{prefix}_normalized_progress": float(1.0 - dist_b1 / (dist_full + EPS)),
    }


def patch_alignment_aggregates(
    delta_x: np.ndarray,
    delta_b1: np.ndarray,
    full_delta: np.ndarray,
    dist_full: float,
) -> dict[str, Any]:
    full_sq = np.sum(full_delta * full_delta, axis=-1)
    alpha = np.sum(delta_x * full_delta, axis=-1) / np.maximum(full_sq, EPS)
    cos = np.sum(delta_x * full_delta, axis=-1) / np.maximum(
        np.linalg.norm(delta_x, axis=-1) * np.linalg.norm(full_delta, axis=-1),
        EPS,
    )
    progress = 1.0 - np.linalg.norm(delta_b1, axis=-1) / np.maximum(
        np.linalg.norm(full_delta, axis=-1),
        EPS,
    )
    return {
        "patch_alpha_mean": float(np.mean(alpha)),
        "patch_alpha_median": float(np.median(alpha)),
        "patch_alpha_std": float(np.std(alpha)),
        "patch_cosine_mean": float(np.mean(cos)),
        "patch_cosine_median": float(np.median(cos)),
        "patch_cosine_std": float(np.std(cos)),
        "patch_progress_mean": float(np.mean(progress)),
        "patch_progress_median": float(np.median(progress)),
        "patch_progress_std": float(np.std(progress)),
        "patch_distance_to_X_mean": float(np.linalg.norm(delta_x, axis=-1).mean()),
        "patch_distance_to_B1_mean": float(np.linalg.norm(delta_b1, axis=-1).mean()),
        "distance_X_to_B1": float(dist_full),
    }


def step_metric_row(layer: str, k: int, step: np.ndarray, full_delta: np.ndarray, grid: tuple[int, int]) -> dict[str, Any]:
    norms = np.linalg.norm(step, axis=-1)
    max_idx = int(np.argmax(norms))
    width = grid[1]
    return {
        **norm_metric_row(layer, "delta_step", k, step),
        "cosine_to_B1_direction": float(cosine_flat(step, full_delta)),
        "max_patch_index": max_idx,
        "max_patch_row": int(max_idx // width),
        "max_patch_col": int(max_idx % width),
        "max_patch_norm": float(norms[max_idx]),
    }


def diff_summary(a: np.ndarray, b: np.ndarray | None) -> dict[str, Any]:
    if b is None:
        return {"status": "missing"}
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return {
        "status": "ok",
        "shape": list(diff.shape),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "l2": float(np.linalg.norm(diff.reshape(-1))),
        "mean_patch_l2": float(np.linalg.norm(diff, axis=-1).mean()),
    }


def fit_shared_pca(
    matrices: list[np.ndarray],
    *,
    n_components: int,
    sign_alignment: str,
) -> dict[str, np.ndarray | str]:
    matrix = np.concatenate([np.asarray(item, dtype=np.float32).reshape(-1, item.shape[-1]) for item in matrices], axis=0)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n = max(1, min(int(n_components), vt.shape[0]))
    components = vt[:n].astype(np.float32, copy=True)
    sign_flips = np.ones(n, dtype=np.int8)
    if sign_alignment == "max_abs_loading_positive":
        for idx in range(n):
            anchor = int(np.argmax(np.abs(components[idx])))
            if components[idx, anchor] < 0:
                components[idx] *= -1.0
                sign_flips[idx] = -1
    elif sign_alignment not in {"none", ""}:
        raise ValueError(f"Unsupported PCA sign_alignment={sign_alignment!r}")
    variance = singular_values**2 / max(matrix.shape[0] - 1, 1)
    total_variance = max(float(np.sum(variance)), EPS)
    explained = (variance[:n] / total_variance).astype(np.float32)
    return {
        "mean": mean.reshape(-1).astype(np.float32),
        "components": components,
        "explained_variance_ratio": explained,
        "sign_flips": sign_flips,
        "sign_alignment": sign_alignment,
    }


def project_pca(delta: np.ndarray, pca: dict[str, Any]) -> np.ndarray:
    flat = np.asarray(delta, dtype=np.float32).reshape(-1, delta.shape[-1])
    projected = (flat - pca["mean"].reshape(1, -1)) @ pca["components"].T
    return projected.astype(np.float32, copy=False)


def shared_rgb_range(
    projections: list[np.ndarray],
    *,
    low_percentile: float,
    high_percentile: float,
) -> dict[str, np.ndarray]:
    values = np.concatenate([pad_scores(proj) for proj in projections], axis=0)
    low = np.nanpercentile(values, low_percentile, axis=0).astype(np.float32)
    high = np.nanpercentile(values, high_percentile, axis=0).astype(np.float32)
    same = np.abs(high - low) < EPS
    high[same] = low[same] + 1.0
    return {"low": low, "high": high}


def pad_scores(scores: np.ndarray) -> np.ndarray:
    out = np.zeros((scores.shape[0], 3), dtype=np.float32)
    out[:, : min(3, scores.shape[1])] = scores[:, : min(3, scores.shape[1])]
    return out


def scores_to_rgb(scores: np.ndarray, grid: tuple[int, int], rgb_range: dict[str, np.ndarray]) -> np.ndarray:
    values = pad_scores(scores).reshape(grid[0], grid[1], 3)
    low = rgb_range["low"].reshape(1, 1, 3)
    high = rgb_range["high"].reshape(1, 1, 3)
    return np.clip((values - low) / np.maximum(high - low, EPS), 0.0, 1.0)


def pca_metric_row(layer: str, family: str, k: int, projection: np.ndarray, all_projections: dict[int, np.ndarray]) -> dict[str, Any]:
    scores = pad_scores(projection)
    first = pad_scores(all_projections[min(all_projections)])
    last = pad_scores(all_projections[max(all_projections)])
    prev_change = None
    if k - 1 in all_projections:
        prev_change = float(np.linalg.norm((scores - pad_scores(all_projections[k - 1])).reshape(-1)))
    return {
        "layer": layer,
        "family": family,
        "k": k,
        "pc1_mean": float(np.mean(scores[:, 0])),
        "pc2_mean": float(np.mean(scores[:, 1])),
        "pc3_mean": float(np.mean(scores[:, 2])),
        "pc1_std": float(np.std(scores[:, 0])),
        "pc2_std": float(np.std(scores[:, 1])),
        "pc3_std": float(np.std(scores[:, 2])),
        "pc_energy": float(np.mean(np.sum(scores * scores, axis=-1))),
        "score_distance_to_first": float(np.linalg.norm((scores - first).reshape(-1))),
        "score_distance_to_last": float(np.linalg.norm((scores - last).reshape(-1))),
        "adjacent_score_change": prev_change,
    }


def get_patch_mask_for_grid(
    config: dict[str, Any],
    reference_run: AnalysisRun,
    grid: tuple[int, int],
    cache: dict[tuple[int, int], np.ndarray | None],
) -> np.ndarray | None:
    if grid in cache:
        return cache[grid]
    object_cfg = dict(config.get("object_mask", {}))
    if not bool(object_cfg.get("enabled", False)):
        cache[grid] = None
        return None
    frames = reference_run.manifest.get("frames", [])
    source_scene_dir = reference_run.manifest.get("source_scene_dir")
    target_frame_id = int(config.get("target_frame_id", 600))
    target_padded = f"{target_frame_id:04d}"
    template = object_cfg.get("path_template")
    candidates: list[Path] = []
    if template and source_scene_dir:
        candidates.append(
            Path(
                str(template).format(
                    source_scene_dir=source_scene_dir,
                    target_frame_id=target_frame_id,
                    target_frame_id_padded=target_padded,
                )
            )
        )
    if frames:
        rgb = Path(str(frames[-1].get("rgb_path", "")))
        if rgb:
            candidates.append(rgb.parent.parent / "instance-filt" / f"{target_frame_id}.png")
    for path in candidates:
        if path.is_file():
            from PIL import Image

            with Image.open(path) as image:
                resized = image.resize((grid[1], grid[0]), resample=Image.Resampling.NEAREST)
                cache[grid] = np.asarray(resized)
                return cache[grid]
    cache[grid] = None
    return None


def scalar_to_rgb(values: np.ndarray, *, cmap: str) -> np.ndarray:
    import matplotlib

    colormap = matplotlib.colormaps.get_cmap(cmap)
    arr = np.asarray(values, dtype=np.float32)
    low = float(np.nanpercentile(arr, 1))
    high = float(np.nanpercentile(arr, 99))
    norm = np.clip((arr - low) / max(high - low, EPS), 0.0, 1.0)
    return np.asarray(colormap(norm)[..., :3], dtype=np.float32)


def resize_rgb(image: np.ndarray, *, patch_size: int, mode: str) -> np.ndarray:
    if patch_size <= 1:
        return image
    from PIL import Image

    pil_image = Image.fromarray(rgb_to_uint8(image))
    resample = Image.Resampling.NEAREST if mode == "nearest" else Image.Resampling.BILINEAR
    width, height = pil_image.size
    resized = pil_image.resize((width * patch_size, height * patch_size), resample=resample)
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_rgb_png(image: np.ndarray, path: Path, *, dpi: int, patch_size: int, upsample_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    out = resize_rgb(image, patch_size=patch_size, mode=upsample_mode)
    Image.fromarray(rgb_to_uint8(out)).save(path, dpi=(dpi, dpi))


def rgb_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_contact_sheet(plt: Any, images: list[tuple[str, np.ndarray]], path: Path, *, dpi: int, cols: int) -> None:
    if not images:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = max(1, min(int(cols), len(images)))
    rows = int(math.ceil(len(images) / cols))
    plt.figure(figsize=(2.5 * cols, 2.35 * rows))
    for idx, (label, image) in enumerate(images, start=1):
        ax = plt.subplot(rows, cols, idx)
        ax.imshow(image)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_gif(images: list[np.ndarray], path: Path) -> None:
    if not images:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    frames = [Image.fromarray(rgb_to_uint8(image)) for image in images]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=450, loop=0)


def overlay_boundaries(rgb: np.ndarray, patch_mask: np.ndarray) -> np.ndarray:
    boundary = object_boundaries(patch_mask)
    out = np.array(rgb, dtype=np.float32, copy=True)
    out[boundary] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return out


def object_boundaries(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return boundary


def save_object_heatmap(
    plt: Any,
    progress_by_object: dict[int, list[float]],
    object_ids: list[int],
    path: Path,
    *,
    dpi: int,
) -> None:
    if not object_ids:
        return
    matrix = np.asarray([progress_by_object[obj] for obj in object_ids], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, max(2.5, 0.32 * len(object_ids))))
    plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.035, pad=0.02)
    plt.xticks(range(16), [str(k) for k in range(16)], fontsize=7)
    plt.yticks(range(len(object_ids)), [object_label(obj) for obj in object_ids], fontsize=7)
    plt.xlabel("staircase k")
    plt.ylabel("object")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_object_curves(
    plt: Any,
    progress_by_object: dict[int, list[float]],
    object_ids: list[int],
    path: Path,
    *,
    dpi: int,
) -> None:
    if not object_ids:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.2))
    xs = list(range(16))
    for obj in object_ids:
        plt.plot(xs, progress_by_object[obj], marker="o", linewidth=1.2, markersize=2.8, label=object_label(obj))
    plt.ylim(-0.05, 1.05)
    plt.xlabel("staircase k")
    plt.ylabel("normalized progress to B1")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def object_label(object_id: int) -> str:
    return "background" if object_id == 0 else f"inst {object_id}"


def dot_flat(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


def cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    return dot_flat(a, b) / (float(np.linalg.norm(a.reshape(-1))) * float(np.linalg.norm(b.reshape(-1))) + EPS)


def pca_array_to_list(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value).reshape(-1)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
    return path


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def build_markdown_summary(
    result: dict[str, Any],
    global_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    object_saturation_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {result.get('setting_name')} Staircase Delta Summary",
        "",
        "## Shared PCA",
        "- Each layer fits separate shared PCA bases for Delta_X, Delta_B1, and Delta_step.",
        "- RGB ranges are fitted jointly across all k for the same layer and delta family.",
        "- PCA signs use max_abs_loading_positive, so channel signs are deterministic.",
        "",
        "## Layer Summary",
        "| layer | k=1 progress | k50 | k80 | k90 | monotonic | max step k | H15 vs B1 mean_abs |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for layer in sorted(result.get("layers", {})):
        progress = [
            row
            for row in global_rows
            if row.get("layer") == layer and row.get("family") == "delta_X"
        ]
        by_k = {int(row["k"]): float(row.get("global_normalized_progress", 0.0)) for row in progress}
        k1 = by_k.get(1)
        k50 = first_k_at(by_k, 0.5)
        k80 = first_k_at(by_k, 0.8)
        k90 = first_k_at(by_k, 0.9)
        monotonic = is_monotonic_non_decreasing([by_k.get(k, float("nan")) for k in range(16)])
        layer_steps = [row for row in step_rows if row.get("layer") == layer]
        max_step = max(layer_steps, key=lambda row: float(row.get("mean_norm") or 0.0))["k"] if layer_steps else None
        endpoint = (
            result.get("layers", {})
            .get(layer, {})
            .get("endpoint_checks", {})
            .get("staircase_k15_vs_v3_B1", {})
            .get("mean_abs")
        )
        lines.append(
            f"| {layer} | {fmt(k1)} | {fmt(k50)} | {fmt(k80)} | {fmt(k90)} | "
            f"{'yes' if monotonic else 'no'} | {fmt(max_step)} | {fmt(endpoint)} |"
        )

    if object_saturation_rows:
        lines.extend(["", "## Object-Level Notes"])
        for layer in sorted({str(row["layer"]) for row in object_saturation_rows}):
            rows = [
                row
                for row in object_saturation_rows
                if row.get("layer") == layer and float(row.get("threshold", 0.0)) == 0.8 and row.get("saturation_k") not in {None, ""}
            ]
            if not rows:
                continue
            earliest = min(rows, key=lambda row: int(row["saturation_k"]))
            latest = max(rows, key=lambda row: int(row["saturation_k"]))
            lines.append(
                f"- {layer}: earliest 80% object is {earliest.get('object_label')} at k={earliest.get('saturation_k')}; "
                f"latest is {latest.get('object_label')} at k={latest.get('saturation_k')}."
            )
    lines.extend(
        [
            "",
            "PCA colors are visualization coordinates only; interpret them together with norms, projection statistics, and object curves.",
            "",
        ]
    )
    return "\n".join(lines)


def first_k_at(values: dict[int, float], threshold: float) -> int | None:
    for k in sorted(values):
        if values[k] >= threshold:
            return k
    return None


def is_monotonic_non_decreasing(values: list[float], *, tol: float = 1e-6) -> bool:
    clean = [value for value in values if not math.isnan(value)]
    return all(b + tol >= a for a, b in zip(clean, clean[1:]))


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)
