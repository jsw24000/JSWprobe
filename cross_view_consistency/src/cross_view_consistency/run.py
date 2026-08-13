"""End-to-end runner for cross-view feature consistency probing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .config import apply_cli_overrides, create_run_dir, load_config, validate_and_resolve_paths
from .feature_backends import build_backend
from .feature_backends.base import FeatureResult
from .frame_sampler import select_frames
from .geometry import GridSpec
from .matcher import match_feature_result
from .metrics import metrics_by_backend, metrics_by_pair
from .pair_sampler import sample_pairs, sample_query_sets
from .scannet_io import frames_to_dataframe, load_scene
from .utils import ensure_dir, save_json, set_seed, to_builtin
from .visualize import save_gt_projection, save_heatmaps, save_match_plot, save_pair_overview, save_summary_plots


def _feature_file_name(result: FeatureResult) -> str:
    return f"{result.backend_name}.pt"


def _backend_summary(result: FeatureResult, feature_path: Path) -> dict:
    return {
        "backend": result.backend_name,
        "layer_or_stage": result.layer_or_stage,
        "status": result.status,
        "feature_shape": list(result.features.shape) if result.features is not None else None,
        "grid_hw": list(result.grid_hw),
        "input_hw": list(result.input_hw),
        "feature_path": str(feature_path),
        "checkpoint_path": result.metadata.get("checkpoint_path", ""),
        "video_rope_disabled": result.metadata.get("video_rope_enabled") is False,
        "rope_disable_method": result.metadata.get("rope_disable_method", ""),
        "hook_name": result.metadata.get("hook_name", ""),
        "failure_reason": result.failure_reason,
        "metadata": result.metadata,
    }


def run_probe(
    config_path: str | Path,
    *,
    tag: str | None = None,
    num_frames: int | None = None,
    pairs_per_bucket: int | None = None,
    queries_per_pair: int | None = None,
    backends: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    cfg = apply_cli_overrides(
        cfg,
        tag=tag,
        num_frames=num_frames,
        pairs_per_bucket=pairs_per_bucket,
        queries_per_pair=queries_per_pair,
        backends=backends,
    )
    cfg = validate_and_resolve_paths(cfg)
    set_seed(int(cfg.get("run", {}).get("seed", 42)))
    run_dir = create_run_dir(cfg)

    summary: dict = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "backend_results": [],
        "assumptions": {},
        "status": "running",
    }
    paths = cfg["paths"]
    scene = load_scene(paths["scannet_root"], paths["scene_id"])
    summary["assumptions"]["depth_color_alignment"] = scene.alignment_note
    input_hw = (int(cfg["data"]["image_resize"]["height"]), int(cfg["data"]["image_resize"]["width"]))
    patch_size = int(cfg.get("dinov2", {}).get("patch_size", 14))
    grid_hw = (input_hw[0] // patch_size, input_hw[1] // patch_size)
    grid = GridSpec(input_hw=input_hw, grid_hw=grid_hw, patch_size=patch_size)
    summary["grid"] = {"input_hw": list(input_hw), "grid_hw": list(grid_hw), "patch_size": patch_size}

    frames = select_frames(scene, cfg)
    frames_df = frames_to_dataframe(frames)
    frames_df.to_csv(run_dir / "selected_frames.csv", index=False)

    pairs_df = sample_pairs(scene, frames, grid, cfg)
    if len(pairs_df) == 0:
        raise RuntimeError("No usable frame pairs found.")
    pairs_df.to_csv(run_dir / "pairs.csv", index=False)

    query_dir = ensure_dir(run_dir / "query_sets")
    query_sets = sample_query_sets(scene, frames, pairs_df, grid, cfg)
    for pair_id, qdf in query_sets.items():
        qdf.to_csv(query_dir / f"{pair_id}_queries.csv", index=False)
    summary["num_frames"] = len(frames)
    summary["num_pairs"] = len(pairs_df)
    summary["num_queries_total"] = int(sum(len(q) for q in query_sets.values()))

    features_dir = ensure_dir(run_dir / "features")
    feature_results: list[FeatureResult] = []
    for backend_name in cfg.get("feature_backends", {}).get("enabled", []):
        backend = build_backend(backend_name)
        results = backend.extract(frames, scene, grid, cfg)
        for result in results:
            feature_path = features_dir / _feature_file_name(result)
            result.save(feature_path)
            summary["backend_results"].append(_backend_summary(result, feature_path))
            if result.status == "success" and result.features is not None:
                feature_results.append(result)

    if not feature_results:
        summary["status"] = "failed_no_successful_backends"
        save_json(to_builtin(summary), run_dir / "run_summary.json")
        raise RuntimeError("All feature backends failed. See run_summary.json.")

    all_matches = []
    for result in tqdm(feature_results, desc="Matching feature backends", unit="backend"):
        matches = match_feature_result(result, pairs_df, query_sets, frames, grid, cfg)
        all_matches.append(matches)
    query_matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

    max_saved = int(cfg.get("run", {}).get("max_saved_queries_per_pair_backend", 512))
    save_query = bool(cfg.get("run", {}).get("save_query_matches", True))
    if save_query and len(query_matches):
        saved = (
            query_matches.groupby(["pair_id", "backend", "layer_or_stage", "method"], group_keys=False)
            .head(max_saved)
            .reset_index(drop=True)
        )
        saved.to_csv(run_dir / "query_matches.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "query_matches.csv", index=False)

    by_pair = metrics_by_pair(query_matches, pairs_df, cfg)
    by_backend = metrics_by_backend(query_matches, cfg)
    by_pair.to_csv(run_dir / "metrics_by_pair.csv", index=False)
    by_backend.to_csv(run_dir / "metrics_by_backend.csv", index=False)

    if cfg.get("visualization", {}).get("enabled", True):
        vis_dir = ensure_dir(run_dir / "visualizations")
        max_pairs = int(cfg["visualization"].get("max_pairs_to_visualize", 12))
        max_lines = int(cfg["visualization"].get("num_lines_per_match_plot", 80))
        correct_px = float(cfg["visualization"].get("correct_px_threshold", 16))
        heatmap_n = int(cfg["visualization"].get("num_heatmap_queries_per_pair_backend", 4))
        for pair in list(pairs_df.itertuples(index=False))[:max_pairs]:
            save_pair_overview(pair, frames, input_hw, vis_dir / f"{pair.pair_id}_overview.png")
            save_gt_projection(pair, frames, input_hw, query_sets[pair.pair_id], vis_dir / f"{pair.pair_id}_gt_projection.png", max_lines=max_lines)
            for result in feature_results:
                m = query_matches[
                    (query_matches["pair_id"] == pair.pair_id)
                    & (query_matches["backend"] == result.backend_name)
                    & (query_matches["method"] == "feature_nn")
                ]
                if len(m):
                    save_match_plot(pair, frames, input_hw, m, vis_dir / f"{pair.pair_id}_{result.backend_name}_matches.png", correct_px, max_lines)
                    save_heatmaps(pair, frames, input_hw, result, query_sets[pair.pair_id], vis_dir / "heatmaps", heatmap_n)
        save_summary_plots(by_backend, vis_dir)

    summary["status"] = "success"
    summary["outputs"] = {
        "config_used": str(run_dir / "config_used.yaml"),
        "selected_frames": str(run_dir / "selected_frames.csv"),
        "pairs": str(run_dir / "pairs.csv"),
        "metrics_by_backend": str(run_dir / "metrics_by_backend.csv"),
        "metrics_by_pair": str(run_dir / "metrics_by_pair.csv"),
        "query_matches": str(run_dir / "query_matches.csv"),
        "visualizations": str(run_dir / "visualizations"),
    }
    save_json(to_builtin(summary), run_dir / "run_summary.json")
    return run_dir

