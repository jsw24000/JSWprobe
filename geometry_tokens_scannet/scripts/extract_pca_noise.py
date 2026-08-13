#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from common import (
    connected_components,
    deep_get,
    deep_set,
    discover_paths,
    frame_output_name,
    infer_patch_grid,
    load_config,
    load_json,
    load_manifest_or_metadata,
    local_color_anomaly,
    overlay_heatmap,
    overlay_mask,
    pooled_pca_rgb,
    read_rgb,
    resize_grid_to_shape,
    robust_normalize,
    safe_layer_name,
    saturation_value,
    save_json,
    save_npy,
    save_rgb,
    select_frames,
    small_component_mask,
    top_percent_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PCA visualization noise masks on the Lingbot patch grid.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--scene-id", default=None, help="Override controlled.scene_id.")
    parser.add_argument("--setting-name", default=None, help="Override controlled.setting_name.")
    parser.add_argument("--condition-id", default=None, help="Override controlled.condition_id.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for debugging.")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.scene_id:
        deep_set(cfg, "controlled.scene_id", args.scene_id)
    if args.setting_name:
        deep_set(cfg, "controlled.setting_name", args.setting_name)
    if args.condition_id:
        deep_set(cfg, "controlled.condition_id", args.condition_id)
    if args.max_frames is not None:
        deep_set(cfg, "frames.max_frames", args.max_frames)


def resolve_pca_images(paths: dict[str, Path], cfg: dict, frames: list[dict]) -> dict[int, Path]:
    token_type = deep_get(cfg, "tokens.token_type", "patch_tokens")
    layer = deep_get(cfg, "pca_noise.layer", deep_get(cfg, "tokens.preferred_layer", "layer_11"))
    layer_safe = safe_layer_name(layer, token_type)
    found: dict[int, Path] = {}

    metrics = paths["pca_metrics"]
    if metrics.exists():
        data = load_json(metrics)
        for _layer_key, layer_data in data.get("layers", {}).items():
            if layer_safe not in _layer_key.replace("/", "_"):
                continue
            for fig in layer_data.get("figures", []):
                path = Path(fig)
                if not path.is_absolute():
                    path = (metrics.parent / path).resolve()
                t = parse_t_index(path.name)
                if t is not None and path.exists():
                    found[t] = path

    figure_dir = paths["pca_figures_dir"]
    if figure_dir.exists():
        for path in sorted(figure_dir.glob(f"*{layer_safe}*t*.png")):
            if "contact_sheet" in path.name:
                continue
            t = parse_t_index(path.name)
            if t is not None:
                found.setdefault(t, path)
        if not found:
            for path in sorted(figure_dir.glob("*t*.png")):
                if "contact_sheet" in path.name:
                    continue
                t = parse_t_index(path.name)
                if t is not None:
                    found.setdefault(t, path)

    allowed_t = {int(frame.get("t", idx)) for idx, frame in enumerate(frames)}
    return {t: path for t, path in sorted(found.items()) if t in allowed_t}


def parse_t_index(name: str) -> int | None:
    match = re.search(r"_t(\d{1,4})(?:\D|$)", name)
    if not match:
        return None
    return int(match.group(1))


def find_existing_mask(paths: dict[str, Path], cfg: dict, frame: dict, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, Path] | None:
    t = int(frame.get("t", 0))
    frame_id = str(frame.get("source_frame_id", ""))
    layer = safe_layer_name(deep_get(cfg, "pca_noise.layer", "layer_11"), deep_get(cfg, "tokens.token_type", "patch_tokens"))
    search_dirs = [paths["pca_figures_dir"], paths["pca_figures_dir"].parent, paths["scene_output"] / "pca_noise"]
    globs = deep_get(cfg, "pca_noise.existing_mask_globs", ["*noise*mask*.npy"])
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in globs:
            for candidate in sorted(directory.glob(pattern)):
                name = candidate.name
                if layer not in name and "pca" in name:
                    pass
                has_t = f"t{t:03d}" in name or f"t{t}" in name
                has_frame = frame_id and frame_id in name
                if not (has_t or has_frame):
                    continue
                arr = np.load(candidate)
                arr = np.squeeze(arr)
                if arr.ndim != 2:
                    continue
                if arr.shape != grid:
                    arr = resize_grid_to_shape(arr.astype(np.float32), grid, nearest=True)
                score = robust_normalize(arr.astype(np.float32))
                mask = arr.astype(bool) if arr.dtype == bool else score >= 0.5
                return score.astype(np.float32), mask.astype(bool), candidate
    return None


def heuristic_noise_mask(patch_rgb: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    radius = int(deep_get(cfg, "pca_noise.local_window_radius", 1))
    anomaly = local_color_anomaly(patch_rgb, radius=radius)
    sat, value = saturation_value(patch_rgb)
    black_thr = float(deep_get(cfg, "pca_noise.black_value_threshold", 0.08))
    sat_thr = float(deep_get(cfg, "pca_noise.saturation_threshold", 0.82))
    black = np.clip((black_thr - value) / max(black_thr, 1e-6), 0.0, 1.0)
    high_sat = np.clip((sat - sat_thr) / max(1.0 - sat_thr, 1e-6), 0.0, 1.0)

    weights = deep_get(cfg, "pca_noise.weights", {})
    score = (
        float(weights.get("local_color_anomaly", 1.0)) * anomaly
        + float(weights.get("black_patch", 0.35)) * black
        + float(weights.get("high_saturation", 0.25)) * high_sat
    )
    score = robust_normalize(score)
    percentile = float(deep_get(cfg, "pca_noise.score_percentile", 90.0))
    min_score = float(deep_get(cfg, "pca_noise.min_score", 0.45))
    threshold = max(min_score, float(np.nanpercentile(score, percentile)))
    candidate = score >= threshold

    max_area = int(deep_get(cfg, "pca_noise.small_component_max_area", 4))
    isolated = small_component_mask(candidate, max_area=max_area)
    keep_large = bool(deep_get(cfg, "pca_noise.keep_large_components", False))
    if keep_large:
        mask = candidate
    else:
        mask = isolated

    isolation_weight = float(weights.get("isolated_component", 0.35))
    score = robust_normalize(score + isolation_weight * isolated.astype(np.float32))
    if keep_large:
        mask = score >= threshold
    else:
        mask = (score >= threshold) & (isolated | candidate)
        mask &= isolated if np.any(isolated) else candidate

    fallback_min = int(deep_get(cfg, "pca_noise.fallback_min_patches", 3))
    if np.count_nonzero(mask) < fallback_min:
        fallback_percent = max(100.0 - percentile, 100.0 * fallback_min / score.size)
        mask = top_percent_mask(score, np.ones_like(mask, dtype=bool), fallback_percent)

    labels, sizes = connected_components(mask)
    info = {
        "method": "heuristic_from_pca_rgb",
        "threshold": threshold,
        "candidate_count": int(np.count_nonzero(candidate)),
        "isolated_count": int(np.count_nonzero(isolated)),
        "mask_count": int(np.count_nonzero(mask)),
        "component_count": len(sizes),
        "component_sizes": sizes,
        "heuristics": {
            "local_color_anomaly": "L2 distance from each patch PCA color to its local neighborhood median.",
            "black_patch": "Boost for very low value PCA patches.",
            "high_saturation": "Boost for highly saturated PCA patches.",
            "isolated_component": "Boost or retain small connected candidate components.",
        },
    }
    return score.astype(np.float32), mask.astype(bool), info


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    paths = discover_paths(cfg)
    manifest = load_manifest_or_metadata(paths, cfg)
    frames = select_frames(manifest.get("frames", []), deep_get(cfg, "frames.max_frames"))
    if not frames:
        raise RuntimeError("No frames found in manifest or fallback metadata.")

    grid = infer_patch_grid(cfg, paths, frames)
    pca_images = resolve_pca_images(paths, cfg, frames)
    if not pca_images:
        raise RuntimeError(
            f"No PCA images found under {paths['pca_figures_dir']} or {paths['pca_metrics']}. "
            "Set paths.pca_figures_dir or paths.pca_metrics_path in the config."
        )

    out_dir = paths["scene_output"] / "pca_noise"
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    score_stack = []
    mask_stack = []
    pca_rgb_stack = []
    frame_index_map = []
    method_infos = {}

    frame_by_t = {int(frame.get("t", idx)): frame for idx, frame in enumerate(frames)}
    for t, pca_path in sorted(pca_images.items()):
        frame = frame_by_t[t]
        existing = find_existing_mask(paths, cfg, frame, grid)
        patch_rgb = pooled_pca_rgb(pca_path, grid)
        if existing is None:
            score, mask, info = heuristic_noise_mask(patch_rgb, cfg)
            mask_source = None
        else:
            score, mask, source = existing
            info = {"method": "existing_mask_file", "mask_source": str(source), "mask_count": int(np.count_nonzero(mask))}
            mask_source = source

        frame_name = frame_output_name(frame)
        fdir = frames_dir / frame_name
        save_npy(fdir / "pca_noise_score.npy", score)
        save_npy(fdir / "pca_noise_mask.npy", mask.astype(bool))
        save_npy(fdir / "pca_patch_rgb.npy", patch_rgb.astype(np.float32))
        pca_rgb = read_rgb(pca_path)
        save_rgb(fdir / "pca_visualization.png", pca_rgb)
        save_rgb(fdir / "pca_noise_score_heatmap.png", overlay_heatmap(np.ones_like(pca_rgb), score, alpha=1.0))
        save_rgb(
            fdir / "pca_noise_mask_overlay.png",
            overlay_mask(pca_rgb, mask, deep_get(cfg, "visualization.pca_noise_color", [0, 190, 255])),
        )
        save_json(info, fdir / "pca_noise_info.json")

        score_stack.append(score)
        mask_stack.append(mask.astype(bool))
        pca_rgb_stack.append(patch_rgb.astype(np.float32))
        frame_index_map.append(
            {
                "stack_index": len(score_stack) - 1,
                "t": int(t),
                "source_frame_id": str(frame.get("source_frame_id", "")),
                "pca_path": str(pca_path),
                "mask_source": None if mask_source is None else str(mask_source),
                "frame_output": str(fdir),
            }
        )
        method_infos[str(t)] = info

    save_npy(out_dir / "pca_noise_score.npy", np.stack(score_stack, axis=0).astype(np.float32))
    save_npy(out_dir / "pca_noise_mask.npy", np.stack(mask_stack, axis=0).astype(bool))
    save_npy(out_dir / "pca_patch_rgb.npy", np.stack(pca_rgb_stack, axis=0).astype(np.float32))
    save_json(frame_index_map, out_dir / "frame_index_map.json")

    metadata = {
        "status": "ok",
        "scene_id": manifest.get("scene_id", deep_get(cfg, "controlled.scene_id")),
        "setting_name": manifest.get("setting_name", deep_get(cfg, "controlled.setting_name")),
        "condition_id": manifest.get("condition_id", deep_get(cfg, "controlled.condition_id")),
        "patch_grid": list(grid),
        "pca_figures_dir": str(paths["pca_figures_dir"]),
        "pca_metrics": str(paths["pca_metrics"]),
        "frame_count": len(score_stack),
        "heuristic_config": deep_get(cfg, "pca_noise", {}),
        "method_infos": method_infos,
    }
    save_json(metadata, out_dir / "metadata.json")
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "frame_count": len(score_stack), "patch_grid": list(grid)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"extract_pca_noise failed: {exc}", file=sys.stderr)
        raise
