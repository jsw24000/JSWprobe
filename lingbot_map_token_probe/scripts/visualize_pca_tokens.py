#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from probe_utils import (
    load_metadata,
    log,
    make_contact_sheet,
    natural_key,
    save_metadata,
    token_files,
    write_summary_md,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize token maps with PCA RGB.")
    parser.add_argument("--input", required=True, help="Scene output dir, or outputs dir for --fit-scope global.")
    parser.add_argument("--output", required=True, help="Output visualization directory.")
    parser.add_argument("--fit-scope", default="scene", choices=["scene", "global"])
    parser.add_argument("--normalization", default="l2", choices=["l2", "zscore", "none"])
    parser.add_argument("--controls-output", default=None, help="Default: <input>/pca_vis_controls")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--contact-cols", type=int, default=4)
    return parser.parse_args()


def scene_dirs_from_input(path: Path) -> List[Path]:
    if (path / "tokens").is_dir():
        return [path]
    scenes = [p for p in path.iterdir() if p.is_dir() and (p / "tokens").is_dir()]
    return sorted(scenes, key=lambda p: natural_key(p.name))


def load_stage_tokens(scene_dir: Path, stage_file: Path) -> tuple[torch.Tensor, dict]:
    payload = torch.load(stage_file, map_location="cpu", weights_only=False)
    tokens = payload["tokens"].float()
    if tokens.ndim == 3:
        hw = payload.get("token_hw")
        if not hw:
            raise ValueError(f"{stage_file} has [T,N,C] tokens but no token_hw.")
        tokens = tokens.reshape(tokens.shape[0], int(hw[0]), int(hw[1]), tokens.shape[-1])
    if tokens.ndim != 4:
        raise ValueError(f"{stage_file} has unsupported token shape {tuple(tokens.shape)}")
    return tokens.contiguous(), payload


def flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.reshape(-1, tokens.shape[-1]).float()


def fit_normalizer(feature_sets: Sequence[torch.Tensor], mode: str) -> dict:
    if mode != "zscore":
        return {"mode": mode}
    x = torch.cat([flatten_tokens(t) for t in feature_sets], dim=0)
    return {
        "mode": mode,
        "mean": x.mean(dim=0),
        "std": x.std(dim=0).clamp_min(1e-6),
    }


def apply_normalizer(x: torch.Tensor, stats: dict) -> torch.Tensor:
    mode = stats.get("mode", "l2")
    if mode == "l2":
        return F.normalize(x, dim=-1, eps=1e-6)
    if mode == "zscore":
        return (x - stats["mean"]) / stats["std"]
    return x


def fit_pca(feature_sets: Sequence[torch.Tensor], normalizer: dict) -> dict:
    x = torch.cat([flatten_tokens(t) for t in feature_sets], dim=0)
    x = apply_normalizer(x, normalizer)
    mean = x.mean(dim=0)
    x_centered = x - mean
    q = min(3, min(x_centered.shape))
    if q < 3:
        raise ValueError(f"Need at least 3 PCA dimensions, got matrix {tuple(x_centered.shape)}")
    _, _, v = torch.pca_lowrank(x_centered, q=3, center=False, niter=4)
    return {"normalizer": normalizer, "mean": mean, "components": v[:, :3]}


def project_with_pca(tokens: torch.Tensor, model: dict) -> torch.Tensor:
    x = flatten_tokens(tokens)
    x = apply_normalizer(x, model["normalizer"])
    y = (x - model["mean"]) @ model["components"]
    return y.reshape(tokens.shape[0], tokens.shape[1], tokens.shape[2], 3)


def robust_uint8(projected: torch.Tensor) -> np.ndarray:
    arr = projected.numpy()
    flat = arr.reshape(-1, 3)
    lo = np.percentile(flat, 1, axis=0)
    hi = np.percentile(flat, 99, axis=0)
    denom = np.maximum(hi - lo, 1e-6)
    arr = (arr - lo) / denom
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).round().astype(np.uint8)


def project_random(tokens: torch.Tensor, normalizer: dict, seed: int) -> torch.Tensor:
    x = flatten_tokens(tokens)
    x = apply_normalizer(x, normalizer)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(x.shape[-1], 3, generator=gen, dtype=x.dtype)
    proj = F.normalize(proj, dim=0, eps=1e-6)
    y = x @ proj
    return y.reshape(tokens.shape[0], tokens.shape[1], tokens.shape[2], 3)


def shuffle_tokens(tokens: torch.Tensor, seed: int) -> torch.Tensor:
    t, h, w, c = tokens.shape
    flat = tokens.reshape(t, h * w, c).clone()
    for frame_id in range(t):
        gen = torch.Generator(device="cpu").manual_seed(seed + frame_id)
        perm = torch.randperm(h * w, generator=gen)
        flat[frame_id] = flat[frame_id, perm]
    return flat.reshape(t, h, w, c)


def frame_paths_for_scene(scene_dir: Path, metadata: dict) -> List[Path]:
    paths = [Path(p) for p in metadata.get("frame_paths", [])]
    if paths and all(p.is_file() for p in paths):
        return paths
    return sorted((scene_dir / "frames").glob("frame_*.*"), key=lambda p: natural_key(p.name))


def save_maps(
    maps_uint8: np.ndarray,
    stage_dir: Path,
    frame_paths: Sequence[Path],
    suffix: str,
    contact_cols: int,
) -> List[Path]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for frame_id, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as src:
            size = src.size
        vis = Image.fromarray(maps_uint8[frame_id]).resize(size, Image.Resampling.BILINEAR)
        out_path = stage_dir / f"{frame_path.stem}_{suffix}.png"
        vis.save(out_path)
        saved.append(out_path)
    make_contact_sheet(saved, stage_dir / "contact_sheet.png", cols=contact_cols)
    return saved


def label_image(img: Image.Image, text: str) -> Image.Image:
    font = ImageFont.load_default()
    strip_h = 20
    out = Image.new("RGB", (img.width, img.height + strip_h), (245, 245, 245))
    out.paste(img.convert("RGB"), (0, strip_h))
    draw = ImageDraw.Draw(out)
    draw.text((5, 4), text, fill=(20, 20, 20), font=font)
    return out


def make_comparisons(
    scene_dir: Path,
    output_dir: Path,
    stage_to_images: Dict[str, Sequence[Path]],
    frame_paths: Sequence[Path],
    contact_cols: int,
) -> None:
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    stages = sorted(stage_to_images, key=natural_key)
    for i, frame_path in enumerate(frame_paths):
        panels = []
        with Image.open(frame_path) as img:
            panels.append(label_image(img.convert("RGB"), "RGB image"))
        for stage in stages:
            with Image.open(stage_to_images[stage][i]) as img:
                panels.append(label_image(img.convert("RGB"), f"{stage} PCA"))
        height = max(p.height for p in panels)
        padded = []
        for panel in panels:
            if panel.height != height:
                canvas = Image.new("RGB", (panel.width, height), (245, 245, 245))
                canvas.paste(panel, (0, 0))
                panel = canvas
            padded.append(panel)
        row = Image.new("RGB", (sum(p.width for p in padded), height), (245, 245, 245))
        x = 0
        for panel in padded:
            row.paste(panel, (x, 0))
            x += panel.width
        out_path = comparison_dir / f"{frame_path.stem}_all_stages.png"
        row.save(out_path)
        saved.append(out_path)
    make_contact_sheet(saved, comparison_dir / "contact_sheet.png", cols=max(1, min(2, contact_cols)))


def stage_names(scene_dirs: Sequence[Path]) -> List[str]:
    names = set()
    for scene_dir in scene_dirs:
        for file in token_files(scene_dir, include_backbone=True):
            names.add(file.stem.replace("_tokens", ""))
    return sorted(names, key=natural_key)


def output_for_scene(base_input: Path, base_output: Path, scene_dir: Path, multi_scene: bool) -> Path:
    if multi_scene:
        return base_output / scene_dir.name
    return base_output


def visualize_scene_with_models(
    scene_dir: Path,
    output_dir: Path,
    pca_models: Dict[str, dict],
    normalizers: Dict[str, dict],
    args: argparse.Namespace,
) -> Dict[str, List[Path]]:
    metadata = load_metadata(scene_dir)
    frame_paths = frame_paths_for_scene(scene_dir, metadata)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found for {scene_dir}")

    stage_outputs: Dict[str, List[Path]] = {}
    for stage_file in token_files(scene_dir, include_backbone=True):
        stage = stage_file.stem.replace("_tokens", "")
        tokens, _ = load_stage_tokens(scene_dir, stage_file)
        log(f"Rendering {scene_dir.name}/{stage}: tokens={tuple(tokens.shape)}")
        maps = robust_uint8(project_with_pca(tokens, pca_models[stage]))
        saved = save_maps(maps, output_dir / stage, frame_paths, "pca_rgb", args.contact_cols)
        stage_outputs[stage] = saved

        if not args.skip_controls:
            controls_root = Path(args.controls_output) if args.controls_output else scene_dir / "pca_vis_controls"
            shuffled = shuffle_tokens(tokens, args.seed)
            shuffled_norm = fit_normalizer([shuffled], args.normalization)
            shuffled_model = fit_pca([shuffled], shuffled_norm)
            shuffled_maps = robust_uint8(project_with_pca(shuffled, shuffled_model))
            save_maps(
                shuffled_maps,
                controls_root / "shuffled_tokens" / stage,
                frame_paths,
                "shuffled_pca_rgb",
                args.contact_cols,
            )

            random_maps = robust_uint8(project_random(tokens, normalizers[stage], args.seed + 1000))
            save_maps(
                random_maps,
                controls_root / "random_projection" / stage,
                frame_paths,
                "random_projection_rgb",
                args.contact_cols,
            )

    make_comparisons(scene_dir, output_dir, stage_outputs, frame_paths, args.contact_cols)

    latest = load_metadata(scene_dir) or metadata
    latest.setdefault("pca", {})
    for stage in stage_outputs:
        latest["pca"][stage] = True
    if not args.skip_controls:
        latest.setdefault("controls", {})
        latest["controls"]["shuffled_tokens"] = True
        latest["controls"]["random_projection"] = True
    save_metadata(scene_dir, latest)
    write_summary_md(scene_dir)
    return stage_outputs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    scene_dirs = scene_dirs_from_input(input_path)
    if not scene_dirs:
        raise FileNotFoundError(f"No scene token directories found under {input_path}")

    torch.manual_seed(args.seed)
    multi_scene = len(scene_dirs) > 1 or not (input_path / "tokens").is_dir()
    stages = stage_names(scene_dirs)
    if not stages:
        raise FileNotFoundError(f"No stage token files found under {input_path}")

    log(f"Fit scope: {args.fit_scope}")
    log(f"Scenes: {[p.name for p in scene_dirs]}")
    log(f"Stages: {stages}")

    if args.fit_scope == "global":
        pca_models: Dict[str, dict] = {}
        normalizers: Dict[str, dict] = {}
        for stage in stages:
            feature_sets = []
            for scene_dir in scene_dirs:
                file = scene_dir / "tokens" / f"{stage}_tokens.pt"
                if file.is_file():
                    tokens, _ = load_stage_tokens(scene_dir, file)
                    feature_sets.append(tokens)
            if not feature_sets:
                continue
            normalizer = fit_normalizer(feature_sets, args.normalization)
            pca_models[stage] = fit_pca(feature_sets, normalizer)
            normalizers[stage] = normalizer

        for scene_dir in scene_dirs:
            scene_output = output_for_scene(input_path, output_path, scene_dir, multi_scene)
            visualize_scene_with_models(scene_dir, scene_output, pca_models, normalizers, args)
    else:
        for scene_dir in scene_dirs:
            pca_models = {}
            normalizers = {}
            for stage_file in token_files(scene_dir, include_backbone=True):
                stage = stage_file.stem.replace("_tokens", "")
                tokens, _ = load_stage_tokens(scene_dir, stage_file)
                normalizer = fit_normalizer([tokens], args.normalization)
                pca_models[stage] = fit_pca([tokens], normalizer)
                normalizers[stage] = normalizer
            scene_output = output_for_scene(input_path, output_path, scene_dir, multi_scene)
            visualize_scene_with_models(scene_dir, scene_output, pca_models, normalizers, args)

    log(f"PCA visualization written to {output_path}")


if __name__ == "__main__":
    main()
