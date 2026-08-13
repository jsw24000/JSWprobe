#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.io import load_config, output_root, write_csv, write_json  # noqa: E402
from revisit_memory.utils.visualization import resize_mask_to_grid  # noqa: E402


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("image_token_pca_vis.py requires torch to load feature .pt files.") from exc


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_features(cfg: dict[str, Any], condition: str) -> dict[str, Any]:
    torch = require_torch()
    path = output_root(cfg) / "features" / condition / "features.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return torch.load(path, map_location="cpu")


def tensor_for(payload: dict[str, Any], group: str, layer: int, frame_id: int) -> np.ndarray:
    frame_ids = list(payload["frame_ids"])
    idx = frame_ids.index(frame_id)
    return payload[group][str(layer)][idx].float().numpy()


def fit_pca(tokens: np.ndarray, n_components: int = 3) -> dict[str, np.ndarray]:
    x = np.asarray(tokens, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components].astype(np.float32)
    # Make signs deterministic so reruns do not arbitrarily invert colors.
    for idx in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[idx])))
        if components[idx, pivot] < 0:
            components[idx] *= -1.0
    denom = max(x.shape[0] - 1, 1)
    explained = (s[:n_components] ** 2) / denom
    total = float(np.sum((s**2) / denom))
    ratio = explained / max(total, 1.0e-12)
    return {
        "mean": mean.astype(np.float32),
        "components": components,
        "explained_variance": explained.astype(np.float32),
        "explained_variance_ratio": ratio.astype(np.float32),
    }


def project(tokens: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(tokens, dtype=np.float32)
    return (x - pca["mean"]) @ pca["components"].T


def robust_rgb(projected: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rgb = (projected - lo.reshape(1, 3)) / np.maximum((hi - lo).reshape(1, 3), 1.0e-6)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def signed_heatmap_rgb(grid_values: np.ndarray, limit: float, pixel_hw: tuple[int, int], mode: str) -> np.ndarray:
    scale = max(float(limit), 1.0e-6)
    normalized = np.clip((grid_values / scale + 1.0) * 0.5, 0.0, 1.0)
    rgb = plt.get_cmap("coolwarm")(normalized)[..., :3].astype(np.float32)
    return upsample_rgb(rgb, pixel_hw, mode)


def intensity_heatmap_rgb(grid_values: np.ndarray, limit: float, pixel_hw: tuple[int, int], mode: str) -> np.ndarray:
    scale = max(float(limit), 1.0e-6)
    normalized = np.clip(grid_values / scale, 0.0, 1.0)
    rgb = plt.get_cmap("magma")(normalized)[..., :3].astype(np.float32)
    return upsample_rgb(rgb, pixel_hw, mode)


def upsample_rgb(grid_rgb: np.ndarray, pixel_hw: tuple[int, int], mode: str) -> np.ndarray:
    grid = np.clip(grid_rgb * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(grid, mode="RGB")
    resample = Image.Resampling.BILINEAR if mode == "bilinear" else Image.Resampling.NEAREST
    return np.asarray(img.resize((pixel_hw[1], pixel_hw[0]), resample=resample), dtype=np.uint8)


def upsample_mask(mask_grid: np.ndarray, pixel_hw: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask_grid > 0).astype(np.uint8) * 255, mode="L")
    return np.asarray(img.resize((pixel_hw[1], pixel_hw[0]), resample=Image.Resampling.NEAREST)) > 0


def save_rgb_with_mask(path: Path, rgb: np.ndarray, mask: np.ndarray, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    ax.imshow(rgb)
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.0)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_mask_grids(cfg: dict[str, Any], frames: list[int], grid_hw: tuple[int, int]) -> dict[int, np.ndarray]:
    root = output_root(cfg)
    with np.load(root / "reconstruction" / "seen_then_removed" / "inputs.npz", allow_pickle=True) as inp:
        frame_ids = inp["frame_ids"].astype(int).tolist()
        masks = {}
        for frame in frames:
            idx = frame_ids.index(frame)
            masks[frame] = resize_mask_to_grid(inp["counterfactual_target_mask"][idx], grid_hw)
    return masks


def make_contact_sheet(
    out_path: Path,
    frames: list[int],
    columns: list[tuple[str, str]],
    rgb_maps: dict[tuple[str, int], np.ndarray],
    mask_maps: dict[int, np.ndarray],
) -> None:
    fig, axes = plt.subplots(len(frames), len(columns), figsize=(4.3 * len(columns), 3.15 * len(frames)), constrained_layout=True)
    if len(frames) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(columns) == 1:
        axes = np.expand_dims(axes, axis=1)
    for row, frame in enumerate(frames):
        for col, (key, label) in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(rgb_maps[(key, frame)])
            mask = mask_maps[frame]
            if mask.any():
                ax.contour(mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
            ax.set_title(f"{frame} · {label}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(cfg: dict[str, Any], frames: list[int], layer: int, blocks: list[str], upsample_mode: str) -> None:
    payloads = {
        "always_absent": load_features(cfg, "always_absent"),
        "seen_then_removed": load_features(cfg, "seen_then_removed"),
    }
    common_frames = sorted(set(payloads["always_absent"]["frame_ids"]).intersection(payloads["seen_then_removed"]["frame_ids"]))
    frames = [frame for frame in frames if frame in common_frames]
    if not frames:
        raise RuntimeError("None of the requested frames are present in both feature files.")

    grid_hw = tuple(int(x) for x in payloads["seen_then_removed"]["token_grid_hw"])
    patch_size = int(cfg["model"]["patch_size"])
    pixel_hw = (grid_hw[0] * patch_size, grid_hw[1] * patch_size)
    mask_grids = load_mask_grids(cfg, frames, grid_hw)
    mask_pixels = {frame: upsample_mask(mask_grids[frame], pixel_hw) for frame in frames}

    root = output_root(cfg)
    base_out = root / "analysis" / "image_token_pca" / f"layer_{layer:02d}_frames_{frames[0]:04d}_{frames[-1]:04d}"
    rows: list[dict[str, Any]] = []

    for block in blocks:
        absent_by_frame = {frame: tensor_for(payloads["always_absent"], block, layer, frame) for frame in frames}
        removed_by_frame = {frame: tensor_for(payloads["seen_then_removed"], block, layer, frame) for frame in frames}
        diff_by_frame = {frame: removed_by_frame[frame] - absent_by_frame[frame] for frame in frames}

        condition_tokens = np.concatenate(
            [absent_by_frame[frame] for frame in frames] + [removed_by_frame[frame] for frame in frames],
            axis=0,
        )
        diff_tokens = np.concatenate([diff_by_frame[frame] for frame in frames], axis=0)
        condition_pca = fit_pca(condition_tokens)
        diff_pca = fit_pca(diff_tokens)

        condition_projected = np.concatenate(
            [project(absent_by_frame[frame], condition_pca) for frame in frames]
            + [project(removed_by_frame[frame], condition_pca) for frame in frames],
            axis=0,
        )
        diff_projected = np.concatenate([project(diff_by_frame[frame], diff_pca) for frame in frames], axis=0)
        condition_lo = np.percentile(condition_projected, 1, axis=0).astype(np.float32)
        condition_hi = np.percentile(condition_projected, 99, axis=0).astype(np.float32)
        diff_lo = np.percentile(diff_projected, 1, axis=0).astype(np.float32)
        diff_hi = np.percentile(diff_projected, 99, axis=0).astype(np.float32)
        diff_component_limits = np.percentile(np.abs(diff_projected), 99, axis=0).astype(np.float32)

        block_out = base_out / block
        block_out.mkdir(parents=True, exist_ok=True)
        rgb_maps: dict[tuple[str, int], np.ndarray] = {}
        component_maps: dict[tuple[str, int], np.ndarray] = {}
        component_intensity_maps: dict[tuple[str, int], np.ndarray] = {}
        for frame in frames:
            absent_rgb_grid = robust_rgb(project(absent_by_frame[frame], condition_pca), condition_lo, condition_hi).reshape(
                (*grid_hw, 3)
            )
            removed_rgb_grid = robust_rgb(project(removed_by_frame[frame], condition_pca), condition_lo, condition_hi).reshape(
                (*grid_hw, 3)
            )
            diff_projection = project(diff_by_frame[frame], diff_pca)
            diff_rgb_grid = robust_rgb(diff_projection, diff_lo, diff_hi).reshape((*grid_hw, 3))
            diff_component_grid = diff_projection.reshape((*grid_hw, 3))
            rgb_maps[("always_absent", frame)] = upsample_rgb(absent_rgb_grid, pixel_hw, upsample_mode)
            rgb_maps[("seen_then_removed", frame)] = upsample_rgb(removed_rgb_grid, pixel_hw, upsample_mode)
            rgb_maps[("removed_minus_absent", frame)] = upsample_rgb(diff_rgb_grid, pixel_hw, upsample_mode)
            for component_idx in range(3):
                key = f"diff_pc{component_idx + 1}"
                limit = float(diff_component_limits[component_idx])
                component_maps[(key, frame)] = signed_heatmap_rgb(
                    diff_component_grid[..., component_idx],
                    limit,
                    pixel_hw,
                    upsample_mode,
                )
                save_rgb_with_mask(
                    block_out / f"frame_{frame:04d}_removed_minus_absent_diff_pc{component_idx + 1}_heatmap.png",
                    component_maps[(key, frame)],
                    mask_pixels[frame],
                    title=(
                        f"Layer {layer} {block} removed-absent diff "
                        f"PC{component_idx + 1} signed projection +/- {limit:.4g}"
                    ),
                )
                intensity_key = f"diff_pc{component_idx + 1}_abs"
                component_intensity_maps[(intensity_key, frame)] = intensity_heatmap_rgb(
                    np.abs(diff_component_grid[..., component_idx]),
                    limit,
                    pixel_hw,
                    upsample_mode,
                )
                save_rgb_with_mask(
                    block_out / f"frame_{frame:04d}_removed_minus_absent_diff_pc{component_idx + 1}_abs_intensity_heatmap.png",
                    component_intensity_maps[(intensity_key, frame)],
                    mask_pixels[frame],
                    title=(
                        f"Layer {layer} {block} removed-absent diff "
                        f"|PC{component_idx + 1} projection| 0..{limit:.4g}"
                    ),
                )
            for key, label in [
                ("always_absent", "always_absent"),
                ("seen_then_removed", "seen_then_removed"),
                ("removed_minus_absent", "removed-absent diff"),
            ]:
                save_rgb_with_mask(
                    block_out / f"frame_{frame:04d}_{key}_pca_rgb.png",
                    rgb_maps[(key, frame)],
                    mask_pixels[frame],
                    title=f"Layer {layer} {block} {label} frame {frame}",
                )

        make_contact_sheet(
            block_out / f"layer{layer:02d}_{block}_absent_removed_diff_pca_rgb_contact_sheet.png",
            frames,
            [
                ("always_absent", "absent"),
                ("seen_then_removed", "removed"),
                ("removed_minus_absent", "diff PCA"),
            ],
            rgb_maps,
            mask_pixels,
        )
        make_contact_sheet(
            block_out / f"layer{layer:02d}_{block}_removed_absent_diff_pca_component_heatmaps_contact_sheet.png",
            frames,
            [
                ("diff_pc1", f"diff PC1 +/- {float(diff_component_limits[0]):.3g}"),
                ("diff_pc2", f"diff PC2 +/- {float(diff_component_limits[1]):.3g}"),
                ("diff_pc3", f"diff PC3 +/- {float(diff_component_limits[2]):.3g}"),
            ],
            component_maps,
            mask_pixels,
        )
        make_contact_sheet(
            block_out / f"layer{layer:02d}_{block}_removed_absent_diff_pca_component_abs_intensity_contact_sheet.png",
            frames,
            [
                ("diff_pc1_abs", f"|diff PC1| 0..{float(diff_component_limits[0]):.3g}"),
                ("diff_pc2_abs", f"|diff PC2| 0..{float(diff_component_limits[1]):.3g}"),
                ("diff_pc3_abs", f"|diff PC3| 0..{float(diff_component_limits[2]):.3g}"),
            ],
            component_intensity_maps,
            mask_pixels,
        )

        rows.append(
            {
                "layer": layer,
                "block_type": block,
                "frames": ",".join(str(frame) for frame in frames),
                "condition_pca_fit_tokens": "concat(always_absent,seen_then_removed)_all_requested_frames",
                "diff_pca_fit_tokens": "seen_then_removed_minus_always_absent_all_requested_frames",
                "condition_pca_explained_variance_ratio_pc1": float(condition_pca["explained_variance_ratio"][0]),
                "condition_pca_explained_variance_ratio_pc2": float(condition_pca["explained_variance_ratio"][1]),
                "condition_pca_explained_variance_ratio_pc3": float(condition_pca["explained_variance_ratio"][2]),
                "diff_pca_explained_variance_ratio_pc1": float(diff_pca["explained_variance_ratio"][0]),
                "diff_pca_explained_variance_ratio_pc2": float(diff_pca["explained_variance_ratio"][1]),
                "diff_pca_explained_variance_ratio_pc3": float(diff_pca["explained_variance_ratio"][2]),
                "condition_rgb_percentile_low": condition_lo.tolist(),
                "condition_rgb_percentile_high": condition_hi.tolist(),
                "diff_rgb_percentile_low": diff_lo.tolist(),
                "diff_rgb_percentile_high": diff_hi.tolist(),
                "diff_component_signed_abs_p99_pc1": float(diff_component_limits[0]),
                "diff_component_signed_abs_p99_pc2": float(diff_component_limits[1]),
                "diff_component_signed_abs_p99_pc3": float(diff_component_limits[2]),
                "contact_sheet": str(block_out / f"layer{layer:02d}_{block}_absent_removed_diff_pca_rgb_contact_sheet.png"),
                "component_heatmap_contact_sheet": str(
                    block_out / f"layer{layer:02d}_{block}_removed_absent_diff_pca_component_heatmaps_contact_sheet.png"
                ),
                "component_abs_intensity_contact_sheet": str(
                    block_out / f"layer{layer:02d}_{block}_removed_absent_diff_pca_component_abs_intensity_contact_sheet.png"
                ),
            }
        )

    write_csv(base_out / "pca_summary.csv", rows)
    write_json(
        base_out / "metadata.json",
        {
            "frames": frames,
            "layer": layer,
            "blocks": blocks,
            "grid_hw": list(grid_hw),
            "pixel_hw": list(pixel_hw),
            "upsample_mode": upsample_mode,
            "comparison_note": (
                "For absent and removed condition visualizations, PCA is fit jointly on both conditions "
                "over all requested frames for each layer/block. For removed-minus-absent vectors, PCA is "
                "fit separately on all requested-frame difference tokens. Diff component heatmaps include signed "
                "scalar projections onto each of the first three diff PCA directions, plus absolute-projection "
                "intensity maps. Both use one 99th-percentile absolute-value color scale per component across "
                "all requested frames."
            ),
            "summary_csv": str(base_out / "pca_summary.csv"),
        },
    )
    print(f"Wrote {base_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PCA RGB visualization for image patch tokens.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    parser.add_argument("--frames", default="68,72,74,76,80,83")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--blocks", default="frame_block,global_block")
    parser.add_argument("--upsample-mode", choices=["nearest", "bilinear"], default="bilinear")
    args = parser.parse_args()
    cfg = load_config(args.config)
    blocks = [part.strip() for part in args.blocks.split(",") if part.strip()]
    run(cfg, parse_int_list(args.frames), int(args.layer), blocks, args.upsample_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
