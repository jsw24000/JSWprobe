#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, load_config, write_json  # noqa: E402
from long_seq_memory.memory_index import preprocessed_patch_grid  # noqa: E402


SPECIAL_TOKENS = 6
DEFAULT_LAYERS = [4, 11, 17, 23]


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return default
    return [int(item) for item in value.split(",") if item.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Expected one file for {pattern}, found {len(matches)}")
    return matches[0]


def required_records(cfg: dict[str, Any], args: argparse.Namespace) -> list[tuple[int, int]]:
    extraction = cfg.get("extraction", {})
    frames = parse_int_list(args.frames, [int(v) for v in extraction.get("raw_qkv_frames", [])])
    layers = parse_int_list(args.layers, [int(v) for v in extraction.get("representative_layers", DEFAULT_LAYERS)])
    return [(frame, layer) for frame in frames for layer in layers]


def resolve_dirs(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    interaction_dir = Path(args.interaction_dir or cfg["interaction_output_dir"]).expanduser()
    default_output = (
        Path(cfg.get("analysis_output_dir", interaction_dir.parent / "analysis"))
        .expanduser()
        .parent
        / "memory_patch_feature_cosine"
    )
    return {
        "qkv_dir": Path(args.qkv_dir or interaction_dir / "raw_qkv_debug" / "qkv").expanduser(),
        "metadata_dir": Path(args.metadata_dir or interaction_dir / "raw_qkv_debug" / "token_metadata").expanduser(),
        "output_dir": Path(args.output_dir or default_output).expanduser(),
    }


def validate_inputs(records: list[tuple[int, int]], qkv_dir: Path, metadata_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = []
    missing = []
    for frame, layer in records:
        qkv_path = find_one(qkv_dir, f"qkv_frame{frame:06d}_layer{layer:02d}_*.npz")
        metadata_path = metadata_dir / f"frame{frame:06d}_layer{layer:02d}_token_metadata.json"
        item = {
            "frame_id": int(frame),
            "layer_id": int(layer),
            "qkv_path": str(qkv_path) if qkv_path else None,
            "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        }
        if qkv_path is None or not metadata_path.exists():
            missing.append(item)
        else:
            available.append(item)
    return available, missing


def infer_patch_grid(cfg: dict[str, Any], frame_id: int, patch_count: int) -> tuple[int, int]:
    dataset_root = Path(cfg.get("dataset_root", "")).expanduser()
    image_path = dataset_root / "images" / f"{frame_id:06d}.png"
    defaults = cfg.get("model_defaults", {})
    image_size = int(defaults.get("image_size", cfg.get("inference", {}).get("image_size", 518)))
    patch_size = int(defaults.get("patch_size", 14))
    if image_path.exists():
        try:
            from PIL import Image

            image = Image.open(image_path)
            grid = preprocessed_patch_grid(image.size[0], image.size[1], image_size=image_size, patch_size=patch_size)
            if grid[0] * grid[1] == patch_count:
                return grid
        except Exception:
            pass
    factors = [(w, patch_count // w) for w in range(1, patch_count + 1) if patch_count % w == 0]
    target_w = image_size // patch_size
    return min(factors, key=lambda wh: (abs(wh[0] - target_w), abs(wh[0] - wh[1])))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a.astype(np.float32)) @ l2_normalize(b.astype(np.float32)).T


def select_old_indices(old_frames: np.ndarray, max_old_frames: int, target_range: tuple[int, int]) -> np.ndarray:
    n = len(old_frames)
    if n <= max_old_frames:
        return np.arange(n, dtype=np.int64)
    keep: set[int] = set()
    keep.update(np.linspace(0, n - 1, max_old_frames, dtype=np.int64).tolist())
    target = np.where((old_frames >= target_range[0]) & (old_frames <= target_range[1]))[0]
    keep.update(target.tolist())
    keep.add(0)
    keep.add(n - 1)
    if len(keep) > max_old_frames:
        required = set(target.tolist()) | {0, n - 1}
        optional = sorted(keep - required)
        budget = max(max_old_frames - len(required), 0)
        if optional and budget:
            optional = [optional[i] for i in np.linspace(0, len(optional) - 1, budget, dtype=np.int64)]
        keep = required | set(optional[:budget])
    return np.asarray(sorted(keep), dtype=np.int64)


def sample_patch_indices(patch_count: int, count: int, seed: int) -> np.ndarray:
    if count >= patch_count:
        return np.arange(patch_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    base = {0, patch_count // 2, patch_count - 1}
    while len(base) < count:
        base.add(int(rng.integers(0, patch_count)))
    return np.asarray(sorted(base), dtype=np.int64)


def old_frame_features(tensor: np.ndarray, old_frame_count: int) -> np.ndarray:
    heads, old_tokens, dim = tensor.shape
    expected = old_frame_count * SPECIAL_TOKENS
    if old_tokens != expected:
        raise ValueError(f"Expected {expected} old tokens, got {old_tokens}")
    # [H, F, 6, D] -> [F, H, D] -> [F, H*D]
    return tensor.reshape(heads, old_frame_count, SPECIAL_TOKENS, dim).mean(axis=2).transpose(1, 0, 2).reshape(old_frame_count, heads * dim)


def sampled_patch_features(
    tensor: np.ndarray,
    patch_frame_count: int,
    patch_count: int,
    patch_indices: np.ndarray,
) -> np.ndarray:
    heads, tokens, dim = tensor.shape
    expected = patch_frame_count * patch_count
    if tokens != expected:
        raise ValueError(f"Expected {expected} patch tokens, got {tokens}")
    # [H, PF, P, D] -> [PF, selectedP, H, D] -> [PF*selectedP, H*D]
    return (
        tensor.reshape(heads, patch_frame_count, patch_count, dim)
        .transpose(1, 2, 0, 3)[:, patch_indices]
        .reshape(patch_frame_count * len(patch_indices), heads * dim)
    )


def sampled_patch_features_from_full_frames(
    tensor: np.ndarray,
    patch_frame_count: int,
    patch_count: int,
    patch_indices: np.ndarray,
) -> np.ndarray:
    heads, tokens, dim = tensor.shape
    tokens_per_frame = SPECIAL_TOKENS + patch_count
    expected = patch_frame_count * tokens_per_frame
    if tokens != expected:
        raise ValueError(f"Expected {expected} retained full-frame tokens, got {tokens}")
    # The SDPA visible-cache layout is [frame0_special, frame0_patches, frame1_special, frame1_patches, ...].
    return (
        tensor.reshape(heads, patch_frame_count, tokens_per_frame, dim)[:, :, SPECIAL_TOKENS:, :]
        .transpose(1, 2, 0, 3)[:, patch_indices]
        .reshape(patch_frame_count * len(patch_indices), heads * dim)
    )


def full_patch_features_for_frame(
    tensor: np.ndarray,
    patch_frame_index: int,
    patch_frame_count: int,
    patch_count: int,
) -> np.ndarray:
    heads, tokens, dim = tensor.shape
    expected = patch_frame_count * patch_count
    if tokens != expected:
        raise ValueError(f"Expected {expected} patch tokens, got {tokens}")
    return (
        tensor.reshape(heads, patch_frame_count, patch_count, dim)
        .transpose(1, 2, 0, 3)[patch_frame_index]
        .reshape(patch_count, heads * dim)
    )


def full_patch_features_for_frame_from_full_frames(
    tensor: np.ndarray,
    patch_frame_index: int,
    patch_frame_count: int,
    patch_count: int,
) -> np.ndarray:
    heads, tokens, dim = tensor.shape
    tokens_per_frame = SPECIAL_TOKENS + patch_count
    expected = patch_frame_count * tokens_per_frame
    if tokens != expected:
        raise ValueError(f"Expected {expected} retained full-frame tokens, got {tokens}")
    return (
        tensor.reshape(heads, patch_frame_count, tokens_per_frame, dim)[:, :, SPECIAL_TOKENS:, :]
        .transpose(1, 2, 0, 3)[patch_frame_index]
        .reshape(patch_count, heads * dim)
    )


def later_mask(old_frames: np.ndarray, patch_frames: np.ndarray) -> np.ndarray:
    return patch_frames[None, :] > old_frames[:, None]


def summarize_heatmap(values: np.ndarray, old_frames: np.ndarray, patch_frames: np.ndarray, target_range: tuple[int, int]) -> dict[str, Any]:
    mask = later_mask(old_frames, patch_frames)
    target_mask = (old_frames >= target_range[0]) & (old_frames <= target_range[1])
    out = {
        "later_mean": float(np.nanmean(np.where(mask, values, np.nan))),
        "later_median": float(np.nanmedian(np.where(mask, values, np.nan))),
        "later_p10": float(np.nanpercentile(np.where(mask, values, np.nan), 10)),
        "later_p90": float(np.nanpercentile(np.where(mask, values, np.nan), 90)),
        "target_history_later_mean": None,
        "non_target_history_later_mean": None,
        "target_minus_non_target": None,
        "max_later_pair": None,
    }
    if target_mask.any():
        target_values = np.where(mask & target_mask[:, None], values, np.nan)
        non_target_values = np.where(mask & ~target_mask[:, None], values, np.nan)
        target_mean = float(np.nanmean(target_values))
        non_target_mean = float(np.nanmean(non_target_values))
        out.update(
            {
                "target_history_later_mean": target_mean,
                "non_target_history_later_mean": non_target_mean,
                "target_minus_non_target": target_mean - non_target_mean,
            }
        )
    later_values = np.where(mask, values, np.nan)
    if np.isfinite(later_values).any():
        idx = np.nanargmax(later_values)
        row, col = np.unravel_index(idx, later_values.shape)
        out["max_later_pair"] = {
            "old_source_frame": int(old_frames[row]),
            "retained_patch_frame": int(patch_frames[col]),
            "cosine": float(values[row, col]),
        }
    return out


def compute_feature_heatmap(
    old_features: np.ndarray,
    patch_features: np.ndarray,
    old_indices: np.ndarray,
    patch_frame_count: int,
    patches_per_frame_sampled: int,
) -> np.ndarray:
    sim = cosine_matrix(old_features[old_indices], patch_features)
    return sim.reshape(len(old_indices), patch_frame_count, patches_per_frame_sampled).mean(axis=2)


def analyze_record(
    cfg: dict[str, Any],
    item: dict[str, Any],
    output_dir: Path,
    target_range: tuple[int, int],
    max_old_frames: int,
    patches_per_frame: int,
) -> dict[str, Any]:
    frame_id = int(item["frame_id"])
    layer_id = int(item["layer_id"])
    metadata = load_json(Path(item["metadata_path"]))
    old_frames_all = np.asarray(metadata["old_memory_frames"], dtype=np.int64)
    patch_frames = np.asarray(metadata["patch_memory_frames"], dtype=np.int64)
    old_frame_count = len(old_frames_all)
    patch_frame_count = len(patch_frames)
    old_token_count = int(metadata["old_memory_token_count"])

    qkv = np.load(Path(item["qkv_path"]))
    k = qkv["k"][0].astype(np.float32)
    v = qkv["v"][0].astype(np.float32)
    heads, total_tokens, dim = k.shape
    patch_count = (int(qkv["q"].shape[2]) - SPECIAL_TOKENS)
    grid_w, grid_h = infer_patch_grid(cfg, frame_id, patch_count)

    retained_start = old_token_count
    tokens_per_retained_frame = SPECIAL_TOKENS + patch_count
    retained_token_count = patch_frame_count * tokens_per_retained_frame
    retained_end = retained_start + retained_token_count
    if retained_end > total_tokens:
        raise ValueError(
            f"Cannot reconstruct retained full-frame span for frame {frame_id} layer {layer_id}: "
            f"old={old_token_count}, retained_tokens={retained_token_count}, total={total_tokens}"
        )
    current_context_tokens = total_tokens - retained_end

    old_indices = select_old_indices(old_frames_all, max_old_frames=max_old_frames, target_range=target_range)
    old_frames = old_frames_all[old_indices]
    patch_indices = sample_patch_indices(patch_count, patches_per_frame, seed=frame_id * 1000 + layer_id)
    actual_patches_per_frame = len(patch_indices)

    k_old = old_frame_features(k[:, :old_token_count, :], old_frame_count)
    v_old = old_frame_features(v[:, :old_token_count, :], old_frame_count)
    k_retained_full = k[:, retained_start:retained_end, :]
    v_retained_full = v[:, retained_start:retained_end, :]
    k_patch = sampled_patch_features_from_full_frames(k_retained_full, patch_frame_count, patch_count, patch_indices)
    v_patch = sampled_patch_features_from_full_frames(v_retained_full, patch_frame_count, patch_count, patch_indices)

    k_heatmap = compute_feature_heatmap(k_old, k_patch, old_indices, patch_frame_count, actual_patches_per_frame)
    v_heatmap = compute_feature_heatmap(v_old, v_patch, old_indices, patch_frame_count, actual_patches_per_frame)

    maps_dir = ensure_dir(output_dir / "maps")
    np.savez_compressed(
        maps_dir / f"memory_patch_cosine_frame{frame_id:06d}_layer{layer_id:02d}.npz",
        old_source_frames=old_frames,
        retained_patch_frames=patch_frames,
        k_cosine_old_source_by_patch_frame=k_heatmap.astype(np.float32),
        v_cosine_old_source_by_patch_frame=v_heatmap.astype(np.float32),
        patch_indices=patch_indices,
    )

    spatial = build_spatial_maps(
        cfg=cfg,
        frame_id=frame_id,
        layer_id=layer_id,
        target_range=target_range,
        old_frames_all=old_frames_all,
        patch_frames=patch_frames,
        k_old=k_old,
        v_old=v_old,
        k_patch_span=k_retained_full,
        v_patch_span=v_retained_full,
        patch_frame_count=patch_frame_count,
        patch_count=patch_count,
        grid_wh=(grid_w, grid_h),
    )
    if spatial is not None:
        np.savez_compressed(maps_dir / f"memory_patch_spatial_frame{frame_id:06d}_layer{layer_id:02d}.npz", **spatial)

    record = {
        "frame_id": frame_id,
        "layer_id": layer_id,
        "qkv_path": item["qkv_path"],
        "metadata_path": item["metadata_path"],
        "heads": int(heads),
        "head_dim": int(dim),
        "patch_grid_wh": [int(grid_w), int(grid_h)],
        "old_memory_frame_count": int(old_frame_count),
        "old_source_frames_sampled": int(len(old_indices)),
        "retained_patch_frame_count": int(patch_frame_count),
        "patches_per_frame_sampled": int(actual_patches_per_frame),
        "old_frame_range": [int(old_frames_all.min()), int(old_frames_all.max())],
        "retained_patch_frame_range": [int(patch_frames.min()), int(patch_frames.max())],
        "tokens_per_retained_frame": int(tokens_per_retained_frame),
        "current_context_tokens_after_retained_span": int(current_context_tokens),
        "k": summarize_heatmap(k_heatmap, old_frames, patch_frames, target_range),
        "v": summarize_heatmap(v_heatmap, old_frames, patch_frames, target_range),
    }
    if spatial is not None:
        record["spatial_map_sources"] = spatial["source_frames"].astype(int).tolist()
        record["spatial_map_patch_frame"] = int(spatial["patch_frame"])
    return record


def build_spatial_maps(
    cfg: dict[str, Any],
    frame_id: int,
    layer_id: int,
    target_range: tuple[int, int],
    old_frames_all: np.ndarray,
    patch_frames: np.ndarray,
    k_old: np.ndarray,
    v_old: np.ndarray,
    k_patch_span: np.ndarray,
    v_patch_span: np.ndarray,
    patch_frame_count: int,
    patch_count: int,
    grid_wh: tuple[int, int],
) -> dict[str, Any] | None:
    del cfg, layer_id
    later_patch_frames = patch_frames[patch_frames > old_frames_all.min()]
    if later_patch_frames.size == 0:
        return None
    patch_frame = int(patch_frames[-1])
    patch_frame_index = int(np.where(patch_frames == patch_frame)[0][0])
    selected_sources = [int(old_frames_all[0]), int(old_frames_all[-1])]
    target = old_frames_all[(old_frames_all >= target_range[0]) & (old_frames_all <= target_range[1])]
    if target.size:
        selected_sources.insert(1, int(target[np.argmin(np.abs(target - ((target_range[0] + target_range[1]) // 2)))]))
    selected_sources = list(dict.fromkeys(selected_sources))
    old_index = {int(frame): idx for idx, frame in enumerate(old_frames_all.tolist())}
    source_indices = np.asarray([old_index[source] for source in selected_sources], dtype=np.int64)

    k_patch_full = full_patch_features_for_frame_from_full_frames(k_patch_span, patch_frame_index, patch_frame_count, patch_count)
    v_patch_full = full_patch_features_for_frame_from_full_frames(v_patch_span, patch_frame_index, patch_frame_count, patch_count)
    k_sim = cosine_matrix(k_old[source_indices], k_patch_full)
    v_sim = cosine_matrix(v_old[source_indices], v_patch_full)
    grid_w, grid_h = grid_wh
    return {
        "frame_id": np.asarray(frame_id, dtype=np.int64),
        "patch_frame": np.asarray(patch_frame, dtype=np.int64),
        "source_frames": np.asarray(selected_sources, dtype=np.int64),
        "k_spatial_cosine": k_sim.reshape(len(selected_sources), grid_h, grid_w).astype(np.float32),
        "v_spatial_cosine": v_sim.reshape(len(selected_sources), grid_h, grid_w).astype(np.float32),
    }


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "font.size": 10,
        }
    )
    return plt


def draw_heatmap(ax: Any, matrix: np.ndarray, x_frames: np.ndarray, y_frames: np.ndarray, title: str) -> Any:
    data = np.asarray(matrix, dtype=np.float64)
    im = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("retained patch frame")
    ax.set_ylabel("old source frame")
    xticks = np.linspace(0, len(x_frames) - 1, min(8, len(x_frames)), dtype=int)
    yticks = np.linspace(0, len(y_frames) - 1, min(8, len(y_frames)), dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x_frames[i])) for i in xticks], rotation=35, ha="right")
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(y_frames[i])) for i in yticks])
    return im


def plot_selected_heatmap(output_dir: Path, frame_id: int, layer_id: int) -> str | None:
    plt = setup_matplotlib()
    path = output_dir / "maps" / f"memory_patch_cosine_frame{frame_id:06d}_layer{layer_id:02d}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    old_frames = z["old_source_frames"]
    patch_frames = z["retained_patch_frames"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "k_cosine_old_source_by_patch_frame", "K cosine"),
        (axes[1], "v_cosine_old_source_by_patch_frame", "V cosine"),
    ]:
        im = draw_heatmap(ax, z[key], patch_frames, old_frames, f"frame {frame_id} layer {layer_id} | {title}")
        fig.colorbar(im, ax=ax, label="cosine")
    figures_dir = ensure_dir(output_dir / "figures")
    out = figures_dir / f"memory_patch_cosine_heatmap_frame{frame_id:06d}_layer{layer_id:02d}.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_layer11_frames(output_dir: Path, frames: list[int]) -> str | None:
    plt = setup_matplotlib()
    rows = []
    for frame in frames:
        path = output_dir / "maps" / f"memory_patch_cosine_frame{frame:06d}_layer11.npz"
        if path.exists():
            rows.append((frame, np.load(path)))
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.3 * len(rows)), squeeze=False, constrained_layout=True)
    for row_idx, (frame, z) in enumerate(rows):
        for col_idx, key in enumerate(["k_cosine_old_source_by_patch_frame", "v_cosine_old_source_by_patch_frame"]):
            ax = axes[row_idx, col_idx]
            title = ("K" if key.startswith("k_") else "V") + f" cosine | frame {frame} L11"
            im = draw_heatmap(ax, z[key], z["retained_patch_frames"], z["old_source_frames"], title)
            fig.colorbar(im, ax=ax, label="cosine")
    figures_dir = ensure_dir(output_dir / "figures")
    out = figures_dir / "memory_patch_cosine_layer11_frames_montage.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_frame_layers(output_dir: Path, frame_id: int, layers: list[int]) -> str | None:
    plt = setup_matplotlib()
    rows = []
    for layer in layers:
        path = output_dir / "maps" / f"memory_patch_cosine_frame{frame_id:06d}_layer{layer:02d}.npz"
        if path.exists():
            rows.append((layer, np.load(path)))
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.3 * len(rows)), squeeze=False, constrained_layout=True)
    for row_idx, (layer, z) in enumerate(rows):
        for col_idx, key in enumerate(["k_cosine_old_source_by_patch_frame", "v_cosine_old_source_by_patch_frame"]):
            ax = axes[row_idx, col_idx]
            title = ("K" if key.startswith("k_") else "V") + f" cosine | frame {frame_id} L{layer}"
            im = draw_heatmap(ax, z[key], z["retained_patch_frames"], z["old_source_frames"], title)
            fig.colorbar(im, ax=ax, label="cosine")
    figures_dir = ensure_dir(output_dir / "figures")
    out = figures_dir / f"memory_patch_cosine_frame{frame_id:06d}_layers_montage.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_summary_bars(output_dir: Path, records: list[dict[str, Any]]) -> str:
    plt = setup_matplotlib()
    ordered = sorted(records, key=lambda r: (r["frame_id"], r["layer_id"]))
    labels = [f"{r['frame_id']}\nL{r['layer_id']}" for r in ordered]
    x = np.arange(len(ordered))
    k_mean = [r["k"]["later_mean"] for r in ordered]
    v_mean = [r["v"]["later_mean"] for r in ordered]
    k_target_delta = [r["k"]["target_minus_non_target"] if r["k"]["target_minus_non_target"] is not None else np.nan for r in ordered]
    v_target_delta = [r["v"]["target_minus_non_target"] if r["v"]["target_minus_non_target"] is not None else np.nan for r in ordered]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
    width = 0.38
    axes[0].bar(x - width / 2, k_mean, width=width, label="K later mean")
    axes[0].bar(x + width / 2, v_mean, width=width, label="V later mean")
    axes[0].set_title("Old Memory Feature vs Later Retained Patch Tokens")
    axes[0].set_ylabel("mean cosine")
    axes[0].legend()
    axes[1].bar(x - width / 2, k_target_delta, width=width, label="K target minus non-target")
    axes[1].bar(x + width / 2, v_target_delta, width=width, label="V target minus non-target")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Target History Segment Bias")
    axes[1].set_ylabel("cosine delta")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    figures_dir = ensure_dir(output_dir / "figures")
    out = figures_dir / "memory_patch_cosine_summary_bars.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_layer_profile(output_dir: Path, records: list[dict[str, Any]]) -> str:
    plt = setup_matplotlib()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["layer_id"])].append(record)
    layers = sorted(grouped)
    k = [float(np.mean([r["k"]["later_mean"] for r in grouped[layer]])) for layer in layers]
    v = [float(np.mean([r["v"]["later_mean"] for r in grouped[layer]])) for layer in layers]
    kt = [float(np.nanmean([r["k"]["target_minus_non_target"] for r in grouped[layer] if r["k"]["target_minus_non_target"] is not None])) for layer in layers]
    vt = [float(np.nanmean([r["v"]["target_minus_non_target"] for r in grouped[layer] if r["v"]["target_minus_non_target"] is not None])) for layer in layers]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(layers, k, marker="o", label="K")
    axes[0].plot(layers, v, marker="o", label="V")
    axes[0].set_title("Mean Cosine By Layer")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("mean cosine")
    axes[0].legend()
    axes[1].plot(layers, kt, marker="o", label="K")
    axes[1].plot(layers, vt, marker="o", label="V")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Target History Bias By Layer")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("target - non-target")
    axes[1].legend()
    figures_dir = ensure_dir(output_dir / "figures")
    out = figures_dir / "memory_patch_cosine_layer_profile.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def plot_spatial_maps(output_dir: Path, frame_id: int, layer_id: int) -> str | None:
    plt = setup_matplotlib()
    path = output_dir / "maps" / f"memory_patch_spatial_frame{frame_id:06d}_layer{layer_id:02d}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    sources = z["source_frames"]
    k = z["k_spatial_cosine"]
    v = z["v_spatial_cosine"]
    fig, axes = plt.subplots(len(sources), 2, figsize=(9, 2.8 * len(sources)), squeeze=False, constrained_layout=True)
    for row, source in enumerate(sources):
        for col, arr in enumerate([k[row], v[row]]):
            ax = axes[row, col]
            im = ax.imshow(arr, aspect="auto")
            ax.set_title(("K" if col == 0 else "V") + f" cosine | old {int(source)} -> patch {int(z['patch_frame'])}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    figures_dir = ensure_dir(output_dir / "figures")
    out = figures_dir / f"memory_patch_cosine_spatial_frame{frame_id:06d}_layer{layer_id:02d}.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.nanmean(arr)),
            "median": float(np.nanmedian(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "p10": float(np.nanpercentile(arr, 10)),
            "p90": float(np.nanpercentile(arr, 90)),
        }

    out = {
        "num_records": len(records),
        "k_later_mean": stats([r["k"]["later_mean"] for r in records]),
        "v_later_mean": stats([r["v"]["later_mean"] for r in records]),
        "k_target_minus_non_target": stats(
            [r["k"]["target_minus_non_target"] for r in records if r["k"]["target_minus_non_target"] is not None]
        ),
        "v_target_minus_non_target": stats(
            [r["v"]["target_minus_non_target"] for r in records if r["v"]["target_minus_non_target"] is not None]
        ),
    }
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["layer_id"])].append(record)
    out["by_layer"] = [
        {
            "layer_id": layer,
            "k_later_mean": float(np.mean([r["k"]["later_mean"] for r in rows])),
            "v_later_mean": float(np.mean([r["v"]["later_mean"] for r in rows])),
            "k_target_minus_non_target": float(
                np.nanmean([r["k"]["target_minus_non_target"] for r in rows if r["k"]["target_minus_non_target"] is not None])
            ),
            "v_target_minus_non_target": float(
                np.nanmean([r["v"]["target_minus_non_target"] for r in rows if r["v"]["target_minus_non_target"] is not None])
            ),
        }
        for layer, rows in sorted(grouped.items())
    ]
    return out


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    agg = summary["aggregate"]
    lines = [
        "# Memory Patch Feature Cosine",
        "",
        f"- Status: `{summary['status']}`",
        f"- Records: {agg['num_records']}",
        f"- Old source frames sampled per record: up to {summary['max_old_frames']}",
        f"- Patch tokens sampled per retained frame: {summary['patches_per_frame']}",
        "",
        "## Interpretation",
        "",
        summary["current_answer"],
        "",
        "## Key Numbers",
        "",
        f"- K later mean cosine: {agg['k_later_mean']['mean']:.6f}",
        f"- V later mean cosine: {agg['v_later_mean']['mean']:.6f}",
        f"- K target-history delta: {agg['k_target_minus_non_target']['mean']:.6f}",
        f"- V target-history delta: {agg['v_target_minus_non_target']['mean']:.6f}",
        "",
        "## Layer Means",
        "",
    ]
    for row in agg["by_layer"]:
        lines.append(
            f"- layer {row['layer_id']}: "
            f"K={row['k_later_mean']:.4f}, V={row['v_later_mean']:.4f}, "
            f"K_target_delta={row['k_target_minus_non_target']:.4f}, "
            f"V_target_delta={row['v_target_minus_non_target']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Patch tokens are sampled for frame-level heatmaps; selected spatial maps use all patch tokens for one retained patch frame.",
            "- Old memory feature is the mean of the six special memory tokens per old source frame.",
        "- The retained cached span is reconstructed as full frames with 6 special tokens followed by patch tokens; current-frame context after that retained span is not analyzed.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--interaction-dir", default=None)
    parser.add_argument("--qkv-dir", default=None)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--max-old-frames", type=int, default=128)
    parser.add_argument("--patches-per-frame", type=int, default=32)
    parser.add_argument("--selected-frame", type=int, default=4414)
    parser.add_argument("--selected-layer", type=int, default=11)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dirs = resolve_dirs(cfg, args)
    output_dir = ensure_dir(dirs["output_dir"])
    records = required_records(cfg, args)
    available, missing = validate_inputs(records, dirs["qkv_dir"], dirs["metadata_dir"])
    if missing:
        summary = {
            "status": "missing_qkv_or_metadata",
            "missing": missing,
            "available_count": len(available),
            "required_count": len(records),
            "current_answer": "Required raw QKV or token metadata is missing; cosine analysis was not run.",
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        raise SystemExit(2)

    target_range = tuple(int(v) for v in cfg.get("extraction", {}).get("target_history_range", [3370, 3440]))
    per_record = []
    for idx, item in enumerate(available, start=1):
        print(f"[{idx}/{len(available)}] frame {item['frame_id']} layer {item['layer_id']}", flush=True)
        per_record.append(
            analyze_record(
                cfg=cfg,
                item=item,
                output_dir=output_dir,
                target_range=target_range,
                max_old_frames=args.max_old_frames,
                patches_per_frame=args.patches_per_frame,
            )
        )

    figures = [
        plot_summary_bars(output_dir, per_record),
        plot_layer_profile(output_dir, per_record),
    ]
    for maybe in [
        plot_selected_heatmap(output_dir, args.selected_frame, args.selected_layer),
        plot_layer11_frames(output_dir, sorted({int(r["frame_id"]) for r in per_record})),
        plot_frame_layers(output_dir, args.selected_frame, sorted({int(r["layer_id"]) for r in per_record})),
        plot_spatial_maps(output_dir, args.selected_frame, args.selected_layer),
    ]:
        if maybe is not None:
            figures.append(maybe)

    agg = aggregate(per_record)
    current_answer = (
        "Direct K/V cosine between sampled old-memory special features and later retained patch-image tokens was computed. "
        f"K-space mean cosine is {agg['k_later_mean']['mean']:.4f}; V-space mean cosine is {agg['v_later_mean']['mean']:.4f}. "
        "Target loop-history features are compared against non-target old history via the target-history delta."
    )
    summary = {
        "status": "complete",
        "config": str(Path(args.config).expanduser()),
        "qkv_dir": str(dirs["qkv_dir"]),
        "metadata_dir": str(dirs["metadata_dir"]),
        "output_dir": str(output_dir),
        "target_history_range": list(target_range),
        "max_old_frames": int(args.max_old_frames),
        "patches_per_frame": int(args.patches_per_frame),
        "aggregate": agg,
        "records": per_record,
        "figures": figures,
        "current_answer": current_answer,
    }
    write_json(output_dir / "per_record_metrics.json", per_record)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary)
    print(f"Wrote memory patch feature cosine summary: {output_dir / 'summary.json'}")
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
