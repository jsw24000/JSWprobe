"""Shared PCA patch-map visualization across related controlled conditions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .base_analysis import (
    AnalysisRun,
    BaseAnalysisMethod,
    infer_patch_grid,
    iter_layer_token_refs,
    patch_tokens_by_frame,
)
from .registry import register_analysis_method


@register_analysis_method("shared_pca_patchmap")
class SharedPCAPatchMapAnalysis(BaseAnalysisMethod):
    """Project patch tokens onto a PCA basis shared across related conditions."""

    supports_group_run = True

    def run(self, run: AnalysisRun) -> dict[str, Any]:
        return self.run_group([run])[0]

    def run_group(self, runs: list[AnalysisRun]) -> list[dict[str, Any]]:
        if not runs:
            return []

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grouped: dict[tuple[str, str, str, str], list[AnalysisRun]] = defaultdict(list)
        for run in runs:
            metadata = run.manifest.get("metadata", {})
            key = (
                run.bundle.model_name,
                str(run.manifest.get("scene_id")),
                str(run.manifest.get("setting_name")),
                str(metadata.get("pair_group_id") or metadata.get("pair_type") or metadata.get("anchor_frame_id") or "all"),
            )
            grouped[key].append(run)

        results: list[dict[str, Any]] = []
        for _group_key, group_runs in grouped.items():
            results.extend(self._run_one_group(group_runs, plt))
        return results

    def _run_one_group(self, runs: list[AnalysisRun], plt: Any) -> list[dict[str, Any]]:
        patch_refs_by_layer: dict[str, list[tuple[AnalysisRun, str, Any, dict[str, Any]]]] = defaultdict(list)
        for run in runs:
            for layer_name, token_type, ref, layer_meta in iter_layer_token_refs(
                run.bundle,
                token_types=["patch_tokens", "d1_stage0_patch_tokens"],
            ):
                if token_type in {"patch_tokens", "d1_stage0_patch_tokens"}:
                    patch_refs_by_layer[f"{layer_name}/{token_type}"].append((run, token_type, ref, layer_meta))

        if not patch_refs_by_layer:
            return [self.skipped(run, "no patch tokens found for shared PCA") for run in runs]

        results_by_run: dict[PathKey, dict[str, Any]] = {
            PathKey(run.bundle_path): {
                "method": self.method_name,
                "status": "ok",
                "model_name": run.bundle.model_name,
                "scene_id": run.manifest.get("scene_id"),
                "setting_name": run.manifest.get("setting_name"),
                "condition_id": run.manifest.get("condition_id"),
                "layers": {},
                "warnings": [],
            }
            for run in runs
        }

        for layer_key, refs in patch_refs_by_layer.items():
            loaded: list[tuple[AnalysisRun, str, np.ndarray, dict[str, Any]]] = []
            warnings: list[dict[str, str]] = []
            for run, token_type, ref, layer_meta in refs:
                seq_len = int(run.manifest.get("seq_len") or run.bundle.seq_len or 0) or None
                try:
                    tokens = patch_tokens_by_frame(ref, seq_len=seq_len, base_dir=run.bundle_path.parent)
                except Exception as exc:
                    warnings.append({"condition_id": run.manifest.get("condition_id"), "warning": str(exc)})
                    continue
                loaded.append((run, token_type, tokens, layer_meta))
            if not loaded:
                for run in runs:
                    results_by_run[PathKey(run.bundle_path)]["warnings"].extend(warnings)
                continue

            map_resolution, upsample_mode = resolve_map_config(self.config)
            contact_sheet_max_side = int(self.figure_config.get("contact_sheet_max_side", 360))
            pca = fit_pca_shared(
                [tokens for _run, _token_type, tokens, _meta in loaded],
                n_components=3,
                max_samples=int(self.config.get("pca_max_samples", 50000)),
                random_seed=int(self.config.get("random_seed", 2026)),
            )
            for run, token_type, tokens, layer_meta in loaded:
                projection = project_pca(tokens, pca)
                layer_result = {
                    "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
                    "mean": pca["mean"].tolist(),
                    "components_shape": list(pca["components"].shape),
                    "patch_shape": list(tokens.shape),
                    "map_resolution": map_resolution,
                    "upsample_mode": upsample_mode if map_resolution in {"pixel", "upsampled", "pixel_level"} else None,
                    "figures": [],
                }
                grid = infer_patch_grid(tokens.shape[1], layer_meta)
                if grid is None:
                    layer_result["warning"] = f"Could not infer patch grid for {tokens.shape[1]} patches"
                else:
                    layer_result["patch_grid"] = list(grid)
                    frame_indices = selected_frame_indices(
                        tokens.shape[0],
                        self.figure_config.get("max_frames_for_contact_sheet", 8),
                        self.config,
                    )
                    layer_result["selected_frame_indices"] = frame_indices
                    layer_result["selected_frame_phases"] = {
                        str(idx): frame_memory_phase(idx, tokens.shape[0], self.config)
                        for idx in frame_indices
                    }
                    images = []
                    for frame_index in frame_indices:
                        patch_image = pca_image(projection[frame_index], grid)
                        image = patch_image
                        pixel_shape = None
                        if map_resolution in {"pixel", "upsampled", "pixel_level"}:
                            pixel_shape = frame_pixel_shape(run, frame_index)
                            if pixel_shape is None:
                                layer_result.setdefault("warnings", []).append(
                                    f"Could not read RGB size for t={frame_index}; saved patch-grid PCA image."
                                )
                            else:
                                image = upsample_rgb_image(patch_image, pixel_shape, mode=upsample_mode)
                        if pixel_shape is not None:
                            layer_result.setdefault("pixel_shapes", {})[str(frame_index)] = list(pixel_shape)
                        fig_path = (
                            run.figures_dir
                            / f"{self.method_name}_{safe_name(layer_key)}_t{frame_index:03d}.png"
                        )
                        save_rgb_image(plt, image, fig_path, dpi=int(self.figure_config.get("dpi", 180)))
                        layer_result["figures"].append(str(fig_path))
                        images.append((frame_index, contact_sheet_image(image, max_side=contact_sheet_max_side)))
                    sheet_path = run.figures_dir / f"{self.method_name}_{safe_name(layer_key)}_contact_sheet.png"
                    save_contact_sheet(plt, images, sheet_path, dpi=int(self.figure_config.get("dpi", 180)))
                    layer_result["contact_sheet"] = str(sheet_path)
                results_by_run[PathKey(run.bundle_path)]["layers"][layer_key] = layer_result

        results: list[dict[str, Any]] = []
        for run in runs:
            result = results_by_run[PathKey(run.bundle_path)]
            if not result["layers"]:
                result = self.skipped(run, "shared PCA could not process any layer", extra={"warnings": result["warnings"]})
            else:
                self.save_json(run.metrics_dir / f"{self.method_name}.json", result)
            results.append(result)
        return results


class PathKey:
    """Hash wrapper for paths that normalizes to resolved strings."""

    def __init__(self, path: Any) -> None:
        self.value = str(path)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PathKey) and self.value == other.value


def fit_pca_shared(
    token_arrays: list[np.ndarray],
    *,
    n_components: int,
    max_samples: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Fit PCA with numpy SVD on a shared sampled patch-token matrix."""

    matrices = [tokens.reshape(-1, tokens.shape[-1]) for tokens in token_arrays]
    matrix = np.concatenate(matrices, axis=0).astype(np.float32, copy=False)
    if matrix.shape[0] > max_samples:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(matrix.shape[0], size=max_samples, replace=False)
        matrix = matrix[indices]
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    variance = singular_values**2 / max(matrix.shape[0] - 1, 1)
    total_variance = np.maximum(variance.sum(), 1e-12)
    explained = variance[:n_components] / total_variance
    return {
        "mean": mean.reshape(-1),
        "components": components,
        "explained_variance_ratio": explained,
    }


def project_pca(tokens: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    """Project [T, P, D] tokens to [T, P, 3] PCA coordinates."""

    flat = tokens.reshape(-1, tokens.shape[-1]).astype(np.float32, copy=False)
    projected = (flat - pca["mean"].reshape(1, -1)) @ pca["components"].T
    return projected.reshape(tokens.shape[0], tokens.shape[1], -1)


def pca_image(projected_frame: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Convert projected patch tokens to an RGB image normalized to [0, 1]."""

    height, width = grid
    image = projected_frame[:, :3].reshape(height, width, 3)
    low = np.nanpercentile(image, 1, axis=(0, 1), keepdims=True)
    high = np.nanpercentile(image, 99, axis=(0, 1), keepdims=True)
    return np.clip((image - low) / np.maximum(high - low, 1e-8), 0.0, 1.0)


def resolve_map_config(config: dict[str, Any]) -> tuple[str, str]:
    """Normalize map resolution and interpolation config.

    ``map_resolution`` selects patch-grid versus pixel-size output. Interpolation
    belongs in ``upsample_mode``; accepting nearest/bilinear here prevents an
    easy-to-make config typo from silently producing patch-sized maps.
    """

    map_resolution = str(config.get("map_resolution", "pixel")).lower()
    upsample_mode = str(config.get("upsample_mode", "nearest")).lower()
    if map_resolution in {"nearest", "bilinear"}:
        upsample_mode = map_resolution
        map_resolution = "pixel"
    if map_resolution not in {"patch", "patch_grid", "pixel", "upsampled", "pixel_level"}:
        raise ValueError(
            f"Unsupported shared_pca_patchmap map_resolution={map_resolution!r}; "
            "expected 'pixel' or 'patch'."
        )
    if upsample_mode not in {"nearest", "bilinear"}:
        raise ValueError(
            f"Unsupported shared_pca_patchmap upsample_mode={upsample_mode!r}; "
            "expected 'nearest' or 'bilinear'."
        )
    return map_resolution, upsample_mode


def frame_pixel_shape(run: AnalysisRun, frame_index: int) -> tuple[int, int] | None:
    """Return the RGB frame shape as ``(height, width)`` for pixel-level maps."""

    path = frame_rgb_path(run, frame_index)
    if path is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return None
    return int(height), int(width)


def frame_rgb_path(run: AnalysisRun, frame_index: int) -> Path | None:
    """Find the RGB path for a frame from the manifest or token bundle."""

    frames = run.manifest.get("frames", [])
    if frame_index < len(frames):
        value = frames[frame_index].get("rgb_path")
        if value:
            return resolve_data_path(value, run.bundle_path.parent)
    if frame_index < len(run.bundle.frame_paths):
        return resolve_data_path(run.bundle.frame_paths[frame_index], run.bundle_path.parent)
    return None


def resolve_data_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def upsample_rgb_image(image: np.ndarray, target_shape: tuple[int, int], *, mode: str) -> np.ndarray:
    """Upsample a patch-grid RGB image to ``target_shape`` for visualization."""

    if mode not in {"nearest", "bilinear"}:
        raise ValueError(f"Unsupported PCA upsample_mode={mode!r}; expected 'nearest' or 'bilinear'.")
    from PIL import Image

    target_height, target_width = target_shape
    pil_image = Image.fromarray(rgb_to_uint8(image))
    resample = Image.Resampling.NEAREST if mode == "nearest" else Image.Resampling.BILINEAR
    resized = pil_image.resize((target_width, target_height), resample=resample)
    return np.asarray(resized, dtype=np.float32) / 255.0


def contact_sheet_image(image: np.ndarray, *, max_side: int) -> np.ndarray:
    """Downsample large pixel-level maps before building a contact sheet."""

    height, width = image.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    target_shape = (max(1, round(height * scale)), max(1, round(width * scale)))
    return upsample_rgb_image(image, target_shape, mode="bilinear")


def selected_frame_indices(
    n_frames: int,
    max_frames: int,
    config: dict[str, Any] | None = None,
) -> list[int]:
    """Select frames for PCA visualization.

    The default policy is Lingbot streaming-memory aware: for long streams it
    samples the scale/anchor frames, the evicted trajectory-memory span, and the
    final local-window span. Set ``frame_selection: legacy`` to recover the old
    early-frame selector.
    """

    if n_frames <= 0 or max_frames <= 0:
        return []

    config = dict(config or {})
    manual_indices = config.get("frame_indices")
    if manual_indices:
        return _dedupe_valid_frames([int(idx) for idx in manual_indices], n_frames)[:max_frames]

    mode = str(config.get("frame_selection", "lingbot_memory_phases")).lower()
    if mode in {"legacy", "canonical"}:
        return _legacy_frame_indices(n_frames, max_frames)
    if mode in {"all", "every"}:
        return list(range(min(n_frames, max_frames)))
    if mode not in {"lingbot_memory_phases", "memory_phases", "streaming_phases"}:
        raise ValueError(
            f"Unsupported shared_pca_patchmap frame_selection={mode!r}; "
            "expected 'lingbot_memory_phases', 'legacy', or 'all'."
        )

    return _lingbot_memory_phase_indices(n_frames, max_frames, config)


def _legacy_frame_indices(n_frames: int, max_frames: int) -> list[int]:
    preferred = [0, 1, 2, 4, 8, 15, 31]
    selected = [idx for idx in preferred if idx < n_frames]
    if len(selected) < min(max_frames, n_frames):
        extra = np.linspace(0, n_frames - 1, min(max_frames, n_frames)).round().astype(int).tolist()
        selected.extend(extra)
    return list(dict.fromkeys(selected))[:max_frames]


def _lingbot_memory_phase_indices(
    n_frames: int,
    max_frames: int,
    config: dict[str, Any],
) -> list[int]:
    scale_frames, _sliding_window, local_start = streaming_phase_bounds(n_frames, config)
    anchor_end = min(n_frames - 1, scale_frames - 1)
    trajectory_start = scale_frames
    trajectory_end = local_start - 1
    last_frame = n_frames - 1

    anchor_candidates: list[int] = []
    _append_frame(anchor_candidates, 0, n_frames)
    _append_frame(anchor_candidates, 1, n_frames)
    _append_frame(anchor_candidates, anchor_end, n_frames)
    _append_frame(anchor_candidates, 2, n_frames)

    trajectory_candidates: list[int] = []
    if trajectory_start <= trajectory_end:
        span = trajectory_end - trajectory_start
        _append_frame(trajectory_candidates, trajectory_start, n_frames)
        _append_frame(trajectory_candidates, round((trajectory_start + trajectory_end) / 2), n_frames)
        _append_frame(trajectory_candidates, trajectory_end, n_frames)
        _append_frame(trajectory_candidates, round(trajectory_start + span * 0.15), n_frames)

    local_candidates: list[int] = []
    if local_start < n_frames:
        span = last_frame - local_start
        _append_frame(local_candidates, local_start, n_frames)
        _append_frame(local_candidates, last_frame, n_frames)
        _append_frame(local_candidates, round(local_start + span * 0.125), n_frames)
        _append_frame(local_candidates, round(local_start + span * 0.5), n_frames)

    phase_candidates = [
        ("anchor", anchor_candidates),
        ("trajectory_memory", trajectory_candidates),
        ("local_window", local_candidates),
    ]
    phase_candidates = [(name, frames) for name, frames in phase_candidates if frames]
    targets = _phase_targets(phase_candidates, max_frames)

    selected: list[int] = []
    for name, candidates in phase_candidates:
        for frame_index in candidates[: targets.get(name, 0)]:
            _append_frame(selected, frame_index, n_frames)

    selected = _dedupe_valid_frames(selected, n_frames)
    if len(selected) < min(max_frames, n_frames):
        extra = np.linspace(0, last_frame, min(max_frames, n_frames)).round().astype(int).tolist()
        selected.extend(extra)
    return sorted(_dedupe_valid_frames(selected, n_frames)[:max_frames])


def _phase_targets(phase_candidates: list[tuple[str, list[int]]], max_frames: int) -> dict[str, int]:
    active_names = [name for name, _frames in phase_candidates]
    if not active_names:
        return {}

    if len(active_names) == 3 and max_frames >= 8:
        targets = {"anchor": 3, "trajectory_memory": 3, "local_window": 2}
        remaining = max_frames - 8
    else:
        targets = {name: 0 for name in active_names}
        remaining = max_frames
        for name in active_names:
            if remaining <= 0:
                break
            targets[name] = 1
            remaining -= 1

    limits = {name: len(frames) for name, frames in phase_candidates}
    while remaining > 0:
        progressed = False
        for name in active_names:
            if remaining <= 0:
                break
            if targets.get(name, 0) < limits[name]:
                targets[name] = targets.get(name, 0) + 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return targets


def streaming_phase_bounds(n_frames: int, config: dict[str, Any] | None = None) -> tuple[int, int, int]:
    """Return ``(scale_frames, sliding_window, local_start)`` for a stream."""

    config = dict(config or {})
    scale_frames = max(1, int(config.get("num_scale_frames", 8)))
    sliding_window = max(1, int(config.get("kv_cache_sliding_window", 64)))
    if n_frames <= 0:
        return scale_frames, sliding_window, 0
    local_start = max(scale_frames, n_frames - sliding_window)
    local_start = min(max(local_start, 0), n_frames)
    return scale_frames, sliding_window, local_start


def frame_memory_phase(
    frame_index: int,
    n_frames: int,
    config: dict[str, Any] | None = None,
) -> str:
    scale_frames, _sliding_window, local_start = streaming_phase_bounds(n_frames, config)
    if frame_index < min(scale_frames, n_frames):
        return "anchor"
    if frame_index < local_start:
        return "trajectory_memory"
    return "local_window"


def _append_frame(frames: list[int], frame_index: int, n_frames: int) -> None:
    if 0 <= frame_index < n_frames and frame_index not in frames:
        frames.append(frame_index)


def _dedupe_valid_frames(frames: list[int], n_frames: int) -> list[int]:
    return list(dict.fromkeys(idx for idx in frames if 0 <= idx < n_frames))


def save_rgb_image(plt: Any, image: np.ndarray, path: Any, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(rgb_to_uint8(image)).save(path, dpi=(dpi, dpi))


def rgb_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_contact_sheet(plt: Any, images: list[tuple[int, np.ndarray]], path: Any, *, dpi: int) -> None:
    if not images:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = len(images)
    plt.figure(figsize=(max(2.2 * cols, 2.2), 2.4))
    for idx, (frame_index, image) in enumerate(images, start=1):
        ax = plt.subplot(1, cols, idx)
        ax.imshow(image)
        ax.set_title(f"t={frame_index}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
