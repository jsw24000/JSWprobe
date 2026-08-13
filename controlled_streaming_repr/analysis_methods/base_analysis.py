"""Base classes and token loading helpers for pluggable analyses."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adapters.token_schema import TokenBundle


@dataclass
class AnalysisRun:
    """A token bundle paired with its source manifest and output paths."""

    bundle: TokenBundle
    manifest: dict[str, Any]
    bundle_path: Path
    metrics_dir: Path
    figures_dir: Path
    config: dict[str, Any] = field(default_factory=dict)


class BaseAnalysisMethod(ABC):
    """Base class for analysis methods.

    Analysis methods read saved token bundles, attention summaries, geometry
    outputs, and manifests. They do not know how a sequence setting generated
    its inputs beyond manifest metadata.
    """

    method_name: str = "base"
    supports_group_run: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.figure_config = self.config.get("figures", {})

    def run_group(self, runs: list[AnalysisRun]) -> list[dict[str, Any]]:
        """Run on a group; default behavior analyzes each run independently."""

        return [self.run(run) for run in runs]

    @abstractmethod
    def run(self, run: AnalysisRun) -> dict[str, Any]:
        """Run an analysis for one token bundle and manifest."""

    def save_json(self, path: str | Path, data: Any) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path

    def skipped(self, run: AnalysisRun, reason: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write and return a skipped-result metrics file."""

        result = {
            "method": self.method_name,
            "status": "skipped",
            "reason": reason,
            "model_name": run.bundle.model_name,
            "scene_id": run.manifest.get("scene_id"),
            "setting_name": run.manifest.get("setting_name"),
            "condition_id": run.manifest.get("condition_id"),
        }
        if extra:
            result.update(extra)
        self.save_json(run.metrics_dir / f"{self.method_name}.json", result)
        return result


def to_numpy(value: Any, *, base_dir: Path | None = None) -> np.ndarray:
    """Load a numpy-compatible object or path into an ndarray."""

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_absolute() and base_dir is not None:
            path = (base_dir / path).resolve()
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix == ".npz":
            loaded = np.load(path, allow_pickle=False)
            first_key = loaded.files[0]
            return loaded[first_key]
        if path.suffix == ".pt":
            try:
                import torch
            except ImportError as exc:
                raise ImportError(f"Loading {path} requires torch.") from exc
            tensor = torch.load(path, map_location="cpu")
            return to_numpy(tensor)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def iter_layer_token_refs(bundle: TokenBundle, token_types: list[str] | None = None) -> list[tuple[str, str, Any, dict[str, Any]]]:
    """Yield layer/token_type references from a bundle."""

    wanted = set(token_types or [])
    refs: list[tuple[str, str, Any, dict[str, Any]]] = []
    for layer_name, layer_values in sorted(bundle.layer_tokens.items()):
        if not isinstance(layer_values, dict):
            continue
        for token_type, ref in layer_values.items():
            if token_type in {"shape", "extra", "patch_start_idx"}:
                continue
            if token_type.endswith("_tokens") or token_type in {"d1_stage0_patch_tokens"}:
                if wanted and token_type not in wanted:
                    continue
                refs.append((layer_name, token_type, ref, layer_values))
    return refs


def frame_embeddings(tokens: Any, seq_len: int | None = None, *, base_dir: Path | None = None) -> np.ndarray:
    """Convert token arrays to one embedding per frame by mean pooling tokens."""

    arr = to_numpy(tokens, base_dir=base_dir).astype(float, copy=False)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        if seq_len is not None and arr.shape[0] == seq_len:
            return arr
        return arr.mean(axis=0, keepdims=True)
    if arr.ndim >= 3:
        if seq_len is not None and arr.shape[0] != seq_len:
            # Some extractors save [layers, frames, tokens, dim] or similar.
            frame_axis = next((axis for axis, size in enumerate(arr.shape[:-1]) if size == seq_len), 0)
            arr = np.moveaxis(arr, frame_axis, 0)
        reduce_axes = tuple(range(1, arr.ndim - 1))
        return arr.mean(axis=reduce_axes)
    raise ValueError(f"Unsupported token shape: {arr.shape}")


def normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize rows in a feature matrix."""

    denom = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(denom, eps)


def cosine_matrix(features: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarity."""

    normalized = normalize_rows(np.asarray(features, dtype=float))
    return normalized @ normalized.T


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return 1 - cosine for corresponding rows."""

    a_norm = normalize_rows(np.asarray(a, dtype=float))
    b_norm = normalize_rows(np.asarray(b, dtype=float))
    return 1.0 - np.sum(a_norm * b_norm, axis=-1)


def infer_patch_grid(n_patches: int, layer_meta: dict[str, Any] | None = None) -> tuple[int, int] | None:
    """Infer a patch grid from metadata or by finding a near-square factor pair."""

    meta = layer_meta or {}
    for key in ("patch_grid", "grid_shape"):
        value = meta.get(key)
        if value is None and isinstance(meta.get("extra"), dict):
            value = meta["extra"].get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
    side = int(round(np.sqrt(n_patches)))
    for height in range(side, 0, -1):
        if n_patches % height == 0:
            width = n_patches // height
            return height, width
    return None


def patch_tokens_by_frame(tokens: Any, seq_len: int | None = None, *, base_dir: Path | None = None) -> np.ndarray:
    """Return patch tokens as [T, P, D], flattening spatial patch axes."""

    arr = to_numpy(tokens, base_dir=base_dir).astype(float, copy=False)
    if arr.ndim < 3:
        raise ValueError(f"Expected patch tokens with frame and patch axes, got {arr.shape}")
    if seq_len is not None and arr.shape[0] != seq_len:
        frame_axis = next((axis for axis, size in enumerate(arr.shape[:-1]) if size == seq_len), 0)
        arr = np.moveaxis(arr, frame_axis, 0)
    if arr.ndim == 3:
        return arr
    return arr.reshape(arr.shape[0], -1, arr.shape[-1])
