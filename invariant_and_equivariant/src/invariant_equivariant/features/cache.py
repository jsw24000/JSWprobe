from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from invariant_equivariant.utils import ensure_dir


def feature_dir(output_dir: Path, model_name: str, layer_name: str) -> Path:
    return ensure_dir(output_dir / "features" / f"model={model_name}" / f"layer={layer_name}")


def feature_path(output_dir: Path, model_name: str, layer_name: str, scene_id: str, state_id: str, camera_id: str) -> Path:
    return feature_dir(output_dir, model_name, layer_name) / f"{scene_id}_{state_id}_{camera_id}.npz"


def save_feature_npz(
    path: Path,
    patch_tokens: np.ndarray | None,
    pooled_raw: np.ndarray,
    pooled_l2: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    ensure_dir(path.parent)
    payload: Dict[str, Any] = {
        "pooled_raw": pooled_raw.astype(np.float32),
        "pooled_l2": pooled_l2.astype(np.float32),
        "metadata_json": np.asarray([__import__("json").dumps(dict(metadata), sort_keys=True)]),
    }
    if patch_tokens is not None:
        payload["patch_tokens"] = patch_tokens
    np.savez_compressed(path, **payload)


def load_feature_vector(path: str | Path, key: str = "pooled_raw") -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data[key], dtype=np.float64)


def load_patch_tokens(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["patch_tokens"])

