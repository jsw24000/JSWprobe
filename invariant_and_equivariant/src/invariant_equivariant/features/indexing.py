from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from invariant_equivariant.features.cache import load_feature_vector
from invariant_equivariant.utils import read_csv, write_csv


INDEX_COLUMNS = [
    "model_name",
    "model_variant",
    "checkpoint_id",
    "layer_name",
    "scene_id",
    "state_id",
    "camera_id",
    "split",
    "target_object_id",
    "target_category",
    "feature_path",
    "feature_dim",
    "mask_area_ratio",
    "patch_coverage_sum",
    "effective_patch_count",
    "valid",
    "position_world_x",
    "position_world_y",
    "position_world_z",
    "position_ref_x",
    "position_ref_y",
    "position_ref_z",
    "position_current_x",
    "position_current_y",
    "position_current_z",
    "grid_row",
    "grid_col",
    "mask_centroid_x",
    "mask_centroid_y",
    "bbox_width",
    "bbox_height",
    "camera_numeric",
]


def index_path(output_dir: Path) -> Path:
    return output_dir / "features" / "feature_index.csv"


def write_feature_index(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(index_path(output_dir), rows, INDEX_COLUMNS)


def read_feature_index(output_dir: Path) -> List[Dict[str, str]]:
    return read_csv(index_path(output_dir))


def rows_for_model_layer(index_rows: Sequence[Mapping[str, Any]], model_name: str, layer_name: str, valid_only: bool = True) -> List[Mapping[str, Any]]:
    rows = [row for row in index_rows if row["model_name"] == model_name and row["layer_name"] == layer_name]
    if valid_only:
        rows = [row for row in rows if str(row["valid"]).lower() in {"true", "1"}]
    return rows


def load_feature_matrix(rows: Sequence[Mapping[str, Any]], key: str = "pooled_raw") -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack([load_feature_vector(row["feature_path"], key=key) for row in rows])


def positions_matrix(rows: Sequence[Mapping[str, Any]], coord: str) -> np.ndarray:
    prefix = {"ref": "position_ref", "current": "position_current", "world": "position_world"}[coord]
    return np.asarray([[float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"])] for row in rows], dtype=np.float64)


def split_masks(rows: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    splits = {}
    for split in ["train", "validation", "test"]:
        splits[split] = np.asarray([row["split"] == split for row in rows], dtype=bool)
    return splits


def frame_feature_lookup(rows: Sequence[Mapping[str, Any]], key: str = "pooled_raw") -> Dict[Tuple[str, str, str], np.ndarray]:
    return {
        (str(row["scene_id"]), str(row["state_id"]), str(row["camera_id"])): load_feature_vector(row["feature_path"], key=key)
        for row in rows
    }
