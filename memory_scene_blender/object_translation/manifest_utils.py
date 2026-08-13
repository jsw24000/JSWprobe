"""Manifest and configuration helpers for object-translation scenes."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} did not contain a mapping")
        return loaded
    except ModuleNotFoundError:
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} did not contain a mapping")
        return loaded


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def np_to_list(array: np.ndarray, digits: Optional[int] = 8) -> List[Any]:
    arr = np.asarray(array, dtype=np.float64)
    if digits is not None:
        arr = np.round(arr, digits)
    return arr.tolist()


def vector_to_list(vector: Sequence[float], digits: Optional[int] = 8) -> List[float]:
    arr = np.asarray(vector, dtype=np.float64)
    if digits is not None:
        arr = np.round(arr, digits)
    return [float(value) for value in arr.tolist()]


def mode_settings(config: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    if mode not in config.get("modes", {}):
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(config.get('modes', {}))}")
    merged = dict(config.get("render", {}))
    merged.update(config["modes"][mode])
    return merged


def normalize_output_root(path: Path, package_dir: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "memory_scene_blender":
        return package_dir.parent / path
    if path.parts and path.parts[0] == "outputs":
        return package_dir / path
    return Path.cwd() / path


def homogeneous_point(point: Sequence[float]) -> np.ndarray:
    return np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float64)


def transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    return (mat @ homogeneous_point(point))[:3]


def transform_delta(matrix: Sequence[Sequence[float]], delta: Sequence[float]) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    return mat[:3, :3] @ np.asarray(delta, dtype=np.float64)


def displacement_group_id(delta_world: Sequence[float], decimals: int = 5) -> str:
    rounded = tuple(round(float(value), decimals) for value in delta_world)
    return "disp_" + "_".join(f"{value:+.{decimals}f}" for value in rounded)


def state_lookup(states: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int], Mapping[str, Any]]:
    return {tuple(state["position_index"]): state for state in states}  # type: ignore[arg-type]


def _grid_pair_specs(rows: int, cols: int) -> List[Tuple[Tuple[int, int], Tuple[int, int], str, List[str]]]:
    specs: List[Tuple[Tuple[int, int], Tuple[int, int], str, List[str]]] = []

    for row in range(rows):
        for col in range(cols - 1):
            specs.append(((row, col), (row, col + 1), "grid_adjacent", ["horizontal", "short-distance"]))
    for row in range(rows - 1):
        for col in range(cols):
            specs.append(((row, col), (row + 1, col), "grid_adjacent", ["vertical", "short-distance"]))
    for row in range(rows - 1):
        for col in range(cols - 1):
            specs.append(((row, col), (row + 1, col + 1), "diagonal", ["diagonal", "medium-distance"]))
            specs.append(((row + 1, col), (row, col + 1), "diagonal", ["diagonal", "medium-distance"]))
    for row in range(rows):
        for col in range(cols - 2):
            specs.append(((row, col), (row, col + 2), "two-step", ["horizontal", "two-step", "medium-distance"]))
            specs.append(((row, col + 2), (row, col), "two-step", ["horizontal", "two-step", "reverse"]))
    for row in range(rows - 2):
        for col in range(cols):
            specs.append(((row, col), (row + 2, col), "two-step", ["vertical", "two-step", "medium-distance"]))
            specs.append(((row + 2, col), (row, col), "two-step", ["vertical", "two-step", "reverse"]))
    for row in range(rows - 2):
        for col in range(cols - 1):
            specs.append(((row, col), (row + 2, col + 1), "medium-distance", ["mixed-axis", "medium-distance"]))
    for row in range(rows - 1):
        for col in range(cols - 2):
            specs.append(((row, col), (row + 1, col + 2), "medium-distance", ["mixed-axis", "medium-distance"]))

    if rows >= 2 and cols >= 2:
        specs.append(((rows - 1, cols - 1), (0, 0), "random_valid_pair", ["diagonal", "reverse"]))
        specs.append(((0, cols - 1), (rows - 1, 0), "random_valid_pair", ["diagonal"]))
    return specs


def build_translation_pairs(
    scene_id: str,
    target_object_id: str,
    states: Sequence[Mapping[str, Any]],
    cameras: Sequence[Mapping[str, Any]],
    frame_lookup: Mapping[Tuple[str, str], str],
    rows: int,
    cols: int,
) -> List[Dict[str, Any]]:
    by_index = state_lookup(states)
    pair_specs = _grid_pair_specs(rows, cols)
    pairs: List[Dict[str, Any]] = []
    seen_keys = set()
    ref_camera = cameras[0]
    ref_w2c = ref_camera["opencv_world_to_camera"]

    for start_idx, end_idx, pair_type, pair_tags in pair_specs:
        state_a = by_index.get(start_idx)
        state_b = by_index.get(end_idx)
        if state_a is None or state_b is None:
            continue
        delta_world = np.asarray(state_b["object_center_world"], dtype=np.float64) - np.asarray(
            state_a["object_center_world"], dtype=np.float64
        )
        distance = float(np.linalg.norm(delta_world))
        delta_ref = transform_delta(ref_w2c, delta_world)
        group_id = displacement_group_id(delta_world)
        direction_ref = delta_ref / np.linalg.norm(delta_ref) if np.linalg.norm(delta_ref) > 1e-9 else delta_ref

        for camera in cameras:
            camera_id = str(camera["camera_id"])
            dedupe_key = (state_a["state_id"], state_b["state_id"], camera_id, pair_type, tuple(pair_tags))
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            delta_current = transform_delta(camera["opencv_world_to_camera"], delta_world)
            pair_id = f"{scene_id}_{state_a['state_id']}_{state_b['state_id']}_{camera_id}_{pair_type}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "scene_id": scene_id,
                    "target_object_id": target_object_id,
                    "camera_id": camera_id,
                    "state_a": state_a["state_id"],
                    "state_b": state_b["state_id"],
                    "frame_a": frame_lookup[(state_a["state_id"], camera_id)],
                    "frame_b": frame_lookup[(state_b["state_id"], camera_id)],
                    "delta_world": vector_to_list(delta_world),
                    "delta_ref_camera": vector_to_list(delta_ref),
                    "delta_current_camera": vector_to_list(delta_current),
                    "distance": float(round(distance, 8)),
                    "direction_unit_ref_camera": vector_to_list(direction_ref),
                    "same_displacement_group": group_id,
                    "pair_type": pair_type,
                    "pair_tags": sorted(set(pair_tags)),
                    "position_index_a": list(start_idx),
                    "position_index_b": list(end_idx),
                }
            )
    return pairs


def _triplet_specs(rows: int, cols: int) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], str]]:
    specs: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], str]] = []
    for row in range(rows):
        for col in range(cols - 2):
            specs.append(((row, col), (row, col + 1), (row, col + 2), "same_axis_horizontal"))
            specs.append(((row, col + 2), (row, col + 1), (row, col), "same_axis_horizontal_reverse"))
    for row in range(rows - 2):
        for col in range(cols):
            specs.append(((row, col), (row + 1, col), (row + 2, col), "same_axis_vertical"))
            specs.append(((row + 2, col), (row + 1, col), (row, col), "same_axis_vertical_reverse"))
    for row in range(rows - 1):
        for col in range(cols - 1):
            specs.append(((row, col), (row, col + 1), (row + 1, col + 1), "orthogonal_then_vertical"))
            specs.append(((row, col), (row + 1, col), (row + 1, col + 1), "orthogonal_then_horizontal"))
            specs.append(((row + 1, col + 1), (row, col + 1), (row, col), "orthogonal_reverse"))
            specs.append(((row, col + 1), (row, col), (row + 1, col), "diagonal_composition"))
    return specs


def build_composition_triplets(
    scene_id: str,
    target_object_id: str,
    states: Sequence[Mapping[str, Any]],
    cameras: Sequence[Mapping[str, Any]],
    rows: int,
    cols: int,
) -> List[Dict[str, Any]]:
    by_index = state_lookup(states)
    ref_w2c = cameras[0]["opencv_world_to_camera"]
    triplets: List[Dict[str, Any]] = []
    seen_keys = set()

    for idx0, idx1, idx2, triplet_type in _triplet_specs(rows, cols):
        state_0 = by_index.get(idx0)
        state_1 = by_index.get(idx1)
        state_2 = by_index.get(idx2)
        if state_0 is None or state_1 is None or state_2 is None:
            continue
        c0 = np.asarray(state_0["object_center_world"], dtype=np.float64)
        c1 = np.asarray(state_1["object_center_world"], dtype=np.float64)
        c2 = np.asarray(state_2["object_center_world"], dtype=np.float64)
        delta_01 = transform_delta(ref_w2c, c1 - c0)
        delta_12 = transform_delta(ref_w2c, c2 - c1)
        delta_02 = transform_delta(ref_w2c, c2 - c0)
        additive = bool(np.allclose(delta_02, delta_01 + delta_12, atol=1e-6))
        for camera in cameras:
            camera_id = str(camera["camera_id"])
            key = (state_0["state_id"], state_1["state_id"], state_2["state_id"], camera_id, triplet_type)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            triplets.append(
                {
                    "triplet_id": f"{scene_id}_{state_0['state_id']}_{state_1['state_id']}_{state_2['state_id']}_{camera_id}_{triplet_type}",
                    "scene_id": scene_id,
                    "target_object_id": target_object_id,
                    "camera_id": camera_id,
                    "state_0": state_0["state_id"],
                    "state_1": state_1["state_id"],
                    "state_2": state_2["state_id"],
                    "delta_01_ref_camera": vector_to_list(delta_01),
                    "delta_12_ref_camera": vector_to_list(delta_12),
                    "delta_02_ref_camera": vector_to_list(delta_02),
                    "is_additively_consistent": additive,
                    "triplet_type": triplet_type,
                    "position_index_0": list(idx0),
                    "position_index_1": list(idx1),
                    "position_index_2": list(idx2),
                }
            )
    return triplets


def build_splits(
    scene_rows: Sequence[Mapping[str, Any]],
    camera_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    scene_ids = [str(row["scene_id"]) for row in scene_rows]
    split_cfg = config.get("splits", {})
    train_count = int(split_cfg.get("train_scene_count", max(0, len(scene_ids) - 4)))
    val_count = int(split_cfg.get("val_scene_count", min(2, max(0, len(scene_ids) - train_count))))
    train = scene_ids[: min(train_count, len(scene_ids))]
    val = scene_ids[len(train) : min(len(scene_ids), len(train) + val_count)]
    test = scene_ids[len(train) + len(val) :]

    primary_cameras = sorted({row["camera_id"] for row in camera_rows if row.get("is_primary")})
    held_out_cameras = sorted({row["camera_id"] for row in camera_rows if row.get("is_held_out")})
    scene_categories = defaultdict(list)
    for row in scene_rows:
        scene_categories[str(row["target_category"])].append(row["scene_id"])

    return {
        "unit": "scene_object_state_group",
        "train_scenes": train,
        "validation_scenes": val,
        "test_scenes": test,
        "primary_cameras": primary_cameras,
        "held_out_cameras": held_out_cameras,
        "seen_target_categories": sorted(scene_categories),
        "held_out_target_instances": [
            row["target_object_id"]
            for idx, row in enumerate(scene_rows)
            if idx % max(1, int(split_cfg.get("held_out_target_instance_mod", 2))) == 0
        ],
        "leakage_guard": "All frames from a scene/state stay in the same split; cameras are not randomized across splits.",
    }


def summarize_dataset(
    scene_rows: Sequence[Mapping[str, Any]],
    camera_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    triplet_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    distances = [float(row["distance"]) for row in pair_rows]
    displacement_counts: Dict[str, int] = defaultdict(int)
    for row in pair_rows:
        displacement_counts[str(row["same_displacement_group"])] += 1
    mask_ratios = [float(row.get("target_mask_area_ratio", 0.0)) for row in frame_rows]
    visible_failures = [row for row in frame_rows if not row.get("visibility_ok", False)]

    def _stats(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }

    return {
        "scene_count": len(scene_rows),
        "camera_count": len(camera_rows),
        "state_count": len(state_rows),
        "frame_count": len(frame_rows),
        "pair_count": len(pair_rows),
        "triplet_count": len(triplet_rows),
        "target_categories": sorted({str(row["target_category"]) for row in scene_rows}),
        "translation_distance_stats": _stats(distances),
        "displacement_group_sample_counts": dict(sorted(displacement_counts.items())),
        "mask_area_ratio_stats": _stats(mask_ratios),
        "visibility_failure_count": len(visible_failures),
    }
