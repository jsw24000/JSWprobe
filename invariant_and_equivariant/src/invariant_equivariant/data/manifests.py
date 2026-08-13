from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from invariant_equivariant.utils import read_json, read_jsonl, split_for_scene


def dataset_root(config: Mapping[str, Any]) -> Path:
    return Path(config["_paths"]["dataset_root"])


def load_manifests(config: Mapping[str, Any]) -> Dict[str, Any]:
    root = dataset_root(config)
    manifests = root / "manifests"
    return {
        "root": root,
        "summary": read_json(root / "dataset_summary.json"),
        "scenes": read_jsonl(manifests / "scenes.jsonl"),
        "states": read_jsonl(manifests / "states.jsonl"),
        "frames": read_jsonl(manifests / "frames.jsonl"),
        "cameras": read_jsonl(manifests / "cameras.jsonl"),
        "pairs": read_jsonl(manifests / "translation_pairs.jsonl"),
        "triplets": read_jsonl(manifests / "composition_triplets.jsonl"),
        "splits": read_json(manifests / "splits.json"),
    }


def selected_state_ids(config: Mapping[str, Any], states: Sequence[Mapping[str, Any]]) -> List[str]:
    selected = config["selection"]["position_indices"]
    if selected == "all":
        return sorted({str(row["state_id"]) for row in states})
    want = {tuple(item) for item in selected}
    return sorted({str(row["state_id"]) for row in states if tuple(row["position_index"]) in want})


def filter_frames(config: Mapping[str, Any], manifests: Mapping[str, Any]) -> List[Dict[str, Any]]:
    scenes = set(config["selection"]["scenes"])
    state_ids = set(selected_state_ids(config, manifests["states"]))
    cameras_cfg = config["selection"]["cameras"]
    cameras = None if cameras_cfg == "all" else set(cameras_cfg)
    rows: List[Dict[str, Any]] = []
    state_lookup = {(row["scene_id"], row["state_id"]): row for row in manifests["states"]}
    for frame in manifests["frames"]:
        if frame["scene_id"] not in scenes:
            continue
        if frame["state_id"] not in state_ids:
            continue
        if cameras is not None and frame["camera_id"] not in cameras:
            continue
        row = dict(frame)
        row["split"] = split_for_scene(config, str(frame["scene_id"]))
        row["state_position_index"] = state_lookup[(frame["scene_id"], frame["state_id"])]["position_index"]
        rows.append(row)
    rows.sort(key=lambda r: (r["scene_id"], r["state_id"], r["camera_id"]))
    return rows


def frame_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return str(row["scene_id"]), str(row["state_id"]), str(row["camera_id"])


def filter_pairs(config: Mapping[str, Any], manifests: Mapping[str, Any], valid_frame_keys: set[Tuple[str, str, str]] | None = None) -> List[Dict[str, Any]]:
    scenes = set(config["selection"]["scenes"])
    state_ids = set(selected_state_ids(config, manifests["states"]))
    cameras_cfg = config["selection"]["cameras"]
    cameras = None if cameras_cfg == "all" else set(cameras_cfg)
    rows: List[Dict[str, Any]] = []
    for pair in manifests["pairs"]:
        if pair["scene_id"] not in scenes or pair["state_a"] not in state_ids or pair["state_b"] not in state_ids:
            continue
        if cameras is not None and pair["camera_id"] not in cameras:
            continue
        split = split_for_scene(config, str(pair["scene_id"]))
        if split == "unused":
            continue
        if valid_frame_keys is not None:
            if (pair["scene_id"], pair["state_a"], pair["camera_id"]) not in valid_frame_keys:
                continue
            if (pair["scene_id"], pair["state_b"], pair["camera_id"]) not in valid_frame_keys:
                continue
        row = dict(pair)
        row["split"] = split
        rows.append(row)
    rows.sort(key=lambda r: (r["scene_id"], r["state_a"], r["state_b"], r["camera_id"], r["pair_type"]))
    return rows


def filter_triplets(config: Mapping[str, Any], manifests: Mapping[str, Any], valid_frame_keys: set[Tuple[str, str, str]] | None = None) -> List[Dict[str, Any]]:
    scenes = set(config["selection"]["scenes"])
    state_ids = set(selected_state_ids(config, manifests["states"]))
    cameras_cfg = config["selection"]["cameras"]
    cameras = None if cameras_cfg == "all" else set(cameras_cfg)
    rows: List[Dict[str, Any]] = []
    for triplet in manifests["triplets"]:
        if triplet["scene_id"] not in scenes:
            continue
        if any(triplet[k] not in state_ids for k in ["state_0", "state_1", "state_2"]):
            continue
        if cameras is not None and triplet["camera_id"] not in cameras:
            continue
        split = split_for_scene(config, str(triplet["scene_id"]))
        if split == "unused":
            continue
        if valid_frame_keys is not None:
            if any((triplet["scene_id"], triplet[k], triplet["camera_id"]) not in valid_frame_keys for k in ["state_0", "state_1", "state_2"]):
                continue
        row = dict(triplet)
        row["split"] = split
        rows.append(row)
    return rows


def state_rank(states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    arr = np.asarray([row["object_center_ref_camera"] for row in states], dtype=np.float64)
    centered = arr - arr.mean(axis=0, keepdims=True)
    if len(centered) == 0:
        return {"rank": 0, "singular_values": []}
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    rank = int((singular_values > max(1e-9, singular_values[0] * 1e-6)).sum()) if singular_values.size else 0
    return {"rank": rank, "singular_values": singular_values.tolist()}

