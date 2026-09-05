#!/usr/bin/env python3
"""Validate the controlled ego/object world-X factorial dataset.

This program deliberately has no Blender dependency.  It validates protocol
metadata and saved sparse tracks from NumPy, and (unless ``--geometry-only``
is selected) also validates rendered files with Pillow/NumPy.

All paths stored by the dataset are required to be relative to the dataset
root.  A geometry-only pass is useful for a generator ``--dry-run``: rendered
checks are then reported as skipped, never silently counted as passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_ROOT = PACKAGE_DIR / "outputs" / "ego_object_x_factorial_v1" / "pilot"

CHECK_NAMES = (
    "required_files",
    "json_finiteness",
    "schema",
    "dataset_counts",
    "factorial_protocol",
    "scene_splits",
    "source_provenance",
    "path_convention",
    "selection_collision_visibility",
    "frame_geometry",
    "frame0_identity",
    "camera_rotation_invariance",
    "endpoint_displacements",
    "relative_equation",
    "track_identity",
    "track_projection_consistency",
    "matched_relative_pair",
    "compensated_target_trajectory",
    "projected_displacement_scale",
    "rendered_outputs",
    "frame0_rgb_identity",
    "compensated_background_change",
)

REQUIRED_MANIFESTS = {
    "scenes": "scenes.jsonl",
    "anchors": "anchors.jsonl",
    "base_cameras": "base_cameras.jsonl",
    "sequences": "sequences.jsonl",
    "frames": "frames.jsonl",
    "matched_relative_groups": "matched_relative_groups.jsonl",
    "track_sets": "track_sets.jsonl",
}

TARGET_IDENTITY_KEYS = (
    "target_object_id",
    "target_object_numeric_id",
    "target_category",
    "target_variant",
    "target_asset_signature",
)

SEQUENCE_REQUIRED_KEYS = {
    "sequence_id",
    "group_id",
    "scene_id",
    "object_anchor_id",
    "base_camera_id",
    "ego_level",
    "object_level",
    "ego_amplitude_m",
    "object_amplitude_m",
    "relative_level",
    "relative_amplitude_m",
    "num_frames",
    "delta_m",
    "motion_axis_world",
    "split",
    "is_compensated_motion",
    "is_static",
    "sequence_dir",
    "tracks_path",
    *TARGET_IDENTITY_KEYS,
}

FRAME_REQUIRED_KEYS = {
    "frame_id",
    "sequence_id",
    "group_id",
    "scene_id",
    "object_anchor_id",
    "base_camera_id",
    "frame_index",
    "alpha_t",
    "actual_ego_dx_m",
    "actual_object_dx_m",
    "actual_relative_dx_m",
    "camera_to_world",
    "world_to_camera",
    "blender_camera_to_world",
    "camera_rotation",
    "camera_world_position",
    "K",
    "object_to_world",
    "image_size",
    "rendered",
    "frame_paths",
}

TRACK_REQUIRED_ARRAYS = {
    "point_id",
    "xyz_object_local",
    "xyz_world",
    "xyz_camera",
    "projected_uv",
    "depth_camera_z_m",
    "range_to_camera_m",
    "in_front_of_camera",
    "in_image",
    "visible",
}

RENDERED_TRACK_ARRAYS = {"depth_buffer_range_m", "depth_buffer_error_m"}


class ValidationState:
    """Collect check-scoped evidence without producing unbounded error logs."""

    def __init__(self, max_messages: int = 500) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.error_counts: Counter[str] = Counter()
        self.warning_counts: Counter[str] = Counter()
        self.skipped: Dict[str, str] = {}
        self.metrics: Dict[str, Any] = {}
        self.max_messages = int(max_messages)

    def error(self, check: str, message: str) -> None:
        self.error_counts[check] += 1
        if len(self.errors) < self.max_messages:
            self.errors.append(f"[{check}] {message}")

    def warning(self, check: str, message: str) -> None:
        self.warning_counts[check] += 1
        if len(self.warnings) < self.max_messages:
            self.warnings.append(f"[{check}] {message}")

    def skip(self, check: str, reason: str) -> None:
        self.skipped.setdefault(check, reason)

    def check_statuses(self) -> Dict[str, Dict[str, Any]]:
        statuses: Dict[str, Dict[str, Any]] = {}
        for name in CHECK_NAMES:
            if self.error_counts[name]:
                status = "failed"
            elif name in self.skipped:
                status = "skipped"
            else:
                status = "passed"
            row: Dict[str, Any] = {
                "status": status,
                "error_count": int(self.error_counts[name]),
                "warning_count": int(self.warning_counts[name]),
            }
            if name in self.skipped:
                row["reason"] = self.skipped[name]
            statuses[name] = row
        return statuses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ego_object_x_factorial_v1 without importing Blender."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset mode root containing config_used.yaml and manifests/.",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Validate metadata, matrices, protocol, and sparse tracks; skip rendered-file checks.",
    )
    return parser.parse_args()


def _read_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(text)
    except ModuleNotFoundError:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _json_is_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _json_is_finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_is_finite(item) for item in value)
    return False


def _safe_relative_path(
    root: Path,
    raw_path: Any,
    label: str,
    state: ValidationState,
) -> Optional[Path]:
    if not isinstance(raw_path, str) or not raw_path:
        state.error("path_convention", f"{label}: path must be a non-empty string")
        return None
    relative = Path(raw_path)
    if relative.is_absolute():
        state.error("path_convention", f"{label}: absolute path is forbidden: {raw_path}")
        return None
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        state.error("path_convention", f"{label}: path escapes dataset root: {raw_path}")
        return None
    return candidate


def _require_keys(
    rows: Sequence[Mapping[str, Any]],
    required: Iterable[str],
    label: str,
    id_key: str,
    state: ValidationState,
) -> None:
    required_set = set(required)
    for index, row in enumerate(rows):
        missing = sorted(required_set - set(row))
        if missing:
            row_id = row.get(id_key, index)
            state.error("schema", f"{label}[{row_id}] missing keys: {', '.join(missing)}")


def _check_unique_ids(
    rows: Sequence[Mapping[str, Any]], key: str, label: str, state: ValidationState
) -> None:
    counts = Counter(str(row.get(key)) for row in rows if key in row)
    for value, count in counts.items():
        if count != 1:
            state.error("schema", f"{label}: duplicate {key}={value!r} ({count} rows)")


def _array(
    value: Any,
    shape: Optional[Tuple[int, ...]],
    label: str,
    check: str,
    state: ValidationState,
) -> Optional[np.ndarray]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        state.error(check, f"{label}: cannot convert to numeric array: {exc}")
        return None
    if shape is not None and result.shape != shape:
        state.error(check, f"{label}: expected shape {shape}, got {result.shape}")
        return None
    if not np.isfinite(result).all():
        state.error(check, f"{label}: contains NaN or Inf")
        return None
    return result


def _matrix4(value: Any, label: str, check: str, state: ValidationState) -> Optional[np.ndarray]:
    matrix = _array(value, (4, 4), label, check, state)
    if matrix is None:
        return None
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-9):
        state.error(check, f"{label}: invalid homogeneous bottom row {matrix[3].tolist()}")
    return matrix


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (matrix @ homogeneous.T).T[:, :3]


def _transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([point, np.ones(1, dtype=np.float64)])
    return (matrix @ homogeneous)[:3]


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _target_world_center(frame: Mapping[str, Any]) -> Any:
    for key in ("target_world_center", "target_center_world"):
        if key in frame:
            return frame[key]
    return None


def _target_camera_center(frame: Mapping[str, Any]) -> Any:
    for key in (
        "target_camera_coordinate_center",
        "target_center_camera",
        "target_center_opencv_camera",
    ):
        if key in frame:
            return frame[key]
    return None


def _frame_path_value(frame: Mapping[str, Any], canonical_key: str) -> Any:
    paths = frame.get("frame_paths")
    if not isinstance(paths, Mapping):
        return None
    aliases = {
        "depth_exr": ("depth_exr", "depth"),
        "depth_npy": ("depth_npy",),
        "rgb": ("rgb",),
        "target_mask": ("target_mask",),
        "normal": ("normal",),
        "albedo": ("albedo",),
        "instance": ("instance",),
        "semantic": ("semantic",),
        "object_id": ("object_id",),
        "frame_metadata": ("frame_metadata", "metadata"),
    }
    for key in aliases[canonical_key]:
        if key in paths:
            return paths[key]
    return None


def load_dataset(
    root: Path, state: ValidationState
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config_path = root / "config_used.yaml"
    summary_path = root / "dataset_summary.json"
    manifests_dir = root / "manifests"
    required = [
        config_path,
        summary_path,
        root / "README.md",
        root / ".ego_object_x_factorial_dataset",
        root / "provenance.json",
        root / "source_rebuild_audit.json",
        manifests_dir / "groups.jsonl",
        manifests_dir / "splits.json",
        root / "selection_report.json",
    ]
    required.extend(manifests_dir / filename for filename in REQUIRED_MANIFESTS.values())
    for path in required:
        if not path.is_file():
            state.error("required_files", f"missing required file: {path.relative_to(root) if root in path.parents else path}")

    config: Dict[str, Any] = {}
    dataset_summary: Dict[str, Any] = {}
    manifests: Dict[str, Any] = {key: [] for key in REQUIRED_MANIFESTS}
    manifests["splits"] = {}
    manifests["selection_report"] = {}
    manifests["provenance"] = {}
    manifests["source_rebuild_audit"] = {}
    manifests["marker"] = {}

    if config_path.is_file():
        try:
            config = _read_config(config_path)
        except Exception as exc:
            state.error("schema", f"cannot parse config_used.yaml: {exc}")
    if summary_path.is_file():
        try:
            dataset_summary = _read_json(summary_path)
        except Exception as exc:
            state.error("schema", f"cannot parse dataset_summary.json: {exc}")
    for key, filename in REQUIRED_MANIFESTS.items():
        path = manifests_dir / filename
        if path.is_file():
            try:
                manifests[key] = _read_jsonl(path)
            except Exception as exc:
                state.error("schema", f"cannot parse manifests/{filename}: {exc}")
    for key, path in (
        ("splits", manifests_dir / "splits.json"),
        ("selection_report", root / "selection_report.json"),
        ("provenance", root / "provenance.json"),
        ("source_rebuild_audit", root / "source_rebuild_audit.json"),
        ("marker", root / ".ego_object_x_factorial_dataset"),
    ):
        if path.is_file():
            try:
                manifests[key] = _read_json(path)
            except Exception as exc:
                state.error("schema", f"cannot parse {path.relative_to(root)}: {exc}")

    finite_payloads: List[Tuple[str, Any]] = [("config_used.yaml", config), ("dataset_summary.json", dataset_summary)]
    finite_payloads.extend((f"manifests/{REQUIRED_MANIFESTS[key]}", rows) for key, rows in manifests.items() if key in REQUIRED_MANIFESTS)
    finite_payloads.extend(
        (
            ("manifests/splits.json", manifests["splits"]),
            ("selection_report.json", manifests["selection_report"]),
            ("provenance.json", manifests["provenance"]),
            ("source_rebuild_audit.json", manifests["source_rebuild_audit"]),
            (".ego_object_x_factorial_dataset", manifests["marker"]),
        )
    )
    for label, payload in finite_payloads:
        if not _json_is_finite(payload):
            state.error("json_finiteness", f"{label} contains non-finite or non-JSON values")
    return config, dataset_summary, manifests


def validate_schema(manifests: Mapping[str, Any], state: ValidationState) -> None:
    scenes = manifests["scenes"]
    anchors = manifests["anchors"]
    cameras = manifests["base_cameras"]
    sequences = manifests["sequences"]
    frames = manifests["frames"]
    matched = manifests["matched_relative_groups"]
    track_sets = manifests["track_sets"]

    _require_keys(scenes, {"scene_id", "split", *TARGET_IDENTITY_KEYS}, "scenes", "scene_id", state)
    _require_keys(
        anchors,
        {"scene_id", "object_anchor_id", "source_state_id", "anchor_sweep"},
        "anchors",
        "object_anchor_id",
        state,
    )
    _require_keys(
        cameras,
        {"group_id", "scene_id", "object_anchor_id", "base_camera_id", "camera_selection"},
        "base_cameras",
        "group_id",
        state,
    )
    _require_keys(sequences, SEQUENCE_REQUIRED_KEYS, "sequences", "sequence_id", state)
    _require_keys(frames, FRAME_REQUIRED_KEYS, "frames", "frame_id", state)
    for index, frame in enumerate(frames):
        if _target_world_center(frame) is None:
            state.error("schema", f"frames[{frame.get('frame_id', index)}] missing target world center")
        if _target_camera_center(frame) is None:
            state.error("schema", f"frames[{frame.get('frame_id', index)}] missing target camera-coordinate center")
    _require_keys(
        matched,
        {"matched_relative_group_id", "scene_id", "base_camera_id", "relative_level", "members"},
        "matched_relative_groups",
        "matched_relative_group_id",
        state,
    )
    for index, row in enumerate(matched):
        if "object_anchor_id" not in row and "anchor_id" not in row:
            state.error("schema", f"matched_relative_groups[{index}] missing object_anchor_id/anchor_id")
    _require_keys(
        track_sets,
        {
            "track_set_id",
            "sequence_id",
            "group_id",
            "scene_id",
            "object_anchor_id",
            "base_camera_id",
            "tracks_path",
            "canonical_points_path",
            "point_count",
            "num_frames",
            "visibility_source",
            "coordinate_convention",
        },
        "track_sets",
        "track_set_id",
        state,
    )

    _check_unique_ids(scenes, "scene_id", "scenes", state)
    _check_unique_ids(cameras, "group_id", "base_cameras", state)
    _check_unique_ids(sequences, "sequence_id", "sequences", state)
    _check_unique_ids(frames, "frame_id", "frames", state)
    _check_unique_ids(matched, "matched_relative_group_id", "matched_relative_groups", state)
    _check_unique_ids(track_sets, "track_set_id", "track_sets", state)
    track_sequence_counts = Counter(str(row.get("sequence_id")) for row in track_sets)
    for sequence_id, count in track_sequence_counts.items():
        if count != 1:
            state.error("schema", f"track_sets: sequence {sequence_id!r} has {count} rows, expected 1")


def _config_protocol(config: Mapping[str, Any], state: ValidationState) -> Dict[str, Any]:
    motion = config.get("motion", {})
    effective = config.get("effective_mode_settings", {})
    validation = config.get("validation", {})
    selection = config.get("selection", {})
    tracks = config.get("tracks", {})
    try:
        levels = [int(value) for value in motion["levels"]]
        delta_m = float(motion["delta_m"])
        axis = np.asarray(motion["axis_world"], dtype=np.float64)
    except Exception as exc:
        state.error("schema", f"config motion protocol is incomplete: {exc}")
        levels = [-2, -1, 0, 1, 2]
        delta_m = 0.04
        axis = np.array([1.0, 0.0, 0.0])
    if levels != [-2, -1, 0, 1, 2]:
        state.error("factorial_protocol", f"motion levels must be [-2,-1,0,1,2], got {levels}")
    if not math.isfinite(delta_m) or delta_m <= 0.0:
        state.error("factorial_protocol", f"delta_m must be positive and finite, got {delta_m}")
    if axis.shape != (3,) or not np.allclose(axis, np.array([1.0, 0.0, 0.0]), atol=1.0e-12):
        state.error("factorial_protocol", f"motion axis must be world +X [1,0,0], got {axis.tolist()}")
        axis = np.array([1.0, 0.0, 0.0])
    return {
        "levels": levels,
        "delta_m": delta_m,
        "axis": axis,
        "num_frames": int(effective.get("num_frames", 0) or 0),
        "resolution": [int(value) for value in effective.get("resolution", [0, 0])],
        "matrix_atol": float(validation.get("matrix_atol", 1.0e-7)),
        "displacement_atol_m": float(validation.get("displacement_atol_m", 1.0e-7)),
        "relative_atol_m": float(validation.get("relative_equation_atol_m", 1.0e-8)),
        "matched_mean_tol_px": float(validation.get("matched_uv_mean_tolerance_px", 0.05)),
        "matched_max_tol_px": float(validation.get("matched_uv_max_tolerance_px", 0.25)),
        "compensated_mean_tol_px": float(validation.get("compensated_uv_mean_tolerance_px", 0.05)),
        "compensated_max_tol_px": float(validation.get("compensated_uv_max_tolerance_px", 0.25)),
        "background_min_mad": float(validation.get("min_background_mean_abs_difference_8bit", 0.25)),
        "background_min_changed_fraction": float(validation.get("min_background_changed_fraction", 0.005)),
        "background_pixel_threshold": float(validation.get("background_changed_pixel_threshold_8bit", 2.0)),
        "min_mask_area_ratio": float(validation.get("min_target_mask_area_ratio", 0.0)),
        "max_edge_touch_ratio": float(validation.get("max_target_edge_touch_ratio", 1.0)),
        "edge_margin_px": int(selection.get("edge_margin_px", 0)),
        "min_in_image_fraction": float(selection.get("min_projected_point_in_image_fraction", 0.0)),
        "min_visible_fraction": float(selection.get("min_visible_surface_fraction", 0.0)),
        "min_initial_mask_area_ratio": float(selection.get("min_initial_mask_area_ratio", 0.0)),
        "max_initial_mask_area_ratio": float(selection.get("max_initial_mask_area_ratio", 1.0)),
        "max_initial_edge_touch_ratio": float(selection.get("max_initial_edge_touch_ratio", 1.0)),
        "track_point_count": int(tracks.get("num_surface_points", 0) or 0),
        "min_common_visible_points": int(tracks.get("min_common_visible_points", 1)),
        "depth_abs_tol_m": float(tracks.get("visibility_depth_abs_tolerance_m", 0.03)),
        "depth_rel_tol": float(tracks.get("visibility_depth_rel_tolerance", 0.015)),
        "visibility_pixel_radius": int(tracks.get("visibility_pixel_radius", 1)),
    }


def validate_counts_and_protocol(
    config: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    manifests: Mapping[str, Any],
    protocol: Mapping[str, Any],
    state: ValidationState,
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, List[Mapping[str, Any]]]]:
    scenes = manifests["scenes"]
    anchors = manifests["anchors"]
    cameras = manifests["base_cameras"]
    sequences = manifests["sequences"]
    frames = manifests["frames"]
    matched_rows = manifests["matched_relative_groups"]
    track_sets = manifests["track_sets"]
    sequence_by_id = {str(row["sequence_id"]): row for row in sequences if "sequence_id" in row}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        if "group_id" in sequence:
            grouped[str(sequence["group_id"])].append(sequence)

    observed = {
        "scene_count": len(scenes),
        "anchor_count": len(anchors),
        "group_count": len(grouped),
        "base_camera_count": len(cameras),
        "sequence_count": len(sequences),
        "frame_count": len(frames),
        "matched_relative_group_count": len(matched_rows),
        "track_set_count": len(track_sets),
        "rendered_frame_count": sum(bool(row.get("rendered", False)) for row in frames),
    }
    state.metrics["counts"] = {"observed": observed}
    for key, value in observed.items():
        if key in dataset_summary:
            try:
                declared = int(dataset_summary[key])
            except Exception:
                state.error("dataset_counts", f"dataset_summary.{key} is not an integer")
                continue
            if declared != value:
                state.error("dataset_counts", f"dataset_summary.{key}={declared}, observed {value}")
    expected: Dict[str, int] = {}
    for key in observed:
        expected_key = f"expected_{key}"
        if expected_key in dataset_summary:
            try:
                expected[key] = int(dataset_summary[expected_key])
            except Exception:
                state.error("dataset_counts", f"dataset_summary.{expected_key} is not an integer")
    for key, value in expected.items():
        if observed[key] != value:
            state.error("dataset_counts", f"expected {key}={value}, observed {observed[key]}")
    state.metrics["counts"]["declared_expected"] = expected

    # Recompute mode expectations from config rather than trusting the summary.
    effective = config.get("effective_mode_settings", {})
    run_scope = config.get("run_scope", {})
    if not isinstance(run_scope, Mapping):
        run_scope = {}
    if run_scope.get("scene_id") is not None:
        expected_scene_count = 1
    elif isinstance(effective.get("scene_ids"), list):
        expected_scene_count = len(effective["scene_ids"])
    else:
        expected_scene_count = int(effective.get("scene_count", 0))
    anchors_per_scene = (
        1 if run_scope.get("anchor_id") is not None else int(effective.get("anchors_per_scene", 0))
    )
    cameras_per_anchor = (
        1
        if run_scope.get("base_camera_id") is not None
        else int(effective.get("base_cameras_per_anchor", 0))
    )
    independently_expected = {
        "scene_count": expected_scene_count,
        "anchor_count": expected_scene_count * anchors_per_scene,
        "group_count": expected_scene_count * anchors_per_scene * cameras_per_anchor,
    }
    independently_expected["base_camera_count"] = independently_expected["group_count"]
    independently_expected["sequence_count"] = independently_expected["group_count"] * 25
    independently_expected["frame_count"] = independently_expected["sequence_count"] * int(
        protocol["num_frames"]
    )
    independently_expected["matched_relative_group_count"] = (
        independently_expected["group_count"] * 9
    )
    independently_expected["track_set_count"] = independently_expected["sequence_count"]
    independently_expected["rendered_frame_count"] = (
        0 if bool(config.get("dry_run", False)) else independently_expected["frame_count"]
    )
    for key, value in independently_expected.items():
        if observed[key] != value:
            state.error(
                "dataset_counts",
                f"config/run-scope expects {key}={value}, observed {observed[key]}",
            )
    state.metrics["counts"]["independently_expected_from_config"] = independently_expected

    levels = list(protocol["levels"])
    expected_conditions = {(ego, obj) for ego in levels for obj in levels}
    delta = float(protocol["delta_m"])
    axis = np.asarray(protocol["axis"], dtype=np.float64)
    matrix_atol = float(protocol["matrix_atol"])
    context_to_group_ids: Dict[Tuple[str, str, str], set[str]] = defaultdict(set)

    scene_ids = {str(row.get("scene_id")) for row in scenes}
    anchor_contexts = {
        (str(row.get("scene_id")), str(row.get("object_anchor_id"))) for row in anchors
    }
    camera_groups = {str(row.get("group_id")) for row in cameras}
    for group_id, rows in sorted(grouped.items()):
        conditions = [(int(row.get("ego_level", 999)), int(row.get("object_level", 999))) for row in rows]
        condition_counts = Counter(conditions)
        if set(conditions) != expected_conditions or any(count != 1 for count in condition_counts.values()):
            missing = sorted(expected_conditions - set(conditions))
            extra = sorted(set(conditions) - expected_conditions)
            duplicates = sorted(condition for condition, count in condition_counts.items() if count > 1)
            state.error(
                "factorial_protocol",
                f"{group_id}: expected exactly 25 unique conditions; count={len(rows)}, "
                f"missing={missing}, extra={extra}, duplicates={duplicates}",
            )
        contexts = {
            (
                str(row.get("scene_id")),
                str(row.get("object_anchor_id")),
                str(row.get("base_camera_id")),
            )
            for row in rows
        }
        if len(contexts) != 1:
            state.error("factorial_protocol", f"{group_id}: sequences do not share one scene/anchor/base-camera context")
        else:
            context = next(iter(contexts))
            context_to_group_ids[context].add(group_id)
            if context[0] not in scene_ids:
                state.error("schema", f"{group_id}: unknown scene {context[0]}")
            if context[:2] not in anchor_contexts:
                state.error("schema", f"{group_id}: unknown anchor {context[0]}/{context[1]}")
        if group_id not in camera_groups:
            state.error("schema", f"{group_id}: no base_cameras manifest row")

        identity_reference: Optional[Tuple[Any, ...]] = None
        for row in rows:
            sequence_id = str(row.get("sequence_id"))
            try:
                ego = int(row["ego_level"])
                obj = int(row["object_level"])
                rel = int(row["relative_level"])
                row_delta = float(row["delta_m"])
                ego_amplitude = float(row["ego_amplitude_m"])
                object_amplitude = float(row["object_amplitude_m"])
                relative_amplitude = float(row["relative_amplitude_m"])
                row_axis = np.asarray(row["motion_axis_world"], dtype=np.float64)
            except Exception as exc:
                state.error("factorial_protocol", f"{sequence_id}: malformed factor fields: {exc}")
                continue
            if rel != obj - ego:
                state.error("factorial_protocol", f"{sequence_id}: relative_level {rel} != object_level-ego_level {obj-ego}")
            if abs(row_delta - delta) > matrix_atol:
                state.error("factorial_protocol", f"{sequence_id}: delta_m {row_delta} != config {delta}")
            if abs(ego_amplitude - ego * delta) > matrix_atol:
                state.error("factorial_protocol", f"{sequence_id}: ego amplitude does not equal ego_level*delta")
            if abs(object_amplitude - obj * delta) > matrix_atol:
                state.error("factorial_protocol", f"{sequence_id}: object amplitude does not equal object_level*delta")
            if abs(relative_amplitude - (obj - ego) * delta) > matrix_atol:
                state.error("factorial_protocol", f"{sequence_id}: relative amplitude is inconsistent")
            if row_axis.shape != (3,) or not np.allclose(row_axis, axis, atol=matrix_atol):
                state.error("factorial_protocol", f"{sequence_id}: motion_axis_world is not config world-X axis")
            expected_compensated = ego == obj and ego != 0
            expected_static = ego == 0 and obj == 0
            if bool(row.get("is_compensated_motion")) != expected_compensated:
                state.error("factorial_protocol", f"{sequence_id}: incorrect is_compensated_motion flag")
            if bool(row.get("is_static")) != expected_static:
                state.error("factorial_protocol", f"{sequence_id}: incorrect is_static flag")
            if protocol["num_frames"] and int(row.get("num_frames", -1)) != protocol["num_frames"]:
                state.error("dataset_counts", f"{sequence_id}: num_frames differs from effective mode settings")
            identity = tuple(json.dumps(row.get(key), sort_keys=True) for key in TARGET_IDENTITY_KEYS)
            if identity_reference is None:
                identity_reference = identity
            elif identity != identity_reference:
                state.error("factorial_protocol", f"{group_id}: target identity differs across conditions")

            _safe_relative_path(Path("."), row.get("sequence_dir"), f"{sequence_id}.sequence_dir", state)
            _safe_relative_path(Path("."), row.get("tracks_path"), f"{sequence_id}.tracks_path", state)

    for context, group_ids in context_to_group_ids.items():
        if len(group_ids) != 1:
            state.error("factorial_protocol", f"context {context} maps to multiple group IDs: {sorted(group_ids)}")

    frame_counts = Counter(str(row.get("sequence_id")) for row in frames)
    for sequence_id, sequence in sequence_by_id.items():
        expected_frames = int(sequence.get("num_frames", -1))
        if frame_counts[sequence_id] != expected_frames:
            state.error(
                "dataset_counts",
                f"{sequence_id}: expected {expected_frames} frame rows, observed {frame_counts[sequence_id]}",
            )
    unknown_frame_sequences = sorted(set(frame_counts) - set(sequence_by_id))
    if unknown_frame_sequences:
        state.error("schema", f"frames reference unknown sequences: {unknown_frame_sequences[:10]}")
    track_sequence_ids = {str(row.get("sequence_id")) for row in track_sets}
    missing_track_sets = sorted(set(sequence_by_id) - track_sequence_ids)
    extra_track_sets = sorted(track_sequence_ids - set(sequence_by_id))
    if missing_track_sets:
        state.error("dataset_counts", f"sequences missing track-set rows: {missing_track_sets[:10]}")
    if extra_track_sets:
        state.error("schema", f"track-set rows reference unknown sequences: {extra_track_sets[:10]}")

    validate_matched_relative_manifest(matched_rows, grouped, state)
    return sequence_by_id, grouped


def validate_matched_relative_manifest(
    rows: Sequence[Mapping[str, Any]],
    sequences_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    state: ValidationState,
) -> None:
    expected: Dict[Tuple[str, str, str, int], set[Tuple[str, int, int]]] = defaultdict(set)
    for sequences in sequences_by_group.values():
        for sequence in sequences:
            key = (
                str(sequence.get("scene_id")),
                str(sequence.get("object_anchor_id")),
                str(sequence.get("base_camera_id")),
                int(sequence.get("relative_level", 999)),
            )
            expected[key].add(
                (
                    str(sequence.get("sequence_id")),
                    int(sequence.get("ego_level", 999)),
                    int(sequence.get("object_level", 999)),
                )
            )
    observed: Dict[Tuple[str, str, str, int], set[Tuple[str, int, int]]] = {}
    for row in rows:
        anchor_id = str(row.get("object_anchor_id", row.get("anchor_id")))
        try:
            key = (
                str(row["scene_id"]),
                anchor_id,
                str(row["base_camera_id"]),
                int(row["relative_level"]),
            )
        except Exception as exc:
            state.error("factorial_protocol", f"malformed matched-relative group: {exc}")
            continue
        members = row.get("members")
        if not isinstance(members, list):
            state.error("factorial_protocol", f"{row.get('matched_relative_group_id')}: members must be a list")
            continue
        member_set: set[Tuple[str, int, int]] = set()
        for member in members:
            try:
                member_set.add(
                    (
                        str(member["sequence_id"]),
                        int(member["ego_level"]),
                        int(member["object_level"]),
                    )
                )
            except Exception as exc:
                state.error("factorial_protocol", f"{row.get('matched_relative_group_id')}: malformed member: {exc}")
        if key in observed:
            state.error("factorial_protocol", f"duplicate matched-relative context {key}")
        observed[key] = member_set
        expected_count = 5 - abs(key[3]) if abs(key[3]) <= 4 else 0
        if len(member_set) != expected_count:
            state.error(
                "factorial_protocol",
                f"{row.get('matched_relative_group_id')}: relative level {key[3]} has {len(member_set)} members, expected {expected_count}",
            )
        if "member_count" in row and int(row["member_count"]) != len(member_set):
            state.error("factorial_protocol", f"{row.get('matched_relative_group_id')}: member_count mismatch")
        expected_multiple = len(member_set) > 1
        if "has_multiple_causal_decompositions" in row and bool(row["has_multiple_causal_decompositions"]) != expected_multiple:
            state.error("factorial_protocol", f"{row.get('matched_relative_group_id')}: incorrect causal-decomposition flag")
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing:
        state.error("factorial_protocol", f"matched-relative manifest missing contexts: {missing[:10]}")
    if extra:
        state.error("factorial_protocol", f"matched-relative manifest has extra contexts: {extra[:10]}")
    for key in set(expected) & set(observed):
        if expected[key] != observed[key]:
            state.error("factorial_protocol", f"matched-relative members disagree for context {key}")


def validate_scene_splits(
    manifests: Mapping[str, Any],
    sequence_by_id: Mapping[str, Mapping[str, Any]],
    state: ValidationState,
) -> None:
    scenes = manifests["scenes"]
    splits = manifests["splits"]
    scene_split: Dict[str, str] = {}
    for row in scenes:
        scene_id = str(row.get("scene_id"))
        split = str(row.get("split"))
        if scene_id in scene_split and scene_split[scene_id] != split:
            state.error("scene_splits", f"{scene_id}: appears in multiple splits")
        scene_split[scene_id] = split

    declared: Dict[str, str] = {}
    if isinstance(splits.get("scene_to_split"), Mapping):
        for scene_id, split in splits["scene_to_split"].items():
            declared[str(scene_id)] = str(split)
    selected_lists_present = any(
        isinstance(splits.get(key), list)
        for key in ("selected_train_scenes", "selected_validation_scenes", "selected_test_scenes")
    )
    list_keys = (
        (
            ("selected_train_scenes", "train"),
            ("selected_validation_scenes", "val"),
            ("selected_test_scenes", "test"),
        )
        if selected_lists_present
        else (
            ("train_scenes", "train"),
            ("validation_scenes", "val"),
            ("val_scenes", "val"),
            ("test_scenes", "test"),
        )
    )
    for key, split in list_keys:
        value = splits.get(key)
        if not isinstance(value, list):
            continue
        for scene_id in value:
            scene_id = str(scene_id)
            if scene_id in declared and declared[scene_id] != split:
                state.error("scene_splits", f"{scene_id}: split manifest assigns both {declared[scene_id]} and {split}")
            declared[scene_id] = split
    if not declared:
        state.error("scene_splits", "splits.json has no scene membership mapping")
    for scene_id, split in scene_split.items():
        normalized = "val" if split == "validation" else split
        declared_normalized = "val" if declared.get(scene_id) == "validation" else declared.get(scene_id)
        if declared_normalized != normalized:
            state.error("scene_splits", f"{scene_id}: scene row split={split}, splits.json={declared.get(scene_id)}")
    extra = sorted(set(declared) - set(scene_split))
    if extra:
        state.warning("scene_splits", f"splits.json includes source scenes not selected in this mode: {extra}")

    groups_to_splits: Dict[str, set[str]] = defaultdict(set)
    for sequence_id, sequence in sequence_by_id.items():
        scene_id = str(sequence.get("scene_id"))
        split = str(sequence.get("split"))
        if scene_id not in scene_split:
            state.error("scene_splits", f"{sequence_id}: unknown scene {scene_id}")
        elif split != scene_split[scene_id]:
            state.error("scene_splits", f"{sequence_id}: split={split}, scene split={scene_split[scene_id]}")
        groups_to_splits[str(sequence.get("group_id"))].add(split)
    for group_id, values in groups_to_splits.items():
        if len(values) != 1:
            state.error("scene_splits", f"{group_id}: conditions cross splits: {sorted(values)}")
    state.metrics["scene_splits"] = {
        "scene_to_split": scene_split,
        "split_scene_counts": dict(Counter(scene_split.values())),
        "unit": splits.get("unit"),
    }


def validate_all_paths(root: Path, manifests: Mapping[str, Any], state: ValidationState) -> None:
    for sequence in manifests["sequences"]:
        sequence_id = str(sequence.get("sequence_id"))
        _safe_relative_path(root, sequence.get("sequence_dir"), f"{sequence_id}.sequence_dir", state)
        _safe_relative_path(root, sequence.get("tracks_path"), f"{sequence_id}.tracks_path", state)
    for row in manifests["track_sets"]:
        track_set_id = str(row.get("track_set_id"))
        _safe_relative_path(root, row.get("tracks_path"), f"{track_set_id}.tracks_path", state)
        _safe_relative_path(root, row.get("canonical_points_path"), f"{track_set_id}.canonical_points_path", state)
    for frame in manifests["frames"]:
        frame_id = str(frame.get("frame_id"))
        paths = frame.get("frame_paths")
        if not isinstance(paths, Mapping):
            state.error("path_convention", f"{frame_id}.frame_paths must be a mapping")
            continue
        for key, value in paths.items():
            _safe_relative_path(root, value, f"{frame_id}.frame_paths.{key}", state)


def validate_selection_report(
    manifests: Mapping[str, Any], protocol: Mapping[str, Any], state: ValidationState
) -> None:
    report = manifests["selection_report"]
    scene_reports = report.get("scene_reports")
    if not isinstance(scene_reports, list):
        state.error("selection_collision_visibility", "selection_report.scene_reports must be a list")
        return
    levels = list(protocol["levels"])
    delta = float(protocol["delta_m"])
    axis = np.asarray(protocol["axis"], dtype=np.float64)
    atol = float(protocol["displacement_atol_m"])
    selected: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    candidate_count = 0
    rejected_count = 0

    for scene_report in scene_reports:
        scene_id = str(scene_report.get("scene_id"))
        if not bool(scene_report.get("selection_is_deterministic", False)):
            state.error("selection_collision_visibility", f"{scene_id}: selection_is_deterministic is not true")
        if not bool(scene_report.get("selection_ok", False)):
            state.error(
                "selection_collision_visibility",
                f"{scene_id}: selection did not complete: {scene_report.get('error')}",
            )
        candidates = scene_report.get("anchor_candidates")
        groups = scene_report.get("selected_groups")
        if not isinstance(candidates, list) or not isinstance(groups, list):
            state.error("selection_collision_visibility", f"{scene_id}: missing candidate/selected group lists")
            continue
        candidate_count += len(candidates)
        rejected_count += sum(not bool(candidate.get("valid", False)) for candidate in candidates)
        for group in groups:
            anchor_id = str(group.get("anchor_id", group.get("object_anchor_id")))
            camera_id = str(group.get("base_camera_id"))
            key = (scene_id, anchor_id, camera_id)
            if key in selected:
                state.error("selection_collision_visibility", f"duplicate selected group {key}")
            selected[key] = group
            _validate_one_selection(group, key, levels, delta, axis, atol, protocol, state)

    manifest_groups = {
        (str(row.get("scene_id")), str(row.get("object_anchor_id")), str(row.get("base_camera_id")))
        for row in manifests["base_cameras"]
    }
    if set(selected) != manifest_groups:
        missing = sorted(manifest_groups - set(selected))
        extra = sorted(set(selected) - manifest_groups)
        state.error(
            "selection_collision_visibility",
            f"selection report/base-camera manifest mismatch; missing={missing}, extra={extra}",
        )

    anchors_by_context: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for anchor in manifests["anchors"]:
        anchor_key = (str(anchor.get("scene_id")), str(anchor.get("object_anchor_id")))
        anchors_by_context[anchor_key] = anchor
        _validate_anchor_sweep(
            anchor.get("anchor_sweep"),
            anchor.get("anchor_root_xy"),
            (anchor_key[0], anchor_key[1], "anchor"),
            levels,
            delta,
            axis,
            atol,
            state,
        )
    for camera_row in manifests["base_cameras"]:
        key = (
            str(camera_row.get("scene_id")),
            str(camera_row.get("object_anchor_id")),
            str(camera_row.get("base_camera_id")),
        )
        anchor = anchors_by_context.get(key[:2])
        combined = {
            "anchor_root_xy": anchor.get("anchor_root_xy") if anchor else None,
            "anchor_sweep": anchor.get("anchor_sweep") if anchor else None,
            "camera_selection": camera_row.get("camera_selection"),
        }
        _validate_one_selection(combined, key, levels, delta, axis, atol, protocol, state)
        selected_row = selected.get(key)
        if selected_row is not None:
            for name, source_value in (
                ("anchor_sweep", combined["anchor_sweep"]),
                ("camera_selection", combined["camera_selection"]),
            ):
                if json.dumps(source_value, sort_keys=True) != json.dumps(selected_row.get(name), sort_keys=True):
                    state.error("selection_collision_visibility", f"{key}: manifest {name} differs from selection report")
    state.metrics["selection"] = {
        "scene_report_count": len(scene_reports),
        "candidate_anchor_count": candidate_count,
        "rejected_anchor_count": rejected_count,
        "selected_group_count": len(selected),
        "evidence_scope": "saved exhaustive anchor-sweep and camera endpoint reports",
    }


def _validate_one_selection(
    row: Mapping[str, Any],
    key: Tuple[str, str, str],
    levels: Sequence[int],
    delta: float,
    axis: np.ndarray,
    atol: float,
    protocol: Mapping[str, Any],
    state: ValidationState,
) -> None:
    sweep = row.get("anchor_sweep")
    camera = row.get("camera_selection")
    if not isinstance(sweep, Mapping) or not isinstance(camera, Mapping):
        state.error("selection_collision_visibility", f"{key}: missing anchor_sweep/camera_selection")
        return
    _validate_anchor_sweep(
        sweep, row.get("anchor_root_xy"), key, levels, delta, axis, atol, state
    )

    if not bool(camera.get("valid", False)):
        state.error("selection_collision_visibility", f"{key}: selected camera is invalid")
    if camera.get("rejection_reasons"):
        state.error("selection_collision_visibility", f"{key}: selected camera has rejection reasons {camera.get('rejection_reasons')}")
    endpoints = camera.get("endpoint_stats")
    expected_conditions = {(ego, obj) for ego in levels for obj in levels}
    if not isinstance(endpoints, list):
        state.error("selection_collision_visibility", f"{key}: camera endpoint_stats must be a list")
        return
    conditions = Counter((int(item.get("ego_level", 999)), int(item.get("object_level", 999))) for item in endpoints)
    if set(conditions) != expected_conditions or any(value != 1 for value in conditions.values()):
        state.error("selection_collision_visibility", f"{key}: camera selection does not cover all 25 endpoint conditions")
    for item in endpoints:
        condition = (item.get("ego_level"), item.get("object_level"))
        if float(item.get("in_image_fraction", -1.0)) < float(protocol["min_in_image_fraction"]):
            state.error("selection_collision_visibility", f"{key}/{condition}: insufficient in-image surface fraction")
        if float(item.get("visible_surface_fraction", -1.0)) < float(protocol["min_visible_fraction"]):
            state.error("selection_collision_visibility", f"{key}/{condition}: insufficient visible surface fraction")
        if float(item.get("projected_bbox_margin_px", -float("inf"))) < float(protocol["edge_margin_px"]):
            state.error("selection_collision_visibility", f"{key}/{condition}: projected bbox violates edge margin")
    initial_ratio = camera.get("initial_mask_area_ratio")
    if initial_ratio is None:
        state.error("selection_collision_visibility", f"{key}: no initial source mask-area evidence")
    else:
        ratio = float(initial_ratio)
        if not (float(protocol["min_initial_mask_area_ratio"]) <= ratio <= float(protocol["max_initial_mask_area_ratio"])):
            state.error("selection_collision_visibility", f"{key}: initial mask area ratio {ratio} is outside selection bounds")
    if bool(camera.get("initial_target_truncated", True)):
        state.error("selection_collision_visibility", f"{key}: initial target is truncated")
    if float(camera.get("initial_edge_touch_ratio", float("inf"))) > float(protocol["max_initial_edge_touch_ratio"]):
        state.error("selection_collision_visibility", f"{key}: initial target touches image edge")


def _validate_anchor_sweep(
    sweep: Any,
    anchor_root_xy: Any,
    key: Tuple[str, str, str],
    levels: Sequence[int],
    delta: float,
    axis: np.ndarray,
    atol: float,
    state: ValidationState,
) -> None:
    if not isinstance(sweep, Mapping):
        state.error("selection_collision_visibility", f"{key}: missing anchor_sweep")
        return
    if not bool(sweep.get("valid", False)):
        state.error("selection_collision_visibility", f"{key}: selected anchor sweep is invalid")
    if not bool(sweep.get("inside_room_for_full_sweep", False)):
        state.error("selection_collision_visibility", f"{key}: full sweep violates room boundary")
    swept_collision_free = sweep.get(
        "collision_free_for_full_swept_volume",
        sweep.get("collision_free_for_full_sweep", False),
    )
    if not bool(swept_collision_free):
        state.error("selection_collision_visibility", f"{key}: full sweep reports a collision")
    checks = sweep.get("checks")
    if not isinstance(checks, list):
        state.error("selection_collision_visibility", f"{key}: anchor sweep checks must be a list")
    else:
        observed_levels = Counter(int(item.get("object_level", 999)) for item in checks)
        if observed_levels != Counter(levels):
            state.error("selection_collision_visibility", f"{key}: anchor sweep does not cover every motion level once")
        base_xy = anchor_root_xy
        base = _array(base_xy, (2,), f"{key}.anchor_root_xy", "selection_collision_visibility", state) if base_xy is not None else None
        for item in checks:
            level = int(item.get("object_level", 999))
            expected_dx = level * delta
            if abs(float(item.get("object_dx_m", float("inf"))) - expected_dx) > atol:
                state.error("selection_collision_visibility", f"{key}: sweep level {level} has wrong displacement")
            if not bool(item.get("inside_room", False)):
                state.error("selection_collision_visibility", f"{key}: sweep level {level} leaves room boundary")
            if not bool(item.get("collision_free", False)):
                state.error("selection_collision_visibility", f"{key}: sweep level {level} collides")
            if base is not None and "root_xy" in item:
                root_xy = _array(item["root_xy"], (2,), f"{key}.level{level}.root_xy", "selection_collision_visibility", state)
                if root_xy is not None:
                    expected_xy = base + axis[:2] * expected_dx
                    if not np.allclose(root_xy, expected_xy, atol=atol):
                        state.error("selection_collision_visibility", f"{key}: sweep root position is inconsistent at level {level}")



def group_frames(
    frames: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    result: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for frame in frames:
        result[str(frame.get("sequence_id"))].append(frame)
    for values in result.values():
        values.sort(key=lambda row: int(row.get("frame_index", -1)))
    return result


def validate_frame_geometry(
    manifests: Mapping[str, Any],
    sequence_by_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    state: ValidationState,
) -> Dict[str, List[Mapping[str, Any]]]:
    frames_by_sequence = group_frames(manifests["frames"])
    matrix_atol = float(protocol["matrix_atol"])
    displacement_atol = float(protocol["displacement_atol_m"])
    relative_atol = float(protocol["relative_atol_m"])
    axis = np.asarray(protocol["axis"], dtype=np.float64)
    global_max = {
        "camera_inverse_error": 0.0,
        "camera_rotation_deviation": 0.0,
        "camera_off_axis_displacement_m": 0.0,
        "camera_displacement_error_m": 0.0,
        "object_displacement_error_m": 0.0,
        "object_linear_transform_deviation": 0.0,
        "intrinsics_deviation": 0.0,
        "relative_equation_error_m": 0.0,
        "target_center_camera_error_m": 0.0,
        "frame0_group_matrix_deviation": 0.0,
    }
    per_sequence_endpoints: List[Dict[str, Any]] = []
    group_frame0: Dict[str, Dict[str, np.ndarray]] = {}

    for sequence_id, sequence in sequence_by_id.items():
        rows = frames_by_sequence.get(sequence_id, [])
        expected_count = int(sequence.get("num_frames", 0))
        indices = [int(row.get("frame_index", -1)) for row in rows]
        if indices != list(range(expected_count)):
            state.error("frame_geometry", f"{sequence_id}: frame indices are {indices}, expected 0..{expected_count-1}")
        if not rows:
            continue
        base_values: Optional[Dict[str, np.ndarray]] = None
        endpoint_camera_error = None
        endpoint_object_error = None
        for frame in rows:
            frame_id = str(frame.get("frame_id"))
            for key in ("group_id", "scene_id", "object_anchor_id", "base_camera_id"):
                if str(frame.get(key)) != str(sequence.get(key)):
                    state.error("frame_geometry", f"{frame_id}: {key} differs from sequence")
            try:
                index = int(frame["frame_index"])
                alpha = float(frame["alpha_t"])
                actual_ego = float(frame["actual_ego_dx_m"])
                actual_object = float(frame["actual_object_dx_m"])
                actual_relative = float(frame["actual_relative_dx_m"])
            except Exception as exc:
                state.error("frame_geometry", f"{frame_id}: malformed time/displacement fields: {exc}")
                continue
            expected_alpha = index / (expected_count - 1) if expected_count > 1 else float("nan")
            if not math.isfinite(alpha) or abs(alpha - expected_alpha) > relative_atol:
                state.error("frame_geometry", f"{frame_id}: alpha_t={alpha}, expected {expected_alpha}")
            expected_ego = float(sequence.get("ego_amplitude_m", 0.0)) * expected_alpha
            expected_object = float(sequence.get("object_amplitude_m", 0.0)) * expected_alpha
            expected_relative = expected_object - expected_ego
            if abs(actual_ego - expected_ego) > displacement_atol:
                state.error("endpoint_displacements", f"{frame_id}: actual ego displacement is not linear protocol value")
            if abs(actual_object - expected_object) > displacement_atol:
                state.error("endpoint_displacements", f"{frame_id}: actual object displacement is not linear protocol value")
            relative_error = abs(actual_relative - (actual_object - actual_ego))
            global_max["relative_equation_error_m"] = max(global_max["relative_equation_error_m"], relative_error)
            if relative_error > relative_atol or abs(actual_relative - expected_relative) > relative_atol:
                state.error("relative_equation", f"{frame_id}: relative displacement equation failed")

            c2w = _matrix4(frame.get("camera_to_world"), f"{frame_id}.camera_to_world", "frame_geometry", state)
            w2c = _matrix4(frame.get("world_to_camera"), f"{frame_id}.world_to_camera", "frame_geometry", state)
            blender_c2w = _matrix4(frame.get("blender_camera_to_world"), f"{frame_id}.blender_camera_to_world", "frame_geometry", state)
            object_to_world = _matrix4(frame.get("object_to_world"), f"{frame_id}.object_to_world", "frame_geometry", state)
            camera_rotation = _array(frame.get("camera_rotation"), (3, 3), f"{frame_id}.camera_rotation", "frame_geometry", state)
            camera_position = _array(frame.get("camera_world_position"), (3,), f"{frame_id}.camera_world_position", "frame_geometry", state)
            k = _array(frame.get("K"), (3, 3), f"{frame_id}.K", "frame_geometry", state)
            target_world = _array(_target_world_center(frame), (3,), f"{frame_id}.target_world_center", "frame_geometry", state)
            target_camera = _array(_target_camera_center(frame), (3,), f"{frame_id}.target_camera_center", "frame_geometry", state)
            if any(value is None for value in (c2w, w2c, blender_c2w, object_to_world, camera_rotation, camera_position, k, target_world, target_camera)):
                continue
            assert c2w is not None and w2c is not None and blender_c2w is not None
            assert object_to_world is not None and camera_rotation is not None and camera_position is not None
            assert k is not None and target_world is not None and target_camera is not None

            inverse_error = _max_abs_difference(c2w @ w2c, np.eye(4))
            global_max["camera_inverse_error"] = max(global_max["camera_inverse_error"], inverse_error)
            if inverse_error > matrix_atol:
                state.error("frame_geometry", f"{frame_id}: camera_to_world/world_to_camera are not inverses (max {inverse_error:.3g})")
            if not np.allclose(camera_position, c2w[:3, 3], atol=matrix_atol):
                state.error("frame_geometry", f"{frame_id}: camera_world_position differs from OpenCV camera_to_world translation")
            if not np.allclose(camera_position, blender_c2w[:3, 3], atol=matrix_atol):
                state.error("frame_geometry", f"{frame_id}: camera position differs between Blender and OpenCV transforms")
            if not np.allclose(camera_rotation, blender_c2w[:3, :3], atol=matrix_atol):
                state.error("frame_geometry", f"{frame_id}: camera_rotation differs from Blender pose rotation")
            orthogonal_error = _max_abs_difference(camera_rotation.T @ camera_rotation, np.eye(3))
            if orthogonal_error > matrix_atol or abs(float(np.linalg.det(camera_rotation)) - 1.0) > matrix_atol * 10:
                state.error("frame_geometry", f"{frame_id}: camera_rotation is not a proper rotation")
            if k[0, 0] <= 0.0 or k[1, 1] <= 0.0 or abs(k[2, 2] - 1.0) > matrix_atol:
                state.error("frame_geometry", f"{frame_id}: invalid camera intrinsics")
            recomputed_target_camera = _transform_point(w2c, target_world)
            center_error = _max_abs_difference(recomputed_target_camera, target_camera)
            global_max["target_center_camera_error_m"] = max(global_max["target_center_camera_error_m"], center_error)
            if center_error > max(matrix_atol * 10.0, 1.0e-7):
                state.error("frame_geometry", f"{frame_id}: target camera center is inconsistent (max {center_error:.3g} m)")

            values = {
                "c2w": c2w,
                "w2c": w2c,
                "blender_c2w": blender_c2w,
                "camera_rotation": camera_rotation,
                "camera_position": camera_position,
                "K": k,
                "object_to_world": object_to_world,
                "target_world": target_world,
            }
            if base_values is None:
                base_values = values
                if index != 0:
                    state.error("frame_geometry", f"{sequence_id}: first frame is not frame 0")
                if abs(actual_ego) > displacement_atol or abs(actual_object) > displacement_atol or abs(actual_relative) > relative_atol:
                    state.error("frame0_identity", f"{frame_id}: frame-0 displacement is nonzero")
                group_id = str(sequence.get("group_id"))
                if group_id not in group_frame0:
                    group_frame0[group_id] = {name: value.copy() for name, value in values.items()}
                else:
                    reference = group_frame0[group_id]
                    for name, value in values.items():
                        deviation = _max_abs_difference(value, reference[name])
                        global_max["frame0_group_matrix_deviation"] = max(global_max["frame0_group_matrix_deviation"], deviation)
                        if deviation > matrix_atol:
                            state.error("frame0_identity", f"{group_id}: frame-0 {name} differs across factorial conditions (max {deviation:.3g})")
            else:
                rotation_deviation = max(
                    _max_abs_difference(camera_rotation, base_values["camera_rotation"]),
                    _max_abs_difference(blender_c2w[:3, :3], base_values["blender_c2w"][:3, :3]),
                    _max_abs_difference(c2w[:3, :3], base_values["c2w"][:3, :3]),
                )
                global_max["camera_rotation_deviation"] = max(global_max["camera_rotation_deviation"], rotation_deviation)
                if rotation_deviation > matrix_atol:
                    state.error("camera_rotation_invariance", f"{frame_id}: camera rotation changed (max {rotation_deviation:.3g})")
                object_linear_deviation = _max_abs_difference(object_to_world[:3, :3], base_values["object_to_world"][:3, :3])
                global_max["object_linear_transform_deviation"] = max(global_max["object_linear_transform_deviation"], object_linear_deviation)
                if object_linear_deviation > matrix_atol:
                    state.error("endpoint_displacements", f"{frame_id}: target orientation or scale changed")
                intrinsics_deviation = _max_abs_difference(k, base_values["K"])
                global_max["intrinsics_deviation"] = max(global_max["intrinsics_deviation"], intrinsics_deviation)
                if intrinsics_deviation > matrix_atol:
                    state.error("camera_rotation_invariance", f"{frame_id}: camera intrinsics changed")

            if base_values is not None:
                camera_delta = camera_position - base_values["camera_position"]
                object_delta = target_world - base_values["target_world"]
                expected_camera_delta = axis * actual_ego
                expected_object_delta = axis * actual_object
                camera_error = float(np.linalg.norm(camera_delta - expected_camera_delta))
                object_error = float(np.linalg.norm(object_delta - expected_object_delta))
                camera_off_axis = float(np.linalg.norm(camera_delta - axis * float(np.dot(camera_delta, axis))))
                global_max["camera_displacement_error_m"] = max(global_max["camera_displacement_error_m"], camera_error)
                global_max["object_displacement_error_m"] = max(global_max["object_displacement_error_m"], object_error)
                global_max["camera_off_axis_displacement_m"] = max(global_max["camera_off_axis_displacement_m"], camera_off_axis)
                if camera_error > displacement_atol or camera_off_axis > displacement_atol:
                    state.error("endpoint_displacements", f"{frame_id}: camera is not a pure intended world-X translation")
                if object_error > displacement_atol:
                    state.error("endpoint_displacements", f"{frame_id}: target is not at intended world-X displacement")
                object_origin_delta = object_to_world[:3, 3] - base_values["object_to_world"][:3, 3]
                if not np.allclose(object_origin_delta, expected_object_delta, atol=displacement_atol):
                    state.error("endpoint_displacements", f"{frame_id}: object root transform has wrong translation")
                if index == expected_count - 1:
                    endpoint_camera_error = camera_error
                    endpoint_object_error = object_error
        if endpoint_camera_error is not None:
            per_sequence_endpoints.append(
                {
                    "sequence_id": sequence_id,
                    "ego_level": int(sequence.get("ego_level", 0)),
                    "object_level": int(sequence.get("object_level", 0)),
                    "camera_error_m": endpoint_camera_error,
                    "object_error_m": endpoint_object_error,
                }
            )

    state.metrics["frame_geometry"] = global_max
    state.metrics["endpoint_displacements"] = {
        "max_camera_error_m": global_max["camera_displacement_error_m"],
        "max_object_error_m": global_max["object_displacement_error_m"],
        "max_camera_off_axis_m": global_max["camera_off_axis_displacement_m"],
        "sequence_count_checked": len(per_sequence_endpoints),
    }
    state.metrics["frame0_identity"] = {
        "max_numeric_deviation": global_max["frame0_group_matrix_deviation"]
    }
    state.metrics["camera_rotation_invariance"] = {
        "max_rotation_matrix_element_deviation": global_max["camera_rotation_deviation"],
        "max_intrinsics_element_deviation": global_max["intrinsics_deviation"],
    }
    state.metrics["relative_equation"] = {
        "max_absolute_error_m": global_max["relative_equation_error_m"]
    }
    return frames_by_sequence


def _load_npz(path: Path, label: str, state: ValidationState) -> Optional[Dict[str, np.ndarray]]:
    if not path.is_file():
        state.error("track_identity", f"{label}: missing NPZ file")
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]).copy() for key in archive.files}
    except Exception as exc:
        state.error("track_identity", f"{label}: cannot read NPZ: {exc}")
        return None


def _finite_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def _load_canonical_points(
    path: Path,
    cache: MutableMapping[Path, Optional[Dict[str, np.ndarray]]],
    state: ValidationState,
) -> Optional[Dict[str, np.ndarray]]:
    if path not in cache:
        cache[path] = _load_npz(path, f"canonical points {path.name}", state)
    return cache[path]


def validate_tracks(
    root: Path,
    manifests: Mapping[str, Any],
    sequence_by_id: Mapping[str, Mapping[str, Any]],
    frames_by_sequence: Mapping[str, Sequence[Mapping[str, Any]]],
    protocol: Mapping[str, Any],
    geometry_only: bool,
    state: ValidationState,
) -> Dict[str, Path]:
    track_rows = {str(row.get("sequence_id")): row for row in manifests["track_sets"]}
    track_paths: Dict[str, Path] = {}
    canonical_cache: Dict[Path, Optional[Dict[str, np.ndarray]]] = {}
    group_canonical: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}
    point_counts: List[int] = []
    visible_fractions: List[float] = []
    maxima = {
        "xyz_world_error_m": 0.0,
        "xyz_camera_error_m": 0.0,
        "projection_error_px": 0.0,
        "depth_z_error_m": 0.0,
        "range_error_m": 0.0,
        "rendered_depth_sample_error_m": 0.0,
    }
    geometry_atol = max(1.0e-7, float(protocol["matrix_atol"]) * 10.0)
    projection_atol = max(1.0e-4, float(protocol["matrix_atol"]) * 1000.0)

    for sequence_id, sequence in sequence_by_id.items():
        row = track_rows.get(sequence_id)
        if row is None:
            continue
        for key in ("group_id", "scene_id", "object_anchor_id", "base_camera_id"):
            if str(row.get(key)) != str(sequence.get(key)):
                state.error("track_identity", f"{sequence_id}: track-set {key} differs from sequence")
        if int(row.get("num_frames", -1)) != int(sequence.get("num_frames", -2)):
            state.error("track_identity", f"{sequence_id}: track-set num_frames differs from sequence")
        if str(row.get("tracks_path")) != str(sequence.get("tracks_path")):
            state.error("track_identity", f"{sequence_id}: track path differs between sequence and track-set manifests")
        visibility_source = str(row.get("visibility_source"))
        if visibility_source not in ("geometry_raycast", "rendered_depth_and_target_mask"):
            state.error("track_identity", f"{sequence_id}: unknown visibility_source={visibility_source!r}")
        convention_text = json.dumps(row.get("coordinate_convention"), sort_keys=True)
        if "OpenCV" not in convention_text or "target-root-local" not in convention_text:
            state.error("track_identity", f"{sequence_id}: coordinate convention does not declare OpenCV and target-root-local coordinates")

        track_path = _safe_relative_path(root, row.get("tracks_path"), f"{sequence_id}.tracks_path", state)
        canonical_path = _safe_relative_path(
            root, row.get("canonical_points_path"), f"{sequence_id}.canonical_points_path", state
        )
        if track_path is None or canonical_path is None:
            continue
        track_paths[sequence_id] = track_path
        track = _load_npz(track_path, f"{sequence_id}.tracks_path", state)
        canonical = _load_canonical_points(canonical_path, canonical_cache, state)
        if track is None or canonical is None:
            continue
        missing = sorted(TRACK_REQUIRED_ARRAYS - set(track))
        if missing:
            state.error("track_identity", f"{sequence_id}: tracks NPZ missing arrays {missing}")
            continue
        if visibility_source == "rendered_depth_and_target_mask":
            rendered_missing = sorted(RENDERED_TRACK_ARRAYS - set(track))
            if rendered_missing:
                state.error("track_projection_consistency", f"{sequence_id}: rendered tracks missing arrays {rendered_missing}")
        canonical_missing = sorted({"point_id", "xyz_object_local"} - set(canonical))
        if canonical_missing:
            state.error("track_identity", f"{sequence_id}: canonical NPZ missing arrays {canonical_missing}")
            continue

        point_ids = np.asarray(track["point_id"])
        local = np.asarray(track["xyz_object_local"], dtype=np.float64)
        canonical_ids = np.asarray(canonical["point_id"])
        canonical_local = np.asarray(canonical["xyz_object_local"], dtype=np.float64)
        if point_ids.ndim != 1:
            state.error("track_identity", f"{sequence_id}: point_id must have shape (P,), got {point_ids.shape}")
            continue
        point_count = len(point_ids)
        point_counts.append(point_count)
        if len(np.unique(point_ids)) != point_count:
            state.error("track_identity", f"{sequence_id}: point_id values are not unique")
        if local.shape != (point_count, 3):
            state.error("track_identity", f"{sequence_id}: xyz_object_local shape {local.shape}, expected {(point_count, 3)}")
            continue
        if not np.isfinite(local).all():
            state.error("track_identity", f"{sequence_id}: xyz_object_local contains NaN/Inf")
        if int(row.get("point_count", -1)) != point_count:
            state.error("track_identity", f"{sequence_id}: track-set point_count differs from NPZ")
        if int(sequence.get("track_point_count", point_count)) != point_count:
            state.error("track_identity", f"{sequence_id}: sequence track_point_count differs from NPZ")
        expected_point_count = int(protocol["track_point_count"])
        if expected_point_count and point_count != expected_point_count:
            state.error("track_identity", f"{sequence_id}: point count {point_count}, config expects {expected_point_count}")
        if not np.array_equal(point_ids, canonical_ids) or canonical_local.shape != local.shape or not np.allclose(
            local, canonical_local, atol=0.0, rtol=0.0
        ):
            state.error("track_identity", f"{sequence_id}: track canonical IDs/coordinates differ from canonical points file")

        group_id = str(sequence.get("group_id"))
        if group_id not in group_canonical:
            group_canonical[group_id] = (point_ids.copy(), local.copy(), sequence_id)
        else:
            ref_ids, ref_local, ref_sequence = group_canonical[group_id]
            if not np.array_equal(point_ids, ref_ids) or not np.allclose(local, ref_local, atol=0.0, rtol=0.0):
                state.error(
                    "track_identity",
                    f"{group_id}: {sequence_id} canonical tracks differ from {ref_sequence}",
                )

        frame_count = int(sequence.get("num_frames", 0))
        expected_shapes = {
            "xyz_world": (frame_count, point_count, 3),
            "xyz_camera": (frame_count, point_count, 3),
            "projected_uv": (frame_count, point_count, 2),
            "depth_camera_z_m": (frame_count, point_count),
            "range_to_camera_m": (frame_count, point_count),
            "in_front_of_camera": (frame_count, point_count),
            "in_image": (frame_count, point_count),
            "visible": (frame_count, point_count),
        }
        for name in RENDERED_TRACK_ARRAYS & set(track):
            expected_shapes[name] = (frame_count, point_count)
        shape_failure = False
        for name, expected_shape in expected_shapes.items():
            if np.asarray(track[name]).shape != expected_shape:
                state.error("track_projection_consistency", f"{sequence_id}: {name} shape {np.asarray(track[name]).shape}, expected {expected_shape}")
                shape_failure = True
        if shape_failure:
            continue
        rows = list(frames_by_sequence.get(sequence_id, []))
        if len(rows) != frame_count:
            continue
        _validate_one_track_geometry(
            sequence_id,
            track,
            local,
            rows,
            visibility_source,
            root,
            protocol,
            geometry_only,
            geometry_atol,
            projection_atol,
            maxima,
            visible_fractions,
            state,
        )

    state.metrics["track_identity"] = {
        "track_set_count_checked": len(track_paths),
        "group_count_checked": len(group_canonical),
        "point_count_distribution": dict(Counter(point_counts)),
        "canonical_identity_rule": "point_id and xyz_object_local must be bitwise-identical within each group",
    }
    state.metrics["track_projection_consistency"] = {
        **maxima,
        "geometry_atol_m": geometry_atol,
        "projection_atol_px": projection_atol,
        "visible_fraction": _summary_stats(visible_fractions),
    }
    return track_paths


def _validate_one_track_geometry(
    sequence_id: str,
    track: Mapping[str, np.ndarray],
    local: np.ndarray,
    frames: Sequence[Mapping[str, Any]],
    visibility_source: str,
    root: Path,
    protocol: Mapping[str, Any],
    geometry_only: bool,
    geometry_atol: float,
    projection_atol: float,
    maxima: MutableMapping[str, float],
    visible_fractions: List[float],
    state: ValidationState,
) -> None:
    for frame_index, frame in enumerate(frames):
        frame_id = str(frame.get("frame_id"))
        object_to_world = _matrix4(frame.get("object_to_world"), f"{frame_id}.object_to_world", "track_projection_consistency", state)
        world_to_camera = _matrix4(frame.get("world_to_camera"), f"{frame_id}.world_to_camera", "track_projection_consistency", state)
        k = _array(frame.get("K", frame.get("intrinsics_K")), (3, 3), f"{frame_id}.K", "track_projection_consistency", state)
        if object_to_world is None or world_to_camera is None or k is None:
            continue
        saved_world = np.asarray(track["xyz_world"][frame_index], dtype=np.float64)
        saved_camera = np.asarray(track["xyz_camera"][frame_index], dtype=np.float64)
        saved_uv = np.asarray(track["projected_uv"][frame_index], dtype=np.float64)
        expected_world = _transform_points(object_to_world, local)
        expected_camera = _transform_points(world_to_camera, expected_world)
        expected_uv = np.full_like(saved_uv, np.nan, dtype=np.float64)
        expected_in_front = expected_camera[:, 2] > 0.0
        projected = (k @ expected_camera[expected_in_front].T).T
        expected_uv[expected_in_front] = projected[:, :2] / projected[:, 2:3]

        world_error = _finite_max_abs(saved_world, expected_world)
        camera_error = _finite_max_abs(saved_camera, expected_camera)
        projection_error = _finite_max_abs(saved_uv[expected_in_front], expected_uv[expected_in_front])
        maxima["xyz_world_error_m"] = max(maxima["xyz_world_error_m"], world_error)
        maxima["xyz_camera_error_m"] = max(maxima["xyz_camera_error_m"], camera_error)
        maxima["projection_error_px"] = max(maxima["projection_error_px"], projection_error)
        if world_error > geometry_atol:
            state.error("track_projection_consistency", f"{frame_id}: xyz_world reconstruction error {world_error:.3g} m")
        if camera_error > geometry_atol:
            state.error("track_projection_consistency", f"{frame_id}: xyz_camera reconstruction error {camera_error:.3g} m")
        if projection_error > projection_atol:
            state.error("track_projection_consistency", f"{frame_id}: projected_uv reconstruction error {projection_error:.3g} px")
        if np.any(~np.isfinite(saved_uv[expected_in_front])):
            state.error("track_projection_consistency", f"{frame_id}: in-front points have non-finite projected_uv")
        if np.any(np.isfinite(saved_uv[~expected_in_front])):
            state.error("track_projection_consistency", f"{frame_id}: behind-camera points should have non-finite projected_uv")

        saved_depth = np.asarray(track["depth_camera_z_m"][frame_index], dtype=np.float64)
        saved_range = np.asarray(track["range_to_camera_m"][frame_index], dtype=np.float64)
        depth_error = _finite_max_abs(saved_depth, expected_camera[:, 2])
        range_error = _finite_max_abs(saved_range, np.linalg.norm(expected_camera, axis=1))
        maxima["depth_z_error_m"] = max(maxima["depth_z_error_m"], depth_error)
        maxima["range_error_m"] = max(maxima["range_error_m"], range_error)
        if depth_error > geometry_atol:
            state.error("track_projection_consistency", f"{frame_id}: axial depth mismatch {depth_error:.3g} m")
        if range_error > geometry_atol:
            state.error("track_projection_consistency", f"{frame_id}: ray range mismatch {range_error:.3g} m")

        in_front = np.asarray(track["in_front_of_camera"][frame_index], dtype=bool)
        in_image = np.asarray(track["in_image"][frame_index], dtype=bool)
        visible = np.asarray(track["visible"][frame_index], dtype=bool)
        width, height = [int(value) for value in frame.get("image_size", [0, 0])]
        expected_in_image = (
            expected_in_front
            & np.isfinite(expected_uv).all(axis=1)
            & (expected_uv[:, 0] >= 0.0)
            & (expected_uv[:, 0] < width)
            & (expected_uv[:, 1] >= 0.0)
            & (expected_uv[:, 1] < height)
        )
        if not np.array_equal(in_front, expected_in_front):
            state.error("track_projection_consistency", f"{frame_id}: in_front_of_camera flags are inconsistent")
        if not np.array_equal(in_image, expected_in_image):
            state.error("track_projection_consistency", f"{frame_id}: in_image flags are inconsistent")
        if np.any(visible & ~in_image):
            state.error("track_projection_consistency", f"{frame_id}: visible points include out-of-image points")
        visible_fractions.append(float(visible.mean()) if len(visible) else 0.0)
        observation = frame.get("track_observation")
        if isinstance(observation, Mapping):
            expected_counts = {
                "point_count": len(visible),
                "in_front_count": int(in_front.sum()),
                "in_image_count": int(in_image.sum()),
                "visible_count": int(visible.sum()),
            }
            for key, value in expected_counts.items():
                if int(observation.get(key, -1)) != value:
                    state.error("track_projection_consistency", f"{frame_id}: track_observation.{key} disagrees with NPZ")

        if visibility_source == "rendered_depth_and_target_mask":
            if not RENDERED_TRACK_ARRAYS.issubset(track):
                continue
            saved_buffer = np.asarray(track["depth_buffer_range_m"][frame_index], dtype=np.float64)
            saved_buffer_error = np.asarray(track["depth_buffer_error_m"][frame_index], dtype=np.float64)
            internal_error = _finite_max_abs(saved_buffer_error, np.abs(saved_buffer - saved_range))
            if internal_error > geometry_atol:
                state.error("track_projection_consistency", f"{frame_id}: saved depth-buffer errors are inconsistent")
            tolerance = np.maximum(
                float(protocol["depth_abs_tol_m"]),
                float(protocol["depth_rel_tol"]) * saved_range,
            )
            expected_visible_internal = in_image & np.isfinite(saved_buffer) & (saved_buffer_error <= tolerance)
            if not np.array_equal(visible, expected_visible_internal):
                state.error("track_projection_consistency", f"{frame_id}: rendered visible flags disagree with saved depth evidence")
            if not geometry_only:
                depth_path = _safe_relative_path(root, _frame_path_value(frame, "depth_npy"), f"{frame_id}.depth_npy", state)
                mask_path = _safe_relative_path(root, _frame_path_value(frame, "target_mask"), f"{frame_id}.target_mask", state)
                if depth_path is not None and mask_path is not None and depth_path.is_file() and mask_path.is_file():
                    try:
                        depth = np.load(depth_path, allow_pickle=False)
                        mask = _load_mask(mask_path)
                        recomputed_buffer, recomputed_error, recomputed_visible = _sample_rendered_visibility(
                            expected_uv,
                            in_image,
                            saved_range,
                            depth,
                            mask,
                            float(protocol["depth_abs_tol_m"]),
                            float(protocol["depth_rel_tol"]),
                            int(protocol["visibility_pixel_radius"]),
                        )
                        sample_error = _finite_max_abs(saved_buffer, recomputed_buffer)
                        maxima["rendered_depth_sample_error_m"] = max(
                            maxima["rendered_depth_sample_error_m"], sample_error
                        )
                        if sample_error > geometry_atol or _finite_max_abs(saved_buffer_error, recomputed_error) > geometry_atol:
                            state.error("track_projection_consistency", f"{frame_id}: track depth samples disagree with depth.npy/mask")
                        if not np.array_equal(visible, recomputed_visible):
                            state.error("track_projection_consistency", f"{frame_id}: visible flags disagree with depth.npy/target_mask")
                    except Exception as exc:
                        state.error("track_projection_consistency", f"{frame_id}: cannot independently resample rendered visibility: {exc}")


def _load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _sample_rendered_visibility(
    uv: np.ndarray,
    in_image: np.ndarray,
    ranges: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    abs_tolerance_m: float,
    rel_tolerance: float,
    pixel_radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError(f"depth/mask shape mismatch: {depth.shape} versus {mask.shape}")
    sampled = np.full(len(uv), np.nan, dtype=np.float64)
    errors = np.full(len(uv), np.nan, dtype=np.float64)
    visible = np.zeros(len(uv), dtype=bool)
    height, width = depth.shape
    radius = int(pixel_radius)
    for index in np.flatnonzero(in_image):
        center_x = int(round(float(uv[index, 0])))
        center_y = int(round(float(uv[index, 1])))
        best_error = float("inf")
        best_depth = float("nan")
        for y in range(max(0, center_y - radius), min(height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(width, center_x + radius + 1)):
                value = float(depth[y, x])
                if not mask[y, x] or not math.isfinite(value) or value <= 0.0:
                    continue
                error = abs(value - float(ranges[index]))
                if error < best_error:
                    best_error = error
                    best_depth = value
        if math.isfinite(best_depth):
            sampled[index] = best_depth
            errors[index] = best_error
            tolerance = max(float(abs_tolerance_m), float(rel_tolerance) * float(ranges[index]))
            visible[index] = best_error <= tolerance
    return sampled, errors, visible


def _load_track_for_pair(
    sequence_id: str, track_paths: Mapping[str, Path], state: ValidationState, check: str
) -> Optional[Dict[str, np.ndarray]]:
    path = track_paths.get(sequence_id)
    if path is None:
        state.error(check, f"{sequence_id}: no valid track path")
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = ("point_id", "projected_uv", "visible")
            if any(key not in archive.files for key in required):
                state.error(check, f"{sequence_id}: pair-comparison arrays are missing")
                return None
            return {key: np.asarray(archive[key]).copy() for key in required}
    except Exception as exc:
        state.error(check, f"{sequence_id}: cannot load tracks for comparison: {exc}")
        return None


def _uv_difference_stats(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    min_common: int,
    label: str,
    check: str,
    state: ValidationState,
) -> Dict[str, Any]:
    if not np.array_equal(left["point_id"], right["point_id"]):
        state.error(check, f"{label}: point_id arrays differ")
        return {**_summary_stats([]), "min_common_visible_per_frame": 0, "per_frame_common_visible": []}
    left_uv = np.asarray(left["projected_uv"], dtype=np.float64)
    right_uv = np.asarray(right["projected_uv"], dtype=np.float64)
    left_visible = np.asarray(left["visible"], dtype=bool)
    right_visible = np.asarray(right["visible"], dtype=bool)
    if left_uv.shape != right_uv.shape or left_visible.shape != right_visible.shape:
        state.error(check, f"{label}: trajectory array shapes differ")
        return {**_summary_stats([]), "min_common_visible_per_frame": 0, "per_frame_common_visible": []}
    values: List[float] = []
    per_frame_counts: List[int] = []
    for index in range(left_uv.shape[0]):
        common = left_visible[index] & right_visible[index]
        common &= np.isfinite(left_uv[index]).all(axis=1) & np.isfinite(right_uv[index]).all(axis=1)
        count = int(common.sum())
        per_frame_counts.append(count)
        if count < min_common:
            state.error(check, f"{label}/frame_{index:03d}: common-visible count {count} < required {min_common}")
        if count:
            values.extend(np.linalg.norm(left_uv[index, common] - right_uv[index, common], axis=1).tolist())
    result = _summary_stats(values)
    result["min_common_visible_per_frame"] = min(per_frame_counts) if per_frame_counts else 0
    result["per_frame_common_visible"] = per_frame_counts
    return result


def validate_counterfactual_pairs(
    sequences_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    track_paths: Mapping[str, Path],
    protocol: Mapping[str, Any],
    state: ValidationState,
) -> None:
    matched_results: List[Dict[str, Any]] = []
    compensated_results: List[Dict[str, Any]] = []
    min_common = int(protocol["min_common_visible_points"])
    for group_id, sequences in sorted(sequences_by_group.items()):
        by_condition = {
            (int(row.get("ego_level", 999)), int(row.get("object_level", 999))): row
            for row in sequences
        }
        for condition in ((1, 0), (0, -1)):
            if condition not in by_condition:
                state.error("matched_relative_pair", f"{group_id}: missing condition {condition}")
        if (1, 0) in by_condition and (0, -1) in by_condition:
            left_id = str(by_condition[(1, 0)]["sequence_id"])
            right_id = str(by_condition[(0, -1)]["sequence_id"])
            left = _load_track_for_pair(left_id, track_paths, state, "matched_relative_pair")
            right = _load_track_for_pair(right_id, track_paths, state, "matched_relative_pair")
            if left is not None and right is not None:
                stats = _uv_difference_stats(left, right, min_common, group_id, "matched_relative_pair", state)
                result = {
                    "group_id": group_id,
                    "left_condition": [1, 0],
                    "right_condition": [0, -1],
                    **stats,
                }
                matched_results.append(result)
                if stats["mean"] is not None and float(stats["mean"]) > float(protocol["matched_mean_tol_px"]):
                    state.error("matched_relative_pair", f"{group_id}: matched-pair mean UV error {stats['mean']:.6g}px exceeds tolerance")
                if stats["max"] is not None and float(stats["max"]) > float(protocol["matched_max_tol_px"]):
                    state.error("matched_relative_pair", f"{group_id}: matched-pair max UV error {stats['max']:.6g}px exceeds tolerance")

        if (0, 0) not in by_condition or (1, 1) not in by_condition:
            state.error("compensated_target_trajectory", f"{group_id}: missing static or (+1,+1) condition")
        else:
            static_id = str(by_condition[(0, 0)]["sequence_id"])
            compensated_id = str(by_condition[(1, 1)]["sequence_id"])
            static_track = _load_track_for_pair(static_id, track_paths, state, "compensated_target_trajectory")
            compensated_track = _load_track_for_pair(compensated_id, track_paths, state, "compensated_target_trajectory")
            if static_track is not None and compensated_track is not None:
                stats = _uv_difference_stats(
                    static_track,
                    compensated_track,
                    min_common,
                    group_id,
                    "compensated_target_trajectory",
                    state,
                )
                result = {
                    "group_id": group_id,
                    "static_condition": [0, 0],
                    "compensated_condition": [1, 1],
                    **stats,
                }
                compensated_results.append(result)
                if stats["mean"] is not None and float(stats["mean"]) > float(protocol["compensated_mean_tol_px"]):
                    state.error("compensated_target_trajectory", f"{group_id}: compensated mean UV error exceeds tolerance")
                if stats["max"] is not None and float(stats["max"]) > float(protocol["compensated_max_tol_px"]):
                    state.error("compensated_target_trajectory", f"{group_id}: compensated max UV error exceeds tolerance")
    state.metrics["matched_relative_pair"] = {
        "definition": "(+1,0) versus (0,-1), common-visible same point_id only",
        "comparisons": matched_results,
        "aggregate": _summary_stats(
            [float(row["mean"]) for row in matched_results if row["mean"] is not None]
        ),
    }
    state.metrics["compensated_target_trajectory"] = {
        "definition": "(0,0) versus (+1,+1), common-visible same point_id only",
        "comparisons": compensated_results,
        "aggregate": _summary_stats(
            [float(row["mean"]) for row in compensated_results if row["mean"] is not None]
        ),
    }


def validate_projected_displacement_scale(
    sequences_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    track_paths: Mapping[str, Path],
    protocol: Mapping[str, Any],
    state: ValidationState,
) -> None:
    by_relative: Dict[int, List[float]] = defaultdict(list)
    representatives: List[Dict[str, Any]] = []
    min_common = int(protocol["min_common_visible_points"])
    for group_id, sequences in sorted(sequences_by_group.items()):
        for sequence in sequences:
            relative_level = int(sequence.get("relative_level", 999))
            if abs(relative_level) != 1:
                continue
            sequence_id = str(sequence.get("sequence_id"))
            track = _load_track_for_pair(sequence_id, track_paths, state, "projected_displacement_scale")
            if track is None:
                continue
            uv = np.asarray(track["projected_uv"], dtype=np.float64)
            visible = np.asarray(track["visible"], dtype=bool)
            if uv.ndim != 3 or uv.shape[0] < 2:
                state.error("projected_displacement_scale", f"{sequence_id}: invalid UV trajectory shape")
                continue
            common = visible[0] & visible[-1]
            common &= np.isfinite(uv[0]).all(axis=1) & np.isfinite(uv[-1]).all(axis=1)
            count = int(common.sum())
            if count < min_common:
                state.error("projected_displacement_scale", f"{sequence_id}: endpoint common-visible count {count} < {min_common}")
            distances = np.linalg.norm(uv[-1, common] - uv[0, common], axis=1) if count else np.asarray([])
            by_relative[relative_level].extend(distances.tolist())
            ego = int(sequence.get("ego_level", 999))
            obj = int(sequence.get("object_level", 999))
            if (ego, obj) in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                representatives.append(
                    {
                        "group_id": group_id,
                        "sequence_id": sequence_id,
                        "condition": [ego, obj],
                        **_summary_stats(distances.tolist()),
                    }
                )
    aggregate_values = [value for values in by_relative.values() for value in values]
    aggregate = _summary_stats(aggregate_values)
    recommendation: Dict[str, Any] = {
        "desired_approximate_range_px": [4.0, 10.0],
        "automatic_per_scene_rescaling_performed": False,
    }
    if aggregate["median"] is not None:
        median = float(aggregate["median"])
        if median < 4.0 or median > 10.0:
            state.warning(
                "projected_displacement_scale",
                f"global +/-1 median displacement {median:.3f}px is outside the desired approximate 4-10px range",
            )
            if median > 0.0:
                recommendation["suggested_uniform_delta_m_for_7px_median"] = float(
                    float(protocol["delta_m"]) * 7.0 / median
                )
    else:
        state.error("projected_displacement_scale", "no common-visible +/-1 relative-motion points")
    state.metrics["projected_displacement_scale"] = {
        "definition": "frame 0 to final frame Euclidean UV displacement for all abs(relative_level)==1 sequences",
        "by_relative_level": {str(key): _summary_stats(values) for key, values in sorted(by_relative.items())},
        "aggregate": aggregate,
        "pure_ego_or_object_representatives": representatives,
        "delta_m": float(protocol["delta_m"]),
        "recommendation": recommendation,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_provenance(
    config: Mapping[str, Any], manifests: Mapping[str, Any], state: ValidationState
) -> None:
    """Check both the saved source guard and the currently addressable source files."""
    provenance = manifests["provenance"]
    rebuild = manifests["source_rebuild_audit"]
    marker = manifests["marker"]

    if marker.get("dataset_name") != "ego_object_x_factorial_v1":
        state.error("source_provenance", "dataset marker has the wrong dataset_name")
    for key in ("mode", "dry_run"):
        if marker.get(key) != config.get("mode_used" if key == "mode" else key):
            state.error("source_provenance", f"dataset marker {key} differs from config_used")
    if not bool(provenance.get("source_is_read_only", False)):
        state.error("source_provenance", "provenance does not declare the source read-only")
    if bool(provenance.get("source_camera_intrinsics_reused", True)):
        state.error("source_provenance", "historical corrupted camera intrinsics were marked as reused")
    if provenance.get("source_camera_bank_semantics") != "spatial_multiview_initial_pose_candidates_only":
        state.error("source_provenance", "source camera bank is not explicitly limited to initial-pose selection")
    if provenance.get("temporal_camera_semantics") != "one_fixed_base_rotation_plus_pure_world_x_translation":
        state.error("source_provenance", "temporal pure-translation semantics are missing")

    before = provenance.get("source_files_sha256")
    after = provenance.get("source_files_sha256_after_generation")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        state.error("source_provenance", "source before/after SHA256 maps are missing")
        before = {}
        after = {}
    elif dict(before) != dict(after):
        state.error("source_provenance", "source hashes changed during generation")
    if not bool(provenance.get("source_integrity_verified_after_generation", False)):
        state.error("source_provenance", "source integrity was not verified after generation")

    audits = rebuild.get("scenes")
    if not isinstance(audits, list) or not audits:
        state.error("source_provenance", "source rebuild audit has no scenes")
        audits = []
    for audit in audits:
        scene_id = audit.get("scene_id", "unknown")
        if not bool(audit.get("match_ok", False)) or audit.get("errors"):
            state.error("source_provenance", f"{scene_id}: rebuilt scene does not exactly match source")

    source_root_raw = provenance.get("source_dataset_root")
    live_results: Dict[str, Any] = {}
    if isinstance(source_root_raw, str) and source_root_raw:
        source_root = Path(source_root_raw).expanduser()
        source_config_raw = config.get("source", {}).get("config") if isinstance(config.get("source"), Mapping) else None
        source_config_path = (
            (PACKAGE_DIR.parent / str(source_config_raw)).resolve()
            if source_config_raw
            else None
        )
        paths: Dict[str, Path] = {
            "config_used": source_root / "config_used.yaml",
            "scenes": source_root / "manifests" / "scenes.jsonl",
            "states": source_root / "manifests" / "states.jsonl",
            "cameras": source_root / "manifests" / "cameras.jsonl",
            "frames": source_root / "manifests" / "frames.jsonl",
            "splits": source_root / "manifests" / "splits.json",
        }
        if source_config_path is not None:
            paths["configured_source_config"] = source_config_path
        for key, expected_hash in before.items():
            path = paths.get(str(key))
            if path is None:
                state.error("source_provenance", f"cannot resolve source hash entry {key!r}")
                continue
            if not path.is_file():
                state.error("source_provenance", f"current source file is missing: {path}")
                live_results[str(key)] = {"path": str(path), "status": "missing"}
                continue
            actual_hash = _sha256_file(path)
            matches = actual_hash == str(expected_hash)
            live_results[str(key)] = {
                "path": str(path),
                "sha256": actual_hash,
                "matches_recorded_before_hash": matches,
            }
            if not matches:
                state.error("source_provenance", f"current source file changed since generation: {path}")
    else:
        state.error("source_provenance", "source_dataset_root is missing from provenance")
    state.metrics["source_provenance"] = {
        "saved_before_after_equal": bool(before) and dict(before) == dict(after),
        "rebuild_scene_count": len(audits),
        "live_source_hashes": live_results,
    }


def _render_required_keys(config: Mapping[str, Any]) -> List[str]:
    render = config.get("render", {})
    required = ["rgb", "depth_exr", "depth_npy", "target_mask", "frame_metadata"]
    for key, flag in (
        ("normal", "normal_enabled"),
        ("albedo", "albedo_enabled"),
        ("instance", "instance_enabled"),
        ("object_id", "instance_enabled"),
        ("semantic", "semantic_enabled"),
    ):
        if bool(render.get(flag, False)):
            required.append(key)
    return required


def _image_shape(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def validate_rendered_outputs(
    root: Path,
    config: Mapping[str, Any],
    manifests: Mapping[str, Any],
    frames_by_sequence: Mapping[str, Sequence[Mapping[str, Any]]],
    sequences_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    protocol: Mapping[str, Any],
    geometry_only: bool,
    state: ValidationState,
) -> None:
    rendered_checks = ("rendered_outputs", "frame0_rgb_identity", "compensated_background_change")
    if geometry_only:
        for check in rendered_checks:
            state.skip(check, "--geometry-only requested; rendered artifacts were intentionally not inspected")
        return
    if bool(config.get("dry_run", False)):
        for check in rendered_checks:
            state.error(check, "dry-run dataset has no rendered evidence; rerun validator with --geometry-only")
        return

    required_keys = _render_required_keys(config)
    raster_keys = set(required_keys) - {"depth_exr", "depth_npy", "frame_metadata"}
    frame_by_id = {str(frame.get("frame_id")): frame for frame in manifests["frames"]}
    checked_files = 0
    mask_ratios: List[float] = []

    for frame in manifests["frames"]:
        frame_id = str(frame.get("frame_id"))
        if not bool(frame.get("rendered", False)):
            state.error("rendered_outputs", f"{frame_id}: rendered flag is false")
        width, height = [int(value) for value in frame.get("image_size", [0, 0])]
        if [width, height] != list(protocol["resolution"]):
            state.error("rendered_outputs", f"{frame_id}: image_size differs from effective resolution")
        resolved: Dict[str, Path] = {}
        for key in required_keys:
            path = _safe_relative_path(root, _frame_path_value(frame, key), f"{frame_id}.{key}", state)
            if path is None:
                continue
            resolved[key] = path
            if not path.is_file() or path.stat().st_size <= 0:
                state.error("rendered_outputs", f"{frame_id}: missing or empty {key}: {path}")
                continue
            checked_files += 1
            if key in raster_keys:
                try:
                    if _image_shape(path) != (width, height):
                        state.error("rendered_outputs", f"{frame_id}: {key} has the wrong resolution")
                except Exception as exc:
                    state.error("rendered_outputs", f"{frame_id}: cannot read {key}: {exc}")

        depth_path = resolved.get("depth_npy")
        if depth_path is not None and depth_path.is_file():
            try:
                depth = np.load(depth_path, allow_pickle=False)
                if depth.shape != (height, width):
                    state.error("rendered_outputs", f"{frame_id}: depth.npy shape is {depth.shape}")
                if not np.isfinite(depth).all():
                    state.error("rendered_outputs", f"{frame_id}: depth.npy contains NaN/Inf")
                if np.any(depth <= 0.0):
                    state.error("rendered_outputs", f"{frame_id}: depth.npy contains non-positive values")
            except Exception as exc:
                state.error("rendered_outputs", f"{frame_id}: cannot read depth.npy: {exc}")

        mask_path = resolved.get("target_mask")
        if mask_path is not None and mask_path.is_file():
            try:
                mask = _load_mask(mask_path)
                pixel_count = int(mask.sum())
                ratio = float(mask.mean())
                mask_ratios.append(ratio)
                if pixel_count != int(frame.get("target_mask_pixel_count", -1)):
                    state.error("rendered_outputs", f"{frame_id}: target-mask pixel count differs from metadata")
                if abs(ratio - float(frame.get("target_mask_area_ratio", -1.0))) > 1.0 / max(1, width * height):
                    state.error("rendered_outputs", f"{frame_id}: target-mask ratio differs from metadata")
                if ratio < float(protocol["min_mask_area_ratio"]):
                    state.error("rendered_outputs", f"{frame_id}: target mask is too small ({ratio:.6f})")
                if bool(frame.get("target_truncated", False)):
                    state.error("rendered_outputs", f"{frame_id}: target mask is truncated")
                if float(frame.get("target_edge_touch_ratio", 1.0)) > float(protocol["max_edge_touch_ratio"]):
                    state.error("rendered_outputs", f"{frame_id}: target mask touches the image boundary")
            except Exception as exc:
                state.error("rendered_outputs", f"{frame_id}: cannot validate target mask: {exc}")

        instance_path = resolved.get("instance")
        object_id_path = resolved.get("object_id")
        if instance_path is not None and object_id_path is not None and instance_path.is_file() and object_id_path.is_file():
            if _sha256_file(instance_path) != _sha256_file(object_id_path):
                state.error("rendered_outputs", f"{frame_id}: instance.png and object_id.png differ")

        metadata_path = resolved.get("frame_metadata")
        if metadata_path is not None and metadata_path.is_file():
            try:
                saved = _read_json(metadata_path)
                if str(saved.get("frame_id")) != frame_id:
                    state.error("rendered_outputs", f"{frame_id}: frame_metadata.json has wrong frame_id")
                for key in ("sequence_id", "group_id", "frame_index", "rendered"):
                    if saved.get(key) != frame.get(key):
                        state.error("rendered_outputs", f"{frame_id}: frame_metadata.{key} differs from manifest")
            except Exception as exc:
                state.error("rendered_outputs", f"{frame_id}: cannot parse frame_metadata.json: {exc}")

    frame0_results: List[Dict[str, Any]] = []
    background_results: List[Dict[str, Any]] = []
    for group_id, sequences in sorted(sequences_by_group.items()):
        frame0_hashes: Dict[str, str] = {}
        by_condition = {
            (int(sequence.get("ego_level", 999)), int(sequence.get("object_level", 999))): sequence
            for sequence in sequences
        }
        for sequence in sequences:
            sequence_id = str(sequence.get("sequence_id"))
            rows = frames_by_sequence.get(sequence_id, [])
            if not rows:
                continue
            frame0 = rows[0]
            rgb_path = _safe_relative_path(root, _frame_path_value(frame0, "rgb"), f"{frame0.get('frame_id')}.rgb", state)
            if rgb_path is not None and rgb_path.is_file():
                frame0_hashes[sequence_id] = _sha256_file(rgb_path)
        unique_hashes = sorted(set(frame0_hashes.values()))
        frame0_results.append({
            "group_id": group_id,
            "sequence_count_checked": len(frame0_hashes),
            "unique_rgb_sha256_count": len(unique_hashes),
        })
        if len(frame0_hashes) != 25 or len(unique_hashes) != 1:
            state.error("frame0_rgb_identity", f"{group_id}: frame-0 RGB is not byte-identical across all 25 conditions")

        static = by_condition.get((0, 0))
        compensated = by_condition.get((1, 1))
        if static is None or compensated is None:
            state.error("compensated_background_change", f"{group_id}: missing static or (+1,+1) sequence")
            continue
        static_frames = frames_by_sequence.get(str(static.get("sequence_id")), [])
        compensated_frames = frames_by_sequence.get(str(compensated.get("sequence_id")), [])
        if not static_frames or not compensated_frames:
            state.error("compensated_background_change", f"{group_id}: missing final frames")
            continue
        left_frame = static_frames[-1]
        right_frame = compensated_frames[-1]
        try:
            left_rgb_path = _safe_relative_path(root, _frame_path_value(left_frame, "rgb"), f"{group_id}.static.rgb", state)
            right_rgb_path = _safe_relative_path(root, _frame_path_value(right_frame, "rgb"), f"{group_id}.compensated.rgb", state)
            left_mask_path = _safe_relative_path(root, _frame_path_value(left_frame, "target_mask"), f"{group_id}.static.mask", state)
            right_mask_path = _safe_relative_path(root, _frame_path_value(right_frame, "target_mask"), f"{group_id}.compensated.mask", state)
            if None in (left_rgb_path, right_rgb_path, left_mask_path, right_mask_path):
                continue
            left_rgb = _load_rgb(left_rgb_path)  # type: ignore[arg-type]
            right_rgb = _load_rgb(right_rgb_path)  # type: ignore[arg-type]
            target_union = _load_mask(left_mask_path) | _load_mask(right_mask_path)  # type: ignore[arg-type]
            background = ~target_union
            if not background.any():
                state.error("compensated_background_change", f"{group_id}: no background pixels outside target masks")
                continue
            channel_difference = np.abs(
                left_rgb.astype(np.int16) - right_rgb.astype(np.int16)
            ).astype(np.float64)
            per_pixel_max = np.max(channel_difference, axis=2)
            mean_abs = float(channel_difference[background].mean())
            changed_fraction = float(
                (per_pixel_max[background] > float(protocol["background_pixel_threshold"])).mean()
            )
            result = {
                "group_id": group_id,
                "background_pixel_count": int(background.sum()),
                "mean_channel_absolute_difference_8bit": mean_abs,
                "changed_fraction": changed_fraction,
                "pixel_change_threshold_8bit": float(protocol["background_pixel_threshold"]),
            }
            background_results.append(result)
            if mean_abs < float(protocol["background_min_mad"]):
                state.error("compensated_background_change", f"{group_id}: compensated background mean change {mean_abs:.4f} is too small")
            if changed_fraction < float(protocol["background_min_changed_fraction"]):
                state.error("compensated_background_change", f"{group_id}: compensated background changed fraction {changed_fraction:.6f} is too small")
        except Exception as exc:
            state.error("compensated_background_change", f"{group_id}: cannot compare compensated background: {exc}")

    state.metrics["rendered_outputs"] = {
        "frame_count_checked": len(manifests["frames"]),
        "required_file_count_checked": checked_files,
        "required_keys": required_keys,
        "target_mask_area_ratio": _summary_stats(mask_ratios),
    }
    state.metrics["frame0_rgb_identity"] = {"groups": frame0_results}
    state.metrics["compensated_background_change"] = {
        "definition": "final-frame static versus (+1,+1), excluding union of target masks",
        "groups": background_results,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def validate_dataset(root: Path, geometry_only: bool) -> Dict[str, Any]:
    state = ValidationState()
    root = root.expanduser().resolve()
    config, dataset_summary, manifests = load_dataset(root, state)
    validate_schema(manifests, state)
    protocol = _config_protocol(config, state)
    sequence_by_id, sequences_by_group = validate_counts_and_protocol(
        config, dataset_summary, manifests, protocol, state
    )
    validate_scene_splits(manifests, sequence_by_id, state)
    validate_source_provenance(config, manifests, state)
    validate_all_paths(root, manifests, state)
    validate_selection_report(manifests, protocol, state)
    frames_by_sequence = validate_frame_geometry(
        manifests, sequence_by_id, protocol, state
    )
    track_paths = validate_tracks(
        root,
        manifests,
        sequence_by_id,
        frames_by_sequence,
        protocol,
        geometry_only,
        state,
    )
    validate_counterfactual_pairs(sequences_by_group, track_paths, protocol, state)
    validate_projected_displacement_scale(sequences_by_group, track_paths, protocol, state)
    validate_rendered_outputs(
        root,
        config,
        manifests,
        frames_by_sequence,
        sequences_by_group,
        protocol,
        geometry_only,
        state,
    )
    payload = _json_safe(
        {
            "validator": "validate_ego_object_x_factorial_dataset.py",
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(root),
            "mode": config.get("mode_used", dataset_summary.get("mode")),
            "geometry_only": bool(geometry_only),
            "ok": not state.errors,
            "checks": state.check_statuses(),
            "metrics": state.metrics,
            "error_count": int(sum(state.error_counts.values())),
            "warning_count": int(sum(state.warning_counts.values())),
            "errors": state.errors,
            "warnings": state.warnings,
            "message_limit": state.max_messages,
        }
    )
    if root.is_dir():
        output_name = "validation_geometry_summary.json" if geometry_only else "validation_summary.json"
        (root / output_name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return payload


def main() -> None:
    args = parse_args()
    result = validate_dataset(args.dataset_root, args.geometry_only)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
