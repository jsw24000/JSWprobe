#!/usr/bin/env python3
"""Validate an object-translation dataset output tree."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_scene_blender.object_translation.label_utils import (
    assert_valid_matrix,
    corners_inside_room,
    image_size,
    json_has_no_nan,
    mask_stats_from_png,
    validate_no_collision,
)
from memory_scene_blender.object_translation.manifest_utils import (
    load_config,
    read_json,
    read_jsonl,
    transform_delta,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate object-translation dataset outputs.")
    parser.add_argument("--dataset-root", type=Path, default=PACKAGE_DIR / "outputs" / "object_translation_v1")
    parser.add_argument("--config", type=Path, default=None, help="Optional source config; defaults to config_used.yaml in the dataset root.")
    parser.add_argument("--strict-visibility", action="store_true", help="Treat per-state visible-camera shortages as hard errors.")
    return parser.parse_args()


def fail(errors: List[str], message: str) -> None:
    errors.append(message)


def load_manifests(root: Path) -> Dict[str, Any]:
    manifests = root / "manifests"
    return {
        "scenes": read_jsonl(manifests / "scenes.jsonl"),
        "cameras": read_jsonl(manifests / "cameras.jsonl"),
        "states": read_jsonl(manifests / "states.jsonl"),
        "frames": read_jsonl(manifests / "frames.jsonl"),
        "pairs": read_jsonl(manifests / "translation_pairs.jsonl"),
        "triplets": read_jsonl(manifests / "composition_triplets.jsonl"),
        "splits": read_json(manifests / "splits.json") if (manifests / "splits.json").exists() else {},
    }


def check_required_files(root: Path, errors: List[str]) -> None:
    required = [
        root / "config_used.yaml",
        root / "dataset_summary.json",
        root / "README.md",
        root / "manifests" / "scenes.jsonl",
        root / "manifests" / "cameras.jsonl",
        root / "manifests" / "states.jsonl",
        root / "manifests" / "frames.jsonl",
        root / "manifests" / "translation_pairs.jsonl",
        root / "manifests" / "composition_triplets.jsonl",
        root / "manifests" / "splits.json",
    ]
    for path in required:
        if not path.exists():
            fail(errors, f"Missing required file: {path}")


def state_maps(states: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(row["scene_id"]), str(row["state_id"])): row for row in states}


def camera_maps(cameras: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(row["scene_id"]), str(row["camera_id"])): row for row in cameras}


def check_cameras_shared(
    frames: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    errors: List[str],
) -> None:
    cameras_by_scene_state: Dict[Tuple[str, str], set] = defaultdict(set)
    for frame in frames:
        cameras_by_scene_state[(str(frame["scene_id"]), str(frame["state_id"]))].add(str(frame["camera_id"]))
    by_scene: Dict[str, List[set]] = defaultdict(list)
    for state in states:
        key = (str(state["scene_id"]), str(state["state_id"]))
        by_scene[str(state["scene_id"])].append(cameras_by_scene_state.get(key, set()))
    for scene_id, camera_sets in by_scene.items():
        if not camera_sets:
            fail(errors, f"{scene_id}: no states found for camera-sharing check")
            continue
        first = camera_sets[0]
        for idx, cameras in enumerate(camera_sets[1:], start=1):
            if cameras != first:
                fail(errors, f"{scene_id}: state camera set mismatch at state index {idx}")


def check_static_and_target_consistency(root: Path, states: Sequence[Mapping[str, Any]], errors: List[str]) -> None:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for state in states:
        grouped[str(state["scene_id"])].append(state)
    for scene_id, rows in grouped.items():
        scene_meta_path = root / scene_id / "scene_metadata.json"
        if not scene_meta_path.exists():
            fail(errors, f"{scene_id}: missing scene_metadata.json")
            continue
        scene_meta = read_json(scene_meta_path)
        signature = scene_meta.get("non_target_layout_signature")
        first_asset_sig = rows[0].get("target_asset_signature")
        first_rot = np.asarray(rows[0]["object_transform_world"], dtype=np.float64)[:3, :3]
        first_scale = rows[0].get("object_local_scale")
        for row in rows:
            if row.get("non_target_layout_signature") != signature:
                fail(errors, f"{scene_id}/{row['state_id']}: static layout signature changed")
            if row.get("target_asset_signature") != first_asset_sig:
                fail(errors, f"{scene_id}/{row['state_id']}: target asset signature changed")
            if row.get("object_local_scale") != first_scale:
                fail(errors, f"{scene_id}/{row['state_id']}: target local scale changed")
            rot = np.asarray(row["object_transform_world"], dtype=np.float64)[:3, :3]
            if not np.allclose(rot, first_rot, atol=1e-7):
                fail(errors, f"{scene_id}/{row['state_id']}: target rotation changed")
            if not row.get("orientation_fixed", False):
                fail(errors, f"{scene_id}/{row['state_id']}: orientation_fixed is not true")


def check_frame_files(frames: Sequence[Mapping[str, Any]], errors: List[str]) -> Dict[str, Any]:
    mask_ratios: List[float] = []
    visibility_failures = 0
    for frame in frames:
        frame_id = str(frame["frame_id"])
        rgb = Path(frame["rgb"])
        normal = Path(frame["normal"])
        mask = Path(frame["target_mask"])
        instance = Path(frame["instance"])
        semantic = Path(frame["semantic"])
        object_id = Path(frame["object_id"])
        depth_exr = Path(frame["depth"])
        depth_npy = Path(frame.get("depth_npy", depth_exr.with_suffix(".npy")))
        for path in [rgb, normal, mask, instance, semantic, object_id, depth_exr, depth_npy, Path(frame["frame_metadata"])]:
            if not path.exists():
                fail(errors, f"{frame_id}: missing output {path}")
                continue

        try:
            rgb_size = image_size(rgb)
            for label, path in [("normal", normal), ("mask", mask), ("instance", instance), ("semantic", semantic), ("object_id", object_id)]:
                if path.exists() and image_size(path) != rgb_size:
                    fail(errors, f"{frame_id}: {label} size does not match RGB")
            if depth_npy.exists():
                depth = np.load(depth_npy)
                if depth.shape[:2] != (rgb_size[1], rgb_size[0]):
                    fail(errors, f"{frame_id}: depth.npy shape {depth.shape} does not match RGB size {rgb_size}")
                finite = np.isfinite(depth)
                if not finite.any():
                    fail(errors, f"{frame_id}: depth.npy has no finite values")
            if mask.exists():
                stats = mask_stats_from_png(mask)
                if stats["mask_pixel_count"] <= 0:
                    fail(errors, f"{frame_id}: target mask is empty")
                saved_count = int(frame.get("target_mask_pixel_count", -1))
                if saved_count >= 0 and abs(saved_count - int(stats["mask_pixel_count"])) > max(4, 0.002 * saved_count):
                    fail(errors, f"{frame_id}: saved mask count {saved_count} disagrees with PNG count {stats['mask_pixel_count']}")
                mask_ratios.append(float(stats["mask_area_ratio"]))
        except Exception as exc:
            fail(errors, f"{frame_id}: image/depth check failed: {exc}")
        if not frame.get("visibility_ok", False):
            visibility_failures += 1
    return {"mask_ratios": mask_ratios, "visibility_failures": visibility_failures}


def check_grounding_collisions(
    root: Path,
    states: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    errors: List[str],
) -> None:
    clearance = float(config.get("positions", {}).get("static_clearance_m", 0.0))
    wall_margin = float(config.get("positions", {}).get("wall_margin_m", 0.0))
    scenes = {row["scene_id"]: read_json(root / str(row["scene_id"]) / "scene_metadata.json") for row in read_jsonl(root / "manifests" / "scenes.jsonl")}
    for state in states:
        scene_id = str(state["scene_id"])
        if abs(float(state.get("ground_contact_z", 0.0))) > 1e-4:
            fail(errors, f"{scene_id}/{state['state_id']}: target is not grounded, z={state.get('ground_contact_z')}")
        scene_meta = scenes.get(scene_id)
        if not scene_meta:
            continue
        if not corners_inside_room(state["bbox_corners_world"], scene_meta["room_dimensions"], wall_margin):
            fail(errors, f"{scene_id}/{state['state_id']}: target bbox violates room margins")
        if not validate_no_collision(state["bbox_corners_world"], scene_meta["static_objects"], clearance):
            fail(errors, f"{scene_id}/{state['state_id']}: target collides with static object footprint")


def check_pair_deltas(
    pairs: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    cameras: Sequence[Mapping[str, Any]],
    errors: List[str],
) -> Dict[str, Any]:
    states_by_key = state_maps(states)
    cameras_by_key = camera_maps(cameras)
    pair_types = Counter()
    pair_tags = Counter()
    displacement_groups = Counter()
    distances: List[float] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        scene_id = str(pair["scene_id"])
        state_a = states_by_key.get((scene_id, str(pair["state_a"])))
        state_b = states_by_key.get((scene_id, str(pair["state_b"])))
        camera = cameras_by_key.get((scene_id, str(pair["camera_id"])))
        ref_camera = cameras_by_key.get((scene_id, "camera_000"))
        if state_a is None or state_b is None or camera is None or ref_camera is None:
            fail(errors, f"{pair_id}: missing referenced state or camera")
            continue
        delta = np.asarray(state_b["object_center_world"], dtype=np.float64) - np.asarray(state_a["object_center_world"], dtype=np.float64)
        if not np.allclose(delta, np.asarray(pair["delta_world"], dtype=np.float64), atol=1e-6):
            fail(errors, f"{pair_id}: delta_world does not match state centers")
        expected_ref = transform_delta(ref_camera["opencv_world_to_camera"], delta)
        expected_current = transform_delta(camera["opencv_world_to_camera"], delta)
        if not np.allclose(expected_ref, np.asarray(pair["delta_ref_camera"], dtype=np.float64), atol=1e-6):
            fail(errors, f"{pair_id}: delta_ref_camera is inconsistent")
        if not np.allclose(expected_current, np.asarray(pair["delta_current_camera"], dtype=np.float64), atol=1e-6):
            fail(errors, f"{pair_id}: delta_current_camera is inconsistent")
        pair_types[str(pair["pair_type"])] += 1
        for tag in pair.get("pair_tags", []):
            pair_tags[str(tag)] += 1
        displacement_groups[str(pair["same_displacement_group"])] += 1
        distances.append(float(pair["distance"]))
    return {
        "pair_type_counts": dict(pair_types),
        "pair_tag_counts": dict(pair_tags),
        "displacement_group_counts": dict(displacement_groups),
        "distance_values": distances,
    }


def check_triplets(
    triplets: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    errors: List[str],
) -> Dict[str, Any]:
    states_by_key = state_maps(states)
    triplet_types = Counter()
    for triplet in triplets:
        triplet_id = str(triplet["triplet_id"])
        d01 = np.asarray(triplet["delta_01_ref_camera"], dtype=np.float64)
        d12 = np.asarray(triplet["delta_12_ref_camera"], dtype=np.float64)
        d02 = np.asarray(triplet["delta_02_ref_camera"], dtype=np.float64)
        if not np.allclose(d01 + d12, d02, atol=1e-6):
            fail(errors, f"{triplet_id}: delta_02 != delta_01 + delta_12")
        if not bool(triplet.get("is_additively_consistent", False)):
            fail(errors, f"{triplet_id}: is_additively_consistent is false")
        scene_id = str(triplet["scene_id"])
        for key_name in ["state_0", "state_1", "state_2"]:
            if (scene_id, str(triplet[key_name])) not in states_by_key:
                fail(errors, f"{triplet_id}: missing {key_name}")
        triplet_types[str(triplet.get("triplet_type", "unknown"))] += 1
    return {"triplet_type_counts": dict(triplet_types)}


def check_matrices_and_camera_convention(cameras: Sequence[Mapping[str, Any]], states: Sequence[Mapping[str, Any]], errors: List[str]) -> None:
    for camera in cameras:
        camera_id = f"{camera['scene_id']}/{camera['camera_id']}"
        for key in [
            "blender_camera_to_world",
            "blender_world_to_camera",
            "opencv_camera_to_world",
            "opencv_world_to_camera",
            "blender_to_opencv_camera",
        ]:
            try:
                assert_valid_matrix(camera[key], f"{camera_id}:{key}")
            except AssertionError as exc:
                fail(errors, str(exc))
        c2w_bl = np.asarray(camera["blender_camera_to_world"], dtype=np.float64)
        w2c_cv = np.asarray(camera["opencv_world_to_camera"], dtype=np.float64)
        origin_cv = w2c_cv @ np.array([*c2w_bl[:3, 3], 1.0])
        if not np.allclose(origin_cv[:3], np.zeros(3), atol=1e-6):
            fail(errors, f"{camera_id}: camera position does not map to OpenCV origin")
        forward_world = c2w_bl @ np.array([0.0, 0.0, -1.0, 1.0])
        forward_cv = w2c_cv @ forward_world
        if not np.allclose(forward_cv[:3], np.array([0.0, 0.0, 1.0]), atol=1e-6):
            fail(errors, f"{camera_id}: Blender -Z forward does not become OpenCV +Z")

    for state in states:
        try:
            assert_valid_matrix(state["object_transform_world"], f"{state['scene_id']}/{state['state_id']}:object_transform_world")
        except AssertionError as exc:
            fail(errors, str(exc))


def check_no_nan(payloads: Sequence[Mapping[str, Any]], label: str, errors: List[str]) -> None:
    for idx, payload in enumerate(payloads):
        if not json_has_no_nan(payload):
            fail(errors, f"{label}[{idx}] contains NaN or Inf")


def check_visibility_by_state(
    frames: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    strict: bool,
    errors: List[str],
) -> Dict[str, Any]:
    settings = config.get("effective_mode_settings", {})
    min_visible = int(settings.get("min_visible_cameras", config.get("visibility", {}).get("min_visible_cameras", 1)))
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for frame in frames:
        if frame.get("visibility_ok", False):
            counts[(str(frame["scene_id"]), str(frame["state_id"]))] += 1
    failures = {f"{scene}/{state}": count for (scene, state), count in counts.items() if count < min_visible}
    if strict:
        for key, count in failures.items():
            fail(errors, f"{key}: visible cameras {count} < required {min_visible}")
    return {"min_visible_cameras": min_visible, "visible_camera_counts": {f"{k[0]}/{k[1]}": v for k, v in counts.items()}, "failures": failures}


def validate(root: Path, config_path: Path | None, strict_visibility: bool) -> Dict[str, Any]:
    errors: List[str] = []
    check_required_files(root, errors)
    if errors:
        return {"ok": False, "errors": errors}

    config = load_config(config_path or (root / "config_used.yaml"))
    manifests = load_manifests(root)
    scenes = manifests["scenes"]
    cameras = manifests["cameras"]
    states = manifests["states"]
    frames = manifests["frames"]
    pairs = manifests["pairs"]
    triplets = manifests["triplets"]

    for label, rows in [("scenes", scenes), ("cameras", cameras), ("states", states), ("frames", frames), ("pairs", pairs), ("triplets", triplets)]:
        check_no_nan(rows, label, errors)

    check_cameras_shared(frames, states, errors)
    check_static_and_target_consistency(root, states, errors)
    frame_stats = check_frame_files(frames, errors)
    check_grounding_collisions(root, states, config, errors)
    pair_stats = check_pair_deltas(pairs, states, cameras, errors)
    triplet_stats = check_triplets(triplets, states, errors)
    check_matrices_and_camera_convention(cameras, states, errors)
    visibility_stats = check_visibility_by_state(frames, config, strict_visibility, errors)

    distances = pair_stats["distance_values"]
    distance_summary = {
        "count": len(distances),
        "min": float(np.min(distances)) if distances else None,
        "max": float(np.max(distances)) if distances else None,
        "mean": float(np.mean(distances)) if distances else None,
    }
    mask_ratios = frame_stats["mask_ratios"]
    mask_summary = {
        "count": len(mask_ratios),
        "min": float(np.min(mask_ratios)) if mask_ratios else None,
        "max": float(np.max(mask_ratios)) if mask_ratios else None,
        "mean": float(np.mean(mask_ratios)) if mask_ratios else None,
    }
    summary = {
        "ok": not errors,
        "scene_count": len(scenes),
        "state_count": len(states),
        "camera_count": len(cameras),
        "frame_count": len(frames),
        "pair_count": len(pairs),
        "triplet_count": len(triplets),
        "translation_distance_distribution": distance_summary,
        "displacement_group_sample_counts": pair_stats["displacement_group_counts"],
        "pair_type_counts": pair_stats["pair_type_counts"],
        "pair_tag_counts": pair_stats["pair_tag_counts"],
        "triplet_type_counts": triplet_stats["triplet_type_counts"],
        "mask_area_ratio_distribution": mask_summary,
        "visibility_failure_count": int(frame_stats["visibility_failures"]),
        "visibility_by_state": visibility_stats,
        "errors": errors,
    }
    write_json(root / "validation_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    summary = validate(root, args.config, args.strict_visibility)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
