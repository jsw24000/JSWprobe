#!/usr/bin/env python3
"""Generate controlled temporal world-X ego/object factorial sequences.

Run inside Blender.  The source ``object_translation_v1`` tree is read-only and
is used only for scene/anchor/camera provenance.  A spatial camera-bank pose is
chosen once per group; each temporal pose then changes only the camera's world-X
translation.  In particular, this script never calls ``look_at`` after motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import bpy  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - Blender-only entrypoint
    raise SystemExit(
        "This generator must run inside Blender. Example:\n"
        "  blender --background --python "
        "memory_scene_blender/scripts/generate_ego_object_x_factorial_dataset.py -- "
        "--config memory_scene_blender/configs/ego_object_x_factorial_v1.yaml --mode smoke --dry-run"
    ) from exc

from memory_scene_blender.ego_object_factorial.geometry import (
    anchor_root_xy,
    choose_anchors_and_cameras,
    configure_factorial_camera,
    geometric_track_observation,
    observation_summary,
    sample_target_surface_points,
)
from memory_scene_blender.ego_object_factorial.protocol import (
    build_matched_relative_groups,
    camera_from_source_extrinsics,
    group_id,
    linear_alphas,
    motion_conditions,
    project_opencv,
    sequence_id,
    translated_camera_payload,
)
from memory_scene_blender.ego_object_factorial.rendering import (
    refine_observation_from_render,
    render_or_reuse_physical_frame,
    save_npz,
    stack_track_observations,
)
from memory_scene_blender.object_translation.assets import asset_world_bbox
from memory_scene_blender.object_translation.manifest_utils import (
    ensure_dir,
    load_config,
    mode_settings,
    np_to_list,
    read_json,
    read_jsonl,
    vector_to_list,
    write_json,
    write_jsonl,
)
from memory_scene_blender.object_translation.scene_builder import (
    build_scene_bundle,
    set_target_position,
)


DATASET_MARKER = ".ego_object_x_factorial_dataset"


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_DIR / "configs" / "ego_object_x_factorial_v1.yaml",
    )
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--output-root", type=Path, default=None, help="Exact mode-specific output directory.")
    parser.add_argument("--scene-id", type=str, default=None, help="Restrict to one source scene, e.g. scene_003.")
    parser.add_argument(
        "--anchor-id",
        type=str,
        default=None,
        help="Restrict to one source anchor/state, e.g. anchor_005 or state_005.",
    )
    parser.add_argument("--base-camera-id", type=str, default=None, help="Force one camera-bank pose.")
    parser.add_argument("--dry-run", action="store_true", help="Write geometry/manifests/tracks but render no images.")
    parser.add_argument("--resume", action="store_true", help="Reuse complete frames only after config identity checks.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate only a marked factorial output directory.")
    parser.add_argument(
        "--allow-full",
        action="store_true",
        help="Required safety acknowledgement before a non-dry-run full render.",
    )
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(
    config: Mapping[str, Any],
    mode: str,
    dry_run: bool,
    scene_id: Optional[str],
    anchor_id: Optional[str],
    base_camera_id: Optional[str],
) -> str:
    payload = {
        "config": config,
        "mode": str(mode),
        "dry_run": bool(dry_run),
        "scene_id": scene_id,
        "anchor_id": anchor_id,
        "base_camera_id": base_camera_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_settings(config: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    settings = dict(config.get("render", {}))
    settings.update(config["modes"][mode])
    return settings


def output_root_for_run(
    config: Mapping[str, Any], settings: Mapping[str, Any], args: argparse.Namespace
) -> Path:
    if args.output_root is not None:
        return resolve_project_path(args.output_root)
    base = resolve_project_path(Path(str(config["output_root"])))
    if args.dry_run:
        return base / "_dry_runs" / str(args.mode)
    return base / str(settings["output_subdir"])


def prepare_output_root(
    output_root: Path,
    source_root: Path,
    expected_digest: str,
    args: argparse.Namespace,
) -> None:
    output_resolved = output_root.resolve()
    source_resolved = source_root.resolve()
    if (
        output_resolved == source_resolved
        or source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise ValueError(
            f"Refusing an output that is the source dataset, inside it, or one of its ancestors: {output_root}"
        )
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    if output_root.exists() and any(output_root.iterdir()):
        marker = output_root / DATASET_MARKER
        if args.overwrite:
            if not marker.exists():
                raise RuntimeError(
                    f"Refusing to delete unmarked directory {output_root}; choose a fresh output or add the marker manually "
                    "only after auditing its contents."
                )
            shutil.rmtree(output_root)
        elif args.resume:
            if not marker.exists():
                raise RuntimeError(f"Cannot resume unmarked directory: {output_root}")
            previous = read_json(marker)
            if previous.get("config_digest") != expected_digest:
                raise RuntimeError(
                    "Resume config/protocol mismatch. Use a fresh output root; historical provenance will not be overwritten."
                )
        else:
            raise FileExistsError(
                f"Output root is not empty: {output_root}. Use --resume for an identical interrupted run, "
                "--overwrite only for this marked new dataset, or choose another --output-root."
            )

    ensure_dir(output_root)
    write_json(
        output_root / DATASET_MARKER,
        {
            "dataset_name": "ego_object_x_factorial_v1",
            "config_digest": expected_digest,
            "mode": args.mode,
            "dry_run": bool(args.dry_run),
            "run_scope": {
                "scene_id": args.scene_id,
                "anchor_id": args.anchor_id,
                "base_camera_id": args.base_camera_id,
            },
        },
    )


def load_source(source_root: Path) -> Dict[str, Any]:
    required = [
        source_root / "config_used.yaml",
        source_root / "manifests" / "scenes.jsonl",
        source_root / "manifests" / "states.jsonl",
        source_root / "manifests" / "cameras.jsonl",
        source_root / "manifests" / "frames.jsonl",
        source_root / "manifests" / "splits.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Source object-translation dataset is incomplete: {missing}")
    return {
        "config": load_config(source_root / "config_used.yaml"),
        "scenes": read_jsonl(source_root / "manifests" / "scenes.jsonl"),
        "states": read_jsonl(source_root / "manifests" / "states.jsonl"),
        "cameras": read_jsonl(source_root / "manifests" / "cameras.jsonl"),
        "frames": read_jsonl(source_root / "manifests" / "frames.jsonl"),
        "splits": read_json(source_root / "manifests" / "splits.json"),
    }


def choose_scene_ids(
    settings: Mapping[str, Any], source_scenes: Sequence[Mapping[str, Any]], forced_scene: Optional[str]
) -> List[str]:
    available = [str(row["scene_id"]) for row in source_scenes]
    if forced_scene is not None:
        if forced_scene not in available:
            raise ValueError(f"Unknown source scene: {forced_scene}")
        return [forced_scene]
    if "scene_ids" in settings:
        chosen = [str(value) for value in settings["scene_ids"]]
    else:
        chosen = available[: int(settings["scene_count"])]
    unknown = sorted(set(chosen) - set(available))
    if unknown:
        raise ValueError(f"Mode selects unavailable source scenes: {unknown}")
    return chosen


def scene_index(scene_id: str) -> int:
    if not scene_id.startswith("scene_"):
        raise ValueError(f"Invalid scene id: {scene_id}")
    return int(scene_id.split("_", 1)[1])


def split_for_scene(scene_id: str, source_splits: Mapping[str, Any]) -> str:
    if scene_id in source_splits.get("train_scenes", []):
        return "train"
    if scene_id in source_splits.get("validation_scenes", []):
        return "val"
    if scene_id in source_splits.get("test_scenes", []):
        return "test"
    raise ValueError(f"Source split manifest does not assign {scene_id}")


def correct_source_cameras(
    source_cameras: Sequence[Mapping[str, Any]],
    source_camera_cfg: Mapping[str, Any],
    resolution: Sequence[int],
) -> List[Dict[str, Any]]:
    """Reuse source extrinsics but intentionally recompute correct image intrinsics.

    The historical generator shadowed its image-height variable with camera
    height, yielding invalid manifest fields such as height=1 and cy=0.625.
    Rendered images are 512x512, so those historical K matrices are never used.
    """
    width, image_height = [int(value) for value in resolution]
    corrected: List[Dict[str, Any]] = []
    for source in sorted(source_cameras, key=lambda row: str(row["camera_id"])):
        payload = camera_from_source_extrinsics(
            source, source_camera_cfg, [width, image_height]
        )
        corrected.append(payload)
    return corrected


def normalized_static_signature(static_metadata: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "logical_name": row["logical_name"],
            "object_id": row["object_id"],
            "category": row["category"],
            "root_matrix_world": row["root_matrix_world"],
            "signature": row["signature"],
        }
        for row in static_metadata
    ]


def compare_static_signatures(
    generated: Sequence[Mapping[str, Any]], source: Sequence[Mapping[str, Any]], atol: float = 1.0e-6
) -> None:
    if len(generated) != len(source):
        raise RuntimeError(f"Static object count drift: generated {len(generated)}, source {len(source)}")
    for generated_row, source_row in zip(generated, source):
        for key in ("logical_name", "object_id", "category", "signature"):
            if generated_row.get(key) != source_row.get(key):
                raise RuntimeError(f"Static scene drift for {source_row.get('logical_name')}: field {key}")
        if not np.allclose(
            np.asarray(generated_row["root_matrix_world"], dtype=np.float64),
            np.asarray(source_row["root_matrix_world"], dtype=np.float64),
            atol=atol,
            rtol=0.0,
        ):
            raise RuntimeError(f"Static transform drift for {source_row.get('logical_name')}")


def audit_rebuilt_scene(
    bundle: Any,
    source_scene: Mapping[str, Any],
    source_scene_metadata: Mapping[str, Any],
    source_states: Sequence[Mapping[str, Any]],
    source_cameras: Sequence[Mapping[str, Any]],
    require_exact: bool,
) -> Dict[str, Any]:
    errors: List[str] = []
    target_variant = int(bundle.target_asset.signature.get("variant", 0))
    if bundle.target_asset.category != source_scene["target_category"]:
        errors.append("target_category_mismatch")
    if target_variant != int(source_scene["target_variant"]):
        errors.append("target_variant_mismatch")
    if bundle.target_asset.signature != source_scene_metadata.get("target_asset_signature"):
        errors.append("target_asset_signature_mismatch")

    generated_positions = {str(row["state_id"]): np.asarray(row["xy"], dtype=np.float64) for row in bundle.positions}
    max_anchor_error = 0.0
    for state in source_states:
        state_id = str(state["state_id"])
        if state_id not in generated_positions:
            errors.append(f"missing_generated_{state_id}")
            continue
        error = float(np.max(np.abs(generated_positions[state_id] - anchor_root_xy(state))))
        max_anchor_error = max(max_anchor_error, error)
    if max_anchor_error > 1.0e-6:
        errors.append("anchor_coordinate_mismatch")

    generated_cameras = {str(row["camera_id"]): row for row in bundle.cameras}
    max_camera_error = 0.0
    for source_camera in source_cameras:
        camera_id = str(source_camera["camera_id"])
        generated_camera = generated_cameras.get(camera_id)
        if generated_camera is None:
            errors.append(f"missing_generated_{camera_id}")
            continue
        error = float(
            np.max(
                np.abs(
                    np.asarray(generated_camera["blender_camera_to_world"], dtype=np.float64)
                    - np.asarray(source_camera["blender_camera_to_world"], dtype=np.float64)
                )
            )
        )
        max_camera_error = max(max_camera_error, error)
    if max_camera_error > 1.0e-6:
        errors.append("camera_extrinsics_mismatch")

    try:
        compare_static_signatures(
            normalized_static_signature(bundle.static_metadata),
            source_scene_metadata["non_target_layout_signature"],
        )
    except RuntimeError as exc:
        errors.append(str(exc))

    report = {
        "scene_id": bundle.scene_id,
        "exact_match_required": bool(require_exact),
        "match_ok": not errors,
        "errors": errors,
        "max_anchor_root_xy_error_m": max_anchor_error,
        "max_camera_extrinsic_abs_error": max_camera_error,
        "source_target_object_id": source_scene["target_object_id"],
        "source_target_category": source_scene["target_category"],
        "source_target_variant": int(source_scene["target_variant"]),
    }
    if errors and require_exact:
        raise RuntimeError(f"Rebuilt {bundle.scene_id} does not match source artifacts: {errors}")
    return report


def target_centers(target_asset: Any, camera: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    bbox = asset_world_bbox(target_asset)
    array = np.asarray(bbox, dtype=np.float64)
    center_world = 0.5 * (array.min(axis=0) + array.max(axis=0))
    center_camera, _uv = project_opencv(
        center_world[None, :], camera["opencv_world_to_camera"], camera["intrinsics"]["K"]
    )
    return center_world, center_camera[0], [vector_to_list(corner) for corner in bbox]


def frame_paths(frame_dir: Path, root: Path) -> Dict[str, str]:
    names = {
        "rgb": "rgb.png",
        "depth": "depth.exr",
        "depth_exr": "depth.exr",
        "depth_npy": "depth.npy",
        "normal": "normal.png",
        "albedo": "albedo.png",
        "target_mask": "target_mask.png",
        "instance": "instance.png",
        "semantic": "semantic.png",
        "object_id": "object_id.png",
        "frame_metadata": "frame_metadata.json",
    }
    return {key: relative_path(frame_dir / filename, root) for key, filename in names.items()}


def source_provenance(
    source_root: Path,
    configured_source_config: Path,
    source: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_paths = {
        "config_used": source_root / "config_used.yaml",
        "configured_source_config": configured_source_config,
        "scenes": source_root / "manifests" / "scenes.jsonl",
        "states": source_root / "manifests" / "states.jsonl",
        "cameras": source_root / "manifests" / "cameras.jsonl",
        "frames": source_root / "manifests" / "frames.jsonl",
        "splits": source_root / "manifests" / "splits.json",
    }
    hashes = {key: sha256_file(path) for key, path in manifest_paths.items()}
    return {
        "source_dataset_root": str(source_root),
        "source_dataset_name": source["config"].get("dataset_name"),
        "source_files_sha256": hashes,
        "source_is_read_only": True,
        "source_camera_bank_semantics": "spatial_multiview_initial_pose_candidates_only",
        "temporal_camera_semantics": "one_fixed_base_rotation_plus_pure_world_x_translation",
        "source_camera_intrinsics_reused": False,
        "source_intrinsics_issue": (
            "Historical camera rows have image-height/K corruption from variable shadowing; only bank extrinsics, "
            "IDs, FOV, and clip planes are reused. K is recomputed for the actual output resolution."
        ),
    }


def write_config_used(
    root: Path,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
    args: argparse.Namespace,
    source_root: Path,
) -> None:
    payload = dict(config)
    payload.update(
        {
            "mode_used": args.mode,
            "dry_run": bool(args.dry_run),
            "effective_mode_settings": dict(settings),
            "resolved_output_root": str(root),
            "resolved_source_dataset_root": str(source_root),
            "camera_intrinsics_policy": "recomputed_from_fov_and_actual_resolution",
            "run_scope": {
                "scene_id": args.scene_id,
                "anchor_id": args.anchor_id,
                "base_camera_id": args.base_camera_id,
            },
        }
    )
    write_json(root / "config_used.yaml", payload)


def write_dataset_readme(root: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# Ego/Object X Factorial V1 ({summary['mode']})

This mode-specific dataset contains controlled temporal sequences.  Each group
fixes `(scene, coarse object anchor, base camera)` and crosses five camera
world-X amplitudes with five target-root world-X amplitudes.  Camera rotation,
intrinsics, target orientation, target scale, lighting, materials, and static
scene geometry remain fixed.  The old spatial camera bank supplies only the
initial pose; it is not treated as a video.

`relative_amplitude_m = object_amplitude_m - ego_amplitude_m` and
`alpha_t = t/(T-1)`.  OpenCV camera coordinates are +X right, +Y down, +Z
forward.  Blender Z-pass values used for visibility are camera-ray ranges,
which are stored separately from OpenCV axial Z in sparse tracks.

Counts: groups={summary['group_count']}, sequences={summary['sequence_count']},
frames={summary['frame_count']}, rendered_frames={summary['rendered_frame_count']}.

Manifests are under `manifests/`.  All paths inside manifests are relative to
this dataset root.  Canonical surface point IDs are stable across all 25
conditions in a group (and, in this version, across every group in a scene).
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def generate_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    if config.get("dataset_name") != "ego_object_x_factorial_v1":
        raise ValueError(f"Unexpected dataset_name in {args.config}: {config.get('dataset_name')}")
    if args.mode == "full" and not args.dry_run and not args.allow_full:
        raise RuntimeError(
            "A full render is outside the current smoke/pilot protocol. Pass --allow-full only in a future "
            "explicitly approved expansion."
        )
    settings = effective_settings(config, args.mode)
    source_root = resolve_project_path(Path(str(config["source"]["dataset_root"])))
    configured_source_config = resolve_project_path(Path(str(config["source"]["config"])))
    source = load_source(source_root)
    if source["config"].get("dataset_name") != config["source"]["expected_dataset_name"]:
        raise ValueError("Source dataset name does not match the explicit protocol configuration")

    digest = config_digest(
        config,
        args.mode,
        bool(args.dry_run),
        args.scene_id,
        args.anchor_id,
        args.base_camera_id,
    )
    output_root = output_root_for_run(config, settings, args)
    prepare_output_root(output_root, source_root, digest, args)
    ensure_dir(output_root / "manifests")
    ensure_dir(output_root / "previews")
    write_config_used(output_root, config, settings, args, source_root)
    provenance = source_provenance(source_root, configured_source_config, source)
    write_json(output_root / "provenance.json", provenance)

    scene_ids = choose_scene_ids(settings, source["scenes"], args.scene_id)
    levels = [int(value) for value in config["motion"]["levels"]]
    delta_m = float(config["motion"]["delta_m"])
    axis_world = np.asarray(config["motion"]["axis_world"], dtype=np.float64)
    if not np.allclose(axis_world, np.array([1.0, 0.0, 0.0]), atol=0.0, rtol=0.0):
        raise ValueError("Version 1 protocol permits only exact world-X motion [1, 0, 0]")
    conditions = motion_conditions(levels, delta_m)
    if len(conditions) != 25:
        raise ValueError("Version 1 requires exactly a 5x5=25 condition grid")
    alphas = linear_alphas(int(settings["num_frames"]))
    width, image_height = [int(value) for value in settings["resolution"]]

    source_scene_by_id = {str(row["scene_id"]): row for row in source["scenes"]}
    source_states_by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    source_cameras_by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in source["states"]:
        source_states_by_scene[str(row["scene_id"])].append(row)
    for row in source["cameras"]:
        source_cameras_by_scene[str(row["scene_id"])].append(row)
    source_frame_lookup = {
        (str(row["scene_id"]), str(row["state_id"]), str(row["camera_id"])): row
        for row in source["frames"]
    }

    scene_rows: List[Dict[str, Any]] = []
    anchor_rows: List[Dict[str, Any]] = []
    base_camera_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    sequence_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    selection_scene_reports: List[Dict[str, Any]] = []
    rebuild_audits: List[Dict[str, Any]] = []

    source_mode = str(config["source"].get("mode", "full"))
    source_build_settings = mode_settings(source["config"], source_mode)
    source_build_settings.update(
        {
            "resolution": list(settings["resolution"]),
            "samples": int(settings["samples"]),
        }
    )

    for scene_id_value in scene_ids:
        print(f"\n=== Building controlled source replica for {scene_id_value} ===")
        source_scene = source_scene_by_id[scene_id_value]
        source_scene_meta = read_json(source_root / scene_id_value / "scene_metadata.json")
        source_states = source_states_by_scene[scene_id_value]
        raw_source_cameras = source_cameras_by_scene[scene_id_value]
        bundle = build_scene_bundle(scene_index(scene_id_value), source["config"], source_build_settings)
        rebuild_audit = audit_rebuilt_scene(
            bundle,
            source_scene,
            source_scene_meta,
            source_states,
            raw_source_cameras,
            bool(config["source"].get("require_exact_scene_match", True)),
        )
        rebuild_audits.append(rebuild_audit)

        corrected_cameras = correct_source_cameras(
            raw_source_cameras,
            source["config"]["camera"],
            settings["resolution"],
        )
        canonical_seed = int(config["seed"]) + scene_index(scene_id_value) * 104729 + int(
            config["tracks"]["sampling_seed_offset"]
        )
        set_target_position(bundle.target_asset, [0.0, 0.0])
        canonical = sample_target_surface_points(
            bundle.target_asset,
            int(config["tracks"]["num_surface_points"]),
            canonical_seed,
        )
        scene_dir = ensure_dir(output_root / scene_id_value)
        canonical_path = scene_dir / "canonical_surface_points.npz"
        save_npz(
            canonical_path,
            canonical,
            {
                "scene_id": scene_id_value,
                "seed": canonical_seed,
                "sampling": "triangle-area-weighted target-root-local surface sampling",
                "shared_across_all_scene_groups": True,
            },
        )

        anchors_needed = 1 if args.anchor_id is not None else int(settings["anchors_per_scene"])
        cameras_needed = 1 if args.base_camera_id is not None else int(settings["base_cameras_per_anchor"])
        selected_groups, selection_report = choose_anchors_and_cameras(
            bundle.target_asset,
            canonical["xyz_object_local"],
            scene_id_value,
            source_states,
            corrected_cameras,
            source_frame_lookup,
            bundle.layout["dimensions"],
            bundle.static_metadata,
            levels,
            delta_m,
            axis_world,
            settings["resolution"],
            config["selection"],
            anchors_needed,
            cameras_needed,
            forced_anchor_id=args.anchor_id,
            forced_camera_id=args.base_camera_id,
        )
        selection_scene_reports.append(selection_report)
        if not selected_groups:
            write_json(
                output_root / "selection_report.json",
                {
                    "selection_schema_version": 1,
                    "delta_m": delta_m,
                    "motion_axis_world": vector_to_list(axis_world),
                    "selection_config": config["selection"],
                    "scene_reports": selection_scene_reports,
                    "selected_groups": [],
                },
            )
            raise RuntimeError(str(selection_report["error"]))

        split = split_for_scene(scene_id_value, source["splits"])
        target_object_id = str(source_scene["target_object_id"])
        target_category = str(source_scene["target_category"])
        target_variant = int(source_scene["target_variant"])
        scene_output_metadata = {
            "scene_id": scene_id_value,
            "split": split,
            "scene_seed": int(source_scene["scene_seed"]),
            "layout_id": source_scene["layout_id"],
            "room_dimensions": bundle.layout["dimensions"],
            "target_object_id": target_object_id,
            "target_object_numeric_id": int(source_scene["target_object_numeric_id"]),
            "target_category": target_category,
            "target_variant": target_variant,
            "target_asset_signature": bundle.target_asset.signature,
            "static_objects": bundle.static_metadata,
            "non_target_layout_signature": normalized_static_signature(bundle.static_metadata),
            "source_scene_metadata": relative_path(source_root / scene_id_value / "scene_metadata.json", source_root),
            "canonical_points_path": relative_path(canonical_path, output_root),
            "canonical_point_count": int(len(canonical["point_id"])),
            "canonical_sampling_seed": canonical_seed,
            "selected_group_count": len(selected_groups),
            "rebuild_audit": rebuild_audit,
        }
        write_json(scene_dir / "scene_metadata.json", scene_output_metadata)
        scene_rows.append(
            {
                **scene_output_metadata,
                "scene_metadata_path": relative_path(scene_dir / "scene_metadata.json", output_root),
            }
        )

        states_by_id = {str(row["state_id"]): row for row in source_states}
        cameras_by_id = {str(row["camera_id"]): row for row in corrected_cameras}
        anchors_written: set[Tuple[str, str]] = set()

        for selected in selected_groups:
            anchor_id = str(selected["anchor_id"])
            source_state_id = str(selected["source_state_id"])
            base_camera_id = str(selected["base_camera_id"])
            source_state = states_by_id[source_state_id]
            base_camera = cameras_by_id[base_camera_id]
            base_xy = anchor_root_xy(source_state)
            current_group_id = group_id(scene_id_value, anchor_id, base_camera_id)
            group_dir = ensure_dir(scene_dir / anchor_id / base_camera_id)
            group_metadata_path = group_dir / "group_metadata.json"

            anchor_key = (scene_id_value, anchor_id)
            if anchor_key not in anchors_written:
                anchor_rows.append(
                    {
                        "scene_id": scene_id_value,
                        "object_anchor_id": anchor_id,
                        "anchor_id": anchor_id,
                        "source_state_id": source_state_id,
                        "source_position_index": list(source_state["position_index"]),
                        "anchor_root_xy": vector_to_list(base_xy),
                        "anchor_object_to_world": source_state["object_transform_world"],
                        "anchor_world_center": source_state["object_center_world"],
                        "split": split,
                        "anchor_sweep": selected["anchor_sweep"],
                    }
                )
                anchors_written.add(anchor_key)

            base_camera_rows.append(
                {
                    "group_id": current_group_id,
                    "scene_id": scene_id_value,
                    "object_anchor_id": anchor_id,
                    "base_camera_id": base_camera_id,
                    "split": split,
                    "intrinsics": base_camera["intrinsics"],
                    "blender_camera_to_world": base_camera["blender_camera_to_world"],
                    "blender_world_to_camera": base_camera["blender_world_to_camera"],
                    "opencv_camera_to_world": base_camera["opencv_camera_to_world"],
                    "opencv_world_to_camera": base_camera["opencv_world_to_camera"],
                    "position": base_camera["position"],
                    "rotation_matrix": base_camera["rotation_matrix"],
                    "fov_degrees": base_camera["fov_degrees"],
                    "clip_start": base_camera["clip_start"],
                    "clip_end": base_camera["clip_end"],
                    "source_intrinsics_reused": False,
                    "camera_selection": selected["camera_selection"],
                }
            )
            group_metadata = {
                "group_id": current_group_id,
                "scene_id": scene_id_value,
                "object_anchor_id": anchor_id,
                "source_state_id": source_state_id,
                "base_camera_id": base_camera_id,
                "split": split,
                "delta_m": delta_m,
                "motion_axis_world": vector_to_list(axis_world),
                "levels": levels,
                "num_frames": int(settings["num_frames"]),
                "condition_count": len(conditions),
                "canonical_points_path": relative_path(canonical_path, output_root),
                "anchor_root_xy": vector_to_list(base_xy),
                "base_camera": base_camera_rows[-1],
                "anchor_sweep": selected["anchor_sweep"],
                "camera_selection": selected["camera_selection"],
                "camera_rotation_policy": "constant; no temporal look_at recomputation",
            }
            write_json(group_metadata_path, group_metadata)
            group_rows.append(
                {
                    **group_metadata,
                    "group_metadata_path": relative_path(group_metadata_path, output_root),
                }
            )

            physical_render_cache: Dict[Tuple[float, float], Path] = {}
            for condition in conditions:
                ego_level = int(condition["ego_level"])
                object_level = int(condition["object_level"])
                current_sequence_id = sequence_id(
                    scene_id_value, anchor_id, base_camera_id, ego_level, object_level
                )
                sequence_dir = ensure_dir(group_dir / "sequences" / current_sequence_id)
                tracks_path = sequence_dir / "tracks.npz"
                sequence_metadata_path = sequence_dir / "sequence_metadata.json"
                sequence_row: Dict[str, Any] = {
                    "sequence_id": current_sequence_id,
                    "group_id": current_group_id,
                    "scene_id": scene_id_value,
                    "object_anchor_id": anchor_id,
                    "anchor_id": anchor_id,
                    "source_state_id": source_state_id,
                    "base_camera_id": base_camera_id,
                    "ego_level": ego_level,
                    "object_level": object_level,
                    "ego_amplitude_m": float(condition["ego_amplitude_m"]),
                    "object_amplitude_m": float(condition["object_amplitude_m"]),
                    "relative_level": int(condition["relative_level"]),
                    "relative_amplitude_m": float(condition["relative_amplitude_m"]),
                    "num_frames": int(settings["num_frames"]),
                    "delta_m": delta_m,
                    "motion_axis_world": vector_to_list(axis_world),
                    "temporal_profile": "alpha_t=t/(T-1)",
                    "split": split,
                    "target_object_id": target_object_id,
                    "target_object_numeric_id": int(source_scene["target_object_numeric_id"]),
                    "target_category": target_category,
                    "target_variant": target_variant,
                    "target_asset_signature": bundle.target_asset.signature,
                    "is_compensated_motion": bool(condition["is_compensated_motion"]),
                    "is_static": bool(condition["is_static"]),
                    "sequence_dir": relative_path(sequence_dir, output_root),
                    "sequence_metadata_path": relative_path(sequence_metadata_path, output_root),
                    "tracks_path": relative_path(tracks_path, output_root),
                    "canonical_points_path": relative_path(canonical_path, output_root),
                    "rendered": not args.dry_run,
                }
                observations: List[Dict[str, np.ndarray]] = []
                sequence_frame_ids: List[str] = []
                for frame_index, alpha_t in enumerate(alphas):
                    actual_ego_dx = float(condition["ego_amplitude_m"] * alpha_t)
                    actual_object_dx = float(condition["object_amplitude_m"] * alpha_t)
                    actual_relative_dx = float(actual_object_dx - actual_ego_dx)
                    target_xy = base_xy + axis_world[:2] * actual_object_dx
                    set_target_position(bundle.target_asset, target_xy)
                    camera = translated_camera_payload(base_camera, actual_ego_dx, axis_world)
                    configure_factorial_camera(camera)
                    object_to_world = np.asarray(bundle.target_asset.root.matrix_world, dtype=np.float64)
                    observation = geometric_track_observation(
                        canonical["xyz_object_local"],
                        object_to_world,
                        camera,
                        settings["resolution"],
                        bundle.target_asset.parts,
                        edge_margin_px=0,
                        ray_tolerance_m=float(config["selection"]["ray_depth_tolerance_m"]),
                    )
                    frame_id = f"{current_sequence_id}__frame_{frame_index:03d}"
                    frame_dir = ensure_dir(sequence_dir / "frames" / f"frame_{frame_index:03d}")

                    if args.dry_run:
                        render_result: Dict[str, Any] = {
                            "render_source": "dry_run_geometry_only",
                            "cache_method": None,
                            "mask_stats": {
                                "mask_pixel_count": 0,
                                "mask_area_ratio": 0.0,
                                "bbox": None,
                                "truncated": False,
                                "edge_touch_ratio": 0.0,
                                "image_size": [width, image_height],
                            },
                            "visible_fraction": float(observation["visible"].mean()),
                            "depth_range_m": None,
                        }
                    else:
                        physical_key = (round(actual_ego_dx, 12), round(actual_object_dx, 12))
                        cached_dir = physical_render_cache.get(physical_key)
                        render_result = render_or_reuse_physical_frame(
                            frame_dir,
                            bundle.target_asset,
                            camera,
                            int(settings["samples"]),
                            settings["resolution"],
                            bool(args.resume),
                            cached_dir,
                        )
                        observation = refine_observation_from_render(
                            observation, frame_dir, config["tracks"]
                        )
                        physical_render_cache.setdefault(physical_key, frame_dir)

                    center_world, center_camera, bbox_corners = target_centers(bundle.target_asset, camera)
                    paths = frame_paths(frame_dir, output_root)
                    frame_row: Dict[str, Any] = {
                        "frame_id": frame_id,
                        "sequence_id": current_sequence_id,
                        "group_id": current_group_id,
                        "scene_id": scene_id_value,
                        "object_anchor_id": anchor_id,
                        "base_camera_id": base_camera_id,
                        "ego_level": ego_level,
                        "object_level": object_level,
                        "relative_level": int(condition["relative_level"]),
                        "frame_index": int(frame_index),
                        "alpha_t": float(alpha_t),
                        "actual_ego_dx_m": actual_ego_dx,
                        "actual_object_dx_m": actual_object_dx,
                        "actual_relative_dx_m": actual_relative_dx,
                        "camera_to_world": camera["opencv_camera_to_world"],
                        "world_to_camera": camera["opencv_world_to_camera"],
                        "opencv_camera_to_world": camera["opencv_camera_to_world"],
                        "opencv_world_to_camera": camera["opencv_world_to_camera"],
                        "blender_camera_to_world": camera["blender_camera_to_world"],
                        "blender_world_to_camera": camera["blender_world_to_camera"],
                        "camera_rotation": camera["rotation_matrix"],
                        "camera_world_position": camera["position"],
                        "intrinsics_K": camera["intrinsics"]["K"],
                        "K": camera["intrinsics"]["K"],
                        "intrinsics": camera["intrinsics"],
                        "object_to_world": np_to_list(object_to_world),
                        "target_object_to_world": np_to_list(object_to_world),
                        "target_world_center": vector_to_list(center_world),
                        "target_camera_coordinate_center": vector_to_list(center_camera),
                        "target_center_world": vector_to_list(center_world),
                        "target_center_camera": vector_to_list(center_camera),
                        "target_bbox_corners_world": bbox_corners,
                        "motion_axis_world": vector_to_list(axis_world),
                        "image_size": [width, image_height],
                        "coordinate_convention": {
                            "world": "Blender world, metres, +Z up",
                            "camera": "OpenCV, +X right, +Y down, +Z forward",
                            "pixel": "top-left origin, u right, v down",
                        },
                        "render_convention": {
                            "camera_type": "perspective",
                            "sensor_fit": "HORIZONTAL",
                            "pixel_aspect": [1.0, 1.0],
                            "principal_point_shift": [0.0, 0.0],
                            "depth_exr_and_npy": "Blender Z-pass camera-ray range in metres",
                            "instance_and_object_id": "identical color-coded PNG plus object_id_palette.json",
                            "semantic": "color-coded PNG plus semantic_palette.json",
                            "normal": "world normal mapped by n*0.5+0.5 into color-managed 8-bit PNG",
                            "albedo": "first material diffuse color rendered as emission",
                        },
                        "frame_dir": relative_path(frame_dir, output_root),
                        "frame_paths": paths,
                        **paths,
                        "rendered": not args.dry_run,
                        "render_source": render_result["render_source"],
                        "cache_method": render_result.get("cache_method"),
                        "target_mask_pixel_count": int(render_result["mask_stats"]["mask_pixel_count"]),
                        "target_mask_area_ratio": float(render_result["mask_stats"]["mask_area_ratio"]),
                        "target_bbox_2d": render_result["mask_stats"].get("bbox"),
                        "target_truncated": bool(render_result["mask_stats"].get("truncated", False)),
                        "target_edge_touch_ratio": float(
                            render_result["mask_stats"].get("edge_touch_ratio", 0.0)
                        ),
                        "render_estimated_visible_fraction": float(render_result["visible_fraction"]),
                        "depth_range_m": render_result.get("depth_range_m"),
                        "track_observation": observation_summary(observation),
                        "look_at_recomputed": False,
                    }
                    write_json(frame_dir / "frame_metadata.json", frame_row)
                    frame_rows.append(frame_row)
                    sequence_frame_ids.append(frame_id)
                    observations.append(observation)

                tracks_payload = stack_track_observations(canonical, observations)
                track_metadata = {
                    "track_schema_version": 1,
                    "sequence_id": current_sequence_id,
                    "group_id": current_group_id,
                    "point_identity": "stable target-root-local physical surface samples",
                    "depth_camera_z_m": "OpenCV axial +Z depth",
                    "range_to_camera_m": "Euclidean camera-ray range",
                    "depth_buffer_range_m": "Blender Z-pass ray range when rendered",
                    "visibility": (
                        "rendered target mask plus Z-pass range consistency"
                        if not args.dry_run
                        else "exact-point Blender ray cast (geometry-only preflight)"
                    ),
                }
                save_npz(tracks_path, tracks_payload, track_metadata)
                sequence_row["frame_ids"] = sequence_frame_ids
                sequence_row["track_point_count"] = int(len(canonical["point_id"]))
                sequence_row["track_visibility_source"] = (
                    "rendered_depth_and_target_mask" if not args.dry_run else "geometry_raycast"
                )
                write_json(sequence_metadata_path, sequence_row)
                sequence_rows.append(sequence_row)
                track_rows.append(
                    {
                        "track_set_id": f"{current_sequence_id}__tracks",
                        "sequence_id": current_sequence_id,
                        "group_id": current_group_id,
                        "scene_id": scene_id_value,
                        "object_anchor_id": anchor_id,
                        "base_camera_id": base_camera_id,
                        "tracks_path": relative_path(tracks_path, output_root),
                        "canonical_points_path": relative_path(canonical_path, output_root),
                        "point_count": int(len(canonical["point_id"])),
                        "num_frames": int(settings["num_frames"]),
                        "visibility_source": sequence_row["track_visibility_source"],
                        "coordinate_convention": track_metadata,
                    }
                )

    matched_relative_rows = build_matched_relative_groups(sequence_rows)
    selected_split_membership: Dict[str, List[str]] = defaultdict(list)
    for row in scene_rows:
        selected_split_membership[str(row["split"])].append(str(row["scene_id"]))
    splits_payload = {
        "unit": "scene",
        "leakage_guard": (
            "Every anchor, camera, condition, frame, and track from one scene retains the source scene split."
        ),
        "train_scenes": list(source["splits"].get("train_scenes", [])),
        "validation_scenes": list(source["splits"].get("validation_scenes", [])),
        "test_scenes": list(source["splits"].get("test_scenes", [])),
        "selected_train_scenes": sorted(selected_split_membership.get("train", [])),
        "selected_validation_scenes": sorted(selected_split_membership.get("val", [])),
        "selected_test_scenes": sorted(selected_split_membership.get("test", [])),
        "source_splits_path": "manifests/splits.json",
    }
    selection_payload = {
        "selection_schema_version": 1,
        "delta_m": delta_m,
        "motion_axis_world": vector_to_list(axis_world),
        "selection_config": config["selection"],
        "scene_reports": selection_scene_reports,
        "selected_groups": [row for scene in selection_scene_reports for row in scene["selected_groups"]],
    }

    manifests_dir = output_root / "manifests"
    write_jsonl(manifests_dir / "scenes.jsonl", scene_rows)
    write_jsonl(manifests_dir / "anchors.jsonl", anchor_rows)
    write_jsonl(manifests_dir / "base_cameras.jsonl", base_camera_rows)
    write_jsonl(manifests_dir / "groups.jsonl", group_rows)
    write_jsonl(manifests_dir / "sequences.jsonl", sequence_rows)
    write_jsonl(manifests_dir / "frames.jsonl", frame_rows)
    write_jsonl(manifests_dir / "matched_relative_groups.jsonl", matched_relative_rows)
    write_jsonl(manifests_dir / "track_sets.jsonl", track_rows)
    write_json(manifests_dir / "splits.json", splits_payload)
    write_json(output_root / "selection_report.json", selection_payload)
    write_json(output_root / "source_rebuild_audit.json", {"scenes": rebuild_audits})

    provenance_after = source_provenance(source_root, configured_source_config, source)
    source_unchanged = (
        provenance_after["source_files_sha256"] == provenance["source_files_sha256"]
    )
    provenance["source_files_sha256_after_generation"] = provenance_after[
        "source_files_sha256"
    ]
    provenance["source_integrity_verified_after_generation"] = bool(source_unchanged)
    write_json(output_root / "provenance.json", provenance)
    if not source_unchanged:
        raise RuntimeError(
            "Source config/manifests changed during generation. New outputs were retained for audit, but the run "
            "must not be treated as valid."
        )

    summary = {
        "dataset_name": config["dataset_name"],
        "schema_version": int(config["schema_version"]),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "output_root": str(output_root),
        "scene_count": len(scene_rows),
        "anchor_count": len(anchor_rows),
        "base_camera_count": len(base_camera_rows),
        "group_count": len(group_rows),
        "sequence_count": len(sequence_rows),
        "frame_count": len(frame_rows),
        "rendered_frame_count": sum(bool(row["rendered"]) for row in frame_rows),
        "matched_relative_group_count": len(matched_relative_rows),
        "track_set_count": len(track_rows),
        "surface_point_count_per_sequence": int(config["tracks"]["num_surface_points"]),
        "delta_m": delta_m,
        "levels": levels,
        "num_frames": int(settings["num_frames"]),
        "resolution": [width, image_height],
        "samples": int(settings["samples"]),
        "selected_scene_ids": scene_ids,
        "run_scope": {
            "scene_id": args.scene_id,
            "anchor_id": args.anchor_id,
            "base_camera_id": args.base_camera_id,
        },
        "full_dataset_was_started": args.mode == "full" and not args.dry_run,
    }
    write_json(output_root / "dataset_summary.json", summary)
    write_dataset_readme(output_root, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    args = parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
