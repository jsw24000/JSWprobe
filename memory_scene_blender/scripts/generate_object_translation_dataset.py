#!/usr/bin/env python3
"""Generate the controlled object-translation Blender dataset.

Run with Blender:

  blender --background --python scripts/generate_object_translation_dataset.py -- \
    --config configs/object_translation_v1.yaml --mode smoke
"""

from __future__ import annotations

import argparse
import shutil
import sys
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
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only outside Blender
    raise SystemExit(
        "This generator must run inside Blender. Example:\n"
        "  blender --background --python memory_scene_blender/scripts/generate_object_translation_dataset.py -- "
        "--config memory_scene_blender/configs/object_translation_v1.yaml --mode smoke"
    ) from exc

from memory_scene_blender.object_translation.assets import CATEGORY_IDS, TARGET_CATEGORIES, asset_world_bbox, logical_asset_metadata
from memory_scene_blender.object_translation.label_utils import frame_label_payload, state_label_payload
from memory_scene_blender.object_translation.manifest_utils import (
    build_composition_triplets,
    build_splits,
    build_translation_pairs,
    ensure_dir,
    load_config,
    mode_settings,
    normalize_output_root,
    read_json,
    summarize_dataset,
    write_json,
    write_jsonl,
)
from memory_scene_blender.object_translation.scene_builder import TARGET_OBJECT_ID, build_scene_bundle, render_frame_passes, set_target_position


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Generate object-translation synthetic data with Blender.")
    parser.add_argument("--config", type=Path, default=PACKAGE_DIR / "configs" / "object_translation_v1.yaml")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Remove and recreate the selected output root.")
    parser.add_argument("--scene-id", type=str, default=None, help="Render only one scene id, e.g. scene_003.")
    parser.add_argument("--scene-index", type=int, default=None, help="Render only one scene index.")
    parser.add_argument("--state-start", type=int, default=None, help="First state index, inclusive.")
    parser.add_argument("--state-end", type=int, default=None, help="Last state index, exclusive.")
    parser.add_argument("--camera-start", type=int, default=None, help="First camera index, inclusive.")
    parser.add_argument("--camera-end", type=int, default=None, help="Last camera index, exclusive.")
    parser.add_argument("--dry-run", action="store_true", help="Build scenes, labels, and manifests without rendering frames.")
    return parser.parse_args(argv)


def selected_scene_indices(args: argparse.Namespace, scene_count: int) -> List[int]:
    if args.scene_id is not None:
        if not args.scene_id.startswith("scene_"):
            raise ValueError("--scene-id must look like scene_003")
        return [int(args.scene_id.split("_", 1)[1])]
    if args.scene_index is not None:
        return [int(args.scene_index)]
    return list(range(scene_count))


def slice_by_range(items: Sequence[Any], start: Optional[int], end: Optional[int]) -> List[Any]:
    return list(items[slice(start, end)])


def write_dataset_readme(output_root: Path, config: Mapping[str, Any], mode: str, summary: Mapping[str, Any]) -> None:
    text = f"""# Object Translation V1

This dataset renders controlled indoor scenes for studying visual representation changes caused by translating one target object on the ground plane.

## Controlled Variables

Within each `scene_xxx`, the room, static furniture, target mesh, target material, target orientation, lighting, render settings, and camera set are fixed. Only the target object's ground-plane XY translation changes across states. Every state is rendered from the same camera IDs, and cameras look at a fixed scene point rather than tracking the target.

## Coordinates

Units are meters. Blender world uses `+Z` up. The target root transform is the target object's object-to-world matrix; its root origin is the intended bottom-center control point. `bbox_corners_world` are world-axis-aligned bounding-box corners computed from the rendered target mesh parts after applying Blender transforms.

Blender camera coordinates use Blender's camera convention: local `-Z` points forward and local `+Y` points up. OpenCV camera coordinates are stored separately with `+X` right, `+Y` down, and `+Z` forward. The saved conversion matrix is:

```text
blender_camera_to_opencv = diag(1, -1, -1, 1)
opencv_world_to_camera = blender_camera_to_opencv @ blender_world_to_camera
```

`camera_000` is the reference camera. Each state stores `object_center_world` and `object_center_ref_camera`; each frame also stores `object_center_current_camera`.

## Files

Manifests live under `manifests/`: `scenes.jsonl`, `cameras.jsonl`, `states.jsonl`, `frames.jsonl`, `translation_pairs.jsonl`, `composition_triplets.jsonl`, and `splits.json`.

Per-frame outputs are aligned pixel-for-pixel:

```text
scene_000/state_000/camera_000/
  rgb.png
  depth.exr
  normal.png
  albedo.png
  target_mask.png
  instance.png
  semantic.png
  object_id.png
  frame_metadata.json
```

## Run

Smoke:

```bash
blender --background --python memory_scene_blender/scripts/generate_object_translation_dataset.py -- \\
  --config memory_scene_blender/configs/object_translation_v1.yaml \\
  --mode smoke
python memory_scene_blender/scripts/render_object_translation_preview.py \\
  --dataset-root memory_scene_blender/outputs/object_translation_v1
python memory_scene_blender/scripts/validate_object_translation_dataset.py \\
  --dataset-root memory_scene_blender/outputs/object_translation_v1
```

Full:

```bash
blender --background --python memory_scene_blender/scripts/generate_object_translation_dataset.py -- \\
  --config memory_scene_blender/configs/object_translation_v1.yaml \\
  --mode full
```

The generator skips complete frames by default. Use `--overwrite` to intentionally recreate the output root.
If this root already contains a smoke run, use `--overwrite` or a different `--output-root` before starting the full 512x512 run; complete frames at a different resolution are rejected instead of mixed.

## Current Run

- Mode: `{mode}`
- Scenes: {summary.get("scene_count")}
- States: {summary.get("state_count")}
- Cameras rows: {summary.get("camera_count")}
- Frames: {summary.get("frame_count")}
- Translation pairs: {summary.get("pair_count")}
- Composition triplets: {summary.get("triplet_count")}
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def target_object_id(scene_index: int, target_category: str, target_variant: int) -> str:
    return f"{target_category}_{target_variant:03d}"


def build_scene_metadata(bundle: Any, config: Mapping[str, Any], target_object_name: str) -> Dict[str, Any]:
    static_signature = [
        {
            "logical_name": row["logical_name"],
            "object_id": row["object_id"],
            "category": row["category"],
            "root_matrix_world": row["root_matrix_world"],
            "signature": row["signature"],
        }
        for row in bundle.static_metadata
    ]
    return {
        "scene_id": bundle.scene_id,
        "dataset_name": config["dataset_name"],
        "scene_seed": int(bundle.seed),
        "layout_id": bundle.layout["layout_id"],
        "room_dimensions": bundle.layout["dimensions"],
        "wall_mode": bundle.layout.get("wall_mode", "north_west"),
        "floor_material": bundle.layout["floor_material"],
        "wall_material": bundle.layout["wall_material"],
        "lighting_preset": bundle.layout["lighting_preset"],
        "target_object_id": target_object_name,
        "target_object_numeric_id": TARGET_OBJECT_ID,
        "target_category": bundle.target_asset.category,
        "target_asset_signature": bundle.target_asset.signature,
        "target_part_names": [obj.name for obj in bundle.target_asset.parts],
        "static_objects": bundle.static_metadata,
        "non_target_layout_signature": static_signature,
        "position_grid": bundle.positions,
        "camera_ids": [camera["camera_id"] for camera in bundle.cameras],
    }


def write_scene_camera_file(scene_dir: Path, cameras: Sequence[Mapping[str, Any]]) -> None:
    write_json(
        scene_dir / "cameras.json",
        {
            "reference_camera_id": cameras[0]["camera_id"],
            "blender_to_opencv_camera": cameras[0]["blender_to_opencv_camera"],
            "cameras": list(cameras),
        },
    )


def write_config_used(output_root: Path, config: Mapping[str, Any], mode: str, settings: Mapping[str, Any]) -> None:
    payload = dict(config)
    payload["mode_used"] = mode
    payload["effective_mode_settings"] = dict(settings)
    write_json(output_root / "config_used.yaml", payload)


def generate_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    settings = mode_settings(config, args.mode)
    output_root = normalize_output_root(args.output_root or Path(config["output_root"]), PACKAGE_DIR)

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)
    ensure_dir(output_root / "manifests")
    ensure_dir(output_root / "previews")
    write_config_used(output_root, config, args.mode, settings)

    scene_rows: List[Dict[str, Any]] = []
    camera_rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    triplet_rows: List[Dict[str, Any]] = []

    scene_indices = selected_scene_indices(args, int(settings["scene_count"]))
    for scene_index in scene_indices:
        print(f"\n=== Building {scene_index=:03d} ===")
        bundle = build_scene_bundle(scene_index, config, settings)
        scene_dir = ensure_dir(output_root / bundle.scene_id)
        write_scene_camera_file(scene_dir, bundle.cameras)

        target_variant = int(bundle.target_asset.signature.get("variant", 0))
        target_object_name = target_object_id(scene_index, bundle.target_asset.category, target_variant)
        scene_metadata = build_scene_metadata(bundle, config, target_object_name)
        write_json(scene_dir / "scene_metadata.json", scene_metadata)
        scene_rows.append(
            {
                "scene_id": bundle.scene_id,
                "scene_path": str(scene_dir),
                "scene_seed": int(bundle.seed),
                "layout_id": bundle.layout["layout_id"],
                "target_object_id": target_object_name,
                "target_object_numeric_id": TARGET_OBJECT_ID,
                "target_category": bundle.target_asset.category,
                "target_variant": target_variant,
                "camera_count": len(bundle.cameras),
                "state_count": len(bundle.positions),
            }
        )

        selected_cameras = slice_by_range(bundle.cameras, args.camera_start, args.camera_end)
        selected_positions = slice_by_range(bundle.positions, args.state_start, args.state_end)
        camera_rows.extend(dict(camera) for camera in selected_cameras)

        state_payloads: List[Dict[str, Any]] = []
        reference_camera = bundle.cameras[0]
        for position in selected_positions:
            set_target_position(bundle.target_asset, position["xy"])
            bbox_corners = asset_world_bbox(bundle.target_asset)
            state_payload = state_label_payload(
                bundle.scene_id,
                target_object_name,
                bundle.target_asset.category,
                position["state_id"],
                np.asarray(bundle.target_asset.root.matrix_world, dtype=np.float64),
                bbox_corners,
                reference_camera,
                int(bundle.seed),
                position["position_index"],
                bundle.target_asset.signature,
            )
            state_payload["target_part_names"] = [obj.name for obj in bundle.target_asset.parts]
            state_payload["non_target_layout_signature"] = scene_metadata["non_target_layout_signature"]
            state_payloads.append(state_payload)
            state_dir = ensure_dir(scene_dir / position["state_id"])
            write_json(state_dir / "state_metadata.json", state_payload)
            state_rows.append(state_payload)

        frame_lookup: Dict[Tuple[str, str], str] = {}
        for state_payload, position in zip(state_payloads, selected_positions):
            set_target_position(bundle.target_asset, position["xy"])
            for camera in selected_cameras:
                frame_dir = ensure_dir(scene_dir / state_payload["state_id"] / camera["camera_id"])
                if args.dry_run:
                    render_result = {
                        "mask_stats": {
                            "mask_pixel_count": 0,
                            "mask_area_ratio": 0.0,
                            "bbox": None,
                            "truncated": False,
                            "edge_touch_ratio": 0.0,
                            "image_size": settings["resolution"],
                        },
                        "visible_fraction": 0.0,
                        "depth_range_m": None,
                    }
                else:
                    print(f"[{bundle.scene_id}/{state_payload['state_id']}/{camera['camera_id']}] render")
                    render_result = render_frame_passes(
                        frame_dir,
                        bundle.target_asset,
                        camera,
                        samples=int(settings["samples"]),
                        overwrite=bool(args.overwrite),
                        expected_resolution=settings["resolution"],
                    )

                if "frame_id" in render_result:
                    frame_meta = dict(render_result)
                else:
                    frame_meta = frame_label_payload(
                        bundle.scene_id,
                        target_object_name,
                        bundle.target_asset.category,
                        state_payload,
                        camera,
                        frame_dir,
                        render_result["mask_stats"],
                        float(render_result["visible_fraction"]),
                        render_result["depth_range_m"],
                        float(settings.get("min_mask_area_ratio", config["visibility"]["min_mask_area_ratio"])),
                    )
                    if not args.dry_run:
                        write_json(frame_dir / "frame_metadata.json", frame_meta)
                frame_lookup[(state_payload["state_id"], camera["camera_id"])] = str(frame_dir)
                frame_rows.append(frame_meta)

        pair_rows.extend(
            build_translation_pairs(
                bundle.scene_id,
                target_object_name,
                state_payloads,
                selected_cameras,
                frame_lookup,
                int(settings["grid_rows"]),
                int(settings["grid_cols"]),
            )
        )
        triplet_rows.extend(
            build_composition_triplets(
                bundle.scene_id,
                target_object_name,
                state_payloads,
                selected_cameras,
                int(settings["grid_rows"]),
                int(settings["grid_cols"]),
            )
        )

    manifests_dir = output_root / "manifests"
    write_jsonl(manifests_dir / "scenes.jsonl", scene_rows)
    write_jsonl(manifests_dir / "cameras.jsonl", camera_rows)
    write_jsonl(manifests_dir / "states.jsonl", state_rows)
    write_jsonl(manifests_dir / "frames.jsonl", frame_rows)
    write_jsonl(manifests_dir / "translation_pairs.jsonl", pair_rows)
    write_jsonl(manifests_dir / "composition_triplets.jsonl", triplet_rows)
    write_json(manifests_dir / "splits.json", build_splits(scene_rows, camera_rows, config))

    summary = summarize_dataset(scene_rows, camera_rows, state_rows, frame_rows, pair_rows, triplet_rows)
    summary["mode"] = args.mode
    summary["output_root"] = str(output_root)
    write_json(output_root / "dataset_summary.json", summary)
    write_dataset_readme(output_root, config, args.mode, summary)
    print(f"\nWrote object-translation dataset to {output_root}")
    print(
        "Counts: "
        f"scenes={summary['scene_count']} states={summary['state_count']} "
        f"cameras={summary['camera_count']} frames={summary['frame_count']} "
        f"pairs={summary['pair_count']} triplets={summary['triplet_count']}"
    )
    return summary


def main() -> None:
    args = parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
