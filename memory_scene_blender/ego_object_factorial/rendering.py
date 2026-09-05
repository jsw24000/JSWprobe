"""Rendering and compact track-output helpers for factorial sequences."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from memory_scene_blender.object_translation.manifest_utils import ensure_dir, read_json
from memory_scene_blender.object_translation.scene_builder import render_frame_passes

from .geometry import depth_buffer_track_visibility, load_mask_blender


RENDERED_PASS_FILENAMES = (
    "rgb.png",
    "depth.exr",
    "depth.npy",
    "normal.png",
    "target_mask.png",
    "instance.png",
    "semantic.png",
    "object_id.png",
    "albedo.png",
    "object_id_palette.json",
    "semantic_palette.json",
)


def complete_rendered_frame(frame_dir: Path) -> bool:
    return all((frame_dir / name).exists() for name in RENDERED_PASS_FILENAMES)


def clone_rendered_passes(source_dir: Path, destination_dir: Path) -> str:
    """Hard-link identical physical renders, falling back to file copies."""
    if not complete_rendered_frame(source_dir):
        raise FileNotFoundError(f"Cached physical frame is incomplete: {source_dir}")
    ensure_dir(destination_dir)
    method = "hardlink"
    for filename in RENDERED_PASS_FILENAMES:
        source = source_dir / filename
        destination = destination_dir / filename
        if destination.exists():
            if source.samefile(destination):
                continue
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
            method = "copy"
    return method


def render_or_reuse_physical_frame(
    frame_dir: Path,
    target_asset: Any,
    camera_payload: Mapping[str, Any],
    samples: int,
    resolution: Sequence[int],
    resume: bool,
    cached_frame_dir: Path | None,
) -> Dict[str, Any]:
    """Render one physical state or reuse an exactly identical cached state."""
    if resume and complete_rendered_frame(frame_dir) and (frame_dir / "frame_metadata.json").exists():
        metadata = read_json(frame_dir / "frame_metadata.json")
        return {
            "render_source": "resume_existing",
            "cache_method": None,
            "mask_stats": {
                "mask_pixel_count": int(metadata["target_mask_pixel_count"]),
                "mask_area_ratio": float(metadata["target_mask_area_ratio"]),
                "bbox": metadata.get("target_bbox_2d"),
                "truncated": bool(metadata.get("target_truncated", False)),
                "edge_touch_ratio": float(metadata.get("target_edge_touch_ratio", 0.0)),
                "image_size": list(metadata["image_size"]),
            },
            "visible_fraction": float(metadata.get("render_estimated_visible_fraction", 0.0)),
            "depth_range_m": metadata.get("depth_range_m"),
        }

    if cached_frame_dir is not None:
        method = clone_rendered_passes(cached_frame_dir, frame_dir)
        cached_metadata = read_json(cached_frame_dir / "frame_metadata.json")
        return {
            "render_source": "identical_physical_state_cache",
            "cache_method": method,
            "mask_stats": {
                "mask_pixel_count": int(cached_metadata["target_mask_pixel_count"]),
                "mask_area_ratio": float(cached_metadata["target_mask_area_ratio"]),
                "bbox": cached_metadata.get("target_bbox_2d"),
                "truncated": bool(cached_metadata.get("target_truncated", False)),
                "edge_touch_ratio": float(cached_metadata.get("target_edge_touch_ratio", 0.0)),
                "image_size": list(cached_metadata["image_size"]),
            },
            "visible_fraction": float(cached_metadata.get("render_estimated_visible_fraction", 0.0)),
            "depth_range_m": cached_metadata.get("depth_range_m"),
        }

    result = render_frame_passes(
        frame_dir,
        target_asset,
        camera_payload,
        samples=int(samples),
        overwrite=False,
        expected_resolution=resolution,
    )
    if "frame_id" in result:
        # This can occur only on an implicit existing-frame path.  The new
        # generator does not silently accept it unless --resume was requested.
        raise RuntimeError(f"Unexpected pre-existing frame without --resume: {frame_dir}")
    return {"render_source": "rendered", "cache_method": None, **result}


def refine_observation_from_render(
    observation: Mapping[str, np.ndarray],
    frame_dir: Path,
    tracks_cfg: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    depth = np.load(frame_dir / "depth.npy", allow_pickle=False)
    mask = load_mask_blender(frame_dir / "target_mask.png")
    return depth_buffer_track_visibility(
        observation,
        depth,
        mask,
        float(tracks_cfg["visibility_depth_abs_tolerance_m"]),
        float(tracks_cfg["visibility_depth_rel_tolerance"]),
        int(tracks_cfg["visibility_pixel_radius"]),
    )


def stack_track_observations(
    canonical: Mapping[str, np.ndarray],
    observations: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    if not observations:
        raise ValueError("Cannot save an empty track sequence")
    payload: Dict[str, np.ndarray] = {
        "point_id": np.asarray(canonical["point_id"], dtype=np.int32),
        "xyz_object_local": np.asarray(canonical["xyz_object_local"], dtype=np.float64),
        "normal_object_local": np.asarray(canonical["normal_object_local"], dtype=np.float64),
        "part_index": np.asarray(canonical["part_index"], dtype=np.int32),
        "polygon_index": np.asarray(canonical["polygon_index"], dtype=np.int32),
        "sampling_triangle_index": np.asarray(canonical["sampling_triangle_index"], dtype=np.int32),
    }
    frame_keys = (
        "xyz_world",
        "xyz_camera",
        "projected_uv",
        "depth_camera_z_m",
        "range_to_camera_m",
        "in_front_of_camera",
        "in_image",
        "visible",
    )
    optional_keys = ("depth_buffer_range_m", "depth_buffer_error_m")
    for key in frame_keys:
        payload[key] = np.stack([np.asarray(row[key]) for row in observations], axis=0)
    for key in optional_keys:
        if all(key in row for row in observations):
            payload[key] = np.stack([np.asarray(row[key]) for row in observations], axis=0)
    return payload


def save_npz(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping[str, Any] | None = None) -> None:
    ensure_dir(path.parent)
    values = {key: np.asarray(value) for key, value in payload.items()}
    if metadata is not None:
        values["metadata_json_utf8"] = np.asarray(
            json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")), dtype=np.str_
        )
    np.savez_compressed(path, **values)
