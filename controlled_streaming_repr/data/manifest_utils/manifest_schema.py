"""Manifest schema helpers for controlled ScanNet sequence experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .scannet_reader import ScanNetFrame


CONTROLLED_MANIFEST_VERSION = "controlled_scannet_v1"


def save_json(path: str | Path, data: Any) -> Path:
    """Save JSON with the repository's standard formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _format_path(path: Path | None, *, path_mode: str, relative_to: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if path_mode == "relative" and relative_to is not None:
        try:
            return str(resolved.relative_to(relative_to.expanduser().resolve()))
        except ValueError:
            return str(resolved)
    return str(resolved)


def manifest_frame_entry(
    frame: ScanNetFrame,
    *,
    t: int,
    role: str,
    path_mode: str = "absolute",
    relative_to: str | Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create one frame entry for a controlled sequence manifest."""

    base = Path(relative_to).resolve() if relative_to else None
    entry: dict[str, Any] = {
        "t": t,
        "source_frame_id": frame.source_frame_id,
        "rgb_path": _format_path(frame.rgb_path, path_mode=path_mode, relative_to=base),
        "depth_path": _format_path(frame.depth_path, path_mode=path_mode, relative_to=base),
        "pose_path": _format_path(frame.pose_path, path_mode=path_mode, relative_to=base),
        "intrinsic_path": _format_path(frame.intrinsic_path, path_mode=path_mode, relative_to=base),
        "role": role,
        "original_order_index": frame.original_order_index,
    }
    entry.update(extra)
    return entry


def make_manifest(
    *,
    scene_id: str,
    setting_name: str,
    condition_id: str,
    seq_len: int,
    source_scene_dir: str | Path,
    frames: list[dict[str, Any]],
    metadata: dict[str, Any],
    path_mode: str = "absolute",
    relative_to: str | Path | None = None,
    dataset: str = "scannet",
) -> dict[str, Any]:
    """Create a versioned controlled ScanNet manifest dictionary."""

    base = Path(relative_to).resolve() if relative_to else None
    scene_dir = _format_path(Path(source_scene_dir), path_mode=path_mode, relative_to=base)
    return {
        "manifest_version": CONTROLLED_MANIFEST_VERSION,
        "dataset": dataset,
        "scene_id": scene_id,
        "setting_name": setting_name,
        "condition_id": condition_id,
        "seq_len": int(seq_len),
        "source_scene_dir": scene_dir,
        "frames": frames,
        "metadata": metadata,
    }


def write_manifest(
    manifest: dict[str, Any],
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a manifest under scene/setting/condition_id.json."""

    root = Path(output_root)
    path = (
        root
        / manifest["scene_id"]
        / manifest["setting_name"]
        / f"{manifest['condition_id']}.json"
    )
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {path}")
    return save_json(path, manifest)


def manifest_record(manifest: dict[str, Any], path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, Any]:
    """Create a compact index record for a manifest."""

    manifest_path = Path(path).resolve()
    if relative_to is not None:
        try:
            display_path = str(manifest_path.relative_to(Path(relative_to).resolve()))
        except ValueError:
            display_path = str(manifest_path)
    else:
        display_path = str(manifest_path)
    return {
        "path": display_path,
        "scene_id": manifest["scene_id"],
        "setting_name": manifest["setting_name"],
        "condition_id": manifest["condition_id"],
        "seq_len": int(manifest["seq_len"]),
    }


def build_manifest_index(
    records: list[dict[str, Any]],
    *,
    project_name: str = CONTROLLED_MANIFEST_VERSION,
    manifest_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build an index grouped by scene, setting, condition, and sequence length."""

    by_scene: dict[str, list[str]] = {}
    by_setting: dict[str, list[str]] = {}
    by_condition: dict[str, str] = {}
    by_seq_len: dict[str, list[str]] = {}
    by_scene_setting: dict[str, dict[str, dict[str, list[str]]]] = {}

    for record in sorted(records, key=lambda item: (item["scene_id"], item["setting_name"], item["seq_len"], item["condition_id"])):
        path = record["path"]
        scene = record["scene_id"]
        setting = record["setting_name"]
        seq_len = str(record["seq_len"])
        condition = record["condition_id"]
        by_scene.setdefault(scene, []).append(path)
        by_setting.setdefault(setting, []).append(path)
        by_condition[condition] = path
        by_seq_len.setdefault(seq_len, []).append(path)
        by_scene_setting.setdefault(scene, {}).setdefault(setting, {}).setdefault(seq_len, []).append(path)

    return {
        "index_version": CONTROLLED_MANIFEST_VERSION,
        "manifest_version": CONTROLLED_MANIFEST_VERSION,
        "project_name": project_name,
        "manifest_root": str(manifest_root) if manifest_root is not None else None,
        "manifest_count": len(records),
        "manifests": sorted(records, key=lambda item: item["path"]),
        "by_scene": by_scene,
        "by_setting": by_setting,
        "by_condition": by_condition,
        "by_seq_len": by_seq_len,
        "by_scene_setting": by_scene_setting,
    }
