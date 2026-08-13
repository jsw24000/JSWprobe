#!/usr/bin/env python3
"""Run Lingbot-map geometry extraction for controlled ScanNet manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from controlled_common import (
    PROJECT_DIR,
    add_common_filters,
    filtered_manifest_records,
    load_config,
    load_json,
    output_root,
    project_name,
    resolve_index_path,
    resolve_manifest_path,
    save_json,
    save_resolved_config,
    setup_logging,
)

from adapters.lingbot_adapter import LingbotAdapter
from adapters.token_schema import TokenBundle, load_token_bundle, save_token_bundle
from run_controlled_extraction import (
    controlled_sequence_id,
    materialize_image_sequence,
    missing_fields,
    token_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled Lingbot-map geometry extraction.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "controlled_scannet_v1.yaml")
    parser.add_argument("--manifest-index", type=Path, default=PROJECT_DIR / "outputs" / "manifests" / "controlled_scannet_v1" / "index.json")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dense-world-points", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_common_filters(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config, "run_controlled_geometry")
    project = project_name(config)
    index_path = resolve_index_path(args.manifest_index, config)
    index = load_json(index_path)
    records = filtered_manifest_records(index, args)

    extraction_cfg = dict(config.get("extraction", {}))
    models = args.models or list(extraction_cfg.get("models", ["lingbot-map"]))
    device = args.device or extraction_cfg.get("device", "cuda")
    skip_existing = bool(extraction_cfg.get("skip_existing", True)) and not args.overwrite
    save_dense_world_points = bool(args.save_dense_world_points or extraction_cfg.get("save_dense_world_points", False))

    logger.info("Selected %d manifests from %s", len(records), index_path)
    logger.info("Models: %s", ", ".join(models))

    summaries: list[dict[str, Any]] = []
    for record in records:
        manifest_path = resolve_manifest_path(record["path"], index_path)
        manifest = load_json(manifest_path)
        for model_name in models:
            summary = process_manifest_model(
                manifest,
                manifest_path=manifest_path,
                model_name=model_name,
                config=config,
                extraction_cfg=extraction_cfg,
                device=device,
                project=project,
                dry_run=args.dry_run,
                skip_existing=skip_existing,
                save_dense_world_points=save_dense_world_points,
            )
            summaries.append(summary)
            logger.info(
                "%s | %s | %s | %s",
                summary.get("status"),
                model_name,
                manifest.get("scene_id"),
                manifest.get("condition_id"),
            )

    if args.dry_run:
        print(json.dumps({"dry_run": True, "runs": summaries}, ensure_ascii=False, indent=2))
        return 0

    run_dir = output_root(config) / "reconstruction_official" / project
    save_resolved_config(config, run_dir)
    save_json(run_dir / "last_geometry_summary.json", {"runs": summaries})
    logger.info("Geometry complete. Summary: %s", run_dir / "last_geometry_summary.json")
    return 0


def process_manifest_model(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    model_name: str,
    config: dict[str, Any],
    extraction_cfg: dict[str, Any],
    device: str,
    project: str,
    dry_run: bool,
    skip_existing: bool,
    save_dense_world_points: bool,
) -> dict[str, Any]:
    geometry_dir = (
        output_root(config)
        / "reconstruction_official"
        / project
        / model_name
        / manifest["scene_id"]
        / manifest["setting_name"]
        / manifest["condition_id"]
        / "geometry"
    )
    token_dir = (
        output_root(config)
        / "tokens"
        / project
        / model_name
        / manifest["scene_id"]
        / manifest["setting_name"]
        / manifest["condition_id"]
    )

    if model_name != "lingbot-map":
        return {
            "status": "skipped",
            "model_name": model_name,
            "scene_id": manifest.get("scene_id"),
            "setting_name": manifest.get("setting_name"),
            "condition_id": manifest.get("condition_id"),
            "seq_len": manifest.get("seq_len"),
            "reason": "Geometry extraction is implemented for lingbot-map only.",
            "output_dir": str(geometry_dir),
        }

    metadata_path = geometry_dir / "metadata.json"
    if skip_existing and metadata_path.is_file() and not dry_run:
        metadata = load_json(metadata_path)
        if metadata.get("status") in {None, "ok"}:
            summary = attach_geometry_to_bundle(
                manifest,
                manifest_path=manifest_path,
                metadata=metadata,
                metadata_path=metadata_path,
                token_dir=token_dir,
                dry_run=False,
            )
            summary["status"] = "attached_existing"
            save_json(geometry_dir / "geometry_summary.json", summary)
            return summary

    return run_lingbot_geometry(
        manifest,
        manifest_path=manifest_path,
        geometry_dir=geometry_dir,
        token_dir=token_dir,
        config=config,
        extraction_cfg=extraction_cfg,
        device=device,
        dry_run=dry_run,
        save_dense_world_points=save_dense_world_points,
    )


def run_lingbot_geometry(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    geometry_dir: Path,
    token_dir: Path,
    config: dict[str, Any],
    extraction_cfg: dict[str, Any],
    device: str,
    dry_run: bool,
    save_dense_world_points: bool,
) -> dict[str, Any]:
    sequence_dir = (
        output_root(config)
        / "controlled_sequences"
        / project_name(config)
        / manifest["scene_id"]
        / manifest["setting_name"]
        / manifest["condition_id"]
    )
    image_folder = sequence_dir / "images"

    if not dry_run:
        materialize_image_sequence(manifest, image_folder, relative_base=PROJECT_DIR)
        save_json(sequence_dir / "sequence_manifest.json", manifest)

    command = build_lingbot_geometry_command(
        image_folder,
        geometry_dir,
        config=config,
        extraction_cfg=extraction_cfg,
        device=device,
        save_dense_world_points=save_dense_world_points,
    )
    if dry_run:
        return {
            "status": "dry_run",
            "model_name": "lingbot-map",
            "scene_id": manifest.get("scene_id"),
            "setting_name": manifest.get("setting_name"),
            "condition_id": manifest.get("condition_id"),
            "seq_len": manifest.get("seq_len"),
            "image_folder": str(image_folder),
            "output_dir": str(geometry_dir),
            "command": command,
        }

    geometry_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        summary = {
            "status": "skipped",
            "model_name": "lingbot-map",
            "scene_id": manifest.get("scene_id"),
            "setting_name": manifest.get("setting_name"),
            "condition_id": manifest.get("condition_id"),
            "seq_len": manifest.get("seq_len"),
            "reason": str(exc),
            "output_dir": str(geometry_dir),
        }
        save_json(geometry_dir / "geometry_summary.json", summary)
        return summary

    metadata_path = geometry_dir / "metadata.json"
    metadata = load_json(metadata_path)
    summary = attach_geometry_to_bundle(
        manifest,
        manifest_path=manifest_path,
        metadata=metadata,
        metadata_path=metadata_path,
        token_dir=token_dir,
        dry_run=False,
    )
    summary["status"] = "ok"
    save_json(geometry_dir / "geometry_summary.json", summary)
    return summary


def build_lingbot_geometry_command(
    image_folder: Path,
    geometry_dir: Path,
    *,
    config: dict[str, Any],
    extraction_cfg: dict[str, Any],
    device: str,
    save_dense_world_points: bool,
) -> list[str]:
    adapter = LingbotAdapter(config)
    command = [
        str(extraction_cfg.get("python_executable") or sys.executable),
        str(PROJECT_DIR / "scripts" / "run_lingbot_official_geometry.py"),
        "--image-folder",
        str(image_folder.resolve()),
        "--output-dir",
        str(geometry_dir.resolve()),
        "--lingbot-repo",
        str(adapter.repo_path.resolve()),
        "--model-path",
        str(adapter.checkpoint_path.resolve()),
        "--device",
        device,
        "--image-size",
        str(int(extraction_cfg.get("image_size", 518))),
        "--patch-size",
        str(int(extraction_cfg.get("patch_size", 14))),
        "--num-scale-frames",
        str(int(extraction_cfg.get("num_scale_frames", 8))),
        "--camera-num-iterations",
        str(int(extraction_cfg.get("camera_num_iterations", 4))),
        "--keyframe-interval",
        str(int(extraction_cfg.get("keyframe_interval", 1))),
        "--kv-cache-sliding-window",
        str(int(extraction_cfg.get("kv_cache_sliding_window", 64))),
        "--max-frame-num",
        str(int(extraction_cfg.get("max_frame_num", 1024))),
    ]
    command.append("--use-sdpa" if bool(extraction_cfg.get("use_sdpa", True)) else "--no-use-sdpa")
    if save_dense_world_points:
        command.append("--save-dense-world-points")
    return command


def attach_geometry_to_bundle(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
    token_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    bundle_json = token_dir / "token_bundle.json"
    bundle_pt = token_dir / "token_bundle.pt"
    bundle = load_token_bundle(bundle_json) if bundle_json.is_file() else geometry_only_bundle(manifest)

    geometry_outputs = geometry_outputs_from_metadata(metadata)
    bundle.geometry_outputs = geometry_outputs
    bundle.output_predictions = None
    bundle.scene_id = manifest.get("scene_id")
    bundle.setting_name = manifest.get("setting_name")
    bundle.condition_id = manifest.get("condition_id", bundle.condition_id)
    bundle.sequence_id = controlled_sequence_id(manifest)
    bundle.seq_len = manifest.get("seq_len")
    bundle.frame_paths = [frame.get("rgb_path") for frame in manifest.get("frames", [])]
    bundle.frame_ids = [frame.get("source_frame_id") for frame in manifest.get("frames", [])]
    bundle.metadata.setdefault("manifest_path", str(manifest_path))
    bundle.metadata.setdefault("manifest_metadata", manifest.get("metadata", {}))
    bundle.metadata["geometry"] = {
        "source_metadata": str(metadata_path.resolve()),
        "output_dir": metadata.get("output_dir"),
        "shapes": metadata.get("shapes", {}),
        "outputs": metadata.get("outputs", {}),
        "ply_stats": metadata.get("ply_stats", {}),
    }
    bundle.metadata["missing_fields"] = missing_fields(bundle)

    summary = {
        "status": "dry_run" if dry_run else "ok",
        "model_name": bundle.model_name,
        "scene_id": bundle.scene_id,
        "setting_name": bundle.setting_name,
        "condition_id": bundle.condition_id,
        "seq_len": bundle.seq_len,
        "frame_count": len(bundle.frame_paths),
        "geometry_dir": str(metadata_path.parent),
        "geometry_metadata": str(metadata_path),
        "token_bundle": str(bundle_json),
        "token_bundle_pt": str(bundle_pt) if bundle_pt.is_file() else None,
        "geometry_outputs": geometry_outputs,
        "missing_fields": bundle.metadata["missing_fields"],
    }
    if dry_run:
        return summary

    token_dir.mkdir(parents=True, exist_ok=True)
    saved_json = save_token_bundle(bundle, bundle_json)
    saved_pt = save_token_bundle(bundle, bundle_pt) if bundle_pt.is_file() else None
    summary["token_bundle"] = str(saved_json)
    summary["token_bundle_pt"] = str(saved_pt) if saved_pt else None
    save_json(token_dir / "token_summary.json", token_summary(bundle, bundle_json=saved_json, bundle_pt=saved_pt))
    return summary


def geometry_only_bundle(manifest: dict[str, Any]) -> TokenBundle:
    frames = manifest.get("frames", [])
    return TokenBundle(
        model_name="lingbot-map",
        sequence_id=controlled_sequence_id(manifest),
        scene_id=manifest.get("scene_id"),
        setting_name=manifest.get("setting_name"),
        condition_id=manifest.get("condition_id"),
        seq_len=manifest.get("seq_len"),
        frame_paths=[frame.get("rgb_path") for frame in frames],
        frame_ids=[frame.get("source_frame_id") for frame in frames],
        metadata={"manifest_metadata": manifest.get("metadata", {})},
    )


def geometry_outputs_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    outputs = metadata.get("outputs", {})
    geometry_outputs: dict[str, Any] = {}
    for key in ("extrinsic", "intrinsic", "pose_enc", "depth", "depth_conf"):
        if outputs.get(key):
            geometry_outputs[key] = outputs[key]
    if outputs.get("world_points_from_depth"):
        geometry_outputs["pointmap"] = outputs["world_points_from_depth"]
    return geometry_outputs


if __name__ == "__main__":
    raise SystemExit(main())
