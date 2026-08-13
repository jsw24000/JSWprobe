#!/usr/bin/env python3
"""Build controlled ScanNet sequence manifests from raw scene folders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from controlled_common import (
    PROJECT_DIR,
    git_commit,
    load_config,
    output_root,
    parse_seq_lens,
    project_name,
    resolve_scannet_root,
    save_resolved_config,
    setup_logging,
)

from data.manifest_utils.manifest_schema import build_manifest_index, manifest_record, save_json, write_manifest
from data.manifest_utils.scannet_reader import ScanNetReader
from sequence_settings.registry import create_sequence_setting, list_sequence_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build controlled ScanNet manifest JSON files.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "controlled_scannet_v1.yaml")
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--seq-lens", nargs="+", default=None, help="Override sequence lengths, e.g. --seq-lens 16 32")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def enabled_settings(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    settings = config.get("sequence", {}).get("settings", {})
    return {
        name: dict(setting_config)
        for name, setting_config in settings.items()
        if setting_config.get("enabled", True)
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config, "build_sequence_manifests")

    seq_lens = parse_seq_lens(args.seq_lens) or list(config.get("sequence", {}).get("seq_lens", [16, 32]))
    project = project_name(config)
    manifest_root = output_root(config) / "manifests" / project
    random_seed = int(config.get("project", {}).get("random_seed", 2026))
    path_mode = str(config.get("project", {}).get("path_mode", "absolute"))

    data_cfg = config.get("data", {})
    scene_ids = data_cfg.get("scene_ids") or None
    max_scenes = int(args.limit_scenes or data_cfg.get("max_scenes", 10))
    if scene_ids:
        scene_ids = list(scene_ids)[:max_scenes]
    scannet_root = resolve_scannet_root(data_cfg, required_scene_ids=scene_ids)
    data_cfg["scannet_root"] = str(scannet_root)

    logger.info("Available sequence settings: %s", ", ".join(list_sequence_settings()))
    logger.info("Using ScanNet root: %s", scannet_root)
    reader = ScanNetReader(
        scannet_root,
        require_color=bool(data_cfg.get("require_color", True)),
        require_depth=bool(data_cfg.get("require_depth", True)),
        require_pose=bool(data_cfg.get("require_pose", True)),
        require_intrinsic=bool(data_cfg.get("require_intrinsic", True)),
        require_valid_pose=True,
    )

    ids_to_scan = scene_ids or reader.discover_scene_ids(max_scenes=max_scenes)
    scenes = []
    for scene_id in ids_to_scan[:max_scenes]:
        try:
            scenes.append(reader.scan_scene(scene_id))
        except Exception as exc:
            logger.warning("Skipping scene %s: %s", scene_id, exc)
    settings = enabled_settings(config)
    manifests: list[dict[str, Any]] = []
    for scene in scenes:
        logger.info("Scene %s: %d valid frames", scene.scene_id, len(scene.frames))
        for setting_name, setting_config in settings.items():
            setting_config["seq_lens"] = seq_lens
            setting = create_sequence_setting(
                setting_name,
                setting_config,
                seq_lens=seq_lens,
                random_seed=random_seed,
                path_mode=path_mode,
                relative_to=PROJECT_DIR if path_mode == "relative" else None,
            )
            setting_manifests = setting.build_manifests(scene)
            annotate_manifest_runtime(setting_manifests, config=config, project=project)
            logger.info("  %s: %d manifests", setting_name, len(setting_manifests))
            manifests.extend(setting_manifests)

    if args.dry_run:
        logger.info("Dry-run: would write %d manifests under %s", len(manifests), manifest_root)
        print_summary(manifests, dry_run=True)
        return 0

    records: list[dict[str, Any]] = []
    for manifest in manifests:
        try:
            path = write_manifest(manifest, manifest_root, overwrite=args.overwrite)
        except FileExistsError:
            path = (
                manifest_root
                / manifest["scene_id"]
                / manifest["setting_name"]
                / f"{manifest['condition_id']}.json"
            )
            logger.warning("Skipping existing manifest: %s", path)
        records.append(manifest_record(manifest, path, relative_to=PROJECT_DIR))

    index = build_manifest_index(records, project_name=project, manifest_root=manifest_root)
    save_json(manifest_root / "index.json", index)
    save_resolved_config(config, manifest_root)
    logger.info("Wrote %d manifest records and index: %s", len(records), manifest_root / "index.json")
    print_summary(manifests, dry_run=False)
    return 0


def print_summary(manifests: list[dict[str, Any]], *, dry_run: bool) -> None:
    counts: dict[str, int] = {}
    for manifest in manifests:
        counts[manifest["setting_name"]] = counts.get(manifest["setting_name"], 0) + 1
    status = "DRY-RUN" if dry_run else "WROTE"
    print(f"{status} controlled manifests: {len(manifests)}")
    for setting_name, count in sorted(counts.items()):
        print(f"  {setting_name}: {count}")


def annotate_manifest_runtime(
    manifests: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    project: str,
) -> None:
    """Attach lightweight provenance without changing sequence construction."""

    now = datetime.now(timezone.utc).isoformat()
    commit = git_commit(PROJECT_DIR)
    for manifest in manifests:
        metadata = manifest.setdefault("metadata", {})
        metadata.setdefault("experiment_name", project)
        metadata.setdefault("experiment_config_path", config.get("_config_path"))
        metadata.setdefault("built_at_utc", now)
        metadata.setdefault("git_commit", commit)


if __name__ == "__main__":
    raise SystemExit(main())
