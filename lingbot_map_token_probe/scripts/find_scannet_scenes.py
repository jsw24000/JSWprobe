#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_utils import (
    candidate_data_roots,
    discover_scenes,
    list_images,
    log,
    natural_key,
    scene_info_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find ScanNet scene directories and RGB image folders."
    )
    parser.add_argument("--data-root", default="/data1", help="Data root to search.")
    parser.add_argument("--max-depth", type=int, default=7, help="Maximum search depth.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--limit", type=int, default=40, help="Maximum scenes to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots, scenes = discover_scenes(args.data_root, max_depth=args.max_depth)
    roots_for_print = candidate_data_roots(args.data_root)

    payload = {
        "requested_data_root": args.data_root,
        "candidate_roots": [
            {"path": str(root), "exists": root.exists()} for root in roots_for_print
        ],
        "found_scene_count": len(scenes),
        "scenes": [scene_info_to_dict(scene) for scene in scenes],
        "recommended_scenes": [
            scene.scene_name for scene in scenes if scene.image_count > 0
        ][:3],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    log(f"Requested data root: {args.data_root}")
    log("Candidate roots:")
    for root in roots_for_print:
        status = "exists" if root.exists() else "missing"
        log(f"  - {root} [{status}]")

    existing = [root for root in roots if root.exists()]
    if existing:
        log("Roots searched:")
        for root in existing:
            log(f"  - {root}")

    log(f"Found {len(scenes)} scene directories.")
    for scene in scenes[: args.limit]:
        rgb = scene.rgb_dir or "(no extracted RGB dir)"
        sens = scene.sens_path or "(no .sens)"
        log(
            f"{scene.scene_name}: images={scene.image_count}, rgb={rgb}, "
            f"sens={sens}, scene_dir={scene.scene_dir}"
        )
        for note in scene.notes:
            log(f"    note: {note}")

    if len(scenes) > args.limit:
        log(f"... skipped {len(scenes) - args.limit} more scenes; use --limit to print more.")

    recommended = [scene for scene in scenes if scene.image_count > 0][:3]
    if recommended:
        log("Recommended scenes for probe:")
        for scene in recommended:
            log(f"  - {scene.scene_name} ({scene.image_count} frames)")
    else:
        log("No scene with RGB images or readable .sens frames was found.")


if __name__ == "__main__":
    main()
