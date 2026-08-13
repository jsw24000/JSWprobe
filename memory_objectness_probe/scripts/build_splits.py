#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_objectness_probe.utils import EXPECTED_VARIANTS, ensure_dir, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build base-scene grouped probe splits.")
    parser.add_argument("--feature_manifest", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, default=PROJECT_ROOT / "outputs" / "splits" / "scene_split_seed0.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_scenes", type=int, default=20)
    parser.add_argument("--val_scenes", type=int, default=5)
    parser.add_argument("--test_scenes", type=int, default=5)
    parser.add_argument("--allow_small", action="store_true", help="Allow requested split sizes to exceed available scenes.")
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.feature_manifest) if row.get("extraction_status") == "success"]
    by_base: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_base.setdefault(str(row["base_scene_id"]), []).append(row)

    complete_bases = [
        base
        for base, group in by_base.items()
        if sorted(str(row["variant_id"]) for row in group) == sorted(EXPECTED_VARIANTS)
    ]
    complete_bases = sorted(complete_bases)
    rng = random.Random(args.seed)
    rng.shuffle(complete_bases)

    requested = args.train_scenes + args.val_scenes + args.test_scenes
    if requested > len(complete_bases) and not args.allow_small:
        raise SystemExit(
            f"Requested {requested} scenes but only {len(complete_bases)} complete base scenes are available. "
            "Use --allow_small for smoke splits or extract more features."
        )

    train_n = min(args.train_scenes, len(complete_bases))
    val_n = min(args.val_scenes, max(0, len(complete_bases) - train_n))
    test_n = min(args.test_scenes, max(0, len(complete_bases) - train_n - val_n))
    train_bases = sorted(complete_bases[:train_n])
    val_bases = sorted(complete_bases[train_n : train_n + val_n])
    test_bases = sorted(complete_bases[train_n + val_n : train_n + val_n + test_n])

    def rows_for(bases: List[str]) -> Dict[str, Any]:
        split_rows = []
        for base in bases:
            split_rows.extend(sorted(by_base[base], key=lambda r: int(r["target_position_id"])))
        return {
            "base_scene_ids": bases,
            "sample_ids": [str(row["sample_id"]) for row in split_rows],
        }

    split = {
        "seed": args.seed,
        "source_feature_manifest": str(args.feature_manifest.resolve()),
        "num_complete_base_scenes": len(complete_bases),
        "train": rows_for(train_bases),
        "val": rows_for(val_bases),
        "test": rows_for(test_bases),
    }
    ensure_dir(args.output_file.parent)
    write_json(args.output_file, split)
    print(f"Complete base scenes: {len(complete_bases)}")
    print(f"train/val/test scenes: {len(train_bases)}/{len(val_bases)}/{len(test_bases)}")
    print(f"Wrote: {args.output_file}")


if __name__ == "__main__":
    main()
