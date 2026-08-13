#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_objectness_probe.dataset import inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect memory-write probe dataset integrity.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "dataset_check")
    parser.add_argument("--min_probe_mask_area", type=float, default=0.0)
    parser.add_argument("--min_probe_visible_ratio", type=float, default=0.0)
    args = parser.parse_args()

    report = inspect_dataset(
        args.data_root,
        args.output_dir,
        min_probe_mask_area=args.min_probe_mask_area,
        min_probe_visible_ratio=args.min_probe_visible_ratio,
    )
    print(f"Dataset root: {report['data_root']}")
    print(f"Samples: {report['num_samples']}  Base scenes: {report['num_base_scenes']}")
    print(f"Variant counts: {report['variant_counts']}")
    print(f"Issues: {report['num_issues']}")
    print(f"Wrote: {args.output_dir / 'dataset_report.json'}")
    print(f"Wrote: {args.output_dir / 'validated_manifest.jsonl'}")


if __name__ == "__main__":
    main()
