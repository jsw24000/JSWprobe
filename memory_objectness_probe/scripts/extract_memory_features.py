#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_objectness_probe.dataset import load_samples
from memory_objectness_probe.feature_store import (
    feature_manifest_row,
    make_feature_payload,
    save_feature,
    summarize_feature_manifest,
    validate_feature_file,
    write_feature_manifests,
)
from memory_objectness_probe.lingbot_runner import LingbotRunner
from memory_objectness_probe.memory_hooks import DEFAULT_EXTRACT_LAYERS
from memory_objectness_probe.utils import ensure_dir, read_jsonl, set_seed


def parse_layers(values: List[str] | None) -> List[int]:
    if not values:
        return list(DEFAULT_EXTRACT_LAYERS)
    out: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if part.startswith("block"):
                part = part[5:]
            out.append(int(part))
    return out


def existing_rows(manifest_path: Path) -> Dict[str, dict]:
    if not manifest_path.exists():
        return {}
    rows = read_jsonl(manifest_path)
    return {str(row["sample_id"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Lingbot-map six-token memory features.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--lingbot_root", type=Path, default=Path("../lingbot-map"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--base_scene_ids", nargs="*", default=None)
    parser.add_argument("--max_base_scenes", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_reconstruction", choices=["none", "minimal", "full"], default="minimal")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=1)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--use_flashinfer", action="store_true", help="Use FlashInfer instead of SDPA fallback.")
    parser.add_argument("--flush_frames", type=int, default=0)
    parser.add_argument("--extract_layers", nargs="*", default=None)
    parser.add_argument(
        "--checkpoint_hash_bytes",
        type=int,
        default=0,
        help="Bytes used for checkpoint SHA256. 0 means full file; positive values compute a prefix hash for faster smoke runs.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_root = Path(args.output_root)
    feature_root = ensure_dir(output_root / "features")
    sample_dir = ensure_dir(feature_root / "samples")
    ensure_dir(output_root / "logs")
    manifest_path = feature_root / "feature_manifest.jsonl"

    lingbot_root = args.lingbot_root.resolve()
    checkpoint = args.checkpoint or (lingbot_root / "checkpoints" / "lingbot-map.pt")
    layers = parse_layers(args.extract_layers)

    samples = load_samples(args.data_root)
    if args.base_scene_ids:
        allowed = set(args.base_scene_ids)
        samples = [s for s in samples if s.base_scene_id in allowed]
    if args.max_base_scenes is not None:
        bases = []
        for sample in samples:
            if sample.base_scene_id not in bases:
                bases.append(sample.base_scene_id)
        allowed = set(bases[: args.max_base_scenes])
        samples = [s for s in samples if s.base_scene_id in allowed]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    rows_by_id = existing_rows(manifest_path) if args.resume else {}
    rows: List[dict] = [rows_by_id[k] for k in sorted(rows_by_id)]

    runner = LingbotRunner(
        lingbot_root=lingbot_root,
        checkpoint_path=checkpoint,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        camera_num_iterations=args.camera_num_iterations,
        use_sdpa=not args.use_flashinfer,
        checkpoint_hash_bytes=args.checkpoint_hash_bytes,
    )

    processed = 0
    for sample in samples:
        feature_path = sample_dir / f"{sample.sample_id}.pt"
        if sample.sample_id in rows_by_id and args.resume and not args.overwrite:
            valid = validate_feature_file(rows_by_id[sample.sample_id]["feature_path"])
            if valid.get("valid"):
                print(f"[skip] {sample.sample_id}")
                continue

        if feature_path.exists() and not args.overwrite:
            valid = validate_feature_file(feature_path)
            if valid.get("valid"):
                payload = None
                try:
                    import torch

                    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
                except Exception:
                    payload = None
                row = feature_manifest_row(sample=sample, feature_path=feature_path, status="success", payload=payload)
                rows_by_id[sample.sample_id] = row
                print(f"[skip-existing] {sample.sample_id}")
                continue

        print(f"[extract] {sample.sample_id}")
        try:
            result = runner.run_sample(
                sample,
                output_root=output_root,
                save_reconstruction=args.save_reconstruction,
                flush_frames=args.flush_frames,
                extract_layers=layers,
            )
            payload = make_feature_payload(
                sample=sample,
                memory_tokens=result["memory_tokens"],
                run_metadata=result["run_metadata"],
            )
            save_feature(feature_path, payload)
            row = feature_manifest_row(sample=sample, feature_path=feature_path, status="success", payload=payload)
            rows_by_id[sample.sample_id] = row
            processed += 1
        except Exception as exc:
            tb = traceback.format_exc()
            log_path = output_root / "logs" / f"{sample.sample_id}.log"
            log_path.write_text(tb, encoding="utf-8")
            row = feature_manifest_row(sample=sample, feature_path=feature_path, status="failed", error=repr(exc))
            rows_by_id[sample.sample_id] = row
            print(f"[failed] {sample.sample_id}: {exc}")

        rows = [rows_by_id[k] for k in sorted(rows_by_id)]
        write_feature_manifests(feature_root, rows)

    rows = [rows_by_id[k] for k in sorted(rows_by_id)]
    write_feature_manifests(feature_root, rows)
    summary = summarize_feature_manifest(feature_root, rows)
    print(f"Rows in feature manifest: {len(rows)}")
    print(f"Success: {summary['success_count']}  Failed: {summary['failure_count']}  Newly processed: {processed}")
    print(f"Wrote: {feature_root / 'feature_manifest.jsonl'}")
    print(f"Wrote: {feature_root / 'extraction_summary.json'}")


if __name__ == "__main__":
    main()
