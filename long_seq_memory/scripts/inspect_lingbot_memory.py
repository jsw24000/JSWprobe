#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, load_config, sha256_file, write_json


KEY_FILES = [
    "benchmark/methods/lingbot_map.py",
    "benchmark/configs/methods/lingbot_map.yaml",
    "demo.py",
    "lingbot_map/models/gct_stream.py",
    "lingbot_map/models/gct_stream_window.py",
    "lingbot_map/aggregator/base.py",
    "lingbot_map/aggregator/stream.py",
    "lingbot_map/layers/block.py",
    "lingbot_map/layers/attention.py",
    "lingbot_map/layers/flashinfer_cache.py",
    "lingbot_map/heads/camera_head.py",
    "lingbot_map/utils/load_fn.py",
]


def find_patterns(path: Path, patterns: list[str]) -> dict[str, list[int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out = {}
    for pattern in patterns:
        rx = re.compile(pattern)
        out[pattern] = [i + 1 for i, line in enumerate(lines) if rx.search(line)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "configs" / "dataset_keble02.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    lingbot_root = Path(cfg["lingbot_root"])
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "static_audit")

    patterns = [
        r"class .*GCTStream",
        r"class .*AggregatorStream",
        r"class .*FlashInfer",
        r"class .*SDPA",
        r"kv_cache",
        r"keyframe_interval",
        r"_skip_append",
        r"scale_token",
        r"register_token",
        r"camera_token",
        r"scaled_dot_product_attention",
        r"prepare_qkv",
        r"append_frame",
        r"evict_frames",
    ]
    files = {}
    for rel in KEY_FILES:
        path = lingbot_root / rel
        files[rel] = {
            "exists": path.exists(),
            "pattern_lines": find_patterns(path, patterns) if path.exists() else {},
        }

    checkpoint = Path(cfg["checkpoint"])
    audit = {
        "lingbot_root": str(lingbot_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
        "files": files,
        "static_findings": {
            "token_order": "camera, register[0:4], scale, image patches",
            "num_special_tokens": 6,
            "default_patch_window_frames": cfg["model_defaults"]["kv_cache_sliding_window"],
            "default_scale_frames": cfg["model_defaults"]["num_scale_frames"],
            "flashinfer_special_stream": "append-only for every appended frame",
            "aggregator_keyframe_skip_append_effective": False,
            "attention_weights_directly_returned": False,
        },
    }
    write_json(out_dir / "lingbot_memory_static_audit.json", audit)
    print(f"Wrote {out_dir / 'lingbot_memory_static_audit.json'}")


if __name__ == "__main__":
    main()
