#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.io import load_config, output_root, write_json  # noqa: E402


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("extract_features.py requires torch to read saved .pt feature files.") from exc


def summarize_condition(cfg, condition: str):
    torch = require_torch()
    feature_path = output_root(cfg) / "features" / condition / "features.pt"
    memory_path = output_root(cfg) / "memory_tokens" / condition / "special_tokens.pt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing {feature_path}. Run run_reconstruction.py first.")
    payload = torch.load(feature_path, map_location="cpu")
    memory = torch.load(memory_path, map_location="cpu") if memory_path.exists() else None
    return {
        "condition": condition,
        "feature_path": str(feature_path),
        "frame_ids": payload["frame_ids"],
        "selected_layers": payload["selected_layers"],
        "patch_start_idx": payload["patch_start_idx"],
        "token_grid_hw": payload["token_grid_hw"],
        "raw_shapes": payload["raw_shapes"],
        "feature_shapes": {
            group: {layer: list(tensor.shape) for layer, tensor in payload[group].items()}
            for group in ["frame_block", "global_block", "frame_special_tokens", "global_special_tokens"]
        },
        "memory_path": str(memory_path) if memory_path.exists() else None,
        "memory_shapes": {
            group: {layer: list(tensor.shape) for layer, tensor in memory[group].items()}
            for group in ["frame_special_tokens", "global_special_tokens"]
        }
        if memory is not None
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize saved streaming feature tensors.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    parser.add_argument("--condition", choices=["always_present", "always_absent", "seen_then_removed"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    conditions = [args.condition] if args.condition else cfg["conditions"]
    summary = {condition: summarize_condition(cfg, condition) for condition in conditions}
    out_path = output_root(cfg) / "features" / "feature_summary.json"
    write_json(out_path, summary)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

