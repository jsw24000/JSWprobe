#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.io import load_config, output_root, project_path, write_json  # noqa: E402


def source_findings(lingbot_root: Path) -> dict[str, Any]:
    files = {
        "demo": lingbot_root / "demo.py",
        "gct_stream": lingbot_root / "lingbot_map" / "models" / "gct_stream.py",
        "aggregator_stream": lingbot_root / "lingbot_map" / "aggregator" / "stream.py",
        "aggregator_base": lingbot_root / "lingbot_map" / "aggregator" / "base.py",
        "flashinfer_cache": lingbot_root / "lingbot_map" / "layers" / "flashinfer_cache.py",
    }
    snippets: dict[str, Any] = {}
    for name, path in files.items():
        text = path.read_text(encoding="utf-8")
        snippets[name] = {
            "path": str(path),
            "contains_inference_streaming": "inference_streaming" in text,
            "contains_keyframe_interval": "keyframe_interval" in text,
            "contains_kv_cache_sliding_window": "kv_cache_sliding_window" in text,
            "contains_scale_patch_pages": "scale_patch_pages" in text,
            "contains_live_window_patch_pages": "live_window_patch_pages" in text,
            "contains_all_special_pages": "all_special_pages" in text,
            "contains_frame_global_concat": "torch.cat([frame_intermediates" in text,
        }
    return {
        "files": snippets,
        "streaming_entry": "GCTStream.inference_streaming processes scale frames first, then frames one by one.",
        "feature_source": "AggregatorBase.forward returns selected outputs as concat(frame_intermediate, global_intermediate) in the last dimension.",
        "cache_source": "FlashInferKVCacheManager stores scale patch pages, live window patch pages, and append-only special pages.",
    }


def import_and_build(cfg: dict[str, Any]) -> dict[str, Any]:
    lingbot_root = project_path(cfg["paths"]["lingbot_root"])
    if str(lingbot_root) not in sys.path:
        sys.path.insert(0, str(lingbot_root))

    if importlib.util.find_spec("torch") is None:
        return {"built": False, "reason": "torch is not importable in this Python environment"}

    import torch
    from lingbot_map.models.gct_stream import GCTStream

    model_cfg = cfg["model"]
    ckpt_path = project_path(cfg["paths"]["model_path"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    checkpoint_has_point_head = any(key.startswith("point_head.") for key in state_dict)
    model = GCTStream(
        img_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        pretrained_path="",
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_local_point=False,
        enable_3d_rope=bool(model_cfg["enable_3d_rope"]),
        max_frame_num=int(model_cfg["max_frame_num"]),
        kv_cache_sliding_window=int(model_cfg["local_window_size"]),
        kv_cache_scale_frames=int(model_cfg["anchor_frame_count"]),
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=bool(model_cfg.get("use_sdpa", False)),
        camera_num_iterations=int(model_cfg["camera_num_iterations"]),
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    info: dict[str, Any] = {
        "built": True,
        "torch_version": torch.__version__,
        "checkpoint": str(ckpt_path),
        "checkpoint_has_point_head": checkpoint_has_point_head,
        "dinov2_preinit_loaded_before_checkpoint": False,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "model_class": type(model).__name__,
        "aggregator_class": type(model.aggregator).__name__,
        "patch_start_idx": int(model.aggregator.patch_start_idx),
        "num_special_tokens": int(model.aggregator.num_special_tokens),
        "num_register_tokens": int(model.aggregator.num_register_tokens),
        "depth": int(model.aggregator.depth),
        "aa_order": list(model.aggregator.aa_order),
        "frame_block_count": len(model.aggregator.frame_blocks),
        "global_block_count": len(model.aggregator.global_blocks),
        "selected_layers_requested": cfg["features"]["selected_layers"],
        "selected_layers_available": [
            idx for idx in cfg["features"]["selected_layers"] if idx < len(model.aggregator.frame_blocks)
        ],
        "frame_block_paths": [f"aggregator.frame_blocks.{idx}" for idx in range(len(model.aggregator.frame_blocks))],
        "global_block_paths": [f"aggregator.global_blocks.{idx}" for idx in range(len(model.aggregator.global_blocks))],
        "camera_head_class": type(model.camera_head).__name__ if model.camera_head is not None else None,
        "point_head_class": type(model.point_head).__name__ if model.point_head is not None else None,
        "depth_head_class": type(model.depth_head).__name__ if model.depth_head is not None else None,
        "output_heads_enabled": {
            "camera_head": model.camera_head is not None,
            "depth_head": model.depth_head is not None,
            "point_head": model.point_head is not None,
            "local_point_head": model.local_point_head is not None,
        },
        "token_layout": {
            "camera_token": 0,
            "register_tokens": [1, 4],
            "scale_token": 5,
            "image_tokens": [int(model.aggregator.patch_start_idx), "end"],
        },
    }
    del model
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LingBot-Map modules used by revisit-memory.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    lingbot_root = project_path(cfg["paths"]["lingbot_root"])
    report = {
        "lingbot_root": str(lingbot_root),
        "source_findings": source_findings(lingbot_root),
        "build_inspection": None,
    }
    try:
        report["build_inspection"] = import_and_build(cfg)
    except Exception as exc:
        report["build_inspection"] = {"built": False, "reason": repr(exc)}

    out_path = output_root(cfg) / "validation" / "lingbot_module_inspection.json"
    write_json(out_path, report)
    print(json.dumps(report["build_inspection"], indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
