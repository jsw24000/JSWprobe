#!/usr/bin/env python3
"""Public Lingbot-map token extraction/adaptation entry point.

This script is intentionally a thin wrapper. The actual model run follows the
official Lingbot-map demo path in ``run_lingbot_official_tokens.py``; this entry
then adapts the saved files into ``TokenBundle`` format for controlled
experiments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_FOLDER = (
    PROJECT_DIR
    / "data"
    / "controlled_sequences"
    / "scene0000_00_lingbot_stream512_stride4"
    / "images"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "tokens"
    / "scene0000_00_lingbot_stream512_stride4"
    / "lingbot_official"
)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from adapters.lingbot_adapter import LingbotAdapter


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        config = yaml.safe_load(text) or {}
    except ImportError:
        config = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            config[key.strip()] = _parse_scalar(value)

    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def parse_layers(value: str) -> list[int] | None:
    value = value.strip()
    if value.lower() in {"all", "*"}:
        return None
    layers = [int(part) for part in re.split(r"[, ]+", value) if part]
    if not layers:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers or 'all'")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/adapt Lingbot-map official intermediate token extraction."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "model_lingbot.yaml")
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--condition-id", default="lingbot_official")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("4,11,17,23"))
    parser.add_argument(
        "--token-slice",
        choices=["patch", "special", "all", "camera", "register", "scale"],
        default="patch",
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument(
        "--use-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SDPA attention. Recommended on this machine.",
    )
    parser.add_argument(
        "--adapt-only",
        action="store_true",
        help="Do not run the model; adapt an existing output-dir metadata.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the extraction command without running Lingbot-map.",
    )
    return parser.parse_args()


def save_bundle_index(bundle: Any, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "token_bundle.json"
    path.write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def bundle_summary(bundle: Any, bundle_path: Path | None = None) -> dict[str, Any]:
    return {
        "status": bundle.metadata.get("status", "ok"),
        "model_name": bundle.model_name,
        "sequence_id": bundle.sequence_id,
        "condition_id": bundle.condition_id,
        "frame_count": len(bundle.frame_paths),
        "layers": {
            label: {
                "shape": values.get("shape"),
                "token_files": {
                    key: value for key, value in values.items() if key.endswith("_tokens")
                },
            }
            for label, values in bundle.layer_tokens.items()
        },
        "token_bundle": str(bundle_path) if bundle_path else None,
        "metadata": bundle.metadata.get("source_metadata"),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    adapter = LingbotAdapter(config)

    if args.dry_run:
        command = adapter.build_token_command(
            args.image_folder,
            args.output_dir,
            layers=args.layers,
            token_slice=args.token_slice,
            dtype=args.dtype,
            device=args.device,
            use_sdpa=args.use_sdpa,
            num_scale_frames=args.num_scale_frames,
            camera_num_iterations=args.camera_num_iterations,
            keyframe_interval=args.keyframe_interval,
            kv_cache_sliding_window=args.kv_cache_sliding_window,
            max_frame_num=args.max_frame_num,
        )
        print(json.dumps({"status": "dry_run", "command": command}, ensure_ascii=False, indent=2))
        return 0

    if args.adapt_only:
        bundle = adapter.load_token_bundle(
            args.output_dir,
            condition_id=args.condition_id,
            sequence_id=args.sequence_id,
        )
    else:
        bundle = adapter.extract_tokens(
            args.image_folder,
            condition_id=args.condition_id,
            sequence_id=args.sequence_id,
            output_dir=args.output_dir,
            layers=args.layers,
            token_slice=args.token_slice,
            dtype=args.dtype,
            device=args.device,
            use_sdpa=args.use_sdpa,
            num_scale_frames=args.num_scale_frames,
            camera_num_iterations=args.camera_num_iterations,
            keyframe_interval=args.keyframe_interval,
            kv_cache_sliding_window=args.kv_cache_sliding_window,
            max_frame_num=args.max_frame_num,
        )

    bundle_path = save_bundle_index(bundle, args.output_dir.resolve())
    print(json.dumps(bundle_summary(bundle, bundle_path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
