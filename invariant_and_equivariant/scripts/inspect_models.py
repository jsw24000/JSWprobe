#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
SRC = EXP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from invariant_equivariant.models.dinov2_extractor import DINOv2Extractor
from invariant_equivariant.models.vggt_extractor import VGGTExtractor
from invariant_equivariant.utils import ensure_dir, load_config, torch_info, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env_dir = ensure_dir(Path(config["_output_dir"]) / "environment")
    inventory = {"torch": torch_info(), "models": []}
    if config["models"]["dinov2"]["enabled"]:
        inventory["models"].append(DINOv2Extractor(config).inventory())
    if config["models"]["vggt"]["enabled"]:
        inventory["models"].append(VGGTExtractor(config).inventory())
    write_json(env_dir / "model_inventory.json", inventory)
    write_json(env_dir / "torch_info.json", inventory["torch"])
    print(inventory)


if __name__ == "__main__":
    main()

