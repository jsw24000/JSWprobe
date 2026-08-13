#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from memory_objectness_probe.metrics import classification_metrics
from memory_objectness_probe.probe_dataset import TASK_NUM_CLASSES, ProbeFeatureDataset, split_sample_ids
from memory_objectness_probe.probe_models import build_probe
from memory_objectness_probe.utils import normalize_layer_name, read_json, write_json
from scripts.train_probe import eval_model, fit_transform_train_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved probe checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature_manifest", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    split = read_json(args.split_file)
    layer = normalize_layer_name(ckpt.get("layer", train_args["layer"]))
    ds = ProbeFeatureDataset(
        args.feature_manifest,
        split_sample_ids(split, args.split),
        task=train_args["task"],
        layer=layer,
        feature_mode=train_args["feature_mode"],
        slot_index=train_args.get("slot_index"),
        slot_group=train_args.get("slot_group"),
    )
    x, y, sample_ids = ds.stack()
    transform = ckpt["transform"]
    mean = transform["standardization"]["mean"]
    std = transform["standardization"]["std"]
    x = (x - mean) / std
    components = transform.get("pca_components")
    if components is not None:
        x = x @ components.T
    model = build_probe(
        train_args["model"],
        ckpt["input_dim"],
        ckpt["num_classes"],
        hidden_dim=train_args.get("hidden_dim", 256),
        dropout=train_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    metrics = eval_model(model, x, y, train_args["task"], device)
    metrics["sample_ids"] = sample_ids
    write_json(args.output_file, metrics)
    print(f"Evaluated {len(sample_ids)} samples from split={args.split}")
    print(f"Wrote: {args.output_file}")


if __name__ == "__main__":
    main()
