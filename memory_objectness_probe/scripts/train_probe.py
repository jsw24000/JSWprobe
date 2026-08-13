#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from memory_objectness_probe.metrics import classification_metrics
from memory_objectness_probe.probe_dataset import TASK_NUM_CLASSES, ProbeFeatureDataset, split_sample_ids
from memory_objectness_probe.probe_models import build_probe
from memory_objectness_probe.utils import ensure_dir, normalize_layer_name, parse_optional_int, read_json, set_seed, write_json


def fit_transform_train_only(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    *,
    pca_dim: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std if x_val.numel() else x_val
    x_test = (x_test - mean) / std if x_test.numel() else x_test

    transform: Dict[str, Any] = {
        "standardization": {"mean": mean.cpu(), "std": std.cpu()},
        "pca_dim": pca_dim,
        "pca_components": None,
    }
    if pca_dim is not None:
        k = min(int(pca_dim), x_train.shape[0], x_train.shape[1])
        x_centered = x_train - x_train.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
        components = vh[:k].contiguous()
        x_train = x_train @ components.T
        x_val = x_val @ components.T if x_val.numel() else x_val
        x_test = x_test @ components.T if x_test.numel() else x_test
        transform["pca_dim"] = k
        transform["pca_components"] = components.cpu()
    return x_train, x_val, x_test, transform


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def eval_model(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, task: str, device: torch.device) -> Dict[str, Any]:
    if x.numel() == 0:
        return {"empty": True}
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=-1).cpu()
        pred = probs.argmax(dim=-1).numpy().tolist()
    scores = probs[:, 1].numpy().tolist() if probs.shape[1] == 2 else None
    return classification_metrics(y.numpy().tolist(), pred, TASK_NUM_CLASSES[task], scores_for_positive=scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear or shallow MLP memory-objectness probe.")
    parser.add_argument("--feature_manifest", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, required=True)
    parser.add_argument("--task", choices=["state5", "presence", "position4"], default="state5")
    parser.add_argument("--layer", required=True)
    parser.add_argument("--feature_mode", choices=["all6_flatten", "all6_mean", "single_slot", "slot_group"], default="all6_mean")
    parser.add_argument("--slot_index", type=int, default=None)
    parser.add_argument("--slot_group", type=str, default=None)
    parser.add_argument("--model", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--pca_dim", default="none")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--limit_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    split = read_json(args.split_file)
    layer = normalize_layer_name(args.layer)
    pca_dim = parse_optional_int(args.pca_dim)

    ds_train = ProbeFeatureDataset(
        args.feature_manifest,
        split_sample_ids(split, "train"),
        task=args.task,
        layer=layer,
        feature_mode=args.feature_mode,
        slot_index=args.slot_index,
        slot_group=args.slot_group,
    )
    ds_val = ProbeFeatureDataset(
        args.feature_manifest,
        split_sample_ids(split, "val"),
        task=args.task,
        layer=layer,
        feature_mode=args.feature_mode,
        slot_index=args.slot_index,
        slot_group=args.slot_group,
    )
    ds_test = ProbeFeatureDataset(
        args.feature_manifest,
        split_sample_ids(split, "test"),
        task=args.task,
        layer=layer,
        feature_mode=args.feature_mode,
        slot_index=args.slot_index,
        slot_group=args.slot_group,
    )

    x_train, y_train, train_ids = ds_train.stack()
    x_val, y_val, val_ids = ds_val.stack()
    x_test, y_test, test_ids = ds_test.stack()
    if args.limit_train_samples is not None and len(x_train) > args.limit_train_samples:
        x_train = x_train[: args.limit_train_samples]
        y_train = y_train[: args.limit_train_samples]
        train_ids = train_ids[: args.limit_train_samples]
    if len(x_train) == 0:
        raise SystemExit("No train examples after filtering. Check split, task, and layer.")
    if not torch.isfinite(x_train).all():
        raise SystemExit("Train features contain non-finite values.")

    x_train, x_val, x_test, transform = fit_transform_train_only(x_train, x_val, x_test, pca_dim=pca_dim)
    input_dim = int(x_train.shape[1])
    num_classes = TASK_NUM_CLASSES[args.task]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_probe(args.model, input_dim, num_classes, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_weights = None
    if args.task == "presence":
        counts = torch.bincount(y_train, minlength=num_classes).float()
        class_weights = (counts.sum() / counts.clamp_min(1.0)).to(device)
        class_weights = class_weights / class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)

    best_score = -1.0
    best_epoch = -1
    best_path = output_dir / "best.pt"
    curves: List[Dict[str, Any]] = []
    patience_left = args.early_stop_patience
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise SystemExit(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
            total += int(yb.numel())

        train_metrics = eval_model(model, x_train, y_train, args.task, device)
        val_metrics = eval_model(model, x_val, y_val, args.task, device)
        monitor_metrics = val_metrics if not val_metrics.get("empty") else train_metrics
        score = float(monitor_metrics.get("balanced_accuracy", monitor_metrics.get("accuracy", 0.0)))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": train_metrics.get("accuracy"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
            "monitor_score": score,
        }
        curves.append(row)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_left = args.early_stop_patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "transform": transform,
                    "args": vars(args),
                    "layer": layer,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    train_metrics = eval_model(model, x_train, y_train, args.task, device)
    val_metrics = eval_model(model, x_val, y_val, args.task, device)
    test_metrics = eval_model(model, x_test, y_test, args.task, device)

    metrics = {
        "task": args.task,
        "layer": layer,
        "feature_mode": args.feature_mode,
        "slot_index": args.slot_index,
        "slot_group": args.slot_group,
        "model": args.model,
        "pca_dim": transform["pca_dim"],
        "input_dim": input_dim,
        "num_classes": num_classes,
        "num_train": int(len(x_train)),
        "num_val": int(len(x_val)),
        "num_test": int(len(x_test)),
        "best_epoch": best_epoch,
        "best_monitor_score": best_score,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "train_sample_ids": train_ids,
        "val_sample_ids": val_ids,
        "test_sample_ids": test_ids,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "confusion_matrix.json", {"val": val_metrics.get("confusion_matrix"), "test": test_metrics.get("confusion_matrix")})
    with (output_dir / "curves.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        writer.writeheader()
        writer.writerows(curves)

    print(f"Train/val/test examples: {len(x_train)}/{len(x_val)}/{len(x_test)}")
    print(f"Best epoch: {best_epoch}, monitor score: {best_score:.4f}")
    print(f"Wrote: {best_path}")
    print(f"Wrote: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
