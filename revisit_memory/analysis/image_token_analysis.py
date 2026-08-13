#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.io import load_config, output_root, write_csv, write_json  # noqa: E402
from revisit_memory.utils.visualization import resize_mask_to_grid, save_heatmap  # noqa: E402


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("image_token_analysis.py requires torch to load feature .pt files.") from exc


def load_features(cfg: dict[str, Any], condition: str):
    torch = require_torch()
    path = output_root(cfg) / "features" / condition / "features.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return torch.load(path, map_location="cpu")


def load_mask_for_frame(cfg: dict[str, Any], condition: str, frame_id: int, grid_hw: tuple[int, int]) -> np.ndarray:
    recon_dir = output_root(cfg) / "reconstruction" / condition
    with np.load(recon_dir / "inputs.npz", allow_pickle=True) as inp_npz:
        frame_ids = inp_npz["frame_ids"].astype(int).tolist()
        idx = frame_ids.index(frame_id)
        mask = inp_npz["counterfactual_target_mask"][idx]
    return resize_mask_to_grid(mask, grid_hw)


def tensor_for(payload: dict[str, Any], group: str, layer: int, frame_id: int):
    frame_ids = list(payload["frame_ids"])
    idx = frame_ids.index(frame_id)
    return payload[group][str(layer)][idx].float()


def l2_map(diff) -> np.ndarray:
    torch = require_torch()
    return torch.linalg.norm(diff, dim=-1).numpy()


def cosine_map(a, b, min_norm: float) -> tuple[np.ndarray, float]:
    torch = require_torch()
    na = torch.linalg.norm(a, dim=-1)
    nb = torch.linalg.norm(b, dim=-1)
    valid = (na >= min_norm) & (nb >= min_norm)
    cos = torch.full_like(na, float("nan"))
    cos[valid] = torch.sum(a[valid] * b[valid], dim=-1) / (na[valid] * nb[valid]).clamp_min(1e-12)
    return cos.numpy(), float(valid.float().mean().item())


def region_stats(metric: np.ndarray, mask: np.ndarray, eps: float) -> dict[str, float]:
    metric = np.asarray(metric, dtype=np.float64)
    valid = np.isfinite(metric)
    inside = mask & valid
    outside = (~mask) & valid
    inside_vals = metric[inside]
    outside_vals = metric[outside]
    inside_mean = float(np.mean(inside_vals)) if inside_vals.size else float("nan")
    outside_mean = float(np.mean(outside_vals)) if outside_vals.size else float("nan")
    ratio = inside_mean / (outside_mean + eps) if np.isfinite(inside_mean) and np.isfinite(outside_mean) else float("nan")
    return {
        "mask_inside_patch_count": int(inside.sum()),
        "mask_outside_patch_count": int(outside.sum()),
        "mask_inside_mean": inside_mean,
        "mask_inside_median": float(np.median(inside_vals)) if inside_vals.size else float("nan"),
        "mask_inside_p95": float(np.percentile(inside_vals, 95)) if inside_vals.size else float("nan"),
        "mask_inside_max": float(np.max(inside_vals)) if inside_vals.size else float("nan"),
        "mask_outside_mean": outside_mean,
        "mask_outside_median": float(np.median(outside_vals)) if outside_vals.size else float("nan"),
        "mask_outside_p95": float(np.percentile(outside_vals, 95)) if outside_vals.size else float("nan"),
        "mask_outside_max": float(np.max(outside_vals)) if outside_vals.size else float("nan"),
        "localization_ratio": float(ratio),
        "valid_patch_fraction": float(valid.mean()),
    }


def sampled_patch_indices(mask_flat: np.ndarray, valid_flat: np.ndarray, outside_budget: int, seed: int) -> np.ndarray:
    inside = np.flatnonzero(mask_flat & valid_flat)
    outside = np.flatnonzero((~mask_flat) & valid_flat)
    if outside_budget > 0 and outside.size > outside_budget:
        rng = np.random.default_rng(seed)
        outside = np.sort(rng.choice(outside, size=outside_budget, replace=False))
    return np.concatenate([inside, outside], axis=0)


def append_patch_samples(
    rows: list[dict[str, Any]],
    metric_values: np.ndarray,
    mask_flat: np.ndarray,
    grid_hw: tuple[int, int],
    *,
    layer: int,
    frame_id: int,
    block_type: str,
    metric_name: str,
    outside_budget: int,
    seed: int,
) -> None:
    flat = np.asarray(metric_values, dtype=np.float32).reshape(-1)
    valid = np.isfinite(flat)
    selected = sampled_patch_indices(mask_flat, valid, outside_budget, seed)
    grid_w = int(grid_hw[1])
    for patch_idx in selected:
        rows.append(
            {
                "metric": metric_name,
                "layer": layer,
                "block_type": block_type,
                "frame": frame_id,
                "patch_index": int(patch_idx),
                "patch_y": int(patch_idx // grid_w),
                "patch_x": int(patch_idx % grid_w),
                "inside_target_mask": bool(mask_flat[patch_idx]),
                "value": float(flat[patch_idx]),
            }
        )


def run(cfg: dict[str, Any], make_figures: bool = True) -> None:
    torch = require_torch()
    payloads = {condition: load_features(cfg, condition) for condition in cfg["conditions"]}
    eval_start, eval_end = cfg["frames"]["second_loop_eval"]
    requested_frames = list(range(int(eval_start), int(eval_end) + 1))
    frame_ids = sorted(set(payloads["seen_then_removed"]["frame_ids"]).intersection(payloads["always_absent"]["frame_ids"]))
    frame_ids = [fid for fid in frame_ids if fid in requested_frames]
    layers = [int(x) for x in payloads["seen_then_removed"]["selected_layers"]]
    grid_hw = tuple(payloads["seen_then_removed"]["token_grid_hw"])
    eps = float(cfg["analysis"]["eps"])
    min_norm = float(cfg["features"]["cosine_min_norm"])

    out_dir = output_root(cfg) / "analysis" / "image_token_l2"
    dir_out = output_root(cfg) / "analysis" / "object_direction"
    out_dir.mkdir(parents=True, exist_ok=True)
    dir_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    patch_sample_rows: list[dict[str, Any]] = []
    maps_for_scale: list[dict[str, Any]] = []
    patch_sample_budget = int(cfg["analysis"].get("image_token_patch_samples_per_frame", 0))

    for layer in layers:
        for frame_id in frame_ids:
            mask = load_mask_for_frame(cfg, "seen_then_removed", frame_id, grid_hw)
            mask_flat = mask.reshape(-1)

            tensors = {}
            for condition in cfg["conditions"]:
                frame_block = tensor_for(payloads[condition], "frame_block", layer, frame_id)
                global_block = tensor_for(payloads[condition], "global_block", layer, frame_id)
                tensors[(condition, "frame_block")] = frame_block
                tensors[(condition, "global_block")] = global_block
                tensors[(condition, "global_update")] = global_block - frame_block

            for block_type in ["frame_block", "global_block"]:
                pair_specs = [
                    ("removed_absent_l2", "seen_then_removed", "always_absent"),
                    ("present_absent_l2", "always_present", "always_absent"),
                ]
                for metric_name, left_condition, right_condition in pair_specs:
                    diff = tensors[(left_condition, block_type)] - tensors[(right_condition, block_type)]
                    metric = l2_map(diff).reshape(grid_hw)
                    stats = region_stats(metric, mask, eps)
                    rows.append(
                        {
                            "metric": metric_name,
                            "comparison": f"{left_condition}_minus_{right_condition}",
                            "layer": layer,
                            "block_type": block_type,
                            "frame": frame_id,
                            **stats,
                        }
                    )
                    maps_for_scale.append(
                        {"layer": layer, "frame": frame_id, "name": f"{block_type}_{metric_name}", "values": metric}
                    )
                    append_patch_samples(
                        patch_sample_rows,
                        metric,
                        mask_flat,
                        grid_hw,
                        layer=layer,
                        frame_id=frame_id,
                        block_type=block_type,
                        metric_name=metric_name,
                        outside_budget=patch_sample_budget,
                        seed=int(cfg["project"]["random_seed"]) + layer * 1000 + frame_id + len(metric_name),
                    )

            update_diff = tensors[("seen_then_removed", "global_update")] - tensors[("always_absent", "global_update")]
            update_metric = l2_map(update_diff).reshape(grid_hw)
            rows.append(
                {
                    "metric": "global_update_delta_l2",
                    "layer": layer,
                    "block_type": "global_update",
                    "frame": frame_id,
                    **region_stats(update_metric, mask, eps),
                }
            )
            maps_for_scale.append({"layer": layer, "frame": frame_id, "name": "global_update_delta_l2", "values": update_metric})
            append_patch_samples(
                patch_sample_rows,
                update_metric,
                mask_flat,
                grid_hw,
                layer=layer,
                frame_id=frame_id,
                block_type="global_update",
                metric_name="global_update_delta_l2",
                outside_budget=patch_sample_budget,
                seed=int(cfg["project"]["random_seed"]) + layer * 1000 + frame_id + 17,
            )

            for block_type in ["frame_block", "global_block"]:
                v_object = tensors[("always_present", block_type)] - tensors[("always_absent", block_type)]
                v_memory = tensors[("seen_then_removed", block_type)] - tensors[("always_absent", block_type)]
                cos, valid_ratio = cosine_map(v_memory, v_object, min_norm)
                cos_grid = cos.reshape(grid_hw)
                direction_rows.append(
                    {
                        "metric": "cosine_alignment",
                        "layer": layer,
                        "block_type": block_type,
                        "frame": frame_id,
                        "valid_patch_fraction_after_norm_threshold": valid_ratio,
                        **region_stats(cos_grid, mask, eps),
                    }
                )
                maps_for_scale.append({"layer": layer, "frame": frame_id, "name": f"{block_type}_cosine_alignment", "values": cos_grid})

            v_object_update = tensors[("always_present", "global_update")] - tensors[("always_absent", "global_update")]
            v_memory_update = tensors[("seen_then_removed", "global_update")] - tensors[("always_absent", "global_update")]
            cos, valid_ratio = cosine_map(v_memory_update, v_object_update, min_norm)
            cos_grid = cos.reshape(grid_hw)
            direction_rows.append(
                {
                    "metric": "cosine_alignment",
                    "layer": layer,
                    "block_type": "global_update",
                    "frame": frame_id,
                    "valid_patch_fraction_after_norm_threshold": valid_ratio,
                    **region_stats(cos_grid, mask, eps),
                }
            )
            maps_for_scale.append({"layer": layer, "frame": frame_id, "name": "global_update_cosine_alignment", "values": cos_grid})

    write_csv(output_root(cfg) / "metrics" / "image_token_metrics.csv", rows)
    write_csv(output_root(cfg) / "metrics" / "object_direction_metrics.csv", direction_rows)
    write_csv(output_root(cfg) / "metrics" / "image_token_patch_samples.csv", patch_sample_rows)

    raw_payload = {
        "grid_hw": grid_hw,
        "frames": frame_ids,
        "layers": layers,
        "maps": {
            f"layer_{m['layer']:02d}/frame_{m['frame']:04d}/{m['name']}": torch.from_numpy(m["values"])
            for m in maps_for_scale
        },
    }
    torch.save(raw_payload, out_dir / "raw_metric_maps.pt")

    if make_figures:
        grouped: dict[tuple[int, str], list[np.ndarray]] = {}
        for m in maps_for_scale:
            grouped.setdefault((m["layer"], m["name"]), []).append(m["values"])
        scales = {}
        percentile = float(cfg["analysis"]["comparable_percentile"])
        for key, vals in grouped.items():
            finite_parts = [v[np.isfinite(v)].reshape(-1) for v in vals if np.isfinite(v).any()]
            cat = np.concatenate(finite_parts) if finite_parts else np.asarray([], dtype=np.float32)
            scales[key] = float(np.percentile(cat, percentile)) if cat.size else 1.0
        for m in maps_for_scale:
            target_dir = dir_out if "cosine" in m["name"] else out_dir
            layer_dir = target_dir / f"layer_{m['layer']:02d}" / m["name"]
            layer_dir.mkdir(parents=True, exist_ok=True)
            save_heatmap(layer_dir / f"frame_{m['frame']:04d}_per_frame.png", m["values"])
            vmax = 1.0 if "cosine" in m["name"] else scales[(m["layer"], m["name"])]
            vmin = -1.0 if "cosine" in m["name"] else 0.0
            save_heatmap(layer_dir / f"frame_{m['frame']:04d}_comparable.png", m["values"], vmin=vmin, vmax=vmax)

    write_json(
        out_dir / "summary.json",
        {
            "frames": frame_ids,
            "layers": layers,
            "grid_hw": list(grid_hw),
            "image_token_start": payloads["seen_then_removed"]["patch_start_idx"],
            "upsample_mode": cfg["features"]["upsample_mode"],
            "cosine_min_norm": min_norm,
            "patch_sample_csv": str(output_root(cfg) / "metrics" / "image_token_patch_samples.csv"),
            "patch_sample_policy": (
                "All finite target-mask patches are kept; up to "
                f"{patch_sample_budget} finite outside-mask patches are sampled per frame/layer/block metric."
            ),
        },
    )
    print(f"Wrote {output_root(cfg) / 'metrics' / 'image_token_metrics.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Image-token L2 and object-direction analysis.")
    parser.add_argument("--config", default="revisit_memory/configs/revisit_memory.yaml")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    make_figures = (not args.no_figures) and bool(cfg["analysis"].get("save_image_token_figures", True))
    run(cfg, make_figures=make_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
