#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revisit_memory.utils.depth_scale import resize_bool_stack, resize_float_stack  # noqa: E402
from revisit_memory.utils.io import load_config, output_root, project_path, write_csv, write_json, write_text  # noqa: E402
from revisit_memory.utils.visualization import normalize_for_uint8, overlay_mask_outline, resize_mask_to_grid  # noqa: E402


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("analyze_memory_transplant.py requires torch.") from exc


def load_features(root: Path, variant: str) -> dict[str, Any]:
    torch = require_torch()
    return torch.load(root / "features" / variant / "features.pt", map_location="cpu")


def load_predictions(root: Path, variant: str) -> dict[str, np.ndarray]:
    with np.load(root / "reconstruction" / variant / "predictions.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_inputs(root: Path, variant: str) -> dict[str, np.ndarray]:
    with np.load(root / "reconstruction" / variant / "inputs.npz", allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def frame_index(payload: dict[str, Any], frame_id: int) -> int:
    return list(payload["frame_ids"]).index(int(frame_id))


def feature_tensor(payload: dict[str, Any], variant: str, group: str, layer: int, frame_id: int):
    idx = frame_index(payload[variant], frame_id)
    return payload[variant][group][str(layer)][idx].float()


def l2_per_patch(diff) -> np.ndarray:
    torch = require_torch()
    return torch.linalg.norm(diff, dim=-1).numpy()


def cosine_distance(a, b, eps: float) -> float:
    torch = require_torch()
    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    denom = (torch.linalg.norm(aa) * torch.linalg.norm(bb)).clamp_min(eps)
    return float((1.0 - torch.sum(aa * bb) / denom).item())


def cosine_similarity(a, b, eps: float) -> float:
    torch = require_torch()
    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    denom = (torch.linalg.norm(aa) * torch.linalg.norm(bb)).clamp_min(eps)
    return float((torch.sum(aa * bb) / denom).item())


def patch_cosine_similarity(a, b, eps: float) -> np.ndarray:
    torch = require_torch()
    aa = a.float()
    bb = b.float()
    denom = (torch.linalg.norm(aa, dim=-1) * torch.linalg.norm(bb, dim=-1)).clamp_min(eps)
    return (torch.sum(aa * bb, dim=-1) / denom).detach().cpu().numpy()


def relative_l2(a, b, eps: float) -> float:
    torch = require_torch()
    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    denom = ((torch.linalg.norm(aa) + torch.linalg.norm(bb)) / 2.0).clamp_min(eps)
    return float((torch.linalg.norm(aa - bb) / denom).item())


def vector_norm(x) -> float:
    torch = require_torch()
    return float(torch.linalg.norm(x.reshape(-1).float()).item())


def region_stats(metric: np.ndarray, mask: np.ndarray, eps: float) -> dict[str, Any]:
    values = np.asarray(metric, dtype=np.float64)
    valid = np.isfinite(values)
    inside = mask & valid
    outside = (~mask) & valid
    inside_vals = values[inside]
    outside_vals = values[outside]
    inside_mean = float(np.mean(inside_vals)) if inside_vals.size else float("nan")
    outside_mean = float(np.mean(outside_vals)) if outside_vals.size else float("nan")
    yy, xx = np.indices(values.shape)
    weights = np.where(valid, np.maximum(values, 0.0), 0.0)
    if float(weights.sum()) > eps:
        cx = float((weights * xx).sum() / weights.sum())
        cy = float((weights * yy).sum() / weights.sum())
    else:
        cx = cy = float("nan")
    if mask.any() and math.isfinite(cx) and math.isfinite(cy):
        mx = float(xx[mask].mean())
        my = float(yy[mask].mean())
        centroid_distance = float(math.sqrt((cx - mx) ** 2 + (cy - my) ** 2))
    else:
        mx = my = centroid_distance = float("nan")
    return {
        "mask_inside_patch_count": int(inside.sum()),
        "mask_outside_patch_count": int(outside.sum()),
        "mask_inside_mean": inside_mean,
        "mask_inside_median": float(np.median(inside_vals)) if inside_vals.size else float("nan"),
        "mask_inside_p95": float(np.percentile(inside_vals, 95)) if inside_vals.size else float("nan"),
        "mask_outside_mean": outside_mean,
        "mask_outside_median": float(np.median(outside_vals)) if outside_vals.size else float("nan"),
        "mask_outside_p95": float(np.percentile(outside_vals, 95)) if outside_vals.size else float("nan"),
        "inside_outside_ratio": inside_mean / (outside_mean + eps)
        if math.isfinite(inside_mean) and math.isfinite(outside_mean)
        else float("nan"),
        "effect_centroid_patch_x": cx,
        "effect_centroid_patch_y": cy,
        "target_mask_center_patch_x": mx,
        "target_mask_center_patch_y": my,
        "effect_centroid_to_target_center_patches": centroid_distance,
    }


def masked_scalar_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    inside_vals = values[mask & valid]
    outside_vals = values[(~mask) & valid]
    inside_mean = float(np.mean(inside_vals)) if inside_vals.size else float("nan")
    outside_mean = float(np.mean(outside_vals)) if outside_vals.size else float("nan")
    return {
        "mask_inside_mean": inside_mean,
        "mask_inside_median": float(np.median(inside_vals)) if inside_vals.size else float("nan"),
        "mask_outside_mean": outside_mean,
        "mask_outside_median": float(np.median(outside_vals)) if outside_vals.size else float("nan"),
        "inside_minus_outside": inside_mean - outside_mean
        if math.isfinite(inside_mean) and math.isfinite(outside_mean)
        else float("nan"),
    }


def heatmap_rgb(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    gray = normalize_for_uint8(values, vmin, vmax)
    try:
        import matplotlib.pyplot as plt

        return (plt.get_cmap("magma")(gray / 255.0)[..., :3] * 255).astype(np.uint8)
    except Exception:
        return np.stack([gray, gray, gray], axis=-1)


def save_heatmap_with_mask(
    path: Path,
    values: np.ndarray,
    mask: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    scale: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = heatmap_rgb(values, vmin=vmin, vmax=vmax)
    if scale > 1:
        image = np.asarray(
            Image.fromarray(image, mode="RGB").resize(
                (image.shape[1] * scale, image.shape[0] * scale),
                resample=Image.Resampling.BILINEAR,
            )
        )
        mask = np.asarray(
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
                (image.shape[1], image.shape[0]),
                resample=Image.Resampling.NEAREST,
            )
        ) > 0
    overlay_mask_outline(image, mask).save(path)


def make_contact_sheet(path: Path, frames: list[int], effects: list[str], images: dict[tuple[int, str], Path]) -> None:
    thumbs = []
    for frame in frames:
        row = []
        for effect in effects:
            row.append(Image.open(images[(frame, effect)]).convert("RGB"))
        thumbs.append(row)
    if not thumbs:
        return
    w, h = thumbs[0][0].size
    pad = 24
    title_h = 26
    sheet = Image.new("RGB", (len(effects) * w, len(frames) * (h + title_h)), "white")
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(sheet)
    except Exception:
        draw = None
    for r, frame in enumerate(frames):
        for c, effect in enumerate(effects):
            x = c * w
            y = r * (h + title_h) + title_h
            sheet.paste(thumbs[r][c], (x, y))
            if draw is not None:
                draw.text((x + pad, y - title_h + 4), f"{frame} {effect}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def effect_specs(available: set[str]) -> dict[str, tuple[str | None, str | None]]:
    specs = {
        "total_RR_minus_AA": ("RR", "AA"),
        "memory_AR_minus_AA": ("AR", "AA"),
        "rolling_RA_minus_AA": ("RA", "AA"),
        "source_AS_minus_AA": ("AS", "AA"),
    }
    if "AN" in available:
        specs["negative_AN_minus_AA"] = ("AN", "AA")
    specs["interaction_RR_minus_RA_minus_AR_plus_AA"] = (None, None)
    return specs


def effect_tensor(payload: dict[str, Any], group: str, layer: int, frame_id: int, effect: str):
    if effect == "interaction_RR_minus_RA_minus_AR_plus_AA":
        return (
            feature_tensor(payload, "RR", group, layer, frame_id)
            - feature_tensor(payload, "RA", group, layer, frame_id)
            - feature_tensor(payload, "AR", group, layer, frame_id)
            + feature_tensor(payload, "AA", group, layer, frame_id)
        )
    left, right = effect_specs(set(payload))[effect]
    return feature_tensor(payload, left, group, layer, frame_id) - feature_tensor(payload, right, group, layer, frame_id)


def interaction_cosine_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Interaction Cosine Summary",
        "",
        "Comparison: `(RR - RA)` vs `(AR - AA)`.",
        "",
        "`RR - RA` is the effect of replacing A trajectory memory with R trajectory memory under the R query/current/local state. `AR - AA` is the same memory replacement under the A query/current/local state.",
        "",
        "Cosine similarity close to 1 means the two change vectors point in the same direction; close to 0 means nearly orthogonal; close to -1 means opposite directions.",
        "",
    ]
    for block_type in ["frame_block", "global_block"]:
        block_rows = [r for r in rows if r["block_type"] == block_type]
        if not block_rows:
            continue
        lines.extend(
            [
                f"## {block_type}",
                "",
                "| Layer | Vector cosine | Angle deg | Mean patch cosine | Mask-in patch cosine | Mask-out patch cosine |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(block_rows, key=lambda r: int(r["layer"])):
            lines.append(
                "| {layer} | {vec:.4f} | {angle:.2f} | {patch:.4f} | {inside:.4f} | {outside:.4f} |".format(
                    layer=int(row["layer"]),
                    vec=float(row["mean_vector_cosine_similarity"]),
                    angle=float(row["mean_vector_angle_degrees"]),
                    patch=float(row["mean_mean_patch_cosine_similarity"]),
                    inside=float(row["mean_mask_inside_mean_patch_cosine"]),
                    outside=float(row["mean_mask_outside_mean_patch_cosine"]),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def load_masks(inputs: dict[str, np.ndarray], frames: list[int], grid_hw: tuple[int, int]) -> dict[int, np.ndarray]:
    frame_ids = inputs["frame_ids"].astype(int).tolist()
    masks = {}
    for frame in frames:
        idx = frame_ids.index(int(frame))
        masks[frame] = resize_mask_to_grid(inputs["counterfactual_target_mask"][idx], grid_hw)
    return masks


def analyze_tokens(cfg: dict[str, Any], root: Path, payload: dict[str, Any], masks: dict[int, np.ndarray]) -> dict[str, Any]:
    torch = require_torch()
    eps = float(cfg["analysis"]["eps"])
    variants = set(payload)
    frames = list(payload["AA"]["frame_ids"])
    layers = [int(x) for x in payload["AA"]["selected_layers"]]
    grid_hw = tuple(int(x) for x in payload["AA"]["token_grid_hw"])
    specs = effect_specs(variants)
    token_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    fraction_rows: list[dict[str, Any]] = []
    interaction_cosine_rows: list[dict[str, Any]] = []
    interaction_cosine_aggregate_rows: list[dict[str, Any]] = []
    heatmap_store: dict[tuple[int, int, str, str], np.ndarray] = {}

    for layer in layers:
        for block_type in ["frame_block", "global_block"]:
            for frame in frames:
                per_effect_norms: dict[str, float] = {}
                for effect, pair in specs.items():
                    if effect.startswith("interaction") and not {"RR", "RA", "AR", "AA"}.issubset(variants):
                        continue
                    if pair[0] is not None and pair[0] not in variants:
                        continue
                    diff = effect_tensor(payload, block_type, layer, frame, effect)
                    metric = l2_per_patch(diff).reshape(grid_hw)
                    heatmap_store[(layer, frame, block_type, effect)] = metric
                    norm = vector_norm(diff)
                    per_effect_norms[effect] = norm
                    row = {
                        "effect": effect,
                        "layer": layer,
                        "block_type": block_type,
                        "frame": frame,
                        "effect_vector_l2": norm,
                        "mean_patch_l2": float(np.mean(metric)),
                        "median_patch_l2": float(np.median(metric)),
                        "p95_patch_l2": float(np.percentile(metric, 95)),
                        **region_stats(metric, masks[frame], eps),
                    }
                    if pair[0] is not None:
                        left = feature_tensor(payload, pair[0], block_type, layer, frame)
                        right = feature_tensor(payload, pair[1], block_type, layer, frame)
                        row["relative_l2"] = relative_l2(left, right, eps)
                        row["cosine_distance"] = cosine_distance(left, right, eps)
                    else:
                        total_norm = per_effect_norms.get("total_RR_minus_AA", float("nan"))
                        row["relative_l2"] = norm / (total_norm + eps) if math.isfinite(total_norm) else float("nan")
                        row["cosine_distance"] = float("nan")
                    token_rows.append(row)

                total = per_effect_norms.get("total_RR_minus_AA", float("nan"))
                memory = per_effect_norms.get("memory_AR_minus_AA", float("nan"))
                rolling = per_effect_norms.get("rolling_RA_minus_AA", float("nan"))
                source = per_effect_norms.get("source_AS_minus_AA", float("nan"))
                fraction_rows.append(
                    {
                        "layer": layer,
                        "block_type": block_type,
                        "frame": frame,
                        "total_effect_l2": total,
                        "memory_effect_l2": memory,
                        "rolling_effect_l2": rolling,
                        "source_effect_l2": source,
                        "memory_fraction": memory / (total + eps) if math.isfinite(memory) and math.isfinite(total) else float("nan"),
                        "rolling_fraction": rolling / (total + eps) if math.isfinite(rolling) and math.isfinite(total) else float("nan"),
                        "source_fraction_of_memory": source / (memory + eps)
                        if math.isfinite(source) and math.isfinite(memory)
                        else float("nan"),
                    }
                )

    if {"RR", "RA", "AR", "AA"}.issubset(variants):
        for layer in layers:
            for block_type in ["frame_block", "global_block"]:
                for frame in frames:
                    delta_memory_under_r_query = feature_tensor(payload, "RR", block_type, layer, frame) - feature_tensor(
                        payload, "RA", block_type, layer, frame
                    )
                    delta_memory_under_a_query = feature_tensor(payload, "AR", block_type, layer, frame) - feature_tensor(
                        payload, "AA", block_type, layer, frame
                    )
                    difference = delta_memory_under_r_query - delta_memory_under_a_query
                    vector_cos = cosine_similarity(delta_memory_under_r_query, delta_memory_under_a_query, eps)
                    vector_cos_clamped = max(-1.0, min(1.0, vector_cos))
                    patch_cos = patch_cosine_similarity(
                        delta_memory_under_r_query, delta_memory_under_a_query, eps
                    ).reshape(grid_hw)
                    valid = np.isfinite(patch_cos)
                    inside = masks[frame] & valid
                    outside = (~masks[frame]) & valid
                    inside_vals = patch_cos[inside]
                    outside_vals = patch_cos[outside]
                    inside_mean = float(np.mean(inside_vals)) if inside_vals.size else float("nan")
                    outside_mean = float(np.mean(outside_vals)) if outside_vals.size else float("nan")
                    interaction_cosine_rows.append(
                        {
                            "comparison": "RR_minus_RA_vs_AR_minus_AA",
                            "layer": layer,
                            "block_type": block_type,
                            "frame": frame,
                            "vector_cosine_similarity": vector_cos,
                            "vector_angle_degrees": float(math.degrees(math.acos(vector_cos_clamped))),
                            "rr_minus_ra_vector_l2": vector_norm(delta_memory_under_r_query),
                            "ar_minus_aa_vector_l2": vector_norm(delta_memory_under_a_query),
                            "difference_vector_l2": vector_norm(difference),
                            "mean_patch_l2_between_change_vectors": float(np.mean(l2_per_patch(difference))),
                            "mean_patch_cosine_similarity": float(np.mean(patch_cos[valid])) if valid.any() else float("nan"),
                            "median_patch_cosine_similarity": float(np.median(patch_cos[valid])) if valid.any() else float("nan"),
                            "p05_patch_cosine_similarity": float(np.percentile(patch_cos[valid], 5)) if valid.any() else float("nan"),
                            "p95_patch_cosine_similarity": float(np.percentile(patch_cos[valid], 95)) if valid.any() else float("nan"),
                            "mask_inside_patch_count": int(inside.sum()),
                            "mask_outside_patch_count": int(outside.sum()),
                            "mask_inside_mean_patch_cosine": inside_mean,
                            "mask_inside_median_patch_cosine": float(np.median(inside_vals)) if inside_vals.size else float("nan"),
                            "mask_outside_mean_patch_cosine": outside_mean,
                            "mask_outside_median_patch_cosine": float(np.median(outside_vals)) if outside_vals.size else float("nan"),
                            "mask_inside_minus_outside_patch_cosine": inside_mean - outside_mean
                            if math.isfinite(inside_mean) and math.isfinite(outside_mean)
                            else float("nan"),
                        }
                    )

    cosine_aggregate_fields = [
        "vector_cosine_similarity",
        "vector_angle_degrees",
        "rr_minus_ra_vector_l2",
        "ar_minus_aa_vector_l2",
        "difference_vector_l2",
        "mean_patch_l2_between_change_vectors",
        "mean_patch_cosine_similarity",
        "median_patch_cosine_similarity",
        "p05_patch_cosine_similarity",
        "p95_patch_cosine_similarity",
        "mask_inside_mean_patch_cosine",
        "mask_inside_median_patch_cosine",
        "mask_outside_mean_patch_cosine",
        "mask_outside_median_patch_cosine",
        "mask_inside_minus_outside_patch_cosine",
    ]
    for layer in layers:
        for block_type in ["frame_block", "global_block"]:
            rows = [
                r
                for r in interaction_cosine_rows
                if r["layer"] == layer and r["block_type"] == block_type
            ]
            if not rows:
                continue
            aggregate = {
                "comparison": "RR_minus_RA_vs_AR_minus_AA",
                "layer": layer,
                "block_type": block_type,
                "frames": ",".join(str(r["frame"]) for r in rows),
            }
            for field in cosine_aggregate_fields:
                aggregate[f"mean_{field}"] = float(np.nanmean([r[field] for r in rows]))
            interaction_cosine_aggregate_rows.append(aggregate)

    for layer in layers:
        for block_type in ["frame_block", "global_block"]:
            for effect in specs:
                rows = [r for r in token_rows if r["layer"] == layer and r["block_type"] == block_type and r["effect"] == effect]
                if not rows:
                    continue
                aggregate_rows.append(
                    {
                        "effect": effect,
                        "layer": layer,
                        "block_type": block_type,
                        "frames": ",".join(str(r["frame"]) for r in rows),
                        "mean_effect_vector_l2": float(np.mean([r["effect_vector_l2"] for r in rows])),
                        "mean_patch_l2": float(np.mean([r["mean_patch_l2"] for r in rows])),
                        "mean_relative_l2": float(np.nanmean([r["relative_l2"] for r in rows])),
                        "mean_cosine_distance": float(np.nanmean([r["cosine_distance"] for r in rows])),
                        "mean_inside_outside_ratio": float(np.nanmean([r["inside_outside_ratio"] for r in rows])),
                        "mean_centroid_distance_patches": float(
                            np.nanmean([r["effect_centroid_to_target_center_patches"] for r in rows])
                        ),
                    }
                )

    write_csv(root / "metrics" / "transplant_token_effect_metrics.csv", token_rows)
    write_csv(root / "metrics" / "transplant_token_effect_aggregate.csv", aggregate_rows)
    write_csv(root / "metrics" / "transplant_effect_fractions.csv", fraction_rows)
    write_csv(root / "metrics" / "transplant_interaction_cosine_metrics.csv", interaction_cosine_rows)
    write_csv(root / "metrics" / "transplant_interaction_cosine_aggregate.csv", interaction_cosine_aggregate_rows)
    write_text(
        root / "analysis" / "interaction_cosine_summary.md",
        interaction_cosine_summary_markdown(interaction_cosine_aggregate_rows),
    )
    torch.save(
        {
            "grid_hw": grid_hw,
            "frames": frames,
            "layers": layers,
            "maps": {
                f"layer_{layer:02d}/frame_{frame:04d}/{block}/{effect}": torch.from_numpy(values)
                for (layer, frame, block, effect), values in heatmap_store.items()
            },
        },
        root / "metrics" / "transplant_raw_effect_maps.pt",
    )
    return {
        "token_rows": token_rows,
        "aggregate_rows": aggregate_rows,
        "fraction_rows": fraction_rows,
        "interaction_cosine_rows": interaction_cosine_rows,
        "interaction_cosine_aggregate_rows": interaction_cosine_aggregate_rows,
        "heatmap_store": heatmap_store,
        "frames": frames,
        "layers": layers,
        "grid_hw": grid_hw,
    }


def make_key_heatmaps(cfg: dict[str, Any], root: Path, token_result: dict[str, Any], masks: dict[int, np.ndarray]) -> list[str]:
    frames = [int(x) for x in cfg["transplant"].get("max_report_heatmap_frames", token_result["frames"])]
    frames = [frame for frame in frames if frame in set(token_result["frames"])]
    effects = [
        "total_RR_minus_AA",
        "memory_AR_minus_AA",
        "rolling_RA_minus_AA",
        "source_AS_minus_AA",
        "interaction_RR_minus_RA_minus_AR_plus_AA",
    ]
    out_paths: list[str] = []
    for block_type in ["frame_block", "global_block"]:
        layer = 11
        patch_size = int(cfg["model"]["patch_size"])
        values_by_effect = [
            token_result["heatmap_store"][(layer, frame, block_type, effect)]
            for frame in frames
            for effect in effects
            if (layer, frame, block_type, effect) in token_result["heatmap_store"]
        ]
        if not values_by_effect:
            continue
        comparable_vmax = float(np.percentile(np.concatenate([v.reshape(-1) for v in values_by_effect]), 99))
        comparable_paths: dict[tuple[int, str], Path] = {}
        for frame in frames:
            for effect in effects:
                key = (layer, frame, block_type, effect)
                if key not in token_result["heatmap_store"]:
                    continue
                values = token_result["heatmap_store"][key]
                out_dir = root / "visualizations" / "token_effect_heatmaps" / f"layer_{layer:02d}" / block_type / effect
                save_heatmap_with_mask(out_dir / f"frame_{frame:04d}_independent.png", values, masks[frame], scale=patch_size)
                comparable = out_dir / f"frame_{frame:04d}_comparable.png"
                save_heatmap_with_mask(comparable, values, masks[frame], vmin=0.0, vmax=comparable_vmax, scale=patch_size)
                comparable_paths[(frame, effect)] = comparable
                out_paths.append(str(comparable))
        sheet = root / "visualizations" / "token_effect_heatmaps" / f"layer11_{block_type}_key_effects_comparable_contact_sheet.png"
        make_contact_sheet(sheet, frames, effects, comparable_paths)
        out_paths.append(str(sheet))
    return out_paths


def analyze_depth_and_pose(cfg: dict[str, Any], root: Path, variants: list[str], masks_pixel: dict[int, np.ndarray]) -> dict[str, Any]:
    eps = float(cfg["analysis"]["eps"])
    preds = {variant: load_predictions(root, variant) for variant in variants}
    inputs = load_inputs(root, "AA")
    frames = preds["AA"]["frame_ids"].astype(int).tolist()
    scale_path = project_path(cfg["project"]["output_root"]) / "runs" / cfg["transplant"]["baseline_run_id"] / "analysis" / "depth" / "shared_depth_scale.json"
    with scale_path.open("r", encoding="utf-8") as f:
        shared_scale = float(json.load(f)["depth_metric_scale"])
    depth_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    aa_depth = preds["AA"]["depth"][..., 0] if preds["AA"]["depth"].ndim == 4 else preds["AA"]["depth"].squeeze(-1)
    aa_depth = aa_depth.astype(np.float32) * np.float32(shared_scale)
    gt_depth = None
    if "gt_depth" in inputs:
        gt_depth = resize_float_stack(inputs["gt_depth"].astype(np.float32), aa_depth.shape[-2:])
    masks_resized = resize_bool_stack(np.stack([masks_pixel[f] for f in frames], axis=0), aa_depth.shape[-2:])
    for variant in variants:
        pred_depth = preds[variant]["depth"][..., 0] if preds[variant]["depth"].ndim == 4 else preds[variant]["depth"].squeeze(-1)
        pred_depth = pred_depth.astype(np.float32) * np.float32(shared_scale)
        diff = pred_depth - aa_depth
        for idx, frame in enumerate(frames):
            mask = masks_resized[idx]
            outside = ~mask
            depth_rows.append(
                {
                    "variant": variant,
                    "effect": f"{variant}_minus_AA",
                    "frame": frame,
                    "shared_depth_scale": shared_scale,
                    "mean_delta_all": float(np.nanmean(diff[idx])),
                    "mean_abs_delta_all": float(np.nanmean(np.abs(diff[idx]))),
                    "mean_delta_inside_mask": float(np.nanmean(diff[idx][mask])) if mask.any() else float("nan"),
                    "mean_abs_delta_inside_mask": float(np.nanmean(np.abs(diff[idx][mask]))) if mask.any() else float("nan"),
                    "mean_delta_outside_mask": float(np.nanmean(diff[idx][outside])) if outside.any() else float("nan"),
                    "mean_abs_delta_outside_mask": float(np.nanmean(np.abs(diff[idx][outside]))) if outside.any() else float("nan"),
                }
            )
            if gt_depth is not None:
                err = pred_depth[idx] - gt_depth[idx]
                depth_rows[-1].update(
                    {
                        "mae_to_gt_all": float(np.nanmean(np.abs(err))),
                        "mae_to_gt_inside_mask": float(np.nanmean(np.abs(err[mask]))) if mask.any() else float("nan"),
                        "mae_to_gt_outside_mask": float(np.nanmean(np.abs(err[outside]))) if outside.any() else float("nan"),
                    }
                )
        for key in ["pose_enc", "pred_cam2world", "pred_intrinsic"]:
            if key not in preds[variant] or key not in preds["AA"]:
                continue
            delta = preds[variant][key] - preds["AA"][key]
            pose_rows.append(
                {
                    "variant": variant,
                    "quantity": key,
                    "mean_abs_delta": float(np.mean(np.abs(delta))),
                    "max_abs_delta": float(np.max(np.abs(delta))),
                    "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / (np.linalg.norm(preds["AA"][key].reshape(-1)) + eps)),
                }
            )
    write_csv(root / "metrics" / "transplant_depth_effect_metrics.csv", depth_rows)
    write_csv(root / "metrics" / "transplant_pose_effect_metrics.csv", pose_rows)
    return {"shared_depth_scale": shared_scale, "depth_rows": depth_rows, "pose_rows": pose_rows}


def fit_pca(tokens: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(tokens, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:3].astype(np.float32)
    for i in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[i])))
        if components[i, pivot] < 0:
            components[i] *= -1.0
    return {"mean": mean.astype(np.float32), "components": components}


def make_diff_pca(cfg: dict[str, Any], root: Path, payload: dict[str, Any], masks: dict[int, np.ndarray]) -> str | None:
    layer = 11
    block = "global_block"
    frames = list(payload["AA"]["frame_ids"])
    effects = ["total_RR_minus_AA", "memory_AR_minus_AA", "rolling_RA_minus_AA"]
    tokens = []
    for effect in effects:
        for frame in frames:
            tokens.append(effect_tensor(payload, block, layer, frame, effect).numpy())
    pca = fit_pca(np.concatenate(tokens, axis=0))
    projected_all = []
    by_key = {}
    for effect in effects:
        for frame in frames:
            diff = effect_tensor(payload, block, layer, frame, effect).numpy()
            proj = (diff - pca["mean"]) @ pca["components"].T
            by_key[(frame, effect)] = proj
            projected_all.append(proj)
    cat = np.concatenate(projected_all, axis=0)
    lo = np.percentile(cat, 1, axis=0)
    hi = np.percentile(cat, 99, axis=0)
    grid_hw = tuple(int(x) for x in payload["AA"]["token_grid_hw"])
    patch_size = int(cfg["model"]["patch_size"])
    pixel_hw = (grid_hw[0] * patch_size, grid_hw[1] * patch_size)
    paths: dict[tuple[int, str], Path] = {}
    out_dir = root / "visualizations" / "diff_pca" / "layer11_global_block"
    for frame in frames:
        for effect in effects:
            proj = by_key[(frame, effect)]
            rgb = (proj - lo.reshape(1, 3)) / np.maximum((hi - lo).reshape(1, 3), 1.0e-6)
            rgb = np.clip(rgb, 0.0, 1.0).reshape((*grid_hw, 3))
            img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").resize(
                (pixel_hw[1], pixel_hw[0]), resample=Image.Resampling.BILINEAR
            )
            mask = Image.fromarray((masks[frame].astype(np.uint8) * 255), mode="L").resize(
                (pixel_hw[1], pixel_hw[0]), resample=Image.Resampling.NEAREST
            )
            out_path = out_dir / f"frame_{frame:04d}_{effect}_pca_rgb.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_mask_outline(np.asarray(img), np.asarray(mask) > 0).save(out_path)
            paths[(frame, effect)] = out_path
    sheet = out_dir / "layer11_global_block_total_memory_rolling_diff_pca_contact_sheet.png"
    make_contact_sheet(sheet, frames, effects, paths)
    return str(sheet)


def analyze_global_update(cfg: dict[str, Any], root: Path, payload: dict[str, Any]) -> None:
    eps = float(cfg["analysis"]["eps"])
    rows = []
    frames = list(payload["AA"]["frame_ids"])
    layers = [int(x) for x in payload["AA"]["selected_layers"]]
    for layer in layers:
        for frame in frames:
            aa_u = feature_tensor(payload, "AA", "global_block", layer, frame) - feature_tensor(
                payload, "AA", "frame_block", layer, frame
            )
            for effect, variant in [
                ("total_RR_minus_AA", "RR"),
                ("memory_AR_minus_AA", "AR"),
                ("rolling_RA_minus_AA", "RA"),
                ("source_AS_minus_AA", "AS"),
            ]:
                if variant not in payload:
                    continue
                u = feature_tensor(payload, variant, "global_block", layer, frame) - feature_tensor(
                    payload, variant, "frame_block", layer, frame
                )
                delta_u = u - aa_u
                incoming = feature_tensor(payload, variant, "frame_block", layer, frame) - feature_tensor(
                    payload, "AA", "frame_block", layer, frame
                )
                du_norm = vector_norm(delta_u)
                incoming_norm = vector_norm(incoming)
                cos_dist = cosine_distance(delta_u, incoming, eps) if incoming_norm > eps and du_norm > eps else float("nan")
                cos_sim = 1.0 - cos_dist if math.isfinite(cos_dist) else float("nan")
                if math.isfinite(cos_sim):
                    parallel = abs(cos_sim) * du_norm
                    orth = math.sqrt(max(du_norm * du_norm - parallel * parallel, 0.0))
                    orth_fraction = orth / (du_norm + eps)
                else:
                    orth_fraction = float("nan")
                rows.append(
                    {
                        "effect": effect,
                        "layer": layer,
                        "frame": frame,
                        "delta_update_l2": du_norm,
                        "incoming_frame_difference_l2": incoming_norm,
                        "cosine_with_incoming_frame_difference": cos_sim,
                        "orthogonal_fraction_of_delta_update": orth_fraction,
                    }
                )
    write_csv(root / "metrics" / "transplant_global_update_metrics.csv", rows)


def summarize_for_report(root: Path, token_result: dict[str, Any], depth_result: dict[str, Any], pca_sheet: str | None) -> str:
    agg = token_result["aggregate_rows"]
    key_rows = [r for r in agg if r["layer"] == 11 and r["block_type"] == "global_block"]
    by_effect = {r["effect"]: r for r in key_rows}
    fractions = [
        r
        for r in token_result["fraction_rows"]
        if r["layer"] == 11 and r["block_type"] == "global_block"
    ]
    mean_fraction = {}
    for key in ["memory_fraction", "rolling_fraction", "source_fraction_of_memory"]:
        vals = [r[key] for r in fractions if math.isfinite(r[key])]
        mean_fraction[key] = float(np.mean(vals)) if vals else float("nan")

    memory = by_effect.get("memory_AR_minus_AA", {}).get("mean_patch_l2", float("nan"))
    rolling = by_effect.get("rolling_RA_minus_AA", {}).get("mean_patch_l2", float("nan"))
    source = by_effect.get("source_AS_minus_AA", {}).get("mean_patch_l2", float("nan"))
    total = by_effect.get("total_RR_minus_AA", {}).get("mean_patch_l2", float("nan"))
    interaction = by_effect.get("interaction_RR_minus_RA_minus_AR_plus_AA", {}).get("mean_patch_l2", float("nan"))
    negative = by_effect.get("negative_AN_minus_AA", {}).get("mean_patch_l2", float("nan"))
    interaction_cosine_key = next(
        (
            r
            for r in token_result.get("interaction_cosine_aggregate_rows", [])
            if r["layer"] == 11 and r["block_type"] == "global_block"
        ),
        {},
    )
    interaction_vector_cosine = interaction_cosine_key.get("mean_vector_cosine_similarity", float("nan"))
    interaction_patch_cosine = interaction_cosine_key.get("mean_mean_patch_cosine_similarity", float("nan"))
    consistency_path = root / "metrics" / "replay_consistency.csv"
    consistency_max_abs = float("nan")
    consistency_max_rel = float("nan")
    if consistency_path.exists():
        import csv

        with consistency_path.open("r", encoding="utf-8", newline="") as f:
            consistency_rows = list(csv.DictReader(f))
        if consistency_rows:
            consistency_max_abs = max(float(r["max_abs"]) for r in consistency_rows)
            consistency_max_rel = max(float(r["relative_l2"]) for r in consistency_rows)
    depth_by_variant: dict[str, dict[str, float]] = {}
    for variant in ["RR", "AR", "RA", "AS", "AN"]:
        rows = [r for r in depth_result["depth_rows"] if r["variant"] == variant]
        if rows:
            depth_by_variant[variant] = {
                "mean_delta_all": float(np.mean([r["mean_delta_all"] for r in rows])),
                "mean_delta_inside_mask": float(np.mean([r["mean_delta_inside_mask"] for r in rows])),
                "mean_delta_outside_mask": float(np.mean([r["mean_delta_outside_mask"] for r in rows])),
                "mean_abs_delta_inside_mask": float(np.mean([r["mean_abs_delta_inside_mask"] for r in rows])),
            }
    if math.isfinite(memory) and math.isfinite(rolling) and memory > rolling:
        primary = "explicit trajectory-memory contribution is larger than the rolling-state branch on layer-11 global image tokens"
    elif math.isfinite(memory) and math.isfinite(rolling):
        primary = "rolling/current-state contribution is larger than the explicit trajectory-memory branch on layer-11 global image tokens"
    else:
        primary = "the mechanism ranking is not available from the generated token metrics"
    if math.isfinite(source) and math.isfinite(negative) and source <= 1.2 * negative:
        source_note = "AS is close to the matched non-target AN control, so the target-visible records alone do not stand out as a specific causal source in this run."
    elif math.isfinite(source) and math.isfinite(memory):
        source_note = "AS is separated from the negative control, suggesting the selected target-visible records contribute to the explicit-memory branch."
    else:
        source_note = "The source-record comparison is inconclusive."

    lines = [
        "# Memory Transplant V1 Report",
        "",
        "## Purpose",
        "",
        "This run asks whether changing only LingBot-Map's long-term trajectory special-token K/V cache changes target-frame processing when the current image, active local window, anchor/scale context, and camera-head state are kept from the query branch.",
        "",
        "## Cache/State Structure",
        "",
        "- Detailed cache notes: `cache_structure.md`.",
        "- Machine-readable layout: `metadata/memory_layout.json`.",
        "- No `lingbot-map/` source files were modified; see `source_changes.md`.",
        "",
        "## Variants",
        "",
        "- AA: A query and A trajectory memory.",
        "- RR: R query and R trajectory memory.",
        "- AR: A query/current/local/anchor state with R trajectory memory.",
        "- RA: R query/current/local/anchor state with A trajectory memory.",
        "- AS: A state with only target-visible source-frame R records transplanted.",
        "- AN: A state with matched non-target R records transplanted.",
        "",
        "## Core Result Snapshot",
        "",
        f"Replay/identity consistency max abs error: `{consistency_max_abs:.6g}`; max relative L2: `{consistency_max_rel:.6g}`.",
        "",
        "| effect | layer11 global mean patch L2 | layer11 global inside/outside |",
        "| --- | ---: | ---: |",
    ]
    for effect in [
        "total_RR_minus_AA",
        "memory_AR_minus_AA",
        "rolling_RA_minus_AA",
        "source_AS_minus_AA",
        "negative_AN_minus_AA",
        "interaction_RR_minus_RA_minus_AR_plus_AA",
    ]:
        row = by_effect.get(effect)
        if row:
            lines.append(f"| {effect} | {row['mean_patch_l2']:.6g} | {row['mean_inside_outside_ratio']:.4g} |")
    lines.extend(
        [
            "",
            "Mean descriptive fractions on layer-11 global tokens:",
            "",
            f"- memory_fraction: `{mean_fraction['memory_fraction']:.4g}`",
            f"- rolling_fraction: `{mean_fraction['rolling_fraction']:.4g}`",
            f"- source_fraction_of_memory: `{mean_fraction['source_fraction_of_memory']:.4g}`",
            f"- interaction vector cosine `(RR-RA)` vs `(AR-AA)`: `{interaction_vector_cosine:.4g}`; mean patch cosine: `{interaction_patch_cosine:.4g}`",
            "",
            "These are descriptive norm ratios only; the query-memory system is nonlinear and the effects need not add linearly.",
            "",
            "## Geometry",
            "",
            f"- Shared depth scale reused from baseline: `{depth_result['shared_depth_scale']:.8g}`.",
            f"- Mean depth delta RR-AA: `{depth_by_variant.get('RR', {}).get('mean_delta_all', float('nan')):.6g}` m-equivalent after shared scale.",
            f"- Mean depth delta AR-AA: `{depth_by_variant.get('AR', {}).get('mean_delta_all', float('nan')):.6g}`.",
            f"- Mean depth delta RA-AA: `{depth_by_variant.get('RA', {}).get('mean_delta_all', float('nan')):.6g}`.",
            f"- Mean depth delta AS-AA: `{depth_by_variant.get('AS', {}).get('mean_delta_all', float('nan')):.6g}`.",
            "- Depth effects: `metrics/transplant_depth_effect_metrics.csv`.",
            "- Pose effects: `metrics/transplant_pose_effect_metrics.csv`.",
            "",
            "## Key Visualizations",
            "",
            "- Layer-11 frame/global token heatmaps: `visualizations/token_effect_heatmaps/`.",
            f"- Layer-11 global diff PCA sheet: `{pca_sheet}`.",
            "",
            "## Interpretation",
            "",
            f"At this stage, the compact layer-11 global-token summary says: {primary}.",
            "",
            f"Mean patch L2 values: total `{total:.6g}`, explicit memory `{memory:.6g}`, rolling `{rolling:.6g}`, source-record `{source:.6g}`, negative control `{negative:.6g}`, interaction `{interaction:.6g}`.",
            "",
            source_note,
            "",
            "The current evidence is therefore not enough to claim object-content memory. It more strongly supports distributed rolling propagation plus a weaker explicit trajectory-memory effect that is diffuse rather than mask-localized. The nontrivial interaction term also supports query-memory coupling rather than an independently readable memory database.",
            "",
            "## Limits And Next Experiment",
            "",
            "This run edits evicted aggregator special K/V records only. It does not transplant patch-token local cache, camera-head cache, or hidden states before K/V projection. The next most decisive experiment is a source-record ablation or target-object relocation run: keep trajectory/timing fixed, move the object in the first loop, and test whether AR/AS responses project to the old object location rather than generic scene boundaries.",
            "",
            "## Files",
            "",
            "- Replay consistency: `metrics/replay_consistency.csv`.",
            "- Token metrics: `metrics/transplant_token_effect_metrics.csv`.",
            "- Aggregate token table: `metrics/transplant_token_effect_aggregate.csv`.",
            "- Effect fractions: `metrics/transplant_effect_fractions.csv`.",
            "- Interaction cosine metrics: `metrics/transplant_interaction_cosine_aggregate.csv` and `analysis/interaction_cosine_summary.md`.",
            "- Global update metrics: `metrics/transplant_global_update_metrics.csv`.",
            "- Source-frame records: `metadata/source_frame_records.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(cfg: dict[str, Any]) -> None:
    root = output_root(cfg)
    variants = [p.name for p in (root / "features").iterdir() if (p / "features.pt").exists()]
    required = {"AA", "RR", "AR", "RA", "AS"}
    missing = sorted(required - set(variants))
    if missing:
        raise FileNotFoundError(f"Missing transplant feature outputs for variants: {missing}. Run run_memory_transplant.py first.")
    variants = sorted(variants, key=lambda x: ["AA", "RR", "AR", "RA", "AS", "AN"].index(x) if x in ["AA", "RR", "AR", "RA", "AS", "AN"] else 99)
    payload = {variant: load_features(root, variant) for variant in variants}
    inputs = load_inputs(root, "AA")
    frames = list(payload["AA"]["frame_ids"])
    grid_hw = tuple(int(x) for x in payload["AA"]["token_grid_hw"])
    masks = load_masks(inputs, frames, grid_hw)
    frame_ids = inputs["frame_ids"].astype(int).tolist()
    masks_pixel = {frame: inputs["counterfactual_target_mask"][frame_ids.index(frame)] for frame in frames}

    token_result = analyze_tokens(cfg, root, payload, masks)
    heatmaps = make_key_heatmaps(cfg, root, token_result, masks)
    depth_result = analyze_depth_and_pose(cfg, root, variants, masks_pixel)
    pca_sheet = make_diff_pca(cfg, root, payload, masks)
    analyze_global_update(cfg, root, payload)
    write_json(
        root / "analysis_summary.json",
        {
            "variants": variants,
            "frames": frames,
            "key_heatmaps": heatmaps,
            "diff_pca_contact_sheet": pca_sheet,
            "shared_depth_scale": depth_result["shared_depth_scale"],
        },
    )
    write_text(root / "report.md", summarize_for_report(root, token_result, depth_result, pca_sheet))
    print(f"Wrote {root / 'report.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze memory transplant outputs.")
    parser.add_argument("--config", default="revisit_memory/configs/memory_transplant_v1.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
