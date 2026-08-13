from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def resize_float(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    return np.asarray(img.resize((hw[1], hw[0]), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def resize_bool(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((np.asarray(arr) > 0).astype(np.uint8) * 255, mode="L")
    return np.asarray(img.resize((hw[1], hw[0]), resample=Image.Resampling.NEAREST)) > 0


def resize_float_stack(stack: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return np.stack([resize_float(arr, hw) for arr in np.asarray(stack)], axis=0)


def resize_bool_stack(stack: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return np.stack([resize_bool(arr, hw) for arr in np.asarray(stack)], axis=0)


def squeeze_depth_stack(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected depth stack shaped [F,H,W] or [F,H,W,1], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def collect_depth_ratios(
    pred_depth: np.ndarray,
    gt_depth_resized: np.ndarray,
    exclude_mask_resized: np.ndarray | None = None,
    *,
    eps: float = 1.0e-6,
    min_unmasked_pixels: int = 1024,
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Collect valid `gt / pred` depth ratios with optional target-mask exclusion."""
    pred = squeeze_depth_stack(pred_depth)
    gt = squeeze_depth_stack(gt_depth_resized)
    if pred.shape != gt.shape:
        raise ValueError(f"pred_depth and gt_depth_resized must have the same shape, got {pred.shape} and {gt.shape}")

    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > eps) & (gt > eps)
    used_mask_exclusion = False
    unmasked_count = None
    if exclude_mask_resized is not None:
        mask = np.asarray(exclude_mask_resized) > 0
        if mask.shape != valid.shape:
            raise ValueError(f"exclude_mask_resized shape {mask.shape} does not match depth shape {valid.shape}")
        unmasked = valid & ~mask
        unmasked_count = int(unmasked.sum())
        if unmasked_count >= int(min_unmasked_pixels):
            valid = unmasked
            used_mask_exclusion = True

    ratios = (gt[valid] / pred[valid]).astype(np.float32)
    original_count = int(ratios.size)
    sampled = False
    if max_samples is not None and ratios.size > int(max_samples):
        rng = np.random.default_rng(seed)
        ratios = ratios[rng.choice(ratios.size, size=int(max_samples), replace=False)]
        sampled = True

    return ratios, {
        "valid_ratio_count_before_sampling": original_count,
        "valid_ratio_count": int(ratios.size),
        "used_mask_exclusion": used_mask_exclusion,
        "unmasked_valid_ratio_count": unmasked_count,
        "sampled": sampled,
        "max_samples": max_samples,
    }


def estimate_depth_scale(
    pred_depth: np.ndarray,
    gt_depth_resized: np.ndarray,
    exclude_mask_resized: np.ndarray | None = None,
    *,
    eps: float = 1.0e-6,
    min_unmasked_pixels: int = 1024,
    max_samples: int = 5_000_000,
    seed: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Estimate one metric scale mapping predicted depth to GT z-depth."""
    ratios, ratio_info = collect_depth_ratios(
        pred_depth,
        gt_depth_resized,
        exclude_mask_resized,
        eps=eps,
        min_unmasked_pixels=min_unmasked_pixels,
        max_samples=max_samples,
        seed=seed,
    )
    if ratios.size == 0:
        return 1.0, {
            "depth_metric_scale": 1.0,
            "valid_ratio_count": 0,
            "scale_policy": "fallback_no_valid_gt_over_pred_ratios",
            **ratio_info,
        }

    scale = float(np.nanmedian(ratios))
    return scale, {
        "depth_metric_scale": scale,
        "ratio_p05": float(np.nanpercentile(ratios, 5)),
        "ratio_p50": scale,
        "ratio_p95": float(np.nanpercentile(ratios, 95)),
        "scale_policy": "global_median_gt_depth_over_pred_depth_excluding_target_when_available",
        **ratio_info,
    }


def estimate_pred_to_reference_scale(
    source_depth: np.ndarray,
    reference_depth: np.ndarray,
    include_mask: np.ndarray | None = None,
    *,
    eps: float = 1.0e-6,
    min_valid_pixels: int = 1024,
    max_samples: int = 1_000_000,
    seed: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Estimate a multiplicative scale mapping source predictions to reference predictions."""
    source = squeeze_depth_stack(np.asarray(source_depth)[None, ...] if np.asarray(source_depth).ndim == 2 else source_depth)
    reference = squeeze_depth_stack(
        np.asarray(reference_depth)[None, ...] if np.asarray(reference_depth).ndim == 2 else reference_depth
    )
    if source.shape != reference.shape:
        raise ValueError(f"source_depth and reference_depth must have the same shape, got {source.shape} and {reference.shape}")
    valid = np.isfinite(source) & np.isfinite(reference) & (source > eps) & (reference > eps)
    if include_mask is not None:
        mask = np.asarray(include_mask) > 0
        if mask.ndim == 2:
            mask = mask[None, ...]
        if mask.shape != valid.shape:
            raise ValueError(f"include_mask shape {mask.shape} does not match depth shape {valid.shape}")
        valid &= mask
    ratios = (reference[valid] / source[valid]).astype(np.float32)
    original_count = int(ratios.size)
    if ratios.size < int(min_valid_pixels):
        return 1.0, {
            "scale": 1.0,
            "valid_ratio_count": original_count,
            "scale_policy": "fallback_too_few_pred_to_reference_ratios",
        }
    sampled = False
    if ratios.size > int(max_samples):
        rng = np.random.default_rng(seed)
        ratios = ratios[rng.choice(ratios.size, size=int(max_samples), replace=False)]
        sampled = True
    scale = float(np.nanmedian(ratios))
    return scale, {
        "scale": scale,
        "valid_ratio_count_before_sampling": original_count,
        "valid_ratio_count": int(ratios.size),
        "ratio_p05": float(np.nanpercentile(ratios, 5)),
        "ratio_p50": scale,
        "ratio_p95": float(np.nanpercentile(ratios, 95)),
        "sampled": sampled,
        "scale_policy": "median_reference_pred_depth_over_source_pred_depth",
    }
