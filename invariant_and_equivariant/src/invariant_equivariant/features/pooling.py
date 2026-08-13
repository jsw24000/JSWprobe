from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_mask(mask_path: str | Path) -> torch.Tensor:
    image = Image.open(mask_path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    mask = (arr > 127).astype(np.float32)
    return torch.from_numpy(mask)[None, None]


def area_weights(mask_path: str | Path, patch_h: int, patch_w: int) -> torch.Tensor:
    mask = load_mask(mask_path)
    return F.interpolate(mask, size=(patch_h, patch_w), mode="area")[0, 0]


def pool_patch_tokens(
    patch_tokens: torch.Tensor,
    mask_path: str | Path,
    thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    """Area-weight object pooling.

    Args:
        patch_tokens: tensor [patch_h, patch_w, dim].
    """
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected [patch_h, patch_w, dim], got {tuple(patch_tokens.shape)}")
    patch_h, patch_w, dim = patch_tokens.shape
    weights = area_weights(mask_path, patch_h, patch_w).to(device=patch_tokens.device, dtype=patch_tokens.dtype)
    denom = weights.sum().clamp_min(1e-8)
    pooled = (patch_tokens * weights[..., None]).sum(dim=(0, 1)) / denom
    l2 = pooled / pooled.norm(p=2).clamp_min(1e-8)
    effective = float((weights > 0.05).sum().item())
    coverage = float(weights.sum().item())
    valid = (
        coverage >= float(thresholds["min_patch_coverage"])
        and effective >= float(thresholds["min_effective_patches"])
    )
    return {
        "pooled_raw": pooled.detach().float().cpu().numpy(),
        "pooled_l2": l2.detach().float().cpu().numpy(),
        "patch_weights": weights.detach().float().cpu().numpy(),
        "patch_coverage_sum": coverage,
        "effective_patch_count": effective,
        "valid_by_patch": bool(valid),
    }


def mask_2d_features(frame: Mapping[str, Any]) -> np.ndarray:
    bbox = frame.get("target_bbox_2d")
    image_size = frame.get("image_size") or [512, 512]
    width, height = float(image_size[0]), float(image_size[1])
    if not bbox:
        vals = [0.0] * 6
    else:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        cx = (x0 + x1) * 0.5 / width
        cy = (y0 + y1) * 0.5 / height
        bw = (x1 - x0 + 1.0) / width
        bh = (y1 - y0 + 1.0) / height
        vals = [cx, cy, bw, bh, float(frame.get("target_mask_area_ratio", 0.0)), float(frame.get("target_truncated", False))]
    cam_num = float(str(frame.get("camera_id", "camera_000")).split("_")[-1])
    return np.asarray([*vals, cam_num], dtype=np.float64)

