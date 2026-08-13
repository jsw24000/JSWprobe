from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def normalize_for_uint8(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if vmin is None:
        vmin = float(np.nanmin(arr[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(arr[finite])) if finite.any() else 1.0
    denom = max(float(vmax - vmin), 1e-12)
    out = np.clip((arr - vmin) / denom, 0.0, 1.0)
    out[~finite] = 0.0
    return (out * 255).astype(np.uint8)


def save_gray(path: str | Path, values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalize_for_uint8(values, vmin, vmax), mode="L").save(path)


def save_heatmap(path: str | Path, values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = normalize_for_uint8(values, vmin, vmax)
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("magma")
        rgba = (cmap(gray / 255.0) * 255).astype(np.uint8)
        Image.fromarray(rgba[..., :3], mode="RGB").save(path)
    except Exception:
        Image.fromarray(gray, mode="L").save(path)


def overlay_mask_outline(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
) -> Image.Image:
    rgb = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return rgb
    padded = np.pad(mask, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    neighbor_all = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    edge = center & ~neighbor_all
    ys, xs = np.nonzero(edge)
    draw = ImageDraw.Draw(rgb)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((x, y), fill=color)
    return rgb


def resize_mask_to_grid(mask: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    image = image.resize((grid_hw[1], grid_hw[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def resize_map_to_image(values: np.ndarray, image_hw: tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    resample = Image.Resampling.BILINEAR if mode == "bilinear" else Image.Resampling.NEAREST
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    fill = float(np.nanmin(arr[finite])) if finite.any() else 0.0
    arr = np.nan_to_num(arr, nan=fill)
    lo, hi = float(arr.min()), float(arr.max())
    scaled = normalize_for_uint8(arr, lo, hi)
    img = Image.fromarray(scaled, mode="L").resize((image_hw[1], image_hw[0]), resample=resample)
    out = np.asarray(img).astype(np.float32) / 255.0
    return out * (hi - lo) + lo


def _draw_bbox_on_axes(ax, bbox_min: np.ndarray, bbox_max: np.ndarray, axes: tuple[int, int]) -> None:
    try:
        import matplotlib.patches as patches

        x0, y0 = float(bbox_min[axes[0]]), float(bbox_min[axes[1]])
        x1, y1 = float(bbox_max[axes[0]]), float(bbox_max[axes[1]])
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.4,
            edgecolor="#00a6d6",
            facecolor="none",
        )
        ax.add_patch(rect)
    except Exception:
        return


def save_scatter_view(
    path: str | Path,
    points: np.ndarray,
    axes: tuple[int, int] = (0, 1),
    title: str | None = None,
    bbox_min: np.ndarray | None = None,
    bbox_max: np.ndarray | None = None,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
        if pts.size:
            ax.scatter(pts[:, axes[0]], pts[:, axes[1]], s=0.2, c="black", alpha=0.5)
        if bbox_min is not None and bbox_max is not None:
            _draw_bbox_on_axes(ax, np.asarray(bbox_min), np.asarray(bbox_max), axes)
        if limits is not None:
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("xyz"[axes[0]])
        ax.set_ylabel("xyz"[axes[1]])
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        Image.new("RGB", (512, 512), "white").save(path)


def save_scatter_comparison(
    path: str | Path,
    point_sets: dict[str, np.ndarray],
    axes: tuple[int, int] = (0, 1),
    bbox_min: np.ndarray | None = None,
    bbox_max: np.ndarray | None = None,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    title: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, len(point_sets), figsize=(6 * len(point_sets), 6), dpi=160, squeeze=False)
        for ax, (name, points) in zip(axs[0], point_sets.items()):
            pts = np.asarray(points).reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.size:
                ax.scatter(pts[:, axes[0]], pts[:, axes[1]], s=0.2, c="black", alpha=0.5)
            if bbox_min is not None and bbox_max is not None:
                _draw_bbox_on_axes(ax, np.asarray(bbox_min), np.asarray(bbox_max), axes)
            if limits is not None:
                ax.set_xlim(*limits[0])
                ax.set_ylim(*limits[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("xyz"[axes[0]])
            ax.set_ylabel("xyz"[axes[1]])
            ax.set_title(name)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        Image.new("RGB", (512, 512), "white").save(path)
