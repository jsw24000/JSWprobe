from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _use_agg_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_trajectory_xy(poses_c2w: np.ndarray, out_path: str | Path, highlights: dict[int, str] | None = None) -> None:
    plt = _use_agg_matplotlib()
    centers = poses_c2w[:, :3, 3]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(centers[:, 0], centers[:, 1], linewidth=1.0, color="#2f6f9f")
    ax.scatter(centers[0, 0], centers[0, 1], c="#2a9d8f", s=30, label="start")
    ax.scatter(centers[-1, 0], centers[-1, 1], c="#e76f51", s=30, label="end")
    if highlights:
        for frame_id, label in highlights.items():
            ax.scatter(centers[frame_id, 0], centers[frame_id, 1], s=50, label=label)
            ax.annotate(str(frame_id), (centers[frame_id, 0], centers[frame_id, 1]))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trajectory_3d(poses_c2w: np.ndarray, out_path: str | Path, highlights: dict[int, str] | None = None) -> None:
    plt = _use_agg_matplotlib()
    centers = poses_c2w[:, :3, 3]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], linewidth=1.0, color="#2f6f9f")
    ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2], c="#2a9d8f", s=30, label="start")
    ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2], c="#e76f51", s=30, label="end")
    if highlights:
        for frame_id, label in highlights.items():
            ax.scatter(centers[frame_id, 0], centers[frame_id, 1], centers[frame_id, 2], s=50, label=label)
            ax.text(centers[frame_id, 0], centers[frame_id, 1], centers[frame_id, 2], str(frame_id))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_frame_jpg(src: str | Path, dst: str | Path) -> None:
    with Image.open(src) as im:
        im.convert("RGB").save(dst, quality=95)


def save_frame_pair_comparison(
    left_path: str | Path,
    right_path: str | Path,
    out_path: str | Path,
    left_label: str,
    right_label: str,
) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    label_h = 34
    width = left.width + right.width
    height = max(left.height, right.height) + label_h
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(left, (0, label_h))
    canvas.paste(right, (left.width, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), left_label, fill=(0, 0, 0))
    draw.text((left.width + 10, 10), right_label, fill=(0, 0, 0))
    canvas.save(out_path, quality=95)
