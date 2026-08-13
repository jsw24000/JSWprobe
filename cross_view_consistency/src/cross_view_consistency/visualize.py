"""Matplotlib visualizations for pair geometry and feature matches."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .matcher import similarity_map_for_query
from .scannet_io import load_rgb_array


def _pair_images(pair, frames, input_hw):
    src = frames[int(pair.src_index)]
    tgt = frames[int(pair.tgt_index)]
    return load_rgb_array(src.color_path, input_hw), load_rgb_array(tgt.color_path, input_hw)


def save_pair_overview(pair, frames, input_hw, out_path: Path) -> None:
    src_img, tgt_img = _pair_images(pair, frames, input_hw)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(src_img)
    axes[0].set_title(f"source {pair.src_frame}")
    axes[1].imshow(tgt_img)
    axes[1].set_title(f"target {pair.tgt_frame}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{pair.pair_id} {pair.bucket} | trans={pair.translation_m:.2f}m rot={pair.rotation_deg:.1f}deg overlap={pair.overlap_ratio:.2f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_gt_projection(pair, frames, input_hw, queries: pd.DataFrame, out_path: Path, max_lines: int = 80) -> None:
    src_img, tgt_img = _pair_images(pair, frames, input_hw)
    n = min(max_lines, len(queries))
    q = queries.iloc[:n]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(src_img)
    axes[1].imshow(tgt_img)
    axes[0].scatter(q["src_u"], q["src_v"], s=8, c="yellow")
    axes[1].scatter(q["gt_tgt_u"], q["gt_tgt_v"], s=8, c="cyan")
    axes[0].set_title(f"source {pair.src_frame}")
    axes[1].set_title(f"GT target {pair.tgt_frame}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{pair.pair_id} GT projections ({len(queries)} valid queries)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_match_plot(pair, frames, input_hw, matches: pd.DataFrame, out_path: Path, correct_px_threshold: float, max_lines: int) -> None:
    src_img, tgt_img = _pair_images(pair, frames, input_hw)
    q = matches[matches["method"] == "feature_nn"].copy()
    if len(q) == 0:
        q = matches.copy()
    q = q.iloc[: min(max_lines, len(q))]
    canvas = np.concatenate([src_img, tgt_img], axis=1)
    w = src_img.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.imshow(canvas)
    for row in q.itertuples(index=False):
        color = "lime" if row.px_error <= correct_px_threshold else "red"
        ax.plot([row.src_u, row.pred_tgt_u + w], [row.src_v, row.pred_tgt_v], color=color, linewidth=0.7, alpha=0.7)
        ax.scatter([row.src_u, row.pred_tgt_u + w], [row.src_v, row.pred_tgt_v], s=5, c=color)
    ax.axis("off")
    r10 = (matches["px_error"] <= 10).mean() if len(matches) else 0
    r16 = (matches["px_error"] <= 16).mean() if len(matches) else 0
    r32 = (matches["px_error"] <= 32).mean() if len(matches) else 0
    med = matches["px_error"].median() if len(matches) else float("nan")
    title_backend = matches["backend"].iloc[0] if len(matches) else "backend"
    ax.set_title(f"{title_backend} | {pair.bucket} | R10={r10:.2f} R16={r16:.2f} R32={r32:.2f} med={med:.1f}px")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_heatmaps(pair, frames, input_hw, feature_result, queries: pd.DataFrame, out_dir: Path, max_queries: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_img, tgt_img = _pair_images(pair, frames, input_hw)
    q = queries.iloc[: min(max_queries, len(queries))]
    for i, row in enumerate(q.itertuples(index=False)):
        sim = similarity_map_for_query(feature_result, int(pair.src_frame), int(pair.tgt_frame), int(row.src_token_idx))
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(src_img)
        axes[0].scatter([row.src_u], [row.src_v], c="yellow", s=30)
        axes[0].set_title("source query")
        axes[1].imshow(tgt_img)
        axes[1].imshow(sim, cmap="magma", alpha=0.55, extent=(0, input_hw[1], input_hw[0], 0))
        axes[1].scatter([row.gt_tgt_u], [row.gt_tgt_v], c="cyan", s=30, label="GT")
        axes[1].set_title("target similarity")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{pair.pair_id} {feature_result.backend_name} query{i:03d}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{pair.pair_id}_{feature_result.backend_name}_query{i:03d}.png", dpi=160)
        plt.close(fig)


def save_summary_plots(metrics_backend: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(metrics_backend) == 0:
        return
    feat = metrics_backend[metrics_backend["method"] == "feature_nn"].copy()
    if len(feat) == 0:
        return
    for col, name, ylabel in [
        ("recall_10px", "summary_recall_by_backend.png", "Recall@10px"),
        ("median_px_error", "summary_median_error_by_backend.png", "Median px error"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = feat["backend"] + " / " + feat["bucket"]
        ax.bar(np.arange(len(feat)), feat[col])
        ax.set_xticks(np.arange(len(feat)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)
    pivot = feat.pivot_table(index="bucket", columns="backend", values="recall_10px", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Recall@10px")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_recall_by_bucket.png", dpi=160)
    plt.close(fig)

