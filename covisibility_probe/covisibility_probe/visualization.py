from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def save_layer_trends(summary: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metric_cols = [
        ("spearman", "Scene-mean Spearman"),
        ("residualized_rank_correlation", "Residualized Rank Correlation"),
        ("ndcg@5", "NDCG@5"),
    ]
    paths: list[Path] = []
    for metric, title in metric_cols:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
        for feature, marker in [
            ("Register head frame-half", "o"),
            ("Register head global-half", "s"),
            ("Register head concat", "^"),
        ]:
            sub = summary[(summary["Feature"] == feature) & (summary["Layer"] != "-")].copy()
            if len(sub) == 0 or metric not in sub:
                continue
            sub["Layer"] = sub["Layer"].astype(int)
            sub = sub.sort_values("Layer")
            y = sub[metric].to_numpy(dtype=np.float64)
            lo = sub.get(f"{metric}_ci_low", pd.Series(np.full(len(sub), np.nan))).to_numpy(dtype=np.float64)
            hi = sub.get(f"{metric}_ci_high", pd.Series(np.full(len(sub), np.nan))).to_numpy(dtype=np.float64)
            yerr = np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)])
            ax.errorbar(sub["Layer"], y, yerr=yerr, marker=marker, capsize=3, label=feature)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out / f"layer_trend_{metric}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def save_time_overlap_heatmap(grid: pd.DataFrame, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(grid) == 0:
        return path
    pivot = grid.pivot(index="time_gap_bin", columns="overlap_bin", values="mean_register_similarity")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("GT overlap bin")
    ax.set_ylabel("Temporal gap bin")
    ax.set_title("Mean register similarity")
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.iloc[y, x]
            text = "" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(x, y, text, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_similarity_hexbin(pairs: pd.DataFrame, column: str, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pairs[pairs["is_trajectory_only"]].copy()
    df = df[np.isfinite(df["overlap_cos"]) & np.isfinite(df[column])]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    if len(df):
        hb = ax.hexbin(df["overlap_cos"], df[column], gridsize=40, mincnt=1, cmap="magma")
        fig.colorbar(hb, ax=ax, label="pair count")
    ax.set_xlabel("GT overlap_cos")
    ax.set_ylabel(column)
    ax.set_title("Trajectory-only similarity vs overlap")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _thumb(path: str | Path, size: tuple[int, int] = (180, 134)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def save_retrieval_cases(
    pairs: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    similarity_column: str,
    out_dir: str | Path,
    max_cases: int = 6,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    by_scene_frame = {
        (r.scene_id, int(r.input_position)): str(r.color_path)
        for r in manifest.itertuples(index=False)
    }
    traj = pairs[pairs["is_trajectory_only"]].copy()
    if len(traj) == 0 or similarity_column not in traj:
        return paths
    candidates = []
    for (scene_id, target), group in traj.groupby(["scene_id", "input_position_j"]):
        if len(group) < 3:
            continue
        best_reg = group.sort_values(similarity_column, ascending=False).head(3)
        best_gt = group.sort_values("overlap_cos", ascending=False).head(1)
        top_reg_overlap = float(best_reg.iloc[0]["overlap_cos"])
        top_gt_overlap = float(best_gt.iloc[0]["overlap_cos"])
        score = abs(top_gt_overlap - top_reg_overlap) + top_gt_overlap
        candidates.append((score, scene_id, int(target), best_reg, best_gt))
    candidates = sorted(candidates, key=lambda x: -x[0])[: int(max_cases)]
    for idx, (_, scene_id, target, best_reg, best_gt) in enumerate(candidates):
        w, h = 5 * 180, 134 + 54
        canvas = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(canvas)
        items = [("target", target, np.nan, 0)]
        for row in best_reg.itertuples(index=False):
            items.append(("reg", int(row.input_position_i), float(row.overlap_cos), int(row.temporal_gap)))
        gt_row = best_gt.iloc[0]
        items.append(("gt-best", int(gt_row["input_position_i"]), float(gt_row["overlap_cos"]), int(gt_row["temporal_gap"])))
        for col, (label, pos, overlap, gap) in enumerate(items[:5]):
            img_path = by_scene_frame.get((scene_id, pos))
            if img_path:
                canvas.paste(_thumb(img_path), (col * 180, 0))
            text = label if label == "target" else f"{label} O={overlap:.2f} dt={gap}"
            draw.text((col * 180 + 4, 140), text, fill=(0, 0, 0))
        path = out / f"retrieval_case_{idx:02d}_{scene_id}_target_{target:04d}.jpg"
        canvas.save(path, quality=92)
        paths.append(path)
    return paths
