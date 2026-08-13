"""View-pair and query-token sampling."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .geometry import GridSpec, estimate_overlap, relative_pose_stats, valid_token_correspondences
from .scannet_io import FrameRecord, SceneInfo


def bucket_for_pair(translation_m: float, rotation_deg: float, buckets: dict) -> str | None:
    for name, spec in buckets.items():
        if (
            translation_m >= float(spec.get("min_translation_m", 0.0))
            and translation_m < float(spec.get("max_translation_m", float("inf")))
            and rotation_deg <= float(spec.get("max_rotation_deg", float("inf")))
        ):
            return name
    return None


def sample_pairs(scene: SceneInfo, frames: list[FrameRecord], grid: GridSpec, cfg: dict) -> pd.DataFrame:
    pair_cfg = cfg["pairs"]
    geom_cfg = cfg["geometry"]
    min_overlap = float(pair_cfg.get("min_overlap_ratio", 0.05))
    min_corr = int(pair_cfg.get("min_valid_correspondences", 200))
    per_bucket = int(pair_cfg.get("pairs_per_bucket", 12))
    buckets = pair_cfg.get("buckets", {})
    seed = int(cfg.get("run", {}).get("seed", 42))

    candidates = []
    combos = list(itertools.combinations(range(len(frames)), 2))
    for src_i, tgt_i in tqdm(combos, desc="Sampling candidate pairs", unit="pair"):
        src = frames[src_i]
        tgt = frames[tgt_i]
        translation, rotation = relative_pose_stats(src, tgt)
        bucket = bucket_for_pair(translation, rotation, buckets)
        overlap = estimate_overlap(scene, src, tgt, geom_cfg, seed=seed + src.frame_id * 997 + tgt.frame_id)
        corr = valid_token_correspondences(scene, src, tgt, grid, geom_cfg)
        num_corr = int(len(corr))
        row = {
            "scene_id": scene.scene_id,
            "src_index": src.index,
            "tgt_index": tgt.index,
            "src_frame": src.frame_id,
            "tgt_frame": tgt.frame_id,
            "bucket": bucket or "unbucketed",
            "translation_m": translation,
            "rotation_deg": rotation,
            "overlap_ratio": overlap,
            "num_valid_corr": num_corr,
        }
        if bucket is not None and overlap >= min_overlap and num_corr >= min_corr:
            candidates.append(row)

    rng = np.random.default_rng(seed)
    selected = []
    for bucket in buckets:
        rows = [r for r in candidates if r["bucket"] == bucket]
        rows = sorted(rows, key=lambda r: (-r["overlap_ratio"], r["translation_m"]))
        if len(rows) > per_bucket:
            # Keep high-overlap candidates but avoid fully deterministic adjacent-only ordering.
            top_pool = rows[: max(per_bucket * 4, per_bucket)]
            take_idx = rng.choice(len(top_pool), size=per_bucket, replace=False)
            rows = [top_pool[i] for i in sorted(take_idx)]
        selected.extend(rows[:per_bucket])

    if not selected:
        # Smoke fallback: use the best geometrically valid pairs even if buckets are sparse.
        fallback = []
        for src_i, tgt_i in combos:
            src = frames[src_i]
            tgt = frames[tgt_i]
            translation, rotation = relative_pose_stats(src, tgt)
            overlap = estimate_overlap(scene, src, tgt, geom_cfg, seed=seed + src.frame_id * 997 + tgt.frame_id)
            corr = valid_token_correspondences(scene, src, tgt, grid, geom_cfg)
            if len(corr) > 0:
                fallback.append(
                    {
                        "scene_id": scene.scene_id,
                        "src_index": src.index,
                        "tgt_index": tgt.index,
                        "src_frame": src.frame_id,
                        "tgt_frame": tgt.frame_id,
                        "bucket": "fallback",
                        "translation_m": translation,
                        "rotation_deg": rotation,
                        "overlap_ratio": overlap,
                        "num_valid_corr": int(len(corr)),
                    }
                )
        fallback = sorted(fallback, key=lambda r: (-r["overlap_ratio"], -r["num_valid_corr"]))
        selected = fallback[: max(1, per_bucket)]

    for pair_id, row in enumerate(selected):
        row["pair_id"] = f"pair_{pair_id:03d}"
    columns = [
        "pair_id",
        "scene_id",
        "src_frame",
        "tgt_frame",
        "bucket",
        "translation_m",
        "rotation_deg",
        "overlap_ratio",
        "num_valid_corr",
        "src_index",
        "tgt_index",
    ]
    return pd.DataFrame(selected, columns=columns)


def sample_query_sets(
    scene: SceneInfo,
    frames: list[FrameRecord],
    pairs_df: pd.DataFrame,
    grid: GridSpec,
    cfg: dict,
) -> dict[str, pd.DataFrame]:
    queries_per_pair = int(cfg["queries"].get("queries_per_pair", 512))
    seed = int(cfg.get("run", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    query_sets: dict[str, pd.DataFrame] = {}
    for row in tqdm(list(pairs_df.itertuples(index=False)), desc="Sampling query sets", unit="pair"):
        src = frames[int(row.src_index)]
        tgt = frames[int(row.tgt_index)]
        corr = valid_token_correspondences(scene, src, tgt, grid, cfg["geometry"])
        if len(corr) > queries_per_pair:
            idx = rng.choice(len(corr), size=queries_per_pair, replace=False)
            corr = corr.iloc[np.sort(idx)].copy()
        else:
            corr = corr.copy()
        corr.insert(0, "query_id", [f"q{i:04d}" for i in range(len(corr))])
        corr.insert(0, "pair_id", row.pair_id)
        corr["src_frame"] = int(row.src_frame)
        corr["tgt_frame"] = int(row.tgt_frame)
        query_sets[row.pair_id] = corr.reset_index(drop=True)
    return query_sets

