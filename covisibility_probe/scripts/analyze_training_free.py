#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch

from covisibility_probe.config import load_config
from covisibility_probe.confound_analysis import residualized_rank_correlation
from covisibility_probe.metrics import (
    bootstrap_ci,
    centered_kernel_alignment,
    matrix_from_descriptor,
    ndcg_at_k,
    pose_similarity,
    spearmanr,
)
from covisibility_probe.overlap import load_overlap_npz
from covisibility_probe.paths import resolve_run_dir
from covisibility_probe.register_similarity import (
    cosine_similarity_matrix,
    register_mean_descriptors,
    register_mean_similarity_matrix,
    register_slot_similarity_matrix,
)
from covisibility_probe.scannet_io import rotation_angle_deg, translation_distance
from covisibility_probe.utils import ensure_dir, read_parquet, save_json, setup_logging, write_parquet


def _np(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _layer_label(layer: int) -> str:
    return f"l{int(layer):02d}"


def _determine_pair_type(i: int, j: int, *, local_window: int, anchor_count: int) -> tuple[str, bool]:
    if i < anchor_count or j < anchor_count:
        return "anchor_related", False
    if (j - i) > local_window:
        return "trajectory", True
    return "local", False


def _feature_matrices(features: dict[str, Any], layers: list[int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrices: dict[str, np.ndarray] = {}
    descriptors: dict[str, np.ndarray] = {}
    dino = _np(features.get("official_dino_pool"))
    if dino is not None:
        matrices["dino_official_similarity"] = cosine_similarity_matrix(dino)
        descriptors["Official DINOv2|"] = dino
    backbone = _np(features.get("lingbot_backbone_pool"))
    if backbone is not None:
        matrices["lingbot_backbone_similarity"] = cosine_similarity_matrix(backbone)
        descriptors["LingBot backbone|"] = backbone

    register_sources = [
        ("register_head_frame", "Register head frame-half"),
        ("register_head_global", "Register head global-half"),
        ("register_head_concat", "Register head concat"),
    ]
    camera_sources = [
        ("camera_head_frame", "Camera head frame-half"),
        ("camera_head_global", "Camera head global-half"),
        ("camera_head_concat", "Camera head concat"),
    ]
    for layer in layers:
        label = _layer_label(layer)
        for key, feature_name in register_sources:
            reg_dict = features.get(key) or {}
            if layer not in reg_dict:
                continue
            reg = _np(reg_dict[layer])
            matrices[f"{key}_slot_similarity_{label}"] = register_slot_similarity_matrix(reg)
            matrices[f"{key}_mean_similarity_{label}"] = register_mean_similarity_matrix(reg)
            descriptors[f"{feature_name}|{layer}"] = reg.reshape(reg.shape[0], -1)
        for key, feature_name in camera_sources:
            cam_dict = features.get(key) or {}
            if layer not in cam_dict or cam_dict[layer] is None:
                continue
            cam = _np(cam_dict[layer])
            matrices[f"{key}_similarity_{label}"] = cosine_similarity_matrix(cam)
            descriptors[f"{feature_name}|{layer}"] = cam
    return matrices, descriptors


def _build_scene_pairs(cfg: dict[str, Any], run_dir: Path, scene_id: str, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    layers = [int(x) for x in cfg["features"]["layers"]]
    features = torch.load(run_dir / "features" / f"{scene_id}.pt", map_location="cpu", weights_only=False)
    overlap = load_overlap_npz(run_dir / "overlap" / f"{scene_id}_overlap.npz")
    df_scene = manifest[manifest["scene_id"] == scene_id].sort_values("input_position").reset_index(drop=True)
    frame_ids = df_scene["frame_id"].astype(int).to_numpy()
    if not np.array_equal(frame_ids, _np(features["frame_ids"]).astype(int)):
        raise RuntimeError(f"Feature frame ids differ from manifest for {scene_id}")
    if not np.array_equal(frame_ids, overlap.frame_ids.astype(int)):
        raise RuntimeError(f"Overlap frame ids differ from manifest for {scene_id}")

    matrices, descriptors = _feature_matrices(features, layers)
    gt_pose = _np(features.get("gt_pose_c2w"))
    n = len(frame_ids)
    local_window = int(cfg.get("analysis", {}).get("local_window", cfg["model"].get("local_window", 32)))
    anchor_count = int(cfg.get("analysis", {}).get("anchor_frame_count", cfg["model"].get("num_scale_frames", 8)))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_type, is_traj = _determine_pair_type(i, j, local_window=local_window, anchor_count=anchor_count)
            trans = translation_distance(gt_pose[i], gt_pose[j]) if gt_pose is not None else float("nan")
            rot = rotation_angle_deg(gt_pose[i], gt_pose[j]) if gt_pose is not None else float("nan")
            row: dict[str, Any] = {
                "scene_id": scene_id,
                "frame_i": int(frame_ids[i]),
                "frame_j": int(frame_ids[j]),
                "input_position_i": int(i),
                "input_position_j": int(j),
                "temporal_gap": int(j - i),
                "pair_type": pair_type,
                "is_trajectory_only": bool(is_traj),
                "overlap_cos": float(overlap.overlap_cos[i, j]),
                "overlap_iou": float(overlap.overlap_iou[i, j]),
                "coverage_i_to_j": float(overlap.coverage_i_to_j[i, j]),
                "coverage_j_to_i": float(overlap.coverage_i_to_j[j, i]),
                "gt_translation_distance": float(trans),
                "gt_rotation_angle_deg": float(rot),
                "pred_translation_distance": float("nan"),
                "pred_rotation_angle_deg": float("nan"),
            }
            for name, mat in matrices.items():
                row[name] = float(mat[i, j])
            for layer in layers:
                label = _layer_label(layer)
                for key in ("camera_head_frame", "camera_head_global", "camera_head_concat"):
                    row.setdefault(f"{key}_similarity_{label}", float("nan"))
                for key in ("register_head_frame", "register_head_global", "register_head_concat"):
                    row.setdefault(f"{key}_slot_similarity_{label}", float("nan"))
                    row.setdefault(f"{key}_mean_similarity_{label}", float("nan"))
            row.setdefault("dino_official_similarity", float("nan"))
            row.setdefault("lingbot_backbone_similarity", float("nan"))
            rows.append(row)
    pairs = pd.DataFrame(rows)
    pairs["time_baseline_similarity"] = -np.log1p(pairs["temporal_gap"].to_numpy(dtype=np.float64))
    pairs["gt_pose_baseline_similarity"] = pose_similarity(
        pairs["gt_translation_distance"].to_numpy(dtype=np.float64),
        pairs["gt_rotation_angle_deg"].to_numpy(dtype=np.float64),
    )
    return pairs, descriptors


def _feature_specs(layers: list[int]) -> list[dict[str, Any]]:
    specs = [
        {"feature": "Time baseline", "layer": "", "column": "time_baseline_similarity", "residualize": False, "residual_controls_dino": False, "cka": False},
        {"feature": "GT pose baseline", "layer": "", "column": "gt_pose_baseline_similarity", "residualize": False, "residual_controls_dino": False, "cka": False},
        {"feature": "Official DINOv2", "layer": "", "column": "dino_official_similarity", "residual_controls_dino": False, "cka": True},
        {"feature": "LingBot backbone", "layer": "", "column": "lingbot_backbone_similarity", "residual_controls_dino": True, "cka": True},
    ]
    for layer in layers:
        label = _layer_label(layer)
        specs.append({"feature": "Camera head concat", "layer": layer, "column": f"camera_head_concat_similarity_{label}", "residual_controls_dino": True, "cka": True})
    for layer in layers:
        label = _layer_label(layer)
        specs.append({"feature": "Register head frame-half", "layer": layer, "column": f"register_head_frame_slot_similarity_{label}", "residual_controls_dino": True, "cka": True})
    for layer in layers:
        label = _layer_label(layer)
        specs.append({"feature": "Register head global-half", "layer": layer, "column": f"register_head_global_slot_similarity_{label}", "residual_controls_dino": True, "cka": True})
    for layer in layers:
        label = _layer_label(layer)
        specs.append({"feature": "Register head concat", "layer": layer, "column": f"register_head_concat_slot_similarity_{label}", "residual_controls_dino": True, "cka": True})
    return specs


def _scene_metrics(scene_pairs: pd.DataFrame, descriptors: dict[str, np.ndarray], overlap_matrix: np.ndarray, specs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    traj = scene_pairs[scene_pairs["is_trajectory_only"]].copy()
    ndcg_k = int(cfg.get("analysis", {}).get("ndcg_k", 5))
    for spec in specs:
        col = spec["column"]
        if col not in traj:
            continue
        sim = traj[col].to_numpy(dtype=np.float64)
        ov = traj["overlap_cos"].to_numpy(dtype=np.float64)
        covariates = {
            "log1p_temporal_gap": np.log1p(traj["temporal_gap"].to_numpy(dtype=np.float64)),
            "gt_translation_distance": traj["gt_translation_distance"].to_numpy(dtype=np.float64),
            "gt_rotation_angle_deg": traj["gt_rotation_angle_deg"].to_numpy(dtype=np.float64),
        }
        if spec.get("residual_controls_dino", True) and "dino_official_similarity" in traj:
            covariates["dino_official_similarity"] = traj["dino_official_similarity"].to_numpy(dtype=np.float64)
        residual = (
            residualized_rank_correlation(sim, ov, covariates)
            if spec.get("residualize", True)
            else float("nan")
        )
        ndcgs = []
        for _, group in traj.groupby("input_position_j"):
            if len(group) > 0:
                ndcgs.append(ndcg_at_k(group[col].to_numpy(dtype=np.float64), group["overlap_cos"].to_numpy(dtype=np.float64), k=ndcg_k))
        cka = float("nan")
        if spec.get("cka"):
            key = f"{spec['feature']}|{spec['layer']}"
            desc = descriptors.get(key)
            if desc is not None:
                cka = centered_kernel_alignment(matrix_from_descriptor(desc), overlap_matrix)
        rows.append(
            {
                "scene_id": scene_pairs["scene_id"].iloc[0],
                "feature": spec["feature"],
                "layer": "" if spec["layer"] == "" else str(spec["layer"]),
                "similarity_column": col,
                "num_trajectory_pairs": int(len(traj)),
                "spearman": spearmanr(sim, ov),
                "residualized_rank_correlation": residual,
                f"ndcg@{ndcg_k}": float(np.nanmean(ndcgs)) if ndcgs else float("nan"),
                "cka": cka,
            }
        )
    return rows


def _summarize(per_scene: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    metrics = ["spearman", "residualized_rank_correlation", f"ndcg@{int(cfg.get('analysis', {}).get('ndcg_k', 5))}", "cka"]
    rows = []
    for (feature, layer), group in per_scene.groupby(["feature", "layer"], dropna=False):
        row = {"Feature": feature, "Layer": layer if layer != "" else "-"}
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=np.float64)
            row[metric] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            lo, hi = bootstrap_ci(vals, iterations=int(cfg.get("analysis", {}).get("bootstrap_iterations", 1000)), seed=int(cfg["run"].get("seed", 42)))
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    order = {spec["feature"]: idx for idx, spec in enumerate(_feature_specs([4, 11, 17, 23]))}
    out = pd.DataFrame(rows)
    out["_order"] = out["Feature"].map(lambda x: order.get(x, 999))
    out["_layer_sort"] = out["Layer"].map(lambda x: -1 if x == "-" else int(x))
    out = out.sort_values(["_order", "_layer_sort"]).drop(columns=["_order", "_layer_sort"]).reset_index(drop=True)
    return out


def _write_markdown(summary: pd.DataFrame, path: Path) -> None:
    cols = ["Feature", "Layer", "spearman", "residualized_rank_correlation", "ndcg@5", "cka"]
    existing = [c for c in cols if c in summary.columns]
    lines = ["| " + " | ".join(existing) + " |", "| " + " | ".join(["---"] * len(existing)) + " |"]
    for _, row in summary.iterrows():
        vals = []
        for col in existing:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if not np.isfinite(val) else f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bin_label(lo: float, hi: float | None) -> str:
    return f"{lo:g}+" if hi is None else f"{lo:g}-{hi:g}"


def _time_bin_metrics(all_pairs: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    layers = [int(x) for x in cfg["features"]["layers"]]
    specs = [s for s in _feature_specs(layers) if s["feature"].startswith("Register")]
    rows = []
    bins = cfg.get("analysis", {}).get("time_gap_bins", [])
    for spec in specs:
        col = spec["column"]
        for scene_id, scene_df in all_pairs[all_pairs["is_trajectory_only"]].groupby("scene_id"):
            for lo, hi in bins:
                lo = float(lo)
                mask = scene_df["temporal_gap"] >= lo
                if hi is not None:
                    mask &= scene_df["temporal_gap"] <= float(hi)
                sub = scene_df[mask]
                if len(sub) == 0:
                    continue
                rows.append(
                    {
                        "scene_id": scene_id,
                        "feature": spec["feature"],
                        "layer": "" if spec["layer"] == "" else str(spec["layer"]),
                        "similarity_column": col,
                        "time_gap_bin": _bin_label(lo, hi),
                        "num_pairs": int(len(sub)),
                        "spearman": spearmanr(sub[col].to_numpy(dtype=np.float64), sub["overlap_cos"].to_numpy(dtype=np.float64)),
                    }
                )
    return pd.DataFrame(rows)


def _time_overlap_grid(all_pairs: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    layer = int(cfg.get("analysis", {}).get("representative_layer", 17))
    col = f"register_head_concat_slot_similarity_{_layer_label(layer)}"
    rows = []
    traj = all_pairs[all_pairs["is_trajectory_only"]]
    for gap_lo, gap_hi in cfg.get("analysis", {}).get("time_gap_bins", []):
        gap_mask = traj["temporal_gap"] >= float(gap_lo)
        if gap_hi is not None:
            gap_mask &= traj["temporal_gap"] <= float(gap_hi)
        for ov_lo, ov_hi in cfg.get("analysis", {}).get("overlap_bins", []):
            ov_mask = (traj["overlap_cos"] >= float(ov_lo)) & (traj["overlap_cos"] < float(ov_hi))
            if float(ov_hi) >= 1.0:
                ov_mask = (traj["overlap_cos"] >= float(ov_lo)) & (traj["overlap_cos"] <= float(ov_hi))
            sub = traj[gap_mask & ov_mask]
            rows.append(
                {
                    "layer": layer,
                    "similarity_column": col,
                    "time_gap_bin": _bin_label(float(gap_lo), gap_hi),
                    "overlap_bin": _bin_label(float(ov_lo), float(ov_hi)),
                    "num_pairs": int(len(sub)),
                    "mean_register_similarity": float(sub[col].mean()) if len(sub) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def analyze(config_path: str | Path, *, run_id: str | None = None, overwrite: bool = False) -> Path:
    cfg = load_config(config_path)
    run_dir = resolve_run_dir(cfg, run_id=run_id)
    manifest = read_parquet(run_dir / "manifests" / "frames.parquet")
    ensure_dir(run_dir / "pairs" / "pairs_by_scene")
    ensure_dir(run_dir / "metrics")
    layers = [int(x) for x in cfg["features"]["layers"]]
    specs = _feature_specs(layers)

    all_pair_dfs = []
    per_scene_rows = []
    for scene_id in manifest["scene_id"].drop_duplicates().tolist():
        pair_path = run_dir / "pairs" / "pairs_by_scene" / f"{scene_id}.parquet"
        if pair_path.is_file() and not overwrite:
            pairs = read_parquet(pair_path)
        else:
            pairs, descriptors = _build_scene_pairs(cfg, run_dir, scene_id, manifest)
            write_parquet(pairs, pair_path)
        all_pair_dfs.append(pairs)
        if "descriptors" not in locals() or pair_path.is_file() and not overwrite:
            features = torch.load(run_dir / "features" / f"{scene_id}.pt", map_location="cpu", weights_only=False)
            _, descriptors = _feature_matrices(features, layers)
        overlap = load_overlap_npz(run_dir / "overlap" / f"{scene_id}_overlap.npz")
        per_scene_rows.extend(_scene_metrics(pairs, descriptors, overlap.overlap_cos, specs, cfg))
        print(f"Analyzed scene: {scene_id}")

    all_pairs = pd.concat(all_pair_dfs, ignore_index=True)
    write_parquet(all_pairs, run_dir / "pairs" / "all_pairs.parquet")
    per_scene = pd.DataFrame(per_scene_rows)
    write_parquet(per_scene, run_dir / "metrics" / "per_scene_metrics.parquet")
    summary = _summarize(per_scene, cfg)
    summary.to_csv(run_dir / "metrics" / "summary_metrics.csv", index=False)
    _write_markdown(summary, run_dir / "metrics" / "summary_metrics.md")
    time_bins = _time_bin_metrics(all_pairs, cfg)
    time_bins.to_csv(run_dir / "metrics" / "time_bin_metrics.csv", index=False)
    grid = _time_overlap_grid(all_pairs, cfg)
    grid.to_csv(run_dir / "metrics" / "time_overlap_grid.csv", index=False)
    save_json(
        {
            "num_pairs": int(len(all_pairs)),
            "num_trajectory_pairs": int(all_pairs["is_trajectory_only"].sum()),
            "pair_type_counts": all_pairs["pair_type"].value_counts().to_dict(),
            "summary_metrics": str(run_dir / "metrics" / "summary_metrics.csv"),
        },
        run_dir / "metrics" / "analysis_summary.json",
    )
    print(f"All pairs written: {run_dir / 'pairs' / 'all_pairs.parquet'}")
    print(f"Summary written: {run_dir / 'metrics' / 'summary_metrics.md'}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    setup_logging()
    analyze(args.config, run_id=args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
