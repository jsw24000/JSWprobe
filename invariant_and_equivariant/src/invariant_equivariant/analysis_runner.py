from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from invariant_equivariant.data.coordinates import delta_from_pair
from invariant_equivariant.data.manifests import filter_pairs, filter_triplets, load_manifests
from invariant_equivariant.features.cache import load_feature_vector
from invariant_equivariant.features.indexing import read_feature_index
from invariant_equivariant.metrics.geometry import effective_rank, orthonormal_basis, subspace_metrics
from invariant_equivariant.metrics.regression import direction_metrics, feature_forward_metrics, regression_metrics
from invariant_equivariant.metrics.retrieval import nearest_state_metrics
from invariant_equivariant.probes.ridge import Standardizer, choose_alpha, fit_ridge, model_to_npz
from invariant_equivariant.utils import ensure_dir, load_config, read_json, write_csv, write_json


def _bool(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def metric_row(config: Mapping[str, Any], model: str, variant: str, layer: str, metric: str, value: float, split: str = "test", coord: str = "ref", ntrain: int = 0, nval: int = 0, ntest: int = 0, feature_space: str = "raw_standardized") -> Dict[str, Any]:
    return {
        "run_name": config["run_name"],
        "model_name": model,
        "model_variant": variant,
        "layer_name": layer,
        "feature_space": feature_space,
        "pooling": config["pooling"]["method"],
        "coordinate_system": coord,
        "split_name": split,
        "seed": int(config["seed"]),
        "metric_name": metric,
        "metric_value": float(value),
        "num_train": int(ntrain),
        "num_val": int(nval),
        "num_test": int(ntest),
    }


def grouped_feature_rows(output_dir: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    rows = [dict(row) for row in read_feature_index(output_dir) if _bool(row["valid"])]
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["layer_name"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: (r["scene_id"], r["state_id"], r["camera_id"]))
    return grouped


def load_x(rows: Sequence[Mapping[str, Any]], feature_key: str = "pooled_raw") -> np.ndarray:
    return np.vstack([load_feature_vector(row["feature_path"], key=feature_key) for row in rows])


def load_y(rows: Sequence[Mapping[str, Any]], coord: str) -> np.ndarray:
    prefix = {"ref": "position_ref", "current": "position_current", "world": "position_world"}[coord]
    return np.asarray([[float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"])] for row in rows], dtype=np.float64)


def masks(rows: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    return {split: np.asarray([row["split"] == split for row in rows], dtype=bool) for split in ["train", "validation", "test"]}


def simple_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, axis in enumerate("xyz"):
        axes[i].scatter(y_true[:, i], y_pred[:, i], s=10, alpha=0.75)
        lo = min(float(y_true[:, i].min()), float(y_pred[:, i].min()))
        hi = max(float(y_true[:, i].max()), float(y_pred[:, i].max()))
        axes[i].plot([lo, hi], [lo, hi], c="black", linewidth=1)
        axes[i].set_title(axis)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_absolute(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["_output_dir"])
    out = ensure_dir(output_dir / "absolute_position")
    grouped = grouped_feature_rows(output_dir)
    alphas = config["analysis"]["ridge_alphas"]
    metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    for (model_name, layer_name), rows in grouped.items():
        variant = rows[0]["model_variant"]
        x = load_x(rows)
        split_masks = masks(rows)
        for coord in config["analysis"]["coordinate_systems"]:
            y = load_y(rows, coord)
            if split_masks["train"].sum() < 2 or split_masks["validation"].sum() < 1 or split_masks["test"].sum() < 1:
                continue
            model = choose_alpha(x[split_masks["train"]], y[split_masks["train"]], x[split_masks["validation"]], y[split_masks["validation"]], alphas)
            pred = model.predict(x)
            ntrain, nval, ntest = int(split_masks["train"].sum()), int(split_masks["validation"].sum()), int(split_masks["test"].sum())
            for split in ["train", "validation", "test"]:
                m = split_masks[split]
                for name, value in regression_metrics(y[m], pred[m]).items():
                    metric_rows.append(metric_row(config, model_name, variant, layer_name, name, value, split, coord, ntrain, nval, ntest))
            for idx, row in enumerate(rows):
                pred_rows.append(
                    {
                        "model_name": model_name,
                        "layer_name": layer_name,
                        "coordinate_system": coord,
                        "scene_id": row["scene_id"],
                        "state_id": row["state_id"],
                        "camera_id": row["camera_id"],
                        "split": row["split"],
                        "gt_x": y[idx, 0],
                        "gt_y": y[idx, 1],
                        "gt_z": y[idx, 2],
                        "pred_x": pred[idx, 0],
                        "pred_y": pred[idx, 1],
                        "pred_z": pred[idx, 2],
                    }
                )
            if coord == config["analysis"]["main_coordinate_system"]:
                model_to_npz(out / f"{model_name}_{layer_name}_ref_probe_weights.npz", model, {"coord": coord})
                s = np.linalg.svd(model.weights.T, compute_uv=False)
                write_csv(out / f"{model_name}_{layer_name}_singular_values.csv", [{"index": i, "singular_value": float(v)} for i, v in enumerate(s)])
                simple_scatter(y[split_masks["test"]], pred[split_masks["test"]], out / f"scatter_{model_name}_{layer_name}_{coord}.png", f"{model_name} {layer_name} {coord}")

    # 2D mask/bbox baseline for reference coordinate.
    all_rows = next(iter(grouped.values())) if grouped else []
    if all_rows:
        xb = np.asarray(
            [
                [float(row["mask_centroid_x"]), float(row["mask_centroid_y"]), float(row["bbox_width"]), float(row["bbox_height"]), float(row["mask_area_ratio"]), float(row["camera_numeric"])]
                for row in all_rows
            ],
            dtype=np.float64,
        )
        split_masks = masks(all_rows)
        y = load_y(all_rows, "ref")
        baseline = choose_alpha(xb[split_masks["train"]], y[split_masks["train"]], xb[split_masks["validation"]], y[split_masks["validation"]], alphas)
        pred = baseline.predict(xb)
        for split in ["train", "validation", "test"]:
            m = split_masks[split]
            for name, value in regression_metrics(y[m], pred[m]).items():
                metric_rows.append(metric_row(config, "2d_mask_bbox_baseline", "mask_bbox_camera_numeric", "mask_bbox", name, value, split, "ref", int(split_masks["train"].sum()), int(split_masks["validation"].sum()), int(split_masks["test"].sum()), "2d"))

    write_csv(out / "metrics.csv", metric_rows)
    write_json(out / "metrics.json", {"metrics": metric_rows})
    write_csv(out / "predictions.csv", pred_rows)
    (out / "report.md").write_text(f"# Absolute Position\n\nRows: {len(metric_rows)}\n", encoding="utf-8")
    return {"metrics": len(metric_rows), "predictions": len(pred_rows)}


def pair_arrays(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], coord: str = "ref", feature_key: str = "pooled_raw"):
    manifests = load_manifests(config)
    valid_keys = {(r["scene_id"], r["state_id"], r["camera_id"]) for r in rows}
    pairs = filter_pairs(config, manifests, valid_keys)
    fmap = {(r["scene_id"], r["state_id"], r["camera_id"]): load_feature_vector(r["feature_path"], key=feature_key) for r in rows}
    x = []
    y = []
    used = []
    for pair in pairs:
        ka = (pair["scene_id"], pair["state_a"], pair["camera_id"])
        kb = (pair["scene_id"], pair["state_b"], pair["camera_id"])
        if ka not in fmap or kb not in fmap:
            continue
        x.append(fmap[kb] - fmap[ka])
        y.append(delta_from_pair(pair, coord))
        used.append(pair)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), used


def pair_arrays_2d(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], coord: str = "ref"):
    manifests = load_manifests(config)
    valid_keys = {(r["scene_id"], r["state_id"], r["camera_id"]) for r in rows}
    pairs = filter_pairs(config, manifests, valid_keys)

    def vec(row: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                float(row["mask_centroid_x"]),
                float(row["mask_centroid_y"]),
                float(row["bbox_width"]),
                float(row["bbox_height"]),
                float(row["mask_area_ratio"]),
                float(row["camera_numeric"]),
            ],
            dtype=np.float64,
        )

    fmap = {(r["scene_id"], r["state_id"], r["camera_id"]): vec(r) for r in rows}
    x = []
    y = []
    used = []
    for pair in pairs:
        ka = (pair["scene_id"], pair["state_a"], pair["camera_id"])
        kb = (pair["scene_id"], pair["state_b"], pair["camera_id"])
        if ka not in fmap or kb not in fmap:
            continue
        delta_2d = fmap[kb] - fmap[ka]
        x.append(np.asarray([*delta_2d[:5], fmap[ka][-1]], dtype=np.float64))
        y.append(delta_from_pair(pair, coord))
        used.append(pair)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), used


def run_delta_decode(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["_output_dir"])
    out = ensure_dir(output_dir / "delta_decode")
    grouped = grouped_feature_rows(output_dir)
    metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    alphas = config["analysis"]["ridge_alphas"]
    rng = np.random.default_rng(int(config["analysis"]["random_seed"]))
    for (model_name, layer_name), rows in grouped.items():
        variant = rows[0]["model_variant"]
        x, y, pairs = pair_arrays(config, rows, "ref")
        if len(pairs) == 0:
            continue
        split_masks = {split: np.asarray([p["split"] == split for p in pairs], dtype=bool) for split in ["train", "validation", "test"]}
        model = choose_alpha(x[split_masks["train"]], y[split_masks["train"]], x[split_masks["validation"]], y[split_masks["validation"]], alphas)
        pred = model.predict(x)
        ntrain, nval, ntest = int(split_masks["train"].sum()), int(split_masks["validation"].sum()), int(split_masks["test"].sum())
        for split in ["train", "validation", "test"]:
            m = split_masks[split]
            metrics = {**regression_metrics(y[m], pred[m]), **direction_metrics(y[m], pred[m])}
            for name, value in metrics.items():
                metric_rows.append(metric_row(config, model_name, variant, layer_name, name, value, split, "ref", ntrain, nval, ntest))
        for idx, pair in enumerate(pairs):
            pred_rows.append({"model_name": model_name, "layer_name": layer_name, "pair_id": pair["pair_id"], "split": pair["split"], "gt_dx": y[idx, 0], "gt_dy": y[idx, 1], "gt_dz": y[idx, 2], "pred_dx": pred[idx, 0], "pred_dy": pred[idx, 1], "pred_dz": pred[idx, 2], "same_displacement_group": pair["same_displacement_group"], "camera_id": pair["camera_id"], "distance": pair["distance"]})
        model_to_npz(out / f"{model_name}_{layer_name}_delta_probe_weights.npz", model, {"coord": "ref"})
        s = np.linalg.svd(model.weights.T, compute_uv=False)
        write_csv(out / f"{model_name}_{layer_name}_singular_values.csv", [{"index": i, "singular_value": float(v)} for i, v in enumerate(s)])

        shuffled_y_train = y[split_masks["train"]].copy()
        rng.shuffle(shuffled_y_train, axis=0)
        shuffled = choose_alpha(x[split_masks["train"]], shuffled_y_train, x[split_masks["validation"]], y[split_masks["validation"]], alphas)
        shuffled_pred = shuffled.predict(x)
        for name, value in {**regression_metrics(y[split_masks["test"]], shuffled_pred[split_masks["test"]]), **direction_metrics(y[split_masks["test"]], shuffled_pred[split_masks["test"]])}.items():
            metric_rows.append(metric_row(config, f"{model_name}_shuffled_label_baseline", variant, layer_name, name, value, "test", "ref", ntrain, nval, ntest))

        zero = np.zeros_like(y)
        for name, value in {**regression_metrics(y[split_masks["test"]], zero[split_masks["test"]]), **direction_metrics(y[split_masks["test"]], zero[split_masks["test"]])}.items():
            metric_rows.append(metric_row(config, f"{model_name}_zero_baseline", variant, layer_name, name, value, "test", "ref", ntrain, nval, ntest))

    if grouped:
        baseline_rows = next(iter(grouped.values()))
        xb, yb, pairs_b = pair_arrays_2d(config, baseline_rows, "ref")
        split_masks = {split: np.asarray([p["split"] == split for p in pairs_b], dtype=bool) for split in ["train", "validation", "test"]}
        if split_masks["train"].sum() > 0 and split_masks["validation"].sum() > 0 and split_masks["test"].sum() > 0:
            model = choose_alpha(xb[split_masks["train"]], yb[split_masks["train"]], xb[split_masks["validation"]], yb[split_masks["validation"]], alphas)
            pred = model.predict(xb)
            ntrain, nval, ntest = int(split_masks["train"].sum()), int(split_masks["validation"].sum()), int(split_masks["test"].sum())
            for name, value in {**regression_metrics(yb[split_masks["test"]], pred[split_masks["test"]]), **direction_metrics(yb[split_masks["test"]], pred[split_masks["test"]])}.items():
                metric_rows.append(metric_row(config, "2d_mask_bbox_delta_baseline", "mask_bbox_camera_numeric", "mask_bbox_delta", name, value, "test", "ref", ntrain, nval, ntest, "2d"))
    write_csv(out / "metrics.csv", metric_rows)
    write_json(out / "metrics.json", {"metrics": metric_rows})
    write_csv(out / "predictions.csv", pred_rows)
    (out / "report.md").write_text(f"# Delta Decode\n\nRows: {len(metric_rows)}\n", encoding="utf-8")
    return {"metrics": len(metric_rows), "predictions": len(pred_rows)}


def features_by_key(rows: Sequence[Mapping[str, Any]], feature_key: str = "pooled_raw", standardize: bool = True):
    x = load_x(rows, feature_key=feature_key)
    std = None
    if standardize:
        train_mask = np.asarray([row["split"] == "train" for row in rows])
        std = Standardizer.fit(x[train_mask])
        x = std.transform(x)
    return {(row["scene_id"], row["state_id"], row["camera_id"]): x[i] for i, row in enumerate(rows)}, x, std


def standardized_features_by_key(rows: Sequence[Mapping[str, Any]]):
    return features_by_key(rows, feature_key="pooled_raw", standardize=True)


def forward_arrays(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], feature_key: str = "pooled_raw", standardize: bool = True):
    manifests = load_manifests(config)
    valid_keys = {(r["scene_id"], r["state_id"], r["camera_id"]) for r in rows}
    pairs = filter_pairs(config, manifests, valid_keys)
    fmap, _xs, _std = features_by_key(rows, feature_key=feature_key, standardize=standardize)
    x = []
    y = []
    used = []
    for pair in pairs:
        ka = (pair["scene_id"], pair["state_a"], pair["camera_id"])
        kb = (pair["scene_id"], pair["state_b"], pair["camera_id"])
        if ka not in fmap or kb not in fmap:
            continue
        x.append(delta_from_pair(pair, "ref"))
        y.append(fmap[kb] - fmap[ka])
        used.append(pair)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), used, fmap


def run_forward(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["_output_dir"])
    out = ensure_dir(output_dir / "forward_equivariance")
    grouped = grouped_feature_rows(output_dir)
    metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    alphas = config["analysis"]["ridge_alphas"]
    for (model_name, layer_name), rows in grouped.items():
        variant = rows[0]["model_variant"]
        variants = [
            ("raw", "pooled_raw", False),
            ("raw_standardized", "pooled_raw", True),
            ("l2_normalized", "pooled_l2", False),
        ]
        for feature_space, feature_key, standardize in variants:
            x, y, pairs, fmap = forward_arrays(config, rows, feature_key=feature_key, standardize=standardize)
            split_masks = {split: np.asarray([p["split"] == split for p in pairs], dtype=bool) for split in ["train", "validation", "test"]}
            if split_masks["train"].sum() < 1 or split_masks["validation"].sum() < 1 or split_masks["test"].sum() < 1:
                continue
            model = choose_alpha(x[split_masks["train"]], y[split_masks["train"]], x[split_masks["validation"]], y[split_masks["validation"]], alphas, standardize_x=False, fit_intercept=False)
            pred = model.predict(x)
            ntrain, nval, ntest = int(split_masks["train"].sum()), int(split_masks["validation"].sum()), int(split_masks["test"].sum())
            for split in ["train", "validation", "test"]:
                m = split_masks[split]
                for name, value in feature_forward_metrics(y[m], pred[m]).items():
                    metric_rows.append(metric_row(config, model_name, variant, layer_name, name, value, split, "ref", ntrain, nval, ntest, feature_space))
            if feature_space != "raw_standardized":
                continue
            model_to_npz(out / f"{model_name}_{layer_name}_forward_B.npz", model, {"coord": "ref", "feature_space": feature_space, "B_shape": list(model.weights.T.shape)})
            by_scene_camera = defaultdict(list)
            for row in rows:
                by_scene_camera[(row["scene_id"], row["camera_id"])].append(row)
            layer_pred_rows = []
            identity_rows = []
            for idx, pair in enumerate(pairs):
                if pair["split"] != "test":
                    continue
                key_a = (pair["scene_id"], pair["state_a"], pair["camera_id"])
                key_b = (pair["scene_id"], pair["state_b"], pair["camera_id"])
                candidates = by_scene_camera[(pair["scene_id"], pair["camera_id"])]
                cand_feat = np.vstack([fmap[(r["scene_id"], r["state_id"], r["camera_id"])] for r in candidates])
                zhat = fmap[key_a] + pred[idx]
                ret = nearest_state_metrics(zhat, key_b, candidates, cand_feat)
                row = {"model_name": model_name, "layer_name": layer_name, "pair_id": pair["pair_id"], "baseline": "forward_map", **ret}
                pred_rows.append(row)
                layer_pred_rows.append(row)
                identity = nearest_state_metrics(fmap[key_a], key_b, candidates, cand_feat)
                id_row = {"model_name": model_name, "layer_name": layer_name, "pair_id": pair["pair_id"], "baseline": "identity", **identity}
                pred_rows.append(id_row)
                identity_rows.append(id_row)
            for prefix, rows_for_metrics in [("state_retrieval", layer_pred_rows), ("identity_state_retrieval", identity_rows)]:
                for name in ["top1", "top3", "grid_distance"]:
                    vals = [r[name] for r in rows_for_metrics if not math.isnan(float(r[name]))]
                    if vals:
                        metric_rows.append(metric_row(config, model_name, variant, layer_name, f"{prefix}_{name}", float(np.mean(vals)), "test", "ref", ntrain, nval, ntest, feature_space))
    write_csv(out / "metrics.csv", metric_rows)
    write_json(out / "metrics.json", {"metrics": metric_rows})
    write_csv(out / "predictions.csv", pred_rows)
    (out / "report.md").write_text(f"# Forward Equivariance\n\nRows: {len(metric_rows)}\n", encoding="utf-8")
    return {"metrics": len(metric_rows), "predictions": len(pred_rows)}


def run_homogeneity(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["_output_dir"])
    out = ensure_dir(output_dir / "homogeneity")
    grouped = grouped_feature_rows(output_dir)
    metric_rows: List[Dict[str, Any]] = []
    for (model_name, layer_name), rows in grouped.items():
        variant = rows[0]["model_variant"]
        _delta_p, dz, pairs2, _ = forward_arrays(config, rows)
        if len(pairs2) == 0:
            continue
        split_counts = {split: sum(1 for p in pairs2 if p["split"] == split) for split in ["train", "validation", "test"]}
        groups = defaultdict(list)
        for idx, pair in enumerate(pairs2):
            if pair["split"] == "test":
                groups[pair["same_displacement_group"]].append(dz[idx])
        if len(groups) < 2:
            continue
        means = {k: np.mean(v, axis=0) for k, v in groups.items() if len(v) > 0}
        overall = np.mean(list(means.values()), axis=0)
        within = np.mean([np.mean(np.sum((np.vstack(v) - means[k]) ** 2, axis=1)) for k, v in groups.items() if len(v) > 0])
        between = np.mean([np.sum((mu - overall) ** 2) for mu in means.values()])
        h = within / max(1e-12, between)
        for name, value in {"within_variance": within, "between_variance": between, "homogeneity_ratio": h}.items():
            metric_rows.append(metric_row(config, model_name, variant, layer_name, name, value, "test", ntrain=split_counts["train"], nval=split_counts["validation"], ntest=split_counts["test"]))
        # Same/different cosine estimates.
        same_cos = []
        diff_cos = []
        for g, vals in groups.items():
            vals = np.vstack(vals)
            if len(vals) >= 2:
                for i in range(len(vals) - 1):
                    a, b = vals[i], vals[i + 1]
                    same_cos.append(float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))))
        group_items = list(groups.items())
        for (g1, v1), (g2, v2) in zip(group_items[:-1], group_items[1:]):
            a, b = v1[0], v2[0]
            diff_cos.append(float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))))
        if same_cos:
            metric_rows.append(metric_row(config, model_name, variant, layer_name, "same_displacement_cosine", float(np.mean(same_cos)), "test", ntrain=split_counts["train"], nval=split_counts["validation"], ntest=split_counts["test"]))
        if diff_cos:
            metric_rows.append(metric_row(config, model_name, variant, layer_name, "different_displacement_cosine", float(np.mean(diff_cos)), "test", ntrain=split_counts["train"], nval=split_counts["validation"], ntest=split_counts["test"]))
    write_csv(out / "metrics.csv", metric_rows)
    write_json(out / "metrics.json", {"metrics": metric_rows})
    (out / "report.md").write_text(f"# Homogeneity\n\nRows: {len(metric_rows)}\n", encoding="utf-8")
    return {"metrics": len(metric_rows)}


def run_subspace(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["_output_dir"])
    out = ensure_dir(output_dir / "subspaces")
    grouped = grouped_feature_rows(output_dir)
    metric_rows: List[Dict[str, Any]] = []
    rank = int(config["analysis"]["subspace_rank"])
    for (model_name, layer_name), rows in grouped.items():
        variant = rows[0]["model_variant"]
        abs_path = output_dir / "absolute_position" / f"{model_name}_{layer_name}_ref_probe_weights.npz"
        fwd_path = output_dir / "forward_equivariance" / f"{model_name}_{layer_name}_forward_B.npz"
        if not abs_path.exists() or not fwd_path.exists():
            continue
        split_counts = {split: sum(1 for r in rows if r["split"] == split) for split in ["train", "validation", "test"]}
        abs_npz = np.load(abs_path)
        fwd_npz = np.load(fwd_path)
        w_abs = abs_npz["weights"].T  # [D, 3]
        b = fwd_npz["weights"].T  # [D, 3]
        s_read = np.linalg.svd(w_abs.T, compute_uv=False)
        s_move = np.linalg.svd(b, compute_uv=False)
        q_read = orthonormal_basis(w_abs.T, rank)
        q_move = orthonormal_basis(b, rank)
        np.save(out / f"{model_name}_{layer_name}_Q_read.npy", q_read)
        np.save(out / f"{model_name}_{layer_name}_Q_move.npy", q_move)
        write_csv(out / f"{model_name}_{layer_name}_singular_values_read.csv", [{"index": i, "singular_value": float(v)} for i, v in enumerate(s_read)])
        write_csv(out / f"{model_name}_{layer_name}_singular_values_move.csv", [{"index": i, "singular_value": float(v)} for i, v in enumerate(s_move)])
        metrics = subspace_metrics(q_read, q_move)
        metrics["effective_rank_read"] = effective_rank(s_read, float(config["analysis"]["effective_rank_threshold"]))
        metrics["effective_rank_move"] = effective_rank(s_move, float(config["analysis"]["effective_rank_threshold"]))
        for name, value in metrics.items():
            metric_rows.append(metric_row(config, model_name, variant, layer_name, name, value, "test", ntrain=split_counts["train"], nval=split_counts["validation"], ntest=split_counts["test"]))
    write_csv(out / "subspace_metrics.csv", metric_rows)
    write_json(out / "metrics.json", {"metrics": metric_rows})
    (out / "report.md").write_text(f"# Subspaces\n\nRows: {len(metric_rows)}\n", encoding="utf-8")
    return {"metrics": len(metric_rows)}


def transform_with_subspace(x_std: np.ndarray, q: np.ndarray, mode: str) -> np.ndarray:
    proj = x_std @ q @ q.T
    if mode.startswith("keep"):
        return proj
    if mode.startswith("remove"):
        return x_std - proj
    return x_std


def run_intervention(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["_output_dir"])
    out = ensure_dir(output_dir / "intervention")
    grouped = grouped_feature_rows(output_dir)
    metric_rows: List[Dict[str, Any]] = []
    alphas = config["analysis"]["ridge_alphas"]
    rng = np.random.default_rng(int(config["analysis"]["random_seed"]))
    for (model_name, layer_name), rows in grouped.items():
        variant = rows[0]["model_variant"]
        q_paths = {
            "read": output_dir / "subspaces" / f"{model_name}_{layer_name}_Q_read.npy",
            "move": output_dir / "subspaces" / f"{model_name}_{layer_name}_Q_move.npy",
        }
        if not all(p.exists() for p in q_paths.values()):
            continue
        x = load_x(rows)
        train_mask = np.asarray([r["split"] == "train" for r in rows])
        std = Standardizer.fit(x[train_mask])
        xstd = std.transform(x)
        y = load_y(rows, "ref")
        split_masks = masks(rows)
        transforms = {"raw": xstd}
        for name, path in q_paths.items():
            q = np.load(path)
            transforms[f"keep_{name}_subspace"] = transform_with_subspace(xstd, q, "keep")
            transforms[f"remove_{name}_subspace"] = transform_with_subspace(xstd, q, "remove")
        rank = int(config["analysis"]["subspace_rank"])
        for seed in range(int(config["analysis"]["random_subspace_repeats"])):
            rand = rng.normal(size=(xstd.shape[1], rank))
            q, _ = np.linalg.qr(rand)
            transforms[f"random_rank_matched_{seed}"] = transform_with_subspace(xstd, q[:, :rank], "remove")
        for feature_space, xt in transforms.items():
            model = choose_alpha(xt[split_masks["train"]], y[split_masks["train"]], xt[split_masks["validation"]], y[split_masks["validation"]], alphas, standardize_x=False)
            pred = model.predict(xt)
            for mname, value in regression_metrics(y[split_masks["test"]], pred[split_masks["test"]]).items():
                metric_rows.append(metric_row(config, model_name, variant, layer_name, mname, value, "test", "ref", int(split_masks["train"].sum()), int(split_masks["validation"].sum()), int(split_masks["test"].sum()), feature_space))
            # Same scene/camera across-position residual similarity.
            sims = []
            by_group = defaultdict(list)
            for idx, row in enumerate(rows):
                if row["split"] == "test":
                    by_group[(row["scene_id"], row["camera_id"])].append(idx)
            for inds in by_group.values():
                for i, j in itertools.combinations(inds, 2):
                    a, b = xt[i], xt[j]
                    sims.append(float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))))
            if sims:
                metric_rows.append(metric_row(config, model_name, variant, layer_name, "same_scene_cross_position_cosine", float(np.mean(sims)), "test", "ref", feature_space=feature_space))
    write_csv(out / "metrics.csv", metric_rows)
    write_json(out / "metrics.json", {"metrics": metric_rows})
    (out / "report.md").write_text(f"# Intervention\n\nRows: {len(metric_rows)}\n", encoding="utf-8")
    return {"metrics": len(metric_rows)}


def run_composition(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["_output_dir"])
    out = ensure_dir(output_dir / "composition")
    grouped = grouped_feature_rows(output_dir)
    manifests = load_manifests(config)
    metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    primitive_tags = set(config["composition"]["primitive_tags"])
    composite_types = set(config["composition"]["composite_pair_types"])
    alphas = config["analysis"]["ridge_alphas"]
    for (model_name, layer_name), rows in grouped.items():
        variant = rows[0]["model_variant"]
        x, y, pairs, fmap = forward_arrays(config, rows)
        primitive_mask = np.asarray([p["split"] == "train" and p["pair_type"] == "grid_adjacent" and bool(primitive_tags.intersection(p.get("pair_tags", []))) for p in pairs], dtype=bool)
        val_mask = np.asarray([p["split"] == "validation" and p["pair_type"] == "grid_adjacent" for p in pairs], dtype=bool)
        comp_mask = np.asarray([p["split"] == "test" and (p["pair_type"] in composite_types or "diagonal" in p.get("pair_tags", [])) for p in pairs], dtype=bool)
        if primitive_mask.sum() < 1 or val_mask.sum() < 1 or comp_mask.sum() < 1:
            continue
        model = choose_alpha(x[primitive_mask], y[primitive_mask], x[val_mask], y[val_mask], alphas, standardize_x=False, fit_intercept=False)
        pred = model.predict(x)
        for name, value in feature_forward_metrics(y[comp_mask], pred[comp_mask]).items():
            metric_rows.append(metric_row(config, model_name, variant, layer_name, f"composite_{name}", value, "test", "ref", int(primitive_mask.sum()), int(val_mask.sum()), int(comp_mask.sum())))
        # Triplet real rendered composite states.
        valid_keys = {(r["scene_id"], r["state_id"], r["camera_id"]) for r in rows}
        triplets = filter_triplets(config, manifests, valid_keys)
        by_scene_camera = defaultdict(list)
        for row in rows:
            by_scene_camera[(row["scene_id"], row["camera_id"])].append(row)
        for triplet in triplets:
            if triplet["split"] != "test":
                continue
            k0 = (triplet["scene_id"], triplet["state_0"], triplet["camera_id"])
            k2 = (triplet["scene_id"], triplet["state_2"], triplet["camera_id"])
            if k0 not in fmap or k2 not in fmap:
                continue
            delta = np.asarray(triplet["delta_02_ref_camera"], dtype=np.float64)
            zhat = fmap[k0] + model.predict(delta[None, :])[0]
            ztrue = fmap[k2]
            err = float(np.sum((ztrue - zhat) ** 2) / max(1e-12, np.sum((ztrue - fmap[k0]) ** 2)))
            cos = float(np.dot(ztrue - fmap[k0], zhat - fmap[k0]) / max(1e-12, np.linalg.norm(ztrue - fmap[k0]) * np.linalg.norm(zhat - fmap[k0])))
            candidates = by_scene_camera[(triplet["scene_id"], triplet["camera_id"])]
            cand_feat = np.vstack([fmap[(r["scene_id"], r["state_id"], r["camera_id"])] for r in candidates])
            ret = nearest_state_metrics(zhat, k2, candidates, cand_feat)
            pred_rows.append({"model_name": model_name, "layer_name": layer_name, "triplet_id": triplet["triplet_id"], "normalized_feature_error": err, "cosine": cos, **ret})
        triplet_rows = [r for r in pred_rows if r["model_name"] == model_name and r["layer_name"] == layer_name]
        for name in ["normalized_feature_error", "cosine", "top1", "top3", "grid_distance"]:
            vals = [float(r[name]) for r in triplet_rows if not math.isnan(float(r[name]))]
            if vals:
                metric_rows.append(metric_row(config, model_name, variant, layer_name, f"triplet_{name}", float(np.mean(vals)), "test", "ref", int(primitive_mask.sum()), int(val_mask.sum()), len(triplet_rows)))
    write_csv(out / "metrics.csv", metric_rows)
    write_json(out / "metrics.json", {"metrics": metric_rows})
    write_csv(out / "predictions.csv", pred_rows)
    (out / "report.md").write_text(f"# Composition\n\nRows: {len(metric_rows)}\n", encoding="utf-8")
    return {"metrics": len(metric_rows), "predictions": len(pred_rows)}
