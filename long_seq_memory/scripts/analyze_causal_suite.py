#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, load_config, write_json  # noqa: E402
from long_seq_memory.trajectory_metrics import rotation_angle_deg  # noqa: E402


DEFAULT_RUNS = {
    "A_special_old_l4_8": "A_special_old_l4_8/reconstruction",
    "Ball_image_old_l4_23": "Ball_image_old_l4_23/reconstruction",
    "Blate_image_old_l17_23": "Blate_image_old_l17_23/reconstruction",
    "C_image_live_special_l8_23": "C_image_live_special_l8_23/reconstruction",
}


def load_pose_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape[-2:] == (3, 4):
        out = np.repeat(np.eye(4)[None], arr.shape[0], axis=0)
        out[:, :3, :4] = arr
        return out
    return arr


def stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def frame_mask(frame_ids: np.ndarray, segment: list[int] | tuple[int, int] | None) -> np.ndarray:
    if not segment:
        return np.ones_like(frame_ids, dtype=bool)
    lo, hi = int(segment[0]), int(segment[1])
    return (frame_ids >= lo) & (frame_ids <= hi)


def load_run(run_dir: Path) -> dict[str, Any]:
    required = {
        "frame_ids": run_dir / "frame_ids.npy",
        "pred": run_dir / "pred_poses_c2w_demo_convention.npy",
        "gt": run_dir / "gt_poses_c2w.npy",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        return {"status": "missing", "run_dir": str(run_dir), "missing": missing}
    return {
        "status": "complete",
        "run_dir": str(run_dir),
        "frame_ids": np.load(required["frame_ids"]),
        "pred": load_pose_array(required["pred"]),
        "gt": load_pose_array(required["gt"]),
    }


def pose_deltas(baseline: dict[str, Any], intervention: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    base_frames = baseline["frame_ids"]
    int_frames = intervention["frame_ids"]
    if not np.array_equal(base_frames, int_frames):
        raise ValueError("Baseline and intervention frame_ids differ; compare only aligned causal runs.")
    base_pred = baseline["pred"]
    int_pred = intervention["pred"]
    gt = baseline["gt"]
    trans_delta = np.linalg.norm(int_pred[:, :3, 3] - base_pred[:, :3, 3], axis=1)
    rot_delta = np.asarray(
        [rotation_angle_deg(base_pred[i, :3, :3], int_pred[i, :3, :3]) for i in range(len(base_frames))],
        dtype=np.float64,
    )
    base_gt_trans = np.linalg.norm(base_pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    int_gt_trans = np.linalg.norm(int_pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    loop_segment = cfg.get("loop_event", {}).get("current_segment")
    selected = [int(v) for v in cfg.get("representation_capture", {}).get("frames", [])]
    selected_rows = []
    frame_to_idx = {int(frame_id): idx for idx, frame_id in enumerate(base_frames.tolist())}
    for frame_id in selected:
        if frame_id not in frame_to_idx:
            continue
        idx = frame_to_idx[frame_id]
        selected_rows.append(
            {
                "frame_id": frame_id,
                "pose_delta_translation_m": float(trans_delta[idx]),
                "pose_delta_rotation_deg": float(rot_delta[idx]),
                "baseline_gt_translation_error_m": float(base_gt_trans[idx]),
                "intervention_gt_translation_error_m": float(int_gt_trans[idx]),
                "gt_translation_error_delta_m": float(int_gt_trans[idx] - base_gt_trans[idx]),
            }
        )
    loop = frame_mask(base_frames, loop_segment)
    return {
        "all_frames": {
            "pose_delta_translation_m": stats(trans_delta),
            "pose_delta_rotation_deg": stats(rot_delta),
            "gt_translation_error_delta_m": stats(int_gt_trans - base_gt_trans),
        },
        "loop_segment": {
            "segment": loop_segment,
            "pose_delta_translation_m": stats(trans_delta[loop]),
            "pose_delta_rotation_deg": stats(rot_delta[loop]),
            "gt_translation_error_delta_m": stats((int_gt_trans - base_gt_trans)[loop]),
        },
        "selected_frames": selected_rows,
        "series": {
            "frame_ids": base_frames.astype(int).tolist(),
            "pose_delta_translation_m": trans_delta.astype(float).tolist(),
            "pose_delta_rotation_deg": rot_delta.astype(float).tolist(),
            "gt_translation_error_delta_m": (int_gt_trans - base_gt_trans).astype(float).tolist(),
        },
    }


def token_groups(num_special: int, token_count: int) -> dict[str, slice]:
    return {
        "camera": slice(0, 1),
        "register": slice(1, min(5, num_special)),
        "scale": slice(min(5, num_special - 1), num_special),
        "special_all": slice(0, num_special),
        "image": slice(num_special, token_count),
        "all": slice(0, token_count),
    }


def compare_representation_dirs(base_dir: Path, int_dir: Path, num_special: int) -> list[dict[str, Any]]:
    if not base_dir.exists() or not int_dir.exists():
        return []
    rows = []
    for base_path in sorted(base_dir.glob("x_frame*_layer*.npz")):
        int_path = int_dir / base_path.name
        if not int_path.exists():
            continue
        with np.load(base_path) as zb, np.load(int_path) as zi:
            xb = zb["tokens"].astype(np.float32)
            xi = zi["tokens"].astype(np.float32)
            frame_id = int(zb["frame_id"])
            layer_id = int(zb["layer_id"])
        if xb.shape != xi.shape:
            continue
        groups = token_groups(num_special, xb.shape[0])
        for group, sl in groups.items():
            base = xb[sl]
            other = xi[sl]
            if base.size == 0:
                continue
            delta = other - base
            per_token_l2 = np.linalg.norm(delta, axis=1)
            base_l2 = np.linalg.norm(base, axis=1)
            rows.append(
                {
                    "frame_id": frame_id,
                    "layer_id": layer_id,
                    "token_group": group,
                    "num_tokens": int(base.shape[0]),
                    "delta_l2_mean": float(per_token_l2.mean()),
                    "delta_l2_median": float(np.median(per_token_l2)),
                    "relative_delta_l2_mean": float(per_token_l2.mean() / max(float(base_l2.mean()), 1e-12)),
                    "delta_l2_max": float(per_token_l2.max()),
                }
            )
    return rows


def compare_dense_outputs(base_dir: Path, int_dir: Path) -> list[dict[str, Any]]:
    dense_base = base_dir / "dense"
    dense_int = int_dir / "dense"
    if not dense_base.exists() or not dense_int.exists():
        return []
    try:
        import torch
    except Exception:
        return []
    rows = []
    for base_path in sorted(dense_base.glob("*.pt")):
        key = base_path.stem
        int_path = dense_int / base_path.name
        base_ids_path = dense_base / f"{key}_frame_ids.npy"
        int_ids_path = dense_int / f"{key}_frame_ids.npy"
        if not int_path.exists() or not base_ids_path.exists() or not int_ids_path.exists():
            continue
        base_ids = np.load(base_ids_path)
        int_ids = np.load(int_ids_path)
        common = sorted(set(base_ids.astype(int).tolist()) & set(int_ids.astype(int).tolist()))
        if not common:
            continue
        xb = torch.load(base_path, map_location="cpu").float()
        xi = torch.load(int_path, map_location="cpu").float()
        base_idx = {int(frame_id): idx for idx, frame_id in enumerate(base_ids.tolist())}
        int_idx = {int(frame_id): idx for idx, frame_id in enumerate(int_ids.tolist())}
        values = []
        for frame_id in common:
            b = xb[:, base_idx[frame_id]]
            i = xi[:, int_idx[frame_id]]
            delta = i - b
            values.append(
                {
                    "frame_id": int(frame_id),
                    "mean_abs": float(delta.abs().mean()),
                    "rmse": float(torch.sqrt((delta * delta).mean())),
                    "relative_rmse": float(torch.linalg.norm(delta) / max(float(torch.linalg.norm(b)), 1e-12)),
                }
            )
        rows.append(
            {
                "key": key,
                "num_common_frames": len(common),
                "frames": values,
                "mean_abs": stats(np.asarray([v["mean_abs"] for v in values])),
                "rmse": stats(np.asarray([v["rmse"] for v in values])),
                "relative_rmse": stats(np.asarray([v["relative_rmse"] for v in values])),
            }
        )
    return rows


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def query_old_mass(record: dict[str, Any], query: str) -> float | None:
    payload = record.get("aggregation", {}).get(query)
    if not payload:
        return None
    values = payload.get("old_memory_mean_attention_mass_by_head_query")
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def summarize_interaction_dir(interaction_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    global_path = interaction_dir / "global_summary" / "global_summary.jsonl"
    patch_path = interaction_dir / "patch_specific" / "patch_specific.jsonl"
    loop_segment = cfg.get("loop_event", {}).get("current_segment")
    by_query: dict[str, list[float]] = defaultdict(list)
    by_query_loop: dict[str, list[float]] = defaultdict(list)
    records = 0
    for record in iter_jsonl(global_path) or []:
        records += 1
        frame_id = int(record.get("frame_id", -1))
        in_loop = bool(loop_segment and int(loop_segment[0]) <= frame_id <= int(loop_segment[1]))
        for query in ["camera_query", "register_queries", "scale_query", "image_queries", "all_queries"]:
            value = query_old_mass(record, query)
            if value is None:
                continue
            by_query[query].append(value)
            if in_loop:
                by_query_loop[query].append(value)
    patch_records = 0
    patch_old_mass_cv = []
    for record in iter_jsonl(patch_path) or []:
        patch_records += 1
        for query_payload in record.get("aggregation", {}).values():
            values = query_payload.get("old_memory_mean_attention_mass_by_head_query")
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float64)
            if arr.size and abs(float(arr.mean())) > 1e-12:
                patch_old_mass_cv.append(float(arr.std() / abs(float(arr.mean()))))
    return {
        "global_summary_records": records,
        "patch_specific_records": patch_records,
        "old_memory_mass_by_query": {query: stats(np.asarray(values)) for query, values in by_query.items()},
        "loop_old_memory_mass_by_query": {query: stats(np.asarray(values)) for query, values in by_query_loop.items()},
        "patch_specific_head_cv_proxy": stats(np.asarray(patch_old_mass_cv)),
    }


def interaction_mean_deltas(baseline: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scope in ["old_memory_mass_by_query", "loop_old_memory_mass_by_query"]:
        out[scope] = {}
        base_scope = baseline.get(scope, {})
        int_scope = intervention.get(scope, {})
        for query, base_stats in base_scope.items():
            int_stats = int_scope.get(query)
            if not int_stats:
                continue
            base_mean = base_stats.get("mean")
            int_mean = int_stats.get("mean")
            if base_mean is None or int_mean is None:
                continue
            out[scope][query] = {
                "baseline_mean": float(base_mean),
                "intervention_mean": float(int_mean),
                "delta": float(int_mean - base_mean),
                "ratio": float(int_mean / max(abs(float(base_mean)), 1e-12)),
            }
    base_cv = baseline.get("patch_specific_head_cv_proxy", {}).get("mean")
    int_cv = intervention.get("patch_specific_head_cv_proxy", {}).get("mean")
    out["patch_specific_head_cv_proxy"] = (
        None
        if base_cv is None or int_cv is None
        else {
            "baseline_mean": float(base_cv),
            "intervention_mean": float(int_cv),
            "delta": float(int_cv - base_cv),
            "ratio": float(int_cv / max(abs(float(base_cv)), 1e-12)),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10})
    return plt


def plot_pose_series(out_dir: Path, name: str, pose: dict[str, Any]) -> str:
    plt = setup_matplotlib()
    frames = np.asarray(pose["series"]["frame_ids"])
    trans = np.asarray(pose["series"]["pose_delta_translation_m"])
    rot = np.asarray(pose["series"]["pose_delta_rotation_deg"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(frames, trans, linewidth=1)
    axes[0].set_ylabel("translation delta (m)")
    axes[0].set_title(f"{name}: intervention - baseline pose delta")
    axes[1].plot(frames, rot, linewidth=1, color="tab:orange")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("rotation delta (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = ensure_dir(out_dir / "figures") / f"{name}_pose_delta_series.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def plot_representation_profiles(out_dir: Path, name: str, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    plt = setup_matplotlib()
    groups = ["camera", "register", "scale", "image"]
    by_group_layer: dict[str, dict[int, list[float]]] = {group: defaultdict(list) for group in groups}
    for row in rows:
        group = row["token_group"]
        if group in by_group_layer:
            by_group_layer[group][int(row["layer_id"])].append(float(row["relative_delta_l2_mean"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    for group in groups:
        layers = sorted(by_group_layer[group])
        if not layers:
            continue
        values = [float(np.mean(by_group_layer[group][layer])) for layer in layers]
        ax.plot(layers, values, marker="o", linewidth=1.5, label=group)
    ax.set_title(f"{name}: representation relative delta by layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean ||X_int - X_base|| / mean ||X_base||")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = ensure_dir(out_dir / "figures") / f"{name}_representation_delta_by_layer.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def default_run_dirs(suite_root: Path) -> dict[str, Path]:
    return {name: suite_root / rel for name, rel in DEFAULT_RUNS.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", default=str(EXPERIMENT_ROOT / "outputs" / "causal_branch_4414"))
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "configs" / "causal_branch_4414_baseline.yaml"))
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--intervention", action="append", default=[], help="NAME=/path/to/reconstruction")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    suite_root = Path(args.suite_root).expanduser()
    baseline_dir = Path(args.baseline_dir).expanduser() if args.baseline_dir else suite_root / "baseline" / "reconstruction"
    intervention_dirs = default_run_dirs(suite_root)
    for item in args.intervention:
        name, value = item.split("=", 1)
        intervention_dirs[name] = Path(value).expanduser()
    output_dir = ensure_dir(args.output_dir or suite_root / "analysis")

    baseline = load_run(baseline_dir)
    report: dict[str, Any] = {
        "status": "complete" if baseline["status"] == "complete" else "missing_baseline",
        "baseline_dir": str(baseline_dir),
        "interventions": {},
    }
    figures = []
    if baseline["status"] != "complete":
        report["baseline_missing"] = baseline["missing"]
        write_json(output_dir / "causal_suite_summary.json", report)
        print(f"Missing baseline outputs. Wrote {output_dir / 'causal_suite_summary.json'}")
        return

    num_special = int(cfg.get("memory_policy", {}).get("patch_start_idx", 6))
    baseline_interaction = summarize_interaction_dir(baseline_dir.parent / "interaction", cfg)
    report["baseline_interaction_summary"] = baseline_interaction
    all_rep_rows = []
    for name, run_dir in intervention_dirs.items():
        intervention = load_run(run_dir)
        if intervention["status"] != "complete":
            report["interventions"][name] = {
                "status": "missing",
                "run_dir": str(run_dir),
                "missing": intervention["missing"],
            }
            continue
        pose = pose_deltas(baseline, intervention, cfg)
        dense = compare_dense_outputs(baseline_dir, run_dir)
        rep_rows = compare_representation_dirs(baseline_dir / "representations", run_dir / "representations", num_special)
        for row in rep_rows:
            row["intervention"] = name
        all_rep_rows.extend(rep_rows)
        interaction = summarize_interaction_dir(run_dir.parent / "interaction", cfg)
        report["interventions"][name] = {
            "status": "complete",
            "run_dir": str(run_dir),
            "pose": {key: value for key, value in pose.items() if key != "series"},
            "dense_outputs": dense,
            "representation_delta_records": len(rep_rows),
            "interaction_summary": interaction,
            "interaction_delta_from_baseline": interaction_mean_deltas(baseline_interaction, interaction),
        }
        figures.append(plot_pose_series(output_dir, name, pose))
        rep_fig = plot_representation_profiles(output_dir, name, rep_rows)
        if rep_fig:
            figures.append(rep_fig)

    write_csv(output_dir / "representation_delta.csv", all_rep_rows)
    report["figures"] = figures
    write_json(output_dir / "causal_suite_summary.json", report)
    (output_dir / "causal_suite_report.md").write_text(
        "\n".join(
            [
                "# Causal Suite Summary",
                "",
                f"- Baseline: `{baseline_dir}`",
                f"- Status: `{report['status']}`",
                f"- Figures: `{len(figures)}`",
                "",
                "## Interventions",
                *[
                    f"- `{name}`: `{payload['status']}`"
                    for name, payload in report["interventions"].items()
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'causal_suite_summary.json'}")
    print(f"Wrote {output_dir / 'causal_suite_report.md'}")


if __name__ == "__main__":
    main()
