#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
SRC = EXP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from invariant_equivariant.data.manifests import filter_frames, filter_pairs, filter_triplets, load_manifests, state_rank
from invariant_equivariant.data.validation import validate_frame_files
from invariant_equivariant.features.pooling import area_weights
from invariant_equivariant.utils import ensure_dir, load_config, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def plot_position_grid(frames: List[Dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    by_scene = defaultdict(list)
    seen = set()
    for frame in frames:
        key = (frame["scene_id"], frame["state_id"])
        if key in seen:
            continue
        seen.add(key)
        by_scene[frame["scene_id"]].append(frame)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), squeeze=False)
    for ax, (scene_id, rows) in zip(axes.ravel(), sorted(by_scene.items())):
        xs = [r["object_center_world"][0] for r in rows]
        ys = [r["object_center_world"][1] for r in rows]
        labels = [r["state_id"].replace("state_", "s") for r in rows]
        ax.scatter(xs, ys, c="tab:red")
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y, label, fontsize=8)
        ax.set_title(scene_id)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_camera_layout(frames: List[Dict[str, Any]], manifests: Dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = {(f["scene_id"], f["camera_id"]) for f in frames}
    cameras = [c for c in manifests["cameras"] if (c["scene_id"], c["camera_id"]) in selected]
    by_scene = defaultdict(list)
    for camera in cameras:
        by_scene[camera["scene_id"]].append(camera)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), squeeze=False)
    for ax, (scene_id, rows) in zip(axes.ravel(), sorted(by_scene.items())):
        for row in rows:
            x, y = row["position"][0], row["position"][1]
            lx, ly = row["look_at"][0], row["look_at"][1]
            ax.scatter([x], [y], c="tab:blue")
            ax.plot([x, lx], [y, ly], c="tab:blue", linewidth=1)
            ax.text(x, y, row["camera_id"].replace("camera_", "c"), fontsize=8)
        ax.set_title(scene_id)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def audit(config_path: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["_output_dir"])
    audit_dir = ensure_dir(output_dir / "audit")
    manifests = load_manifests(config)
    frames = filter_frames(config, manifests)
    pairs = filter_pairs(config, manifests)
    triplets = filter_triplets(config, manifests)

    invalid = validate_frame_files(frames)
    mask_rows: List[Dict[str, Any]] = []
    for frame in frames:
        weights = area_weights(frame["target_mask"], config["input_size"] // config["patch_size"], config["input_size"] // config["patch_size"])
        mask_rows.append(
            {
                "scene_id": frame["scene_id"],
                "state_id": frame["state_id"],
                "camera_id": frame["camera_id"],
                "mask_area_ratio": frame["target_mask_area_ratio"],
                "patch_coverage_sum": float(weights.sum().item()),
                "effective_patch_count": int((weights > 0.05).sum().item()),
                "visibility_ok": frame.get("visibility_ok", False),
            }
        )

    states_selected = [
        s for s in manifests["states"]
        if s["scene_id"] in config["selection"]["scenes"]
        and (config["selection"]["position_indices"] == "all" or tuple(s["position_index"]) in {tuple(x) for x in config["selection"]["position_indices"]})
    ]
    rank = state_rank(states_selected)

    def vector_rank(vectors: List[Any]) -> Dict[str, Any]:
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.size == 0:
            return {"rank": 0, "singular_values": []}
        centered = arr - arr.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        rank_value = int((singular_values > max(1e-9, singular_values[0] * 1e-6)).sum()) if singular_values.size else 0
        return {"rank": rank_value, "singular_values": singular_values.tolist()}

    world_delta_rank = vector_rank([p["delta_world"] for p in pairs])
    ref_delta_rank = vector_rank([p["delta_ref_camera"] for p in pairs])

    coord_rows: List[Dict[str, Any]] = []
    state_lookup = {(s["scene_id"], s["state_id"]): s for s in manifests["states"]}
    for pair in pairs:
        a = np.asarray(state_lookup[(pair["scene_id"], pair["state_a"])]["object_center_world"], dtype=np.float64)
        b = np.asarray(state_lookup[(pair["scene_id"], pair["state_b"])]["object_center_world"], dtype=np.float64)
        saved = np.asarray(pair["delta_world"], dtype=np.float64)
        coord_rows.append(
            {
                "pair_id": pair["pair_id"],
                "max_abs_delta_world_error": float(np.max(np.abs((b - a) - saved))),
                "split": pair["split"],
            }
        )

    disp_counter = Counter(pair["same_displacement_group"] for pair in pairs)
    pair_type_counter = Counter(pair["pair_type"] for pair in pairs)
    tag_counter = Counter(tag for pair in pairs for tag in pair.get("pair_tags", []))
    triplet_failures = sum(0 if t.get("is_additively_consistent") else 1 for t in triplets)
    camera_sets = defaultdict(set)
    for frame in frames:
        camera_sets[(frame["scene_id"], frame["state_id"])].add(frame["camera_id"])
    shared_camera_ok = len({tuple(sorted(v)) for v in camera_sets.values()}) == 1

    write_csv(audit_dir / "invalid_samples.csv", invalid)
    write_csv(audit_dir / "coordinate_checks.csv", coord_rows)
    write_csv(
        audit_dir / "displacement_statistics.csv",
        [{"same_displacement_group": k, "count": v} for k, v in sorted(disp_counter.items())],
    )
    write_csv(audit_dir / "mask_statistics.csv", mask_rows)
    plot_position_grid(frames, audit_dir / "position_grid.png")
    plot_camera_layout(frames, manifests, audit_dir / "camera_layout.png")

    summary = {
        "scene_count": len(set(f["scene_id"] for f in frames)),
        "state_count": len(set((f["scene_id"], f["state_id"]) for f in frames)),
        "camera_count": len(set(f["camera_id"] for f in frames)),
        "frame_count": len(frames),
        "pair_count": len(pairs),
        "triplet_count": len(triplets),
        "invalid_file_count": len(invalid),
        "shared_camera_set_ok": shared_camera_ok,
        "coordinate_max_error": max((r["max_abs_delta_world_error"] for r in coord_rows), default=0.0),
        "triplet_failure_count": triplet_failures,
        "motion_rank_ref_camera": rank["rank"],
        "motion_singular_values_ref_camera": rank["singular_values"],
        "displacement_rank_world": world_delta_rank["rank"],
        "displacement_singular_values_world": world_delta_rank["singular_values"],
        "displacement_rank_ref_camera": ref_delta_rank["rank"],
        "displacement_singular_values_ref_camera": ref_delta_rank["singular_values"],
        "pair_type_counts": dict(pair_type_counter),
        "pair_tag_counts": dict(tag_counter),
        "displacement_group_count": len(disp_counter),
        "mask_area_ratio_min": float(min(r["mask_area_ratio"] for r in mask_rows)) if mask_rows else None,
        "mask_area_ratio_mean": float(np.mean([r["mask_area_ratio"] for r in mask_rows])) if mask_rows else None,
        "patch_coverage_min": float(min(r["patch_coverage_sum"] for r in mask_rows)) if mask_rows else None,
    }
    write_json(audit_dir / "audit_summary.json", summary)
    (audit_dir / "audit_report.md").write_text(
        "# Audit Report\n\n"
        f"- Frames: {summary['frame_count']}\n"
        f"- Pairs: {summary['pair_count']}\n"
        f"- Triplets: {summary['triplet_count']}\n"
        f"- Shared camera set OK: {summary['shared_camera_set_ok']}\n"
        f"- Motion rank in reference camera coordinates: {summary['motion_rank_ref_camera']}\n"
        f"- Displacement rank in world coordinates: {summary['displacement_rank_world']}\n"
        f"- Displacement rank in reference camera coordinates: {summary['displacement_rank_ref_camera']}\n"
        f"- Invalid files: {summary['invalid_file_count']}\n"
        f"- Triplet failures: {summary['triplet_failure_count']}\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = audit(args.config)
    print(summary)


if __name__ == "__main__":
    main()
