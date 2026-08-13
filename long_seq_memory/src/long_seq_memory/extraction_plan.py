from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from long_seq_memory.dataset import load_scene
from long_seq_memory.io_utils import ensure_dir, write_json
from long_seq_memory.schedule import build_keyframe_schedule, schedule_config_from_dict, written_frame_ids


def inclusive_range(span: list[int] | tuple[int, int], stride: int = 1) -> list[int]:
    start, end = int(span[0]), int(span[1])
    return list(range(start, end + 1, max(int(stride), 1)))


def representative_layers(requested: list[int], depth: int) -> list[int]:
    requested = [int(layer) for layer in requested]
    if requested and min(requested) >= 0 and max(requested) < depth:
        return requested
    if depth <= 1:
        return [0]
    anchors = [0.2, 0.45, 0.7, 1.0]
    return sorted(set(min(depth - 1, round((depth - 1) * value)) for value in anchors))


def choose_control_frames(cfg: dict[str, Any], count: int = 3) -> list[int]:
    scene = load_scene(cfg["dataset_root"])
    poses = scene.poses_c2w
    loop = cfg.get("loop_event", {})
    history = int(loop.get("history_frame", 3403))
    current = int(loop.get("current_frame", 4414))
    current_segment = tuple(loop.get("current_segment", [4380, 4445]))
    history_segment = tuple(loop.get("history_segment", [3370, 3440]))
    centers = poses[:, :3, 3]
    loop_motion = float(np.linalg.norm(centers[current] - centers[current - 1])) if current > 0 else 0.0
    thirds = np.linspace(0, len(centers) - 1, count + 2, dtype=int)[1:-1]
    controls = []
    for anchor in thirds:
        lo = max(8, int(anchor) - 400)
        hi = min(len(centers) - 1, int(anchor) + 400)
        candidates = []
        for frame_id in range(lo, hi + 1):
            if current_segment[0] <= frame_id <= current_segment[1]:
                continue
            if history_segment[0] <= frame_id <= history_segment[1]:
                continue
            if abs(frame_id - current) < 500 or abs(frame_id - history) < 500:
                continue
            if np.linalg.norm(centers[frame_id] - centers[history]) < 2.0:
                continue
            if np.linalg.norm(centers[frame_id] - centers[current]) < 2.0:
                continue
            motion = float(np.linalg.norm(centers[frame_id] - centers[frame_id - 1])) if frame_id > 0 else 0.0
            candidates.append((abs(motion - loop_motion), abs(frame_id - int(anchor)), frame_id))
        if candidates:
            controls.append(int(sorted(candidates)[0][2]))
    controls = sorted(set(controls))
    if len(controls) >= count:
        return controls[:count]

    fallback = []
    for frame_id in range(8, len(centers)):
        if frame_id in controls:
            continue
        if current_segment[0] <= frame_id <= current_segment[1]:
            continue
        if history_segment[0] <= frame_id <= history_segment[1]:
            continue
        if abs(frame_id - current) < 300 or abs(frame_id - history) < 300:
            continue
        if np.linalg.norm(centers[frame_id] - centers[history]) < 1.0:
            continue
        if np.linalg.norm(centers[frame_id] - centers[current]) < 1.0:
            continue
        motion = float(np.linalg.norm(centers[frame_id] - centers[frame_id - 1])) if frame_id > 0 else 0.0
        spread_penalty = min(abs(frame_id - picked) for picked in controls) if controls else 0
        fallback.append((abs(motion - loop_motion), -spread_penalty, frame_id))
    for _, _, frame_id in sorted(fallback):
        controls.append(int(frame_id))
        controls = sorted(set(controls))
        if len(controls) >= count:
            break
    return controls[:count]


def build_extraction_plan(cfg: dict[str, Any]) -> dict[str, Any]:
    extraction = cfg.get("extraction", {})
    scfg = schedule_config_from_dict(cfg)
    schedule_rows = build_keyframe_schedule(scfg)
    depth = int(cfg.get("model_profile", {}).get("num_gca_layers", cfg.get("model_defaults", {}).get("num_gca_layers", 24)))
    rep_layers = representative_layers(list(extraction.get("representative_layers", [4, 11, 17, 23])), depth)
    source_frames = [int(row["source_frame_id"]) for row in schedule_rows]
    stride = int(extraction.get("global_summary_stride", 10))
    loop_range = extraction.get("loop_current_range", cfg.get("loop_event", {}).get("current_segment", [4380, 4445]))
    target_range = extraction.get("target_history_range", cfg.get("loop_event", {}).get("history_segment", [3370, 3440]))
    loop_current_frames = inclusive_range(loop_range, int(extraction.get("loop_current_stride", 1)))
    retained_history = [
        frame_id
        for frame_id in written_frame_ids(schedule_rows, max(loop_current_frames) if loop_current_frames else None)
        if int(target_range[0]) <= frame_id <= int(target_range[1])
    ]
    selected = [int(v) for v in extraction.get("selected_current_frames", [4390, 4400, 4408, 4414, 4420, 4430])]
    controls = choose_control_frames(cfg, int(extraction.get("auto_control_frames", 3)))
    raw_requested = [int(v) for v in extraction.get("raw_qkv_frames", [4400, 4414, 4430])]
    raw_max = int(extraction.get("raw_qkv_max_frames", 4))
    raw_frames = sorted(set(raw_requested + controls[: max(0, raw_max - len(set(raw_requested)))]))[:raw_max]
    return {
        "status": "complete",
        "run_name": cfg.get("run_name"),
        "source_frame_id_convention": "original Oxford frame id, zero-based",
        "levels": {
            "global_summary": {
                "frames": [frame for frame in source_frames if (frame - source_frames[0]) % max(stride, 1) == 0],
                "layers": list(range(depth)),
                "stores_raw_qkv": False,
            },
            "loop_dense": {
                "current_frames": loop_current_frames,
                "target_history_range": list(target_range),
                "retained_history_frames": retained_history,
                "layers": list(range(depth)),
                "stores_raw_qkv": False,
            },
            "patch_specific": {
                "selected_current_frames": selected,
                "auto_control_frames": controls,
                "layers": rep_layers,
                "stores_raw_qkv": False,
            },
            "raw_qkv_debug": {
                "frames": raw_frames,
                "layers": rep_layers,
                "budget_gb": float(extraction.get("raw_qkv_budget_gb", 8)),
                "storage_dtype": extraction.get("raw_qkv_storage_dtype", "float16"),
            },
        },
    }


def write_extraction_plan(cfg: dict[str, Any]) -> dict[str, str]:
    plan = build_extraction_plan(cfg)
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "planning" / cfg.get("run_name", "run"))
    write_json(out_dir / "feature_extraction_plan.json", plan)
    return {"output_dir": str(out_dir), "plan": str(out_dir / "feature_extraction_plan.json")}
