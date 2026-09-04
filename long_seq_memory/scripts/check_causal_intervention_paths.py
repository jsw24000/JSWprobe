#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from long_seq_memory.io_utils import ensure_dir, load_config, write_json  # noqa: E402
from long_seq_memory.memory_index import memory_state_summary, preprocessed_patch_grid  # noqa: E402
from long_seq_memory.schedule import build_keyframe_schedule, schedule_config_from_dict, written_frame_ids  # noqa: E402


DEFAULT_CONFIGS = [
    "configs/causal_branch_4414_baseline.yaml",
    "configs/causal_branch_4414_A_special_old_l4_8.yaml",
    "configs/causal_branch_4414_Ball_image_old_l4_23.yaml",
    "configs/causal_branch_4414_Blate_image_old_l17_23.yaml",
    "configs/causal_branch_4414_C_image_live_special_l8_23.yaml",
]


def parse_config_paths(values: list[str] | None) -> list[Path]:
    raw = values or DEFAULT_CONFIGS
    return [(EXPERIMENT_ROOT / item if not Path(item).is_absolute() else Path(item)).resolve() for item in raw]


def path_status(path: str | Path, kind: str = "path") -> dict[str, Any]:
    p = Path(path).expanduser()
    return {
        "kind": kind,
        "path": str(p),
        "exists": p.exists(),
        "is_dir": p.is_dir(),
        "is_file": p.is_file(),
    }


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def required_frames_for_config(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = cfg.get("dataset", {})
    start = int(dataset.get("start_frame", cfg.get("start_frame", 0)))
    end = int(dataset.get("end_frame", cfg.get("end_frame", cfg.get("sequence", {}).get("expected_frames", 0))))
    stride = int(dataset.get("input_stride", cfg.get("stride", 1)))
    frames = list(range(start, end, stride))
    max_frame = max(frames) if frames else None
    return {
        "start_frame": start,
        "end_frame_exclusive": end,
        "input_stride": stride,
        "required_input_frames": len(frames),
        "max_required_frame": max_frame,
        "minimum_indexed_frames": 0 if max_frame is None else int(max_frame) + 1,
    }


def check_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["dataset_root"]).expanduser()
    sequence_expected = int(cfg.get("sequence", {}).get("expected_frames", 0))
    required = required_frames_for_config(cfg)
    images_dir = root / "images"
    timestamp_file = root / "timestamp_sync" / "frame_sync.csv"
    poses_file = root / "poses_c2w.txt"
    image_count = count_files(images_dir, "*.png")
    pose_rows = count_nonempty_lines(poses_file)
    sync_rows = max(count_nonempty_lines(timestamp_file) - 1, 0) if timestamp_file.exists() else 0
    minimum_indexed = int(required["minimum_indexed_frames"])
    out = {
        "root": path_status(root, "dataset_root"),
        "images_dir": path_status(images_dir, "images_dir"),
        "image_count": image_count,
        "sequence_expected_frames": sequence_expected,
        "required_frames": required,
        "minimum_indexed_frames": minimum_indexed,
        "intrinsics": path_status(root / "intrinsics.txt", "intrinsics"),
        "poses": path_status(poses_file, "gt_trajectory"),
        "pose_rows": pose_rows,
        "timestamp_sync": path_status(timestamp_file, "timestamp_sync"),
        "timestamp_sync_rows": sync_rows,
        "lidar_tls_dir": path_status(root / "lidar_tls", "lidar_tls"),
    }
    out["ready_for_reconstruction"] = bool(
        out["root"]["exists"]
        and out["images_dir"]["exists"]
        and image_count >= minimum_indexed
        and out["intrinsics"]["exists"]
        and out["poses"]["exists"]
        and pose_rows >= minimum_indexed
        and out["timestamp_sync"]["exists"]
        and sync_rows >= minimum_indexed
    )
    out["needs_redownload_or_preprocess"] = not out["ready_for_reconstruction"]
    return out


def check_existing_run_outputs(cfg: dict[str, Any]) -> dict[str, Any]:
    recon = Path(cfg["reconstruction_output_dir"]).expanduser()
    interaction = Path(cfg["interaction_output_dir"]).expanduser()
    required_recon = [
        "frame_ids.npy",
        "gt_poses_c2w.npy",
        "pred_poses_c2w_demo_convention.npy",
        "pred_intrinsics.npy",
        "run_manifest.json",
    ]
    out = {
        "reconstruction_dir": str(recon),
        "interaction_dir": str(interaction),
        "reconstruction_exists": recon.exists(),
        "interaction_exists": interaction.exists(),
        "required_reconstruction_files": {
            name: (recon / name).exists() for name in required_recon
        },
        "dense_dir_exists": (recon / "dense").exists(),
        "representation_index_exists": (recon / "representation_capture_index.json").exists(),
        "causal_stats_exists": (recon / "causal_intervention_stats.jsonl").exists(),
        "interaction_index_exists": (interaction / "interaction_index.json").exists(),
    }
    out["has_minimal_reconstruction"] = all(out["required_reconstruction_files"].values())
    return out


def infer_patch_count(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = cfg.get("model_defaults", {})
    image_size = int(defaults.get("image_size", cfg.get("inference", {}).get("image_size", 518)))
    patch_size = int(defaults.get("patch_size", 14))
    root = Path(cfg["dataset_root"]).expanduser()
    intrinsics_path = root / "intrinsics.txt"
    if intrinsics_path.exists():
        parts = intrinsics_path.read_text(encoding="utf-8").strip().split()
        width, height = int(parts[4]), int(parts[5])
        grid = preprocessed_patch_grid(width, height, image_size=image_size, patch_size=patch_size)
        return {"patch_grid_wh": list(grid), "patch_count": int(grid[0] * grid[1]), "source": "intrinsics"}
    return {"patch_grid_wh": [37, 28], "patch_count": 1036, "source": "fallback_previous_keble02"}


def intervention_frames(cfg: dict[str, Any], fallback_frame: int) -> list[int]:
    intervention = cfg.get("causal_intervention", {}) or {}
    if not intervention.get("enabled", False):
        return []
    frames = intervention.get("frames")
    if frames is None:
        return []
    if isinstance(frames, str) and frames.strip().lower() == "all":
        return [int(fallback_frame)]
    return [int(frame_id) for frame_id in frames]


def estimate_mask_for_frame(cfg: dict[str, Any], frame_id: int) -> dict[str, Any]:
    intervention = cfg.get("causal_intervention", {}) or {}
    if not intervention.get("enabled", False):
        return {"enabled": False}
    selected_frames = intervention_frames(cfg, frame_id)
    applies_to_frame = (
        isinstance(intervention.get("frames"), str)
        and str(intervention.get("frames")).strip().lower() == "all"
    ) or int(frame_id) in set(selected_frames)
    schedule_rows = build_keyframe_schedule(schedule_config_from_dict(cfg))
    written = written_frame_ids(schedule_rows, frame_id)
    by_frame = {int(row["source_frame_id"]): row for row in schedule_rows}
    defaults = cfg.get("model_defaults", {})
    scale_frames = int(defaults.get("num_scale_frames", 8))
    sliding_window = int(defaults.get("kv_cache_sliding_window", 64))
    num_special = int(intervention.get("num_special_tokens", cfg.get("memory_policy", {}).get("patch_start_idx", 6)))
    patch_info = infer_patch_count(cfg)
    patch_count = int(patch_info["patch_count"])
    tokens_per_frame = patch_count + num_special
    row = memory_state_summary(
        current_frame=frame_id,
        processed_frames=written,
        target_frame=int(cfg.get("loop_event", {}).get("history_frame", 3403)),
        scale_frames=scale_frames,
        sliding_window=sliding_window,
        num_register_tokens=int(cfg.get("memory_policy", {}).get("num_register_tokens", 4)),
    )
    old_tokens = len(row["old_memory_frames"]) * num_special
    current_visible_only = 0 if frame_id in set(row["patch_memory_frames"]) else 1
    live_frames = len(row["patch_memory_frames"]) + current_visible_only
    live_tokens = live_frames * tokens_per_frame
    q_tokens = tokens_per_frame
    mode = intervention["mode"]
    if mode == "special_to_old_memory":
        q_selected = num_special
        k_selected = old_tokens
    elif mode == "image_to_old_memory":
        q_selected = patch_count
        k_selected = old_tokens
    elif mode == "image_to_live_special":
        q_selected = patch_count
        live_special = live_frames * num_special
        k_selected = live_special + (old_tokens if intervention.get("include_old_memory_special_keys", False) else 0)
    else:
        q_selected = 0
        k_selected = 0
    return {
        "enabled": True,
        "mode": mode,
        "frame_selected": bool(applies_to_frame),
        "layers": [int(v) for v in intervention.get("layers", [])],
        "frame_id": int(frame_id),
        "frame_is_memory_write": bool(by_frame.get(int(frame_id), {}).get("expected_memory_write", False)),
        "patch_info": patch_info,
        "tokens_per_frame": int(tokens_per_frame),
        "query_tokens_per_current_frame": int(q_tokens),
        "selected_query_tokens_per_layer": int(q_selected),
        "selected_key_tokens_per_layer": int(k_selected),
        "masked_pairs_per_layer": int(q_selected * k_selected),
        "old_memory_frame_count": len(row["old_memory_frames"]),
        "patch_memory_frame_count": len(row["patch_memory_frames"]),
        "old_memory_frame_range": row["old_memory_frame_range"],
        "patch_memory_frame_range": [min(row["patch_memory_frames"]), max(row["patch_memory_frames"])] if row["patch_memory_frames"] else None,
    }


def estimate_mask_for_selected_frames(cfg: dict[str, Any], fallback_frame: int) -> list[dict[str, Any]]:
    frames = intervention_frames(cfg, fallback_frame)
    if not frames:
        return []
    return [estimate_mask_for_frame(cfg, frame_id) for frame_id in frames]


def run_commands(configs: list[Path]) -> list[dict[str, str]]:
    commands = []
    for config in configs:
        commands.append(
            {
                "config": str(config),
                "command": f"python scripts/run_long_sequence.py --config {config} --allow-large-output --confirm-full-run",
            }
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--probe-frame", type=int, default=4414)
    parser.add_argument("--output-dir", default=str(EXPERIMENT_ROOT / "outputs" / "causal_branch_4414" / "path_check"))
    args = parser.parse_args()

    config_paths = parse_config_paths(args.configs)
    reports = []
    output_dirs = []
    for path in config_paths:
        cfg = load_config(path)
        output_dirs.append(str(Path(cfg["reconstruction_output_dir"]).expanduser()))
        reports.append(
            {
                "config": str(path),
                "run_name": cfg.get("run_name"),
                "dataset": check_dataset(cfg),
                "lingbot_root": path_status(cfg["lingbot_root"], "lingbot_root"),
                "checkpoint": path_status(cfg["checkpoint"], "checkpoint"),
                "existing_outputs": check_existing_run_outputs(cfg),
                "intervention_mask_probe": estimate_mask_for_frame(cfg, args.probe_frame),
                "intervention_selected_frame_probes": estimate_mask_for_selected_frames(cfg, args.probe_frame),
                "representation_capture": cfg.get("representation_capture", {"enabled": False}),
            }
        )

    duplicate_outputs = sorted({item for item in output_dirs if output_dirs.count(item) > 1})
    overall_ready = all(
        item["dataset"]["ready_for_reconstruction"]
        and item["lingbot_root"]["exists"]
        and item["checkpoint"]["exists"]
        for item in reports
    )
    max_required_indexed_frames = max(
        (int(item["dataset"]["minimum_indexed_frames"]) for item in reports),
        default=0,
    )
    contamination_risks = []
    for item in reports:
        for probe in item.get("intervention_selected_frame_probes", []):
            if probe.get("frame_is_memory_write", False):
                contamination_risks.append(
                    {
                        "run_name": item["run_name"],
                        "frame_id": probe["frame_id"],
                        "reason": "selected intervention frame is a memory-write frame",
                    }
                )
    payload = {
        "status": "ready_for_formal_runs" if overall_ready and not duplicate_outputs and not contamination_risks else "blocked_or_needs_attention",
        "probe_frame": int(args.probe_frame),
        "duplicate_reconstruction_output_dirs": duplicate_outputs,
        "state_contamination_risks": contamination_risks,
        "configs": reports,
        "suggested_preprocess_commands": {
            "fast_reconstruction_ready": (
                "python /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/scripts/lingbot_oxford_preprocess.py "
                "--dataset_dir /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/downloads "
                "--output_dir /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/lingbot_ready_loop "
                "--sequence 2024-03-12-keble-college-02 "
                f"--max_frames {max_required_indexed_frames} --images_only"
            ),
            "with_tls_depth_outputs_heavy": (
                "python /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/scripts/lingbot_oxford_preprocess.py "
                "--dataset_dir /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/downloads "
                "--output_dir /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/lingbot_ready_loop "
                "--sequence 2024-03-12-keble-college-02 "
                f"--max_frames {max_required_indexed_frames}"
            ),
        },
        "suggested_run_commands": run_commands(config_paths),
        "post_run_analysis_command": (
            "python scripts/analyze_causal_suite.py "
            "--suite-root /home/3dsm/Desktop/JSWprobe/long_seq_memory/outputs/causal_branch_4414 "
            "--config configs/causal_branch_4414_baseline.yaml"
        ),
    }
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "causal_path_check.json", payload)
    lines = [
        "# Causal Intervention Path Check",
        "",
        f"- Status: `{payload['status']}`",
        f"- Probe frame: `{args.probe_frame}`",
        f"- Duplicate reconstruction dirs: `{duplicate_outputs}`",
        f"- State contamination risks: `{contamination_risks}`",
        "",
        "## Suggested Preprocess",
        "",
        f"- Fast reconstruction-ready: `{payload['suggested_preprocess_commands']['fast_reconstruction_ready']}`",
        f"- With TLS depth outputs, heavier: `{payload['suggested_preprocess_commands']['with_tls_depth_outputs_heavy']}`",
        "",
        "## Configs",
    ]
    for item in reports:
        dataset = item["dataset"]
        mask = item["intervention_mask_probe"]
        lines.append(f"- `{item['run_name']}`")
        lines.append(
            f"  - dataset_ready: `{dataset['ready_for_reconstruction']}`; "
            f"images: `{dataset['image_count']}/{dataset['minimum_indexed_frames']}`; "
            f"poses: `{dataset['pose_rows']}/{dataset['minimum_indexed_frames']}`; "
            f"sync_rows: `{dataset['timestamp_sync_rows']}/{dataset['minimum_indexed_frames']}`"
        )
        lines.append(f"  - checkpoint_exists: `{item['checkpoint']['exists']}`")
        lines.append(f"  - has_existing_reconstruction: `{item['existing_outputs']['has_minimal_reconstruction']}`")
        if mask.get("enabled"):
            lines.append(
                "  - mask_probe: "
                f"`{mask['mode']}` layers={mask['layers']} "
                f"frame_selected={mask['frame_selected']} "
                f"q={mask['selected_query_tokens_per_layer']} "
                f"k={mask['selected_key_tokens_per_layer']} "
                f"pairs/layer={mask['masked_pairs_per_layer']}"
            )
            for selected_probe in item.get("intervention_selected_frame_probes", []):
                lines.append(
                    "  - selected_frame_probe: "
                    f"frame={selected_probe['frame_id']} "
                    f"memory_write={selected_probe['frame_is_memory_write']} "
                    f"q={selected_probe['selected_query_tokens_per_layer']} "
                    f"k={selected_probe['selected_key_tokens_per_layer']} "
                    f"pairs/layer={selected_probe['masked_pairs_per_layer']}"
                )
    (out_dir / "causal_path_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'causal_path_check.json'}")
    print(f"Wrote {out_dir / 'causal_path_check.md'}")
    print(f"status={payload['status']}")
    if not overall_ready:
        print("dataset/checkpoint/lingbot paths are not all ready; inspect causal_path_check.md before running formal reconstruction.")


if __name__ == "__main__":
    main()
