from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from long_seq_memory.dataset import load_scene
from long_seq_memory.io_utils import ensure_dir, write_json
from long_seq_memory.memory_index import preprocessed_patch_grid
from long_seq_memory.schedule import (
    build_keyframe_schedule,
    old_special_frames_at,
    retained_full_frames_at,
    schedule_config_from_dict,
    schedule_report,
    written_frame_ids,
    write_schedule_outputs,
)


BYTES_PER_GB = 1024**3


def _cfg_get(cfg: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, cfg.get("model_defaults", {}).get(key, cfg.get(key, default)))


def _dtype_bytes(dtype: str) -> int:
    normalized = str(dtype).replace("torch.", "").lower()
    if normalized in {"bfloat16", "float16", "fp16", "half"}:
        return 2
    if normalized in {"float64", "fp64", "double"}:
        return 8
    return 4


def _load_timing_reference(cfg: dict[str, Any]) -> dict[str, Any]:
    path = cfg.get("resource_estimation", {}).get("timing_reference_dir")
    if not path:
        path = Path(cfg["output_dir"]) / "reconstruction" / "loop_3200_4600"
    ref_dir = Path(path).expanduser()
    out: dict[str, Any] = {
        "reference_dir": str(ref_dir),
        "available": False,
        "mean_frame_s": None,
        "median_frame_s": None,
        "num_timing_samples": 0,
        "gpu_peak_allocated_gb": None,
        "gpu_peak_reserved_gb": None,
    }
    time_path = ref_dir / "per_frame_time_s.npy"
    if time_path.exists():
        values = np.load(time_path, allow_pickle=True).astype(np.float64)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            out.update(
                {
                    "available": True,
                    "mean_frame_s": float(mean(values.tolist())),
                    "median_frame_s": float(median(values.tolist())),
                    "num_timing_samples": int(values.size),
                }
            )
    manifest_path = ref_dir / "run_manifest.json"
    if manifest_path.exists():
        import json

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        out["gpu_peak_allocated_gb"] = manifest.get("gpu_peak_allocated_gb")
        out["gpu_peak_reserved_gb"] = manifest.get("gpu_peak_reserved_gb")
    return out


def _range_len_inclusive(span: list[int] | tuple[int, int], stride: int = 1) -> int:
    start, end = int(span[0]), int(span[1])
    if end < start:
        return 0
    return ((end - start) // max(int(stride), 1)) + 1


def _feature_size_estimate(
    cfg: dict[str, Any],
    schedule_rows: list[dict[str, Any]],
    patch_count: int,
    tokens_per_frame: int,
    special_tokens: int,
    hidden_dim: int,
    dtype_bytes: int,
    depth: int,
) -> dict[str, Any]:
    extraction = cfg.get("extraction", {})
    num_frames = len(schedule_rows)
    global_stride = int(extraction.get("global_summary_stride", 10))
    query_types = ["image_queries", "camera_query", "register_queries", "scale_query", "all_queries"]
    memory_types = ["camera", "register", "scale"]
    metric_count = 2
    global_samples = math.ceil(num_frames / max(global_stride, 1))
    top_k = int(extraction.get("top_k_history_frames", 32))
    age_bins = int(extraction.get("age_histogram_bins", 32))

    # Heuristic row model: per sampled frame/layer/query, store scalars,
    # top-k ids/scores, and age histogram for attention and contribution.
    global_row_bytes = (
        20 * 8
        + top_k * (4 + 4) * metric_count
        + age_bins * 4 * metric_count
        + len(memory_types) * 8 * metric_count
    )
    global_summary_gb = global_samples * depth * len(query_types) * global_row_bytes / BYTES_PER_GB

    loop_current_range = extraction.get("loop_current_range", cfg.get("loop_event", {}).get("current_segment", [4380, 4445]))
    loop_stride = int(extraction.get("loop_current_stride", 1))
    target_history_range = extraction.get("target_history_range", cfg.get("loop_event", {}).get("history_segment", [3370, 3440]))
    loop_current_count = _range_len_inclusive(loop_current_range, loop_stride)
    retained_history_frames = [
        fid
        for fid in written_frame_ids(schedule_rows, int(loop_current_range[1]))
        if int(target_history_range[0]) <= fid <= int(target_history_range[1])
    ]
    history_count = len(retained_history_frames)

    dense_values = loop_current_count * max(history_count, 1) * depth * len(query_types) * len(memory_types) * metric_count
    loop_dense_gb = dense_values * 4 / BYTES_PER_GB
    representative_layers = list(extraction.get("representative_layers", [4, 11, 17, 23]))
    heads = int(cfg.get("model_profile", {}).get("num_heads", 16))
    rep_head_values = loop_current_count * max(history_count, 1) * len(representative_layers) * heads * len(query_types) * len(memory_types)
    loop_dense_gb += rep_head_values * 4 / BYTES_PER_GB

    selected_frames = list(extraction.get("selected_current_frames", [4390, 4400, 4408, 4414, 4420, 4430]))
    num_controls = int(extraction.get("auto_control_frames", 3))
    patch_events = len(selected_frames) + num_controls
    patch_values = patch_events * len(representative_layers) * patch_count * len(memory_types) * 16 * 3
    patch_specific_gb = patch_values * 4 / BYTES_PER_GB

    raw_frames = list(extraction.get("raw_qkv_frames", [4400, 4414, 4430]))
    raw_max_frames = int(extraction.get("raw_qkv_max_frames", 4))
    raw_count = min(raw_max_frames, len(set(raw_frames)) + 1)
    raw_frame_candidates = sorted(set(raw_frames + selected_frames))
    raw_frame_for_worst_case = raw_frame_candidates[-1] if raw_frame_candidates else int(loop_current_range[1])
    retained = retained_full_frames_at(
        schedule_rows,
        raw_frame_for_worst_case,
        int(_cfg_get(cfg, "inference", "num_scale_frames", 8)),
        int(_cfg_get(cfg, "inference", "kv_cache_sliding_window", 64)),
    )
    old = old_special_frames_at(
        schedule_rows,
        raw_frame_for_worst_case,
        int(_cfg_get(cfg, "inference", "num_scale_frames", 8)),
        int(_cfg_get(cfg, "inference", "kv_cache_sliding_window", 64)),
    )
    visible_tokens = len(retained) * tokens_per_frame + len(old) * special_tokens + tokens_per_frame
    raw_qkv_gb = raw_count * len(representative_layers) * (
        tokens_per_frame * hidden_dim + visible_tokens * hidden_dim * 2
    ) * dtype_bytes / BYTES_PER_GB

    total = global_summary_gb + loop_dense_gb + patch_specific_gb + raw_qkv_gb
    return {
        "global_summary": {
            "stride": global_stride,
            "sampled_frames": global_samples,
            "estimated_gb": global_summary_gb,
        },
        "loop_dense": {
            "current_range": list(loop_current_range),
            "target_history_range": list(target_history_range),
            "current_count": loop_current_count,
            "retained_history_frames_in_target_range": retained_history_frames,
            "estimated_gb": loop_dense_gb,
        },
        "patch_specific": {
            "selected_current_frames": selected_frames,
            "auto_control_frames": num_controls,
            "patch_count": patch_count,
            "estimated_gb": patch_specific_gb,
        },
        "raw_qkv_debug": {
            "requested_raw_frames": raw_frames,
            "estimated_raw_frame_count": raw_count,
            "representative_layers": representative_layers,
            "worst_case_frame": raw_frame_for_worst_case,
            "visible_tokens_per_layer_worst_case": visible_tokens,
            "estimated_gb": raw_qkv_gb,
            "budget_gb": float(extraction.get("raw_qkv_budget_gb", 8)),
            "within_budget": raw_qkv_gb <= float(extraction.get("raw_qkv_budget_gb", 8)),
        },
        "total_estimated_gb": total,
    }


def estimate_resources(cfg: dict[str, Any]) -> dict[str, Any]:
    scfg = schedule_config_from_dict(cfg)
    schedule_rows = build_keyframe_schedule(scfg)
    scene = load_scene(cfg["dataset_root"])
    image_size = int(_cfg_get(cfg, "inference", "image_size", 518))
    patch_size = int(cfg.get("model_defaults", {}).get("patch_size", 14))
    patch_grid = preprocessed_patch_grid(scene.intrinsics.width, scene.intrinsics.height, image_size, patch_size)
    patch_count = patch_grid[0] * patch_grid[1]
    special_tokens = int(cfg.get("memory_policy", {}).get("patch_start_idx", 6))
    tokens_per_frame = patch_count + special_tokens
    depth = int(cfg.get("model_profile", {}).get("num_gca_layers", cfg.get("model_defaults", {}).get("num_gca_layers", 24)))
    heads = int(cfg.get("model_profile", {}).get("num_heads", 16))
    head_dim = int(cfg.get("model_profile", {}).get("head_dim", 64))
    hidden_dim = heads * head_dim
    dtype = str(_cfg_get(cfg, "inference", "dtype", "bfloat16"))
    bytes_per_scalar = _dtype_bytes(dtype)
    final_frame = int(schedule_rows[-1]["source_frame_id"]) if schedule_rows else scfg.end_frame

    retained_final = retained_full_frames_at(schedule_rows, final_frame, scfg.num_scale_frames, scfg.sliding_window)
    old_final = old_special_frames_at(schedule_rows, final_frame, scfg.num_scale_frames, scfg.sliding_window)
    kv_tokens_per_layer = len(retained_final) * tokens_per_frame + len(old_final) * special_tokens
    kv_cache_gb = kv_tokens_per_layer * hidden_dim * 2 * bytes_per_scalar * depth / BYTES_PER_GB

    outputs_cfg = cfg.get("outputs", {})
    pre_h = patch_grid[1] * patch_size
    pre_w = patch_grid[0] * patch_size
    frame_count = len(schedule_rows)
    pose_gb = frame_count * (16 + 9) * 4 / BYTES_PER_GB
    dense_gb = 0.0
    if outputs_cfg.get("save_depth_all_frames", False):
        depth_frames = frame_count
    else:
        stride = int(outputs_cfg.get("save_depth_stride", 0) or 0)
        depth_frames = math.ceil(frame_count / stride) if stride > 0 else 0
    if depth_frames:
        dense_gb += depth_frames * pre_h * pre_w * 2 / BYTES_PER_GB
    if outputs_cfg.get("save_world_points_all_frames", False):
        point_frames = frame_count
    else:
        stride = int(outputs_cfg.get("save_world_points_stride", 0) or 0)
        point_frames = math.ceil(frame_count / stride) if stride > 0 else 0
    if point_frames:
        dense_gb += point_frames * pre_h * pre_w * 3 * 2 / BYTES_PER_GB

    timing = _load_timing_reference(cfg)
    frame_s = timing.get("mean_frame_s") or 1.0
    run_time_s = float(frame_s) * frame_count

    feature = _feature_size_estimate(
        cfg,
        schedule_rows,
        patch_count=patch_count,
        tokens_per_frame=tokens_per_frame,
        special_tokens=special_tokens,
        hidden_dim=hidden_dim,
        dtype_bytes=bytes_per_scalar,
        depth=depth,
    )
    reconstruction_output_gb = pose_gb + dense_gb + 0.05
    total_output_gb = reconstruction_output_gb + feature["total_estimated_gb"]
    reference_peak = timing.get("gpu_peak_allocated_gb")
    recommended_peak = max(kv_cache_gb + 12.0, float(reference_peak or 0.0))
    loop = cfg.get("loop_event", {})
    report = schedule_report(
        schedule_rows,
        scfg,
        history_frame=int(loop.get("history_frame", 3403)),
        current_frame=int(loop.get("current_frame", 4414)),
        history_segment=tuple(loop.get("history_segment", [3370, 3440])),
    )

    return {
        "status": "complete",
        "run_name": cfg.get("run_name"),
        "config": cfg.get("_config_path"),
        "model_profile": {
            "num_gca_layers": depth,
            "num_heads": heads,
            "head_dim": head_dim,
            "hidden_dim": hidden_dim,
            "dtype": dtype,
            "bytes_per_scalar": bytes_per_scalar,
        },
        "sequence": {
            "num_input_frames": frame_count,
            "start_frame": scfg.start_frame,
            "end_frame_exclusive": scfg.end_frame,
            "input_stride": scfg.input_stride,
        },
        "tokenization": {
            "patch_grid_wh": list(patch_grid),
            "patch_count": patch_count,
            "special_tokens_per_frame": special_tokens,
            "tokens_per_frame": tokens_per_frame,
        },
        "keyframes": {
            "keyframe_interval": scfg.keyframe_interval,
            "num_scale_frames": scfg.num_scale_frames,
            "total_memory_writes": len(written_frame_ids(schedule_rows)),
            "retained_full_frames_at_end": len(retained_final),
            "old_special_frames_at_end": len(old_final),
            "old_special_tokens_per_layer_at_end": len(old_final) * special_tokens,
            "schedule_report": report,
        },
        "gpu_memory_estimate": {
            "kv_cache_gb": kv_cache_gb,
            "reference_peak_allocated_gb": reference_peak,
            "recommended_peak_allocated_gb": recommended_peak,
            "note": "KV estimate excludes model weights and transient activations; recommended peak uses the focused run as a floor.",
        },
        "runtime_estimate": {
            "timing_reference": timing,
            "estimated_seconds": run_time_s,
            "estimated_hours": run_time_s / 3600.0,
        },
        "disk_estimate": {
            "reconstruction_output_gb": reconstruction_output_gb,
            "feature_extraction": feature,
            "total_output_gb": total_output_gb,
            "target_budget_gb": float(cfg.get("resource_estimation", {}).get("output_budget_gb", 30)),
            "hard_budget_gb": float(cfg.get("resource_estimation", {}).get("hard_output_budget_gb", 40)),
        },
        "warnings": [
            "This is a pre-run estimate. Formal results require the actual full reconstruction and interaction extraction outputs.",
            "Resume is not enabled unless a real KV checkpoint is written and restored.",
        ],
    }


def write_resource_estimate(
    cfg: dict[str, Any],
    estimate: dict[str, Any],
    allow_large_output: bool = False,
    filename_prefix: str = "resource_estimate",
) -> dict[str, Any]:
    run_name = cfg.get("run_name", "run")
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "planning" / run_name)
    hard = float(estimate["disk_estimate"]["hard_budget_gb"])
    total = float(estimate["disk_estimate"]["total_output_gb"])
    blocked = total > hard and not allow_large_output
    if blocked:
        estimate["status"] = "incomplete"
        estimate["blocked_by_output_budget"] = True
    write_json(out_dir / f"{filename_prefix}.json", estimate)
    md = render_resource_estimate_markdown(estimate)
    (out_dir / f"{filename_prefix}.md").write_text(md, encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "json": str(out_dir / f"{filename_prefix}.json"),
        "markdown": str(out_dir / f"{filename_prefix}.md"),
        "blocked_by_output_budget": blocked,
    }


def render_resource_estimate_markdown(estimate: dict[str, Any]) -> str:
    keyframes = estimate["keyframes"]
    disk = estimate["disk_estimate"]
    feature = disk["feature_extraction"]
    gpu = estimate["gpu_memory_estimate"]
    runtime = estimate["runtime_estimate"]
    report = keyframes["schedule_report"]
    lines = [
        f"# Resource Estimate: {estimate.get('run_name')}",
        "",
        f"- Status: `{estimate.get('status')}`",
        f"- Frames: {estimate['sequence']['num_input_frames']}",
        f"- Keyframe interval: {keyframes['keyframe_interval']}",
        f"- Total memory writes: {keyframes['total_memory_writes']}",
        f"- Old special tokens per layer at end: {keyframes['old_special_tokens_per_layer_at_end']}",
        f"- KV cache estimate: {gpu['kv_cache_gb']:.2f} GB",
        f"- Recommended GPU peak budget: {gpu['recommended_peak_allocated_gb']:.2f} GB",
        f"- Estimated runtime: {runtime['estimated_hours']:.2f} h",
        f"- Reconstruction output: {disk['reconstruction_output_gb']:.2f} GB",
        f"- Feature output: {feature['total_estimated_gb']:.2f} GB",
        f"- Total output: {disk['total_output_gb']:.2f} GB",
        "",
        "## Loop Keyframes",
        "",
        f"- Frame {report['history_frame']} written: `{report['history_frame_written']}`",
        f"- Frame {report['current_frame']} written: `{report['current_frame_written']}`",
        f"- Nearest retained memory frames to {report['history_frame']} at {report['current_frame']}: "
        f"{report['nearest_retained_history_to_history_frame_at_current']}",
        f"- Written keyframes in {report['history_segment']}: {report['written_keyframes_in_history_segment']}",
        "",
        "## Feature Output",
        "",
        f"- Level 1 global summary: {feature['global_summary']['estimated_gb']:.2f} GB",
        f"- Level 2 loop dense: {feature['loop_dense']['estimated_gb']:.2f} GB",
        f"- Level 3 patch-specific: {feature['patch_specific']['estimated_gb']:.2f} GB",
        f"- Raw Q/K/V debug: {feature['raw_qkv_debug']['estimated_gb']:.2f} GB "
        f"(budget {feature['raw_qkv_debug']['budget_gb']:.2f} GB)",
        "",
    ]
    if estimate.get("blocked_by_output_budget"):
        lines.extend(
            [
                "## Blocked",
                "",
                "Estimated output exceeds the hard budget. Re-run with `--allow-large-output` only after accepting this cost.",
                "",
            ]
        )
    return "\n".join(lines)


def prepare_planning_outputs(cfg: dict[str, Any]) -> dict[str, Any]:
    scfg = schedule_config_from_dict(cfg)
    rows = build_keyframe_schedule(scfg)
    loop = cfg.get("loop_event", {})
    report = schedule_report(
        rows,
        scfg,
        history_frame=int(loop.get("history_frame", 3403)),
        current_frame=int(loop.get("current_frame", 4414)),
        history_segment=tuple(loop.get("history_segment", [3370, 3440])),
    )
    return write_schedule_outputs(cfg["output_dir"], cfg.get("run_name", "run"), rows, report)
