from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from long_seq_memory.io_utils import ensure_dir, write_json, write_table


@dataclass(frozen=True)
class ScheduleConfig:
    start_frame: int
    end_frame: int
    input_stride: int
    num_scale_frames: int
    keyframe_interval: int
    sliding_window: int


def schedule_config_from_dict(cfg: dict[str, Any]) -> ScheduleConfig:
    dataset = cfg.get("dataset", {})
    inference = cfg.get("inference", {})
    defaults = cfg.get("model_defaults", {})
    return ScheduleConfig(
        start_frame=int(dataset.get("start_frame", cfg.get("start_frame", 0))),
        end_frame=int(dataset.get("end_frame", cfg.get("end_frame", cfg.get("sequence", {}).get("expected_frames", 0)))),
        input_stride=int(dataset.get("input_stride", cfg.get("stride", 1))),
        num_scale_frames=int(inference.get("num_scale_frames", defaults.get("num_scale_frames", 8))),
        keyframe_interval=int(inference.get("keyframe_interval", defaults.get("keyframe_interval", 1))),
        sliding_window=int(inference.get("kv_cache_sliding_window", defaults.get("kv_cache_sliding_window", 64))),
    )


def build_keyframe_schedule(scfg: ScheduleConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keyframe_index = -1
    source_frames = list(range(scfg.start_frame, scfg.end_frame, scfg.input_stride))
    for input_index, source_frame_id in enumerate(source_frames):
        is_scale = input_index < scfg.num_scale_frames
        is_keyframe = (not is_scale) and (
            scfg.keyframe_interval <= 1
            or ((input_index - scfg.num_scale_frames) % scfg.keyframe_interval == 0)
        )
        expected_write = is_scale or is_keyframe
        if expected_write:
            keyframe_index += 1
        rows.append(
            {
                "input_index": input_index,
                "source_frame_id": source_frame_id,
                "is_scale_frame": is_scale,
                "is_keyframe": is_keyframe,
                "keyframe_index": keyframe_index if expected_write else None,
                "expected_memory_write": expected_write,
            }
        )
    return rows


def written_frame_ids(schedule_rows: list[dict[str, Any]], up_to_frame: int | None = None) -> list[int]:
    out = []
    for row in schedule_rows:
        fid = int(row["source_frame_id"])
        if up_to_frame is not None and fid > up_to_frame:
            break
        if bool(row["expected_memory_write"]):
            out.append(fid)
    return out


def retained_full_frames_at(
    schedule_rows: list[dict[str, Any]],
    current_frame: int,
    num_scale_frames: int,
    sliding_window: int,
) -> list[int]:
    writes = written_frame_ids(schedule_rows, current_frame)
    scale = writes[:num_scale_frames]
    live_candidates = [f for f in writes[num_scale_frames:] if f <= current_frame]
    live = live_candidates[-sliding_window:] if sliding_window > 0 else []
    out = []
    seen = set()
    for frame_id in scale + live:
        if frame_id not in seen:
            out.append(frame_id)
            seen.add(frame_id)
    return out


def old_special_frames_at(
    schedule_rows: list[dict[str, Any]],
    current_frame: int,
    num_scale_frames: int,
    sliding_window: int,
) -> list[int]:
    writes = written_frame_ids(schedule_rows, current_frame)
    retained = set(retained_full_frames_at(schedule_rows, current_frame, num_scale_frames, sliding_window))
    return [f for f in writes if f not in retained]


def nearest_retained_history(
    retained_frames: list[int],
    target_frame: int,
    max_items: int = 8,
) -> list[int]:
    before = [f for f in retained_frames if f <= target_frame]
    after = [f for f in retained_frames if f > target_frame]
    merged = sorted(before[-max_items:] + after[:max_items], key=lambda f: (abs(f - target_frame), f))
    return merged[:max_items]


def frames_in_range(frames: list[int], span: tuple[int, int]) -> list[int]:
    return [int(frame_id) for frame_id in frames if span[0] <= int(frame_id) <= span[1]]


def schedule_report(
    schedule_rows: list[dict[str, Any]],
    scfg: ScheduleConfig,
    history_frame: int,
    current_frame: int,
    history_segment: tuple[int, int],
) -> dict[str, Any]:
    by_frame = {int(row["source_frame_id"]): row for row in schedule_rows}
    retained_at_current = retained_full_frames_at(schedule_rows, current_frame, scfg.num_scale_frames, scfg.sliding_window)
    old_at_current = old_special_frames_at(schedule_rows, current_frame, scfg.num_scale_frames, scfg.sliding_window)
    long_memory_at_current = sorted(set(old_at_current + retained_at_current))
    segment_writes = [
        int(row["source_frame_id"])
        for row in schedule_rows
        if history_segment[0] <= int(row["source_frame_id"]) <= history_segment[1]
        and bool(row["expected_memory_write"])
    ]
    all_writes = written_frame_ids(schedule_rows)
    return {
        "start_frame": scfg.start_frame,
        "end_frame": scfg.end_frame,
        "input_stride": scfg.input_stride,
        "num_input_frames": len(schedule_rows),
        "num_scale_frames": scfg.num_scale_frames,
        "keyframe_interval": scfg.keyframe_interval,
        "sliding_window": scfg.sliding_window,
        "total_memory_writes": len(all_writes),
        "history_frame": history_frame,
        "current_frame": current_frame,
        "history_frame_written": bool(by_frame.get(history_frame, {}).get("expected_memory_write", False)),
        "current_frame_written": bool(by_frame.get(current_frame, {}).get("expected_memory_write", False)),
        "retained_full_frames_at_current": retained_at_current,
        "old_special_frames_at_current": old_at_current,
        "long_memory_frames_at_current_note": "Includes retained full patch frames and old special-token frames.",
        "old_special_frame_range_at_current": [min(old_at_current), max(old_at_current)] if old_at_current else None,
        "history_frame_in_old_special_at_current": history_frame in old_at_current,
        "history_frame_in_retained_full_at_current": history_frame in retained_at_current,
        "nearest_retained_history_to_history_frame_at_current": nearest_retained_history(
            long_memory_at_current, history_frame
        ),
        "nearest_full_patch_frames_to_history_frame_at_current": nearest_retained_history(
            retained_at_current, history_frame
        ),
        "history_segment": list(history_segment),
        "written_keyframes_in_history_segment": segment_writes,
        "retained_memory_frames_in_history_segment_at_current": frames_in_range(long_memory_at_current, history_segment),
        "retained_full_patch_frames_in_history_segment_at_current": frames_in_range(retained_at_current, history_segment),
        "num_written_keyframes_in_history_segment": len(segment_writes),
        }


def write_schedule_outputs(
    output_dir: str | Path,
    run_name: str,
    schedule_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    out_dir = ensure_dir(Path(output_dir) / "planning" / run_name)
    table_status = write_table(
        out_dir / "keyframe_schedule.parquet",
        schedule_rows,
        metadata={"run_name": run_name, "report": report},
    )
    write_json(out_dir / "keyframe_schedule_report.json", report)
    return {"output_dir": str(out_dir), "table": table_status, "report_path": str(out_dir / "keyframe_schedule_report.json")}
