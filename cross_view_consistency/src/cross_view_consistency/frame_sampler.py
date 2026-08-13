"""Frame selection for ScanNet scenes."""

from __future__ import annotations

from .scannet_io import FrameRecord, SceneInfo, make_frame_record


def select_frames(scene: SceneInfo, cfg: dict) -> list[FrameRecord]:
    fs = cfg["frame_sampling"]
    start = int(fs.get("start_frame", 0))
    stride = max(1, int(fs.get("stride", 5)))
    target = int(fs.get("num_frames", 32))
    allow_skip = bool(fs.get("allow_skip_missing", True))

    frames: list[FrameRecord] = []
    frame_id = start
    misses = 0
    max_misses = int(fs.get("max_consecutive_misses", 2000))
    while len(frames) < target and misses < max_misses:
        rec = make_frame_record(scene, frame_id, len(frames))
        if rec is not None and rec.valid_pose and rec.valid_depth:
            frames.append(rec)
            misses = 0
        else:
            misses += 1
            if not allow_skip:
                reason = "missing files" if rec is None else "invalid pose/depth"
                raise RuntimeError(f"Frame {frame_id} is unusable: {reason}")
        frame_id += stride
    if len(frames) < target:
        raise RuntimeError(f"Only found {len(frames)} usable frames, requested {target}.")
    for idx, fr in enumerate(frames):
        fr.index = idx
    return frames

