from __future__ import annotations

from dataclasses import dataclass

from .scannet_io import FrameRecord, SceneInfo


@dataclass(frozen=True)
class SceneSelection:
    selected_scene_ids: list[str]
    skipped: list[dict[str, object]]


def valid_frames(scene: SceneInfo) -> list[FrameRecord]:
    return [f for f in scene.frames if f.valid_pose and f.valid_depth]


def sample_scene_frames(
    scene: SceneInfo,
    *,
    frames_per_scene: int,
    raw_frame_stride: int,
) -> list[FrameRecord]:
    frames = valid_frames(scene)
    sampled = frames[:: max(1, int(raw_frame_stride))]
    return sampled[: int(frames_per_scene)]


def choose_scenes(
    scenes: list[SceneInfo],
    *,
    num_scenes: int,
    frames_per_scene: int,
    raw_frame_stride: int,
    preferred_debug_scenes: list[str] | None = None,
) -> SceneSelection:
    by_id = {s.scene_id: s for s in scenes}
    order: list[str] = []
    for sid in preferred_debug_scenes or []:
        if sid in by_id and sid not in order:
            order.append(sid)
    for sid in sorted(by_id):
        if sid not in order:
            order.append(sid)

    selected: list[str] = []
    skipped: list[dict[str, object]] = []
    for sid in order:
        scene = by_id[sid]
        n_valid = len(valid_frames(scene))
        n_sampled = len(sample_scene_frames(scene, frames_per_scene=frames_per_scene, raw_frame_stride=raw_frame_stride))
        if n_sampled == 0:
            skipped.append({"scene_id": sid, "reason": "no_valid_sampled_frames", "valid_frames": n_valid})
            continue
        selected.append(sid)
        if len(selected) >= num_scenes:
            break
    return SceneSelection(selected_scene_ids=selected, skipped=skipped)

