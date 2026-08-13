"""Explicit prefix + fixed-current-frame controlled sequence setting."""

from __future__ import annotations

from typing import Any

from data.manifest_utils.scannet_reader import ScanNetFrame, ScanNetScene

from .base_setting import BaseSequenceSetting, frame_pattern
from .registry import register_sequence_setting


@register_sequence_setting("prefix_induced_modulation")
class PrefixInducedModulationSetting(BaseSequenceSetting):
    """Build hand-specified prefix conditions that all end in the same frame."""

    def build_manifests(self, scene: ScanNetScene) -> list[dict[str, Any]]:
        expected_scene = self.config.get("scene_id")
        if expected_scene and not _scene_matches(str(expected_scene), scene.scene_id):
            raise ValueError(
                f"prefix_induced_modulation expected scene_id={expected_scene}, got {scene.scene_id}"
            )

        sequences = self.config.get("sequences") or {}
        if not isinstance(sequences, dict) or not sequences:
            raise ValueError("prefix_induced_modulation requires a non-empty sequences mapping")

        target_frame_id = int(self.config.get("target_frame_id"))
        total_length = int(self.config.get("sequence_total_length", self.seq_lens[0]))
        prefix_length = int(self.config.get("prefix_length", total_length - 1))
        if prefix_length + 1 != total_length:
            raise ValueError(
                f"prefix_length + 1 must equal sequence_total_length, got {prefix_length} + 1 != {total_length}"
            )
        if any(seq_len != total_length for seq_len in self.seq_lens):
            raise ValueError(
                f"prefix_induced_modulation uses explicit sequence_total_length={total_length}; "
                f"received seq_lens={self.seq_lens}"
            )

        requested_ids = _requested_frame_ids(sequences)
        missing = _missing_frame_ids(scene.frames, requested_ids)
        if missing:
            raise ValueError(
                f"Required raw frame ids are missing or invalid for {scene.scene_id}: {missing}. "
                f"Available valid frame ids: {_available_frame_summary(scene.frames)}"
            )

        frame_by_id = {frame.original_order_index: frame for frame in scene.frames}
        manifests: list[dict[str, Any]] = []
        for sequence_name, sequence_cfg in sequences.items():
            frame_ids = [int(value) for value in sequence_cfg.get("frame_ids", [])]
            if len(frame_ids) != total_length:
                raise ValueError(
                    f"{sequence_name} has {len(frame_ids)} frames; expected sequence_total_length={total_length}"
                )
            if frame_ids[-1] != target_frame_id:
                raise ValueError(
                    f"{sequence_name} must end with target_frame_id={target_frame_id}; got {frame_ids[-1]}"
                )

            sequence_role = str(sequence_cfg.get("role", sequence_name))
            frames = [frame_by_id[frame_id] for frame_id in frame_ids]
            entries = self._frame_entries(
                frames,
                sequence_name=sequence_name,
                sequence_role=sequence_role,
                requested_frame_ids=frame_ids,
                target_frame_id=target_frame_id,
            )
            condition_id = str(sequence_cfg.get("condition_id", sequence_name))
            manifests.append(
                self._manifest(
                    scene,
                    condition_id=condition_id,
                    seq_len=total_length,
                    frames=entries,
                    metadata=self._metadata(
                        sequence_name=sequence_name,
                        sequence_role=sequence_role,
                        frame_ids=frame_ids,
                        frames=frames,
                        target_frame_id=target_frame_id,
                        total_length=total_length,
                        prefix_length=prefix_length,
                    ),
                )
            )
        return manifests

    def _frame_entries(
        self,
        frames: list[ScanNetFrame],
        *,
        sequence_name: str,
        sequence_role: str,
        requested_frame_ids: list[int],
        target_frame_id: int,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        last_t = len(frames) - 1
        for t, (frame, requested_id) in enumerate(zip(frames, requested_frame_ids)):
            is_target = t == last_t
            entries.append(
                self._frame_entry(
                    frame,
                    t=t,
                    role="target_current_X" if is_target else sequence_role,
                    sequence_name=sequence_name,
                    sequence_role=sequence_role,
                    requested_raw_frame_id=int(requested_id),
                    target_frame_id=int(target_frame_id),
                    is_target_frame=is_target,
                    prefix_position=None if is_target else t,
                )
            )
        return entries

    def _metadata(
        self,
        *,
        sequence_name: str,
        sequence_role: str,
        frame_ids: list[int],
        frames: list[ScanNetFrame],
        target_frame_id: int,
        total_length: int,
        prefix_length: int,
    ) -> dict[str, Any]:
        prefix_ids = frame_ids[:-1]
        roles = [sequence_role for _ in prefix_ids] + ["target_current_X"]
        return {
            "experiment_name": self.config.get("experiment_name", "controlled_16_exp_v3"),
            "setting_params": {
                "scene_id": self.config.get("scene_id"),
                "target_frame_id": int(target_frame_id),
                "sequence_total_length": int(total_length),
                "prefix_length": int(prefix_length),
                "stride": int(self.config.get("stride", 1)),
                "token_type": self.config.get("token_type", "image_tokens"),
                "extract_last_frame_only": bool(self.config.get("extract_last_frame_only", True)),
                "layers": [int(value) for value in self.config.get("layers", [])],
            },
            "sequence_name": sequence_name,
            "sequence_role": sequence_role,
            "target_frame_id": int(target_frame_id),
            "prefix_frame_ids": prefix_ids,
            "full_frame_ids": frame_ids,
            "resolved_source_frame_ids": [frame.source_frame_id for frame in frames],
            "actual_image_paths": [str(frame.rgb_path) for frame in frames],
            "actual_depth_paths": [str(frame.depth_path) if frame.depth_path else None for frame in frames],
            "actual_pose_paths": [str(frame.pose_path) if frame.pose_path else None for frame in frames],
            "actual_intrinsic_paths": [
                str(frame.intrinsic_path) if frame.intrinsic_path else None for frame in frames
            ],
            "target_t": len(frame_ids) - 1,
            "all_frames_found": True,
            "order_policy": "prefix order exactly as configured; target X appended last",
            "seq_len": int(total_length),
            "frame_pattern": frame_pattern(roles),
            "notes": "Fixed-current-frame prefix modulation sequence.",
        }


def _scene_matches(config_scene_id: str, actual_scene_id: str) -> bool:
    return actual_scene_id == config_scene_id or actual_scene_id.startswith(f"{config_scene_id}_")


def _requested_frame_ids(sequences: dict[str, Any]) -> list[int]:
    requested: list[int] = []
    for sequence_cfg in sequences.values():
        requested.extend(int(value) for value in sequence_cfg.get("frame_ids", []))
    return sorted(set(requested))


def _missing_frame_ids(frames: list[ScanNetFrame], requested_ids: list[int]) -> list[int]:
    available = {frame.original_order_index for frame in frames}
    return [frame_id for frame_id in requested_ids if frame_id not in available]


def _available_frame_summary(frames: list[ScanNetFrame]) -> dict[str, Any]:
    ids = [frame.original_order_index for frame in frames]
    if not ids:
        return {"count": 0, "first": [], "last": []}
    return {
        "count": len(ids),
        "min": min(ids),
        "max": max(ids),
        "first": ids[:20],
        "last": ids[-20:],
    }
