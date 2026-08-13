"""Controlled 128-frame B1/X sequence pairs for exp v5."""

from __future__ import annotations

from typing import Any

from data.manifest_utils.scannet_reader import ScanNetFrame, ScanNetScene

from .base_setting import BaseSequenceSetting, frame_pattern
from .registry import register_sequence_setting


@register_sequence_setting("controlled_multi_length")
class ControlledMultiLengthSetting(BaseSequenceSetting):
    """Build normal B1 streams and repeated-target X controls by segment."""

    def build_manifests(self, scene: ScanNetScene) -> list[dict[str, Any]]:
        expected_scene = self.config.get("scene_id")
        if expected_scene and not _scene_matches(str(expected_scene), scene.scene_id):
            raise ValueError(
                f"controlled_multi_length expected scene_id={expected_scene}, got {scene.scene_id}"
            )

        total_length = int(self.config.get("sequence_total_length", self.seq_lens[0]))
        if any(seq_len != total_length for seq_len in self.seq_lens):
            raise ValueError(
                f"controlled_multi_length uses sequence_total_length={total_length}; "
                f"received seq_lens={self.seq_lens}"
            )
        if total_length <= 0:
            raise ValueError("sequence_total_length must be positive")

        stride = int(self.config.get("stride", 1))
        if stride <= 0:
            raise ValueError("stride must be positive")
        start_frame_ids = [int(value) for value in self.config.get("start_frame_ids", [])]
        if not start_frame_ids:
            segment_count = int(self.config.get("segment_count", 3))
            start_frame_ids = [idx * total_length * stride for idx in range(segment_count)]

        frame_by_id = {frame.original_order_index: frame for frame in scene.frames}
        requested_ids = sorted(
            {
                start + offset * stride
                for start in start_frame_ids
                for offset in range(total_length)
            }
        )
        missing = [frame_id for frame_id in requested_ids if frame_id not in frame_by_id]
        if missing:
            raise ValueError(
                f"Required raw frame ids are missing or invalid for {scene.scene_id}: {missing[:20]}"
            )

        manifests: list[dict[str, Any]] = []
        for segment_number, start_frame_id in enumerate(start_frame_ids):
            segment_id = f"seg{segment_number:03d}"
            b1_ids = [start_frame_id + offset * stride for offset in range(total_length)]
            target_frame_id = b1_ids[-1]
            x_ids = [target_frame_id for _ in range(total_length)]

            manifests.append(
                self._build_sequence_manifest(
                    scene,
                    segment_id=segment_id,
                    segment_number=segment_number,
                    sequence_role="B1",
                    frame_ids=b1_ids,
                    target_frame_id=target_frame_id,
                    x_repeated_target_frame_id=None,
                    frame_by_id=frame_by_id,
                    total_length=total_length,
                    stride=stride,
                )
            )
            manifests.append(
                self._build_sequence_manifest(
                    scene,
                    segment_id=segment_id,
                    segment_number=segment_number,
                    sequence_role="X",
                    frame_ids=x_ids,
                    target_frame_id=target_frame_id,
                    x_repeated_target_frame_id=target_frame_id,
                    frame_by_id=frame_by_id,
                    total_length=total_length,
                    stride=stride,
                )
            )
        return manifests

    def _build_sequence_manifest(
        self,
        scene: ScanNetScene,
        *,
        segment_id: str,
        segment_number: int,
        sequence_role: str,
        frame_ids: list[int],
        target_frame_id: int,
        x_repeated_target_frame_id: int | None,
        frame_by_id: dict[int, ScanNetFrame],
        total_length: int,
        stride: int,
    ) -> dict[str, Any]:
        condition_id = f"{segment_id}_{sequence_role}"
        frames = [frame_by_id[frame_id] for frame_id in frame_ids]
        entries = self._frame_entries(
            frames,
            segment_id=segment_id,
            segment_number=segment_number,
            sequence_role=sequence_role,
            requested_frame_ids=frame_ids,
            target_frame_id=target_frame_id,
        )
        return self._manifest(
            scene,
            condition_id=condition_id,
            seq_len=total_length,
            frames=entries,
            metadata=self._metadata(
                segment_id=segment_id,
                segment_number=segment_number,
                sequence_role=sequence_role,
                frame_ids=frame_ids,
                frames=frames,
                target_frame_id=target_frame_id,
                x_repeated_target_frame_id=x_repeated_target_frame_id,
                total_length=total_length,
                stride=stride,
            ),
        )

    def _frame_entries(
        self,
        frames: list[ScanNetFrame],
        *,
        segment_id: str,
        segment_number: int,
        sequence_role: str,
        requested_frame_ids: list[int],
        target_frame_id: int,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        last_t = len(frames) - 1
        for t, (frame, requested_id) in enumerate(zip(frames, requested_frame_ids)):
            is_target = t == last_t
            role = "target_current_X" if is_target else f"{sequence_role}_history"
            entries.append(
                self._frame_entry(
                    frame,
                    t=t,
                    role=role,
                    sequence_name=f"{segment_id}_{sequence_role}",
                    sequence_role=sequence_role,
                    segment_id=segment_id,
                    segment_number=int(segment_number),
                    requested_raw_frame_id=int(requested_id),
                    target_frame_id=int(target_frame_id),
                    is_target_frame=is_target,
                    prefix_position=None if is_target else t,
                    original_segment_position=t if sequence_role == "B1" else None,
                    x_repeat_id=t if sequence_role == "X" else None,
                )
            )
        return entries

    def _metadata(
        self,
        *,
        segment_id: str,
        segment_number: int,
        sequence_role: str,
        frame_ids: list[int],
        frames: list[ScanNetFrame],
        target_frame_id: int,
        x_repeated_target_frame_id: int | None,
        total_length: int,
        stride: int,
    ) -> dict[str, Any]:
        prefix_ids = frame_ids[:-1]
        roles = [f"{sequence_role}_history" for _ in prefix_ids] + ["target_current_X"]
        return {
            "experiment_name": self.config.get("experiment_name", "controlled_multi_length_exp_v5"),
            "sequence_name": f"{segment_id}_{sequence_role}",
            "sequence_role": sequence_role,
            "segment_id": segment_id,
            "segment_number": int(segment_number),
            "target_frame_id": int(target_frame_id),
            "target_t": int(total_length - 1),
            "prefix_frame_ids": prefix_ids,
            "full_frame_ids": frame_ids,
            "x_repeated_target_frame_id": (
                int(x_repeated_target_frame_id) if x_repeated_target_frame_id is not None else None
            ),
            "resolved_source_frame_ids": [frame.source_frame_id for frame in frames],
            "actual_image_paths": [str(frame.rgb_path) for frame in frames],
            "actual_depth_paths": [str(frame.depth_path) if frame.depth_path else None for frame in frames],
            "actual_pose_paths": [str(frame.pose_path) if frame.pose_path else None for frame in frames],
            "actual_intrinsic_paths": [
                str(frame.intrinsic_path) if frame.intrinsic_path else None
                for frame in frames
            ],
            "segment_start_frame_id": int(frame_ids[0]),
            "segment_end_frame_id": int(target_frame_id),
            "stride": int(stride),
            "seq_len": int(total_length),
            "frame_pattern": frame_pattern(roles),
            "order_policy": (
                "normal contiguous B1 segment"
                if sequence_role == "B1"
                else "target frame repeated for every stream position"
            ),
            "setting_params": {
                "scene_id": self.config.get("scene_id"),
                "sequence_total_length": int(total_length),
                "stride": int(stride),
                "start_frame_ids": [int(value) for value in self.config.get("start_frame_ids", [])],
                "token_type": self.config.get("token_type", "image_tokens"),
                "extract_last_frame_only": bool(self.config.get("extract_last_frame_only", True)),
                "layers": [int(value) for value in self.config.get("layers", [])],
            },
            "all_frames_found": True,
            "notes": "128-frame B1/X pair for controlled_multi_length_exp_v5.",
        }


def _scene_matches(config_scene_id: str, actual_scene_id: str) -> bool:
    return actual_scene_id == config_scene_id or actual_scene_id.startswith(f"{config_scene_id}_")
