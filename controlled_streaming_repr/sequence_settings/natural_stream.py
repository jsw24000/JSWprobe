"""Natural stream setting: contiguous ScanNet frames in original order."""

from __future__ import annotations

from typing import Any

from data.manifest_utils.scannet_reader import ScanNetFrame, ScanNetScene

from .base_setting import BaseSequenceSetting, frame_pattern, uniform_indices
from .registry import register_sequence_setting


@register_sequence_setting("natural_stream")
class NaturalStreamSetting(BaseSequenceSetting):
    """Generate unmodified contiguous frame streams from a scene."""

    def build_manifests(self, scene: ScanNetScene) -> list[dict[str, Any]]:
        if not scene.frames:
            return []

        manifests: list[dict[str, Any]] = []
        stride = max(1, int(self.config.get("stride", 1)))
        num_segments = max(1, int(self.config.get("num_segments_per_scene", 1)))
        start_indices = self.config.get("start_indices")
        start_strategy = str(self.config.get("start_strategy", "beginning")).lower()

        for seq_len in self.seq_lens:
            max_start = len(scene.frames) - 1 - (seq_len - 1) * stride
            if max_start < 0:
                continue
            if start_indices is not None:
                starts = [int(value) for value in start_indices if 0 <= int(value) <= max_start]
            elif start_strategy == "uniform":
                starts = uniform_indices(len(scene.frames), num_segments, min_index=0, max_index=max_start)
            else:
                starts = [0]

            for segment_number, start in enumerate(starts[:num_segments]):
                segment = [scene.frames[start + t * stride] for t in range(seq_len)]
                manifests.append(
                    self._build_stream_manifest(
                        scene,
                        segment=segment,
                        segment_number=segment_number,
                        seq_len=seq_len,
                        start=start,
                        stride=stride,
                    )
                )
        return manifests

    def _build_stream_manifest(
        self,
        scene: ScanNetScene,
        *,
        segment: list[ScanNetFrame],
        segment_number: int,
        seq_len: int,
        start: int,
        stride: int,
    ) -> dict[str, Any]:
        roles = ["N" for _ in segment]
        entries = [
            self._frame_entry(
                frame,
                t=t,
                role=roles[t],
                original_segment_position=t,
                natural_stream_index=t,
            )
            for t, frame in enumerate(segment)
        ]
        condition_id = f"natural_seq{seq_len}_seg{segment_number:03d}"
        metadata: dict[str, Any] = {
            "setting_params": {
                "num_segments_per_scene": int(self.config.get("num_segments_per_scene", 1)),
                "start_strategy": self.config.get("start_strategy", "beginning"),
                "start_indices": self.config.get("start_indices"),
                "stride": stride,
                "require_valid_pose": self.config.get("require_valid_pose", True),
            },
            "order_variant": "natural",
            "start_valid_index": start,
            "end_valid_index": start + (seq_len - 1) * stride,
            "start_frame_id": segment[0].source_frame_id,
            "end_frame_id": segment[-1].source_frame_id,
            "source_frame_ids": [frame.source_frame_id for frame in segment],
            "stride": stride,
            "seq_len": seq_len,
            "frame_pattern": frame_pattern(roles),
            "notes": "Contiguous ScanNet stream in original order; no repetition, alternation, shuffle, or reverse.",
        }
        return self._manifest(
            scene,
            condition_id=condition_id,
            seq_len=seq_len,
            frames=entries,
            metadata=metadata,
        )
