"""Repeated-frame controlled setting: A, A, A, ..."""

from __future__ import annotations

from typing import Any

from data.manifest_utils.scannet_reader import ScanNetScene

from .base_setting import BaseSequenceSetting, frame_pattern, uniform_indices
from .registry import register_sequence_setting


@register_sequence_setting("repeated_frame_stability")
class RepeatedFrameStabilitySetting(BaseSequenceSetting):
    """Generate repeated copies of a small set of anchor frames."""

    def build_manifests(self, scene: ScanNetScene) -> list[dict[str, Any]]:
        frames = scene.frames
        if not frames:
            return []

        num_anchors = int(self.config.get("num_anchors_per_scene", 3))
        avoid_ratio = float(self.config.get("avoid_boundary_ratio", 0.1))
        anchor_indices = uniform_indices(
            len(frames),
            num_anchors,
            avoid_boundary_ratio=avoid_ratio,
        )

        manifests: list[dict[str, Any]] = []
        for anchor_number, frame_index in enumerate(anchor_indices):
            anchor = frames[frame_index]
            for seq_len in self.seq_lens:
                roles = ["A"] * seq_len
                entries = [
                    self._frame_entry(
                        anchor,
                        t=t,
                        role="A",
                        repeat_id=t,
                    )
                    for t in range(seq_len)
                ]
                condition_id = f"repeat_A_seq{seq_len}_anchor{anchor_number:03d}"
                metadata: dict[str, Any] = {
                    "setting_params": {
                        "num_anchors_per_scene": num_anchors,
                        "anchor_strategy": self.config.get("anchor_strategy", "uniform_valid"),
                        "avoid_boundary_ratio": avoid_ratio,
                        "require_valid_pose": self.config.get("require_valid_pose", True),
                    },
                    "anchor_frame_id": anchor.source_frame_id,
                    "anchor_original_index": anchor.original_order_index,
                    "seq_len": seq_len,
                    "frame_pattern": frame_pattern(roles),
                    "notes": "",
                }
                manifests.append(
                    self._manifest(
                        scene,
                        condition_id=condition_id,
                        seq_len=seq_len,
                        frames=entries,
                        metadata=metadata,
                    )
                )
        return manifests
