"""Order perturbation setting: normal, reverse, and deterministic shuffle."""

from __future__ import annotations

from typing import Any

from data.manifest_utils.scannet_reader import ScanNetFrame, ScanNetScene

from .base_setting import BaseSequenceSetting, frame_pattern, uniform_indices
from .registry import register_sequence_setting


@register_sequence_setting("order_perturbation")
class OrderPerturbationSetting(BaseSequenceSetting):
    """Generate order variants over the same frame segment."""

    def build_manifests(self, scene: ScanNetScene) -> list[dict[str, Any]]:
        if not scene.frames:
            return []

        manifests: list[dict[str, Any]] = []
        num_segments = int(self.config.get("num_segments_per_scene", 3))
        strides = [int(value) for value in self.config.get("base_stride_candidates", [5, 10, 15])]
        for seq_len in self.seq_lens:
            for stride in strides[:1] if strides else [1]:
                max_start = len(scene.frames) - (seq_len - 1) * stride - 1
                if max_start < 0:
                    continue
                starts = uniform_indices(len(scene.frames), num_segments, min_index=0, max_index=max_start)
                for segment_number, start in enumerate(starts):
                    segment = [scene.frames[start + t * stride] for t in range(seq_len)]
                    manifests.extend(
                        self._build_segment_manifests(
                            scene,
                            segment=segment,
                            segment_number=segment_number,
                            seq_len=seq_len,
                            stride=stride,
                        )
                    )
        return manifests

    def _build_segment_manifests(
        self,
        scene: ScanNetScene,
        *,
        segment: list[ScanNetFrame],
        segment_number: int,
        seq_len: int,
        stride: int,
    ) -> list[dict[str, Any]]:
        variants = {
            "normal": list(range(seq_len)),
            "reverse": list(reversed(range(seq_len))),
            "shuffle": deterministic_interleave_permutation(seq_len),
        }
        manifests: list[dict[str, Any]] = []
        original_frame_ids = [frame.source_frame_id for frame in segment]
        for variant_name, permutation in variants.items():
            ordered = [segment[index] for index in permutation]
            roles = [chr(ord("A") + min(index, 25)) for index in permutation]
            entries = [
                self._frame_entry(
                    frame,
                    t=t,
                    role=roles[t],
                    original_segment_position=permutation[t],
                    permutation_index=permutation[t],
                )
                for t, frame in enumerate(ordered)
            ]
            condition_id = f"order_{variant_name}_seq{seq_len}_seg{segment_number:03d}"
            metadata: dict[str, Any] = {
                "setting_params": {
                    "num_segments_per_scene": int(self.config.get("num_segments_per_scene", 3)),
                    "base_stride_candidates": list(self.config.get("base_stride_candidates", [5, 10, 15])),
                    "shuffle_seed": int(self.config.get("shuffle_seed", 2026)),
                    "shuffle_mode": self.config.get("shuffle_mode", "deterministic_interleave"),
                    "require_valid_pose": self.config.get("require_valid_pose", True),
                },
                "order_variant": variant_name,
                "original_frame_ids": original_frame_ids,
                "used_permutation": permutation,
                "stride": stride,
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


def deterministic_interleave_permutation(seq_len: int) -> list[int]:
    """Return a reproducible non-random permutation for shuffle_fixed.

    For length 8 this returns ``[0, 3, 1, 5, 2, 7, 4, 6]``, matching the
    motivating example while extending deterministically to longer sequences.
    """

    if seq_len <= 2:
        return list(range(seq_len))
    high_stride = list(range(max(1, seq_len // 2 - 1), seq_len, 2))
    used: set[int] = set()
    permutation: list[int] = []
    for low in range((seq_len + 1) // 2):
        if low not in used:
            permutation.append(low)
            used.add(low)
        if high_stride:
            high = high_stride.pop(0)
            if high not in used:
                permutation.append(high)
                used.add(high)
    permutation.extend(index for index in range(seq_len) if index not in used)
    return permutation
