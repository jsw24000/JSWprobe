"""Two-state alternation settings: A/B/A/B and A/Z/A/Z."""

from __future__ import annotations

from typing import Any

from data.manifest_utils.pose_utils import safe_pose_distances
from data.manifest_utils.scannet_reader import ScanNetFrame, ScanNetScene

from .base_setting import BaseSequenceSetting, frame_pattern, uniform_indices
from .registry import register_sequence_setting


@register_sequence_setting("two_state_alternation")
class TwoStateAlternationSetting(BaseSequenceSetting):
    """Generate near-pair and far-pair alternation manifests."""

    def build_manifests(self, scene: ScanNetScene) -> list[dict[str, Any]]:
        if len(scene.frames) < 2:
            return []

        manifests: list[dict[str, Any]] = []
        paired_states = self._select_paired_states(scene.frames)

        for pair_number, pair in enumerate(paired_states):
            anchor = pair["anchor"]
            near_other = pair["near_other"]
            near_stats = pair["near_stats"]
            far_other = pair["far_other"]
            far_stats = pair["far_stats"]
            pair_group_id = f"shared_A_{anchor.source_frame_id}_pair{pair_number:03d}"

            for pair_type, other_role, other, stats, prefix in (
                ("near", "B", near_other, near_stats, "abab_near"),
                ("far", "Z", far_other, far_stats, "azaz_far"),
            ):
                if pair_type not in pair.get("emit_pair_types", {"near", "far"}):
                    continue
                if other is None:
                    continue
                for seq_len in self.seq_lens:
                    roles = ["A" if t % 2 == 0 else other_role for t in range(seq_len)]
                    entries = [
                        self._frame_entry(
                            anchor if role == "A" else other,
                            t=t,
                            role=role,
                            alternation_id=t // 2,
                        )
                        for t, role in enumerate(roles)
                    ]
                    label = pair.get("label")
                    label_part = f"_{label}" if label else ""
                    condition_id = f"{prefix}{label_part}_seq{seq_len}_pair{pair_number:03d}"
                    metadata: dict[str, Any] = {
                        "setting_params": self._setting_params(),
                        "pair_group_id": pair_group_id,
                        "pair_number": pair_number,
                        "selection_mode": pair.get("selection_mode", "auto"),
                        "A_frame_id": anchor.source_frame_id,
                        "shared_A_frame_id": anchor.source_frame_id,
                        "paired_B_frame_id": near_other.source_frame_id if near_other is not None else None,
                        "paired_Z_frame_id": far_other.source_frame_id if far_other is not None else None,
                        f"{other_role}_frame_id": other.source_frame_id,
                        "pair_type": pair_type,
                        "index_gap": abs(other.original_order_index - anchor.original_order_index),
                        "pose_translation_distance": stats.get("translation"),
                        "pose_rotation_distance_deg": stats.get("rotation_deg"),
                        "near_index_gap": (
                            abs(near_other.original_order_index - anchor.original_order_index)
                            if near_other is not None
                            else None
                        ),
                        "near_pose_translation_distance": near_stats.get("translation"),
                        "near_pose_rotation_distance_deg": near_stats.get("rotation_deg"),
                        "far_index_gap": (
                            abs(far_other.original_order_index - anchor.original_order_index)
                            if far_other is not None
                            else None
                        ),
                        "far_pose_translation_distance": far_stats.get("translation"),
                        "far_pose_rotation_distance_deg": far_stats.get("rotation_deg"),
                        "seq_len": seq_len,
                        "frame_pattern": frame_pattern(roles),
                        "notes": pair.get(
                            "notes",
                            "near and far conditions with the same pair_number share the same A frame",
                        ),
                    }
                    if label:
                        metadata["pair_label"] = label
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

    def _setting_params(self) -> dict[str, Any]:
        return {
            "num_pairs_per_scene": int(self.config.get("num_pairs_per_scene", 3)),
            "near_gap_candidates": list(self.config.get("near_gap_candidates", [5, 10, 15])),
            "far_min_index_gap": int(self.config.get("far_min_index_gap", 80)),
            "far_max_index_gap": self.config.get("far_max_index_gap"),
            "prefer_pose_distance_for_far": bool(self.config.get("prefer_pose_distance_for_far", True)),
            "far_min_translation": float(self.config.get("far_min_translation", 0.8)),
            "if_pose_distance_unavailable": self.config.get("if_pose_distance_unavailable", "fallback_to_index_gap"),
            "require_valid_pose": self.config.get("require_valid_pose", True),
            "pairing_policy": (
                "manual_pairs"
                if self.config.get("manual_pairs")
                else "shared_A_for_near_and_far"
            ),
            "manual_pair_count": len(self.config.get("manual_pairs") or []),
        }

    def _select_paired_states(self, frames: list[ScanNetFrame]) -> list[dict[str, Any]]:
        manual_pairs = self.config.get("manual_pairs") or []
        if manual_pairs:
            return self._select_manual_paired_states(frames, manual_pairs)

        num_pairs = int(self.config.get("num_pairs_per_scene", 3))
        gaps = [int(gap) for gap in self.config.get("near_gap_candidates", [5, 10, 15])]
        max_gap = max(gaps) if gaps else 1
        min_far_gap = int(self.config.get("far_min_index_gap", 80))
        required_forward_gap = max(max_gap, min_far_gap)
        anchors = uniform_indices(
            len(frames),
            num_pairs,
            min_index=0,
            max_index=max(0, len(frames) - required_forward_gap - 1),
        )

        pairs: list[dict[str, Any]] = []
        for anchor_index in anchors:
            near = self._select_near_other(frames, anchor_index)
            far = self._select_far_other(frames, anchor_index)
            if near is None or far is None:
                continue
            near_other, near_stats = near
            far_other, far_stats = far
            pairs.append(
                {
                    "anchor": frames[anchor_index],
                    "near_other": near_other,
                    "near_stats": near_stats,
                    "far_other": far_other,
                    "far_stats": far_stats,
                }
            )
        return pairs[:num_pairs]

    def _select_manual_paired_states(
        self,
        frames: list[ScanNetFrame],
        manual_pairs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for pair_config in manual_pairs:
            if not isinstance(pair_config, dict):
                continue

            anchor = self._lookup_frame(
                frames,
                pair_config.get("A_frame_id")
                or pair_config.get("anchor_frame_id")
                or pair_config.get("a_frame_id")
                or pair_config.get("A"),
            )
            if anchor is None:
                continue

            near_other = self._lookup_frame(
                frames,
                pair_config.get("B_frame_id")
                or pair_config.get("near_frame_id")
                or pair_config.get("b_frame_id")
                or pair_config.get("B"),
            )
            far_other = self._lookup_frame(
                frames,
                pair_config.get("Z_frame_id")
                or pair_config.get("far_frame_id")
                or pair_config.get("z_frame_id")
                or pair_config.get("Z"),
            )
            if near_other is None and far_other is None:
                continue

            emit_pair_types = pair_config.get("emit_pair_types")
            if emit_pair_types is None:
                emit = set()
                if near_other is not None:
                    emit.add("near")
                if far_other is not None:
                    emit.add("far")
            else:
                emit = {str(value) for value in emit_pair_types}

            pairs.append(
                {
                    "anchor": anchor,
                    "near_other": near_other,
                    "near_stats": (
                        safe_pose_distances(anchor.pose_path, near_other.pose_path)
                        if near_other is not None
                        else {}
                    ),
                    "far_other": far_other,
                    "far_stats": (
                        safe_pose_distances(anchor.pose_path, far_other.pose_path)
                        if far_other is not None
                        else {}
                    ),
                    "selection_mode": "manual",
                    "emit_pair_types": emit,
                    "label": pair_config.get("label") or pair_config.get("pair_label"),
                    "notes": pair_config.get("notes", "manual A/B or A/Z pair"),
                }
            )
        return pairs

    def _lookup_frame(self, frames: list[ScanNetFrame], frame_id: Any) -> ScanNetFrame | None:
        if frame_id is None:
            return None
        key = str(frame_id).strip()
        if not key:
            return None

        by_source_id = {frame.source_frame_id: frame for frame in frames}
        if key in by_source_id:
            return by_source_id[key]

        if key.isdigit():
            numeric_id = int(key)
            for frame in frames:
                if frame.original_order_index == numeric_id:
                    return frame
        return None

    def _select_near_other(
        self,
        frames: list[ScanNetFrame],
        anchor_index: int,
    ) -> tuple[ScanNetFrame, dict[str, Any]] | None:
        gaps = [int(gap) for gap in self.config.get("near_gap_candidates", [5, 10, 15])]
        anchor = frames[anchor_index]
        for gap in gaps:
            other_index = anchor_index + gap
            if other_index >= len(frames):
                continue
            other = frames[other_index]
            stats = safe_pose_distances(anchor.pose_path, other.pose_path)
            return other, stats
        return None

    def _select_far_other(
        self,
        frames: list[ScanNetFrame],
        anchor_index: int,
    ) -> tuple[ScanNetFrame, dict[str, Any]] | None:
        min_gap = int(self.config.get("far_min_index_gap", 80))
        max_gap_value = self.config.get("far_max_index_gap")
        max_gap = int(max_gap_value) if max_gap_value is not None else None
        min_translation = float(self.config.get("far_min_translation", 0.8))
        prefer_pose = bool(self.config.get("prefer_pose_distance_for_far", True))
        anchor = frames[anchor_index]

        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for other_index in range(len(frames)):
            gap = abs(frames[other_index].original_order_index - anchor.original_order_index)
            if gap < min_gap:
                continue
            if max_gap is not None and gap > max_gap:
                continue
            stats = safe_pose_distances(anchor.pose_path, frames[other_index].pose_path)
            translation = stats.get("translation")
            if prefer_pose and translation is not None and translation < min_translation:
                continue
            score = float(translation if translation is not None else gap)
            candidates.append((score, other_index, stats))
        if candidates:
            _, other_index, stats = max(candidates, key=lambda item: item[0])
            return frames[other_index], stats

        fallback_index = min(len(frames) - 1, anchor_index + min_gap)
        if fallback_index == anchor_index and len(frames) > 1:
            fallback_index = len(frames) - 1
        if fallback_index == anchor_index:
            return None
        stats = safe_pose_distances(anchor.pose_path, frames[fallback_index].pose_path)
        return frames[fallback_index], stats
