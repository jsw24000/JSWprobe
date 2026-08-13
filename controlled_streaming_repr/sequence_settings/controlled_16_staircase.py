"""Controlled 16-frame staircase sequence setting for exp v4."""

from __future__ import annotations

from typing import Any

from data.manifest_utils.scannet_reader import ScanNetFrame, ScanNetScene

from .base_setting import BaseSequenceSetting, frame_pattern
from .registry import register_sequence_setting


@register_sequence_setting("controlled_16_staircase")
class Controlled16StaircaseSetting(BaseSequenceSetting):
    """Build staircase prefixes by replacing C suffix frames with B1 suffix frames."""

    default_replacement_order = "suffix"

    def build_manifests(self, scene: ScanNetScene) -> list[dict[str, Any]]:
        expected_scene = self.config.get("scene_id")
        if expected_scene and not _scene_matches(str(expected_scene), scene.scene_id):
            raise ValueError(
                f"controlled_16_staircase expected scene_id={expected_scene}, got {scene.scene_id}"
            )

        total_length = int(self.config.get("sequence_total_length", self.seq_lens[0]))
        prefix_length = int(self.config.get("prefix_length", total_length - 1))
        target_frame_id = int(self.config.get("target_frame_id"))
        if total_length != 16 or prefix_length != 15:
            raise ValueError(
                "controlled_16_staircase is intentionally fixed to 16 total frames "
                f"with 15 prefix frames; got total_length={total_length}, prefix_length={prefix_length}"
            )
        if any(seq_len != total_length for seq_len in self.seq_lens):
            raise ValueError(
                f"controlled_16_staircase uses sequence_total_length={total_length}; "
                f"received seq_lens={self.seq_lens}"
            )

        references = self.config.get("reference_sequences") or self.config.get("sequences") or {}
        if not isinstance(references, dict) or not references:
            raise ValueError("controlled_16_staircase requires reference_sequences copied from v3")

        relevant_name = str(self.config.get("relevant_sequence_name", "B1_local_relevant"))
        unrelated_name = str(self.config.get("unrelated_sequence_name", "C"))
        x_name = str(self.config.get("x_sequence_name", "X"))
        unrelated_reference_prefix = str(
            self.config.get("unrelated_reference_prefix")
            or ("X" if unrelated_name == x_name else "U")
        )
        replacement_order = str(self.config.get("replacement_order", self.default_replacement_order)).lower()
        if replacement_order not in {"prefix", "suffix"}:
            raise ValueError(f"replacement_order must be 'prefix' or 'suffix'; got {replacement_order!r}")

        relevant_ids = _sequence_frame_ids(references, relevant_name, total_length)
        unrelated_ids = _sequence_frame_ids(references, unrelated_name, total_length)
        x_ids = _sequence_frame_ids(references, x_name, total_length)
        for name, ids in ((relevant_name, relevant_ids), (unrelated_name, unrelated_ids), (x_name, x_ids)):
            if ids[-1] != target_frame_id:
                raise ValueError(f"{name} must end in target_frame_id={target_frame_id}; got {ids[-1]}")
        if any(frame_id != target_frame_id for frame_id in x_ids):
            raise ValueError(f"{x_name} must be the v3 repeated-target X baseline; got {x_ids}")

        requested_ids = sorted(set(relevant_ids + unrelated_ids + x_ids))
        missing = _missing_frame_ids(scene.frames, requested_ids)
        if missing:
            raise ValueError(
                f"Required raw frame ids are missing or invalid for {scene.scene_id}: {missing}. "
                f"Available valid frame ids: {_available_frame_summary(scene.frames)}"
            )

        frame_by_id = {frame.original_order_index: frame for frame in scene.frames}
        relevant_prefix = relevant_ids[:-1]
        unrelated_prefix = unrelated_ids[:-1]
        manifests: list[dict[str, Any]] = []
        for k in range(prefix_length + 1):
            condition_id = f"staircase_k{k:02d}"
            prefix_ids = _staircase_prefix_ids(
                k=k,
                prefix_length=prefix_length,
                relevant_prefix=relevant_prefix,
                unrelated_prefix=unrelated_prefix,
                replacement_order=replacement_order,
            )
            frame_ids = prefix_ids + [target_frame_id]
            source_plan = _source_plan(
                k=k,
                prefix_length=prefix_length,
                relevant_prefix=relevant_prefix,
                unrelated_prefix=unrelated_prefix,
                target_frame_id=target_frame_id,
                relevant_name=relevant_name,
                unrelated_name=unrelated_name,
                unrelated_reference_prefix=unrelated_reference_prefix,
                replacement_order=replacement_order,
            )
            frames = [frame_by_id[frame_id] for frame_id in frame_ids]
            entries = self._frame_entries(
                frames,
                source_plan=source_plan,
                condition_id=condition_id,
                k=k,
                target_frame_id=target_frame_id,
                sequence_role=_sequence_role(replacement_order),
            )
            manifests.append(
                self._manifest(
                    scene,
                    condition_id=condition_id,
                    seq_len=total_length,
                    frames=entries,
                    metadata=self._metadata(
                        condition_id=condition_id,
                        k=k,
                        frame_ids=frame_ids,
                        source_plan=source_plan,
                        target_frame_id=target_frame_id,
                        total_length=total_length,
                        prefix_length=prefix_length,
                        relevant_name=relevant_name,
                        unrelated_name=unrelated_name,
                        x_name=x_name,
                        relevant_ids=relevant_ids,
                        unrelated_ids=unrelated_ids,
                        x_ids=x_ids,
                        unrelated_reference_prefix=unrelated_reference_prefix,
                        replacement_order=replacement_order,
                    ),
                )
            )

        _validate_staircase_endpoints(
            manifests,
            relevant_ids=relevant_ids,
            unrelated_ids=unrelated_ids,
            target_frame_id=target_frame_id,
            prefix_length=prefix_length,
            replacement_order=replacement_order,
        )
        return manifests

    def _frame_entries(
        self,
        frames: list[ScanNetFrame],
        *,
        source_plan: list[dict[str, Any]],
        condition_id: str,
        k: int,
        target_frame_id: int,
        sequence_role: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for t, (frame, source) in enumerate(zip(frames, source_plan)):
            is_target = bool(source["is_target_frame"])
            entries.append(
                self._frame_entry(
                    frame,
                    t=t,
                    role=str(source["role"]),
                    sequence_name=condition_id,
                    sequence_role=sequence_role,
                    requested_raw_frame_id=int(source["frame_id"]),
                    target_frame_id=int(target_frame_id),
                    is_target_frame=is_target,
                    prefix_position=None if is_target else t,
                    staircase_k=int(k),
                    staircase_condition_id=condition_id,
                    source_sequence=source["source_sequence"],
                    source_reference=source["source_reference"],
                    source_reference_position=source["source_reference_position"],
                    is_relevant_history=bool(source["is_relevant_history"]),
                    is_unrelated_history=bool(source["is_unrelated_history"]),
                    replaced_unrelated_frame_id=source.get("replaced_unrelated_frame_id"),
                    replaced_unrelated_reference=source.get("replaced_unrelated_reference"),
                )
            )
        return entries

    def _metadata(
        self,
        *,
        condition_id: str,
        k: int,
        frame_ids: list[int],
        source_plan: list[dict[str, Any]],
        target_frame_id: int,
        total_length: int,
        prefix_length: int,
        relevant_name: str,
        unrelated_name: str,
        x_name: str,
        relevant_ids: list[int],
        unrelated_ids: list[int],
        x_ids: list[int],
        unrelated_reference_prefix: str,
        replacement_order: str,
    ) -> dict[str, Any]:
        roles = [str(item["role"]) for item in source_plan]
        return {
            "experiment_name": self.config.get("experiment_name", "controlled_16_staircase_exp_v4"),
            "sequence_name": condition_id,
            "sequence_role": _sequence_role(replacement_order),
            "staircase_k": int(k),
            "staircase_replacement_order": replacement_order,
            "num_relevant_history_frames": int(k),
            "num_unrelated_history_frames": int(prefix_length - k),
            "target_frame_id": int(target_frame_id),
            "prefix_frame_ids": frame_ids[:-1],
            "full_frame_ids": frame_ids,
            "target_t": total_length - 1,
            "seq_len": int(total_length),
            "frame_pattern": frame_pattern(roles),
            "order_policy": _order_policy(replacement_order, unrelated_reference_prefix),
            "source_reference_sequences": {
                "relevant_sequence_name": relevant_name,
                "unrelated_sequence_name": unrelated_name,
                "x_sequence_name": x_name,
                "reference_v3_project": self.config.get("reference_v3_project", "controlled_16_exp_v3"),
                "reference_v3_setting": self.config.get("reference_v3_setting", "prefix_induced_modulation"),
                "B1_frame_ids": relevant_ids,
                "C_frame_ids": unrelated_ids,
                "X_frame_ids": x_ids,
            },
            "setting_params": {
                "scene_id": self.config.get("scene_id"),
                "target_frame_id": int(target_frame_id),
                "sequence_total_length": int(total_length),
                "prefix_length": int(prefix_length),
                "token_type": self.config.get("token_type", "image_tokens"),
                "extract_last_frame_only": bool(self.config.get("extract_last_frame_only", True)),
                "layers": [int(value) for value in self.config.get("layers", [])],
                "replacement_order": replacement_order,
                "unrelated_reference_prefix": unrelated_reference_prefix,
                "construction": _construction(prefix_length, replacement_order, unrelated_reference_prefix),
            },
            "staircase_frame_manifest": source_plan,
            "all_frames_found": True,
            "notes": "Derived only from v3 B1/C/X frame ids; no new frame selection is performed.",
        }


@register_sequence_setting("controlled_16_staircase_x_to_B1")
class Controlled16StaircaseXToB1Setting(Controlled16StaircaseSetting):
    """Build X-to-B1 staircase prefixes by replacing X suffix frames with B1 suffix frames."""

    default_replacement_order = "suffix"


@register_sequence_setting("controlled_16_staircase_reverse")
class Controlled16StaircaseReverseSetting(Controlled16StaircaseSetting):
    """Build staircase prefixes by replacing C prefix frames with B1 prefix frames."""

    default_replacement_order = "prefix"


@register_sequence_setting("controlled_16_staircase_x_to_B1_reversed")
class Controlled16StaircaseXToB1ReversedSetting(Controlled16StaircaseSetting):
    """Build X-to-B1 staircase prefixes by replacing X prefix frames with B1 prefix frames."""

    default_replacement_order = "prefix"


def _source_plan(
    *,
    k: int,
    prefix_length: int,
    relevant_prefix: list[int],
    unrelated_prefix: list[int],
    target_frame_id: int,
    relevant_name: str,
    unrelated_name: str,
    unrelated_reference_prefix: str,
    replacement_order: str,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for position in range(prefix_length):
        use_relevant = _uses_relevant(
            position=position,
            k=k,
            prefix_length=prefix_length,
            replacement_order=replacement_order,
        )
        source_sequence = relevant_name if use_relevant else unrelated_name
        frame_id = relevant_prefix[position] if use_relevant else unrelated_prefix[position]
        reference_prefix = "R" if use_relevant else unrelated_reference_prefix
        role = _history_role(
            use_relevant=use_relevant,
            replacement_order=replacement_order,
            unrelated_name=unrelated_name,
        )
        plan.append(
            {
                "t": position,
                "frame_id": int(frame_id),
                "source_sequence": source_sequence,
                "source_reference": f"{reference_prefix}{position + 1:02d}",
                "source_reference_position": int(position + 1),
                "role": role,
                "is_relevant_history": bool(use_relevant),
                "is_unrelated_history": not use_relevant,
                "is_target_frame": False,
                "replaced_unrelated_frame_id": int(unrelated_prefix[position]) if use_relevant else None,
                "replaced_unrelated_reference": f"{unrelated_reference_prefix}{position + 1:02d}" if use_relevant else None,
            }
        )
    plan.append(
        {
            "t": prefix_length,
            "frame_id": int(target_frame_id),
            "source_sequence": "target_0600",
            "source_reference": "target_0600",
            "source_reference_position": None,
            "role": "target_current_X",
            "is_relevant_history": False,
            "is_unrelated_history": False,
            "is_target_frame": True,
            "replaced_unrelated_frame_id": None,
            "replaced_unrelated_reference": None,
        }
    )
    return plan


def _staircase_prefix_ids(
    *,
    k: int,
    prefix_length: int,
    relevant_prefix: list[int],
    unrelated_prefix: list[int],
    replacement_order: str,
) -> list[int]:
    return [
        relevant_prefix[position]
        if _uses_relevant(
            position=position,
            k=k,
            prefix_length=prefix_length,
            replacement_order=replacement_order,
        )
        else unrelated_prefix[position]
        for position in range(prefix_length)
    ]


def _uses_relevant(*, position: int, k: int, prefix_length: int, replacement_order: str) -> bool:
    if replacement_order == "prefix":
        return position < k
    if replacement_order == "suffix":
        return position >= prefix_length - k
    raise ValueError(f"Unknown replacement_order: {replacement_order}")


def _sequence_role(replacement_order: str) -> str:
    return "staircase_relevant_prefix" if replacement_order == "prefix" else "staircase_relevant_suffix"


def _history_role(*, use_relevant: bool, replacement_order: str, unrelated_name: str) -> str:
    side = "prefix" if replacement_order == "prefix" else "suffix"
    opposite = "suffix" if replacement_order == "prefix" else "prefix"
    if use_relevant:
        return f"relevant_B1_{side}"
    unrelated_label = "target_X" if unrelated_name == "X" else f"unrelated_{unrelated_name}"
    return f"{unrelated_label}_{opposite}"


def _order_policy(replacement_order: str, unrelated_reference_prefix: str = "U") -> str:
    if replacement_order == "prefix":
        return f"R prefix followed by a contiguous {unrelated_reference_prefix} suffix, then fixed target_0600"
    return f"{unrelated_reference_prefix} prefix followed by a contiguous R suffix, then fixed target_0600"


def _construction(prefix_length: int, replacement_order: str, unrelated_reference_prefix: str = "U") -> str:
    if replacement_order == "prefix":
        return f"S_k = R01..Rk, {unrelated_reference_prefix}(k+1)..{unrelated_reference_prefix}{prefix_length:02d}, target_0600"
    return (
        f"S_k = {unrelated_reference_prefix}01..{unrelated_reference_prefix}({prefix_length}-k), "
        f"R({prefix_length + 1}-k)..R{prefix_length:02d}, target_0600"
    )


def _sequence_frame_ids(references: dict[str, Any], name: str, total_length: int) -> list[int]:
    if name not in references:
        available = ", ".join(sorted(str(key) for key in references))
        raise ValueError(f"Missing reference sequence {name!r}; available: {available}")
    values = references[name].get("frame_ids", [])
    frame_ids = [int(value) for value in values]
    if len(frame_ids) != total_length:
        raise ValueError(f"{name} has {len(frame_ids)} frames; expected {total_length}")
    return frame_ids


def _validate_staircase_endpoints(
    manifests: list[dict[str, Any]],
    *,
    relevant_ids: list[int],
    unrelated_ids: list[int],
    target_frame_id: int,
    prefix_length: int,
    replacement_order: str,
) -> None:
    if len(manifests) != prefix_length + 1:
        raise ValueError(f"Expected {prefix_length + 1} staircase manifests; got {len(manifests)}")
    by_k = {int(manifest["metadata"]["staircase_k"]): manifest for manifest in manifests}
    expected_keys = set(range(prefix_length + 1))
    if set(by_k) != expected_keys:
        raise ValueError(f"Unexpected staircase k values: {sorted(by_k)}")
    for k, manifest in by_k.items():
        frame_ids = [int(frame["original_order_index"]) for frame in manifest.get("frames", [])]
        expected = _staircase_prefix_ids(
            k=k,
            prefix_length=prefix_length,
            relevant_prefix=relevant_ids[:prefix_length],
            unrelated_prefix=unrelated_ids[:prefix_length],
            replacement_order=replacement_order,
        ) + [target_frame_id]
        if frame_ids != expected:
            raise ValueError(f"staircase_k{k:02d} frame ids {frame_ids} != expected {expected}")
        relevant_count = sum(1 for frame in manifest["frames"][:-1] if frame.get("is_relevant_history"))
        if relevant_count != k:
            raise ValueError(f"staircase_k{k:02d} has {relevant_count} relevant frames; expected {k}")
    if [int(frame["original_order_index"]) for frame in by_k[0]["frames"]] != unrelated_ids:
        raise ValueError("staircase_k00 does not match the configured unrelated baseline exactly")
    if [int(frame["original_order_index"]) for frame in by_k[prefix_length]["frames"]] != relevant_ids:
        raise ValueError("staircase_k15 does not match the configured relevant baseline exactly")


def _scene_matches(config_scene_id: str, actual_scene_id: str) -> bool:
    return actual_scene_id == config_scene_id or actual_scene_id.startswith(f"{config_scene_id}_")


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
