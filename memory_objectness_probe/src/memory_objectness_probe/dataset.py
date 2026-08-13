from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import (
    EXPECTED_VARIANTS,
    VARIANT_TO_POSITION,
    finite_stats,
    list_images,
    read_json,
    read_jsonl,
    read_manifest_csv,
    stable_sample_id,
    write_json,
    write_jsonl,
)


@dataclass
class SampleRecord:
    sample_id: str
    base_scene_id: str
    variant_id: str
    scene_path: Path
    metadata_path: Path
    target_present: int
    target_position_id: int
    probe_frame: int
    target_world_position: Optional[Any] = None
    frame15_target_mask_area: Optional[float] = None
    frame15_target_visible_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def rgb_dir(self) -> Path:
        return self.scene_path / "rgb"

    @property
    def cameras_path(self) -> Path:
        return self.scene_path / "cameras.json"

    @property
    def probe_rgb_path(self) -> Path:
        return self.rgb_dir / f"frame_{self.probe_frame:04d}.png"

    @property
    def probe_target_mask_path(self) -> Path:
        return self.scene_path / f"target_mask_frame_{self.probe_frame:04d}.png"

    def to_manifest_row(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "base_scene_id": self.base_scene_id,
            "variant_id": self.variant_id,
            "scene_path": str(self.scene_path),
            "metadata_path": str(self.metadata_path),
            "target_present": int(self.target_present),
            "target_position_id": int(self.target_position_id),
            "probe_frame": int(self.probe_frame),
            "frame15_target_mask_area": self.frame15_target_mask_area,
            "frame15_target_visible_ratio": self.frame15_target_visible_ratio,
        }


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _metadata_record(scene_path: Path) -> Optional[SampleRecord]:
    metadata_path = scene_path / "metadata.json"
    if not metadata_path.exists():
        return None
    meta = read_json(metadata_path)
    base_scene_id = str(meta["base_scene_id"])
    variant_id = str(meta["variant_id"])
    probe_frame = int(meta.get("probe_frame", 15))
    return SampleRecord(
        sample_id=stable_sample_id(base_scene_id, variant_id),
        base_scene_id=base_scene_id,
        variant_id=variant_id,
        scene_path=scene_path.resolve(),
        metadata_path=metadata_path.resolve(),
        target_present=_coerce_int(meta.get("target_present")),
        target_position_id=_coerce_int(meta.get("target_position_id")),
        probe_frame=probe_frame,
        target_world_position=meta.get("target_world_position"),
        frame15_target_mask_area=_coerce_float(meta.get("frame15_target_mask_area")),
        frame15_target_visible_ratio=_coerce_float(meta.get("frame15_target_visible_ratio")),
        metadata=meta,
    )


def load_samples(data_root: str | Path) -> List[SampleRecord]:
    data_root = Path(data_root).resolve()
    manifest_jsonl = data_root / "manifest.jsonl"
    manifest_csv = data_root / "manifest.csv"

    if manifest_jsonl.exists():
        rows = read_jsonl(manifest_jsonl)
    elif manifest_csv.exists():
        rows = read_manifest_csv(manifest_csv)
    else:
        samples = []
        for path in sorted(data_root.glob("base_scene_*/*/metadata.json")):
            rec = _metadata_record(path.parent)
            if rec is not None:
                samples.append(rec)
        return sorted(samples, key=lambda r: (r.base_scene_id, r.target_position_id, r.variant_id))

    samples: List[SampleRecord] = []
    for row in rows:
        scene_path = Path(row["scene_path"])
        if not scene_path.is_absolute():
            scene_path = (data_root / scene_path).resolve()
        metadata_path = scene_path / "metadata.json"
        meta = read_json(metadata_path) if metadata_path.exists() else {}
        base_scene_id = str(row.get("base_scene_id", meta.get("base_scene_id")))
        variant_id = str(row.get("variant_id", meta.get("variant_id")))
        probe_frame = _coerce_int(row.get("probe_frame", meta.get("probe_frame")), 15)
        samples.append(
            SampleRecord(
                sample_id=stable_sample_id(base_scene_id, variant_id),
                base_scene_id=base_scene_id,
                variant_id=variant_id,
                scene_path=scene_path.resolve(),
                metadata_path=metadata_path.resolve(),
                target_present=_coerce_int(row.get("target_present", meta.get("target_present"))),
                target_position_id=_coerce_int(row.get("target_position_id", meta.get("target_position_id"))),
                probe_frame=probe_frame,
                target_world_position=meta.get("target_world_position"),
                frame15_target_mask_area=_coerce_float(
                    row.get("frame15_target_mask_area", meta.get("frame15_target_mask_area"))
                ),
                frame15_target_visible_ratio=_coerce_float(
                    row.get("frame15_target_visible_ratio", meta.get("frame15_target_visible_ratio"))
                ),
                metadata=meta,
            )
        )
    return sorted(samples, key=lambda r: (r.base_scene_id, r.target_position_id, r.variant_id))


def group_by_base_scene(samples: Sequence[SampleRecord]) -> Dict[str, List[SampleRecord]]:
    groups: Dict[str, List[SampleRecord]] = {}
    for sample in samples:
        groups.setdefault(sample.base_scene_id, []).append(sample)
    for group in groups.values():
        group.sort(key=lambda r: (r.target_position_id, r.variant_id))
    return dict(sorted(groups.items()))


def _pose_signature(sample: SampleRecord) -> Optional[List[Any]]:
    cameras_path = sample.cameras_path
    if not cameras_path.exists():
        return None
    data = read_json(cameras_path)
    frames = data.get("frames", [])
    signature = []
    for frame in frames:
        pose = np.asarray(frame.get("cam2world"), dtype=np.float64)
        signature.append((int(frame.get("frame", len(signature))), tuple(np.round(pose.reshape(-1), 7).tolist())))
    return signature


def inspect_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    min_probe_mask_area: float = 0.0,
    min_probe_visible_ratio: float = 0.0,
) -> Dict[str, Any]:
    samples = load_samples(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    issues: List[Dict[str, Any]] = []
    variant_counts: Dict[str, int] = {v: 0 for v in EXPECTED_VARIANTS}
    mask_areas: List[float] = []
    visible_ratios: List[float] = []
    present_mask_areas: List[float] = []
    present_visible_ratios: List[float] = []

    for sample in samples:
        variant_counts[sample.variant_id] = variant_counts.get(sample.variant_id, 0) + 1
        expected_pos = VARIANT_TO_POSITION.get(sample.variant_id)
        if expected_pos is None:
            issues.append({"sample_id": sample.sample_id, "issue": "unknown_variant", "variant_id": sample.variant_id})
        elif sample.target_position_id != expected_pos:
            issues.append(
                {
                    "sample_id": sample.sample_id,
                    "issue": "target_position_id_mismatch",
                    "expected": expected_pos,
                    "actual": sample.target_position_id,
                }
            )
        expected_present = 0 if sample.variant_id == "absent" else 1
        if sample.target_present != expected_present:
            issues.append(
                {
                    "sample_id": sample.sample_id,
                    "issue": "target_present_mismatch",
                    "expected": expected_present,
                    "actual": sample.target_present,
                }
            )
        if not sample.probe_rgb_path.exists():
            issues.append({"sample_id": sample.sample_id, "issue": "missing_probe_rgb", "path": str(sample.probe_rgb_path)})
        if not sample.cameras_path.exists():
            issues.append({"sample_id": sample.sample_id, "issue": "missing_cameras_json", "path": str(sample.cameras_path)})
        else:
            cameras = read_json(sample.cameras_path)
            if not cameras.get("intrinsic"):
                issues.append({"sample_id": sample.sample_id, "issue": "missing_camera_intrinsics"})
            frames = cameras.get("frames", [])
            if len(frames) <= sample.probe_frame:
                issues.append(
                    {
                        "sample_id": sample.sample_id,
                        "issue": "camera_pose_missing_probe_frame",
                        "num_camera_frames": len(frames),
                        "probe_frame": sample.probe_frame,
                    }
                )

        rgb_count = len(list_images(sample.rgb_dir))
        if rgb_count <= sample.probe_frame:
            issues.append(
                {
                    "sample_id": sample.sample_id,
                    "issue": "rgb_sequence_too_short",
                    "rgb_count": rgb_count,
                    "probe_frame": sample.probe_frame,
                }
            )

        area = sample.frame15_target_mask_area
        ratio = sample.frame15_target_visible_ratio
        if area is not None:
            mask_areas.append(float(area))
        if ratio is not None:
            visible_ratios.append(float(ratio))
        if sample.target_present:
            if area is None:
                issues.append({"sample_id": sample.sample_id, "issue": "present_missing_mask_area"})
            else:
                present_mask_areas.append(float(area))
                if area < min_probe_mask_area:
                    issues.append(
                        {
                            "sample_id": sample.sample_id,
                            "issue": "present_mask_area_below_threshold",
                            "mask_area": area,
                            "threshold": min_probe_mask_area,
                        }
                    )
            if ratio is None:
                issues.append({"sample_id": sample.sample_id, "issue": "present_missing_visible_ratio"})
            else:
                present_visible_ratios.append(float(ratio))
                if ratio < min_probe_visible_ratio:
                    issues.append(
                        {
                            "sample_id": sample.sample_id,
                            "issue": "present_visible_ratio_below_threshold",
                            "visible_ratio": ratio,
                            "threshold": min_probe_visible_ratio,
                        }
                    )
            if not sample.probe_target_mask_path.exists():
                issues.append(
                    {"sample_id": sample.sample_id, "issue": "present_missing_target_mask", "path": str(sample.probe_target_mask_path)}
                )
        else:
            if sample.target_position_id != 0:
                issues.append({"sample_id": sample.sample_id, "issue": "absent_position_not_zero"})

    groups = group_by_base_scene(samples)
    base_scene_issues: List[Dict[str, Any]] = []
    for base_scene_id, group in groups.items():
        variants = sorted(sample.variant_id for sample in group)
        if variants != sorted(EXPECTED_VARIANTS):
            base_scene_issues.append(
                {
                    "base_scene_id": base_scene_id,
                    "issue": "variant_set_mismatch",
                    "expected": list(EXPECTED_VARIANTS),
                    "actual": variants,
                }
            )

        signatures = {sample.variant_id: _pose_signature(sample) for sample in group}
        non_null = [sig for sig in signatures.values() if sig is not None]
        if non_null and any(sig != non_null[0] for sig in non_null):
            base_scene_issues.append({"base_scene_id": base_scene_id, "issue": "camera_trajectory_differs_across_variants"})

    issues.extend(base_scene_issues)

    report = {
        "data_root": str(Path(data_root).resolve()),
        "num_samples": len(samples),
        "num_base_scenes": len(groups),
        "expected_num_samples": 150,
        "expected_num_base_scenes": 30,
        "variant_counts": variant_counts,
        "complete_base_scenes": [
            base_scene_id
            for base_scene_id, group in groups.items()
            if sorted(sample.variant_id for sample in group) == sorted(EXPECTED_VARIANTS)
        ],
        "issues": issues,
        "num_issues": len(issues),
        "mask_area_stats_all": finite_stats(mask_areas),
        "visible_ratio_stats_all": finite_stats(visible_ratios),
        "mask_area_stats_present": finite_stats(present_mask_areas),
        "visible_ratio_stats_present": finite_stats(present_visible_ratios),
        "min_probe_mask_area": min_probe_mask_area,
        "min_probe_visible_ratio": min_probe_visible_ratio,
    }

    write_json(output_dir / "dataset_report.json", report)
    write_jsonl(output_dir / "validated_manifest.jsonl", [sample.to_manifest_row() for sample in samples])
    return report
