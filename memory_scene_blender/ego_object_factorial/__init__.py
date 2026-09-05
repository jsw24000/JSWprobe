"""Controlled temporal ego/object factorial data-generation helpers."""

from .protocol import (
    MOTION_AXIS_WORLD,
    build_matched_relative_groups,
    camera_from_source_extrinsics,
    condition_key,
    linear_alphas,
    motion_conditions,
    project_opencv,
    sequence_id,
    translated_camera_payload,
)

__all__ = [
    "MOTION_AXIS_WORLD",
    "build_matched_relative_groups",
    "camera_from_source_extrinsics",
    "condition_key",
    "linear_alphas",
    "motion_conditions",
    "project_opencv",
    "sequence_id",
    "translated_camera_payload",
]
