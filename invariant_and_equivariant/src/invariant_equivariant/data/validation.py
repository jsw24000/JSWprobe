from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from PIL import Image


def image_exists_and_size(path: str) -> tuple[bool, tuple[int, int] | None]:
    p = Path(path)
    if not p.exists():
        return False, None
    with Image.open(p) as image:
        return True, image.size


def validate_frame_files(frames: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    invalid: List[Dict[str, Any]] = []
    for frame in frames:
        rgb_ok, rgb_size = image_exists_and_size(str(frame["rgb"]))
        mask_ok, mask_size = image_exists_and_size(str(frame["target_mask"]))
        if not rgb_ok or not mask_ok or rgb_size != mask_size:
            invalid.append(
                {
                    "scene_id": frame["scene_id"],
                    "state_id": frame["state_id"],
                    "camera_id": frame["camera_id"],
                    "reason": "missing_or_mismatched_rgb_mask",
                    "rgb_exists": rgb_ok,
                    "mask_exists": mask_ok,
                    "rgb_size": rgb_size,
                    "mask_size": mask_size,
                }
            )
    return invalid

