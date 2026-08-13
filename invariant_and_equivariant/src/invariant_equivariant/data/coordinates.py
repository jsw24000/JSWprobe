from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def position_from_frame(frame: Mapping[str, Any], coordinate_system: str) -> np.ndarray:
    if coordinate_system == "ref":
        return np.asarray(frame["object_center_ref_camera"], dtype=np.float64)
    if coordinate_system == "current":
        return np.asarray(frame["object_center_current_camera"], dtype=np.float64)
    if coordinate_system == "world":
        return np.asarray(frame["object_center_world"], dtype=np.float64)
    raise ValueError(f"Unknown coordinate system: {coordinate_system}")


def delta_from_pair(pair: Mapping[str, Any], coordinate_system: str) -> np.ndarray:
    if coordinate_system == "ref":
        return np.asarray(pair["delta_ref_camera"], dtype=np.float64)
    if coordinate_system == "current":
        return np.asarray(pair["delta_current_camera"], dtype=np.float64)
    if coordinate_system == "world":
        return np.asarray(pair["delta_world"], dtype=np.float64)
    raise ValueError(f"Unknown coordinate system: {coordinate_system}")

