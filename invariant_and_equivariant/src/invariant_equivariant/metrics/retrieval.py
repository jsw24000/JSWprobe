from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


def nearest_state_metrics(
    predicted: np.ndarray,
    true_key: Tuple[str, str, str],
    candidates: Sequence[Mapping[str, Any]],
    candidate_features: np.ndarray,
) -> Dict[str, float]:
    if len(candidates) == 0:
        return {"top1": 0.0, "top3": 0.0, "grid_distance": float("nan")}
    dists = np.linalg.norm(candidate_features - predicted[None, :], axis=1)
    order = np.argsort(dists)
    keys = [(row["scene_id"], row["state_id"], row["camera_id"]) for row in candidates]
    top1 = keys[int(order[0])] == true_key
    top3 = true_key in [keys[int(i)] for i in order[: min(3, len(order))]]
    true_row = next((row for row in candidates if (row["scene_id"], row["state_id"], row["camera_id"]) == true_key), None)
    pred_row = candidates[int(order[0])]
    grid_distance = float("nan")
    if true_row is not None:
        if "state_position_index" in true_row and "state_position_index" in pred_row:
            a = np.asarray(true_row["state_position_index"], dtype=np.float64)
            b = np.asarray(pred_row["state_position_index"], dtype=np.float64)
            grid_distance = float(np.linalg.norm(a - b))
        elif all(k in true_row and k in pred_row for k in ["grid_row", "grid_col"]):
            a = np.asarray([float(true_row["grid_row"]), float(true_row["grid_col"])], dtype=np.float64)
            b = np.asarray([float(pred_row["grid_row"]), float(pred_row["grid_col"])], dtype=np.float64)
            grid_distance = float(np.linalg.norm(a - b))
    return {"top1": float(top1), "top3": float(top3), "grid_distance": grid_distance}
