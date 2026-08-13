from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from long_seq_memory.trajectory_metrics import loop_pair_metrics


@dataclass(frozen=True)
class PoseCandidate:
    frame_i: int
    frame_j: int
    center_distance_m: float
    relative_rotation_deg: float
    view_direction_cosine: float
    pose_candidate_score: float


def pose_candidate_score(poses_c2w: np.ndarray, i: int, j: int) -> PoseCandidate:
    metrics = loop_pair_metrics(poses_c2w, i, j)
    dist = metrics["center_distance_m"]
    view = metrics["view_direction_cosine"]
    rot = metrics["relative_rotation_deg"]
    distance_term = float(np.exp(-dist / 2.0))
    view_term = float(max(0.0, (view + 1.0) / 2.0))
    rot_term = float(np.exp(-rot / 90.0))
    return PoseCandidate(
        frame_i=i,
        frame_j=j,
        center_distance_m=dist,
        relative_rotation_deg=rot,
        view_direction_cosine=view,
        pose_candidate_score=distance_term * view_term * rot_term,
    )


def tls_frame_overlap(*_args, **_kwargs):
    raise NotImplementedError(
        "TLS/LiDAR z-buffer overlap is not implemented yet. Use pose_candidate_score "
        "as pose_candidate, not geometric_overlap."
    )
