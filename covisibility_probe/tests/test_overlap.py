from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covisibility_probe.overlap import build_overlap_matrices, intersection_count, pair_overlap_from_voxels
from covisibility_probe.scannet_io import load_pose_c2w, valid_pose_matrix


def test_same_frame_overlap_is_one() -> None:
    vox = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.int64)
    result = build_overlap_matrices([0, 1], [vox, vox.copy()])
    assert np.allclose(result.overlap_cos[0, 1], 1.0)
    assert np.allclose(result.overlap_iou[0, 1], 1.0)


def test_overlap_symmetric_and_range() -> None:
    a = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    b = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.int64)
    result = build_overlap_matrices([0, 1], [a, b])
    assert np.allclose(result.overlap_cos, result.overlap_cos.T)
    assert result.overlap_cos.min() >= 0
    assert result.overlap_cos.max() <= 1


def test_empty_depth_safe_overlap() -> None:
    a = np.empty((0, 3), dtype=np.int64)
    b = np.array([[1, 2, 3]], dtype=np.int64)
    oc, oi, ca, cb, inter = pair_overlap_from_voxels(a, b)
    assert (oc, oi, ca, cb, inter) == (0.0, 0.0, 0.0, 0.0, 0)


def test_intersection_count() -> None:
    a = np.array([[0, 0, 0], [1, 2, 3], [9, 9, 9]], dtype=np.int64)
    b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    assert intersection_count(a, b) == 1


def test_pose_direction_conversion(tmp_path: Path) -> None:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = np.linalg.inv(c2w)
    pose_path = tmp_path / "pose.txt"
    np.savetxt(pose_path, w2c)
    loaded, valid = load_pose_c2w(pose_path, convention="world_to_camera")
    assert valid
    assert valid_pose_matrix(loaded)
    assert np.allclose(loaded, c2w)

