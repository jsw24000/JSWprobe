"""ScanNet scene IO helpers.

The code assumes the common ScanNet export layout:
scene/color, scene/depth, scene/pose, scene/intrinsic. Poses are interpreted as
camera-to-world transforms. Color and depth may have different resolutions; the
geometry module maps resized RGB token coordinates through color and depth image
coordinates using the recorded image sizes and depth intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class FrameRecord:
    index: int
    frame_id: int
    color_path: Path
    depth_path: Path
    pose_path: Path
    valid_pose: bool
    valid_depth: bool
    color_size: tuple[int, int]
    depth_size: tuple[int, int]
    pose_c2w: np.ndarray


@dataclass
class SceneInfo:
    scene_id: str
    scene_dir: Path
    intrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_color: np.ndarray
    extrinsic_depth: np.ndarray
    alignment_note: str


def read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path).astype(np.float64)


def load_scene(scannet_root: str | Path, scene_id: str) -> SceneInfo:
    scene_dir = Path(scannet_root) / scene_id
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"ScanNet scene not found: {scene_dir}")
    intrinsic_dir = scene_dir / "intrinsic"
    k_color = read_matrix(intrinsic_dir / "intrinsic_color.txt")[:3, :3]
    k_depth = read_matrix(intrinsic_dir / "intrinsic_depth.txt")[:3, :3]
    ext_color = read_matrix(intrinsic_dir / "extrinsic_color.txt")
    ext_depth = read_matrix(intrinsic_dir / "extrinsic_depth.txt")
    note = (
        "Using exported color/depth frames with identity color/depth extrinsics when present; "
        "token RGB coordinates are mapped to depth coordinates by image-size scaling and depth intrinsics."
    )
    return SceneInfo(scene_id, scene_dir, k_color, k_depth, ext_color, ext_depth, note)


def _frame_file(scene: SceneInfo, subdir: str, frame_id: int, exts: tuple[str, ...]) -> Path | None:
    for ext in exts:
        p = scene.scene_dir / subdir / f"{frame_id}{ext}"
        if p.is_file():
            return p
    return None


def load_pose(path: Path) -> tuple[np.ndarray, bool]:
    if not path.is_file():
        return np.eye(4, dtype=np.float64), False
    pose = read_matrix(path)
    valid = pose.shape == (4, 4) and np.isfinite(pose).all() and abs(np.linalg.det(pose[:3, :3])) > 1e-6
    return pose, bool(valid)


def depth_has_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    arr = np.asarray(Image.open(path))
    return bool(np.any(arr > 0))


def make_frame_record(scene: SceneInfo, frame_id: int, index: int) -> FrameRecord | None:
    color_path = _frame_file(scene, "color", frame_id, (".jpg", ".png"))
    depth_path = _frame_file(scene, "depth", frame_id, (".png", ".jpg"))
    pose_path = scene.scene_dir / "pose" / f"{frame_id}.txt"
    if color_path is None or depth_path is None:
        return None
    pose, valid_pose = load_pose(pose_path)
    valid_depth = depth_has_valid(depth_path)
    color_size = Image.open(color_path).size
    depth_size = Image.open(depth_path).size
    return FrameRecord(
        index=index,
        frame_id=frame_id,
        color_path=color_path,
        depth_path=depth_path,
        pose_path=pose_path,
        valid_pose=valid_pose,
        valid_depth=valid_depth,
        color_size=color_size,
        depth_size=depth_size,
        pose_c2w=pose,
    )


def load_rgb(path: str | Path, size_hw: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size_hw is not None:
        h, w = size_hw
        img = img.resize((w, h), Image.Resampling.BICUBIC)
    return img


def load_rgb_array(path: str | Path, size_hw: tuple[int, int]) -> np.ndarray:
    return np.asarray(load_rgb(path, size_hw=size_hw)).astype(np.uint8)


def load_depth_m(path: str | Path, depth_scale: float) -> np.ndarray:
    depth = np.asarray(Image.open(path)).astype(np.float32)
    return depth / float(depth_scale)


def frames_to_dataframe(frames: list[FrameRecord]) -> pd.DataFrame:
    rows = []
    for fr in frames:
        rows.append(
            {
                "index": fr.index,
                "frame_id": fr.frame_id,
                "color_path": str(fr.color_path),
                "depth_path": str(fr.depth_path),
                "pose_path": str(fr.pose_path),
                "valid_pose": fr.valid_pose,
                "valid_depth": fr.valid_depth,
            }
        )
    return pd.DataFrame(rows)

