from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


COLOR_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEPTH_EXTENSIONS = (".png", ".tiff", ".tif")


@dataclass(frozen=True)
class FrameRecord:
    scene_id: str
    input_position: int
    frame_id: int
    color_path: Path
    depth_path: Path
    pose_path: Path
    color_size: tuple[int, int]
    depth_size: tuple[int, int]
    pose_c2w: np.ndarray
    valid_pose: bool
    valid_depth: bool


@dataclass(frozen=True)
class SceneInfo:
    scene_id: str
    scene_dir: Path
    intrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_color: np.ndarray | None
    extrinsic_depth: np.ndarray | None
    frames: list[FrameRecord]
    alignment_note: str


def numeric_frame_id(path: Path) -> int | None:
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def _collect_by_id(directory: Path, extensions: Iterable[str]) -> dict[int, Path]:
    if not directory.is_dir():
        return {}
    allowed = {e.lower() for e in extensions}
    out: dict[int, Path] = {}
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in allowed:
            idx = numeric_frame_id(p)
            if idx is not None:
                out[idx] = p.resolve()
    return out


def read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path).astype(np.float64)


def _read_optional_matrix(path: Path) -> np.ndarray | None:
    return read_matrix(path) if path.is_file() else None


def valid_pose_matrix(pose: np.ndarray) -> bool:
    return pose.shape == (4, 4) and np.isfinite(pose).all() and abs(np.linalg.det(pose[:3, :3])) > 1e-8


def load_pose_c2w(path: Path, convention: str = "camera_to_world") -> tuple[np.ndarray, bool]:
    if not path.is_file():
        return np.eye(4, dtype=np.float64), False
    pose = read_matrix(path)
    valid = valid_pose_matrix(pose)
    if not valid:
        return np.eye(4, dtype=np.float64), False
    if convention == "camera_to_world":
        return pose, True
    if convention == "world_to_camera":
        return np.linalg.inv(pose), True
    raise ValueError(f"Unknown pose convention: {convention}")


def load_depth_m(path: str | Path, depth_scale: float = 1000.0) -> np.ndarray:
    arr = np.asarray(Image.open(path)).astype(np.float32)
    return arr / float(depth_scale)


def depth_has_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    arr = np.asarray(Image.open(path))
    return bool(np.any(arr > 0))


def discover_scene_ids(scannet_root: str | Path) -> list[str]:
    root = Path(scannet_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("scene"))


def load_scene(
    scannet_root: str | Path,
    scene_id: str,
    *,
    pose_convention: str = "camera_to_world",
    check_depth_nonzero: bool = False,
) -> SceneInfo:
    scene_dir = Path(scannet_root) / scene_id
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"ScanNet scene not found: {scene_dir}")

    intrinsic_dir = scene_dir / "intrinsic"
    k_color_path = intrinsic_dir / "intrinsic_color.txt"
    k_depth_path = intrinsic_dir / "intrinsic_depth.txt"
    if not k_color_path.is_file() or not k_depth_path.is_file():
        raise FileNotFoundError(f"Missing ScanNet intrinsics under: {intrinsic_dir}")

    k_color = read_matrix(k_color_path)[:3, :3]
    k_depth = read_matrix(k_depth_path)[:3, :3]
    ext_color = _read_optional_matrix(intrinsic_dir / "extrinsic_color.txt")
    ext_depth = _read_optional_matrix(intrinsic_dir / "extrinsic_depth.txt")

    color_by_id = _collect_by_id(scene_dir / "color", COLOR_EXTENSIONS)
    depth_by_id = _collect_by_id(scene_dir / "depth", DEPTH_EXTENSIONS)
    pose_by_id = _collect_by_id(scene_dir / "pose", (".txt",))

    frames: list[FrameRecord] = []
    for pos, frame_id in enumerate(sorted(set(color_by_id) & set(depth_by_id) & set(pose_by_id))):
        color_path = color_by_id[frame_id]
        depth_path = depth_by_id[frame_id]
        pose_path = pose_by_id[frame_id]
        pose, valid_pose = load_pose_c2w(pose_path, convention=pose_convention)
        try:
            color_size = Image.open(color_path).size
            depth_size = Image.open(depth_path).size
        except Exception:
            continue
        valid_depth = depth_has_valid(depth_path) if check_depth_nonzero else depth_path.is_file()
        frames.append(
            FrameRecord(
                scene_id=scene_id,
                input_position=pos,
                frame_id=int(frame_id),
                color_path=color_path,
                depth_path=depth_path,
                pose_path=pose_path,
                color_size=color_size,
                depth_size=depth_size,
                pose_c2w=pose,
                valid_pose=valid_pose,
                valid_depth=valid_depth,
            )
        )

    note = (
        "ScanNet pose files are interpreted as camera-to-world. "
        "Visible-surface overlap is computed at depth resolution with intrinsic_depth. "
        "RGB is used for model input only."
    )
    return SceneInfo(scene_id, scene_dir, k_color, k_depth, ext_color, ext_depth, frames, note)


def scene_health(scene: SceneInfo) -> dict[str, object]:
    valid = [f for f in scene.frames if f.valid_pose and f.valid_depth]
    color_sizes = sorted({f.color_size for f in valid})
    depth_sizes = sorted({f.depth_size for f in valid})
    return {
        "scene_id": scene.scene_id,
        "scene_dir": str(scene.scene_dir),
        "num_frames_total": len(scene.frames),
        "num_valid_frames": len(valid),
        "has_rgb": bool(scene.frames),
        "has_depth": any(f.depth_path.is_file() for f in scene.frames),
        "has_pose": any(f.pose_path.is_file() for f in scene.frames),
        "has_intrinsic_color": scene.intrinsic_color.shape == (3, 3),
        "has_intrinsic_depth": scene.intrinsic_depth.shape == (3, 3),
        "color_sizes": [list(x) for x in color_sizes[:4]],
        "depth_sizes": [list(x) for x in depth_sizes[:4]],
        "first_frame_id": valid[0].frame_id if valid else None,
        "last_frame_id": valid[-1].frame_id if valid else None,
    }


def frame_table_rows(frames: list[FrameRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pos, fr in enumerate(frames):
        rows.append(
            {
                "scene_id": fr.scene_id,
                "input_position": pos,
                "frame_id": fr.frame_id,
                "color_path": str(fr.color_path),
                "depth_path": str(fr.depth_path),
                "pose_path": str(fr.pose_path),
                "valid_pose": bool(fr.valid_pose),
                "valid_depth": bool(fr.valid_depth),
                "color_width": int(fr.color_size[0]),
                "color_height": int(fr.color_size[1]),
                "depth_width": int(fr.depth_size[0]),
                "depth_height": int(fr.depth_size[1]),
            }
        )
    return rows


def rotation_angle_deg(c2w_a: np.ndarray, c2w_b: np.ndarray) -> float:
    r_rel = c2w_b[:3, :3].T @ c2w_a[:3, :3]
    trace = float(np.trace(r_rel))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def translation_distance(c2w_a: np.ndarray, c2w_b: np.ndarray) -> float:
    return float(np.linalg.norm(c2w_b[:3, 3] - c2w_a[:3, 3]))
