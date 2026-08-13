from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class SceneData:
    root: Path
    image_paths: list[Path]
    poses_c2w: np.ndarray
    intrinsics: Intrinsics
    frame_sync: list[dict[str, Any]]

    @property
    def num_frames(self) -> int:
        return len(self.image_paths)

    def timestamp(self, frame_id: int) -> float | None:
        if not self.frame_sync:
            return None
        value = self.frame_sync[frame_id].get("image_timestamp")
        return None if value in ("", None) else float(value)

    def center(self, frame_id: int) -> np.ndarray:
        return self.poses_c2w[frame_id, :3, 3]

    def image_path(self, frame_id: int) -> Path:
        return self.image_paths[frame_id]


def load_image_paths(scene_root: str | Path) -> list[Path]:
    image_dir = Path(scene_root) / "images"
    return sorted(image_dir.glob("*.png"))


def load_intrinsics(scene_root: str | Path) -> Intrinsics:
    path = Path(scene_root) / "intrinsics.txt"
    vals = path.read_text(encoding="utf-8").strip().split()
    if len(vals) != 6:
        raise ValueError(f"Expected 6 intrinsics values in {path}, found {len(vals)}")
    fx, fy, cx, cy = map(float, vals[:4])
    width, height = map(int, vals[4:])
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def load_poses(scene_root: str | Path) -> np.ndarray:
    path = Path(scene_root) / "poses_c2w.txt"
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 16:
        raise ValueError(f"Expected 16 values per pose row in {path}, found {raw.shape[1]}")
    return raw.reshape(-1, 4, 4)


def _cast_csv_value(value: str) -> Any:
    if value == "":
        return value
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def read_csv_dicts(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: _cast_csv_value(v) for k, v in row.items()} for row in reader]


def load_frame_sync(scene_root: str | Path) -> list[dict[str, Any]]:
    path = Path(scene_root) / "timestamp_sync" / "frame_sync.csv"
    if not path.exists():
        return []
    return read_csv_dicts(path)


def load_scene(scene_root: str | Path) -> SceneData:
    scene_root = Path(scene_root).expanduser().resolve()
    return SceneData(
        root=scene_root,
        image_paths=load_image_paths(scene_root),
        poses_c2w=load_poses(scene_root),
        intrinsics=load_intrinsics(scene_root),
        frame_sync=load_frame_sync(scene_root),
    )


def first_image_size(image_paths: list[Path]) -> tuple[int, int] | None:
    if not image_paths:
        return None
    with Image.open(image_paths[0]) as im:
        return im.size


def frame_ids(start: int, end: int, stride: int = 1) -> list[int]:
    return list(range(int(start), int(end), int(stride)))
