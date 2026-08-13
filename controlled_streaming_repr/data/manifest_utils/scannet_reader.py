"""Reader for raw ScanNet scene folders used by controlled sequence settings.

The reader scans a ScanNet ``sceneXXXX_XX`` directory, matches RGB/depth/pose
files by numeric frame id, validates poses when requested, and returns sorted
frame records. It never copies RGB/depth data; downstream sequence settings use
the returned paths to write lightweight JSON manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .pose_utils import is_valid_pose


COLOR_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEPTH_EXTENSIONS = (".png", ".tiff", ".tif")


@dataclass(frozen=True)
class ScanNetFrame:
    """A single valid raw ScanNet frame with paths to all required assets."""

    scene_id: str
    source_frame_id: str
    original_order_index: int
    rgb_path: Path
    depth_path: Path | None
    pose_path: Path | None
    intrinsic_path: Path | None


@dataclass(frozen=True)
class ScanNetScene:
    """A scanned ScanNet scene and its numerically sorted valid frames."""

    scene_id: str
    scene_dir: Path
    frames: list[ScanNetFrame]


def numeric_frame_id(path: Path) -> int | None:
    """Parse the numeric frame id from a ScanNet file stem."""

    stem = path.stem
    if stem.isdigit():
        return int(stem)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def frame_id_string(index: int, width: int = 6) -> str:
    """Format a numeric frame id using ScanNet's common zero-padded style."""

    return f"{index:0{width}d}"


def _collect_by_id(directory: Path, extensions: Iterable[str]) -> dict[int, Path]:
    if not directory.is_dir():
        return {}
    allowed = {ext.lower() for ext in extensions}
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        frame_id = numeric_frame_id(path)
        if frame_id is None:
            continue
        result[frame_id] = path.resolve()
    return result


def _find_intrinsic(scene_dir: Path) -> Path | None:
    intrinsic_dir = scene_dir / "intrinsic"
    candidates = [
        intrinsic_dir / "intrinsic_color.txt",
        intrinsic_dir / "intrinsic_depth.txt",
        intrinsic_dir / "intrinsic.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if intrinsic_dir.is_dir():
        text_files = sorted(intrinsic_dir.glob("*.txt"))
        if text_files:
            return text_files[0].resolve()
    return None


class ScanNetReader:
    """Scan raw ScanNet scenes and expose valid frame metadata."""

    def __init__(
        self,
        scannet_root: str | Path,
        *,
        require_color: bool = True,
        require_depth: bool = True,
        require_pose: bool = True,
        require_intrinsic: bool = True,
        require_valid_pose: bool = True,
    ) -> None:
        self.scannet_root = Path(scannet_root).expanduser().resolve()
        self.require_color = require_color
        self.require_depth = require_depth
        self.require_pose = require_pose
        self.require_intrinsic = require_intrinsic
        self.require_valid_pose = require_valid_pose

    def discover_scene_ids(self, max_scenes: int = 10) -> list[str]:
        """Return scene ids found under ``scannet_root`` in sorted order."""

        if not self.scannet_root.is_dir():
            return []
        scene_dirs = [
            path.name
            for path in self.scannet_root.iterdir()
            if path.is_dir() and path.name.startswith("scene")
        ]
        return sorted(scene_dirs)[:max_scenes]

    def scan_scene(self, scene_id: str) -> ScanNetScene:
        """Read and sort valid frames from a single ScanNet scene."""

        scene_dir = (self.scannet_root / scene_id).resolve()
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"ScanNet scene directory not found: {scene_dir}")

        color_by_id = _collect_by_id(scene_dir / "color", COLOR_EXTENSIONS)
        depth_by_id = _collect_by_id(scene_dir / "depth", DEPTH_EXTENSIONS)
        pose_by_id = _collect_by_id(scene_dir / "pose", (".txt",))
        intrinsic_path = _find_intrinsic(scene_dir)

        if self.require_intrinsic and intrinsic_path is None:
            raise FileNotFoundError(f"No intrinsic text file found in: {scene_dir / 'intrinsic'}")

        candidate_ids = set(color_by_id)
        if not self.require_color:
            candidate_ids |= set(depth_by_id) | set(pose_by_id)

        frames: list[ScanNetFrame] = []
        width = max((len(str(path.stem)) for path in color_by_id.values()), default=6)
        for frame_index in sorted(candidate_ids):
            rgb_path = color_by_id.get(frame_index)
            depth_path = depth_by_id.get(frame_index)
            pose_path = pose_by_id.get(frame_index)
            if self.require_color and rgb_path is None:
                continue
            if self.require_depth and depth_path is None:
                continue
            if self.require_pose and pose_path is None:
                continue
            if self.require_valid_pose and pose_path is not None and not is_valid_pose(pose_path):
                continue

            frame_id = frame_id_string(frame_index, width=width)
            frames.append(
                ScanNetFrame(
                    scene_id=scene_id,
                    source_frame_id=frame_id,
                    original_order_index=frame_index,
                    rgb_path=rgb_path or Path(),
                    depth_path=depth_path,
                    pose_path=pose_path,
                    intrinsic_path=intrinsic_path,
                )
            )

        return ScanNetScene(scene_id=scene_id, scene_dir=scene_dir, frames=frames)

    def scan_scenes(self, scene_ids: list[str] | None = None, max_scenes: int = 10) -> list[ScanNetScene]:
        """Scan up to ``max_scenes`` selected or discovered scenes."""

        ids = list(scene_ids or self.discover_scene_ids(max_scenes=max_scenes))
        return [self.scan_scene(scene_id) for scene_id in ids[:max_scenes]]
