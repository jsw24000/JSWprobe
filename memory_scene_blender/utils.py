"""Shared pure-Python helpers for the memory-scene experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .config import CAMERA, FRAME, RENDER, CameraConfig, FrameConfig


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def np_to_list(array: np.ndarray) -> List[Any]:
    return array.astype(float).tolist()


def look_at_cam2world(location: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """Build a Blender camera-to-world matrix that points -Z toward target."""
    loc = np.asarray(location, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    forward = tgt - loc
    forward /= np.linalg.norm(forward)

    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up_guess)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward
    mat[:3, 3] = loc
    return mat


def loop_angle_for_index(
    loop_index: int,
    frame_cfg: Optional[FrameConfig] = None,
    camera_cfg: Optional[CameraConfig] = None,
) -> float:
    frame_cfg = frame_cfg or FRAME
    camera_cfg = camera_cfg or CAMERA
    if camera_cfg.path_mode == "piecewise_angles":
        if not camera_cfg.loop_waypoint_angles_rad:
            raise ValueError("piecewise_angles camera path requires loop_waypoint_angles_rad")

        waypoints = sorted(camera_cfg.loop_waypoint_angles_rad, key=lambda item: item[0])
        if loop_index <= waypoints[0][0]:
            return float(waypoints[0][1])
        if loop_index >= waypoints[-1][0]:
            return float(waypoints[-1][1])

        for (left_index, left_angle), (right_index, right_angle) in zip(waypoints, waypoints[1:]):
            if left_index <= loop_index <= right_index:
                span = right_index - left_index
                if span <= 0:
                    raise ValueError(f"Invalid camera waypoint span: {left_index} -> {right_index}")
                t = (loop_index - left_index) / span
                return float(left_angle + (right_angle - left_angle) * t)

        raise ValueError(f"Loop index {loop_index} is outside camera waypoint range")
    if camera_cfg.path_mode != "orbit":
        raise ValueError(f"Unsupported camera path mode: {camera_cfg.path_mode!r}")
    return camera_cfg.loop_start_angle_rad + 2.0 * math.pi * (loop_index / frame_cfg.loop_frames)


def camera_location_from_angle(angle: float, camera_cfg: Optional[CameraConfig] = None) -> np.ndarray:
    camera_cfg = camera_cfg or CAMERA
    return np.array(
        [
            camera_cfg.radius_x * math.cos(angle),
            camera_cfg.radius_y * math.sin(angle),
            camera_cfg.height_m,
        ],
        dtype=np.float64,
    )


def generate_camera_poses(
    frame_cfg: Optional[FrameConfig] = None,
    camera_cfg: Optional[CameraConfig] = None,
) -> List[np.ndarray]:
    """Generate 8 anchor poses followed by two identical 44-frame loops."""
    frame_cfg = frame_cfg or FRAME
    camera_cfg = camera_cfg or CAMERA
    poses: List[np.ndarray] = []

    for frame in frame_cfg.anchor_frames:
        t = frame / max(1, frame_cfg.anchor_count - 1)
        angle = camera_cfg.loop_start_angle_rad + camera_cfg.anchor_angle_offset_rad * (1.0 - t)
        poses.append(look_at_cam2world(camera_location_from_angle(angle, camera_cfg), camera_cfg.look_at))

    first_loop: List[np.ndarray] = []
    for loop_index in range(frame_cfg.loop_frames):
        angle = loop_angle_for_index(loop_index, frame_cfg, camera_cfg)
        first_loop.append(look_at_cam2world(camera_location_from_angle(angle, camera_cfg), camera_cfg.look_at))

    poses.extend(first_loop)
    poses.extend([pose.copy() for pose in first_loop])

    if len(poses) != frame_cfg.total_frames:
        raise ValueError(f"Expected {frame_cfg.total_frames} camera poses, got {len(poses)}")
    return poses


def camera_loop_metadata(
    frame_cfg: Optional[FrameConfig] = None,
    camera_cfg: Optional[CameraConfig] = None,
) -> List[Dict[str, Any]]:
    frame_cfg = frame_cfg or FRAME
    camera_cfg = camera_cfg or CAMERA
    rows: List[Dict[str, Any]] = []
    for frame in range(frame_cfg.total_frames):
        if frame < frame_cfg.anchor_count:
            segment = "anchor"
            loop_index = None
            paired_frame = None
            angle = camera_cfg.loop_start_angle_rad + camera_cfg.anchor_angle_offset_rad * (
                1.0 - frame / max(1, frame_cfg.anchor_count - 1)
            )
        elif frame <= frame_cfg.first_loop_end:
            segment = "first_loop"
            loop_index = frame - frame_cfg.first_loop_start
            paired_frame = frame + frame_cfg.loop_frames
            angle = loop_angle_for_index(loop_index, frame_cfg, camera_cfg)
        else:
            segment = "second_loop"
            loop_index = frame - frame_cfg.second_loop_start
            paired_frame = frame - frame_cfg.loop_frames
            angle = loop_angle_for_index(loop_index, frame_cfg, camera_cfg)
        rows.append(
            {
                "frame": frame,
                "segment": segment,
                "loop_index": loop_index,
                "paired_frame": paired_frame,
                "angle_rad": angle,
            }
        )
    return rows


def camera_intrinsics_matrix(width: int, height: int, lens_mm: float, sensor_width_mm: float = 36.0) -> np.ndarray:
    """Approximate Blender perspective intrinsics for the default horizontal sensor fit."""
    fx = lens_mm / sensor_width_mm * width
    fy = fx
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def visibility_pixel_threshold(width: int, height: int) -> int:
    area_threshold = int(round(width * height * RENDER.visibility_area_ratio))
    return max(RENDER.visibility_pixel_threshold, area_threshold)


def visible_frames_from_counts(pixel_counts: Mapping[int, int], threshold: int) -> List[int]:
    return [int(frame) for frame, count in sorted(pixel_counts.items()) if int(count) >= threshold]


def compute_memory_gap(first_visible: Sequence[int], second_revisit_visible: Sequence[int]) -> Dict[str, Any]:
    if not first_visible or not second_revisit_visible:
        return {
            "last_first_loop_visible_frame": None,
            "first_second_loop_revisit_frame": None,
            "gap_frames": None,
            "gap_ok": False,
        }

    last_first = max(frame for frame in first_visible if FRAME.first_loop_start <= frame <= FRAME.first_loop_end)
    first_second = min(frame for frame in second_revisit_visible if FRAME.second_loop_start <= frame <= FRAME.second_loop_end)
    gap = first_second - last_first
    return {
        "last_first_loop_visible_frame": int(last_first),
        "first_second_loop_revisit_frame": int(first_second),
        "gap_frames": int(gap),
        "gap_ok": bool(gap > FRAME.local_window),
    }


def assert_paired_camera_poses(poses: Sequence[np.ndarray], atol: float = 1e-8) -> None:
    for frame in FRAME.first_loop_frames:
        paired = frame + FRAME.loop_frames
        if not np.allclose(poses[frame], poses[paired], atol=atol):
            raise AssertionError(f"Camera pose mismatch between frame {frame} and paired frame {paired}")


def summarize_frame_list(frames: Sequence[int], max_items: int = 24) -> str:
    frames = list(frames)
    if not frames:
        return "[]"
    if len(frames) <= max_items:
        return "[" + ", ".join(str(x) for x in frames) + "]"
    head = ", ".join(str(x) for x in frames[: max_items // 2])
    tail = ", ".join(str(x) for x in frames[-max_items // 2 :])
    return f"[{head}, ..., {tail}]"


def make_contact_sheet(
    image_paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
    thumb_width: int = 220,
    pad: int = 10,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not image_paths:
        return

    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        scale = thumb_width / img.width
        thumb_size = (thumb_width, int(round(img.height * scale)))
        img = img.resize(thumb_size, Image.Resampling.LANCZOS)
        thumbs.append(img)

    font = ImageFont.load_default()
    label_height = 18
    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    cell_w = thumb_width + pad * 2
    cell_h = thumbs[0].height + label_height + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)

    for idx, (thumb, label) in enumerate(zip(thumbs, labels)):
        row, col = divmod(idx, cols)
        x = col * cell_w + pad
        y = row * cell_h + pad
        sheet.paste(thumb, (x, y))
        draw.text((x, y + thumb.height + 3), label, fill=(20, 20, 20), font=font)

    ensure_dir(output_path.parent)
    sheet.save(output_path)


def compare_rgb_dirs(dir_a: Path, dir_b: Path, frames: Iterable[int]) -> Dict[str, Any]:
    from PIL import Image

    frame_stats = []
    for frame in frames:
        path_a = dir_a / f"frame_{frame:04d}.png"
        path_b = dir_b / f"frame_{frame:04d}.png"
        if not path_a.exists() or not path_b.exists():
            continue
        arr_a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.int16)
        arr_b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.int16)
        diff = np.abs(arr_a - arr_b)
        frame_stats.append(
            {
                "frame": int(frame),
                "max_abs": int(diff.max()),
                "mean_abs": float(diff.mean()),
            }
        )

    if not frame_stats:
        return {"frames_compared": 0, "max_abs": None, "mean_abs": None, "per_frame": []}

    return {
        "frames_compared": len(frame_stats),
        "max_abs": int(max(row["max_abs"] for row in frame_stats)),
        "mean_abs": float(np.mean([row["mean_abs"] for row in frame_stats])),
        "per_frame": frame_stats,
    }
