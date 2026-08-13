#!/usr/bin/env python3
"""Render lightweight preview sheets for object-translation outputs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_scene_blender.object_translation.manifest_utils import ensure_dir, read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create preview images for an object-translation dataset.")
    parser.add_argument("--dataset-root", type=Path, default=PACKAGE_DIR / "outputs" / "object_translation_v1")
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--thumb-width", type=int, default=180)
    return parser.parse_args()


def open_rgb(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = width / image.width
    return image.resize((width, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)


def make_contact_sheet(
    image_paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
    thumb_width: int = 180,
    cols: int = 4,
) -> None:
    if not image_paths:
        return
    thumbs = [open_rgb(path, thumb_width) for path in image_paths]
    font = ImageFont.load_default()
    label_h = 18
    pad = 8
    cols = max(1, min(cols, len(thumbs)))
    rows = int(math.ceil(len(thumbs) / cols))
    cell_w = thumb_width + 2 * pad
    cell_h = thumbs[0].height + label_h + 2 * pad
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (246, 246, 242))
    draw = ImageDraw.Draw(sheet)
    for idx, (thumb, label) in enumerate(zip(thumbs, labels)):
        row, col = divmod(idx, cols)
        x = col * cell_w + pad
        y = row * cell_h + pad
        sheet.paste(thumb, (x, y))
        draw.text((x, y + thumb.height + 3), label, fill=(20, 20, 20), font=font)
    ensure_dir(output_path.parent)
    sheet.save(output_path)


def frame_map(frames: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], Mapping[str, Any]]:
    return {(str(row["scene_id"]), str(row["state_id"]), str(row["camera_id"])): row for row in frames}


def world_to_canvas(x: float, y: float, room_w: float, room_d: float, size: int, pad: int) -> Tuple[int, int]:
    px = pad + (x + room_w * 0.5) / room_w * (size - 2 * pad)
    py = pad + (room_d * 0.5 - y) / room_d * (size - 2 * pad)
    return int(round(px)), int(round(py))


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    corners: Sequence[Sequence[float]],
    room_w: float,
    room_d: float,
    size: int,
    pad: int,
    outline: Tuple[int, int, int],
    fill: Optional[Tuple[int, int, int]] = None,
) -> None:
    arr = np.asarray(corners, dtype=np.float64)
    x0, x1 = float(arr[:, 0].min()), float(arr[:, 0].max())
    y0, y1 = float(arr[:, 1].min()), float(arr[:, 1].max())
    p0 = world_to_canvas(x0, y0, room_w, room_d, size, pad)
    p1 = world_to_canvas(x1, y1, room_w, room_d, size, pad)
    box = [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])]
    draw.rectangle(box, outline=outline, fill=fill, width=2)


def render_topdown(scene_meta: Mapping[str, Any], states: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    room_w, room_d, _room_h = [float(v) for v in scene_meta["room_dimensions"]]
    size = 720
    pad = 48
    image = Image.new("RGB", (size, size), (244, 242, 236))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    room_box = [pad, pad, size - pad, size - pad]
    draw.rectangle(room_box, outline=(80, 80, 76), width=3)
    draw.line([room_box[0], room_box[1], room_box[2], room_box[1]], fill=(70, 70, 72), width=8)
    draw.line([room_box[0], room_box[1], room_box[0], room_box[3]], fill=(70, 70, 72), width=8)

    for obj in scene_meta["static_objects"]:
        if obj.get("is_room_shell") or obj.get("is_target"):
            continue
        draw_bbox(draw, obj["bbox_corners_world"], room_w, room_d, size, pad, outline=(82, 105, 120), fill=(196, 211, 216))

    for state in states:
        x, y, _z = [float(v) for v in state["object_bottom_center_world"]]
        px, py = world_to_canvas(x, y, room_w, room_d, size, pad)
        draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=(210, 82, 54), outline=(90, 30, 20), width=1)
        draw.text((px + 9, py - 8), str(state["state_id"]).replace("state_", "s"), fill=(35, 35, 35), font=font)

    draw.text((pad, 18), f"{scene_meta['scene_id']} target positions", fill=(30, 30, 30), font=font)
    ensure_dir(output_path.parent)
    image.save(output_path)


def render_camera_map(scene_meta: Mapping[str, Any], cameras: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    room_w, room_d, _room_h = [float(v) for v in scene_meta["room_dimensions"]]
    size = 720
    pad = 48
    image = Image.new("RGB", (size, size), (241, 244, 242))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    camera_xy = [(float(cam["position"][0]), float(cam["position"][1])) for cam in cameras]
    look_xy = [(float(cam["look_at"][0]), float(cam["look_at"][1])) for cam in cameras]
    xs = [-room_w * 0.5, room_w * 0.5, *(x for x, _y in camera_xy), *(x for x, _y in look_xy)]
    ys = [-room_d * 0.5, room_d * 0.5, *(y for _x, y in camera_xy), *(y for _x, y in look_xy)]
    margin = 0.35
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin

    def map_xy(x: float, y: float) -> Tuple[int, int]:
        px = pad + (x - xmin) / max(1e-6, xmax - xmin) * (size - 2 * pad)
        py = pad + (ymax - y) / max(1e-6, ymax - ymin) * (size - 2 * pad)
        return int(round(px)), int(round(py))

    room_tl = map_xy(-room_w * 0.5, room_d * 0.5)
    room_br = map_xy(room_w * 0.5, -room_d * 0.5)
    room_box = [min(room_tl[0], room_br[0]), min(room_tl[1], room_br[1]), max(room_tl[0], room_br[0]), max(room_tl[1], room_br[1])]
    draw.rectangle(room_box, outline=(82, 82, 76), width=3)
    draw.line([room_box[0], room_box[1], room_box[2], room_box[1]], fill=(60, 60, 60), width=8)
    draw.line([room_box[0], room_box[1], room_box[0], room_box[3]], fill=(60, 60, 60), width=8)
    for obj in scene_meta["static_objects"]:
        if obj.get("is_room_shell") or obj.get("is_target"):
            continue
        arr = np.asarray(obj["bbox_corners_world"], dtype=np.float64)
        p0 = map_xy(float(arr[:, 0].min()), float(arr[:, 1].min()))
        p1 = map_xy(float(arr[:, 0].max()), float(arr[:, 1].max()))
        draw.rectangle(
            [min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])],
            outline=(120, 120, 120),
            fill=(210, 216, 214),
            width=2,
        )

    for camera in cameras:
        x, y, _z = [float(v) for v in camera["position"]]
        lx, ly, _lz = [float(v) for v in camera["look_at"]]
        px, py = map_xy(x, y)
        qx, qy = map_xy(lx, ly)
        color = (48, 97, 170) if camera.get("is_primary") else (150, 88, 170)
        draw.line([px, py, qx, qy], fill=color, width=2)
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=color)
        draw.text((px + 8, py - 7), camera["camera_id"].replace("camera_", "c"), fill=(25, 25, 25), font=font)
    draw.text((pad, 18), f"{scene_meta['scene_id']} cameras", fill=(30, 30, 30), font=font)
    ensure_dir(output_path.parent)
    image.save(output_path)


def overlay_mask(rgb_path: Path, mask_path: Path, output_path: Path) -> None:
    rgb = Image.open(rgb_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(rgb.size)
    overlay = Image.new("RGBA", rgb.size, (220, 50, 35, 0))
    overlay.putalpha(mask.point(lambda value: 110 if value > 127 else 0))
    out = Image.alpha_composite(rgb, overlay).convert("RGB")
    ensure_dir(output_path.parent)
    out.save(output_path)


def render_overlay_sheet(frames: Sequence[Mapping[str, Any]], output_path: Path, thumb_width: int) -> None:
    temp_paths: List[Path] = []
    labels: List[str] = []
    for idx, frame in enumerate(frames[:8]):
        out = output_path.parent / f"_overlay_{idx:02d}.png"
        overlay_mask(Path(frame["rgb"]), Path(frame["target_mask"]), out)
        temp_paths.append(out)
        labels.append(f"{frame['state_id']} {frame['camera_id']}")
    make_contact_sheet(temp_paths, labels, output_path, thumb_width=thumb_width, cols=4)
    for path in temp_paths:
        if path.exists():
            path.unlink()


def render_previews(root: Path, scene_id: Optional[str], thumb_width: int) -> Dict[str, Any]:
    scenes = read_jsonl(root / "manifests" / "scenes.jsonl")
    states = read_jsonl(root / "manifests" / "states.jsonl")
    cameras = read_jsonl(root / "manifests" / "cameras.jsonl")
    frames = read_jsonl(root / "manifests" / "frames.jsonl")
    triplets = read_jsonl(root / "manifests" / "composition_triplets.jsonl")
    if not scenes:
        raise RuntimeError(f"No scenes found under {root}")
    chosen = scene_id or str(scenes[0]["scene_id"])
    preview_dir = ensure_dir(root / "previews")
    scene_meta = read_json(root / chosen / "scene_metadata.json")
    scene_states = [row for row in states if row["scene_id"] == chosen]
    scene_cameras = [row for row in cameras if row["scene_id"] == chosen]
    scene_frames = [row for row in frames if row["scene_id"] == chosen]
    fmap = frame_map(frames)

    outputs: List[str] = []
    topdown = preview_dir / f"{chosen}_topdown_positions.png"
    camera_map = preview_dir / f"{chosen}_camera_map.png"
    render_topdown(scene_meta, scene_states, topdown)
    render_camera_map(scene_meta, scene_cameras, camera_map)
    outputs.extend([str(topdown), str(camera_map)])

    same_camera = [row for row in scene_frames if row["camera_id"] == "camera_000"]
    same_camera = sorted(same_camera, key=lambda row: row["state_id"])
    path = preview_dir / f"{chosen}_camera_000_states_contact_sheet.png"
    make_contact_sheet([Path(row["rgb"]) for row in same_camera], [str(row["state_id"]) for row in same_camera], path, thumb_width=thumb_width, cols=4)
    outputs.append(str(path))

    same_state = [row for row in scene_frames if row["state_id"] == "state_000"]
    same_state = sorted(same_state, key=lambda row: row["camera_id"])
    path = preview_dir / f"{chosen}_state_000_cameras_contact_sheet.png"
    make_contact_sheet([Path(row["rgb"]) for row in same_state], [str(row["camera_id"]) for row in same_state], path, thumb_width=thumb_width, cols=4)
    outputs.append(str(path))

    path = preview_dir / f"{chosen}_rgb_mask_overlay_contact_sheet.png"
    render_overlay_sheet(same_camera or scene_frames, path, thumb_width=thumb_width)
    outputs.append(str(path))

    scene_triplets = [row for row in triplets if row["scene_id"] == chosen]
    if scene_triplets:
        triplet = scene_triplets[0]
        triplet_frames = [
            fmap[(chosen, triplet["state_0"], triplet["camera_id"])],
            fmap[(chosen, triplet["state_1"], triplet["camera_id"])],
            fmap[(chosen, triplet["state_2"], triplet["camera_id"])],
        ]
        path = preview_dir / f"{chosen}_composition_triplet_contact_sheet.png"
        make_contact_sheet(
            [Path(row["rgb"]) for row in triplet_frames],
            [str(triplet["state_0"]), str(triplet["state_1"]), str(triplet["state_2"])],
            path,
            thumb_width=thumb_width,
            cols=3,
        )
        outputs.append(str(path))

    summary = {"scene_id": chosen, "preview_count": len(outputs), "previews": outputs}
    write_json(preview_dir / f"{chosen}_preview_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = render_previews(args.dataset_root, args.scene_id, args.thumb_width)
    print(summary)


if __name__ == "__main__":
    main()
