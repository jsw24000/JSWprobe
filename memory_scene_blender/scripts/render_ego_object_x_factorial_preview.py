#!/usr/bin/env python3
"""Create compact temporal previews for ego/object factorial datasets.

The dataset root is a mode-specific directory (for example ``.../smoke`` or
``.../pilot``).  Manifest paths and frame paths are interpreted relative to
that root, so previews remain portable when a dataset is moved.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_scene_blender.object_translation.manifest_utils import (  # noqa: E402
    ensure_dir,
    read_jsonl,
    write_json,
)


ROW_SPECS: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("static", (0, 0)),
    ("pure ego", (1, 0)),
    ("pure object", (0, -1)),
    ("compensated", (1, 1)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 4-condition x 3-timepoint contact sheets for each factorial group."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PACKAGE_DIR / "outputs" / "ego_object_x_factorial_v1" / "pilot",
        help="Mode-specific dataset root containing manifests/sequences.jsonl and frames.jsonl.",
    )
    parser.add_argument(
        "--group-id",
        action="append",
        default=None,
        help="Render only this group ID; may be supplied more than once.",
    )
    parser.add_argument(
        "--scene-id", type=str, default=None, help="Render only groups in this scene."
    )
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived preview sheets and preview_summary.json.",
    )
    return parser.parse_args()


def _group_id(sequence: Mapping[str, Any]) -> str:
    explicit = sequence.get("group_id", sequence.get("factorial_group_id"))
    if explicit is not None:
        return str(explicit)
    anchor_id = sequence.get("object_anchor_id", sequence.get("anchor_id"))
    required = {
        "scene_id": sequence.get("scene_id"),
        "object_anchor_id": anchor_id,
        "base_camera_id": sequence.get("base_camera_id"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise KeyError(f"Cannot derive group ID; sequence is missing {missing}: {sequence}")
    keys = ("scene_id", "object_anchor_id", "base_camera_id")
    return "__".join(str(required[key]) for key in keys)


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "group"


def _frame_path_value(frame: Mapping[str, Any], kind: str) -> Optional[str]:
    aliases = (kind,) if kind != "target_mask" else ("target_mask", "mask")
    for container_key in ("frame_paths", "paths"):
        container = frame.get(container_key)
        if isinstance(container, Mapping):
            for alias in aliases:
                value = container.get(alias)
                if value:
                    return str(value)
    # Tolerate the direct path keys used by the older object-translation
    # manifest while preferring the new explicit frame_paths mapping.
    for alias in aliases:
        value = frame.get(alias)
        if value:
            return str(value)
    return None


def _resolve_relative_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidates = [root / path, root.parent / path]
    if path.parts and path.parts[0] == "memory_scene_blender":
        candidates.append(PROJECT_ROOT / path)
    if path.parts and path.parts[0] == "outputs":
        candidates.append(PACKAGE_DIR / path)
    candidates.extend((PACKAGE_DIR / "outputs" / path, Path.cwd() / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return the canonical mode-root interpretation for a useful error/summary
    # even when a render was interrupted before the file was written.
    return root / path


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _open_rgb(path: Path, size: Tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _open_mask_overlay(rgb_path: Path, mask_path: Path, size: Tuple[int, int]) -> Image.Image:
    rgb = _open_rgb(rgb_path, size).convert("RGBA")
    with Image.open(mask_path) as source:
        mask = source.convert("L").resize(size, Image.Resampling.NEAREST)
    alpha = mask.point(lambda value: 112 if value > 127 else 0)
    overlay = Image.new("RGBA", size, (232, 48, 42, 0))
    overlay.putalpha(alpha)
    return Image.alpha_composite(rgb, overlay).convert("RGB")


def _placeholder(size: Tuple[int, int], message: str) -> Image.Image:
    image = Image.new("RGB", size, (232, 232, 228))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(175, 70, 62), width=2)
    draw.multiline_text(
        (8, 8), message, fill=(110, 35, 30), font=ImageFont.load_default(), spacing=3
    )
    return image


def _thumbnail_size(
    root: Path,
    selected_frames: Sequence[Optional[Mapping[str, Any]]],
    thumb_width: int,
) -> Tuple[int, int]:
    for frame in selected_frames:
        if frame is None:
            continue
        raw = _frame_path_value(frame, "rgb")
        if raw is None:
            continue
        path = _resolve_relative_path(root, raw)
        if path.is_file():
            with Image.open(path) as image:
                height = max(1, int(round(thumb_width * image.height / image.width)))
            return thumb_width, height
    return thumb_width, thumb_width


def _render_sheet(
    root: Path,
    group_id: str,
    frame_indices: Sequence[int],
    cells: Sequence[Tuple[str, Tuple[int, int], Optional[Mapping[str, Any]]]],
    output_path: Path,
    thumb_width: int,
    mask_overlay: bool,
) -> Dict[str, List[str]]:
    selected_frames = [frame for _label, _condition, frame in cells]
    thumb_size = _thumbnail_size(root, selected_frames, thumb_width)
    font = ImageFont.load_default()
    pad = 8
    title_h = 28
    col_header_h = 23
    row_label_w = 136
    cell_caption_h = 18
    cell_w = thumb_size[0] + 2 * pad
    cell_h = thumb_size[1] + cell_caption_h + 2 * pad
    sheet = Image.new(
        "RGB",
        (
            row_label_w + len(frame_indices) * cell_w,
            title_h + col_header_h + len(ROW_SPECS) * cell_h,
        ),
        (247, 247, 243),
    )
    draw = ImageDraw.Draw(sheet)
    kind = "target-mask overlay" if mask_overlay else "RGB"
    draw.text((8, 8), f"{group_id} | {kind}", fill=(20, 20, 20), font=font)
    for column, frame_index in enumerate(frame_indices):
        x = row_label_w + column * cell_w + pad
        role = ("start", "middle", "last")[column]
        draw.text((x, title_h + 4), f"frame {frame_index} ({role})", fill=(35, 35, 35), font=font)

    missing_rgb: List[str] = []
    missing_masks: List[str] = []
    for cell_index, (row_label, condition, frame) in enumerate(cells):
        row, column = divmod(cell_index, len(frame_indices))
        x = row_label_w + column * cell_w + pad
        y = title_h + col_header_h + row * cell_h + pad
        if column == 0:
            draw.multiline_text(
                (8, y + 4),
                f"{row_label}\n(e={condition[0]:+d}, o={condition[1]:+d})",
                fill=(25, 25, 25),
                font=font,
                spacing=4,
            )

        frame_index = int(frame_indices[column])
        identifier = f"{group_id} e={condition[0]} o={condition[1]} frame={frame_index}"
        if frame is None:
            thumb = _placeholder(thumb_size, "missing manifest row")
            missing_rgb.append(identifier)
            if mask_overlay:
                missing_masks.append(identifier)
        else:
            rgb_raw = _frame_path_value(frame, "rgb")
            rgb_path = _resolve_relative_path(root, rgb_raw) if rgb_raw is not None else None
            if rgb_path is None or not rgb_path.is_file():
                thumb = _placeholder(thumb_size, "missing RGB")
                missing_rgb.append(identifier)
            elif mask_overlay:
                mask_raw = _frame_path_value(frame, "target_mask")
                mask_path = _resolve_relative_path(root, mask_raw) if mask_raw is not None else None
                if mask_path is None or not mask_path.is_file():
                    thumb = _open_rgb(rgb_path, thumb_size)
                    missing_masks.append(identifier)
                else:
                    thumb = _open_mask_overlay(rgb_path, mask_path, thumb_size)
            else:
                thumb = _open_rgb(rgb_path, thumb_size)

        sheet.paste(thumb, (x, y))
        sequence_id = (
            "missing sequence" if frame is None else str(frame.get("sequence_id", "unknown"))
        )
        draw.text((x, y + thumb_size[1] + 3), sequence_id[-34:], fill=(55, 55, 55), font=font)

    ensure_dir(output_path.parent)
    sheet.save(output_path)
    return {"missing_rgb": missing_rgb, "missing_masks": missing_masks}


def _context(sequence: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "scene_id": str(sequence.get("scene_id", "")),
        "object_anchor_id": str(sequence.get("object_anchor_id", sequence.get("anchor_id", ""))),
        "base_camera_id": str(sequence.get("base_camera_id", "")),
    }


def render_previews(
    dataset_root: Path,
    requested_group_ids: Optional[Sequence[str]] = None,
    scene_id: Optional[str] = None,
    thumb_width: int = 240,
    overwrite: bool = False,
) -> Dict[str, Any]:
    root = dataset_root.expanduser().resolve()
    if int(thumb_width) <= 0:
        raise ValueError("thumb_width must be positive")
    sequence_manifest = root / "manifests" / "sequences.jsonl"
    frame_manifest = root / "manifests" / "frames.jsonl"
    if not sequence_manifest.is_file() or not frame_manifest.is_file():
        raise FileNotFoundError(
            f"Expected mode-specific manifests at {sequence_manifest} and {frame_manifest}"
        )

    sequences = read_jsonl(sequence_manifest)
    frames = read_jsonl(frame_manifest)
    if not sequences:
        raise RuntimeError(f"No sequences found in {sequence_manifest}")

    sequences_by_group: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        sequences_by_group[_group_id(sequence)].append(sequence)

    requested = set(str(value) for value in requested_group_ids or [])
    group_ids = [
        group
        for group in sorted(sequences_by_group)
        if (not requested or group in requested)
        and (
            scene_id is None
            or any(str(row.get("scene_id")) == scene_id for row in sequences_by_group[group])
        )
    ]
    if requested:
        unknown = sorted(requested.difference(sequences_by_group))
        if unknown:
            raise ValueError(f"Unknown requested group IDs: {unknown}")
    if not group_ids:
        raise RuntimeError("No factorial groups matched the requested filters")

    frame_lookup: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for frame in frames:
        key = (str(frame["sequence_id"]), int(frame["frame_index"]))
        if key in frame_lookup:
            raise ValueError(f"Duplicate frame manifest row for {key}")
        frame_lookup[key] = frame

    preview_dir = root / "previews"
    plans: List[Tuple[str, Path, Path]] = []
    for group in group_ids:
        stem = _safe_filename(group)
        plans.append(
            (
                group,
                preview_dir / f"{stem}_rgb_contact_sheet.png",
                preview_dir / f"{stem}_target_mask_overlay_contact_sheet.png",
            )
        )
    summary_path = preview_dir / "preview_summary.json"
    existing = [path for _group, rgb, mask in plans for path in (rgb, mask) if path.exists()]
    if summary_path.exists():
        existing.append(summary_path)
    if existing and not overwrite:
        shown = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(
            f"Preview outputs already exist ({shown}); pass --overwrite to replace them"
        )

    group_summaries: List[Dict[str, Any]] = []
    for group, rgb_output, mask_output in plans:
        group_sequences = sequences_by_group[group]
        by_condition: Dict[Tuple[int, int], Mapping[str, Any]] = {}
        for sequence in group_sequences:
            condition = (int(sequence["ego_level"]), int(sequence["object_level"]))
            if condition in by_condition:
                raise ValueError(f"{group}: duplicate sequence for condition {condition}")
            by_condition[condition] = sequence

        declared_lengths = sorted(
            {
                int(sequence["num_frames"])
                for sequence in group_sequences
                if sequence.get("num_frames") is not None
            }
        )
        if declared_lengths:
            num_frames = declared_lengths[0]
        else:
            group_sequence_ids = {str(sequence["sequence_id"]) for sequence in group_sequences}
            indices = [
                index
                for (sequence_id_key, index) in frame_lookup
                if sequence_id_key in group_sequence_ids
            ]
            num_frames = max(indices) + 1 if indices else 0
        if num_frames < 2:
            raise ValueError(
                f"{group}: cannot select start/middle/last from num_frames={num_frames}"
            )
        frame_indices = (0, num_frames // 2, num_frames - 1)

        cells: List[Tuple[str, Tuple[int, int], Optional[Mapping[str, Any]]]] = []
        missing_conditions: List[str] = []
        missing_frame_rows: List[str] = []
        for row_label, condition in ROW_SPECS:
            sequence = by_condition.get(condition)
            if sequence is None:
                missing_conditions.append(f"e={condition[0]},o={condition[1]}")
            for frame_index in frame_indices:
                frame = None
                if sequence is not None:
                    frame = frame_lookup.get((str(sequence["sequence_id"]), int(frame_index)))
                    if frame is None:
                        missing_frame_rows.append(
                            f"{sequence['sequence_id']} frame={frame_index}"
                        )
                cells.append((row_label, condition, frame))

        rgb_status = _render_sheet(
            root, group, frame_indices, cells, rgb_output, int(thumb_width), mask_overlay=False
        )
        mask_status = _render_sheet(
            root, group, frame_indices, cells, mask_output, int(thumb_width), mask_overlay=True
        )
        context = _context(group_sequences[0])
        group_summaries.append(
            {
                "group_id": group,
                **context,
                "sequence_count": len(group_sequences),
                "expected_factorial_condition_count": 25,
                "has_all_25_conditions": len(by_condition) == 25,
                "declared_num_frames_values": declared_lengths,
                "selected_frame_indices": list(frame_indices),
                "displayed_conditions": [
                    {"label": label, "ego_level": condition[0], "object_level": condition[1]}
                    for label, condition in ROW_SPECS
                ],
                "missing_display_conditions": missing_conditions,
                "missing_frame_manifest_rows": missing_frame_rows,
                "missing_rgb_files": rgb_status["missing_rgb"],
                "missing_target_masks": mask_status["missing_masks"],
                "rgb_contact_sheet": _relative_to_root(rgb_output, root),
                "target_mask_overlay_contact_sheet": _relative_to_root(mask_output, root),
            }
        )

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "dataset_root_semantics": "mode-specific; all reported preview paths are relative to it",
        "sequence_manifest": "manifests/sequences.jsonl",
        "frame_manifest": "manifests/frames.jsonl",
        "group_count": len(group_summaries),
        "groups": group_summaries,
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = render_previews(
        args.dataset_root,
        requested_group_ids=args.group_id,
        scene_id=args.scene_id,
        thumb_width=args.thumb_width,
        overwrite=args.overwrite,
    )
    print(f"Wrote {summary['group_count']} group preview(s) and previews/preview_summary.json")


if __name__ == "__main__":
    main()
