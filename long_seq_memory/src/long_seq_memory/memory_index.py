from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TokenLayout:
    num_register_tokens: int = 4
    has_scale_token: bool = True
    patch_count: int = 0

    @property
    def camera_index(self) -> int:
        return 0

    @property
    def register_indices(self) -> list[int]:
        return list(range(1, 1 + self.num_register_tokens))

    @property
    def scale_index(self) -> int | None:
        if not self.has_scale_token:
            return None
        return 1 + self.num_register_tokens

    @property
    def num_special_tokens(self) -> int:
        return 1 + self.num_register_tokens + (1 if self.has_scale_token else 0)

    @property
    def patch_start_idx(self) -> int:
        return self.num_special_tokens

    @property
    def tokens_per_frame(self) -> int:
        return self.num_special_tokens + self.patch_count

    def token_type(self, local_token_index: int) -> tuple[str, int]:
        if local_token_index == self.camera_index:
            return "camera", 0
        if local_token_index in self.register_indices:
            return "register", local_token_index - 1
        if self.scale_index is not None and local_token_index == self.scale_index:
            return "scale", 0
        if local_token_index >= self.patch_start_idx:
            return "image_patch", local_token_index - self.patch_start_idx
        return "other", local_token_index


@dataclass
class MemoryTokenRecord:
    memory_slot: int
    source_frame_id: int
    source_local_index: int
    token_type: str
    token_type_index: int
    timestamp: float | None
    temporal_position: int
    layer_id: int | None = None
    head_id: int | None = None
    stream: str = "special"


def preprocessed_patch_grid(
    original_width: int,
    original_height: int,
    image_size: int = 518,
    patch_size: int = 14,
    mode: str = "crop",
) -> tuple[int, int]:
    if mode != "crop":
        raise NotImplementedError("Only LingBot canonical crop mode is implemented here.")
    new_width = image_size
    new_height = round(original_height * (new_width / original_width) / patch_size) * patch_size
    if new_height > image_size:
        new_height = image_size
    return new_width // patch_size, new_height // patch_size


def retained_patch_frames(processed_frames: list[int], scale_frames: int, sliding_window: int) -> list[int]:
    scale = processed_frames[:scale_frames]
    window = processed_frames[-sliding_window:] if sliding_window > 0 else []
    out = []
    seen = set()
    for frame_id in scale + window:
        if frame_id not in seen:
            out.append(frame_id)
            seen.add(frame_id)
    return out


def old_special_frames(processed_frames: list[int], scale_frames: int, sliding_window: int) -> list[int]:
    live = set(retained_patch_frames(processed_frames, scale_frames, sliding_window))
    return [frame_id for frame_id in processed_frames if frame_id not in live]


def memory_state_summary(
    current_frame: int,
    processed_frames: list[int],
    target_frame: int,
    scale_frames: int = 8,
    sliding_window: int = 64,
    num_register_tokens: int = 4,
) -> dict:
    patch_frames = retained_patch_frames(processed_frames, scale_frames, sliding_window)
    old_frames = old_special_frames(processed_frames, scale_frames, sliding_window)
    local_start = min(patch_frames[-sliding_window:]) if patch_frames else None
    local_end = max(patch_frames[-sliding_window:]) if patch_frames else None
    return {
        "current_frame": current_frame,
        "local_frame_range": [local_start, local_end],
        "patch_memory_frames": patch_frames,
        "old_memory_frames": old_frames,
        "old_memory_frame_range": [min(old_frames), max(old_frames)] if old_frames else None,
        "num_old_memory_frames": len(old_frames),
        "num_camera_tokens": len(old_frames),
        "num_register_tokens": len(old_frames) * num_register_tokens,
        "contains_target_frame": target_frame in old_frames or target_frame in patch_frames,
        "target_frame_in_old_special_memory": target_frame in old_frames,
        "target_frame_in_patch_window": target_frame in patch_frames,
    }


def build_special_memory_records(
    frames: Iterable[int],
    layout: TokenLayout,
    timestamps: dict[int, float] | None = None,
    layer_id: int | None = None,
    head_id: int | None = None,
) -> list[MemoryTokenRecord]:
    records: list[MemoryTokenRecord] = []
    slot = 0
    for temporal_position, frame_id in enumerate(frames):
        for local_idx in range(layout.num_special_tokens):
            token_type, type_idx = layout.token_type(local_idx)
            records.append(
                MemoryTokenRecord(
                    memory_slot=slot,
                    source_frame_id=int(frame_id),
                    source_local_index=local_idx,
                    token_type=token_type,
                    token_type_index=type_idx,
                    timestamp=None if timestamps is None else timestamps.get(int(frame_id)),
                    temporal_position=temporal_position,
                    layer_id=layer_id,
                    head_id=head_id,
                    stream="special",
                )
            )
            slot += 1
    return records


def write_records_csv(path: str | Path, records: list[MemoryTokenRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(MemoryTokenRecord(0, 0, 0, "", 0, None, 0).__dict__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_memory_state_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
