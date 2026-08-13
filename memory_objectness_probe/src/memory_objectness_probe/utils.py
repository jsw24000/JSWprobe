from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np


VARIANT_TO_POSITION = {
    "absent": 0,
    "pos1": 1,
    "pos2": 2,
    "pos3": 3,
    "pos4": 4,
}

POSITION_TO_VARIANT = {v: k for k, v in VARIANT_TO_POSITION.items()}
EXPECTED_VARIANTS = tuple(VARIANT_TO_POSITION.keys())


def repo_root_from_file(file: str | Path) -> Path:
    return Path(file).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=True)
        f.write("\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize_cell(row.get(k)) for k in fieldnames})


def read_manifest_csv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    for row in rows:
        for key in ("target_present", "target_position_id", "probe_frame"):
            if key in row and row[key] not in ("", None):
                row[key] = int(row[key])
        for key in ("frame15_target_mask_area", "frame15_target_visible_ratio"):
            if key in row and row[key] not in ("", None):
                row[key] = float(row[key])
    return rows


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def stable_sample_id(base_scene_id: str, variant_id: str) -> str:
    return f"{base_scene_id}__{variant_id}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def sha256_file(path: str | Path, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    remaining = max_bytes
    with Path(path).open("rb") as f:
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk_size = 1024 * 1024
            if remaining is not None:
                chunk_size = min(chunk_size, remaining)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def add_src_to_path(project_root: str | Path) -> None:
    src = Path(project_root).resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def add_lingbot_to_path(lingbot_root: str | Path) -> None:
    root = Path(lingbot_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def finite_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def parse_optional_int(value: str | int | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if str(value).lower() in {"none", "null", ""}:
        return None
    return int(value)


def normalize_layer_name(layer: str | int) -> str:
    text = str(layer)
    if text.startswith("block"):
        suffix = text[5:]
        if suffix.isdigit():
            return f"block{int(suffix):02d}"
        return text
    if text.isdigit():
        return f"block{int(text):02d}"
    return text


def torch_load_cpu(path: str | Path) -> Any:
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=False)


def list_images(rgb_dir: str | Path) -> List[Path]:
    return sorted(Path(rgb_dir).glob("frame_*.png"))
