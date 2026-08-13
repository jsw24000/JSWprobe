from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAME_RE = re.compile(r"(\d{4,6})")
RUN_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(project_path(path))
    return cfg


def sanitize_run_id(run_id: str) -> str:
    cleaned = RUN_ID_RE.sub("_", run_id.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("run_id cannot be empty after sanitization")
    return cleaned


def current_run_id(cfg: dict[str, Any]) -> str | None:
    run_id = os.environ.get("REVISIT_MEMORY_RUN_ID") or cfg.get("project", {}).get("run_id")
    return sanitize_run_id(run_id) if run_id else None


def output_root(cfg: dict[str, Any]) -> Path:
    base = project_path(cfg["project"]["output_root"])
    run_id = current_run_id(cfg)
    if run_id:
        return ensure_dir(base / "runs" / run_id)
    return ensure_dir(base)


def ensure_dir(path: str | Path) -> Path:
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with project_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: str | Path, text: str) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with project_path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def data_root(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["paths"]["data_root"])


def condition_dir(cfg: dict[str, Any], condition: str) -> Path:
    return data_root(cfg) / condition


def parse_frame_id(path: str | Path) -> int:
    match = FRAME_RE.search(Path(path).stem)
    if not match:
        raise ValueError(f"Could not parse frame id from {path}")
    return int(match.group(1))


def rgb_path(cfg: dict[str, Any], condition: str, frame_id: int) -> Path:
    return condition_dir(cfg, condition) / "rgb" / f"frame_{frame_id:04d}.png"


def depth_path(cfg: dict[str, Any], condition: str, frame_id: int) -> Path:
    return condition_dir(cfg, condition) / "depth" / f"depth_{frame_id:04d}.exr"


def camera_json_path(cfg: dict[str, Any], condition: str) -> Path:
    return condition_dir(cfg, condition) / "cameras.json"


def metadata_path(cfg: dict[str, Any], condition: str) -> Path:
    return condition_dir(cfg, condition) / "metadata.json"


def list_frame_ids(cfg: dict[str, Any], condition: str) -> list[int]:
    frames = [parse_frame_id(p) for p in (condition_dir(cfg, condition) / "rgb").glob("frame_*.png")]
    return sorted(frames)


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(project_path(path)).convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(project_path(path)))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def load_npy_mask(path: str | Path, positive_id: int | None = None) -> np.ndarray:
    arr = np.load(project_path(path))
    return arr > 0 if positive_id is None else arr == positive_id


def load_depth_exr(path: str | Path) -> np.ndarray:
    path = project_path(path)
    cv2_error: Exception | None = None
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2  # type: ignore

        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"cv2 could not read {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth.astype(np.float32)
    except ImportError:
        pass
    except Exception as exc:
        cv2_error = exc

    try:
        import imageio.v3 as iio  # type: ignore

        depth = iio.imread(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read EXR depth {path}. Install opencv-python/imageio with EXR support, "
            f"or set OPENCV_IO_ENABLE_OPENEXR=1 before importing cv2. cv2 error: {cv2_error!r}"
        ) from exc


def find_counterfactual_mask(cfg: dict[str, Any], condition: str, frame_id: int) -> Path | None:
    base = condition_dir(cfg, condition) / "masks"
    candidates = [
        base / "counterfactual_target" / f"counterfactual_target_{frame_id:04d}.png",
        base / "target_binary" / f"target_binary_{frame_id:04d}.png",
        base / "target_binary" / f"frame_{frame_id:04d}.png",
        base / "target_binary" / f"frame_{frame_id:04d}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_target_presence_mask(cfg: dict[str, Any], condition: str, frame_id: int, target_object_id: int) -> Path | None:
    base = condition_dir(cfg, condition) / "masks"
    candidates = [
        base / "target_binary" / f"target_binary_{frame_id:04d}.png",
        base / "target_binary" / f"frame_{frame_id:04d}.png",
        base / "target_binary" / f"frame_{frame_id:04d}.npy",
        base / "object_id" / f"frame_{frame_id:04d}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_any_target_mask(path: str | Path, target_object_id: int | None = None) -> np.ndarray:
    path = project_path(path)
    if path.suffix == ".npy":
        return load_npy_mask(path, positive_id=target_object_id)
    return load_mask(path)


def target_object(cfg: dict[str, Any]) -> dict[str, Any]:
    shared_path = data_root(cfg) / "shared_scene_metadata.json"
    shared = read_json(shared_path)
    target_id = shared.get("target_object_id")
    for obj in shared.get("objects", []):
        if obj.get("is_target") or obj.get("object_id") == target_id:
            return obj
    raise ValueError(f"No target object found in {shared_path}")


def target_bbox(cfg: dict[str, Any]) -> dict[str, Any]:
    obj = target_object(cfg)
    center = np.asarray(obj["location"], dtype=np.float32)
    dims = np.asarray(obj["dimensions"], dtype=np.float32)
    return {
        "name": obj.get("name"),
        "object_id": obj.get("object_id"),
        "center": center.tolist(),
        "dimensions": dims.tolist(),
        "min": (center - dims / 2.0).tolist(),
        "max": (center + dims / 2.0).tolist(),
    }


def frame_ranges_to_ids(ranges: Iterable[Iterable[int]]) -> list[int]:
    ids: set[int] = set()
    for start, end in ranges:
        ids.update(range(int(start), int(end) + 1))
    return sorted(ids)


def load_reconstruction_npz(cfg: dict[str, Any], condition: str) -> dict[str, np.ndarray]:
    path = output_root(cfg) / "reconstruction" / condition / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing reconstruction output: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
