from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import yaml


def project_dir_from_config(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return value


def load_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.resolve()
    exp_root = project_dir_from_config(config_path)
    cfg = load_yaml(config_path)
    paths_cfg = load_yaml(exp_root / cfg.get("paths_config", "configs/paths.yaml"))

    def resolve(value: str) -> str:
        expanded = os.path.expandvars(value)
        path = Path(expanded)
        if path.is_absolute():
            return str(path)
        return str((exp_root / path).resolve())

    cfg["_config_path"] = str(config_path)
    cfg["_experiment_root"] = str(exp_root)
    cfg["_paths"] = {key: resolve(value) if isinstance(value, str) else value for key, value in paths_cfg.items()}
    cfg["_output_dir"] = str((exp_root / cfg["output_dir"]).resolve())
    return cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def copy_config_used(config_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    shutil.copyfile(config_path, output_dir / "config_used.yaml")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def command_output(cmd: Sequence[str], cwd: Optional[Path] = None) -> str:
    return subprocess.check_output(list(cmd), cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.STDOUT)


def torch_info() -> Dict[str, Any]:
    try:
        import torch

        info = {
            "python": sys.executable,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            info["device_name_0"] = torch.cuda.get_device_name(0)
            info["device_capability_0"] = list(torch.cuda.get_device_capability(0))
        return info
    except Exception as exc:
        return {"python": sys.executable, "torch_error": repr(exc)}


def setup_import_paths(config: Mapping[str, Any]) -> None:
    for key in ("dinov2_repo", "vggt_repo"):
        path = config["_paths"].get(key)
        if path and path not in sys.path:
            sys.path.insert(0, path)


def split_for_scene(config: Mapping[str, Any], scene_id: str) -> str:
    for split, scene_ids in config["splits"].items():
        if scene_id in scene_ids:
            return "validation" if split == "val" else split
    return "unused"


def as_float_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=np.float64)

