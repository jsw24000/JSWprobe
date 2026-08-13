from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format=LOGGER_FORMAT)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def import_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def require_module(module_name: str, extra: str = "") -> None:
    if not import_available(module_name):
        suffix = f" {extra}" if extra else ""
        raise RuntimeError(f"Missing Python module '{module_name}'.{suffix}")


def sha256_file(path: str | Path, max_bytes: int | None = None) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        remaining = max_bytes
        while True:
            if remaining is not None and remaining <= 0:
                break
            chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def git_commit(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        out = subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


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


def write_parquet(df: Any, path: str | Path) -> None:
    require_module(
        "pyarrow",
        "Install with `pip install -r covisibility_probe/requirements-extra.txt` inside the active experiment environment.",
    )
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=False)


def read_parquet(path: str | Path) -> Any:
    require_module(
        "pyarrow",
        "Install with `pip install -r covisibility_probe/requirements-extra.txt` inside the active experiment environment.",
    )
    import pandas as pd

    return pd.read_parquet(path)


def command_exists(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}

