"""Small IO helpers for experiment artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, data: Any) -> Path:
    """Save JSON with UTF-8 text and stable indentation."""

    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(path: str | Path, obj: Any) -> Path:
    """Save a Python object with pickle."""

    out = Path(path)
    ensure_dir(out.parent)
    with out.open("wb") as f:
        pickle.dump(obj, f)
    return out


def load_pickle(path: str | Path) -> Any:
    """Load a Python object from pickle."""

    with Path(path).open("rb") as f:
        return pickle.load(f)

