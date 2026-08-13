from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT.parent


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherited = cfg.pop("inherits", None)
    if inherited:
        base = load_config(path.parent / inherited)
        cfg = deep_update(base, cfg)

    cfg["_config_path"] = str(path)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base) if base is not None else PROJECT_ROOT) / path


def json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is expected, but keep this safe.
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(data), f, indent=2, sort_keys=True)
        f.write("\n")


def write_table(path: str | Path, rows: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a small table, preferring Parquet when a backend is available.

    Returns a status dictionary. If Parquet cannot be written in the current
    Python environment, a CSV sidecar is written and the status makes that
    explicit instead of pretending the requested Parquet exists.
    """
    path = Path(path)
    ensure_dir(path.parent)
    status: dict[str, Any] = {
        "requested_path": str(path),
        "num_rows": len(rows),
        "format": None,
        "path": None,
        "parquet_available": False,
    }
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        status.update({"format": "parquet", "path": str(path), "parquet_available": True})
    except Exception as exc:
        csv_path = path.with_suffix(path.suffix + ".csv") if path.suffix else path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(json_ready(row))
        status.update({
            "format": "csv_fallback",
            "path": str(csv_path),
            "parquet_available": False,
            "parquet_error": str(exc),
        })

    if metadata is not None:
        write_json(path.with_suffix(path.suffix + ".metadata.json"), {"table": status, "metadata": metadata})
    return status


def read_table(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    parquet_error: Exception | None = None
    if path.exists():
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict(orient="records")
        except Exception as exc:
            parquet_error = exc
    csv_path = path.with_suffix(path.suffix + ".csv") if path.suffix else path.with_suffix(".csv")
    if not csv_path.exists():
        if path.exists() and parquet_error is not None:
            raise RuntimeError(f"Table exists at {path}, but no Parquet reader is available: {parquet_error}") from parquet_error
        raise FileNotFoundError(f"No table found at {path} or {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [{key: _cast_table_value(value) for key, value in row.items()} for row in csv.DictReader(f)]


def _cast_table_value(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def command_output(args: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_environment(lingbot_root: str | Path | None = None) -> dict[str, Any]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "project_git_commit": command_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT),
    }
    if lingbot_root is not None:
        env["lingbot_git_commit"] = command_output(["git", "rev-parse", "HEAD"], cwd=lingbot_root)
    return env
