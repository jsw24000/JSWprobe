from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import auto_scannet_root, project_root, resolve_project_path


_ENV_RE = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


def _expand_env_string(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(3)
        return os.environ.get(name, default or "")

    return _ENV_RE.sub(repl, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand_env(cfg)
    cfg["_config_path"] = str(cfg_path)
    return resolve_config(cfg)


def resolve_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    paths = cfg.setdefault("paths", {})
    base = project_root()

    for key in ("lingbot_repo", "lingbot_checkpoint", "dinov2_repo", "dinov2_checkpoint", "vggt_repo"):
        value = paths.get(key)
        if value and value != "auto":
            paths[key] = str(resolve_project_path(value, base=base))

    scannet_root = paths.get("scannet_root", "auto")
    if scannet_root in (None, "", "auto"):
        found = auto_scannet_root()
        paths["scannet_root"] = str(found) if found else None
    else:
        paths["scannet_root"] = str(resolve_project_path(scannet_root, base=base))

    output_root = cfg.setdefault("run", {}).get("output_root", "outputs")
    cfg["run"]["output_root"] = str(resolve_project_path(output_root, base=base))
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

