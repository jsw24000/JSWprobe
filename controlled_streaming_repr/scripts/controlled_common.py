"""Shared helpers for controlled ScanNet command-line scripts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML or JSON config and attach resolved path metadata."""

    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        config = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML configs. Install pyyaml or use JSON.") from exc
        config = yaml.safe_load(text) or {}
    config = expand_env_values(config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    config["_project_dir"] = str(PROJECT_DIR)
    return config


def expand_env_values(value: Any) -> Any:
    """Expand environment variables, including ${VAR:-default} strings."""

    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if not isinstance(value, str):
        return value

    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(3)
        env_value = os.environ.get(name)
        if env_value:
            return env_value
        return default or ""

    return os.path.expandvars(pattern.sub(repl, value))


def output_root(config: dict[str, Any]) -> Path:
    """Resolve configured output root against the project directory."""

    root = Path(config.get("project", {}).get("output_root", "outputs")).expanduser()
    if not root.is_absolute():
        root = PROJECT_DIR / root
    return root.resolve()


def project_name(config: dict[str, Any]) -> str:
    return str(config.get("project", {}).get("name", "controlled_scannet_v1"))


def git_commit(path: str | Path = PROJECT_DIR) -> str | None:
    """Return the current git commit when ``path`` belongs to a git repo."""

    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def setup_logging(config: dict[str, Any], script_name: str) -> logging.Logger:
    """Log to terminal, the legacy logs dir, and a project-specific logs dir."""

    log_root = output_root(config) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    log_paths = [log_root / f"{script_name}.log", log_root / project_name(config) / f"{script_name}.log"]
    seen: set[Path] = set()
    for log_path in log_paths:
        if log_path in seen:
            continue
        seen.add(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def resolve_scannet_root(data_cfg: dict[str, Any], *, required_scene_ids: list[str] | None = None) -> Path:
    """Resolve ScanNet scene root from the experiment config or data_scannet.yaml."""

    configured = data_cfg.get("scannet_root")
    if configured and str(configured).lower() not in {"auto", "from_data_scannet_config", ""}:
        return Path(str(configured)).expanduser().resolve()

    aux_path = PROJECT_DIR / "configs" / "data_scannet.yaml"
    aux = load_config(aux_path) if aux_path.is_file() else {}
    env_var = str(data_cfg.get("env_var") or aux.get("env_var") or "SCANNET_RAW_ROOT")
    candidates: list[Path] = []
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(Path(str(path)).expanduser() for path in aux.get("candidate_roots", []))
    if aux.get("verified_raw_scans_root"):
        candidates.append(Path(str(aux["verified_raw_scans_root"])).expanduser())

    scenes = [str(scene) for scene in (required_scene_ids or [])]
    seen: set[str] = set()
    for candidate in candidates:
        for root in _candidate_scannet_roots(candidate):
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            if _root_has_required_scenes(root, scenes):
                return root.resolve()

    scene_hint = f" containing {', '.join(scenes)}" if scenes else ""
    raise FileNotFoundError(
        f"Could not resolve ScanNet scans root{scene_hint}. "
        f"Set {env_var} or update configs/data_scannet.yaml."
    )


def _candidate_scannet_roots(candidate: Path) -> list[Path]:
    """Return plausible ScanNet scene roots derived from one configured path."""

    options = [
        candidate,
        candidate / "scans",
        candidate / "scans_extracted",
        candidate / "ScanNet" / "scans",
        candidate / "ScanNet" / "scans_extracted",
    ]
    roots: list[Path] = []
    for option in options:
        if option.is_dir() and (
            option.name in {"scans", "scans_extracted"}
            or any(child.is_dir() and child.name.startswith("scene") for child in option.glob("scene????_??"))
        ):
            roots.append(option)
    return roots


def _root_has_required_scenes(root: Path, scenes: list[str]) -> bool:
    if not root.is_dir():
        return False
    if not scenes:
        return any(child.is_dir() and child.name.startswith("scene") for child in root.glob("scene????_??"))
    return all((root / scene).is_dir() for scene in scenes)


def save_resolved_config(config: dict[str, Any], out_dir: str | Path, filename: str = "resolved_config.json") -> Path:
    """Save the fully resolved runtime config as JSON."""

    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def parse_seq_lens(values: list[str] | None) -> list[int] | None:
    """Parse CLI sequence lengths from repeated args or comma-separated values."""

    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(part) for part in re.split(r"[, ]+", value) if part)
    return parsed


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def resolve_index_path(path: str | Path, config: dict[str, Any] | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if config:
        project = Path(config.get("_project_dir", PROJECT_DIR))
        return (project / candidate).resolve()
    return (PROJECT_DIR / candidate).resolve()


def resolve_manifest_path(record_path: str | Path, index_path: str | Path) -> Path:
    """Resolve a manifest path stored in an index."""

    path = Path(record_path)
    if path.is_absolute():
        return path
    index_parent = Path(index_path).resolve().parent
    project_candidate = (PROJECT_DIR / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return (index_parent / path).resolve()


def add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--setting", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)


def record_matches_filters(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if getattr(args, "setting", None) and record.get("setting_name") != args.setting:
        return False
    if getattr(args, "scene", None) and record.get("scene_id") != args.scene:
        return False
    if getattr(args, "condition_id", None) and record.get("condition_id") != args.condition_id:
        return False
    if getattr(args, "seq_len", None) and int(record.get("seq_len")) != int(args.seq_len):
        return False
    return True


def filtered_manifest_records(index: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    records = [record for record in index.get("manifests", []) if record_matches_filters(record, args)]
    max_runs = getattr(args, "max_runs", None)
    if max_runs is not None:
        records = records[: int(max_runs)]
    return records
