"""Config loading, CLI overrides, and run-directory setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, load_yaml, project_root, resolve_path, save_yaml, timestamp


def load_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    cfg["_config_path"] = str(Path(config_path).resolve())
    cfg["_project_root"] = str(project_root())
    return cfg


def apply_cli_overrides(
    cfg: dict[str, Any],
    *,
    tag: str | None = None,
    num_frames: int | None = None,
    pairs_per_bucket: int | None = None,
    queries_per_pair: int | None = None,
    backends: str | None = None,
) -> dict[str, Any]:
    if tag:
        cfg.setdefault("run", {})["tag"] = tag
    if num_frames is not None:
        cfg.setdefault("frame_sampling", {})["num_frames"] = int(num_frames)
    if pairs_per_bucket is not None:
        cfg.setdefault("pairs", {})["pairs_per_bucket"] = int(pairs_per_bucket)
    if queries_per_pair is not None:
        cfg.setdefault("queries", {})["queries_per_pair"] = int(queries_per_pair)
    if backends:
        enabled = [x.strip() for x in backends.split(",") if x.strip()]
        cfg.setdefault("feature_backends", {})["enabled"] = enabled
    return cfg


def validate_and_resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    root = project_root()
    paths = cfg.setdefault("paths", {})
    for key in ["scannet_root", "lingbot_repo", "lingbot_checkpoint", "dinov2_repo", "dinov2_checkpoint"]:
        if key in paths:
            resolved = resolve_path(paths.get(key), root)
            paths[key] = str(resolved) if resolved is not None else ""
    return cfg


def create_run_dir(cfg: dict[str, Any]) -> Path:
    run_cfg = cfg.setdefault("run", {})
    scene_id = cfg.get("paths", {}).get("scene_id", "scene")
    tag = run_cfg.get("tag", "run")
    output_root = resolve_path(run_cfg.get("output_root", "outputs"), project_root() / "cross_view_consistency")
    assert output_root is not None
    ensure_dir(output_root)
    run_dir = output_root / f"{timestamp()}_{scene_id}_{tag}"
    ensure_dir(run_dir)
    save_yaml(cfg, run_dir / "config_used.yaml")
    return run_dir

