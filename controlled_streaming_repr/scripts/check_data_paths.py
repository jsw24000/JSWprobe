#!/usr/bin/env python3
"""Check external ScanNet raw data paths for this project."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "data_scannet.yaml"
SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")


DEFAULT_CONFIG: dict[str, Any] = {
    "env_var": "SCANNET_RAW_ROOT",
    "verified_raw_scans_root": "/home/data1/ScanNet/scans",
    "candidate_roots": [
        "/home/data1/ScanNet/scans",
        "/home/data1/ScanNet",
        "/home/data1",
        "/home/data1/3dsm/S2VGGT",
        "/disk1/3dsm/S2VGGT",
        "/disk1/scannet",
        "/data1",
    ],
    "previous_experiment_scenes": [
        "scene0000_00",
        "scene0002_00",
        "scene0006_00",
        "scene0010_00",
        "scene0015_00",
        "scene0016_00",
        "scene0024_00",
        "scene0030_00",
    ],
}


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small YAML shape used by configs/data_scannet.yaml."""

    if not path.exists():
        return dict(DEFAULT_CONFIG)

    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {**DEFAULT_CONFIG, **loaded}
    except ImportError:
        pass

    data: dict[str, Any] = {}
    current_list: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            current_list = stripped[:-1]
            data[current_list] = []
            continue

        if stripped.startswith("-") and current_list:
            data[current_list].append(stripped[1:].strip().strip('"').strip("'"))
            continue

        current_list = None
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            data[key.strip()] = value.strip().strip('"').strip("'")

    return {**DEFAULT_CONFIG, **data}


def _scene_root_from_candidate(path: Path) -> Path | None:
    """Return a scans directory if this candidate looks like ScanNet data."""

    if path.is_dir() and path.name == "scans":
        return path
    if (path / "scans").is_dir():
        return path / "scans"
    if path.is_dir() and any(
        child.is_dir() and SCENE_RE.match(child.name)
        for child in path.glob("scene????_??")
    ):
        return path
    return None


def _count_scene_dirs(scans_root: Path) -> int:
    return sum(1 for p in scans_root.iterdir() if p.is_dir() and SCENE_RE.match(p.name))


def _scene_status(scans_root: Path, scene: str) -> dict[str, Any]:
    scene_dir = scans_root / scene
    sens = scene_dir / f"{scene}.sens"
    aggregation = scene_dir / f"{scene}.aggregation.json"
    filt_instance = scene_dir / f"{scene}_2d-instance-filt.zip"
    raw_instance = scene_dir / f"{scene}_2d-instance.zip"
    return {
        "scene": scene,
        "scene_dir": scene_dir,
        "exists": scene_dir.is_dir(),
        "sens": sens.is_file(),
        "aggregation": aggregation.is_file(),
        "filtered_instance_zip": filt_instance.is_file(),
        "raw_instance_zip": raw_instance.is_file(),
    }


def main() -> int:
    config = _parse_simple_yaml(CONFIG_PATH)
    env_var = str(config.get("env_var", "SCANNET_RAW_ROOT"))
    env_value = os.environ.get(env_var)

    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(Path(p).expanduser() for p in config.get("candidate_roots", []))
    verified = config.get("verified_raw_scans_root")
    if verified:
        candidates.append(Path(str(verified)).expanduser())

    print("ScanNet raw 数据路径检查")
    print(f"项目目录: {PROJECT_DIR}")
    print(f"配置文件: {CONFIG_PATH}")
    print(f"可选环境变量: {env_var}")
    if env_value:
        print(f"环境变量当前值: {env_value}")
    else:
        print("环境变量当前值: 未设置")
    print()

    scans_root: Path | None = None
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        exists = candidate.exists()
        found_root = _scene_root_from_candidate(candidate) if exists else None
        if scans_root is None and found_root is not None:
            scans_root = found_root

        mark = "✓" if found_root is not None else ("-" if exists else "✗")
        detail = "可作为 scans root" if found_root is not None else (
            "存在但不是 scans root" if exists else "缺失"
        )
        print(f"{mark} {candidate}: {detail}")

    print()
    if scans_root is None:
        print("未找到可用 ScanNet scans root。请设置 SCANNET_RAW_ROOT=/path/to/ScanNet/scans。")
        return 1

    print(f"选定 raw scans root: {scans_root}")
    print(f"发现 scene 目录数量: {_count_scene_dirs(scans_root)}")
    print()

    scenes = config.get("previous_experiment_scenes", [])
    print("旧实验使用过的 scene 检查:")
    all_ok = True
    for scene in scenes:
        status = _scene_status(scans_root, str(scene))
        ok = bool(status["exists"] and status["sens"])
        all_ok = all_ok and ok
        mark = "✓" if ok else "✗"
        print(
            f"{mark} {scene}: dir={status['exists']}, sens={status['sens']}, "
            f"aggregation={status['aggregation']}, "
            f"filt_zip={status['filtered_instance_zip']}, "
            f"raw_zip={status['raw_instance_zip']}"
        )

    print()
    if all_ok:
        print("检查通过：raw ScanNet 数据已找到；当前不需要额外 export 全局变量。")
        print(f"如需换机器覆盖路径，可设置: export {env_var}=/path/to/ScanNet/scans")
        return 0

    print("部分旧实验 scene 不完整；请检查 raw 数据盘或用环境变量覆盖路径。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

