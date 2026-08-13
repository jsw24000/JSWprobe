#!/usr/bin/env python3
"""Check project paths without importing or running heavy models."""

from __future__ import annotations

from pathlib import Path


def _format_path_status(label: str, path: Path, expect_symlink: bool = False) -> tuple[bool, str]:
    exists = path.exists()
    is_link = path.is_symlink()
    ok = exists and (is_link if expect_symlink else True)

    if expect_symlink and is_link:
        detail = f"软链接 -> {path.readlink()}"
    elif expect_symlink and exists:
        detail = "存在，但不是软链接"
    elif exists:
        detail = "存在"
    else:
        detail = "缺失"

    mark = "✓" if ok else "✗"
    return ok, f"{mark} {label}: {detail} ({path})"


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    main_dir = project_dir.parent

    checks = [
        ("根目录 lingbot-map", main_dir / "lingbot-map", False),
        ("根目录 vggt", main_dir / "vggt", False),
        ("third_party/lingbot-map", project_dir / "third_party" / "lingbot-map", True),
        ("third_party/vggt", project_dir / "third_party" / "vggt", True),
        ("configs", project_dir / "configs", False),
        ("data", project_dir / "data", False),
        ("data/controlled_sequences", project_dir / "data" / "controlled_sequences", False),
        ("data/manifest_utils", project_dir / "data" / "manifest_utils", False),
        ("outputs/manifests", project_dir / "outputs" / "manifests", False),
        ("outputs/tokens", project_dir / "outputs" / "tokens", False),
        ("outputs/figures", project_dir / "outputs" / "figures", False),
        ("outputs/metrics", project_dir / "outputs" / "metrics", False),
        ("outputs/logs", project_dir / "outputs" / "logs", False),
    ]

    print("项目路径检查")
    print(f"主实验目录: {main_dir}")
    print(f"项目目录: {project_dir}")
    print()

    all_ok = True
    for label, path, expect_symlink in checks:
        ok, line = _format_path_status(label, path, expect_symlink)
        all_ok = all_ok and ok
        print(line)

    print()
    if all_ok:
        print("检查通过：基础目录和软链接均可用。")
        return 0

    print("检查未通过：请先修复上面标记为缺失或类型不符的路径。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
