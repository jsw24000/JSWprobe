#!/usr/bin/env python3
"""Create a controlled experiment interface folder."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"


def validate_name(name: str) -> str:
    if not name:
        raise ValueError("Experiment name cannot be empty.")
    if "/" in name or "\\" in name:
        raise ValueError("Experiment name must be a folder name, not a path.")
    if name.startswith("."):
        raise ValueError("Experiment name must not start with '.'.")
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a new controlled experiment interface.")
    parser.add_argument("--name", required=True, help="Experiment folder name, e.g. exp04_xxx")
    args = parser.parse_args()

    name = validate_name(args.name)
    exp_dir = EXPERIMENTS_DIR / name
    if exp_dir.exists():
        print(f"实验文件夹已存在，不覆盖: {exp_dir}")
        return 1

    exp_dir.mkdir(parents=False, exist_ok=False)

    readme = f"""# {name}

Controlled experiment interface.

请在这里补充：

- scientific question
- controlled input conditions
- geometry source
- token extraction plan
- analysis methods

当前目录只定义实验接口；具体运行脚本可按该接口读取 geometry 和 token_bundle。
"""

    config = f"""experiment_name: "{name}"
status: "interface"
model: "lingbot-map"
sequence_id: null
geometry_source: null
token_bundle: null
conditions: []
analysis_methods:
  - "pca"
  - "gram"
  - "rsa"
  - "cka"
  - "token_displacement"
notes: "Generated controlled experiment interface."
"""

    (exp_dir / "README.md").write_text(readme, encoding="utf-8")
    (exp_dir / "config.yaml").write_text(config, encoding="utf-8")
    print(f"已创建实验模板: {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
