from __future__ import annotations

from typing import Mapping, Sequence


def scenes_for_split(config: Mapping, split: str) -> Sequence[str]:
    if split == "validation" and "validation" not in config["splits"]:
        return config["splits"].get("val", [])
    return config["splits"].get(split, [])

