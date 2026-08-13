"""Base interface for controlled sequence settings.

Sequence settings generate manifests only. They do not run models, extract
tokens, compute metrics, or create figures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from data.manifest_utils.manifest_schema import make_manifest, manifest_frame_entry
from data.manifest_utils.scannet_reader import ScanNetFrame, ScanNetScene


class BaseSequenceSetting(ABC):
    """Base class for manifest-only controlled sequence generators."""

    setting_name: str = "base"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        seq_lens: list[int] | None = None,
        random_seed: int = 2026,
        path_mode: str = "absolute",
        relative_to: str | Path | None = None,
    ) -> None:
        self.config = dict(config or {})
        self.seq_lens = [int(value) for value in (seq_lens or self.config.get("seq_lens", [16, 32]))]
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        self.path_mode = path_mode
        self.relative_to = Path(relative_to).resolve() if relative_to else None

    @abstractmethod
    def build_manifests(self, scene: ScanNetScene) -> list[dict[str, Any]]:
        """Return manifests generated from a scanned scene."""

    def _frame_entry(self, frame: ScanNetFrame, *, t: int, role: str, **extra: Any) -> dict[str, Any]:
        return manifest_frame_entry(
            frame,
            t=t,
            role=role,
            path_mode=self.path_mode,
            relative_to=self.relative_to,
            **extra,
        )

    def _manifest(
        self,
        scene: ScanNetScene,
        *,
        condition_id: str,
        seq_len: int,
        frames: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return make_manifest(
            scene_id=scene.scene_id,
            setting_name=self.setting_name,
            condition_id=condition_id,
            seq_len=seq_len,
            source_scene_dir=scene.scene_dir,
            frames=frames,
            metadata=metadata,
            path_mode=self.path_mode,
            relative_to=self.relative_to,
        )


def uniform_indices(
    n_items: int,
    count: int,
    *,
    avoid_boundary_ratio: float = 0.0,
    min_index: int | None = None,
    max_index: int | None = None,
) -> list[int]:
    """Select up to ``count`` approximately uniform integer indices."""

    if n_items <= 0 or count <= 0:
        return []
    lower = int(np.floor(n_items * avoid_boundary_ratio)) if min_index is None else min_index
    upper = int(np.ceil(n_items * (1.0 - avoid_boundary_ratio))) - 1 if max_index is None else max_index
    lower = max(0, lower)
    upper = min(n_items - 1, upper)
    if lower > upper:
        lower, upper = 0, n_items - 1
    available = list(range(lower, upper + 1))
    if len(available) <= count:
        return available
    positions = np.linspace(0, len(available) - 1, count)
    selected = [available[int(round(position))] for position in positions]
    return list(dict.fromkeys(selected))


def frame_pattern(roles: list[str], max_items: int = 16) -> list[str]:
    """Return a compact frame pattern list for manifest metadata."""

    if len(roles) <= max_items:
        return roles
    return roles[: max_items - 1] + ["..."]
