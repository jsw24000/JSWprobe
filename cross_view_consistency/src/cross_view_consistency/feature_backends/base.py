"""Base classes for feature extraction backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class FeatureResult:
    features: torch.Tensor | None
    frame_ids: list[int]
    grid_hw: tuple[int, int]
    input_hw: tuple[int, int]
    backend_name: str
    layer_or_stage: str
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    failure_reason: str = ""

    def save(self, path: str | Path) -> None:
        payload = {
            "features": self.features.cpu() if self.features is not None else None,
            "frame_ids": self.frame_ids,
            "grid_hw": self.grid_hw,
            "input_hw": self.input_hw,
            "backend_name": self.backend_name,
            "layer_or_stage": self.layer_or_stage,
            "metadata": self.metadata,
            "status": self.status,
            "failure_reason": self.failure_reason,
        }
        torch.save(payload, path)

    @classmethod
    def failed(
        cls,
        *,
        backend_name: str,
        frame_ids: list[int],
        grid_hw: tuple[int, int],
        input_hw: tuple[int, int],
        reason: str,
        metadata: dict[str, Any] | None = None,
        status: str = "failed_unavailable",
    ) -> "FeatureResult":
        return cls(
            features=None,
            frame_ids=frame_ids,
            grid_hw=grid_hw,
            input_hw=input_hw,
            backend_name=backend_name,
            layer_or_stage="",
            metadata=metadata or {},
            status=status,
            failure_reason=reason,
        )


class FeatureBackend:
    name: str = "base"

    def extract(self, frames, scene, grid, cfg) -> list[FeatureResult]:
        raise NotImplementedError

