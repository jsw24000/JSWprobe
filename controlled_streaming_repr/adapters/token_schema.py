"""Shared token schema used by model adapters and analysis methods."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenBundle:
    """A lightweight container for token outputs from different models.

    `layer_tokens` follows this recommended structure:

    {
        "layer_0": {
            "patch_tokens": tensor_or_path_or_none,
            "camera_tokens": tensor_or_path_or_none,
            "register_tokens": tensor_or_path_or_none,
            "memory_tokens": tensor_or_path_or_none,
            "extra": {},
        },
        ...
    }

    Values intentionally use `Any` so this schema can store either in-memory
    arrays or paths to large `.npy`/`.pt` files without requiring torch at
    project setup time.
    """

    model_name: str
    sequence_id: str
    condition_id: str
    frame_paths: list[str]
    scene_id: str | None = None
    setting_name: str | None = None
    seq_len: int | None = None
    frame_ids: list[str] = field(default_factory=list)
    layer_tokens: dict[str, dict[str, Any]] = field(default_factory=dict)
    attention: dict[str, Any] = field(default_factory=dict)
    geometry_outputs: dict[str, Any] = field(default_factory=dict)
    output_predictions: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary for JSON/pickle serialization helpers."""

        return {
            "model_name": self.model_name,
            "sequence_id": self.sequence_id,
            "scene_id": self.scene_id,
            "setting_name": self.setting_name,
            "condition_id": self.condition_id,
            "seq_len": self.seq_len,
            "frame_paths": self.frame_paths,
            "frame_ids": self.frame_ids,
            "layer_tokens": self.layer_tokens,
            "attention": self.attention,
            "geometry_outputs": self.geometry_outputs,
            "output_predictions": self.output_predictions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenBundle":
        """Create a bundle from a dictionary, tolerating older bundle files."""

        geometry_outputs = data.get("geometry_outputs") or {}
        output_predictions = data.get("output_predictions")
        if not geometry_outputs and isinstance(output_predictions, dict):
            geometry_outputs = output_predictions
        return cls(
            model_name=data.get("model_name", "unknown_model"),
            sequence_id=data.get("sequence_id") or data.get("condition_id", "unknown_sequence"),
            scene_id=data.get("scene_id"),
            setting_name=data.get("setting_name"),
            condition_id=data.get("condition_id", "unknown_condition"),
            seq_len=data.get("seq_len"),
            frame_paths=list(data.get("frame_paths", [])),
            frame_ids=list(data.get("frame_ids", [])),
            layer_tokens=dict(data.get("layer_tokens", {})),
            attention=dict(data.get("attention", {})),
            geometry_outputs=dict(geometry_outputs),
            output_predictions=output_predictions,
            metadata=dict(data.get("metadata", {})),
        )

    def save_json(self, path: str | Path) -> Path:
        """Save a path-based bundle index as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path


def load_token_bundle(path: str | Path) -> TokenBundle:
    """Load a ``TokenBundle`` from JSON or a torch ``.pt`` file."""

    bundle_path = Path(path)
    if bundle_path.suffix == ".pt":
        try:
            import torch
        except ImportError as exc:
            raise ImportError(f"Loading {bundle_path} requires torch.") from exc
        loaded = torch.load(bundle_path, map_location="cpu")
        if isinstance(loaded, TokenBundle):
            return loaded
        if isinstance(loaded, dict):
            return TokenBundle.from_dict(loaded)
        raise TypeError(f"Unsupported token bundle object in {bundle_path}: {type(loaded)!r}")

    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    return TokenBundle.from_dict(data)


def save_token_bundle(bundle: TokenBundle, path: str | Path) -> Path:
    """Save a bundle as JSON, or as torch ``.pt`` when requested."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".pt":
        try:
            import torch
        except ImportError:
            fallback = output_path.with_suffix(".json")
            return bundle.save_json(fallback)
        torch.save(bundle.to_dict(), output_path)
        return output_path
    return bundle.save_json(output_path)
