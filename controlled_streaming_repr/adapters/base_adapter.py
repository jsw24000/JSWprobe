"""Base interface for model-specific token extraction adapters."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

from .token_schema import TokenBundle


class BaseModelAdapter(ABC):
    """Common interface for model adapters.

    Concrete adapters should hide model-specific loading and forward details,
    while returning a shared `TokenBundle` schema.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model: Any = None

    def _resolve_path(self, path_value: str | Path | None) -> Path | None:
        """Resolve a path, using config file location when available."""

        if path_value is None:
            return None

        path = Path(path_value)
        if path.is_absolute():
            return path

        config_dir = self.config.get("_config_dir")
        if config_dir:
            return (Path(config_dir) / path).resolve()

        return (Path.cwd() / path).resolve()

    def check_environment(self) -> dict[str, Any]:
        """Return environment diagnostics for this adapter."""

        return {
            "adapter": self.__class__.__name__,
            "status": "not_implemented",
        }

    def load_model(self) -> Any:
        """Load the underlying model.

        Current adapters intentionally keep this unimplemented until weights
        and exact hook points are ready.
        """

        raise NotImplementedError

    def extract_tokens(
        self,
        input_sequence: Any,
        condition_id: str,
        **kwargs: Any,
    ) -> TokenBundle:
        """Extract tokens from an input sequence."""

        raise NotImplementedError

