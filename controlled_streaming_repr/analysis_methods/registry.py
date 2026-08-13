"""Registry for pluggable controlled analysis methods."""

from __future__ import annotations

from typing import Any

from .base_analysis import BaseAnalysisMethod


ANALYSIS_METHOD_REGISTRY: dict[str, type[BaseAnalysisMethod]] = {}


def register_analysis_method(name: str):
    """Decorator for registering an analysis method."""

    def decorator(cls: type[BaseAnalysisMethod]) -> type[BaseAnalysisMethod]:
        if name in ANALYSIS_METHOD_REGISTRY:
            raise KeyError(f"Analysis method already registered: {name}")
        cls.method_name = name
        ANALYSIS_METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def get_analysis_method_class(name: str) -> type[BaseAnalysisMethod]:
    """Return a registered analysis class."""

    _import_builtin_methods()
    try:
        return ANALYSIS_METHOD_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(ANALYSIS_METHOD_REGISTRY))
        raise KeyError(f"Unknown analysis method: {name}. Available: {available}") from exc


def create_analysis_method(name: str, config: dict[str, Any] | None = None) -> BaseAnalysisMethod:
    """Instantiate a registered analysis method."""

    return get_analysis_method_class(name)(config=config)


def list_analysis_methods() -> list[str]:
    """List registered analysis method names."""

    _import_builtin_methods()
    return sorted(ANALYSIS_METHOD_REGISTRY)


def _import_builtin_methods() -> None:
    from . import (  # noqa: F401
        attention_mass,
        frame_gram,
        geometry_stability,
        multi_length_delta,
        patch_affinity,
        prefix_delta,
        shared_pca_patchmap,
        staircase_delta,
        token_drift,
    )
