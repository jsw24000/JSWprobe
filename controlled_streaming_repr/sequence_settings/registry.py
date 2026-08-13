"""Registry for manifest-only controlled sequence settings."""

from __future__ import annotations

from typing import Any

from .base_setting import BaseSequenceSetting


SEQUENCE_SETTING_REGISTRY: dict[str, type[BaseSequenceSetting]] = {}


def register_sequence_setting(name: str):
    """Decorator used by built-in and future sequence settings."""

    def decorator(cls: type[BaseSequenceSetting]) -> type[BaseSequenceSetting]:
        if name in SEQUENCE_SETTING_REGISTRY:
            raise KeyError(f"Sequence setting already registered: {name}")
        cls.setting_name = name
        SEQUENCE_SETTING_REGISTRY[name] = cls
        return cls

    return decorator


def get_sequence_setting_class(name: str) -> type[BaseSequenceSetting]:
    """Return a registered setting class by name."""

    _import_builtin_settings()
    try:
        return SEQUENCE_SETTING_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(SEQUENCE_SETTING_REGISTRY))
        raise KeyError(f"Unknown sequence setting: {name}. Available: {available}") from exc


def create_sequence_setting(name: str, config: dict[str, Any] | None = None, **kwargs: Any) -> BaseSequenceSetting:
    """Instantiate a registered sequence setting."""

    return get_sequence_setting_class(name)(config=config, **kwargs)


def list_sequence_settings() -> list[str]:
    """List registered sequence setting names."""

    _import_builtin_settings()
    return sorted(SEQUENCE_SETTING_REGISTRY)


def _import_builtin_settings() -> None:
    from . import (  # noqa: F401
        controlled_multi_length,
        controlled_16_staircase,
        natural_stream,
        order_perturbation,
        prefix_induced_modulation,
        repeated_frame,
        two_state_alternation,
    )
