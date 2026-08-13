"""Pluggable analysis methods for controlled token bundles."""

from .registry import (
    create_analysis_method,
    get_analysis_method_class,
    list_analysis_methods,
    register_analysis_method,
)

__all__ = [
    "create_analysis_method",
    "get_analysis_method_class",
    "list_analysis_methods",
    "register_analysis_method",
]
