"""VGGT adapter interface for a future non-streaming baseline."""

from __future__ import annotations

from typing import Any

from .base_adapter import BaseModelAdapter
from .token_schema import TokenBundle


class VGGTAdapter(BaseModelAdapter):
    """Interface-only adapter for VGGT.

    Future implementation notes:
    - Extract VGGT patch tokens, aggregated tokens, and camera-related tokens.
    - Keep VGGT as the non-streaming reconstruction comparison model.
    - Do not call `from_pretrained` or load large weights in this skeleton.
    """

    def check_environment(self) -> dict[str, Any]:
        repo_path = self._resolve_path(self.config.get("repo_path", "../third_party/vggt"))
        weights_path = self._resolve_path(self.config.get("weights_path"))
        repo_exists = bool(repo_path and repo_path.exists())
        weights_exists = bool(weights_path and weights_path.exists())

        warnings: list[str] = []
        if not repo_exists:
            warnings.append("VGGT repo path does not exist.")
        if not weights_path:
            warnings.append("VGGT weights_path is not configured.")
        elif not weights_exists:
            warnings.append("VGGT weights path does not exist; skeleton will not run model.")

        return {
            "adapter": self.__class__.__name__,
            "model_name": self.config.get("model_name", "vggt"),
            "repo_path": str(repo_path) if repo_path else None,
            "repo_exists": repo_exists,
            "weights_path": str(weights_path) if weights_path else None,
            "weights_exists": weights_exists,
            "weights_ready": bool(self.config.get("weights_ready", False)),
            "warnings": warnings,
            "status": "ok_with_warnings" if warnings else "ok",
        }

    def load_model(self) -> Any:
        raise RuntimeError("当前仅为 VGGT adapter 接口骨架，不实际加载模型。")

    def extract_tokens(
        self,
        input_sequence: Any,
        condition_id: str,
        **kwargs: Any,
    ) -> TokenBundle:
        # TODO: connect VGGT hooks for patch / aggregated / camera-related tokens.
        # TODO: define exact token names after weights and model entry points are ready.
        raise NotImplementedError(
            "VGGT token extraction is interface-only; no model is run."
        )
