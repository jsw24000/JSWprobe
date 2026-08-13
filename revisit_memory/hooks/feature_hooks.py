from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CapturedFeatureBatch:
    selected_layers: list[int]
    patch_start_idx: int
    token_grid_hw: tuple[int, int]
    frame_image_tokens: dict[int, torch.Tensor] = field(default_factory=dict)
    global_image_tokens: dict[int, torch.Tensor] = field(default_factory=dict)
    frame_special_tokens: dict[int, torch.Tensor] = field(default_factory=dict)
    global_special_tokens: dict[int, torch.Tensor] = field(default_factory=dict)
    raw_shapes: dict[int, tuple[int, ...]] = field(default_factory=dict)


class AggregateFeatureCapture:
    """Monkey-patch wrapper for `model._aggregate_features`.

    LingBot-Map's aggregator returns selected block-group outputs as
    `[B, S, P, 2C]`, where the final dimension is
    `[frame_block_output, global_block_output]`. This wrapper stores a detached
    copy after each model forward without changing model math.
    """

    def __init__(self, model: Any, selected_layers: list[int], save_dtype: torch.dtype = torch.float16):
        self.model = model
        self.selected_layers = selected_layers
        self.save_dtype = save_dtype
        self.original = None
        self.last: CapturedFeatureBatch | None = None

    def __enter__(self):
        self.original = self.model._aggregate_features

        def wrapped(*args, **kwargs):
            outputs, patch_start_idx = self.original(*args, **kwargs)
            if len(outputs) != len(self.selected_layers):
                raise RuntimeError(
                    f"Expected {len(self.selected_layers)} selected aggregator outputs for layers "
                    f"{self.selected_layers}, got {len(outputs)}."
                )
            images = args[0] if args else kwargs.get("images")
            h = int(images.shape[-2])
            w = int(images.shape[-1])
            patch = int(getattr(self.model, "patch_size", 14))
            batch = CapturedFeatureBatch(
                selected_layers=list(self.selected_layers),
                patch_start_idx=int(patch_start_idx),
                token_grid_hw=(h // patch, w // patch),
            )
            for layer_id, tensor in zip(self.selected_layers, outputs):
                if tensor.shape[-1] % 2 != 0:
                    raise RuntimeError(
                        f"Selected layer {layer_id} returned an odd channel count {tensor.shape[-1]}; "
                        "cannot split concat(frame_block, global_block)."
                    )
                split = tensor.shape[-1] // 2
                frame = tensor[..., :split].detach().to("cpu", dtype=self.save_dtype)
                global_ = tensor[..., split:].detach().to("cpu", dtype=self.save_dtype)
                batch.frame_special_tokens[layer_id] = frame[..., :patch_start_idx, :].contiguous()
                batch.global_special_tokens[layer_id] = global_[..., :patch_start_idx, :].contiguous()
                batch.frame_image_tokens[layer_id] = frame[..., patch_start_idx:, :].contiguous()
                batch.global_image_tokens[layer_id] = global_[..., patch_start_idx:, :].contiguous()
                batch.raw_shapes[layer_id] = tuple(tensor.shape)
            self.last = batch
            return outputs, patch_start_idx

        self.model._aggregate_features = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original is not None:
            self.model._aggregate_features = self.original
        return False


def token_type_names(num_register_tokens: int = 4) -> list[str]:
    return ["camera"] + [f"register_{idx}" for idx in range(num_register_tokens)] + ["scale"]
