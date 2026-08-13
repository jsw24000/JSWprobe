from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from .utils import normalize_layer_name


DEFAULT_EXTRACT_LAYERS = (4, 11, 17, 23)
SLOT_NAMES = ["camera", "register0", "register1", "register2", "register3", "scale"]
SLOT_INDICES = {
    "camera": [0],
    "register": [1, 2, 3, 4],
    "scale": [5],
    "slot0": [0],
    "slot1": [1],
    "slot2": [2],
    "slot3": [3],
    "slot4": [4],
    "slot5": [5],
}


class MemoryTokenRecorder:
    """Record per-frame six-token outputs from GCTStream._aggregate_features.

    GCTStream._aggregate_features returns selected aggregator block outputs as
    tensors shaped [B, S, P, D].  In the streaming model, the first P slots are
    special tokens and patch_start_idx == 6, so slicing [:patch_start_idx]
    returns camera + register0..3 + scale tokens for each frame.
    """

    def __init__(
        self,
        model: Any,
        layer_indices: Sequence[int] = DEFAULT_EXTRACT_LAYERS,
    ):
        self.model = model
        self.layer_indices = [int(layer) for layer in layer_indices]
        self.records: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.call_log: List[Dict[str, Any]] = []
        self.patch_start_idx: Optional[int] = None
        self._original = None

    def __enter__(self) -> "MemoryTokenRecorder":
        self._original = self.model._aggregate_features

        def wrapped_aggregate_features(*args, **kwargs):
            before_total = int(getattr(self.model.aggregator, "total_frames_processed", 0))
            outputs, patch_start_idx = self._original(*args, **kwargs)
            self.patch_start_idx = int(patch_start_idx)
            if self.patch_start_idx != 6:
                raise RuntimeError(f"Expected 6 special tokens, got patch_start_idx={self.patch_start_idx}")

            if outputs:
                _, seq_len, _, _ = outputs[0].shape
                frame_ids = [before_total + i for i in range(seq_len)]
                for layer_index, tensor in zip(self.layer_indices, outputs):
                    layer_name = normalize_layer_name(layer_index)
                    detached = tensor.detach().float().cpu()
                    for local_idx, frame_id in enumerate(frame_ids):
                        self.records[int(frame_id)][layer_name] = detached[0, local_idx, : self.patch_start_idx, :].contiguous()

                self.call_log.append(
                    {
                        "before_total_frames_processed": before_total,
                        "seq_len": int(seq_len),
                        "frame_ids": frame_ids,
                        "patch_start_idx": int(patch_start_idx),
                        "layer_names": [normalize_layer_name(i) for i in self.layer_indices[: len(outputs)]],
                        "shapes": {normalize_layer_name(i): list(t.shape) for i, t in zip(self.layer_indices, outputs)},
                    }
                )
            return outputs, patch_start_idx

        self.model._aggregate_features = wrapped_aggregate_features
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original is not None:
            self.model._aggregate_features = self._original

    def get_frame_tokens(self, frame_id: int) -> Dict[str, torch.Tensor]:
        return dict(self.records.get(int(frame_id), {}))

    def available_layers(self) -> List[str]:
        seen = set()
        layers: List[str] = []
        for record in self.records.values():
            for layer in record:
                if layer not in seen:
                    layers.append(layer)
                    seen.add(layer)
        return layers
