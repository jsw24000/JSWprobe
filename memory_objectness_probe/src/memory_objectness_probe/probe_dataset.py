from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .utils import normalize_layer_name, read_jsonl, torch_load_cpu


TASK_NUM_CLASSES = {
    "state5": 5,
    "presence": 2,
    "position4": 4,
}


def label_for_task(payload: Mapping[str, Any], task: str) -> Optional[int]:
    pos = int(payload["target_position_id"])
    if task == "state5":
        return pos
    if task == "presence":
        return 0 if pos == 0 else 1
    if task == "position4":
        if pos == 0:
            return None
        return pos - 1
    raise ValueError(f"Unknown task: {task}")


def tensor_for_mode(
    tokens: torch.Tensor,
    *,
    feature_mode: str,
    slot_index: Optional[int] = None,
    slot_group: Optional[str] = None,
    slot_indices: Optional[Mapping[str, Sequence[int]]] = None,
) -> torch.Tensor:
    if feature_mode == "all6_flatten":
        return tokens.reshape(-1)
    if feature_mode == "all6_mean":
        return tokens.mean(dim=0)
    if feature_mode == "single_slot":
        if slot_index is None:
            raise ValueError("single_slot requires --slot_index")
        return tokens[int(slot_index)]
    if feature_mode == "slot_group":
        if not slot_group:
            raise ValueError("slot_group requires --slot_group")
        slot_indices = slot_indices or {}
        if slot_group in slot_indices:
            indices = list(slot_indices[slot_group])
        elif slot_group.startswith("slot") and slot_group[4:].isdigit():
            indices = [int(slot_group[4:])]
        else:
            indices = [int(part) for part in slot_group.split(",")]
        return tokens[indices].mean(dim=0)
    raise ValueError(f"Unknown feature mode: {feature_mode}")


@dataclass
class ProbeExample:
    sample_id: str
    base_scene_id: str
    variant_id: str
    feature_path: Path
    label: int


class ProbeFeatureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        feature_manifest: str | Path,
        sample_ids: Sequence[str],
        *,
        task: str,
        layer: str | int,
        feature_mode: str,
        slot_index: Optional[int] = None,
        slot_group: Optional[str] = None,
    ):
        rows = read_jsonl(feature_manifest)
        wanted = set(sample_ids)
        self.layer = normalize_layer_name(layer)
        self.task = task
        self.feature_mode = feature_mode
        self.slot_index = slot_index
        self.slot_group = slot_group
        self.examples: List[ProbeExample] = []
        self._cache: Dict[str, torch.Tensor] = {}

        for row in rows:
            if row.get("extraction_status") != "success":
                continue
            sample_id = str(row["sample_id"])
            if sample_id not in wanted:
                continue
            payload = torch_load_cpu(row["feature_path"])
            label = label_for_task(payload, task)
            if label is None:
                continue
            if self.layer not in payload["memory_tokens"]:
                continue
            self.examples.append(
                ProbeExample(
                    sample_id=sample_id,
                    base_scene_id=str(row["base_scene_id"]),
                    variant_id=str(row["variant_id"]),
                    feature_path=Path(row["feature_path"]),
                    label=int(label),
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        example = self.examples[idx]
        payload = torch_load_cpu(example.feature_path)
        tokens = payload["memory_tokens"][self.layer].float()
        x = tensor_for_mode(
            tokens,
            feature_mode=self.feature_mode,
            slot_index=self.slot_index,
            slot_group=self.slot_group,
            slot_indices=payload.get("slot_indices", {}),
        )
        y = torch.tensor(example.label, dtype=torch.long)
        return x, y, example.sample_id

    def stack(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        xs, ys, ids = [], [], []
        for i in range(len(self)):
            x, y, sample_id = self[i]
            xs.append(x)
            ys.append(y)
            ids.append(sample_id)
        if not xs:
            return torch.empty(0), torch.empty(0, dtype=torch.long), []
        return torch.stack(xs), torch.stack(ys), ids


def split_sample_ids(split_data: Mapping[str, Any], split: str) -> List[str]:
    value = split_data[split]
    if isinstance(value, dict):
        return list(value.get("sample_ids", []))
    return list(value)
