from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covisibility_probe.lingbot_adapter import split_aggregator_head_outputs


def test_split_aggregator_head_outputs_shapes_and_halves() -> None:
    tensor = torch.arange(1 * 2 * 10 * 16, dtype=torch.float32).reshape(1, 2, 10, 16)
    out = split_aggregator_head_outputs([tensor], [4], patch_start_idx=6, dtype=torch.float16)

    reg_concat = out["register_head_concat"][4]
    reg_frame = out["register_head_frame"][4]
    reg_global = out["register_head_global"][4]
    cam_concat = out["camera_head_concat"][4]

    assert reg_concat.shape == (2, 4, 16)
    assert reg_frame.shape == (2, 4, 8)
    assert reg_global.shape == (2, 4, 8)
    assert cam_concat.shape == (2, 16)
    assert out["camera_head_frame"][4].shape == (2, 8)
    assert out["camera_head_global"][4].shape == (2, 8)

    assert torch.equal(reg_frame.float(), reg_concat[..., :8].float())
    assert torch.equal(reg_global.float(), reg_concat[..., 8:].float())
    assert torch.equal(cam_concat.float(), tensor[0, :, 0, :].float())
    assert reg_concat.dtype == torch.float16


def test_split_aggregator_head_outputs_requires_four_registers() -> None:
    tensor = torch.zeros(1, 1, 5, 16)
    try:
        split_aggregator_head_outputs([tensor], [4], patch_start_idx=5)
    except ValueError as exc:
        assert "at least 6 special tokens" in str(exc)
    else:
        raise AssertionError("Expected invalid special-token layout to raise ValueError")


def test_split_aggregator_head_outputs_clones_slices() -> None:
    tensor = torch.randn(1, 1, 10, 16)
    out = split_aggregator_head_outputs([tensor], [4], patch_start_idx=6, dtype=torch.float32)
    assert out["register_head_concat"][4].data_ptr() != tensor.data_ptr()
    assert out["register_head_frame"][4].data_ptr() != out["register_head_concat"][4].data_ptr()
