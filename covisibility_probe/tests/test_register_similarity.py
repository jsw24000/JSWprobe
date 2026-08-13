from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from covisibility_probe.register_similarity import (
    register_mean_descriptors,
    register_mean_similarity_matrix,
    register_slot_similarity_matrix,
)


def test_identical_register_slot_similarity_is_one() -> None:
    rng = np.random.default_rng(0)
    regs = rng.normal(size=(3, 4, 8)).astype(np.float32)
    sim = register_slot_similarity_matrix(regs)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)


def test_slotwise_similarity_range_reasonable() -> None:
    rng = np.random.default_rng(1)
    regs = rng.normal(size=(5, 4, 16)).astype(np.float32)
    sim = register_slot_similarity_matrix(regs)
    assert np.nanmin(sim) >= -1.0 - 1e-5
    assert np.nanmax(sim) <= 1.0 + 1e-5


def test_mean_pooling_descriptor_dim() -> None:
    regs = np.ones((7, 4, 12), dtype=np.float32)
    desc = register_mean_descriptors(regs)
    assert desc.shape == (7, 12)
    sim = register_mean_similarity_matrix(regs)
    assert sim.shape == (7, 7)


def test_float16_save_read_consistency(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    regs = torch.from_numpy(rng.normal(size=(4, 4, 8)).astype(np.float32)).half()
    path = tmp_path / "features.pt"
    torch.save({"regs": regs}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)["regs"].float().numpy()
    assert np.allclose(loaded, regs.float().numpy(), atol=1e-3)

