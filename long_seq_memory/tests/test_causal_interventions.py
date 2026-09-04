import pytest

from long_seq_memory.interaction_hooks import (
    causal_intervention_from_config,
    representation_layers_from_config,
)


def test_causal_intervention_config_parser():
    cfg = {
        "memory_policy": {"patch_start_idx": 6},
        "causal_intervention": {
            "enabled": True,
            "name": "A",
            "mode": "special_to_old_memory",
            "layers": [4, 5, 6, 7, 8],
            "frames": [4414],
        },
    }
    parsed = causal_intervention_from_config(cfg)
    assert parsed is not None
    assert parsed.name == "A"
    assert parsed.mode == "special_to_old_memory"
    assert parsed.layers == {4, 5, 6, 7, 8}
    assert parsed.frame_ids == {4414}
    assert parsed.num_special_tokens == 6


def test_causal_intervention_applies_only_to_selected_current_frame():
    cfg = {
        "causal_intervention": {
            "enabled": True,
            "mode": "image_to_old_memory",
            "layers": [17, 18],
            "frames": [4414],
        }
    }
    parsed = causal_intervention_from_config(cfg)
    assert parsed is not None
    parsed.current_frame_id = 4413
    assert parsed.applies_to_current_frame() is False
    parsed.current_frame_id = 4414
    assert parsed.applies_to_current_frame() is True


def test_causal_intervention_all_frames_escape_hatch_is_explicit():
    cfg = {
        "causal_intervention": {
            "enabled": True,
            "mode": "image_to_old_memory",
            "layers": [17, 18],
            "frames": "all",
        }
    }
    parsed = causal_intervention_from_config(cfg)
    assert parsed is not None
    assert parsed.frame_ids is None
    parsed.current_frame_id = 1
    assert parsed.applies_to_current_frame() is True


def test_representation_layers_all_means_every_layer():
    assert representation_layers_from_config("all", 24) is None
    assert representation_layers_from_config([4, 11, 99], 24) == {4, 11}


def test_bad_causal_intervention_mode_raises():
    cfg = {
        "causal_intervention": {
            "enabled": True,
            "mode": "too_much_magic",
            "layers": [0],
        }
    }
    with pytest.raises(ValueError):
        causal_intervention_from_config(cfg)


def test_causal_intervention_requires_explicit_frames():
    cfg = {
        "causal_intervention": {
            "enabled": True,
            "mode": "image_to_old_memory",
            "layers": [17, 18],
        }
    }
    with pytest.raises(ValueError):
        causal_intervention_from_config(cfg)
