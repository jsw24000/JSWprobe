from long_seq_memory.memory_index import (
    TokenLayout,
    memory_state_summary,
    old_special_frames,
    retained_patch_frames,
)


def test_token_layout_defaults():
    layout = TokenLayout(num_register_tokens=4, has_scale_token=True, patch_count=10)
    assert layout.num_special_tokens == 6
    assert layout.patch_start_idx == 6
    assert layout.token_type(0) == ("camera", 0)
    assert layout.token_type(4) == ("register", 3)
    assert layout.token_type(5) == ("scale", 0)
    assert layout.token_type(6) == ("image_patch", 0)


def test_memory_state_old_frame():
    frames = list(range(100))
    assert retained_patch_frames(frames, scale_frames=8, sliding_window=64)[0] == 0
    assert 20 in old_special_frames(frames, scale_frames=8, sliding_window=64)
    summary = memory_state_summary(99, frames, target_frame=20, scale_frames=8, sliding_window=64)
    assert summary["target_frame_in_old_special_memory"] is True
