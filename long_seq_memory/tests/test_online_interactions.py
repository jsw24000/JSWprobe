import pytest

torch = pytest.importorskip("torch")

from long_seq_memory.interaction_hooks import QKVRecord
from long_seq_memory.online_interactions import compact_summary, compute_interaction_summary


def test_online_summary_dense_and_compact():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 20, 4)
    v = torch.randn(1, 2, 20, 4)
    row = {
        "current_frame": 20,
        "old_memory_frames": [5, 10],
        "old_memory_frame_range": [5, 10],
        "patch_memory_frames": [0, 15, 20],
    }
    record = QKVRecord(frame_id=20, layer_id=1, q=q, k=k, v=v, capture_kind="visible_memory_qkv")

    summary = compute_interaction_summary(record, row, dense_by_source=True)
    all_queries = summary["aggregation"]["all_queries"]

    assert summary["old_memory_token_count"] == 12
    assert all_queries["by_source_frame"]["source_frames"] == [5, 10]
    assert len(all_queries["camera"]["by_source_frame"]["attention_mass_by_head_frame"]) == 2
    assert all_queries["normalization"]["attention_mass"] == "full_context_softmax_denominator"

    compact = compact_summary(summary)
    assert "by_source_frame" not in compact["aggregation"]["all_queries"]
    assert "by_source_frame" not in compact["aggregation"]["all_queries"]["camera"]
