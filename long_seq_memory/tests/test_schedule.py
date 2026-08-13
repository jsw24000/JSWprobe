from long_seq_memory.schedule import (
    ScheduleConfig,
    build_keyframe_schedule,
    old_special_frames_at,
    retained_full_frames_at,
    schedule_report,
)


def test_kf5_loop_frames_and_segment_writes():
    cfg = ScheduleConfig(
        start_frame=0,
        end_frame=5968,
        input_stride=1,
        num_scale_frames=8,
        keyframe_interval=5,
        sliding_window=64,
    )
    rows = build_keyframe_schedule(cfg)
    report = schedule_report(rows, cfg, history_frame=3403, current_frame=4414, history_segment=(3370, 3440))

    assert report["total_memory_writes"] == 1200
    assert report["history_frame_written"] is True
    assert report["current_frame_written"] is False
    assert 3403 in old_special_frames_at(rows, 4414, 8, 64)
    assert 3403 not in retained_full_frames_at(rows, 4414, 8, 64)
    assert report["written_keyframes_in_history_segment"] == [
        3373,
        3378,
        3383,
        3388,
        3393,
        3398,
        3403,
        3408,
        3413,
        3418,
        3423,
        3428,
        3433,
        3438,
    ]


def test_kf10_does_not_write_3403():
    cfg = ScheduleConfig(
        start_frame=0,
        end_frame=5968,
        input_stride=1,
        num_scale_frames=8,
        keyframe_interval=10,
        sliding_window=64,
    )
    rows = build_keyframe_schedule(cfg)
    report = schedule_report(rows, cfg, history_frame=3403, current_frame=4414, history_segment=(3370, 3440))

    assert report["total_memory_writes"] == 604
    assert report["history_frame_written"] is False
    assert report["current_frame_written"] is False
    assert 3403 not in report["retained_memory_frames_in_history_segment_at_current"]
    assert report["written_keyframes_in_history_segment"] == [3378, 3388, 3398, 3408, 3418, 3428, 3438]
