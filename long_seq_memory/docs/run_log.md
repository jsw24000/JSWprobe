# Run Log

Date: 2026-07-31

## Completed Gates

### Static Audit

Command:

```bash
python long_seq_memory/scripts/inspect_lingbot_memory.py \
  --config long_seq_memory/configs/dataset_keble02.yaml
```

Output:

- `outputs/static_audit/lingbot_memory_static_audit.json`

Key findings:

- token order: camera, 4 register tokens, scale, image patches;
- `num_special_tokens = 6`;
- patch window default: 64 frames plus 8 scale frames;
- FlashInfer does not work on this Blackwell/CUDA combination, so runtime configs use SDPA fallback;
- normal attention weights are not returned by the model.

### Dataset Validation

Command:

```bash
python long_seq_memory/scripts/validate_dataset.py \
  --config long_seq_memory/configs/dataset_keble02.yaml
```

Output:

- `outputs/dataset_validation/validation.json`
- `outputs/dataset_validation/loop_3403_4414.json`
- `outputs/dataset_validation/trajectory_gt_xy.png`
- `outputs/dataset_validation/trajectory_gt_3d.png`
- `outputs/dataset_validation/frame_3403.jpg`
- `outputs/dataset_validation/frame_4414.jpg`
- `outputs/dataset_validation/frame_pair_comparison.png`

Result:

- validation passed;
- 5968 images;
- 5968 GT poses;
- symlinks valid;
- frame 3403 to 4414 distance: 0.1189915183 m;
- time gap: 50.799432 s;
- relative rotation: 158.492595 deg;
- view-direction cosine: -0.929576.

This is a strong spatial revisit, but the pose-only same-view test is negative. Strict TLS/LiDAR overlap is still not implemented.

### 32-Frame Smoke

Command:

```bash
conda run -n lingbot-map python long_seq_memory/scripts/run_reconstruction.py \
  --config long_seq_memory/configs/smoke_32.yaml
```

Output:

- `outputs/reconstruction/smoke_32/run_manifest.json`
- `outputs/reconstruction/smoke_32/memory_state_summary.jsonl`
- `outputs/reconstruction/smoke_32/pred_poses_c2w_demo_convention.npy`

Result:

- 32 output poses;
- processed image shape: 3x392x518;
- estimated tokens per frame: 1042;
- GPU peak allocated: about 8.54 GB.

### 128-Frame Smoke

Command:

```bash
conda run -n lingbot-map python long_seq_memory/scripts/run_reconstruction.py \
  --config long_seq_memory/configs/smoke_128.yaml
```

Output:

- `outputs/reconstruction/smoke_128/run_manifest.json`
- `outputs/reconstruction/smoke_128/memory_state_summary.jsonl`

Result:

- 128 output poses;
- frame 96 old memory range: 8..32;
- frame 127 old memory range: 8..63;
- confirms transition from live patch window to old special memory.

### Small Attention Extraction

Command:

```bash
conda run -n lingbot-map python long_seq_memory/scripts/run_reconstruction.py \
  --config long_seq_memory/configs/attention_smoke_96.yaml

conda run -n lingbot-map python long_seq_memory/scripts/extract_memory_interactions.py \
  --run_dir long_seq_memory/outputs/reconstruction/attention_smoke_96 \
  --chunk_size 64
```

Output:

- `outputs/reconstruction/attention_smoke_96/qkv/qkv_frame000096_layer23_0000.npz`
- `outputs/reconstruction/attention_smoke_96/interactions/frame000096_layer23_interaction_summary.json`

Captured tensor shapes:

- Q: `[1, 16, 1042, 64]`
- visible K: `[1, 16, 75174, 64]`
- visible V: `[1, 16, 75174, 64]`
- old special memory tokens at frame 96: 150

The Q/K/V file is about 435 MB, so focused-loop Q/K/V capture should be narrowed or aggregated online before enabling many event frames.

### Focused Loop Reconstruction

Command:

```bash
conda run -n lingbot-map python long_seq_memory/scripts/run_loop_experiment.py \
  --config long_seq_memory/configs/loop_3403_4414.yaml
```

Output:

- `outputs/reconstruction/loop_3200_4600/run_manifest.json`
- `outputs/reconstruction/loop_3200_4600/memory_state_summary.jsonl`
- `outputs/reconstruction/loop_3200_4600/pred_poses_c2w_demo_convention.npy`
- `outputs/reconstruction/loop_3200_4600/trajectory_metrics.json`

Result:

- frame range: 3200..4599;
- 1400 output poses;
- GPU peak allocated: about 20.26 GB;
- at current frame 4414:
  - old memory range: 3208..4350;
  - local patch frame range: 4351..4414;
  - frame 3403 is present in old special memory;
  - frame 3403 is not present in the live patch window.

## Not Completed Yet

- Focused-loop Q/K/V extraction around 4414 is not enabled yet, because one full visible K/V capture is large.
- TLS/LiDAR geometric overlap is not implemented yet; current overlap output is marked `pose_candidate_only`.
- Final loop retrieval analysis figures are not generated yet.

## 2026-08-01 Full-Sequence Planning Update

Added:

- `configs/full_5968_direct_kf5.yaml`
- `configs/full_5968_direct_kf10.yaml`
- keyframe dry-run scheduler;
- resource estimator;
- full sequence runner with `--resource-profile-only`;
- feature extraction planner;
- real, input-driven `analyze_loop_retrieval.py` that writes incomplete status instead of fake results.

Planning outputs:

- `outputs/planning/full_5968_direct_kf5/keyframe_schedule.parquet`
- `outputs/planning/full_5968_direct_kf5/resource_estimate.json`
- `outputs/planning/full_5968_direct_kf5/feature_extraction_plan.json`
- corresponding kf10 files under `outputs/planning/full_5968_direct_kf10/`

kf5 dry-run result:

- total memory writes: 1200;
- frame 3403 is written to long-term KV;
- frame 4414 is not a keyframe;
- retained history keyframes in `[3370, 3440]`: 3373, 3378, 3383, 3388, 3393, 3398, 3403, 3408, 3413, 3418, 3423, 3428, 3433, 3438.

kf10 dry-run result:

- total memory writes: 604;
- frame 3403 is not written;
- frame 4414 is not a keyframe;
- retained history keyframes in `[3370, 3440]`: 3378, 3388, 3398, 3408, 3418, 3428, 3438.

Short 0-128 validation:

```bash
conda run -n lingbot-map python scripts/run_long_sequence.py \
  --config configs/full_5968_direct_kf5.yaml \
  --end-frame 128
```

Output:

- `outputs/full_5968_direct_kf5/reconstruction_0_128/run_manifest.json`
- `outputs/full_5968_direct_kf5/reconstruction_0_128/keyframe_schedule.parquet`
- `outputs/full_5968_direct_kf5/reconstruction_0_128/memory_state_summary.jsonl`
- pose/GT/frame ID arrays;
- stride-saved dense outputs.

Result:

- 128 predicted poses;
- total memory writes: 32;
- SDPA skip-append patch installed and marked effective;
- depth frames saved at 0, 10, ..., 120;
- world-point frames saved at 0, 20, ..., 120.

## 2026-08-01 Full kf5 Reconstruction

Command:

```bash
conda run -n lingbot-map python scripts/run_long_sequence.py \
  --config configs/full_5968_direct_kf5.yaml \
  --confirm-full-run
```

Output:

- `outputs/full_5968_direct_kf5/reconstruction/run_manifest.json`
- `outputs/full_5968_direct_kf5/reconstruction/frame_ids.npy`
- `outputs/full_5968_direct_kf5/reconstruction/gt_poses_c2w.npy`
- `outputs/full_5968_direct_kf5/reconstruction/pred_poses_c2w_demo_convention.npy`
- `outputs/full_5968_direct_kf5/reconstruction/pred_intrinsics.npy`
- `outputs/full_5968_direct_kf5/reconstruction/per_frame_time_s.npy`
- `outputs/full_5968_direct_kf5/reconstruction/memory_state_summary.jsonl`
- `outputs/full_5968_direct_kf5/reconstruction/keyframe_schedule.parquet`
- stride-saved dense outputs under `dense/`
- raw per-frame trajectory metrics under `trajectory_metrics.json`

Result:

- 5968 input frames, source frame IDs 0..5967;
- 5968 predicted poses and 5968 GT poses saved;
- keyframe interval: 5;
- total long-term memory writes: 1200;
- SDPA skip-append patch installed and effective;
- no reset detected by manifest;
- at frame 4414, frame 3403 is present in old special memory and not in the live patch window;
- reconstruction output size: about 1.9 GB;
- mean per-frame measured model time: about 0.167 s, median about 0.160 s;
- depth saved every 10 frames, world points every 20 frames.

The startup message `Failed to get device capability: SM 12.x requires CUDA >= 12.9` was observed again. It is the FlashInfer capability probe failing on the Blackwell/CUDA stack; this run used SDPA fallback and completed.

Raw trajectory metrics:

- translation mean: 52.52 m, median: 48.16 m;
- rotation mean: 123.92 deg, median: 122.01 deg.

These metrics are raw GT-vs-pred values without Sim(3), scale, or convention alignment and should not be reported as final ATE.

## 2026-08-01 Online Interaction Implementation

Added:

- `configs/full_5968_direct_kf5_online_interactions.yaml`;
- `src/long_seq_memory/online_interactions.py`;
- online aggregation wiring in `lingbot_adapter.py`.

The intended interaction pipeline is now:

```text
capture visible-memory Q/K/V in target layer
compute full-context attention denominator in memory
slice old trajectory-memory tokens
aggregate by source frame, memory token type, and query type
write Level 1 / Level 2 summaries
discard raw Q/K/V unless Level 3 raw-debug criteria match
```

Configured outputs:

- Level 1: `outputs/full_5968_direct_kf5/interaction/global_summary/global_summary.jsonl`
- Level 2: `outputs/full_5968_direct_kf5/interaction/loop_dense/loop_dense.jsonl`
- Level 3 patch summaries: `outputs/full_5968_direct_kf5/interaction/patch_specific/patch_specific.jsonl`
- Level 3 raw Q/K/V: `outputs/full_5968_direct_kf5/interaction/raw_qkv_debug/qkv/`
- Level 3 token metadata: `outputs/full_5968_direct_kf5/interaction/raw_qkv_debug/token_metadata/`

Level 3 raw frames:

- control frame: 2755;
- loop/debug frames: 4400, 4414, 4430;
- layers: 4, 11, 17, 23.

Planning check:

- Level 1 frames: 597 sampled current frames;
- Level 2 current window: 4380..4445, 66 frames;
- retained target history keyframes: 3373, 3378, 3383, 3388, 3393, 3398, 3403, 3408, 3413, 3418, 3423, 3428, 3433, 3438;
- estimated total output: about 5.11 GB;
- raw Q/K/V estimate: about 4.98 GB, within the 8 GB raw budget.

Command:

```bash
conda run -n lingbot-map python scripts/run_long_sequence.py \
  --config configs/full_5968_direct_kf5_online_interactions.yaml \
  --confirm-full-run
```

This interaction replay has not been run yet.
