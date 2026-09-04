# Long Sequence Memory Current-Frame Causal Interventions

This protocol tests the path:

```text
old memory -> special tokens -> image tokens
```

The first formal debug target is current frame `4414`. All frames before `4414`
run normally. The causal branch starts only when frame `4414` enters the
forward pass, so the pre-4414 old trajectory memory, local-window tokens, and
rolling state should match the baseline run under deterministic eval.

Do not use the older full-sequence intervention configs as the main entry point
for this causal question. They answer a different question: what happens after a
connection is disrupted for a long prefix.

## Dataset Gate

Run this before any formal reconstruction:

```bash
cd /home/3dsm/Desktop/JSWprobe/long_seq_memory
python scripts/check_causal_intervention_paths.py
```

The expected processed dataset root is:

```text
/home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/lingbot_ready_loop/keble-college-02
```

If `dataset_ready` is false, recreate that directory from the single Oxford
Spires sequence before running baseline or interventions. The one-sequence
download pattern is recorded at:

```text
/home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/dataset_download_one_sequence.yaml
```

Use the local HuggingFace downloader wrapper when the files are missing:

```bash
cd /home/3dsm/Desktop/JSWprobe/long_seq_memory
python scripts/download_oxford_minimal.py
```

If the local cache should be checked without network, use:

```bash
python scripts/download_oxford_minimal.py --dry-run
```

Then run the existing preprocess entry point:

```bash
python /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/scripts/lingbot_oxford_preprocess.py \
  --dataset_dir /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/downloads \
  --output_dir /home/3dsm/Desktop/JSWprobe/oxford_spires_minimal/lingbot_ready_loop \
  --sequence 2024-03-12-keble-college-02 \
  --max_frames 4415 \
  --images_only
```

This fast command is enough for reconstruction-ready RGB, intrinsics, poses,
and timestamp sync. Remove `--images_only` only if you also want the preprocess
step to generate TLS-derived depth maps and `ground_truth.ply`; that path is
much heavier. The raw LiDAR/TLS payload remains in the minimal downloads folder
either way.

## Branch-4414 Runs

All branch runs use the same Oxford sequence, checkpoint, SDPA path,
keyframe-only long-term KV writes, and full input stride `1`. They process
frames `0..4414` and save poses for every processed frame. Depth/world-points
and representation `X_l` are saved only for diagnostic frames `3403` and
`4414`.

Baseline:

```bash
python scripts/run_long_sequence.py \
  --config configs/causal_branch_4414_baseline.yaml \
  --allow-large-output \
  --confirm-full-run
```

A, cut early special-query access to old trajectory memory only at frame 4414:

```bash
python scripts/run_long_sequence.py \
  --config configs/causal_branch_4414_A_special_old_l4_8.yaml \
  --allow-large-output \
  --confirm-full-run
```

B-all, cut image-query access to old trajectory memory across all configured
memory-attention layers at frame 4414:

```bash
python scripts/run_long_sequence.py \
  --config configs/causal_branch_4414_Ball_image_old_l4_23.yaml \
  --allow-large-output \
  --confirm-full-run
```

B-late, cut only the late image-query access to old trajectory memory at frame
4414:

```bash
python scripts/run_long_sequence.py \
  --config configs/causal_branch_4414_Blate_image_old_l17_23.yaml \
  --allow-large-output \
  --confirm-full-run
```

C, cut the second hop from live special tokens into image tokens at frame 4414:

```bash
python scripts/run_long_sequence.py \
  --config configs/causal_branch_4414_C_image_live_special_l8_23.yaml \
  --allow-large-output \
  --confirm-full-run
```

## Intervention Definitions

A masks only:

```text
camera/register/scale query -> old trajectory-memory K/V
frame 4414
layers 4-8
```

All other communication remains available, including special -> current image,
special -> local window, image -> old memory, image -> special, and ordinary
live-memory attention.

B-all masks only:

```text
image patch query -> old trajectory-memory K/V
frame 4414
layers 4-23
```

B-late masks only:

```text
image patch query -> old trajectory-memory K/V
frame 4414
layers 17-23
```

C masks only:

```text
image patch query -> live retained/current camera/register/scale K/V
frame 4414
layers 8-23
```

C deliberately starts after the early/middle layers where special tokens can
read old memory. It does not mask old trajectory-memory K/V directly, so it is
separate from B. The interpretation has a known confound: live special tokens
contain history, current-frame, local-window, and trajectory information. Treat
C as a first causal screening result, not as a final decomposition of the
history-only component.

## State-Contamination Check

Before launching a branch run, inspect:

```text
outputs/causal_branch_4414/path_check/causal_path_check.md
```

For branch-at-current-frame experiments, selected intervention frames should be
non-memory-write frames when later frames in the same run will be analyzed. With
the current `start_frame=0`, `num_scale_frames=8`, and `keyframe_interval=5`,
frame `4414` is a non-memory-write frame.

If you later run multiple selected current frames in a single stream, keep this
check. A selected frame that writes modified KV can contaminate later branch
states.

## Post-Run Analysis

After baseline and any available interventions finish:

```bash
python scripts/analyze_causal_suite.py \
  --suite-root /home/3dsm/Desktop/JSWprobe/long_seq_memory/outputs/causal_branch_4414 \
  --config configs/causal_branch_4414_baseline.yaml
```

The summary is written to:

```text
/home/3dsm/Desktop/JSWprobe/long_seq_memory/outputs/causal_branch_4414/analysis/causal_suite_summary.json
```

The analysis compares:

- intervention-vs-baseline camera pose deltas;
- raw GT-vs-pred translation error deltas;
- selected-frame depth and world-point deltas when saved;
- token representation `Delta X_l` by layer and token group;
- online old-memory mass summaries when interaction outputs exist.
