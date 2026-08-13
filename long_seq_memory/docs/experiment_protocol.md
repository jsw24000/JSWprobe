# Experiment Protocol

Frame indices are 0-based throughout.

## Gate 1: Static Audit

Run:

```bash
python long_seq_memory/scripts/inspect_lingbot_memory.py \
  --config long_seq_memory/configs/dataset_keble02.yaml
```

Expected output:

- `outputs/static_audit/lingbot_memory_static_audit.json`
- updated notes in `docs/lingbot_memory_architecture.md`

Do not proceed if the source path or checkpoint is missing.

## Gate 2: Dataset Validation

Run:

```bash
python long_seq_memory/scripts/validate_dataset.py \
  --config long_seq_memory/configs/dataset_keble02.yaml
```

Expected output:

- `outputs/dataset_validation/validation.json`
- `outputs/dataset_validation/loop_3403_4414.json`
- GT trajectory plots and frame comparison images

Do not proceed if the loop pair distance or frame/pose alignment fails.

## Gate 3: 32-Frame Smoke

Run:

```bash
conda run -n lingbot-map python long_seq_memory/scripts/run_reconstruction.py \
  --config long_seq_memory/configs/smoke_32.yaml
```

Check:

- model loads;
- output pose count is 32;
- GT frame IDs match selected image paths;
- `memory_state_summary.jsonl` is written.

## Gate 4: 128-Frame Smoke

Run only after Gate 3:

```bash
conda run -n lingbot-map python long_seq_memory/scripts/run_reconstruction.py \
  --config long_seq_memory/configs/smoke_128.yaml
```

Check:

- sequence crosses the 64-frame local patch window;
- old special memory grows;
- source-frame mapping remains monotonic;
- selected Q/K/V output size is manageable.

## Gate 5: Focused Loop

## Gate 5: Small Attention Extraction

Run only after Gate 4:

```bash
conda run -n lingbot-map python long_seq_memory/scripts/run_reconstruction.py \
  --config long_seq_memory/configs/attention_smoke_96.yaml

conda run -n lingbot-map python long_seq_memory/scripts/extract_memory_interactions.py \
  --run_dir long_seq_memory/outputs/reconstruction/attention_smoke_96 \
  --chunk_size 64
```

Check:

- one selected Q/K/V capture exists;
- frame/layer match the config;
- old special memory token count is nonzero;
- interaction summary is generated from real Q/K/V.

## Gate 6: Focused Loop

Run only after Gate 5:

```bash
conda run -n lingbot-map python long_seq_memory/scripts/run_loop_experiment.py \
  --config long_seq_memory/configs/loop_3403_4414.yaml
```

Required checks before analysis:

- frame 3403 and frame 4414 are in one uninterrupted streaming run;
- frame 3403 appears in old special memory when frame 4414 is processed;
- frame 3403 is outside the live patch window at frame 4414;
- no state reset occurred between these frames.

## Full Run

`configs/full_5968.yaml` is prepared but marked as `prepared_only_do_not_run_before_loop_smoke`.
Do not run it until focused-loop Q/K/V extraction and memory indexing are stable.
