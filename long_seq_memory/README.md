# Long Sequence Memory Experiment

This directory contains the Oxford Spires / LingBot-Map long-sequence memory
experiment. It is intentionally self-contained: new configs, scripts, logs,
caches, and outputs live here, while `../lingbot-map` and
`../oxford_spires_minimal` are treated as read-only inputs.

Note: the user request names `Lingbot-map`, but the directory present in this
workspace is `lingbot-map`. The configs use the existing lowercase path.

## Current Inputs

- LingBot-Map source: `../lingbot-map`
- Checkpoint: `../lingbot-map/checkpoints/lingbot-map.pt`
- Conda env found locally: `lingbot-map`
- Oxford scene: `../oxford_spires_minimal/lingbot_ready_loop/keble-college-02`
- Frame indexing: 0-based everywhere in this experiment
- Strong revisit candidate: frame `3403` to frame `4414`

## Stage Order

1. Static memory audit:
   `python scripts/inspect_lingbot_memory.py --config configs/dataset_keble02.yaml`
2. Dataset validation:
   `python scripts/validate_dataset.py --config configs/dataset_keble02.yaml`
3. 32-frame smoke:
   `conda run -n lingbot-map python scripts/run_reconstruction.py --config configs/smoke_32.yaml`
4. 128-frame smoke:
   `conda run -n lingbot-map python scripts/run_reconstruction.py --config configs/smoke_128.yaml`
5. Small attention extraction smoke:
   `conda run -n lingbot-map python scripts/run_reconstruction.py --config configs/attention_smoke_96.yaml`
6. Focused loop run:
   `conda run -n lingbot-map python scripts/run_loop_experiment.py --config configs/loop_3403_4414.yaml`

Do not run the full 5968-frame config until the focused loop extraction is
stable and the memory index confirms that frame 3403 is retained when frame
4414 is processed.

## Full Direct Streaming Plans

Prepared configs:

- `configs/full_5968_direct_kf5.yaml`
- `configs/full_5968_direct_kf10.yaml`

Planning and resource estimates:

```bash
conda run -n lingbot-map python scripts/estimate_full_run_resources.py \
  --config configs/full_5968_direct_kf5.yaml

conda run -n lingbot-map python scripts/estimate_full_run_resources.py \
  --config configs/full_5968_direct_kf10.yaml
```

Short precheck without model execution:

```bash
conda run -n lingbot-map python scripts/run_long_sequence.py \
  --config configs/full_5968_direct_kf5.yaml \
  --end-frame 1024 \
  --resource-profile-only
```

Short reconstruction validation:

```bash
conda run -n lingbot-map python scripts/run_long_sequence.py \
  --config configs/full_5968_direct_kf5.yaml \
  --end-frame 128
```

Formal full run, after accepting the estimate:

```bash
conda run -n lingbot-map python scripts/run_long_sequence.py \
  --config configs/full_5968_direct_kf5.yaml \
  --confirm-full-run
```

Full analysis must be run only after real reconstruction and interaction
outputs exist:

```bash
conda run -n lingbot-map python scripts/run_long_sequence.py \
  --config configs/full_5968_direct_kf5_online_interactions.yaml \
  --confirm-full-run

conda run -n lingbot-map python scripts/analyze_loop_retrieval.py \
  --config configs/full_5968_direct_kf5.yaml
```

`full_5968_direct_kf5_online_interactions.yaml` replays the kf5 stream and
computes interaction summaries online. It writes Level 1 global summaries,
Level 2 loop-dense summaries, and only saves raw Q/K/V for the four Level 3
debug frames.

## Outputs

All generated artifacts should be written under `outputs/`. Real measurements
and plots must come from actual runs; scripts should fail or mark a field as
unavailable rather than inventing values.

See `docs/run_log.md` for the current completed runs and remaining work.
