# Memory Objectness Probe

Independent experiment for probing whether Lingbot-map compact trajectory
memory encodes target presence and coarse target position at frame 15.

## Paths On This Machine

From this directory:

- Dataset: `../memory_scene_blender/outputs/memory_write_probe_v1`
- Lingbot-map source: `../lingbot-map`
- Checkpoint: `../lingbot-map/checkpoints/lingbot-map.pt`
- Conda env: `lingbot-map`

The scripts do not modify Lingbot-map. The extractor inserts `--lingbot_root`
at the front of `sys.path` so it uses the requested local source tree.

## Data Check

```bash
cd /home/3dsm/Desktop/JSWprobe/memory_objectness_probe

conda run -n lingbot-map python scripts/inspect_dataset.py \
  --data_root ../memory_scene_blender/outputs/memory_write_probe_v1 \
  --output_dir outputs/dataset_check
```

Outputs:

```text
outputs/dataset_check/dataset_report.json
outputs/dataset_check/validated_manifest.jsonl
```

## Smoke Extraction

```bash
conda run -n lingbot-map python scripts/extract_memory_features.py \
  --data_root ../memory_scene_blender/outputs/memory_write_probe_v1 \
  --lingbot_root ../lingbot-map \
  --output_root outputs \
  --base_scene_ids base_scene_000 \
  --save_reconstruction minimal \
  --overwrite
```

Expected result: five feature files under `outputs/features/samples/`, one for
each of `absent`, `pos1`, `pos2`, `pos3`, and `pos4`.

## Full Extraction

```bash
conda run -n lingbot-map python scripts/extract_memory_features.py \
  --data_root ../memory_scene_blender/outputs/memory_write_probe_v1 \
  --lingbot_root ../lingbot-map \
  --output_root outputs \
  --save_reconstruction minimal \
  --resume
```

Feature outputs:

```text
outputs/features/samples/<sample_id>.pt
outputs/features/feature_manifest.jsonl
outputs/features/feature_manifest.csv
outputs/features/extraction_summary.json
outputs/reconstructions/<sample_id>/run_metadata.json
```

Each `.pt` payload stores raw CPU float32 tensors:

```python
memory_tokens["block17"].shape == [6, 2048]
```

The six slot order is:

```text
camera, register0, register1, register2, register3, scale
```

## Build Split

```bash
conda run -n lingbot-map python scripts/build_splits.py \
  --feature_manifest outputs/features/feature_manifest.jsonl \
  --output_file outputs/splits/scene_split_seed0.json \
  --seed 0 \
  --train_scenes 20 \
  --val_scenes 5 \
  --test_scenes 5
```

Splits are grouped by `base_scene_id`; the five variants of a base scene never
cross split boundaries.

## Probe Smoke Test

Use a real available layer. Current extractor names layers as `block04`,
`block11`, `block17`, and `block23`; numeric aliases like `17` also work.

```bash
conda run -n lingbot-map python scripts/train_probe.py \
  --feature_manifest outputs/features/feature_manifest.jsonl \
  --split_file outputs/splits/scene_split_seed0.json \
  --task state5 \
  --layer 17 \
  --feature_mode all6_mean \
  --model linear \
  --max_epochs 2 \
  --limit_train_samples 20 \
  --output_dir outputs/probe_runs/smoke_state5
```

## Full Probe Commands

Five-class linear probe:

```bash
conda run -n lingbot-map python scripts/train_probe.py \
  --feature_manifest outputs/features/feature_manifest.jsonl \
  --split_file outputs/splits/scene_split_seed0.json \
  --task state5 \
  --layer 17 \
  --feature_mode all6_flatten \
  --model linear \
  --pca_dim 64 \
  --max_epochs 200 \
  --early_stop_patience 20 \
  --output_dir outputs/probe_runs/state5_layer17_linear
```

Presence probe:

```bash
conda run -n lingbot-map python scripts/train_probe.py \
  --feature_manifest outputs/features/feature_manifest.jsonl \
  --split_file outputs/splits/scene_split_seed0.json \
  --task presence \
  --layer 17 \
  --feature_mode all6_flatten \
  --model linear \
  --pca_dim 64 \
  --max_epochs 200 \
  --early_stop_patience 20 \
  --output_dir outputs/probe_runs/presence_layer17_linear
```

Four-position probe:

```bash
conda run -n lingbot-map python scripts/train_probe.py \
  --feature_manifest outputs/features/feature_manifest.jsonl \
  --split_file outputs/splits/scene_split_seed0.json \
  --task position4 \
  --layer 17 \
  --feature_mode all6_flatten \
  --model linear \
  --pca_dim 64 \
  --max_epochs 200 \
  --early_stop_patience 20 \
  --output_dir outputs/probe_runs/position4_layer17_linear
```

Shallow MLP control:

```bash
conda run -n lingbot-map python scripts/train_probe.py \
  --feature_manifest outputs/features/feature_manifest.jsonl \
  --split_file outputs/splits/scene_split_seed0.json \
  --task state5 \
  --layer 17 \
  --feature_mode all6_flatten \
  --model mlp \
  --pca_dim 64 \
  --hidden_dim 256 \
  --dropout 0.1 \
  --max_epochs 200 \
  --early_stop_patience 20 \
  --output_dir outputs/probe_runs/state5_layer17_mlp
```

## Broad Probe Sweep

This is the current broad coverage run. It covers three targets
(`state5`, `presence`, `position4`), four extracted layers (`block04`,
`block11`, `block17`, `block23`), and multiple token readouts:

```text
all6_mean
all6_flatten + train-only PCA64
slot0 ... slot5
group_camera / group_register / group_scale
```

Linear probes are run for every readout. With `--include_mlp_controls`, shallow
MLP controls are also run for `all6_mean` and `all6_flatten_pca64`.

```bash
conda run -n lingbot-map python scripts/run_probe_sweep.py \
  --include_mlp_controls \
  --summary_dir outputs/probe_summaries/broad_probe_sweep \
  --max_epochs 200 \
  --early_stop_patience 20 \
  --seed 0
```

Expected broad sweep size:

```text
3 tasks x 4 layers x (11 linear readouts + 2 MLP controls) = 156 runs
```

Useful outputs:

```text
outputs/probe_summaries/broad_probe_sweep/sweep_plan.csv
outputs/probe_summaries/broad_probe_sweep/sweep_status.json
outputs/probe_summaries/broad_probe_sweep/logs/<run_name>.stdout.log
outputs/probe_summaries/broad_probe_sweep/logs/<run_name>.stderr.log
outputs/probe_runs/<run_name>/metrics.json
outputs/probe_runs/<run_name>/best_model.pt
```

Summarize only the runs from this sweep plan:

```bash
conda run -n lingbot-map python scripts/summarize_probe_runs.py \
  --probe_root outputs/probe_runs \
  --output_dir outputs/probe_summaries/broad_probe_sweep \
  --plan_file outputs/probe_summaries/broad_probe_sweep/sweep_plan.json
```

Summary outputs:

```text
outputs/probe_summaries/broad_probe_sweep/probe_summary.csv
outputs/probe_summaries/broad_probe_sweep/probe_summary.json
outputs/probe_summaries/broad_probe_sweep/best_by_task.json
outputs/probe_summaries/broad_probe_sweep/probe_summary.md
```

## Visualize Probe Sweep

Generate a consistent set of PNG/PDF figures from `probe_summary.csv`:

```bash
conda run -n lingbot-map python scripts/plot_probe_summary.py \
  --summary_csv outputs/probe_summaries/broad_probe_sweep/probe_summary.csv \
  --output_dir outputs/probe_summaries/broad_probe_sweep/figures
```

Figure outputs:

```text
outputs/probe_summaries/broad_probe_sweep/figures/01_layer_trends.png
outputs/probe_summaries/broad_probe_sweep/figures/02_task_layer_heatmaps.png
outputs/probe_summaries/broad_probe_sweep/figures/03_readout_family_distributions.png
outputs/probe_summaries/broad_probe_sweep/figures/04_single_slot_heatmaps.png
outputs/probe_summaries/broad_probe_sweep/figures/05_top_runs_by_task.png
outputs/probe_summaries/broad_probe_sweep/figures/06_val_vs_test_scatter.png
outputs/probe_summaries/broad_probe_sweep/figures/07_mlp_vs_linear_deltas.png
outputs/probe_summaries/broad_probe_sweep/figures/08_best_epoch_by_task.png
outputs/probe_summaries/broad_probe_sweep/figures/figure_index.md
```

For quick legacy coverage, there is also a smaller bash helper:

```bash
bash scripts/sweep_probe.sh
```

## Recommended Follow-Up Probes

The broad sweep is a first-pass coverage grid. To make the conclusions more
robust, repeat the best readouts across several base-scene splits:

```bash
for seed in 1 2 3 4; do
  conda run -n lingbot-map python scripts/build_splits.py \
    --feature_manifest outputs/features/feature_manifest.jsonl \
    --output_file outputs/splits/scene_split_seed${seed}.json \
    --seed ${seed} \
    --train_scenes 20 \
    --val_scenes 5 \
    --test_scenes 5

  conda run -n lingbot-map python scripts/run_probe_sweep.py \
    --tasks state5,presence,position4 \
    --layers block11,block17,block23 \
    --split_file outputs/splits/scene_split_seed${seed}.json \
    --output_root outputs/probe_runs_seed${seed} \
    --summary_dir outputs/probe_summaries/broad_probe_sweep_seed${seed} \
    --include_mlp_controls \
    --max_epochs 200 \
    --early_stop_patience 20 \
    --seed ${seed}
done
```

For a tighter confirmation sweep, restrict to the readouts that looked most
promising on seed 0 by editing `default_linear_feature_specs()` in
`scripts/run_probe_sweep.py` or by adding a small curated runner.
