# Revisit Memory Experiment

This folder is an independent experiment for testing whether LingBot-Map's long-range streaming state preserves information about an object seen in the first loop and whether that history changes geometry or image-token features during a second revisit.

No BlenderProc data is copied here. `configs/revisit_memory.yaml` points to:

```text
memory_scene_blender/outputs/lingbot_memory_basic_v1
```

## Source Facts Used

I inspected the local LingBot-Map code rather than assuming module names:

- Streaming entry: `lingbot_map.models.gct_stream.GCTStream.inference_streaming`.
- The official flow first processes scale/anchor frames together, then processes remaining frames one by one with `causal_inference=True`.
- The aggregator has `aggregator.frame_blocks` and `aggregator.global_blocks`, each with 24 blocks.
- `AggregatorBase.forward` returns selected layer outputs as `torch.cat([frame_intermediate, global_intermediate], dim=-1)`, so this experiment splits the final channel dimension into Frame Block and Global Block features.
- Token layout in streaming mode is camera token, 4 register tokens, scale token, then image patch tokens. Thus `patch_start_idx = 6`.
- FlashInfer cache stores scale patch pages, live-window patch pages, and append-only special pages. The experiment logs source-frame IDs for those categories.
- The runner builds LingBot-Map with its camera head and depth head enabled. Point/local point heads are disabled for this experiment.

## Core Configuration

Default config:

- anchor/scale frames: `8`
- local patch window: `16`
- keyframe interval: `1`
- 3D RoPE / streaming temporal encoding: enabled
- selected layers: `4, 11, 17, 23`
- saved feature frames by default: frames `12-39` and `56-83` with stride `4`, plus key frames
  `23, 30, 39, 56, 67, 74, 83`
- primary evaluation interval: frames `56-83`
- first-loop target pixel interval: frames `12-39`
- depth comparison scale: one shared `median(gt_depth / pred_depth)` scale from target-free background
  frames `0-11`, pooled across all conditions; 56-83 target-mask-outside pair alignment is logged as
  a diagnostic, not as the main metric
- lightweight qualitative point cloud export: one GT-pose second-loop overview PLY per condition

The explicit local-window check is:

```text
56 - 39 = 17 > local_window_size = 16
```

At frame 56, the history-only local window is frames `40-55`, so frames `12-39` are outside the local image-token window. Anchor frames `0-7` are also checked for target pixels.

## Environment

Use the same Python environment that can run LingBot-Map:

```bash
conda activate lingbot-map
cd /home/3dsm/Desktop/JSWprobe
pip install -e lingbot-map
```

The current config expects:

```text
lingbot-map/checkpoints/lingbot-map.pt
```

No separate DINOv2 pretraining file is used. The LingBot-Map checkpoint is the only model-weight source for this experiment.

FlashInfer is preferred when the installed wheel supports the GPU/CUDA pair. On the current Blackwell / CUDA 12.8 environment, FlashInfer reports that SM 12.x requires CUDA >= 12.9, so this experiment defaults to `use_sdpa: true`. This still uses LingBot-Map's streaming/causal KV-cache path, just with PyTorch SDPA instead of FlashInfer kernels.

## Commands

Validate paired inputs:

```bash
python revisit_memory/scripts/validate_inputs.py \
  --config revisit_memory/configs/revisit_memory.yaml
```

If the active environment cannot read EXR files, structure/RGB/camera/mask validation can still be run with:

```bash
python revisit_memory/scripts/validate_inputs.py \
  --config revisit_memory/configs/revisit_memory.yaml \
  --skip-depth
```

Inspect LingBot modules:

```bash
python revisit_memory/scripts/inspect_lingbot_modules.py \
  --config revisit_memory/configs/revisit_memory.yaml
```

Run sanity checks after reconstructions/features exist:

```bash
python revisit_memory/scripts/sanity_checks.py \
  --config revisit_memory/configs/revisit_memory.yaml
```

Generate the default layer-11 PCA RGB token visualization for frames `68,72,74,76,80,83`:

```bash
python revisit_memory/analysis/image_token_pca_vis.py \
  --config revisit_memory/configs/revisit_memory.yaml
```

Run one condition:

```bash
python revisit_memory/scripts/run_reconstruction.py \
  --config revisit_memory/configs/revisit_memory.yaml \
  --condition seen_then_removed
```

Smoke test through frame 56:

```bash
python revisit_memory/scripts/run_all.py \
  --config revisit_memory/configs/revisit_memory.yaml \
  --smoke
```

Full experiment:

```bash
python revisit_memory/scripts/run_all.py \
  --config revisit_memory/configs/revisit_memory.yaml \
  --run-id full_001
```

Memory transplant experiment, reusing `full_shared_scale_v1` as the baseline reference:

```bash
REVISIT_MEMORY_RUN_ID=memory_transplant_v1 \
/home/3dsm/miniconda3/envs/lingbot-map/bin/python \
revisit_memory/scripts/run_memory_transplant.py \
  --config revisit_memory/configs/memory_transplant_v1.yaml \
  --run-id memory_transplant_v1
```

Rerun only the transplant analysis after replay outputs exist:

```bash
REVISIT_MEMORY_RUN_ID=memory_transplant_v1 \
/home/3dsm/miniconda3/envs/lingbot-map/bin/python \
revisit_memory/analysis/analyze_memory_transplant.py \
  --config revisit_memory/configs/memory_transplant_v1.yaml
```

Rerun only analyses after reconstructions exist:

```bash
python revisit_memory/scripts/run_all.py \
  --config revisit_memory/configs/revisit_memory.yaml \
  --analysis-only \
  --run-id full_001
```

## Run Isolation

`run_all.py` writes every run to a separate directory:

```text
revisit_memory/outputs/runs/<run-id>/
```

If `--run-id` is omitted, `run_all.py` creates a timestamped run id such as:

```text
20260720_181530_smoke
20260720_203012_full
```

Use an explicit run id for important experiments, for example:

```bash
python revisit_memory/scripts/run_all.py \
  --config revisit_memory/configs/revisit_memory.yaml \
  --run-id full_sdpa_v1
```

Then use the same id for analysis-only reruns:

```bash
python revisit_memory/scripts/run_all.py \
  --config revisit_memory/configs/revisit_memory.yaml \
  --analysis-only \
  --run-id full_sdpa_v1
```

The root of each run contains `run_manifest.json`, plus its own `validation/`, `reconstruction/`, `features/`, `memory_tokens/`, `metrics/`, `analysis/`, and `report/` directories. This prevents smoke, full, and repeated analysis runs from overwriting or mixing with one another.

Input/data sanity failures stop the pipeline. Feature-equivalence checks use mean relative L2 as the hard gate; isolated max-token outliers are recorded as `pass_with_caveats` by default so downstream depth, point-cloud, camera, image-token, and memory-token analyses still run. Set `validation.feature_equivalence_hard_fail_on_max_relative_l2: true` for a stricter debugging pass.

When running individual scripts without `run_all.py`, set:

```bash
export REVISIT_MEMORY_RUN_ID=full_sdpa_v1
```

before invoking the script.

## Outputs

Main layout:

```text
outputs/
├── validation/
├── reconstruction/{condition}/
├── features/{condition}/
├── memory_tokens/{condition}/
├── analysis/
├── metrics/
└── report/summary.md
```

Per condition, reconstruction saves:

- `predictions.npz`: raw LingBot-Map depth-head output, depth confidence, pose encoding, predicted camera pose, and predicted intrinsics. Model input images are omitted by default.
- `inputs.npz`: GT camera pose, original and preprocessed intrinsics, counterfactual mask, mask pixel counts, and GT depth when EXR reading is available.
- `cache_log.json`: local window IDs, scale frame IDs, append-only special-token source frame IDs, and frame-56 checks.
- `pointclouds/pred_depth_metric_scaled/gt_camera/second_loop_0056_0083_overview.ply`: coarse sampled qualitative point cloud from LingBot depth-head predictions, globally scaled to GT metric depth, then reprojected with GT camera poses.
- `pointclouds/depth_metric_scale.json`: the global median `gt_depth / pred_depth` scale used for metric qualitative PLY export.
- `features.pt`: selected image tokens for configured sampled frames.
- `special_tokens.pt`: selected special/context tokens for all processed frames.

Point clouds in the analysis are derived from LingBot-Map depth-head predictions reprojected with GT camera poses. Because LingBot depth and camera pose are in an arbitrary but internally consistent scale, metric GT-pose point clouds and GT-bbox occupancy use a single global median `gt_depth / pred_depth` scale when GT depth is available. The reconstruction runner does not build or save a pointmap head.

Depth-result comparison is separate from point-cloud export. `analysis/depth/shared_depth_scale.json` stores the shared 0-11 background scale used for all three conditions. `metrics/depth_condition_differences.csv` compares `seen_then_removed - always_absent` on 56-83 after this shared scale, and also logs a target-mask-outside background alignment diagnostic for each frame.

Image-token PCA visualizations are written under `analysis/image_token_pca/`. For the default layer-11 selected-frame view, absent/removed condition colors come from one PCA fit jointly on both conditions over all selected frames for the same block. The `removed - absent` RGB maps use a separate PCA fit on all selected-frame difference tokens. The same diff PCA also outputs signed PC1/PC2/PC3 scalar-projection heatmaps and absolute-projection intensity heatmaps with one per-component color scale shared across all selected frames.

## Analysis Scope

Implemented first-stage analyses:

- input/feature sanity checks for identical first-loop prefixes
- pointcloud occupancy inside the target bbox, including `ghost_occupancy_delta`
- depth MAE and signed residual in/out of the counterfactual target mask
- same-scale depth deltas for `seen_then_removed` vs `always_absent`, with first-prefix/anchor scale sanity rows
- camera translation/rotation error with one full-sequence Sim(3) alignment
- image-token L2 maps, Global Block update differences, region percentiles, and patch-level L2 samples
- layer-11 image-token PCA RGB maps for `always_absent`, `seen_then_removed`, and `seen_then_removed - always_absent`, plus signed diff-PC projection heatmaps
- object-direction cosine alignment using `always_present - always_absent`
- frame/global special memory-token L2/norm analysis across the full sequence, plus band summaries and sampled source frames

The report explicitly distinguishes geometry ghost, image-token residual, camera-state changes, memory-token-only differences, and no detectable memory effect.

## Disk Use

Feature tensors and ASCII PLY files can be large. The default config uses sampled feature frames rather than a dense save: it saves every 4th frame in `12-39` and `56-83`, plus several key frames, omits model input images from `predictions.npz`, writes only one coarse reconstruction PLY per condition, disables analysis PLY files, and disables dense depth/image-token PNG dumps. Use smoke mode first before a full run.

For denser outputs, edit `configs/revisit_memory.yaml`:

- `frames.default_feature_stride`: set to `1` for dense feature analysis over `frames.default_feature_ranges`.
- `frames.default_feature_extra_frames`: add important viewpoints that should always be saved.
- `exports.reconstruction_pointclouds.depth_sources`: add `pred_depth` for the raw unscaled model-depth PLY, or `gt_depth` for a GT-depth/GT-camera geometry baseline.
- `exports.reconstruction_pointclouds.camera_sources`: add `pred_camera` if you want to inspect LingBot camera-head pose quality.
- `exports.reconstruction_pointclouds.segments`: add `first_loop`, `all_processed`, or `revisit_frame`.
- `analysis.pointcloud_save_ply`, `analysis.save_depth_figures`, `analysis.save_image_token_figures`: set to `true` only when those artifacts are needed.
