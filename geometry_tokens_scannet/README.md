# geometry_tokens_scannet

This is a side-by-side experiment next to `lingbot-map` and `controlled_streaming_repr`.
It tests whether Lingbot-map intermediate-layer PCA "noise" patches tend to land on
ScanNet patch locations whose depth is hard to predict from simple 2D cues.

The goal is not to prove that PCA noise is a geometry token. A positive overlap or
correlation is only preliminary evidence that the visual noise may encode geometry
beyond low-level 2D appearance. A weak or negative result is still useful.

## Default Data Assumption

The default config targets:

- scene: `scene0002_00`
- controlled experiment: `controlled_128_exp_v1`
- setting: `order_perturbation`
- condition: `order_normal_seq128_seg000`
- model: `lingbot-map`
- PCA layer: `layer_17/patch_tokens`

The scripts first read the controlled manifest and token metadata produced by
`controlled_streaming_repr`. In the current workspace this resolves to the 128-frame
ScanNet sequence with stride 5. The patch grid is inferred from token shape and RGB
aspect ratio. For the current Lingbot-map outputs, `1036` patch tokens become `28 x 37`.

No source files in `controlled_streaming_repr` or `lingbot-map` are modified.

## Configure Paths

Edit `configs/default.yaml` if automatic discovery does not match your machine:

- `paths.controlled_root`: sibling `controlled_streaming_repr` directory.
- `paths.scannet_root`: ScanNet extracted scans root. Can also come from `SCANNET_RAW_ROOT`.
- `paths.manifest_path`: explicit controlled sequence manifest JSON.
- `paths.token_dir`: explicit directory containing `layer_XX_patch_tokens.npy`.
- `paths.pca_figures_dir`: explicit directory containing `shared_pca_patchmap_*_tXXX.png`.
- `paths.pca_metrics_path`: explicit `shared_pca_patchmap.json`.
- `tokens.patch_grid`: set manually if token shape cannot be found.

For `scene0000_00`, override `controlled.scene_id`, `setting_name`, and
`condition_id`, or pass command-line overrides.

## Run

From this directory:

```bash
python scripts/run_all.py --config configs/default.yaml
```

Useful debugging variants:

```bash
python scripts/run_all.py --max-frames 8
python scripts/extract_geometry_tokens.py --config configs/default.yaml
python scripts/extract_pca_noise.py --config configs/default.yaml
python scripts/compare_geometry_pca_noise.py --config configs/default.yaml
```

Scene override example:

```bash
python scripts/run_all.py \
  --scene-id scene0000_00 \
  --setting-name natural_stream \
  --condition-id natural_seq128_seg000
```

## Method

1. Load ScanNet RGB, depth, and intrinsics from the controlled manifest. RGB and depth
   are pooled to the Lingbot patch grid.
2. Build low-level 2D cues per patch:
   `x`, `y`, mean RGB, Sobel edge magnitude, and optional RGB gradient magnitude.
3. Fit a simple Ridge regressor from cues to patch depth. The default is sequence-level
   fitting across all valid patches. If `sklearn` is installed, `RidgeCV` is used;
   otherwise a small NumPy RidgeCV fallback is used.
4. Compute patch depth residuals and normalize them per frame.
5. Define Geometry Tokens as top 5%, 10%, and 20% residual patches among valid depth
   patches.
6. Load PCA noise masks when available. If only PCA RGB images exist, generate noise
   scores from:
   - local PCA color anomaly versus neighborhood median;
   - very dark PCA patches;
   - highly saturated PCA patches;
   - small connected candidate components.
7. Compare Geometry Tokens with PCA noise patches on common frame indices.

PCA heuristic parameters are under `pca_noise` in the config.

## Metrics

For each top-k threshold:

- `overlap_ratio`: intersection divided by the smaller set size.
- `iou`: intersection over union.
- `precision_at_geometry_topK`: intersection divided by number of geometry tokens.
- `recall_at_geometry_topK`: intersection divided by number of PCA noise patches.
- `noise_to_nearest_geometry_mean_patch_distance`: mean Euclidean patch-grid distance
  from each PCA noise patch to the closest geometry token.
- `pearson_score` and `spearman_score`: correlation between residual score and PCA
  noise score when both are available.

## Outputs

All default outputs go to:

```text
outputs/scene0002/
```

Important files:

```text
outputs/scene0002/geometry/geometry_score.npy
outputs/scene0002/geometry/geometry_mask_top5.npy
outputs/scene0002/geometry/geometry_mask_top10.npy
outputs/scene0002/geometry/geometry_mask_top20.npy
outputs/scene0002/pca_noise/pca_noise_score.npy
outputs/scene0002/pca_noise/pca_noise_mask.npy
outputs/scene0002/metrics.json
outputs/scene0002/comparison/metrics_per_frame.csv
outputs/scene0002/comparison/*_compare_top10.png
outputs/scene0002/comparison/summary_overlap_metrics.png
outputs/scene0002/comparison/score_correlation_scatter.png
```

Each processed frame also has per-frame NumPy arrays and visual overlays under
`geometry/frames/` and `pca_noise/frames/`.

## Interpreting Negative Results

Low overlap or weak correlation does not rule out geometry in Lingbot-map tokens.
Possible causes include:

- PCA noise may come from positional encoding rather than scene geometry.
- Streaming state, memory tokens, or KV-cache modulation may create visual artifacts.
- PCA normalization may amplify rare token directions unrelated to depth residuals.
- The PCA RGB image may be an upsampled visualization artifact.
- RGB-depth alignment and patch-grid pooling can introduce spatial error.
- Depth residuals from a linear 2D-cue Ridge model are only a proxy for geometry.
- Different Lingbot layers can encode different mixtures of geometry, appearance,
  motion history, and reconstruction state.
