# Covisibility Probe

Training-free analysis for one question: do LingBot-Map's four per-frame register tokens form a view-level representation space that reflects true shared visible 3D surface overlap between two frames?

The experiment compares official DINOv2, LingBot's pre-aggregator DINO backbone tokens, LingBot register tokens at the exact head-input concat position, and an optional camera-token control. It does not train a probe, predict patch masks, or modify model parameters.

## Research Question

For frames `i` and `j`, the main target is whether register similarity `Similarity(R_i, R_j)` is stably associated with GT visible-surface overlap `O_ij`. The important negative controls are time proximity, camera pose proximity, and DINO appearance similarity. A positive raw correlation alone is not enough, because natural RGB-D video has strong temporal and pose shortcuts.

The current code supports negative and partial results. For example, high raw correlation but low residualized correlation should be read as evidence that the register space may mostly track time, pose, or appearance under this setup.

## Why Registers

LingBot-Map uses special tokens in the streaming aggregator. In the local source inspected for this project, the special-token layout is:

| Token | Index |
| --- | ---: |
| camera | 0 |
| register 0..3 | 1..4 |
| scale | 5 |
| image patches | 6.. |

The four register tokens are the focus because the streaming cache keeps special tokens from old frames after dense image patch K/V has left the local window. The camera token is saved only as a control for pose-oriented special-token behavior.

## Relation To Co-Visibility Analysis

This is related to Co-VGGT-style co-visibility probing in that the supervision signal is geometric overlap from RGB-D, pose, and intrinsics. It differs in scope: this project asks whether a compact view-level register descriptor already organizes frames by visible-surface overlap. It does not train pair classifiers, decode masks, or evaluate patch-level correspondences.

## Streaming Order

Main runs feed frames in natural ScanNet order:

```text
I_1 -> I_2 -> ... -> I_T
```

The debug and main configs set:

```yaml
natural_order: true
local_window: 32
video_rope: true
adaptive_keyframe_selection: false
keyframe_interval: 1
reset_state: false
```

The implementation mirrors LingBot streaming: the first `num_scale_frames` frames are processed as the initial scale/anchor block, then later frames are processed one at a time with the same KV cache.

## Trajectory-Only Pairs

The main analysis uses only pairs where the historical frame is outside the dense local window:

```text
i < j - local_window
```

Pairs touching the initial scale/anchor frames are marked `anchor_related` and excluded from the main metric. Pairs inside the local dense window are marked `local`. All pair types remain in the raw pair tables.

## Head-Input Feature Definitions

Layer indices are 0-based LingBot aggregator block-group indices. The selected layers are `4, 11, 17, 23`.

No forward hook or monkey patch is used in the current implementation. LingBot's local `AggregatorBase.forward` appends selected outputs as:

```python
concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
```

Each selected output has shape `[B, S, P, 2C]`. This is the same concat token representation consumed by LingBot prediction heads. The experiment extracts the camera/register special tokens from this returned `aggregated_tokens_list`.

Definitions:

| Feature | Exact tensor point |
| --- | --- |
| `register_head_frame` | register tokens from the frame half of the head-input concat, after the complete frame block |
| `register_head_global` | register tokens from the global half of the head-input concat, after the complete global block including FFN |
| `register_head_concat` | full `torch.cat([frame_half, global_half], dim=-1)` register tokens consumed by the heads |
| `camera_head_concat` | full head-input camera token control |
| `camera_head_frame` / `camera_head_global` | frame/global halves of the camera token control, saved for audit |

The earlier internal pre/post-GCA hook path has been removed so that all LingBot features are taken from the same location used by the original output heads. This run does not extract attention `q`, `k`, or `v` vectors.

## GT Visible-Surface Overlap

For each selected frame, depth pixels are sampled with configurable stride, backprojected using `intrinsic_depth`, transformed with ScanNet pose interpreted as camera-to-world, and quantized into world voxels:

```text
q(X) = floor(X / voxel_size)
```

Default:

```yaml
pixel_stride: 4
voxel_size: 0.05
depth_scale: 1000.0
```

The main GT is:

```text
overlap_cos = |V_i intersection V_j| / sqrt(|V_i| |V_j|)
```

The pair table also saves IoU and directional coverage:

```text
coverage_i_to_j = |V_i intersection V_j| / |V_j|
coverage_j_to_i = |V_i intersection V_j| / |V_i|
```

The debug run audits self-overlap, range, symmetry, adjacent-vs-far overlap, and a sampled strict reprojection comparison.

## Similarity

The main register similarity is fixed-slot cosine average:

```text
S_slot = mean_k cosine(r_i,k, r_j,k), k=1..4
```

The pair table also saves mean-pooled register cosine:

```text
z_i = normalize(mean_k r_i,k)
S_mean = z_i dot z_j
```

Official DINOv2, LingBot backbone, and camera-token controls use ordinary cosine similarity. For LingBot register tokens, the primary comparison uses `register_head_concat`; `register_head_frame` and `register_head_global` show whether the head-input signal comes mainly from the per-frame stream, the cross-frame/global stream, or their concatenation.

## Confound Controls

The main metrics are computed on trajectory-only pairs. For each scene independently, the residualized rank correlation controls:

```text
log1p(temporal_gap)
GT translation distance
GT rotation angle
official DINO similarity
```

All variables are rank-transformed and standardized, then Ridge residuals are correlated. The output is named `residualized_rank_correlation`. It is not a causal estimate.

Time-bin metrics are also saved. Within fixed temporal-gap bins, a real overlap signal should still vary with GT overlap.

## Metrics

The summary reports:

| Metric | Meaning |
| --- | --- |
| Spearman | per-scene Spearman between similarity and `overlap_cos`, then scene-mean |
| residualized rank correlation | per-scene residual correlation after time/pose/DINO controls |
| NDCG@5 | can similarity retrieve high-overlap far-history frames for each target |
| CKA | centered kernel alignment between feature Gram and GT overlap Gram, per scene |

Scene-level bootstrap confidence intervals are saved in `summary_metrics.csv`.

## Paths

The environment inspection found:

```text
ScanNet root: /home/data1/3dsm/ScanNet/scans_extracted
LingBot repo: /home/3dsm/Desktop/JSWprobe/lingbot-map
LingBot checkpoint: /home/3dsm/Desktop/JSWprobe/lingbot-map/checkpoints/lingbot-map.pt
DINOv2 repo: /home/3dsm/Desktop/JSWprobe/dinov2
DINOv2 checkpoint: /home/3dsm/Desktop/JSWprobe/weights/dinov2/dinov2_vitl14_reg4_pretrain.pth
VGGT repo: /home/3dsm/Desktop/JSWprobe/vggt
```

Only three ScanNet scenes are currently available: `scene0000_00`, `scene0002_00`, and `scene0006_00`. A main config requesting 20 scenes will record this shortage in `scene_split.json` rather than silently pretending 20 scenes exist.

## Commands

Use the `lingbot-map` conda environment on this machine:

```bash
conda activate lingbot-map
pip install -r requirements-extra.txt
```

Inspect:

```bash
python scripts/inspect_environment.py
```

Debug pipeline:

```bash
python scripts/run_pipeline.py --config configs/debug.yaml
```

Main pipeline:

```bash
python scripts/run_pipeline.py --config configs/main.yaml
```

Run stages separately:

```bash
python scripts/build_manifest.py --config configs/debug.yaml
python scripts/build_overlap_gt.py --config configs/debug.yaml --run-id <run_id>
python scripts/extract_features.py --config configs/debug.yaml --run-id <run_id>
python scripts/analyze_training_free.py --config configs/debug.yaml --run-id <run_id>
python scripts/make_visualizations.py --config configs/debug.yaml --run-id <run_id>
```

Resume a range:

```bash
python scripts/run_pipeline.py \
  --config configs/debug.yaml \
  --run-id <run_id> \
  --start-stage analyze_training_free \
  --end-stage make_visualizations
```

## Outputs

Each run writes:

```text
outputs/<run_id>/
metadata/
manifests/frames.parquet
features/<scene_id>.pt
overlap/
pairs/pairs_by_scene/<scene_id>.parquet
pairs/all_pairs.parquet
metrics/per_scene_metrics.parquet
metrics/summary_metrics.csv
metrics/summary_metrics.md
figures/
```

Feature files contain frame ids, input positions, official DINO descriptors, LingBot backbone descriptors, raw head-input register tensors per layer (`register_head_frame`, `register_head_global`, `register_head_concat`), camera-token controls, and GT camera-to-world poses.

## Reading Figures

Layer trend plots compare `Register head frame-half`, `Register head global-half`, and `Register head concat` over layers `4, 11, 17, 23`.

The time-by-overlap heatmap uses the configured representative layer, default `17`, with `register_head_concat` similarity. It shows whether similarity increases with GT overlap inside the same temporal bin.

The hexbin plot shows trajectory-only pair density over GT overlap and `register_head_concat` similarity. Retrieval cases show a target frame, register Top-3 history frames, and the GT-overlap-best history frame.

## Debug Run

The latest completed head-input debug run is:

```text
outputs/20260719_headconcat_debug
```

It selected `scene0000_00` and `scene0002_00`, 80 frames each with raw stride 5. The model input is `392 x 518`, patch grid `28 x 37`, and runtime `patch_start_idx=6`.

Overlap audit:

| Scene | voxel-vs-reprojection Spearman | adjacent > far |
| --- | ---: | --- |
| scene0000_00 | 0.9684 | true |
| scene0002_00 | 0.9909 | true |

The debug run produced 6,320 pairs, of which 1,560 are trajectory-only. The saved LingBot register shapes are:

| Feature | Shape per scene |
| --- | --- |
| `register_head_frame` | `[80, 4, 1024]` |
| `register_head_global` | `[80, 4, 1024]` |
| `register_head_concat` | `[80, 4, 2048]` |

The smoke-test summary is in `outputs/20260719_headconcat_debug/metrics/summary_metrics.md`. Tests pass:

```text
12 passed
```

These debug metrics are a smoke-test result, not a final research conclusion.

## Possible Outcomes

If the global half is clearly better than the frame half, that suggests cross-frame/global processing strengthens a view-level overlap geometry at the representation actually used by the heads.

If raw correlation is high but residualized correlation is low, the representation may mostly encode time, pose, or DINO appearance.

If register does not beat DINO, then this training-free register descriptor is not yet evidence for additional surface co-visibility information.

If layers differ, later analysis should treat layer choice as a design variable rather than reporting only the best-looking layer.

## Limits

This experiment cannot prove that GCA causally uses overlap information. It cannot prove registers contain precise patch-level masks. A cosine failure also cannot prove the information is absent; it may be nonlinear or slot-permuted.

Predicted LingBot pose is optional and disabled by default in the debug/main configs to keep the first pass light. GT pose is always saved as camera-to-world.

## Next Steps

1. Check whether GCA attention retrieves high-overlap historical frames.
2. Test whether post-GCA image tokens gain patch-level history coverage.
3. Train a very small pair probe to test nonlinear information not visible to cosine similarity.
