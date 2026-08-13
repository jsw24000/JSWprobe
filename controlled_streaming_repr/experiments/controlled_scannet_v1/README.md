# controlled_scannet_v1

This experiment set uses controlled ScanNet video stimuli to probe memory and
state evolution in streaming 3D reconstruction models. The goal is not to prove
that tokens directly solve a downstream segmentation task. The goal is to expose
how Lingbot-map representations, attention, memory/cache state, and geometry
outputs change when the visual input stream is controlled.

## Sequence Settings

`repeated_frame_stability` builds streams of the form `A, A, A, ...`. It asks
whether deep GCA tokens, camera tokens, memory tokens, attention summaries, and
geometry outputs continue to evolve when the current image is exactly unchanged.

`two_state_alternation` builds streams of the form `A, B, A, B, ...` for near
pairs and `A, Z, A, Z, ...` for far pairs. It asks whether representations are
bound more strongly to image content, streaming time state, local pose windows,
or anchor memory.

`order_perturbation` builds `normal`, `reverse`, and deterministic `shuffle`
variants from the same source frames. It asks how shallow visual tokens, deep
GCA tokens, camera/register/memory tokens, and geometry outputs respond to frame
order.

This first version intentionally does not include occlusion, cross-scene
interference, augmentation, or activation patching.

## Decoupled Design

Sequence settings only scan raw ScanNet scenes and write manifest JSON files.
They do not run models, visualize tokens, or compute metrics.

Analysis methods only read saved `TokenBundle` files, attention summaries,
geometry outputs, and manifests. They do not know how a controlled sequence was
generated beyond the manifest metadata.

The connection is:

```text
ScanNet raw scene
  -> sequence setting registry
  -> sequence manifests
  -> model adapters
  -> token bundles
  -> analysis method registry
  -> metrics + figures + summary
```

Any setting can be combined with any analysis method.

## Data And Manifests

`data/controlled_sequences` is an input frame pool. It can hold prepared
scene/frame files or symlinks, but it is not where generated experiment
manifests are written.

`sequence_settings` defines how frames are combined into controlled sequences.
For example, it decides whether a manifest is repeated-frame, two-state
alternation, or order perturbation.

`outputs/manifests` records the actual generated controlled sequences for a
run. Each JSON manifest stores paths and metadata only; it does not copy large
RGB/depth/pose files.

Token extraction reads the manifest and writes a `TokenBundle`. Analysis reads
both the manifest and the `TokenBundle`, then writes metrics and figures.

## Analysis Methods

`frame_gram` pools tokens per frame and writes an `N x N` cosine similarity
matrix plus a heatmap.

`token_drift` computes drift to the first frame and drift to the previous frame,
plus role-based same/cross similarity when role metadata is present.

`shared_pca_patchmap` fits one shared PCA basis across related conditions before
projecting patch tokens. It should not be replaced by per-image PCA.

`patch_affinity` computes selected patch-to-patch cosine affinity heatmaps
without materializing every possible frame pair.

`attention_mass` reads saved Lingbot-map attention summaries. Raw attention is
not required and is disabled by default.

`geometry_stability` summarizes predicted camera/depth/pointmap stability. These
metrics are auxiliary evidence and should not be over-interpreted as hard
conclusions.

## Recommended First Run

Step 1: dry-run manifest generation for one scene and `seq_len=16`.

```bash
cd /home/3dsm/Desktop/JSWprobe/controlled_streaming_repr
python scripts/build_sequence_manifests.py \
  --config configs/controlled_scannet_v1.yaml \
  --limit-scenes 1 \
  --seq-lens 16 \
  --dry-run
```

Step 2: write manifests.

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_scannet_v1.yaml \
  --limit-scenes 1 \
  --seq-lens 16 \
  --overwrite
```

Step 3: run Lingbot-map extraction for the smallest repeated-frame condition.

```bash
python scripts/run_controlled_extraction.py \
  --config configs/controlled_scannet_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --setting repeated_frame_stability \
  --seq-len 16 \
  --max-runs 1
```

Step 4: run the first two lightweight analyses.

```bash
python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_scannet_v1.yaml \
  --analysis-config configs/analysis_controlled_v1.yaml \
  --manifest-index outputs/manifests/controlled_scannet_v1/index.json \
  --models lingbot-map \
  --methods frame_gram token_drift \
  --setting repeated_frame_stability \
  --seq-len 16 \
  --max-runs 1
```

Step 5: add VGGT and `shared_pca_patchmap` after the Lingbot path is healthy.

Step 6: expand to `seq_len=32` and multiple scenes after checking disk use.

## Notes

PCA maps must use a shared basis within the intended comparison group.

Gram and PCA figures are descriptive phenomena, not standalone conclusions.

Attention can be very large, so the default config saves summaries rather than
raw attention.

Geometry outputs are auxiliary evidence for representation analysis.

Lingbot-map d1/stage0 tokens serve as the DINO-like visual reference in this
project. DINO is not added as a separate model in this stage.
