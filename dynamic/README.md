# Dynamic representation analysis: E1 pilot

For the current four-model workflow, use `configs/e1_four_models.yaml`; commands
and architecture differences are in [Four-model comparison](#four-model-e1-comparison-vggt--omega--dinov2--dinov3). The original two-model protocol below is retained.

E1 asks whether representation geometry separates **observable relative translation**
`r=o-e` from its **physical camera/object causes (e,o)** across network computation
and joint multi-frame context. This is controlled intervention feature geometry,
not a probe-accuracy or absolute-feature clustering experiment.

The read-only source is
`../memory_scene_blender/outputs/ego_object_x_factorial_v1/pilot`: 2 scenes,
4 `(scene,anchor,base_camera)` groups, 25 conditions/group, 8 frames/sequence,
512×512, world-X translation step δ=0.04 m. Levels are integers -2…2;
physical amplitudes are level×δ. All groups belong to the source train split.
No readout is trained, and no held-out generalization claim is made.

VGGT-Omega observes `[I7]` (Single), `[I0,I7]` (Pair), or `[I0,…,I7]` (Full).
The target is always I7. Single gives it the first/reference special tokens;
Pair and Full give it subsequent-frame tokens. Pair vs Full is the cleaner
context comparison. This model is not streaming, so layers are computation,
not video time. DINOv3 independently processes I7 as a strong single-image
visual baseline; there is no artificial DINO Pair/Full pooling.

Primary features interpolate the patch lattice at the same canonical physical
surface points. All 25 final frames must share visibility, positive camera Z,
in-image projection, and ≥8 model-pixel mask-boundary distance. The audited
core sets contain 26/25/31/38 points, shared across all models and regimes.
No fallback was needed. The code stops for insufficient common points; a
metric-specific fallback must be explicitly implemented and audited before
using a new dataset that needs one. No metric silently changes point sets.

Omega caches blocks `[0,2,4,6,9,11,14,17,20,23]`. Each 2048-dimensional cached
token is split into 1024-dimensional frame-block output (`pre`) and
inter-frame-block output (`post`). Patch, camera and all 16 individual
registers are retained. Register-only blocks 2/6/9/14/20 must have identical
patch pre/post. DINOv3 ViT-L/16 has 24 blocks; blocks `[5,11,17,23]` are read
with the shared output LayerNorm. Its local checkpoint is in Transformers
format; the corresponding already installed implementation loads it offline.
The upstream source and actual runtime source are both recorded.

Two auxiliary representations check robustness: mean over ≥90% target-occupied
patches (entity pooling, not physical correspondence), and Omega DenseHead
fused features `[256,128,128]` just before depth projection, sampled at the
same physical points. DenseHead operates per frame; endpoint-only decoding
preserves all context already aggregated from Single/Pair/Full.

Every condition uses its own **same-regime static (0,0) endpoint** reference.
Metrics are tangent consistency and norm scaling at ±δ/±2δ, signed/absolute
ego-object tangent cosine, sensitivities, nonzero-same-r normalized causal
distance, and compensated r=0 response. See [protocol](docs/e1_protocol.md)
and [feature definitions](docs/feature_definitions.md).

The independent reporting unit is the group, not surface points. We show each
group and their median, keep point distributions, and perform no significance
tests. Four groups share only two scenes, so even group diversity is limited.
Feature separation does not establish causal identification; static context,
background, illumination and view changes remain possible explanations.

Run from any working directory:

```bash
bash /home/3dsm/Desktop/JSWprobe/dynamic/scripts/run_e1_pipeline.sh
```

The pipeline audits sources, tests coordinates and analytic metric oracles,
extracts and validates the 10-condition smoke, then extracts/validates the full
pilot, computes metrics, generates five figure groups and builds the report.
Existing complete compatible feature shards are reused and hash-validated.
To create an isolated new run, append `--output-root /absolute/fresh/path`.
All scripts accept `--config`, `--dataset-root`, `--output-root`, `--device`,
`--dinov3-checkpoint`, `--vggt-omega-checkpoint`, and corresponding `--*-repo`.
A different model architecture or image resolution needs a new audited adapter;
those changes are intentionally not silently inferred from CLI arguments.
Use `E1_PYTHON=/path/to/python` to override the shell interpreter.

Outputs are in `outputs/e1_pilot/{audit,features,metrics,figures,report}`.
Read [E1_REPORT](outputs/e1_pilot/report/E1_REPORT.md) for measured results and
`E1_SUMMARY.json` for machine-readable tables. The source dataset, external
model repositories and checkpoint bytes are never modified or copied here.
Compact features are stored float32 (~1.1 GB) to avoid additional float16
quantization before subtracting small intervention responses. Model attention
uses bf16 autocast; DenseHead and metric arithmetic use float32.

## Four-model E1 comparison (VGGT / Omega / DINOv2 / DINOv3)

Use the shared pipeline with the new explicit configuration:

```bash
cd /home/3dsm/Desktop/JSWprobe
bash dynamic/scripts/run_e1_pipeline.sh --config dynamic/configs/e1_four_models.yaml
```

This runs an 80-shard smoke gate followed by 800 full shards (100 sequences ×
DINOv2 Single + DINOv3 Single + VGGT Single/Pair/Full + Omega Single/Pair/Full),
then the same metric implementation and a joint report. Processing is sequential
on one GPU. Outputs are in `dynamic/outputs/e1_four_models/`; the original
`e1_pilot/` results and its two-model configuration are preserved. Repeating the
same command validates and reuses compatible existing shards. It does not
re-forward completed sequences. Model/config/input mismatches fail instead of
silently overwriting the cache.

To run only the two newly added models in a separate output tree:

```bash
bash dynamic/scripts/run_e1_pipeline.sh \
  --config dynamic/configs/e1_four_models.yaml \
  --models dinov2 vggt \
  --output-root /home/3dsm/Desktop/JSWprobe/dynamic/outputs/e1_vggt_dinov2
```

The same `--models` flag supports any subset. All stage scripts share the model
registry and accept the same configuration/path/device overrides. The extraction
script additionally accepts `--model NAME` to process one model under an already
audited configuration; the smoke/full completeness gates still require every
model selected by `--models`/the config. Do not launch multiple writers into the
same output root concurrently.

To stop after a smoke check, run these three stages:

```bash
E1_PYTHON=/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python
"$E1_PYTHON" dynamic/scripts/audit_e1.py --config dynamic/configs/e1_four_models.yaml
"$E1_PYTHON" dynamic/scripts/extract_e1_features.py --config dynamic/configs/e1_four_models.yaml --stage smoke
"$E1_PYTHON" dynamic/scripts/validate_extraction.py --config dynamic/configs/e1_four_models.yaml --stage smoke
```

Important audited architecture differences:

| Model | Local variant | Patch | Input | Aggregator registers | Selected blocks |
|---|---|---|---|---|---|
| DINOv3 | ViT-L/16 LVD1689M | 16 | 512×512 | N/A (encoder has 4) | 5,11,17,23 |
| DINOv2 | ViT-L/14-reg4 | 14 | 518×518 padded | N/A (encoder has 4) | 5,11,17,23 |
| Omega | 1B-512 | 16 | 512×512 | 16 | 0,2,4,6,9,11,14,17,20,23 |
| VGGT | 1B | 14 | 518×518 padded | 4 | 0,2,4,6,9,11,14,17,20,23 |

All four actual selected models have depth 24 and embedding dimension 1024.
The DINOv2 checkpoint is loaded strictly from
`/home/3dsm/Desktop/dinov2/weights/dinov2/dinov2_vitl14_reg4_pretrain.pth` using
native local source. VGGT uses `/home/3dsm/Desktop/vggt/checkpoints/model.pt`
and the corresponding local source. No weights or repositories are downloaded.
DINOv2's moved directory has no `.git`; all Python source hashes are recorded.

The two patch-14 models receive the same original pixels, with 3 white pixels
padded on every side. Masks pad with background; UV shifts by +3. No resize,
crop, frame reduction or point reselection occurs. These required spatial
adapters are explicitly tested and audited. Their patch-grid coordinates are
`(UV+3)/14-0.5`, versus `UV/16-0.5` for the patch-16 models. All models retain
the exact 26/25/31/38 physical points and the ≥8px interior criterion.

VGGT has global inter-frame attention at every block, so Omega's register-only
equality gate does not apply to it. Its pre/post halves still mean frame-output
versus inter-frame-output and each is analyzed independently. All 4 VGGT
registers are saved and analyzed individually, with their median in the main
plot. The dense VGGT hook is `depth_head.scratch.output_conv2` input, runtime
`[1,128,518,518]`, before the prediction head. It is a different decoder stage
and scale from Omega's `[1,256,128,128]`; comparisons retain this limitation.

Results: `outputs/e1_four_models/report/E1_REPORT.md`, `E1_SUMMARY.json`,
`figures/figure3_four_model_causal_distance.{png,pdf}` and the shared CSV schemas.
The joint report is generated from measured model-specific tables, not the
interpretive prose of the earlier two-model run.

## E2 V2 four-panel experiment

E2 is an additive, separately named pipeline using the validated 24-context V2
dataset and `outputs/e2_full_v2/`. It preserves E1 adapters and metrics, adds
physical-context-level X/Y point matching, relative/common response geometry,
rank-safe 2-D tangent-like subspaces, decomposed same-r controls, and cross-scale
locality diagnostics. Start with the no-forward planner:

```bash
python dynamic/scripts/plan_e2.py --config dynamic/configs/e2_full_v2.yaml --panel all
```

Exact staged commands, metric definitions, cache behavior, output tables, and
scientific wording constraints are in [the E2 protocol](docs/e2_protocol.md).
