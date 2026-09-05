# Dynamic representation analysis: E1 pilot

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
