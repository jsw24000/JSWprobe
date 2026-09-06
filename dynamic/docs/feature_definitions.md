# Feature definitions and coordinate audit

## Models actually found

- DINOv3 source `/home/3dsm/Desktop/dinov3`; local weights
  `/home/3dsm/Desktop/dinov3_weights/dinov3-vitl16-pretrain-lvd1689m/model.safetensors`.
  The on-disk config declares DINOv3ViTModel: ViT-L, depth=24, D=1024,
  patch=16, 4 registers. Runtime: installed Transformers 4.57.1
  `models/dinov3_vit/modeling_dinov3_vit.py`, strict local loading with no missing,
  unexpected or mismatched keys. The local Meta repository is recorded for
  provenance; the runtime does not pretend that the HF weights are native pth.
- Omega source `/home/3dsm/Desktop/vggt-omega`; weights
  `checkpoints/vggt_omega_1b_512.pt`, 1B-512 without text head, strict full-state load.
  Aggregator depth=24, D=1024, patch=16, 16 registers. Runtime cache selection
  is changed without modifying checkpoint or external source.

`audit/model_audit.json` records hashes, source commits/status and environment;
`*_runtime.json` records inspected model dimensions and implementation paths.

## Original render → model → lattice

The generator's `camera_intrinsics_from_fov` sets principal point `(W/2,H/2)`
and Blender projection gives UV relative to top-left image **edge**. Thus the
center of array pixel `[y,x]` is `(x+0.5,y+0.5)`, and the image spans
`[0,W]×[0,H]`. The generator's depth visibility sampling rounds UV in a 3×3
neighborhood; we retain that visibility information, then apply a stricter
interior filter. We do not copy that rounded lookup into feature interpolation.

`SpatialTransform` explicitly stores original size, crop origin/size, resize,
and padding offsets. This pilot uses identity spatial transforms for both
models: RGB/mask/UV remain 512×512, with no crop or padding. The transform also supports symmetric padding for the patch-14 models
described in the four-model extension below. This matches Omega's official image
loader at square 512. DINO's RoPE builds its grid from actual input size,
so the checkpoint's default 224 image processor resize is deliberately disabled.
An executed 512 forward validates a 32×32 lattice for both models.

For edge-coordinate input UV and a `Gh×Gw` feature map:

```
u_grid = u_model * Gw / W_model - 0.5
v_grid = v_model * Gh / H_model - 0.5
```

Patch centers `(8+16j,8+16i)` map exactly to `(j,i)`. Image edges 0 and 512
map to -0.5 and 31.5. `grid_sample(align_corners=False,padding_mode='border')`
uses normalized coordinates `2*u_model/W_model-1`, similarly for v.
Row-major flattened patch index is `i*Gw+j`. Tests verify centers, all-edge
behavior, resize, row/column ordering and a known bilinear ramp. Boundary
extension is explicit; retained object core points are far from image edges.

Dense fused `[256,128,128]` is sampled as a stride-4 **latent lattice**, with
nominal centers `(2+4j,2+4i)`. Learned convolution/deconvolution receptive fields
are not pixel-local; this mapping does not assert exact pixel-level features.
Bilinear interpolation for either map does not restore lost spatial information.

## Visibility and physical correspondence

`point_id` and `xyz_object_local` are checked identical across each group's
25 conditions. Tracks contain 192 canonical points with local/world/camera
XYZ, UV, axial Z, ray range, front/in-image/visible flags, sampled rendered
range and depth error. Visibility exists for all 192 samples in every frame,
with false values for invisible samples; it is not a dense surface guarantee.
The generation implementation compares Blender Z-pass ray range against
point-to-camera range (not axial Z), with abs/rel tolerances and target mask.

A core point must be visible/front/in-image in **every final condition**, and
its transformed UV must be ≥8 px inside the target silhouette/hole boundary.
We bilinearly sample Euclidean distance-to-background at UV−0.5 and subtract
0.5 pixel conservatively. This excludes strong mask boundaries; the original
visibility tolerance can still admit near self-occlusion. An 8 px margin reduces
mixing but cannot guarantee every receptive field is pure target. The separate
pool requires ≥90% patch occupancy. This distinction is retained in reports.

Canonical point IDs are saved per shard. If >64 pass, deterministic local-XYZ
farthest-point sampling begins with lowest point ID. This dataset passes with
26/25/31/38 points and no fallback. A dataset with <16 causes a hard gate;
metric-specific fallback is permissible but has not been needed/implemented.

## Tensor meanings

Omega cached block output: `[B,S,17+32*32,2048]`.
`[..., :1024]` is frame-attention output before this block's inter-frame
attention; `[..., 1024:]` is its inter-frame output. At endpoint:

- index 0: camera `[1,1024]`;
- indices 1:17: register set `[16,1024]`, never averaged before metrics;
- indices 17:1041: row-major patch lattice, sampled to `[P,1024]`;
- ≥0.9-occupancy patch average: `[1,1024]`.

Saved names are `L{index}__{pre|post}__{patch|pool|camera|register}`.
Raw pre/post halves are not concatenated for representation metrics.
Every register-only block must have raw patch max absolute difference ≤1e-6
or extraction stops before any science analysis. This is checked on the full
endpoint patch grid, not only the sampled points.

DINO output hook runs on blocks 5/11/17/23, then applies the shared `model.norm`
(as in native `get_intermediate_layers(norm=True)`). `[1,1029,1024]` tokens have
CLS at 0, four registers at 1:5, and patches at 5:1029. Only patch/clean pool
are saved as `L{index}__norm__{patch|pool}`. These depths are quartile positions
of the audited depth; DINO and Omega depth indices are not homologous modules.

Omega DenseHead hook: `dense_head.proj.register_forward_pre_hook`, the fused
feature **after positional embedding**, before its depth projection and pixel
shuffle. Runtime `[1,256,128,128]`; save `L23__fused__dense=[P,256]`. Only the
endpoint is decoded, using endpoint slices of the already joint-context cached
layers 4/11/17/23. The head's convolutions mix no frames, so this is equivalent
to the endpoint of full head evaluation in eval mode (checked in fidelity test).

Forward uses `eval()` plus `inference_mode()`. Parameters remain float32;
attention computation uses official bf16 autocast, DenseHead float32. Saved
readouts are float32; metrics convert to float32 before baseline subtraction.
The first-vs-subsequent camera/register initialization is never equated across
Single and Pair/Full; static referencing is always within the same regime.

## Four-model extension

The original definitions above continue to apply to DINOv3/Omega. In four-model
configs, `model_registry.py` selects the model-specific adapter, capabilities,
regimes, spatial transform and expected output schema. Metrics and group
aggregation are unchanged. Dataset-level core points are chosen once in the
original 512-coordinate system and explicitly verified in every model input.

For VGGT and DINOv2: the local audited patch size is 14, depth 24, D=1024.
Keep the original 512 pixels and pad 3px symmetrically to 518. The transform's
resize scale is `(input_width-2*pad_x)/crop_width`, likewise y; here exactly 1.
RGB padding is white (1 before normalization); masks use background (0).
UV_model=UV_original+3; the 37×37 lattice has centers at model UV=7+14j.
The unit test checks exact RGB preservation, UV shift, corners, lattice ordering,
and unchanged physical-point boundary distance. No old feature file is modified.

DINOv2 uses native `dinov2_vitl14_reg(pretrained=False)`, then a strict load of
its audited local pth. `get_intermediate_layers(n=[5,11,17,23],norm=True)` returns
`[1,1369,1024]` patch-only tensors (CLS/registers are excluded by the official
method). Patch/pool readout names remain `Lx__norm__patch/pool`. There is no
DINOv2 pair/full aggregation or register analysis, matching the DINOv3 baseline.

VGGT official aggregator outputs concatenate frame/global intermediates, just
as E1 expects: `[1,S,1374,2048]`, pre/post split at 1024. Camera=0,
registers=1:5, patches=5:1374. Selected layers match Omega's indices for easy
comparison, but **none** are register-only in VGGT. The official implementation
materializes all layers transiently; only selected endpoint compact readouts
are saved. Dense decoding uses only the endpoint's already contextualized
features from layers 4/11/17/23, with a full-head equivalence check.

The VGGT dense hook captures `depth_head.scratch.output_conv2` input:
`[1,128,518,518]`, after interpolation to model input resolution and position
embedding, before the depth/confidence prediction convolutions. Sampling uses
that map's nominal pixel-center lattice and the same transformed physical UV.
Omega's dense auxiliary remains the original 256-channel stride-4 lattice.
Channel count and decoder computations differ, so these auxiliary representations
are not architecture-identical controls. All saved readouts/metrics are float32.
