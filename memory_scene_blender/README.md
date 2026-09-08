# Lingbot-map Memory Scene BlenderProc Experiment

This directory generates a clean, paired BlenderProc dataset for probing whether Lingbot-map memory retains a target object after it has disappeared.

The generator creates three strictly paired 96-frame sequences:

- `always_present`: target exists in both loops.
- `always_absent`: target never exists.
- `seen_then_removed`: target exists through frame 51, then is removed for frames 52-95.

The camera trajectory, intrinsics, room layout, static objects, materials, and lighting are deterministic and shared across all conditions.

Three scene profiles are available:

- `basic`: the original multi-object scene, written to `lingbot_memory_basic_v1`.
- `simplified`: a reduced scene with only the room shell, one large central occluder block, and the teal target cylinder, written to `lingbot_memory_simplified_v1`.
- `simplified_v2`: the same reduced scene with the teal target cylinder moved to the other end of the central occluder and a matched camera path, written to `lingbot_memory_simplified_v2`.

## Run

From the project root:

```bash
source .venv-blenderproc/bin/activate
blenderproc run memory_scene_blender/generate_memory_scene.py
```

Generate the simplified scene:

```bash
source .venv-blenderproc/bin/activate
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified
```

Generate the simplified v2 scene:

```bash
source .venv-blenderproc/bin/activate
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified_v2
```

If the same output directory already exists, add `--overwrite`.

Recommended quick geometry check before the full render:

```bash
source .venv-blenderproc/bin/activate
blenderproc run memory_scene_blender/generate_memory_scene.py --dry-run
```

For the simplified scene:

```bash
source .venv-blenderproc/bin/activate
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified --dry-run
```

For the simplified v2 scene:

```bash
source .venv-blenderproc/bin/activate
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified_v2 --dry-run
```

The dry run uses geometric ray checks and can be conservative for tiny partial silhouettes. The rendered mask sanity checks are the authoritative visibility/gap checks.

For a faster low-resolution smoke render:

```bash
source .venv-blenderproc/bin/activate
blenderproc run memory_scene_blender/generate_memory_scene.py --resolution 320 240 --samples 8
```

Smoke render the simplified scene:

```bash
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified --resolution 320 240 --samples 8
```

Smoke render the simplified v2 scene:

```bash
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified_v2 --resolution 320 240 --samples 8
```

Generate the memory-write probe dataset:

```bash
source .venv-blenderproc/bin/activate
blenderproc run memory_scene_blender/generate_memory_write_probe_v1.py \
  --output-root memory_scene_blender/outputs/memory_write_probe_v1 \
  --num-base-scenes 30 \
  --num-frames 16 \
  --probe-frame 15 \
  --tail-frames 0 \
  --seed 0
```

The new generator also accepts underscore aliases for these options, e.g. `--output_root`, `--num_base_scenes`, and `--probe_frame`.

Generate the object-translation v1 smoke dataset:

```bash
blender --background --python memory_scene_blender/scripts/generate_object_translation_dataset.py -- \
  --config memory_scene_blender/configs/object_translation_v1.yaml \
  --mode smoke \
  --overwrite
python memory_scene_blender/scripts/render_object_translation_preview.py \
  --dataset-root memory_scene_blender/outputs/object_translation_v1
python memory_scene_blender/scripts/validate_object_translation_dataset.py \
  --dataset-root memory_scene_blender/outputs/object_translation_v1
```

Generate the full object-translation v1 dataset:

```bash
blender --background --python memory_scene_blender/scripts/generate_object_translation_dataset.py -- \
  --config memory_scene_blender/configs/object_translation_v1.yaml \
  --mode full
```

This object-translation generator uses direct Blender Python rather than BlenderProc, skips complete frames by default, and supports `--overwrite`, `--scene-id`, `--state-start/--state-end`, and `--camera-start/--camera-end`.
If the default output root already contains a smoke run, start the full run with `--overwrite` or pass a different `--output-root`; complete frames at a different resolution are rejected instead of mixed.

## Ego/Object World-X Factorial Sequences

`ego_object_x_factorial_v1` is a separate temporal dataset for comparing the
observable target motion `r = o - e` with its camera/object causal
decomposition.  It reads the `object_translation_v1` scene, 4x4 coarse anchor,
and spatial camera-bank metadata without modifying that dataset.  A bank camera
is selected as a fixed base pose; temporal camera poses preserve its rotation
exactly and add only world-X translation.  The historical camera-bank
intrinsics are deliberately not reused because their saved image-height field
is corrupted; intrinsics are recomputed from the declared 60-degree FOV and
the actual square render resolution.

The default roots are mode-specific so a dry run, smoke render, and pilot
cannot silently mix:

```text
memory_scene_blender/outputs/ego_object_x_factorial_v1/
  _dry_runs/smoke/
  _dry_runs/pilot/
  smoke/
  pilot/
```

Geometry-only smoke preflight (no image rendering):

```bash
blender --background --python memory_scene_blender/scripts/generate_ego_object_x_factorial_dataset.py -- \
  --config memory_scene_blender/configs/ego_object_x_factorial_v1.yaml \
  --mode smoke \
  --dry-run
python memory_scene_blender/scripts/validate_ego_object_x_factorial_dataset.py \
  --dataset-root memory_scene_blender/outputs/ego_object_x_factorial_v1/_dry_runs/smoke \
  --geometry-only
```

Render and validate the one-group, 200-frame smoke dataset:

```bash
blender --background --python memory_scene_blender/scripts/generate_ego_object_x_factorial_dataset.py -- \
  --config memory_scene_blender/configs/ego_object_x_factorial_v1.yaml \
  --mode smoke
python memory_scene_blender/scripts/validate_ego_object_x_factorial_dataset.py \
  --dataset-root memory_scene_blender/outputs/ego_object_x_factorial_v1/smoke
python memory_scene_blender/scripts/render_ego_object_x_factorial_preview.py \
  --dataset-root memory_scene_blender/outputs/ego_object_x_factorial_v1/smoke
```

Only after smoke validation passes, render and validate the four-group,
800-frame pilot:

```bash
blender --background --python memory_scene_blender/scripts/generate_ego_object_x_factorial_dataset.py -- \
  --config memory_scene_blender/configs/ego_object_x_factorial_v1.yaml \
  --mode pilot
python memory_scene_blender/scripts/validate_ego_object_x_factorial_dataset.py \
  --dataset-root memory_scene_blender/outputs/ego_object_x_factorial_v1/pilot
python memory_scene_blender/scripts/render_ego_object_x_factorial_preview.py \
  --dataset-root memory_scene_blender/outputs/ego_object_x_factorial_v1/pilot
```

An interrupted identical run may use `--resume`.  `--overwrite` is accepted
only for an output directory carrying this generator's dataset marker; it can
never target or sit inside `object_translation_v1`.  `--scene-id`,
`--anchor-id`, and `--base-camera-id` provide bounded debugging subsets.  The
`full` mode is implemented for future expansion but is not part of the current
pilot protocol.  A future non-dry-run full launch additionally requires the
explicit `--allow-full` acknowledgement.

Quick geometry-only check:

```bash
blenderproc run memory_scene_blender/generate_memory_write_probe_v1.py \
  --output-root memory_scene_blender/outputs/memory_write_probe_v1_dryrun \
  --num-base-scenes 3 \
  --dry-run \
  --overwrite
```

Low-resolution smoke render:

```bash
blenderproc run memory_scene_blender/generate_memory_write_probe_v1.py \
  --output-root memory_scene_blender/outputs/memory_write_probe_v1_smoke \
  --num-base-scenes 1 \
  --resolution 320 240 \
  --samples 8 \
  --overwrite
```

Render one condition only:

```bash
blenderproc run memory_scene_blender/generate_memory_scene.py --condition seen_then_removed
```

Render one simplified condition only:

```bash
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified --condition seen_then_removed
```

Render one simplified v2 condition only:

```bash
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified_v2 --condition seen_then_removed
```

Save a debug Blender scene for visual inspection:

```bash
blenderproc run memory_scene_blender/generate_memory_scene.py --condition always_present --save-blend memory_scene_blender/outputs/lingbot_memory_basic_v1/debug_always_present_frame28.blend --preview-frame 28
```

Save a simplified debug scene:

```bash
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified --condition always_present --save-blend memory_scene_blender/outputs/lingbot_memory_simplified_v1/debug_always_present_frame28.blend --preview-frame 28
```

Save a simplified v2 debug scene:

```bash
blenderproc run memory_scene_blender/generate_memory_scene.py --scene-profile simplified_v2 --condition always_present --save-blend memory_scene_blender/outputs/lingbot_memory_simplified_v2/debug_always_present_frame28.blend --preview-frame 28
```

## Output Layout

Default output root:

```text
memory_scene_blender/outputs/lingbot_memory_basic_v1/
  shared_scene_metadata.json
  sanity_checks.json
  always_present/
    rgb/frame_0000.png
    depth/depth_0000.exr
    masks/object_id/frame_0000.npy
    masks/object_id_png/frame_0000.png
    masks/category_id/frame_0000.npy
    masks/category_id_png/frame_0000.png
    masks/target_binary/target_binary_0000.png
    masks/instance_color/actual_color_0000.png
    cameras.json
    metadata.json
    contact_sheet.png
    summary.txt
  always_absent/
    ...
    masks/counterfactual_target/counterfactual_target_0052.png
  seen_then_removed/
    ...
    masks/counterfactual_target/counterfactual_target_0052.png
    masks/counterfactual_target_color/counterfactual_color_0052.png
```

Memory-write probe output:

```text
memory_scene_blender/outputs/memory_write_probe_v1/
  experiment_config.json
  manifest.jsonl
  manifest.csv
  summary.json
  reference_visibility/
    pos1/
      masks/target_mask_frame_0015.png
    ...
  base_scene_000/
    base_scene_metadata.json
    absent/
      rgb/frame_0000.png
      depth/depth_0000.exr
      masks/target_binary/target_binary_0015.png
      target_mask_frame_0015.png
      cameras.json
      metadata.json
      contact_sheet.png
      summary.txt
    pos1/
      ...
    pos2/
      ...
    pos3/
      ...
    pos4/
      ...
```

Depth is written as OpenEXR from Blender's Z pass. Instance/object masks are decoded from deterministic color-coded mask renders and saved as integer `.npy` maps, plus 16-bit PNG label maps and binary target/counterfactual target PNG masks for convenient inspection. For the memory-write probe generator, the target object uses fixed object ID `20`; its visibility ratio is computed from the frame-15 target mask area divided by the unoccluded reference mask area for the same target position.

Object-translation v1 output:

```text
memory_scene_blender/outputs/object_translation_v1/
  config_used.yaml
  dataset_summary.json
  README.md
  validation_summary.json
  manifests/
    scenes.jsonl
    cameras.jsonl
    states.jsonl
    frames.jsonl
    translation_pairs.jsonl
    composition_triplets.jsonl
    splits.json
  scene_000/
    scene_metadata.json
    cameras.json
    state_000/
      state_metadata.json
      camera_000/
        rgb.png
        depth.exr
        depth.npy
        normal.png
        albedo.png
        target_mask.png
        instance.png
        semantic.png
        object_id.png
        frame_metadata.json
  previews/
```

Ego/object factorial mode output:

```text
memory_scene_blender/outputs/ego_object_x_factorial_v1/pilot/
  config_used.yaml
  provenance.json
  source_rebuild_audit.json
  selection_report.json
  dataset_summary.json
  validation_summary.json
  manifests/
    scenes.jsonl
    anchors.jsonl
    base_cameras.jsonl
    groups.jsonl
    sequences.jsonl
    frames.jsonl
    matched_relative_groups.jsonl
    track_sets.jsonl
    splits.json
  scene_000/
    scene_metadata.json
    canonical_surface_points.npz
    anchor_xxx/camera_xxx/
      group_metadata.json
      sequences/<sequence_id>/
        sequence_metadata.json
        tracks.npz
        frames/frame_000/
          rgb.png
          depth.exr
          depth.npy
          normal.png
          albedo.png
          target_mask.png
          instance.png
          semantic.png
          object_id.png
          frame_metadata.json
  previews/
```

For object-translation v1, Blender world coordinates use meters with `+Z` up. Blender camera coordinates use local `-Z` forward and `+Y` up; OpenCV camera coordinates are stored separately with `+X` right, `+Y` down, and `+Z` forward. The saved conversion is `diag(1, -1, -1, 1)`, and `camera_000` is the reference camera.

## Frame Design

- Frames `0-7`: anchor segment. The target is geometrically checked and pixel-checked to be absent from anchor frames.
- Frames `8-51`: first loop, 44 frames.
- Frames `52-95`: second loop, exactly the same 44 camera poses as the first loop.
- Local window assumption: `16` frames.

The generator computes:

- target-visible frames from rendered masks;
- second-loop counterfactual target projection frames;
- the gap from the last first-loop target observation to the first second-loop revisit of that target location;
- whether the gap is greater than 16 frames.

## Notes

- No external scene assets are used.
- Materials are simple Blender procedural/noise materials and fixed matte colors.
- The central partition is a thick short wall that blocks the target from the anchor viewpoints.
- In the simplified profile, the non-target geometry is intentionally limited to the room shell plus one long central occluder block. The target has no decoys or nearby distractor objects.
- In the simplified v2 profile, the target is moved from `(1.62, -0.74, 0.42)` to `(1.62, 0.74, 0.42)`. The loop start/end camera positions remain matched to the simplified v1 path, while the intervening loop cameras are interpolated to preserve the same target-visible frame window.
- The script writes a per-condition contact sheet and `summary.txt` for fast visual checks.


## Ego/object factorial V2: confirmation across physical contexts

`ego_object_factorial_v2` builds novel procedural scenes without requiring any
`object_translation_v1` output or claiming exact source-scene reconstruction.
**Core `tx_d004` is the direct E1 replication panel:** world-X translation,
0.04 m per level, levels `[-2,-1,0,1,2]`, 25 Cartesian conditions, 8 linear-time
frames, 512×512, and 8 full-render samples. Lighting/materials belong to scene
context; there are no rotation, Z, non-rigid, or lighting-control interventions.

The 12 existing targets (four furniture categories × variants 0–2) each appear
in two of six backgrounds, yielding 24 contexts. Crossing uses
`b=(2*v+c+offset)%6`, offset 0 or 1; every background has one target per category.
The extension uses variant `c%2` in each category, retaining both of that target's
background contexts: 8 contexts, 2/category, covering all six backgrounds.
It adds X/Y translation at 0.02/0.04/0.06 m. Core `tx_d004` is stored once and
belongs to both panels in extension contexts. Total: **24 contexts, 64 groups,
1,600 sequences, 12,800 frames**. Auxiliary X/Y scales must be analyzed separately
from primary confirmation aggregation unless an explicit factor analysis is intended.

The static asset builder indexes layouts modulo 3. To obtain six distinct
room/static combinations without changing asset definitions, background room
indices are `[0,1,2,1,2,0]`, with static IDs `[0,1,2,3,4,5]`. Thus the second trio
uses different room/static pairings, rather than repeating the first trio.

V2 selects the first valid anchor/camera in deterministic candidate order.
An extension's single anchor and base camera must pass swept-room/collision and
all temporal camera/visibility checks for all six families. Initial size is a
projected bounding-box estimate; the post-render validator checks actual masks.
Failure preserves a selection report and stops; no planned context is replaced.
Canonical files live at `canonical_targets/<target_id>/canonical_surface_points.npz`;
the seed depends only on dataset seed and target identity. Render caching uses
six rounded world-translation coordinates at physical-context scope. Reused
render bytes do not share sequence metadata or tracks.

The `manifests/` directory retains scenes, anchors, base_cameras, groups, sequences,
frames, matched_relative_groups and track_sets, and adds targets, backgrounds,
contexts and motion_families. All manifest paths are dataset-root-relative.
`scene_id=context_id`; explicit identity columns carry background, target,
physical context, motion family and panel membership. Matched-relative groups
use `(group_id, relative_level)`, so axes/scales cannot merge. `splits.json`
explicitly leaves contexts unassigned; define task-specific splits before fitting.
`context_plan.json`, config/provenance hashes, seeds, Blender/Git information,
`selection_report.json`, and `reports/{coverage_summary.json,coverage_matrix.csv,
selection_summary.csv,projected_motion_scale.csv}` support audits, not scientific
conclusions. The 12×6 coverage matrix describes the full planned crossing;
coverage summary also reports actually generated counts.

Run commands from `/home/3dsm/Desktop/JSWprobe`. Existing nonempty output roots
are refused; choose a fresh `--output-root` to repeat a smoke. There is no automatic
overwrite or resume. Geometry outputs are separate from rendered outputs.

```bash
# Six-family geometry smoke, no image rendering
/home/3dsm/.local/bin/blender --background --python-exit-code 1 \
  --python memory_scene_blender/scripts/generate_ego_object_factorial_v2.py -- \
  --config memory_scene_blender/configs/ego_object_factorial_v2.yaml \
  --mode smoke --dry-run --smoke-all-families --output-root /tmp/ego_object_v2_geometry_smoke
python memory_scene_blender/scripts/validate_ego_object_factorial_v2.py \
  --dataset-root /tmp/ego_object_v2_geometry_smoke --geometry-only

# Real small smoke: one context, tx_d004, 25 sequences, 200 frames, 4 samples
/home/3dsm/.local/bin/blender --background --python-exit-code 1 \
  --python memory_scene_blender/scripts/generate_ego_object_factorial_v2.py -- \
  --config memory_scene_blender/configs/ego_object_factorial_v2.yaml --mode smoke
python memory_scene_blender/scripts/validate_ego_object_factorial_v2.py \
  --dataset-root memory_scene_blender/outputs/ego_object_factorial_v2/smoke
python memory_scene_blender/scripts/render_ego_object_factorial_v2_preview.py \
  --dataset-root memory_scene_blender/outputs/ego_object_factorial_v2/smoke
```

Full commands below are for later manual execution. Implementation/smoke testing
does **not** create the full dataset. A real full render requires `--allow-full`.

```bash
/home/3dsm/.local/bin/blender --background --python-exit-code 1 \
  --python memory_scene_blender/scripts/generate_ego_object_factorial_v2.py -- \
  --config memory_scene_blender/configs/ego_object_factorial_v2.yaml \
  --mode full --allow-full
python memory_scene_blender/scripts/validate_ego_object_factorial_v2.py \
  --dataset-root memory_scene_blender/outputs/ego_object_factorial_v2/full
python memory_scene_blender/scripts/render_ego_object_factorial_v2_preview.py \
  --dataset-root memory_scene_blender/outputs/ego_object_factorial_v2/full --max-groups 2
```

To repair only the two small-target contexts, render into a fresh staging root.
The chair keeps more image-edge margin at 65% of its original camera distance;
the smaller side table uses 62%. This produces exactly 2 contexts, 7 groups,
175 sequences and 1400 frames. The apply step first checks every rendered frame,
then transactionally replaces those context directories and manifest rows; a
failed full validation restores the old data.

```bash
/home/3dsm/.local/bin/blender --background --python-exit-code 1 \
  --python memory_scene_blender/scripts/generate_ego_object_factorial_v2.py -- \
  --config memory_scene_blender/configs/ego_object_factorial_v2.yaml \
  --mode full --allow-full \
  --context-id bg_003__chair_v01 \
  --context-id bg_003__side_table_v00 \
  --camera-distance-scale 0.65 \
  --context-camera-distance-scale bg_003__side_table_v00=0.62 \
  --output-root memory_scene_blender/outputs/ego_object_factorial_v2/repair_bg003_near

# Read-only acceptance of the staged replacement.
python memory_scene_blender/scripts/apply_ego_object_factorial_v2_context_repair.py \
  --dataset-root memory_scene_blender/outputs/ego_object_factorial_v2/full \
  --replacement-root memory_scene_blender/outputs/ego_object_factorial_v2/repair_bg003_near

# Install only after the read-only command reports ok=true.
python memory_scene_blender/scripts/apply_ego_object_factorial_v2_context_repair.py \
  --dataset-root memory_scene_blender/outputs/ego_object_factorial_v2/full \
  --replacement-root memory_scene_blender/outputs/ego_object_factorial_v2/repair_bg003_near \
  --apply
```

Later feature extraction uses `dynamic/configs/e1_v2_core_template.yaml`, which
selects `motion_family_ids: [tx_d004]`. The adapter filters before grouping/audit;
heterogeneous axes/scales are rejected, including at analysis entry. Extraction
validation uses the same selection. V1 configs need no changes. After full render
validation, the existing staged extraction workflow can be run manually:

```bash
/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python dynamic/scripts/audit_e1.py \
  --config dynamic/configs/e1_v2_core_template.yaml
/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python dynamic/scripts/extract_e1_features.py \
  --config dynamic/configs/e1_v2_core_template.yaml --stage smoke
/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python dynamic/scripts/validate_extraction.py \
  --config dynamic/configs/e1_v2_core_template.yaml --stage smoke
/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python dynamic/scripts/extract_e1_features.py \
  --config dynamic/configs/e1_v2_core_template.yaml --stage full
/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python dynamic/scripts/validate_extraction.py \
  --config dynamic/configs/e1_v2_core_template.yaml --stage full
```

This task does not redesign E1 scientific reports: some narrative remains pilot
specific, so review it before using report text for V2. V1 generators, configs,
asset definitions, exact-rebuild selection, X-axis defaults, and existing outputs
are unchanged. Blender-free regression tests:

```bash
python -m unittest discover -s memory_scene_blender/tests
/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python -m unittest discover -s dynamic/tests
```

Additional reproducible smoke audits (fresh report paths required):

```bash
/home/3dsm/.local/bin/blender --background --python-exit-code 1 \
  --python memory_scene_blender/tests/blender_v2_canonical_smoke.py -- \
  --output-report /tmp/ego_object_v2_canonical_identity.json
python memory_scene_blender/tests/v2_validation_mutations.py \
  --dataset-root /tmp/ego_object_v2_geometry_smoke \
  --output-report /tmp/ego_object_v2_validation_mutations.json
```

The canonical audit constructs all 24 scenes and compares 12 target identities
across backgrounds; it renders zero frames. Mutation checks use temporary copies
and first require a passing baseline. Geometry validation reports its effective
precision floors (2e-6 m for same-r coordinates and 2e-7 m for object displacement),
accounting for Blender float32 transforms and the existing eight-decimal matrix
serialization. UV tolerances remain configuration controlled.


V2 depth calibration on this machine: Blender 5.2 `BLENDER_EEVEE` Z-pass is
**axial camera Z**, confirmed against scene ray casts. V2 keeps native axial
`depth.exr` and `native_depth.npy`, while `depth.npy` contains Euclidean ray range
computed using pixel-center rays and saved K. This lets the existing sparse-track
visibility helper consume its declared range units. The validator checks this
conversion and then independently resamples mask/depth visibility. Other Blender
versions/backends are rejected until calibrated. V1 depth artifacts/code are unchanged.

The initial rendered smoke at `outputs/ego_object_factorial_v2/smoke` predates this
correction and is retained with its failed validation. The corrected, validated
copy is `outputs/ego_object_factorial_v2/smoke_depth_checked`; RGB and native EXR
bytes are reused, with normalized NPY depth and recomputed visibility. Reproduce
that conversion only to a fresh destination:

```bash
python memory_scene_blender/scripts/normalize_ego_object_v2_smoke_depth.py \
  --source-root memory_scene_blender/outputs/ego_object_factorial_v2/smoke \
  --output-root /tmp/ego_object_v2_corrected_smoke
```

New generation performs depth normalization automatically. The conversion tool
accepts only the initial V2 smoke schema, never V1 or full datasets.
