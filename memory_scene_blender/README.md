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
