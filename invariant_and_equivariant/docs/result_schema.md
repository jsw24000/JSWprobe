# Result Schema

All metric rows use snake_case fields.

Required fields:

```text
run_name
model_name
model_variant
layer_name
feature_space
pooling
coordinate_system
split_name
seed
metric_name
metric_value
num_train
num_val
num_test
```

Feature index rows include:

```text
model_name
model_variant
checkpoint_id
layer_name
scene_id
state_id
camera_id
split
target_object_id
target_category
feature_path
feature_dim
mask_area_ratio
effective_patch_count
valid
position_world_x/y/z
position_ref_x/y/z
position_current_x/y/z
```

The smoke pipeline writes CSV, JSON, NPZ, PNG, and Markdown outputs only. Parquet can be added later if a stable dependency is introduced.

E9 move-control outputs live under `e9_move_controls/`:

```text
metrics.csv          summary metric rows following the common schema
metrics.json         JSON copy of the metric rows
summary.csv          one compact row per model/layer
control_details.csv  per-shuffle and per-scene-half stability records
*_bases.npz          true move, PCA-z, and PCA-delta-z rank-matched bases
report.md            protocol and compact result table
```
