# controlled_16_exp_v3

This experiment probes prefix-induced modulation of Lingbot-map final-frame
image tokens while the current ScanNet frame is fixed.

## Conditions

All sequences use `scene0002_00`, total length 16, and target frame `600` as
the final input frame.

- `A_low_overlap`: frames `360..444` with stride 6, then `600`.
- `B1_local_relevant`: frames `510..594` with stride 6, then `600`.
- `B2_long_relevant`: frames `150..234` with stride 6, then `600`.

The target frame is never re-ordered into its original video position. It is
always appended as the final stream frame.

## Analysis

`prefix_delta` reads the final-frame patch tokens for layers 4, 11, 17, and 23,
then computes:

- `B1_minus_A = Z_X_B1 - Z_X_A`
- `B2_minus_A = Z_X_B2 - Z_X_A`
- `B1_minus_B2 = Z_X_B1 - Z_X_B2`

The method writes delta norm heatmaps, per-comparison delta PCA maps, fixed-PCA
condition maps, layer-wise metrics, manifest summary, and a markdown summary.

## Recommended Commands

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_16_exp_v3.yaml \
  --dry-run

python scripts/build_sequence_manifests.py \
  --config configs/controlled_16_exp_v3.yaml \
  --overwrite

python scripts/run_controlled_extraction.py \
  --config configs/controlled_16_exp_v3.yaml \
  --manifest-index outputs/manifests/controlled_16_exp_v3/index.json \
  --models lingbot-map \
  --setting prefix_induced_modulation \
  --seq-len 16

python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_16_exp_v3.yaml \
  --analysis-config configs/analysis_controlled_16_exp_v3.yaml \
  --manifest-index outputs/manifests/controlled_16_exp_v3/index.json \
  --models lingbot-map \
  --methods prefix_delta \
  --setting prefix_induced_modulation \
  --seq-len 16
```
