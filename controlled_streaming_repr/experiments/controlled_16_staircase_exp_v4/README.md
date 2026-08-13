# controlled_16_staircase_exp_v4

This experiment extends `controlled_16_exp_v3` with a 16-step staircase prefix.
It keeps `scene0002_00`, target frame `0600`, total length 16, Lingbot-map
extraction settings, token hooks, layers, and output schema aligned with v3.

## Sequence Definition

The v4 setting is `controlled_16_staircase`. It derives all frames from the v3
configuration, without reselecting or reprocessing source frames:

- Relevant prefix `R01..R15`: v3 `B1_local_relevant` frames `510..594`.
- Unrelated prefix `U01..U15`: v3 `C` frames `920..1004`.
- Target: frame `0600`, always the last input.
- X baseline: v3 `X`, repeated `0600`, reused by analysis.

For `k=0..15`:

```text
S_k = [U01, ..., U(15-k), R(16-k), ..., R15, target_0600]
```

Endpoints:

- `staircase_k00` exactly matches v3 `C`.
- `staircase_k15` exactly matches v3 `B1_local_relevant`.

Each manifest frame records the source file, frame id, `Rxx`/`Uxx` reference,
source sequence, whether it is relevant history, and whether it is the target.

## Reverse Staircase Sub-Experiment

The reverse sub-experiment setting is `controlled_16_staircase_reverse`. It uses
the same v3 `B1_local_relevant`, `C`, `X`, scene, target, layers, and sequence
length, but replaces the unrelated prefix from left to right:

```text
S_k = [R01, ..., Rk, U(k+1), ..., U15, target_0600]
```

Endpoints are unchanged:

- `staircase_k00` exactly matches v3 `C`.
- `staircase_k15` exactly matches v3 `B1_local_relevant`.

This reverse setting is intended to test whether early relevant frames have a
different marginal effect from the suffix replacement used by the original v4
staircase.

## X-to-B1 Staircase Sub-Experiment

The X-to-B1 sub-experiment keeps the same v4 scene, target, B1 sequence, model
settings, and output root. It changes only the non-relevant prefix source:
instead of using unrelated C frames, the baseline prefix is the v3 repeated
target X sequence.

The suffix-replacement setting is `controlled_16_staircase_x_to_B1`:

```text
S_k = [X01, ..., X(15-k), R(16-k), ..., R15, target_0600]
```

The prefix-replacement setting is `controlled_16_staircase_x_to_B1_reversed`:

```text
S_k = [R01, ..., Rk, X(k+1), ..., X15, target_0600]
```

Endpoints:

- `staircase_k00` exactly matches v3 `X`.
- `staircase_k15` exactly matches v3 `B1_local_relevant`.

This isolates the effect of progressively adding normal B1 history without
introducing an unrelated visual scene in the remaining prefix slots.

## Commands

Build manifests:

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_16_staircase_exp_v4.yaml \
  --dry-run

python scripts/build_sequence_manifests.py \
  --config configs/controlled_16_staircase_exp_v4.yaml \
  --overwrite
```

Build original + reverse manifests under the v4 output root:

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_16_staircase_exp_v4_with_reverse.yaml \
  --overwrite
```

Build original + reverse + X-to-B1 manifests under the same v4 output root:

```bash
python scripts/build_sequence_manifests.py \
  --config configs/controlled_16_staircase_exp_v4_with_x_to_B1.yaml \
  --overwrite
```

Validate manifests and write the aggregate staircase manifest:

```bash
python scripts/validate_staircase_v4.py \
  --config configs/controlled_16_staircase_exp_v4.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json
```

Validate reverse manifests:

```bash
python scripts/validate_staircase_v4.py \
  --config configs/controlled_16_staircase_exp_v4_with_reverse.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --setting controlled_16_staircase_reverse
```

Validate X-to-B1 manifests:

```bash
python scripts/validate_staircase_v4.py \
  --config configs/controlled_16_staircase_exp_v4_with_x_to_B1.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --setting controlled_16_staircase_x_to_B1

python scripts/validate_staircase_v4.py \
  --config configs/controlled_16_staircase_exp_v4_with_x_to_B1.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --setting controlled_16_staircase_x_to_B1_reversed
```

Extract Lingbot-map tokens:

```bash
python scripts/run_controlled_extraction.py \
  --config configs/controlled_16_staircase_exp_v4.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --setting controlled_16_staircase \
  --seq-len 16
```

Extract Lingbot-map tokens for the reverse setting:

```bash
python scripts/run_controlled_extraction.py \
  --config configs/controlled_16_staircase_exp_v4_with_reverse.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --setting controlled_16_staircase_reverse \
  --seq-len 16
```

Extract Lingbot-map tokens for X-to-B1:

```bash
python scripts/run_controlled_extraction.py \
  --config configs/controlled_16_staircase_exp_v4_with_x_to_B1.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --setting controlled_16_staircase_x_to_B1 \
  --seq-len 16

python scripts/run_controlled_extraction.py \
  --config configs/controlled_16_staircase_exp_v4_with_x_to_B1.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --setting controlled_16_staircase_x_to_B1_reversed \
  --seq-len 16
```

Optional geometry reconstruction:

```bash
python scripts/run_controlled_geometry.py \
  --config configs/controlled_16_staircase_exp_v4.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --setting controlled_16_staircase \
  --seq-len 16
```

Validate token shapes and the `H_15 ~= H_B1` endpoint after extraction:

```bash
python scripts/validate_staircase_v4.py \
  --config configs/controlled_16_staircase_exp_v4.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --check-tokens
```

Run analysis:

```bash
python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_16_staircase_exp_v4.yaml \
  --analysis-config configs/analysis_controlled_16_staircase_exp_v4.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --methods staircase_delta \
  --setting controlled_16_staircase \
  --seq-len 16
```

Run reverse analysis with contact-sheet-only PCA figures:

```bash
python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_16_staircase_exp_v4_with_reverse.yaml \
  --analysis-config configs/analysis_controlled_16_staircase_reverse_exp_v4.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --methods staircase_delta \
  --setting controlled_16_staircase_reverse \
  --seq-len 16
```

Run X-to-B1 analysis:

```bash
python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_16_staircase_exp_v4_with_x_to_B1.yaml \
  --analysis-config configs/analysis_controlled_16_staircase_x_to_B1_exp_v4.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --methods staircase_delta \
  --setting controlled_16_staircase_x_to_B1 \
  --seq-len 16

python scripts/run_controlled_analysis.py \
  --experiment-config configs/controlled_16_staircase_exp_v4_with_x_to_B1.yaml \
  --analysis-config configs/analysis_controlled_16_staircase_x_to_B1_reversed_exp_v4.yaml \
  --manifest-index outputs/manifests/controlled_16_staircase_exp_v4/index.json \
  --models lingbot-map \
  --methods staircase_delta \
  --setting controlled_16_staircase_x_to_B1_reversed \
  --seq-len 16
```

## Analysis Outputs

`staircase_delta` reads only the final-frame patch tokens for layers 4, 11, 17,
and 23. It computes:

- `Delta_X(k) = H_k - H_X`
- `Delta_B1(k) = H_k - H_B1`
- `Delta_step(k) = H_k - H_(k-1)`

For each layer and delta family, PCA is fitted once on all relevant k/patch
difference tokens. RGB normalization is also shared across k, and PCA signs are
stabilized by making the largest absolute loading positive.

Primary outputs:

- Shared PCA basis `.npz` files with mean, components, explained variance,
  RGB range, fitted k values, and sign flips.
- Per-k PCA RGB images, contact sheets, GIFs, RGB arrays, and PCA score arrays.
- Global and patch-aggregated norm/progress metrics.
- Stepwise marginal-change metrics.
- Object-level CSVs and plots from `instance-filt/600.png` when available.
- Layer-wise markdown summary.

Output roots:

```text
outputs/manifests/controlled_16_staircase_exp_v4/
outputs/controlled_sequences/controlled_16_staircase_exp_v4/
outputs/tokens/controlled_16_staircase_exp_v4/
outputs/figures/controlled_16_staircase_exp_v4/
outputs/metrics/controlled_16_staircase_exp_v4/
outputs/reconstruction_official/controlled_16_staircase_exp_v4/
```

Reverse outputs use the same roots and a separate setting directory:

```text
outputs/manifests/controlled_16_staircase_exp_v4/scene0002_00/controlled_16_staircase_reverse/
outputs/tokens/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_reverse/
outputs/figures/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_reverse/
outputs/metrics/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_reverse/
```

X-to-B1 outputs use the same v4 roots and separate setting directories:

```text
outputs/manifests/controlled_16_staircase_exp_v4/scene0002_00/controlled_16_staircase_x_to_B1/
outputs/manifests/controlled_16_staircase_exp_v4/scene0002_00/controlled_16_staircase_x_to_B1_reversed/
outputs/tokens/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_x_to_B1/
outputs/tokens/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_x_to_B1_reversed/
outputs/figures/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_x_to_B1/
outputs/figures/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_x_to_B1_reversed/
outputs/metrics/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_x_to_B1/
outputs/metrics/controlled_16_staircase_exp_v4/lingbot-map/scene0002_00/controlled_16_staircase_x_to_B1_reversed/
```
