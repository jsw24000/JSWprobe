#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/noise_slot_svd_debias.yaml}
PYTHON=${PYTHON:-python}

$PYTHON scripts/generate_noise_slot_samples.py --config "$CONFIG"
$PYTHON scripts/estimate_noise_slot_subspace.py --config "$CONFIG"
$PYTHON scripts/visualize_noise_slot_basis.py --config "$CONFIG"
$PYTHON scripts/visualize_correspondence_noise_debiased.py --config "$CONFIG"
$PYTHON scripts/evaluate_noise_debiased_correspondence.py --config "$CONFIG"
$PYTHON scripts/make_noise_slot_debias_summary.py --config "$CONFIG"

echo "[noise-slot] Done. See outputs/<scene>/05_noise_slot_svd_debias/06_summary/debias_summary.md"
