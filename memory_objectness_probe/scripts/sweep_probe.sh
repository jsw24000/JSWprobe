#!/usr/bin/env bash
set -euo pipefail

FEATURE_MANIFEST="${FEATURE_MANIFEST:-outputs/features/feature_manifest.jsonl}"
SPLIT_FILE="${SPLIT_FILE:-outputs/splits/scene_split_seed0.json}"
LAYERS="${LAYERS:-block04 block11 block17 block23}"
TASKS="${TASKS:-state5 presence position4}"
MODES="${MODES:-all6_mean all6_flatten}"
MODEL="${MODEL:-linear}"
PCA_DIM="${PCA_DIM:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"

for task in ${TASKS}; do
  for layer in ${LAYERS}; do
    for mode in ${MODES}; do
      extra=()
      if [[ "${mode}" == "all6_flatten" ]]; then
        extra=(--pca_dim "${PCA_DIM}")
      else
        extra=(--pca_dim none)
      fi
      conda run -n lingbot-map python scripts/train_probe.py \
        --feature_manifest "${FEATURE_MANIFEST}" \
        --split_file "${SPLIT_FILE}" \
        --task "${task}" \
        --layer "${layer}" \
        --feature_mode "${mode}" \
        --model "${MODEL}" \
        "${extra[@]}" \
        --max_epochs "${MAX_EPOCHS}" \
        --early_stop_patience "${PATIENCE}" \
        --output_dir "outputs/probe_runs/${task}_${layer}_${mode}_${MODEL}"
    done

    for slot in 0 1 2 3 4 5; do
      conda run -n lingbot-map python scripts/train_probe.py \
        --feature_manifest "${FEATURE_MANIFEST}" \
        --split_file "${SPLIT_FILE}" \
        --task "${task}" \
        --layer "${layer}" \
        --feature_mode single_slot \
        --slot_index "${slot}" \
        --model "${MODEL}" \
        --pca_dim none \
        --max_epochs "${MAX_EPOCHS}" \
        --early_stop_patience "${PATIENCE}" \
        --output_dir "outputs/probe_runs/${task}_${layer}_slot${slot}_${MODEL}"
    done

    for group in camera register scale; do
      conda run -n lingbot-map python scripts/train_probe.py \
        --feature_manifest "${FEATURE_MANIFEST}" \
        --split_file "${SPLIT_FILE}" \
        --task "${task}" \
        --layer "${layer}" \
        --feature_mode slot_group \
        --slot_group "${group}" \
        --model "${MODEL}" \
        --pca_dim none \
        --max_epochs "${MAX_EPOCHS}" \
        --early_stop_patience "${PATIENCE}" \
        --output_dir "outputs/probe_runs/${task}_${layer}_${group}_${MODEL}"
    done
  done
done
