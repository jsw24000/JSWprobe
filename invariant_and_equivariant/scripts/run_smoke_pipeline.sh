#!/usr/bin/env bash
set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi
FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

CONFIG="configs/smoke.yaml"
RUN_DIR="outputs/smoke_v1"
LOG_DIR="${RUN_DIR}/logs"
ENV_DIR="${RUN_DIR}/environment"
STAMP_DIR="${RUN_DIR}/.stamps"

mkdir -p "${LOG_DIR}" "${ENV_DIR}" "${STAMP_DIR}" "${RUN_DIR}/figures" "${RUN_DIR}/tables"
cp "${CONFIG}" "${RUN_DIR}/config_used.yaml"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

if [[ ! -f "${ENV_DIR}/pip_freeze_before.txt" || "${FORCE}" == "1" ]]; then
  pip freeze > "${ENV_DIR}/pip_freeze_before.txt"
fi

run_step() {
  local name="$1"
  local marker="$2"
  shift 2
  if [[ "${FORCE}" == "0" && -f "${marker}" ]]; then
    echo "[skip] ${name}"
    return 0
  fi
  echo "[run] ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  touch "${marker}"
}

run_step inspect_models "${STAMP_DIR}/inspect_models.done" \
  python scripts/inspect_models.py --config "${CONFIG}"

run_step audit_dataset "${STAMP_DIR}/audit_dataset.done" \
  python scripts/audit_dataset.py --config "${CONFIG}"

run_step extract_dinov2 "${STAMP_DIR}/extract_dinov2.done" \
  python scripts/extract_features.py --config "${CONFIG}" --model dinov2 "${FORCE_ARGS[@]}"

run_step extract_vggt "${STAMP_DIR}/extract_vggt.done" \
  python scripts/extract_features.py --config "${CONFIG}" --model vggt "${FORCE_ARGS[@]}"

run_step absolute_position "${STAMP_DIR}/absolute_position.done" \
  python scripts/run_absolute_position.py --config "${CONFIG}"

run_step delta_decode "${STAMP_DIR}/delta_decode.done" \
  python scripts/run_delta_decode.py --config "${CONFIG}"

run_step decode_composition "${STAMP_DIR}/decode_composition.done" \
  python scripts/run_decode_composition.py --config "${CONFIG}"

run_step forward_equivariance "${STAMP_DIR}/forward_equivariance.done" \
  python scripts/run_forward_equivariance.py --config "${CONFIG}"

run_step homogeneity "${STAMP_DIR}/homogeneity.done" \
  python scripts/run_homogeneity.py --config "${CONFIG}"

run_step subspace_analysis "${STAMP_DIR}/subspace_analysis.done" \
  python scripts/run_subspace_analysis.py --config "${CONFIG}"

run_step intervention "${STAMP_DIR}/intervention.done" \
  python scripts/run_intervention.py --config "${CONFIG}"

run_step composition "${STAMP_DIR}/composition.done" \
  python scripts/run_composition.py --config "${CONFIG}"

run_step build_report "${STAMP_DIR}/build_report.done" \
  python scripts/build_report.py --run-dir "${RUN_DIR}"

pip freeze > "${ENV_DIR}/pip_freeze_after.txt"
echo "[done] smoke outputs are in ${RUN_DIR}"
