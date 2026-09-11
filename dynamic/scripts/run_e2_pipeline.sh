#!/usr/bin/env bash
set -euo pipefail
DYNAMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E2_PYTHON="${E2_PYTHON:-/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python}"
CONFIG="$DYNAMIC_ROOT/configs/e2_full_v2.yaml"
PANEL=""
STAGE="full"
EXTRA=()
while (($#)); do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --panel) PANEL="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
if [[ -z "$PANEL" ]]; then echo "--panel is required" >&2; exit 2; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$E2_PYTHON" -m unittest discover -s "$DYNAMIC_ROOT/tests" -v
"$E2_PYTHON" "$DYNAMIC_ROOT/scripts/plan_e2.py" --config "$CONFIG" --panel "$PANEL" "${EXTRA[@]}"
"$E2_PYTHON" "$DYNAMIC_ROOT/scripts/audit_e2.py" --config "$CONFIG" "${EXTRA[@]}"
if [[ "$STAGE" == "full" ]]; then
  "$E2_PYTHON" "$DYNAMIC_ROOT/scripts/extract_e2_features.py" --config "$CONFIG" --panel "$PANEL" --stage smoke "${EXTRA[@]}"
  "$E2_PYTHON" "$DYNAMIC_ROOT/scripts/validate_e2_extraction.py" --config "$CONFIG" --panel "$PANEL" --stage smoke "${EXTRA[@]}"
fi
"$E2_PYTHON" "$DYNAMIC_ROOT/scripts/extract_e2_features.py" --config "$CONFIG" --panel "$PANEL" --stage "$STAGE" "${EXTRA[@]}"
"$E2_PYTHON" "$DYNAMIC_ROOT/scripts/validate_e2_extraction.py" --config "$CONFIG" --panel "$PANEL" --stage "$STAGE" "${EXTRA[@]}"
if [[ "$STAGE" == "full" ]]; then
  "$E2_PYTHON" "$DYNAMIC_ROOT/scripts/run_e2_analysis.py" --config "$CONFIG" --panel "$PANEL" "${EXTRA[@]}"
  "$E2_PYTHON" "$DYNAMIC_ROOT/scripts/build_e2_report.py" --config "$CONFIG" --panels "$PANEL" "${EXTRA[@]}"
fi
