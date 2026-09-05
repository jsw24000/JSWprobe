#!/usr/bin/env bash
set -euo pipefail
DYNAMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
E1_PYTHON="${E1_PYTHON:-/home/3dsm/miniconda3/envs/repr_vggt_dinov3/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$E1_PYTHON" -m unittest discover -s "$DYNAMIC_ROOT/tests" -v
"$E1_PYTHON" "$DYNAMIC_ROOT/scripts/audit_e1.py" "$@"
"$E1_PYTHON" "$DYNAMIC_ROOT/scripts/extract_e1_features.py" --stage smoke "$@"
"$E1_PYTHON" "$DYNAMIC_ROOT/scripts/validate_extraction.py" --stage smoke "$@"
"$E1_PYTHON" "$DYNAMIC_ROOT/scripts/extract_e1_features.py" --stage full "$@"
"$E1_PYTHON" "$DYNAMIC_ROOT/scripts/validate_extraction.py" --stage full "$@"
"$E1_PYTHON" "$DYNAMIC_ROOT/scripts/check_forward_fidelity.py" "$@"
"$E1_PYTHON" "$DYNAMIC_ROOT/scripts/run_e1_analysis.py" "$@"
"$E1_PYTHON" "$DYNAMIC_ROOT/scripts/build_e1_report.py" "$@"
