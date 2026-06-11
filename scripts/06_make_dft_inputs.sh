#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" -m mataltmag_hybrid.dft.make_vasp_inputs --config configs/dft.yaml
