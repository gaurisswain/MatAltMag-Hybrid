#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" -m mataltmag_hybrid.io.prepare_data --config configs/paths.yaml
