#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" -m mataltmag_hybrid.models.predict_candidates --config configs/train.yaml --out results/candidates/ranked_candidates.csv --top-n 3
