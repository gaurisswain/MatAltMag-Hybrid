#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" -m mataltmag_hybrid.features.featurize_all --config configs/features.yaml --out data/interim/explicit_features.parquet
