#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
"${PYTHON:-python3}" -m mataltmag_hybrid.gnn.reproduce_inference --config configs/paths.yaml --mataltmag-root external/MatAltMag --out data/interim/gnn_outputs.csv
"${PYTHON:-python3}" -m mataltmag_hybrid.gnn.extract_embeddings --config configs/model.yaml --out data/interim/gnn_embeddings.parquet
