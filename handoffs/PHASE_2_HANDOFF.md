# Phase 2 Handoff

## Status

Completed for the Materials Project available-CIF subset. The 1,025 dataset material IDs without CIFs are explicitly excluded and recorded; no synthetic embeddings were generated.

## Previous handoffs checked

- `handoffs/PHASE_0_HANDOFF.md`
- `handoffs/PHASE_1_HANDOFF.md`

## Commands run

- `PYTHONPATH=src python3 -c "from mataltmag_hybrid.gnn.adapter import MatAltMagEmbeddingExtractor; ..."`
- `bash scripts/02_extract_gnn.sh`
- `PYTHONPATH=/private/tmp/mataltmag_gdown python3 -m gdown --folder 'https://drive.google.com/drive/folders/1Dbb3u-_LGZ8trq1w4o173GWeGjtksghx?usp=sharing' --output external/MatAltMag/checkpoints`
- `PYTHONPATH=/private/tmp/mataltmag_gdown python3 -m gdown --folder 'https://drive.google.com/drive/folders/1xYQrIfC71z-IlD33hkTdunTsUgMLb_hA?usp=drive_link' --output external/MatAltMag/checkpoints/drive_link_checkpoint`
- `PYTHONPATH=src python3 -m mataltmag_hybrid.io.download_cifs --config configs/paths.yaml --chunk-size 200`
- `PYTHONPATH=src python3 -m mataltmag_hybrid.io.download_cifs --config configs/paths.yaml --retry-missing` was attempted and interrupted after repeated zero-document responses
- `bash scripts/02_extract_gnn.sh`
- `python3 -m compileall -q src tests`
- `python3 -m pytest -q tests/test_gnn_adapter.py`
- `.venv-pytest/bin/python -m pytest -q`

## Outputs produced

- `src/mataltmag_hybrid/gnn/adapter.py`
- `src/mataltmag_hybrid/gnn/reproduce_inference.py`
- `src/mataltmag_hybrid/gnn/extract_embeddings.py`
- `scripts/02_extract_gnn.sh`
- `data/interim/gnn_outputs.csv` from the reproduction step before embedding extraction blocked
- `external/MatAltMag/checkpoints/checkpoint-27/pytorch_model.bin`
- `external/MatAltMag/checkpoints/checkpoint-27/optimizer.bin`
- `external/MatAltMag/checkpoints/checkpoint-27/scheduler.bin`
- `external/MatAltMag/checkpoints/checkpoint-27/random_states_0.pkl`
- `data/raw/mataltmag/cifs/download_manifest.csv`
- real upstream encoder extraction wiring in `src/mataltmag_hybrid/gnn/adapter.py`
- `configs/model.yaml` now points to `external/MatAltMag/checkpoints/checkpoint-27/pytorch_model.bin`
- `data/interim/gnn_embeddings.parquet`
- `data/interim/gnn_embedding_parts/part_*.parquet`
- `results/metrics/embedding_export_check.json`

## Tests/checks passed

- Deterministic fixture embedding check passed.
- Public module entrypoints exist:
  - `python -m mataltmag_hybrid.gnn.reproduce_inference`
  - `python -m mataltmag_hybrid.gnn.extract_embeddings`
- Reproduction step produced 42,524 GNN probability rows with unique material IDs.
- Real checkpoint smoke test produced a `(2, 513)` table: `material_id` plus 512 encoder embedding columns.
- `data/interim/gnn_embeddings.parquet` has 67,128 rows, 513 columns, unique material IDs, and 512 `emb_*` columns.
- `results/metrics/embedding_export_check.json` records 68,153 requested rows, 67,128 exported rows, and 1,025 missing-CIF rows.
- `python3 -m compileall -q src tests` passed.
- `.venv-pytest/bin/python -m pytest -q` passed on 2026-06-03: 10 tests passed.

## Blocked gates or missing external inputs

- Materials Project returned 67,128 CIFs for 68,153 dataset material IDs; 1,025 IDs were not returned and are listed in `data/raw/mataltmag/cifs/download_failures.csv`.
- System `python3 -m pytest -q` remains unavailable because Homebrew Python is externally managed; pytest is installed in local `.venv-pytest`.

## Next phase entrypoint

`bash scripts/03_featurize.sh`
