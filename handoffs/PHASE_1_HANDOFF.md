# Phase 1 Handoff

## Status

Completed reproduction from the upstream GitHub-provided `out/output.csv`. No fabricated GNN probabilities were produced.

## Previous handoffs checked

- `handoffs/PHASE_0_HANDOFF.md`

## Commands run

- `bash scripts/01_prepare_data.sh`
- `git clone https://github.com/zfgao66/MatAltMag.git external/MatAltMag`
- staged upstream raw CSVs and `atom_init.json` into `data/raw/mataltmag/`
- `bash scripts/02_extract_gnn.sh`

## Outputs produced

- `data/interim/gnn_outputs.csv`
- updated `data/processed/dataset_index.parquet` with `gnn_probability`

## Tests/checks passed

- `data/interim/gnn_outputs.csv` has 42,524 rows and unique `material_id` values.
- `data/processed/dataset_index.parquet` has 68,153 rows and unique `material_id` values.
- Candidate rows remain excluded from supervised labels.
- Regression compile check from Phase 0 passed: `python3 -m compileall -q src tests`.

## Blocked gates or missing external inputs

- `bash scripts/02_extract_gnn.sh` now advances past reproduction but blocks in Phase 2 because `external/MatAltMag/checkpoints/model.pt` is missing.
- Upstream GitHub does not include CIF files or trained checkpoint weights; the README points to Materials Project download for CIFs and Google Drive for weights.

## Next phase entrypoint

`bash scripts/02_extract_gnn.sh`
