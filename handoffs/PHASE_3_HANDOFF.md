# Phase 3 Handoff

## Status

Feature export completed with Materials Project stability metadata populated for every MP-returned structure. CIF inputs and MP summary rows are available for 67,128 rows; the 1,025 Materials Project unavailable rows are retained with unknown symmetry values and null stability fields.

## Previous handoffs checked

- `handoffs/PHASE_0_HANDOFF.md`
- `handoffs/PHASE_1_HANDOFF.md`
- `handoffs/PHASE_2_HANDOFF.md`

## Commands run

- `PYTHONPATH=src python3 -c "from mataltmag_hybrid.features.composition import composition_features; ..."`
- `PYTHONPATH=src python3 -m mataltmag_hybrid.io.download_cifs --config configs/paths.yaml --chunk-size 200`
- `PYTHONPATH=src python3 -m mataltmag_hybrid.io.download_mp_metadata --config configs/paths.yaml --out data/raw/materials_project/mp_summary.csv --chunk-size 500`
- `bash scripts/03_featurize.sh`
- `python3 -m compileall -q src tests`

## Outputs produced

- `src/mataltmag_hybrid/features/symmetry.py`
- `src/mataltmag_hybrid/features/composition.py`
- `src/mataltmag_hybrid/features/stability.py`
- `src/mataltmag_hybrid/features/featurize_all.py`
- `src/mataltmag_hybrid/features/validate_features.py`
- `src/mataltmag_hybrid/io/download_mp_metadata.py`
- `scripts/03_featurize.sh`
- `data/raw/materials_project/mp_summary.csv`
- `data/raw/mataltmag/cifs/*.cif` for 67,128 Materials Project structures
- `data/raw/mataltmag/cifs/download_manifest.csv`
- `data/raw/mataltmag/cifs/download_failures.csv`
- `data/interim/explicit_features.parquet`
- `data/processed/feature_schema.json`
- `results/metrics/feature_quality_report.csv`

## Tests/checks passed

- Composition fixture check for `Fe2O3` passed.
- Feature quality report utility is implemented.
- CIF coverage check: 67,128 CIFs for 68,153 dataset material IDs; no extra CIF IDs.
- `data/interim/explicit_features.parquet` has 68,153 rows, 23 columns, and unique material IDs.
- Feature table joins one-to-one with the dataset index for all 68,153 rows.
- Feature table joins one-to-one with embeddings for 67,128 rows.
- 1,025 rows have `crystal_system=unknown`, matching missing-CIF coverage.
- MP summary cache has 67,128 unique material IDs.
- Stability fields `formation_energy_per_atom`, `energy_above_hull`, `band_gap`, `density`, and `volume_per_atom` are non-null for 67,128 feature rows and null for the same 1,025 unavailable rows.
- `python3 -m compileall -q src tests` passed.

## Blocked gates or missing external inputs

- No Phase 3 blocker remains. The 1,025 missing rows are upstream MP/CIF coverage gaps and are documented in the generated artifacts.

## Next phase entrypoint

`bash scripts/04_train.sh`
