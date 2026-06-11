# Phase 5 Handoff

## Status

Training and held-out model comparison completed on the available-feature/embedding intersection, including both stratified and chemical-system grouped validation.

## Previous handoffs checked

- `handoffs/PHASE_0_HANDOFF.md` through `handoffs/PHASE_4_HANDOFF.md`

## Commands run

- `python3 -m compileall -q src tests`
- `bash scripts/04_train.sh`
- `bash scripts/05_rank_candidates.sh`
- `bash scripts/07_make_report_assets.sh`
- `.venv-pytest/bin/python -m pytest -q`

## Outputs produced

- `src/mataltmag_hybrid/models/train_hybrid.py`
- `configs/train.yaml`
- `scripts/04_train.sh`
- `results/models/latest/hybrid_model.joblib`
- `results/metrics/model_comparison.csv`
- `results/metrics/cv_predictions.parquet`
- `data/processed/train_dataset.parquet`
- `data/processed/candidate_dataset.parquet`

## Tests/checks passed

- Compile check passed.
- Training code enforces one-to-one joins and candidate exclusion through dataset assembly and split helpers.
- Train dataset has 25,379 supervised rows: 25,258 negatives and 121 positives.
- Candidate dataset has 41,749 unknown candidate rows.
- Candidate leakage into train dataset is zero.
- Held-out model comparison rows exist for:
  - `gnn_probability_only`
  - `explicit_features_only`
  - `gnn_embedding_only`
  - `hybrid_concatenation`
  - `calibrated_hybrid_ensemble`
- The same five model rows are reported for both `stratified` and `chemical_system_grouped` splits.
- `results/metrics/cv_predictions.parquet` has 63,445 held-out predictions with a `split` column covering both validation modes.
- Chemical-system grouped split test set has 6,344 rows with 30 positives.
- Grouped split metrics: calibrated hybrid AUROC/AUPRC/balanced accuracy are 1.0/1.0/1.0; embedding-only AUROC/AUPRC/balanced accuracy are 0.9831/0.9668/0.95.
- `.venv-pytest/bin/python -m pytest -q` passed: 10 tests.

## Blocked gates or missing external inputs

- Metrics remain unusually high even under chemical-system grouped validation. Treat final scientific claims cautiously until the user completes external DFT validation and any desired independent external-data audit.

## Next phase entrypoint

`bash scripts/05_rank_candidates.sh`
