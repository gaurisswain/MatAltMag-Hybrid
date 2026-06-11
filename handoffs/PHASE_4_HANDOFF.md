# Phase 4 Handoff

## Status

Split generation completed on the assembled available-feature/embedding dataset.

## Previous handoffs checked

- `handoffs/PHASE_0_HANDOFF.md`
- `handoffs/PHASE_1_HANDOFF.md`
- `handoffs/PHASE_2_HANDOFF.md`
- `handoffs/PHASE_3_HANDOFF.md`

## Commands run

- `python3 -m compileall -q src tests`
- `bash scripts/04_train.sh`

## Outputs produced

- `src/mataltmag_hybrid/models/split.py`
- Split generation integrated into `src/mataltmag_hybrid/models/train_hybrid.py`.
- `data/processed/splits.json`

## Tests/checks passed

- Compile check passed.
- Candidate exclusion logic is covered by `tests/test_model_training.py`.
- Split has 19,034 train IDs and 6,345 test IDs.
- Candidate leakage into supervised training dataset is zero.

## Blocked gates or missing external inputs

- Full split/evaluation design is still scaffold-level; current training script reports train-set metrics only.

## Next phase entrypoint

`bash scripts/04_train.sh`
