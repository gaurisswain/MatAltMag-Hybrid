# Phase 6 Handoff

## Status

Candidate ranking completed on the available-feature/embedding candidate dataset and refreshed after calibrated model training plus Materials Project stability metadata.

## Previous handoffs checked

- `handoffs/PHASE_0_HANDOFF.md` through `handoffs/PHASE_5_HANDOFF.md`

## Commands run

- `python3 -m compileall -q src tests`
- `bash scripts/05_rank_candidates.sh`
- `bash scripts/06_make_dft_inputs.sh`

## Outputs produced

- `src/mataltmag_hybrid/models/predict_candidates.py`
- `scripts/05_rank_candidates.sh`
- Required ranking CSV column list in code.
- `results/candidates/ranked_candidates.csv`
- `results/candidates/top_candidates_for_dft.csv`
- `results/candidates/top20_candidate_cards.md`

## Tests/checks passed

- Compile check passed.
- Required ranking columns are covered by `tests/test_candidate_ranking.py`.
- Ranking CSV has 41,749 rows and all required columns.
- Top DFT shortlist has 3 selected rows.
- Ranking was regenerated after saving the calibrated hybrid model and refreshing explicit features with `energy_above_hull`.
- Current top DFT shortlist is `mp-1205353`, `mp-6294`, and `mp-562216`.
- All three selected rows have non-null `formation_energy_per_atom`, `energy_above_hull`, and `band_gap`.

## Blocked gates or missing external inputs

- Ranking still uses placeholder identical values for explicit-only, embedding-only, and uncertainty fields; candidate-specific uncertainty should be implemented before final scientific reporting if uncertainty is a claimed result.

## Next phase entrypoint

`bash scripts/06_make_dft_inputs.sh`
