# Phase 7 Handoff

## Status

DFT input-generation completed for the current top three ranked candidates. DFT execution and validation are external per user instruction.

## Previous handoffs checked

- `handoffs/PHASE_0_HANDOFF.md` through `handoffs/PHASE_6_HANDOFF.md`

## Commands run

- `python3 -m compileall -q src tests`
- `bash scripts/06_make_dft_inputs.sh`

## Outputs produced

- `src/mataltmag_hybrid/dft/make_vasp_inputs.py`
- `src/mataltmag_hybrid/dft/select_candidates.py`
- `scripts/06_make_dft_inputs.sh`
- `results/dft/candidate_001_mp-1205353/`
- `results/dft/candidate_002_mp-6294/`
- `results/dft/candidate_003_mp-562216/`

## Tests/checks passed

- Compile check passed.
- DFT generator code writes `INCAR`, `KPOINTS`, `metadata.yaml`, and analysis README only.
- `POTCAR` is neither required nor generated.
- Verified no `POTCAR` exists under `results/dft`.
- Previous generated DFT folders from older ranking runs may still exist under `results/dft`; use `results/candidates/top_candidates_for_dft.csv` as the active shortlist.

## Blocked gates or missing external inputs

- VASP calculations have not been run here; only input folders were generated. The user will complete DFT validation externally.

## Next phase entrypoint

`bash scripts/07_make_report_assets.sh`
