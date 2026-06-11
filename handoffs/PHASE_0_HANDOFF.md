# Phase 0 Handoff

## Status

Submitted with scaffold complete. Re-run on 2026-06-02 completed after downloading the upstream MatAltMag GitHub repository and staging its raw CSV files.

## Commands run

- `mkdir -p ...` for the required repository skeleton.
- `cp /Users/ashutoshjaiswal/Downloads/MatAltMag_Hybrid_README.md README.md`
- `cp /Users/ashutoshjaiswal/Downloads/MatAltMag_Hybrid_Codex_Brief.md docs/codex_brief.md`
- `cp /Users/ashutoshjaiswal/Downloads/MatAltMag_Hybrid_Phasewise_Plan.md docs/phasewise_plan.md`
- `chmod +x scripts/*.sh`
- `python3 -m compileall -q src tests`
- `bash scripts/01_prepare_data.sh`
- `bash scripts/07_make_report_assets.sh`
- `bash scripts/01_prepare_data.sh` on 2026-06-02
- `python3 -m compileall -q src tests` on 2026-06-02
- `python3 -m pytest -q` on 2026-06-02
- `python3 -m venv --system-site-packages .venv-pytest` on 2026-06-03
- `.venv-pytest/bin/python -m pip install pytest` on 2026-06-03
- `.venv-pytest/bin/python -m pytest -q` on 2026-06-03
- `git clone https://github.com/zfgao66/MatAltMag.git external/MatAltMag` on 2026-06-02
- copied GitHub-provided `label0.csv`, `label1.csv`, `candidate.csv`, `atom_init.json`, `out/output.csv`, and `out/Candidate_for_DFT_validate.csv` into `data/raw/mataltmag/`
- `bash scripts/01_prepare_data.sh` on 2026-06-02 after staging GitHub data

## Outputs produced

- Repository skeleton under `/Users/ashutoshjaiswal/mataltmag-hybrid`.
- `README.md`
- `docs/codex_brief.md`
- `docs/phasewise_plan.md`
- `docs/data_contract.md`
- `pyproject.toml`
- `environment.yml`
- `configs/paths.yaml`
- `configs/model.yaml`
- `configs/features.yaml`
- `configs/train.yaml`
- `configs/dft.yaml`
- Python package under `src/mataltmag_hybrid/`.
- Phase scripts under `scripts/`.
- Tests under `tests/`.
- Minimal valid notebook placeholders under `notebooks/`.
- `report/reproducibility_checklist.md`
- `data/processed/dataset_index.parquet`

## Tests/checks passed

- `python3 -m compileall -q src tests` passed.
- Deterministic fixture embedding direct check passed with `PYTHONPATH=src`.
- Composition feature direct check passed with `PYTHONPATH=src`.
- `bash scripts/07_make_report_assets.sh` passed.
- `python3 -m compileall -q src tests` passed on 2026-06-02.
- `data/processed/dataset_index.parquet` has 68,153 rows with unique `material_id` values.
- Candidate rows are marked `label=unknown`; no candidate row has a supervised label.
- `.venv-pytest/bin/python -m pytest -q` passed on 2026-06-03: 10 tests passed.

## Tests/checks not run

- System `python3 -m pytest -q` is not used because Homebrew Python is externally managed; pytest is installed in local `.venv-pytest`.

## Blocked gates or missing external inputs

- CIF files are still absent under `data/raw/mataltmag/cifs/`; later feature phases remain blocked until they are downloaded or provided.
- Trained checkpoint files are not included in the GitHub checkout.

## Next phase entrypoint

`bash scripts/02_extract_gnn.sh`
