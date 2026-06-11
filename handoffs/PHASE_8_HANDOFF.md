# Phase 8 Handoff

## Status

Interpretability/report-analysis assets completed from the generated embeddings, refreshed stability-aware features, rankings, and trained calibrated model.

## Previous handoffs checked

- `handoffs/PHASE_0_HANDOFF.md` through `handoffs/PHASE_7_HANDOFF.md`

## Commands run

- `python3 -m compileall -q src tests`
- `bash scripts/07_make_report_assets.sh`
- `.venv-pytest/bin/python -m pytest -q`
- reran `bash scripts/07_make_report_assets.sh` after calibrated model training
- reran `bash scripts/07_make_report_assets.sh` after MP metadata refresh and grouped validation

## Outputs produced

- `src/mataltmag_hybrid/analysis/embedding_plots.py`
- `src/mataltmag_hybrid/analysis/shap_analysis.py`
- `src/mataltmag_hybrid/analysis/ablations.py`
- `notebooks/05_interpretability.ipynb`
- `results/figures/embedding_umap_by_label.png`
- `results/figures/embedding_umap_by_space_group.png`
- `results/figures/shap_summary.png`
- `results/metrics/embedding_pca_projection.csv`
- `results/metrics/surrogate_gnn_explanation.csv`
- `results/metrics/hybrid_feature_importance.csv`
- `results/metrics/error_analysis.csv`

## Tests/checks passed

- Compile check passed.
- `.venv-pytest/bin/python -m pytest -q` passed: 10 tests.
- Report asset script completed successfully end to end.
- Hybrid feature importance supports the calibrated model and was regenerated.
- Current top feature-importance rows include `stoich_entropy`, `magnetic_elements`, `num_elements`, and `n_inequivalent_magnetic_sites`, followed by several embedding dimensions.

## Blocked gates or missing external inputs

- Current outputs are lightweight PCA/permutation/tree-importance analyses, not full SHAP or UMAP/t-SNE.
- DFT-dependent interpretation remains external.

## Next phase entrypoint

`bash scripts/07_make_report_assets.sh`
