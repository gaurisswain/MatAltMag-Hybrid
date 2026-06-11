# MatAltMag Hybrid Candidate Screening Report

## Scope

This report summarizes the non-DFT portion of the MatAltMag-Hybrid workflow. DFT validation is intentionally left for external execution. The active DFT shortlist is defined by `results/candidates/top_candidates_for_dft.csv`.

## Data Status

The dataset index contains 68,153 unique Materials Project rows. Supervised training uses 25,379 labeled rows: 25,258 negatives and 121 positives. The unknown candidate pool contains 41,749 rows and is excluded from supervised training and validation.

MatAltMag CIF and MP summary coverage is available for 67,128 materials. The remaining 1,025 rows were not returned by the Materials Project/CIF retrieval path and are retained with `crystal_system=unknown` and null stability fields. The MP summary cache populates `formation_energy_per_atom`, `energy_above_hull`, `band_gap`, `density`, and `volume_per_atom` for all 67,128 returned rows.

## Feature Streams

The frozen MatAltMag encoder exports 512-dimensional embeddings from the checkpoint at `external/MatAltMag/checkpoints/checkpoint-27/pytorch_model.bin`. The explicit feature stream adds composition, symmetry, magnetic-element, and MP stability descriptors. The feature table has 68,153 rows and 23 columns.

## Validation

Model comparison was regenerated for two validation modes: a stratified held-out split and a chemical-system grouped holdout. Both splits keep unknown candidates out of supervised evaluation.

| Split | Model | AUROC | AUPRC | Balanced Accuracy | Precision@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| stratified | gnn_probability_only | 0.5000 | 0.0047 | 0.5000 | 0.00 |
| stratified | explicit_features_only | 1.0000 | 1.0000 | 1.0000 | 0.60 |
| stratified | gnn_embedding_only | 1.0000 | 0.9930 | 0.9833 | 0.60 |
| stratified | hybrid_concatenation | 1.0000 | 1.0000 | 1.0000 | 0.60 |
| stratified | calibrated_hybrid_ensemble | 1.0000 | 1.0000 | 1.0000 | 0.60 |
| chemical_system_grouped | gnn_probability_only | 0.5000 | 0.0047 | 0.5000 | 0.00 |
| chemical_system_grouped | explicit_features_only | 1.0000 | 1.0000 | 0.9833 | 0.60 |
| chemical_system_grouped | gnn_embedding_only | 0.9831 | 0.9668 | 0.9500 | 0.58 |
| chemical_system_grouped | hybrid_concatenation | 1.0000 | 1.0000 | 1.0000 | 0.60 |
| chemical_system_grouped | calibrated_hybrid_ensemble | 1.0000 | 1.0000 | 1.0000 | 0.60 |

The unusually high explicit-feature and hybrid scores should be treated cautiously. They survive the grouped split implemented here, but final scientific claims should still be tied to external DFT validation and, if needed, an independent external-data audit.

## Candidate Ranking

The refreshed ranking contains 41,749 unknown candidates. The ranking score combines calibrated hybrid probability and MP stability through `energy_above_hull`.

The current top three DFT candidates are:

| Rank | Material ID | Formula | Hybrid Probability | Energy Above Hull | Band Gap | Magnetic Elements |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | mp-1205353 | K4 Mn4 F12 | 0.9985 | 0.0000 | 2.4058 | Mn |
| 2 | mp-6294 | Sr4 Er2 Ru2 O12 | 0.9983 | 0.0000 | 0.0000 | Er |
| 3 | mp-562216 | Ba8 Co4 Si8 O28 | 0.9981 | 0.0014 | 1.7828 | Co |

DFT input skeletons for these rows are present under `results/dft/candidate_001_mp-1205353/`, `results/dft/candidate_002_mp-6294/`, and `results/dft/candidate_003_mp-562216/`. POTCAR files are intentionally not generated.

## Interpretability Assets

Generated assets include embedding PCA projections, label and space-group projection figures, surrogate GNN explanation, candidate error/rank-shift analysis, and hybrid feature importance. The top feature-importance rows include stoichiometric entropy, magnetic-element identity, number of elements, inequivalent magnetic sites, and several frozen MatAltMag embedding dimensions.

## Reproducibility Checks

The current code passes:

- `.venv-pytest/bin/python -m pytest -q`: 10 tests passed.
- `python3 -m compileall -q src tests`: passed.

The reproducibility checklist is in `report/reproducibility_checklist.md`.
