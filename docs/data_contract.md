# Data Contract

This repository is a reproducible scaffold for MatAltMag-Hybrid. It must not fabricate scientific outputs. When required external files are absent, phase scripts stop with explicit validation errors.

## Raw MatAltMag Inputs

Expected under `data/raw/mataltmag/`:

- `label0.csv`
- `label1.csv`
- `candidate.csv`
- `Candidate_for_DFT_validate.csv`
- `output.csv` when available
- `atom_init.json`
- `cifs/<material_id>.cif`

## Canonical Dataset Index

`data/processed/dataset_index.parquet` columns:

- `material_id`
- `formula`
- `source_split`: `label0`, `label1`, `confirmed_50`, or `unconfirmed_candidate`
- `label`: `0`, `1`, or `unknown`
- `is_candidate`
- `gnn_probability`

All `material_id` values must be unique. Rows with `is_candidate=true` or `label=unknown` are excluded from supervised training.

## Ranked Candidate Output

`results/candidates/ranked_candidates.csv` must include both original and hybrid probabilities, DFT selection flags, and rationale. It is valid only after real candidate inputs, features, embeddings, and a trained model exist.
