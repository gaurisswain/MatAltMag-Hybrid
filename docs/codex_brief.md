# Codex Brief: Build MatAltMag-Hybrid

You are implementing a Python research codebase called `mataltmag-hybrid`. The goal is to extend Gao et al.'s MatAltMag model with an explicit feature stream and a calibrated ensemble classifier.

## Non-negotiable requirements

1. Do not edit `external/MatAltMag` directly. Wrap it.
2. Use the trained MatAltMag model as a frozen feature extractor.
3. Export original GNN probabilities and frozen embeddings for every material.
4. Build explicit symmetry/composition/stability features from CIFs and Materials Project metadata.
5. Train and evaluate these models:
   - original GNN probability only
   - explicit features only
   - GNN embedding only
   - hybrid concatenation
   - calibrated hybrid ensemble
6. Never train on the ~300 unconfirmed candidates.
7. Produce `results/candidates/ranked_candidates.csv` with both original and hybrid probabilities.
8. Generate DFT input folders for the top 2–3 candidates.
9. Include tests for feature shape, ID alignment, deterministic embeddings, and no candidate leakage.

## Start by creating this structure

```text
configs/
data/raw/
data/interim/
data/processed/
external/MatAltMag/
src/mataltmag_hybrid/
scripts/
results/metrics/
results/figures/
results/candidates/
results/dft/
tests/
notebooks/
report/
```

## First files to implement

1. `src/mataltmag_hybrid/config.py`
2. `src/mataltmag_hybrid/io/load_mataltmag.py`
3. `src/mataltmag_hybrid/gnn/adapter.py`
4. `src/mataltmag_hybrid/gnn/reproduce_inference.py`
5. `src/mataltmag_hybrid/gnn/extract_embeddings.py`
6. `src/mataltmag_hybrid/features/symmetry.py`
7. `src/mataltmag_hybrid/features/composition.py`
8. `src/mataltmag_hybrid/features/stability.py`
9. `src/mataltmag_hybrid/models/train_hybrid.py`
10. `src/mataltmag_hybrid/models/predict_candidates.py`

## Data schemas

### GNN outputs

```text
material_id: str
gnn_probability: float
gnn_rank: int
source_file: str
```

### Embeddings

```text
material_id: str
emb_000: float
emb_001: float
...
```

### Explicit features

```text
material_id: str
space_group_number: int
crystal_system: str
point_group: str
has_inversion: bool
n_symmetry_ops: int
n_rotation_2: int
n_rotation_3: int
n_rotation_4: int
n_rotation_6: int
n_mirror_ops: int
n_magnetic_sites: int
n_inequivalent_magnetic_sites: int
magnetic_elements: str
tm_3d_fraction: float
rare_earth_4f_fraction: float
num_elements: int
stoich_entropy: float
formation_energy_per_atom: float
energy_above_hull: float
band_gap: float
density: float
volume_per_atom: float
```

### Ranked candidates

```text
rank_hybrid: int
material_id: str
formula: str
space_group_number: int
crystal_system: str
gnn_probability: float
hybrid_probability: float
hybrid_probability_std: float
explicit_only_probability: float
embedding_only_probability: float
formation_energy_per_atom: float
energy_above_hull: float
band_gap: float
n_magnetic_sites: int
magnetic_elements: str
dft_priority_score: float
dft_selected: bool
rationale: str
```

## Modeling guidance

Use scikit-learn first. Avoid overengineering.

Recommended models:

- `LogisticRegression(class_weight="balanced")`
- `ExtraTreesClassifier(class_weight="balanced")`
- `HistGradientBoostingClassifier()`
- optional small PyTorch MLP

Use calibrated probabilities:

- `CalibratedClassifierCV(method="sigmoid")` for small validation sets.
- Isotonic only if enough positives exist in calibration folds.

Metrics:

- AUROC
- AUPRC
- precision@K
- recall@K
- enrichment@K
- balanced accuracy
- Brier score
- expected calibration error

## DFT input generation

Generate VASP folders only; do not include POTCAR.

For each candidate:

```text
relax/
static/
bands/
analysis/
metadata.yaml
```

Use spin-polarized no-SOC settings for validation. Initialize AFM patterns using alternating MAGMOM values on magnetic atoms. Band calculations should use a converged charge density from the static calculation and line-mode KPOINTS.

## Testing requirements

Implement tests for:

1. All material IDs are unique.
2. Candidate materials never appear in supervised training splits.
3. Feature tables join one-to-one with dataset index.
4. No all-null or all-constant features are used for training.
5. Embedding extraction returns deterministic values for fixed input and checkpoint.
6. Ranking output contains required columns.
7. DFT folder generator does not write or require POTCAR.

## Final command target

The project should eventually run as:

```bash
bash scripts/01_prepare_data.sh
bash scripts/02_extract_gnn.sh
bash scripts/03_featurize.sh
bash scripts/04_train.sh
bash scripts/05_rank_candidates.sh
bash scripts/06_make_dft_inputs.sh
bash scripts/07_make_report_assets.sh
```
