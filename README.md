# MatAltMag-Hybrid: Hybrid Altermagnet Classifier

A reproducible research codebase for extending Gao et al.'s MatAltMag model with explicit symmetry and composition features, re-ranking unconfirmed altermagnet candidates, and preparing top candidates for DFT validation.

## 1. Project goal

Build a hybrid altermagnet classifier that combines:

1. A frozen/pre-trained MatAltMag GNN encoder used as a black-box crystal-structure feature extractor.
2. Explicit handcrafted features from crystal symmetry, magnetic-element composition, stoichiometry, and Materials Project stability metadata.

The model outputs two scores for each unconfirmed candidate:

- `gnn_probability`: original MatAltMag probability.
- `hybrid_probability`: ensemble probability from frozen GNN embedding plus explicit features.

The final deliverable is a ranked candidate CSV and DFT validation package for the top two or three candidates.

## 2. Scientific objective

The project tests whether explicit physics-inspired descriptors improve few-shot altermagnet discovery beyond the published GNN classifier. If the hybrid model improves held-out performance, use it to prioritize candidates for DFT. If it does not improve performance, pivot to interpretability: explain what the published GNN has learned using embeddings, surrogate models, SHAP/permutation importance, and chemical-family analysis.

## 3. Repository layout

```text
mataltmag-hybrid/
├── README.md
├── environment.yml
├── pyproject.toml
├── configs/
│   ├── paths.yaml
│   ├── model.yaml
│   ├── features.yaml
│   ├── train.yaml
│   └── dft.yaml
├── external/
│   └── MatAltMag/                    # cloned Gao et al. repository; not edited directly
├── data/
│   ├── raw/
│   │   ├── mataltmag/                # label0.csv, label1.csv, candidate.csv, CIFs, atom_init.json
│   │   └── materials_project/        # MP summary metadata cache
│   ├── interim/
│   │   ├── gnn_outputs.csv
│   │   ├── gnn_embeddings.parquet
│   │   ├── explicit_features.parquet
│   │   └── dataset_index.parquet
│   └── processed/
│       ├── train_dataset.parquet
│       ├── candidate_dataset.parquet
│       └── feature_schema.json
├── src/
│   └── mataltmag_hybrid/
│       ├── __init__.py
│       ├── config.py
│       ├── io/
│       │   ├── load_mataltmag.py
│       │   ├── materials_project.py
│       │   └── cache.py
│       ├── gnn/
│       │   ├── adapter.py
│       │   ├── extract_embeddings.py
│       │   └── reproduce_inference.py
│       ├── features/
│       │   ├── symmetry.py
│       │   ├── composition.py
│       │   ├── stability.py
│       │   ├── featurize_all.py
│       │   └── validate_features.py
│       ├── models/
│       │   ├── split.py
│       │   ├── train_hybrid.py
│       │   ├── calibrate.py
│       │   ├── predict_candidates.py
│       │   └── baselines.py
│       ├── analysis/
│       │   ├── metrics.py
│       │   ├── ablations.py
│       │   ├── embedding_plots.py
│       │   ├── shap_analysis.py
│       │   └── report_tables.py
│       └── dft/
│           ├── select_candidates.py
│           ├── make_vasp_inputs.py
│           ├── parse_vasp_outputs.py
│           └── band_split_analysis.py
├── notebooks/
│   ├── 01_reproduce_mataltmag.ipynb
│   ├── 02_feature_audit.ipynb
│   ├── 03_model_eval.ipynb
│   ├── 04_candidate_ranking.ipynb
│   ├── 05_interpretability.ipynb
│   └── 06_dft_figures.ipynb
├── scripts/
│   ├── 00_clone_mataltmag.sh
│   ├── 01_prepare_data.sh
│   ├── 02_extract_gnn.sh
│   ├── 03_featurize.sh
│   ├── 04_train.sh
│   ├── 05_rank_candidates.sh
│   ├── 06_make_dft_inputs.sh
│   └── 07_make_report_assets.sh
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── candidates/
│   │   ├── ranked_candidates.csv
│   │   ├── top_candidates_for_dft.csv
│   │   └── candidate_cards.md
│   └── dft/
│       ├── candidate_001/
│       ├── candidate_002/
│       └── candidate_003/
├── tests/
│   ├── test_features.py
│   ├── test_gnn_adapter.py
│   ├── test_model_training.py
│   └── test_candidate_ranking.py
└── report/
    ├── main.tex or main.md
    ├── figures/
    └── tables/
```

## 4. Environment

Recommended starting environment:

```bash
conda create -n mataltmag-hybrid python=3.10 -y
conda activate mataltmag-hybrid
pip install torch==2.0.1 accelerate==0.20.0 pymatgen PyYAML tqdm pandas numpy scikit-learn scipy matplotlib seaborn shap umap-learn pyarrow mp-api matminer joblib typer rich pytest
```

Notes:

- Use the MatAltMag versions where needed for reproduction.
- Do not commit Materials Project API keys, VASP outputs with licensed pseudopotentials, or large checkpoint files.
- Store secrets in environment variables, for example `MP_API_KEY`.

## 5. Data contract

Expected raw files from MatAltMag:

```text
data/raw/mataltmag/
├── atom_init.json
├── label0.csv                         # non-altermagnetic labels
├── label1.csv                         # known altermagnetic positives
├── candidate.csv                      # candidate set used by Gao et al.
├── Candidate_for_DFT_validate.csv      # ~300 unconfirmed high-probability candidates
├── output.csv                         # original MatAltMag predictions, if available
└── cifs/
    ├── mp-xxxx.cif
    └── ...
```

The canonical index table should have at least:

```text
material_id, formula, source_split, label, is_candidate, gnn_probability
```

where:

- `source_split = label0 | label1 | confirmed_50 | unconfirmed_candidate`
- `label = 0 | 1 | unknown`
- `is_candidate = true | false`

## 6. Pipeline

### Step 1: Reproduce MatAltMag inference

```bash
python -m mataltmag_hybrid.gnn.reproduce_inference \
  --config configs/paths.yaml \
  --mataltmag-root external/MatAltMag \
  --out data/interim/gnn_outputs.csv
```

Expected output:

```text
data/interim/gnn_outputs.csv
material_id, gnn_probability, gnn_rank, source_file
```

Validation checks:

- Material IDs match MatAltMag candidate inputs.
- Candidate probabilities approximately match `out/output.csv` or `Candidate_for_DFT_validate.csv` if those files contain scores.
- The original top-ranked candidates remain top-ranked after reproduction.

### Step 2: Extract frozen GNN embeddings

```bash
python -m mataltmag_hybrid.gnn.extract_embeddings \
  --config configs/model.yaml \
  --out data/interim/gnn_embeddings.parquet
```

Implementation options:

1. Preferred: add a non-invasive wrapper around MatAltMag model that returns the pooled crystal representation before the classifier head.
2. Backup: register a PyTorch forward hook on the classifier input layer and save the tensor passed into the head.
3. Last resort: fork `model.py` into this repository and add `return_embedding=True` while keeping the original output path unchanged.

Expected output:

```text
material_id, emb_000, emb_001, ..., emb_N
```

### Step 3: Compute explicit features

```bash
python -m mataltmag_hybrid.features.featurize_all \
  --config configs/features.yaml \
  --out data/interim/explicit_features.parquet
```

Feature groups:

#### Symmetry features

- Space group number.
- Crystal system.
- Point group symbol.
- Centrosymmetric flag.
- Presence/count of inversion operations.
- Presence/count of 2-fold, 3-fold, 4-fold, and 6-fold rotations.
- Presence/count of mirror operations.
- Number of symmetry-equivalent magnetic sites.
- Number of inequivalent magnetic sites.
- Magnetic atom Wyckoff-letter distribution.
- Primitive-cell atom count.
- Conventional-cell atom count.
- Magnetic atoms per primitive cell.
- Heuristic flags for pairs of magnetic atoms related by inversion, translation, rotation, or mirror.
- Exclusion flags such as `is_P1_or_Pminus1`.

#### Composition features

- Number of elements.
- Stoichiometric entropy.
- Fraction of 3d transition metals.
- Fraction of 4f rare earths.
- Magnetic element count.
- Magnetic element identity one-hot or multi-hot.
- Mean/std/min/max of atomic number, atomic mass, electronegativity, covalent radius.
- Magpie-style elemental statistics.
- Formula complexity metrics.

#### Materials Project metadata

- Formation energy per atom.
- Energy above hull.
- Band gap.
- Density.
- Volume per atom.
- Theoretical/experimental flag if available.
- Number of sites.

Expected output:

```text
material_id, sg_number, crystal_system_*, has_inversion, n_magnetic_sites, tm_3d_fraction, rare_earth_fraction, e_above_hull, band_gap, ...
```

### Step 4: Build train/validation/test splits

```bash
python -m mataltmag_hybrid.models.split \
  --labels data/processed/train_dataset.parquet \
  --out data/processed/splits.json
```

Rules:

- Use only labeled data for model training and validation.
- Do not train on the ~300 unconfirmed candidates.
- Keep a fixed random seed and save split IDs.
- Use stratified splits because positives are scarce.
- Add a grouped split by formula prototype or chemical system if possible, to check whether the model generalizes beyond near-duplicate chemistries.

Recommended evaluation sets:

1. `cv_random`: stratified repeated K-fold.
2. `cv_grouped`: grouped by reduced formula or chemical system.
3. `external_confirmed_50`: optional external test set from Gao et al.'s newly confirmed 50 materials, if curated cleanly and not already included in training labels.

### Step 5: Train baselines and hybrid ensemble

```bash
python -m mataltmag_hybrid.models.train_hybrid \
  --config configs/train.yaml \
  --out results/metrics/model_summary.json
```

Models to train:

1. `published_gnn`: original MatAltMag probability only.
2. `explicit_only`: symmetry + composition + MP metadata features.
3. `embedding_only`: frozen GNN embedding only.
4. `hybrid_concat`: frozen GNN embedding + explicit features.
5. `hybrid_ensemble`: averaged calibrated predictions from multiple models.

Recommended ensemble members:

- Balanced logistic regression.
- Random forest or extra-trees classifier.
- Histogram gradient boosting classifier.
- Small MLP classifier on scaled features.

Class imbalance handling:

- Use `class_weight="balanced"` where supported.
- Report average precision and precision@K, not only accuracy.
- Use calibration curves because candidate ranking depends on probability quality.

Main metrics:

```text
AUROC, AUPRC, balanced_accuracy, F1, recall@K, precision@K, enrichment@K, Brier score, expected calibration error
```

Success criterion:

- Hybrid model is at least not worse than original GNN on held-out positives.
- Ideally hybrid improves AUPRC, recall@K, or enrichment@K.
- If performance is statistically indistinguishable, use hybrid model as a ranking aid and pivot the paper toward interpretability.

### Step 6: Rank unconfirmed candidates

```bash
python -m mataltmag_hybrid.models.predict_candidates \
  --candidates data/processed/candidate_dataset.parquet \
  --model-dir results/models/latest \
  --out results/candidates/ranked_candidates.csv
```

Required columns:

```text
rank_hybrid, material_id, formula, space_group_number, crystal_system,
gnn_probability, hybrid_probability, hybrid_probability_std,
explicit_only_probability, embedding_only_probability,
formation_energy_per_atom, energy_above_hull, band_gap,
n_magnetic_sites, magnetic_elements, dft_priority_score, dft_selected, rationale
```

Recommended ranking formula:

```text
dft_priority_score =
    0.45 * calibrated_hybrid_probability
  + 0.20 * calibrated_gnn_probability
  + 0.15 * stability_score
  + 0.10 * simplicity_score
  + 0.10 * diversity_score
  - 0.10 * uncertainty_penalty
```

Candidate selection rules for DFT:

- Prefer high hybrid probability and high GNN probability.
- Prefer low energy above hull.
- Prefer simple cells and 3d magnetic elements over complex 4f-heavy systems for speed.
- Avoid selecting three near-duplicates from the same chemistry family.
- Avoid materials already confirmed in Gao et al.'s table unless using them as sanity checks.

### Step 7: Prepare DFT validation inputs

```bash
python -m mataltmag_hybrid.dft.select_candidates \
  --ranked results/candidates/ranked_candidates.csv \
  --top-n 3 \
  --out results/candidates/top_candidates_for_dft.csv

python -m mataltmag_hybrid.dft.make_vasp_inputs \
  --candidates results/candidates/top_candidates_for_dft.csv \
  --config configs/dft.yaml \
  --out results/dft/
```

For each selected material, create:

```text
results/dft/candidate_001/
├── metadata.yaml
├── POSCAR
├── POTCAR.placeholder.txt
├── relax/
│   ├── INCAR
│   ├── KPOINTS
│   └── run.sh
├── static/
│   ├── INCAR
│   ├── KPOINTS
│   └── run.sh
├── bands/
│   ├── INCAR
│   ├── KPOINTS
│   └── run.sh
└── analysis/
    ├── parse_band_split.py
    └── expected_outputs.md
```

DFT validation logic:

1. Start from Materials Project relaxed structure.
2. Generate plausible collinear AFM orderings on magnetic atoms.
3. Run spin-polarized calculations without SOC.
4. Confirm near-zero net magnetization for the AFM state.
5. Run static self-consistent calculation.
6. Run non-self-consistent band calculation on a high-symmetry path using the converged density.
7. Inspect spin-resolved bands for non-degenerate spin-up/spin-down bands at generic k-points.
8. Save band plots and quantitative spin-splitting values.

### Step 8: Interpretability fallback

Run this regardless of whether the hybrid model wins, but make it the central thesis if the hybrid model does not beat the original GNN.

```bash
python -m mataltmag_hybrid.analysis.shap_analysis \
  --model-dir results/models/latest \
  --out results/figures/shap_summary.png

python -m mataltmag_hybrid.analysis.embedding_plots \
  --embeddings data/interim/gnn_embeddings.parquet \
  --features data/interim/explicit_features.parquet \
  --out results/figures/embedding_umap.png
```

Interpretability questions:

- Do known positives cluster in the frozen GNN embedding space?
- Does the GNN separate materials by magnetic element, space group, or chemical family?
- Can explicit features predict the original GNN probability?
- Which explicit features explain hybrid probability?
- Are high-probability candidates enriched in specific space groups or magnetic-element families?

## 7. Report outputs

Required figures:

1. Full workflow diagram.
2. Data split diagram.
3. Model comparison table.
4. ROC and precision-recall curves.
5. Calibration curve.
6. UMAP/t-SNE embedding plot.
7. SHAP/permutation feature-importance plot.
8. Top candidate ranking table.
9. DFT band structures for selected candidates.
10. Spin-splitting plot or table at generic k-points.

Suggested report structure:

```text
1. Introduction
2. Background: altermagnetism and MatAltMag
3. Data and reproduction of Gao et al. inference
4. Hybrid feature design
5. Model training and evaluation
6. Candidate ranking
7. DFT validation of top candidates
8. Interpretability and failure analysis
9. Conclusion
```

## 8. Reproducibility checklist

Before considering the project complete, verify:

- [ ] Raw data sources are documented.
- [ ] All train/test split IDs are saved.
- [ ] MatAltMag inference is reproduced or discrepancy is explained.
- [ ] Embeddings are exported with fixed checkpoint hash.
- [ ] Explicit feature schema is saved.
- [ ] Candidate ranking CSV has both GNN and hybrid probabilities.
- [ ] Model metrics include confidence intervals across folds.
- [ ] Top DFT candidates have candidate cards with selection rationale.
- [ ] VASP input folders are generated reproducibly.
- [ ] Band-splitting analysis script parses outputs without manual editing.
- [ ] Report figures are regenerated by one command.

## 9. Main commands

```bash
# Prepare data and reproduce original inference
bash scripts/01_prepare_data.sh
bash scripts/02_extract_gnn.sh

# Compute explicit features
bash scripts/03_featurize.sh

# Train and evaluate hybrid model
bash scripts/04_train.sh

# Rank unconfirmed candidates
bash scripts/05_rank_candidates.sh

# Prepare DFT inputs for top candidates
bash scripts/06_make_dft_inputs.sh

# Generate report figures/tables
bash scripts/07_make_report_assets.sh
```

## 10. Definition of done

Minimum useful completion:

- Working pipeline from MatAltMag inputs to ranked candidate CSV.
- Reproduced original GNN probabilities or documented mismatch.
- Hybrid model trained and compared against original GNN.
- Candidate list with original and hybrid probabilities.
- Clear top two or three DFT candidates with rationale.
- Report draft with model results, figures, and either DFT validation or interpretability fallback.

Stretch completion:

- At least one unconfirmed candidate shows non-relativistic spin splitting in spin-polarized no-SOC DFT band structure and near-zero net magnetic moment.

## 11. Citation note

This project extends Gao et al.'s MatAltMag implementation. Cite the original National Science Review paper and the MatAltMag GitHub repository in the report and README of any public derivative repository.
