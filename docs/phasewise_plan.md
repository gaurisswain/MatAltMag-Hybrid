# Phasewise Execution Plan: MatAltMag Hybrid Altermagnet Classifier

## Phase 0 — Lock the project contract

### Goal
Convert the thesis idea into a precise, testable computational workflow.

### Tasks
- Define the final target output: `results/candidates/ranked_candidates.csv`.
- Define the training labels: positives, negatives, optional external confirmed positives, and unknown candidates.
- Decide whether the 50 Gao-discovered materials are used as external validation or excluded from training.
- Freeze all random seeds and create a `configs/` folder.
- Create the folder structure and initial tests.

### Outputs
- `README.md`
- `configs/paths.yaml`
- `configs/train.yaml`
- `data/processed/dataset_index.parquet`
- `docs/data_contract.md`

### Exit criteria
- Every material ID has a unique row and a known source: `label0`, `label1`, `confirmed_50`, or `unconfirmed_candidate`.
- The ~300 unconfirmed candidates are marked `label=unknown` and never used for supervised training.

---

## Phase 1 — Reproduce Gao et al. MatAltMag inference

### Goal
Run the published pipeline well enough that the original GNN probability can be treated as a trustworthy baseline.

### Tasks
- Clone the MatAltMag repository into `external/MatAltMag`.
- Set up the legacy-compatible Python environment.
- Download or place required files: `atom_init.json`, `label0.csv`, `label1.csv`, `candidate.csv`, CIF files, trained checkpoint, and output files.
- Run original `predict.py` through a wrapper script.
- Save original predictions in a clean, project-owned CSV.

### Outputs
- `data/interim/gnn_outputs.csv`
- `results/metrics/reproduction_check.json`
- `notebooks/01_reproduce_mataltmag.ipynb`

### Exit criteria
- For every unconfirmed candidate, the project has a `gnn_probability`.
- Any mismatch with Gao's `out/output.csv` or `Candidate_for_DFT_validate.csv` is documented.

---


## Phase 2 — Extract frozen GNN embeddings

### Goal
Use Gao et al.'s GNN encoder as a black-box feature extractor.

### Tasks
- Identify the pooled crystal representation immediately before the MatAltMag classifier head.
- Implement `mataltmag_hybrid.gnn.adapter.MatAltMagEmbeddingExtractor`.
- Export embeddings for all labeled and candidate materials.
- Save checkpoint metadata: path, file hash, architecture args, embedding dimension.

### Outputs
- `data/interim/gnn_embeddings.parquet`
- `results/metrics/embedding_export_check.json`
- `tests/test_gnn_adapter.py`

### Exit criteria
- Embedding matrix has one row per material ID.
- Embeddings are deterministic across repeated runs with the same checkpoint and seed.
- Original predictions can still be reproduced after adding the hook/wrapper.

---

## Phase 3 — Build explicit symmetry/composition/stability features

### Goal
Create a second feature stream containing interpretable physics and chemistry descriptors.

### Tasks
- Parse each CIF with pymatgen.
- Use symmetry analysis to compute space group, crystal system, point group, symmetry operations, and magnetic-site descriptors.
- Compute composition features: magnetic element counts, 3d/4f fractions, Magpie-style statistics, stoichiometric complexity.
- Query/cache Materials Project summary fields: formation energy per atom, energy above hull, band gap, density, volume, number of sites.
- Validate every feature for missingness, constants, extreme values, and leakage.

### Outputs
- `data/interim/explicit_features.parquet`
- `data/processed/feature_schema.json`
- `results/metrics/feature_quality_report.csv`
- `notebooks/02_feature_audit.ipynb`

### Exit criteria
- Feature table joins cleanly with labels and embeddings.
- Missing values are handled consistently.
- No feature directly leaks the target label or candidate status.

---

## Phase 4 — Create fair training/evaluation splits

### Goal
Measure whether the hybrid model actually improves over the published GNN.

### Tasks
- Build fixed stratified splits for scarce positives.
- Build grouped splits by chemical system or reduced formula to detect chemical-family leakage.
- Keep all unconfirmed candidates out of training and validation.
- Optionally reserve Gao's 50 newly confirmed materials as an external test set if they can be curated cleanly.

### Outputs
- `data/processed/train_dataset.parquet`
- `data/processed/candidate_dataset.parquet`
- `data/processed/splits.json`

### Exit criteria
- Every reported metric can be traced to a saved split.
- Positive/negative counts are reported for each split.

---

## Phase 5 — Train baselines and the hybrid ensemble

### Goal
Train an ensemble classifier on frozen GNN embeddings plus explicit features.

### Tasks
- Train baseline 1: original GNN probability only.
- Train baseline 2: explicit features only.
- Train baseline 3: frozen GNN embedding only.
- Train model 4: concatenated hybrid features.
- Train model 5: calibrated ensemble.
- Calibrate probabilities using Platt scaling or isotonic calibration.
- Compare models using AUPRC, AUROC, precision@K, recall@K, enrichment@K, Brier score, and calibration error.

### Outputs
- `results/models/latest/`
- `results/metrics/model_comparison.csv`
- `results/metrics/cv_predictions.parquet`
- `results/figures/pr_curve.png`
- `results/figures/roc_curve.png`
- `results/figures/calibration_curve.png`

### Exit criteria
- Hybrid is compared directly against the published GNN baseline.
- The winning model is selected by held-out performance, not candidate-set aesthetics.
- If hybrid does not win, the failure is turned into an interpretability result.

---

## Phase 6 — Re-rank the ~300 unconfirmed candidates

### Goal
Produce the central thesis artifact: a ranked list with both original and hybrid probabilities.

### Tasks
- Run the trained ensemble on unconfirmed candidates only.
- Add uncertainty using ensemble standard deviation or bootstrap variation.
- Add DFT-priority score using probability, stability, simplicity, diversity, and uncertainty.
- Generate candidate cards for the top 20.

### Outputs
- `results/candidates/ranked_candidates.csv`
- `results/candidates/top20_candidate_cards.md`
- `results/candidates/top_candidates_for_dft.csv`
- `notebooks/04_candidate_ranking.ipynb`

### Exit criteria
- CSV has both `gnn_probability` and `hybrid_probability`.
- Top candidates include rationale, not just scores.
- Top DFT candidates are not obvious duplicates.

---

## Phase 7 — DFT validation of top two or three candidates

### Goal
Validate whether selected candidates show altermagnetic electronic structure signatures.

### Tasks
- Select two or three candidates using ranking plus practical DFT filters.
- Generate VASP input folders for relaxation, static calculation, and band structure.
- Use spin-polarized collinear calculations without SOC.
- Initialize plausible AFM orderings with alternating `MAGMOM` on magnetic atoms.
- Confirm near-zero net moment for the AFM state.
- Run static self-consistent calculation.
- Run non-self-consistent band calculation along a high-symmetry path using the converged charge density.
- Plot spin-resolved bands.
- Quantify spin splitting at generic k-points.

### Outputs
- `results/dft/candidate_*/relax/`
- `results/dft/candidate_*/static/`
- `results/dft/candidate_*/bands/`
- `results/dft/candidate_*/analysis/band_splitting.csv`
- `results/figures/dft_band_candidate_001.png`
- `results/figures/dft_band_candidate_002.png`

### Exit criteria
- Each selected candidate has a complete DFT provenance record.
- At least one figure shows spin-resolved non-degenerate bands at generic k-points without SOC, or the failure is clearly documented.

---

## Phase 8 — Interpretability and failure-mode analysis

### Goal
Produce a publishable fallback even if hybrid classification does not outperform the original model.

### Tasks
- Plot UMAP/t-SNE embeddings colored by label, space group, magnetic element, and GNN probability.
- Train a surrogate model that predicts GNN probability from explicit features.
- Run SHAP/permutation importance on explicit and hybrid models.
- Analyze false positives and false negatives.
- Identify whether GNN learned chemical family, symmetry class, magnetic-element identity, or structural motifs.

### Outputs
- `results/figures/embedding_umap_by_label.png`
- `results/figures/embedding_umap_by_space_group.png`
- `results/figures/shap_summary.png`
- `results/metrics/surrogate_gnn_explanation.csv`
- `results/metrics/error_analysis.csv`

### Exit criteria
- A negative result still answers a scientific question: whether handcrafted features add information beyond MatAltMag embeddings.

---

## Phase 9 — Report/preprint assembly

### Goal
Write a 12–15 page supervisor-ready report or preprint draft.

### Tasks
- Assemble figures and tables from reproducible scripts.
- Write methods precisely enough for another student to rerun the work.
- Include a clean model comparison table.
- Include the ranked candidate list and selection rationale.
- Include DFT band structures if completed, or interpretability fallback if DFT is inconclusive.

### Outputs
- `report/main.pdf`
- `report/main.tex` or `report/main.md`
- `results/candidates/ranked_candidates.csv`
- `results/candidates/top_candidates_for_dft.csv`

### Exit criteria
- The report can defend one of these claims:
  1. The hybrid model improves candidate prioritization over the original GNN.
  2. The hybrid model does not improve performance, but interpretability analysis reveals what MatAltMag learned and where it fails.

---

# Codex implementation order

Give Codex these implementation chunks in order:

1. Create repository skeleton, config loader, logging, and tests.
2. Implement MatAltMag inference wrapper.
3. Implement embedding extractor with forward hook.
4. Implement CIF parser and symmetry/composition features.
5. Implement Materials Project metadata cache.
6. Implement dataset assembly and split generation.
7. Implement baseline and hybrid model training.
8. Implement evaluation metrics and report tables.
9. Implement candidate ranking and candidate cards.
10. Implement DFT input generation.
11. Implement VASP output parsing and band-splitting analysis.
12. Implement interpretability notebooks/scripts.
