# MatAltMag-Hybrid: Hybrid Altermagnet Classifier

A reproducible research codebase extending Gao et al.'s MatAltMag GNN with
explicit symmetry and composition features, re-ranking unconfirmed altermagnet
candidates and preparing top candidates for DFT validation.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Scientific Objective](#2-scientific-objective)
3. [How the Models Work — Plain English](#3-how-the-models-work--plain-english)
4. [Repository Layout](#4-repository-layout)
5. [Environment](#5-environment)
6. [Data Contract](#6-data-contract)
7. [Pipeline](#7-pipeline)
8. [DFT Validation](#8-dft-validation)
9. [Key Fixes and Design Decisions](#9-key-fixes-and-design-decisions)
10. [Final DFT Candidates](#10-final-dft-candidates)
11. [Report Outputs](#11-report-outputs)
12. [Reproducibility Checklist](#12-reproducibility-checklist)
13. [Citation Note](#13-citation-note)

---

## 1. Project Goal

Build a hybrid altermagnet classifier that combines:

1. A frozen pre-trained MatAltMag GNN encoder used as a black-box crystal
   structure feature extractor.
2. Explicit handcrafted features from crystal symmetry, magnetic-element
   composition, stoichiometry, and Materials Project stability metadata.

The model outputs two scores per candidate:

- `gnn_probability`: original MatAltMag probability.
- `hybrid_probability`: ensemble probability from frozen GNN embedding plus
  explicit features.

The final deliverable is a ranked candidate CSV and DFT validation package
for the top candidates, filtered by physical symmetry constraints.

---

## 2. Scientific Objective

Test whether explicit physics-inspired descriptors improve few-shot altermagnet
discovery beyond the published GNN classifier.

- If the hybrid model improves held-out performance: use it to prioritise
  candidates for DFT.
- If performance is statistically indistinguishable: pivot to interpretability —
  explain what the GNN has learned using embeddings, SHAP, and chemical-family
  analysis.

Altermagnetism requires two magnetic sublattices related by a crystal rotation
(not inversion). This symmetry constraint is enforced as a hard filter at the
candidate selection stage using the space group number.

---

## 3. How the Models Work — Plain English

This section explains each component of the pipeline in order, so each model
can be pointed to and explained independently.

### 3.1 The Original GNN — MatAltMag (Gao et al.)

**What it is:** A Crystal Graph Neural Network (CGNN) trained to classify
materials as altermagnetic or not.

**How it works:** A crystal is represented as a graph where atoms are nodes
and bonds are edges. The network passes information between neighbouring atoms
through several convolutional layers. After several rounds of message passing,
a pooling operation compresses the entire crystal into a single fixed-length
vector — a numerical fingerprint of the structure. A small classification head
then converts this fingerprint into a probability.

**What it produces:** A `gnn_probability` score for each material, between 0
and 1, representing the model's confidence that the material is altermagnetic.

**Where it lives:** `external/MatAltMag/` (cloned, not edited).
Inference is reproduced via `src/mataltmag_hybrid/gnn/reproduce_inference.py`.

---

### 3.2 GNN Embedding Extraction

**What it is:** The same GNN, used purely as a feature extractor with its
weights frozen (not updated during training).

**How it works:** Instead of taking the final classification probability, a
hook is registered on the layer just before the classification head. This
captures the 512-dimensional internal representation — the crystal fingerprint
— before it is converted to a probability. This vector encodes structural
information the GNN learned during its original training.

**What it produces:** A 512-dimensional embedding vector per material,
saved as `data/interim/gnn_embeddings.parquet` with columns
`emb_000` through `emb_511`.

**Where it lives:** `src/mataltmag_hybrid/gnn/extract_embeddings.py`.

**Why freeze it:** Freezing prevents the training of the hybrid model from
distorting the GNN's learned representations. The GNN is treated as a
pre-trained expert whose knowledge is borrowed, not overwritten.

---

### 3.3 Explicit Feature Engineering

**What it is:** Three families of human-designed, physics-motivated features
computed directly from the crystal structure and database metadata.

**Why it matters:** The GNN learns patterns implicitly from the crystal graph
but has no guaranteed awareness of the specific symmetry conditions that make
altermagnetism possible. These explicit features inject that domain knowledge
directly.

#### Symmetry Features (`src/mataltmag_hybrid/features/symmetry.py`)

Computed from the CIF file using pymatgen's `SpacegroupAnalyzer`:

- Space group number and crystal system
- `has_inversion`: whether the space group contains an inversion centre.
  This is the single most important feature — altermagnetism is
  symmetry-forbidden in centrosymmetric space groups.
- Counts of rotation operations (2-fold, 3-fold, 4-fold, 6-fold)
- Number of magnetic sites and their Wyckoff letter distribution
- Flags for magnetic atom pairs related by inversion vs rotation

#### Composition Features (`src/mataltmag_hybrid/features/composition.py`)

Computed from the chemical formula:

- Fraction of 3d transition metals (Mn, Fe, Co, Ni, Cr)
- Fraction of 4f rare earth elements
- Mean and spread of atomic number, electronegativity, covalent radius
- Stoichiometric entropy and formula complexity

#### Stability Features (`src/mataltmag_hybrid/features/stability.py`)

Fetched from the Materials Project cache:

- Formation energy per atom
- Energy above the convex hull (key stability indicator)
- Band gap and density

**What it produces:** `data/interim/explicit_features.parquet`.

**Where it lives:** `src/mataltmag_hybrid/features/featurize_all.py` runs all
three and merges them into one table.

---

### 3.4 The Hybrid Model (`src/mataltmag_hybrid/models/train_hybrid.py`)

**What it is:** A lightweight ensemble of classifiers trained on top of the
concatenated GNN embedding and explicit features.

**How it works:** Five model variants are trained and compared:

| Model name | Features used |
|---|---|
| `gnn_probability_only` | Original GNN score as a single feature |
| `explicit_features_only` | Symmetry + composition + stability only |
| `gnn_embedding_only` | 512-dim frozen GNN embedding only |
| `hybrid_concatenation` | GNN embedding + explicit features combined |
| `calibrated_hybrid_ensemble` | Same as above with probability calibration |

Each variant is evaluated on two split types:

- `stratified`: standard random train/test split with class balancing
- `chemical_system_grouped`: held-out chemical families to test
  generalisation beyond near-duplicate chemistries

The final model saved to disk is the `calibrated_hybrid_ensemble` trained on
all labeled data, using `ExtraTreesClassifier` wrapped in
`CalibratedClassifierCV` (Platt scaling) to ensure probability scores are
well-calibrated for ranking.

**Key metrics reported:** AUROC, AUPRC, balanced accuracy, Brier score,
precision@K, recall@K, enrichment@K.

**Important caveat:** The AUROC on the stratified split was ~0.9999, which
strongly suggests data leakage from near-duplicate chemistries. The grouped
split metric is the more honest performance estimate. This is documented as a
known limitation and motivates the interpretability analysis.

**Where it lives:** `src/mataltmag_hybrid/models/train_hybrid.py`.

---

### 3.5 Candidate Ranking (`src/mataltmag_hybrid/models/predict_candidates.py`)

**What it is:** The trained hybrid model applied to all 41,749 unconfirmed
candidate materials to produce a ranked list for DFT validation.

**How it works:**

1. Loads `candidate_dataset.parquet` (41,749 materials with all features).
2. Scores each material with the trained model.
3. Applies a hard centrosymmetry filter using the space group number — all
   materials whose space group has inversion symmetry are excluded from the
   top of the ranking (altermagnetism is symmetry-forbidden for these).
4. Computes a composite DFT priority score:

```
dft_priority_score =
    0.45 × hybrid_probability
  + 0.20 × gnn_probability
  + 0.15 × stability_score      (from energy above hull)
  + 0.10 × simplicity_score     (fewer magnetic sites = faster DFT)
  - 0.10 × hull_penalty         (penalise hull > 0.1 eV/atom)
```

5. Cross-references Gao et al.'s 381-row `Candidate_for_DFT_validate.csv`
   shortlist and flags those materials.
6. Writes `results/candidates/ranked_candidates.csv` with 41,749 rows.

**Centrosymmetric space groups excluded from top ranking:**
SG 2, 10–15, 47–74, 83–88, 123–142, 147–148, 162–167, 175–176,
191–194, 200–206, 221–230 (68 space groups total).

Of 41,749 candidates, approximately 23,500 are centrosymmetric and are
ranked below all eligible candidates regardless of model score. This is a
hard physics constraint, not a soft preference.

**Where it lives:** `src/mataltmag_hybrid/models/predict_candidates.py`.

---

### 3.6 DFT Input Generation (`src/mataltmag_hybrid/dft/make_vasp_inputs.py`)

**What it is:** Automated generation of VASP input files for the top
candidates selected for DFT validation.

**How it works:** For each selected candidate:

1. Loads the crystal structure from the CIF file.
2. Checks for inversion symmetry using pymatgen — skips centrosymmetric
   structures with a clear error message.
3. Generates a collinear AFM MAGMOM string by alternating +/− moments on
   magnetic sites and assigning small seed moments to non-magnetic atoms.
4. Builds a DFT+U block (LDAU) for correlated elements (Mn, Fe, Co, Ni,
   Cr, rare earths) using standard Dudarev U values.
5. Generates a proper KPOINTS file for the bands stage using
   `pymatgen.symmetry.bandstructure.HighSymmKpath`, with five additional
   generic off-symmetry k-points appended for spin-splitting detection.
6. Writes static and bands INCARs with `ISPIN=2`, `ISYM=0`, `NUPDOWN=0`,
   `ISTART=0`, `ICHARG=2`.

**Where it lives:** `src/mataltmag_hybrid/dft/make_vasp_inputs.py`.

---

### 3.7 Band Splitting Analysis (`src/mataltmag_hybrid/dft/band_split_analysis.py`)

**What it is:** Post-processing script that reads VASP EIGENVAL output and
quantifies spin splitting at every k-point near the Fermi level.

**How it works:**

1. Parses EIGENVAL to extract spin-up and spin-down band energies at each
   k-point.
2. Reads the Fermi energy from OUTCAR.
3. For each k-point, computes the maximum and mean energy difference between
   spin-up and spin-down bands within a 2 eV window around the Fermi level.
4. Reports the net magnetisation from the static OUTCAR to confirm AFM
   ground state.

**Altermagnetic signature to look for:**
- Spin splitting > 50 meV at the generic off-symmetry k-points
- Spin splitting ≈ 0 at standard high-symmetry points (Γ, X, M, etc.)
- Net magnetisation < 0.5 μB (confirms AFM, not ferromagnetic)

**Where it lives:** `src/mataltmag_hybrid/dft/band_split_analysis.py`.

---

## 4. Repository Layout

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
│   └── MatAltMag/
├── data/
│   ├── raw/
│   │   ├── mataltmag/          # label0.csv, label1.csv, candidate.csv, CIFs
│   │   └── materials_project/  # MP summary metadata cache
│   ├── interim/
│   │   ├── gnn_outputs.csv
│   │   ├── gnn_embeddings.parquet
│   │   ├── explicit_features.parquet
│   │   └── dataset_index.parquet
│   └── processed/
│       ├── train_dataset.parquet
│       ├── candidate_dataset.parquet    # 41,749 candidates with all features
│       └── feature_schema.json
├── src/
│   └── mataltmag_hybrid/
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
│       │   ├── predict_candidates.py   # scores all 41,749 candidates
│       │   └── baselines.py
│       ├── analysis/
│       │   ├── metrics.py
│       │   ├── ablations.py
│       │   ├── embedding_plots.py
│       │   ├── shap_analysis.py
│       │   └── report_tables.py
│       └── dft/
│           ├── select_candidates.py    # centrosymmetry filter
│           ├── make_vasp_inputs.py     # MAGMOM + LDAU + KPOINTS generation
│           ├── parse_vasp_outputs.py
│           └── band_split_analysis.py  # EIGENVAL parser + splitting metrics
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── candidates/
│   │   ├── ranked_candidates.csv       # 41,749 rows, all candidates scored
│   │   ├── top_candidates_for_dft.csv  # final 3 selected candidates
│   │   ├── inversion_audit.csv         # centrosymmetry audit of full list
│   │   └── candidate_cards.md
│   └── dft/
│       ├── candidate_001_mp-1189260/   # Nb6Cr2S12,   SG 182
│       ├── candidate_002_mp-1208846/   # Sr4Co2Si4O14, SG 113
│       └── candidate_003_mp-1227180/   # Ca2Lu2Mn4O12, SG 26
└── report/
```

---

## 5. Environment

```bash
conda create -n mataltmag-hybrid python=3.10 -y
conda activate mataltmag-hybrid
pip install -e .
pip install torch==2.0.1 pymatgen PyYAML tqdm pandas numpy scikit-learn \
    scipy matplotlib seaborn shap umap-learn pyarrow mp-api matminer \
    joblib typer rich pytest
```

Install the package in editable mode before running any scripts:

```bash
pip install -e .
```

Do not commit Materials Project API keys, VASP outputs with licensed
pseudopotentials, or large checkpoint files. Store secrets as environment
variables, for example `MP_API_KEY`.

---

## 6. Data Contract

```text
data/raw/mataltmag/
├── atom_init.json
├── label0.csv                          # non-altermagnetic labels
├── label1.csv                          # known altermagnetic positives
├── candidate.csv                       # 42,523 candidate materials
├── Candidate_for_DFT_validate.csv      # 381 Gao et al. high-priority candidates
├── output.csv                          # original MatAltMag predictions
└── cifs/                               # 62,968 CIF files (mp-XXXXX.cif)
```

---

## 7. Pipeline

### Step 1 — Reproduce MatAltMag inference

```bash
python -m mataltmag_hybrid.gnn.reproduce_inference \
  --config configs/paths.yaml \
  --mataltmag-root external/MatAltMag \
  --out data/interim/gnn_outputs.csv
```

### Step 2 — Extract frozen GNN embeddings

```bash
python -m mataltmag_hybrid.gnn.extract_embeddings \
  --config configs/model.yaml \
  --out data/interim/gnn_embeddings.parquet
```

### Step 3 — Compute explicit features

```bash
python -m mataltmag_hybrid.features.featurize_all \
  --config configs/features.yaml \
  --out data/interim/explicit_features.parquet
```

### Step 4 — Train hybrid model

```bash
python -m mataltmag_hybrid.models.train_hybrid \
  --config configs/train.yaml \
  --out results/metrics/model_summary.json
```

### Step 5 — Rank all 41,749 candidates

```bash
python -m mataltmag_hybrid.models.predict_candidates \
  --config configs/train.yaml
```

This scores all candidates, applies the centrosymmetry filter using space
group numbers, cross-references Gao et al.'s 381-row shortlist, and writes
`results/candidates/ranked_candidates.csv`.

### Step 6 — Generate DFT inputs

```bash
python -m mataltmag_hybrid.dft.make_vasp_inputs --config configs/dft.yaml
```

### Step 7 — Run VASP on DFT server (two-stage, no relax)

Since Materials Project structures are pre-relaxed at the PBE level, the
relax stage is skipped. The workflow is:

```
MP POSCAR → Static (SCF, ICHARG=2) → copy CHGCAR → Bands (non-SCF, ICHARG=11)
```

Submit static first. After convergence, copy CHGCAR to the bands folder,
then submit bands.

### Step 8 — Analyse band splitting

```bash
# After copying EIGENVAL and OUTCAR back from the DFT server
for cand in results/dft/candidate_*/; do
    python -m mataltmag_hybrid.dft.band_split_analysis --dft-dir "$cand"
done
```

---

## 8. DFT Validation

### INCAR settings

All calculations use:

| Tag | Value | Reason |
|---|---|---|
| ISPIN | 2 | Spin-polarised calculation |
| ISYM | 0 | Disable symmetry — required for AFM spin channels |
| NUPDOWN | 0 | Constrain net moment to zero — enforces AFM state |
| ISTART | 0 | Start from scratch, no WAVECAR |
| ICHARG | 2 (static) / 11 (bands) | Build charge from atoms / read from CHGCAR |
| LDAUTYPE | 2 | Dudarev GGA+U scheme |
| LMAXMIX | 4 | Required for d-electron DFT+U |

### DFT+U values used

| Element | U (eV) | l |
|---|---|---|
| Cr | 3.7 | 2 (d) |
| Mn | 3.9 | 2 (d) |
| Fe | 5.3 | 2 (d) |
| Co | 3.32 | 2 (d) |
| Ni | 6.45 | 2 (d) |
| Er | 8.0 | 3 (f) |

### KPOINTS

- Static: 8×8×8 Gamma-centred mesh
- Bands: full high-symmetry path from `pymatgen.symmetry.bandstructure.HighSymmKpath`
  plus five generic off-symmetry k-points for spin-splitting detection:
  [0.10,0.20,0.30], [0.15,0.35,0.10], [0.22,0.11,0.44],
  [0.33,0.17,0.28], [0.41,0.29,0.13]

Generic k-points are essential — altermagnetic spin splitting is
symmetry-forced to zero at all standard high-symmetry points and only
non-zero at generic positions in the Brillouin zone.

### Convergence checks before proceeding to bands

```bash
# Must print at least one line
grep "reached required accuracy" static/OUTCAR

# Net moment must be < 0.5 μB
grep "number of electron" static/OUTCAR | tail -3

# Magnetic atoms must show alternating non-zero moments
grep "magnetization (x)" static/OUTCAR | tail -25
```

---

## 9. Key Fixes and Design Decisions

### 9.1 Centrosymmetry filter

**Problem:** The original pipeline ranked centrosymmetric materials (SG 14,
15, 74) as top candidates. Altermagnetism is symmetry-forbidden in all 68
centrosymmetric space groups — any AFM ordering in these materials produces
degenerate spin-up/down bands with no splitting.

**Fix:** Hard filter in `select_candidates.py` and `predict_candidates.py`
using the canonical set of 68 centrosymmetric space group numbers. The
`has_inversion` column in `candidate_dataset.parquet` was found to be all-False
(computed incorrectly during featurisation) and is not used. Space group number
lookup is used instead.

### 9.2 MAGMOM not set in original INCARs

**Problem:** The original `make_vasp_inputs.py` generated INCARs without a
MAGMOM tag. VASP defaulted to ferromagnetic ordering with moment = 1 μB,
never producing the intended AFM state.

**Fix:** MAGMOM is now generated per-material by iterating through the
structure and assigning alternating +/− moments on magnetic sites with
small seed moments (0.6 μB) on non-magnetic atoms. `NUPDOWN = 0` is added
to constrain net moment to zero throughout SCF.

### 9.3 Band KPOINTS was a placeholder

**Problem:** The bands KPOINTS file contained the literal string "Line mode
path placeholder; replace with pymatgen high-symmetry path." VASP either
errored or produced meaningless results.

**Fix:** `pymatgen.symmetry.bandstructure.HighSymmKpath` now generates the
correct symmetry path automatically from the structure, with five generic
k-points appended.

### 9.4 band_split_analysis.py was empty

**Problem:** The analysis script contained only a docstring. No spin
splitting was ever measured.

**Fix:** Full implementation parsing EIGENVAL and OUTCAR, reporting
per-k-point maximum and mean spin splitting in eV near the Fermi level.

### 9.5 predict_candidates.py only scored 3 materials

**Problem:** The original script only scored the 3 pre-selected candidates
instead of the full 41,749-row candidate pool.

**Fix:** Complete rewrite scoring all candidates, applying the centrosymmetry
filter, cross-referencing Gao et al.'s shortlist, and computing the composite
DFT priority score.

### 9.6 AUROC of 0.9999 — data leakage

**Observation:** The stratified split AUROC of ~0.9999 is almost certainly
due to near-duplicate chemistries split across train and test sets. The
grouped split by chemical system gives a more honest estimate.

**Status:** Documented as a known limitation. The grouped split metric is
the one cited in the report. The model is used for ranking, not for
absolute probability claims.

---

## 10. Final DFT Candidates

Three candidates selected after centrosymmetry filtering, ranked by DFT
priority score. All are in Gao et al.'s 381-row shortlist.

| # | Material ID | Formula | SG | SG Symbol | Magnetic Element | Hull (eV/atom) |
|---|---|---|---|---|---|---|
| 1 | mp-1189260 | Nb6Cr2S12 | 182 | P6₃22 | Cr (±5.0 μB) | 0.000 |
| 2 | mp-1208846 | Sr4Co2Si4O14 | 113 | P-42₁m | Co (±3.0 μB) | 0.000 |
| 3 | mp-1227180 | Ca2Lu2Mn4O12 | 26 | Pmc2₁ | Mn (±4.0 μB) | 0.027 |

**Why these three:**

- **mp-1189260 (SG 182):** Chiral hexagonal space group with 6₃ screw and
  2-fold rotation axes. These rotations can relate the two Cr sublattices,
  satisfying the core symmetry requirement for altermagnetism. Cr in
  sulphide environments carries well-defined moments. On the convex hull.

- **mp-1208846 (SG 113):** Tetragonal non-centrosymmetric space group with
  improper -4 axis and 2₁ screw. Co is a reliable 3d magnetic element with
  well-converging DFT+U. On the convex hull.

- **mp-1227180 (SG 26):** Orthorhombic polar space group with a 2₁ screw
  axis — a rotation operation sufficient to relate two Mn sublattices. Mn
  in oxide environments gives large, stable moments (4 μB) and converges
  reliably with U = 3.9 eV.

**Candidates excluded from top 3 and why:**

| SG | Example | Reason excluded |
|---|---|---|
| 14, 15, 74 | mp-6294, mp-562216, mp-1205353 | Centrosymmetric — altermagnetism forbidden |
| 1 (P1) | mp-1283659, mp-1274129 | No symmetry operations — no protected sublattice relation |
| 6 (Pm) | mp-1227194, mp-1305431 | Mirror only — no rotation axis to relate sublattices |

---

## 11. Report Outputs

Required figures:

1. Full workflow diagram
2. Data split diagram (stratified vs grouped)
3. Model comparison table (5 model variants × 2 split types)
4. ROC and precision-recall curves
5. Calibration curve
6. UMAP/t-SNE embedding plot coloured by label and chemical family
7. SHAP/permutation feature-importance plot
8. Top candidate ranking table with centrosymmetry annotation
9. DFT band structures for selected candidates
10. Spin-splitting values at generic k-points

Suggested report structure:

```
1. Introduction
2. Background: altermagnetism and MatAltMag
3. Data and reproduction of Gao et al. inference
4. Hybrid feature design
5. Model training and evaluation
6. Candidate ranking and symmetry filtering
7. DFT validation of top candidates
8. Interpretability and failure analysis
9. Conclusion
```

---

## 12. Reproducibility Checklist

- [ ] Raw data sources documented
- [ ] All train/test split IDs saved to `data/processed/splits.json`
- [ ] MatAltMag inference reproduced or discrepancy explained
- [ ] Embeddings exported with fixed checkpoint hash
- [ ] Explicit feature schema saved to `data/processed/feature_schema.json`
- [ ] Centrosymmetry audit saved to `results/candidates/inversion_audit.csv`
- [ ] Candidate ranking CSV has both GNN and hybrid probabilities for all 41,749 candidates
- [ ] Model metrics include both stratified and grouped split results
- [ ] Top DFT candidates have candidate cards with selection rationale
- [ ] VASP INCARs include MAGMOM, ISYM=0, NUPDOWN=0, ISTART=0, ICHARG=2
- [ ] VASP bands KPOINTS includes generic off-symmetry k-points
- [ ] Band-splitting analysis run for all three candidates
- [ ] AUROC leakage documented and grouped-split metric cited in report

---

## 13. Citation Note

This project extends Gao et al.'s MatAltMag implementation. Cite the
original National Science Review paper and the MatAltMag GitHub repository
in the report and README of any public derivative repository.

```
Gao et al., "High-throughput identification of altermagnetic materials"
National Science Review, 2024.
https://github.com/Junwen-MX/MatAltMag
```
