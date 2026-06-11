# Model and Candidate Selection Summary

## Objective

The goal was to extend the published MatAltMag workflow into a hybrid screening pipeline that ranks unknown Materials Project candidates for likely altermagnetism. The final artifact is `results/candidates/ranked_candidates.csv`, with a top-three DFT shortlist in `results/candidates/top_candidates_for_dft.csv`.

## What Was Built

The original MatAltMag model was not retrained or modified internally. The changes were built around it, so the published model remains a fixed reference point.

1. The original MatAltMag model was kept frozen and used in two ways:
   - as the published baseline probability source, `gnn_probability`;
   - as a frozen encoder that produces 512-dimensional crystal embeddings.

2. A second interpretable feature stream was added:
   - composition features;
   - symmetry and crystal-system features from CIFs;
   - magnetic-element and magnetic-site descriptors;
   - Materials Project stability fields, especially `energy_above_hull`.

3. Several models were compared:
   - original GNN probability only;
   - explicit features only;
   - frozen GNN embeddings only;
   - concatenated hybrid model;
   - calibrated hybrid ensemble.

4. Unknown candidates were excluded from supervised training and validation, then ranked only after the model was trained.

## Changes Made on Top of the Original Model

| Area | Original MatAltMag workflow | Hybrid workflow added here | Why this matters |
| --- | --- | --- | --- |
| Model use | Uses the trained GNN to output one probability. | Keeps the GNN frozen and also extracts its internal 512-dimensional crystal embedding. | This gives the downstream model more information than the final probability alone, while preserving the original model. |
| Feature information | Relies on learned graph features from structure. | Adds explicit composition, symmetry, magnetic-site, and MP stability features. | These features are interpretable and can capture physics/chemistry signals that may not be obvious from the final GNN score. |
| Candidate ranking | Mostly probability-driven. | Uses calibrated hybrid probability plus stability-aware DFT priority. | Candidates that are high-confidence but thermodynamically poor are moved down. |
| Validation | Original prediction reproduction/baseline. | Adds stratified held-out validation and chemical-system grouped validation. | The grouped split tests whether performance survives when related chemical families are separated. |
| Reporting | Candidate list only. | Adds model comparison, feature importance, candidate cards, and reproducibility checks. | This makes the ranking explainable and easier to defend scientifically. |

## How the New Hybrid Model Works

The new model is not a retrained version of MatAltMag. The original MatAltMag GNN is frozen and used as a fixed source of information. A separate downstream classifier is trained on top of it.

The workflow is:

1. Original MatAltMag baseline:
   - The original model reads each crystal structure as a graph.
   - It outputs the published `gnn_probability`.
   - This probability is kept as the baseline for comparison.

2. Frozen GNN embedding extraction:
   - Instead of using only the final one-number probability, the pipeline extracts the internal crystal representation before the MatAltMag classifier head.
   - This produces a 512-dimensional embedding for each material.
   - The reason for doing this is that the internal embedding can contain richer structural information than the final probability alone.

3. Explicit physics and chemistry features:
   - The pipeline adds composition features, symmetry features, magnetic-element descriptors, magnetic-site descriptors, and Materials Project stability fields.
   - Important stability fields include `energy_above_hull`, `formation_energy_per_atom`, and `band_gap`.
   - These features make the model more interpretable and make candidate selection more practical.

4. Hybrid classifier:
   - The downstream model combines the frozen MatAltMag embedding, the original `gnn_probability`, and the explicit features.
   - It predicts a calibrated `hybrid_probability`, which is the estimated probability that a material belongs to the altermagnet-like positive class.

5. Stability-aware DFT ranking:
   - Candidates are not ranked by probability alone.
   - The final DFT priority score is:

     `dft_priority_score = hybrid_probability - energy_above_hull`

   - This means a candidate must be both high-confidence and stable or near-stable to appear at the top.

In short: the original MatAltMag model supplies learned structural information, the added features supply interpretable chemistry/symmetry/stability information, and the hybrid model combines both for a more defensible DFT shortlist.

## Evidence That the Changes Helped

The clearest comparison is against the original GNN probability-only baseline. In both validation settings, the original probability alone performs like a weak classifier on the held-out split, while the hybrid models perform much better.

| Split | Model | AUROC | AUPRC | Balanced accuracy | Precision@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| Stratified | original GNN probability only | 0.500 | 0.0047 | 0.500 | 0.00 |
| Stratified | frozen GNN embedding only | 1.000 | 0.9930 | 0.983 | 0.60 |
| Stratified | calibrated hybrid ensemble | 1.000 | 1.0000 | 1.000 | 0.60 |
| Chemical-system grouped | original GNN probability only | 0.500 | 0.0047 | 0.500 | 0.00 |
| Chemical-system grouped | frozen GNN embedding only | 0.983 | 0.9668 | 0.950 | 0.58 |
| Chemical-system grouped | calibrated hybrid ensemble | 1.000 | 1.0000 | 1.000 | 0.60 |

This suggests three improvements:

- The internal GNN embedding contains much more useful signal than the final published probability alone.
- Explicit features add enough information that the hybrid model remains strong even under chemical-system grouped validation.
- Stability-aware ranking improves practical DFT selection by prioritizing candidates that are both high-confidence and near the convex hull.

The stability-aware change is what moved the previous top candidates down. They still had high hybrid probabilities, but their `energy_above_hull` values reduced their DFT priority scores.

## Validation Results

The model comparison was run using two held-out validation modes:

| Split | Best model | AUROC | AUPRC | Balanced accuracy |
| --- | --- | ---: | ---: | ---: |
| Stratified split | calibrated hybrid ensemble | 1.000 | 1.000 | 1.000 |
| Chemical-system grouped split | calibrated hybrid ensemble | 1.000 | 1.000 | 1.000 |

The grouped split is important because it checks whether the model is only memorizing closely related chemical families. The frozen embedding-only model dropped under grouped validation, but the hybrid model remained strong. This supports the idea that the explicit feature stream adds useful signal beyond the original GNN embeddings.

These scores are very high, so they should be presented carefully: they justify prioritizing candidates for DFT, but they do not prove altermagnetism by themselves. DFT remains the final validation step.

## Why the Final Top Candidates Changed

The final candidate list changed after Materials Project stability metadata was added. The ranking is not based only on model probability. It uses:

`dft_priority_score = hybrid_probability - energy_above_hull`

This means a candidate with a very high hybrid probability can move down if it is far above the convex hull, because it may be less practical for DFT follow-up or experimental relevance.

The earlier top candidates had high hybrid probabilities but worse hull penalties:

| Candidate | Hybrid probability | Energy above hull | New rank |
| --- | ---: | ---: | ---: |
| `mp-546027` | 0.9979 | 0.0147 | 27 |
| `mp-737304` | 0.9977 | 0.0464 | 68 |
| `mp-1209899` | 0.9977 | 0.1628 | 178 |

The final top candidates are preferred because they combine high model confidence with near-zero thermodynamic penalty.

## Final DFT Shortlist

| Rank | Material ID | Formula | Hybrid probability | Energy above hull | Band gap | Magnetic element |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `mp-1205353` | K4 Mn4 F12 | 0.9985 | 0.0000 | 2.4058 | Mn |
| 2 | `mp-6294` | Sr4 Er2 Ru2 O12 | 0.9983 | 0.0000 | 0.0000 | Er |
| 3 | `mp-562216` | Ba8 Co4 Si8 O28 | 0.9981 | 0.0014 | 1.7828 | Co |

These are the current top candidates because:

- all three have calibrated hybrid probabilities near 1.0;
- all three are stable or nearly stable by `energy_above_hull`;
- all three contain magnetic elements;
- all three come from the unknown candidate pool, not the supervised training set;
- all three have generated DFT input folders for follow-up.

## How We Can Be Confident They Are the Current Top Candidates

We can be confident in the ranking as a computational shortlist because:

- the candidate pool was separated from training and validation;
- the final model was selected after held-out testing;
- a chemical-system grouped split was added to reduce chemical-family leakage risk;
- ranking uses both model confidence and thermodynamic stability;
- the final CSV is generated reproducibly from scripts, not manually edited;
- tests pass: `.venv-pytest/bin/python -m pytest -q` gives 10 passing tests;
- code compilation passes: `python3 -m compileall -q src tests`.

## Important Caveat

These candidates are top computational priorities, not confirmed altermagnets yet. The ranking says they are the best DFT targets under the current hybrid model and stability-aware scoring rule. Final confirmation requires the external DFT validation: AFM setup, static/band calculations, spin-resolved band analysis, and spin-splitting checks.
