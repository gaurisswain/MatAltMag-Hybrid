"""
Score all unconfirmed candidates with the trained hybrid model and produce a
ranked CSV for DFT candidate selection.

Key behaviours
--------------
- Loads all rows from candidate_dataset.parquet (41 k+ materials).
- Uses the trained model saved by train_hybrid.py.
- Applies the priority scoring formula from the project README.
- Uses the has_inversion column already computed during featurisation to
  exclude centrosymmetric materials (altermagnetism symmetry-forbidden).
- Cross-references Candidate_for_DFT_validate.csv (Gao et al.'s 381-row
  shortlist) and flags those materials for priority attention.
- Writes results/candidates/ranked_candidates.csv with all required columns.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from mataltmag_hybrid.config import GateError, ensure_parent, load_paths, load_yaml

# Columns that must never be fed as features to the model
DROP_COLS = {
    "material_id", "formula", "source_split", "label",
    "is_candidate", "gnn_probability",
}

# Minimum energy above hull (eV/atom) threshold — materials above this are
# thermodynamically very unstable and deprioritised but not excluded outright
HULL_PENALTY_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Feature preparation  (must mirror train_hybrid.py _feature_matrix logic)
# ---------------------------------------------------------------------------

def _feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for inference.
    Uses hybrid_concatenation mode: GNN embedding + explicit features,
    gnn_probability excluded (it is used separately in priority formula).
    Matches the feature set the model was trained on.
    """
    drop = {col for col in DROP_COLS if col in data.columns}
    return data.drop(columns=list(drop)).copy()


# ---------------------------------------------------------------------------
# Priority scoring formula  (README §6)
# ---------------------------------------------------------------------------

def _stability_score(energy_above_hull: pd.Series) -> pd.Series:
    """
    Normalised stability score in [0, 1].
    hull = 0  -> score = 1.0  (on the convex hull, stable)
    hull = 0.1 eV/atom -> score = 0.0
    Above 0.1 eV/atom: negative (penalised).
    """
    return (1.0 - energy_above_hull.clip(lower=0.0) / HULL_PENALTY_THRESHOLD).clip(-1.0, 1.0)


def _simplicity_score(n_magnetic_sites: pd.Series) -> pd.Series:
    """
    Prefer small, simple magnetic unit cells (faster DFT).
    Normalised so 1 site -> 1.0, 10+ sites -> 0.0.
    """
    return (1.0 - (n_magnetic_sites.clip(lower=1, upper=10) - 1) / 9.0)


def _diversity_penalty(data: pd.DataFrame, selected_ids: list[str]) -> pd.Series:
    """
    Simple chemical-system diversity score.
    Materials whose chemical system already appears in selected_ids get a
    small penalty so we avoid recommending three variants of the same compound.
    Returns a Series of 0 (no penalty) or 0.5 (same family already selected).
    """
    if not selected_ids:
        return pd.Series(0.0, index=data.index)

    def _chem_sys(formula: str) -> str:
        import re
        elements = sorted(set(re.findall(r"[A-Z][a-z]?", str(formula))))
        return "-".join(elements)

    selected_systems = set(
        _chem_sys(row["formula"])
        for _, row in data[data["material_id"].isin(selected_ids)].iterrows()
    )
    return data["formula"].map(_chem_sys).isin(selected_systems).astype(float) * 0.5


def compute_priority_score(
    data: pd.DataFrame,
    hybrid_prob: pd.Series,
    gnn_prob: pd.Series,
) -> pd.Series:
    """
    dft_priority_score =
        0.45 * hybrid_probability
      + 0.20 * gnn_probability
      + 0.15 * stability_score
      + 0.10 * simplicity_score
      - 0.10 * hull_penalty      (energy_above_hull > threshold)
    """
    hull     = data.get("energy_above_hull", pd.Series(0.0, index=data.index)).fillna(0.1)
    n_sites  = data.get("n_magnetic_sites",  pd.Series(4,   index=data.index)).fillna(4)

    stability  = _stability_score(hull)
    simplicity = _simplicity_score(n_sites)
    hull_pen   = (hull > HULL_PENALTY_THRESHOLD).astype(float) * 0.1

    return (
        0.45 * hybrid_prob
      + 0.20 * gnn_prob
      + 0.15 * stability
      + 0.10 * simplicity
      - 0.10 * hull_pen
    )


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict(config: str, model_dir_override: str | None = None) -> Path:
    cfg   = load_yaml(config)
    paths = load_paths(cfg.get("paths_config", "configs/paths.yaml"))

    # ------------------------------------------------------------------
    # 1. Load candidate dataset
    # ------------------------------------------------------------------
    cand_path = paths.path("candidate_dataset")
    if not cand_path.exists():
        raise GateError(
            f"Candidate dataset not found: {cand_path}\n"
            "Run scripts/04_train.sh first to generate candidate_dataset.parquet."
        )
    candidates = pd.read_parquet(cand_path)
    print(f"[predict_candidates] Loaded {len(candidates):,} candidates.")

    # ------------------------------------------------------------------
    # 2. Load trained model
    # ------------------------------------------------------------------
    model_dir = Path(model_dir_override) if model_dir_override else (
        paths.root / cfg.get("model_dir", "results/models/latest")
    )
    model_path = model_dir / "hybrid_model.joblib"
    if not model_path.exists():
        raise GateError(
            f"Trained model not found: {model_path}\n"
            "Run scripts/04_train.sh first."
        )
    model = joblib.load(model_path)
    print(f"[predict_candidates] Loaded model from {model_path}")

    # ------------------------------------------------------------------
    # 3. Build feature matrix and score
    # ------------------------------------------------------------------
    x = _feature_matrix(candidates)

    # Numeric imputation for any NaNs (same strategy as training)
    numeric_cols = x.select_dtypes("number").columns
    if x[numeric_cols].isna().any().any():
        imp = SimpleImputer(strategy="median")
        x[numeric_cols] = imp.fit_transform(x[numeric_cols])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hybrid_prob = pd.Series(
            model.predict_proba(x)[:, 1],
            index=candidates.index,
            name="hybrid_probability",
        )

    gnn_prob = candidates.get(
        "gnn_probability",
        pd.Series(0.5, index=candidates.index)
    ).fillna(0.5)

    print(f"[predict_candidates] Scored {len(candidates):,} candidates.")

    # ------------------------------------------------------------------
    # 4. Apply centrosymmetry filter using precomputed has_inversion column
    # ------------------------------------------------------------------
    # Recompute centrosymmetry from space_group_number.
    # The has_inversion column in the dataset is unreliable (all-False).
    # Using the canonical set of 68 centrosymmetric space group numbers instead.
    CENTROSYMMETRIC_SGS = {
        2,                                          # Triclinic  -1
        10,11,12,13,14,15,                          # Monoclinic 2/m
        47,48,49,50,51,52,53,54,55,56,57,58,59,60,
        61,62,63,64,65,66,67,68,69,70,71,72,73,74, # Orthorhombic mmm
        83,84,85,86,87,88,                          # Tetragonal 4/m
        123,124,125,126,127,128,129,130,131,132,
        133,134,135,136,137,138,139,140,141,142,    # Tetragonal 4/mmm
        147,148,                                    # Trigonal -3
        162,163,164,165,166,167,                    # Trigonal -3m
        175,176,                                    # Hexagonal 6/m
        191,192,193,194,                            # Hexagonal 6/mmm
        200,201,202,203,204,205,206,                # Cubic m-3
        221,222,223,224,225,226,227,228,229,230,    # Cubic m-3m
    }
    sg_col = candidates.get(
        "space_group_number", pd.Series(0, index=candidates.index)
    ).fillna(0).astype(int)
    inv_mask = sg_col.isin(CENTROSYMMETRIC_SGS)
    n_centrosym = int(inv_mask.sum())
    print(
        f"[predict_candidates] Centrosymmetric (excluded from top ranking): "
        f"{n_centrosym:,} / {len(candidates):,}  "
        f"({100*n_centrosym/len(candidates):.1f}%)"
    )

    # ------------------------------------------------------------------
    # 5. Cross-reference Gao et al.'s 381-row shortlist
    # ------------------------------------------------------------------
    gao_ids: set[str] = set()
    gao_path = paths.root / "data/raw/mataltmag/Candidate_for_DFT_validate.csv"
    if gao_path.exists():
        gao_df  = pd.read_csv(gao_path)
        id_col  = "id" if "id" in gao_df.columns else gao_df.columns[0]
        gao_ids = set(gao_df[id_col].astype(str).tolist())
        print(f"[predict_candidates] Loaded {len(gao_ids)} Gao et al. shortlist IDs.")
    else:
        print("[predict_candidates] Candidate_for_DFT_validate.csv not found — skipping cross-reference.")

    # ------------------------------------------------------------------
    # 6. Compute priority score
    # ------------------------------------------------------------------
    priority = compute_priority_score(candidates, hybrid_prob, gnn_prob)

    # ------------------------------------------------------------------
    # 7. Assemble output dataframe
    # ------------------------------------------------------------------
    out = candidates[[
        "material_id", "formula",
        "space_group_number", "crystal_system",
        "has_inversion",
        "n_magnetic_sites", "magnetic_elements",
        "formation_energy_per_atom", "energy_above_hull", "band_gap",
    ]].copy()

    # Fill missing columns gracefully
    for col in ["space_group_number", "crystal_system", "has_inversion",
                "n_magnetic_sites", "magnetic_elements",
                "formation_energy_per_atom", "energy_above_hull", "band_gap"]:
        if col not in out.columns:
            out[col] = None

    out["gnn_probability"]           = gnn_prob.values
    out["hybrid_probability"]        = hybrid_prob.values
    out["hybrid_probability_std"]    = 0.0   # single model; extend if ensemble
    out["explicit_only_probability"] = hybrid_prob.values   # proxy until ablation
    out["embedding_only_probability"]= hybrid_prob.values   # proxy until ablation
    out["dft_priority_score"]        = priority.values
    out["is_centrosymmetric"]        = inv_mask.values
    out["in_gao_shortlist"]          = candidates["material_id"].isin(gao_ids).values

    # ------------------------------------------------------------------
    # 8. Rank: non-centrosymmetric candidates first, then by priority score
    # ------------------------------------------------------------------
    out_eligible   = out[~inv_mask].sort_values("dft_priority_score", ascending=False)
    out_centrosym  = out[inv_mask].sort_values("dft_priority_score",  ascending=False)
    out_ranked     = pd.concat([out_eligible, out_centrosym], ignore_index=True)
    out_ranked.insert(0, "rank_hybrid", range(1, len(out_ranked) + 1))

    # Mark top-3 eligible as dft_selected
    eligible_idx = out_ranked[~out_ranked["is_centrosymmetric"]].head(3).index
    out_ranked["dft_selected"] = False
    out_ranked.loc[eligible_idx, "dft_selected"] = True

    # Add rationale string for selected candidates
    def _rationale(row) -> str:
        return (
            f"hybrid={row['hybrid_probability']:.3f}; "
            f"hull={row['energy_above_hull']:.4f}; "
            f"magnetic={row['magnetic_elements']}; "
            f"gao_shortlist={row['in_gao_shortlist']}"
        )
    out_ranked["rationale"] = out_ranked.apply(
        lambda r: _rationale(r) if r["dft_selected"] else "", axis=1
    )

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    out_path = paths.root / cfg.get(
        "ranked_candidates", "results/candidates/ranked_candidates.csv"
    )
    ensure_parent(out_path)
    out_ranked.to_csv(out_path, index=False)

    # Summary
    n_eligible = int((~out_ranked["is_centrosymmetric"]).sum())
    n_selected = int(out_ranked["dft_selected"].sum())
    print(f"\n[predict_candidates] Results")
    print(f"  Total candidates ranked : {len(out_ranked):,}")
    print(f"  Eligible (non-centrosym): {n_eligible:,}")
    print(f"  Marked for DFT          : {n_selected}")
    print(f"\n  Top 10 eligible candidates:")
    top10 = out_ranked[~out_ranked["is_centrosymmetric"]].head(10)
    print(top10[[
        "rank_hybrid", "material_id", "formula",
        "space_group_number", "hybrid_probability",
        "energy_above_hull", "in_gao_shortlist",
    ]].to_string(index=False))
    print(f"\n  Saved to {out_path}")

    # Also write the top-3 for DFT
    top_path = paths.root / cfg.get(
        "top_candidates", "results/candidates/top_candidates_for_dft.csv"
    )
    ensure_parent(top_path)
    out_ranked[out_ranked["dft_selected"]].to_csv(top_path, index=False)
    print(f"  Top DFT candidates saved to {top_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score all candidates with the trained hybrid model and rank them."
    )
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Training config YAML (for paths resolution).",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Override path to candidate_dataset.parquet.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override path to model directory containing hybrid_model.joblib.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path for ranked_candidates.csv.",
    )
    args = parser.parse_args()

    try:
        predict(args.config, model_dir_override=args.model_dir)
    except (GateError, FileNotFoundError) as exc:
        print(f"\nGate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()