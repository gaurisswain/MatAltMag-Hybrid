from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from mataltmag_hybrid.config import ensure_parent, load_paths
from mataltmag_hybrid.io.cache import read_table


def write_surrogate_gnn_explanation(paths_config: str = "configs/paths.yaml") -> Path:
    paths = load_paths(paths_config)
    index = read_table(paths.path("dataset_index"))
    features = read_table(paths.path("explicit_features"))
    data = index.merge(features, on="material_id", validate="one_to_one")
    data["gnn_probability"] = pd.to_numeric(data["gnn_probability"], errors="coerce")
    data = data.loc[data["gnn_probability"].notna()].copy()
    if data.empty:
        raise ValueError("No rows with gnn_probability are available for surrogate analysis.")

    drop_cols = {"material_id", "formula", "source_split", "label", "is_candidate", "gnn_probability"}
    x = data.drop(columns=[col for col in drop_cols if col in data.columns])
    y = data["gnn_probability"]
    numeric = x.select_dtypes("number").columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
        ]
    )
    model = make_pipeline(pre, ExtraTreesRegressor(n_estimators=80, random_state=20260602, n_jobs=1))
    model.fit(x, y)
    score = model.score(x, y)
    perm = permutation_importance(model, x, y, n_repeats=5, random_state=20260602, n_jobs=1)
    out = pd.DataFrame(
        {
            "feature": x.columns,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
            "surrogate_r2_train": score,
            "n_rows": len(data),
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    out_path = paths.root / "results" / "metrics" / "surrogate_gnn_explanation.csv"
    ensure_parent(out_path)
    out.to_csv(out_path, index=False)
    return out_path


def write_candidate_error_analysis(paths_config: str = "configs/paths.yaml") -> Path:
    paths = load_paths(paths_config)
    ranked_path = paths.path("ranked_candidates")
    ranked = pd.read_csv(ranked_path)
    ranked["gnn_probability"] = pd.to_numeric(ranked["gnn_probability"], errors="coerce")
    ranked["hybrid_probability"] = pd.to_numeric(ranked["hybrid_probability"], errors="coerce")
    ranked["probability_delta"] = ranked["hybrid_probability"] - ranked["gnn_probability"]
    ranked["rank_gnn"] = ranked["gnn_probability"].rank(ascending=False, method="first").astype("Int64")
    ranked["rank_shift_hybrid_minus_gnn"] = ranked["rank_hybrid"] - ranked["rank_gnn"]
    cols = [
        "material_id",
        "formula",
        "rank_hybrid",
        "rank_gnn",
        "rank_shift_hybrid_minus_gnn",
        "gnn_probability",
        "hybrid_probability",
        "probability_delta",
        "space_group_number",
        "crystal_system",
        "magnetic_elements",
        "dft_selected",
    ]
    out = ranked[cols].sort_values("rank_shift_hybrid_minus_gnn")
    out_path = paths.root / "results" / "metrics" / "error_analysis.csv"
    ensure_parent(out_path)
    out.to_csv(out_path, index=False)
    return out_path
