from __future__ import annotations

import pandas as pd


def feature_quality_report(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        if col == "material_id":
            continue
        series = features[col]
        rows.append(
            {
                "feature": col,
                "dtype": str(series.dtype),
                "missing_fraction": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "all_null": bool(series.isna().all()),
                "all_constant": bool(series.nunique(dropna=True) <= 1),
            }
        )
    return pd.DataFrame(rows)


def assert_trainable_features(df: pd.DataFrame) -> None:
    numeric = df.drop(columns=[col for col in ["material_id", "label", "is_candidate"] if col in df.columns]).select_dtypes("number")
    bad = [col for col in numeric.columns if numeric[col].isna().all() or numeric[col].nunique(dropna=True) <= 1]
    if bad:
        raise ValueError(f"All-null or all-constant training feature(s): {bad}")

