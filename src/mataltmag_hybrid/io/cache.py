from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists() and path.suffix == ".parquet" and path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except ImportError:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return
    df.to_csv(path, index=False)
