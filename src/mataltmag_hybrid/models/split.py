from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def supervised_rows(index: pd.DataFrame) -> pd.DataFrame:
    return index.loc[(index["is_candidate"].astype(str).str.lower() != "true") & (index["label"].astype(str) != "unknown")].copy()


def candidate_rows(index: pd.DataFrame) -> pd.DataFrame:
    return index.loc[(index["is_candidate"].astype(str).str.lower() == "true") | (index["label"].astype(str) == "unknown")].copy()


def make_split(index: pd.DataFrame, seed: int, out: Path) -> dict[str, list[str]]:
    labeled = supervised_rows(index)
    if len(labeled) < 4 or labeled["label"].nunique() < 2:
        split = {"train": labeled["material_id"].astype(str).tolist(), "test": []}
    else:
        train, test = train_test_split(
            labeled["material_id"].astype(str),
            test_size=0.25,
            random_state=seed,
            stratify=labeled["label"].astype(int),
        )
        split = {"train": list(train), "test": list(test)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split

