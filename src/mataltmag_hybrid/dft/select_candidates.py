from __future__ import annotations

import pandas as pd


def select_for_dft(ranked: pd.DataFrame, n: int) -> pd.DataFrame:
    if "dft_selected" in ranked.columns and ranked["dft_selected"].any():
        return ranked.loc[ranked["dft_selected"]].head(n)
    return ranked.sort_values("dft_priority_score", ascending=False).head(n)

