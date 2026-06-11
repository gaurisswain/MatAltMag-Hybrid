from __future__ import annotations

import pandas as pd

STABILITY_DEFAULTS = {
    "formation_energy_per_atom": pd.NA,
    "energy_above_hull": pd.NA,
    "band_gap": pd.NA,
    "density": pd.NA,
    "volume_per_atom": pd.NA,
}


def stability_lookup(material_id: str, mp_cache: pd.DataFrame) -> dict[str, object]:
    if mp_cache.empty or "material_id" not in mp_cache.columns:
        return dict(STABILITY_DEFAULTS)
    rows = mp_cache.loc[mp_cache["material_id"].astype(str) == str(material_id)]
    if rows.empty:
        return dict(STABILITY_DEFAULTS)
    row = rows.iloc[0].to_dict()
    return {key: row.get(key, pd.NA) for key in STABILITY_DEFAULTS}

