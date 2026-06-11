from __future__ import annotations

from pathlib import Path

import pandas as pd

MP_FIELDS = [
    "material_id",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "density",
    "volume_per_atom",
]


def load_mp_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MP_FIELDS)
    df = pd.read_csv(path)
    missing = [col for col in MP_FIELDS if col not in df.columns]
    for col in missing:
        df[col] = pd.NA
    return df[MP_FIELDS]

