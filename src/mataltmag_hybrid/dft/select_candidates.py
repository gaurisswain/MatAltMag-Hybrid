"""Select top DFT candidates from ranked list, filtering centrosymmetric structures."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd


def _has_inversion(cif_path: str) -> bool:
    """
    Return True if structure has inversion symmetry.
    Altermagnetic spin splitting is forbidden in centrosymmetric space groups.
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Issues encountered while parsing CIF"
            )
            structure = Structure.from_file(cif_path)
        ops = SpacegroupAnalyzer(structure).get_symmetry_operations()
        return any(
            abs(op.rotation_matrix + [[1,0,0],[0,1,0],[0,0,1]]).max() < 0.01
            and abs(op.translation_vector).max() < 0.01
            for op in ops
        )
    except Exception:
        return True  # conservative: exclude if uncertain


def select_for_dft(
    ranked: pd.DataFrame,
    n: int,
    cif_dir: str = "data/raw/mataltmag/cifs",
) -> pd.DataFrame:
    """
    Return the top n candidates that pass the centrosymmetry filter.

    Pulls from the full ranked list in order of hybrid_probability until
    n valid candidates are found or the list is exhausted.
    """
    # Start from dft_selected=True rows, then fall back to full ranked list
    pre_selected = ranked[ranked.get("dft_selected", pd.Series(False)) == True].copy()
    remainder = ranked[~ranked["material_id"].isin(
        pre_selected["material_id"]
    )].copy()
    ordered = pd.concat([pre_selected, remainder], ignore_index=True)

    selected = []
    skipped = []

    for _, row in ordered.iterrows():
        mid = str(row["material_id"])
        cif_path = f"{cif_dir}/{mid}.cif"

        if _has_inversion(cif_path):
            skipped.append(mid)
            print(
                f"[select_for_dft] Skipping {mid} "
                f"(SG {row.get('space_group_number','?')}): "
                f"centrosymmetric — altermagnetism forbidden.",
                file=sys.stderr,
            )
            continue

        selected.append(row)
        if len(selected) == n:
            break

    if skipped:
        print(
            f"[select_for_dft] {len(skipped)} centrosymmetric candidate(s) "
            f"skipped: {skipped}",
            file=sys.stderr,
        )
    if not selected:
        return pd.DataFrame()

    result = pd.DataFrame(selected)
    result["dft_selected"] = True
    return result.reset_index(drop=True)