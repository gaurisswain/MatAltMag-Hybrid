from __future__ import annotations

import math
import re
from collections import Counter

MAGNETIC_3D = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"}
RARE_EARTH_4F = {"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"}


def parse_formula(formula: str) -> Counter[str]:
    parts = re.findall(r"([A-Z][a-z]?)([0-9.]*)", str(formula))
    counts: Counter[str] = Counter()
    for element, amount in parts:
        counts[element] += float(amount) if amount else 1.0
    return counts


def composition_features(formula: str) -> dict[str, object]:
    counts = parse_formula(formula)
    total = sum(counts.values()) or 1.0
    magnetic = [el for el in counts if el in MAGNETIC_3D or el in RARE_EARTH_4F]
    fractions = [value / total for value in counts.values()]
    entropy = -sum(frac * math.log(frac) for frac in fractions if frac > 0)
    return {
        "n_magnetic_sites": int(sum(counts[el] for el in magnetic)),
        "n_inequivalent_magnetic_sites": len(magnetic),
        "magnetic_elements": ";".join(sorted(magnetic)),
        "tm_3d_fraction": sum(counts[el] for el in MAGNETIC_3D if el in counts) / total,
        "rare_earth_4f_fraction": sum(counts[el] for el in RARE_EARTH_4F if el in counts) / total,
        "num_elements": len(counts),
        "stoich_entropy": entropy,
    }

