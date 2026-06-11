from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from mataltmag_hybrid.config import GateError, load_yaml, repo_root
from mataltmag_hybrid.dft.select_candidates import select_for_dft


INCAR_RELAX = """SYSTEM = MatAltMag-Hybrid relax
ISPIN = 2
ENCUT = 520
EDIFF = 1E-5
EDIFFG = -0.02
IBRION = 2
ISIF = 3
NSW = 100
LORBIT = 11
"""

INCAR_STATIC = """SYSTEM = MatAltMag-Hybrid static
ISPIN = 2
ENCUT = 520
EDIFF = 1E-6
IBRION = -1
NSW = 0
LORBIT = 11
"""

INCAR_BANDS = """SYSTEM = MatAltMag-Hybrid bands
ISPIN = 2
ICHARG = 11
ENCUT = 520
EDIFF = 1E-6
LORBIT = 11
"""


def make_inputs(config: str = "configs/dft.yaml") -> Path:
    cfg = load_yaml(config)
    root = repo_root()
    ranked_path = root / cfg.get("top_candidates", "results/candidates/top_candidates_for_dft.csv")
    if not ranked_path.exists():
        ranked_path = root / cfg.get("ranked_candidates", "results/candidates/ranked_candidates.csv")
    if not ranked_path.exists():
        raise GateError(f"DFT input generation requires ranked candidates: {ranked_path}")
    ranked = pd.read_csv(ranked_path)
    selected = select_for_dft(ranked, int(cfg.get("n_select", 3)))
    if selected.empty:
        raise GateError("DFT input generation blocked: no selected candidates.")
    output_dir = root / cfg.get("output_dir", "results/dft")
    for idx, row in enumerate(selected.to_dict("records"), start=1):
        candidate_dir = output_dir / f"candidate_{idx:03d}_{row['material_id']}"
        for subdir in ("relax", "static", "bands", "analysis"):
            (candidate_dir / subdir).mkdir(parents=True, exist_ok=True)
        (candidate_dir / "relax" / "INCAR").write_text(INCAR_RELAX, encoding="utf-8")
        (candidate_dir / "static" / "INCAR").write_text(INCAR_STATIC, encoding="utf-8")
        (candidate_dir / "bands" / "INCAR").write_text(INCAR_BANDS, encoding="utf-8")
        (candidate_dir / "relax" / "KPOINTS").write_text("Automatic mesh\n0\nGamma\n6 6 6\n0 0 0\n", encoding="utf-8")
        (candidate_dir / "static" / "KPOINTS").write_text("Automatic mesh\n0\nGamma\n8 8 8\n0 0 0\n", encoding="utf-8")
        (candidate_dir / "bands" / "KPOINTS").write_text("Line mode path placeholder; replace with pymatgen high-symmetry path.\n", encoding="utf-8")
        (candidate_dir / "metadata.yaml").write_text(yaml.safe_dump(row, sort_keys=True), encoding="utf-8")
        (candidate_dir / "analysis" / "README.md").write_text("Place parsed band splitting outputs here. POTCAR is intentionally not generated.\n", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dft.yaml")
    args = parser.parse_args()
    try:
        make_inputs(args.config)
    except GateError as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
