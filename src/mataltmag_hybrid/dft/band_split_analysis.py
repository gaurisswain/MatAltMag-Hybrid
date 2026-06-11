"""Parse VASP EIGENVAL outputs and quantify spin splitting at generic k-points."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_eigenval(eigenval_path: Path) -> dict:
    lines = eigenval_path.read_text().splitlines()
    ispin   = int(lines[0].split()[3])
    nkpts   = int(lines[5].split()[1])
    nbands  = int(lines[5].split()[2])
    kpoints, spin_up, spin_dn = [], [], []
    i = 7
    for _ in range(nkpts):
        kpoints.append(list(map(float, lines[i].split()[:3])))
        up, dn = [], []
        for b in range(nbands):
            vals = lines[i + 1 + b].split()
            up.append(float(vals[1]))
            if ispin == 2:
                dn.append(float(vals[2]))
        spin_up.append(up)
        spin_dn.append(dn)
        i += nbands + 2
    return {
        "kpoints": np.array(kpoints),
        "spin_up": np.array(spin_up),
        "spin_dn": np.array(spin_dn),
        "ispin":   ispin,
        "nkpts":   nkpts,
        "nbands":  nbands,
    }


def find_fermi(outcar_path: Path) -> float:
    for line in reversed(outcar_path.read_text().splitlines()):
        if "E-fermi" in line:
            return float(line.split()[2])
    raise ValueError(f"E-fermi not found in {outcar_path}")


def net_magnetization(outcar_path: Path) -> float:
    val = None
    for line in outcar_path.read_text().splitlines():
        if "number of electron" in line and "magnetization" in line:
            val = float(line.split()[-1])
    return val if val is not None else float("nan")


def spin_splitting(bands_dir: Path, efermi: float) -> pd.DataFrame:
    data = parse_eigenval(bands_dir / "EIGENVAL")
    if data["ispin"] != 2:
        raise ValueError("Calculation is not spin-polarised (ISPIN != 2).")
    rows = []
    for ki, (kpt, up, dn) in enumerate(
        zip(data["kpoints"], data["spin_up"], data["spin_dn"])
    ):
        up_rel = np.array(up) - efermi
        dn_rel = np.array(dn) - efermi
        window = (abs(up_rel) < 2.0) | (abs(dn_rel) < 2.0)
        if not window.any():
            continue
        split = np.abs(up_rel[window] - dn_rel[window])
        rows.append({
            "kpt_index":             ki,
            "kx":                    round(float(kpt[0]), 4),
            "ky":                    round(float(kpt[1]), 4),
            "kz":                    round(float(kpt[2]), 4),
            "max_spin_splitting_eV": round(float(split.max()), 4),
            "mean_spin_splitting_eV":round(float(split.mean()), 4),
            "n_bands_in_window":     int(window.sum()),
        })
    return pd.DataFrame(rows)


def analyze(dft_dir: str) -> None:
    root      = Path(dft_dir)
    bands_dir = root / "bands"
    outcar_static = root / "static" / "OUTCAR"
    outcar_bands  = bands_dir / "OUTCAR"

    print(f"\n{'='*60}")
    print(f"Candidate : {root.name}")

    # --- Net magnetisation ---
    if outcar_static.exists():
        netmag = net_magnetization(outcar_static)
        status = "✓ AFM confirmed" if abs(netmag) < 0.5 else "⚠ NOT AFM"
        print(f"Net magnetisation (static) : {netmag:.3f} μB  [{status}]")
    else:
        print("static/OUTCAR not found — skipping magnetisation check.")

    # --- Spin splitting ---
    eigenval = bands_dir / "EIGENVAL"
    if not eigenval.exists():
        print("bands/EIGENVAL not found — run VASP bands calculation first.")
        return

    outcar_ref = outcar_bands if outcar_bands.exists() else outcar_static
    try:
        efermi = find_fermi(outcar_ref) if outcar_ref.exists() else 0.0
    except ValueError:
        efermi = 0.0

    df = spin_splitting(bands_dir, efermi)
    out_path = root / "analysis" / "spin_splitting.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSpin splitting at {len(df)} k-points near Fermi level (±2 eV):")
    print(df.sort_values("max_spin_splitting_eV", ascending=False)
            .head(10).to_string(index=False))

    max_split = df["max_spin_splitting_eV"].max() if not df.empty else 0.0
    if max_split > 0.05:
        print(f"\n✓ MAX splitting = {max_split:.4f} eV — altermagnetic signature PRESENT")
    else:
        print(f"\n✗ MAX splitting = {max_split:.4f} eV — no significant splitting detected")

    print(f"\nFull results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dft-dir", required=True,
                        help="Path to candidate DFT directory, e.g. results/dft/candidate_001_mp-xxxxx")
    args = parser.parse_args()
    analyze(args.dft_dir)


if __name__ == "__main__":
    main()