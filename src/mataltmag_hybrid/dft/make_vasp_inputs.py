from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from mataltmag_hybrid.config import GateError, load_yaml, repo_root
from mataltmag_hybrid.dft.select_candidates import select_for_dft

# ---------------------------------------------------------------------------
# DFT+U parameters (standard GGA+U values from Materials Project / literature)
# Format: element -> (U_eff in eV, angular momentum l)
# l = 2 for d-electrons (transition metals), l = 3 for f-electrons (rare earths)
# ---------------------------------------------------------------------------
LDAU_PARAMS: dict[str, tuple[float, int]] = {
    "Mn": (3.9,  2),
    "Fe": (5.3,  2),
    "Co": (3.32, 2),
    "Ni": (6.45, 2),
    "Cr": (3.7,  2),
    "V":  (3.25, 2),
    "Mo": (4.38, 2),
    "W":  (6.2,  2),
    "Cu": (7.71, 2),
    "Er": (8.0,  3),
    "Nd": (7.0,  3),
    "Gd": (7.66, 3),
    "Ce": (4.5,  3),
    "Pr": (6.0,  3),
    "Sm": (7.4,  3),
    "Eu": (7.4,  3),
    "Tb": (7.66, 3),
    "Dy": (7.66, 3),
    "Ho": (7.66, 3),
    "Tm": (7.66, 3),
    "Yb": (7.66, 3),
}

# Default magnetic moments (μB) per element for MAGMOM initialisation
DEFAULT_MOMENTS: dict[str, float] = {
    "Mn": 4.0,
    "Fe": 4.0,
    "Co": 3.0,
    "Ni": 2.0,
    "Cr": 3.0,
    "V":  3.0,
    "Mo": 2.0,
    "W":  2.0,
    "Cu": 1.0,
    "Er": 7.0,
    "Nd": 3.0,
    "Gd": 7.0,
    "Ce": 1.0,
    "Pr": 2.0,
    "Sm": 5.0,
    "Eu": 7.0,
    "Tb": 9.0,
    "Dy": 10.0,
    "Ho": 10.0,
    "Tm": 7.0,
    "Yb": 0.0,
}

# Small seed moment on non-magnetic atoms (helps VASP converge spin density)
NON_MAGNETIC_SEED: float = 0.6

# Generic off-symmetry k-points appended to band path for spin-splitting detection.
# These sit at generic positions in the Brillouin zone where altermagnetic splitting
# is non-zero (unlike high-symmetry points where it is symmetry-forced to zero).
GENERIC_KPOINTS: list[list[float]] = [
    [0.10, 0.20, 0.30],
    [0.15, 0.35, 0.10],
    [0.22, 0.11, 0.44],
    [0.33, 0.17, 0.28],
    [0.41, 0.29, 0.13],
]

# ---------------------------------------------------------------------------
# INCAR templates
# {magmom} and {ldau_block} are filled per candidate.
# ISYM = 0 is mandatory: symmetry reduction incorrectly merges spin channels
# in collinear AFM calculations and must be disabled.
# ---------------------------------------------------------------------------

INCAR_RELAX_TEMPLATE = """\
SYSTEM = MatAltMag-Hybrid relax
ISPIN  = 2
ISYM   = 0
ENCUT  = 520
EDIFF  = 1E-5
EDIFFG = -0.02
IBRION = 2
ISIF   = 3
NSW    = 100
LORBIT = 11
MAGMOM = {magmom}
{ldau_block}"""

INCAR_STATIC_TEMPLATE = """\
SYSTEM = MatAltMag-Hybrid static
ISPIN  = 2
ISYM   = 0
ENCUT  = 520
EDIFF  = 1E-6
IBRION = -1
NSW    = 0
LORBIT = 11
MAGMOM = {magmom}
{ldau_block}"""

INCAR_BANDS_TEMPLATE = """\
SYSTEM = MatAltMag-Hybrid bands
ISPIN  = 2
ISYM   = 0
ICHARG = 11
ENCUT  = 520
EDIFF  = 1E-6
LORBIT = 11
MAGMOM = {magmom}
{ldau_block}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_magnetic_elements(magnetic_elements_field: str) -> list[str]:
    """
    Parse the magnetic_elements column from the candidate CSV.
    Handles comma-separated strings like 'Mn', 'Er', 'Mn,Co', etc.
    Returns a list of clean element symbols.
    """
    if not magnetic_elements_field or str(magnetic_elements_field).lower() in ("nan", "none", ""):
        return []
    return [el.strip() for el in str(magnetic_elements_field).split(",") if el.strip()]


def _make_magmom(structure, magnetic_elements: list[str]) -> str:
    """
    Build a VASP MAGMOM string for a collinear AFM ordering.

    Strategy:
      - Magnetic sites: alternate +moment / -moment as we iterate through the
        structure. This is the simplest possible AFM seed; if the true ground
        state has a different ordering pattern it will still converge provided
        the initial moments are large enough.
      - Non-magnetic sites: small positive seed (NON_MAGNETIC_SEED) so VASP
        does not artificially collapse the spin density.

    For structures with only one magnetic site (ferrimagnetic risk) we still
    apply a +/- pattern — VASP will relax to the true ground state from there.
    """
    moments = []
    sign = 1
    for site in structure:
        el = site.species_string
        if el in magnetic_elements:
            m = DEFAULT_MOMENTS.get(el, 3.0)
            moments.append(sign * m)
            sign *= -1          # flip sign for next magnetic atom
        else:
            moments.append(NON_MAGNETIC_SEED)
    return " ".join(str(round(m, 1)) for m in moments)


def _make_ldau_block(structure, magnetic_elements: list[str]) -> str:
    """
    Build the LDAU block for GGA+U calculations.

    Returns an empty string if none of the elements in the structure need U.
    Uses LDAUTYPE = 2 (Dudarev scheme, single U-J parameter).
    LMAXMIX = 4 is required for d-electrons; 6 for f-electrons.
    """
    # Ordered list of unique species as they appear (VASP expects one value per species)
    seen: list[str] = []
    for site in structure:
        el = site.species_string
        if el not in seen:
            seen.append(el)

    needs_u = [el for el in seen if el in LDAU_PARAMS]
    if not needs_u:
        return ""

    ldaul_vals = []
    ldauu_vals = []
    ldauj_vals = []
    max_l = 0
    for el in seen:
        if el in LDAU_PARAMS:
            u, l = LDAU_PARAMS[el]
            ldaul_vals.append(str(l))
            ldauu_vals.append(str(u))
            ldauj_vals.append("0.0")
            max_l = max(max_l, l)
        else:
            ldaul_vals.append("-1")
            ldauu_vals.append("0.0")
            ldauj_vals.append("0.0")

    lmaxmix = 6 if max_l >= 3 else 4  # 6 for f-electrons, 4 for d-electrons

    return (
        "LDAU     = .TRUE.\n"
        "LDAUTYPE = 2\n"
        f"LDAUL    = {' '.join(ldaul_vals)}\n"
        f"LDAUU    = {' '.join(ldauu_vals)}\n"
        f"LDAUJ    = {' '.join(ldauj_vals)}\n"
        f"LMAXMIX  = {lmaxmix}"
    )


def _make_kpoints_relax() -> str:
    """Gamma-centred mesh for structural relaxation."""
    return "Automatic mesh\n0\nGamma\n6 6 6\n0 0 0\n"


def _make_kpoints_static() -> str:
    """Denser Gamma-centred mesh for static self-consistent calculation."""
    return "Automatic mesh\n0\nGamma\n8 8 8\n0 0 0\n"


def _make_kpoints_bands(structure) -> str:
    """
    Build a proper KPOINTS file for a non-self-consistent band structure run.

    Uses pymatgen's HighSymmKpath to generate the conventional high-symmetry
    path for the space group, then appends GENERIC_KPOINTS at low-symmetry
    positions in the Brillouin zone.

    The generic k-points are critical for detecting altermagnetic spin splitting:
    by symmetry, the splitting is zero along most standard high-symmetry lines
    and only non-zero at generic k-points.

    Falls back to a simple line-mode path if pymatgen raises an error.
    """
    try:
        from pymatgen.symmetry.bandstructure import HighSymmKpath

        kpath = HighSymmKpath(structure)
        kpts, labels = kpath.get_kpoints(line_density=40, coords_are_cartesian=False)

        lines = [
            "Explicit k-path for band structure + generic k-points for spin-splitting",
            str(len(kpts) + len(GENERIC_KPOINTS)),
            "Reciprocal",
        ]
        for kpt, label in zip(kpts, labels):
            tag = f"! {label}" if label else "!"
            lines.append(f"  {kpt[0]:.6f}  {kpt[1]:.6f}  {kpt[2]:.6f}  0  {tag}")

        lines.append("! --- generic off-symmetry k-points for spin-splitting check ---")
        for i, gk in enumerate(GENERIC_KPOINTS, start=1):
            lines.append(f"  {gk[0]:.6f}  {gk[1]:.6f}  {gk[2]:.6f}  0  ! generic_{i:02d}")

        return "\n".join(lines) + "\n"

    except Exception as exc:
        warnings.warn(
            f"pymatgen HighSymmKpath failed ({exc}); falling back to manual line-mode path.",
            stacklevel=2,
        )
        # Minimal fallback: a small manual path + generic points
        fallback = [
            "Fallback line-mode band path + generic k-points",
            str(8 + len(GENERIC_KPOINTS)),
            "Reciprocal",
            "  0.000000  0.000000  0.000000  0  ! Gamma",
            "  0.500000  0.000000  0.000000  0  ! X",
            "  0.500000  0.500000  0.000000  0  ! M",
            "  0.000000  0.000000  0.000000  0  ! Gamma",
            "  0.000000  0.000000  0.500000  0  ! Z",
            "  0.500000  0.000000  0.500000  0  ! R",
            "  0.500000  0.500000  0.500000  0  ! A",
            "  0.000000  0.000000  0.000000  0  ! Gamma",
            "! --- generic off-symmetry k-points ---",
        ]
        for i, gk in enumerate(GENERIC_KPOINTS, start=1):
            fallback.append(f"  {gk[0]:.6f}  {gk[1]:.6f}  {gk[2]:.6f}  0  ! generic_{i:02d}")
        return "\n".join(fallback) + "\n"


def _has_inversion(structure) -> bool:
    """
    Return True if the structure has inversion symmetry.

    Altermagnetic spin splitting is forbidden in centrosymmetric space groups
    (inversion maps spin-up sublattice onto spin-down sublattice, forcing
    band degeneracy). We skip such candidates here to avoid wasted DFT runs.
    """
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        ops = SpacegroupAnalyzer(structure).get_symmetry_operations()
        return any(
            abs(op.rotation_matrix + [[1, 0, 0], [0, 1, 0], [0, 0, 1]]).max() < 0.01
            and abs(op.translation_vector).max() < 0.01
            for op in ops
        )
    except Exception:
        # Conservative default: if we cannot determine, treat as centrosymmetric
        return True


def _load_structure(cif_path: Path, material_id: str):
    """
    Load a pymatgen Structure from a CIF file.
    Raises GateError with a clear message if the file is missing.
    """
    if not cif_path.exists():
        raise GateError(
            f"CIF not found for {material_id}: {cif_path}\n"
            "Ensure data/raw/mataltmag/cifs/ contains all candidate CIFs."
        )
    from pymatgen.core import Structure
    return Structure.from_file(str(cif_path))


def _write_metadata(candidate_dir: Path, row: dict, structure, magmom: str, has_inv: bool) -> None:
    """Write a metadata YAML summarising what was generated and why."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        sga = SpacegroupAnalyzer(structure)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
    except Exception:
        sg_symbol = row.get("space_group_number", "unknown")
        sg_number = row.get("space_group_number", -1)

    meta = {
        "material_id":         str(row["material_id"]),
        "formula":             str(row.get("formula", "")),
        "space_group_symbol":  str(sg_symbol),
        "space_group_number":  int(sg_number),
        "has_inversion":       bool(has_inv),
        "magnetic_elements":   str(row.get("magnetic_elements", "")),
        "gnn_probability":     float(row.get("gnn_probability", 0.0)),
        "hybrid_probability":  float(row.get("hybrid_probability", 0.0)),
        "energy_above_hull":   float(row.get("energy_above_hull", float("nan"))),
        "magmom_string":       magmom,
        "dft_notes": (
            "ISYM=0 mandatory for spin-polarised AFM. "
            "MAGMOM alternates +/- on magnetic sites. "
            "KPOINTS for bands includes generic k-points for spin-splitting detection. "
            "POTCAR must be added manually (not committed due to licensing)."
        ),
    }
    (candidate_dir / "metadata.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )


def _write_analysis_readme(analysis_dir: Path, row: dict) -> None:
    """Write instructions for post-processing the band calculation."""
    mid = row.get("material_id", "unknown")
    text = f"""\
# DFT Analysis — {mid}

## What to check after VASP completes

### 1. Convergence (relax and static)
    grep "reached required accuracy" relax/OUTCAR
    grep "reached required accuracy" static/OUTCAR

### 2. Magnetic ground state (static)
    grep "mag=" static/OUTCAR | tail -5
    # Each Mn/Co/Fe/Er site should show alternating +/- moments.
    # Net magnetisation should be near zero (< 0.1 μB per formula unit).

    grep "number of electron.*magnetization" static/OUTCAR | tail -3

### 3. Spin splitting (bands)
    # Run the analysis script from repo root:
    python -m mataltmag_hybrid.dft.band_split_analysis --dft-dir results/dft/{mid}

    # The script reads EIGENVAL and reports spin splitting at every k-point,
    # including the generic off-symmetry points appended at the end of KPOINTS.
    # An altermagnetic signature looks like:
    #   - Splitting > 50 meV at generic k-points
    #   - Splitting ~0 meV at high-symmetry points (Gamma, X, M, ...)
    #   - Net magnetisation still ~0

## Files expected after VASP runs
    relax/OUTCAR, relax/CONTCAR, relax/OSZICAR
    static/OUTCAR, static/CHGCAR
    bands/EIGENVAL, bands/OUTCAR

## POTCAR
    Build POTCAR by concatenating PAW PBE pseudopotentials in the same element
    order as POSCAR. Example (adjust elements to match your POSCAR):
        cat $VASP_PP/Mn/POTCAR $VASP_PP/F/POTCAR > relax/POTCAR
        cp relax/POTCAR static/POTCAR
        cp relax/POTCAR bands/POTCAR
"""
    (analysis_dir / "README.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def make_inputs(config: str = "configs/dft.yaml") -> Path:
    """
    Generate VASP input directories for the top DFT candidates.

    Returns the output directory path.
    Raises GateError if prerequisites are missing or no valid candidates survive
    the centrosymmetry filter.
    """
    cfg = load_yaml(config)
    root = repo_root()

    # Locate ranked candidates CSV
    ranked_path = root / cfg.get("top_candidates", "results/candidates/top_candidates_for_dft.csv")
    if not ranked_path.exists():
        ranked_path = root / cfg.get("ranked_candidates", "results/candidates/ranked_candidates.csv")
    if not ranked_path.exists():
        raise GateError(
            f"DFT input generation requires a ranked candidates CSV: {ranked_path}\n"
            "Run scripts/05_rank_candidates.sh first."
        )

    ranked = pd.read_csv(ranked_path)
    selected = select_for_dft(ranked, int(cfg.get("n_select", 3)))
    if selected.empty:
        raise GateError("DFT input generation blocked: no candidates marked dft_selected=True.")

    cif_dir = root / cfg.get("cif_dir", "data/raw/mataltmag/cifs")
    output_dir = root / cfg.get("output_dir", "results/dft")
    skipped: list[str] = []
    written: list[str] = []
    candidate_counter = 0

    for row in selected.to_dict("records"):
        mid = str(row["material_id"])
        cif_path = cif_dir / f"{mid}.cif"

        # Load structure (raises GateError if CIF is missing)
        structure = _load_structure(cif_path, mid)

        # Centrosymmetry guard ------------------------------------------------
        # Altermagnetism requires the two spin sublattices to be related by a
        # crystal rotation, NOT by inversion. If the space group has inversion,
        # any AFM ordering will produce degenerate spin-up/down bands and the
        # material is a conventional antiferromagnet, not an altermagnet.
        has_inv = _has_inversion(structure)
        if has_inv:
            print(
                f"[make_vasp_inputs] SKIPPING {mid} ({row.get('formula','?')}): "
                f"space group #{row.get('space_group_number','?')} is centrosymmetric — "
                f"altermagnetism symmetry-forbidden.",
                file=sys.stderr,
            )
            skipped.append(mid)
            continue

        candidate_counter += 1
        candidate_dir = output_dir / f"candidate_{candidate_counter:03d}_{mid}"
        for subdir in ("relax", "static", "bands", "analysis"):
            (candidate_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Parse magnetic elements
        mag_els = _parse_magnetic_elements(str(row.get("magnetic_elements", "")))

        # Build MAGMOM and LDAU strings
        magmom   = _make_magmom(structure, mag_els)
        ldau     = _make_ldau_block(structure, mag_els)

        # Write INCARs
        (candidate_dir / "relax"  / "INCAR").write_text(
            INCAR_RELAX_TEMPLATE.format(magmom=magmom, ldau_block=ldau).strip() + "\n",
            encoding="utf-8",
        )
        (candidate_dir / "static" / "INCAR").write_text(
            INCAR_STATIC_TEMPLATE.format(magmom=magmom, ldau_block=ldau).strip() + "\n",
            encoding="utf-8",
        )
        (candidate_dir / "bands"  / "INCAR").write_text(
            INCAR_BANDS_TEMPLATE.format(magmom=magmom, ldau_block=ldau).strip() + "\n",
            encoding="utf-8",
        )

        # Write KPOINTS
        (candidate_dir / "relax"  / "KPOINTS").write_text(_make_kpoints_relax(),  encoding="utf-8")
        (candidate_dir / "static" / "KPOINTS").write_text(_make_kpoints_static(), encoding="utf-8")
        (candidate_dir / "bands"  / "KPOINTS").write_text(_make_kpoints_bands(structure), encoding="utf-8")

        # Write POSCAR from relaxed CIF structure
        from pymatgen.io.vasp.inputs import Poscar
        poscar = Poscar(structure)
        (candidate_dir / "relax" / "POSCAR").write_text(poscar.get_str(), encoding="utf-8")
        # static and bands use CONTCAR from previous step; provide a copy as fallback
        (candidate_dir / "static" / "POSCAR").write_text(poscar.get_str(), encoding="utf-8")
        (candidate_dir / "bands"  / "POSCAR").write_text(poscar.get_str(), encoding="utf-8")

        # Write placeholder run scripts for cluster submission
        for stage in ("relax", "static", "bands"):
            _write_run_script(candidate_dir / stage / "run.sh", stage, mid)

        # Write metadata and analysis README
        _write_metadata(candidate_dir, row, structure, magmom, has_inv)
        _write_analysis_readme(candidate_dir / "analysis", row)

        # POTCAR placeholder note (never committed — licensing)
        (candidate_dir / "POTCAR.placeholder.txt").write_text(
            f"Build POTCAR for {mid} by concatenating PAW PBE pseudopotentials.\n"
            "See analysis/README.md for instructions.\n"
            "Do NOT commit POTCAR files to the repository.\n",
            encoding="utf-8",
        )

        written.append(mid)
        print(f"[make_vasp_inputs] Written inputs for candidate_{candidate_counter:03d}_{mid}")

    # Summary
    if skipped:
        print(
            f"\n[make_vasp_inputs] Skipped {len(skipped)} centrosymmetric candidate(s): "
            f"{skipped}",
            file=sys.stderr,
        )
    if not written:
        raise GateError(
            "No valid DFT candidates remain after centrosymmetry filtering.\n"
            "All top candidates have inversion symmetry — re-rank with a stricter "
            "symmetry filter or extend the candidate pool."
        )

    print(f"\n[make_vasp_inputs] Done. {len(written)} candidate(s) written to {output_dir}")
    return output_dir


def _write_run_script(path: Path, stage: str, material_id: str) -> None:
    """
    Write a minimal SLURM submission script.
    Edit the #SBATCH directives to match your cluster configuration.
    """
    prev = {"static": "relax/CONTCAR", "bands": "static/CHGCAR"}.get(stage, "")
    copy_line = ""
    if stage == "static":
        copy_line = "cp ../relax/CONTCAR POSCAR  # use relaxed structure\n"
    elif stage == "bands":
        copy_line = (
            "cp ../static/CHGCAR .       # use converged charge density\n"
            "cp ../static/WAVECAR . 2>/dev/null || true\n"
        )

    script = f"""\
#!/bin/bash
#SBATCH --job-name={material_id}_{stage}
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --output={stage}.out
#SBATCH --error={stage}.err

# Adjust module names to match your cluster
module load vasp/6.3.0

cd $SLURM_SUBMIT_DIR
{copy_line}
mpirun -np $SLURM_NTASKS vasp_std > vasp.log 2>&1
echo "VASP finished with exit code $?"
"""
    path.write_text(script, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate VASP DFT input files for top altermagnet candidates."
    )
    parser.add_argument(
        "--config",
        default="configs/dft.yaml",
        help="Path to DFT config YAML (default: configs/dft.yaml)",
    )
    args = parser.parse_args()
    try:
        make_inputs(args.config)
    except GateError as exc:
        print(f"\nGate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()