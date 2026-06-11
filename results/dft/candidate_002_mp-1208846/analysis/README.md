# DFT Analysis — mp-1208846

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
    python -m mataltmag_hybrid.dft.band_split_analysis --dft-dir results/dft/mp-1208846

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
