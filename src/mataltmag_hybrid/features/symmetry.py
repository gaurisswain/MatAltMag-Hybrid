from __future__ import annotations

from pathlib import Path


def symmetry_features(cif_path: Path | None) -> dict[str, object]:
    if cif_path is None or not cif_path.exists():
        return {
            "space_group_number": -1,
            "crystal_system": "unknown",
            "point_group": "unknown",
            "has_inversion": False,
            "n_symmetry_ops": 0,
            "n_rotation_2": 0,
            "n_rotation_3": 0,
            "n_rotation_4": 0,
            "n_rotation_6": 0,
            "n_mirror_ops": 0,
        }
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        return {
            "space_group_number": -1,
            "crystal_system": "pymatgen_unavailable",
            "point_group": "pymatgen_unavailable",
            "has_inversion": False,
            "n_symmetry_ops": 0,
            "n_rotation_2": 0,
            "n_rotation_3": 0,
            "n_rotation_4": 0,
            "n_rotation_6": 0,
            "n_mirror_ops": 0,
        }
    structure = Structure.from_file(str(cif_path))
    analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
    operations = analyzer.get_symmetry_operations()
    rotations = [op.rotation_matrix for op in operations]
    traces = [int(round(matrix.trace())) for matrix in rotations]
    return {
        "space_group_number": analyzer.get_space_group_number(),
        "crystal_system": analyzer.get_crystal_system(),
        "point_group": analyzer.get_point_group_symbol(),
        "has_inversion": any((matrix == -1).all() for matrix in rotations),
        "n_symmetry_ops": len(operations),
        "n_rotation_2": traces.count(-1),
        "n_rotation_3": traces.count(0),
        "n_rotation_4": traces.count(1),
        "n_rotation_6": traces.count(2),
        "n_mirror_ops": traces.count(1),
    }

