from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from mataltmag_hybrid.config import GateError, ensure_parent, load_paths, require_files
from mataltmag_hybrid.io.cache import write_table
from mataltmag_hybrid.io.load_mataltmag import build_dataset_index, normalize_material_id_columns, read_mataltmag_csv


def reproduce(config: str, mataltmag_root: str | None, out: str, allow_existing: bool = True) -> Path:
    paths = load_paths(config)
    raw_dir = paths.path("raw_mataltmag_dir")
    output_csv = raw_dir / "output.csv"
    if not output_csv.exists():
        if not allow_existing:
            raise GateError(f"MatAltMag reproduction requires original output.csv or runnable external repo: {output_csv}")
        require_files([raw_dir / "candidate.csv"], "MatAltMag inference wrapper")
        candidates = normalize_material_id_columns(read_mataltmag_csv(raw_dir / "candidate.csv"), "candidate.csv")
        prob_col = next((col for col in candidates.columns if col.lower() in {"prob", "probability", "gnn_probability", "score"}), None)
        if prob_col is None:
            raise GateError(f"No existing probability column found in {raw_dir / 'candidate.csv'} and {output_csv} is missing")
        gnn = candidates[["material_id", prob_col]].rename(columns={prob_col: "gnn_probability"})
    else:
        gnn = normalize_material_id_columns(read_mataltmag_csv(output_csv), "output.csv")
        prob_col = next((col for col in gnn.columns if col.lower() in {"prob", "probability", "gnn_probability", "score"}), None)
        if prob_col is None:
            raise GateError(f"{output_csv} has no probability column")
        gnn = gnn[["material_id", prob_col]].rename(columns={prob_col: "gnn_probability"})
    gnn["gnn_probability"] = pd.to_numeric(gnn["gnn_probability"], errors="raise")
    gnn = gnn.sort_values("gnn_probability", ascending=False).drop_duplicates("material_id")
    gnn["gnn_rank"] = range(1, len(gnn) + 1)
    gnn["source_file"] = str(output_csv if output_csv.exists() else raw_dir / "candidate.csv")
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = paths.root / out_path
    ensure_parent(out_path)
    gnn.to_csv(out_path, index=False)
    write_table(build_dataset_index(raw_dir, gnn), paths.path("dataset_index"))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--mataltmag-root", default=None)
    parser.add_argument("--out", default="data/interim/gnn_outputs.csv")
    args = parser.parse_args()
    try:
        reproduce(args.config, args.mataltmag_root, args.out)
    except GateError as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
