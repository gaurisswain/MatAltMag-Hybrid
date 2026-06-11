from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mataltmag_hybrid.config import GateError, ensure_parent, load_paths, load_yaml
from mataltmag_hybrid.features.composition import composition_features
from mataltmag_hybrid.features.stability import stability_lookup
from mataltmag_hybrid.features.symmetry import symmetry_features
from mataltmag_hybrid.features.validate_features import feature_quality_report
from mataltmag_hybrid.io.cache import read_table, write_table
from mataltmag_hybrid.io.materials_project import load_mp_cache


def featurize(config: str, out: str) -> Path:
    cfg = load_yaml(config)
    paths = load_paths(cfg.get("paths_config", "configs/paths.yaml"))
    index_path = paths.path("dataset_index")
    if not index_path.exists() and not index_path.with_suffix(".csv").exists():
        raise GateError(f"Feature generation requires dataset index: {index_path}")
    index = read_table(index_path)
    mp_cache = load_mp_cache(paths.root / cfg.get("materials_project_cache", "data/raw/materials_project/mp_summary.csv"))
    rows = []
    for row in index.to_dict("records"):
        material_id = str(row["material_id"])
        cif_path = paths.path("cif_dir") / f"{material_id}.cif"
        feature_row = {"material_id": material_id}
        feature_row.update(symmetry_features(cif_path))
        feature_row.update(composition_features(str(row.get("formula", ""))))
        feature_row.update(stability_lookup(material_id, mp_cache))
        rows.append(feature_row)
    features = pd.DataFrame(rows)
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = paths.root / out_path
    ensure_parent(out_path)
    write_table(features, out_path)
    schema_path = paths.root / cfg.get("feature_schema", "data/processed/feature_schema.json")
    ensure_parent(schema_path)
    schema_path.write_text(json.dumps({col: str(dtype) for col, dtype in features.dtypes.items()}, indent=2), encoding="utf-8")
    report_path = paths.root / cfg.get("quality_report", "results/metrics/feature_quality_report.csv")
    ensure_parent(report_path)
    feature_quality_report(features).to_csv(report_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/features.yaml")
    parser.add_argument("--out", default="data/interim/explicit_features.parquet")
    args = parser.parse_args()
    try:
        featurize(args.config, args.out)
    except (GateError, FileNotFoundError) as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
