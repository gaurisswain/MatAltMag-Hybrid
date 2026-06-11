from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from mataltmag_hybrid.config import GateError, ensure_parent, load_paths
from mataltmag_hybrid.io.cache import read_table
from mataltmag_hybrid.io.materials_project import MP_FIELDS


def download_mp_metadata(config: str = "configs/paths.yaml", out: str = "data/raw/materials_project/mp_summary.csv", chunk_size: int = 500) -> Path:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise GateError("Materials Project metadata download requires MP_API_KEY in the environment")
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise GateError("Materials Project metadata download requires mp-api") from exc

    paths = load_paths(config)
    index = read_table(paths.path("dataset_index"))
    ids = index["material_id"].astype(str).drop_duplicates().tolist()
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = paths.root / out_path
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=MP_FIELDS)
    done = set(existing["material_id"].astype(str)) if "material_id" in existing else set()
    missing = [material_id for material_id in ids if material_id not in done]
    rows = []
    with MPRester(api_key) as mpr:
        for start in range(0, len(missing), chunk_size):
            batch = missing[start : start + chunk_size]
            docs = mpr.materials.summary.search(
                material_ids=batch,
                fields=["material_id", "formation_energy_per_atom", "energy_above_hull", "band_gap", "density", "volume", "nsites"],
            )
            for doc in docs:
                nsites = getattr(doc, "nsites", pd.NA)
                volume = getattr(doc, "volume", pd.NA)
                rows.append(
                    {
                        "material_id": str(doc.material_id),
                        "formation_energy_per_atom": getattr(doc, "formation_energy_per_atom", pd.NA),
                        "energy_above_hull": getattr(doc, "energy_above_hull", pd.NA),
                        "band_gap": getattr(doc, "band_gap", pd.NA),
                        "density": getattr(doc, "density", pd.NA),
                        "volume_per_atom": (volume / nsites) if pd.notna(volume) and pd.notna(nsites) and nsites else pd.NA,
                    }
                )
    ensure_parent(out_path)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates("material_id", keep="last")
    combined = combined.reindex(columns=MP_FIELDS)
    combined.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--out", default="data/raw/materials_project/mp_summary.csv")
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()
    try:
        path = download_mp_metadata(args.config, args.out, args.chunk_size)
    except GateError as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
