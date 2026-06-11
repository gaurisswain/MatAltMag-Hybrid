from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from mataltmag_hybrid.config import GateError, load_paths
from mataltmag_hybrid.io.cache import read_table


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def download_cifs(config: str, limit: int | None = None, chunk_size: int = 200) -> Path:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise GateError("Materials Project CIF download requires MP_API_KEY in the environment")

    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise GateError("Materials Project CIF download requires mp-api") from exc

    paths = load_paths(config)
    index_path = paths.path("dataset_index")
    if not index_path.exists():
        raise GateError(f"Missing dataset index: {index_path}. Run scripts/01_prepare_data.sh first.")

    index = read_table(index_path)
    material_ids = index["material_id"].astype(str).drop_duplicates().tolist()
    if limit is not None:
        material_ids = material_ids[:limit]

    cif_dir = paths.path("cif_dir")
    cif_dir.mkdir(parents=True, exist_ok=True)
    missing = [material_id for material_id in material_ids if not (cif_dir / f"{material_id}.cif").exists()]
    failures: list[dict[str, str]] = []

    with MPRester(api_key) as mpr:
        for batch in chunks(missing, chunk_size):
            try:
                docs = mpr.materials.summary.search(material_ids=batch, fields=["material_id", "structure"])
            except Exception as exc:
                failures.extend({"material_id": material_id, "error": str(exc)} for material_id in batch)
                continue

            found = set()
            for doc in docs:
                material_id = str(doc.material_id)
                found.add(material_id)
                try:
                    doc.structure.to(filename=str(cif_dir / f"{material_id}.cif"))
                except Exception as exc:
                    failures.append({"material_id": material_id, "error": str(exc)})

            for material_id in sorted(set(batch) - found):
                failures.append({"material_id": material_id, "error": "not returned by Materials Project"})

    manifest = cif_dir / "download_manifest.csv"
    existing = sorted(path.stem for path in cif_dir.glob("*.cif"))
    pd.DataFrame({"material_id": existing}).to_csv(manifest, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(cif_dir / "download_failures.csv", index=False)
    return manifest


def retry_missing_cifs(config: str, chunk_size: int = 200) -> Path:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise GateError("Materials Project CIF download requires MP_API_KEY in the environment")

    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise GateError("Materials Project CIF download requires mp-api") from exc

    paths = load_paths(config)
    index = read_table(paths.path("dataset_index"))
    cif_dir = paths.path("cif_dir")
    cif_dir.mkdir(parents=True, exist_ok=True)
    material_ids = index["material_id"].astype(str).drop_duplicates().tolist()
    missing = [material_id for material_id in material_ids if not (cif_dir / f"{material_id}.cif").exists()]
    failures: list[dict[str, str]] = []

    with MPRester(api_key) as mpr:
        for material_id in missing:
            try:
                structure = mpr.get_structure_by_material_id(material_id)
                structure.to(filename=str(cif_dir / f"{material_id}.cif"))
            except Exception as exc:
                failures.append({"material_id": material_id, "error": str(exc)})

    manifest = cif_dir / "download_manifest.csv"
    existing = sorted(path.stem for path in cif_dir.glob("*.cif"))
    pd.DataFrame({"material_id": existing}).to_csv(manifest, index=False)
    failures_path = cif_dir / "download_failures.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--retry-missing", action="store_true")
    args = parser.parse_args()
    try:
        if args.retry_missing:
            manifest = retry_missing_cifs(args.config, args.chunk_size)
        else:
            manifest = download_cifs(args.config, args.limit, args.chunk_size)
    except GateError as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
