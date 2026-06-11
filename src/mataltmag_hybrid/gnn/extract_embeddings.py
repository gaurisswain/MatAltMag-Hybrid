from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mataltmag_hybrid.config import GateError, ensure_parent, load_paths, load_yaml
from mataltmag_hybrid.gnn.adapter import MatAltMagEmbeddingExtractor
from mataltmag_hybrid.io.cache import read_table, write_table


def extract(config: str, out: str, fixture: bool = False) -> Path:
    cfg = load_yaml(config)
    paths = load_paths(cfg.get("paths_config", "configs/paths.yaml"))
    index_path = paths.path("dataset_index")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing dataset index: {index_path}. Run scripts/01_prepare_data.sh first.")
    index = read_table(index_path)
    material_ids = index["material_id"].astype(str).tolist()
    cif_dir = paths.root / cfg.get("cif_dir", "data/raw/mataltmag/cifs")
    missing_cifs: list[str] = []
    if not fixture and not bool(cfg.get("allow_fixture", False)) and bool(cfg.get("skip_missing_cifs", False)):
        available = []
        for material_id in material_ids:
            if (cif_dir / f"{material_id}.cif").exists():
                available.append(material_id)
            else:
                missing_cifs.append(material_id)
        material_ids = available
    extractor = MatAltMagEmbeddingExtractor(
        checkpoint_path=paths.root / cfg.get("checkpoint_path", "external/MatAltMag/checkpoints/model.pt"),
        mataltmag_root=paths.root / cfg.get("mataltmag_root", "external/MatAltMag"),
        cif_dir=cif_dir,
        atom_init_path=paths.root / cfg.get("atom_init_path", "data/raw/mataltmag/atom_init.json"),
        embedding_dim=int(cfg.get("embedding_dim", 16)),
        seed=int(cfg.get("seed", 20260602)),
        fixture_mode=fixture or bool(cfg.get("allow_fixture", False)),
        batch_size=int(cfg.get("batch_size", 128)),
        radius=float(cfg.get("radius", 20)),
        max_num_nbr=int(cfg.get("max_num_nbr", 12)),
        dmin=float(cfg.get("dmin", 0)),
        step=float(cfg.get("step", 0.2)),
        n_conv=int(cfg.get("n_conv", 3)),
        head_output_dim=int(cfg.get("head_output_dim", 2)),
        drop_rate=float(cfg.get("drop_rate", 0)),
        sample_size=int(cfg.get("sample_size", 10)),
        device=str(cfg.get("device", "cpu")),
        part_dir=paths.root / cfg["part_dir"] if cfg.get("part_dir") else None,
        part_size=int(cfg.get("part_size", 2048)),
    )
    embeddings = extractor.extract(material_ids)
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = paths.root / out_path
    ensure_parent(out_path)
    write_table(embeddings, out_path)
    check_path = paths.root / cfg.get("embedding_check", "results/metrics/embedding_export_check.json")
    ensure_parent(check_path)
    check_path.write_text(
        json.dumps(
            {
                "requested_rows": int(len(index)),
                "exported_rows": int(len(embeddings)),
                "missing_cif_rows": int(len(missing_cifs)),
                "missing_cif_ids_sample": missing_cifs[:50],
                "embedding_dim": int(len([col for col in embeddings.columns if col.startswith("emb_")])) if not embeddings.empty else int(cfg.get("embedding_dim", 16)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--out", default="data/interim/gnn_embeddings.parquet")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    try:
        extract(args.config, args.out, args.fixture)
    except (GateError, FileNotFoundError) as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
