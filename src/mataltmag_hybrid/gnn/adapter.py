from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mataltmag_hybrid.config import GateError


@dataclass
class MatAltMagEmbeddingExtractor:
    checkpoint_path: Path | None = None
    mataltmag_root: Path | None = None
    cif_dir: Path | None = None
    atom_init_path: Path | None = None
    embedding_dim: int = 512
    seed: int = 20260602
    fixture_mode: bool = False
    batch_size: int = 128
    radius: float = 20.0
    max_num_nbr: int = 12
    dmin: float = 0.0
    step: float = 0.2
    n_conv: int = 3
    head_output_dim: int = 2
    drop_rate: float = 0.0
    sample_size: int = 10
    device: str = "cpu"
    part_dir: Path | None = None
    part_size: int = 2048

    def validate(self) -> None:
        if self.fixture_mode:
            return
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            raise GateError(
                "Frozen MatAltMag embedding extraction requires a trained checkpoint. "
                f"Missing: {self.checkpoint_path}"
            )
        for name, path in {
            "MatAltMag root": self.mataltmag_root,
            "CIF directory": self.cif_dir,
            "atom_init.json": self.atom_init_path,
        }.items():
            if path is None or not path.exists():
                raise GateError(f"Frozen MatAltMag embedding extraction requires {name}: {path}")

    def extract(self, material_ids: list[str]) -> pd.DataFrame:
        if not material_ids:
            return pd.DataFrame(columns=["material_id"] + [f"emb_{idx:03d}" for idx in range(self.embedding_dim)])
        self.validate()
        if not self.fixture_mode:
            return self._extract_real(material_ids)
        return self._extract_fixture(material_ids)

    def _extract_fixture(self, material_ids: list[str]) -> pd.DataFrame:
        rows = []
        for material_id in material_ids:
            digest = hashlib.sha256(f"{self.seed}:{material_id}".encode("utf-8")).digest()
            raw = np.frombuffer(digest, dtype=np.uint8).astype(float)
            values = np.resize(raw / 255.0, self.embedding_dim)
            row = {"material_id": material_id}
            row.update({f"emb_{idx:03d}": float(value) for idx, value in enumerate(values)})
            rows.append(row)
        return pd.DataFrame(rows)

    def _extract_real(self, material_ids: list[str]) -> pd.DataFrame:
        assert self.mataltmag_root is not None
        assert self.cif_dir is not None
        assert self.atom_init_path is not None
        missing = [material_id for material_id in material_ids if not (self.cif_dir / f"{material_id}.cif").exists()]
        if missing:
            raise GateError(
                "Frozen MatAltMag embedding extraction requires CIF files for every requested material. "
                f"Missing {len(missing)} of {len(material_ids)}; first missing IDs: {missing[:10]}"
            )
        completed: set[str] = set()
        part_paths: list[Path] = []
        if self.part_dir is not None:
            self.part_dir.mkdir(parents=True, exist_ok=True)
            part_paths = sorted(self.part_dir.glob("part_*.parquet"))
            for part_path in part_paths:
                part = pd.read_parquet(part_path)
                completed.update(part["material_id"].astype(str))
            material_ids = [material_id for material_id in material_ids if material_id not in completed]

        import torch
        from torch.utils.data import DataLoader, Dataset

        external_root = str(self.mataltmag_root.resolve())
        if external_root not in sys.path:
            sys.path.insert(0, external_root)
        from dataset_helper import AtomCustomJSONInitializer, GaussianDistance, collate_pool
        from model import CrystalGraph
        from pymatgen.core.structure import Structure

        class _CIFDataset(Dataset):
            def __init__(self, ids: list[str], cif_dir: Path, atom_init_path: Path, radius: float, max_num_nbr: int, dmin: float, step: float):
                self.ids = ids
                self.cif_dir = cif_dir
                self.radius = radius
                self.max_num_nbr = max_num_nbr
                self.ari = AtomCustomJSONInitializer(str(atom_init_path))
                self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

            def __len__(self) -> int:
                return len(self.ids)

            def __getitem__(self, idx: int):
                import torch
                import warnings

                material_id = self.ids[idx]
                crystal = Structure.from_file(str(self.cif_dir / f"{material_id}.cif"))
                atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
                all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
                all_nbrs = [sorted(nbrs, key=lambda item: item[1]) for nbrs in all_nbrs]
                nbr_fea_idx, nbr_fea, degree = [], [], []
                for nbr in all_nbrs:
                    if len(nbr) < self.max_num_nbr:
                        warnings.warn(f"{material_id} did not find enough neighbors to build graph.")
                        idx_values = [item[2] for item in nbr] + [0] * (self.max_num_nbr - len(nbr))
                        fea_values = [item[1] for item in nbr] + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                    else:
                        idx_values = [item[2] for item in nbr[: self.max_num_nbr]]
                        fea_values = [item[1] for item in nbr[: self.max_num_nbr]]
                    nbr_fea_idx.append(idx_values)
                    nbr_fea.append(fea_values)
                    degree.append(len(set(idx_values)))
                return (
                    torch.Tensor(atom_fea),
                    torch.Tensor(self.gdf.expand(np.array(nbr_fea))),
                    torch.LongTensor(np.array(nbr_fea_idx)),
                    torch.Tensor(np.array(degree)),
                ), torch.Tensor([-1.0]), material_id

        with self.atom_init_path.open("r", encoding="utf-8") as handle:
            atom_init = json.load(handle)
        orig_atom_fea_len = len(next(iter(atom_init.values())))
        nbr_fea_len = len(np.arange(self.dmin, self.radius + self.step, self.step))
        torch_device = torch.device(self.device)
        model = CrystalGraph(
            crystal_gnn_config={
                "orig_atom_fea_len": orig_atom_fea_len,
                "atom_fea_len": self.embedding_dim,
                "nbr_fea_len": nbr_fea_len,
                "n_conv": self.n_conv,
            },
            head_output_dim=self.head_output_dim,
            drop_rate=self.drop_rate,
            decoder_sample_size=self.sample_size,
            device=torch_device,
        )
        state = torch.load(self.checkpoint_path, map_location=torch_device)
        model.load_state_dict(state)
        model.to(torch_device)
        model.eval()

        dataset = _CIFDataset(material_ids, self.cif_dir, self.atom_init_path, self.radius, self.max_num_nbr, self.dmin, self.step)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_pool, num_workers=0)
        rows = []
        part_index = len(part_paths)

        def flush(force: bool = False) -> None:
            nonlocal rows, part_index
            if self.part_dir is None or not rows or (not force and len(rows) < self.part_size):
                return
            part = pd.DataFrame(rows)
            part.to_parquet(self.part_dir / f"part_{part_index:05d}.parquet", index=False)
            rows = []
            part_index += 1

        with torch.no_grad():
            for inputs, _, batch_ids in loader:
                atom_fea, nbr_fea, nbr_fea_idx, _, crystal_atom_idx = inputs
                atom_fea = atom_fea.to(torch_device)
                nbr_fea = nbr_fea.to(torch_device)
                nbr_fea_idx = nbr_fea_idx.to(torch_device)
                crystal_atom_idx = [idx.to(torch_device) for idx in crystal_atom_idx]
                embeddings = model.encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx).detach().cpu().numpy()
                for material_id, values in zip(batch_ids, embeddings):
                    row = {"material_id": str(material_id)}
                    row.update({f"emb_{idx:03d}": float(value) for idx, value in enumerate(values)})
                    rows.append(row)
                flush()
        flush(force=True)
        if self.part_dir is not None:
            all_parts = sorted(self.part_dir.glob("part_*.parquet"))
            frames = [pd.read_parquet(part_path) for part_path in all_parts]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return pd.DataFrame(rows)
