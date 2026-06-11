from __future__ import annotations

import pandas as pd

from mataltmag_hybrid.io.load_mataltmag import build_dataset_index


def test_unique_material_ids(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"material_id": ["m1"], "formula": ["FeO"]}).to_csv(raw / "label0.csv", index=False)
    pd.DataFrame({"material_id": ["m2"], "formula": ["MnO"]}).to_csv(raw / "label1.csv", index=False)
    pd.DataFrame({"material_id": ["m3"], "formula": ["CrO"]}).to_csv(raw / "candidate.csv", index=False)
    index = build_dataset_index(raw)
    assert index["material_id"].is_unique
    assert set(index["source_split"]) == {"label0", "label1", "unconfirmed_candidate"}


def test_upstream_no_header_mataltmag_csvs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "label0.csv").write_text("0,mp-1\n1,mp-2\n")
    (raw / "label1.csv").write_text("Fe2 O3,mp-3\n")
    (raw / "candidate.csv").write_text("0,mp-4\n")

    index = build_dataset_index(raw)

    assert index["material_id"].tolist() == ["mp-1", "mp-2", "mp-3", "mp-4"]
    assert index.loc[index["material_id"] == "mp-3", "formula"].item() == "Fe2 O3"
    assert index["material_id"].is_unique


def test_labeled_rows_take_precedence_over_candidate_rows(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"material_id": ["m1", "m2"], "formula": ["FeO", "MnO"]}).to_csv(raw / "label0.csv", index=False)
    pd.DataFrame({"material_id": ["m1"], "formula": ["FeO"]}).to_csv(raw / "label1.csv", index=False)
    pd.DataFrame({"material_id": ["m1", "m3"], "formula": ["FeO", "CrO"]}).to_csv(raw / "candidate.csv", index=False)

    index = build_dataset_index(raw)

    assert index["material_id"].is_unique
    assert index.loc[index["material_id"] == "m1", "label"].item() == "1"
    assert set(index.loc[index["is_candidate"], "material_id"]) == {"m3"}
