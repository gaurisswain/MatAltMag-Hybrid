from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from mataltmag_hybrid.dft.make_vasp_inputs import make_inputs
from mataltmag_hybrid.models.predict_candidates import REQUIRED_RANKING_COLUMNS


def test_required_ranking_columns():
    assert "material_id" in REQUIRED_RANKING_COLUMNS
    assert "hybrid_probability" in REQUIRED_RANKING_COLUMNS
    assert "dft_selected" in REQUIRED_RANKING_COLUMNS


def test_dft_generator_never_writes_potcar(tmp_path, monkeypatch):
    ranked = tmp_path / "ranked.csv"
    pd.DataFrame(
        {
            "material_id": ["mp-1"],
            "dft_selected": [True],
            "dft_priority_score": [1.0],
        }
    ).to_csv(ranked, index=False)
    config = tmp_path / "dft.yaml"
    config.write_text(
        f"ranked_candidates: {ranked}\noutput_dir: {tmp_path / 'dft'}\nn_select: 1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    out = make_inputs(str(config))
    assert out.exists()
    assert not any("POTCAR" in files for _, _, files in os.walk(out))

