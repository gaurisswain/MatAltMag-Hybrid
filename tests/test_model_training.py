from __future__ import annotations

import pandas as pd

from mataltmag_hybrid.models.split import candidate_rows, supervised_rows


def test_candidates_excluded_from_supervised_training():
    df = pd.DataFrame(
        {
            "material_id": ["neg", "pos", "cand"],
            "label": [0, 1, "unknown"],
            "is_candidate": [False, False, True],
        }
    )
    assert set(supervised_rows(df)["material_id"]) == {"neg", "pos"}
    assert set(candidate_rows(df)["material_id"]) == {"cand"}


def test_one_to_one_join_alignment():
    labels = pd.DataFrame({"material_id": ["a", "b"], "label": [0, 1], "is_candidate": [False, False]})
    features = pd.DataFrame({"material_id": ["a", "b"], "x": [1.0, 2.0]})
    embeddings = pd.DataFrame({"material_id": ["a", "b"], "emb_000": [0.1, 0.2]})
    joined = labels.merge(features, on="material_id", validate="one_to_one").merge(embeddings, on="material_id", validate="one_to_one")
    assert len(joined) == 2

