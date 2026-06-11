from __future__ import annotations

import pandas as pd

from mataltmag_hybrid.features.composition import composition_features
from mataltmag_hybrid.features.validate_features import feature_quality_report


def test_composition_features_have_expected_fields():
    features = composition_features("Fe2O3")
    assert features["n_magnetic_sites"] == 2
    assert features["magnetic_elements"] == "Fe"
    assert features["num_elements"] == 2


def test_feature_quality_flags_constant_columns():
    report = feature_quality_report(pd.DataFrame({"material_id": ["a", "b"], "x": [1, 1], "y": [1, 2]}))
    by_name = report.set_index("feature")
    assert bool(by_name.loc["x", "all_constant"])
    assert not bool(by_name.loc["y", "all_constant"])

