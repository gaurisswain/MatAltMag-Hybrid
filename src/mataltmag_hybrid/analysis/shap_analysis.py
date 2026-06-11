from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mataltmag_hybrid.config import ensure_parent, load_paths, load_yaml
from mataltmag_hybrid.io.cache import read_table


def write_hybrid_feature_importance(config: str = "configs/train.yaml", top_n: int = 30) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_yaml(config)
    paths = load_paths(cfg.get("paths_config", "configs/paths.yaml"))
    model_path = paths.root / cfg.get("model_dir", "results/models/latest") / "hybrid_model.joblib"
    train = read_table(paths.path("train_dataset"))
    model = joblib.load(model_path)
    drop_cols = {"material_id", "formula", "source_split", "label", "is_candidate"}
    x = train.drop(columns=[col for col in drop_cols if col in train.columns])
    pipeline = model
    pre = pipeline.named_steps["columntransformer"]
    names = pre.get_feature_names_out()
    if "extratreesclassifier" in pipeline.named_steps:
        values = pipeline.named_steps["extratreesclassifier"].feature_importances_
    elif "calibratedclassifiercv" in pipeline.named_steps:
        calibrated = pipeline.named_steps["calibratedclassifiercv"]
        values = np.mean([cc.estimator.feature_importances_ for cc in calibrated.calibrated_classifiers_], axis=0)
    else:
        raise ValueError(f"Unsupported model steps for feature importance: {list(pipeline.named_steps)}")
    importances = pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    )
    out_csv = paths.root / "results" / "metrics" / "hybrid_feature_importance.csv"
    ensure_parent(out_csv)
    importances.to_csv(out_csv, index=False)

    plot_data = importances.head(top_n).iloc[::-1]
    out_png = paths.root / "results" / "figures" / "shap_summary.png"
    ensure_parent(out_png)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    ax.barh(plot_data["feature"], plot_data["importance"])
    ax.set_xlabel("ExtraTrees feature importance")
    ax.set_title("Hybrid Model Feature Importance")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return [out_csv, out_png]
