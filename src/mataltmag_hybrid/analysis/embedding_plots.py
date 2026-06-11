from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from mataltmag_hybrid.config import ensure_parent, load_paths
from mataltmag_hybrid.io.cache import read_table


def write_embedding_pca_plots(paths_config: str = "configs/paths.yaml", max_points: int = 12000) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = load_paths(paths_config)
    index = read_table(paths.path("dataset_index"))
    embeddings = read_table(paths.path("gnn_embeddings"))
    features = read_table(paths.path("explicit_features"))
    data = index.merge(embeddings, on="material_id", validate="one_to_one").merge(
        features[["material_id", "crystal_system", "space_group_number", "magnetic_elements"]],
        on="material_id",
        validate="one_to_one",
    )
    emb_cols = [col for col in data.columns if col.startswith("emb_")]
    sample = data.sample(n=min(max_points, len(data)), random_state=20260602) if len(data) > max_points else data
    coords = PCA(n_components=2, random_state=20260602).fit_transform(sample[emb_cols])
    plot_data = sample[["material_id", "label", "source_split", "crystal_system", "magnetic_elements"]].copy()
    plot_data["pc1"] = coords[:, 0]
    plot_data["pc2"] = coords[:, 1]

    out_paths: list[Path] = []
    specs = [
        ("label", "embedding_umap_by_label.png"),
        ("crystal_system", "embedding_umap_by_space_group.png"),
    ]
    for color_col, filename in specs:
        out = paths.root / "results" / "figures" / filename
        ensure_parent(out)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
        for value, group in plot_data.groupby(color_col, dropna=False):
            ax.scatter(group["pc1"], group["pc2"], s=5, alpha=0.55, label=str(value))
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"Frozen MatAltMag Embeddings by {color_col.replace('_', ' ')}")
        ax.legend(markerscale=2, fontsize=7, frameon=False, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        out_paths.append(out)

    csv_out = paths.root / "results" / "metrics" / "embedding_pca_projection.csv"
    ensure_parent(csv_out)
    plot_data.to_csv(csv_out, index=False)
    out_paths.append(csv_out)
    return out_paths
