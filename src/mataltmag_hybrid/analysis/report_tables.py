from __future__ import annotations

from pathlib import Path


def _mark(path: str) -> str:
    return "x" if Path(path).exists() else " "


def write_reproducibility_checklist(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Reproducibility Checklist\n\n"
        f"- [{_mark('data/raw/mataltmag/label0.csv')}] Raw MatAltMag files present\n"
        f"- [{_mark('data/processed/train_dataset.parquet')}] Candidate rows excluded from supervised training\n"
        f"- [{_mark('data/interim/gnn_embeddings.parquet')}] Embeddings exported from frozen checkpoint\n"
        f"- [{_mark('results/metrics/embedding_export_check.json')}] Missing-CIF exclusions documented\n"
        f"- [{_mark('results/metrics/feature_quality_report.csv')}] Feature quality report generated\n"
        f"- [{_mark('results/metrics/model_comparison.csv')}] Model comparison table generated\n"
        f"- [{_mark('results/candidates/ranked_candidates.csv')}] Ranked candidates generated from trained model\n"
        f"- [{_mark('results/figures/embedding_umap_by_label.png')}] Embedding projection figure generated\n"
        f"- [{_mark('results/metrics/surrogate_gnn_explanation.csv')}] Surrogate GNN explanation table generated\n"
        f"- [{_mark('results/metrics/error_analysis.csv')}] Candidate error/rank-shift analysis generated\n"
        f"- [{' ' if list(Path('results/dft').rglob('POTCAR')) else 'x'}] DFT folders omit POTCAR\n",
        encoding="utf-8",
    )
