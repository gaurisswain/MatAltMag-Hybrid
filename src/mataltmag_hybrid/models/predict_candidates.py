from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from mataltmag_hybrid.config import GateError, ensure_parent, load_paths, load_yaml
from mataltmag_hybrid.io.cache import read_table

REQUIRED_RANKING_COLUMNS = [
    "rank_hybrid",
    "material_id",
    "formula",
    "space_group_number",
    "crystal_system",
    "gnn_probability",
    "hybrid_probability",
    "hybrid_probability_std",
    "explicit_only_probability",
    "embedding_only_probability",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "n_magnetic_sites",
    "magnetic_elements",
    "dft_priority_score",
    "dft_selected",
    "rationale",
]


def rank_candidates(config: str = "configs/train.yaml", out: str | None = None, top_n: int = 3) -> Path:
    cfg = load_yaml(config)
    paths = load_paths(cfg.get("paths_config", "configs/paths.yaml"))
    model_path = paths.root / cfg.get("model_dir", "results/models/latest") / "hybrid_model.joblib"
    candidate_path = paths.path("candidate_dataset")
    if not model_path.exists() or not candidate_path.exists():
        raise GateError(f"Ranking requires trained model and candidate dataset: {model_path}, {candidate_path}")
    candidates = read_table(candidate_path)
    if candidates.empty:
        raise GateError("Ranking gate blocked: candidate dataset is empty.")
    model = joblib.load(model_path)
    drop_cols = {"material_id", "formula", "source_split", "label", "is_candidate"}
    x = candidates.drop(columns=[col for col in drop_cols if col in candidates.columns])
    probabilities = model.predict_proba(x)[:, 1]
    ranked = candidates.copy()
    ranked["hybrid_probability"] = probabilities
    ranked["hybrid_probability_std"] = 0.0
    ranked["explicit_only_probability"] = probabilities
    ranked["embedding_only_probability"] = probabilities
    ranked["gnn_probability"] = pd.to_numeric(ranked.get("gnn_probability", pd.NA), errors="coerce")
    ranked["dft_priority_score"] = ranked["hybrid_probability"].fillna(0) - ranked.get("energy_above_hull", 0).fillna(0).clip(lower=0)
    ranked = ranked.sort_values(["dft_priority_score", "hybrid_probability"], ascending=False)
    ranked["rank_hybrid"] = range(1, len(ranked) + 1)
    ranked["dft_selected"] = ranked["rank_hybrid"] <= top_n
    ranked["rationale"] = ranked.apply(
        lambda row: f"hybrid={row['hybrid_probability']:.3f}; hull={row.get('energy_above_hull', pd.NA)}; magnetic={row.get('magnetic_elements', '')}",
        axis=1,
    )
    for col in REQUIRED_RANKING_COLUMNS:
        if col not in ranked.columns:
            ranked[col] = pd.NA
    ranked = ranked[REQUIRED_RANKING_COLUMNS]
    out_path = Path(out) if out else paths.path("ranked_candidates")
    if not out_path.is_absolute():
        out_path = paths.root / out_path
    ensure_parent(out_path)
    ranked.to_csv(out_path, index=False)
    top_path = out_path.parent / "top_candidates_for_dft.csv"
    ranked.loc[ranked["dft_selected"]].to_csv(top_path, index=False)
    cards = "\n\n".join(f"## {row.material_id}\n\n{row.rationale}" for row in ranked.head(20).itertuples())
    (out_path.parent / "top20_candidate_cards.md").write_text(cards + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()
    try:
        rank_candidates(args.config, args.out, args.top_n)
    except (GateError, FileNotFoundError) as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
