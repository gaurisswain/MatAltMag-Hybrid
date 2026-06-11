from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from mataltmag_hybrid.config import GateError, ensure_parent, load_paths, load_yaml
from mataltmag_hybrid.features.composition import parse_formula
from mataltmag_hybrid.features.validate_features import assert_trainable_features
from mataltmag_hybrid.io.cache import read_table, write_table
from mataltmag_hybrid.models.split import make_split, supervised_rows


DROP_COLS = {"material_id", "formula", "source_split", "label", "is_candidate"}


def _assemble(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [paths.path("dataset_index"), paths.path("explicit_features"), paths.path("gnn_embeddings")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise GateError(f"Training requires missing table(s): {', '.join(missing)}")
    index = read_table(paths.path("dataset_index"))
    features = read_table(paths.path("explicit_features"))
    embeddings = read_table(paths.path("gnn_embeddings"))
    data = index.merge(features, on="material_id", validate="one_to_one").merge(embeddings, on="material_id", validate="one_to_one")
    return index, data


def _feature_matrix(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    base = data.drop(columns=[col for col in DROP_COLS if col in data.columns]).copy()
    emb_cols = [col for col in base.columns if col.startswith("emb_")]
    if mode == "gnn_probability_only":
        return base[["gnn_probability"]] if "gnn_probability" in base.columns else pd.DataFrame(index=base.index)
    if mode == "explicit_features_only":
        drop = set(emb_cols) | {"gnn_probability"}
        return base.drop(columns=[col for col in drop if col in base.columns])
    if mode == "gnn_embedding_only":
        return base[emb_cols]
    if mode in {"hybrid_concatenation", "calibrated_hybrid_ensemble"}:
        return base.drop(columns=["gnn_probability"], errors="ignore")
    raise ValueError(f"Unknown feature mode: {mode}")


def _preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes("number").columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
        ]
    )


def _base_estimator(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=96, random_state=seed, class_weight="balanced", n_jobs=1)


def _scores(y_true: pd.Series, proba) -> dict[str, float]:
    y_true = y_true.astype(int)
    pred = proba >= 0.5
    out = {
        "auroc_test": roc_auc_score(y_true, proba) if y_true.nunique() == 2 else pd.NA,
        "auprc_test": average_precision_score(y_true, proba),
        "balanced_accuracy_test": balanced_accuracy_score(y_true, pred),
        "brier_score_test": brier_score_loss(y_true, proba),
    }
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    for k in (10, 25, 50):
        kk = min(k, len(y_true))
        order = pd.Series(proba, index=y_true.index).sort_values(ascending=False).index[:kk]
        positives_at_k = int(y_true.loc[order].sum()) if kk else 0
        out[f"precision_at_{k}"] = float(positives_at_k / kk) if kk else 0.0
        out[f"recall_at_{k}"] = float(positives_at_k / y_true.sum()) if y_true.sum() else 0.0
        out[f"enrichment_at_{k}"] = float(out[f"precision_at_{k}"] / prevalence) if prevalence else pd.NA
    return out


def _chemical_system(formula: str) -> str:
    elements = sorted(parse_formula(str(formula)).keys())
    return "-".join(elements) if elements else str(formula)


def _group_holdout_split(trainable: pd.DataFrame, seed: int, test_fraction: float = 0.25) -> tuple[set[str], set[str]]:
    groups = trainable.assign(_group=trainable["formula"].map(_chemical_system))
    positive_groups = groups.loc[groups["label"].astype(int) == 1, "_group"].drop_duplicates().sample(frac=1, random_state=seed).tolist()
    negative_groups = groups.loc[groups["label"].astype(int) == 0, "_group"].drop_duplicates().sample(frac=1, random_state=seed).tolist()
    target_pos = max(1, int(groups["label"].astype(int).sum() * test_fraction))
    target_neg = max(1, int((groups["label"].astype(int) == 0).sum() * test_fraction))
    selected: set[str] = set()
    pos_count = neg_count = 0
    for group in positive_groups:
        selected.add(group)
        part = groups.loc[groups["_group"] == group, "label"].astype(int)
        pos_count += int(part.sum())
        neg_count += int((part == 0).sum())
        if pos_count >= target_pos:
            break
    for group in negative_groups:
        if group in selected:
            continue
        selected.add(group)
        part = groups.loc[groups["_group"] == group, "label"].astype(int)
        neg_count += int((part == 0).sum())
        if neg_count >= target_neg:
            break
    test_ids = set(groups.loc[groups["_group"].isin(selected), "material_id"].astype(str))
    train_ids = set(groups.loc[~groups["_group"].isin(selected), "material_id"].astype(str))
    if not train_ids or not test_ids:
        raise GateError("Grouped split failed to produce non-empty train/test sets.")
    return train_ids, test_ids


def train(config: str) -> Path:
    cfg = load_yaml(config)
    paths = load_paths(cfg.get("paths_config", "configs/paths.yaml"))
    _, data = _assemble(paths)
    trainable = supervised_rows(data)
    minimum = int(cfg.get("minimum_labeled_rows", 4))
    if len(trainable) < minimum or trainable["label"].nunique() < 2:
        raise GateError(
            f"Training gate blocked: need at least {minimum} labeled rows with both classes; "
            f"found {len(trainable)} rows and {trainable['label'].nunique()} class(es)."
        )
    assert_trainable_features(trainable)
    split = make_split(data, int(cfg.get("seed", 20260602)), paths.root / cfg.get("splits_out", "data/processed/splits.json"))
    seed = int(cfg.get("seed", 20260602))
    train_ids = set(split["train"])
    test_ids = set(split["test"])
    train_part = trainable.loc[trainable["material_id"].isin(train_ids)].copy()
    test_part = trainable.loc[trainable["material_id"].isin(test_ids)].copy()
    y_train = train_part["label"].astype(int)
    y_test = test_part["label"].astype(int)
    model_dir = paths.root / cfg.get("model_dir", "results/models/latest")
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    prediction_rows = []
    final_model = None
    split_specs = [
        ("stratified", set(split["train"]), set(split["test"])),
        ("chemical_system_grouped", *_group_holdout_split(trainable, seed)),
    ]
    for split_name, train_ids_current, test_ids_current in split_specs:
        train_part_current = trainable.loc[trainable["material_id"].isin(train_ids_current)].copy()
        test_part_current = trainable.loc[trainable["material_id"].isin(test_ids_current)].copy()
        y_train_current = train_part_current["label"].astype(int)
        y_test_current = test_part_current["label"].astype(int)
        for name in [
        "gnn_probability_only",
        "explicit_features_only",
        "gnn_embedding_only",
        "hybrid_concatenation",
        "calibrated_hybrid_ensemble",
        ]:
            x_train = _feature_matrix(train_part_current, name)
            x_test = _feature_matrix(test_part_current, name)
            row = {
                "split": split_name,
                "model": name,
                "n_train": len(train_part_current),
                "n_test": len(test_part_current),
                "n_split_train": len(train_ids_current),
                "n_split_test": len(test_ids_current),
                "n_test_positive": int(y_test_current.sum()),
                "n_test_negative": int((y_test_current == 0).sum()),
            }
            if x_train.empty or x_train.select_dtypes("number").isna().all().all():
                row["status"] = "skipped"
                row["skip_reason"] = "no usable feature values for supervised rows"
                metrics.append(row)
                continue
            if name == "gnn_probability_only":
                model = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(class_weight="balanced", max_iter=500))
            elif name == "calibrated_hybrid_ensemble":
                model = make_pipeline(_preprocessor(x_train), CalibratedClassifierCV(_base_estimator(seed), method="sigmoid", cv=3))
            else:
                model = make_pipeline(_preprocessor(x_train), _base_estimator(seed))
            model.fit(x_train, y_train_current)
            proba_test = model.predict_proba(x_test)[:, 1]
            row.update({"status": "ok", "skip_reason": ""})
            row.update(_scores(y_test_current, proba_test))
            metrics.append(row)
            for material_id, y_true, proba in zip(test_part_current["material_id"], y_test_current, proba_test):
                prediction_rows.append({"split": split_name, "model": name, "material_id": material_id, "label": int(y_true), "probability": float(proba)})
            if split_name == "stratified" and name == "calibrated_hybrid_ensemble":
                final_x = _feature_matrix(trainable, name)
                final_y = trainable["label"].astype(int)
                final_model = make_pipeline(_preprocessor(final_x), CalibratedClassifierCV(_base_estimator(seed), method="sigmoid", cv=3)).fit(final_x, final_y)

    if final_model is None:
        final_x = _feature_matrix(trainable, "hybrid_concatenation")
        final_y = trainable["label"].astype(int)
        final_model = make_pipeline(_preprocessor(final_x), _base_estimator(seed)).fit(final_x, final_y)
    joblib.dump(final_model, model_dir / "hybrid_model.joblib")
    metrics_path = paths.root / cfg.get("metrics_out", "results/metrics/model_comparison.csv")
    ensure_parent(metrics_path)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    write_table(pd.DataFrame(prediction_rows), paths.root / "results" / "metrics" / "cv_predictions.parquet")
    write_table(trainable, paths.path("train_dataset"))
    candidates = data.loc[~data["material_id"].isin(trainable["material_id"])]
    write_table(candidates, paths.path("candidate_dataset"))
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    try:
        train(args.config)
    except (GateError, FileNotFoundError, ValueError) as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
