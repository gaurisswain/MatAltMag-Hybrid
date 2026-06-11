from __future__ import annotations

from pathlib import Path

import pandas as pd

from mataltmag_hybrid.config import GateError


ID_COLUMNS = ("material_id", "mp_id", "id", "Materials_id", "name")
FORMULA_COLUMNS = ("formula", "pretty_formula", "formula_pretty", "composition")


def read_mataltmag_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if any(col in df.columns for col in ID_COLUMNS):
        return df

    raw = pd.read_csv(path, header=None)
    if raw.empty:
        return raw
    if raw.shape[1] == 1:
        return pd.DataFrame({"material_id": raw.iloc[:, 0].astype(str)})

    first = raw.iloc[:, 0].astype(str)
    second = raw.iloc[:, 1].astype(str)
    first_is_row_number = first.str.fullmatch(r"\d+").all()
    second_is_numeric = pd.to_numeric(second, errors="coerce").notna().all()
    first_is_material_id = first.str.match(r"^(mp-|mvc-)").all()
    if first_is_material_id and second_is_numeric:
        return pd.DataFrame({"material_id": first, "gnn_probability": second})
    if first_is_row_number:
        return pd.DataFrame({"material_id": second})
    return pd.DataFrame({"formula": first, "material_id": second})


def normalize_material_id_columns(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = df.copy()
    id_col = next((col for col in ID_COLUMNS if col in df.columns), None)
    if id_col is None:
        raise GateError(f"{source_file} has no recognized material-id column; expected one of {ID_COLUMNS}")
    df["material_id"] = df[id_col].astype(str)
    if "formula" not in df.columns:
        formula_col = next((col for col in FORMULA_COLUMNS if col in df.columns), None)
        df["formula"] = df[formula_col].astype(str) if formula_col else df["material_id"]
    return df


def load_labeled_index(raw_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for filename, label, split in [
        ("label0.csv", 0, "label0"),
        ("label1.csv", 1, "label1"),
    ]:
        path = raw_dir / filename
        if not path.exists():
            raise GateError(f"Phase 0/1 requires raw MatAltMag label file: {path}")
        df = normalize_material_id_columns(read_mataltmag_csv(path), filename)
        df = df[["material_id", "formula"]].copy()
        df["source_split"] = split
        df["label"] = label
        df["is_candidate"] = False
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_candidate_index(raw_dir: Path) -> pd.DataFrame:
    candidates = []
    for filename in ("Candidate_for_DFT_validate.csv", "candidate.csv"):
        path = raw_dir / filename
        if path.exists():
            df = normalize_material_id_columns(read_mataltmag_csv(path), filename)
            keep = ["material_id", "formula"]
            out = df[keep].copy()
            out["source_split"] = "unconfirmed_candidate"
            out["label"] = "unknown"
            out["is_candidate"] = True
            candidates.append(out)
    if not candidates:
        raise GateError(f"Phase 0/1 requires candidate CSV under {raw_dir}")
    return pd.concat(candidates, ignore_index=True).drop_duplicates("material_id")


def build_dataset_index(raw_dir: Path, gnn_outputs: pd.DataFrame | None = None) -> pd.DataFrame:
    labels = load_labeled_index(raw_dir)
    if labels["material_id"].duplicated().any():
        labels = labels.sort_values(["material_id", "label"], ascending=[True, False]).drop_duplicates(
            "material_id", keep="first"
        )

    candidates = load_candidate_index(raw_dir)
    candidates = candidates.loc[~candidates["material_id"].isin(labels["material_id"])].copy()

    index = pd.concat([labels, candidates], ignore_index=True)
    if index["material_id"].duplicated().any():
        duplicated = index.loc[index["material_id"].duplicated(), "material_id"].tolist()
        raise GateError(f"Duplicate material_id values in dataset index: {duplicated[:10]}")
    index["label"] = index["label"].astype(str)
    index["gnn_probability"] = pd.NA
    if gnn_outputs is not None and not gnn_outputs.empty:
        index = index.drop(columns=["gnn_probability"]).merge(
            gnn_outputs[["material_id", "gnn_probability"]], how="left", on="material_id"
        )
    return index
