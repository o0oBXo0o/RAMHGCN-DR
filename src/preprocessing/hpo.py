"""Build disease features from a versioned HPO annotation snapshot."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def normalize_disease_id(value: object) -> str:
    """Return the numeric component of an MIM/OMIM identifier."""
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("OMIM:", "").replace("MIM:", "").replace("MIM", "")
    return re.sub(r"[^0-9]", "", text)


def load_diseases(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if "Disease" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "Disease"})
    frame["Disease"] = frame["Disease"].astype(str).str.strip()
    frame["mim_numeric"] = frame["Disease"].map(normalize_disease_id)
    if frame["mim_numeric"].eq("").any():
        raise ValueError("At least one disease identifier cannot be normalized")
    if frame["mim_numeric"].duplicated().any():
        raise ValueError("Disease identifiers must be unique")
    return frame


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases: Dict[str, str] = {}
    for column in frame.columns:
        canonical = column.strip().lower()
        if canonical in ("databaseid", "database_id", "db_object_id"):
            aliases[column] = "database_id"
        elif canonical in ("diseasename", "disease_name", "db_name"):
            aliases[column] = "disease_name"
        elif canonical in ("qualifier",):
            aliases[column] = "qualifier"
        elif canonical in ("hpo_id", "hpoid", "hpo id"):
            aliases[column] = "hpo_id"
        elif canonical in ("aspect",):
            aliases[column] = "aspect"
    frame = frame.rename(columns=aliases)
    required = {"database_id", "disease_name", "hpo_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("HPOA is missing columns: {}".format(sorted(missing)))
    return frame


def load_hpo_annotations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        keep_default_na=False,
    )
    frame = _standardize_columns(frame)
    frame["mim_numeric"] = frame["database_id"].map(normalize_disease_id)
    if "qualifier" in frame.columns:
        frame = frame[frame["qualifier"].str.upper() != "NOT"].copy()
    if "aspect" in frame.columns:
        frame = frame[frame["aspect"].str.upper().isin(["P", "O", ""])].copy()
    return frame[frame["hpo_id"].str.startswith("HP:")].copy()


def build_ic_features(
    diseases: pd.DataFrame,
    annotations: pd.DataFrame,
    l2_normalize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the source IC weighting and optional row-wise L2 normalization."""
    selected = annotations[
        annotations["mim_numeric"].isin(diseases["mim_numeric"])
    ].copy()
    total_diseases = annotations["mim_numeric"].nunique()
    frequencies = (
        annotations[["mim_numeric", "hpo_id"]]
        .drop_duplicates()
        .groupby("hpo_id")["mim_numeric"]
        .nunique()
        .reset_index(name="disease_frequency")
    )
    frequencies["ic"] = -np.log(
        (frequencies["disease_frequency"] + 1.0) / (total_diseases + 1.0)
    )
    selected = selected.merge(frequencies[["hpo_id", "ic"]], on="hpo_id", how="left")
    selected["value"] = selected["ic"].fillna(0.0)
    mapping_columns = ["mim_numeric", "database_id", "disease_name", "hpo_id", "ic"]
    long_mapping = selected[mapping_columns].drop_duplicates()
    feature_frame = selected.pivot_table(
        index="mim_numeric",
        columns="hpo_id",
        values="value",
        aggfunc="max",
        fill_value=0.0,
    )
    feature_frame = feature_frame.reindex(diseases["mim_numeric"], fill_value=0.0)
    feature_frame.index = diseases["Disease"].values
    feature_frame.index.name = "Disease"
    missing = diseases[
        ~diseases["mim_numeric"].isin(selected["mim_numeric"])
    ].copy()
    if l2_normalize and feature_frame.shape[1]:
        values = normalize(feature_frame.values, norm="l2", axis=1)
        feature_frame = pd.DataFrame(
            values,
            index=feature_frame.index,
            columns=feature_frame.columns,
        )
    return feature_frame, long_mapping, missing


def generate_disease_features(
    disease_path: Path,
    annotation_path: Path,
    output_dir: Path,
    l2_normalize: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    diseases = load_diseases(disease_path)
    annotations = load_hpo_annotations(annotation_path)
    features, mapping, missing = build_ic_features(
        diseases, annotations, l2_normalize=l2_normalize
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "disease_hpo_ic.csv")
    mapping.to_csv(output_dir / "disease_hpo_terms.tsv", sep="\t", index=False)
    missing.to_csv(output_dir / "diseases_without_hpo.tsv", sep="\t", index=False)
    return features, {
        "diseases": int(len(diseases)),
        "hpo_dimensions": int(features.shape[1]),
        "selected_annotations": int(len(mapping)),
        "diseases_without_hpo": int(len(missing)),
    }
