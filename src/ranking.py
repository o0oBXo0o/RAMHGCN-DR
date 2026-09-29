"""Known-positive masking and global/bidirectional Top-K exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from data import MultiplexDataset
from runtime import write_json


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def export_rankings(
    embeddings: np.ndarray,
    dataset: MultiplexDataset,
    output_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_matrix = sigmoid(
        embeddings[: dataset.n_drugs]
        @ embeddings[dataset.n_drugs : dataset.n_nodes].T
    )
    known = dataset.known_pairs()
    if bool(config["mask_known_positives"]):
        for drug, disease_global in known:
            score_matrix[drug, disease_global - dataset.n_drugs] = np.nan

    drug_index = np.repeat(np.arange(dataset.n_drugs), dataset.n_diseases)
    disease_index = np.tile(np.arange(dataset.n_diseases), dataset.n_drugs)
    scores = score_matrix.ravel()
    valid = ~np.isnan(scores)
    drug_index, disease_index, scores = drug_index[valid], disease_index[valid], scores[valid]
    order = np.argsort(-scores)
    drug_index, disease_index, scores = drug_index[order], disease_index[order], scores[order]
    drug_ids = np.asarray(dataset.drug_ids, dtype=object)
    drug_names = np.asarray(dataset.drug_names, dtype=object)
    disease_ids = np.asarray(dataset.disease_ids, dtype=object)
    disease_names = np.asarray(dataset.disease_names, dtype=object)
    global_frame = pd.DataFrame(
        {
            "rank": np.arange(1, len(scores) + 1, dtype=np.int64),
            "drug_id": drug_ids[drug_index],
            "drug_name": drug_names[drug_index],
            "disease_id": disease_ids[disease_index],
            "disease_name": disease_names[disease_index],
            "association_score": scores.astype(np.float64),
        }
    )

    top_k = int(config["top_k"])
    drug_rows = []
    for drug in range(dataset.n_drugs):
        row = np.nan_to_num(score_matrix[drug], nan=-np.inf)
        indices = np.argpartition(row, -min(top_k, len(row)))[-min(top_k, len(row)) :]
        indices = indices[np.argsort(-row[indices])]
        for rank, disease in enumerate(indices, start=1):
            if np.isfinite(row[disease]):
                drug_rows.append(
                    [rank, dataset.drug_ids[drug], dataset.drug_names[drug], dataset.disease_ids[disease], dataset.disease_names[disease], float(row[disease])]
                )
    drug_frame = pd.DataFrame(
        drug_rows,
        columns=["rank", "drug_id", "drug_name", "disease_id", "disease_name", "association_score"],
    )

    disease_rows = []
    for disease in range(dataset.n_diseases):
        column = np.nan_to_num(score_matrix[:, disease], nan=-np.inf)
        indices = np.argpartition(column, -min(top_k, len(column)))[-min(top_k, len(column)) :]
        indices = indices[np.argsort(-column[indices])]
        for rank, drug in enumerate(indices, start=1):
            if np.isfinite(column[drug]):
                disease_rows.append(
                    [rank, dataset.drug_ids[drug], dataset.drug_names[drug], dataset.disease_ids[disease], dataset.disease_names[disease], float(column[drug])]
                )
    disease_frame = pd.DataFrame(
        disease_rows,
        columns=["rank", "drug_id", "drug_name", "disease_id", "disease_name", "association_score"],
    )

    paths = {
        "global_csv": output_dir / "global_ranking.csv",
        "drug_topk_csv": output_dir / "topk_diseases_by_drug.csv",
        "disease_topk_csv": output_dir / "topk_drugs_by_disease.csv",
    }
    global_frame.to_csv(paths["global_csv"], index=False)
    drug_frame.to_csv(paths["drug_topk_csv"], index=False)
    disease_frame.to_csv(paths["disease_topk_csv"], index=False)
    if bool(config.get("export_excel", True)):
        paths.update(
            {
                "global_xlsx": output_dir / "global_ranking.xlsx",
                "drug_topk_xlsx": output_dir / "topk_diseases_by_drug.xlsx",
                "disease_topk_xlsx": output_dir / "topk_drugs_by_disease.xlsx",
            }
        )
        global_frame.to_excel(paths["global_xlsx"], index=False)
        drug_frame.to_excel(paths["drug_topk_xlsx"], index=False)
        disease_frame.to_excel(paths["disease_topk_xlsx"], index=False)

    manifest = {
        "candidate_pairs": int(len(global_frame)),
        "known_pairs_masked": int(len(known)),
        "known_positive_intersection": 0,
        "drug_topk_rows": int(len(drug_frame)),
        "disease_topk_rows": int(len(disease_frame)),
        "score_interpretation": "model-derived association score; not a calibrated clinical probability",
        "files": {name: str(path) for name, path in paths.items()},
        "top10": global_frame.head(10).to_dict(orient="records"),
    }
    write_json(output_dir / "ranking_manifest.json", manifest)
    return manifest
