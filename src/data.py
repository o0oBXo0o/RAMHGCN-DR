"""Loading and validation for the versioned real Multiplex inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.io import loadmat

from config import resolve_path
from similarity import combine_similarity_blocks, load_similarity_network


Pair = Tuple[int, int]
RELATION_LABELS = {"1": "DD", "2": "DGD", "3": "DPD", "4": "DCD"}
RELATION_KEYS = {
    "DD": "IUI_Drug_Disease",
    "DGD": "IUI_Drug_Gene_Disease",
    "DPD": "IUI_Drug_Gene_Pathway_Gene_Disease",
    "DCD": "IUI_Drug_Gene_Complex_Gene_Disease",
}


@dataclass
class RelationLabels:
    positive: List[Pair]
    negative: List[Pair]


@dataclass
class MultiplexDataset:
    features: np.ndarray
    relation_order: List[str]
    relation_adjacencies: Dict[str, sparse.csr_matrix]
    node_similarity: sparse.csr_matrix
    similarity_stats: Dict[str, Any]
    relation_pairs: Dict[str, List[Pair]]
    labelled_edges: Dict[str, RelationLabels]
    drug_ids: List[str]
    drug_names: List[str]
    disease_ids: List[str]
    disease_names: List[str]

    @property
    def n_drugs(self) -> int:
        return len(self.drug_ids)

    @property
    def n_diseases(self) -> int:
        return len(self.disease_ids)

    @property
    def n_nodes(self) -> int:
        return self.n_drugs + self.n_diseases

    def known_pairs(self) -> Set[Pair]:
        result: Set[Pair] = set()
        for values in self.relation_pairs.values():
            result.update(values)
        return result

    def adjacency_stack(
        self, device: torch.device, excluded_pairs: Optional[Set[Pair]] = None
    ) -> torch.Tensor:
        excluded = excluded_pairs or set()
        matrices = []
        for relation in self.relation_order:
            if not excluded:
                matrix = self.relation_adjacencies[relation].toarray().astype(np.float32)
            else:
                matrix = self.node_similarity.toarray().astype(np.float32)
                for first, second in self.relation_pairs[relation]:
                    if (first, second) in excluded:
                        continue
                    matrix[first, second] = 1.0
                    matrix[second, first] = 1.0
            matrices.append(matrix)
        return torch.as_tensor(np.stack(matrices, axis=2), dtype=torch.float32, device=device)


def _read_labelled_edges(split_dir: Path, relation_order: Sequence[str]) -> Dict[str, RelationLabels]:
    result = {name: RelationLabels([], []) for name in relation_order}
    for filename in ("train.txt", "valid.txt", "test.txt"):
        path = split_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                words = line.split()
                if not words:
                    continue
                if len(words) != 4:
                    raise ValueError("{}:{} must have four columns".format(path, line_number))
                label, first, second, outcome = words
                relation = RELATION_LABELS.get(label)
                if relation not in result:
                    raise ValueError("Unknown relation label {} at {}:{}".format(label, path, line_number))
                pair = (int(first), int(second))
                if int(outcome) == 1:
                    result[relation].positive.append(pair)
                elif int(outcome) == 0:
                    result[relation].negative.append(pair)
                else:
                    raise ValueError("Binary outcome required at {}:{}".format(path, line_number))
    return result


def load_dataset(config: Dict[str, Any]) -> MultiplexDataset:
    data_config = config["data"]
    mat_path = resolve_path(data_config["mat_file"])
    mapping_path = resolve_path(data_config["mapping_file"])
    split_dir = resolve_path(data_config["split_dir"])
    relation_order = list(data_config["relation_order"])
    mat = loadmat(str(mat_path))
    features_raw = mat["full_feature"]
    features = (
        features_raw.toarray().astype(np.float32)
        if sparse.issparse(features_raw)
        else np.asarray(features_raw, dtype=np.float32)
    )

    mapping = pd.read_excel(mapping_path)
    if "STT" in mapping.columns:
        mapping = mapping.sort_values("STT").reset_index(drop=True)
    boundary_values = np.flatnonzero(mapping["ID"].astype(str).str.startswith("MIM").to_numpy())
    if len(boundary_values) == 0:
        raise ValueError("No MIM disease boundary found in mapping")
    boundary = int(boundary_values[0])
    drug_ids = mapping.iloc[:boundary]["ID"].astype(str).tolist()
    drug_names = mapping.iloc[:boundary]["Name"].astype(str).tolist()
    disease_ids = mapping.iloc[boundary:]["ID"].astype(str).tolist()
    disease_names = mapping.iloc[boundary:]["Name"].astype(str).tolist()

    similarity_config = data_config["similarity"]
    diagonal = float(similarity_config.get("diagonal", 1.0))
    drug_similarity = load_similarity_network(
        resolve_path(similarity_config["drug_file"]), drug_ids, diagonal=diagonal
    )
    disease_similarity = load_similarity_network(
        resolve_path(similarity_config["disease_file"]), disease_ids, diagonal=diagonal
    )
    node_similarity = combine_similarity_blocks(drug_similarity, disease_similarity)
    similarity_stats = {
        "drug": {
            "source_rows": drug_similarity.source_rows,
            "source_entities": drug_similarity.source_entities,
            "retained_rows": drug_similarity.retained_rows,
            "filtered_rows": drug_similarity.filtered_rows,
            "pairs": drug_similarity.pair_count,
            "covered_entities": drug_similarity.covered_entities,
        },
        "disease": {
            "source_rows": disease_similarity.source_rows,
            "source_entities": disease_similarity.source_entities,
            "retained_rows": disease_similarity.retained_rows,
            "filtered_rows": disease_similarity.filtered_rows,
            "pairs": disease_similarity.pair_count,
            "covered_entities": disease_similarity.covered_entities,
        },
        "diagonal": {"value": diagonal},
    }

    adjacencies: Dict[str, sparse.csr_matrix] = {}
    pairs: Dict[str, List[Pair]] = {}
    for relation in relation_order:
        key = RELATION_KEYS[relation]
        matrix = sparse.csr_matrix(mat[key], dtype=np.float32)
        adjacencies[relation] = (matrix + node_similarity).tocsr()
        cross = matrix[:boundary, boundary:].tocoo()
        pairs[relation] = [
            (int(first), boundary + int(second))
            for first, second in zip(cross.row.tolist(), cross.col.tolist())
        ]

    labelled = _read_labelled_edges(split_dir, relation_order)
    return MultiplexDataset(
        features=features,
        relation_order=relation_order,
        relation_adjacencies=adjacencies,
        node_similarity=node_similarity,
        similarity_stats=similarity_stats,
        relation_pairs=pairs,
        labelled_edges=labelled,
        drug_ids=drug_ids,
        drug_names=drug_names,
        disease_ids=disease_ids,
        disease_names=disease_names,
    )


def validate_dataset(dataset: MultiplexDataset, config: Dict[str, Any]) -> Dict[str, Any]:
    expected_nodes = config["data"]["expected_nodes"]
    expected_relations = config["data"]["expected_relations"]
    expected_similarity = config["data"]["similarity"]["expected_pairs"]
    expected_similarity_coverage = config["data"]["similarity"]["expected_covered_entities"]
    expected_similarity_source_rows = config["data"]["similarity"]["expected_source_rows"]
    expected_similarity_source_entities = config["data"]["similarity"]["expected_source_entities"]
    problems = []
    if dataset.n_drugs != int(expected_nodes["drugs"]):
        problems.append("drug count mismatch")
    if dataset.n_diseases != int(expected_nodes["diseases"]):
        problems.append("disease count mismatch")
    if dataset.features.shape[0] != dataset.n_nodes:
        problems.append("feature row count does not equal node count")
    if dataset.node_similarity.shape != (dataset.n_nodes, dataset.n_nodes):
        problems.append("node similarity shape mismatch")
    drug_similarity = dataset.node_similarity[: dataset.n_drugs, : dataset.n_drugs]
    disease_similarity = dataset.node_similarity[dataset.n_drugs :, dataset.n_drugs :]
    cross_similarity_nonzero = (
        dataset.node_similarity[: dataset.n_drugs, dataset.n_drugs :].nnz
        + dataset.node_similarity[dataset.n_drugs :, : dataset.n_drugs].nnz
    )
    similarities = {
        "drug": {
            "shape": list(drug_similarity.shape),
            "source_rows": int(dataset.similarity_stats["drug"]["source_rows"]),
            "source_entities": int(dataset.similarity_stats["drug"]["source_entities"]),
            "retained_rows": int(dataset.similarity_stats["drug"]["retained_rows"]),
            "filtered_rows": int(dataset.similarity_stats["drug"]["filtered_rows"]),
            "pairs": int(dataset.similarity_stats["drug"]["pairs"]),
            "covered_entities": int(dataset.similarity_stats["drug"]["covered_entities"]),
            "symmetric": bool((drug_similarity != drug_similarity.T).nnz == 0),
            "unit_diagonal": bool(np.allclose(drug_similarity.diagonal(), 1.0)),
        },
        "disease": {
            "shape": list(disease_similarity.shape),
            "source_rows": int(dataset.similarity_stats["disease"]["source_rows"]),
            "source_entities": int(dataset.similarity_stats["disease"]["source_entities"]),
            "retained_rows": int(dataset.similarity_stats["disease"]["retained_rows"]),
            "filtered_rows": int(dataset.similarity_stats["disease"]["filtered_rows"]),
            "pairs": int(dataset.similarity_stats["disease"]["pairs"]),
            "covered_entities": int(dataset.similarity_stats["disease"]["covered_entities"]),
            "symmetric": bool((disease_similarity != disease_similarity.T).nnz == 0),
            "unit_diagonal": bool(np.allclose(disease_similarity.diagonal(), 1.0)),
        },
        "cross_block_nonzero": int(cross_similarity_nonzero),
    }
    for name in ("drug", "disease"):
        if similarities[name]["source_rows"] != int(expected_similarity_source_rows[name]):
            problems.append("{} similarity source row count mismatch".format(name))
        if similarities[name]["source_entities"] != int(expected_similarity_source_entities[name]):
            problems.append("{} similarity source entity count mismatch".format(name))
        if similarities[name]["source_rows"] != (
            similarities[name]["retained_rows"] + similarities[name]["filtered_rows"]
        ):
            problems.append("{} similarity row accounting mismatch".format(name))
        if similarities[name]["pairs"] != int(expected_similarity[name]):
            problems.append("{} similarity pair count mismatch".format(name))
        if similarities[name]["covered_entities"] != int(expected_similarity_coverage[name]):
            problems.append("{} similarity coverage mismatch".format(name))
        if not similarities[name]["symmetric"]:
            problems.append("{} similarity is not symmetric".format(name))
        if not similarities[name]["unit_diagonal"]:
            problems.append("{} similarity diagonal mismatch".format(name))
    if cross_similarity_nonzero:
        problems.append("similarity matrix has cross-type entries")

    relations = {}
    for name in dataset.relation_order:
        matrix_count = len(dataset.relation_pairs[name])
        labelled_count = len(dataset.labelled_edges[name].positive)
        expected = int(expected_relations[name])
        exact_match = set(dataset.relation_pairs[name]) == set(dataset.labelled_edges[name].positive)
        relations[name] = {
            "matrix_positive_pairs": matrix_count,
            "labelled_positive_pairs": labelled_count,
            "expected": expected,
            "exact_pair_set_match": exact_match,
            "negative_pairs": len(dataset.labelled_edges[name].negative),
        }
        if matrix_count != expected or labelled_count != expected or not exact_match:
            problems.append("relation {} mismatch".format(name))
        if set(dataset.labelled_edges[name].positive) & set(dataset.labelled_edges[name].negative):
            problems.append("relation {} has positive/negative overlap".format(name))
    report = {
        "status": "valid" if not problems else "invalid",
        "nodes": {
            "drugs": dataset.n_drugs,
            "diseases": dataset.n_diseases,
            "total": dataset.n_nodes,
        },
        "features": {
            "shape": list(dataset.features.shape),
            "dtype": str(dataset.features.dtype),
            "all_zero_rows": int(np.sum(np.all(dataset.features == 0, axis=1))),
        },
        "relations": relations,
        "similarities": similarities,
        "known_unique_pairs": len(dataset.known_pairs()),
        "unknown_candidate_pairs": dataset.n_drugs * dataset.n_diseases - len(dataset.known_pairs()),
        "problems": problems,
    }
    if problems:
        raise ValueError("Dataset validation failed: {}".format("; ".join(problems)))
    return report
