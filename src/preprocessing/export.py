"""Create model matrices and labeled splits using the source 22/05 algorithm."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat

from .relations import StringPair


IndexPair = Tuple[int, int]


def indexed_relation_pairs(
    relations: Dict[str, Set[StringPair]],
    drug_ids: Sequence[str],
    disease_ids: Sequence[str],
    relation_order: Sequence[str],
) -> Dict[str, List[IndexPair]]:
    drug_index = {value: index for index, value in enumerate(drug_ids)}
    disease_index = {value: len(drug_ids) + index for index, value in enumerate(disease_ids)}
    result: Dict[str, List[IndexPair]] = {}
    for relation in relation_order:
        result[relation] = sorted(
            (
                (drug_index[drug], disease_index[disease])
                for drug, disease in relations[relation]
            ),
            key=lambda value: (value[0], value[1]),
        )
    return result


def symmetric_adjacency(pairs: Sequence[IndexPair], nodes: int) -> sparse.csr_matrix:
    array = np.asarray(pairs, dtype=np.int64)
    if not len(array):
        return sparse.csr_matrix((nodes, nodes), dtype=np.float32)
    rows = np.concatenate([array[:, 0], array[:, 1]])
    cols = np.concatenate([array[:, 1], array[:, 0]])
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.coo_matrix((values, (rows, cols)), shape=(nodes, nodes)).tocsr()


def _positive_splits(
    pairs: Sequence[IndexPair], seed: int
) -> Tuple[List[IndexPair], List[IndexPair], List[IndexPair]]:
    array = np.asarray(pairs, dtype=np.int64)
    array = array[np.random.default_rng(seed).permutation(len(array))]
    train_size = int(0.80 * len(array))
    validation_size = int(0.10 * len(array))
    train = [tuple(map(int, row)) for row in array[:train_size]]
    validation = [
        tuple(map(int, row))
        for row in array[train_size : train_size + validation_size]
    ]
    test = [tuple(map(int, row)) for row in array[train_size + validation_size :]]
    return train, validation, test


def _lines(relation_id: int, edges: Sequence[IndexPair], label: int) -> List[str]:
    return [
        "{}\t{}\t{}\t{}\n".format(relation_id, first, second, label)
        for first, second in edges
    ]


def create_source_splits(
    indexed_pairs: Dict[str, List[IndexPair]],
    relation_order: Sequence[str],
    drug_count: int,
    disease_count: int,
    seed: int,
    output_dir: Path,
) -> Tuple[Dict[str, Dict[str, List[IndexPair]]], Dict[str, Any]]:
    positive_union: Set[IndexPair] = set()
    for relation in relation_order:
        positive_union.update(indexed_pairs[relation])
    negative_pool = [
        (drug, drug_count + disease)
        for drug in range(drug_count)
        for disease in range(disease_count)
        if (drug, drug_count + disease) not in positive_union
    ]
    np.random.default_rng(seed).shuffle(negative_pool)
    cursor = 0
    splits: Dict[str, Dict[str, List[IndexPair]]] = {}
    output = {"train": [], "valid": [], "test": [], "all_edges": []}
    relation_details: Dict[str, Any] = {}
    for relation_id, relation in enumerate(relation_order, start=1):
        train, validation, test = _positive_splits(
            indexed_pairs[relation], seed + relation_id
        )
        # Source extraction from sparse matrices sorts positive pairs in each split.
        train, validation, test = sorted(train), sorted(validation), sorted(test)
        required = len(train) + len(validation) + len(test)
        selected_negative = negative_pool[cursor : cursor + required]
        if len(selected_negative) != required:
            raise ValueError("Insufficient global negative candidates")
        cursor += required
        negative_train = selected_negative[: len(train)]
        negative_validation = selected_negative[len(train) : len(train) + len(validation)]
        negative_test = selected_negative[len(train) + len(validation) :]
        splits[relation] = {"train": train, "valid": validation, "test": test}
        train_lines = _lines(relation_id, train, 1) + _lines(relation_id, negative_train, 0)
        valid_lines = _lines(relation_id, validation, 1) + _lines(relation_id, negative_validation, 0)
        test_lines = _lines(relation_id, test, 1) + _lines(relation_id, negative_test, 0)
        output["train"].extend(train_lines)
        output["valid"].extend(valid_lines)
        output["test"].extend(test_lines)
        output["all_edges"].extend(train_lines + valid_lines + test_lines)
        relation_details[relation] = {
            "positive": {"train": len(train), "valid": len(validation), "test": len(test)},
            "negative": {
                "train": len(negative_train),
                "valid": len(negative_validation),
                "test": len(negative_test),
            },
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in output.items():
        path = output_dir / "{}.txt".format(name)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(values))
    return splits, {
        "known_unique_pairs": int(len(positive_union)),
        "unknown_candidate_pairs": int(len(negative_pool)),
        "negative_samples_allocated": int(cursor),
        "relations": relation_details,
    }


def _matlab_relation_container(
    splits: Dict[str, Dict[str, List[IndexPair]]],
    relation_order: Sequence[str],
    split_name: str,
    nodes: int,
) -> np.ndarray:
    result = np.empty((len(relation_order), 1), dtype=object)
    for index, relation in enumerate(relation_order):
        result[index, 0] = symmetric_adjacency(splits[relation][split_name], nodes)
    return result


def save_model_input(
    output_path: Path,
    indexed_pairs: Dict[str, List[IndexPair]],
    splits: Dict[str, Dict[str, List[IndexPair]]],
    relation_order: Sequence[str],
    features: np.ndarray,
    drug_count: int,
    disease_count: int,
    seed: int,
) -> Dict[str, Any]:
    nodes = drug_count + disease_count
    adjacencies = {
        relation: symmetric_adjacency(indexed_pairs[relation], nodes)
        for relation in relation_order
    }
    drug_indices = np.arange(drug_count, dtype=np.int32)
    np.random.default_rng(seed).shuffle(drug_indices)
    train_size = int(0.8 * drug_count)
    valid_size = int(0.1 * drug_count)
    train_idx = drug_indices[:train_size].reshape(-1, 1)
    valid_idx = drug_indices[train_size : train_size + valid_size].reshape(-1, 1)
    test_idx = drug_indices[train_size + valid_size :].reshape(-1, 1)
    keys = {
        "DD": "IUI_Drug_Disease",
        "DGD": "IUI_Drug_Gene_Disease",
        "DPD": "IUI_Drug_Gene_Pathway_Gene_Disease",
        "DCD": "IUI_Drug_Gene_Complex_Gene_Disease",
    }
    payload: Dict[str, Any] = {
        keys[name]: adjacencies[name].astype(np.float32) for name in relation_order
    }
    feature_sparse = sparse.csr_matrix(features.astype(np.float32))
    payload.update(
        {
            "feature": feature_sparse,
            "full_feature": feature_sparse,
            "label": np.ones((drug_count, 1), dtype=np.int32),
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "Drug_num": drug_count,
            "Disease_num": disease_count,
            "train": _matlab_relation_container(splits, relation_order, "train", nodes),
            "valid": _matlab_relation_container(splits, relation_order, "valid", nodes),
            "test": _matlab_relation_container(splits, relation_order, "test", nodes),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(output_path), payload)
    return {
        "nodes": int(nodes),
        "feature_shape": list(features.shape),
        "feature_nonzero": int(np.count_nonzero(features)),
        "relation_pairs": {
            relation: int(len(indexed_pairs[relation])) for relation in relation_order
        },
        "node_split": {
            "train": int(len(train_idx)),
            "valid": int(len(valid_idx)),
            "test": int(len(test_idx)),
        },
    }


def save_relation_tables(
    relations: Dict[str, Set[StringPair]],
    relation_order: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for relation in relation_order:
        frame = pd.DataFrame(sorted(relations[relation]), columns=["drug_id", "disease_id"])
        frame.to_csv(output_dir / "{}.tsv".format(relation), sep="\t", index=False)
