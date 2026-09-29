"""Load weighted drug-drug and disease-disease similarity networks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SimilarityNetwork:
    matrix: sparse.csr_matrix
    source_rows: int
    source_entities: int
    retained_rows: int
    filtered_rows: int
    pair_count: int
    covered_entities: int


def load_similarity_network(
    path: Path,
    entity_ids: Sequence[str],
    diagonal: float = 1.0,
) -> SimilarityNetwork:
    """Convert an ID-score-ID edge list to a symmetric matrix in model ID order."""
    index = {str(identifier): position for position, identifier in enumerate(entity_ids)}
    if len(index) != len(entity_ids):
        raise ValueError("Similarity entity identifiers must be unique")
    pairs: Dict[Tuple[int, int], float] = {}
    covered = set()
    source_entities = set()
    source_rows = 0
    retained_rows = 0
    filtered_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if not fields or fields == [""]:
                continue
            if len(fields) != 3:
                raise ValueError(
                    "{}:{} must contain ID1, score and ID2".format(path, line_number)
                )
            first_id, score_text, second_id = fields
            source_rows += 1
            source_entities.update((first_id, second_id))
            try:
                score = float(score_text)
            except ValueError as error:
                raise ValueError(
                    "{}:{} has a non-numeric score".format(path, line_number)
                ) from error
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError(
                    "{}:{} score must be finite and within [0, 1]".format(
                        path, line_number
                    )
                )
            if first_id not in index or second_id not in index:
                filtered_rows += 1
                continue
            retained_rows += 1
            first, second = index[first_id], index[second_id]
            key = (first, second) if first <= second else (second, first)
            if key in pairs and not math.isclose(
                pairs[key], score, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "{}:{} conflicts with a duplicate similarity pair".format(
                        path, line_number
                    )
                )
            pairs[key] = score
            covered.update((first, second))

    rows = []
    columns = []
    values = []
    for (first, second), score in sorted(pairs.items()):
        rows.append(first)
        columns.append(second)
        values.append(score)
        if first != second:
            rows.append(second)
            columns.append(first)
            values.append(score)
    matrix = sparse.coo_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(len(entity_ids), len(entity_ids)),
        dtype=np.float32,
    ).tocsr()
    matrix.setdiag(float(diagonal))
    matrix.eliminate_zeros()
    return SimilarityNetwork(
        matrix=matrix,
        source_rows=source_rows,
        source_entities=len(source_entities),
        retained_rows=retained_rows,
        filtered_rows=filtered_rows,
        pair_count=len(pairs),
        covered_entities=len(covered),
    )


def combine_similarity_blocks(
    drug_network: SimilarityNetwork,
    disease_network: SimilarityNetwork,
) -> sparse.csr_matrix:
    """Place drug and disease similarity matrices on the model node diagonal."""
    return sparse.block_diag(
        (drug_network.matrix, disease_network.matrix),
        format="csr",
        dtype=np.float32,
    )
