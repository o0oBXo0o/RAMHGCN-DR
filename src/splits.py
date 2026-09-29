"""Deterministic relation-wise cross-validation fold construction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from data import MultiplexDataset, Pair, RelationLabels


@dataclass
class FoldSplit:
    fold: int
    train: Dict[str, RelationLabels]
    validation: Dict[str, RelationLabels]
    test: Dict[str, RelationLabels]
    excluded_positive_pairs: Set[Pair]


def _split_equal(
    positive: Sequence[Pair], negative: Sequence[Pair], folds: int, rng: random.Random
) -> List[Tuple[List[Pair], List[Pair]]]:
    if len(positive) != len(negative):
        raise ValueError("Positive and negative lists must be balanced")
    positives = list(positive)
    negatives = list(negative)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    base, remainder = divmod(len(positives), folds)
    result = []
    start = 0
    for fold in range(folds):
        size = base + (1 if fold < remainder else 0)
        result.append((positives[start : start + size], negatives[start : start + size]))
        start += size
    return result


def _empty(relation_order: Sequence[str]) -> Dict[str, RelationLabels]:
    return {name: RelationLabels([], []) for name in relation_order}


def assert_disjoint(split: FoldSplit, relations: Sequence[str]) -> None:
    for relation in relations:
        train = set(split.train[relation].positive) | set(split.train[relation].negative)
        validation = set(split.validation[relation].positive) | set(split.validation[relation].negative)
        test = set(split.test[relation].positive) | set(split.test[relation].negative)
        if train & validation or train & test or validation & test:
            raise AssertionError("Example overlap in fold {} relation {}".format(split.fold, relation))


def make_relation_folds(
    dataset: MultiplexDataset, folds: int, seed: int
) -> List[FoldSplit]:
    """Split each of the four relation channels into balanced deterministic folds."""
    rng = random.Random(seed)
    chunks = {}
    for relation in dataset.relation_order:
        labelled = dataset.labelled_edges[relation]
        n_positive = len(labelled.positive)
        if len(labelled.negative) >= n_positive:
            sampled_negative = rng.sample(labelled.negative, n_positive)
        else:
            sampled_negative = [
                labelled.negative[index % len(labelled.negative)]
                for index in range(n_positive)
            ]
        chunks[relation] = _split_equal(
            labelled.positive, sampled_negative, folds, rng
        )

    result = []
    for fold in range(folds):
        train = _empty(dataset.relation_order)
        validation = _empty(dataset.relation_order)
        test = _empty(dataset.relation_order)
        held_out = set()
        for relation in dataset.relation_order:
            test_pos, test_neg = chunks[relation][fold]
            valid_pos, valid_neg = chunks[relation][(fold + 1) % folds]
            test[relation] = RelationLabels(list(test_pos), list(test_neg))
            validation[relation] = RelationLabels(list(valid_pos), list(valid_neg))
            for index, (pos, neg) in enumerate(chunks[relation]):
                if index in (fold, (fold + 1) % folds):
                    continue
                train[relation].positive.extend(pos)
                train[relation].negative.extend(neg)
            held_out.update(test_pos)
            held_out.update(valid_pos)
        split = FoldSplit(fold, train, validation, test, held_out)
        assert_disjoint(split, dataset.relation_order)
        result.append(split)
    return result
