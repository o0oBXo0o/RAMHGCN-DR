"""Binary link-prediction metrics and threshold policies."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc as curve_area,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from data import Pair, RelationLabels


METRIC_NAMES = ["auroc", "auprc", "f1", "precision", "recall", "accuracy", "mcc"]


def pair_scores(embeddings: np.ndarray, pairs: Sequence[Pair]) -> np.ndarray:
    values = np.asarray(pairs, dtype=np.int64)
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    return np.sum(embeddings[values[:, 0]] * embeddings[values[:, 1]], axis=1).astype(np.float64)


def labels_and_scores(
    embeddings: np.ndarray, edges: RelationLabels
) -> Tuple[np.ndarray, np.ndarray]:
    positive = pair_scores(embeddings, edges.positive)
    negative = pair_scores(embeddings, edges.negative)
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.int32), np.zeros(len(negative), dtype=np.int32)]
    )
    return labels, np.concatenate([positive, negative])


def metric_values(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    predictions = (scores >= threshold).astype(np.int32)
    fpr, tpr, _ = roc_curve(labels, scores)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    return {
        "auroc": float(curve_area(fpr, tpr)),
        "auprc": float(curve_area(recall_curve, precision_curve)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "threshold": float(threshold),
    }


def choose_mcc_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)[::-1]
    best_threshold = float(candidates[0]) if len(candidates) else 0.5
    best_mcc = -float("inf")
    for threshold in candidates:
        value = matthews_corrcoef(labels, scores >= threshold)
        if np.isfinite(value) and value > best_mcc + 1e-12:
            best_mcc = float(value)
            best_threshold = float(threshold)
    return best_threshold


def evaluate_edges(
    embeddings: np.ndarray, edges: RelationLabels, threshold_policy: str, threshold: Optional[float] = None
) -> Dict[str, float]:
    labels, scores = labels_and_scores(embeddings, edges)
    if len(np.unique(labels)) != 2:
        raise ValueError("Both positive and negative examples are required")
    if threshold is None:
        if threshold_policy != "validation_mcc":
            raise ValueError("Unknown threshold policy: {}".format(threshold_policy))
        threshold = choose_mcc_threshold(labels, scores)
    return metric_values(labels, scores, threshold)


def macro_evaluate(
    embeddings: np.ndarray,
    edges_by_relation: Dict[str, RelationLabels],
    relations: Sequence[str],
    threshold_policy: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    per_relation = {
        relation: evaluate_edges(
            embeddings,
            edges_by_relation[relation],
            threshold_policy=threshold_policy,
            threshold=None if thresholds is None else thresholds[relation],
        )
        for relation in relations
    }
    macro = {
        name: float(np.mean([per_relation[relation][name] for relation in relations]))
        for name in METRIC_NAMES
    }
    return macro, per_relation


def summarize_folds(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    result = {}
    for name in METRIC_NAMES:
        values = np.asarray([fold[name] for fold in fold_metrics], dtype=np.float64)
        result[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return result
