"""Configuration loading, path resolution and schema validation for RAMHGCN."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PROTOCOL = "relation_wise_10cv"


def require(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError("Missing required configuration key: {}".format(key))
    return mapping[key]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def validate_config(config: Dict[str, Any]) -> None:
    for section in ("project", "data", "model", "training", "evaluation", "ranking", "runtime"):
        require(config, section)
    protocol = str(require(config["training"], "protocol"))
    if protocol != SUPPORTED_PROTOCOL:
        raise ValueError("Unsupported protocol: {}".format(protocol))
    order = list(require(config["data"], "relation_order"))
    if order != ["DD", "DGD", "DPD", "DCD"]:
        raise ValueError("relation_order must be [DD, DGD, DPD, DCD]")
    folds = int(require(config["training"], "outer_folds"))
    if folds < 3:
        raise ValueError("outer_folds must be at least 3 (test, validation and training folds)")
    if int(require(config["training"], "epochs")) < 1:
        raise ValueError("epochs must be positive")
    similarity = require(config["data"], "similarity")
    for key in (
        "drug_file",
        "disease_file",
        "expected_pairs",
        "expected_source_rows",
        "expected_source_entities",
    ):
        require(similarity, key)
    require(similarity, "expected_covered_entities")
    for group in (
        "expected_pairs",
        "expected_covered_entities",
        "expected_source_rows",
        "expected_source_entities",
    ):
        values = require(similarity, group)
        require(values, "drug")
        require(values, "disease")
    diagonal = float(similarity.get("diagonal", 1.0))
    if diagonal != 1.0:
        raise ValueError("similarity diagonal must be 1.0")
    if list(config["training"].get("supervision_relations", [])) != order:
        raise ValueError("supervision_relations must include [DD, DGD, DPD, DCD]")
    if config["model"].get("relation_weighting") != "softmax":
        raise ValueError("RAMHGCN requires non-negative normalized softmax relation weights")
    if config["evaluation"].get("threshold_policy") != "validation_mcc":
        raise ValueError("threshold_policy must be validation_mcc")
    if config["training"].get("checkpoint_metric") != "validation_auprc":
        raise ValueError("checkpoint_metric must be validation_auprc")
    if config["evaluation"].get("best_fold_metric") != "validation_auprc":
        raise ValueError("best_fold_metric must be validation_auprc")
    if config["training"].get("adjacency_policy") != "full_multiplex_graph":
        raise ValueError("adjacency_policy must be full_multiplex_graph")


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_repository_root"] = str(REPOSITORY_ROOT)
    validate_config(config)
    return config
