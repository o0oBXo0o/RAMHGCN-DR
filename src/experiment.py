"""End-to-end experiment orchestration and machine-readable artifacts."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from config import REPOSITORY_ROOT, resolve_path
from data import MultiplexDataset, load_dataset, validate_dataset
from metrics import METRIC_NAMES
from ranking import export_rankings
from runtime import choose_device, runtime_manifest, write_json
from training import run_cross_validation


def _serializable_fold(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if key != "embedding"}


def _fold_table(folds: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for value in folds:
        row = {
            "fold": value["fold"],
            "selected_epoch": value["selected_epoch"],
        }
        row.update({"validation_{}".format(k): v for k, v in value["validation"].items()})
        row.update({"test_{}".format(k): v for k, v in value["test"].items()})
        row.update({"weight_{}".format(k): v for k, v in value["relation_weights"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = resolve_path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config_snapshot = copy.deepcopy(config)
    config_snapshot.pop("_config_path", None)
    config_snapshot.pop("_repository_root", None)
    (output_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    dataset = load_dataset(config)
    validation = validate_dataset(dataset, config)
    write_json(output_dir / "data_validation.json", validation)
    device = choose_device(config["runtime"]["device"])
    protocol = config["training"]["protocol"]
    result = run_cross_validation(dataset, config, device, output_dir)

    embedding = result["ranking_embedding"]
    embedding_name = "ranking_embedding.npy"
    np.save(output_dir / embedding_name, embedding)
    serializable_folds = [_serializable_fold(value) for value in result["folds"]]
    _fold_table(serializable_folds).to_csv(output_dir / "fold_metrics.csv", index=False)
    write_json(output_dir / "fold_details.json", serializable_folds)

    ranking_manifest = None
    if bool(config["ranking"]["enabled"]):
        ranking_manifest = export_rankings(
            embedding, dataset, output_dir / "rankings", config["ranking"]
        )

    elapsed = time.time() - started
    metrics = {
        "status": "completed",
        "protocol": protocol,
        "elapsed_seconds": elapsed,
        "best_fold": result["best_fold"],
        "summary": result["summary"],
        "folds": [value["test"] for value in serializable_folds],
        "embedding_file": str(output_dir / embedding_name),
        "embedding_shape": list(embedding.shape),
        "ranking_model_policy": result["ranking_model_policy"],
    }
    write_json(output_dir / "metrics.json", metrics)

    source_files = list((REPOSITORY_ROOT / "src").rglob("*.py")) + list(
        (REPOSITORY_ROOT / "scripts").glob("*.py")
    )
    data_files = [
        resolve_path(config["data"]["mat_file"]),
        resolve_path(config["data"]["mapping_file"]),
        resolve_path(config["data"]["similarity"]["drug_file"]),
        resolve_path(config["data"]["similarity"]["disease_file"]),
    ] + sorted(resolve_path(config["data"]["split_dir"]).glob("*.txt"))
    provenance = runtime_manifest(
        REPOSITORY_ROOT,
        Path(config["_config_path"]),
        source_files,
        data_files,
    )
    provenance.update(
        {
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "elapsed_seconds": elapsed,
            "device_used": str(device),
            "source_version": "RAMHGCN",
        }
    )
    write_json(output_dir / "provenance.json", provenance)
    manifest = {
        "status": "completed",
        "protocol": protocol,
        "output_dir": str(output_dir),
        "metrics": str(output_dir / "metrics.json"),
        "fold_metrics": str(output_dir / "fold_metrics.csv"),
        "provenance": str(output_dir / "provenance.json"),
        "ranking": ranking_manifest,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest
