"""Orchestrate raw HPO, chemistry, relation projection, split and MAT creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from config import REPOSITORY_ROOT, resolve_path
from runtime import sha256_file, write_json
from .chemistry import generate_drug_features
from .export import (
    create_source_splits,
    indexed_relation_pairs,
    save_model_input,
    save_relation_tables,
)
from .hpo import generate_disease_features, load_diseases
from .relations import project_relations


def _path_map(values: Dict[str, str]) -> Dict[str, Path]:
    return {name: resolve_path(value) for name, value in values.items()}


def load_preprocessing_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "preprocessing" not in config:
        raise ValueError("Configuration must contain a preprocessing section")
    config["_config_path"] = str(config_path)
    return config


def _portable_output_label(output_dir: Path) -> str:
    """Prefer a repository-relative label, but support external temp outputs."""
    try:
        return str(output_dir.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(output_dir)


def prepare_data(config: Dict[str, Any], output_dir_override: str = "") -> Dict[str, Any]:
    values = config["preprocessing"]
    output_dir = resolve_path(output_dir_override or values["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Preprocessing output already exists and will not be overwritten: {}".format(output_dir)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _path_map(values["inputs"])
    missing_inputs = [str(path) for path in inputs.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError("Missing preprocessing inputs: {}".format(missing_inputs))
    relation_order = list(values.get("relation_order", ["DD", "DGD", "DPD", "DCD"]))
    if relation_order != ["DD", "DGD", "DPD", "DCD"]:
        raise ValueError("relation_order must be [DD, DGD, DPD, DCD]")
    seed = int(config.get("project", {}).get("seed", 42))

    drug_entities, drug_features, drug_stats = generate_drug_features(
        inputs["drug_smiles"], aggregation=values.get("drug_atom_aggregation", "mean")
    )
    disease_features, disease_stats = generate_disease_features(
        inputs["diseases"],
        inputs["hpo_annotations"],
        output_dir / "intermediate",
        l2_normalize=bool(values.get("hpo_l2_normalize", True)),
    )
    disease_entities = load_diseases(inputs["diseases"])
    drug_ids = drug_entities["drug_id"].astype(str).tolist()
    disease_ids = disease_entities["Disease"].astype(str).tolist()

    relations, relation_stats = project_relations(
        inputs["direct_drug_disease"],
        inputs["drug_targets"],
        inputs["phenotype_genes"],
        inputs["pathway_genes"],
        inputs["protein_complex_genes"],
        set(drug_ids),
        set(disease_ids),
        distinct_pathway_genes=bool(values.get("distinct_pathway_genes", True)),
        distinct_complex_genes=bool(values.get("distinct_complex_genes", False)),
    )
    expected = values.get("expected_relation_pairs", {})
    observed = relation_stats["projected_relation_pairs"]
    for relation in relation_order:
        if relation in expected and int(expected[relation]) != int(observed[relation]):
            raise ValueError(
                "Relation {} count mismatch: {} != {}".format(
                    relation, observed[relation], expected[relation]
                )
            )
    indexed = indexed_relation_pairs(relations, drug_ids, disease_ids, relation_order)
    splits, split_stats = create_source_splits(
        indexed,
        relation_order,
        len(drug_ids),
        len(disease_ids),
        seed,
        output_dir / "splits",
    )
    disease_matrix = disease_features.to_numpy(dtype=np.float32)
    features = np.zeros(
        (len(drug_ids) + len(disease_ids), drug_features.shape[1] + disease_matrix.shape[1]),
        dtype=np.float32,
    )
    features[: len(drug_ids), : drug_features.shape[1]] = drug_features
    features[len(drug_ids) :, drug_features.shape[1] :] = disease_matrix
    model_input_path = output_dir / "multiplex_graph.mat"
    export_stats = save_model_input(
        model_input_path,
        indexed,
        splits,
        relation_order,
        features,
        len(drug_ids),
        len(disease_ids),
        seed,
    )
    save_relation_tables(relations, relation_order, output_dir / "relations")
    pd.DataFrame({"drug_id": drug_ids}).to_csv(output_dir / "drug_ids.tsv", sep="\t", index=False)
    pd.DataFrame({"disease_id": disease_ids}).to_csv(output_dir / "disease_ids.tsv", sep="\t", index=False)
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    (output_dir / "preprocessing.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    output_files = [model_input_path] + sorted((output_dir / "splits").glob("*.txt"))
    manifest = {
        "status": "completed",
        "source_algorithm": "modular reconstruction of the 22/05 preprocessing flow",
        "seed": seed,
        "output_dir": _portable_output_label(output_dir),
        "raw_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "output_sha256": {
            str(path.relative_to(output_dir)): sha256_file(path) for path in output_files
        },
        "drug_features": drug_stats,
        "disease_features": disease_stats,
        "relations": relation_stats,
        "splits": split_stats,
        "model_input": export_stats,
    }
    write_json(output_dir / "preprocessing_manifest.json", manifest)
    return manifest
