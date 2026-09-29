"""Training and relation-wise cross-validation for RAMHGCN."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as functional
from torch import nn
from tqdm import tqdm

from data import MultiplexDataset, Pair, RelationLabels
from metrics import macro_evaluate, summarize_folds
from model import RAMHGCNEncoder, pair_logits, parameter_counts
from graph_input import build_adjacency_stack
from runtime import seed_everything
from splits import FoldSplit, make_relation_folds


def _model(config: Dict[str, Any], input_dimension: int, device: torch.device) -> RAMHGCNEncoder:
    values = config["model"]
    relation_order = list(config["data"]["relation_order"])
    disabled = set(values.get("disabled_relations", []))
    unknown = disabled.difference(relation_order)
    if unknown:
        raise ValueError("Unknown disabled relations: {}".format(sorted(unknown)))
    relation_mask = [0.0 if name in disabled else 1.0 for name in relation_order]
    return RAMHGCNEncoder(
        input_dimension=input_dimension,
        embedding_dimension=int(values["embedding_dim"]),
        relation_mask=relation_mask,
    ).to(device)


def _optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    values = config["training"]
    arguments = {
        "lr": float(values["learning_rate"]),
        "weight_decay": float(values["weight_decay"]),
    }
    name = str(values["optimizer"]).lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **arguments)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **arguments)
    raise ValueError("optimizer must be AdamW or Adam")


def _edge_tensor(
    edges: Dict[str, RelationLabels], relations: Sequence[str], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    positive: List[Pair] = []
    negative: List[Pair] = []
    for relation in relations:
        positive.extend(edges[relation].positive)
        negative.extend(edges[relation].negative)
    if not positive or not negative:
        raise ValueError("Training requires positive and negative pairs")
    return (
        torch.as_tensor(positive, dtype=torch.long, device=device),
        torch.as_tensor(negative, dtype=torch.long, device=device),
    )


def _loss(embeddings: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    positive_logits = pair_logits(embeddings, positive)
    negative_logits = -pair_logits(embeddings, negative)
    return -(
        torch.mean(functional.logsigmoid(positive_logits))
        + torch.mean(functional.logsigmoid(negative_logits))
    )


def _cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _save_checkpoint(
    path: Path,
    state: Dict[str, torch.Tensor],
    fold: int,
    selected_epoch: int,
    relation_weights: Sequence[float],
    protocol: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": state,
            "fold": fold,
            "selected_epoch": selected_epoch,
            "relation_weights": list(relation_weights),
            "protocol": protocol,
        },
        path,
    )


def _relation_fold(
    dataset: MultiplexDataset,
    split: FoldSplit,
    config: Dict[str, Any],
    features: torch.Tensor,
    adjacency: torch.Tensor,
    device: torch.device,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    relations = list(config["training"]["supervision_relations"])
    model = _model(config, dataset.features.shape[1], device)
    optimizer = _optimizer(model, config)
    positive, negative = _edge_tensor(split.train, relations, device)
    epochs = int(config["training"]["epochs"])
    best_validation_auprc = -float("inf")
    best_epoch = -1
    best_embedding = None
    best_state = None
    best_weights = None

    for epoch in tqdm(range(1, epochs + 1), desc="RAMHGCN fold {}".format(split.fold + 1), leave=False):
        model.train()
        embeddings = model(features, adjacency)
        loss = _loss(embeddings, positive, negative)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            evaluated = model(features, adjacency).detach().cpu().numpy()
        selection_thresholds = {relation: 0.0 for relation in relations}
        validation, _ = macro_evaluate(
            evaluated,
            split.validation,
            relations,
            threshold_policy=config["evaluation"]["threshold_policy"],
            thresholds=selection_thresholds,
        )
        validation_auprc = validation["auprc"]
        if validation_auprc > best_validation_auprc:
            best_validation_auprc = float(validation_auprc)
            best_epoch = epoch
            best_embedding = evaluated.copy()
            best_state = _cpu_state(model)
            best_weights = model.relation_weights().cpu().numpy().astype(float).tolist()

    if best_state is None or best_embedding is None:
        raise RuntimeError("No checkpoint selected for fold {}".format(split.fold + 1))
    best_validation, validation_relations = macro_evaluate(
        best_embedding,
        split.validation,
        relations,
        threshold_policy=config["evaluation"]["threshold_policy"],
    )
    thresholds = {
        relation: validation_relations[relation]["threshold"] for relation in relations
    }
    best_test, best_test_relations = macro_evaluate(
        best_embedding,
        split.test,
        relations,
        threshold_policy=config["evaluation"]["threshold_policy"],
        thresholds=thresholds,
    )
    _save_checkpoint(
        checkpoint_path, best_state, split.fold + 1, best_epoch, best_weights, "relation_wise_10cv"
    )
    declared, active = parameter_counts(model)
    return {
        "fold": split.fold + 1,
        "selected_epoch": best_epoch,
        "validation": best_validation,
        "test": best_test,
        "validation_per_relation": validation_relations,
        "test_per_relation": best_test_relations,
        "validation_thresholds": thresholds,
        "relation_weights": {
            name: value for name, value in zip(dataset.relation_order, best_weights)
        },
        "checkpoint": str(checkpoint_path),
        "embedding": best_embedding,
        "parameter_counts": {"declared": declared, "forward_active": active},
    }


def run_cross_validation(
    dataset: MultiplexDataset, config: Dict[str, Any], device: torch.device, output_dir: Path
) -> Dict[str, Any]:
    seed = int(config["project"]["seed"])
    deterministic = bool(config["runtime"].get("deterministic", True))
    seed_everything(seed, deterministic)
    folds = make_relation_folds(
        dataset, int(config["training"]["outer_folds"]), seed
    )
    features = torch.as_tensor(dataset.features, dtype=torch.float32, device=device)
    adjacency = build_adjacency_stack(dataset, config, device)
    results = []
    for split in folds:
        result = _relation_fold(
            dataset,
            split,
            config,
            features,
            adjacency,
            device,
            output_dir / "checkpoints" / "fold_{:02d}.pt".format(split.fold + 1),
        )
        results.append(result)
    best = max(results, key=lambda value: value["validation"]["auprc"])
    summary = summarize_folds([value["test"] for value in results])
    return {
        "protocol": "relation_wise_10cv",
        "summary": summary,
        "folds": results,
        "splits": folds,
        "ranking_embedding": best["embedding"],
        "ranking_model_policy": "best fold selected by validation AUPRC",
        "best_fold": best["fold"],
    }
