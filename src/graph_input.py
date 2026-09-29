"""Fold-specific adjacency construction, normalization and self-loop policy."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import torch

from data import MultiplexDataset, Pair


def _symmetric_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=1)
    inverse = torch.zeros_like(degree)
    positive = degree > 0
    inverse[positive] = torch.pow(degree[positive], -0.5)
    return inverse[:, None] * adjacency * inverse[None, :]


def build_adjacency_stack(
    dataset: MultiplexDataset,
    config: Dict[str, Any],
    device: torch.device,
    excluded_pairs: Optional[Set[Pair]] = None,
) -> torch.Tensor:
    stack = dataset.adjacency_stack(device, excluded_pairs=excluded_pairs)
    normalization = str(config["model"].get("adjacency_normalization", "none"))
    add_self_loops = bool(config["model"].get("self_loops", False))
    if normalization == "none" and not add_self_loops:
        return stack
    result = []
    identity = torch.eye(dataset.n_nodes, dtype=stack.dtype, device=device)
    for index in range(stack.shape[2]):
        adjacency = stack[:, :, index]
        if add_self_loops:
            adjacency = adjacency + identity
        if normalization == "symmetric":
            adjacency = _symmetric_normalize(adjacency)
        elif normalization != "none":
            raise ValueError("adjacency_normalization must be none or symmetric")
        result.append(adjacency)
    return torch.stack(result, dim=2)
