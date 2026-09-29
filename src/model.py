"""Relation-adaptive multiplex graph encoder and dot-product decoder."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = torch.mm(inputs.float(), self.weight)
        output = torch.spmm(adjacency, support)
        return output if self.bias is None else output + self.bias


class RAMHGCNEncoder(nn.Module):
    """Relation-adaptive multiplex encoder with two graph-convolution layers."""

    def __init__(
        self,
        input_dimension: int,
        embedding_dimension: int,
        relation_mask: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.gc1 = GraphConvolution(input_dimension, embedding_dimension)
        self.gc2 = GraphConvolution(embedding_dimension, embedding_dimension)
        self.relation_logits = Parameter(torch.zeros(4, 1), requires_grad=True)
        mask = [1.0, 1.0, 1.0, 1.0] if relation_mask is None else list(relation_mask)
        if len(mask) != 4 or not any(float(value) != 0.0 for value in mask):
            raise ValueError("relation_mask must keep at least one of four relations")
        self.register_buffer("relation_mask", torch.as_tensor(mask, dtype=torch.float32).reshape(4, 1))

    def effective_relation_weights(self) -> torch.Tensor:
        logits = self.relation_logits.masked_fill(self.relation_mask == 0, -torch.inf)
        return torch.softmax(logits, dim=0)

    def aggregate_adjacency(self, adjacency_stack: torch.Tensor) -> torch.Tensor:
        return torch.matmul(adjacency_stack, self.effective_relation_weights()).squeeze(2)

    def forward(self, features: torch.Tensor, adjacency_stack: torch.Tensor) -> torch.Tensor:
        adjacency = self.aggregate_adjacency(adjacency_stack)
        first = self.gc1(features, adjacency)
        second = self.gc2(first, adjacency)
        return (first + second) / 2.0

    def relation_weights(self) -> torch.Tensor:
        return self.effective_relation_weights().detach().reshape(-1)


def pair_logits(embeddings: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    return torch.sum(embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], dim=1)


def parameter_counts(model: nn.Module) -> Tuple[int, int]:
    declared = sum(parameter.numel() for parameter in model.parameters())
    active_names = {"relation_logits", "gc1.weight", "gc1.bias", "gc2.weight", "gc2.bias"}
    active = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in active_names
    )
    return declared, active
