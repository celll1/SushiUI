"""Materialized CPU oracle for Sol-Attn routing and correction semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SolReferenceResult:
    output: torch.Tensor
    routes: torch.Tensor


def _blocks(tensor: torch.Tensor, block_size: int) -> list[torch.Tensor]:
    return list(tensor.split(block_size, dim=1))


def sol_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: Optional[float] = None,
    tau: float = 1.0,
    threshold_type: str = "diag",
    block_size: int = 64,
    sink_start: int = 0,
    sink_tokens: int = 0,
    exact_neighbor_blocks: int = 1,
) -> SolReferenceResult:
    """Evaluate the paper equations directly; intended only for small tests."""
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Sol reference requires equal BSHD query/key/value tensors")
    if threshold_type not in {"diag", "exact"}:
        raise ValueError("threshold_type must be 'diag' or 'exact'")
    if block_size <= 0 or exact_neighbor_blocks < 0:
        raise ValueError("block sizes must be positive and neighbor radius non-negative")
    tokens = query.shape[1]
    if not 0 <= sink_start <= tokens or not 0 <= sink_tokens <= tokens - sink_start:
        raise ValueError("sink range must lie inside the sequence")

    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    q = query.float()
    k = key.float()
    v = value.float()
    q_blocks = _blocks(q, block_size)
    k_blocks = _blocks(k, block_size)
    k_centroids = torch.stack([part.mean(dim=1) for part in k_blocks], dim=1)
    q_centroids = torch.stack([part.mean(dim=1) for part in q_blocks], dim=1)
    proxy = torch.einsum("bqhd,bkhd->bhqk", q_centroids, k_centroids) * scale

    mean = proxy.mean(dim=-1, keepdim=True)
    if threshold_type == "exact":
        variance = proxy.var(dim=-1, correction=0, keepdim=True)
    else:
        k_variance = k_centroids.var(dim=1, correction=0)
        variance = torch.einsum(
            "bqhd,bhd->bhq", q_centroids.square(), k_variance
        ).unsqueeze(-1) * (scale * scale)
    threshold = mean + float(tau) * torch.sqrt(variance + 1.0e-6)
    routes = proxy > threshold

    block_count = len(q_blocks)
    indices = torch.arange(block_count, device=query.device)
    routes |= (
        (indices[:, None] - indices[None, :]).abs() <= exact_neighbor_blocks
    )[None, None]
    if sink_tokens:
        first = sink_start // block_size
        last = math.ceil((sink_start + sink_tokens) / block_size)
        routes[..., first:last] = True

    outputs = []
    for query_block, q_part in enumerate(q_blocks):
        logits = []
        for key_block, k_part in enumerate(k_blocks):
            exact = torch.einsum("bqhd,bkhd->bhqk", q_part, k_part)
            approximate = (
                q_part * k_centroids[:, key_block].unsqueeze(1)
            ).sum(dim=-1).permute(0, 2, 1).unsqueeze(-1).expand_as(exact)
            route = routes[:, :, query_block, key_block].unsqueeze(-1).unsqueeze(-1)
            logits.append(torch.where(route, exact, approximate) * scale)
        weights = torch.softmax(torch.cat(logits, dim=-1), dim=-1)
        values = v.transpose(1, 2)
        outputs.append((weights @ values).transpose(1, 2))
    output = torch.cat(outputs, dim=1).to(query.dtype)
    return SolReferenceResult(output=output, routes=routes)


__all__ = ["SolReferenceResult", "sol_attention_reference"]
