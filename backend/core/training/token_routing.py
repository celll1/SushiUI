"""TREAD token routing utilities (arXiv 2501.04765).

Architecture-agnostic gather/scatter helpers for routing a random subset of
tokens through a *span* of transformer blocks during TRAINING ONLY. The dropped
tokens bypass the span and are restored to their pre-span values at re-entry
(identity transport — the single-route formulation of the paper, where tokens
"stored at layer i are reintroduced at layer j"). Kept tokens are processed by
the span and scattered back into their original positions.

Routing is training-only; inference / sampling always runs the full network on
all tokens (do NOT attach a route config during sampling).

Reference:
  Krause et al., "TREAD: Token Routing for Efficient Architecture-agnostic
  Diffusion Training", arXiv:2501.04765.

The helpers operate on a flattened token stream [B, N, D] so any DiT can reuse
them: the caller is responsible for flattening its residual stream to [B, N, D]
(and any per-token side tensors such as RoPE rows / positional embeddings) and
reshaping back afterwards.
"""

from typing import Optional

import torch


def select_kept_indices(
    num_tokens: int,
    drop_ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly select the token indices that are KEPT for the routed span.

    Uniform random selection per call (per step), matching the paper's
    "randomly selected tokens ... without any dynamic adaptations based on
    iterations or timesteps". Returns a sorted 1-D LongTensor of length
    ``round(num_tokens * (1 - drop_ratio))`` (clamped to [1, num_tokens]).

    NOTE: a single index set is shared across the batch. For batch_size == 1
    this is identical to per-sample selection; for batch_size > 1 it is a minor
    deviation that keeps RoPE / positional gathers batch-shared (a single
    ``index_select`` along the sequence dim).
    """
    keep = num_tokens - int(round(num_tokens * float(drop_ratio)))
    keep = max(1, min(num_tokens, keep))
    perm = torch.randperm(num_tokens, device=device, generator=generator)
    return perm[:keep].sort().values


def gather_tokens(x_bnd: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather the kept token rows. x_bnd: [B, N, D], idx: [keep] -> [B, keep, D]."""
    return x_bnd.index_select(1, idx)


def scatter_tokens(
    full_bnd: torch.Tensor, kept_bnd: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    """Scatter processed kept tokens back into the full stream (identity carry).

    Bypassed tokens retain their pre-span values from ``full_bnd`` (the paper's
    residual/identity transport); kept tokens take the processed ``kept_bnd``
    values. Returns a NEW tensor (autograd-friendly: index_copy propagates grads
    to ``kept_bnd`` at kept positions and to ``full_bnd`` elsewhere).

    full_bnd: [B, N, D], kept_bnd: [B, keep, D], idx: [keep].
    """
    out = full_bnd.clone()
    out.index_copy_(1, idx, kept_bnd.to(out.dtype))
    return out
