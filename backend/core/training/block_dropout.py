"""Low-rate stochastic depth for DiT training (per-batch block dropout).

Training-only regularization: on each training step a small set of transformer
blocks is independently dropped (replaced by identity) with probability
``skip_rate``. Dropped blocks contribute nothing; executed blocks have their
residual DELTA rescaled by ``1 / (1 - skip_rate)`` so the expected residual
contribution is unbiased vs. the full network run at eval (the standard
"inverted" stochastic-depth convention, cf. torchvision ``StochasticDepth`` and
Huang et al., "Deep Networks with Stochastic Depth", arXiv:1603.09382).

Crucially, NOT every block is eligible. Empirical block-removal studies of
pretrained DiTs (e.g. the BlockSkip / layer-pruning line of work) show the
MIDDLE blocks are semantically critical — dropping them destroys the subject —
while EARLY and LATE blocks tolerate removal far better. So a contiguous middle
span ``[protect_start, protect_end)`` is PROTECTED (never dropped); only the
front ``[0, protect_start)`` and back ``[protect_end, num_blocks)`` blocks are
eligible. This mirrors the FLUX finding that the last blocks + a protected core
survive pruning; for Anima's 28 blocks the defaults protect the middle 16
(6..21) and leave the first 6 + last 6 eligible.

Inference / sampling ALWAYS runs the full network (no drop, no scaling): the
trainer only attaches the config during a training forward.
"""

from typing import List, Optional, Tuple

import torch


def eligible_blocks(
    num_blocks: int, protect_start: int, protect_end: int
) -> List[int]:
    """Block indices eligible for dropout: front + back, excluding the protected
    middle span ``[protect_start, protect_end)``.

    ``protect_start``/``protect_end`` are clamped to ``[0, num_blocks]`` and to
    ``protect_start <= protect_end``; an empty protected span means every block
    is eligible, a full-width span means none are.
    """
    ps = max(0, min(num_blocks, int(protect_start)))
    pe = max(ps, min(num_blocks, int(protect_end)))
    return [i for i in range(num_blocks) if i < ps or i >= pe]


def compute_skip_mask(
    num_blocks: int,
    skip_rate: float,
    protect_start: int,
    protect_end: int,
    device: torch.device,
    exclude: Optional[set] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[bool], List[int]]:
    """Sample which blocks are dropped for one training step.

    Each ELIGIBLE block (front/back, outside the protected middle span, and not
    in ``exclude``) is independently dropped with probability ``skip_rate``.
    Protected / excluded blocks are never dropped.

    Returns ``(skip_mask, eligible)`` where ``skip_mask[i]`` is True if block
    ``i`` is dropped this step, and ``eligible`` is the list of eligible indices
    (for logging). ``skip_rate <= 0`` yields an all-False mask.
    """
    skip_mask = [False] * num_blocks
    elig = eligible_blocks(num_blocks, protect_start, protect_end)
    if exclude:
        elig = [i for i in elig if i not in exclude]
    if skip_rate <= 0.0 or not elig:
        return skip_mask, elig
    # One Bernoulli draw per eligible block (independent per-block dropout).
    draws = torch.rand(len(elig), device=device, generator=generator)
    for pos, idx in enumerate(elig):
        if float(draws[pos]) < skip_rate:
            skip_mask[idx] = True
    return skip_mask, elig
