"""fp32 reference oracle for the adapter algebras (test-only).

An oracle that shares code with the thing it validates proves nothing, so
nothing here calls into ``layers.py``: each delta weight is written from its
algebraic definition -- an explicit sum of outer products for a low-rank
product, an explicit block assembly for the Kronecker product -- in fp32, at
whatever cost. It is imported by ``backend/tests/*`` only; no runtime path may
depend on it.

The functions take the branch tensors under the names ``branch_tensors()``
produces, so a layer's live state can be handed straight over. Extra keys
(``alpha``, which LyCORIS carries as a tensor) are ignored.

Strength is the design doc's contract, ``W_eff(s) = W_base + s * (W_adapter -
W_base)``, not upstream's; see ``dora_effective_weight`` and "Runtime hazards"
item 2 in ``docs/guides/LYCORIS_ADAPTER_DESIGN.md``.

KNOWN BLIND SPOTS -- two conventions this file MIRRORS from ``layers.py``
rather than deriving, so no comparison against it can catch them: the LoKr
operand order (``kron(w1, w2)``; the swap has the SAME output shape), and the
``dora_scale`` reshape to ``(out, 1)``, which reads a ``wd_on_out=False``
``(1, in)`` vector as row magnitudes on a square weight. Both are open
questions for the upstream LyCORIS check.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

__all__ = [
    "adapter_delta_weight",
    "adapter_scale",
    "dora_effective_delta_weight",
    "dora_effective_weight",
    "loha_delta_weight",
    "lokr_delta_weight",
    "lora_delta_weight",
]

_F32 = torch.float32


def _f32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(_F32)


def _get(tensors: Mapping[str, torch.Tensor], name: str, algorithm: str) -> torch.Tensor:
    try:
        return _f32(tensors[name])
    except KeyError:
        raise KeyError(
            f"{algorithm} oracle needs tensor {name!r}; got {sorted(tensors)}"
        ) from None


def _low_rank_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` as the sum of its rank-1 terms -- the definition, not ``mm``."""
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dimensions disagree: {tuple(a.shape)} @ {tuple(b.shape)}")
    terms = [torch.outer(a[:, k], b[k, :]) for k in range(a.shape[1])]
    return torch.stack(terms, dim=0).sum(dim=0)


def _kronecker_product(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """``kron(w1, w2)`` assembled block by block, so it shares nothing with
    ``torch.kron`` -- which is what the layer under test calls."""
    rows = [
        torch.cat([w1[i, j] * w2 for j in range(w1.shape[1])], dim=1)
        for i in range(w1.shape[0])
    ]
    return torch.cat(rows, dim=0)


def adapter_scale(algorithm: str, rank: Optional[int], alpha: Optional[float]) -> float:
    """``alpha / rank``, mirroring ``layers.py``.

    CONVENTION, pending the upstream LyCORIS check: LoKr's unfactored form is
    rank 0 and the layer then uses ``alpha`` bare rather than 1.0.
    """
    if alpha is None:
        return 1.0
    if rank:
        return float(alpha) / float(rank)
    if algorithm == "lokr":
        return float(alpha)
    raise ValueError(f"{algorithm} scales by alpha/rank, so rank {rank!r} is unusable")


def lora_delta_weight(
    tensors: Mapping[str, torch.Tensor],
    *,
    rank: Optional[int],
    alpha: Optional[float],
    strength: float = 1.0,
) -> torch.Tensor:
    """``strength * (alpha/rank) * up @ down``, shape ``[out, in]``."""
    down = _get(tensors, "lora_down.weight", "lora")   # [rank, in]
    up = _get(tensors, "lora_up.weight", "lora")       # [out, rank]
    scale = adapter_scale("lora", rank, alpha)
    return _low_rank_product(up, down) * (scale * float(strength))


def loha_delta_weight(
    tensors: Mapping[str, torch.Tensor],
    *,
    rank: Optional[int],
    alpha: Optional[float],
    strength: float = 1.0,
) -> torch.Tensor:
    """``strength * (alpha/rank) * (w1_a @ w1_b) ⊙ (w2_a @ w2_b)``."""
    w1 = _low_rank_product(_get(tensors, "hada_w1_a", "loha"),
                           _get(tensors, "hada_w1_b", "loha"))
    w2 = _low_rank_product(_get(tensors, "hada_w2_a", "loha"),
                           _get(tensors, "hada_w2_b", "loha"))
    scale = adapter_scale("loha", rank, alpha)
    return (w1 * w2) * (scale * float(strength))


def lokr_delta_weight(
    tensors: Mapping[str, torch.Tensor],
    *,
    rank: Optional[int],
    alpha: Optional[float],
    strength: float = 1.0,
) -> torch.Tensor:
    """``strength * (alpha/rank) * kron(w1, w2)``, ``w2`` factored when rank > 0."""
    w1 = _get(tensors, "lokr_w1", "lokr")
    if "lokr_w2" in tensors:
        w2 = _get(tensors, "lokr_w2", "lokr")
    else:
        w2 = _low_rank_product(_get(tensors, "lokr_w2_a", "lokr"),
                               _get(tensors, "lokr_w2_b", "lokr"))
    scale = adapter_scale("lokr", rank, alpha)
    return _kronecker_product(w1, w2) * (scale * float(strength))


_DELTA_BY_ALGORITHM = {
    "lora": lora_delta_weight,
    "loha": loha_delta_weight,
    "lokr": lokr_delta_weight,
}


def adapter_delta_weight(
    algorithm: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    rank: Optional[int],
    alpha: Optional[float],
    strength: float = 1.0,
) -> torch.Tensor:
    try:
        fn = _DELTA_BY_ALGORITHM[algorithm]
    except KeyError:
        raise ValueError(
            f"no oracle for algorithm {algorithm!r}; have "
            f"{sorted(_DELTA_BY_ALGORITHM)}") from None
    return fn(tensors, rank=rank, alpha=alpha, strength=strength)


def dora_effective_weight(
    base_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    dora_scale: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """``W_base + s * (W_adapter - W_base)`` for the weight-decomposed families.

    ``W_adapter`` renormalizes each output row of ``W_base + delta`` and rescales
    it by that row's ``dora_scale``. ``delta_weight`` must therefore be the
    additive branch at UNIT strength: the strength rides on the interpolation,
    not on the delta.
    """
    w0 = _f32(base_weight)
    v = w0 + _f32(delta_weight)
    # Rows of a real weight are never all-zero; the clamp only keeps a
    # degenerate test fixture from producing NaN.
    v_norm = torch.linalg.vector_norm(v, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
    w_adapter = _f32(dora_scale).reshape(-1, 1) * (v / v_norm)
    return w0 + (w_adapter - w0) * float(strength)


def dora_effective_delta_weight(
    base_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    dora_scale: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """``W_eff(s) - W_base``: what a branch adds on top of the base forward."""
    return dora_effective_weight(base_weight, delta_weight, dora_scale,
                                 strength=strength) - _f32(base_weight)
