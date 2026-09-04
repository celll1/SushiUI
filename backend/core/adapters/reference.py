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

KNOWN BLIND SPOTS -- conventions this file SHARES with ``layers.py`` rather
than deriving, so no comparison between the two can catch them. Each was
checked by hand against upstream at
``03270a3839102e63b48578c80e7c024036de74d7``:

1. the LoKr operand order (``kron(w1, w2)``; the swap has the SAME output
   shape) -- agrees with ``make_kron``;
2. LoKr's scale derivation, including which operand's rank is the divisor --
   agrees with ``rank_scale`` in ``kernels/autograd/lokr.py``.

The ``dora_scale`` axis is written from the definition here rather than copied,
but that too is only evidence once read against upstream ``_weight_decompose``,
which a test cannot do. The ROW form -- upstream's ``wd_on_out=True`` default,
stored ``(out,)`` or ``(out, 1)`` -- no longer rests on that agreement alone:
``adapter_oracle_gate_cheap_test`` compares it against PEFT's
``DoraLinearLayer``, a third implementation installed in this venv, which norms
along ``dim=1`` and holds one magnitude per output row, and onto whose
``lora_magnitude_vector`` diffusers maps a Kohya ``dora_scale`` directly.
Measured agreement 4.1e-7; a reversed row order is 1.04 off. The COLUMN form
(``(1, in)``) has no such witness -- PEFT does not implement it -- and remains
two mirrors of one reading.
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
    "lokr_scale",
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


def _apply_scalar(delta: torch.Tensor,
                  tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """A trained ``scalar`` (``use_scalar=True``), folded in BEFORE the strength
    multiplier.

    Absent from any real file -- upstream's ``custom_state_dict`` folds it into
    the saved ``w1``/``hada_w1_a`` and emits no key -- so this arm exists for
    live layer state, not for a checkpoint."""
    scalar = tensors.get("scalar")
    return delta if scalar is None else delta * _f32(scalar)


def _kronecker_product(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """``kron(w1, w2)`` assembled block by block, so it shares nothing with
    ``torch.kron`` -- which is what the layer under test calls."""
    rows = [
        torch.cat([w1[i, j] * w2 for j in range(w1.shape[1])], dim=1)
        for i in range(w1.shape[0])
    ]
    return torch.cat(rows, dim=0)


def adapter_scale(algorithm: str, rank: Optional[int], alpha: Optional[float]) -> float:
    """``alpha / rank`` for the algebras whose scale is a declared constant.

    LoKr is NOT one of them: its divisor comes from the tensor set, so it has
    ``lokr_scale`` below instead.
    """
    if alpha is None:
        return 1.0
    if rank:
        return float(alpha) / float(rank)
    raise ValueError(f"{algorithm} scales by alpha/rank, so rank {rank!r} is unusable")


def lokr_scale(alpha: Optional[float], w1_a: Optional[torch.Tensor],
               w2_a: Optional[torch.Tensor]) -> float:
    """LoKr's scale, from WHICH OPERANDS ARE FACTORED rather than from a
    declared rank -- upstream's ``rank_scale`` in ``kernels/autograd/lokr.py``.

    With both operands stored full there is no rank, upstream sets
    ``alpha = lora_dim`` so ``alpha/rank`` is 1, and it writes that ``lora_dim``
    into the file's ``alpha``. Dividing by nothing and using the stored value
    would scale the adapter by ``lora_dim`` (4, 8, 32...).

    w1 first, as upstream does; no representable checkpoint distinguishes the
    two orders.
    """
    if w1_a is not None:
        return 1.0 if alpha is None else float(alpha) / float(w1_a.shape[1])
    if w2_a is not None:
        return 1.0 if alpha is None else float(alpha) / float(w2_a.shape[1])
    return 1.0


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
    delta = (w1 * w2) * adapter_scale("loha", rank, alpha)
    return _apply_scalar(delta, tensors) * float(strength)


def lokr_delta_weight(
    tensors: Mapping[str, torch.Tensor],
    *,
    rank: Optional[int] = None,
    alpha: Optional[float],
    strength: float = 1.0,
) -> torch.Tensor:
    """``strength * scale * kron(w1, w2)``, either operand full or factored.

    ``rank`` is accepted for signature uniformity and IGNORED: see
    ``lokr_scale``.
    """
    if "lokr_w1" in tensors:
        w1, w1_a = _get(tensors, "lokr_w1", "lokr"), None
    else:
        w1_a = _get(tensors, "lokr_w1_a", "lokr")
        w1 = _low_rank_product(w1_a, _get(tensors, "lokr_w1_b", "lokr"))
    if "lokr_w2" in tensors:
        w2, w2_a = _get(tensors, "lokr_w2", "lokr"), None
    else:
        w2_a = _get(tensors, "lokr_w2_a", "lokr")
        w2 = _low_rank_product(w2_a, _get(tensors, "lokr_w2_b", "lokr"))
    delta = _kronecker_product(w1, w2) * lokr_scale(alpha, w1_a, w2_a)
    return _apply_scalar(delta, tensors) * float(strength)


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

    ``W_adapter`` renormalizes each row (or column -- see below) of
    ``W_base + delta`` and rescales it by that row's ``dora_scale``.
    ``delta_weight`` must therefore be the additive branch at UNIT strength: the
    strength rides on the interpolation, not on the delta.
    """
    w0 = _f32(base_weight)
    v = w0 + _f32(delta_weight)
    out_features, in_features = w0.shape
    magnitudes = _f32(dora_scale)
    shape = tuple(magnitudes.shape)
    # The axis IS the shape: wd_on_out=True (upstream's default) stores one
    # magnitude per output row, wd_on_out=False one per input column, and
    # nothing else records which. Written from that definition, not from
    # ``layers.py`` -- on a square weight a mirrored reshape proves nothing.
    if shape in ((out_features,), (out_features, 1)):
        axis, view = 1, (out_features, 1)
    elif shape == (1, in_features):
        axis, view = 0, (1, in_features)
    else:
        raise ValueError(
            f"dora_scale shape {shape} is neither ({out_features}, 1) nor "
            f"(1, {in_features}) for a [{out_features}, {in_features}] weight")
    # A real weight has no all-zero row or column; the clamp only keeps a
    # degenerate test fixture from producing NaN.
    v_norm = torch.linalg.vector_norm(v, ord=2, dim=axis, keepdim=True).clamp_min(1e-12)
    w_adapter = magnitudes.reshape(view) * (v / v_norm)
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
