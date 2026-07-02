"""
The unified attention conduit.

``dispatch_attention`` is the ONE backend-agnostic entry point used by every
architecture that routes attention through this module. It:

    1. Normalizes the backend string (``normal``/``none``/``sdpa``/``None`` ->
       ``native``; unknown -> native+warn; ``sla`` passthrough).
    2. Short-circuits non-fungible passthrough backends (``sla``) at the TOP,
       before registry resolution (R2).
    3. Resolves capability guards (MODE / mask / head_dim / GQA), downgrading to
       native with a one-time reason log where required.
    4. Transposes BHSD<->BSHD at the boundary so every architecture feeds its
       native tensors while the kernels see ONE canonical layout.
    5. Auto-enables GQA on the native path when ``H_kv != H`` (R3).
    6. Dispatches to the selected backend fn, falling back to native if the
       kernel fails (returns ``None``). A model never sees a raised kernel
       error or a wrong-shape output.

Canonical layout: BSHD == ``[batch, seq_len, num_heads, head_dim]`` -- what
FlashAttention / SageAttention want natively. Architectures holding
``[B, H, S, D]`` pass ``layout="BHSD"`` and the conduit transposes for them.
"""

from enum import Enum
from typing import Optional

import torch

from .config import (
    _PASSTHROUGH,
    normalize_backend,
    resolve_backend,
    to_diffusers_backend,  # noqa: F401 - re-exported for callers via __init__
)
from .registry import BACKENDS

# One-time dedup for the "which backend is actually running" info log.
_backend_used_logged = set()


class AttentionMode(str, Enum):
    """Attention execution mode.

    A ``str`` enum so ``mode == "training"`` works across module boundaries
    without importing this enum (used by ``config.resolve_backend``).
    """

    INFERENCE = "inference"
    TRAINING = "training"


def _log_backend_used(backend: str, fell_back_from: Optional[str] = None) -> None:
    """Log the effective backend once per (backend, origin) pair."""
    dedup_key = f"{backend}<-{fell_back_from}"
    if dedup_key in _backend_used_logged:
        return
    _backend_used_logged.add(dedup_key)
    if fell_back_from is not None and fell_back_from != backend:
        print(f"[Attention] using {backend.upper()} backend (fallback from {fell_back_from.upper()})")
    else:
        print(f"[Attention] using {backend.upper()} backend")


def _dispatch_passthrough(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    layout: str,
    enable_gqa: bool,
) -> torch.Tensor:
    """Handle a non-fungible passthrough backend (currently only ``sla``).

    SLA (Sparse-Linear Attention) is a separate, non-fungible backend owned by
    the Z-Image SLA subsystem. When an SLA model is loaded, its attention
    modules run their OWN forward (with the extra ``proj_l`` projection); such a
    model never actually reaches this conduit with ``backend='sla'``.

    This branch exists so that (a) ``normalize_backend`` can preserve the
    ``'sla'`` string verbatim without clobbering it to native, and (b) if the
    string ever does reach the conduit in a build WITHOUT an SLA kernel present
    (as in this branch), we degrade gracefully to native math rather than
    warning "unknown backend" or crashing. A real SLA kernel plugs in HERE.
    """
    _log_backend_used(backend)
    # Reserved SLA hook. No SLA kernel is present in this build, so fall back to
    # native math (same result the legacy shim produced for an unrecognized
    # backend), keeping the input layout intact.
    return dispatch_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        backend="native",
        mode=AttentionMode.INFERENCE,
        layout=layout,
        enable_gqa=enable_gqa,
    )


def dispatch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
    mode: AttentionMode = AttentionMode.INFERENCE,
    layout: str = "BSHD",
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Backend-agnostic attention dispatch.

    Args:
        query: ``[B, S_q, H, D]`` (BSHD) or ``[B, H, S_q, D]`` (BHSD, see ``layout``).
        key:   ``[B, S_k, H_kv, D]`` (BSHD) or ``[B, H_kv, S_k, D]`` (BHSD).
        value: same layout/shape as ``key``.
        attn_mask: Optional mask in SDPA score space (bool or additive float).
        dropout_p: Attention dropout probability.
        is_causal: Apply causal masking.
        scale: Softmax scale; ``None`` uses ``1/sqrt(head_dim)``.
        backend: "native"|"flash"|"sage" (aliases "normal"/"none"/"sdpa"/None
            -> native); "sla" passes through.
        mode: ``AttentionMode.INFERENCE`` or ``AttentionMode.TRAINING`` (gates
            inference-only backends).
        layout: "BSHD" (canonical) or "BHSD" (transposed at the boundary).
        enable_gqa: Force GQA on the native path (auto-enabled when H_kv != H).

    Returns:
        Attention output in the SAME layout as the inputs.
    """
    backend = normalize_backend(backend)

    # R2: short-circuit non-fungible passthrough backends BEFORE registry
    # resolution so a required backend (sla) is never rewritten to native.
    if backend in _PASSTHROUGH:
        return _dispatch_passthrough(
            backend, query, key, value, attn_mask, dropout_p, is_causal, scale, layout, enable_gqa
        )

    resolved = resolve_backend(backend, mode, query, key, attn_mask, layout)

    # Boundary transpose: BHSD -> canonical BSHD for the kernels.
    if layout == "BHSD":
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
    else:
        q, k, v = query, key, value

    # R3: auto-enable GQA on the native path when q/kv head counts differ.
    # Computed on the canonical BSHD view (heads at dim 2). FlashAttention
    # broadcasts GQA natively and ignores this flag; sage is already downgraded
    # when heads are unequal.
    gqa = enable_gqa or (k.shape[2] != q.shape[2])

    out = BACKENDS[resolved].fn(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=gqa,
    )

    if out is None and resolved != "native":
        # Kernel failed at runtime -> fall back to native (never raise).
        _log_backend_used("native", fell_back_from=resolved)
        out = BACKENDS["native"].fn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=gqa,
        )
    elif out is not None:
        _log_backend_used(resolved)

    if out is None:
        # Native is the terminal fallback; if it also failed, surface the error
        # rather than returning a wrong-shape/None tensor to the model.
        raise RuntimeError("[Attention] native SDPA failed; see prior error log")

    # Boundary transpose back to the caller's layout.
    if layout == "BHSD":
        out = out.transpose(1, 2).contiguous()

    return out
