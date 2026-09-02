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
from .observed import note_backend
from .registry import BACKENDS

# One-time dedup for the "which backend is actually running" info log.
_backend_used_logged = set()


def _repeat_kv_bshd(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand kv heads on the canonical BSHD layout ([B, S, H_kv, D], heads at
    dim 2) so the native path never calls SDPA's own ``enable_gqa`` broadcast.

    Equivalent to ``torch.repeat_interleave(x, dim=2, repeats=n_rep)`` -- the
    same grouping convention ``F.scaled_dot_product_attention(enable_gqa=True)``
    applies internally (each kv head is shared by ``n_rep`` consecutive query
    heads), so the result is numerically identical.
    """
    if n_rep == 1:
        return x
    b, s, h_kv, d = x.shape
    x = x[:, :, :, None, :].expand(b, s, h_kv, n_rep, d)
    return x.reshape(b, s, h_kv * n_rep, d)


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

    # SDPA's own enable_gqa broadcast is far slower than pre-expanding K/V
    # (measured ~9x on a 32q/8kv, head_dim 128 shape -- see
    # backend/core/models/sensenova/vendor/modeling_qwen3.py). Only the native
    # kernel pays for enable_gqa, so only pre-expand there; flash/sage already
    # broadcast GQA natively or are downgraded to native beforehand.
    dispatch_k, dispatch_v = k, v
    if resolved == "native" and gqa and k.shape[2] != q.shape[2]:
        n_rep = q.shape[2] // k.shape[2]
        dispatch_k = _repeat_kv_bshd(k, n_rep)
        dispatch_v = _repeat_kv_bshd(v, n_rep)
        gqa = False

    out = BACKENDS[resolved].fn(
        q,
        dispatch_k,
        dispatch_v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=gqa,
    )

    if out is None and resolved != "native":
        # Kernel failed at runtime -> fall back to native (never raise).
        _log_backend_used("native", fell_back_from=resolved)
        note_backend("native")
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
        # Record the backend that actually produced this call's output, so the
        # generation's row can name what RAN rather than what was requested.
        note_backend(resolved)

    if out is None:
        # Native is the terminal fallback; if it also failed, surface the error
        # rather than returning a wrong-shape/None tensor to the model.
        raise RuntimeError("[Attention] native SDPA failed; see prior error log")

    # Boundary transpose back to the caller's layout.
    if layout == "BHSD":
        out = out.transpose(1, 2).contiguous()

    return out


def dispatch_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
    mode: AttentionMode = AttentionMode.INFERENCE,
) -> torch.Tensor:
    """Variable-length (packed) attention over ``[total, H, D]`` tensors.

    Segment ``i`` of the queries (``cu_seqlens_q[i]:cu_seqlens_q[i+1]``) attends
    only to segment ``i`` of the keys/values. Segments may have different key
    lengths; nothing is padded. ``is_causal`` applies within a segment, with the
    query's last row aligned to the key's last column (FlashAttention's
    bottom-right convention), which reduces to plain causal when the two lengths
    are equal.

    ``flash`` resolves to ``flash_attn_varlen_func``. Every other backend (and
    any kernel failure) runs one SDPA call per segment on the same tensors,
    which is exact and needs no mask. Both accept unequal q/kv head counts.
    """
    backend = normalize_backend(backend)
    if backend in _PASSTHROUGH:
        raise ValueError(f"varlen attention is not implemented for the {backend!r} backend")
    resolved = resolve_backend(backend, mode, query, key, None, "BSHD")
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError("varlen attention takes packed [total, H, D] tensors")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must both hold n_segments + 1 offsets")

    if resolved == "flash":
        try:
            from flash_attn import flash_attn_varlen_func

            original_dtype = query.dtype
            if original_dtype in (torch.float16, torch.bfloat16):
                q, k, v = query, key, value
            else:
                q, k, v = (t.to(torch.bfloat16) for t in (query, key, value))
            out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q.to(torch.int32), cu_seqlens_k.to(torch.int32),
                int(max_seqlen_q), int(max_seqlen_k),
                dropout_p=dropout_p, softmax_scale=scale, causal=is_causal,
            )
            _log_backend_used("flash")
            note_backend("flash")
            return out.to(original_dtype) if out.dtype != original_dtype else out
        except ImportError:
            _log_backend_used("native", fell_back_from="flash")
        except Exception as exc:  # noqa: BLE001 - fall back, never raise into the model
            print(f"[Attention] flash varlen error ({exc}); falling back to per-segment SDPA")
            _log_backend_used("native", fell_back_from="flash")

    starts_q = cu_seqlens_q.tolist()
    starts_k = cu_seqlens_k.tolist()
    n_rep = query.shape[1] // key.shape[1]
    outputs = []
    for i in range(len(starts_q) - 1):
        q = query[starts_q[i]:starts_q[i + 1]].transpose(0, 1).unsqueeze(0)  # [1, H, Sq, D]
        k = key[starts_k[i]:starts_k[i + 1]].transpose(0, 1).unsqueeze(0)
        v = value[starts_k[i]:starts_k[i + 1]].transpose(0, 1).unsqueeze(0)
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        seq_q, seq_k = q.shape[2], k.shape[2]
        if is_causal and seq_q != seq_k:
            rows = torch.arange(seq_q, device=q.device).unsqueeze(1) + (seq_k - seq_q)
            cols = torch.arange(seq_k, device=q.device).unsqueeze(0)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=cols <= rows, dropout_p=dropout_p, scale=scale)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        outputs.append(out.squeeze(0).transpose(0, 1))
    note_backend("native")
    return torch.cat(outputs, dim=0)
