"""
Per-backend attention callables for the unified attention conduit.

Each backend function receives tensors in the CANONICAL BSHD layout
(``[batch, seq_len, num_heads, head_dim]``) and returns a tensor in the SAME
BSHD layout. Layout adaptation for architectures that hold ``[B, H, S, D]``
(BHSD) is performed by the conduit (see ``dispatch.py``) at the boundary, so
these functions never have to reason about BHSD.

Contract for every backend fn:
    fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
       scale=None, enable_gqa=False) -> Optional[torch.Tensor]

    * On success:  returns the attention output in BSHD layout.
    * On failure:  wraps the kernel in ``try/except`` and returns ``None`` so
                   the conduit can fall back to native SDPA. A backend fn MUST
                   NOT raise into the model.

Capability gating (mode / mask / head_dim / GQA / dtype) is handled by
``config.resolve_backend`` BEFORE these functions are called; each function
therefore assumes it is only invoked when its capabilities are satisfied
(e.g. ``_sage_attn`` is never called with a non-None mask).

References verified against the installed builds:
    * sageattention.core.sageattn(q, k, v, tensor_layout=..., is_causal=...,
      sm_scale=..., return_lse=...)  -- NHD == [B, S, H, D] == our canonical.
    * flash_attn.flash_attn_func(q, k, v, dropout_p=..., softmax_scale=...,
      causal=...)  -- expects BSHD, fp16/bf16 only.
"""

from typing import Optional

import torch
import torch.nn.functional as F

_HALF_DTYPES = (torch.float16, torch.bfloat16)

# Kernel-fallback warnings are emitted from per-call hot paths, so they are
# deduped -- but the dedup is keyed by (generation id, message), NOT by message
# alone. A process-lifetime dedup (what this was) meant only the FIRST
# generation after a backend start could ever record an
# ``attention_kernel_fallback``; every later generation silently ran on native
# SDPA (different dtype from the flash path, so NOT numerically equivalent)
# with nothing on its row to say so. Worse, if the first fallback fired outside
# a generation (model load, warm-up, a route with no ``start_generation()``),
# the warning was swallowed for the entire process lifetime.
#
# The console print keeps the once-per-process dedup: it is a log, and the
# fallback fires per attention call.
_warned_kernels: set = set()      # (generation_id, message) -- warnings[] channel
_logged_kernels: set = set()      # message -- console only


def _warn_kernel_fallback(message: str) -> None:
    """Best-effort: surface an attention-kernel fallback once per generation.

    Lazily imported so this attention module never hard-depends on the api
    package at import time. Never raises.
    """
    if message not in _logged_kernels:
        _logged_kernels.add(message)
        print(f"[Attention] {message}")
    try:
        from api.generation_status import add_warning, current_generation_id
        gen_id = current_generation_id()
    except Exception:
        return
    if gen_id == 0:
        # Outside a generation there is nothing to attach the warning to; do
        # NOT record a dedup entry, so the next generation still reports it.
        return
    key = (gen_id, message)
    if key in _warned_kernels:
        return
    _warned_kernels.add(key)
    if len(_warned_kernels) > 512:
        _warned_kernels.clear()
        _warned_kernels.add(key)
    try:
        add_warning(message, code="attention_kernel_fallback")
    except Exception:
        pass


def _process_mask(attn_mask: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
    """
    Convert an attention mask into an additive float mask suitable for
    ``F.scaled_dot_product_attention``.

    * ``None``            -> ``None``
    * 2D bool/other       -> broadcast to ``[B, 1, 1, S]``
    * bool mask           -> ``0.0`` where True, ``-inf`` where False
    * float/additive mask -> returned unchanged

    Mirrors the extracted Z-Image behaviour (``zimage_utils._process_mask``);
    duplicated locally to keep ``backends.py`` free of a circular import with
    the ``zimage_utils`` shim (which imports the conduit).
    """
    if attn_mask is None:
        return None

    if attn_mask.ndim == 2:
        attn_mask = attn_mask[:, None, None, :]

    if attn_mask.dtype == torch.bool:
        new_mask = torch.zeros_like(attn_mask, dtype=dtype)
        new_mask.masked_fill_(~attn_mask, float("-inf"))
        return new_mask

    return attn_mask


def _native_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> Optional[torch.Tensor]:
    """
    PyTorch scaled-dot-product attention (the always-available fallback).

    Inputs/outputs are BSHD. SDPA wants BHSD, so we transpose in and out. The
    ``enable_gqa`` flag is forwarded so an unequal q/kv head count does not
    raise (see conduit R3). Returns ``None`` only on an unexpected error, which
    the conduit treats as a hard failure (native is the terminal fallback).
    """
    try:
        # BSHD -> BHSD for SDPA.
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        processed_mask = _process_mask(attn_mask, q.dtype)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=processed_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

        # BHSD -> BSHD.
        return out.transpose(1, 2).contiguous()
    except Exception as e:  # noqa: BLE001 - never raise into the model
        print(f"[Attention] native SDPA error: {e}")
        return None


def _flash_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> Optional[torch.Tensor]:
    """
    FlashAttention-2 (``flash_attn_func``), BSHD in/out.

    FlashAttention only accepts fp16/bf16; non-half inputs are cast to bf16 and
    the output is cast back to the original dtype (mirrors the reference dtype
    handling). Custom masks are not supported by the kernel; the mask guard in
    ``resolve_backend`` guarantees ``attn_mask is None`` here, so ``attn_mask``
    is intentionally ignored. FlashAttention broadcasts unequal q/kv heads
    natively, so ``enable_gqa`` is a no-op. Returns ``None`` on any failure.
    """
    try:
        from flash_attn import flash_attn_func

        original_dtype = query.dtype
        needs_conversion = original_dtype not in _HALF_DTYPES

        if needs_conversion:
            q = query.to(torch.bfloat16)
            k = key.to(torch.bfloat16)
            v = value.to(torch.bfloat16)
        else:
            q, k, v = query, key, value

        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=is_causal,
        )

        if needs_conversion:
            out = out.to(original_dtype)

        return out.contiguous()
    except ImportError:
        _warn_kernel_fallback("flash_attn not available; falling back to native attention")
        return None
    except Exception as e:  # noqa: BLE001 - never raise into the model
        _warn_kernel_fallback(f"flash_attn error ({e}); falling back to native attention")
        return None


def _sage_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> Optional[torch.Tensor]:
    """
    SageAttention (INT8-quantized attention), BSHD in/out.

    R1 fix: our canonical layout is BSHD == ``[B, S, H, D]`` == sageattention's
    ``NHD``. We therefore call ``sageattn(..., tensor_layout='NHD')`` -- NOT
    ``'HND'`` as the legacy reference did (the legacy code mislabelled BSHD as
    HND, feeding transposed strides to the kernel).

    Notes on the verified API:
        * The top-level ``sageattn`` takes ``sm_scale`` (not ``scale``) and does
          NOT accept ``attn_mask`` (masks are only on the low-level kernels).
          Sage is gated ``supports_mask=False`` upstream, so ``attn_mask`` is
          always ``None`` here and is intentionally not forwarded.
        * Sage requires fp16/bf16 inputs; non-half tensors are cast to bf16 and
          the output is cast back.
        * Sage does not support GQA (gated ``supports_gqa=False`` upstream), so
          ``enable_gqa`` is never True here.
    Returns ``None`` on any failure.
    """
    try:
        from sageattention import sageattn

        original_dtype = query.dtype
        needs_conversion = original_dtype not in _HALF_DTYPES

        if needs_conversion:
            q = query.to(torch.bfloat16)
            k = key.to(torch.bfloat16)
            v = value.to(torch.bfloat16)
        else:
            q, k, v = query, key, value

        # Sage kernels prefer contiguous tensors.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out = sageattn(
            q,
            k,
            v,
            tensor_layout="NHD",  # NHD == [B, S, H, D] == canonical BSHD (R1)
            is_causal=is_causal,
            sm_scale=scale,
        )

        if needs_conversion:
            out = out.to(original_dtype)

        return out.contiguous()
    except ImportError:
        _warn_kernel_fallback("sageattention not available; falling back to native attention")
        return None
    except Exception as e:  # noqa: BLE001 - never raise into the model
        _warn_kernel_fallback(f"sageattention error ({e}); falling back to native attention")
        return None


def _tq_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> Optional[torch.Tensor]:
    """
    TQ-Attention (Triton-Quantized attention with a full backward), BSHD in/out.

    Sage-compatible signature. Our canonical layout BSHD == ``[B, S, H, D]`` ==
    tq_attention's ``NHD``, so we call ``tq_attention(..., layout='NHD')``.

    Notes on the verified API (venv tq_attention.tq_attention):
        * Takes ``sm_scale`` (not ``scale``); does NOT accept an additive/boolean
          mask -> gated ``supports_mask=False`` upstream, so ``attn_mask`` is
          always ``None`` here and is intentionally not forwarded.
        * Supports a real backward (Triton) -> usable in TRAINING (``trainable=True``).
        * Broadcasts unequal q/kv heads (GQA verified) -> ``enable_gqa`` unused.
        * head_dim must be a supported power of 2 (64/128); other dims are gated
          out upstream (``allowed_head_dims={64, 128}``) and never reach here.
        * Quantized kernel: non-half inputs are cast to bf16 and the output cast
          back (matches the sage/flash quant path).
    Returns ``None`` on any failure (conduit falls back to native).
    """
    try:
        from tq_attention import tq_attention as _tq

        original_dtype = query.dtype
        needs_conversion = original_dtype not in _HALF_DTYPES

        if needs_conversion:
            q = query.to(torch.bfloat16)
            k = key.to(torch.bfloat16)
            v = value.to(torch.bfloat16)
        else:
            q, k, v = query, key, value

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out = _tq(
            q,
            k,
            v,
            layout="NHD",  # NHD == [B, S, H, D] == canonical BSHD
            is_causal=is_causal,
            sm_scale=scale,
            # Pin the exact-P Triton backward for training reproducibility.
            # tq_attention 0.6.0 changed the backward_mode default "triton"->"auto",
            # which silently switches training gradients to the FA2 composite
            # backward whenever flash-attn happens to be installed (INT8 / no-QJL /
            # head_dim in {64,128} -- always true on the conduit tq path). That makes
            # gradients environment-dependent (flash-attn present or not). Forcing
            # "triton" preserves the pre-0.6.0 deterministic default so training runs
            # are reproducible regardless of the local flash-attn install. Inference
            # (no grad) never touches the backward, so this is a no-op there.
            backward_mode="triton",
        )

        if needs_conversion:
            out = out.to(original_dtype)

        return out.contiguous()
    except ImportError:
        _warn_kernel_fallback("tq_attention not available; falling back to native attention")
        return None
    except Exception as e:  # noqa: BLE001 - never raise into the model
        _warn_kernel_fallback(f"tq_attention error ({e}); falling back to native attention")
        return None
