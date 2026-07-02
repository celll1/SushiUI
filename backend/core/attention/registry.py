"""
Attention backend registry.

A single frozen :class:`AttentionBackend` descriptor per backend captures its
capabilities (trainability, mask support, head-dim limits, dtype needs, GQA
support) alongside the callable that runs its kernel. ``resolve_backend``
(``config.py``) reads these descriptors to decide when a requested backend must
be downgraded to native; the conduit (``dispatch.py``) reads ``fn`` to run it.

Adding a future backend (e.g. TQ) is a ONE-branch change: add one
``AttentionBackend`` entry here and one callable in ``backends.py``. No conduit
edits are required.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Set

from .backends import _flash_attn, _native_sdpa, _sage_attn


@dataclass(frozen=True)
class AttentionBackend:
    """Immutable capability descriptor for an attention backend.

    Attributes:
        name: Canonical backend string ("native" | "flash" | "sage" | ...).
        fn: Kernel callable. Signature:
            ``fn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=None, enable_gqa=False) -> Tensor | None``. Tensors are BSHD;
            returns BSHD or ``None`` on failure.
        trainable: If False, the backend is refused in TRAINING mode (no
            backward kernel) and downgraded to native.
        supports_mask: If False, the backend is downgraded to native whenever
            ``attn_mask is not None``.
        max_head_dim: If not None, downgrade to native when ``head_dim > max``.
        allowed_head_dims: If not None, downgrade to native when
            ``head_dim not in`` this set.
        needs_half_dtype: If True, the kernel requires fp16/bf16; the backend fn
            casts q/k/v to bf16 and casts the output back (informational; the
            actual cast lives in the backend fn).
        supports_gqa: If False, downgrade to native when ``H_kv != H``.
    """

    name: str
    fn: Callable
    trainable: bool
    supports_mask: bool
    max_head_dim: Optional[int]
    allowed_head_dims: Optional[Set[int]]
    needs_half_dtype: bool
    supports_gqa: bool


# ---------------------------------------------------------------------------
# Backend table.
#
# sage:
#   * INT8-quantized, inference-only (no backward) -> trainable=False.
#   * No custom-mask support at the top-level sageattn API -> supports_mask=False.
#   * Installed SageAttention2 build supports head_dim in {64, 96, 128} with a
#     hard cap of 128; head_dims outside this set (SD1.5 40/80/160,
#     Ideogram4 256, MiniT2I l16 52) are refused -> native. Confirmed against
#     the installed sageattention.core kernels.
#   * Requires equal q/kv head counts -> supports_gqa=False (Z-Image GQA
#     auto-downgrades to native for sage).
#
# flash:
#   * FlashAttention-2, supports backward -> trainable=True.
#   * No custom mask (only causal) -> supports_mask=False.
#   * head_dim cap 256 (FA-2 supports 256 on Ampere+/Hopper) -> covers
#     Ideogram4 D=256 as borderline; mask guard still forces native when the
#     block-diagonal segment mask is present.
#   * Broadcasts unequal q/kv heads natively -> supports_gqa=True.
#
# native:
#   * PyTorch SDPA -- the terminal fallback. Handles everything (mask, GQA via
#     enable_gqa, any head_dim, any dtype). Always trainable.
# ---------------------------------------------------------------------------
BACKENDS = {
    "native": AttentionBackend(
        name="native",
        fn=_native_sdpa,
        trainable=True,
        supports_mask=True,
        max_head_dim=None,
        allowed_head_dims=None,
        needs_half_dtype=False,
        supports_gqa=True,
    ),
    "flash": AttentionBackend(
        name="flash",
        fn=_flash_attn,
        trainable=True,
        supports_mask=False,
        max_head_dim=256,
        allowed_head_dims=None,
        needs_half_dtype=True,
        supports_gqa=True,
    ),
    "sage": AttentionBackend(
        name="sage",
        fn=_sage_attn,
        trainable=False,
        supports_mask=False,
        max_head_dim=128,
        allowed_head_dims={64, 96, 128},
        needs_half_dtype=True,
        supports_gqa=False,
    ),
    # "tq": reserved -- add ONE AttentionBackend entry here + one fn in
    # backends.py to enable Triton-Quantized attention. No conduit edits.
}
