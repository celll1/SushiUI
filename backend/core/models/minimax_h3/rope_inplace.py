"""In-place rotary embedding for MiniMax-H3's ``MiniMaxH3AttnProcessor``.

WHY. ``_apply_rotary_emb`` (call sites: ``transformer_minimax_h3.py``'s
``MiniMaxH3AttnProcessor.__call__``, once for query and once for key) builds a
fresh ``torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()``
tensor while the pre-rope input is still live. At production scale
(``S = 97,159``, ``num_heads = 56``, ``head_dim = 128``, bf16) that concat is
1.297 GiB, once per call -- 2.594 GiB total across q and k, per block, and it
is the single mover of peak ALLOCATED across the whole 50-block forward
(measured: -0.946 GiB allocated at both ``blocks_to_swap=0`` and
``blocks_to_swap=40`` when this is skipped, `scratchpad/integrated_probe.py`
on the real ``w4a8_mixed`` checkpoint at ``S=97,159``).

``rotary_dim`` (``cos.shape[-1]``, 96 on the released checkpoints) is smaller
than ``head_dim`` (128): the trailing ``head_dim - rotary_dim`` channels are
the "pass-through" part the rotation never touches, so the rotated ``[:half]``
/ ``[half:rotary_dim]`` halves can be written straight back over the leading
``rotary_dim`` channels of the input tensor without ever materializing the
concat. This is REAL-NUMBER-exact by construction (it is the same six
elementwise ops the stock ``rotate_half`` expansion performs, just written to
their own storage instead of a freshly allocated one) and, unlike
``core.models.minimax_h3.ff_chunking``'s GEMM-noise caveat, there is no matmul
here at all -- every op is elementwise multiply/subtract/add, so there is no
reduction-order freedom for a BLAS kernel to exploit. Verified
``torch.equal`` bit-exact against the stock concat path at production shape
on an RTX 6000 Ada (`scratchpad/iso_probe.py`, `scratchpad/int_patch_probe.py`)
and is re-verified against tiny CPU shapes in
``backend/tests/minimax_h3_activation_memory_test.py``.

Deliberately NOT ``addcmul_``/``lerp_``-style fused ops: those fuse the
multiply-add into a single FMA, which rounds differently from the two
separate ``mul`` then ``sub``/``add`` the stock expression performs. Using
the same op *sequence* as the stock code, just targeting existing storage, is
what makes this bit-exact rather than merely close.

SAFETY INVARIANT (why in-place is safe here at all). This function is called
on ``query`` / ``key`` AFTER ``attn.norm_q`` / ``attn.norm_k``
(``MiniMaxH3AttnProcessor.__call__``, ``transformer_minimax_h3.py``): both are
``nn.RMSNorm`` instances, and ``RMSNorm.forward`` always materializes a fresh
output tensor (it is a normalize-then-scale op, not a view), so the tensor
this function mutates is never aliased with anything else live in the
caller -- in particular never with ``value``, even under
``attn.fused_projections`` where ``query, key, value = attn.to_qkv(...).chunk(3, dim=-1)``
makes q/k/v VIEWS into one buffer BEFORE the norm. If qk-norm were ever made
optional (it is not, today -- ``MiniMaxH3Attention.__init__`` always
constructs ``norm_q`` / ``norm_k``), this function would have to stop being
called directly on the processor's ``query`` / ``key`` and would instead need
its own defensive ``.clone()``, or it would corrupt ``value`` silently through
the shared ``to_qkv`` buffer.

Guarded exactly like ``ff_chunking.chunked_feed_forward``: in-place mutation
breaks autograd (the original tensor's data is required, unmodified, for
several backward formulas -- multiply's, in particular), so this falls back to
the stock, allocating expression whenever ``torch.is_grad_enabled()`` or the
input already carries a grad requirement. Unlike ``chunked_feed_forward``,
there is no sequence-length short-circuit here: this is not a chunking
strategy (there is no loop over ``row_budget``-sized slices to skip) -- the
in-place rewrite touches the same six elementwise ops per call at every
sequence length, so it is unconditionally at least as cheap as the stock
concat and importing ``FF_CHUNK_ROW_BUDGET`` here would document a threshold
that does not gate anything.
"""

from __future__ import annotations

import torch


def _apply_rotary_emb_stock(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The original, allocating expression -- used verbatim as the autograd fallback."""
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]

    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


def apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    r"""
    Rotate the leading ``rotary_dim`` channels of every head in place and pass the remaining channels
    through unchanged. ``hidden_states`` is ``(batch_size, seq_len, num_heads, head_dim)`` and
    ``cos``/``sin`` are ``(seq_len, rotary_dim)``. Inference-only; see module docstring for the guard
    and the safety invariant this relies on.
    """
    if torch.is_grad_enabled() or hidden_states.requires_grad:
        return _apply_rotary_emb_stock(hidden_states, cos, sin)

    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    c = cos.to(hidden_states.dtype)[None, :, None, :]
    s = sin.to(hidden_states.dtype)[None, :, None, :]
    x1 = hidden_states[..., :half]
    x2 = hidden_states[..., half:rotary_dim]
    # Both products read x1/x2 before either slice is overwritten below.
    o1 = x1 * c[..., :half] - x2 * s[..., :half]
    o2 = x2 * c[..., half:] + x1 * s[..., half:]
    hidden_states[..., :half] = o1
    hidden_states[..., half:rotary_dim] = o2
    return hidden_states
