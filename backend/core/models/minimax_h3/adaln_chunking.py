"""Sequence-dim chunking of MiniMax-H3's AdaLN modulation, gated residual add,
and the output tail (``MiniMaxH3AdaLayerNormOut``).

WHY. All three of these are ROW-WISE ops over the packed sequence -- an
``index_select`` gather that expands a <=9-row (block modulation) or <=1025-row
(output tail, over distinct timesteps) table to the full sequence length, an
elementwise ``RMSNorm`` (normalizes within a row, never across rows), and an
elementwise multiply/add -- no GEMM, no cross-row reduction anywhere in any of
the three. Splitting them over the sequence axis and concatenating (or writing
into a preallocated buffer) is therefore exact by construction, more directly
so than ``core.models.minimax_h3.ff_chunking``'s SwiGLU chunking, which has to
reason about GEMM reduction-order noise because a Linear is involved. Nothing
here calls a Linear over a chunked axis in a way that changes reduction order:
``adaln_proj.linear`` / ``norm_out.linear`` are still called ONCE, over the
untouched small ``(num_rows, ...)`` table, before any chunk loop begins --
only the ``index_select`` expansion and the elementwise arithmetic are
chunked.

MEASURED (real ``w4a8_mixed`` checkpoint, ``S=97,159``, RTX 6000 Ada,
``scratchpad/integrated_probe.py`` / ``scratchpad/int_patch_probe.py``):

  * Output tail (``chunked_norm_out``, replaces ``MiniMaxH3AdaLayerNormOut.forward``):
    -3.895 GiB peak reserved at both ``blocks_to_swap=0`` and
    ``blocks_to_swap=40``. Six full-size ``(1, S, 5376)`` bf16 tensors
    (0.973 GiB each) plus the output heads' ``.to(float32)`` cast (1.946 GiB)
    collapse to one preallocated ``out_dtype`` buffer written chunk-by-chunk.
    Isolated timing (`scratchpad/iso_probe.py`): unchunked and chunked are
    within run-to-run noise of each other.
  * Block modulation + gated residual (``chunked_ada_modulate`` /
    ``gated_residual_add``, both called twice per block -- once for
    attention's pre-norm, once for the feed-forward's): -1.949 / -2.274 GiB
    peak reserved and -0.972 GiB peak allocated at ``blocks_to_swap=40``
    (module-level isolate measured faster, not slower: modulation
    0.090s -> 0.048s, gated add 0.049s -> 0.016s per call --
    `scratchpad/iso_probe.py`).

ROW BUDGET. Reuses ``core.models.minimax_h3.ff_chunking.FF_CHUNK_ROW_BUDGET``
(12,288) rather than a second constant: all three ops here chunk the exact
same packed-sequence axis, at the exact same production length
(``S = 97,159``) the FF budget was swept against, and the AdaLN-curve
projections that feed the modulation tables here are orders of magnitude
smaller than the SwiGLU activations that budget was tuned for -- there is no
reason the efficient chunk count would differ per op. A second constant would
only document the same number twice.

GUARD. Every function here refuses to chunk (and, for the gated residual,
refuses to mutate in place) whenever ``torch.is_grad_enabled()`` or the
relevant input already carries a grad requirement, falling back to the exact
stock expression -- same pattern, same reasoning as
``ff_chunking.chunked_feed_forward``: chunking under autograd would save MORE
activations for backward (one graph node per chunk), and in-place mutation
under autograd can corrupt values several backward formulas still need. Also
short-circuits whenever ``seq_len <= row_budget`` (the common case for short
clips), taking the plain stock expression with zero chunking overhead -- see
``ff_chunking.py:158`` for the same pattern.

FBCACHE-ALIASING HAZARD (gated residual only). ``gated_residual_add`` mutates
its ``residual`` argument in place under the no-grad, long-sequence path.
Inside ``MiniMaxH3TransformerBlock.forward``, ``residual = hidden_states`` is
the SAME tensor object the block *received* -- so for block 0, an in-place
gated add mutates the very tensor
``core.models.minimax_h3_block_loop_wrapper.MiniMaxH3BlockLoopWrapper._custom_forward``
keeps around as ``original_hidden_states`` (used at ``:291``/``:309``/``:314``
to compute FBCache's stored residual and to reconstruct a cache hit's output).
Without a fix, block 0's in-place add would zero out
``first_residual = hidden_states - original_hidden_states`` (both sides would
be the same, now-identical object), silently breaking the FBCache guard with
no exception anywhere. Fixed at the wrapper, not here: ``_custom_forward``
clones ``hidden_states`` into ``original_hidden_states`` once, before the
block loop, ONLY when an FBCache is attached (one 0.973 GiB copy, on a path
that is already mutually exclusive with block swap -- see
``MiniMaxH3BlockLoopWrapper.attach_fbcache``). The stock (non-wrapper)
``MiniMaxH3Transformer3DModel.forward`` never keeps an aliased reference to
any pre-block-loop tensor, so it needs no such clone; Spectrum forecasts the
model's OUTPUT rather than reading the stream mid-forward, so it is
unaffected by this in-place change either way.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.models.minimax_h3.ff_chunking import FF_CHUNK_ROW_BUDGET


# ---------------------------------------------------------------------------
# Block-level AdaLN modulation: norm(hidden_states) * (1 + scale) + shift,
# scale/shift gathered per row from a <=9-row table via `adaln_indices`.
# ---------------------------------------------------------------------------

def _ada_modulate_apply(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    adaln_indices: torch.Tensor,
) -> torch.Tensor:
    normed = norm(hidden_states)
    return normed * (1.0 + scale.index_select(0, adaln_indices)) + shift.index_select(0, adaln_indices)


def chunked_ada_modulate(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    adaln_indices: torch.Tensor,
    row_budget: int = FF_CHUNK_ROW_BUDGET,
) -> torch.Tensor:
    """``norm(hidden_states) * (1 + scale[adaln_indices]) + shift[adaln_indices]``, chunked over
    the sequence axis into a preallocated buffer. Inference-only and a no-op below ``row_budget``
    rows; see module docstring."""
    if torch.is_grad_enabled() or hidden_states.requires_grad:
        return _ada_modulate_apply(norm, hidden_states, scale, shift, adaln_indices)

    seq_len = hidden_states.shape[1]
    if seq_len <= row_budget:
        return _ada_modulate_apply(norm, hidden_states, scale, shift, adaln_indices)

    out = torch.empty_like(hidden_states)
    for s in range(0, seq_len, row_budget):
        e = min(s + row_budget, seq_len)
        idx = adaln_indices[s:e]
        normed = norm(hidden_states[:, s:e])
        out[:, s:e] = normed * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)
    return out


# ---------------------------------------------------------------------------
# Gated residual add: residual + gate[adaln_indices] * delta, in place when safe.
# ---------------------------------------------------------------------------

def gated_residual_add(
    residual: torch.Tensor,
    gate: torch.Tensor,
    adaln_indices: torch.Tensor,
    delta: torch.Tensor,
    row_budget: int = FF_CHUNK_ROW_BUDGET,
) -> torch.Tensor:
    """``residual + gate[adaln_indices] * delta``. Mutates ``residual`` in place, chunked over the
    sequence axis, whenever it is safe to (inference, long sequence); returns the plain
    out-of-place expression otherwise. See the module docstring's FBCache-aliasing hazard note --
    the caller (``MiniMaxH3BlockLoopWrapper``) is responsible for not handing this function a
    tensor it still needs unmodified."""
    if torch.is_grad_enabled() or residual.requires_grad or delta.requires_grad:
        return residual + gate.index_select(0, adaln_indices) * delta

    seq_len = residual.shape[1]
    if seq_len <= row_budget:
        residual.add_(gate.index_select(0, adaln_indices) * delta)
        return residual

    for s in range(0, seq_len, row_budget):
        e = min(s + row_budget, seq_len)
        residual[:, s:e].add_(gate.index_select(0, adaln_indices[s:e]) * delta[:, s:e])
    return residual


# ---------------------------------------------------------------------------
# Output tail: MiniMaxH3AdaLayerNormOut's norm + shift/scale modulation, writing
# directly into the output heads' dtype to absorb the caller's `.to(...)` cast.
# ---------------------------------------------------------------------------

def _norm_out_apply(
    norm_out: nn.Module,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    timestep_indices: torch.Tensor,
) -> torch.Tensor:
    if norm_out.apply_silu:
        temb = nn.functional.silu(temb)
    shift, scale = norm_out.linear(temb.to(norm_out.linear.weight.dtype)).chunk(2, dim=-1)
    hidden_states = norm_out.norm(hidden_states)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
        0, timestep_indices
    )


def chunked_norm_out(
    norm_out: nn.Module,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    timestep_indices: torch.Tensor,
    out_dtype: torch.dtype,
    row_budget: int = FF_CHUNK_ROW_BUDGET,
) -> torch.Tensor:
    """``MiniMaxH3AdaLayerNormOut``'s body, chunked over the sequence axis and writing straight
    into ``out_dtype`` (the output heads' parameter dtype) to absorb the cast the caller used to
    apply afterwards (``.to(self.proj_out.weight.dtype)``) -- see both call sites in
    ``transformer_minimax_h3.py`` and ``minimax_h3_block_loop_wrapper.py``, which now pass
    ``out_dtype`` in and no longer cast the result themselves. Every branch here (grad fallback,
    short-sequence short-circuit, and the chunked loop) returns a tensor already at ``out_dtype``,
    so a caller-side ``.to(out_dtype)`` on top of this function's result would be a genuine no-op
    (same dtype -> ``Tensor.to`` returns the same object, no copy) rather than a safety net that
    is expected to fire.
    """
    if torch.is_grad_enabled() or hidden_states.requires_grad or temb.requires_grad:
        return _norm_out_apply(norm_out, hidden_states, temb, timestep_indices).to(out_dtype)

    seq_len = hidden_states.shape[1]
    if seq_len <= row_budget:
        return _norm_out_apply(norm_out, hidden_states, temb, timestep_indices).to(out_dtype)

    if norm_out.apply_silu:
        temb = nn.functional.silu(temb)
    shift, scale = norm_out.linear(temb.to(norm_out.linear.weight.dtype)).chunk(2, dim=-1)
    out = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=out_dtype)
    for s in range(0, seq_len, row_budget):
        e = min(s + row_budget, seq_len)
        idx = timestep_indices[s:e]
        chunk = norm_out.norm(hidden_states[:, s:e])
        chunk = chunk * (1.0 + scale.index_select(0, idx)) + shift.index_select(0, idx)
        out[:, s:e] = chunk.to(out_dtype)
    return out
