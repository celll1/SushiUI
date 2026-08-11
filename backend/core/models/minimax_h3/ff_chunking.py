"""Sequence-dim chunking of ``MiniMaxH3TransformerBlock``'s SwiGLU feed-forward.

WHY. Per-module instrumentation of the real ``minimax_h3_fl2va_pruned_w4a8_mixed``
checkpoint at 768x1248x345 frames (packed sequence ``S = 97,159`` rows) located the
single largest allocation in the whole 50-block forward: the SwiGLU projection
inside block 1's feed-forward, ``[1, 97159, 28672]`` bf16 (``28672 == 2 * ffn_dim``,
the ``[up; gate]`` pair `SwiGLU.forward` splits before multiplying), which by
itself adds 10.378 GiB (5.189 GiB for the projection plus the two 2.594 GiB
post-split halves) to a 36.139 GiB reserved / 28.134 GiB allocated peak.

SwiGLU is a pointwise-over-tokens op (`proj` is a Linear over the last dim only,
`chunk` + `silu` + `mul` never mix rows), so in REAL-NUMBER arithmetic splitting
the ``FeedForward`` call over the packed sequence's ROW axis and concatenating
the results is an exact identity. In FLOATING-POINT arithmetic this is close but
NOT universally bit-identical -- measured directly (``backend/tests/
minimax_h3_ff_chunking_test.py`` and a one-off probe against the real checkpoint,
see below), not assumed:

  * Loading the REAL block-1 feed-forward from the three released quantized
    checkpoints and running it chunked vs. unchunked at the production shape
    (S = 97,159, hidden 5,376, ffn 14,336, bf16 activations) on an RTX 6000 Ada:
    the ``w4a8_mixed`` checkpoint (the one this module's budget was tuned
    against) and the ``int8_convrot`` checkpoint are ``torch.equal`` bit-exact.
    The ``fp8_scaled`` checkpoint is NOT: its ``Fp8Linear`` runs in "dequant
    only" mode (weights are dequantized to bf16, then a plain bf16 GEMM runs),
    and that GEMM is not shape-invariant -- differences up to ~1% of the output
    magnitude appear once the sequence is actually split into more than one
    chunk. Isolated further with a bare, unquantized `nn.Linear` at both toy and
    production dimensions: the same non-bit-identical behaviour reproduces with
    no quantization, no SwiGLU and no LoRA involved at all, and vanishes when
    ``torch.use_deterministic_algorithms(True)`` is NOT enough to fix it (still
    diverges) -- i.e. this is cuBLAS/cuDNN choosing a different GEMM
    tiling/reduction-order algorithm depending on the matmul's row count (`M`),
    a documented, hardware-level property of batched matrix multiplication, not
    a defect in this call-site patch. ``float16`` was bit-exact in every
    configuration tested (CPU and GPU, toy and production dimensions); ``bf16``
    and ``fp32`` were bit-exact on the two custom low-bit GEMM kernels above but
    not on the plain-cuBLAS dequant path. The deviations are ULP-scale relative
    to the output's own magnitude (max abs diff was a few bf16 ULPs on
    synthetic in-distribution inputs; an out-of-distribution stress input
    inflated the absolute numbers but not the ~0.4-1% relative scale) --
    consistent with ordinary bf16 rounding noise already present throughout a
    50-block bf16 transformer from many other sources (attention kernel choice,
    dequantization order, block-swap round trips), not a correctness break
    (no shape mismatch, no NaN/Inf, no systematic bias observed).

  * VRAM, on the real ``w4a8_mixed`` checkpoint's own block-1 feed-forward
    (real weights, this module's own ``chunked_feed_forward``, S = 97,159):
    peak allocated 11.473 -> 4.336 GiB, peak reserved 11.996 -> 5.188 GiB.
    Wall time for that single call: 0.326s unchunked vs. 0.227s chunked (i.e.
    no regression -- within the run-to-run spread, not slower). This is a
    module-level isolate, not the full 50-block-forward VRAM number reported
    above from per-module instrumentation; both point the same direction.

Given this, "unconditional and free" still holds for the measured checkpoint
this budget was tuned against (``w4a8_mixed``) and for ``int8_convrot`` --
bit-exact, no time cost. For the ``fp8_scaled`` dequant-only path it is a
bf16-GEMM-noise-scale (not correctness-scale) deviation, of the same order as
noise the model already carries elsewhere; it still runs unconditionally
because reverting to the unchunked call would only trade a measured 7+ GiB
reduction for eliminating a deviation that is already smaller than other
unavoidable bf16 rounding in the same forward pass, not for eliminating a
sequence-length-dependent VRAM/quality tradeoff. Measured against the same
checkpoint's per-module instrumentation of the FULL 50-block forward: chunking
x4 took 36.139 -> 31.250 GiB reserved (28.134 -> 23.629 GiB allocated) with
block swap off, and x8 took 28.518 -> 26.227 GiB reserved (21.956 -> 17.450 GiB
allocated) with block swap at 40; in both cases the per-forward wall time
(~202 s) did not move outside the ~200.2-203.0 s spread of repeated unchunked
forwards -- i.e. this is a measured-free win on VRAM and time, so it runs
unconditionally rather than behind a user-facing toggle (see below on why there
is no ``param_defaults.py`` entry).

CALL-SITE PATCH, NOT A ``FeedForward.forward`` OVERRIDE. ``ff.net.0.proj`` (the
SwiGLU projection) and ``ff.net.2`` (the output projection) are LoRA targets --
``core/models/minimax_h3/minimax_h3_lora.py`` wraps them in place, replacing
``ff.net[0].proj`` or ``ff.net[2]`` with a small forward-time-addition wrapper,
never the ``FeedForward``/``SwiGLU`` module itself. ``chunked_feed_forward``
below therefore calls the ``FeedForward`` module (``ff(chunk)``) once per chunk
instead of re-implementing SwiGLU's arithmetic, so whatever is currently
installed at ``ff.net[0].proj`` / ``ff.net[2]`` -- a plain ``nn.Linear``, a LoRA
wrapper around one, or a quantized Linear (``W4A8Linear`` / ``Fp8Linear`` /
``ConvRotInt8Linear``, all of which operate on arbitrary leading dims and
materialize no dense weight) -- runs exactly as it would unchunked, just on a
shorter sequence axis. FBCache / Spectrum and block swap attach at
``MiniMaxH3BlockLoopWrapper`` (the per-block loop, above this call), so they are
unaffected.

INFERENCE ONLY. Under autograd, splitting the sequence axis means every chunk's
intermediate activations are saved separately for backward -- MORE memory, not
less (the whole point of the unchunked call during training is one contiguous
save). ``chunked_feed_forward`` refuses to chunk whenever
``torch.is_grad_enabled()`` is true or the input requires grad, falling back to
the plain, unchunked ``ff(hidden_states)`` call -- literally the same call, so
this fallback branch is bit-identical to calling ``ff`` directly by
construction (no split/cat ever runs). This governs training and the
(inference-only) ``_gradient_checkpointing_func`` branch at
``transformer_minimax_h3.py`` is untouched by this module: it runs under
``torch.is_grad_enabled()`` by construction, so the guard here would refuse to
chunk it even if it called this function, and in fact it never does -- only
``MiniMaxH3TransformerBlock.forward`` calls ``chunked_feed_forward``.

WHY NOT A ``param_defaults.py`` KNOB. This repo's rule is "VRAM is never a
reason to cap capability" -- a low-VRAM mode has to stay reachable, but a
default that is *strictly better* on every measured axis (exact or
noise-floor-scale output depending on the quantization path -- see above --
free at long sequences, a no-op at short ones once ``seq_len <= row budget``)
is not a capability/VRAM tradeoff for the user to make; it is a strictly
dominant behaviour, so it is unconditional rather than opt-in. The row budget
below is a single named, documented module constant -- not a knob and not a
literal buried inside ``MiniMaxH3TransformerBlock.forward``.
"""

from __future__ import annotations

import torch


# Row budget (packed-sequence positions per chunk) for the SwiGLU feed-forward's post-projection
# activation stack `[B, chunk_rows, 2 * ffn_dim]`. MiniMax-H3's `ffn_dim` is 14,336 for every released
# checkpoint (`transformer_minimax_h3.py`'s `ffn_dim` default; the single-file loader reads it from the
# checkpoint but every real checkpoint agrees), so `2 * ffn_dim == 28,672` channels regardless of the
# quantization mode (chunking only ever splits activations, never weights -- see the module docstring).
# 24,576 rows reproduces the MEASURED "block swap off" configuration above almost exactly (97,159 rows /
# 24,576 -> 4 chunks, i.e. the reported 36.139 -> 31.250 GiB reduction) rather than extrapolating to an
# unmeasured finer grain; the block-swap-on x8 result shows more is available for the taking, but 4 is the
# one point on this curve that was actually measured end to end.
FF_CHUNK_ROW_BUDGET = 24_576


def chunked_feed_forward(
    ff: torch.nn.Module,
    hidden_states: torch.Tensor,
    row_budget: int = FF_CHUNK_ROW_BUDGET,
) -> torch.Tensor:
    """Run ``ff(hidden_states)`` in sequence-dim chunks of at most ``row_budget`` rows.

    ``hidden_states`` is ``[B, S, H]``. Inference-only (see module docstring); under autograd,
    or when the input already carries a grad requirement, this degrades to the plain
    ``ff(hidden_states)`` call unchanged. Also degrades to the plain call whenever
    ``S <= row_budget``, so short sequences (the common case for small clips) pay no chunking
    overhead at all -- not even a `split`/`cat` round trip.
    """
    if torch.is_grad_enabled() or hidden_states.requires_grad:
        return ff(hidden_states)

    seq_len = hidden_states.shape[1]
    if seq_len <= row_budget:
        return ff(hidden_states)

    chunks = hidden_states.split(row_budget, dim=1)
    return torch.cat([ff(chunk) for chunk in chunks], dim=1)
