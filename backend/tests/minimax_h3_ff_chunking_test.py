"""Regression tests for MiniMax-H3's sequence-chunked feed-forward
(``core.models.minimax_h3.ff_chunking.chunked_feed_forward``), which replaces
``ff_output = self.ff(norm_hidden_states)`` at the block's feed-forward call site
(``core/models/minimax_h3/vendor/transformer_minimax_h3.py``) to shrink the
single largest activation in the model (the SwiGLU projection's
``[B, S, 2 * ffn_dim]`` tensor) by splitting the call over the packed sequence
axis.

NUMERIC CLAIM, MEASURED NOT ASSUMED: SwiGLU has no cross-token operation, so in
REAL-NUMBER arithmetic chunking the sequence axis is an exact identity. In
FLOATING-POINT arithmetic this held (``torch.equal``, exact) for every shape
tested on the CPU reference BLAS path and for two of the three real released
quantized checkpoints' GEMM kernels (``w4a8_mixed``, ``int8_convrot``) at
production scale on an RTX 6000 Ada -- but NOT universally: the third
checkpoint's ``Fp8Linear`` "dequant to bf16, then plain GEMM" path, and a bare
``nn.Linear`` isolated from everything else (no quantization, no SwiGLU, no
LoRA), both showed sub-percent, ULP-scale deviations once chunking actually
activates on GPU, because cuBLAS is free to pick a different tiling / reduction
order for a GEMM depending on its row count -- a documented property of
batched matrix multiplication, not a bug in this call-site patch (see
``core/models/minimax_h3/ff_chunking.py``'s module docstring for the full
measurement). This suite therefore checks NUMERIC closeness with a tolerance
sized to that noise floor, not blind ``torch.equal`` -- an ``allclose`` that
would also catch anything shape/logic-broken (which would show up as an
enormous, not a ULP-scale, deviation) instead of asserting something this
codebase measured to be false on GPU.

The tests do not depend on GPU or the real 50-block MiniMax-H3 checkpoint: a
tiny ``FeedForward(activation_fn="swiglu")`` reproduces the exact call-site
shape (the real ``self.ff`` in ``MiniMaxH3TransformerBlock``) at a size this
suite can run on CPU in a few milliseconds. The real-checkpoint, production-shape
measurement (bit-exact for ``w4a8_mixed``/``int8_convrot``, ULP-scale deviation
for ``fp8_scaled``, VRAM 11.996 -> 5.188 GiB reserved) was run as a one-off
probe against the real model files and is reported in the module docstring and
task writeup -- it is not repeated here because it needs a ~20 GiB checkpoint
this test suite must not depend on to run anywhere.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(REPO_ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from diffusers.models.attention import FeedForward  # noqa: E402

from core.models.minimax_h3.ff_chunking import chunked_feed_forward  # noqa: E402


def _make_ff(dtype: torch.dtype) -> FeedForward:
    torch.manual_seed(0)
    ff = FeedForward(dim=32, inner_dim=48, activation_fn="swiglu", bias=False)
    return ff.to(dtype=dtype)


# Tolerance sized to the measured noise floor per dtype (see module docstring above):
# fp32's cuBLAS/MKL reduction-order noise measured ~1e-7 absolute; fp16 was bit-exact in
# every configuration tested but is given slack rather than asserted exact, since that
# exactness is an emergent property of the BLAS build, not a code guarantee; bf16's
# reduction-order noise measured up to ~1 ULP (~0.0078 at unit magnitude).
_TOLERANCE = {
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float16: dict(rtol=5e-3, atol=5e-3),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# (seq_len, row_budget): includes shapes that are NOT multiples of the chunk
# count, S == 1, and a budget smaller than S (forcing more than one chunk, with
# a final short chunk).
_SHAPES = [
    (100, 24),   # 100 / 24 -> 5 chunks, last chunk has 4 rows (not a clean multiple)
    (97, 32),    # prime-ish length against a budget that does not divide it
    (1, 8),      # S == 1: budget larger than S, degrades to the unchunked call
    (5, 2),      # budget much smaller than S
]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("seq_len,row_budget", _SHAPES)
def test_chunked_matches_unchunked_within_gemm_noise_floor(dtype, seq_len, row_budget):
    ff = _make_ff(dtype)
    torch.manual_seed(1)
    hidden_states = torch.randn(2, seq_len, 32, dtype=dtype)  # B == 2: do not assume B == 1

    with torch.no_grad():
        unchunked = ff(hidden_states)
        chunked = chunked_feed_forward(ff, hidden_states, row_budget=row_budget)

    assert chunked.shape == unchunked.shape
    assert torch.allclose(chunked, unchunked, **_TOLERANCE[dtype]), (
        f"chunked_feed_forward diverged from the unchunked call beyond the measured GEMM "
        f"reduction-order noise floor (dtype={dtype}, seq_len={seq_len}, row_budget={row_budget}), "
        f"max abs diff={ (chunked.float() - unchunked.float()).abs().max().item() }"
    )


def test_chunk_count_can_exceed_a_naive_expectation_and_still_match():
    """row_budget=1 forces one chunk per row (chunk count == seq_len, the most
    chunks this function can ever produce since `split` never emits an empty
    chunk) -- still must stay within the GEMM noise floor."""
    ff = _make_ff(torch.float32)
    torch.manual_seed(2)
    hidden_states = torch.randn(1, 7, 32, dtype=torch.float32)

    with torch.no_grad():
        unchunked = ff(hidden_states)
        chunked = chunked_feed_forward(ff, hidden_states, row_budget=1)

    assert torch.allclose(chunked, unchunked, **_TOLERANCE[torch.float32])


# ---------------------------------------------------------------------------
# The guard that matters: chunking is inference-only. These assertions are
# exact call-count checks on Python control flow, not numerics -- they fail
# deterministically if the grad-mode guard in chunked_feed_forward is ever
# removed or loosened, independent of any floating-point noise question above.
# ---------------------------------------------------------------------------

def test_chunking_is_skipped_under_autograd():
    """Under torch.is_grad_enabled(), chunked_feed_forward must take the plain,
    unchunked path -- chunking the sequence axis under autograd SAVES MORE
    activations for backward (one graph node per chunk), not fewer, so it must
    never engage there. This test is written against a proxy (call count) that
    DOES break if the guard is removed: with the guard gone, the grad-enabled
    call below would still produce numerically-close output (see the tolerance
    tests above) but would silently take on backward's higher memory
    footprint -- exactly the regression the guard exists to prevent, and
    exactly the kind of change a numeric-only test could not catch.
    """
    ff = _make_ff(torch.float32)
    torch.manual_seed(3)
    hidden_states = torch.randn(1, 40, 32, dtype=torch.float32, requires_grad=True)

    calls = {"n": 0}
    real_forward = ff.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    ff.forward = counting_forward
    try:
        chunked_feed_forward(ff, hidden_states, row_budget=8)
    finally:
        ff.forward = real_forward

    # Grad-enabled: the guard must take the single, unchunked call (1 call to
    # `ff`), not the 5-chunk split (`40 / 8 == 5` calls) it would take under
    # torch.no_grad(). If the guard is removed, this becomes 5 and the test fails.
    assert calls["n"] == 1


def test_chunking_engages_under_no_grad_for_long_sequences():
    ff = _make_ff(torch.float32)
    torch.manual_seed(4)
    hidden_states = torch.randn(1, 40, 32, dtype=torch.float32)

    calls = {"n": 0}
    real_forward = ff.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    ff.forward = counting_forward
    try:
        with torch.no_grad():
            chunked_feed_forward(ff, hidden_states, row_budget=8)
    finally:
        ff.forward = real_forward

    assert calls["n"] == 5  # 40 / 8 rows-per-chunk


def test_short_sequence_takes_the_zero_overhead_unchunked_path():
    ff = _make_ff(torch.float32)
    torch.manual_seed(5)
    hidden_states = torch.randn(1, 10, 32, dtype=torch.float32)

    calls = {"n": 0}
    real_forward = ff.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    ff.forward = counting_forward
    try:
        with torch.no_grad():
            chunked_feed_forward(ff, hidden_states, row_budget=8192)
    finally:
        ff.forward = real_forward

    assert calls["n"] == 1  # seq_len (10) <= row_budget (8192): plain call, no split/cat


def test_requires_grad_input_skips_chunking_even_under_no_grad_context():
    """`hidden_states.requires_grad` is checked independently of
    `torch.is_grad_enabled()` (belt-and-suspenders: a caller could construct a
    grad-tracking tensor and then evaluate it under `torch.no_grad()`, e.g. a
    higher-order-gradient or double-backward context where the outer scope
    disables grad tracking for NEW ops but the input tensor itself still
    carries a grad_fn from outside). This must still take the unchunked path.
    """
    ff = _make_ff(torch.float32)
    torch.manual_seed(6)
    # A leaf tensor's `requires_grad` is set at creation and is unaffected by
    # `torch.no_grad()` scopes entered afterwards -- only new ops built inside
    # such a scope stop being tracked. So this reproduces the "requires_grad
    # input, no_grad context" combination for real.
    hidden_states = torch.randn(1, 40, 32, dtype=torch.float32, requires_grad=True)

    calls = {"n": 0}
    real_forward = ff.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    ff.forward = counting_forward
    try:
        with torch.no_grad():
            # torch.is_grad_enabled() is False here, but hidden_states.requires_grad is True.
            assert torch.is_grad_enabled() is False
            assert hidden_states.requires_grad is True
            chunked_feed_forward(ff, hidden_states, row_budget=8)
    finally:
        ff.forward = real_forward

    assert calls["n"] == 1
