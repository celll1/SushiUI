"""Regression tests for MiniMax-H3's three activation-memory reductions:

  * ``core.models.minimax_h3.rope_inplace.apply_rotary_emb`` -- A3, in-place RoPE.
  * ``core.models.minimax_h3.adaln_chunking.chunked_ada_modulate`` /
    ``.gated_residual_add`` -- A2, chunked block-level AdaLN modulation and an
    in-place gated residual add.
  * ``core.models.minimax_h3.adaln_chunking.chunked_norm_out`` -- A1, the
    chunked output-tail norm that also absorbs the caller's ``.to(out_dtype)``
    cast.

All three are inference-only (guarded on ``torch.is_grad_enabled()`` /
``.requires_grad``, exactly like ``core.models.minimax_h3.ff_chunking
.chunked_feed_forward``) and, for the chunked ones, short-circuit whenever the
packed sequence is no longer than the row budget. This suite tests the guard,
the short-circuit, and bit-exactness against the stock expression with tiny
CPU modules across dtype/shape combinations that are NOT multiples of the
budget -- it does not load the real ~12 GiB checkpoint (see
``scratchpad/verify_shipped.py`` for the production-shape, real-checkpoint,
GPU bit-exactness run this suite's tiny-module tests are modelled on: bit-exact,
``maxdiff == 0.0``, at ``S = 97,159`` for both ``blocks_to_swap in (0, 40)``).

Unlike ``ff_chunking``'s SwiGLU chunking, none of the three ops here calls a
Linear over a chunked sequence axis in a way that could introduce GEMM
reduction-order noise (``adaln_proj.linear`` / ``norm_out.linear`` are always
called once, over the small un-chunked modulation table, before any chunk
loop begins) -- every chunked/in-place op below is elementwise or an
``index_select`` gather. The exactness assertions are therefore
``torch.equal``, not ``torch.allclose``.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(REPO_ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.models.minimax_h3.adaln_chunking import (  # noqa: E402
    _ada_modulate_apply,
    _norm_out_apply,
    chunked_ada_modulate,
    chunked_norm_out,
    chunked_norm_out_proj_fused,
    gated_residual_add,
)
from core.models.minimax_h3.rope_inplace import (  # noqa: E402
    _apply_rotary_emb_stock,
    apply_rotary_emb,
)

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# (seq_len, row_budget): not-multiples-of-the-budget, seq_len == 1, and a budget
# much smaller than seq_len -- same spirit as ff_chunking_test.py's _SHAPES.
_SHAPES = [
    (100, 24),
    (97, 32),
    (1, 8),
    (5, 2),
]


def _make_norm(hidden, dtype):
    torch.manual_seed(0)
    return nn.RMSNorm(hidden, eps=1e-5).to(dtype)


# ---------------------------------------------------------------------------
# A2a: chunked_ada_modulate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("seq_len,row_budget", _SHAPES)
def test_chunked_ada_modulate_matches_stock(dtype, seq_len, row_budget):
    hidden = 12
    num_rows = 9
    norm = _make_norm(hidden, dtype)
    torch.manual_seed(1)
    hidden_states = torch.randn(2, seq_len, hidden, dtype=dtype)
    scale = torch.randn(num_rows, hidden, dtype=dtype)
    shift = torch.randn(num_rows, hidden, dtype=dtype)
    adaln_indices = torch.randint(0, num_rows, (seq_len,))

    with torch.no_grad():
        stock = _ada_modulate_apply(norm, hidden_states, scale, shift, adaln_indices)
        chunked = chunked_ada_modulate(norm, hidden_states, scale, shift, adaln_indices, row_budget=row_budget)

    assert chunked.shape == stock.shape
    assert torch.equal(chunked, stock)


def test_ada_modulate_is_skipped_under_autograd():
    hidden = 12
    norm = _make_norm(hidden, torch.float32)
    torch.manual_seed(2)
    hidden_states = torch.randn(1, 40, hidden, dtype=torch.float32, requires_grad=True)
    scale = torch.randn(9, hidden)
    shift = torch.randn(9, hidden)
    adaln_indices = torch.randint(0, 9, (40,))

    calls = {"n": 0}
    real_forward = norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm.forward = counting_forward
    try:
        chunked_ada_modulate(norm, hidden_states, scale, shift, adaln_indices, row_budget=8)
    finally:
        norm.forward = real_forward

    # Grad-enabled: 1 call (unchunked), not 5 (40 / 8).
    assert calls["n"] == 1


def test_ada_modulate_chunks_under_no_grad_for_long_sequences():
    hidden = 12
    norm = _make_norm(hidden, torch.float32)
    torch.manual_seed(3)
    hidden_states = torch.randn(1, 40, hidden, dtype=torch.float32)
    scale = torch.randn(9, hidden)
    shift = torch.randn(9, hidden)
    adaln_indices = torch.randint(0, 9, (40,))

    calls = {"n": 0}
    real_forward = norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm.forward = counting_forward
    try:
        with torch.no_grad():
            chunked_ada_modulate(norm, hidden_states, scale, shift, adaln_indices, row_budget=8)
    finally:
        norm.forward = real_forward

    assert calls["n"] == 5


def test_ada_modulate_short_sequence_takes_zero_overhead_path():
    hidden = 12
    norm = _make_norm(hidden, torch.float32)
    torch.manual_seed(4)
    hidden_states = torch.randn(1, 10, hidden, dtype=torch.float32)
    scale = torch.randn(9, hidden)
    shift = torch.randn(9, hidden)
    adaln_indices = torch.randint(0, 9, (10,))

    calls = {"n": 0}
    real_forward = norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm.forward = counting_forward
    try:
        with torch.no_grad():
            chunked_ada_modulate(norm, hidden_states, scale, shift, adaln_indices, row_budget=8192)
    finally:
        norm.forward = real_forward

    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# A2b: gated_residual_add
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("seq_len,row_budget", _SHAPES)
def test_gated_residual_add_matches_stock(dtype, seq_len, row_budget):
    hidden = 12
    num_rows = 9
    torch.manual_seed(5)
    residual = torch.randn(2, seq_len, hidden, dtype=dtype)
    delta = torch.randn(2, seq_len, hidden, dtype=dtype)
    gate = torch.randn(num_rows, hidden, dtype=dtype)
    adaln_indices = torch.randint(0, num_rows, (seq_len,))

    stock = residual + gate.index_select(0, adaln_indices) * delta
    with torch.no_grad():
        out = gated_residual_add(residual.clone(), gate, adaln_indices, delta.clone(), row_budget=row_budget)

    assert torch.equal(out, stock)


def test_gated_residual_add_is_out_of_place_under_autograd():
    """Under grad, must NOT mutate `residual` -- the stock `residual + ...`
    expression allocates a new tensor. Checked via storage identity (a
    numerics-independent proxy, same spirit as ff_chunking_test.py's call-count
    checks): if the guard were removed, `residual` would be mutated in place
    and its data pointer would be unchanged after the call, silently breaking
    any other code (e.g. FBCache) still holding a reference to it."""
    hidden = 12
    torch.manual_seed(6)
    residual = torch.randn(1, 40, hidden, dtype=torch.float32, requires_grad=True)
    delta = torch.randn(1, 40, hidden, dtype=torch.float32)
    gate = torch.randn(9, hidden)
    adaln_indices = torch.randint(0, 9, (40,))
    ptr_before = residual.data_ptr()

    out = gated_residual_add(residual, gate, adaln_indices, delta, row_budget=8)

    assert out.data_ptr() != ptr_before
    assert residual.data_ptr() == ptr_before  # untouched


def test_gated_residual_add_mutates_in_place_under_no_grad():
    hidden = 12
    torch.manual_seed(7)
    residual = torch.randn(1, 40, hidden, dtype=torch.float32)
    delta = torch.randn(1, 40, hidden, dtype=torch.float32)
    gate = torch.randn(9, hidden)
    adaln_indices = torch.randint(0, 9, (40,))
    ptr_before = residual.data_ptr()

    with torch.no_grad():
        out = gated_residual_add(residual, gate, adaln_indices, delta, row_budget=8)

    assert out.data_ptr() == ptr_before  # in place, chunked (40 / 8 == 5 chunks)


def test_gated_residual_add_mutates_in_place_on_the_short_circuit_path_too():
    """Even below `row_budget` (no chunk loop), the no-grad path is still
    in-place -- there is no `seq_len <= row_budget` short-circuit to the
    OUT-OF-PLACE stock expression here (unlike the modulate/norm-out chunking
    functions), because a single whole-tensor in-place add has no allocation
    to save on being skipped."""
    hidden = 12
    torch.manual_seed(8)
    residual = torch.randn(1, 10, hidden, dtype=torch.float32)
    delta = torch.randn(1, 10, hidden, dtype=torch.float32)
    gate = torch.randn(9, hidden)
    adaln_indices = torch.randint(0, 9, (10,))
    ptr_before = residual.data_ptr()

    with torch.no_grad():
        out = gated_residual_add(residual, gate, adaln_indices, delta, row_budget=8192)

    assert out.data_ptr() == ptr_before


# ---------------------------------------------------------------------------
# A1: chunked_norm_out
# ---------------------------------------------------------------------------

class _FakeNormOut(nn.Module):
    def __init__(self, hidden, time_embed_dim, dtype, apply_silu=False):
        super().__init__()
        self.apply_silu = apply_silu
        self.norm = nn.RMSNorm(hidden, eps=1e-5).to(dtype)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden).to(dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seq_len,row_budget", _SHAPES)
def test_chunked_norm_out_matches_stock(dtype, out_dtype, seq_len, row_budget):
    hidden, time_embed_dim, num_timesteps = 12, 6, 9
    norm_out = _FakeNormOut(hidden, time_embed_dim, dtype)
    torch.manual_seed(9)
    hidden_states = torch.randn(2, seq_len, hidden, dtype=dtype)
    temb = torch.randn(num_timesteps, time_embed_dim, dtype=dtype)
    timestep_indices = torch.randint(0, num_timesteps, (seq_len,))

    with torch.no_grad():
        stock = _norm_out_apply(norm_out, hidden_states, temb, timestep_indices).to(out_dtype)
        chunked = chunked_norm_out(norm_out, hidden_states, temb, timestep_indices, out_dtype, row_budget=row_budget)

    assert chunked.dtype == out_dtype
    assert torch.equal(chunked, stock)


def test_norm_out_is_skipped_under_autograd():
    hidden, time_embed_dim = 12, 6
    norm_out = _FakeNormOut(hidden, time_embed_dim, torch.float32)
    torch.manual_seed(10)
    hidden_states = torch.randn(1, 40, hidden, dtype=torch.float32, requires_grad=True)
    temb = torch.randn(9, time_embed_dim)
    timestep_indices = torch.randint(0, 9, (40,))

    calls = {"n": 0}
    real_forward = norm_out.norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm_out.norm.forward = counting_forward
    try:
        chunked_norm_out(norm_out, hidden_states, temb, timestep_indices, torch.float32, row_budget=8)
    finally:
        norm_out.norm.forward = real_forward

    assert calls["n"] == 1


def test_norm_out_chunks_under_no_grad_for_long_sequences():
    hidden, time_embed_dim = 12, 6
    norm_out = _FakeNormOut(hidden, time_embed_dim, torch.float32)
    torch.manual_seed(11)
    hidden_states = torch.randn(1, 40, hidden, dtype=torch.float32)
    temb = torch.randn(9, time_embed_dim)
    timestep_indices = torch.randint(0, 9, (40,))

    calls = {"n": 0}
    real_forward = norm_out.norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm_out.norm.forward = counting_forward
    try:
        with torch.no_grad():
            chunked_norm_out(norm_out, hidden_states, temb, timestep_indices, torch.float32, row_budget=8)
    finally:
        norm_out.norm.forward = real_forward

    assert calls["n"] == 5


def test_norm_out_short_sequence_takes_zero_overhead_path():
    hidden, time_embed_dim = 12, 6
    norm_out = _FakeNormOut(hidden, time_embed_dim, torch.float32)
    torch.manual_seed(12)
    hidden_states = torch.randn(1, 10, hidden, dtype=torch.float32)
    temb = torch.randn(9, time_embed_dim)
    timestep_indices = torch.randint(0, 9, (10,))

    calls = {"n": 0}
    real_forward = norm_out.norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm_out.norm.forward = counting_forward
    try:
        with torch.no_grad():
            chunked_norm_out(norm_out, hidden_states, temb, timestep_indices, torch.float32, row_budget=8192)
    finally:
        norm_out.norm.forward = real_forward

    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# AP3: chunked_norm_out_proj_fused (opt-in output-tail head fusion). Unlike
# chunked_norm_out, this is NOT bit-exact with the unfused path (folding
# proj_out/audio_proj_out into the chunk loop changes each GEMM's row count
# per call), so the value-matching test below uses torch.allclose, not
# torch.equal -- see adaln_chunking.py's "Head fusion" note for the measured
# production-shape deviation this tolerance is set well above.
# ---------------------------------------------------------------------------

def _unfused_heads(norm_out, hidden_states, temb, timestep_indices, proj_out, audio_proj_out):
    normed = chunked_norm_out(norm_out, hidden_states, temb, timestep_indices, proj_out.weight.dtype)
    return proj_out(normed), audio_proj_out(normed)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("seq_len,row_budget", _SHAPES)
def test_fused_output_heads_close_to_unfused(dtype, seq_len, row_budget):
    hidden, time_embed_dim, num_timesteps = 12, 6, 9
    video_dim, audio_dim = 4, 3
    norm_out = _FakeNormOut(hidden, time_embed_dim, dtype)
    proj_out = nn.Linear(hidden, video_dim).to(dtype)
    audio_proj_out = nn.Linear(hidden, audio_dim).to(dtype)
    torch.manual_seed(20)
    hidden_states = torch.randn(2, seq_len, hidden, dtype=dtype)
    temb = torch.randn(num_timesteps, time_embed_dim, dtype=dtype)
    timestep_indices = torch.randint(0, num_timesteps, (seq_len,))

    with torch.no_grad():
        video_stock, audio_stock = _unfused_heads(
            norm_out, hidden_states, temb, timestep_indices, proj_out, audio_proj_out)
        video_fused, audio_fused = chunked_norm_out_proj_fused(
            norm_out, hidden_states, temb, timestep_indices, proj_out, audio_proj_out, row_budget=row_budget)

    assert video_fused.shape == video_stock.shape
    assert audio_fused.shape == audio_stock.shape
    assert video_fused.dtype == proj_out.weight.dtype
    assert torch.allclose(video_fused, video_stock, atol=1e-3, rtol=1e-3)
    assert torch.allclose(audio_fused, audio_stock, atol=1e-3, rtol=1e-3)


def test_fuse_output_proj_is_skipped_under_autograd():
    hidden, time_embed_dim = 12, 6
    norm_out = _FakeNormOut(hidden, time_embed_dim, torch.float32)
    proj_out = nn.Linear(hidden, 4)
    audio_proj_out = nn.Linear(hidden, 3)
    torch.manual_seed(21)
    hidden_states = torch.randn(1, 40, hidden, dtype=torch.float32, requires_grad=True)
    temb = torch.randn(9, time_embed_dim)
    timestep_indices = torch.randint(0, 9, (40,))

    calls = {"n": 0}
    real_forward = norm_out.norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm_out.norm.forward = counting_forward
    try:
        chunked_norm_out_proj_fused(
            norm_out, hidden_states, temb, timestep_indices, proj_out, audio_proj_out, row_budget=8)
    finally:
        norm_out.norm.forward = real_forward

    # Grad-enabled: 1 call (the unfused `_norm_out_apply` fallback), not 5 (40 / 8).
    assert calls["n"] == 1


def test_fuse_output_proj_chunks_under_no_grad_for_long_sequences():
    hidden, time_embed_dim = 12, 6
    norm_out = _FakeNormOut(hidden, time_embed_dim, torch.float32)
    proj_out = nn.Linear(hidden, 4)
    audio_proj_out = nn.Linear(hidden, 3)
    torch.manual_seed(22)
    hidden_states = torch.randn(1, 40, hidden, dtype=torch.float32)
    temb = torch.randn(9, time_embed_dim)
    timestep_indices = torch.randint(0, 9, (40,))

    calls = {"n": 0}
    real_forward = norm_out.norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm_out.norm.forward = counting_forward
    try:
        with torch.no_grad():
            chunked_norm_out_proj_fused(
                norm_out, hidden_states, temb, timestep_indices, proj_out, audio_proj_out, row_budget=8)
    finally:
        norm_out.norm.forward = real_forward

    assert calls["n"] == 5


def test_fuse_output_proj_short_sequence_takes_zero_overhead_path():
    hidden, time_embed_dim = 12, 6
    norm_out = _FakeNormOut(hidden, time_embed_dim, torch.float32)
    proj_out = nn.Linear(hidden, 4)
    audio_proj_out = nn.Linear(hidden, 3)
    torch.manual_seed(23)
    hidden_states = torch.randn(1, 10, hidden, dtype=torch.float32)
    temb = torch.randn(9, time_embed_dim)
    timestep_indices = torch.randint(0, 9, (10,))

    calls = {"n": 0}
    real_forward = norm_out.norm.forward

    def counting_forward(x):
        calls["n"] += 1
        return real_forward(x)

    norm_out.norm.forward = counting_forward
    try:
        with torch.no_grad():
            chunked_norm_out_proj_fused(
                norm_out, hidden_states, temb, timestep_indices, proj_out, audio_proj_out, row_budget=8192)
    finally:
        norm_out.norm.forward = real_forward

    assert calls["n"] == 1


def test_fuse_output_proj_flag_defaults_off_on_the_real_model():
    """The opt-in flag itself: `MiniMaxH3Transformer3DModel.fuse_output_proj`
    must default False, so a model built with no explicit setting takes the
    unfused (bit-exact-with-history) path."""
    model = _build_tiny_minimax_h3_model()
    assert model.fuse_output_proj is False


# ---------------------------------------------------------------------------
# A3: apply_rotary_emb (in-place RoPE)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("seq_len", [1, 5, 97, 100])
def test_apply_rotary_emb_matches_stock(dtype, seq_len):
    heads, head_dim, rotary_dim = 3, 16, 12  # rotary_dim < head_dim: pass-through channels present
    torch.manual_seed(13)
    hidden_states = torch.randn(2, seq_len, heads, head_dim, dtype=dtype)
    cos = torch.randn(seq_len, rotary_dim).cos()
    sin = torch.randn(seq_len, rotary_dim).sin()

    stock = _apply_rotary_emb_stock(hidden_states.clone(), cos, sin)
    with torch.no_grad():
        shipped = apply_rotary_emb(hidden_states.clone(), cos, sin)

    assert torch.equal(shipped, stock)


def test_apply_rotary_emb_is_out_of_place_under_autograd():
    heads, head_dim, rotary_dim, seq_len = 3, 16, 12, 40
    torch.manual_seed(14)
    hidden_states = torch.randn(1, seq_len, heads, head_dim, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(seq_len, rotary_dim).cos()
    sin = torch.randn(seq_len, rotary_dim).sin()
    ptr_before = hidden_states.data_ptr()

    out = apply_rotary_emb(hidden_states, cos, sin)

    assert out.data_ptr() != ptr_before


def test_apply_rotary_emb_is_in_place_under_no_grad():
    heads, head_dim, rotary_dim, seq_len = 3, 16, 12, 40
    torch.manual_seed(15)
    hidden_states = torch.randn(1, seq_len, heads, head_dim, dtype=torch.float32)
    cos = torch.randn(seq_len, rotary_dim).cos()
    sin = torch.randn(seq_len, rotary_dim).sin()
    ptr_before = hidden_states.data_ptr()

    with torch.no_grad():
        out = apply_rotary_emb(hidden_states, cos, sin)

    assert out.data_ptr() == ptr_before


def test_apply_rotary_emb_requires_grad_input_skips_inplace_even_under_no_grad_context():
    """Same belt-and-suspenders case as ff_chunking_test.py's analogous test: a
    leaf tensor's `requires_grad` survives entry into a `torch.no_grad()` block."""
    heads, head_dim, rotary_dim, seq_len = 3, 16, 12, 40
    torch.manual_seed(16)
    hidden_states = torch.randn(1, seq_len, heads, head_dim, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(seq_len, rotary_dim).cos()
    sin = torch.randn(seq_len, rotary_dim).sin()
    ptr_before = hidden_states.data_ptr()

    with torch.no_grad():
        assert torch.is_grad_enabled() is False
        assert hidden_states.requires_grad is True
        out = apply_rotary_emb(hidden_states, cos, sin)

    assert out.data_ptr() != ptr_before


# ---------------------------------------------------------------------------
# FBCache-aliasing regression (A2's hazard, fixed at the wrapper, not in
# adaln_chunking.py itself -- exercised here through the real vendor classes).
# ---------------------------------------------------------------------------

class _FakeFBCache:
    """Duck-typed double for the FBCache the wrapper expects: `use_cache` /
    `get` / `store`. Always misses (forces the `store` branch), which is the
    branch that reads `hidden_states - original_hidden_states` -- exactly the
    aliasing hazard site."""

    def __init__(self):
        self.stored = None

    def use_cache(self, video_residual, step, guard_indicator=None):
        return False

    def get(self):  # pragma: no cover - never reached, use_cache always misses
        raise AssertionError("get() should not be called when use_cache() returns False")

    def store(self, delta):
        self.stored = delta.clone()


def _build_tiny_minimax_h3_model():
    from core.models.minimax_h3.vendor.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    torch.manual_seed(17)
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=16,
        rope_freq_dim=2,
        rope_theta=10000.0,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
        adaln_curve_grid=None,
    )
    model.eval()
    return model


def _tiny_packed_layout():
    # 8 rows: 2 text, 4 video, 2 audio. token_tags: 0=video, 1=text, 2=audio.
    text_indices = torch.tensor([0, 1])
    video_indices = torch.tensor([2, 3, 4, 5])
    audio_indices = torch.tensor([6, 7])
    token_tags = torch.tensor([1, 1, 0, 0, 0, 0, 2, 2])
    timestep_indices = torch.zeros(8, dtype=torch.long)
    position_ids = torch.zeros(8, 3, dtype=torch.long)
    return dict(
        text_indices=text_indices,
        video_indices=video_indices,
        audio_indices=audio_indices,
        token_tags=token_tags,
        timestep_indices=timestep_indices,
        position_ids=position_ids,
    )


def test_fbcache_original_hidden_states_survives_block_zeros_in_place_gated_add():
    """The regression this guards: `MiniMaxH3TransformerBlock.forward`'s gated
    residual add mutates its `residual` argument in place at inference, and for
    block 0 that argument IS the tensor `MiniMaxH3BlockLoopWrapper._custom_forward`
    keeps as `original_hidden_states`. Without the wrapper's clone-before-the-loop
    fix, `first_residual = hidden_states - original_hidden_states` (and, on the
    store branch reached here, `hidden_states - original_hidden_states` again)
    would be exactly zero -- both sides the same, now-identical object -- for a
    single-block model. With the fix, the stored delta is the real, non-trivial
    attention + feed-forward contribution.
    """
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper

    model = _build_tiny_minimax_h3_model()
    wrapper = MiniMaxH3BlockLoopWrapper(model)
    fbcache = _FakeFBCache()
    wrapper.attach_fbcache(fbcache, rows_per_frame=4, condition_video_rows=0)

    layout = _tiny_packed_layout()
    torch.manual_seed(18)
    kw = dict(
        hidden_states=torch.randn(1, 4, 2),  # video_patch_dim == in_channels == 2
        audio_hidden_states=torch.randn(1, 2, 2),
        encoder_hidden_states=torch.randn(1, 2, 8),
        timestep=torch.tensor([0.5]),
        return_dict=False,
        **layout,
    )

    with torch.no_grad():
        wrapper(**kw)

    assert fbcache.stored is not None
    # The regression signature: a broken (unfixed) aliasing bug makes this
    # EXACTLY all-zeros (not merely close to it) because both operands of the
    # subtraction would be the identical mutated object.
    assert not torch.equal(fbcache.stored, torch.zeros_like(fbcache.stored))
    assert fbcache.stored.abs().max().item() > 1e-6


def test_fbcache_store_delta_matches_an_independently_computed_reference():
    """Stronger version of the above: recompute the expected pre-/post-block-0
    stream independently (by driving the SAME model's stage-1 projections and
    block-0 call manually, from a cloned copy of the input) and check the
    wrapper's stored FBCache delta against it exactly -- not just "nonzero"."""
    from core.models.minimax_h3.vendor.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper

    model = _build_tiny_minimax_h3_model()
    wrapper = MiniMaxH3BlockLoopWrapper(model)
    fbcache = _FakeFBCache()
    wrapper.attach_fbcache(fbcache, rows_per_frame=4, condition_video_rows=0)

    layout = _tiny_packed_layout()
    torch.manual_seed(19)
    video_in = torch.randn(1, 4, 2)
    audio_in = torch.randn(1, 2, 2)
    text_in = torch.randn(1, 2, 8)
    timestep = torch.tensor([0.5])

    with torch.no_grad():
        wrapper(
            hidden_states=video_in, audio_hidden_states=audio_in, encoder_hidden_states=text_in,
            timestep=timestep, return_dict=False, **layout,
        )

        # Independent reference: replicate stage 1 + block 0 manually.
        video_embeds = model.proj_in(video_in.to(model.proj_in.weight.dtype))
        audio_embeds = model.audio_proj_in(audio_in.to(model.audio_proj_in.weight.dtype))
        text_embeds = model.context_embedder(text_in.to(model.context_embedder.weight.dtype))
        text_embeds = model.token_refiner(text_embeds)
        pre_block = text_embeds.new_zeros((1, 8, text_embeds.shape[-1]))
        pre_block = pre_block.index_copy(1, layout["text_indices"], text_embeds)
        pre_block = pre_block.index_copy(1, layout["video_indices"], video_embeds.to(text_embeds.dtype))
        pre_block = pre_block.index_copy(1, layout["audio_indices"], audio_embeds.to(text_embeds.dtype))
        pre_block_reference = pre_block.clone()

        temb = model.time_embedder(model.time_proj(timestep).to(model.time_embedder.linear_1.weight.dtype))
        adaln_indices = layout["timestep_indices"] * MINIMAX_H3_MODALITY_NUM + layout["token_tags"]
        rotary_emb = model.rope(layout["position_ids"])
        post_block_reference = model.transformer_blocks[0](pre_block.clone(), temb, adaln_indices, rotary_emb)

    expected_delta = post_block_reference - pre_block_reference
    assert torch.equal(fbcache.stored, expected_delta)
