"""``core.models.common.gguf_q8_0_linear`` -- design doc phase 12 ("Q8_0
residency").

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_gguf_q8_0_linear_test.py -v

Everything here runs on tiny SYNTHETIC tensors (no GGUF file, no checkpoint)
except the end-to-end builder test at the bottom, which uses a REAL (not
zero-filled placeholder) Q8_0-encoded tiny GGUF fixture
(``tests.minimax_music3_gguf_fixture.write_tiny_pruned_gguf_q8_0_text_encoder_
and_official_tree`` / ``encode_q8_0_tensor``). The residency NUMBERS (host
RSS, VRAM, the dequant error distribution against the real staged checkpoint)
are one-shot measurements against a multi-GB file this repo does not ship as
a fixture, so they stay in ``tmp/`` and are recorded in
``docs/guides/MINIMAX_MUSIC3_DESIGN.md`` instead -- what belongs here are the
FRAGILE PROPERTIES: the placement invariant, cache invalidation (all three
triggers), the dequant round trip, row-split exactness, and the bias
refusal.
"""

import copy
import os
import pickle
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.common.gguf_container import Q8_0_BLOCK_SIZE  # noqa: E402
from core.models.common.gguf_q8_0_linear import (  # noqa: E402
    GGUFQ8_0Linear,
    dequantize_q8_0,
    install_packed_q8_0_linears,
)
from tests.minimax_music3_gguf_fixture import encode_q8_0_tensor  # noqa: E402


def _random_q8_0_pair(out_features: int, in_features: int, seed: int = 0):
    """A synthetic (codes, scale) pair with REAL Q8_0 rounding (via
    ``encode_q8_0_tensor`` + a matching decode), for tests that need a
    non-trivial, non-hand-crafted quantized weight."""
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(out_features, in_features, generator=generator)
    raw = encode_q8_0_tensor(source)
    blocks_per_row = in_features // Q8_0_BLOCK_SIZE
    n_blocks = out_features * blocks_per_row
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 34)
    scale = torch.from_numpy(arr[:, :2].copy().view(np.float16)).reshape(out_features, blocks_per_row)
    codes = torch.from_numpy(arr[:, 2:].copy().view(np.int8)).reshape(out_features, in_features)
    return codes, scale, source


# ---------------------------------------------------------------------------
# dequantize_q8_0: manual-reference correctness, shape/dtype validation.
# ---------------------------------------------------------------------------

def test_dequantize_q8_0_matches_a_hand_computed_reference():
    # Two rows, two blocks of 32 -- hand-pick codes/scale so the expected
    # dense values are exactly representable in float32, no rounding
    # ambiguity in the test's OWN reference computation.
    out_features, blocks_per_row = 2, 2
    in_features = blocks_per_row * Q8_0_BLOCK_SIZE
    codes = torch.zeros(out_features, in_features, dtype=torch.int8)
    codes[0, 0:32] = 10
    codes[0, 32:64] = -20
    codes[1, 0:32] = 127
    codes[1, 32:64] = -127
    scale = torch.tensor([[0.5, 0.25], [1.0, 2.0]], dtype=torch.float16)

    got = dequantize_q8_0(codes, scale, torch.float32)

    expected = torch.zeros(out_features, in_features, dtype=torch.float32)
    expected[0, 0:32] = 10 * 0.5
    expected[0, 32:64] = -20 * 0.25
    expected[1, 0:32] = 127 * 1.0
    expected[1, 32:64] = -127 * 2.0
    assert torch.equal(got, expected)


def test_dequantize_q8_0_rejects_non_int8_codes():
    codes = torch.zeros(1, 32, dtype=torch.float32)
    scale = torch.zeros(1, 1, dtype=torch.float16)
    with pytest.raises(ValueError, match="int8"):
        dequantize_q8_0(codes, scale, torch.float32)


def test_dequantize_q8_0_rejects_non_float16_scale():
    codes = torch.zeros(1, 32, dtype=torch.int8)
    scale = torch.zeros(1, 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="float16"):
        dequantize_q8_0(codes, scale, torch.float32)


def test_dequantize_q8_0_rejects_a_shape_mismatch():
    codes = torch.zeros(2, 64, dtype=torch.int8)
    scale = torch.zeros(2, 3, dtype=torch.float16)  # should be (2, 2) for in_features=64
    with pytest.raises(ValueError, match="consistent"):
        dequantize_q8_0(codes, scale, torch.float32)


# ---------------------------------------------------------------------------
# Row-split exactness: dequantize-then-split == split-then-dequantize.
# ---------------------------------------------------------------------------

def test_row_split_of_packed_data_matches_split_of_the_dequantized_tensor():
    out_features, in_features = 24, 64  # 2 blocks/row
    codes, scale, _source = _random_q8_0_pair(out_features, in_features, seed=1)

    whole = dequantize_q8_0(codes, scale, torch.float32)

    sizes = [10, 6, 8]  # sums to 24, an uneven (GQA-shaped) split
    code_chunks = torch.split(codes, sizes, dim=0)
    scale_chunks = torch.split(scale, sizes, dim=0)
    whole_chunks = torch.split(whole, sizes, dim=0)

    for c, s, expected in zip(code_chunks, scale_chunks, whole_chunks):
        got = dequantize_q8_0(c.contiguous(), s.contiguous(), torch.float32)
        assert torch.equal(got, expected)


# ---------------------------------------------------------------------------
# GGUFQ8_0Linear: placement invariant, forward correctness.
# ---------------------------------------------------------------------------

def _build_module(out_features=16, in_features=64, compute_dtype=torch.float32, seed=2):
    codes, scale, source = _random_q8_0_pair(out_features, in_features, seed=seed)
    module = GGUFQ8_0Linear(codes, scale, None, compute_dtype)
    return module, source


def test_forward_matches_dequantize_then_linear():
    module, _source = _build_module()
    x = torch.randn(3, module.in_features)
    got = module(x)
    expected = torch.nn.functional.linear(x, dequantize_q8_0(module.qweight, module.qscale, module.compute_dtype))
    assert torch.equal(got, expected)


def test_dtype_only_apply_leaves_packed_buffers_as_int8_and_float16():
    # `.float()`/`.double()` are dtype-only `_apply` calls, runnable without
    # CUDA -- `qweight`/`qscale` must stay int8/float16 regardless (they are
    # never forwarded to `fn` at all), unlike `bias`, which would follow.
    module, _source = _build_module()
    module.double()
    assert module.qweight.dtype is torch.int8
    assert module.qscale.dtype is torch.float16


def test_apply_invalidates_the_cache():
    module, _source = _build_module()
    x = torch.randn(2, module.in_features)
    module(x)
    assert module._dequant_cache is not None
    module.float()  # a dtype-only _apply call; still must drop the cache
    assert module._dequant_cache is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_qweight_and_qscale_never_move_to_cuda_via_to():
    module, _source = _build_module()
    module = module.to("cuda")
    assert module.qweight.device.type == "cpu"
    assert module.qscale.device.type == "cpu"

    x = torch.randn(2, module.in_features, device="cuda")
    y = module(x)
    assert y.device.type == "cuda"
    assert module.qweight.device.type == "cpu"
    assert module.qscale.device.type == "cpu"
    assert module._dequant_cache.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_self_heal_moves_a_stranded_packed_buffer_back_to_cpu():
    # Simulates a mover that bypasses `_apply` entirely (see the module
    # docstring's `diffusers`/`accelerate` note): reassign the buffers
    # directly, exactly the shape `set_module_tensor_to_device` or a raw
    # `buffer.data = buffer.data.to(...)` would leave behind.
    module, _source = _build_module()
    module._buffers["qweight"] = module.qweight.cuda()
    module._buffers["qscale"] = module.qscale.cuda()
    assert module.qweight.device.type == "cuda"

    x = torch.randn(2, module.in_features)  # CPU input this time
    y = module(x)
    assert y.device.type == "cpu"
    assert module.qweight.device.type == "cpu"
    assert module.qscale.device.type == "cpu"


# ---------------------------------------------------------------------------
# F1: load_state_dict must invalidate the cache.
# ---------------------------------------------------------------------------

def test_load_state_dict_invalidates_the_cache():
    module, _source = _build_module(seed=3)
    x = torch.randn(2, module.in_features)
    old_output = module(x)
    assert module._dequant_cache is not None

    new_codes, new_scale, _new_source = _random_q8_0_pair(module.out_features, module.in_features, seed=4)
    assert not torch.equal(new_codes, module.qweight), "test fixture bug: new codes must differ from old"

    module.load_state_dict({"qweight": new_codes, "qscale": new_scale}, strict=False)
    assert torch.equal(module.qweight, new_codes), "qweight actually replaced"

    new_output = module(x)
    assert not torch.equal(new_output, old_output), "forward reflects NEW weights"
    assert not torch.equal(module(x), old_output), "forward does not still equal OLD weights"


# ---------------------------------------------------------------------------
# F6: the dense mirror is cached per (device, dtype), not device only.
# ---------------------------------------------------------------------------

def test_cache_is_keyed_on_dtype_not_only_device():
    module, _source = _build_module(compute_dtype=torch.float32, seed=5)
    x32 = torch.randn(2, module.in_features, dtype=torch.float32)
    x16 = x32.to(torch.float16)

    out32 = module(x32)
    cache32 = module._dequant_cache
    assert cache32.dtype is torch.float32

    out16 = module(x16)
    cache16 = module._dequant_cache
    assert cache16.dtype is torch.float16
    assert cache16 is not cache32

    # Both outputs must be correct for THEIR OWN dtype, not merely present.
    expected32 = torch.nn.functional.linear(x32, dequantize_q8_0(module.qweight, module.qscale, torch.float32))
    expected16 = torch.nn.functional.linear(x16, dequantize_q8_0(module.qweight, module.qscale, torch.float16))
    assert torch.equal(out32, expected32)
    assert torch.equal(out16, expected16)

    # Reusing fp32 again must rebuild (not silently reuse the fp16 mirror).
    out32_again = module(x32)
    assert module._dequant_cache.dtype is torch.float32
    assert torch.equal(out32_again, expected32)


# ---------------------------------------------------------------------------
# F8: the dense mirror must not survive deepcopy / pickle.
# ---------------------------------------------------------------------------

def test_deepcopy_excludes_the_dense_mirror():
    module, _source = _build_module(seed=6)
    x = torch.randn(2, module.in_features)
    module(x)
    assert module._dequant_cache is not None

    copied = copy.deepcopy(module)
    assert copied._dequant_cache is None
    assert module._dequant_cache is not None  # the original is untouched
    # The copy still computes correctly (its buffers were copied too).
    assert torch.equal(copied(x), module(x))


def test_pickle_round_trip_excludes_the_dense_mirror():
    module, _source = _build_module(seed=7)
    x = torch.randn(2, module.in_features)
    module(x)
    assert module._dequant_cache is not None

    restored = pickle.loads(pickle.dumps(module))
    assert restored._dequant_cache is None
    assert torch.equal(restored(x), module(x))


# ---------------------------------------------------------------------------
# install_packed_q8_0_linears: shape/bias validation.
# ---------------------------------------------------------------------------

def test_install_packed_q8_0_linears_replaces_a_linear():
    root = nn.Module()
    root.proj = nn.Linear(64, 16, bias=False)
    codes, scale, _source = _random_q8_0_pair(16, 64, seed=8)

    installed = install_packed_q8_0_linears(root, {"proj.weight": (codes, scale)}, torch.float32)
    assert installed == 1
    assert isinstance(root.proj, GGUFQ8_0Linear)


def test_install_packed_q8_0_linears_refuses_a_biased_linear():
    root = nn.Module()
    root.proj = nn.Linear(64, 16, bias=True)
    codes, scale, _source = _random_q8_0_pair(16, 64, seed=9)

    with pytest.raises(NotImplementedError, match="bias"):
        install_packed_q8_0_linears(root, {"proj.weight": (codes, scale)}, torch.float32)


def test_install_packed_q8_0_linears_refuses_a_shape_mismatch():
    root = nn.Module()
    root.proj = nn.Linear(32, 16, bias=False)  # in_features=32, but the packed weight below is 64
    codes, scale, _source = _random_q8_0_pair(16, 64, seed=10)

    with pytest.raises(ValueError, match="but the packed weight is"):
        install_packed_q8_0_linears(root, {"proj.weight": (codes, scale)}, torch.float32)


def test_install_packed_q8_0_linears_refuses_a_non_linear_target():
    root = nn.Module()
    root.proj = nn.LayerNorm(64)
    codes, scale, _source = _random_q8_0_pair(16, 64, seed=11)

    with pytest.raises(TypeError, match="not nn.Linear"):
        install_packed_q8_0_linears(root, {"proj.weight": (codes, scale)}, torch.float32)


# ---------------------------------------------------------------------------
# End-to-end packed builder: a REAL (non-placeholder) Q8_0-encoded tiny GGUF
# text encoder, through the full loader path.
# ---------------------------------------------------------------------------

def test_pruned_gguf_q8_0_text_encoder_builder_round_trip(tmp_path):
    from core.models.minimax_music3 import loader
    from tests.minimax_music3_gguf_fixture import (
        write_tiny_pruned_gguf_q8_0_text_encoder_and_official_tree,
    )

    fixture = write_tiny_pruned_gguf_q8_0_text_encoder_and_official_tree(tmp_path)
    language_model, rvq_depth_decoder, _depth_config = (
        loader.build_language_model_and_depth_decoder_from_pruned_gguf_q8_0_text_encoder(
            fixture["text_encoder_path"], fixture["official"], torch.float32,
        )
    )

    assert type(language_model).__name__ == "Qwen3ForCausalLM"
    assert type(rvq_depth_decoder).__name__ == "MiniMaxMusic3RVQDepthDecoder"
    assert not hasattr(language_model, "lm_head")
    assert hasattr(language_model, "lm_head_pruned")

    lm_packed = [m for m in language_model.modules() if isinstance(m, GGUFQ8_0Linear)]
    depth_packed = [m for m in rvq_depth_decoder.modules() if isinstance(m, GGUFQ8_0Linear)]
    # 1 LM layer * (qkv->3 + gate_up->2 + o_proj + down_proj = 7) + lm_head_pruned = 8
    assert len(lm_packed) == 8
    # 1 depth layer * 7 + projection + 2 audio_heads = 10
    assert len(depth_packed) == 10

    no_meta = [
        n for n, t in list(language_model.named_parameters()) + list(language_model.named_buffers())
        if getattr(t, "is_meta", False)
    ] + [
        n for n, t in list(rvq_depth_decoder.named_parameters()) + list(rvq_depth_decoder.named_buffers())
        if getattr(t, "is_meta", False)
    ]
    assert no_meta == []

    # Every packed buffer starts host-resident.
    for m in lm_packed + depth_packed:
        assert m.qweight.device.type == "cpu"
        assert m.qscale.device.type == "cpu"

    # Correctness, within Q8_0's own tolerance (NOT bit-identical -- lossy by
    # construction, and NOT elementwise `allclose` either: with a single
    # 32-wide block per row there is little averaging, so an individual
    # near-zero output element can show a large RELATIVE error while the
    # tensor as a whole is accurate -- the same relative-RMS metric used to
    # characterize the real staged checkpoint's own error distribution, not
    # a looser elementwise bound). o_proj's forward against a random
    # activation must be close to what the PRE-QUANTIZATION source weight
    # would have produced.
    def _relative_rms(got: torch.Tensor, expected: torch.Tensor) -> float:
        diff = (got - expected).float()
        return float(diff.pow(2).mean().sqrt() / expected.float().pow(2).mean().sqrt())

    o_proj = language_model.model.layers[0].self_attn.o_proj
    assert isinstance(o_proj, GGUFQ8_0Linear)
    x = torch.randn(4, o_proj.in_features)
    got = o_proj(x)
    expected = torch.nn.functional.linear(x, fixture["q8_0_source_tensors"]["model.layers.0.self_attn.o_proj.weight"])
    assert _relative_rms(got, expected) < 0.1

    # lm_head_pruned is also Q8_0 on the real checkpoint -- confirm the
    # PATCHED leaf module itself is packed too, not only the body layers.
    assert isinstance(language_model.lm_head_pruned, GGUFQ8_0Linear)
    lm_head_expected = fixture["q8_0_source_tensors"]["model.lm_head_pruned.weight"]
    x_head = torch.randn(2, language_model.lm_head_pruned.in_features)
    got_head = language_model.lm_head_pruned(x_head)
    expected_head = torch.nn.functional.linear(x_head, lm_head_expected)
    assert _relative_rms(got_head, expected_head) < 0.1

    # The DENSE (non-quantized) side is bit-identical, unaffected by any of
    # this -- exactly as the plain pruned GGUF builder's own test checks.
    got_norm = language_model.model.norm.weight.to(torch.float32)
    assert torch.equal(got_norm, fixture["dense_tensors"]["model.norm.weight"])
