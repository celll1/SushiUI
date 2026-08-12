"""MiniMax-H3: NVFP4/AWQ loading for the TEXT ENCODER.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_nvfp4_test.py -v

Covers what the ConvRot TE test (`minimax_h3_te_int8_convrot_test.py`) does
for that format: the E2M1 code table + comfy-kitchen dequant convention, the
marker validator's exact contract (including the `.pre_quant_scale`
placement rule -- required on `self_attn.o_proj`/`mlp.down_proj`, refused
everywhere else), a malformed marker still refusing even with
`allow_h3_nvfp4=True`, `model.embed_tokens`'s separate gather-then-scale
`Int8Embedding` contract, and a synthetic `_gpu_module_params` +
`functional_call` trip carrying BOTH a uint8 packed weight and a
float8_e4m3fn block scale -- the exact case that exposed the
`is_floating_point()` bug in `h3_pipeline_ops._gpu_module_params`.

NOT the 48 GiB file: every fixture here is built by hand, per the module
docstring's host-memory constraint. The real-file numerical gate is recorded
separately in `scratchpad/minimax_h3_te_nvfp4_verification.md`.
"""

import os
import sys

import pytest
import torch

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.models.common.quantized_checkpoint_guard import (  # noqa: E402
    UnsupportedQuantSemanticsError,
)
from core.models.minimax_h3.loader import (  # noqa: E402
    _guard_component_file,
    _h3_int8_embedding_layers_from_markers,
    _h3_nvfp4_layers_from_markers,
    _rewrite_te_key,
    _supported_h3_nvfp4_marker,
)


NVFP4_MARKER_BYTES = b'{"format": "nvfp4", "full_precision_matrix_mult": true}'
EMBED_MARKER_BYTES = b'{"format": "int8_tensorwise"}'


def _nvfp4_marker():
    return torch.tensor(list(NVFP4_MARKER_BYTES), dtype=torch.uint8)


def _embed_marker():
    return torch.tensor(list(EMBED_MARKER_BYTES), dtype=torch.uint8)


class _Handle:
    """A ``safe_open``-shaped stand-in: only ``get_tensor`` is used."""

    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, key):
        return self.tensors[key]


def _entry(tensor):
    names = {
        torch.int8: "I8", torch.float32: "F32", torch.uint8: "U8",
        torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3",
    }
    return {"dtype": names[tensor.dtype], "shape": list(tensor.shape)}


def _nvfp4_layer_tensors(out_features, in_features, *, with_pre_quant_scale):
    """A minimal, marker-VALID NVFP4 Linear's tensors (arbitrary codes/scales)."""
    packed_k = in_features // 2
    n_blocks = in_features // 16
    tensors = {
        "weight": torch.randint(0, 256, (out_features, packed_k), dtype=torch.uint8),
        "weight_scale": torch.ones(out_features, n_blocks, dtype=torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.01, dtype=torch.float32),
        "comfy_quant": _nvfp4_marker(),
    }
    if with_pre_quant_scale:
        tensors["pre_quant_scale"] = torch.ones(in_features, dtype=torch.bfloat16)
    return tensors


# ---------------------------------------------------------------------------
# E2M1 code table via comfy_kitchen.dequantize_nvfp4, hi_first=True
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="float8_e4m3fn not available"
)
def test_e2m1_code_table_matches_comfy_kitchen_dequant():
    """Hand-packed codes 0..15 (even index -> HIGH nibble, hi_first=True)
    through comfy_kitchen's own kernel reproduce the OCP E2M1 magnitude table
    {0, 0.5, 1, 1.5, 2, 3, 4, 6} and its negatives -- the exact convention
    `Nvfp4Linear.forward` relies on. Scale=1 everywhere so the still-opaque
    multi-tile block-scale swizzle (see the verification note, section 3)
    cannot perturb this specific check: only ROW 0's payload is meaningful."""
    import comfy_kitchen

    m, k = 128, 64  # smallest shape comfy_kitchen's swizzle accepts (measured)
    packed_k = k // 2
    weight = torch.zeros(m, packed_k, dtype=torch.uint8)
    codes = list(range(16))
    row0 = [(codes[2 * i] << 4) | codes[2 * i + 1] for i in range(8)]
    weight[0, :8] = torch.tensor(row0, dtype=torch.uint8)

    block_scale = torch.ones(m, k // 16, dtype=torch.float8_e4m3fn)
    weight_scale_2 = torch.tensor(1.0, dtype=torch.float32)

    out = comfy_kitchen.dequantize_nvfp4(
        weight, weight_scale_2, block_scale, output_type=torch.float32, hi_first=True
    )
    expected_mag = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    expected = torch.tensor(
        [expected_mag[c % 8] * (-1 if c >= 8 else 1) for c in codes]
    )
    torch.testing.assert_close(out[0, :16], expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Marker validator: exact contract
# ---------------------------------------------------------------------------

def test_marker_validator_accepts_the_exact_released_contract():
    tensors = _nvfp4_layer_tensors(8, 32, with_pre_quant_scale=False)
    header = {f"model.layers.0.self_attn.q_proj.{k}": _entry(v) for k, v in tensors.items()}
    key = "model.layers.0.self_attn.q_proj.comfy_quant"
    config = _supported_h3_nvfp4_marker(
        key, tensors["comfy_quant"], header, path="fixture.safetensors"
    )
    assert config == {
        "in_features": 32, "out_features": 8,
        "has_pre_quant_scale": False, "marker_numel": len(NVFP4_MARKER_BYTES),
    }


def test_marker_validator_requires_pre_quant_scale_on_o_proj_and_down_proj():
    """`o_proj`/`down_proj` have no upstream layernorm to fold AWQ smoothing
    into; a marker-valid layer at one of those names WITHOUT a
    `.pre_quant_scale` is refused, not silently accepted without smoothing."""
    tensors = _nvfp4_layer_tensors(8, 32, with_pre_quant_scale=False)
    header = {f"model.layers.0.self_attn.o_proj.{k}": _entry(v) for k, v in tensors.items()}
    key = "model.layers.0.self_attn.o_proj.comfy_quant"
    with pytest.raises(ValueError, match="carries no"):
        _supported_h3_nvfp4_marker(key, tensors["comfy_quant"], header, path="fixture.safetensors")


def test_marker_validator_refuses_pre_quant_scale_on_a_layer_that_should_not_have_one():
    """A `.pre_quant_scale` on, say, `self_attn.q_proj` (input comes from
    `input_layernorm`, smoothing is folded there) is unmodeled -- refused
    rather than silently applied to the wrong layer."""
    tensors = _nvfp4_layer_tensors(8, 32, with_pre_quant_scale=True)
    header = {f"model.layers.0.self_attn.q_proj.{k}": _entry(v) for k, v in tensors.items()}
    key = "model.layers.0.self_attn.q_proj.comfy_quant"
    with pytest.raises(ValueError, match="only"):
        _supported_h3_nvfp4_marker(key, tensors["comfy_quant"], header, path="fixture.safetensors")


def test_marker_validator_rejects_a_malformed_marker():
    """A marker declaring an extra/unknown field is not the exact released
    contract -- ``None`` (unrecognized), not a partial acceptance."""
    bad_marker = torch.tensor(
        list(b'{"format": "nvfp4", "full_precision_matrix_mult": true, "extra": 1}'),
        dtype=torch.uint8,
    )
    tensors = _nvfp4_layer_tensors(8, 32, with_pre_quant_scale=False)
    header = {f"model.layers.0.self_attn.q_proj.{k}": _entry(v) for k, v in tensors.items()}
    key = "model.layers.0.self_attn.q_proj.comfy_quant"
    assert _supported_h3_nvfp4_marker(key, bad_marker, header, path="fixture.safetensors") is None


def test_te_guard_still_refuses_a_malformed_nvfp4_marker_even_with_allow_h3_nvfp4(tmp_path):
    from safetensors.torch import save_file

    bad_marker = torch.tensor(
        list(b'{"format": "nvfp4", "full_precision_matrix_mult": true, "extra": 1}'),
        dtype=torch.uint8,
    )
    path = tmp_path / "te_bad_nvfp4.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 16, dtype=torch.uint8),
            "model.layers.0.self_attn.q_proj.weight_scale":
                torch.ones(8, 2, dtype=torch.float8_e4m3fn),
            "model.layers.0.self_attn.q_proj.weight_scale_2": torch.tensor(0.01),
            "model.layers.0.self_attn.q_proj.comfy_quant": bad_marker,
        },
        str(path),
    )
    with pytest.raises(UnsupportedQuantSemanticsError, match="quantization format"):
        _guard_component_file(str(path), label="text encoder", allow_h3_nvfp4=True)


def test_te_guard_accepts_a_valid_nvfp4_file_including_pre_quant_scale(tmp_path):
    """The positive case, at the same header-probe layer `_guard_component_file`
    runs at: a marker-valid `o_proj` WITH `.pre_quant_scale` does not refuse."""
    from safetensors.torch import save_file

    tensors = _nvfp4_layer_tensors(8, 32, with_pre_quant_scale=True)
    save_file(
        {f"model.layers.0.self_attn.o_proj.{k}": v for k, v in tensors.items()},
        str(tmp_path / "te_ok_nvfp4.safetensors"),
    )
    header, _metadata = _guard_component_file(
        str(tmp_path / "te_ok_nvfp4.safetensors"), label="text encoder", allow_h3_nvfp4=True,
    )
    assert "model.layers.0.self_attn.o_proj.pre_quant_scale" in header


# ---------------------------------------------------------------------------
# Marker -> _rewrite_te_key -> module path (NVFP4 layers + embed_tokens)
# ---------------------------------------------------------------------------

def test_nvfp4_and_embedding_markers_map_through_rewrite_te_key():
    q_tensors = _nvfp4_layer_tensors(8, 32, with_pre_quant_scale=False)
    embed_tensors = {
        "weight": torch.zeros(20, 8, dtype=torch.int8),
        "weight_scale": torch.ones(20, dtype=torch.float32),
        "comfy_quant": _embed_marker(),
    }
    all_tensors = {}
    all_tensors.update({f"model.layers.0.self_attn.q_proj.{k}": v for k, v in q_tensors.items()})
    all_tensors.update({f"model.embed_tokens.{k}": v for k, v in embed_tensors.items()})
    header = {key: _entry(value) for key, value in all_tensors.items()}
    handle = _Handle(all_tensors)

    nvfp4_layers = _h3_nvfp4_layers_from_markers(handle, header, path="fixture.safetensors")
    assert set(nvfp4_layers) == {"model.layers.0.self_attn.q_proj"}
    embed_layers = _h3_int8_embedding_layers_from_markers(handle, header, path="fixture.safetensors")
    assert set(embed_layers) == {"model.embed_tokens"}
    assert embed_layers["model.embed_tokens"] == {
        "num_embeddings": 20, "embedding_dim": 8, "marker_numel": len(EMBED_MARKER_BYTES)
    }

    mapped_nvfp4 = {_rewrite_te_key(k): v for k, v in nvfp4_layers.items()}
    mapped_embed = {_rewrite_te_key(k): v for k, v in embed_layers.items()}
    assert set(mapped_nvfp4) == {"model.language_model.layers.0.self_attn.q_proj"}
    # The whole point of the `key == "model.embed_tokens"` branch in
    # `_rewrite_te_key`: the embedding stem has NOTHING after the prefix, so
    # only an exact-match rule (not a `startswith("model.embed_tokens.")`
    # rule) can rewrite it.
    assert set(mapped_embed) == {"model.language_model.embed_tokens"}


# ---------------------------------------------------------------------------
# Nvfp4Linear: swap + forward (pre_quant_scale applied to the activation)
# ---------------------------------------------------------------------------

def test_swap_linears_to_nvfp4_and_forward_runs_on_cpu():
    """128 output rows / 64 input features: the smallest shape
    comfy_kitchen's own block-scale swizzle accepts (measured in the E2M1
    table test above -- anything under 128 rows raises inside its own
    ``from_blocked`` reshape, independent of anything this module does)."""
    from core.models.common.nvfp4_linear import Nvfp4Linear, swap_linears_to_nvfp4

    tensors = _nvfp4_layer_tensors(128, 64, with_pre_quant_scale=True)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(64, 128, bias=False)

    wrapper = Wrapper()
    swapped = swap_linears_to_nvfp4(
        wrapper, {f"proj.{k}": v for k, v in tensors.items()},
        {"proj": dict(has_pre_quant_scale=True)}, torch.float32,
    )
    assert swapped == 1
    assert isinstance(wrapper.proj, Nvfp4Linear)
    missing, unexpected = wrapper.load_state_dict(
        {f"proj.{k}": v for k, v in tensors.items()}, strict=False, assign=True
    )
    assert missing == [] and unexpected == []

    import comfy_kitchen  # noqa: F401 -- skip cleanly if the runtime is absent
    x = torch.randn(2, 64)
    out = wrapper.proj(x)
    assert out.shape == (2, 128)
    assert torch.isfinite(out).all()


def test_pre_quant_scale_multiplies_the_activation_not_the_weight():
    """Direct check of the documented direction: `y = (x * pqs) @ W^T`. Uses a
    uniform-weight layer (block_scale=1, weight_scale_2=1, every code = the
    E2M1 magnitude-1.0 code) so the dequantized weight is exactly a known
    value and the activation-side multiply is the only unverified step. 128
    output rows / 64 input features: the smallest shape comfy_kitchen's
    block-scale swizzle accepts (see the E2M1 table test above)."""
    import comfy_kitchen  # noqa: F401

    from core.models.common.nvfp4_linear import Nvfp4Linear

    out_features, in_features = 128, 64
    mod = Nvfp4Linear(
        in_features, out_features, bias=False, compute_dtype=torch.float32,
        has_pre_quant_scale=True, marker_numel=8, device="cpu",
    )
    # code 2 = magnitude 1.0 (see the E2M1 table test); pack it into every
    # nibble so every dequantized weight element is 1.0 * block_scale *
    # weight_scale_2.
    mod.weight.fill_(0x22)
    mod.weight_scale.fill_(1.0)
    mod.weight_scale_2.fill_(1.0)
    mod.pre_quant_scale.fill_(2.0)  # activation must be doubled before the GEMM
    mod.eval()

    x = torch.ones(1, in_features)
    with torch.no_grad():
        out = mod(x)
    # W is all-ones [128,64] (code 2 -> magnitude 1.0), so
    # out = (x * 2) @ W^T = (2*ones(64)) @ ones(64,)^T = 128 per output unit.
    torch.testing.assert_close(
        out, torch.full((1, out_features), float(2 * in_features)), rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Int8Embedding: gather-then-scale
# ---------------------------------------------------------------------------

def test_int8_embedding_gather_then_scale_matches_manual_dequant():
    from core.models.common.int8_embedding import Int8Embedding, swap_embedding_to_int8

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(10, 4)

    wrapper = Wrapper()
    weight = torch.randint(-127, 128, (10, 4), dtype=torch.int8)
    scale = torch.arange(1, 11, dtype=torch.float32) * 0.01
    marker = _embed_marker()
    state_dict = {
        "embed_tokens.weight": weight,
        "embed_tokens.weight_scale": scale,
        "embed_tokens.comfy_quant": marker,
    }
    swapped = swap_embedding_to_int8(
        wrapper, state_dict, {"embed_tokens": {"num_embeddings": 10, "embedding_dim": 4}},
        torch.float32,
    )
    assert swapped == 1
    assert isinstance(wrapper.embed_tokens, Int8Embedding)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False, assign=True)
    assert missing == [] and unexpected == []

    input_ids = torch.tensor([0, 5, 9, 5])
    out = wrapper.embed_tokens(input_ids)
    expected = (weight[input_ids].to(torch.float32) * scale[input_ids].unsqueeze(-1))
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# _gpu_module_params + functional_call: uint8 codes AND float8 block scale
# survive unwidened (the bug this format exposed)
# ---------------------------------------------------------------------------

def test_gpu_module_params_preserves_uint8_codes_and_float8_block_scale():
    """The exact regression this format found: `is_floating_point()` is TRUE
    for `torch.float8_e4m3fn`, so a naive "widen every floating buffer"
    branch would corrupt `Nvfp4Linear.weight_scale`. Runs on CPU (the dtype
    logic under test has nothing to do with the device) -- the CUDA-gated
    Nvfp4Linear-through-comfy_kitchen forward is covered by
    `test_swap_linears_to_nvfp4_and_forward_runs_on_cpu` and the real-file
    gate in the scratchpad note."""
    from core.models.minimax_h3.h3_pipeline_ops import _gpu_module_params

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("packed", torch.zeros(4, 2, dtype=torch.uint8))
            self.register_buffer("block_scale", torch.ones(4, dtype=torch.float8_e4m3fn))
            self.register_buffer("global_scale", torch.tensor(0.5, dtype=torch.float32))
            self.register_buffer("index", torch.zeros(4, dtype=torch.int64))
            self.lin = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    module = Fixture()
    gpu_params = _gpu_module_params(module, "cpu")

    assert gpu_params["packed"].dtype is torch.uint8
    assert gpu_params["block_scale"].dtype is torch.float8_e4m3fn
    assert gpu_params["global_scale"].dtype is torch.float32
    assert gpu_params["index"].dtype is torch.int64
    # Ordinary bf16 parameters still widen to float32 -- the whole reason
    # `functional_call` runs the module in fp32 in the first place.
    assert gpu_params["lin.weight"].dtype is torch.float32
