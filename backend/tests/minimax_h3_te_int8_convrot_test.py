"""MiniMax-H3: ConvRot INT8 loading for the TEXT ENCODER.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_int8_convrot_test.py -v

Covers what `minimax_h3_int8_convrot_test.py` (the DiT) and
`minimax_h3_te_selection_test.py` (file selection) do not: the TE builder's
own marker -> `_rewrite_te_key` -> module-path mapping, the `[out, 1] ->
[out]` `weight_scale` reshape `swap_linears_to_convrot_int8` requires, that a
file whose markers do NOT match the exact ConvRot contract is still refused
even with `allow_h3_int8_convrot=True`, that the census/verify pattern
`_build_text_encoder` now runs on its non-ConvRot layers still catches an
unswappable quantized tensor, and a synthetic numerical check of
`h3_pipeline_ops._gpu_module_params` + `torch.func.functional_call` against a
real `ConvRotInt8Linear` (not the 48 GiB file -- see the module docstring's
host-memory constraint).
"""

import os
import sys

import pytest
import torch

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from model_root import model_path  # noqa: E402

from core.models.common.quantized_checkpoint_guard import (  # noqa: E402
    UnsupportedQuantSemanticsError,
    quantized_state_dict_report,
    scaled_quantization_report,
    verify_quantized_swap,
)
from core.models.minimax_h3.loader import (  # noqa: E402
    _guard_component_file,
    _int8_convrot_layers_from_markers,
    _rewrite_te_key,
)


MARKER_BYTES = (
    b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
)


def _marker_tensor():
    return torch.tensor(list(MARKER_BYTES), dtype=torch.uint8)


class _Handle:
    """A ``safe_open``-shaped stand-in: only ``get_tensor`` is used."""

    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, key):
        return self.tensors[key]


def _entry(tensor):
    names = {torch.int8: "I8", torch.float32: "F32", torch.uint8: "U8", torch.bfloat16: "BF16"}
    return {"dtype": names[tensor.dtype], "shape": list(tensor.shape)}


# ---------------------------------------------------------------------------
# Marker -> `_rewrite_te_key` -> module path
# ---------------------------------------------------------------------------

def test_marker_layers_map_through_rewrite_te_key_to_the_live_module_path():
    """The exact mapping `_build_text_encoder` performs, run standalone.

    Source keys use the file's flat naming (`model.layers.N....`); the built
    `Qwen3VLForConditionalGeneration` needs `model.language_model.layers.N....`
    -- `_rewrite_te_key` is the ONLY translation, no fan-out (unlike the DiT's
    fused qkv), so the rewritten source key IS the target module path.
    """
    marker = _marker_tensor()
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 256, dtype=torch.int8),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(8, 1, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.comfy_quant": marker,
        "model.layers.3.mlp.down_proj.weight": torch.zeros(4, 512, dtype=torch.int8),
        "model.layers.3.mlp.down_proj.weight_scale": torch.ones(4, 1, dtype=torch.float32),
        "model.layers.3.mlp.down_proj.comfy_quant": marker,
        # An ordinary bf16 tensor with no marker: must NOT show up in the map.
        "model.embed_tokens.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
    }
    header = {key: _entry(value) for key, value in tensors.items()}
    handle = _Handle(tensors)

    source_layers = _int8_convrot_layers_from_markers(handle, header, path="fixture.safetensors")
    assert set(source_layers) == {
        "model.layers.0.self_attn.q_proj",
        "model.layers.3.mlp.down_proj",
    }
    assert source_layers["model.layers.0.self_attn.q_proj"] == {
        "convrot_groupsize": 256, "marker_numel": len(MARKER_BYTES)
    }

    mapped = {_rewrite_te_key(source): cfg for source, cfg in source_layers.items()}
    assert set(mapped) == {
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.3.mlp.down_proj",
    }


# ---------------------------------------------------------------------------
# The [out, 1] -> [out] weight_scale reshape
# ---------------------------------------------------------------------------

def test_weight_scale_reshape_is_a_zero_copy_view_and_matches_int8linear_shape():
    """`Int8Linear.weight_scale` registers `(out_features,)`; the file's
    validated scale is `[out, 1]`. `_build_text_encoder` reshapes it on the
    exact tensor the swap and the `load_state_dict` both read -- verified here
    as a standalone operation, plus that it shares storage with the original
    (no host copy of a 48 GiB file's sidecar tensors)."""
    scale = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    reshaped = scale.reshape(-1)
    assert tuple(reshaped.shape) == (8,)
    assert reshaped.data_ptr() == scale.data_ptr()
    assert torch.equal(reshaped, torch.arange(8, dtype=torch.float32))


# ---------------------------------------------------------------------------
# A file whose markers do NOT match the exact contract is still refused
# ---------------------------------------------------------------------------

def test_te_guard_refuses_a_convrot_declaration_with_the_wrong_groupsize(tmp_path):
    """`allow_h3_int8_convrot=True` waives ONE exact contract, not any ConvRot
    declaration. A groupsize the marker validator does not recognize (128
    instead of 256) must still be refused, the same way the DiT's guard test
    pins it."""
    from safetensors.torch import save_file

    bad_marker = torch.tensor(
        list(b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 128}'),
        dtype=torch.uint8,
    )
    path = tmp_path / "te_bad_groupsize.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 128, dtype=torch.int8),
            "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(8, 1, dtype=torch.float32),
            "model.layers.0.self_attn.q_proj.comfy_quant": bad_marker,
        },
        str(path),
    )
    with pytest.raises(UnsupportedQuantSemanticsError, match="HADAMARD-ROTATED"):
        _guard_component_file(str(path), label="text encoder", allow_h3_int8_convrot=True)


def test_te_guard_still_refuses_nvfp4_pre_quant_scale(tmp_path):
    """The co-distributed `..._nvfp4_awq` file: `allow_h3_int8_convrot` does not
    touch `.pre_quant_scale` at all, so it refuses exactly as before."""
    from safetensors.torch import save_file

    path = tmp_path / "te_nvfp4.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.pre_quant_scale": torch.ones(8, dtype=torch.bfloat16),
        },
        str(path),
    )
    with pytest.raises(UnsupportedQuantSemanticsError, match="pre_quant_scale"):
        _guard_component_file(str(path), label="text encoder", allow_h3_int8_convrot=True)


# ---------------------------------------------------------------------------
# Census/verify catching an unexpected (unswapped) quantized tensor
# ---------------------------------------------------------------------------

def test_census_verify_pattern_catches_a_scaled_layer_the_te_builder_cannot_swap():
    """`_build_text_encoder` swaps ONLY the validated ConvRot layers; it has no
    Int8Linear/Fp8Linear swap for anything else. This is the exact
    census+verify call the builder makes on the non-ConvRot remainder, with
    the always-0 swap count that remainder gets -- proving a plain scaled
    int8 layer (no `.comfy_quant`, so the header guard never sees it) is still
    refused rather than silently cast into a bf16 parameter."""
    state_dict = {
        "model.language_model.layers.7.self_attn.k_proj.weight": torch.zeros(8, 8, dtype=torch.int8),
        "model.language_model.layers.7.self_attn.k_proj.weight_scale": torch.ones(8, dtype=torch.float32),
    }
    census = quantized_state_dict_report(
        state_dict, arch="MiniMax-H3", path="fixture.safetensors", label="text encoder")
    report = scaled_quantization_report(
        census, arch="MiniMax-H3", path="fixture.safetensors", label="text encoder")
    assert report is not None
    with pytest.raises(RuntimeError, match="weight-only QUANTIZED"):
        verify_quantized_swap(report, 0, arch="MiniMax-H3", path="fixture.safetensors", label="text encoder")


def test_census_verify_pattern_is_silent_on_an_ordinary_bf16_te():
    """No `.weight_scale`, no int8/float8 `.weight` -- the ordinary bf16 file,
    unaffected by any of this."""
    state_dict = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
    }
    census = quantized_state_dict_report(
        state_dict, arch="MiniMax-H3", path="fixture.safetensors", label="text encoder")
    assert census is None
    verify_quantized_swap(None, 0, arch="MiniMax-H3", path="fixture.safetensors", label="text encoder")


# ---------------------------------------------------------------------------
# Synthetic `_gpu_module_params` + `functional_call` numerical check
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Comfy-Kitchen CUDA ConvRot path")
def test_gpu_module_params_functional_call_matches_a_plain_convrot_forward():
    """NOT the 48 GiB file (module docstring's host-memory constraint) -- two
    tiny `ConvRotInt8Linear` layers built directly, run through the exact
    helper `encode_presentation` uses per decoder layer.

    `_gpu_module_params` casts every FLOATING buffer to fp32 and leaves
    non-floating ones alone. For a ConvRot module that means: the int8
    `.weight` (non-floating) passes through UNCHANGED -- comfy-kitchen's
    kernel needs the raw codes, not a promoted float -- and `.weight_scale`
    (already float32) is a no-op cast, not a silent narrowing. This is why no
    quant-aware special-casing turned out to be needed in `_gpu_module_params`
    itself: verified here rather than assumed.
    """
    from core.models.common.convrot_int8_linear import (
        require_convrot_int8_runtime, swap_linears_to_convrot_int8,
    )
    from core.models.minimax_h3.h3_pipeline_ops import _gpu_module_params

    require_convrot_int8_runtime()
    torch.manual_seed(0)
    device = "cuda"

    class TwoLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(256, 256, bias=False)
            self.b = torch.nn.Linear(256, 256, bias=False)

        def forward(self, x):
            return self.b(self.a(x))

    module = TwoLayer().to(device)

    def build_hadamard(n):
        base = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float64, device=device,
        )
        h = base
        while h.shape[0] < n:
            h = torch.kron(h, base)
        return (h / (n ** 0.5)).to(torch.float32)

    def rotate_and_quantize(weight, groupsize):
        out_features, in_features = weight.shape
        h = build_hadamard(groupsize)
        blocks = weight.to(torch.float64).reshape(out_features, in_features // groupsize, groupsize)
        rotated = (blocks @ h.to(torch.float64).T).reshape(out_features, in_features).to(torch.float32)
        scale = rotated.abs().amax(dim=1) / 127.0
        codes = torch.round(rotated / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        return codes, scale

    weight_a = torch.randn(256, 256, device=device, dtype=torch.float32)
    weight_b = torch.randn(256, 256, device=device, dtype=torch.float32)
    # Groupsize 256 -- the one exact contract this repo implements (the
    # Hadamard construction also requires a power of 4, so this is not an
    # arbitrary test choice).
    codes_a, scale_a = rotate_and_quantize(weight_a, 256)
    codes_b, scale_b = rotate_and_quantize(weight_b, 256)

    state_dict = {
        "a.weight": codes_a, "a.weight_scale": scale_a,
        "a.comfy_quant": torch.zeros(1, dtype=torch.uint8, device=device),
        "b.weight": codes_b, "b.weight_scale": scale_b,
        "b.comfy_quant": torch.zeros(1, dtype=torch.uint8, device=device),
    }
    layer_configs = {
        "a": {"convrot_groupsize": 256, "marker_numel": 1},
        "b": {"convrot_groupsize": 256, "marker_numel": 1},
    }
    swapped = swap_linears_to_convrot_int8(module, state_dict, layer_configs, torch.bfloat16)
    assert swapped == 2
    missing, unexpected = module.load_state_dict(state_dict, strict=False, assign=True)
    assert missing == [] and unexpected == []
    module.eval()

    x = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        plain = module(x)

        gpu_params = _gpu_module_params(module, device)
        # Buffer dtypes must be exactly what the kernel needs, not blindly
        # promoted: int8 codes stay int8, the f32 scale stays f32.
        assert gpu_params["a.weight"].dtype is torch.int8
        assert gpu_params["a.weight_scale"].dtype is torch.float32
        assert gpu_params["b.weight"].dtype is torch.int8
        assert gpu_params["b.weight_scale"].dtype is torch.float32

        via_functional_call = torch.func.functional_call(module, gpu_params, args=(x,))

    torch.testing.assert_close(via_functional_call, plain, rtol=0, atol=0)

    ref_a = torch.nn.functional.linear(x.float(), weight_a)
    ref = torch.nn.functional.linear(ref_a.to(torch.bfloat16).float(), weight_b)
    rel_rms = (via_functional_call.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert torch.isfinite(via_functional_call).all()
    assert rel_rms < 0.05


# ---------------------------------------------------------------------------
# Selection resolves to int8_convrot on the real distribution (header-only)
# ---------------------------------------------------------------------------

def test_real_distribution_selects_int8_convrot_by_default():
    """Header-only against the real tree (`<MODEL_ROOT>/minimax_h3`, as the rest
    of the MiniMax-H3 test suite already assumes -- see minimax_h3_training_test.py
    and minimax_h3_lora_conversion_test.py). Zero tensor bytes read."""
    root = model_path("minimax_h3")
    if not os.path.isdir(root):
        pytest.skip(f"{root} not present on this machine")
    from core.models.minimax_h3.loader import detect_minimax_h3_layout

    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    assert layout["text_encoder"] is not None
    assert layout["text_encoder"].endswith("qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    assert layout["text_encoder_reason"] == "preferred"
