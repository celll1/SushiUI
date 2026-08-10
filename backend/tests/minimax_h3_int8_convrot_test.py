import os
import sys

import pytest
import torch
import torch.nn.functional as F


BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.models.common.convrot_int8_linear import (  # noqa: E402
    ConvRotInt8Linear,
    swap_linears_to_convrot_int8,
)
from core.models.minimax_h3.loader import (  # noqa: E402
    _guard_component_file,
    _map_dit_state_dict,
    _supported_int8_convrot_marker,
)


MARKER_BYTES = (
    b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
)


def _marker():
    return torch.tensor(list(MARKER_BYTES), dtype=torch.uint8)


def _entry(tensor):
    names = {
        torch.int8: "I8",
        torch.float32: "F32",
        torch.uint8: "U8",
    }
    return {"dtype": names[tensor.dtype], "shape": list(tensor.shape)}


class _Handle:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, key):
        return self.tensors[key]


def test_real_marker_contract_requires_per_row_int8_weight_and_groupsize_256():
    marker = _marker()
    header = {
        "blocks.0.mlp.fc2.weight": {"dtype": "I8", "shape": [8, 256]},
        "blocks.0.mlp.fc2.weight_scale": {"dtype": "F32", "shape": [8, 1]},
        "blocks.0.mlp.fc2.comfy_quant": _entry(marker),
    }
    assert _supported_int8_convrot_marker(
        "blocks.0.mlp.fc2.comfy_quant", marker, header, path="fixture.safetensors"
    ) == {"convrot_groupsize": 256, "marker_numel": len(MARKER_BYTES)}

    bad = dict(header)
    bad["blocks.0.mlp.fc2.weight_scale"] = {"dtype": "F32", "shape": []}
    with pytest.raises(ValueError, match="weight_scale"):
        _supported_int8_convrot_marker(
            "blocks.0.mlp.fc2.comfy_quant", marker, bad, path="fixture.safetensors"
        )


def test_h3_guard_allows_only_the_validated_convrot_contract(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "h3_int8_convrot.safetensors"
    save_file(
        {
            "blocks.0.mlp.fc2.weight": torch.zeros(8, 256, dtype=torch.int8),
            "blocks.0.mlp.fc2.weight_scale": torch.ones(8, 1, dtype=torch.float32),
            "blocks.0.mlp.fc2.comfy_quant": _marker(),
        },
        str(path),
    )
    with pytest.raises(RuntimeError, match="HADAMARD-ROTATED"):
        _guard_component_file(str(path), label="transformer")
    header, _metadata = _guard_component_file(
        str(path), label="transformer", allow_h3_int8_convrot=True
    )
    assert header["blocks.0.mlp.fc2.weight"]["dtype"] == "I8"


def test_qkv_and_swiglu_per_row_scales_and_markers_follow_weight_rows():
    qkv_weight = torch.arange(24 * 256, dtype=torch.int32).to(torch.int8).view(24, 256)
    qkv_scale = torch.arange(1, 25, dtype=torch.float32).view(24, 1)
    fc1_weight = torch.arange(8 * 256, dtype=torch.int32).to(torch.int8).view(8, 256)
    fc1_scale = torch.arange(1, 9, dtype=torch.float32).view(8, 1)
    tensors = {
        "blocks.0.attn.qkv_proj.weight": qkv_weight,
        "blocks.0.attn.qkv_proj.weight_scale": qkv_scale,
        "blocks.0.attn.qkv_proj.comfy_quant": _marker(),
        "blocks.0.mlp.fc1.weight": fc1_weight,
        "blocks.0.mlp.fc1.weight_scale": fc1_scale,
        "blocks.0.mlp.fc1.comfy_quant": _marker(),
    }
    header = {key: _entry(value) for key, value in tensors.items()}
    source_layers = {
        "blocks.0.attn.qkv_proj": {"convrot_groupsize": 256},
        "blocks.0.mlp.fc1": {"convrot_groupsize": 256},
    }
    mapped, stats = _map_dit_state_dict(
        _Handle(tensors),
        header,
        {"num_attention_heads": 2, "attention_head_dim": 4, "adaln_curve_grid": 1},
        torch.bfloat16,
        int8_convrot_layers=source_layers,
    )

    assert torch.equal(mapped["transformer_blocks.0.attn.to_q.weight"], qkv_weight[:8])
    assert torch.equal(mapped["transformer_blocks.0.attn.to_v.weight_scale"], qkv_scale[16:, 0])
    assert torch.equal(
        mapped["transformer_blocks.0.attn.to_k.comfy_quant"], _marker()
    )
    assert torch.equal(mapped["transformer_blocks.0.ff.net.0.proj.weight"][:4], fc1_weight[4:])
    assert torch.equal(
        mapped["transformer_blocks.0.ff.net.0.proj.weight_scale"][:4], fc1_scale[4:, 0]
    )
    assert stats["qkv_split"] == 1
    assert stats["swiglu_scale_swapped"] == 1


def test_convrot_linear_uses_online_rotation_and_exact_unrotation_for_grad():
    from comfy_kitchen import int8_linear
    from comfy_kitchen.backends.eager.quantization import (
        dequantize_int8_convrot_weight,
        quantize_int8_convrot_weight,
    )

    torch.manual_seed(9)
    source_weight = torch.randn(12, 256, dtype=torch.float32)
    qdata, scale = quantize_int8_convrot_weight(source_weight, 256)
    layer = ConvRotInt8Linear(
        256,
        12,
        False,
        torch.float32,
        convrot_groupsize=256,
        marker_numel=len(MARKER_BYTES),
    )
    layer.weight = qdata
    layer.weight_scale = scale.reshape(-1)
    layer.comfy_quant = _marker()
    x = torch.randn(5, 256, dtype=torch.float32)

    with torch.no_grad():
        actual = layer(x)
        expected = int8_linear(
            x,
            qdata,
            scale,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=256,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    restored = dequantize_int8_convrot_weight(qdata, scale, 256)
    grad_x = x.detach().requires_grad_(True)
    layer(grad_x).sum().backward()
    oracle_x = x.detach().requires_grad_(True)
    F.linear(oracle_x, restored).sum().backward()
    torch.testing.assert_close(grad_x.grad, oracle_x.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Comfy-Kitchen CUDA ConvRot path")
def test_cuda_path_handles_real_h3_hidden_width_without_dense_weight_rebuild():
    from comfy_kitchen.backends.eager.quantization import (
        dequantize_int8_convrot_weight,
        quantize_int8_convrot_weight,
    )

    torch.manual_seed(11)
    source_weight = torch.randn(5376, 5376, dtype=torch.float32)
    qdata, scale = quantize_int8_convrot_weight(source_weight, 256)
    layer = ConvRotInt8Linear(
        5376,
        5376,
        False,
        torch.bfloat16,
        convrot_groupsize=256,
        marker_numel=len(MARKER_BYTES),
    ).cuda()
    layer.weight = qdata.cuda()
    layer.weight_scale = scale.reshape(-1).cuda()
    layer.comfy_quant = _marker().cuda()
    x = torch.randn(8, 5376, device="cuda", dtype=torch.bfloat16)

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        actual = layer(x)
    peak = torch.cuda.max_memory_allocated()
    restored = dequantize_int8_convrot_weight(qdata, scale, 256).to(
        device="cuda", dtype=torch.bfloat16
    )
    expected = F.linear(x, restored)
    rel_rms = (actual.float() - expected.float()).pow(2).mean().sqrt() / \
        expected.float().pow(2).mean().sqrt()
    assert torch.isfinite(actual).all()
    assert rel_rms < 0.03
    assert peak - baseline < 48 * 1024 * 1024


def test_swap_installs_marker_bearing_convrot_module_without_dense_allocation():
    model = torch.nn.Module()
    model.fc = torch.nn.Linear(256, 8, bias=False, device="meta")
    state_dict = {
        "fc.weight": torch.zeros(8, 256, dtype=torch.int8),
        "fc.weight_scale": torch.ones(8, dtype=torch.float32),
        "fc.comfy_quant": _marker(),
    }
    configs = {
        "fc": {"convrot_groupsize": 256, "marker_numel": len(MARKER_BYTES)}
    }
    assert swap_linears_to_convrot_int8(model, state_dict, configs, torch.bfloat16) == 1
    assert isinstance(model.fc, ConvRotInt8Linear)
    assert model.fc.weight.is_meta
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    assert missing == []
    assert unexpected == []
    assert model.fc.weight.dtype is torch.int8
    assert torch.equal(model.fc.comfy_quant, _marker())


def test_convrot_gemm_path_is_recorded_as_fixed_comfy_kitchen_operator():
    from api.generation_utils import extract_fp8_gemm_info
    from api.quantized_gemm import report_quantized_gemm_outcome

    class _Manager:
        current_model_info = {"type": "minimax_h3"}
        minimax_h3_components = {
            "transformer": torch.nn.Sequential(
                ConvRotInt8Linear(
                    256,
                    8,
                    False,
                    torch.bfloat16,
                    convrot_groupsize=256,
                    marker_numel=len(MARKER_BYTES),
                )
            )
        }

    label = extract_fp8_gemm_info(_Manager())
    assert label == "convrot_int8(comfy-kitchen)"
    message = report_quantized_gemm_outcome("w8a8", label, "minimax_h3")
    assert "does not control this checkpoint's fixed quantized operator" in message
