import json
import os
import sys

import pytest
import torch


BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.models.common.w4a8_linear import W4A8Linear  # noqa: E402
from core.memory_management.block_offloading import TransformerBlockOffloader  # noqa: E402
from core.models.minimax_h3.loader import (  # noqa: E402
    _map_dit_state_dict,
    _mapped_w4a8_layer_configs,
    _w4a8_layers_from_metadata,
)


class _Handle:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, key):
        return self.tensors[key]


def _entry(tensor):
    names = {
        torch.int8: "I8",
        torch.float32: "F32",
        torch.float8_e4m3fn: "F8_E4M3",
    }
    return {"dtype": names[tensor.dtype], "shape": list(tensor.shape)}


def test_w4a8_qkv_and_swiglu_sidecars_follow_weight_rows():
    qkv = {
        "blocks.0.attn.qkv_proj.weight": torch.arange(24 * 128, dtype=torch.int32).to(torch.int8).view(24, 128),
        "blocks.0.attn.qkv_proj.weight_s_rel": torch.arange(24 * 16, dtype=torch.float32).to(torch.float8_e4m3fn).view(24, 16),
        "blocks.0.attn.qkv_proj.weight_s_channel": torch.arange(24, dtype=torch.float32),
        "blocks.0.attn.qkv_proj.weight_codebook": torch.arange(16, dtype=torch.float32),
    }
    fc1 = {
        "blocks.0.mlp.fc1.weight": torch.arange(8 * 128, dtype=torch.int32).to(torch.int8).view(8, 128),
        "blocks.0.mlp.fc1.weight_s_rel": torch.arange(8 * 16, dtype=torch.float32).to(torch.float8_e4m3fn).view(8, 16),
        "blocks.0.mlp.fc1.weight_s_channel": torch.arange(8, dtype=torch.float32),
        "blocks.0.mlp.fc1.weight_codebook": torch.arange(16, dtype=torch.float32),
    }
    tensors = {**qkv, **fc1}
    layers = {
        name: {
            "format": "asym_w4a8_int8",
            "convrot": True,
            "convrot_groupsize": 256,
            "group_size": 16,
        }
        for name in ("blocks.0.attn.qkv_proj", "blocks.0.mlp.fc1")
    }
    metadata = {"_quantization_metadata": json.dumps({"layers": layers})}
    header = {key: _entry(value) for key, value in tensors.items()}
    parsed = _w4a8_layers_from_metadata(metadata, header, path="fixture.safetensors")
    mapped, _stats = _map_dit_state_dict(
        _Handle(tensors),
        header,
        {"num_attention_heads": 2, "attention_head_dim": 4, "adaln_curve_grid": 1},
        torch.bfloat16,
        w4a8_layers=parsed,
    )

    assert torch.equal(mapped["transformer_blocks.0.attn.to_q.weight"], qkv["blocks.0.attn.qkv_proj.weight"][:8])
    assert torch.equal(mapped["transformer_blocks.0.attn.to_v.weight_s_channel"], qkv["blocks.0.attn.qkv_proj.weight_s_channel"][16:])
    assert mapped["transformer_blocks.0.attn.to_q.weight_codebook"] is qkv["blocks.0.attn.qkv_proj.weight_codebook"]
    assert torch.equal(mapped["transformer_blocks.0.ff.net.0.proj.weight"][:4], fc1["blocks.0.mlp.fc1.weight"][4:])
    assert torch.equal(mapped["transformer_blocks.0.ff.net.0.proj.weight_s_channel"][:4], fc1["blocks.0.mlp.fc1.weight_s_channel"][4:])
    assert len(_mapped_w4a8_layer_configs(parsed)) == 4


def test_w4a8_linear_matches_comfy_kitchen_dequant_oracle():
    ck = pytest.importorskip("comfy_kitchen.tensor")
    torch.manual_seed(4)
    weight = torch.randn(12, 256, dtype=torch.bfloat16)
    qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(weight)
    layer = W4A8Linear(256, 12, False, torch.bfloat16)
    layer.weight = qdata
    layer.weight_s_rel = s_rel
    layer.weight_s_channel = s_channel
    layer.weight_codebook = codebook
    x = torch.randn(3, 256, dtype=torch.bfloat16)

    actual = layer(x)
    expected = ck.w4a8_int8_linear(
        x, qdata, s_rel, s_channel, codebook=codebook, correction=correction
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    dequantized = ck.dequantize_w4a8_int8_weight(
        qdata, s_rel, s_channel, codebook=codebook, correction=correction
    )
    float_reference = torch.nn.functional.linear(x, dequantized)
    rel_rms = (actual.float() - float_reference.float()).pow(2).mean().sqrt() / \
        float_reference.float().pow(2).mean().sqrt()
    assert rel_rms < 0.03

    grad_input = x.detach().clone().requires_grad_(True)
    grad_output = layer(grad_input)
    grad_output.float().sum().backward()
    oracle_input = x.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(oracle_input, dequantized).float().sum().backward()
    torch.testing.assert_close(grad_input.grad, oracle_input.grad, rtol=0, atol=0)


def test_w4a8_metadata_rejects_missing_sidecar_before_tensor_read():
    layers = {
        "blocks.0.mlp.fc2": {
            "format": "asym_w4a8_int8",
            "convrot": True,
            "convrot_groupsize": 256,
            "group_size": 16,
        }
    }
    header = {
        "blocks.0.mlp.fc2.weight": {"dtype": "I8", "shape": [8, 128]},
    }
    with pytest.raises(ValueError, match="missing weight/s_rel/s_channel"):
        _w4a8_layers_from_metadata(
            {"_quantization_metadata": json.dumps({"layers": layers})},
            header,
            path="broken.safetensors",
        )


def test_file_metadata_rejects_unknown_non_w4_semantics():
    metadata = {
        "_quantization_metadata": json.dumps({
            "layers": {
                "blocks.0.mlp.fc2": {
                    "format": "int8_tensorwise",
                    "weight_permutation": "interleaved",
                }
            }
        })
    }
    with pytest.raises(ValueError, match="unknown quantization field"):
        _w4a8_layers_from_metadata(metadata, {}, path="unknown.safetensors")


def test_w4a8_gemm_path_is_recorded_in_generation_metadata():
    from api.generation_utils import extract_fp8_gemm_info

    class _Manager:
        current_model_info = {"type": "minimax_h3"}
        minimax_h3_components = {
            "transformer": torch.nn.Sequential(
                W4A8Linear(256, 12, False, torch.bfloat16)
            )
        }

    assert extract_fp8_gemm_info(_Manager()) == "w4a8_int8(comfy-kitchen)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA block-swap lifecycle")
def test_w4a8_block_swap_moves_sidecars_with_weight():
    """A swapped-out W4A8Linear block must leave nothing of itself on the GPU.

    Before this was fixed, ``weighs_to_device``/``_build_weight_swap_jobs`` moved
    only ``.weight``: the quantization sidecar buffers (``weight_s_rel`` --
    22.97 MiB/block on the real MiniMax-H3 checkpoint --, ``weight_s_channel``,
    ``weight_codebook``) stayed GPU-resident for every block regardless of
    ``blocks_to_swap``, a fixed ~1.1 GiB VRAM cost block swap could not reclaim.
    """
    ck = pytest.importorskip("comfy_kitchen.tensor")

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.randn(256, 256, dtype=torch.bfloat16)
            qdata, s_rel, s_channel, _correction, codebook = \
                ck.quantize_w4a8_int8_weight(weight)
            self.proj = W4A8Linear(256, 256, False, torch.bfloat16)
            self.proj.weight = qdata
            self.proj.weight_s_rel = s_rel
            self.proj.weight_s_channel = s_channel
            self.proj.weight_codebook = codebook

        def forward(self, x):
            return self.proj(x)

    device = torch.device("cuda")
    blocks = torch.nn.ModuleList([_Block(), _Block(), _Block()])
    offloader = TransformerBlockOffloader(
        blocks=blocks,
        blocks_to_swap=2,
        device=device,
        target_dtype=torch.bfloat16,
    )
    sidecars = ("weight_s_rel", "weight_s_channel", "weight_codebook")
    try:
        offloader.prepare_block_devices_before_forward()
        assert blocks[0].proj.weight.device.type == "cuda"
        assert all(getattr(blocks[0].proj, name).device.type == "cuda" for name in sidecars)
        assert blocks[1].proj.weight.device.type == "cpu"
        assert all(getattr(blocks[1].proj, name).device.type == "cpu" for name in sidecars), (
            "a swapped-out block's sidecar buffers were left GPU-resident")

        offloader.wait_for_block(1)
        offloader.submit_move_blocks_forward(1)
        offloader.wait_for_block(2)
        assert blocks[1].proj.weight.device.type == "cpu"
        assert all(getattr(blocks[1].proj, name).device.type == "cpu" for name in sidecars)
        assert blocks[2].proj.weight.device.type == "cuda"
        assert all(getattr(blocks[2].proj, name).device.type == "cuda" for name in sidecars)

        output = blocks[2](torch.randn(1, 256, device=device, dtype=torch.bfloat16))
        assert output.device.type == "cuda"
        assert torch.isfinite(output).all()
    finally:
        offloader.thread_pool.shutdown(wait=True)
        blocks.to("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA block-swap lifecycle")
def test_plain_linear_block_swap_moves_only_its_own_tensors():
    """A block with no quantization sidecars must move exactly the bytes it
    moved before this change (no behaviour change for the common case)."""
    from core.memory_management.block_offloading import weight_sidecar_names

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(64, 64, bias=True, dtype=torch.bfloat16)

        def forward(self, x):
            return self.proj(x)

    device = torch.device("cuda")
    blocks = torch.nn.ModuleList([_Block(), _Block(), _Block()])
    assert weight_sidecar_names(blocks[0].proj) == []

    offloader = TransformerBlockOffloader(
        blocks=blocks,
        blocks_to_swap=2,
        device=device,
        target_dtype=torch.bfloat16,
    )
    try:
        offloader.prepare_block_devices_before_forward()
        assert blocks[0].proj.weight.device.type == "cuda"
        assert blocks[1].proj.weight.device.type == "cpu"

        offloader.wait_for_block(1)
        offloader.submit_move_blocks_forward(1)
        offloader.wait_for_block(2)
        assert blocks[1].proj.weight.device.type == "cpu"
        assert blocks[2].proj.weight.device.type == "cuda"

        output = blocks[2](torch.randn(1, 64, device=device, dtype=torch.bfloat16))
        assert output.device.type == "cuda"
        assert torch.isfinite(output).all()
    finally:
        offloader.thread_pool.shutdown(wait=True)
        blocks.to("cpu")
