"""Small native forward/backward oracles; CUDA initialization is forbidden."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def forbidden_cuda(*args, **kwargs):
    raise AssertionError("CPU native test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.model_definition import create_transformer, transformer_parameter_census


def tiny_config(padding):
    return {"dim": 24, "n_layers": 2, "n_refiner_layers": 1, "n_heads": 3,
            "n_kv_heads": 3, "cap_feat_dim": 8, "in_channels": 16,
            "axes_dims": [2, 2, 4], "axes_lens": [128, 32, 32],
            "all_patch_size": [2], "all_f_patch_size": [1],
            "alignment_padding_mode": padding, "multi_frame_output": padding == "learned"}


@pytest.mark.parametrize("module", ["configuration_bailingmm2", "modeling_bailingmm2",
    "modeling_bailing_moe_v2", "processing_bailingmm2", "tokenization_bailing",
    "image_processing_bailingmm2", "qwen2_5_vit", "diffusion.autoencoder_kl_qwenimage",
    "diffusion.pipeline", "diffusion.generator"])
def test_native_imports_do_not_initialize_cuda(module):
    importlib.import_module("core.models.ming_image_01.vendor." + module)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("padding", ["zero_masked", "learned"])
def test_native_ragged_reference_forward_backward_and_masks(padding):
    torch.manual_seed(41)
    model = create_transformer(tiny_config(padding), device="cpu", dtype=torch.float32)
    if padding == "learned":
        torch.nn.init.constant_(model.x_pad_token, 0.02)
        torch.nn.init.constant_(model.cap_pad_token, 0.03)
    frames = 3 if padding == "learned" else 1
    targets = [torch.randn(16, frames, 6, 8, requires_grad=True),
               torch.randn(16, frames, 8, 8, requires_grad=True)]
    refs = [torch.randn(16, 1, 6, 8), torch.randn(16, 1, 8, 8)]
    queries = [torch.randn(5, 8, requires_grad=True), torch.randn(8, 8, requires_grad=True)]
    direct = [torch.randn(3, 24, requires_grad=True), torch.randn(5, 24, requires_grad=True)]
    masks = []
    handle = model.layers[0].register_forward_pre_hook(lambda _m, args: masks.append(args[1].detach().clone()))
    result = model(targets, torch.tensor([0.2, 0.7]), queries, ref_x=refs, cap_feats_2=direct).sample
    handle.remove()
    for output, target in zip(result, targets):
        expected_frames = frames + 1 if padding == "learned" else 1
        assert output.shape == (16, expected_frames, *target.shape[-2:])
        assert torch.isfinite(output).all()
    # Supervise only target frames; native Layer returns appended references too.
    loss = sum(output[:, :frames].square().mean() for output in result)
    loss.backward()
    assert all(target.grad is not None and torch.isfinite(target.grad).all() for target in targets)
    assert all(q.grad is not None and q.grad.abs().sum() > 0 for q in queries + direct)
    assert model.layers[0].attention.to_q.weight.grad.abs().sum() > 0
    if padding == "learned":
        assert model.x_pad_token.grad is not None
        assert model.cap_pad_token.grad is not None
        assert masks[0][0].all()  # First item has 48 native image tokens plus live alignment padding.
    else:
        assert (~masks[0]).any()
    assert not torch.cuda.is_initialized()


def test_meta_parameter_census_matches_native_model():
    masked = transformer_parameter_census(tiny_config("zero_masked"))
    learned = transformer_parameter_census(tiny_config("learned"))
    native = create_transformer(tiny_config("learned"), device="cpu", dtype=torch.float32)
    assert learned["total_parameters"] == sum(p.numel() for p in native.parameters())
    assert learned["total_parameters"] - masked["total_parameters"] == 48
    assert learned["groups"]["x_pad_token"] == learned["groups"]["cap_pad_token"] == 24
    assert not torch.cuda.is_initialized()
