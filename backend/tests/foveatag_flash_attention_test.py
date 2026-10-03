"""External FlashAttention must preserve shifted-window connectivity and backward."""

import os

import pytest
import torch

from core.tagger.foveatag.encoder import (
    RMSNorm, SpatialAttention, WindowAttentionBlock, _external_flash_attention, _window_flash_plan,
)


@pytest.mark.parametrize("height,width,shift", [(8, 8, 0), (8, 8, 2), (5, 7, 0), (5, 7, 2)])
def test_packed_windows_match_masked_output_and_qkv_gradients(height, width, shift):
    from core.tagger.foveatag.encoder import _shift_regions, _window_partition
    window, batch = 4, 2
    padded_height, padded_width = ((height + 3) // 4 * 4, (width + 3) // 4 * 4)
    valid = torch.zeros(batch, padded_height, padded_width, dtype=torch.bool)
    valid[:, :height, :width] = True
    if shift:
        valid = torch.roll(valid, (-shift, -shift), (1, 2))
    valid = _window_partition(valid[..., None], window).squeeze(-1)
    regions = _window_partition(_shift_regions(batch, padded_height, padded_width,
        window, shift, torch.device("cpu"))[..., None], window).squeeze(-1)
    allowed = valid[:, :, None] & valid[:, None, :] & (regions[:, :, None] == regions[:, None, :])
    allowed |= (~valid)[:, :, None] & torch.eye(16, dtype=torch.bool)[None]
    values = [torch.randn(valid.shape[0], 2, 16, 8, requires_grad=True) for _ in range(3)]
    reference = torch.nn.functional.scaled_dot_product_attention(*values, attn_mask=allowed[:, None])
    expected_gradients = torch.autograd.grad(reference.square().sum(), values)
    packed_values = [value.detach().clone().requires_grad_() for value in values]
    # The registry's CPU native varlen reference exercises the identical packing/unpacking.
    actual = _external_flash_attention(*packed_values,
        _window_flash_plan(batch, height, width, window, shift, "cpu"), training=True)
    actual_gradients = torch.autograd.grad(actual.square().sum(), packed_values)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    for left, right in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(os.environ.get("FOVEATAG_FLASH_GPU_TEST") != "1", reason="explicit GPU verification")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["global", "window", "shifted_padded"])
def test_real_external_flash_forward_and_backward(dtype, kind, monkeypatch):
    from core.attention.registry import BACKENDS
    calls = []
    descriptor = BACKENDS["flash"]
    from dataclasses import replace
    def dense(*args, **kwargs):
        calls.append("dense")
        return descriptor.fn(*args, **kwargs)
    def varlen(*args, **kwargs):
        calls.append("varlen")
        return descriptor.varlen_fn(*args, **kwargs)
    monkeypatch.setitem(BACKENDS, "flash", replace(descriptor, fn=dense, varlen_fn=varlen))
    if kind == "global":
        reference = SpatialAttention(192, 4).cuda().to(dtype).train()
        accelerated = SpatialAttention(192, 4).cuda().to(dtype).train()
        value = torch.randn(2, 23, 192, device="cuda", dtype=dtype)
        coordinates = torch.rand(2, 23, 2, device="cuda", dtype=dtype)
        attention = accelerated
    else:
        shift = kind == "shifted_padded"
        reference = WindowAttentionBlock(192, 4, 4, shift, 2).cuda().to(dtype).train()
        accelerated = WindowAttentionBlock(192, 4, 4, shift, 2).cuda().to(dtype).train()
        with torch.no_grad():
            reference.attention_scale.fill_(.5)
            reference.mlp_scale.fill_(.5)
        height, width = (5, 7) if shift else (8, 8)
        value = torch.randn(2, 192, height, width, device="cuda", dtype=dtype)
        coordinates = torch.rand(2, height, width, 2, device="cuda", dtype=dtype)
        attention = accelerated.attention
    accelerated.load_state_dict(reference.state_dict())
    for instance in (reference, accelerated):
        for module in instance.modules():
            if isinstance(module, RMSNorm):
                module.use_fp32 = True
    attention.use_sdpa = True
    attention.use_external_flash = True
    left, right = value.clone().requires_grad_(), value.clone().requires_grad_()
    expected, actual = reference(left, coordinates), accelerated(right, coordinates)
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.012)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=.08, atol=2e-5)
    for (_, original), (_, optimized) in zip(reference.named_parameters(), accelerated.named_parameters()):
        torch.testing.assert_close(original.grad, optimized.grad, rtol=.08, atol=2e-4)
    assert calls == (["dense"] if kind == "global" else ["varlen"])


@pytest.mark.skipif(os.environ.get("FOVEATAG_FLASH_GPU_TEST") != "1", reason="explicit GPU verification")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_external_flash_checkpointed_encoder_gradients(dtype):
    from core.tagger.foveatag.encoder import FoveaEncoder, FoveaEncoderConfig
    config = FoveaEncoderConfig(widths=(16, 32, 192, 256), depths=(1, 1, 2, 1),
        window_size=4, attention_heads=(4, 4), scale_dropout=0)
    reference = FoveaEncoder(config).cuda().train()
    checked = FoveaEncoder(config).cuda().train()
    checked.load_state_dict(reference.state_dict())
    for instance in (reference, checked):
        instance.set_sdpa(True, external_flash=True)
    checked.gradient_checkpointing_enable()
    image = torch.randn(1, 3, 72, 104, device="cuda")
    geometry = torch.tensor([[.5, .5, 1., 1., 0., 0., 0., 0., 1., 1.]], device="cuda")
    for instance in (reference, checked):
        with torch.autocast("cuda", dtype=dtype):
            result = instance(image, geometry)
            loss = result.local_map.float().square().mean() + result.global_summary.float().square().mean()
        loss.backward()
    for (name, left), (_, right) in zip(reference.named_parameters(), checked.named_parameters()):
        assert left.grad is not None and torch.isfinite(left.grad).all(), name
        assert torch.isfinite(right.grad).all(), name
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
