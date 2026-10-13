"""Offloaded saved activations come back with their original strides."""
import pytest
import torch
import torch.nn.functional as F

from core.memory_management.activation_dispatcher import AsyncActivationOffloader, offload_activations

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def attention_grads(mode):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 512, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    # BSHD tensors transposed to BHSD: the efficient-attention output is saved non-contiguous.
    engine = AsyncActivationOffloader(64 * 2**20) if mode == "async" else None
    with offload_activations(mode is not None, threshold_bytes=1, transfer_mode=mode or "sync", async_engine=engine):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        (out.float() * torch.linspace(-1, 1, out.shape[-1], device="cuda")).sum().backward()
    return [t.grad.float() for t in (q, k, v)]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_offloaded_efficient_attention_matches_resident(mode):
    reference = attention_grads(None)
    for actual, expected in zip(attention_grads(mode), reference):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_unpacked_tensor_keeps_strides():
    seen = {}

    class Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            saved = x.transpose(0, 1) * 1
            ctx.save_for_backward(saved)
            seen["forward"] = (saved.shape, saved.stride())
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            (saved,) = ctx.saved_tensors
            seen["backward"] = (saved.shape, saved.stride())
            return grad

    x = torch.randn(64, 32, 16, device="cuda", requires_grad=True)
    with offload_activations(True, threshold_bytes=1, transfer_mode="sync"):
        Probe.apply(x).sum().backward()
    assert seen["backward"] == seen["forward"]
