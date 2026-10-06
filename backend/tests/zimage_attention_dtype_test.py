"""FP32 normalization under autocast must preserve a homogeneous attention call."""
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("rotary", [False, True])
def test_fp32_norm_with_bf16_projection_backward(rotary):
    from core.models.zimage_transformer import ZImageAttention
    model = ZImageAttention(24, 2, 2).float()
    value = torch.randn(1,4,24,requires_grad=True)
    freqs = torch.ones(1,4,6,dtype=torch.complex64) if rotary else None
    calls = []
    def attention(q,k,v,**kwargs):
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        calls.append(q.dtype)
        return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
    with patch("core.zimage_utils.dispatch_attention", side_effect=attention), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(value,freqs_cis=freqs)
        loss = output.float().square().mean()
    loss.backward()
    assert calls == [torch.bfloat16] and output.dtype == torch.bfloat16
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(model.norm_q.weight.grad).all()
