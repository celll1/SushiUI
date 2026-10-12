"""Attention arithmetic, dtype boundaries and FP64-gradient references."""
import pytest
import torch

from core.models.components.global_summary_attention import global_summary_attention
from core.training.partition import PartitionBox


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_autocast_preserves_reference_gradients_and_output_dtype(dtype):
    torch.manual_seed(721)
    inputs = [torch.randn(*shape).to(dtype).requires_grad_(True) for shape in
              ((8, 32), (2, 127, 32), (2, 127, 32), (2, 63, 32))]
    do = torch.randn(2, 63, 32).to(dtype)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = global_summary_attention(*inputs)
    grads = torch.autograd.grad(out, inputs, do)
    refs = [x.detach().double().requires_grad_(True) for x in inputs]
    sq, k, v, q = refs
    summaries = ((sq.expand(2, -1, -1) @ k.transpose(-1, -2)) / 32**.5).softmax(-1) @ v
    expected = ((q @ summaries.transpose(-1, -2)) / 32**.5).softmax(-1) @ summaries
    expected_grads = torch.autograd.grad(expected, refs, do.double())
    assert out.dtype == dtype
    tolerance = {torch.bfloat16: .004, torch.float16: .0006,
                 torch.float32: 2e-6, torch.float64: 1e-12}[dtype]
    assert (out.double() - expected).norm() / expected.norm() < tolerance
    for actual, reference in zip(grads, expected_grads):
        assert actual.dtype == dtype and actual.isfinite().all()
        assert (actual.double() - reference).norm() / reference.norm() < tolerance


@pytest.mark.parametrize('architecture', ['anima', 'qwen', 'ming'])
@pytest.mark.parametrize('autocast', [False, True])
def test_each_adapter_keeps_products_fp32_and_restores_dtype(monkeypatch, architecture, autocast):
    from core.models.anima.partition_global import AnimaPartitionGlobalAdapter
    from core.models.qwen_image_21.partition_global import QwenPartitionGlobalAdapter
    from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter
    cls = {'anima': AnimaPartitionGlobalAdapter, 'qwen': QwenPartitionGlobalAdapter,
           'ming': MingPartitionGlobalAdapter}[architecture]
    model = cls(4, 12, rank=8, summary_tokens=3, dtype=torch.bfloat16)
    torch.nn.init.normal_(model.out.weight, std=.1)
    shape = (2, 8, 12, 4) if architecture == 'qwen' else (2, 4, 1 if architecture == 'anima' else 3, 8, 12)
    x = torch.randn(*shape, dtype=torch.bfloat16, requires_grad=True)
    original = torch.einsum
    seen = []
    def checked(equation, *args, **kwargs):
        assert not torch.is_autocast_enabled('cpu')
        assert all(a.dtype == torch.float32 for a in args)
        seen.append(equation)
        return original(equation, *args, **kwargs)
    monkeypatch.setattr(torch, 'einsum', checked)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        out = model(x, PartitionBox(1, 2, 3, 5))
        out.float().square().sum().backward()
    assert len(seen) == 4 and out.dtype == torch.bfloat16
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
               for p in model.parameters())
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
