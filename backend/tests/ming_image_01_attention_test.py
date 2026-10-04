"""CPU native oracle, ragged KV packing and lifecycle attention routing."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from core.models.ming_image_01.attention import install_attention, joint_attention, MingAttentionProcessor
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.residual_unet import restore_topology
from core.attention.registry import BACKENDS
from core.attention.backends import _native_varlen
from ming_image_01_rewiring_test import config, spec_for


def dense_oracle(q, k, v, mask):
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
        v.transpose(1, 2), attn_mask=mask[:, None, None, :]).transpose(1, 2)


@pytest.mark.parametrize("padding", ["zero_masked", "learned"])
@pytest.mark.parametrize("rewired", [False, True])
def test_conduit_matches_native_dit_and_gradients_with_ragged_context(padding, rewired):
    torch.manual_seed(23)
    original = create_transformer(config(padding), device="cpu", dtype=torch.float32)
    if padding == "learned":
        torch.nn.init.constant_(original.x_pad_token, .02)
        torch.nn.init.constant_(original.cap_pad_token, .03)
    if rewired:
        restore_topology(original, spec_for(original))
    conduit = deepcopy(original)
    assert install_attention(conduit, "normal") >= len(conduit.layers)
    assert original.state_dict().keys() == conduit.state_dict().keys()
    frames = 3 if padding == "learned" else 1
    images = [torch.randn(16, frames, 16, 16) for _ in range(2)]
    references = [torch.randn(16, 1, 16, 16), torch.randn(16, 2, 16, 16)]
    queries = [torch.randn(5, 8) for _ in range(2)]
    captions = [torch.randn(3, 24), torch.randn(35, 24)]
    outputs, leaves = [], []
    for model in (original, conduit):
        x, q, c = [[value.clone().requires_grad_() for value in group] for group in (images, queries, captions)]
        result = model(x, torch.tensor([.2, .6]), q, cap_feats_2=c, ref_x=references).sample
        sum(value.square().mean() for value in result).backward()
        outputs.append(result)
        leaves.append(x + q + c)
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    for a, b in zip(*leaves):
        torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=2e-6)
    for name, parameter in original.named_parameters():
        other = dict(conduit.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=5e-4, atol=2e-6)
    assert not torch.cuda.is_initialized()


def test_packed_flash_keeps_all_query_rows_and_matches_key_mask_gradients(monkeypatch):
    torch.manual_seed(3)
    mask = torch.tensor([[True, True, False, True, False], [True, False, True, True, True]])
    calls = []
    def kernel(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        calls.append((cu_q.clone(), cu_k.clone(), max_q, max_k))
        return _native_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs)
    monkeypatch.setitem(BACKENDS, "flash", replace(BACKENDS["flash"], needs_half_dtype=False, varlen_fn=kernel))
    values = [torch.randn(2, 5, 3, 8) for _ in range(3)]
    first = [value.clone().requires_grad_() for value in values]
    second = [value.clone().requires_grad_() for value in values]
    expected = dense_oracle(*first, mask)
    actual = joint_attention(*second, mask, backend="flash")
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert calls and calls[0][0].tolist() == [0, 5, 10] and calls[0][1].tolist() == [0, 3, 7]
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    for a, b in zip(first, second):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("shape", [(2, 5), (2, 1, 1, 5), (1, 5)])
@pytest.mark.parametrize("chunk", [0, 2])
def test_all_valid_key_mask_keeps_flash_and_matches_gradients(monkeypatch, shape, chunk):
    from core.models.ming_image_01 import attention
    calls = []
    def dispatch(q, k, v, *, attn_mask, backend, mode):
        calls.append((attn_mask, backend))
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    monkeypatch.setattr(attention, "dispatch_attention", dispatch)
    values = [torch.randn(2, 5, 3, 8) for _ in range(3)]
    first = [v.clone().requires_grad_() for v in values]
    second = [v.clone().requires_grad_() for v in values]
    expected = dense_oracle(*first, torch.ones(2, 5, dtype=torch.bool))
    actual = joint_attention(*second, torch.ones(shape, dtype=torch.bool),
                             backend="flash", query_chunk_tokens=chunk)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert calls and all(mask is None and backend == "flash" for mask, backend in calls)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for a, b in zip(first, second):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)


def test_training_sage_guard_and_all_masked_dense_semantics(monkeypatch):
    calls = []
    def kernel(*args, **kwargs):
        calls.append(True)
        return _native_varlen(*args, **kwargs)
    monkeypatch.setitem(BACKENDS, "sage", replace(BACKENDS["sage"], needs_half_dtype=False,
        allowed_head_dims={8}, varlen_fn=kernel))
    q, k, v = [torch.randn(2, 4, 2, 8, requires_grad=True) for _ in range(3)]
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    joint_attention(q, k, v, mask, backend="sage").sum().backward()
    assert not calls and all(value.grad is not None for value in (q, k, v))
    with torch.no_grad():
        joint_attention(q, k, v, mask, backend="sage")
    assert calls == [True]
    mask[1] = False
    actual = joint_attention(q, k, v, mask, backend="native")
    torch.testing.assert_close(actual, dense_oracle(q, k, v, mask))
    assert actual[1].count_nonzero() == 0
    assert not torch.cuda.is_initialized()


def test_attention_request_and_training_installation_reach_native_processors():
    from ming_image_01_pipeline_test import manager, params
    from core.training.arch.ming_image_01 import MingImage01ArchHandler
    from types import SimpleNamespace
    owner, components = manager("generation_edit")
    owner._generate_txt2img_ming_image_01(params(attention_type="flash"))
    assert components.transformer._ming_attention_backend == "flash"
    assert all(isinstance(layer.attention.processor, MingAttentionProcessor) for layer in components.transformer.layers)
    trainer = SimpleNamespace(transformer=components.transformer, attention_backend="sage",
        _resolve_training_backend=lambda backend: "native")
    MingImage01ArchHandler().setup_attention_backend(trainer)
    assert trainer.transformer._ming_attention_backend == "native"
    with pytest.raises(ValueError, match="SLA"):
        install_attention(trainer.transformer, "sla")
    assert not torch.cuda.is_initialized()
