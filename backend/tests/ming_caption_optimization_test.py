"""Numerical oracles for hidden-state-only conditioning and sparse dispatch."""

import copy
import inspect
import textwrap
import types

import pytest
import torch

from ming_image_01_conditioning_test import inputs, make_conditioner
from core.models.ming_image_01.vendor import modeling_bailingmm2 as native_module


def legacy_conditioning(native):
    source = textwrap.dedent(inspect.getsource(type(native).get_condition_embeds_for_image_gen))
    source = source.replace("self.model.model(", "self.model.forward(")
    source = source.replace("self.connector.model(", "self.connector(")
    namespace = dict(vars(native_module))
    exec(compile(source, "<legacy conditioning oracle>", "exec"), namespace)
    native.get_condition_embeds_for_image_gen = types.MethodType(
        namespace["get_condition_embeds_for_image_gen"], native)


@pytest.mark.parametrize("trainable", [False, True])
def test_decoder_only_matches_legacy_outputs_and_gradients(trainable):
    torch.manual_seed(320)
    optimized = make_conditioner().eval()
    optimized.set_trainable({"language", "projections", "connector", "queries"} if trainable else set())
    reference = copy.deepcopy(optimized)
    legacy_conditioning(reference.native)
    calls = []
    handles = [module.register_forward_hook(lambda *_: calls.append(True)) for module in
               (optimized.native.model.lm_head, optimized.native.connector.lm_head)]
    actual, expected = optimized(**inputs()), reference(**inputs())
    for name in ("queries", "direct"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    assert not calls
    for handle in handles:
        handle.remove()
    if trainable:
        (actual.queries.square().mean() + actual.direct.square().mean()).backward()
        (expected.queries.square().mean() + expected.direct.square().mean()).backward()
        for (name, parameter), (_, original) in zip(optimized.named_parameters(), reference.named_parameters()):
            assert (parameter.grad is None) == (original.grad is None), name
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("tokens", [1, 7, 32])
def test_sparse_dispatch_matches_dense_count_oracle(tokens):
    model = make_conditioner()
    block = model.native.model.model.layers[1].mlp
    torch.manual_seed(tokens)
    x = torch.randn(tokens, 24)
    ids = torch.stack([torch.randperm(4)[:2] for _ in range(tokens)])
    weights = torch.rand(tokens, 2).softmax(-1)
    counts = ids.new_zeros(tokens, 4).scatter_(1, ids, 1).sum(0).tolist()
    order = ids.flatten().argsort()
    sorted_x = x[order // 2]
    pieces, offset = [], 0
    with torch.no_grad():
        for expert, count in zip(block.experts, counts):
            if count:
                pieces.append(expert(sorted_x[offset:offset + count]))
                offset += count
        result = torch.empty_like(torch.cat(pieces))
        result[order] = torch.cat(pieces)
        expected = (result.reshape(tokens, 2, -1) * weights.unsqueeze(-1)).sum(1)
    torch.testing.assert_close(block.moe_infer(x, ids, weights), expected, rtol=0, atol=0)


def test_equal_length_batch_matches_single_and_different_lengths_need_trim():
    model = make_conditioner().eval()
    model.set_trainable(set())
    ids = torch.tensor([[1, 3, 5, 6], [1, 7, 8, 9]])
    together = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    for index in range(2):
        alone = model(input_ids=ids[index:index + 1], attention_mask=torch.ones_like(ids[index:index + 1]))
        torch.testing.assert_close(together.queries[index:index + 1], alone.queries, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(together.direct[index:index + 1], alone.direct, atol=1e-6, rtol=1e-5)
    shorter = model(input_ids=ids[:1, :2], attention_mask=torch.ones_like(ids[:1, :2]))
    padded = ids.clone()
    padded[0, 2:] = 0
    mask = torch.ones_like(padded)
    mask[0, 2:] = 0
    mixed = model(input_ids=padded, attention_mask=mask)
    assert mixed.direct.shape[1] != shorter.direct.shape[1]
