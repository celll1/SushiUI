from copy import deepcopy

import pytest
import torch

from core.models.anima.anima_models import Block
from core.models.anima.anima_attention import AttentionParams
from core.experiments.anima_projected_block import project_block


def source(lora=True):
    torch.manual_seed(63)
    result = Block(16, 12, 2, mlp_ratio=4., use_adaln_lora=lora, adaln_lora_dim=8)
    with torch.no_grad():
        for name, param in result.named_parameters():
            if name.startswith('adaln_'):
                param.normal_(std=.15)
    return result


def inputs(lora=True):
    return (torch.randn(2, 1, 2, 3, 16, requires_grad=True),
            torch.randn(2, 1, 16, requires_grad=True),
            torch.randn(2, 4, 12, requires_grad=True),
            AttentionParams(attn_mode='native'),
            torch.randn(2, 1, 48, requires_grad=True) if lora else None)


@pytest.mark.parametrize('lora', [False, True])
@pytest.mark.parametrize('heads', [[0, 1], [1, 0]])
def test_full_width_forward_and_input_gradient_oracle(lora, heads):
    original = source(lora)
    saved = deepcopy(original.state_dict())
    rng = torch.get_rng_state().clone()
    projected = project_block(original, heads)
    assert torch.equal(rng, torch.get_rng_state())
    x, emb, context, attention, adaln = inputs(lora)
    expected = original(x, emb, context, attention, adaln_lora_B_T_3D=adaln)
    actual = projected(x, emb, context, attention, adaln_lora_B_T_3D=adaln)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    variables = (x, emb, context, adaln) if lora else (x, emb, context)
    reference = torch.autograd.grad(expected.square().mean(), variables, retain_graph=True)
    observed = torch.autograd.grad(actual.square().mean(), variables)
    for actual_grad, expected_grad in zip(observed, reference):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['none', 'standard', 'cpu_offload'])
def test_narrow_tensors_conditioning_complement_and_optimizer(mode):
    original = source()
    original.gradient_checkpoint_mode = mode
    candidate = project_block(original, [1])
    assert candidate.student.x_dim == 8 and candidate.student.self_attn.head_dim == 8
    assert candidate.student.self_attn.n_heads == 1
    assert sum(p.numel() for p in candidate.parameters()) < sum(p.numel() for p in original.parameters())
    indices = candidate.channels
    torch.testing.assert_close(candidate.student.cross_attn.k_proj.weight, original.cross_attn.k_proj.weight[indices])
    modulation = torch.cat([indices + part*16 for part in range(3)])
    torch.testing.assert_close(candidate.student.adaln_modulation_self_attn[2].weight,
                               original.adaln_modulation_self_attn[2].weight[modulation])
    x, emb, context, attention, adaln = inputs()
    output = candidate(x, emb, context, attention, adaln_lora_B_T_3D=adaln)
    torch.testing.assert_close(output[..., :8], x[..., :8], rtol=0, atol=0)
    assert not torch.equal(output[..., 8:], x[..., 8:])
    before = candidate.student.mlp.layer2.weight.detach().clone()
    optimizer = torch.optim.SGD(candidate.parameters(), lr=.1)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
    optimizer.step()
    assert not torch.equal(before, candidate.student.mlp.layer2.weight)
    restored = project_block(original, [1])
    restored.load_state_dict(candidate.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, emb, context, attention, adaln_lora_B_T_3D=adaln),
                               candidate(x, emb, context, attention, adaln_lora_B_T_3D=adaln), rtol=0, atol=0)


@pytest.mark.parametrize('heads', [[], [True], [2], [-1], [0, 0], '0'])
def test_invalid_head_selection(heads):
    with pytest.raises(ValueError):
        project_block(source(), heads)


def test_invalid_neurons_and_incompatible_execution():
    original = source()
    with pytest.raises(ValueError, match='MLP selection'):
        project_block(original, [0], mlp_neurons=[0])
    candidate = project_block(original, [0])
    x, emb, context, attention, adaln = inputs()
    with pytest.raises(ValueError, match='shift/scale/gate'):
        candidate(x, emb, context, attention)
    with pytest.raises(ValueError, match='without extra positions'):
        candidate(x, emb, context, attention, adaln_lora_B_T_3D=adaln, extra_per_block_pos_emb=torch.zeros_like(x))
    original.set_offload_recompute_callback(lambda: None)
    with pytest.raises(ValueError, match='offload'):
        project_block(original, [0])


@pytest.mark.parametrize('heads', [[0, 1], [1]])
def test_trainable_bridge_initial_parity_rng_and_dense_output_gradient(heads):
    original = source()
    selected = project_block(original, heads)
    rng = torch.get_rng_state().clone()
    bridged = project_block(original, heads, trainable_bridges=True)
    assert torch.equal(rng, torch.get_rng_state())
    x, emb, context, attention, adaln = inputs()
    reference = selected(x, emb, context, attention, adaln_lora_B_T_3D=adaln)
    output = bridged(x, emb, context, attention, adaln_lora_B_T_3D=adaln)
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
    optimizer = torch.optim.SGD(bridged.parameters(), lr=.05)
    output.square().mean().backward()
    assert bridged.encoder.grad is not None and torch.isfinite(bridged.encoder.grad).all()
    assert bridged.decoder.grad.abs().sum() > 0
    if heads == [1]:
        assert bridged.decoder.grad[:8].abs().sum() > 0
        old_output = output.detach().clone()
        for param in bridged.parameters():
            if param is not bridged.decoder:
                param.grad = None
        optimizer.step()
        updated = bridged(x, emb, context, attention, adaln_lora_B_T_3D=adaln)
        assert not torch.equal(updated[..., :8], old_output[..., :8])
    restored = project_block(original, heads, trainable_bridges=True)
    restored.load_state_dict(bridged.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, emb, context, attention, adaln_lora_B_T_3D=adaln),
                               bridged(x, emb, context, attention, adaln_lora_B_T_3D=adaln), rtol=0, atol=0)
