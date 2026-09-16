import copy

import pytest
import torch

from core.models.sensenova.latent_refiner import (
    REFINER_INPUTS,
    REFINER_NORM,
    LatentRefiner,
    apply_latent_refiner,
    validate_gen_refiner_declaration,
)


def _declaration(width=16, depth=2):
    return {
        "version": 1,
        "width": width,
        "depth": depth,
        "inputs": list(REFINER_INPUTS),
        "norm": REFINER_NORM,
        "detach_anchor_step": None,
        "detach_steps": None,
        "detach_accum": None,
    }


def _state(module):
    return {f"fm_modules.fm_refiner.{key}": value.detach().clone()
            for key, value in module.state_dict().items()}


def test_zero_init_is_exact_identity_and_batch_timesteps():
    module = LatentRefiner(4, 16, 2)
    x0 = torch.randn(3, 4, 12, 10)
    z = torch.randn_like(x0)
    actual = apply_latent_refiner(
        module, x0, z, torch.tensor([0.1, 0.5, 0.9]), 1.7
    )
    assert torch.equal(actual, x0)


def test_checkpointed_and_plain_gradients_match():
    source = LatentRefiner(4, 16, 2)
    torch.nn.init.normal_(source.out.weight, std=0.01)
    plain = copy.deepcopy(source)
    checked = copy.deepcopy(source)
    x0 = torch.randn(2, 4, 8, 8, requires_grad=True)
    z = torch.randn_like(x0)

    plain(x0, z, torch.tensor([0.2, 0.7]), 1.3).square().mean().backward()
    checked(x0.detach(), z, torch.tensor([0.2, 0.7]), 1.3,
            checkpoint_blocks=True).square().mean().backward()
    for left, right in zip(plain.parameters(), checked.parameters()):
        assert torch.allclose(left.grad, right.grad, rtol=1e-5, atol=1e-6)


def test_receptive_field_is_strictly_local():
    module = LatentRefiner(4, 16, 2)
    torch.nn.init.normal_(module.out.weight, std=0.01)
    x0 = torch.randn(1, 4, 25, 25)
    z = torch.randn_like(x0)
    changed = x0.clone()
    changed[:, :, 12, 12] += 1
    baseline = module(x0, z, torch.tensor(0.5), 1.0) - x0
    perturbed = module(changed, z, torch.tensor(0.5), 1.0) - changed
    difference = (baseline - perturbed).abs().amax(dim=1)[0]
    radius = module.receptive_field_radius
    yy, xx = torch.meshgrid(torch.arange(25), torch.arange(25), indexing="ij")
    outside = (yy - 12).abs().maximum((xx - 12).abs()) > radius
    assert torch.equal(difference[outside], torch.zeros_like(difference[outside]))


def test_declaration_and_tensor_payload_are_fail_closed():
    module = LatentRefiner(4, 16, 2)
    config = {"gen_in_channels": 4, "gen_refiner": _declaration()}
    state = _state(module)
    assert validate_gen_refiner_declaration(config, state) == config["gen_refiner"]

    missing = dict(state)
    missing.pop("fm_modules.fm_refiner.gate")
    with pytest.raises(ValueError, match="tensor set differs"):
        validate_gen_refiner_declaration(config, missing)

    undeclared = {"gen_in_channels": 4}
    with pytest.raises(ValueError, match="without gen_refiner declaration"):
        validate_gen_refiner_declaration(undeclared, state)


def test_anneal_gate_must_match_checkpoint_clock():
    module = LatentRefiner(4, 16, 1)
    module.gate.fill_(0.5)
    declaration = _declaration(depth=1)
    declaration.update({
        "detach_anchor_step": 10,
        "detach_steps": 20,
        "detach_accum": 1,
    })
    config = {"gen_in_channels": 4, "gen_refiner": declaration}
    validate_gen_refiner_declaration(config, _state(module), checkpoint_step=20)
    with pytest.raises(ValueError, match="disagrees with detach clock"):
        validate_gen_refiner_declaration(config, _state(module), checkpoint_step=21)
