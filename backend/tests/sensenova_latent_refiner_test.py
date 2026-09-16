import copy
from types import SimpleNamespace

import pytest
import torch

from core.models.sensenova.latent_refiner import (
    REFINER_INPUTS,
    REFINER_NORM,
    LatentRefiner,
    apply_latent_refiner,
    validate_gen_refiner_declaration,
)
from core.training.adapters.sensenova_adapter import SenseNovaFullParameterAdapter
from core.training.arch.sensenova import SenseNovaArchHandler
from sensenova_full_finetune_adapter_test import (
    _attach_small_fm_modules,
    _full_ft_trainer,
    _materialized,
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


def _training_adapter(mode):
    branch = "gen"
    transformer = _materialized(branch)
    _attach_small_fm_modules(transformer)
    refiner = LatentRefiner(4, 16, 1).to(torch.bfloat16)
    transformer.fm_modules["fm_refiner"] = refiner
    trainer = _full_ft_trainer(
        branch,
        transformer,
        sensenova_train_fm_modules=True,
        sensenova_refiner_training_mode=mode,
        sensenova_refiner_lr_factor=0.25,
        learning_rate=1e-6,
    )
    trainer.config["sensenova_full_finetune_save_format"] = "mixed"
    return SenseNovaFullParameterAdapter(trainer), transformer, refiner


@pytest.mark.parametrize(
    "mode,refiner_trainable,base_trainable",
    [("joint", True, True), ("refiner_only", True, False), ("base_only", False, True)],
)
def test_training_modes_have_disjoint_parameter_boundaries(
    mode, refiner_trainable, base_trainable
):
    adapter, transformer, refiner = _training_adapter(mode)
    adapter.prepare_models_for_training()
    groups = adapter.arch_param_groups()
    selected = {id(parameter) for group in groups for parameter in group["params"]}
    refiner_ids = {id(parameter) for parameter in refiner.parameters()}
    base_ids = {
        id(parameter)
        for name, module in transformer.fm_modules.items()
        if name != "fm_refiner"
        for parameter in module.parameters()
    }
    assert bool(selected & refiner_ids) is refiner_trainable
    assert bool(selected & base_ids) is base_trainable
    assert all(parameter.requires_grad is refiner_trainable for parameter in refiner.parameters())


def test_refiner_only_uses_its_lr_factor_and_one_group():
    adapter, _transformer, refiner = _training_adapter("refiner_only")
    adapter.prepare_models_for_training()
    groups = adapter.arch_param_groups()
    assert [group["name"] for group in groups] == ["generation_refiner"]
    assert groups[0]["lr"] == pytest.approx(2.5e-7)
    assert {id(parameter) for parameter in groups[0]["params"]} == {
        id(parameter) for parameter in refiner.parameters()
    }


def _resolver_trainer(*, module=None, declaration=None, **config):
    transformer = torch.nn.Module()
    transformer.fm_modules = torch.nn.ModuleDict()
    if module is not None:
        transformer.fm_modules["fm_refiner"] = module
    transformer.gen_in_channels = 4
    transformer.gen_patch_size = 8
    transformer.gen_vit_patch_size = 4
    transformer.gen_vae_scale_factor = 8
    transformer.patch_size = 2
    transformer.downsample_ratio = 0.5
    settings = {
        "sensenova_latent_refiner": "inherit",
        "sensenova_refiner_training_mode": "joint",
        "train_unet": True,
        "gradient_accumulation_steps": 1,
    }
    settings.update(config)
    return SimpleNamespace(
        transformer=transformer,
        config=settings,
        sensenova_config_dict={
            "gen_in_channels": 4,
            **({"gen_refiner": declaration} if declaration is not None else {}),
        },
        sensenova_checkpoint_step=20,
        trains_base_weights=True,
        weight_dtype=torch.bfloat16,
    )


def test_refiner_only_without_a_refiner_is_refused_instead_of_training_the_base():
    trainer = _resolver_trainer(sensenova_refiner_training_mode="refiner_only")
    with pytest.raises(ValueError, match="requires an attached checkpoint refiner"):
        SenseNovaArchHandler().resolve_latent_refiner(trainer)


def test_anneal_resume_preserves_its_original_clock_and_checks_accumulation():
    module = LatentRefiner(4, 16, 1)
    declaration = _declaration(depth=1)
    declaration.update({
        "detach_anchor_step": 10,
        "detach_steps": 20,
        "detach_accum": 1,
    })
    trainer = _resolver_trainer(
        module=module,
        declaration=declaration,
        sensenova_latent_refiner="detach",
        sensenova_refiner_detach_steps=999,
    )
    SenseNovaArchHandler().resolve_latent_refiner(trainer)
    assert trainer.sensenova_config_dict["gen_refiner"] == declaration
    assert trainer.sensenova_refiner_state == "anneal"
    assert trainer.sensenova_refiner_training_mode == "base_only"

    changed = _resolver_trainer(
        module=copy.deepcopy(module),
        declaration=declaration,
        gradient_accumulation_steps=2,
    )
    with pytest.raises(ValueError, match="cannot change gradient_accumulation_steps"):
        SenseNovaArchHandler().resolve_latent_refiner(changed)


def test_inherit_refuses_a_shape_override_on_an_attached_refiner():
    module = LatentRefiner(4, 16, 1)
    trainer = _resolver_trainer(
        module=module,
        declaration=_declaration(depth=1),
        sensenova_refiner_width=32,
    )
    with pytest.raises(ValueError, match="width cannot change"):
        SenseNovaArchHandler().resolve_latent_refiner(trainer)
