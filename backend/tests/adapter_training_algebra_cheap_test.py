"""The training engine for LoHa/LoKr: census, export, resume, refusals.

What the per-architecture gate
(``adapter_lycoris_training_roundtrip_cheap_test.py``) does not cover, because it
is architecture-independent: that every factor reaches the optimizer exactly
once through BOTH fused paths, that the exporter folds ``scalar`` away, that a
failed resume restores the layer's alpha, and that the two capability axes are
asked separately.

Run with:
    venv/Scripts/python.exe -m pytest \
        backend/tests/adapter_training_algebra_cheap_test.py -v
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lora_roundtrip_common import randomise_branch_tensors  # noqa: E402

from core.adapters import (  # noqa: E402
    LoHaLinearLayer, LoKrLinearLayer, LoRALinearLayer,
)
from core.adapters.capability import (  # noqa: E402
    AXIS_GENERATION, AXIS_TRAINING, ENABLED_ADAPTER_PAIRS,
    TRAINABLE_ADAPTER_PAIRS, adapter_training_refusal_reason,
)
from core.adapters.layers import new_adapter_branch  # noqa: E402
from core.training.adapters.base_adapter import (  # noqa: E402
    LORA_COMPONENT_UNET, BaseLoRAAdapter, TrainingAdapterSpec,
    resolve_training_adapter_spec,
)
from core.training.arch import ARCH_REGISTRY  # noqa: E402

D = 8
RANK = 4
ALPHA = 6

#: factors per branch, by algebra. LoKr's default form is full ``w1`` plus a
#: factored ``w2``, so three.
FACTOR_COUNT = {"lora": 2, "loha": 4, "lokr": 3}
ALGEBRAS = sorted(FACTOR_COUNT)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(D, D, bias=False)
        self.to_v = nn.Linear(D, D, bias=False)

    def forward(self, x):
        return self.to_q(x) + self.to_v(x)


class _Adapter(BaseLoRAAdapter):
    """The smallest real adapter: it goes through the shared ``build_branch``."""

    def apply_lora_to_unet(self, lora_layers):
        model = self.trainer.transformer
        for name in ("to_q", "to_v"):
            layer = self.build_branch(getattr(model, name), name)
            setattr(model, name, layer)
            self.register_lora_layer(lora_layers, name, layer,
                                     LORA_COMPONENT_UNET)
        return len(lora_layers)

    def apply_lora_to_text_encoders(self, lora_layers):
        return 0

    def setup_trainable_parameters(self, lora_layers):
        return self.component_param_groups(
            lora_layers, {LORA_COMPONENT_UNET: lambda: 1e-4})

    def checkpoint_metadata(self, lora_layers, step, epoch):
        return {"model_type": "test", "step": str(step), "epoch": str(epoch)}


def build(algorithm, options=None, seed=3):
    torch.manual_seed(0)
    model = _Block()
    trainer = SimpleNamespace(transformer=model, config={},
                              adapter_algorithm=algorithm,
                              adapter_config=options or {})
    adapter = _Adapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    randomise_branch_tensors(layers, seed=seed, std=0.2)
    return adapter, layers, model


# --- the spec -------------------------------------------------------------

def test_a_missing_field_normalizes_to_ordinary_lora():
    spec = resolve_training_adapter_spec(SimpleNamespace(config={}))
    assert (spec.algorithm, spec.weight_decompose) == ("lora", False)
    assert spec.is_ordinary_lora and spec.metadata() == {}


def test_the_train_config_section_is_not_a_second_source():
    """It would reach the layer without train_runner's preflight, which reads
    the network block."""
    trainer = SimpleNamespace(config={"adapter_algorithm": "lokr"})
    assert resolve_training_adapter_spec(trainer).algorithm == "lora"


@pytest.mark.parametrize("value", ["dora", "locon", "", None])
def test_an_unknown_algorithm_is_refused_by_name(value):
    if value in (None, ""):
        assert TrainingAdapterSpec(algorithm=value).is_ordinary_lora
        return
    with pytest.raises(ValueError, match="adapter_algorithm"):
        TrainingAdapterSpec(algorithm=value)


def test_weight_decompose_is_accepted_as_a_field_and_refused_as_a_value():
    with pytest.raises(ValueError, match="Phase 3"):
        TrainingAdapterSpec(algorithm="loha", weight_decompose=True)


@pytest.mark.parametrize("algorithm", ["loha", "lokr"])
def test_use_scalar_cannot_be_trained(algorithm):
    # It would be folded away at save and forced to 1 at load, so a resume
    # would rebuild a different layer.
    with pytest.raises(ValueError, match="use_scalar"):
        build(algorithm, {"use_scalar": True})


def test_an_option_the_algebra_does_not_have_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="decompose_both"):
        build("loha", {"decompose_both": True})


def test_lokr_options_reach_the_layer():
    _adapter, layers, _model = build("lokr", {"decompose_both": True})
    layer = layers["to_q"]
    assert layer.decompose_both and hasattr(layer, "lokr_w1_a")


# --- the optimizer census -------------------------------------------------

@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_every_factor_reaches_the_optimizer_exactly_once(algorithm):
    adapter, layers, _model = build(algorithm)
    groups = adapter.setup_trainable_parameters(layers)
    params = [p for group in groups for p in group["params"]]
    expected = FACTOR_COUNT[algorithm] * len(layers)
    assert len(params) == expected
    assert len({id(p) for p in params}) == expected, "a factor was handed over twice"

    # And every one of the layer's own parameters is among them: a factor the
    # census misses trains nothing while the loss falls normally.
    owned = {id(p) for layer in layers.values() for p in layer.parameters()
             if p.requires_grad}
    assert owned == {id(p) for p in params}


@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_the_alpha_of_a_lycoris_layer_is_not_an_optimizer_parameter(algorithm):
    _adapter, layers, _model = build(algorithm)
    layer = layers["to_q"]
    assert "alpha" not in [n for n, _p in layer.named_parameters()]
    if algorithm != "lora":
        assert "alpha" in layer.branch_tensors()


@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_fused_optimizer_groups_complete_and_clear_every_gradient(algorithm):
    from core.training.optimizers.fused_optimizer_groups import (
        FusedOptimizerGroups, create_optimizer_groups,
    )

    adapter, layers, model = build(algorithm)
    groups = adapter.setup_trainable_parameters(layers)
    params = [p for group in groups for p in group["params"]]
    optimizers = create_optimizer_groups(params, "adamw", 3, 1e-3)
    fused = FusedOptimizerGroups(optimizers, 0.0)
    fused.register_hooks()

    assert sum(fused.num_parameters_per_group) == len(params)
    assert len(fused.parameter_optimizer_map) == len(params), \
        "two groups claimed the same parameter"

    before = [p.detach().clone() for p in params]
    model(torch.randn(2, D)).pow(2).mean().backward()

    assert fused.step_incomplete_groups() == [], \
        "a group did not complete inside the backward"
    assert all(p.grad is None for p in params), \
        "the hooks did not clear the gradients they applied"
    moved = sum(1 for p, old in zip(params, before) if not torch.equal(p, old))
    assert moved == len(params), f"only {moved}/{len(params)} factors were updated"


@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_the_fused_backward_hooks_fire_once_per_factor(algorithm):
    """The other fused path: one per-parameter hook, Adafactor's ``step_param``."""
    from core.training.base_trainer import BaseTrainer
    from core.training.optimizer_factory import OptimizerFactory

    adapter, layers, model = build(algorithm)
    groups = adapter.setup_trainable_parameters(layers)
    params = [p for group in groups for p in group["params"]]
    optimizer = OptimizerFactory.create_optimizer("adafactor", groups, 1e-3)

    trainer = SimpleNamespace(
        log_prefix="[test]", optimizer=optimizer, use_grad_scaler=False,
        optimizer_schedule_free=False, optimizer_stochastic_rounding=False,
        optimizer_update_census=False, _fused_grad_norm=None,
        _update_census=None, use_fused_backward=False,
        _fused_backward_target_module=lambda: model)
    BaseTrainer._setup_fused_backward_pass(trainer, "adafactor")
    assert trainer.use_fused_backward

    applied = []
    inner = optimizer.step_param
    optimizer.step_param = lambda p, pg: (applied.append(id(p)), inner(p, pg))[1]

    before = [p.detach().clone() for p in params]
    model(torch.randn(2, D)).pow(2).mean().backward()

    assert sorted(applied) == sorted(id(p) for p in params), \
        "a factor's hook never fired, or fired twice"
    assert all(p.grad is None for p in params)
    assert all(not torch.equal(p, old) for p, old in zip(params, before))


# --- export and resume ----------------------------------------------------

@pytest.mark.parametrize("algorithm,folded", [("loha", "hada_w1_a"),
                                              ("lokr", "lokr_w1")])
def test_export_folds_scalar_into_the_first_factor_and_drops_the_key(algorithm,
                                                                     folded):
    """Upstream folds at save and forces ``scalar := 1`` at load, so emitting
    the key bare leaves every other reader ``1/scalar`` too strong."""
    base = nn.Linear(D, D, bias=False)
    cls = {"loha": LoHaLinearLayer, "lokr": LoKrLinearLayer}[algorithm]
    layer = cls(base, RANK, ALPHA, "x", torch.float32, use_scalar=True)
    with torch.no_grad():
        layer.scalar.fill_(0.25)

    exported = layer.export_tensors()
    assert "scalar" in layer.branch_tensors(), "the live view keeps it"
    assert "scalar" not in exported
    assert torch.equal(exported[folded],
                       layer.branch_tensors()[folded].detach() * 0.25)


@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_a_saved_checkpoint_carries_exactly_the_resume_tensors(algorithm,
                                                               tmp_path):
    from safetensors.torch import load_file

    adapter, layers, _model = build(algorithm)
    path = tmp_path / "a.safetensors"
    adapter.save_checkpoint(layers, 5, 1, path)
    saved = load_file(str(path))
    for name, layer in layers.items():
        for key in layer.branch_tensors():
            assert f"{name}.{key}" in saved, f"resume would refuse: {name}.{key}"


@pytest.mark.parametrize("algorithm", ["loha", "lokr"])
def test_a_failed_resume_restores_the_layers_alpha(algorithm, tmp_path):
    """``alpha`` is a freshly built VALUE, not a live buffer: a rollback that
    ``copy_``s into it restores a throwaway and leaves the scale moved."""
    from core.training.lora_trainer import LoRATrainer

    adapter, layers, _model = build(algorithm)
    path = tmp_path / "a.safetensors"
    adapter.save_checkpoint(layers, 5, 1, path)

    _a2, resumed, _m2 = build(algorithm, seed=99)
    for layer in resumed.values():
        layer.set_alpha(99.0)
    alphas_before = [layer.alpha for layer in resumed.values()]

    # The second layer's load raises after the first has landed.
    victim = list(resumed.values())[1]
    victim.load_tensors = lambda tensors: (_ for _ in ()).throw(
        RuntimeError("boom"))
    trainer = SimpleNamespace(lora_layers=resumed, log_prefix="[test]",
                              lora_rank=RANK, lora_alpha=ALPHA)
    with pytest.raises(RuntimeError, match="boom"):
        LoRATrainer.load_checkpoint(trainer, str(path))

    assert [layer.alpha for layer in resumed.values()] == alphas_before


# --- the two capability axes ----------------------------------------------

def test_require_will_not_answer_without_an_axis():
    capability = ARCH_REGISTRY["krea2"].adapter_capability
    with pytest.raises(TypeError):
        capability.require("loha", False)


def test_the_training_axis_is_a_subset_of_the_generation_one():
    for arch, trainable in TRAINABLE_ADAPTER_PAIRS.items():
        assert trainable <= ENABLED_ADAPTER_PAIRS[arch], arch


@pytest.mark.parametrize("arch", sorted(TRAINABLE_ADAPTER_PAIRS))
def test_every_architecture_can_train_ordinary_lora(arch):
    capability = ARCH_REGISTRY[arch].adapter_capability
    assert capability.supports("lora", False, AXIS_TRAINING)


@pytest.mark.parametrize("arch", ["minimax_h3", "sensenova"])
def test_a_generation_flip_does_not_open_training(arch):
    """These two LOAD a LoHa and do not train one; the axes must disagree."""
    capability = ARCH_REGISTRY[arch].adapter_capability
    assert capability.supports("loha", False, AXIS_GENERATION)
    assert not capability.supports("loha", False, AXIS_TRAINING)
    with pytest.raises(ValueError, match="cannot be trained"):
        capability.require("loha", False, AXIS_TRAINING)
    capability.require("loha", False, AXIS_GENERATION)  # still fine


def test_an_architecture_that_loads_nothing_reports_the_load_refusal():
    """sd15 cannot train a LoHa BECAUSE it cannot load one; saying only
    'training is not enabled' would send the reader looking in the wrong place."""
    reason = adapter_training_refusal_reason("sd15", "loha", False)
    assert "not enabled" in reason and "sd15" in reason


def test_the_adapter_refuses_an_algebra_its_architecture_cannot_train():
    handler = ARCH_REGISTRY["sensenova"]
    trainer = SimpleNamespace(transformer=_Block(), config={},
                              adapter_algorithm="loha", arch=handler)
    with pytest.raises(ValueError, match="cannot be trained"):
        _Adapter(trainer, RANK, ALPHA, torch.float32)


def test_an_ordinary_lora_run_is_unaffected_by_the_capability_check():
    handler = ARCH_REGISTRY["sensenova"]
    trainer = SimpleNamespace(transformer=_Block(), config={}, arch=handler)
    adapter = _Adapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    assert all(isinstance(layer, LoRALinearLayer) for layer in layers.values())


def test_the_branch_factory_refuses_weight_decomposition():
    with pytest.raises(ValueError, match="Phase 3"):
        new_adapter_branch("lora", nn.Linear(D, D), rank=RANK, alpha=ALPHA,
                           weight_decompose=True)


# --- the run-level contract, refused before the model loads ---------------

def _contract(network, method="lora", architecture="krea2", train=None):
    from core.training import train_runner

    class _Loader:
        @staticmethod
        def detect_model_type(_path):
            return architecture

    import core.model_loader as model_loader
    original = model_loader.ModelLoader
    model_loader.ModelLoader = _Loader
    try:
        train_runner._assert_adapter_algebra_contract(
            method, network, "model", train or {})
    finally:
        model_loader.ModelLoader = original


def test_a_yaml_with_no_network_algebra_keys_is_an_ordinary_lora_run():
    _contract({"type": "lora", "linear": 16})


def test_the_contract_accepts_an_enabled_pair():
    _contract({"adapter_algorithm": "lokr"}, architecture="krea2")


def test_the_contract_refuses_an_architecture_that_cannot_train_it():
    with pytest.raises(ValueError, match="sensenova"):
        _contract({"adapter_algorithm": "loha"}, architecture="sensenova")


@pytest.mark.parametrize("method", ["relora", "full_finetune", "controlnet"])
def test_the_contract_refuses_every_method_but_lora(method):
    with pytest.raises(ValueError, match="network.type='lora' only"):
        _contract({"adapter_algorithm": "loha"}, method=method)


def test_the_contract_refuses_weight_decomposition_by_name():
    with pytest.raises(ValueError, match="Phase 3"):
        _contract({"adapter_algorithm": "lora", "weight_decompose": True})


def test_the_contract_refuses_an_unknown_algebra():
    with pytest.raises(ValueError, match="adapter_algorithm"):
        _contract({"adapter_algorithm": "locon"})


def test_block_swap_is_refused_before_the_model_loads():
    """The config preflight sees blocks_to_swap, so the refusal does not wait
    for the checkpoint to be resident."""
    with pytest.raises(ValueError, match="blocks_to_swap"):
        _contract({"adapter_algorithm": "loha"}, train={"blocks_to_swap": 8})
    _contract({"adapter_algorithm": "loha"}, train={"blocks_to_swap": 0})
    # Ordinary LoRA still swaps.
    _contract({"adapter_algorithm": "lora"}, train={"blocks_to_swap": 8})


def test_a_stale_adapter_config_is_refused_before_the_model_loads():
    """Even on an ordinary LoRA run, where nothing else would ever read it."""
    with pytest.raises(ValueError, match="factor"):
        _contract({"adapter_algorithm": "lora", "adapter_config": {"factor": 8}})
    with pytest.raises(ValueError, match="decompose_both"):
        _contract({"adapter_algorithm": "loha",
                   "adapter_config": {"decompose_both": True}})
    _contract({"adapter_algorithm": "lokr", "adapter_config": {"factor": 8}})


def test_the_specific_training_refusal_reaches_the_capability_payload():
    """The bespoke sentence is in core.adapters, which api may import; an
    ArchHandler's is not reachable from there (arch_capabilities must not pull
    the trainer stack in), and a sentence no client can read is not a reason."""
    from api.arch_capabilities import adapter_families_payload

    payload = adapter_families_payload()
    for arch in ("minimax_h3", "sensenova"):
        served = payload[arch]["untrainable"]["loha"]
        assert "gate of their own" in served, arch
        # ...and the handler renders the same words, from the same table.
        assert served == ARCH_REGISTRY[arch].adapter_capability.refusal_reason(
            "loha", False, AXIS_TRAINING)


def test_block_swap_is_refused_with_a_lycoris_algebra():
    """No offloader moves a bare parameter, and what that costs a TRAINING step
    is unmeasured, so the combination is refused rather than run."""
    from core.training.lora_trainer import require_trainable_algebra

    handler = ARCH_REGISTRY["krea2"]
    trainer = SimpleNamespace(config={}, adapter_algorithm="loha",
                              blocks_to_swap=8)
    with pytest.raises(ValueError, match="blocks_to_swap"):
        require_trainable_algebra(trainer, handler)

    trainer.blocks_to_swap = 0
    require_trainable_algebra(trainer, handler)

    # An ordinary LoRA still swaps, on every architecture.
    require_trainable_algebra(
        SimpleNamespace(config={}, blocks_to_swap=8), handler)
