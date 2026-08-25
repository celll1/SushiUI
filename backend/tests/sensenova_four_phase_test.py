"""U-2-4: the four-phase eviction split of SENSENOVA_TRAINING_DESIGN.md 8.3.2.

The acceptance criterion is gradient parity: the understanding-side gradient the
split produces must equal the one a single ``loss.backward()`` produces. A split
that merely returns finite numbers is worthless, so every parity test here is
paired with a NEGATIVE CONTROL that runs the same tree with the mechanism
disabled and asserts the specific wrong answer it gives.

TOLERANCE. Zero -- bitwise. Derived, not chosen: the split performs the same
operations on the same values in the same order as the single backward. The
recomputed understanding forward is the same function applied to the same inputs
with the same weights (``attention_dropout`` is asserted zero elsewhere so it is
deterministic, the property gradient checkpointing already depends on), so it
reproduces the original bitwise; the boundary gradient the generation backward
deposits in ``leaf.grad`` is an accumulation into an absent buffer, i.e. the
value itself; and feeding that value into ``autograd.backward`` runs the same
understanding backward graph the single call would have run. Nothing rounds
twice and nothing is reordered, so any difference at all is a defect rather than
float noise. ``float64`` here keeps that argument checkable without a GPU;
``float32`` is asserted bitwise too, and both hold.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.param_defaults import TRAINING_DEFAULTS
from core.models.sensenova.mot_weight_selector import select_mot_weight_modules
from core.training.ops import sensenova_ops
from core.training.sensenova_four_phase import (
    SenseNovaFourPhaseBackward,
    install_four_phase_backward,
)
from core.training.sensenova_phase_eviction import SenseNovaTrainingPhaseEvictor


# --------------------------------------------------------------------------
# A synthetic stand-in for the und/gen pair: stage A produces "K/V", stage B
# consumes it. Small enough to differentiate exactly, shaped like the real cut.
# --------------------------------------------------------------------------


class UnderstandingStage(nn.Module):
    """Produces one (keys, values) pair per layer from the prompt tokens."""

    def __init__(self, layers: int, width: int, dtype: torch.dtype):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(width, width, bias=False, dtype=dtype) for _ in range(layers)]
        )
        self.k = nn.ModuleList(
            [nn.Linear(width, width, bias=False, dtype=dtype) for _ in range(layers)]
        )
        self.v = nn.ModuleList(
            [nn.Linear(width, width, bias=False, dtype=dtype) for _ in range(layers)]
        )

    def forward(self, tokens: torch.Tensor):
        hidden = tokens
        out = []
        for block, k, v in zip(self.blocks, self.k, self.v):
            hidden = torch.tanh(block(hidden))
            out.append((k(hidden), v(hidden)))
        return out


class GenerationStage(nn.Module):
    def __init__(self, layers: int, width: int, dtype: torch.dtype):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(width, width, bias=False, dtype=dtype) for _ in range(layers)]
        )

    def forward(self, image: torch.Tensor, cache):
        hidden = image
        for block, (keys, values) in zip(self.blocks, cache):
            attention = torch.softmax(hidden @ keys.transpose(-1, -2), dim=-1) @ values
            hidden = torch.tanh(block(hidden) + attention)
        return hidden


class Pair(nn.Module):
    def __init__(self, layers=3, width=4, dtype=torch.float64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.und = UnderstandingStage(layers, width, dtype)
        self.gen = GenerationStage(layers, width, dtype)


def _inputs(width=4, dtype=torch.float64, seed=7):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(5, width, generator=generator, dtype=dtype)
    image = torch.randn(6, width, generator=generator, dtype=dtype)
    target = torch.randn(6, width, generator=generator, dtype=dtype)
    return tokens, image, target


def _und_grads(model: Pair):
    return {name: p.grad.clone() for name, p in model.und.named_parameters()}


def _single_backward(model: Pair, tokens, image, target):
    model.zero_grad(set_to_none=True)
    cache = model.und(tokens)
    loss = ((model.gen(image, cache) - target) ** 2).mean()
    loss.backward()
    return float(loss), _und_grads(model)


def _split_backward(model: Pair, tokens, image, target, *, cut=True):
    """The three-phase sequence. ``cut=False`` is the negative control."""
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        cache = model.und(tokens)
    if cut:
        leaves = [
            (k.detach().requires_grad_(True), v.detach().requires_grad_(True))
            for k, v in cache
        ]
    else:
        # The defect: the boundary is handed on DETACHED. The generation loss is
        # identical and finite, and the understanding half receives nothing.
        leaves = [(k.detach(), v.detach()) for k, v in cache]
    loss = ((model.gen(image, leaves) - target) ** 2).mean()
    loss.backward()
    flat_leaves = [t for pair in leaves for t in pair]
    if any(t.grad is None for t in flat_leaves):
        return float(loss), None
    grads = [t.grad.clone() for t in flat_leaves]
    recomputed = model.und(tokens)
    tensors = [t for pair in recomputed for t in pair]
    torch.autograd.backward(tensors, grad_tensors=grads)
    return float(loss), _und_grads(model)


# --------------------------------------------------------------------------
# (D) gradient parity + negative control
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_split_backward_matches_single_backward_bitwise(dtype):
    tokens, image, target = _inputs(dtype=dtype)
    reference = Pair(dtype=dtype)
    split = Pair(dtype=dtype)
    split.load_state_dict(reference.state_dict())

    single_loss, single_grads = _single_backward(reference, tokens, image, target)
    split_loss, split_grads = _split_backward(split, tokens, image, target)

    assert split_loss == single_loss
    assert set(split_grads) == set(single_grads)
    for name, expected in single_grads.items():
        assert torch.equal(split_grads[name], expected), (
            f"{name}: max abs diff "
            f"{(split_grads[name] - expected).abs().max().item()}"
        )


def test_negative_control_uncut_boundary_trains_nothing_while_the_loss_is_fine():
    """Without the cut the loss is identical and the understanding half is dead."""
    tokens, image, target = _inputs()
    reference = Pair()
    broken = Pair()
    broken.load_state_dict(reference.state_dict())

    single_loss, single_grads = _single_backward(reference, tokens, image, target)
    broken_loss, broken_grads = _split_backward(broken, tokens, image, target, cut=False)

    assert broken_loss == single_loss
    assert broken_grads is None
    assert all(p.grad is None for p in broken.und.parameters())
    assert any(g.abs().max() > 0 for g in single_grads.values())


def test_negative_control_skipping_phase_three_leaves_the_und_half_unupdated():
    tokens, image, target = _inputs()
    model = Pair()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        cache = model.und(tokens)
    leaves = [
        (k.detach().requires_grad_(True), v.detach().requires_grad_(True))
        for k, v in cache
    ]
    loss = ((model.gen(image, leaves) - target) ** 2).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.gen.parameters())
    assert all(p.grad is None for p in model.und.parameters())


def test_recomputed_forward_reproduces_its_own_forward_bitwise():
    """The parity argument's premise, checked rather than assumed."""
    tokens, _, _ = _inputs()
    model = Pair()
    with torch.no_grad():
        first = model.und(tokens)
        second = model.und(tokens)
    for (k1, v1), (k2, v2) in zip(first, second):
        assert torch.equal(k1, k2)
        assert torch.equal(v1, v2)


# --------------------------------------------------------------------------
# (C) the graph-cut helper the production path uses
# --------------------------------------------------------------------------


class _Cache:
    def __init__(self, layers):
        self.layers = layers


class _Layer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


def _fake_trainer():
    trainer = type("T", (), {})()
    trainer.transformer = None
    return trainer


def test_cut_returns_leaves_and_capture_takes_their_gradient():
    context = SenseNovaFourPhaseBackward(_fake_trainer())
    source = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    graph_keys = source * 2
    graph_values = source * 3
    cache = context.cut(_Cache([_Layer(graph_keys, graph_values)]), ("inputs",))

    leaf_keys = cache.layers[0].keys
    leaf_values = cache.layers[0].values
    assert leaf_keys.is_leaf and leaf_keys.requires_grad and leaf_keys.grad_fn is None
    assert torch.equal(leaf_keys, graph_keys.detach())

    (leaf_keys.sum() + 2 * leaf_values.sum()).backward()
    context.capture()
    assert context.pending_count == 1
    assert leaf_keys.grad is None  # consumed, not left to be double-counted


def test_capture_refuses_a_boundary_the_generation_backward_never_reached():
    context = SenseNovaFourPhaseBackward(_fake_trainer())
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("inputs",))
    with pytest.raises(RuntimeError, match="left no gradient"):
        context.capture()


def test_cut_refuses_a_second_cut_before_capture():
    context = SenseNovaFourPhaseBackward(_fake_trainer())
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    with pytest.raises(RuntimeError, match="never captured"):
        context.cut(_Cache([_Layer(keys, keys)]), ("b",))


def test_flush_refuses_while_a_cut_is_outstanding():
    context = SenseNovaFourPhaseBackward(_fake_trainer())
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("a",))
    context._pending.append((("b",), [torch.zeros(2, 3)]))
    with pytest.raises(RuntimeError, match="uncaptured boundary cut"):
        context.flush()


def test_install_four_phase_backward_binds_to_the_trainer():
    trainer = _fake_trainer()
    context = install_four_phase_backward(trainer)
    assert trainer.sensenova_four_phase is context


# --------------------------------------------------------------------------
# The boundary-leaf assertion (replaces grad_fn for the cut cache)
# --------------------------------------------------------------------------


def test_boundary_leaf_assertion_accepts_leaves_and_rejects_both_failure_modes():
    leaf = torch.randn(2, 3, requires_grad=True)
    sensenova_ops._assert_prefix_cache_boundary_leaf(_Cache([_Layer(leaf, leaf)]))

    detached = torch.randn(2, 3)
    with pytest.raises(ValueError, match="grad-requiring LEAVES"):
        sensenova_ops._assert_prefix_cache_boundary_leaf(
            _Cache([_Layer(detached, detached)])
        )

    graph = torch.randn(2, 3, requires_grad=True) * 2
    with pytest.raises(ValueError, match="grad-requiring LEAVES"):
        sensenova_ops._assert_prefix_cache_boundary_leaf(_Cache([_Layer(graph, graph)]))


def test_immutable_prefix_cache_routes_boundary_leaf_past_the_grad_fn_check():
    leaf = torch.randn(2, 3, requires_grad=True)
    cache = _Cache([_Layer(leaf, leaf)])
    sensenova_ops._assert_immutable_prefix_cache(cache, 1, boundary_leaf=True)
    # The single-backward assertion still refuses it -- a leaf there means the
    # understanding half silently receives nothing.
    with pytest.raises(ValueError, match="carry no"):
        sensenova_ops._assert_immutable_prefix_cache(cache, 1, trainable=True)


# --------------------------------------------------------------------------
# (B) the evictor's fourth phase
# --------------------------------------------------------------------------


class _Half(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))


class _EvictLayer(nn.Module):
    def __init__(self, with_und_lora=False):
        super().__init__()
        self.proj = _Half()
        self.proj_mot_gen = _Half()
        if with_und_lora:
            self.proj.lora_down = nn.Linear(2, 1, bias=False)
            self.proj.lora_up = nn.Linear(1, 2, bias=False)


def _evict_transformer(*, count=42, with_und_lora=False):
    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.model = nn.Module()
    root.language_model.model.layers = nn.ModuleList(
        [_EvictLayer(with_und_lora=with_und_lora) for _ in range(count)]
    )
    return root


def _evictor(**kwargs):
    return SenseNovaTrainingPhaseEvictor(_evict_transformer(), "cpu", **kwargs)


def test_four_phase_state_machine_walks_the_designed_cycle():
    evictor = _evictor(four_phase=True)
    assert evictor.state == "full"
    evictor.enter_prefix()
    assert evictor.state == "prefix"
    evictor.assert_understanding_resident()
    evictor.enter_denoise()
    assert evictor.state == "denoise"
    evictor.assert_generation_resident()
    evictor.enter_und_backward()
    assert evictor.state == "und_backward"
    evictor.assert_understanding_resident()


def test_und_backward_to_prefix_is_the_designed_no_op():
    evictor = _evictor(four_phase=True)
    evictor.enter_prefix()
    evictor.enter_denoise()
    evictor.enter_und_backward()
    moves = []
    evictor._transition = lambda *a, **k: moves.append(a)
    evictor.enter_prefix()
    assert evictor.state == "prefix"
    assert moves == []


def test_three_state_evictor_refuses_the_fourth_phase():
    evictor = _evictor()
    evictor.enter_prefix()
    evictor.enter_denoise()
    with pytest.raises(RuntimeError, match="requires the four-phase evictor"):
        evictor.enter_und_backward()


def test_und_backward_requires_a_completed_denoise():
    evictor = _evictor(four_phase=True)
    evictor.enter_prefix()
    with pytest.raises(RuntimeError, match="requires a completed denoise"):
        evictor.enter_und_backward()


def test_assert_understanding_resident_refuses_the_denoise_phase():
    evictor = _evictor(four_phase=True)
    evictor.enter_prefix()
    evictor.enter_denoise()
    with pytest.raises(
        RuntimeError, match="understanding work requires prefix or und_backward state"
    ):
        evictor.assert_understanding_resident()


def test_teardown_accepts_the_new_state():
    evictor = _evictor(four_phase=True)
    evictor.enter_prefix()
    evictor.enter_denoise()
    evictor.enter_und_backward()
    evictor.teardown()
    assert evictor.state == "closed"


# --------------------------------------------------------------------------
# (E) census x four-phase ordering -- 12 records this as UNTESTED
# --------------------------------------------------------------------------


def test_eviction_refuses_a_half_whose_gradients_the_hooks_never_consumed():
    """The failure 8.3's table names: a fused hook skips a parameter, and the
    half is staged to CPU anyway, so it is never updated while the loss falls."""
    evictor = _evictor(four_phase=True)
    evictor.enter_prefix()
    evictor.enter_denoise()
    # A hook that ran leaves grad=None; this one did not run.
    evictor._gen_modules[0].weight.grad = torch.zeros(2)
    with pytest.raises(RuntimeError, match="still holds a gradient"):
        evictor.enter_und_backward()


def test_eviction_allows_the_half_once_the_hooks_have_nulled_the_gradients():
    evictor = _evictor(four_phase=True)
    evictor.enter_prefix()
    evictor.enter_denoise()
    for module in evictor._gen_modules:
        module.weight.grad = torch.zeros(2)
    for module in evictor._gen_modules:  # what the fused hook does after step_param
        module.weight.grad = None
    evictor.enter_und_backward()
    assert evictor.state == "und_backward"


def test_three_state_evictor_does_not_take_the_gradient_check():
    """Unchanged behaviour for the frozen-understanding route it serves."""
    evictor = _evictor()
    evictor.enter_prefix()
    evictor._und_modules[0].weight.grad = torch.zeros(2)
    evictor.enter_denoise()
    assert evictor.state == "denoise"


# --------------------------------------------------------------------------
# (B) the layer-selection discriminator, re-derived (8.4)
# --------------------------------------------------------------------------


class _Int8Like(nn.Module):
    """Owns no Parameter at all -- what a `parameters()` rule silently drops."""

    def __init__(self):
        super().__init__()
        self.register_buffer("qweight", torch.ones(2, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(1))


class _MixedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _Int8Like()       # frozen half, quantized
        self.proj_mot_gen = _Half()   # trained half, materialized


def test_persistence_discriminator_still_selects_the_quantized_frozen_half():
    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.model = nn.Module()
    root.language_model.model.layers = nn.ModuleList([_MixedLayer() for _ in range(42)])
    selection = select_mot_weight_modules(root)
    assert len(selection.und_modules) == 42
    assert len(selection.gen_modules) == 42
    # The rule that shipped inert twice would have selected zero of them.
    assert all(
        not list(module.parameters(recurse=False))
        for module in selection.und_modules
    )


def test_four_phase_selection_tolerates_understanding_adapters():
    root = _evict_transformer(with_und_lora=True)
    with pytest.raises(RuntimeError, match="asymmetric"):
        select_mot_weight_modules(root, require_exact_symmetry=True)
    selection = select_mot_weight_modules(
        root, require_exact_symmetry=True, allow_understanding_adapters=True
    )
    assert len(selection.gen_modules) == 42
    assert len(selection.und_modules) == 42 * 3  # base + lora_down + lora_up


# --------------------------------------------------------------------------
# (E) contract wiring
# --------------------------------------------------------------------------


class _ContractTrainer:
    def __init__(self, **config):
        self.config = dict(config)
        self.training_method = config.get("training_method", "full_finetune")
        for key, value in config.items():
            setattr(self, key, value)


def _full_ft_four_phase_config(**overrides):
    base = {
        "training_method": "full_finetune",
        "train_text_encoder": True,
        "sensenova_mot_phase_eviction": True,
        "sensenova_four_phase_eviction": True,
    }
    base.update(overrides)
    return base


def test_four_phase_contract_accepts_the_designed_configuration():
    sensenova_ops.assert_four_phase_contract(
        _ContractTrainer(**_full_ft_four_phase_config())
    )


@pytest.mark.parametrize(
    "override, message",
    [
        ({"train_text_encoder": False}, "requires train_text_encoder"),
        ({"sensenova_mot_phase_eviction": False}, "requires sensenova_mot_phase_eviction"),
        ({"training_method": "lora"}, "full_finetune"),
    ],
)
def test_four_phase_contract_refuses_each_missing_precondition(override, message):
    """``match=`` is load-bearing: without it any one clause firing for another
    clause's reason still passes, which is the whole point of testing three."""
    trainer = _ContractTrainer(**_full_ft_four_phase_config(**override))
    with pytest.raises(ValueError) as excinfo:
        sensenova_ops.assert_four_phase_contract(trainer)
    assert message in " ".join(str(excinfo.value).split())


def test_fused_backward_backstop_refuses_an_uninstalled_hook_path():
    trainer = _ContractTrainer(**_full_ft_four_phase_config())
    trainer.use_fused_backward = False
    with pytest.raises(RuntimeError, match="requires the fused backward pass"):
        sensenova_ops.assert_four_phase_fused_backward(trainer)
    trainer.use_fused_backward = True
    sensenova_ops.assert_four_phase_fused_backward(trainer)


def test_flag_is_opt_in_and_lives_in_param_defaults():
    assert TRAINING_DEFAULTS["sensenova_four_phase_eviction"] is False


def test_flag_reaches_the_product_rather_than_only_hand_written_yaml():
    """It is served by /schema/training-defaults, so a request must accept it and
    the spec must document it -- otherwise the endpoint advertises a key nothing
    consumes and the feature is reachable only from a hand-edited config."""
    import yaml

    from api.routes import TrainingRunCreateRequest

    assert "sensenova_four_phase_eviction" in TrainingRunCreateRequest.model_fields
    spec = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "openapi.yaml").read_text(
            encoding="utf-8"
        )
    )
    properties = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    assert "sensenova_four_phase_eviction" in properties
    assert properties["sensenova_four_phase_eviction"]["default"] is False


def test_config_builder_carries_the_flag_from_run_params():
    from core.training.training_config import _build_train_section

    section = _build_train_section(
        {"sensenova_four_phase_eviction": True},
        total_steps=None,
        epochs=1,
        train_unet=True,
        train_text_encoder=True,
        include_block_swap=True,
    )
    assert section["sensenova_four_phase_eviction"] is True


def test_mnt_cost_is_announced_rather_than_refused():
    trainer = _ContractTrainer(
        **_full_ft_four_phase_config(multi_noise_timesteps=4)
    )
    emitted = []
    original = sensenova_ops.emit_training_warning
    sensenova_ops.emit_training_warning = lambda message, **kw: emitted.append(kw)
    try:
        sensenova_ops.assert_four_phase_contract(trainer)
    finally:
        sensenova_ops.emit_training_warning = original
    assert [entry["code"] for entry in emitted] == ["sensenova_four_phase_mnt_cost"]

    assert not sensenova_ops.warn_four_phase_mnt_cost(
        _ContractTrainer(**_full_ft_four_phase_config())
    )


def test_installer_ignores_the_flag_off_the_full_finetune_route():
    """F7: BaseTrainer sets the attribute on every trainer and LoRATrainer calls
    this installer, so the symmetry backstop must not be relaxed by the flag
    alone when the front-line runner check did not run."""
    from core.training.sensenova_phase_eviction import install_training_phase_eviction

    trainer = type("T", (), {})()
    trainer.transformer = _evict_transformer()
    trainer.device = "cpu"
    trainer.sensenova_four_phase_eviction = True
    trainer.config = {"training_method": "lora"}
    evictor = install_training_phase_eviction(trainer)
    assert evictor.four_phase is False

    # trains_base_weights is the channel FullParameterTrainer actually sets.
    trainer.trains_base_weights = True
    assert install_training_phase_eviction(trainer).four_phase is True


def test_discard_clears_an_uncaptured_cut_so_the_next_batch_can_start():
    """F5: the recoverable-OOM skip abandons a batch whose prefix was already
    cut. Without discard() the next cut() raises, and that exception is not an
    OOM, so it kills a run the skip path exists to keep alive."""
    context = SenseNovaFourPhaseBackward(_fake_trainer())
    keys = torch.randn(2, 3, requires_grad=True) * 1.0
    context.cut(_Cache([_Layer(keys, keys)]), ("abandoned",))
    context.discard()
    context.cut(_Cache([_Layer(keys, keys)]), ("next batch",))
    assert context.pending_count == 0


def test_runner_lifts_the_refusal_only_when_four_phase_is_armed():
    # Patched, not relied on by path string: the runner's own detection reads the
    # checkpoint through ModelLoader and only falls back to the filename when that
    # raises, so a path-only test passes for the wrong reason wherever the file is
    # absent -- and would stop testing the runner at all if it were renamed.
    from unittest.mock import patch

    from core.model_loader import ModelLoader
    from core.training import train_runner

    def config(**overrides):
        base = {
            "batch_size": 1,
            "blocks_to_swap": 0,
            "train_unet": True,
            "train_text_encoder": True,
            "sensenova_mot_phase_eviction": True,
            "optimizer": "adafactor",
            "gradient_accumulation_steps": 1,
            "num_optimizer_groups": 0,
            "use_ema": False,
            "sensenova_full_finetune_save_format": "mixed",
        }
        base.update(overrides)
        return base

    path = "model"
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="cannot be combined"):
            train_runner._apply_sensenova_training_contract(
                path, "full_finetune", config(), {}
            )

        armed = config(sensenova_four_phase_eviction=True)
        assert train_runner._apply_sensenova_training_contract(
            path, "full_finetune", armed, {}
        )

        with pytest.raises(ValueError, match="network.type='full_finetune'"):
            train_runner._apply_sensenova_training_contract(
                path, "lora", config(sensenova_four_phase_eviction=True), {}
            )
