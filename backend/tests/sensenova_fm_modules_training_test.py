"""`sensenova_train_fm_modules`: the 16 tensors a full fine-tune leaves frozen.

A SenseNova full fine-tune's scope is the set of decoder Linears the INT8 load
dequantized, which is a property of the quantization layout rather than a claim
about what is worth training. `transformer.fm_modules` -- the generation ViT
embeddings, the timestep and noise-scale embedders and the fm_head that emits
the pixel prediction -- is not quantized, so it is not materialized, so it never
reached the optimizer: two real checkpoints 4,960 steps apart hold all 16 of its
tensors byte-identical while the generation decoder moved 3.09e-3 relative.

This file pins the option that makes that a choice, and pins the guard it must
not weaken: the decoder-Linear count stays exactly
`SENSENOVA_BRANCH_LINEAR_COUNTS[branch]`, because that check is what catches an
unmaterialized INT8 module, and fm parameters are collected on a separate path.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_fm_modules_training_test.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensenova_full_finetune_adapter_test import (  # noqa: E402
    _collected,
    _full_ft_trainer,
    _materialized,
)

from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from core.training.adapters import SenseNovaFullParameterAdapter  # noqa: E402
from core.training.adapters.base_adapter import (  # noqa: E402
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_UNET,
)

# The distributed checkpoint's fm_modules index, tensor for tensor.
FM_TENSORS = 16


def _mlp() -> nn.Module:
    """`mlp.0` / `mlp.2`, the TimestepEmbedder layout."""
    embedder = nn.Module()
    embedder.mlp = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
    return embedder


def _attach_fm_modules(transformer: nn.Module) -> nn.Module:
    vision = nn.Module()
    vision.embeddings = nn.Module()
    vision.embeddings.patch_embedding = nn.Linear(4, 4)
    vision.embeddings.dense_embedding = nn.Linear(4, 4)
    head = nn.Module()
    head.conv1 = nn.Conv2d(2, 2, 1)
    head.conv2 = nn.Conv2d(2, 2, 1)
    transformer.fm_modules = nn.ModuleDict({
        "vision_model_mot_gen": vision,
        "timestep_embedder": _mlp(),
        "fm_head": head,
        "noise_scale_embedder": _mlp(),
    }).to(torch.bfloat16)
    return transformer


def _adapter(branch: str, *, fm: bool, attach: bool = True):
    transformer = _materialized(branch)
    if attach:
        _attach_fm_modules(transformer)
    trainer = _full_ft_trainer(branch, transformer, sensenova_train_fm_modules=fm)
    return SenseNovaFullParameterAdapter(trainer), transformer


def _fm_ids(transformer: nn.Module) -> set:
    return {id(p) for p in transformer.fm_modules.parameters()}


def test_the_option_is_off_by_default():
    assert TRAINING_DEFAULTS["sensenova_train_fm_modules"] is False


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL: the shipped behaviour, which the flag off must reproduce
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch,groups_expected", [("gen", 1), ("und", 1), ("both", 2)])
def test_off_leaves_fm_modules_frozen_and_uncollected(branch, groups_expected):
    adapter, transformer = _adapter(branch, fm=False)
    adapter.prepare_models_for_training()
    groups = adapter.setup_trainable_parameters()

    assert len(groups) == groups_expected
    tensors, _ = _collected(groups)
    assert tensors == 294 * groups_expected
    assert len(list(transformer.fm_modules.parameters())) == FM_TENSORS
    assert not any(p.requires_grad for p in transformer.fm_modules.parameters())
    assert not (_fm_ids(transformer) & {id(p) for g in groups for p in g["params"]})


def test_off_is_the_same_when_fm_modules_is_absent_entirely():
    """The flag off must not depend on the container existing."""
    adapter, transformer = _adapter("gen", fm=False, attach=False)
    adapter.prepare_models_for_training()
    assert _collected(adapter.setup_trainable_parameters()) == (294, 294 * 4 * 8)
    assert not hasattr(transformer, "fm_modules")


# ---------------------------------------------------------------------------
# On
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch", ["gen", "both"])
def test_on_adds_the_16_tensors_to_the_generation_group(branch):
    adapter, transformer = _adapter(branch, fm=True)
    adapter.prepare_models_for_training()
    groups = adapter.setup_trainable_parameters()

    fm_ids = _fm_ids(transformer)
    assert all(p.requires_grad for p in transformer.fm_modules.parameters())
    # Generation group first, and it is the ONLY group that carries them.
    assert len(groups[0]["params"]) == 294 + FM_TENSORS
    assert fm_ids <= {id(p) for p in groups[0]["params"]}
    assert groups[0]["lr"] == adapter.trainer.unet_lr
    for group in groups[1:]:
        assert not (fm_ids & {id(p) for p in group["params"]})
    # Counted once across the whole optimizer, not once per group.
    tensors, _ = _collected(groups)
    assert tensors == 294 * len(groups) + FM_TENSORS


def test_on_does_not_widen_the_decoder_linear_guard():
    """`_resolve_scope`'s exact count is the unmaterialized-INT8 detector."""
    adapter, transformer = _adapter("gen", fm=True)
    branch, targets = adapter._resolve_scope()
    assert branch == "gen" and len(targets) == 294
    assert not (_fm_ids(transformer) & {id(m) for _, _, _, m in targets})


def test_the_collected_fm_parameters_are_floating_point():
    """Asked of the tensors rather than assumed from "these are not INT8"."""
    adapter, transformer = _adapter("gen", fm=True)
    adapter.prepare_models_for_training()
    assert {p.dtype for p in transformer.fm_modules.parameters()} == {torch.bfloat16}
    assert all(p.dtype.is_floating_point for p in adapter._fm_parameters("gen"))


def test_a_non_float_fm_parameter_is_refused_rather_than_silently_untrained():
    adapter, transformer = _adapter("gen", fm=True)
    transformer.fm_modules["fm_head"].conv1.weight = nn.Parameter(
        torch.zeros(2, 2, 1, 1, dtype=torch.int8), requires_grad=False
    )
    with pytest.raises(RuntimeError, match="non-floating-point"):
        adapter.prepare_models_for_training()


def test_an_absent_container_with_the_flag_on_is_refused():
    adapter, _ = _adapter("gen", fm=True, attach=False)
    with pytest.raises(RuntimeError, match="no fm_modules container"):
        adapter.prepare_models_for_training()


# ---------------------------------------------------------------------------
# Understanding-only: warn and proceed, never fail
# ---------------------------------------------------------------------------

def test_understanding_only_warns_and_trains_the_decoder_linears_alone(capsys):
    adapter, transformer = _adapter("und", fm=True)
    adapter.prepare_models_for_training()
    groups = adapter.setup_trainable_parameters()

    assert "sensenova_train_fm_modules_branch_mismatch" in capsys.readouterr().out
    assert _collected(groups) == (294, 294 * 4 * 8)
    assert not any(p.requires_grad for p in transformer.fm_modules.parameters())


def test_the_branch_warning_is_emitted_once_not_per_call(capsys):
    """Three methods resolve the scope; one run is one warning."""
    adapter, _ = _adapter("und", fm=True)
    adapter.prepare_models_for_training()
    adapter.setup_trainable_parameters()
    adapter.grad_norm_components()
    out = capsys.readouterr().out
    assert out.count("sensenova_train_fm_modules_branch_mismatch") == 1


# ---------------------------------------------------------------------------
# Gradient-norm attribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch", ["gen", "both"])
def test_fm_parameters_report_under_the_generation_component(branch):
    """Otherwise they land in whatever bucket the module walk reached first."""
    adapter, transformer = _adapter(branch, fm=True)
    adapter.prepare_models_for_training()
    components = adapter.grad_norm_components()
    fm_components = {components[i] for i in _fm_ids(transformer)}
    assert fm_components == {LORA_COMPONENT_UNET}
    if branch == "both":
        assert LORA_COMPONENT_TEXT_ENCODER_1 in set(components.values())


def test_off_registers_no_fm_component():
    adapter, transformer = _adapter("gen", fm=False)
    adapter.prepare_models_for_training()
    components = adapter.grad_norm_components()
    assert not (_fm_ids(transformer) & set(components))
