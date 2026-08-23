"""Component attribution of LoRA layers for gradient-norm reporting.

Guards the defect that motivated ``BaseLoRAAdapter.register_lora_layer``: the
grad-norm split used to be inferred from substrings of the LoRA key, so an
architecture whose keys are plain module paths (SenseNova) or use another
prefix (FLUX.2 text encoder) contributed to the total only and left
``grad_norm_unet`` at 0.0.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.adapters.base_adapter import (
    LORA_COMPONENT_TEXT_ENCODER,
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_TEXT_ENCODER_2,
    LORA_COMPONENT_UNET,
)
from core.training.adapters.flux2_adapter import FLUX2LoRAAdapter
from core.training.adapters.sd15_adapter import LoRALinearLayer, SD15LoRAAdapter
from core.training.adapters.sdxl_adapter import SDXLLoRAAdapter
from core.training.adapters.sensenova_adapter import SenseNovaLoRAAdapter
from core.training.base_trainer import BaseTrainer


# --------------------------------------------------------------------------
# Minimal module trees (no real model is loaded anywhere in this file)
# --------------------------------------------------------------------------


class Transformer2DModel(nn.Module):
    """Class name is load-bearing: the SDXL adapter selects by it."""

    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(2, 2)
        self.proj_out = nn.Linear(2, 2)


class _Unet(nn.Module):
    def __init__(self, n_blocks: int = 2):
        super().__init__()
        self.attentions = nn.ModuleList([Transformer2DModel() for _ in range(n_blocks)])


class _ClipMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)


class _ClipLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _ClipMlp()


class _ClipTextEncoder(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        encoder = nn.Module()
        encoder.layers = nn.ModuleList([_ClipLayer() for _ in range(n_layers)])
        text_model = nn.Module()
        text_model.encoder = encoder
        self.text_model = text_model


class _QwenMlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(2, 2, bias=False))


class _QwenAttn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(2, 2, bias=False))


class _QwenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _QwenMlp()
        self.self_attn = _QwenAttn()


class _QwenTextEncoder(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        model = nn.Module()
        model.layers = nn.ModuleList([_QwenLayer() for _ in range(n_layers)])
        self.model = model


class _SenseNovaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen"):
            setattr(self, name, nn.Linear(2, 2, bias=False))


class _SenseNovaMlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(2, 2, bias=False))


class _SenseNovaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SenseNovaAttention()
        self.mlp_mot_gen = _SenseNovaMlp()


class _SenseNovaTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        core = nn.Module()
        core.layers = nn.ModuleList([_SenseNovaBlock() for _ in range(42)])
        language_model = nn.Module()
        language_model.model = core
        self.language_model = language_model


# --------------------------------------------------------------------------
# Per-adapter component registration
# --------------------------------------------------------------------------


def test_sdxl_registers_unet_te1_and_te2_separately():
    trainer = SimpleNamespace(
        unet=_Unet(),
        text_encoder=_ClipTextEncoder(3),
        text_encoder_2=_ClipTextEncoder(4),
    )
    adapter = SDXLLoRAAdapter(trainer, 2, 4)
    layers = {}

    assert adapter.apply_lora_to_unet(layers) == 4
    assert adapter.apply_lora_to_text_encoders(layers) == (3 + 4) * 2

    components = adapter.lora_components
    assert set(components) == set(layers)
    by_component = {}
    for name, component in components.items():
        by_component.setdefault(component, []).append(name)

    assert len(by_component[LORA_COMPONENT_UNET]) == 4
    assert len(by_component[LORA_COMPONENT_TEXT_ENCODER_1]) == 6
    assert len(by_component[LORA_COMPONENT_TEXT_ENCODER_2]) == 8
    # The TE1/TE2 split must keep matching the sd-scripts key namespace.
    assert all(n.startswith("lora_te1_") for n in by_component[LORA_COMPONENT_TEXT_ENCODER_1])
    assert all(n.startswith("lora_te2_") for n in by_component[LORA_COMPONENT_TEXT_ENCODER_2])
    assert all(n.startswith("lora_unet_") for n in by_component[LORA_COMPONENT_UNET])


def test_sd15_registers_its_single_clip_as_text_encoder_1():
    trainer = SimpleNamespace(unet=_Unet(1), text_encoder=_ClipTextEncoder(2))
    adapter = SD15LoRAAdapter(trainer, 2, 4)
    layers = {}

    assert adapter.apply_lora_to_unet(layers) == 2
    assert adapter.apply_lora_to_text_encoders(layers) == 4

    components = adapter.lora_components
    assert sum(c == LORA_COMPONENT_UNET for c in components.values()) == 2
    assert sum(c == LORA_COMPONENT_TEXT_ENCODER_1 for c in components.values()) == 4
    assert LORA_COMPONENT_TEXT_ENCODER_2 not in components.values()


def test_sensenova_registers_all_294_generation_layers_as_dit():
    trainer = SimpleNamespace(transformer=_SenseNovaTransformer(), unet_lr=2e-4)
    adapter = SenseNovaLoRAAdapter(trainer, 2, 4)
    layers = {}

    assert adapter.apply_lora_to_unet(layers) == 294
    components = adapter.lora_components
    assert len(components) == 294
    assert set(components.values()) == {LORA_COMPONENT_UNET}
    # Keys are plain module paths — the substring heuristic matched none of them.
    assert all(name.startswith("language_model.model.layers.") for name in components)

    # Re-application over already-wrapped targets must repopulate the registry too.
    repopulated = {}
    fresh = SenseNovaLoRAAdapter(trainer, 2, 4)
    assert fresh.apply_lora_to_unet(repopulated) == 0
    assert len(fresh.lora_components) == 294
    assert set(fresh.lora_components.values()) == {LORA_COMPONENT_UNET}


def test_flux2_text_encoder_lora_is_registered_as_text_encoder():
    trainer = SimpleNamespace(text_encoder=_QwenTextEncoder(2), train_text_encoder=True)
    adapter = FLUX2LoRAAdapter(trainer, 2, 4)
    layers = {}

    assert adapter.apply_lora_to_text_encoders(layers) == 2 * 7
    components = adapter.lora_components
    assert set(components.values()) == {LORA_COMPONENT_TEXT_ENCODER}
    assert all(name.startswith("lora_te_") for name in components)


def test_registering_an_unknown_component_is_rejected():
    adapter = SD15LoRAAdapter(SimpleNamespace(), 2, 4)
    with pytest.raises(ValueError, match="Unknown LoRA component"):
        adapter.register_lora_layer({}, "x", nn.Linear(2, 2), "decoder")


# --------------------------------------------------------------------------
# Grad-norm aggregation
# --------------------------------------------------------------------------


def _lora_layer(grad_value: float) -> LoRALinearLayer:
    layer = LoRALinearLayer(nn.Linear(2, 2, bias=False), rank=1, alpha=1, lora_name="x")
    for p in (layer.lora_down.weight, layer.lora_up.weight):
        p.grad = torch.full_like(p, grad_value)
    return layer


class _FakeTrainer:
    log_prefix = "[test]"

    def __init__(self, lora_layers, adapter):
        self.lora_layers = lora_layers
        self.adapter = adapter

    _calculate_grad_norms = BaseTrainer._calculate_grad_norms


def test_grad_norms_follow_the_registered_components():
    adapter = SD15LoRAAdapter(SimpleNamespace(), 2, 4)
    layers = {}
    for name, component in (
        ("dit", LORA_COMPONENT_UNET),
        ("te1", LORA_COMPONENT_TEXT_ENCODER_1),
        ("te2", LORA_COMPONENT_TEXT_ENCODER_2),
        ("te", LORA_COMPONENT_TEXT_ENCODER),
    ):
        adapter.register_lora_layer(layers, name, _lora_layer(1.0), component)

    total, te, te1, te2, unet, ve = _FakeTrainer(layers, adapter)._calculate_grad_norms()

    assert unet > 0.0
    assert te1 > 0.0 and te2 > 0.0
    assert te == pytest.approx((te1 ** 2 + te2 ** 2 + unet ** 2) ** 0.5)
    assert total == pytest.approx((te ** 2 + unet ** 2) ** 0.5)
    assert ve == 0.0


def test_sensenova_style_module_path_keys_reach_the_dit_bucket():
    trainer = SimpleNamespace(transformer=_SenseNovaTransformer(), unet_lr=2e-4)
    adapter = SenseNovaLoRAAdapter(trainer, 2, 4)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    for layer in layers.values():
        for p in (layer.lora_down.weight, layer.lora_up.weight):
            p.grad = torch.full_like(p, 0.5)

    total, te, te1, te2, unet, ve = _FakeTrainer(layers, adapter)._calculate_grad_norms()

    assert unet == pytest.approx(total)
    assert (te, te1, te2, ve) == (0.0, 0.0, 0.0, 0.0)


def test_unregistered_layers_land_in_the_unet_bucket_with_a_warning(capsys):
    adapter = SD15LoRAAdapter(SimpleNamespace(), 2, 4)
    layers = {"mystery.module.path": _lora_layer(1.0)}  # no register_lora_layer call

    total, _, _, _, unet, _ = _FakeTrainer(layers, adapter)._calculate_grad_norms()

    assert unet == pytest.approx(total) and unet > 0.0
    out = capsys.readouterr().out
    assert "without a registered component" in out
    assert "mystery.module.path" in out
