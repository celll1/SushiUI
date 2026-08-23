from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.sensenova.sensenova_lora import (
    apply_lora_group,
    iter_sensenova_lora_targets,
    load_lora_safetensors,
    normalise_lora_state_dict,
    restore_originals,
)
from core.training.adapters.sd15_adapter import LoRALinearLayer
from core.training.adapters.sensenova_adapter import SenseNovaLoRAAdapter


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in (
            "q_proj_mot_gen",
            "k_proj_mot_gen",
            "v_proj_mot_gen",
            "o_proj_mot_gen",
        ):
            setattr(self, name, nn.Linear(2, 2, bias=False))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(2, 2, bias=False))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp_mot_gen = _Mlp()


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        core = nn.Module()
        core.layers = nn.ModuleList([_Block() for _ in range(42)])
        language_model = nn.Module()
        language_model.model = core
        self.language_model = language_model


def _adapter(transformer: nn.Module, *, rank: int = 2, alpha: int = 4):
    trainer = SimpleNamespace(transformer=transformer, unet_lr=2e-4)
    return SenseNovaLoRAAdapter(trainer, rank, alpha), trainer


def test_sensenova_adapter_wraps_only_the_generation_namespace():
    transformer = _Transformer()
    adapter, _ = _adapter(transformer)
    layers = {}

    assert adapter.apply_lora_to_unet(layers) == 294
    assert len(layers) == 294
    assert all(isinstance(layer, LoRALinearLayer) for layer in layers.values())
    assert all(name.startswith("language_model.model.layers.") for name in layers)
    assert sum(".self_attn." in name for name in layers) == 168
    assert sum(".mlp_mot_gen." in name for name in layers) == 126
    assert adapter.apply_lora_to_unet(layers) == 0
    layers[next(iter(layers))].lora_name = "wrong"
    with pytest.raises(RuntimeError, match="wrong namespace"):
        adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0

    groups = adapter.setup_trainable_parameters(layers)
    assert len(groups) == 1
    assert groups[0]["lr"] == 2e-4
    assert len(groups[0]["params"]) == 588


def test_sensenova_adapter_refuses_incomplete_or_mixed_target_trees():
    adapter, _ = _adapter(None)
    with pytest.raises(RuntimeError, match="loaded transformer"):
        adapter.apply_lora_to_unet({})

    incomplete = _Transformer()
    del incomplete.language_model.model.layers[0].self_attn.q_proj_mot_gen
    adapter, _ = _adapter(incomplete)
    with pytest.raises(RuntimeError, match="exactly 294"):
        adapter.apply_lora_to_unet({})

    mixed = _Transformer()
    first = mixed.language_model.model.layers[0].self_attn.q_proj_mot_gen
    mixed.language_model.model.layers[0].self_attn.q_proj_mot_gen = LoRALinearLayer(
        first, rank=2, alpha=4, lora_name="prewrapped"
    )
    adapter, _ = _adapter(mixed)
    with pytest.raises(RuntimeError, match="mixed or unsupported"):
        adapter.apply_lora_to_unet({})


def test_sensenova_adapter_checkpoint_round_trips_through_runtime_loader(tmp_path):
    transformer = _Transformer()
    adapter, _ = _adapter(transformer)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    with torch.no_grad():
        for index, layer in enumerate(layers.values()):
            layer.lora_down.weight.fill_(index / 1000)
            layer.lora_up.weight.fill_(-index / 1000)

    repopulated = {}
    assert adapter.apply_lora_to_unet(repopulated) == 0
    assert len(repopulated) == 294
    assert all(repopulated[name] is layer for name, layer in layers.items())

    checkpoint = tmp_path / "sensenova_lora.safetensors"
    assert len(adapter.setup_trainable_parameters(repopulated)[0]["params"]) == 588
    adapter.save_checkpoint(repopulated, step=7, epoch=2, output_path=checkpoint)
    raw, file_format = load_lora_safetensors(str(checkpoint))
    grouped = normalise_lora_state_dict(raw)

    assert file_format == "neo_hf_lora"
    assert len(raw) == 882
    assert len(grouped) == 294
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert metadata["tensor_kind"] == "neo_hf_lora"
    assert metadata["model_type"] == "sensenova"
    assert metadata["modelspec.architecture"] == "sensenova"
    assert metadata["lora_targets"] == "generation"
    assert metadata["lora_rank"] == "2"
    assert metadata["lora_alpha"] == "4"
    assert metadata["step"] == "7"
    assert metadata["epoch"] == "2"

    inference_transformer = _Transformer()
    last_name = next(reversed(layers))
    original = dict(inference_transformer.named_modules())[last_name]
    originals = {}
    wrapped_keys = set()
    assert apply_lora_group(
        inference_transformer,
        grouped,
        strength=1.0,
        lora_original_modules=originals,
        wrapped_keys=wrapped_keys,
    ) == 294
    runtime_layer = dict(inference_transformer.named_modules())[last_name]
    torch.testing.assert_close(
        runtime_layer.lora_down.weight,
        layers[last_name].lora_down.weight,
    )
    torch.testing.assert_close(
        runtime_layer.lora_up.weight,
        layers[last_name].lora_up.weight,
    )
    assert grouped[last_name]["alpha"].item() == 4
    assert runtime_layer.scale == 2
    sample = torch.tensor([[1.0, -2.0]])
    assert not torch.equal(runtime_layer(sample), original(sample))
    assert restore_originals(inference_transformer, originals, wrapped_keys) == 294
    assert dict(inference_transformer.named_modules())[last_name] is original

    resume_transformer = _Transformer()
    resume_adapter, _ = _adapter(resume_transformer)
    resume_layers = {}
    resume_adapter.apply_lora_to_unet(resume_layers)
    with torch.no_grad():
        for layer in resume_layers.values():
            layer.lora_down.weight.zero_()
            layer.lora_up.weight.zero_()
    from core.training.lora_trainer import LoRATrainer

    resume_owner = SimpleNamespace(log_prefix="[test]", lora_layers=resume_layers)
    assert LoRATrainer.load_checkpoint(resume_owner, str(checkpoint)) == 7
    torch.testing.assert_close(
        resume_layers[last_name].lora_down.weight,
        layers[last_name].lora_down.weight,
    )
    torch.testing.assert_close(
        resume_layers[last_name].lora_up.weight,
        layers[last_name].lora_up.weight,
    )


def test_sensenova_runtime_strength_zero_is_exact_and_restores_identity():
    transformer = _Transformer()
    targets = list(iter_sensenova_lora_targets(transformer))
    grouped = {
        path: {
            "down": torch.ones((1, module.in_features), dtype=torch.float32),
            "up": torch.ones((module.out_features, 1), dtype=torch.float32),
            "alpha": torch.tensor(1.0),
        }
        for path, _parent, _attr, module in targets
    }
    original_ids = {path: id(module) for path, _p, _a, module in targets}
    sample = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    path, parent, attr, original = targets[0]
    expected = original(sample)
    originals = {}
    wrapped_keys = set()

    assert apply_lora_group(
        transformer,
        grouped,
        strength=0.0,
        lora_original_modules=originals,
        wrapped_keys=wrapped_keys,
    ) == 294
    wrapped = getattr(parent, attr)
    assert torch.equal(wrapped(sample), expected)
    assert restore_originals(transformer, originals, wrapped_keys) == 294
    restored_ids = {
        name: id(module)
        for name, _p, _a, module in iter_sensenova_lora_targets(transformer)
    }
    assert restored_ids == original_ids
    assert getattr(parent, attr) is original
    assert path in originals
