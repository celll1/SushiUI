"""Atomic validation and commit gates for LoRA training resume."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _path in (_REPO, _BACKEND):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.adapters import LoRALinearLayer  # noqa: E402
from core.training.lora_trainer import LoRATrainer  # noqa: E402
from core.training.adapters.krea2_adapter import Krea2LoRAAdapter  # noqa: E402


def _layer(seed):
    torch.manual_seed(seed)
    return LoRALinearLayer(nn.Linear(4, 3), rank=2, alpha=2, lora_name="x")


def _trainer(layers):
    trainer = object.__new__(LoRATrainer)
    trainer.log_prefix = "[test]"
    trainer.lora_rank = 2
    trainer.lora_alpha = 2
    trainer.lora_layers = layers
    return trainer


def _state(layers):
    return {
        f"{stem}.{name}": tensor.detach().clone()
        for stem, layer in layers.items()
        for name, tensor in layer.branch_tensors().items()
    }


def _write(path, tensors, **metadata):
    values = {name: tensor.detach().clone() for name, tensor in tensors.items()}
    save_file(values, str(path), metadata={key: str(value) for key, value in metadata.items()})


def test_branch_construction_is_non_destructive_and_training_registration_freezes():
    base = nn.Linear(4, 3)
    branch = LoRALinearLayer(base, rank=2, alpha=2, lora_name="x")
    assert all(parameter.requires_grad for parameter in base.parameters())

    adapter = Krea2LoRAAdapter(SimpleNamespace(), 2, 2)
    layers = {}
    adapter.register_lora_layer(layers, "x", branch, "unet")

    assert layers == {"x": branch}
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_complete_checkpoint_loads_and_ignores_compatible_surplus(tmp_path):
    layers = {"a": _layer(1), "b": _layer(2)}
    before = _state(layers)
    loaded = {name: torch.full_like(value, 7) for name, value in before.items()}
    loaded["a.alpha"] = torch.tensor(2.0)
    loaded["inactive_component.extra"] = torch.ones(1)
    path = tmp_path / "complete.safetensors"
    _write(path, loaded, step=9, lora_rank=2, lora_alpha=2)

    assert LoRATrainer.load_checkpoint(_trainer(layers), str(path)) == 9
    assert all(torch.equal(tensor, loaded[name])
               for name, tensor in _state(layers).items())


@pytest.mark.parametrize("defect", ["missing", "shape"])
def test_preflight_rejects_before_copying_any_layer(tmp_path, defect):
    layers = {"a": _layer(3), "b": _layer(4)}
    before = _state(layers)
    loaded = {name: torch.full_like(value, 5) for name, value in before.items()}
    key = "b.lora_up.weight"
    if defect == "missing":
        del loaded[key]
    else:
        loaded[key] = torch.zeros(4, 2)
    path = tmp_path / f"{defect}.safetensors"
    _write(path, loaded)

    with pytest.raises(ValueError):
        LoRATrainer.load_checkpoint(_trainer(layers), str(path))

    assert all(torch.equal(tensor, before[name])
               for name, tensor in _state(layers).items())


def test_metadata_conflict_is_rejected_before_copy(tmp_path):
    layers = {"a": _layer(5)}
    before = _state(layers)
    path = tmp_path / "wrong-rank.safetensors"
    _write(path, before, lora_rank=8)

    with pytest.raises(ValueError, match="conflicts"):
        LoRATrainer.load_checkpoint(_trainer(layers), str(path))
    assert all(torch.equal(tensor, before[name])
               for name, tensor in _state(layers).items())


def test_commit_failure_rolls_back_every_layer(tmp_path):
    class _FailingLayer(LoRALinearLayer):
        def load_tensors(self, tensors):
            first = next(iter(self.branch_tensors()))
            self.branch_tensors()[first].data.copy_(tensors[first])
            raise RuntimeError("synthetic commit failure")

    layers = {"a": _layer(6), "b": _FailingLayer(
        nn.Linear(4, 3), rank=2, alpha=2, lora_name="b")}
    before = _state(layers)
    loaded = {name: torch.full_like(value, 3) for name, value in before.items()}
    path = tmp_path / "commit-failure.safetensors"
    _write(path, loaded)

    with pytest.raises(RuntimeError, match="synthetic"):
        LoRATrainer.load_checkpoint(_trainer(layers), str(path))

    assert all(torch.equal(tensor, before[name])
               for name, tensor in _state(layers).items())
