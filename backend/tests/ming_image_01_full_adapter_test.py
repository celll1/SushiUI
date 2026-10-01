"""CPU full-parameter component ownership and shared trainer adapter selection."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming full adapter attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.training.adapters.ming_image_01_adapter import MingImage01FullParameterAdapter
from core.training.full_parameter_trainer import FullParameterTrainer
from ming_image_01_training_batch_test import components


def owner(value, groups, rates, *, train_dit=True):
    return SimpleNamespace(ming_image_01_components=value, transformer=value.transformer, ming_conditioner_trainable_groups=groups,
        ming_conditioner_lrs=rates, train_unet=train_dit, unet_lr=.001, learning_rate=.001,
        gradient_checkpointing=False, is_ming_image_01=True, log_prefix="[CPU]")


@pytest.mark.parametrize("groups,train_dit", [(set(), True), ({"projections"}, False),
                                               ({"language", "vision", "connector", "queries", "projections"}, True)])
def test_actual_shared_factory_and_unique_optimizer_ownership(groups, train_dit):
    value = components("generation_edit")
    rates = {name: .0001 * (index + 1) for index, name in enumerate(sorted(groups))}
    trainer = owner(value, groups, rates, train_dit=train_dit)
    FullParameterTrainer._create_adapter(trainer)
    adapter = trainer.adapter
    assert isinstance(adapter, MingImage01FullParameterAdapter)
    adapter.prepare_models_for_training()
    param_groups = adapter.arch_param_groups()
    expected = [name for name in ("dit", "language", "vision", "connector", "queries", "projections")
                if name == "dit" and train_dit or name in groups]
    assert [group["name"] for group in param_groups] == expected
    actual_ids = [id(p) for group in param_groups for p in group["params"]]
    expected_ids = {id(p) for module in (value.transformer, value.conditioner) for p in module.parameters() if p.requires_grad}
    assert len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == expected_ids
    assert set(adapter.grad_norm_components()) == expected_ids
    assert all(not p.requires_grad for p in value.vae.parameters()) and not value.vae.training
    assert value.conditioner.cacheable == (not groups)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("groups,rates", [({"foreign"}, {}), ({"language"}, {}), ({"language"}, {"language": 0})])
def test_bad_training_ownership_or_rates_refused(groups, rates):
    trainer = owner(components("generation_edit"), groups, rates)
    adapter = MingImage01FullParameterAdapter(trainer, conditioner_groups=groups, conditioner_lrs=rates)
    with pytest.raises(ValueError):
        adapter.prepare_models_for_training()


@pytest.mark.parametrize("fail", [False, True])
def test_export_preserves_run_metadata_and_restores_all_component_devices(tmp_path, monkeypatch, fail):
    from core.training.adapters import ming_image_01_adapter as module
    value = components("generation_edit")
    trainer = owner(value, {"language", "projections"}, {"language": .0001, "projections": .0002})
    adapter = MingImage01FullParameterAdapter(trainer, conditioner_groups=trainer.ming_conditioner_trainable_groups,
                                             conditioner_lrs=trainer.ming_conditioner_lrs)
    adapter.prepare_models_for_training()
    calls = []
    def save(components, path, **kwargs):
        calls.append(kwargs)
        assert all(p.device.type == "cpu" for part in (value.transformer, value.conditioner, value.vae) for p in part.parameters())
        if fail: raise OSError("save failed")
        return path / "manifest.json"
    monkeypatch.setattr(module, "save_bundle", save)
    before = {id(p): (p.device, p.dtype, p.requires_grad) for part in (value.transformer, value.conditioner, value.vae) for p in part.parameters()}
    if fail:
        with pytest.raises(OSError, match="save failed"): adapter.write_checkpoint(4, 2, tmp_path / "full")
    else:
        assert adapter.write_checkpoint(4, 2, tmp_path / "full") == tmp_path / "full/manifest.json"
    metadata = calls[0]["training_metadata"]
    assert metadata["step"] == 4 and metadata["epoch"] == 2 and metadata["kind"] == "full_parameter_weights"
    assert metadata["conditioner_groups"] == ["language", "projections"]
    assert [g["name"] for g in metadata["optimizer_groups"]] == ["dit", "language", "projections"]
    assert all(before[id(p)] == (p.device, p.dtype, p.requires_grad) for part in (value.transformer, value.conditioner, value.vae) for p in part.parameters())


def test_full_te_checkpointing_preserves_both_conditioning_gradients():
    from core.models.ming_image_01.training_data import prepare_training_batch
    from core.training.ops.ming_image_01_ops import train_step
    from PIL import Image
    value = components("generation_edit")
    groups = {"language", "vision", "connector", "queries", "projections"}
    trainer = owner(value, groups, {name: .0001 for name in groups})
    trainer.gradient_checkpointing = True
    adapter = MingImage01FullParameterAdapter(trainer, conditioner_groups=groups, conditioner_lrs=trainer.ming_conditioner_lrs)
    adapter.prepare_models_for_training()
    assert value.transformer.is_gradient_checkpointing
    batch = prepare_training_batch(value, prompts=["word5"],
        references=[[Image.new("RGBA", (32, 32), (30, 40, 50, 100))]],
        target_pixels=torch.zeros(1, 4, 1, 32, 32), frame_weights=torch.ones(1, 1), resolution=32)
    batch.conditioning.queries.retain_grad()
    batch.conditioning.direct.retain_grad()
    trainer.device = torch.device("cpu")
    trainer.training_dtype = torch.float32
    trainer.mixed_precision = False
    trainer.reconstruction_loss_weight = .2
    loss, *_ = train_step(trainer, batch, timesteps=torch.tensor([.5]))
    loss.backward()
    assert batch.conditioning.queries.grad.abs().sum() > 0
    assert batch.conditioning.direct.grad.abs().sum() > 0
    for name, modules in value.conditioner.component_groups().items():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for module in modules for p in module.parameters()), name
    assert not torch.cuda.is_initialized()
