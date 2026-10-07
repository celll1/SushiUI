import pytest
import torch

from core.memory_management.frozen_backing import FrozenCPUBacking, trim_mapped_pages


def test_park_reuses_original_storage_and_preserves_dynamic_buffers():
    model = torch.nn.Linear(3, 2).requires_grad_(False)
    model.register_buffer("runtime", torch.ones(2))
    original = model.weight.detach()
    backing = FrozenCPUBacking((model,))
    model.weight.data = model.weight.clone()
    model.runtime.add_(3)
    backing.park()
    assert model.weight.data_ptr() == original.data_ptr()
    assert torch.equal(model.runtime, torch.full((2,), 4.0))
    backing.stage("cpu")
    backing.park()
    assert model.weight.data_ptr() == original.data_ptr()


def test_trainable_module_is_rejected():
    with pytest.raises(ValueError, match="frozen CPU"):
        FrozenCPUBacking((torch.nn.Linear(2, 2),))


@pytest.mark.parametrize("change", ["trainable", "replacement", "gradient", "dtype"])
def test_changed_parameter_cannot_be_discarded(change):
    model = torch.nn.Linear(2, 2).requires_grad_(False)
    backing = FrozenCPUBacking((model,))
    if change == "trainable":
        model.weight.requires_grad_(True)
    elif change == "replacement":
        model.weight = torch.nn.Parameter(model.weight.clone(), requires_grad=False)
    elif change == "gradient":
        model.weight.grad = torch.ones_like(model.weight)
    else:
        model.to(torch.float64)
    with pytest.raises(RuntimeError, match="parameter changed"):
        backing.park()


def test_private_cpu_pages_remain_readable():
    tensor = torch.randn(5)
    expected = tensor.clone()
    assert trim_mapped_pages((tensor, tensor)) == 0
    assert torch.equal(tensor, expected)


def test_file_mapped_pages_remain_readable(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensor = torch.randn(4096)
    path = tmp_path / "weights.safetensors"
    save_file({"weight": tensor}, str(path))
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        mapped = handle.get_tensor("weight")
        trim_mapped_pages((mapped,))
        assert torch.equal(tensor, mapped)


def test_ltx_sample_failure_parks_conditioner():
    from types import SimpleNamespace
    from core.training.ops.ltx2_ops import generate_sample

    events = []
    def fail(**kwargs):
        raise RuntimeError("synthetic sample failure")
    vae = torch.nn.Identity()
    vae.spatial_compression_ratio = 16
    vae.temporal_compression_ratio = 4
    trainer = SimpleNamespace(log_prefix="[test]", transformer=torch.nn.Identity(), vae=vae,
                              text_encoder=torch.nn.Identity(), device="cpu", ltx2_pipeline=fail,
                              _ltx2_frozen_conditioner=object(),
                              move_text_encoder_to_cpu=lambda: events.append("park"))
    assert generate_sample(trainer, "synthetic", height=32, width=32, seed=1) is None
    assert events == ["park"]


def test_trainer_cleanup_drops_backing_owner(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: None)
    monkeypatch.setattr(torch._C, "_cuda_init", lambda *a, **k: None)
    from core.training.base_trainer import BaseTrainer

    events = []
    owner = SimpleNamespace(park=lambda: events.append("park"))
    trainer = SimpleNamespace(log_prefix="[test]", _ltx2_frozen_conditioner=owner,
                              _shutdown_db_executor=lambda: None,
                              _layer_offload_conductors=lambda: ())
    BaseTrainer.cleanup(trainer)
    assert events == ["park"]
    assert trainer._ltx2_frozen_conditioner is None
