"""CPU native shared-handler identity, objective and staged run boundary."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming arch test attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.training.arch import ARCH_REGISTRY, get_arch_handler, resolve_arch_name
from core.training.arch.base_arch import TrainStepContext
from core.models.ming_image_01.training_data import MingTrainingBatch
from core.models.ming_image_01.conditioning import MingConditioning
from api.arch_capabilities import TRAINING_DECLARED_ARCHS, TRAINING_UNSUPPORTED, TRAINING_METHODS
from ming_image_01_training_batch_test import components, trainer


def test_registry_identity_native_sign_and_tiny_wiring():
    value = trainer(components("generation_edit"))
    value.is_ming_image_01 = True
    handler = get_arch_handler(value)
    assert resolve_arch_name(value) == handler.name == "ming_image_01"
    assert set(ARCH_REGISTRY) == set(TRAINING_DECLARED_ARCHS)
    assert set(TRAINING_UNSUPPORTED[handler.name]) == set(TRAINING_METHODS)
    assert handler.timestep_convention == "t0" and handler.velocity_sign == "x0_minus_eps"
    assert handler.wiring.te_out_dim == 8 and handler.wiring.latent_channels == 16
    assert handler.depth_blocks(value) is value.transformer.layers


def test_shared_handler_native_batch_backward_and_context_refusal():
    value = trainer(components("generation_edit"))
    value.is_ming_image_01 = True
    handler = get_arch_handler(value)
    clean = torch.zeros(1, 16, 1, 4, 4)
    batch = MingTrainingBatch(clean, MingConditioning(torch.randn(1, 4, 8), torch.randn(1, 3, 24)), torch.ones(1, 1), None)
    result = handler.train_step(value, TrainStepContext(latents=clean, ming_batch=batch, timesteps=torch.tensor([.5])))
    result[0].backward()
    assert value.transformer.layers[0].attention.to_q.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="aligned native"):
        handler.train_step(value, TrainStepContext(latents=clean))
    assert not torch.cuda.is_initialized()


def test_handler_vae_keeps_layer_axis_independent():
    components_ = components("layer_decompose")
    value = trainer(components_)
    value.vae = components_.vae
    value.is_ming_image_01 = True
    handler = get_arch_handler(value)
    pixels = torch.zeros(1, 4, 3, 32, 32)
    latents = handler.vae_encode(value, pixels)
    decoded = handler.vae_decode(value, latents, latent_h=4, latent_w=4)
    assert latents.shape == (1, 16, 3, 4, 4) and decoded.shape == pixels.shape
    with pytest.raises(ValueError): handler.vae_encode(value, pixels[:, :3])


def test_normal_training_start_refuses_staged_ming_before_dataset_or_weight_load(monkeypatch):
    from core.model_loader import ModelLoader
    from core.training.train_runner import _prepare_training_process_config
    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda path: "ming_image_01")
    config = {"config": {"process": [{"train": {}, "network": {"type": "full_finetune"}}]}}
    with pytest.raises(ValueError, match="optimization-loop"):
        _prepare_training_process_config(config, "no-weights-loaded")
    assert not torch.cuda.is_initialized()
