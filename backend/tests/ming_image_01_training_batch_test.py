"""CPU independent RGBA frame encoding and shared trainer objective integration."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
def forbidden_cuda(*a, **k):
    raise AssertionError("Ming training batch attempted CUDA initialization")
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.training_data import MingTrainingBatch, encode_frames, prepare_training_batch
from core.models.ming_image_01.conditioning import MingConditioning
from core.training.ops.ming_image_01_ops import train_step
from ming_image_01_sampling_test import make_sampler
from ming_image_01_conditioning_test import make_conditioner
from ming_image_01_prompting_test import processor


def components(profile):
    value = make_sampler(profile).components
    value.conditioner = make_conditioner()
    value.conditioner.set_trainable({"language", "projections"})
    value.processor = processor()
    return value


def trainer(value, **kw):
    kw.setdefault("config", {})
    return SimpleNamespace(transformer=value.transformer, device=torch.device("cpu"),
        training_dtype=torch.float32, timestep_sampler=None, mixed_precision=False,
        reconstruction_loss_weight=.25, _collect_concept_sample_loss=True, **kw)


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_real_native_rgba_batch_backward_and_optimizer_step(profile):
    value = components(profile)
    frames = 3 if profile == "layer_decompose" else 1
    targets = torch.rand(2, 4, frames, 32, 32) * 2 - 1
    references = [[Image.new("RGBA", (32, 32), (50, 60, 70, 100))],
                  [Image.new("RGBA", (32, 32), (80, 90, 100, 200))] * 2]
    weights = torch.ones(2, frames)
    if frames > 1: weights[:, 0] = 0
    batch = prepare_training_batch(value, prompts=["word5", "word6"], references=references,
                                  target_pixels=targets, frame_weights=weights, resolution=32)
    assert batch.clean.shape == (2, 16, frames, 4, 4)
    assert [reference.shape[1] for reference in batch.reference_latents] == [1, 2]
    assert not batch.clean.requires_grad
    subset = batch.sliced(1, 2)
    assert subset.clean.shape[0] == 1 and subset.reference_latents[0].shape[1] == 2
    before = value.transformer.layers[0].attention.to_q.weight.detach().clone()
    optim = torch.optim.AdamW([p for module in (value.transformer, value.conditioner)
                               for p in module.parameters() if p.requires_grad], lr=.001)
    owner = trainer(value)
    loss, prediction, reconstruction = train_step(owner, batch, timesteps=torch.tensor([.2, .8]),
                                                 cfg_drop_mask=torch.tensor([False, False]))
    assert torch.isfinite(loss) and not prediction.requires_grad and not reconstruction.requires_grad
    loss.backward()
    assert value.transformer.layers[0].attention.to_q.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in value.conditioner.native.model.parameters())
    assert all(p.grad is None for p in value.vae.parameters())
    optim.step()
    assert not torch.equal(before, value.transformer.layers[0].attention.to_q.weight)
    assert owner._pending_concept_sample_loss.shape == (2,)
    assert not torch.cuda.is_initialized()


def test_frame_encoding_is_independent_and_preserves_order_and_alpha():
    value = components("layer_decompose")
    pixels = torch.zeros(1, 4, 3, 32, 32)
    pixels[:, 3, 0] = -1
    pixels[:, 3, 2] = 1
    combined = encode_frames(value.vae, pixels)
    separate = torch.cat([encode_frames(value.vae, pixels[:, :, index:index+1]) for index in range(3)], dim=2)
    torch.testing.assert_close(combined, separate, rtol=0, atol=0)
    assert not torch.equal(combined[:, :, 0], combined[:, :, 2])


class PredictionOracle(torch.nn.Module):
    in_channels = 16
    config = SimpleNamespace(multi_frame_output=False, cap_feat_dim=8, dim=24)
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(-.5))
    def forward(self, **kw):
        return ([torch.ones_like(item) * self.weight for item in kw["x"]],)


def test_shared_mnt_noise_sign_reconstruction_and_per_item_metric_oracle():
    clean = torch.zeros(2, 16, 1, 4, 4)
    batch = MingTrainingBatch(clean, MingConditioning(torch.ones(2, 2, 8), torch.ones(2, 2, 24)), torch.ones(2, 1), None)
    value = SimpleNamespace(transformer=PredictionOracle())
    owner = trainer(value, _active_mnt_noise=torch.ones_like(clean))
    owner.reconstruction_loss_weight = .4
    loss, prediction, reconstruction = train_step(owner, batch, timesteps=torch.tensor([.5, .5]))
    torch.testing.assert_close(prediction, torch.tensor(.25))
    torch.testing.assert_close(reconstruction, torch.tensor(.0625))
    torch.testing.assert_close(loss, torch.tensor(.175))
    torch.testing.assert_close(owner._pending_concept_sample_loss, torch.full((2,), .175))
    loss.backward()
    torch.testing.assert_close(value.transformer.weight.grad, torch.tensor(.7))


@pytest.mark.parametrize("problem", ["no_reference", "zero_weights", "wrong_frames", "nan_pixels"])
def test_bad_supervision_refused(problem):
    value = components("layer_decompose")
    targets = torch.zeros(1, 4, 3, 32, 32)
    references = [[Image.new("RGBA", (32, 32))]]
    weights = torch.ones(1, 3)
    if problem == "no_reference": references = [[]]
    elif problem == "zero_weights": weights.zero_()
    elif problem == "wrong_frames": targets = targets[:, :, :1]
    else: targets[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        prepare_training_batch(value, prompts=["word5"], references=references,
                               target_pixels=targets, frame_weights=weights, resolution=32)


def test_design_mixed_plain_and_edit_reference_batch_backward():
    value = components("generation_edit")
    batch = prepare_training_batch(value, prompts=["word5", "word6"],
        references=[[], [Image.new("RGBA", (32, 32), (20, 30, 40, 50))]],
        target_pixels=torch.zeros(2, 4, 1, 32, 32), frame_weights=torch.ones(2, 1), resolution=32)
    assert batch.reference_latents[0] is None and batch.reference_latents[1].shape == (16, 1, 4, 4)
    loss, *_ = train_step(trainer(value), batch, timesteps=torch.tensor([.2, .8]))
    loss.backward()
    assert torch.isfinite(loss) and value.transformer.layers[0].attention.to_q.weight.grad.abs().sum() > 0
    assert not torch.cuda.is_initialized()
