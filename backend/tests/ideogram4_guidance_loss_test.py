"""Ideogram 4 guidance-target loss: the unconditional transformer is the teacher."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.ops import ideogram4_ops

GH, GW, CHANNELS, TEXT_LEN = 2, 3, 8, 5


class Recorder(torch.nn.Module):
    dtype = torch.float32

    def __init__(self, offset):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25))
        self.offset = offset
        self.calls = []

    def forward(self, hidden_states, timestep, encoder_hidden_states, position_ids, segment_ids, indicator,
                return_dict):
        out = hidden_states * self.scale + self.offset + encoder_hidden_states.float().mean((1, 2)).view(-1, 1, 1)
        self.calls.append({"x": hidden_states.detach().clone(), "t": timestep.detach().clone(),
                           "context": encoder_hidden_states.detach().clone(), "position_ids": position_ids,
                           "grad": torch.is_grad_enabled(), "out": out})
        return (out,)


def _run(weight):
    torch.manual_seed(2)
    model, uncond = Recorder(0.0), Recorder(0.5)
    latents, noise = torch.randn(2, GH * GW, CHANNELS), torch.randn(2, GH * GW, CHANNELS)
    trainer = SimpleNamespace(
        config={"guidance_loss_weight": weight, "guidance_loss_mix_mode": "blend",
                "guidance_loss_scale": 3.0, "guidance_loss_schedule": "constant"},
        transformer=model, transformer_uncond=uncond, ideogram4_train_uncond=False,
        device=torch.device("cpu"), training_dtype=torch.float32, mixed_precision=False,
        timestep_sampler=None, crop_decode_loss_enable=False, _active_mnt_noise=noise,
        log_extra_metric=lambda *a: None, log_prefix="[test]")
    loss, _, _ = ideogram4_ops.train_step(
        trainer, latents=latents, encoder_features=torch.ones(2, 3, TEXT_LEN, 4),
        encoder_mask=torch.ones(2, TEXT_LEN), timesteps=torch.tensor([.3, .6]), latent_h=GH, latent_w=GW)
    return model, uncond, latents, noise, loss


def test_teacher_is_the_unconditional_transformer_on_image_only_inputs():
    model, uncond, latents, noise, loss = _run(.5)
    (teacher,), (student,) = uncond.calls, model.calls
    assert not teacher["grad"] and student["grad"]
    max_text = student["x"].shape[1] - GH * GW
    torch.testing.assert_close(teacher["x"], student["x"][:, max_text:])
    torch.testing.assert_close(teacher["position_ids"], student["position_ids"][:, max_text:])
    assert not teacher["context"].any() and teacher["context"].shape[1] == GH * GW
    target = latents - noise
    prediction, null = student["out"][:, max_text:], teacher["out"]
    guided = null + 3 * (target - null)
    expected = (.5 * (prediction - target).square().flatten(1).mean(1)
                + .5 * (prediction - guided).square().flatten(1).mean(1)).mean()
    torch.testing.assert_close(loss, expected)


def test_zero_weight_never_runs_the_teacher():
    _, uncond, *_ = _run(0.0)
    assert uncond.calls == []
