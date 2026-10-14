"""Anima guidance-target loss: the null-condition teacher and the mixed objective."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None
from core.training.ops.anima_ops import train_step

AUX_KEYS = ("source_mask", "t5_input_ids", "t5_attn_mask")


class Recorder(nn.Module):
    """Velocity depends on the context so the null and caption branches differ."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 4, 1)
        self.vae_latent_bridge = SimpleNamespace(noise_alignment=object())
        self.calls = []

    def forward(self, x, timesteps, context, alignment_noise, **kwargs):
        self.calls.append({"x": x.detach().clone(), "timesteps": timesteps.detach().clone(),
                           "context": context.detach().clone(), "noise": alignment_noise.detach().clone(),
                           "grad": torch.is_grad_enabled()})
        shift = context.float().mean((1, 2)).view(-1, 1, 1)
        return (self.projection(x.squeeze(2)) + shift[:, None]).unsqueeze(2)


def _run(config, cfg_drop_mask=None):
    torch.manual_seed(3)
    model = Recorder()
    clean, noise = torch.randn(2, 4, 8, 12), torch.randn(2, 4, 8, 12)
    sigma = torch.tensor([.3, .7])
    metrics = {}
    null = (torch.zeros(5, 6), {key: torch.ones(5, dtype=torch.long) for key in AUX_KEYS})
    trainer = SimpleNamespace(config=config, device=torch.device("cpu"), training_dtype=torch.float32,
        transformer=model, transformer_original=model, reconstruction_loss_weight=0.0,
        mixed_precision=False, _active_mnt_noise=noise, log_prefix="[test]",
        _guidance_null_encoding=null, log_extra_metric=metrics.__setitem__)
    loss, _, _ = train_step(trainer, latents=clean, prompt_embeds=torch.ones(2, 3, 6),
        anima_aux={key: torch.ones(2, 3, dtype=torch.long) for key in AUX_KEYS},
        timesteps=sigma, cfg_drop_mask=cfg_drop_mask)
    return model, clean, noise, sigma, loss, metrics


def test_teacher_sees_null_condition_on_the_same_noisy_input_and_target_is_exact():
    model, clean, noise, sigma, loss, metrics = _run(
        {"guidance_loss_weight": .5, "guidance_loss_mix_mode": "blend",
         "guidance_loss_scale": 3.0, "guidance_loss_schedule": "constant"})
    teacher, student = model.calls
    view = sigma.view(-1, 1, 1, 1)
    noisy = (1 - view) * clean + view * noise
    assert not teacher["grad"] and student["grad"]
    torch.testing.assert_close(teacher["x"].squeeze(2), noisy)
    torch.testing.assert_close(teacher["timesteps"], sigma)
    torch.testing.assert_close(teacher["noise"].squeeze(2), noise)
    assert teacher["context"].abs().sum() == 0 and teacher["context"].shape == (2, 5, 6)

    prediction = model.projection(noisy) + 1
    with torch.no_grad():
        uncond = model.projection(noisy)
    target = noise - clean
    guided = uncond + 3 * (target - uncond)
    per_image = .5 * (prediction - target).square().mean((1, 2, 3)) + \
        .5 * (prediction - guided).square().mean((1, 2, 3))
    torch.testing.assert_close(loss, per_image.mean())
    for actual, reference in zip(torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True),
                                 torch.autograd.grad(per_image.mean(), tuple(model.parameters()))):
        torch.testing.assert_close(actual, reference)
    assert metrics["guidance_loss_selected_fraction"] == 1.0


def test_cfg_null_rows_keep_ordinary_mse_and_skip_the_teacher():
    model, clean, noise, sigma, loss, _ = _run(
        {"guidance_loss_weight": 1.0, "guidance_loss_mix_mode": "blend",
         "guidance_loss_scale": 2.0, "guidance_loss_schedule": "constant"},
        cfg_drop_mask=torch.tensor([False, True]))
    teacher, _ = model.calls
    assert teacher["x"].shape[0] == 1
    view = sigma.view(-1, 1, 1, 1)
    noisy = (1 - view) * clean + view * noise
    prediction = model.projection(noisy) + 1
    target = noise - clean
    with torch.no_grad():
        uncond = model.projection(noisy[:1])
    guided = uncond + 2 * (target[:1] - uncond)
    expected = torch.stack([(prediction[0] - guided[0]).square().mean(),
                            (prediction[1] - target[1]).square().mean()]).mean()
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("config", [{}, {"guidance_loss_weight": 0.0}])
def test_zero_weight_runs_no_extra_forward(config):
    model, *_ = _run(config)
    assert len(model.calls) == 1


def test_partition_training_refuses_guidance():
    from core.models.anima.partition_training import validate_training_options
    with pytest.raises(ValueError, match="guidance"):
        validate_training_options({"dit_partition_training_enabled": True, "guidance_loss_weight": .5})
