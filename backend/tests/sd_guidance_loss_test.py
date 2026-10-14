"""SD1.5 / SDXL guidance-target loss: DDPM noise-level mapping and the UNet teacher."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from diffusers import DDPMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.ops import sd_sdxl_ops


class FakeUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.75))
        self.calls = []

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None):
        out = sample * self.scale + encoder_hidden_states.float().flatten(1).mean(1).view(-1, 1, 1, 1)
        if added_cond_kwargs is not None:
            out = out + added_cond_kwargs["text_embeds"].float().mean(1).view(-1, 1, 1, 1)
        self.calls.append({"x": sample.detach().clone(), "t": timestep.clone(),
                           "grad": torch.is_grad_enabled(), "out": out})
        return SimpleNamespace(sample=out)


def _run(sdxl, prediction_target="epsilon", min_snr_gamma=0.0, cfg_drop_mask=None, weight=.5):
    torch.manual_seed(1)
    unet, scheduler = FakeUNet(), DDPMScheduler(num_train_timesteps=1000)
    if prediction_target == "velocity":
        scheduler.register_to_config(prediction_type="v_prediction")
    latents, noise = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
    pooled = torch.ones(2, 3) if sdxl else None
    null = (torch.full((1, 6, 8), -1.0), torch.zeros(1, 3) if sdxl else None)
    trainer = SimpleNamespace(
        config={"guidance_loss_weight": weight, "guidance_loss_mix_mode": "blend",
                "guidance_loss_scale": 2.5, "guidance_loss_schedule": "constant"},
        unet=unet, is_sdxl=sdxl, device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, timestep_sampler=None, noise_process="ddpm",
        prediction_target=prediction_target, noise_scheduler=scheduler, min_snr_gamma=min_snr_gamma,
        reconstruction_loss_weight=0.0, snr_regularization_loss=None, energy_regularization_loss=None,
        crop_decode_loss_enable=False, _active_mnt_noise=noise, _guidance_null_encoding=null,
        log_extra_metric=lambda *a: None, log_prefix="[test]")
    loss, _, _ = sd_sdxl_ops.train_step(
        trainer, latents=latents, text_embeddings=torch.ones(2, 5, 8), pooled_embeddings=pooled,
        time_ids=torch.zeros(2, 6) if sdxl else None, timesteps=torch.tensor([.2, .7]),
        cfg_drop_mask=cfg_drop_mask)
    return unet, scheduler, latents, noise, loss


def test_noise_level_is_the_equal_snr_flow_sigma():
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    timesteps = torch.tensor([0, 250, 500, 999])
    trainer = SimpleNamespace(noise_scheduler=scheduler)
    level, eps_scale = sd_sdxl_ops.guidance_noise_level(trainer, timesteps, "ddpm", "epsilon")
    alpha_bar = scheduler.alphas_cumprod[timesteps]
    torch.testing.assert_close((level / (1 - level)) ** 2, (1 - alpha_bar) / alpha_bar, rtol=1e-4, atol=0)
    torch.testing.assert_close(eps_scale, ((1 - alpha_bar) / alpha_bar).sqrt())
    assert bool((level[1:] > level[:-1]).all()) and 0 < float(level[0]) < float(level[-1]) < 1
    flow_level, flow_scale = sd_sdxl_ops.guidance_noise_level(trainer, torch.tensor([.3]), "flow", "velocity")
    torch.testing.assert_close(flow_level, flow_scale)


@pytest.mark.parametrize("sdxl,prediction", [(False, "epsilon"), (True, "velocity")])
def test_teacher_runs_on_the_null_and_target_is_exact(sdxl, prediction):
    unet, scheduler, latents, noise, loss = _run(sdxl, prediction)
    teacher, student = unet.calls
    assert not teacher["grad"] and student["grad"]
    assert torch.equal(teacher["x"], student["x"]) and torch.equal(teacher["t"], student["t"])
    if prediction == "epsilon":
        target = noise
    else:
        alpha_bar = scheduler.alphas_cumprod[student["t"]].view(-1, 1, 1, 1)
        target = alpha_bar.sqrt() * noise - (1 - alpha_bar).sqrt() * latents
    uncond, prediction_out = teacher["out"], student["out"]
    guided = uncond + 2.5 * (target - uncond)
    expected = (.5 * (prediction_out - target).square().flatten(1).mean(1)
                + .5 * (prediction_out - guided).square().flatten(1).mean(1)).mean()
    torch.testing.assert_close(loss, expected)


def test_min_snr_weights_the_guided_per_sample_loss():
    from core.training.base_trainer import apply_snr_weight
    unet, scheduler, _, noise, loss = _run(False, min_snr_gamma=5.0)
    teacher, student = unet.calls
    guided = teacher["out"] + 2.5 * (noise - teacher["out"])
    per_sample = (.5 * (student["out"] - noise).square().flatten(1).mean(1)
                  + .5 * (student["out"] - guided).square().flatten(1).mean(1))
    torch.testing.assert_close(loss, apply_snr_weight(per_sample, student["t"], scheduler, 5.0).mean())


def test_zero_weight_and_cfg_null_rows():
    assert len(_run(False, weight=0.0)[0].calls) == 1
    unet, *_ = _run(True, cfg_drop_mask=torch.tensor([True, False]))
    assert unet.calls[0]["x"].shape[0] == 1
