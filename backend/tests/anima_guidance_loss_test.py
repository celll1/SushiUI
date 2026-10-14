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


@pytest.mark.parametrize("cap", [0.0, 0.05])
def test_partitioned_guidance_matches_the_stitched_full_image_objective(cap):
    from core.models.anima.anima_models import Anima, LLMAdapter
    from core.models.anima.checkpointing import configure
    from core.models.anima.partition_training import backward_regions, training_plan
    from core.training import guidance_loss

    torch.manual_seed(11)
    model = Anima(max_img_h=64, max_img_w=64, max_frames=1, in_channels=4, out_channels=4,
                  model_channels=24, num_blocks=2, num_heads=2, mlp_ratio=1., crossattn_emb_channels=12,
                  use_adaln_lora=True, adaln_lora_dim=8, use_llm_adapter=False)
    model.llm_adapter = LLMAdapter(12, 12, 12, num_layers=1, num_heads=2, self_attn=True)
    model.use_llm_adapter = True
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, .05)
    config = dict(dit_partition_training_enabled=True, dit_partition_mode="fixed",
                  dit_partition_fixed_count=2, dit_partition_halo_tokens=2,
                  guidance_loss_weight=.5, guidance_loss_mix_mode="blend", guidance_loss_scale=3.0,
                  guidance_loss_schedule="sigma", guidance_loss_extrapolation_max_rms=cap)
    model.train()
    configure(model, config)
    null = (torch.zeros(4, 12), {name: torch.ones(4, dtype=torch.long)
                                 for name in ("source_mask", "t5_input_ids", "t5_attn_mask")})
    metrics = {}
    trainer = SimpleNamespace(config=config, transformer=model, device=torch.device("cpu"),
        training_dtype=torch.float32, reconstruction_loss_weight=0.0, mixed_precision=False,
        _guidance_null_encoding=null, log_extra_metric=metrics.__setitem__)
    calls = []
    handle = model.register_forward_hook(
        lambda module, args, kwargs, out: calls.append((torch.is_grad_enabled(), kwargs["partition_box"],
                                                        kwargs["context"].detach().clone(), out.detach())),
        with_kwargs=True)
    clean = torch.randn(2, 4, 16, 24)
    noise, sigma = torch.randn_like(clean), torch.tensor([.3, .7])
    plan = training_plan(trainer, clean)
    try:
        total, _, _ = backward_regions(
            trainer, clean=clean, noise=noise, sigma=sigma, prompt_embeds=torch.ones(2, 3, 12),
            anima_aux={name: torch.ones(2, 3, dtype=torch.long)
                       for name in ("source_mask", "t5_input_ids", "t5_attn_mask")},
            plan=plan, backward_scale=1.0, cfg_drop_mask=torch.tensor([False, True]))
    finally:
        handle.remove()

    teachers = [call for call in calls if not call[0]]
    students = [call for call in calls if call[0]]
    assert len(teachers) == len(students) == len(plan.regions)
    assert all(call[2].shape[0] == 1 and not call[2].any() for call in teachers)
    assert all(bool((call[2] == 1).all()) for call in students)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    def stitch(outputs, rows):
        full = torch.zeros(rows, *clean.shape[1:])
        for region, (_, box, _, out) in zip(plan.regions, outputs):
            assert box == region.input
            core = region.core_in_input
            full[..., 2*region.core.top:2*region.core.bottom, 2*region.core.left:2*region.core.right] = \
                out.squeeze(2)[..., 2*core.top:2*core.bottom, 2*core.left:2*core.right]
        return full

    prediction, uncond = stitch(students, 2), torch.zeros_like(clean)
    uncond[:1] = stitch(teachers, 1)
    target = noise - clean
    condition = guidance_loss.condition(config, torch.tensor([True, False]))
    step_plan = guidance_loss.plan(trainer, sigma, torch.tensor([False, True]))
    effective = guidance_loss.effective_scale(trainer, step_plan, target, uncond, sigma)
    expected, _, _ = guidance_loss.objective(prediction, target, uncond, sigma, condition,
                                             mix=step_plan.mix, guide_scale_override=effective)
    assert total == pytest.approx(float(expected), rel=1e-5)
    assert metrics["guidance_loss_selected_fraction"] == .5
