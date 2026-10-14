"""Guidance-target loss across the flow architectures that share the caption-stage null.

Each arch's step must run its null-condition teacher (no grad, selected rows,
same noisy input) and regress onto u + g * (v - u).
"""
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

BATCH = 2
CONFIG = {"guidance_loss_weight": .5, "guidance_loss_mix_mode": "blend",
          "guidance_loss_scale": 3.0, "guidance_loss_schedule": "constant"}


class Recorder(nn.Module):
    """Velocity = linear(x) + mean(context), recording every call."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.5, 1.5, channels))
        self.calls = []

    def velocity(self, x, context, channel_dim):
        shape = [1] * x.dim()
        shape[channel_dim] = -1
        shift = context.float().flatten(1).mean(1).view(-1, *([1] * (x.dim() - 1)))
        out = x * self.weight.view(shape) + shift
        self.calls.append({"x": x.detach().clone(), "context": context.detach().clone(),
                           "grad": torch.is_grad_enabled(), "out": out})
        return out


def _trainer(model, noise, **extra):
    return SimpleNamespace(config=dict(CONFIG), device=torch.device("cpu"), training_dtype=torch.float32,
        transformer=model, transformer_original=model, mixed_precision=False, timestep_sampler=None,
        noise_scheduler=None, snr_regularization_loss=None, energy_regularization_loss=None,
        reconstruction_loss_weight=0.0, _active_mnt_noise=noise, log_prefix="[test]",
        log_extra_metric=lambda name, value: None,
        stash_cfg_null_per_sample_loss=lambda prediction, target: None, **extra)


def _zimage(cfg_drop_mask):
    from core.training.ops.zimage_ops import train_step

    class ZImage(Recorder):
        def forward(self, x, t, cap_feats, cap_mask):
            return self.velocity(x, cap_feats, 1), None

    model, clean, noise = ZImage(4), torch.randn(BATCH, 4, 6, 6), torch.randn(BATCH, 4, 6, 6)
    null = (torch.full((7, 8), -1.0), torch.ones(7, dtype=torch.bool))
    trainer = _trainer(model, noise, _guidance_null_encoding=null)
    loss, _, _ = train_step(trainer, latents=clean, prompt_embeds=torch.ones(BATCH, 5, 8),
                            attention_mask=torch.ones(BATCH, 5, dtype=torch.bool),
                            timesteps=torch.tensor([.3, .7]), cfg_drop_mask=cfg_drop_mask)
    return model, loss, clean - noise, lambda out: out.squeeze(2)


def _krea2(cfg_drop_mask):
    from core.training.ops.krea2_ops import train_step

    class Krea2(Recorder):
        dtype = torch.float32

        def forward(self, hidden_states, encoder_hidden_states, timestep, position_ids,
                    encoder_attention_mask, return_dict):
            return (self.velocity(hidden_states, encoder_hidden_states, 2),)

    model, clean, noise = Krea2(8), torch.randn(BATCH, 9, 8), torch.randn(BATCH, 9, 8)
    null = (torch.full((1, 6, 2, 4), -1.0), torch.ones(6, dtype=torch.long))
    trainer = _trainer(model, noise, _guidance_null_encoding=null, krea2_discrete_flow_shift=1.0)
    loss, _, _ = train_step(trainer, latents=clean, encoder_features=torch.ones(BATCH, 5, 2, 4),
                            encoder_mask=torch.ones(BATCH, 5, dtype=torch.long), timesteps=torch.tensor([.3, .7]),
                            latent_h=3, latent_w=3, cfg_drop_mask=cfg_drop_mask)
    return model, loss, noise - clean, lambda out: out


def _flux2(cfg_drop_mask):
    from core.training.base_trainer import BaseTrainer
    from core.training.ops.flux2_ops import train_step

    class Flux2(Recorder):
        def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, guidance,
                    return_dict):
            return (self.velocity(hidden_states, encoder_hidden_states, 2),)

    model, clean, noise = Flux2(8), torch.randn(BATCH, 9, 8), torch.randn(BATCH, 9, 8)
    null = (torch.full((1, 6, 8), -1.0), None)
    trainer = _trainer(model, noise, _guidance_null_encoding=null, flux2_transformer_wrapper=None,
                       noise_process="flow", prediction_target="velocity")
    trainer._flux2_prepare_text_ids = lambda embeds: BaseTrainer._flux2_prepare_text_ids(trainer, embeds)
    loss, _, _ = train_step(trainer, latents=clean, prompt_embeds=torch.ones(BATCH, 5, 8),
                            img_ids=torch.zeros(BATCH, 9, 4, dtype=torch.long),
                            txt_ids=torch.zeros(BATCH, 5, 4, dtype=torch.long),
                            timesteps=torch.tensor([.3, .7]), cfg_drop_mask=cfg_drop_mask)
    return model, loss, noise - clean, lambda out: out


def _ltx2(cfg_drop_mask):
    from core.training.ops.ltx2_ops import _pack_latents, train_step

    class Rope:
        def prepare_video_coords(self, b, t, h, w, device, fps=1.0):
            return torch.zeros(b, 3, t * h * w, 2)

    class Ltx2(Recorder):
        rope = Rope()

        def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, **_kwargs):
            return self.velocity(hidden_states, encoder_hidden_states, 2), torch.zeros_like(audio_hidden_states)

    model, clean, noise = Ltx2(6), torch.randn(BATCH, 6, 2, 2, 2), torch.randn(BATCH, 6, 2, 2, 2)
    null = (torch.full((7, 8), -1.0), {"audio_text_embedding": torch.zeros(7, 8),
                                       "mask": torch.ones(7, dtype=torch.long)})
    trainer = _trainer(model, noise, _guidance_null_encoding=null, vae=None, tread_config=None,
                       blockskip_config=None, ltx2_block_loop_wrapper=None)
    loss, _, _ = train_step(trainer, latents=clean, prompt_embeds=torch.ones(BATCH, 5, 8),
                            ltx2_aux={"audio_text_embedding": torch.zeros(BATCH, 5, 8),
                                      "mask": torch.ones(BATCH, 5, dtype=torch.long)},
                            timesteps=torch.tensor([.3, .7]), cfg_drop_mask=cfg_drop_mask)
    return model, loss, _pack_latents(noise - clean), lambda out: out


def _lens(cfg_drop_mask):
    from core.training.ops import lens_ops

    class Lens(Recorder):
        def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep, img_shapes):
            return self.velocity(hidden_states, torch.stack(encoder_hidden_states, 1), 2)

    model, clean, noise = Lens(8), torch.randn(BATCH, 9, 8), torch.randn(BATCH, 9, 8)
    trainer = _trainer(model, noise)
    null = lambda features, mask: lens_ops.apply_cfg_null_collated(
        features, mask, torch.ones(features.shape[0], dtype=torch.bool))
    loss, _, _ = lens_ops.train_step(trainer, latents=clean, encoder_features=torch.ones(BATCH, 2, 5, 4),
                                     encoder_mask=torch.ones(BATCH, 5, dtype=torch.bool),
                                     timesteps=torch.tensor([.3, .7]), latent_h=3, latent_w=3,
                                     null_condition=null, guidance_ineligible=cfg_drop_mask)
    return model, loss, noise - clean, lambda out: out


def _minit2i(cfg_drop_mask):
    from core.training.ops import minit2i_ops

    class MiniT2I(Recorder):
        dtype = torch.float32

        def forward(self, x, t, embeds, mask):
            return self.velocity(x, mask.float().unsqueeze(-1), 1)

    model, images = MiniT2I(3), torch.randn(BATCH, 3, 4, 4)
    noise = torch.randn(BATCH, 3, 4, 4)
    trainer = _trainer(model, noise, minit2i_noise_scale=2.0)
    null = lambda embeds, mask: minit2i_ops.apply_cfg_null_collated(
        embeds, mask, torch.ones(embeds.shape[0], dtype=torch.bool))
    t = torch.tensor([.3, .7])
    loss, _, _ = minit2i_ops.train_step(trainer, images=images, text_embeds=torch.ones(BATCH, 5, 4),
                                        attention_mask=torch.ones(BATCH, 5, dtype=torch.long),
                                        timesteps=t, null_condition=null, guidance_ineligible=cfg_drop_mask)
    x_t = model.calls[-1]["x"]
    denom = (1 - t).clamp_min(.05).view(-1, 1, 1, 1)
    return model, loss, (images - x_t) / denom, lambda out: (out - x_t) / denom


ARCHS = {"zimage": _zimage, "krea2": _krea2, "flux2": _flux2, "ltx2": _ltx2, "lens": _lens, "minit2i": _minit2i}
NULL_VALUE = {"lens": 0, "minit2i": 0}


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_teacher_runs_on_the_null_and_target_is_exact(arch):
    model, loss, target, image = ARCHS[arch](None)
    teacher, student = model.calls
    assert not teacher["grad"] and student["grad"]
    assert torch.equal(teacher["x"], student["x"])
    null = NULL_VALUE.get(arch, -1)
    assert bool((teacher["context"] == null).all()) and bool((student["context"] == 1).all())
    prediction, uncond = image(student["out"]), image(teacher["out"])
    guided = uncond + 3 * (target - uncond)
    per_image = .5 * (prediction - target).square().flatten(1).mean(1) + \
        .5 * (prediction - guided).square().flatten(1).mean(1)
    torch.testing.assert_close(loss, per_image.mean())


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_cfg_null_rows_are_not_taught(arch):
    model, *_ = ARCHS[arch](torch.tensor([False, True]))
    assert model.calls[0]["x"].shape[0] == 1 and not model.calls[0]["grad"]


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_zero_weight_runs_no_teacher(arch, monkeypatch):
    monkeypatch.setitem(CONFIG, "guidance_loss_weight", 0.0)
    model, *_ = ARCHS[arch](None)
    assert len(model.calls) == 1


def test_distilled_checkpoints_are_refused():
    from core.training.arch.flux2 import Flux2ArchHandler
    from core.training.arch.krea2 import Krea2ArchHandler
    assert Flux2ArchHandler().guidance_loss_refusal(SimpleNamespace(is_distilled=True))
    assert Flux2ArchHandler().guidance_loss_refusal(SimpleNamespace(is_distilled=False)) is None
    assert Krea2ArchHandler().guidance_loss_refusal(SimpleNamespace(krea2_is_distilled=True))
    from core.training.arch.zimage import ZImageArchHandler
    assert ZImageArchHandler().guidance_loss_refusal(SimpleNamespace(model_path="M:/z-image-turbo.safetensors"))
    assert ZImageArchHandler().guidance_loss_refusal(SimpleNamespace(model_path="M:/z-image.safetensors")) is None
