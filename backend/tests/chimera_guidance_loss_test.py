"""SenseNova SDXL Chimera guidance-target loss (plain flow-velocity contract)."""
from types import SimpleNamespace

import pytest
import torch

from core.models.sensenova_sdxl_chimera.attention_processor import ChimeraAttnProcessor
from core.training.arch.sensenova_sdxl_chimera import SenseNovaSDXLChimeraArchHandler
from core.training.ops.sensenova_sdxl_chimera_ops import train_step


class FakeUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.5))
        self.attn_processors = {"fake": ChimeraAttnProcessor()}
        self.calls = []

    def forward(self, sample, timestep, *, encoder_hidden_states, **_kwargs):
        out = sample * self.scale + encoder_hidden_states.float().flatten(1).mean(1).view(-1, 1, 1, 1)
        self.calls.append({"x": sample.detach().clone(), "grad": torch.is_grad_enabled(), "out": out,
                           "positions": self.attn_processors["fake"].context.context_positions.clone()})
        return (out,)


def _run(weight, cfg_drop_mask=None):
    torch.manual_seed(5)
    unet = FakeUNet()
    null = (torch.full((1, 2, 6), -1.0), {
        "pooled_text_embeds": torch.zeros(1, 5),
        "context_positions": torch.full((1, 2, 3), 7.0),
        "context_attention_mask": torch.ones(1, 2, dtype=torch.bool),
    })
    trainer = SimpleNamespace(
        config={"chimera_training_stage": "unet", "guidance_loss_weight": weight,
                "guidance_loss_mix_mode": "blend", "guidance_loss_scale": 2.0,
                "guidance_loss_schedule": "constant"},
        device="cpu", training_dtype=torch.float32, unet=unet,
        _guidance_null_encoding=null, log_extra_metric=lambda *a: None)
    ctx = SimpleNamespace(
        text_embeddings=torch.ones(2, 3, 6),
        attention_mask={"pooled_text_embeds": torch.zeros(2, 5),
                        "context_positions": torch.zeros(2, 3, 3),
                        "context_attention_mask": torch.ones(2, 3, dtype=torch.bool)},
        latents=torch.randn(2, 4, 2, 2), timesteps=torch.tensor([.25, .75]), time_ids=None,
        cfg_drop_mask=cfg_drop_mask)
    loss, _, _ = train_step(trainer, ctx)
    return unet, ctx, loss


def test_teacher_uses_the_null_context_then_restores_the_caption_context():
    unet, ctx, loss = _run(.5)
    teacher, student = unet.calls
    assert not teacher["grad"] and student["grad"]
    assert bool((teacher["positions"] == 7).all()) and bool((student["positions"] == 0).all())
    assert torch.equal(teacher["x"], student["x"])
    noise = (teacher["x"] - ctx.timesteps.view(-1, 1, 1, 1) * ctx.latents) / (1 - ctx.timesteps.view(-1, 1, 1, 1))
    target = ctx.latents - noise
    prediction, uncond = student["out"], teacher["out"]
    guided = uncond + 2 * (target - uncond)
    expected = (.5 * (prediction - target).square().flatten(1).mean(1)
                + .5 * (prediction - guided).square().flatten(1).mean(1)).mean()
    torch.testing.assert_close(loss, expected)


def test_zero_weight_and_cfg_null_rows():
    assert len(_run(0.0)[0].calls) == 1
    unet, *_ = _run(1.0, cfg_drop_mask=torch.tensor([True, False]))
    assert unet.calls[0]["x"].shape[0] == 1


@pytest.mark.parametrize("prediction", ["endpoint_observable_residual", "polar_flow_v4"])
def test_other_prediction_contracts_are_refused(prediction):
    handler = SenseNovaSDXLChimeraArchHandler()
    trainer = SimpleNamespace(config={"chimera_training_stage": "unet"}, chimera_prediction={"type": prediction})
    assert "flow_velocity" in handler.guidance_loss_refusal(trainer)
    trainer.chimera_prediction = {"type": "flow_velocity"}
    assert handler.guidance_loss_refusal(trainer) is None
