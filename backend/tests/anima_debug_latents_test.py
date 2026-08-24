"""Anima training debug-latent dumps.

Guards the defect where ``anima_ops.train_step`` declared ``debug_save_path`` /
``debug_captions`` / ``debug_reference_image_paths`` in its signature and never
read them, so ``debug_latents`` silently produced nothing for the arch while the
caller looked correct.

No real model is loaded: the DiT is a stand-in that analytically returns the
exact rectified-flow target, which pins the sign of the x0 reconstruction.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.ops import anima_ops


LATENT_SHAPE = (2, 4, 8, 8)  # [B, C, H, W] — channel count is irrelevant here


class PerfectVelocityDiT(nn.Module):
    """Returns the exact rectified-flow target for the given latents.

    ``x_t = (1 - sigma) * x_0 + sigma * noise`` so ``noise`` is recoverable from
    ``x_t`` and the known ``x_0``; the target is ``v = noise - x_0``.
    """

    def __init__(self, clean_latents: torch.Tensor):
        super().__init__()
        self.clean = clean_latents
        self.seen_x = None

    def forward(self, x, timesteps, context, padding_mask,
                target_input_ids, target_attention_mask, source_attention_mask):
        self.seen_x = x
        x_t = x.squeeze(2)
        sigma = timesteps.view(-1, *([1] * (x_t.dim() - 1))).to(x_t.dtype)
        noise = (x_t - (1.0 - sigma) * self.clean) / sigma
        return (noise - self.clean).unsqueeze(2)


class ConstantVelocityDiT(nn.Module):
    def __init__(self, value: float = 0.25):
        super().__init__()
        self.value = value

    def forward(self, x, timesteps, context, padding_mask,
                target_input_ids, target_attention_mask, source_attention_mask):
        return torch.full_like(x, self.value)


def _make_trainer(transformer):
    return SimpleNamespace(
        log_prefix="[test]",
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        timestep_sampler=None,
        mixed_precision=False,
        transformer=transformer,
        reconstruction_loss_weight=0.0,
        tread_config=None,
        block_skip_config=None,
        blockskip_config=None,
    )


def _aux(batch: int = LATENT_SHAPE[0], length: int = 5):
    return {
        "source_mask": torch.ones(batch, length, dtype=torch.long),
        "t5_input_ids": torch.zeros(batch, length, dtype=torch.long),
        "t5_attn_mask": torch.ones(batch, length, dtype=torch.long),
    }


def _run(transformer, debug_save_path, sigma=0.3, **kwargs):
    trainer = _make_trainer(transformer)
    latents = torch.randn(*LATENT_SHAPE)
    prompt_embeds = torch.zeros(LATENT_SHAPE[0], 5, 8)
    timesteps = torch.full((LATENT_SHAPE[0],), sigma)
    return latents, anima_ops.train_step(
        trainer,
        latents=latents,
        prompt_embeds=prompt_embeds,
        anima_aux=_aux(),
        timesteps=timesteps,
        debug_save_path=debug_save_path,
        **kwargs,
    )


def test_no_debug_path_writes_nothing(tmp_path):
    _run(ConstantVelocityDiT(), None)
    assert list(tmp_path.iterdir()) == []


def test_debug_path_writes_dump_with_endpoint_keys(tmp_path):
    out = tmp_path / "step_000010"
    _run(
        ConstantVelocityDiT(),
        out,
        debug_captions=["a caption", "second"],
        debug_reference_image_paths=[None, "/tmp/ref.png"],
    )

    files = list(out.glob("latents_t*.pt"))
    assert len(files) == 1
    # visualize_debug_latent parses the timestep out of the filename as a float.
    assert float(files[0].stem.replace("latents_t", "")) == pytest.approx(0.3)

    data = torch.load(files[0], map_location="cpu")
    # Keys consumed by routes.visualize_debug_latent.
    for key in ("latents", "noisy_latents", "predicted_velocity",
                "actual_velocity", "predicted_latent", "timestep",
                "loss", "recon_loss", "model_type"):
        assert key in data, key
    assert data["model_type"] == "anima"
    assert data["caption"] == "a caption"
    assert data["reference_image_path"] == "/tmp/ref.png"
    # latent_to_image() indexes [C, H, W] after dropping a leading batch dim.
    for key in ("latents", "noisy_latents", "predicted_velocity",
                "actual_velocity", "predicted_latent"):
        assert data[key].shape == (1,) + LATENT_SHAPE[1:], key


def test_predicted_latent_matches_noising_definition(tmp_path):
    """A perfect velocity prediction must reconstruct x_0 exactly.

    Anima noises with ``x_t = (1 - sigma) x_0 + sigma * noise`` and targets
    ``v = noise - x_0``, so ``x_0 = x_t - sigma * v``. With Z-Image's opposite
    sign (``x_t + sigma * v``) this lands on ``x_0 + 2 sigma v`` instead.
    """
    out = tmp_path / "step_000000"
    latents = torch.randn(*LATENT_SHAPE)
    trainer = _make_trainer(PerfectVelocityDiT(latents))
    sigma = 0.4
    anima_ops.train_step(
        trainer,
        latents=latents,
        prompt_embeds=torch.zeros(LATENT_SHAPE[0], 5, 8),
        anima_aux=_aux(),
        timesteps=torch.full((LATENT_SHAPE[0],), sigma),
        debug_save_path=out,
    )

    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert torch.allclose(data["predicted_latent"], latents[0:1], atol=1e-4)
    assert torch.allclose(data["predicted_velocity"], data["actual_velocity"], atol=1e-5)
    assert data["recon_loss"] == pytest.approx(0.0, abs=1e-8)
    # noisy = x_0 + sigma * v, the same relation read from the other direction.
    assert torch.allclose(
        data["noisy_latents"], latents[0:1] + sigma * data["actual_velocity"], atol=1e-4
    )


def test_dump_failure_does_not_break_train_step(tmp_path, monkeypatch, capsys):
    """A diagnostic that only runs every N steps must not kill a long run."""
    def _boom(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(torch, "save", _boom)
    _latents, result = _run(ConstantVelocityDiT(), tmp_path / "step_000010")
    loss, pred_loss_value, _recon = result

    assert torch.isfinite(loss)
    assert isinstance(pred_loss_value, float)
    out = capsys.readouterr().out
    assert "[debug_latents] save failed" in out
    assert "No space left on device" in out


def test_predicted_latent_is_xt_minus_sigma_v(tmp_path):
    """Sign check that does not depend on the prediction being correct."""
    out = tmp_path / "step_000001"
    sigma = 0.7
    _run(ConstantVelocityDiT(0.25), out, sigma=sigma)
    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    expected = data["noisy_latents"] - sigma * data["predicted_velocity"]
    assert torch.allclose(data["predicted_latent"], expected, atol=1e-5)
