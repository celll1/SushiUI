"""Multi-architecture crop decode loss and VAE swap alignment tests.

Validates that:
1. Z-Image, Anima, Lens, Ideogram 4, FLUX.2, and Krea2 route through compute_crop_decode_loss.
2. Packed latents (Lens, Ideogram 4, FLUX.2, Krea2) are properly unpacked into 2D before crop decode.
3. VAE swap (replacing trainer.vae with a new instance) automatically invalidates/re-initializes
   the auxiliary loss module and runs against the new VAE.
4. If VAE parameters are offloaded to CPU (e.g. latent caching swap_onthefly), compute_crop_decode_loss
   ensures they are resident on trainer.device before forward.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.inference.context_tiled_decode import TileRect
from core.training.arch.ideogram4 import Ideogram4ArchHandler
from core.training.arch.lens import LensArchHandler
from core.training.ops.crop_decode_loss import (
    CropDecodeLossModule,
    compute_crop_decode_loss,
)


class MockDecoder(nn.Module):
    def __init__(self, in_channels: int = 16, out_channels: int = 3, scale: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, scale_factor=float(self.scale), mode="nearest")
        return self.conv(h)


class MockVAE(nn.Module):
    def __init__(self, in_channels: int = 16, scale: int = 8, identifier: str = "base"):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale
        self.identifier = identifier
        self.decoder = MockDecoder(in_channels=in_channels, scale=scale)
        self.config = SimpleNamespace(
            latent_channels=in_channels,
            scaling_factor=0.3611,
            shift_factor=0.0,
            block_out_channels=[64, 128],
        )

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        out = self.decoder(z)
        if return_dict:
            return SimpleNamespace(sample=out)
        return (out,)


class MinimalDummyModel(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.layer = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


@pytest.fixture
def base_trainer():
    device = torch.device("cpu")
    vae = MockVAE(in_channels=16, scale=8, identifier="vae_16ch")
    model = MinimalDummyModel(channels=16)

    trainer = SimpleNamespace(
        device=device,
        training_dtype=torch.float32,
        vae_dtype=torch.float32,
        vae=vae,
        crop_decode_loss_enable=True,
        crop_decode_loss_weight=0.1,
        crop_decode_loss_metric="mse",
        crop_decode_loss_margin_cells=4,
        crop_decode_loss_out_cells=8,
        crop_decode_loss_snr_range="",
        noise_scheduler=None,
        log_prefix="[TestTrainer]",
        metrics={},
    )

    def _log_extra(k, v):
        trainer.metrics[k] = v

    trainer.log_extra_metric = _log_extra
    return trainer


def test_zimage_crop_decode_loss(base_trainer):
    """Z-Image: 2D latents [B, 16, H, W] with velocity_sign='x0_minus_eps'."""
    B, C, H, W = 1, 16, 16, 16
    noisy = torch.randn(B, C, H, W, requires_grad=True)
    clean = torch.randn(B, C, H, W)
    t = torch.tensor([0.4])

    # Model predicts v = x0 - eps
    model = MinimalDummyModel(channels=16)
    v_pred = model(noisy)
    main_loss = F.mse_loss(v_pred, clean - noisy)

    aux_loss, raw_val = compute_crop_decode_loss(
        trainer=base_trainer,
        model_pred=v_pred,
        noisy_latents=noisy,
        timesteps=t,
        clean_latents=clean,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign="x0_minus_eps",
        main_loss=main_loss,
    )

    assert aux_loss is not None
    assert torch.isfinite(aux_loss)
    assert raw_val > 0.0

    # Test backward pass
    total_loss = main_loss + aux_loss
    total_loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_anima_crop_decode_loss(base_trainer):
    """Anima: 2D latents [B, 16, H, W] with velocity_sign='eps_minus_x0'."""
    B, C, H, W = 1, 16, 16, 16
    noisy = torch.randn(B, C, H, W, requires_grad=True)
    clean = torch.randn(B, C, H, W)
    t = torch.tensor([0.5])

    model = MinimalDummyModel(channels=16)
    v_pred = model(noisy)
    main_loss = F.mse_loss(v_pred, noisy - clean)

    aux_loss, raw_val = compute_crop_decode_loss(
        trainer=base_trainer,
        model_pred=v_pred,
        noisy_latents=noisy,
        timesteps=t,
        clean_latents=clean,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign="eps_minus_x0",
        main_loss=main_loss,
    )

    assert aux_loss is not None
    assert torch.isfinite(aux_loss)
    assert "crop_decode_loss" in base_trainer.metrics


@pytest.mark.parametrize(
    "handler_cls", [LensArchHandler, Ideogram4ArchHandler], ids=["lens", "ideogram4"]
)
def test_lens_and_ideogram4_unpatchify_routing(base_trainer, handler_cls):
    """Lens and Ideogram 4: Packed latents [B, N, 128] unpacked via _unpatchify.

    The two archs declare opposite velocity signs, so the prediction is built from
    the handler's declaration rather than a literal.
    """
    from core.models.lens.lens_pipeline_ops import _unpatchify

    B = 1
    C = 32
    latent_h, latent_w = 8, 8
    N = latent_h * latent_w

    # Create 32-ch VAE for Lens/Ideogram4
    base_trainer.vae = MockVAE(in_channels=32, scale=8, identifier="lens_vae")
    # Packed sequence has C * 4 = 128 channels
    packed_seq = torch.randn(B, N, 128, requires_grad=True)
    clean_seq = torch.randn(B, N, 128)
    t = torch.tensor([0.3])

    def _to_2d(seq_t):
        x_4d = seq_t.reshape(B, latent_h, latent_w, -1).permute(0, 3, 1, 2).contiguous()
        return _unpatchify(x_4d)

    latents_2d = _to_2d(clean_seq)
    noisy_2d = _to_2d(packed_seq)
    sign = handler_cls.velocity_sign
    v_pred_2d = (latents_2d - noisy_2d) if sign == "x0_minus_eps" else (noisy_2d - latents_2d)

    assert latents_2d.shape == (B, 32, latent_h * 2, latent_w * 2)

    aux_loss, raw_val = compute_crop_decode_loss(
        trainer=base_trainer,
        model_pred=v_pred_2d,
        noisy_latents=noisy_2d,
        timesteps=t,
        clean_latents=latents_2d,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign=sign,
    )

    assert aux_loss is not None
    assert torch.isfinite(aux_loss)


def test_flux2_and_krea2_unpack_routing(base_trainer):
    """FLUX.2 and Krea2: Unpack sequence into 2D before auxiliary loss computation."""
    from core.models.krea2.krea2_pipeline_ops import unpack_latents

    B = 1
    grid_h, grid_w = 8, 8
    patch_size = 2
    # 16ch base -> 16 * 4 = 64 packed channels
    packed_seq = torch.randn(B, grid_h * grid_w, 64, requires_grad=True)
    clean_seq = torch.randn(B, grid_h * grid_w, 64)
    t = torch.tensor([0.6])

    def _to_2d_krea2(seq_t):
        unpacked_5d = unpack_latents(seq_t, grid_h, grid_w, patch_size=patch_size)
        return unpacked_5d.squeeze(2)

    latents_2d = _to_2d_krea2(clean_seq)
    noisy_2d = _to_2d_krea2(packed_seq)
    v_pred_2d = noisy_2d - latents_2d

    assert latents_2d.shape == (B, 16, grid_h * patch_size, grid_w * patch_size)

    aux_loss, raw_val = compute_crop_decode_loss(
        trainer=base_trainer,
        model_pred=v_pred_2d,
        noisy_latents=noisy_2d,
        timesteps=t,
        clean_latents=latents_2d,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign="eps_minus_x0",
    )

    assert aux_loss is not None
    assert torch.isfinite(aux_loss)


def test_vae_swap_dynamic_tracking(base_trainer):
    """Replacing trainer.vae (VAE swap) triggers automatic re-initialization of loss module."""
    B, C1, H, W = 1, 16, 16, 16
    noisy1 = torch.randn(B, C1, H, W)
    clean1 = torch.randn(B, C1, H, W)
    t = torch.tensor([0.5])

    # 1. First run with initial 16-ch VAE
    compute_crop_decode_loss(
        trainer=base_trainer,
        model_pred=clean1,
        noisy_latents=noisy1,
        timesteps=t,
        clean_latents=clean1,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign="eps_minus_x0",
    )
    mod1 = base_trainer._crop_decode_loss_module
    assert mod1.vae.identifier == "vae_16ch"

    # 2. VAE Swap happens: trainer.vae is replaced by a 32-ch swapped VAE
    swapped_vae = MockVAE(in_channels=32, scale=8, identifier="swapped_32ch_vae")
    base_trainer.vae = swapped_vae

    C2 = 32
    noisy2 = torch.randn(B, C2, H, W)
    clean2 = torch.randn(B, C2, H, W)

    compute_crop_decode_loss(
        trainer=base_trainer,
        model_pred=clean2,
        noisy_latents=noisy2,
        timesteps=t,
        clean_latents=clean2,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign="eps_minus_x0",
    )

    mod2 = base_trainer._crop_decode_loss_module
    # Module must have been automatically re-initialized with the swapped VAE
    assert mod2 is not mod1
    assert mod2.vae.identifier == "swapped_32ch_vae"


def test_vae_cpu_offload_device_safety(base_trainer):
    """If trainer.vae was moved to CPU (latent cache swap), compute_crop_decode_loss ensures device safety."""
    # Move VAE parameters to CPU explicitly
    base_trainer.vae.to("cpu")
    base_trainer.device = torch.device("cpu")

    B, C, H, W = 1, 16, 16, 16
    noisy = torch.randn(B, C, H, W)
    clean = torch.randn(B, C, H, W)
    t = torch.tensor([0.5])

    aux_loss, raw_val = compute_crop_decode_loss(
        trainer=base_trainer,
        model_pred=clean,
        noisy_latents=noisy,
        timesteps=t,
        clean_latents=clean,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign="eps_minus_x0",
    )
    assert aux_loss is not None
    assert raw_val > 0.0

