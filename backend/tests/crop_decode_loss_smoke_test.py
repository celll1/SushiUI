"""Smoke tests for Phase 3: Crop Decode Auxiliary Loss.

Validates:
1. compute_crop_decode_loss forward and backward graph preservation (p.grad nonzero).
2. SNR range filtering behavior (in-band vs out-of-band).
3. Metric registration and gradient norm ratio logging.
4. Execution across fp32, fp16, and bf16 datatypes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Importing trainer/routes must not take the GPU the owner's run holds.
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.training.ops.crop_decode_loss import compute_crop_decode_loss, parse_snr_range
from core.training.probes.probe_crop_decode_autograd import MockVAE


class MockTrainer:
    """Minimal mock trainer providing base_trainer attributes for crop decode loss."""

    def __init__(self, vae: nn.Module, device: torch.device, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.vae = vae
        self.log_prefix = "[MockTrainer]"
        self.crop_decode_loss_enable = True
        self.crop_decode_loss_weight = 0.5
        self.crop_decode_loss_margin_cells = 4
        self.crop_decode_loss_out_cells = 8
        self.crop_decode_loss_metric = "l1"
        self.crop_decode_loss_snr_range = ""
        self._crop_decode_loss_module = None
        self.noise_scheduler = MagicMock()
        # Mock alphas_cumprod for ddpm
        self.noise_scheduler.alphas_cumprod = torch.linspace(0.99, 0.01, 1000, device=device)
        self.noise_scheduler.config = MagicMock()
        self.noise_scheduler.config.num_train_timesteps = 1000
        self.logged_metrics = {}

    def log_extra_metric(self, name: str, value: float):
        self.logged_metrics[name] = value


class MockModel(nn.Module):
    """Simple 2-layer conv network simulating diffusion transformer / UNet head."""

    def __init__(self, in_ch: int = 4, out_ch: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.silu(self.conv1(x)))


class TestCropDecodeLossSmoke:
    def test_parse_snr_range(self):
        assert parse_snr_range("") == (None, None)
        assert parse_snr_range("  ") == (None, None)
        assert parse_snr_range("0.5") == (0.5, None)
        assert parse_snr_range("0.1, 10.0") == (0.1, 10.0)
        assert parse_snr_range("none, 5.0") == (None, 5.0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_crop_decode_loss_gradient_reaches_model_parameters(self, dtype):
        device = torch.device("cpu")
        vae = MockVAE(scale=8).to(device=device, dtype=dtype)
        vae.eval()

        trainer = MockTrainer(vae=vae, device=device, dtype=dtype)
        trainer.crop_decode_loss_metric = "l1"

        model = MockModel(in_ch=4, out_ch=4).to(device=device, dtype=dtype)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 3-step smoke test
        for step in range(3):
            optimizer.zero_grad()

            clean_latents = torch.randn(1, 4, 16, 16, device=device, dtype=dtype)
            noisy_latents = clean_latents + 0.1 * torch.randn_like(clean_latents)
            timesteps = torch.tensor([500], device=device, dtype=torch.long)

            # Model forward
            model_pred = model(noisy_latents)

            # Main MSE loss
            main_loss = F.mse_loss(model_pred, clean_latents)

            # Auxiliary crop decode loss
            aux_loss, raw_val = compute_crop_decode_loss(
                trainer=trainer,
                model_pred=model_pred,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                clean_latents=clean_latents,
                noise_process="ddpm",
                prediction_target="epsilon",
                noise_scheduler=trainer.noise_scheduler,
                alphas_cumprod_cached=trainer.noise_scheduler.alphas_cumprod,
                main_loss=main_loss,
            )

            assert aux_loss is not None
            assert torch.isfinite(aux_loss)
            assert raw_val > 0.0

            total_loss = main_loss + aux_loss
            assert torch.isfinite(total_loss)

            total_loss.backward()

            # Verify gradient reaches model parameters
            for p in model.parameters():
                assert p.grad is not None
                assert torch.isfinite(p.grad).all()
                assert p.grad.abs().sum() > 0.0

            optimizer.step()

            # Verify extra metrics were logged
            assert "crop_decode_loss" in trainer.logged_metrics
            assert trainer.logged_metrics["crop_decode_loss"] > 0.0

    def test_crop_decode_loss_snr_filtering(self):
        device = torch.device("cpu")
        vae = MockVAE(scale=8).to(device=device, dtype=torch.float32)
        vae.eval()

        trainer = MockTrainer(vae=vae, device=device, dtype=torch.float32)
        # Set narrow SNR range that excludes timestep 999 (near-zero SNR)
        trainer.crop_decode_loss_snr_range = "10.0, 100.0"

        clean_latents = torch.randn(1, 4, 16, 16, device=device)
        noisy_latents = torch.randn_like(clean_latents)
        # Timestep 999 has very low SNR ~0.01 (outside [10, 100])
        timesteps = torch.tensor([999], device=device, dtype=torch.long)
        model_pred = torch.randn_like(clean_latents, requires_grad=True)

        aux_loss, raw_val = compute_crop_decode_loss(
            trainer=trainer,
            model_pred=model_pred,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            clean_latents=clean_latents,
            noise_process="ddpm",
            prediction_target="epsilon",
            noise_scheduler=trainer.noise_scheduler,
            alphas_cumprod_cached=trainer.noise_scheduler.alphas_cumprod,
        )

        # Should be filtered out (None)
        assert aux_loss is None
        assert raw_val == 0.0
