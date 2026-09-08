"""Tests for autograd-enabled context crop decode (Phase 2-1).

Validates backward graph preservation, geometry slicing, and coordinate math
using CPU-only lightweight mock autoencoders.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Importing trainer/routes must not take the GPU the owner's run holds.
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.inference.context_tiled_decode import TileRect
from core.training.ops.crop_decode import decode_crop_with_context, make_crop_rect


class MockVAE(nn.Module):
    """Minimal mock VAE decoder: 4ch in -> 3ch out with 8x spatial upsampling (ConvTranspose2d)."""

    def __init__(self, scale: int = 8):
        super().__init__()
        self.scale = scale
        self.spatial_compression_ratio = scale
        # 8x upsample via single strided transpose conv
        self.conv = nn.ConvTranspose2d(4, 3, kernel_size=scale, stride=scale)
        # Freeze parameters to simulate eval/frozen VAE
        for p in self.parameters():
            p.requires_grad = False

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        out = self.conv(z)
        if return_dict:
            from types import SimpleNamespace
            return SimpleNamespace(sample=out)
        return (out,)


class TestCropDecodeAutograd:
    def test_make_crop_rect_clamping(self):
        # Latent: 64x64, Crop: y in [10, 30], x in [20, 40], margin: 16
        rect = make_crop_rect(lat_h=64, lat_w=64, y0=10, y1=30, x0=20, x1=40, margin_cells=16)
        assert rect.y0 == 10 and rect.y1 == 30
        assert rect.x0 == 20 and rect.x1 == 40
        # py0 should clamp at 0 (10 - 16 < 0)
        assert rect.py0 == 0
        assert rect.py1 == 30 + 16  # 46
        # px0 should be 20 - 16 = 4
        assert rect.px0 == 4
        assert rect.px1 == 40 + 16  # 56

    def test_decode_crop_preserves_autograd_graph(self):
        vae = MockVAE(scale=8)
        vae.eval()

        # Latent with requires_grad=True
        latent = torch.randn(1, 4, 32, 32, requires_grad=True)

        rect = make_crop_rect(lat_h=32, lat_w=32, y0=8, y1=24, x0=8, x1=24, margin_cells=4)

        # Output should have spatial shape: (24-8)*8 = 128 x 128
        decoded_crop = decode_crop_with_context(vae, latent, rect, scale=8)
        assert decoded_crop.shape == (1, 3, 128, 128)
        assert decoded_crop.requires_grad is True

        # Backprop dummy scalar loss
        loss = decoded_crop.sum()
        loss.backward()

        assert latent.grad is not None
        # Padded window cells should have nonzero gradients
        grad_window = latent.grad[0, :, rect.py0:rect.py1, rect.px0:rect.px1]
        assert grad_window.abs().sum() > 0.0

        # Cells outside the padded window must have strictly ZERO gradients
        grad_outside_top = latent.grad[0, :, :rect.py0, :]
        if grad_outside_top.numel() > 0:
            assert grad_outside_top.abs().sum() == 0.0

    def test_decode_crop_value_matches_manual_crop(self):
        vae = MockVAE(scale=4)
        vae.eval()

        latent = torch.randn(1, 4, 16, 16)
        rect = make_crop_rect(lat_h=16, lat_w=16, y0=4, y1=12, x0=4, x1=12, margin_cells=2)

        # decode_crop_with_context output
        crop_out = decode_crop_with_context(vae, latent, rect, scale=4)

        # Manual padded decode & slice
        padded_latent = latent[..., rect.py0:rect.py1, rect.px0:rect.px1]
        padded_dec = vae.decode(padded_latent, return_dict=False)[0]
        ty0 = (rect.y0 - rect.py0) * 4
        tx0 = (rect.x0 - rect.px0) * 4
        th = (rect.y1 - rect.y0) * 4
        tw = (rect.x1 - rect.x0) * 4
        manual_slice = padded_dec[..., ty0:ty0 + th, tx0:tx0 + tw]

        assert torch.allclose(crop_out, manual_slice, atol=1e-6)

    def test_probe_measure_parity_mock(self):
        from core.training.probes.probe_crop_decode_autograd import (
            MockVAE as ProbeMockVAE,
            measure_parity_and_cost,
        )

        vae = ProbeMockVAE(scale=8)
        vae.eval()

        # Run minimal sweep with 2 margins
        data = measure_parity_and_cost(
            vae=vae,
            device=torch.device("cpu"),
            dtype=torch.float32,
            resolution=128,
            crop_cells=8,
            margins=[0, 4],
            warmup=1,
            repeat=1,
            seed=42,
        )

        assert "results" in data
        assert len(data["results"]) == 2
        r0 = data["results"][0]
        r1 = data["results"][1]
        assert r0["margin_cells"] == 0
        assert r1["margin_cells"] == 4
        # Margin 4 should have higher gradient cosine similarity than Margin 0
        assert r1["grad_cosine"] > r0["grad_cosine"]

    def test_crop_decode_loss_parameter_defaults(self):
        from api.param_defaults import TRAINING_DEFAULTS

        assert "crop_decode_loss_enable" in TRAINING_DEFAULTS
        assert TRAINING_DEFAULTS["crop_decode_loss_enable"] is False
        assert TRAINING_DEFAULTS["crop_decode_loss_weight"] == 0.0
        assert TRAINING_DEFAULTS["crop_decode_loss_margin_cells"] == 16
        assert TRAINING_DEFAULTS["crop_decode_loss_out_cells"] == 32
        assert TRAINING_DEFAULTS["crop_decode_loss_metric"] == "lpips"
        assert TRAINING_DEFAULTS["crop_decode_loss_snr_range"] == ""

    def test_crop_decode_loss_metrics_registered(self):
        from core.training.metric_registry import EXTRA_METRIC_DEFS

        assert "crop_decode_loss" in EXTRA_METRIC_DEFS
        assert "crop_decode_grad_norm_ratio" in EXTRA_METRIC_DEFS
        assert EXTRA_METRIC_DEFS["crop_decode_loss"]["family"] == "loss"
        assert EXTRA_METRIC_DEFS["crop_decode_grad_norm_ratio"]["family"] == "bounded_diagnostic"


