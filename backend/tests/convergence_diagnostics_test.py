"""Tests for convergence diagnostics instrumentation (Phase 1).

Validates statistical measures for symptoms A and B, registration in EXTRA_METRIC_DEFS,
and parameter default wiring without requiring a real GPU.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

# Importing trainer/routes must not take the GPU the owner's run holds.
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from api.param_defaults import TRAINING_DEFAULTS
from core.training.diagnostics.convergence_stats import (
    cell_periodicity_power,
    channel_mean_std_gap,
    low_frequency_power_ratio,
    pixel_stats_gap,
    trajectory_gap,
)
from core.training.metric_registry import EXTRA_METRIC_DEFS


class TestConvergenceStats:
    def test_channel_mean_std_gap_identical(self):
        x = torch.randn(2, 4, 16, 16)
        mean_err, std_err = channel_mean_std_gap(x, x)
        assert mean_err == pytest.approx(0.0, abs=1e-6)
        assert std_err == pytest.approx(0.0, abs=1e-6)

    def test_channel_mean_std_gap_bias(self):
        x = torch.zeros(1, 4, 8, 8)
        y = torch.ones(1, 4, 8, 8) * 2.5
        mean_err, std_err = channel_mean_std_gap(x, y)
        assert mean_err == pytest.approx(2.5, abs=1e-5)
        assert std_err == pytest.approx(0.0, abs=1e-5)

    def test_low_frequency_power_ratio_smooth_vs_noise(self):
        # Smooth image (low frequency dominant)
        h, w = 32, 32
        y, x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        smooth = torch.sin(x * math.pi) * torch.cos(y * math.pi)
        smooth_ratio = low_frequency_power_ratio(smooth, radius_fraction=0.25)

        # High frequency noise (checkerboard)
        checker = ((torch.arange(h).unsqueeze(1) + torch.arange(w).unsqueeze(0)) % 2).float() * 2.0 - 1.0
        checker_ratio = low_frequency_power_ratio(checker, radius_fraction=0.25)

        assert 0.0 <= smooth_ratio <= 1.0
        assert 0.0 <= checker_ratio <= 1.0
        assert smooth_ratio > checker_ratio

    def test_pixel_stats_gap_identical(self):
        img = torch.rand(1, 3, 32, 32)
        stats = pixel_stats_gap(img, img)
        assert stats["lum_err"] == pytest.approx(0.0, abs=1e-6)
        assert stats["sat_err"] == pytest.approx(0.0, abs=1e-6)
        assert stats["var_ratio"] == pytest.approx(1.0, abs=1e-5)

    def test_pixel_stats_gap_luminance_shift(self):
        img1 = torch.zeros(1, 3, 16, 16)
        img2 = torch.ones(1, 3, 16, 16) * 0.5
        stats = pixel_stats_gap(img1, img2)
        assert stats["lum_err"] == pytest.approx(0.5, abs=1e-5)

    def test_cell_periodicity_power_grid_detection(self):
        # Random uniform noise: no 8px periodicity
        torch.manual_seed(42)
        noise = torch.randn(1, 64, 64)
        noise_power = cell_periodicity_power(noise, period=8)
        assert abs(noise_power) < 0.2

        # 8px grid artifact: periodic pattern every 8 pixels
        grid = torch.zeros(1, 64, 64)
        grid[:, ::8, :] += 1.0
        grid[:, :, ::8] += 1.0
        grid_power = cell_periodicity_power(grid, period=8)
        assert grid_power > 0.5

    def test_trajectory_gap(self):
        gap = trajectory_gap(0.1, 0.4)
        assert gap == pytest.approx(0.3, abs=1e-6)


class TestMetricRegistryIntegration:
    EXPECTED_METRICS = [
        "diag_latent_mean_err",
        "diag_latent_std_err",
        "diag_low_freq_power_ratio",
        "diag_pixel_luminance_err",
        "diag_cell_periodicity_power",
        "diag_trajectory_gap",
    ]

    def test_all_diag_metrics_registered(self):
        for name in self.EXPECTED_METRICS:
            assert name in EXTRA_METRIC_DEFS, f"{name} missing from EXTRA_METRIC_DEFS"
            entry = EXTRA_METRIC_DEFS[name]
            assert entry.get("sampling") == "periodic"
            assert entry.get("family") == "bounded_diagnostic"


class TestParameterDefaults:
    def test_diagnostics_defaults_exist(self):
        assert "convergence_diagnostics_enable" in TRAINING_DEFAULTS
        assert TRAINING_DEFAULTS["convergence_diagnostics_enable"] is False
        assert "convergence_diagnostics_interval" in TRAINING_DEFAULTS
        assert TRAINING_DEFAULTS["convergence_diagnostics_interval"] == 100


class TestBaseTrainerDiagnosticsHook:
    def test_run_convergence_diagnostics_executes_safely(self, tmp_path):
        from core.training.base_trainer import BaseTrainer

        # Construct minimal mock trainer
        trainer = MagicMock(spec=BaseTrainer)
        trainer.output_dir = tmp_path
        trainer.log_prefix = "[TestTrainer]"
        trainer._sample_prompts = [{"positive": "test prompt"}]
        trainer._diag_gt_img = None
        trainer._diag_gt_latent = None
        trainer._last_predicted_latent = torch.randn(1, 4, 16, 16)
        trainer.extra_metrics = {}

        logged_metrics = {}
        def fake_log_extra_metric(name, value):
            logged_metrics[name] = value
        trainer.log_extra_metric = fake_log_extra_metric

        rollout_img = Image.new("RGB", (64, 64), color=(128, 128, 128))

        # Call real unbound method on trainer instance
        BaseTrainer._run_convergence_diagnostics(
            trainer,
            current_step=100,
            rollout_sample=rollout_img,
            reference_image_path=None,
        )

        # Confirm images saved
        assert (tmp_path / "samples" / "step_000100_diag_rollout.png").exists()

        # Confirm metrics logged
        assert "diag_low_freq_power_ratio" in logged_metrics
        assert "diag_cell_periodicity_power" in logged_metrics
        assert "diag_trajectory_gap" in logged_metrics
        assert not math.isnan(logged_metrics["diag_low_freq_power_ratio"])
