from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference.custom_sampling import calculate_dynamic_cfg, cfg_schedule_peak


@pytest.mark.parametrize("schedule", ("linear", "quadratic", "cosine", "exponential"))
def test_dynamic_cfg_rises_from_noise_minimum_to_clean_maximum(schedule):
    values = [
        calculate_dynamic_cfg(
            sigma=10.0 * (1.0 - progress),
            sigma_max=10.0,
            cfg_base=7.0,
            cfg_schedule_type=schedule,
            cfg_schedule_min=1.0,
            cfg_schedule_max=7.0,
            cfg_schedule_power=2.0,
            denoise_progress=progress,
        )
        for progress in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(7.0)
    assert values == sorted(values)


def test_sigma_fallback_has_the_same_noise_to_clean_direction():
    noisy = calculate_dynamic_cfg(
        sigma=10.0, sigma_max=10.0, cfg_base=7.0,
        cfg_schedule_type="linear", cfg_schedule_min=1.0,
    )
    clean = calculate_dynamic_cfg(
        sigma=0.0, sigma_max=10.0, cfg_base=7.0,
        cfg_schedule_type="linear", cfg_schedule_min=1.0,
    )
    assert noisy == pytest.approx(1.0)
    assert clean == pytest.approx(7.0)


def test_explicit_progress_does_not_require_scheduler_sigmas():
    midpoint = calculate_dynamic_cfg(
        sigma=0.0, sigma_max=0.0, cfg_base=7.0,
        cfg_schedule_type="linear", cfg_schedule_min=1.0,
        denoise_progress=0.5,
    )
    assert midpoint == pytest.approx(4.0)


def test_snr_adaptive_starts_at_noise_minimum_before_first_observation():
    assert calculate_dynamic_cfg(
        sigma=10.0, sigma_max=10.0, cfg_base=7.0,
        cfg_schedule_type="snr_based", cfg_schedule_min=1.0,
        cfg_rescale_snr_alpha=0.25, snr=None, denoise_progress=0.0,
    ) == pytest.approx(1.0)


def test_constant_schedule_remains_exactly_the_requested_cfg():
    assert calculate_dynamic_cfg(
        sigma=10.0, sigma_max=10.0, cfg_base=7.0,
        cfg_schedule_type="constant", cfg_schedule_min=1.0,
        denoise_progress=0.0,
    ) == pytest.approx(7.0)


def test_schedule_peak_keeps_the_unconditional_branch_when_base_is_one():
    assert cfg_schedule_peak(1.0, "linear", 1.0, 7.0) == pytest.approx(7.0)
    assert cfg_schedule_peak(1.0, "constant", 7.0, 9.0) == pytest.approx(1.0)


def test_cfg_schedule_capability_matches_classic_image_cfg_architectures():
    from api.arch_capabilities import ARCH_UNSUPPORTED

    supported = {
        "sd15", "sdxl", "zimage", "flux2", "ideogram4", "lens",
        "minit2i", "anima", "krea2", "sensenova",
        "sensenova_sdxl_chimera",
    }
    unsupported = {"ltx2", "acestep", "minimax_h3", "minimax_music3", "yue2"}
    assert all("cfg_schedule" not in ARCH_UNSUPPORTED.get(arch, {}) for arch in supported)
    assert all("cfg_schedule" in ARCH_UNSUPPORTED.get(arch, {}) for arch in unsupported)
