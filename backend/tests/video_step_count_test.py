"""Video step counts: the per-arch scheduler floor is a request-time contract.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/video_step_count_test.py -v

WHY THIS FILE EXISTS
--------------------
`num_inference_steps` does not mean the same thing on the two video
architectures, and the difference is only visible from inside their schedulers:

* LTX-2.3 loads diffusers' ``FlowMatchEulerDiscreteScheduler``: N steps build N
  timesteps (the terminal sigma is appended separately), so N drives N model
  evaluations and N=1 is a legal one-step request.
* MiniMax-H3's vendored scheduler builds a ``linspace(1, 0, N)`` SIGMA GRID with
  the terminal 0 included and sets ``timesteps = 1 - sigmas[:-1]``, so N grid
  points drive N-1 evaluations and N=1 drives none. Its ``set_timesteps``
  refuses N < 2.

A live API test found `num_inference_steps=1` on MiniMax-H3 answering HTTP 500
after 71.6 s -- the request had already paid for a full text encode before the
sampler raised. The floor therefore lives on the arch's ``TemporalSpec`` and is
enforced at the route by ``validate_video_steps`` before any weight is touched.

The tests below pin three things together: the declared floor, the real
scheduler's behaviour at that floor (so the spec cannot drift away from the code
it describes), and the 400 the route helper raises.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.error_handlers import ValidationError  # noqa: E402
from api.generation_utils import validate_video_steps  # noqa: E402
from core.models.components.wiring import (  # noqa: E402
    LTX2_TEMPORAL,
    MINIMAX_H3_TEMPORAL,
    temporal_spec_for_arch,
)


# --------------------------------------------------------------------------
# The declared spec
# --------------------------------------------------------------------------

def test_declared_floors():
    """The two archs differ, and the difference is what the payload serves."""
    assert MINIMAX_H3_TEMPORAL.min_inference_steps == 2
    assert MINIMAX_H3_TEMPORAL.steps_are_sigma_grid_points is True
    assert LTX2_TEMPORAL.min_inference_steps == 1
    assert LTX2_TEMPORAL.steps_are_sigma_grid_points is False


def test_capability_payload_carries_the_step_contract():
    """A client building a step-count control reads it from arch-capabilities."""
    from api.arch_capabilities import video_constraints_payload

    payload = video_constraints_payload()
    for arch in ("ltx2", "minimax_h3"):
        spec = temporal_spec_for_arch(arch)
        assert payload[arch]["min_inference_steps"] == spec.min_inference_steps
        assert payload[arch]["steps_are_sigma_grid_points"] is spec.steps_are_sigma_grid_points


# --------------------------------------------------------------------------
# The spec vs. the real schedulers
# --------------------------------------------------------------------------

def test_h3_scheduler_agrees_with_its_declared_floor():
    """N grid points -> N-1 evaluations, and N below the floor raises."""
    from core.models.minimax_h3.vendor.scheduling_minimax_h3 import MiniMaxH3Scheduler

    floor = MINIMAX_H3_TEMPORAL.min_inference_steps
    scheduler = MiniMaxH3Scheduler()
    scheduler.set_timesteps(floor)
    assert scheduler.timesteps.numel() == floor - 1

    scheduler_20 = MiniMaxH3Scheduler()
    scheduler_20.set_timesteps(20)
    # The documented client-visible fact: a "20-step" request runs 19 evaluations.
    assert scheduler_20.timesteps.numel() == 19

    with pytest.raises(ValueError):
        MiniMaxH3Scheduler().set_timesteps(floor - 1)


def test_ltx2_scheduler_agrees_with_its_declared_floor():
    """N steps -> N evaluations; a single-step request is legal."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    floor = LTX2_TEMPORAL.min_inference_steps
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(floor)
    assert scheduler.timesteps.numel() == floor
    assert torch.isclose(scheduler.sigmas[-1], torch.zeros(())).item()

    scheduler_8 = FlowMatchEulerDiscreteScheduler()
    scheduler_8.set_timesteps(8)
    assert scheduler_8.timesteps.numel() == 8


# --------------------------------------------------------------------------
# The route-level guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("steps", [1, 0, -1])
def test_h3_below_floor_is_a_validation_error(steps):
    with pytest.raises(ValidationError) as excinfo:
        validate_video_steps({"num_inference_steps": steps}, "minimax_h3")
    err = excinfo.value
    assert err.status_code == 400
    # The message has to be actionable: it must name the floor and explain the
    # grid-point semantics that produce it.
    assert "at least 2" in str(err.message)
    assert "grid points" in err.detail
    assert str(steps) in err.detail


@pytest.mark.parametrize("steps", [2, 3, 20])
def test_h3_at_or_above_floor_passes(steps):
    validate_video_steps({"num_inference_steps": steps}, "minimax_h3")


@pytest.mark.parametrize("steps", [1, 8])
def test_ltx2_single_step_is_untouched(steps):
    """LTX-2.3's shipped behaviour does not change: 1 step stays legal."""
    validate_video_steps({"num_inference_steps": steps}, "ltx2")


@pytest.mark.parametrize("arch", [None, "", "sdxl", "flux2", "not_an_arch"])
def test_non_video_archs_are_left_alone(arch):
    validate_video_steps({"num_inference_steps": 1}, arch)


def test_custom_steps_key():
    """The key is a parameter, so a route using another name can reuse this."""
    with pytest.raises(ValidationError):
        validate_video_steps({"steps": 1}, "minimax_h3", steps_key="steps")
    validate_video_steps({"steps": 4}, "minimax_h3", steps_key="steps")


def test_missing_key_is_not_an_error():
    """A request dict without the key cannot be judged, and must not 400."""
    validate_video_steps({}, "minimax_h3")
