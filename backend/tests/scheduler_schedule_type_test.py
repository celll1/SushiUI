"""``get_scheduler`` resolves a schedule from the request alone, not history.

Three defects this pins, all caused by writing overrides INTO
``pipeline.scheduler.config`` (which is the source scheduler's live
``_internal_dict`` -- diffusers' ``FrozenDict`` does not reject writes) and then
calling ``from_config`` with no kwargs:

* ``extract_init_dict`` drops every key listed in ``_use_default_values``, and
  an SD1.5 single-file load lists exactly ``prediction_type`` and
  ``timestep_spacing`` there, so a v-pred SD1.5 checkpoint sampled as epsilon
  with ``linspace`` spacing from its first generation onward.
* With spacing dropped, ``uniform`` and ``exponential`` (which then meant
  trailing spacing) produced identical sigmas on SD1.5 (measured, 20 steps:
  linspace/trailing sigma0=14.6146, leading sigma0=11.0283).
* Keys only written for the schedule that needs them survived into the next
  request, because ``get_scheduler`` reads back the scheduler it last returned.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/scheduler_schedule_type_test.py -v
"""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO = Path(os.path.abspath(__file__)).parents[2]
_BACKEND = _REPO / "backend"
for _p in (str(_REPO), str(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusers import (  # noqa: E402
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config  # noqa: E402
from diffusers.schedulers.scheduling_utils import SchedulerMixin  # noqa: E402

from core.inference.schedulers import (  # noqa: E402
    SAMPLER_MAP,
    _accepted_config_keys,
    get_scheduler,
    unsupported_schedule_overrides,
)
from core.model_loader import ModelLoader  # noqa: E402

# scheduler/scheduler_config.json of the two repos the single-file loaders
# resolve their scheduler config from. Verbatim, so the omissions
# (SD1.5 has no prediction_type / timestep_spacing) are the ones production sees.
SD15_SCHEDULER_CONFIG = {
    "_class_name": "PNDMScheduler",
    "_diffusers_version": "0.6.0",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "trained_betas": None,
    "clip_sample": False,
}

SDXL_SCHEDULER_CONFIG = {
    "_class_name": "EulerDiscreteScheduler",
    "_diffusers_version": "0.19.0.dev0",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "use_karras_sigmas": False,
}


class Pipe:
    """The only thing ``get_scheduler`` reads off a pipeline."""

    def __init__(self, scheduler):
        self.scheduler = scheduler


def sd15_source():
    return Pipe(PNDMScheduler.from_config(dict(SD15_SCHEDULER_CONFIG)))


def sd15_v_pred_source():
    pipe = sd15_source()
    ModelLoader._configure_v_prediction_scheduler(pipe)
    return pipe


def sdxl_source():
    return Pipe(EulerDiscreteScheduler.from_config(dict(SDXL_SCHEDULER_CONFIG)))


def defaults_only_source():
    """Built with no arguments: every key lands in ``_use_default_values``."""
    return Pipe(EulerDiscreteScheduler())


ALL_SOURCES = {
    "sd15": sd15_source,
    "sd15_v_pred": sd15_v_pred_source,
    "sdxl": sdxl_source,
    "defaults_only": defaults_only_source,
}

SCHEDULE_TYPES = ("uniform", "karras", "exponential")


def schedule_of(scheduler):
    return {
        "prediction_type": scheduler.config.get("prediction_type"),
        "use_karras_sigmas": scheduler.config.get("use_karras_sigmas"),
        "use_exponential_sigmas": scheduler.config.get("use_exponential_sigmas"),
        "timestep_spacing": scheduler.config.get("timestep_spacing"),
    }


def sigmas_of(scheduler, steps=20):
    scheduler.set_timesteps(steps)
    return scheduler.sigmas.clone()


def step_once(scheduler, steps=20, seed=0):
    """One real denoise step, so an assertion is about the solver and not a
    config string the scheduler class never reads."""
    scheduler.set_timesteps(steps)
    sample = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(seed))
    sample = sample * scheduler.init_noise_sigma
    model_out = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(seed + 1))
    return scheduler.step(model_out, scheduler.timesteps[0], sample).prev_sample


@contextlib.contextmanager
def captured_warnings():
    """Stub ``api.generation_status`` so the emission path is exercised without
    importing the api package (whose ``__init__`` pulls in all of routes.py)."""
    saved = {k: sys.modules.get(k) for k in ("api", "api.generation_status")}
    recorded = []
    pkg = types.ModuleType("api")
    pkg.__path__ = []
    status = types.ModuleType("api.generation_status")
    status.add_warning = lambda message, code=None: recorded.append(
        {"code": code, "message": message})
    pkg.generation_status = status
    sys.modules["api"] = pkg
    sys.modules["api.generation_status"] = status
    try:
        yield recorded
    finally:
        for key, module in saved.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module


# ---------------------------------------------------------------------------
# 1. The source scheduler is read-only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source_name", sorted(ALL_SOURCES))
@pytest.mark.parametrize("schedule_type", SCHEDULE_TYPES)
@pytest.mark.parametrize("sampler", ["euler", "euler_a", "dpmpp_2m", "dpmpp_sde"])
def test_source_config_is_not_rewritten(source_name, schedule_type, sampler):
    pipe = ALL_SOURCES[source_name]()
    before = dict(pipe.scheduler.config)

    get_scheduler(pipeline=pipe, sampler=sampler, schedule_type=schedule_type)

    assert dict(pipe.scheduler.config) == before


# ---------------------------------------------------------------------------
# 2. v-prediction survives, first generation and every one after it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("schedule_type", SCHEDULE_TYPES)
def test_sd15_v_prediction_survives_first_generation(schedule_type):
    scheduler = get_scheduler(pipeline=sd15_v_pred_source(), sampler="euler",
                              schedule_type=schedule_type)

    # What pipeline.py reads to pick guidance_rescale=0.7.
    assert scheduler.config.get("prediction_type") == "v_prediction"
    assert scheduler.config["timestep_spacing"] == "trailing"


def test_sd15_v_prediction_survives_repeated_generations():
    pipe = sd15_v_pred_source()
    for _ in range(4):
        # pipeline.py assigns the result back, so the next call's source is
        # the previous call's output.
        pipe.scheduler = get_scheduler(pipeline=pipe, sampler="euler",
                                       schedule_type="uniform")
        assert pipe.scheduler.config.get("prediction_type") == "v_prediction"
        assert pipe.scheduler.config["timestep_spacing"] == "trailing"


def test_sd15_v_pred_solver_is_not_the_epsilon_solver():
    v_pred = get_scheduler(pipeline=sd15_v_pred_source(), sampler="euler",
                           schedule_type="uniform")
    epsilon = get_scheduler(pipeline=sd15_source(), sampler="euler",
                            schedule_type="uniform")
    assert not torch.allclose(step_once(v_pred), step_once(epsilon))


def test_sdxl_explicit_prediction_type_is_carried():
    scheduler = get_scheduler(pipeline=sdxl_source(), sampler="euler",
                              schedule_type="uniform")
    assert scheduler.config["prediction_type"] == "epsilon"



@pytest.mark.parametrize("source_name", sorted(ALL_SOURCES))
def test_consecutive_switches_leave_no_residue(source_name):
    pipe = ALL_SOURCES[source_name]()
    for schedule_type in ("uniform", "karras", "exponential", "uniform",
                          "karras", "uniform"):
        pipe.scheduler = get_scheduler(pipeline=pipe, sampler="euler",
                                       schedule_type=schedule_type)
        fresh = get_scheduler(pipeline=ALL_SOURCES[source_name](), sampler="euler",
                              schedule_type=schedule_type)

        assert schedule_of(pipe.scheduler) == schedule_of(fresh), schedule_type
        assert torch.allclose(sigmas_of(pipe.scheduler), sigmas_of(fresh)), schedule_type
        assert torch.allclose(step_once(pipe.scheduler), step_once(fresh)), schedule_type


def test_exponential_after_karras_is_not_karras():
    pipe = sd15_source()
    pipe.scheduler = get_scheduler(pipeline=pipe, sampler="euler",
                                   schedule_type="karras")
    pipe.scheduler = get_scheduler(pipeline=pipe, sampler="euler",
                                   schedule_type="exponential")

    assert pipe.scheduler.config["use_karras_sigmas"] is False
    plain = get_scheduler(pipeline=sd15_source(), sampler="euler",
                          schedule_type="exponential")
    assert torch.allclose(sigmas_of(pipe.scheduler), sigmas_of(plain))


def test_karras_after_exponential_is_not_exponential():
    pipe = sd15_source()
    pipe.scheduler = get_scheduler(pipeline=pipe, sampler="euler",
                                   schedule_type="exponential")
    pipe.scheduler = get_scheduler(pipeline=pipe, sampler="euler",
                                   schedule_type="karras")

    assert pipe.scheduler.config["use_exponential_sigmas"] is False
    plain = get_scheduler(pipeline=sd15_source(), sampler="euler",
                          schedule_type="karras")
    assert torch.allclose(sigmas_of(pipe.scheduler), sigmas_of(plain))


def test_karras_spacing_does_not_depend_on_the_previous_schedule():
    from_exponential = sd15_source()
    from_exponential.scheduler = get_scheduler(
        pipeline=from_exponential, sampler="euler", schedule_type="exponential")
    from_exponential.scheduler = get_scheduler(
        pipeline=from_exponential, sampler="euler", schedule_type="karras")

    from_uniform = sd15_source()
    from_uniform.scheduler = get_scheduler(
        pipeline=from_uniform, sampler="euler", schedule_type="uniform")
    from_uniform.scheduler = get_scheduler(
        pipeline=from_uniform, sampler="euler", schedule_type="karras")

    assert (from_exponential.scheduler.config["timestep_spacing"]
            == from_uniform.scheduler.config["timestep_spacing"] == "leading")
    assert torch.allclose(sigmas_of(from_exponential.scheduler),
                          sigmas_of(from_uniform.scheduler))



def test_sd15_uniform_and_exponential_differ():
    uniform = get_scheduler(pipeline=sd15_source(), sampler="euler",
                            schedule_type="uniform")
    exponential = get_scheduler(pipeline=sd15_source(), sampler="euler",
                                schedule_type="exponential")

    # exponential chooses sigmas, so it leaves spacing where uniform puts it.
    assert uniform.config["timestep_spacing"] == "leading"
    assert exponential.config["timestep_spacing"] == "leading"
    assert exponential.config["use_exponential_sigmas"] is True
    assert not torch.allclose(sigmas_of(uniform), sigmas_of(exponential))
    assert not torch.allclose(step_once(uniform), step_once(exponential))
    # Measured on the SD1.5 betas at 20 steps: same endpoint, different curve.
    assert float(sigmas_of(uniform)[0]) == pytest.approx(11.0283, abs=1e-3)
    assert float(sigmas_of(exponential)[0]) == pytest.approx(11.0283, abs=1e-3)
    assert float(sigmas_of(uniform)[5]) == pytest.approx(3.3478, abs=1e-3)
    assert float(sigmas_of(exponential)[5]) == pytest.approx(2.5350, abs=1e-3)


def test_exponential_is_the_exponential_sigma_schedule():
    exponential = get_scheduler(pipeline=sdxl_source(), sampler="euler",
                                schedule_type="exponential")
    reference = EulerDiscreteScheduler.from_config(
        dict(SDXL_SCHEDULER_CONFIG), use_exponential_sigmas=True,
        use_karras_sigmas=False, timestep_spacing="leading",
        prediction_type="epsilon")

    assert torch.allclose(sigmas_of(exponential), sigmas_of(reference))
    assert torch.allclose(step_once(exponential), step_once(reference))

    karras = get_scheduler(pipeline=sdxl_source(), sampler="euler",
                           schedule_type="karras")
    assert not torch.allclose(sigmas_of(exponential), sigmas_of(karras))


@pytest.mark.parametrize("schedule_type", ["uniform", "karras"])
def test_exponential_key_does_not_disturb_the_other_schedules(schedule_type):
    scheduler = get_scheduler(pipeline=sdxl_source(), sampler="euler",
                              schedule_type=schedule_type)
    reference = EulerDiscreteScheduler.from_config(
        dict(SDXL_SCHEDULER_CONFIG), use_karras_sigmas=(schedule_type == "karras"),
        timestep_spacing="leading", prediction_type="epsilon")

    assert scheduler.config["use_exponential_sigmas"] is False
    assert torch.allclose(sigmas_of(scheduler), sigmas_of(reference))
    assert torch.allclose(step_once(scheduler), step_once(reference))


def test_karras_changes_the_sigma_schedule():
    uniform = get_scheduler(pipeline=sdxl_source(), sampler="euler",
                            schedule_type="uniform")
    karras = get_scheduler(pipeline=sdxl_source(), sampler="euler",
                           schedule_type="karras")

    assert karras.config["use_karras_sigmas"] is True
    assert not torch.allclose(sigmas_of(uniform), sigmas_of(karras))
    assert not torch.allclose(step_once(uniform), step_once(karras))

    reference = EulerDiscreteScheduler.from_config(
        dict(SDXL_SCHEDULER_CONFIG), use_karras_sigmas=True,
        timestep_spacing="leading", prediction_type="epsilon")
    assert torch.allclose(step_once(karras), step_once(reference))


# ---------------------------------------------------------------------------
# 5. Combinations the sampler cannot honour are reported, not silently dropped
# ---------------------------------------------------------------------------

def test_karras_on_a_sampler_without_karras_sigmas_warns():
    for sampler in ("euler_a", "ddim", "ddpm", "pndm"):
        warnings = unsupported_schedule_overrides(sampler, "karras")
        assert [w["code"] for w in warnings] == ["unsupported_param"], sampler
        assert "use_karras_sigmas" in warnings[0]["message"]

    for sampler in ("euler", "dpmpp_2m", "unipc", "heun", "lms"):
        assert unsupported_schedule_overrides(sampler, "karras") == [], sampler


def test_no_warning_when_the_dropped_override_is_the_neutral_value():
    # euler_a takes no use_karras_sigmas, but uniform asks for False anyway.
    assert unsupported_schedule_overrides("euler_a", "uniform") == []


def test_dpmpp_sde_ignores_timestep_spacing_and_says_so():
    # DPMSolverSinglestepScheduler has no timestep_spacing argument at all, so
    # only a v-prediction model -- the one thing that asks for spacing other
    # than the default -- has anything to be told about.
    for schedule_type in SCHEDULE_TYPES:
        assert unsupported_schedule_overrides("dpmpp_sde", schedule_type) == [], \
            schedule_type
    v_pred = unsupported_schedule_overrides("dpmpp_sde", "uniform", "v_prediction")
    assert [w["code"] for w in v_pred] == ["unsupported_param"]
    assert "trailing" in v_pred[0]["message"]


def test_exponential_on_a_sampler_without_exponential_sigmas_warns():
    for sampler in ("euler_a", "ddim", "ddpm", "pndm"):
        warnings = unsupported_schedule_overrides(sampler, "exponential")
        assert [w["code"] for w in warnings] == ["unsupported_param"], sampler
        assert "use_exponential_sigmas" in warnings[0]["message"]

    for sampler in ("euler", "dpmpp_2m", "dpmpp_sde", "unipc", "heun", "lms",
                    "dpm2", "dpm2_a"):
        assert unsupported_schedule_overrides(sampler, "exponential") == [], sampler


def test_warnings_reach_the_generation_status_store():
    with captured_warnings() as recorded:
        get_scheduler(pipeline=sdxl_source(), sampler="euler_a",
                      schedule_type="karras")
    assert [w["code"] for w in recorded] == ["unsupported_param"]

    with captured_warnings() as recorded:
        get_scheduler(pipeline=sdxl_source(), sampler="euler",
                      schedule_type="karras")
    assert recorded == []


def test_unsupported_key_is_not_claimed_in_the_built_config():
    """An ignored setting must not read back as if it applied: diffusers copies
    unconsumed source-config keys onto the new scheduler's config."""
    karras_source = Pipe(get_scheduler(pipeline=sdxl_source(), sampler="euler",
                                       schedule_type="karras"))
    assert karras_source.scheduler.config["use_karras_sigmas"] is True

    ancestral = get_scheduler(pipeline=karras_source, sampler="euler_a",
                              schedule_type="karras")
    assert isinstance(ancestral, EulerAncestralDiscreteScheduler)
    assert "use_karras_sigmas" not in ancestral.config


def test_unknown_sampler_still_raises():
    with pytest.raises(ValueError):
        get_scheduler(pipeline=sdxl_source(), sampler="nope",
                      schedule_type="uniform")



class StandInScheduler(SchedulerMixin, ConfigMixin):
    """Takes ``use_karras_sigmas`` but lists it in ``ignore_for_config``.

    That is the one case where a key is in the ``__init__`` signature and still
    never reaches it: ``extract_init_dict`` subtracts ``ignore_for_config`` from
    its expected keys, so the kwarg lands in ``unused_kwargs``. No class in
    ``SAMPLER_MAP`` uses it today (pinned below), so only a stand-in can show it.
    """

    ignore_for_config = ["use_karras_sigmas"]

    @register_to_config
    def __init__(self, num_train_timesteps=1000, prediction_type="epsilon",
                 timestep_spacing="leading", use_karras_sigmas=False):
        self.received_karras_sigmas = use_karras_sigmas


def test_no_current_sampler_ignores_any_config_key():
    """The measurement the stand-in above stands in for."""
    for sampler, scheduler_class in SAMPLER_MAP.items():
        assert list(scheduler_class.ignore_for_config) == [], sampler


def test_ignored_key_is_not_counted_as_accepted():
    assert "use_karras_sigmas" in inspect.signature(
        StandInScheduler.__init__).parameters
    assert "use_karras_sigmas" not in _accepted_config_keys(StandInScheduler)
    assert {"prediction_type", "timestep_spacing"} <= _accepted_config_keys(
        StandInScheduler)


def test_ignored_override_is_reported_not_silently_dropped(monkeypatch):
    monkeypatch.setitem(SAMPLER_MAP, "stand_in", StandInScheduler)

    with captured_warnings() as recorded:
        scheduler = get_scheduler(pipeline=sdxl_source(), sampler="stand_in",
                                  schedule_type="karras")

    # diffusers dropped the kwarg, so karras was not applied ...
    assert scheduler.received_karras_sigmas is False
    # ... and nothing may read back as if it had been.
    assert "use_karras_sigmas" not in scheduler.config
    assert [w["code"] for w in recorded] == ["unsupported_param"]
    assert "use_karras_sigmas" in recorded[0]["message"]
