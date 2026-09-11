"""Training sample generation solves with the target the loss trains.

``from_single_file`` reports ``prediction_type="epsilon"`` for every SD/SDXL
checkpoint that is not SD2 (diffusers ``single_file_utils``), so a v-pred run
loaded from a ``.safetensors`` file used to compute a velocity loss while its
in-training samples were denoised by an epsilon solver.
``temp_pipeline.sampling_scheduler_source`` makes ``trainer.prediction_target``
decide, with the scheduler's own config as the fallback.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_sample_v_prediction_scheduler_test.py -v
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(os.path.abspath(__file__)).parents[2]
_BACKEND = _REPO / "backend"
for _p in (str(_REPO), str(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diffusers import EulerDiscreteScheduler, PNDMScheduler  # noqa: E402

from core.inference.schedulers import get_scheduler  # noqa: E402
from core.training.temp_pipeline import sampling_scheduler_source  # noqa: E402

# scheduler/scheduler_config.json of the repo an SD1.5 single-file load resolves
# its scheduler from. Verbatim: the two keys it OMITS are the whole point --
# they land in ``_use_default_values``, which ``from_config`` discards.
# ``single_file_utils.SCHEDULER_DEFAULT_CONFIG`` (the legacy ldm path) sets both
# explicitly and so cannot reproduce this.
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


def single_file_scheduler():
    """What ``StableDiffusion(XL)Pipeline.from_single_file`` hands the trainer."""
    scheduler = PNDMScheduler.from_config(dict(SD15_SCHEDULER_CONFIG))
    assert sorted(scheduler.config["_use_default_values"]) == [
        "prediction_type", "timestep_spacing"]
    return scheduler


def fake_trainer(**overrides):
    base = dict(
        original_scheduler=single_file_scheduler(),
        prediction_target="epsilon",
        noise_process="ddpm",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def step_once(scheduler, seed=0):
    """One denoise step, so the assertion is about the solver, not the config."""
    scheduler.set_timesteps(4)
    sample = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(seed))
    sample = sample * scheduler.init_noise_sigma
    model_out = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(seed + 1))
    return scheduler.step(model_out, scheduler.timesteps[0], sample).prev_sample



def test_v_pred_single_file_source_is_realigned():
    trainer = fake_trainer(prediction_target="velocity")
    assert trainer.original_scheduler.config["prediction_type"] == "epsilon"

    source = sampling_scheduler_source(trainer)

    assert source.scheduler is trainer.original_scheduler
    assert source.scheduler.config["prediction_type"] == "v_prediction"
    assert source.scheduler.config["timestep_spacing"] == "trailing"


def test_v_pred_sample_scheduler_solves_velocity():
    trainer = fake_trainer(prediction_target="velocity")
    scheduler = get_scheduler(
        pipeline=sampling_scheduler_source(trainer), sampler="euler",
        schedule_type="uniform",
    )

    # What ops/sd_sdxl_ops.py reads to pick guidance_rescale.
    assert scheduler.config.get("prediction_type") == "v_prediction"
    assert isinstance(scheduler, EulerDiscreteScheduler)

    reference = EulerDiscreteScheduler.from_config(
        dict(SD15_SCHEDULER_CONFIG),
        prediction_type="v_prediction", timestep_spacing="trailing",
        use_karras_sigmas=False,
    )
    assert torch.allclose(step_once(scheduler), step_once(reference))

    epsilon_solver = get_scheduler(
        pipeline=sampling_scheduler_source(fake_trainer()), sampler="euler",
        schedule_type="uniform",
    )
    assert not torch.allclose(step_once(scheduler), step_once(epsilon_solver))


# ---------------------------------------------------------------------------
# Everything that is not a DDPM-family v-pred run keeps its scheduler
# ---------------------------------------------------------------------------

def test_epsilon_run_scheduler_untouched():
    trainer = fake_trainer()
    before = dict(trainer.original_scheduler.config)

    scheduler = get_scheduler(
        pipeline=sampling_scheduler_source(trainer), sampler="euler",
        schedule_type="uniform",
    )

    # get_scheduler builds from a copy; the trainer's own scheduler, which the
    # run keeps training with, must come back untouched.
    assert dict(trainer.original_scheduler.config) == before
    assert scheduler.config["prediction_type"] == "epsilon"
    assert scheduler.config["timestep_spacing"] == "leading"


def test_flow_arch_scheduler_untouched():
    """Flow archs share one scheduler object between training and sampling;
    stamping v_prediction/trailing on it would corrupt the training schedule."""
    trainer = fake_trainer(prediction_target="velocity", noise_process="flow")
    before = dict(trainer.original_scheduler.config)

    sampling_scheduler_source(trainer)

    assert dict(trainer.original_scheduler.config) == before


def test_diffusers_directory_v_pred_not_rewritten():
    # A directory load whose scheduler_config.json states both keys, so neither
    # is a default value.
    scheduler = EulerDiscreteScheduler.from_config(
        dict(SD15_SCHEDULER_CONFIG),
        prediction_type="v_prediction", timestep_spacing="trailing",
    )
    trainer = fake_trainer(original_scheduler=scheduler, prediction_target="velocity")
    before = dict(scheduler.config)

    source = sampling_scheduler_source(trainer)

    assert dict(source.scheduler.config) == before


def test_missing_prediction_target_falls_back_to_scheduler_config():
    trainer = SimpleNamespace(original_scheduler=single_file_scheduler())
    before = dict(trainer.original_scheduler.config)

    source = sampling_scheduler_source(trainer)

    assert dict(source.scheduler.config) == before


def test_absent_scheduler_is_carried_not_raised():
    source = sampling_scheduler_source(SimpleNamespace(prediction_target="velocity"))
    assert source.scheduler is None



def test_every_training_sample_site_uses_the_helper():
    sites = 0
    for path in sorted((_BACKEND / "core" / "training").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "get_scheduler"):
                continue
            pipeline_arg = next(
                (kw.value for kw in node.keywords if kw.arg == "pipeline"), None)
            assert pipeline_arg is not None, f"{path.name}: get_scheduler without pipeline="
            assert (getattr(pipeline_arg.func, "id", None)
                    == "sampling_scheduler_source"), (
                f"{path.name}:{node.lineno} builds a sampling scheduler without "
                f"aligning it to the run's prediction_target")
            sites += 1
    assert sites == 5, f"expected 5 training sample-generation sites, found {sites}"
