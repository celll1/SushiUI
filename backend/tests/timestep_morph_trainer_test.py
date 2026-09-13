"""Trainer-side wiring of the resume-time timestep morph.

Exercises the BaseTrainer methods directly on a stub carrying only the
attributes they read, so no model, optimizer or CUDA context is involved:

  - the successful-optimizer-update counter and what it does to lambda,
  - resolution of the morph source (checkpoint record wins over config),
  - retarget mid-transition, completion, and the disabled-mid-flight refusal,
  - MNT windows partitioned where they cross an optimizer boundary.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/timestep_morph_trainer_test.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.timestep_sampler import (  # noqa: E402
    MorphingTimestepSampler,
    QuantileTableSampler,
    TimestepSampler,
    sampler_expr,
    sampler_to_config,
)

UNIFORM = {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0}
LOGIT = {"distribution": "logit_normal", "mean": 1.0, "std": 1.0}

_METHODS = (
    "_advance_optimizer_update",
    "_arm_timestep_morph",
    "_log_timestep_morph",
    "_mnt_update_offsets",
    "_stratified_mnt_timesteps",
    "_stratified_mnt_timesteps_morphing",
    "_timestep_morph_state",
    "timestep_morph_status",
)


class _StubTrainer:
    """Just enough trainer for the morph methods; no model, optimizer or GPU."""

    log_prefix = "[test]"
    device = torch.device("cpu")
    use_fused_backward = False
    fused_optimizer_groups = None

    def __init__(self, **config):
        self.config = {"stratified_timesteps": True, **config}
        self.timestep_sampler = None
        self._timestep_morph = None
        self._optimizer_update_step = 0
        self._resume_optimizer_update_step = 0
        self._resume_timestep_morph = None
        self.arch = None


for _name in _METHODS:
    setattr(_StubTrainer, _name, BaseTrainer.__dict__[_name])


def _target(config=LOGIT):
    return TimestepSampler.from_config(dict(config))


def _morph_config(target=LOGIT, **morph):
    block = {"enabled": True, "steps": 100, "curve": "linear",
             "interpolation": "quantile", "from": None}
    block.update(morph)
    return {**target, "morph": block}


class UpdateCounterTest(unittest.TestCase):
    def test_counter_drives_lambda(self):
        trainer = _StubTrainer()
        sampler = MorphingTimestepSampler(_target(UNIFORM), _target(), steps=10,
                                          curve="linear")
        for _ in range(5):
            trainer._advance_optimizer_update(sampler)
        self.assertEqual(trainer._optimizer_update_step, 5)
        self.assertAlmostEqual(sampler.lam, 0.5, places=6)

    def test_offsets_follow_the_accumulation_boundary(self):
        trainer = _StubTrainer()
        self.assertEqual(trainer._mnt_update_offsets(0, 8, 4), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(trainer._mnt_update_offsets(2, 8, 4), [0, 0, 1, 1, 1, 1, 2, 2])
        self.assertEqual(trainer._mnt_update_offsets(0, 4, 1), [0, 1, 2, 3])

    def test_fused_path_counts_every_backward(self):
        trainer = _StubTrainer()
        trainer.use_fused_backward = True
        self.assertEqual(trainer._mnt_update_offsets(0, 4, 8), [0, 1, 2, 3])


class ArmMorphTest(unittest.TestCase):
    def test_fresh_run_with_no_source_trains_at_the_target(self):
        trainer = _StubTrainer()
        target = _target()
        armed = trainer._arm_timestep_morph(target, _morph_config(), "t0")
        self.assertIs(armed, target)
        self.assertIsNone(trainer._timestep_morph)

    def test_configured_source_starts_a_morph_at_the_current_update(self):
        trainer = _StubTrainer()
        trainer._resume_optimizer_update_step = 4000
        armed = trainer._arm_timestep_morph(
            _target(), _morph_config(**{"from": UNIFORM}), "t0")
        self.assertIsInstance(armed, MorphingTimestepSampler)
        self.assertEqual(armed.start_update, 4000)
        self.assertEqual(armed.lam, 0.0)
        self.assertEqual(sampler_to_config(armed.source), sampler_to_config(_target(UNIFORM)))

    def test_unchanged_distribution_is_not_morphed(self):
        trainer = _StubTrainer()
        config = _morph_config(target=UNIFORM,
                               **{"from": {"distribution": "uniform"}})
        armed = trainer._arm_timestep_morph(_target(UNIFORM), config, "t0")
        self.assertNotIsInstance(armed, MorphingTimestepSampler)

    def test_checkpoint_record_continues_the_same_transition(self):
        trainer = _StubTrainer()
        target = _target()
        in_flight = MorphingTimestepSampler(_target(UNIFORM), target, steps=100,
                                            curve="linear", start_update=1000)
        in_flight.set_optimizer_update_step(1040)
        trainer._resume_timestep_morph = in_flight.state()
        trainer._resume_optimizer_update_step = 1040
        armed = trainer._arm_timestep_morph(target, _morph_config(steps=999), "t0")
        # The record's own steps/start win: the transition resumes at lambda=0.4,
        # it does not restart with the freshly configured length.
        self.assertEqual(armed.start_update, 1000)
        self.assertEqual(armed.steps, 100)
        self.assertAlmostEqual(armed.lam, 0.4, places=6)

    def test_finished_record_drops_back_to_the_plain_target(self):
        trainer = _StubTrainer()
        target = _target()
        in_flight = MorphingTimestepSampler(_target(UNIFORM), target, steps=100,
                                            start_update=1000)
        trainer._resume_timestep_morph = in_flight.state()
        trainer._resume_optimizer_update_step = 1200
        armed = trainer._arm_timestep_morph(target, _morph_config(), "t0")
        self.assertIs(armed, target)
        self.assertIsNone(trainer._timestep_morph_state())

    def test_retarget_mid_transition_starts_from_the_law_in_force(self):
        trainer = _StubTrainer()
        in_flight = MorphingTimestepSampler(_target(UNIFORM), _target(), steps=100,
                                            curve="linear", start_update=0)
        in_flight.set_optimizer_update_step(50)
        trainer._resume_timestep_morph = in_flight.state()
        trainer._resume_optimizer_update_step = 50
        new_target = _target({"distribution": "normal", "mean": 0.3, "std": 0.1})
        armed = trainer._arm_timestep_morph(
            new_target, _morph_config(target={"distribution": "normal", "mean": 0.3,
                                              "std": 0.1}, steps=200), "t0")
        self.assertEqual(armed.start_update, 50)
        self.assertEqual(armed.steps, 200)
        # The new source is the halfway law, matching the old morph's draws now.
        u = torch.linspace(0.05, 0.95, 32)
        self.assertTrue(torch.allclose(armed.source.icdf(u), in_flight.icdf(u), atol=1e-6))
        # ... and it stays pinned there while the new transition runs.
        armed.set_optimizer_update_step(150)
        self.assertTrue(torch.allclose(armed.source.icdf(u), in_flight.icdf(u), atol=1e-6))

    def test_disabling_morph_mid_flight_switches_immediately(self):
        trainer = _StubTrainer()
        in_flight = MorphingTimestepSampler(_target(UNIFORM), _target(), steps=100)
        in_flight.set_optimizer_update_step(10)
        trainer._resume_timestep_morph = in_flight.state()
        trainer._resume_optimizer_update_step = 10
        new_target = _target({"distribution": "normal"})
        armed = trainer._arm_timestep_morph(
            new_target, {"distribution": "normal", "morph": {"enabled": False}}, "t0")
        self.assertIs(armed, new_target)

    def test_disabling_morph_with_an_unchanged_target_also_switches(self):
        # Otherwise "disabled" would silently keep an in-flight transition
        # running, which is the one thing the checkbox is for.
        trainer = _StubTrainer()
        target = _target()
        in_flight = MorphingTimestepSampler(_target(UNIFORM), target, steps=100)
        in_flight.set_optimizer_update_step(10)
        trainer._resume_timestep_morph = in_flight.state()
        trainer._resume_optimizer_update_step = 10
        armed = trainer._arm_timestep_morph(
            target, {**LOGIT, "morph": {"enabled": False}}, "t0")
        self.assertIs(armed, target)

    def test_unreadable_record_falls_back_to_the_configured_source(self):
        trainer = _StubTrainer()
        trainer._resume_timestep_morph = {"version": 99, "from": {}, "to": {},
                                          "steps": 1, "curve": "linear",
                                          "interpolation": "quantile",
                                          "start_update": 0}
        armed = trainer._arm_timestep_morph(
            _target(), _morph_config(**{"from": UNIFORM}), "t0")
        self.assertIsInstance(armed, MorphingTimestepSampler)

    def test_over_deep_nesting_is_flattened(self):
        trainer = _StubTrainer()
        sampler = _target(UNIFORM)
        for _ in range(4):
            sampler = MorphingTimestepSampler(sampler, _target(), steps=10)
        sampler.set_optimizer_update_step(5)
        trainer._resume_timestep_morph = sampler.state()
        trainer._resume_optimizer_update_step = 5
        new_target = _target({"distribution": "normal"})
        armed = trainer._arm_timestep_morph(
            new_target, _morph_config(target={"distribution": "normal"}), "t0")
        self.assertIsInstance(armed.source, QuantileTableSampler)

    def test_status_and_state_describe_the_live_morph(self):
        trainer = _StubTrainer()
        trainer._resume_optimizer_update_step = 100
        armed = trainer._arm_timestep_morph(
            _target(), _morph_config(**{"from": UNIFORM}), "t0")
        trainer.timestep_sampler = armed
        for _ in range(50):
            trainer._advance_optimizer_update(armed)
        status = trainer.timestep_morph_status()
        self.assertTrue(status["active"])
        self.assertAlmostEqual(status["lam"], 0.5, places=6)
        self.assertEqual(status["optimizer_update_step"], 150)
        self.assertEqual(status["start_update"], 100)
        state = trainer._timestep_morph_state()
        self.assertEqual(state["to"], sampler_expr(_target()))
        self.assertEqual(state["start_update"], 100)


class PartitionedStratificationTest(unittest.TestCase):
    def test_window_is_cut_at_the_optimizer_boundary(self):
        trainer = _StubTrainer()
        morphing = MorphingTimestepSampler(_target(UNIFORM), _target(), steps=8,
                                           curve="linear")
        trainer._timestep_morph = morphing
        trainer.timestep_sampler = morphing
        block = trainer._stratified_mnt_timesteps(morphing, 8, 1, global_step=0,
                                                  gradient_accumulation_steps=4)
        self.assertEqual(tuple(block.shape), (8, 1))
        # Position is left where training expects it, not at a predicted one.
        self.assertEqual(morphing.update_step, 0)

    def test_mixture_morph_falls_back_to_independent_draws(self):
        trainer = _StubTrainer()
        morphing = MorphingTimestepSampler(_target(UNIFORM), _target(), steps=8,
                                           interpolation="mixture")
        trainer._timestep_morph = morphing
        trainer.timestep_sampler = morphing
        self.assertIsNone(trainer._stratified_mnt_timesteps(morphing, 4, 1))

    def test_no_morph_keeps_the_plain_stratified_path(self):
        trainer = _StubTrainer()
        sampler = _target(UNIFORM)
        block = trainer._stratified_mnt_timesteps(sampler, 4, 2)
        self.assertEqual(tuple(block.shape), (4, 2))


class CheckpointStateRoundTripTest(unittest.TestCase):
    """save -> load -> arm, through a real state.json on disk."""

    class _StateStub(_StubTrainer):
        run_name = "run"
        lr_scheduler = None
        lr_schedulers: list = []
        optimizer = None

        def __init__(self, output_dir, **config):
            super().__init__(**config)
            self.output_dir = Path(output_dir)

    _StateStub.save_training_state = BaseTrainer.__dict__["save_training_state"]
    _StateStub.load_training_state = BaseTrainer.__dict__["load_training_state"]

    def test_an_in_flight_morph_survives_a_crash_resume(self):
        import tempfile

        target = _target()
        with tempfile.TemporaryDirectory() as tmp:
            saver = self._StateStub(tmp)
            morphing = MorphingTimestepSampler(_target(UNIFORM), target, steps=100,
                                               curve="linear", start_update=1000)
            morphing.set_optimizer_update_step(1040)
            saver._timestep_morph = morphing
            saver._optimizer_update_step = 1040
            saver.save_training_state(step=40, epoch=0, batch_idx=3)

            resumed = self._StateStub(tmp)
            state = resumed.load_training_state(40)
            self.assertEqual(state["optimizer_update_step"], 1040)

            # No API update between the crash and the resume: morph.from is
            # still null, and the record alone has to carry the transition.
            armed = resumed._arm_timestep_morph(target, _morph_config(steps=9999), "t0")
            self.assertIsInstance(armed, MorphingTimestepSampler)
            self.assertEqual(armed.start_update, 1000)
            self.assertEqual(armed.steps, 100)
            self.assertAlmostEqual(armed.lam, 0.4, places=6)

    def test_a_finished_morph_is_not_persisted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            saver = self._StateStub(tmp)
            morphing = MorphingTimestepSampler(_target(UNIFORM), _target(), steps=10,
                                               start_update=0)
            morphing.set_optimizer_update_step(10)
            saver._timestep_morph = morphing
            saver._optimizer_update_step = 10
            saver.save_training_state(step=10, epoch=0, batch_idx=1)
            resumed = self._StateStub(tmp)
            self.assertIsNone(resumed.load_training_state(10)["timestep_morph"])

    def test_a_pre_morph_checkpoint_resumes_as_no_morph(self):
        import json
        import random
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_step_000005_state.json"
            version, rng_state, gauss_next = random.getstate()
            path.write_text(json.dumps({
                "global_step": 5, "epoch": 0, "batch_idx": 1,
                "random_state": {"version": version, "state": list(rng_state),
                                 "gauss_next": gauss_next},
            }))
            resumed = self._StateStub(tmp)
            resumed.load_training_state(5)
            self.assertEqual(resumed._resume_optimizer_update_step, 0)
            target = _target()
            self.assertIs(resumed._arm_timestep_morph(target, _morph_config(), "t0"),
                          target)


if __name__ == "__main__":
    unittest.main()
