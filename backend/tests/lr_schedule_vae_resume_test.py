"""P7: the VAE trainer on the shared LR registry, through its own artifact.

Run from the repository root with the repo's virtualenv interpreter:

    venv/Scripts/python.exe -m pytest backend/tests/lr_schedule_vae_resume_test.py -q

Why this file exists
--------------------
``docs/guides/LR_SCHEDULER_DESIGN.md`` §17.4 makes resume and extension a
NECESSARY CONDITION of this phase: swapping the builder is the easy half, and
the half that can break silently is that a ``LambdaLR``'s ``state_dict()``
carries ``last_epoch`` and ``base_lrs`` and nothing its lambda closed over. The
timeline is closed over. So a resume that restored only the scheduler's own
state would land on the right POSITION of a curve rebuilt from this session's
``total_steps`` -- which is exactly the "an extension rewinds the plateau"
defect (§7.3), still happening with the warp installed.

Everything here drives the real ``VaeTrainer.build_optimizer`` /
``save_checkpoint`` / ``load_checkpoint`` against a real temp directory, with a
real ``torch.optim.AdamW`` over four-element parameters standing in for the
decoder. No VAE, no dataset, no GPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.lr_schedules import LR_SCHEDULER_NAMES
from core.training.vae.vae_config import (
    VaeConfigError, VALID_LR_SCHEDULERS, _UNSHAPED_LR_SCHEDULERS,
)
from core.training.vae.vae_trainer import (
    VaeTrainer, _LR_SCHEDULE_STATE_VERSION, _split_lr_scheduler_state,
)

_REPO = Path(_BACKEND).parent

_NAMES = ["decoder.a", "decoder.b"]


def _trainer(root: Path, *, total_steps: int, scheduler: str = "cosine",
             warmup: int = 0, lr: float = 1e-5, accum: int = 1) -> VaeTrainer:
    """A VaeTrainer with everything build/save/load touch, and nothing else."""
    trainer = VaeTrainer.__new__(VaeTrainer)
    trainer.log_prefix = "[VaeTrainer]"
    trainer.device = torch.device("cpu")
    trainer.output_dir = root
    trainer.checkpoints_dir = root / "checkpoints"
    trainer.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    trainer.run_name = "vae_p7"
    trainer.run_id = None          # keeps _record_checkpoint_row off the DB
    trainer.ema = None
    trainer._ema_updates = 0
    trainer._ema_retained_init = 1.0
    trainer.resume_seq = 0
    trainer.train_sampler = None
    trainer.global_step = 0
    trainer.stopped = False
    trainer.train_encoder = False
    trainer._base_vae_identity = None
    trainer.lr_timeline = None
    trainer.lr_schedule_spec = None
    trainer.trainable_names = list(_NAMES)
    trainer.trainable_params = [torch.nn.Parameter(torch.zeros(4))
                                for _ in _NAMES]
    trainer.cfg = {
        "total_steps": total_steps,
        "lr_scheduler": scheduler,
        "lr_warmup_steps": warmup,
        "learning_rate": lr,
        "optimizer": "adamw",
        "optimizer_weight_decay": 0.0,
        "gradient_accumulation_steps": accum,
        "train_decoder": True,
        "train_encoder": False,
        "decoder_blocks": "all",
        "encoder_blocks": "all",
        "max_step_saves_to_keep": 0,   # keeps _prune_checkpoints off
    }
    trainer.build_optimizer()
    return trainer


def _advance(trainer: VaeTrainer, steps: int) -> None:
    """The train loop's update branch: optimizer, scheduler, step counter."""
    for _ in range(steps):
        for param in trainer.trainable_params:
            param.grad = torch.ones_like(param)
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.global_step += 1


def _curve(trainer: VaeTrainer, steps) -> list:
    lr_lambda = trainer.lr_scheduler.lr_lambdas[0]
    return [lr_lambda(s) for s in steps]


class VaeScheduleBuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_every_offered_name_builds(self):
        """The UI's whole vocabulary, through the real build_optimizer."""
        from torch.optim.lr_scheduler import LambdaLR
        for name in VALID_LR_SCHEDULERS:
            with self.subTest(lr_scheduler=name):
                trainer = _trainer(self.root, total_steps=1000, scheduler=name,
                                   warmup=100)
                self.assertIsInstance(trainer.lr_scheduler, LambdaLR)
                # One lambda per param group: lr_utils' re-assertion tests the
                # length, and a bare callable would take its fallback path.
                self.assertEqual(len(trainer.lr_scheduler.lr_lambdas),
                                 len(trainer.optimizer.param_groups))
                values = _curve(trainer, range(0, 1001, 25))
                self.assertTrue(all(0.0 <= v <= 1.0 for v in values), values)
                self.assertEqual(values[0], 0.0)      # warmup starts at 0
                self.assertAlmostEqual(_curve(trainer, [100])[0], 1.0)

    def test_a_withheld_registry_name_is_not_offered(self):
        for name in _UNSHAPED_LR_SCHEDULERS:
            self.assertIn(name, LR_SCHEDULER_NAMES)
            self.assertNotIn(name, VALID_LR_SCHEDULERS)

    def test_the_step_axis_is_not_divided_by_accumulation(self):
        """This loop counts optimizer steps, so total/warmup are ALREADY
        scheduler advances -- dividing them here would shrink the curve to
        1/accum of the run (design §18.9)."""
        trainer = _trainer(self.root, total_steps=1000, scheduler="cosine",
                           warmup=100, accum=4)
        self.assertEqual(trainer.lr_schedule_spec.total_steps, 1000)
        self.assertEqual(trainer.lr_schedule_spec.warmup_steps, 100)
        self.assertAlmostEqual(_curve(trainer, [100])[0], 1.0)
        self.assertAlmostEqual(_curve(trainer, [1000])[0], 0.0)

    def test_constant_now_warms_up(self):
        """The refusal that came off: diffusers' constant never received the
        warmup, the registry's does."""
        trainer = _trainer(self.root, total_steps=1000, scheduler="constant",
                           warmup=100)
        self.assertAlmostEqual(_curve(trainer, [50])[0], 0.5)
        self.assertAlmostEqual(_curve(trainer, [100])[0], 1.0)
        self.assertAlmostEqual(_curve(trainer, [999])[0], 1.0)

    def test_an_unknown_name_falls_back_to_a_constant_lr_and_says_so(self):
        """build_optimizer's existing except-branch, which the conditional
        artifact policy (_CKPT_CONDITIONAL) is written around."""
        trainer = _trainer(self.root, total_steps=100, scheduler="nope")
        self.assertIsNone(trainer.lr_scheduler)
        self.assertIsNone(trainer.lr_timeline)


class VaeScheduleCheckpointTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _save_at(self, trainer: VaeTrainer, step: int) -> Path:
        _advance(trainer, step - trainer.global_step)
        return trainer.save_checkpoint(step)

    def test_the_payload_carries_version_events_and_position(self):
        trainer = _trainer(self.root, total_steps=1000,
                           scheduler="plateau_cosine_floor")
        ckpt = self._save_at(trainer, 900)
        payload = torch.load(ckpt / "lr_scheduler.pt", map_location="cpu",
                             weights_only=False)
        self.assertEqual(payload["lr_schedule_version"],
                         _LR_SCHEDULE_STATE_VERSION)
        self.assertEqual(payload["scheduler_step"], 900)
        self.assertEqual([e["kind"] for e in payload["events"]], ["total_steps"])
        self.assertEqual(payload["events"][0]["value"], 1000)
        self.assertEqual(payload["scheduler"]["last_epoch"], 900)

    def test_a_plain_resume_restores_position_and_curve(self):
        trainer = _trainer(self.root, total_steps=1000, scheduler="cosine",
                           warmup=100)
        ckpt = self._save_at(trainer, 400)
        before = _curve(trainer, range(0, 1001, 50))

        resumed = _trainer(self.root, total_steps=1000, scheduler="cosine",
                           warmup=100)
        resumed.load_checkpoint(ckpt)

        self.assertEqual(resumed.global_step, 400)
        self.assertEqual(resumed.lr_scheduler.last_epoch, 400)
        self.assertEqual(_curve(resumed, range(0, 1001, 50)), before)
        # The LR in force is the config's base times the multiplier HERE.
        expected = 1e-5 * _curve(resumed, [400])[0]
        self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], expected,
                               places=12)

    def test_an_extension_does_not_rewind_the_plateau(self):
        """§7.3, the defect the timeline exists for. Without the events in the
        file, D moves 850 -> 1700 and step 901 is back on the plateau."""
        trainer = _trainer(self.root, total_steps=1000,
                           scheduler="plateau_cosine_floor")
        ckpt = self._save_at(trainer, 900)
        before = _curve(trainer, range(0, 901))
        at_900 = before[900]
        self.assertLess(at_900, 1.0)          # decaying already

        extended = _trainer(self.root, total_steps=2000,
                            scheduler="plateau_cosine_floor")
        extended.load_checkpoint(ckpt)

        # Everything before the anchor is bit-identical...
        self.assertEqual(_curve(extended, range(0, 901)), before)
        # ...the curve is continuous at it, and still decaying rather than back
        # at the base rate.
        after = _curve(extended, [901, 1500, 2000])
        # Continuous at the seam and still falling: the decay is stretched over
        # the new remainder (11 real steps per nominal one), not restarted.
        self.assertLess(after[0], at_900)
        self.assertLess(at_900 - after[0], 0.01)
        self.assertLess(after[1], after[0])
        # The floor is reached at the NEW end, not 1000 steps early.
        self.assertAlmostEqual(after[2], 0.25, places=12)
        self.assertGreater(_curve(extended, [1999])[0], 0.25)

    def test_the_extension_anchor_survives_the_next_checkpoint(self):
        """Two sessions of events, so the warp composes rather than resetting."""
        trainer = _trainer(self.root, total_steps=1000,
                           scheduler="plateau_cosine_floor")
        first = self._save_at(trainer, 900)

        extended = _trainer(self.root, total_steps=2000,
                            scheduler="plateau_cosine_floor")
        extended.load_checkpoint(first)
        self.assertEqual([e["kind"] for e in extended.lr_timeline.events],
                         ["total_steps", "total_steps"])
        second = self._save_at(extended, 1200)
        mid = _curve(extended, range(0, 1201))

        third = _trainer(self.root, total_steps=2000,
                         scheduler="plateau_cosine_floor")
        third.load_checkpoint(second)
        self.assertEqual([(e["kind"], e["at"], e["value"])
                          for e in third.lr_timeline.events],
                         [("total_steps", 0, 1000), ("total_steps", 900, 2000)])
        self.assertEqual(_curve(third, range(0, 1201)), mid)

    def test_a_shortened_run_holds_the_floor_from_the_new_end(self):
        trainer = _trainer(self.root, total_steps=1000,
                           scheduler="plateau_cosine_floor")
        ckpt = self._save_at(trainer, 900)

        shrunk = _trainer(self.root, total_steps=950,
                          scheduler="plateau_cosine_floor")
        shrunk.load_checkpoint(ckpt)
        self.assertAlmostEqual(_curve(shrunk, [950])[0], 0.25, places=12)
        self.assertAlmostEqual(_curve(shrunk, [1200])[0], 0.25, places=12)

    def test_an_old_format_file_migrates(self):
        """Before P7 the file WAS the LambdaLR state_dict. It still resumes:
        the position is restored, and with no recorded timeline the current
        total becomes the nominal axis (§7.4)."""
        trainer = _trainer(self.root, total_steps=1000, scheduler="cosine",
                           warmup=100)
        ckpt = self._save_at(trainer, 400)
        # Rewrite the artifact in the pre-P7 format, and correct the manifest
        # size so the resume guard classifies it "ok" rather than damaged.
        legacy = ckpt / "lr_scheduler.pt"
        torch.save(trainer.lr_scheduler.state_dict(), legacy)
        state_path = ckpt / "train_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["artifacts"]["lr_scheduler.pt"] = legacy.stat().st_size
        state_path.write_text(json.dumps(state), encoding="utf-8")

        resumed = _trainer(self.root, total_steps=1000, scheduler="cosine",
                           warmup=100)
        resumed.load_checkpoint(ckpt)
        self.assertEqual(resumed.lr_scheduler.last_epoch, 400)
        self.assertEqual([e["kind"] for e in resumed.lr_timeline.events],
                         ["total_steps"])
        self.assertEqual(resumed.lr_timeline.events[0]["value"], 1000)
        self.assertEqual(_curve(resumed, range(0, 1001, 50)),
                         _curve(trainer, range(0, 1001, 50)))

    def test_an_old_format_file_takes_the_current_total_as_nominal(self):
        """The one resume §7.4 cannot protect: the total it was built with is
        not recorded anywhere, so an extension IS a rebuild -- once."""
        trainer = _trainer(self.root, total_steps=1000,
                           scheduler="plateau_cosine_floor")
        ckpt = self._save_at(trainer, 900)
        legacy = ckpt / "lr_scheduler.pt"
        torch.save(trainer.lr_scheduler.state_dict(), legacy)
        state_path = ckpt / "train_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["artifacts"]["lr_scheduler.pt"] = legacy.stat().st_size
        state_path.write_text(json.dumps(state), encoding="utf-8")

        extended = _trainer(self.root, total_steps=2000,
                            scheduler="plateau_cosine_floor")
        extended.load_checkpoint(ckpt)
        self.assertEqual(len(extended.lr_timeline.events), 1)
        self.assertAlmostEqual(_curve(extended, [901])[0], 1.0)  # plateau again
        # ...and from here on it IS protected: the next checkpoint carries the
        # timeline.
        second = self._save_at(extended, 1000)
        payload = torch.load(second / "lr_scheduler.pt", map_location="cpu",
                             weights_only=False)
        self.assertEqual(payload["events"][0]["value"], 2000)

    def test_a_payload_from_a_newer_build_is_refused_not_ignored(self):
        trainer = _trainer(self.root, total_steps=1000, scheduler="cosine")
        ckpt = self._save_at(trainer, 100)
        path = ckpt / "lr_scheduler.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["lr_schedule_version"] = _LR_SCHEDULE_STATE_VERSION + 1
        torch.save(payload, path)
        state_path = ckpt / "train_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["artifacts"]["lr_scheduler.pt"] = path.stat().st_size
        state_path.write_text(json.dumps(state), encoding="utf-8")

        resumed = _trainer(self.root, total_steps=1000, scheduler="cosine")
        with self.assertRaises(VaeConfigError) as caught:
            resumed.load_checkpoint(ckpt)
        self.assertIn("format version", str(caught.exception))

    def test_the_split_reads_both_formats(self):
        state = {"last_epoch": 7, "base_lrs": [1e-5]}
        self.assertEqual(_split_lr_scheduler_state(state, 99), (state, [], 7))
        new = {"lr_schedule_version": 1, "scheduler_step": 5,
               "events": [{"kind": "total_steps", "at": 0, "value": 10}],
               "scheduler": state}
        self.assertEqual(_split_lr_scheduler_state(new, 99),
                         (state, new["events"], 5))


class VaeScheduleVocabularyMirrorTest(unittest.TestCase):
    """The public VAE scheduler vocabulary matches the backend registry."""

    def test_the_openapi_enum_is_the_offered_vocabulary(self):
        spec = yaml.safe_load((_REPO / "openapi.yaml").read_text(encoding="utf-8"))
        props = spec["components"]["schemas"]["VaeTrainingDefaults"]["properties"]
        self.assertEqual(props["lr_scheduler"]["enum"],
                         list(VALID_LR_SCHEDULERS))

if __name__ == "__main__":
    unittest.main()
