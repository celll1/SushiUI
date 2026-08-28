"""Disk-space-aware checkpoint retention.

Run 121 (SenseNova both-branch full FT) died at step 39672: 32.7 GiB free on a
volume that needed 60.85 GiB for one checkpoint set, ``max_step_saves_to_keep``
2, and a trainer that SAVES then prunes -- so keep=N transiently needs N+1 sets
(~182 GiB). The failed write left a truncated 14.04 GB ``*_optimizer.pt``, which
a later host-resident resume treats as FATAL (``OptimizerStateFileUnreadable``).

What is exercised here, all with fakes -- no real large file, no real full
volume: free space is a monkeypatched reading, checkpoint "sets" are a few bytes
each, and the ENOSPC is a raised exception with the real message text
safetensors and torch produce.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/checkpoint_space_retention_test.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import tempfile  # noqa: E402

import core.training.base_trainer as bt  # noqa: E402
from core.training.checkpoint_space import (  # noqa: E402
    GIB,
    KEEP_FLOOR_AFTER_WRITE,
    KEEP_FLOOR_BEFORE_WRITE,
    CheckpointSaveSpaceError,
    estimate_set_bytes,
    is_disk_full_error,
    plan_retention,
)

# Run 121's real numbers.
SET_BYTES = int(60.85 * GIB)
FREE_BYTES = int(32.7 * GIB)

# The two writers' verbatim failure text (the safetensors one carries a
# localized OS string; only the numeric tail is matchable).
SAFETENSORS_ENOSPC = (
    "Error while serializing: I/O error: "
    "ディスクに十分な空き領域"
    "がありません。 (os error 112)"
)
TORCH_SHORT_WRITE = (
    "[enforce fail at inline_container.cc:668] . unexpected pos 15080239872 vs 15080239704"
)


class FakeSafetensorError(Exception):
    pass


# ---------------------------------------------------------------------------
# Fake trainer: the real retention methods over a temp directory
# ---------------------------------------------------------------------------

class FakeTrainer:
    """The shipped methods, with only the save itself faked."""

    _safe_unlink = bt.BaseTrainer._safe_unlink
    _cleanup_old_checkpoints = bt.BaseTrainer._cleanup_old_checkpoints
    _cleanup_old_optimizer_states = bt.BaseTrainer._cleanup_old_optimizer_states
    _run_checkpoint_cleanup = bt.BaseTrainer._run_checkpoint_cleanup
    _existing_checkpoint_set_sizes = bt.BaseTrainer._existing_checkpoint_set_sizes
    _estimate_checkpoint_set_bytes = bt.BaseTrainer._estimate_checkpoint_set_bytes
    _plan_checkpoint_space = bt.BaseTrainer._plan_checkpoint_space
    _announce_checkpoint_space_plan = bt.BaseTrainer._announce_checkpoint_space_plan
    _delete_partial_step_artifacts = bt.BaseTrainer._delete_partial_step_artifacts
    _periodic_save_with_space_guard = bt.BaseTrainer._periodic_save_with_space_guard

    def __init__(self, output_dir: Path, fail_saves: int = 0):
        self.output_dir = Path(output_dir)
        self.run_name = "run121"
        self.log_prefix = "[Test]"
        self.run_id = None
        self.config = {"optimizer": "adamw8bit_ringbuffer"}
        self.events = []
        self.fail_saves = fail_saves
        self.saved_steps = []

    # -- fakes ---------------------------------------------------------
    def _save_checkpoint_bundle(self, step, epoch, batch_idx, multi_noise_timesteps):
        self.events.append(("save", step))
        if self.fail_saves > 0:
            self.fail_saves -= 1
            # A real failure leaves the artefact it got partway through.
            write_set(self.output_dir, self.run_name, step, weight_bytes=8, optimizer_bytes=3)
            raise FakeSafetensorError(SAFETENSORS_ENOSPC)
        write_set(self.output_dir, self.run_name, step)
        self.saved_steps.append(step)

    def _cleanup_old_checkpoints(self, keep):  # noqa: F811 - records, then real
        self.events.append(("prune", keep))
        bt.BaseTrainer._cleanup_old_checkpoints(self, keep)

    def _cleanup_old_optimizer_states(self, keep):  # noqa: F811
        self.events.append(("prune_opt", keep))
        bt.BaseTrainer._cleanup_old_optimizer_states(self, keep)


def write_set(directory: Path, run_name: str, step: int,
              weight_bytes: int = 16, optimizer_bytes: int = 32) -> None:
    """One checkpoint set on disk: weights + optimizer + state sidecars."""
    base = f"{run_name}_step_{step:06d}"
    (directory / f"{base}.safetensors").write_bytes(b"w" * weight_bytes)
    (directory / f"{base}_optimizer.pt").write_bytes(b"o" * optimizer_bytes)
    (directory / f"{base}_state.json").write_text("{}")


def steps_on_disk(directory: Path, suffix: str = ".safetensors") -> list:
    out = []
    for p in directory.glob(f"*_step_*{suffix}"):
        if suffix == ".safetensors" and p.name.endswith("_optimizer.pt"):
            continue
        if bt.QUARANTINE_ENTRY_MARKER in p.name:
            continue
        out.append(int(p.name.rsplit("_step_", 1)[1][:6]))
    return sorted(out)


def patch_free(test, value):
    original = bt._volume_free_bytes
    bt._volume_free_bytes = lambda path: value
    test.addCleanup(lambda: setattr(bt, "_volume_free_bytes", original))


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)


# ---------------------------------------------------------------------------
# 1. The preflight arithmetic
# ---------------------------------------------------------------------------

class RetentionPlanTest(unittest.TestCase):
    def test_ample_space_keeps_the_request_and_does_not_reorder(self):
        """MUTANT: pruning unconditionally before the write. With room to
        spare, the retention pass must stay where it was (after the save) --
        pruning first briefly holds one complete set fewer."""
        plan = plan_retention(
            free=500 * GIB, required=SET_BYTES,
            set_sizes_newest_first=[SET_BYTES, SET_BYTES], requested_keep=2,
        )
        self.assertEqual(plan.effective_keep, 2)
        self.assertFalse(plan.reduced)
        self.assertTrue(plan.fits_as_is)
        self.assertFalse(plan.prune_first)

    def test_run_121_reduces_keep_and_prunes_first(self):
        """MUTANT: the shipped ordering (save, then prune). 32.7 GiB free with
        two 60.85 GiB sets on disk and keep=2 is the incident: keep must drop to
        the floor and the prune must move ahead of the write."""
        plan = plan_retention(
            free=FREE_BYTES, required=SET_BYTES,
            set_sizes_newest_first=[SET_BYTES, SET_BYTES], requested_keep=2,
        )
        self.assertEqual(plan.effective_keep, KEEP_FLOOR_BEFORE_WRITE)
        self.assertTrue(plan.prune_first)
        self.assertTrue(plan.fits)
        self.assertGreaterEqual(plan.free_bytes + plan.reclaim_bytes, SET_BYTES)

    def test_keep_is_reduced_only_as_far_as_needed(self):
        """MUTANT: dropping straight to the floor whenever space is tight.
        Five sets, room for one more after deleting two -> keep 4, not 2."""
        sizes = [10, 10, 10, 10, 10]
        plan = plan_retention(free=5, required=20, set_sizes_newest_first=sizes,
                              requested_keep=6)
        self.assertEqual(plan.effective_keep, 4)

    def test_floor_is_never_breached_and_a_hopeless_save_is_flagged(self):
        """MUTANT: letting the search run to keep=0/1 before the write. Even
        when nothing can make the save fit, the newest set on disk (the run's
        resume target) survives and the plan reports fits=False."""
        plan = plan_retention(free=1, required=10_000,
                              set_sizes_newest_first=[10, 10, 10], requested_keep=3)
        self.assertEqual(plan.effective_keep, KEEP_FLOOR_BEFORE_WRITE)
        self.assertFalse(plan.fits)

    def test_keep_all_stays_keep_all_until_it_cannot(self):
        plan = plan_retention(free=10_000, required=10,
                              set_sizes_newest_first=[10, 10], requested_keep=0)
        self.assertEqual(plan.effective_keep, 0)
        tight = plan_retention(free=1, required=10,
                               set_sizes_newest_first=[10, 10], requested_keep=0)
        self.assertEqual(tight.effective_keep, 2)

    def test_unreadable_volume_changes_nothing(self):
        """MUTANT: treating an unknown free-space reading as zero, which would
        prune to the floor on every save on a volume shutil cannot stat."""
        plan = plan_retention(free=None, required=SET_BYTES,
                              set_sizes_newest_first=[SET_BYTES], requested_keep=3)
        self.assertEqual(plan.effective_keep, 3)
        self.assertFalse(plan.prune_first)

    def test_headroom_means_an_exact_fit_is_not_a_fit(self):
        plan = plan_retention(free=100, required=100,
                              set_sizes_newest_first=[50, 50], requested_keep=3)
        self.assertTrue(plan.prune_first)


class DiskFullClassifierTest(unittest.TestCase):
    def test_every_writer_that_can_report_enospc_is_recognized(self):
        """MUTANT: matching only OSError.errno. Neither of the two writers that
        actually failed on run 121 raises an OSError."""
        self.assertTrue(is_disk_full_error(FakeSafetensorError(SAFETENSORS_ENOSPC)))
        self.assertTrue(is_disk_full_error(RuntimeError(TORCH_SHORT_WRITE)))
        self.assertTrue(is_disk_full_error(OSError(28, "No space left on device")))

    def test_unrelated_failures_are_not_disk_full(self):
        self.assertFalse(is_disk_full_error(RuntimeError("CUDA out of memory")))
        self.assertFalse(is_disk_full_error(PermissionError(13, "Access is denied")))


class EstimateTest(unittest.TestCase):
    def test_ringbuffer_estimate_is_in_the_measured_ballpark(self):
        """16.21B params, bf16 weights, uint8 moment pair: the estimate must
        land near run 121's measured 60.85 GiB set, not an order out."""
        estimate = estimate_set_bytes(16_210_000_000, 2, "adamw8bit_ringbuffer")
        self.assertLess(abs(estimate - SET_BYTES) / SET_BYTES, 0.05)

    def test_unknown_optimizer_overestimates_rather_than_under(self):
        self.assertGreater(
            estimate_set_bytes(1_000, 2, "some_new_optimizer"),
            estimate_set_bytes(1_000, 2, "adafactor"),
        )


# ---------------------------------------------------------------------------
# 2. The estimate and the floor, against a real directory
# ---------------------------------------------------------------------------

class MeasuredEstimateTest(TempDirCase):
    def test_estimate_uses_the_largest_set_not_the_newest(self):
        """MUTANT: measuring the NEWEST set. Run 121's newest set is the
        truncated one; sizing the next save from it under-books the space."""
        trainer = FakeTrainer(self.dir)
        write_set(self.dir, trainer.run_name, 100, weight_bytes=64, optimizer_bytes=64)
        write_set(self.dir, trainer.run_name, 200, weight_bytes=64, optimizer_bytes=1)
        sizes = trainer._existing_checkpoint_set_sizes()
        self.assertEqual(len(sizes), 2)
        self.assertEqual(trainer._estimate_checkpoint_set_bytes(sizes), max(sizes))


class OptimizerRetentionTest(TempDirCase):
    def test_optimizer_states_prune_harder_than_the_weights(self):
        """MUTANT: deleting optimizer states only as a side effect of pruning
        their parent checkpoint (the shipped behaviour). Four sets kept, one
        optimizer state: the .pt files must go while the weights stay."""
        trainer = FakeTrainer(self.dir)
        for step in (100, 200, 300, 400):
            write_set(self.dir, trainer.run_name, step)

        bt.BaseTrainer._cleanup_old_optimizer_states(trainer, 1)

        self.assertEqual(steps_on_disk(self.dir), [100, 200, 300, 400])
        self.assertEqual(steps_on_disk(self.dir, "_optimizer.pt"), [400])

    def test_the_resume_targets_state_survives_even_out_of_step_order(self):
        """MUTANT: keeping the newest N .pt files and nothing else. An
        optimizer state written for a step whose weights are the newest
        checkpoint must not be deleted because a later .pt exists."""
        trainer = FakeTrainer(self.dir)
        write_set(self.dir, trainer.run_name, 100)
        write_set(self.dir, trainer.run_name, 200)
        # A newer optimizer state with no weights beside it (an emergency save
        # whose checkpoint write failed).
        (self.dir / f"{trainer.run_name}_step_000300_optimizer.pt").write_bytes(b"o")

        bt.BaseTrainer._cleanup_old_optimizer_states(trainer, 1)

        self.assertEqual(steps_on_disk(self.dir, "_optimizer.pt"), [200, 300])

    def test_keep_all_and_quarantined_states_are_untouched(self):
        trainer = FakeTrainer(self.dir)
        write_set(self.dir, trainer.run_name, 100)
        write_set(self.dir, trainer.run_name, 200)
        quarantined = (self.dir /
                       f"{trainer.run_name}{bt.QUARANTINE_ENTRY_MARKER}000100_optimizer.pt")
        quarantined.write_bytes(b"o")

        bt.BaseTrainer._cleanup_old_optimizer_states(trainer, 0)
        self.assertEqual(steps_on_disk(self.dir, "_optimizer.pt"), [100, 200])

        bt.BaseTrainer._cleanup_old_optimizer_states(trainer, 1)
        self.assertTrue(quarantined.exists())


# ---------------------------------------------------------------------------
# 3. The save path
# ---------------------------------------------------------------------------

class PeriodicSaveSpaceGuardTest(TempDirCase):
    def _trainer(self, free, fail_saves=0):
        trainer = FakeTrainer(self.dir, fail_saves=fail_saves)
        patch_free(self, free)
        return trainer

    def test_ample_space_saves_then_prunes(self):
        """MUTANT: moving the retention pass before the write unconditionally."""
        trainer = self._trainer(free=10 * GIB)
        write_set(self.dir, trainer.run_name, 100)
        trainer._periodic_save_with_space_guard(
            step=200, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=2, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )
        kinds = [e[0] for e in trainer.events]
        self.assertEqual(kinds[0], "save")
        self.assertIn("prune", kinds)
        self.assertLess(kinds.index("save"), kinds.index("prune"))

    def test_tight_space_prunes_before_writing(self):
        """MUTANT: the shipped save-then-prune ordering. With three sets on
        disk and room for well under two, the prune must precede the save --
        that ordering is the whole reason keep=2 needed 3 sets of space."""
        trainer = self._trainer(free=40)
        for step in (100, 200, 300):
            write_set(self.dir, trainer.run_name, step)
        trainer._periodic_save_with_space_guard(
            step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )
        kinds = [e[0] for e in trainer.events]
        self.assertLess(kinds.index("prune"), kinds.index("save"))
        self.assertIn(400, steps_on_disk(self.dir))

    def test_the_last_complete_set_is_never_pruned_before_the_write(self):
        """MUTANT: a pre-write floor of 1 (i.e. deleting every old set to make
        room). The save that follows may itself fail; the newest existing set
        is what the run would resume from."""
        trainer = self._trainer(free=1)
        write_set(self.dir, trainer.run_name, 100)
        trainer._periodic_save_with_space_guard(
            step=200, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=5, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )
        self.assertIn(100, steps_on_disk(self.dir))

    def test_reduced_retention_reaches_the_warning_channel(self):
        """MUTANT: reducing retention silently. A run that quietly stops
        keeping the checkpoints the user asked for must say so."""
        emitted = []
        original = bt.emit_training_warning
        bt.emit_training_warning = lambda message, **kw: emitted.append((message, kw))
        self.addCleanup(lambda: setattr(bt, "emit_training_warning", original))

        trainer = self._trainer(free=1)
        for step in (100, 200, 300):
            write_set(self.dir, trainer.run_name, step)
        trainer._periodic_save_with_space_guard(
            step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )
        self.assertEqual(len(emitted), 1)
        self.assertIn("retention reduced from 3 to 2", emitted[0][0])
        self.assertEqual(emitted[0][1].get("code"), "checkpoint_retention_reduced")

        # Same reduction next interval: console only, no second persisted notice.
        trainer._periodic_save_with_space_guard(
            step=500, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )
        self.assertEqual(len(emitted), 1)

    def test_enospc_is_retried_once_after_pruning(self):
        """MUTANT: letting the first ENOSPC end the run. One failure, then a
        prune, then a successful retry -- the run continues."""
        trainer = self._trainer(free=40, fail_saves=1)
        for step in (100, 200, 300):
            write_set(self.dir, trainer.run_name, step)
        trainer._periodic_save_with_space_guard(
            step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )
        self.assertEqual([e for e in trainer.events if e[0] == "save"],
                         [("save", 400), ("save", 400)])
        self.assertEqual(trainer.saved_steps, [400])

    def test_a_failed_write_leaves_no_partial_artefact(self):
        """MUTANT: leaving the half-written file where it fell. Run 121's
        truncated 14.04 GB optimizer .pt is fatal for a host-resident resume
        and blocks it until a human deletes the file."""
        trainer = self._trainer(free=40, fail_saves=1)
        for step in (100, 200, 300):
            write_set(self.dir, trainer.run_name, step)
        trainer._periodic_save_with_space_guard(
            step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )
        # The retry's own (complete) set is the only step-400 optimizer file,
        # and it is the full-size one, not the 3-byte partial.
        partial = self.dir / f"{trainer.run_name}_step_000400_optimizer.pt"
        self.assertEqual(partial.stat().st_size, 32)

    def test_a_hopeless_enospc_fails_with_the_numbers(self):
        """MUTANT: re-raising the raw SafetensorError. The shipped failure is a
        localized OS string with no free/required/volume in it."""
        trainer = self._trainer(free=40, fail_saves=5)
        for step in (100, 200, 300):
            write_set(self.dir, trainer.run_name, step)
        with self.assertRaises(CheckpointSaveSpaceError) as caught:
            trainer._periodic_save_with_space_guard(
                step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
                max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
                save_every_n_steps=100,
            )
        message = str(caught.exception)
        self.assertIn("free", message)
        self.assertIn("required", message)
        self.assertIn("step 400", message)
        self.assertIsNotNone(caught.exception.free_bytes)
        # And nothing of step 400 is left behind.
        self.assertNotIn(400, steps_on_disk(self.dir))
        self.assertNotIn(400, steps_on_disk(self.dir, "_optimizer.pt"))

    def test_a_non_space_failure_is_not_swallowed(self):
        """MUTANT: treating every save failure as ENOSPC and retrying it."""
        trainer = self._trainer(free=10 * GIB)

        def boom(*args, **kwargs):
            trainer.events.append(("save", 400))
            raise RuntimeError("CUDA error: device-side assert triggered")

        trainer._save_checkpoint_bundle = boom
        with self.assertRaises(RuntimeError):
            trainer._periodic_save_with_space_guard(
                step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
                max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
                save_every_n_steps=100,
            )
        self.assertEqual(len([e for e in trainer.events if e[0] == "save"]), 1)


class PartialArtefactCleanupTest(TempDirCase):
    def test_a_completed_periodic_save_is_not_mistaken_for_a_partial_one(self):
        """MUTANT: deleting every file for the step unconditionally. The
        emergency handler can run LATER in the same iteration as a periodic
        save that already succeeded at this step; its own failed
        save_checkpoint must not take that good set with it."""
        trainer = FakeTrainer(self.dir)
        write_set(self.dir, trainer.run_name, 400)
        trainer._last_periodic_checkpoint_step = 400

        trainer._delete_partial_step_artifacts(400)

        self.assertEqual(steps_on_disk(self.dir), [400])


class SaveOptimizerStatePartialTest(TempDirCase):
    def test_a_failed_torch_save_deletes_its_own_output(self):
        """MUTANT: no try/except around torch.save. This is exactly how run
        121's truncated optimizer file was produced -- by the EMERGENCY
        handler, which no space preflight covers."""
        import torch

        trainer = FakeTrainer(self.dir)
        trainer.save_optimizer_state = bt.BaseTrainer.save_optimizer_state.__get__(trainer)
        target = self.dir / f"{trainer.run_name}_step_000400_optimizer.pt"

        original_save = torch.save
        original_all = bt.all_optimizers

        def truncating_save(payload, path, *args, **kwargs):
            Path(path).write_bytes(b"partial")
            raise RuntimeError(TORCH_SHORT_WRITE)

        class _Optimizer:
            def state_dict(self):
                return {"state": {}}

        torch.save = truncating_save
        bt.all_optimizers = lambda trainer_: [_Optimizer()]
        self.addCleanup(lambda: setattr(torch, "save", original_save))
        self.addCleanup(lambda: setattr(bt, "all_optimizers", original_all))

        with self.assertRaises(RuntimeError):
            trainer.save_optimizer_state(step=400)
        self.assertFalse(target.exists())


# ---------------------------------------------------------------------------
# 4. The parameter's own round trip
# ---------------------------------------------------------------------------

class ParameterRoundTripTest(unittest.TestCase):
    def test_it_survives_a_config_panel_edit(self):
        """MUTANT: omitting the ("save",) entry in _YAML_FIELD_LOCATIONS. The
        extractor would then look for it in process.train, find nothing, and
        every config edit would silently reset it to the default."""
        import yaml

        from api.param_defaults import TRAINING_DEFAULTS
        from api.routes import _extract_request_params_from_yaml
        from core.training.training_config import TrainingConfigGenerator

        self.assertEqual(TRAINING_DEFAULTS["max_optimizer_saves_to_keep"], 1)

        generator = TrainingConfigGenerator()
        text = generator.generate_full_finetune_config(
            run_name="r", base_model_path="m.safetensors", dataset_path="d",
            output_dir="o", p={"total_steps": 10, "max_optimizer_saves_to_keep": 4},
        )
        process = yaml.safe_load(text)["config"]["process"][0]
        self.assertEqual(process["save"]["max_optimizer_saves_to_keep"], 4)

        params = _extract_request_params_from_yaml(process, "full_finetune")
        self.assertEqual(params["max_optimizer_saves_to_keep"], 4)

    def test_an_unset_value_lands_on_the_shared_default(self):
        import yaml

        from api.param_defaults import TRAINING_DEFAULTS
        from core.training.training_config import TrainingConfigGenerator

        text = TrainingConfigGenerator().generate_lora_config(
            run_name="r", base_model_path="m.safetensors", dataset_path="d",
            output_dir="o", p={"total_steps": 10},
        )
        process = yaml.safe_load(text)["config"]["process"][0]
        self.assertEqual(process["save"]["max_optimizer_saves_to_keep"],
                         TRAINING_DEFAULTS["max_optimizer_saves_to_keep"])


if __name__ == "__main__":
    unittest.main()
