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
    survivors_after_prune,
)

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



# Bytes each bundle stage writes, complete and half-written. The bundle is NOT
# atomic on disk: run 121's weights were complete and valid when the optimizer
# stage hit ENOSPC, which is the case the whole-set delete used to destroy.
_STAGE_FILE_SUFFIX = {
    "weights": ".safetensors",
    "state": "_state.json",
    "optimizer": "_optimizer.pt",
}
_STAGE_FULL_BYTES = {"weights": 16, "state": 2, "optimizer": 32}
_STAGE_PARTIAL_BYTES = {"weights": 8, "state": 1, "optimizer": 3}


class FakeTrainer:
    """The shipped methods, with only the individual writers faked."""

    _safe_unlink = bt.BaseTrainer._safe_unlink
    _cleanup_old_checkpoints = bt.BaseTrainer._cleanup_old_checkpoints
    _cleanup_old_optimizer_states = bt.BaseTrainer._cleanup_old_optimizer_states
    _run_checkpoint_cleanup = bt.BaseTrainer._run_checkpoint_cleanup
    _existing_checkpoint_set_parts = bt.BaseTrainer._existing_checkpoint_set_parts
    _existing_checkpoint_set_sizes = bt.BaseTrainer._existing_checkpoint_set_sizes
    _estimate_checkpoint_set_bytes = bt.BaseTrainer._estimate_checkpoint_set_bytes
    _plan_checkpoint_space = bt.BaseTrainer._plan_checkpoint_space
    _announce_checkpoint_space_plan = bt.BaseTrainer._announce_checkpoint_space_plan
    _begin_checkpoint_bundle = bt.BaseTrainer._begin_checkpoint_bundle
    _note_checkpoint_bundle_stage = bt.BaseTrainer._note_checkpoint_bundle_stage
    _completed_checkpoint_bundle_stages = bt.BaseTrainer._completed_checkpoint_bundle_stages
    # run_id is always None below, so both are a no-op -- bound only so the
    # shared _save_checkpoint_bundle / _cleanup_old_checkpoints code they're
    # called from doesn't AttributeError on this fake.
    _record_checkpoint_db_row = bt.BaseTrainer._record_checkpoint_db_row
    _delete_checkpoint_db_row = bt.BaseTrainer._delete_checkpoint_db_row
    _delete_partial_step_artifacts = bt.BaseTrainer._delete_partial_step_artifacts
    _save_checkpoint_bundle = bt.BaseTrainer._save_checkpoint_bundle
    _periodic_save_with_space_guard = bt.BaseTrainer._periodic_save_with_space_guard

    def __init__(self, output_dir: Path, fail_saves: int = 0,
                 fail_stage: str = "optimizer"):
        self.output_dir = Path(output_dir)
        self.run_name = "run121"
        self.log_prefix = "[Test]"
        self.run_id = None
        self.config = {"optimizer": "adamw8bit_ringbuffer"}
        self.events = []
        self.fail_saves = fail_saves
        self.fail_stage = fail_stage
        self.saved_steps = []
        self._attempt_fails = False

    def _write_stage(self, step, stage):
        path = self.output_dir / (
            f"{self.run_name}_step_{step:06d}{_STAGE_FILE_SUFFIX[stage]}")
        failing = self._attempt_fails and stage == self.fail_stage
        sizes = _STAGE_PARTIAL_BYTES if failing else _STAGE_FULL_BYTES
        path.write_bytes(b"x" * sizes[stage])
        if failing:
            raise FakeSafetensorError(SAFETENSORS_ENOSPC)

    def save_checkpoint(self, step, epoch):
        self.events.append(("save", step))
        self._attempt_fails = self.fail_saves > 0
        if self._attempt_fails:
            self.fail_saves -= 1
        self._write_stage(step, "weights")

    def save_training_state(self, step, epoch, batch_idx, multi_noise_timesteps):
        self._write_stage(step, "state")

    def save_optimizer_state(self, step):
        self._write_stage(step, "optimizer")

    def save_ema_state(self, step):
        pass

    def _save_ema_checkpoint(self, step, epoch):
        self.saved_steps.append(step)

    def _cleanup_old_checkpoints(self, keep):  # noqa: F811 - records, then real
        self.events.append(("prune", keep))
        bt.BaseTrainer._cleanup_old_checkpoints(self, keep)

    def _cleanup_old_optimizer_states(self, keep, current_step=None):  # noqa: F811
        self.events.append(("prune_opt", keep))
        bt.BaseTrainer._cleanup_old_optimizer_states(self, keep, current_step=current_step)


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



class RetentionPlanTest(unittest.TestCase):
    def test_ample_space_keeps_the_request_and_does_not_reorder(self):
        plan = plan_retention(
            free=500 * GIB, required=SET_BYTES,
            set_sizes_newest_first=[SET_BYTES, SET_BYTES], requested_keep=2,
        )
        self.assertEqual(plan.effective_keep, 2)
        self.assertFalse(plan.reduced)
        self.assertTrue(plan.fits_as_is)
        self.assertFalse(plan.prune_first)

    def test_run_121_reduces_keep_and_prunes_first(self):
        plan = plan_retention(
            free=FREE_BYTES, required=SET_BYTES,
            set_sizes_newest_first=[SET_BYTES, SET_BYTES], requested_keep=2,
        )
        self.assertEqual(plan.effective_keep, KEEP_FLOOR_BEFORE_WRITE)
        self.assertTrue(plan.prune_first)
        self.assertTrue(plan.fits)
        self.assertGreaterEqual(plan.free_bytes + plan.reclaim_bytes, SET_BYTES)

    def test_keep_is_reduced_only_as_far_as_needed(self):
        sizes = [10, 10, 10, 10, 10]
        plan = plan_retention(free=5, required=20, set_sizes_newest_first=sizes,
                              requested_keep=6)
        self.assertEqual(plan.effective_keep, 4)

    def test_floor_is_never_breached_and_a_hopeless_save_is_flagged(self):
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
        plan = plan_retention(free=None, required=SET_BYTES,
                              set_sizes_newest_first=[SET_BYTES], requested_keep=3)
        self.assertEqual(plan.effective_keep, 3)
        self.assertFalse(plan.prune_first)

    def test_keep_one_plans_the_prune_the_trainer_will_actually_run(self):
        plan = plan_retention(free=5, required=20, set_sizes_newest_first=[10, 10],
                              requested_keep=1, floor=1)
        self.assertEqual(survivors_after_prune(plan.effective_keep, 2), 1)
        self.assertLessEqual(plan.reclaim_bytes, 10)
        self.assertFalse(plan.fits)

    def test_headroom_means_an_exact_fit_is_not_a_fit(self):
        plan = plan_retention(free=100, required=100,
                              set_sizes_newest_first=[50, 50], requested_keep=3)
        self.assertTrue(plan.prune_first)


class DiskFullClassifierTest(unittest.TestCase):
    def test_every_writer_that_can_report_enospc_is_recognized(self):
        self.assertTrue(is_disk_full_error(FakeSafetensorError(SAFETENSORS_ENOSPC)))
        self.assertTrue(is_disk_full_error(RuntimeError(TORCH_SHORT_WRITE)))
        self.assertTrue(is_disk_full_error(OSError(28, "No space left on device")))

    def test_unrelated_failures_are_not_disk_full(self):
        self.assertFalse(is_disk_full_error(RuntimeError("CUDA out of memory")))
        self.assertFalse(is_disk_full_error(PermissionError(13, "Access is denied")))

    def test_another_inline_container_assertion_is_not_a_full_disk(self):
        self.assertFalse(is_disk_full_error(RuntimeError(
            "[enforce fail at inline_container.cc:250] . file not found: archive/data.pkl"
        )))

    def test_a_rewrapped_writer_error_is_still_recognized(self):
        try:
            raise FakeSafetensorError(SAFETENSORS_ENOSPC)
        except FakeSafetensorError as inner:
            wrapped = RuntimeError("Failed to save checkpoint shard 3/9")
            wrapped.__cause__ = inner
            self.assertTrue(is_disk_full_error(wrapped))

        try:
            raise OSError(28, "No space left on device")
        except OSError:
            try:
                raise RuntimeError("checkpoint write failed")   # implicit context
            except RuntimeError as chained:
                self.assertTrue(is_disk_full_error(chained))


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



class MeasuredEstimateTest(TempDirCase):
    def test_estimate_uses_the_largest_set_not_the_newest(self):
        trainer = FakeTrainer(self.dir)
        write_set(self.dir, trainer.run_name, 100, weight_bytes=64, optimizer_bytes=64)
        write_set(self.dir, trainer.run_name, 200, weight_bytes=64, optimizer_bytes=1)
        parts = trainer._existing_checkpoint_set_parts()
        self.assertEqual(len(parts), 2)
        self.assertEqual(trainer._estimate_checkpoint_set_bytes(parts),
                         max(w + s for w, s in parts))

    def test_a_set_with_no_sidecar_does_not_halve_the_estimate(self):
        trainer = FakeTrainer(self.dir)
        for step in (100, 200):
            write_set(self.dir, trainer.run_name, step,
                      weight_bytes=64, optimizer_bytes=64)
            (self.dir / f"{trainer.run_name}_step_{step:06d}_optimizer.pt").unlink()

        parts = trainer._existing_checkpoint_set_parts()
        # Nothing on disk is bigger than the weights half...
        self.assertEqual(max(w + s for w, s in parts), 64 + 2)
        # ...but a full save writes weights AND a sidecar. The first save's
        # optimizer state is measured from the run's own history, not from the
        # deflated set: the fall-back to the parameter count only applies with
        # nothing measured at all.
        write_set(self.dir, trainer.run_name, 300, weight_bytes=1, optimizer_bytes=64)
        parts = trainer._existing_checkpoint_set_parts()
        self.assertEqual(trainer._estimate_checkpoint_set_bytes(parts), 64 + 64 + 2)


class OptimizerRetentionTest(TempDirCase):
    def test_optimizer_states_prune_harder_than_the_weights(self):
        trainer = FakeTrainer(self.dir)
        for step in (100, 200, 300, 400):
            write_set(self.dir, trainer.run_name, step)

        bt.BaseTrainer._cleanup_old_optimizer_states(trainer, 1)

        self.assertEqual(steps_on_disk(self.dir), [100, 200, 300, 400])
        self.assertEqual(steps_on_disk(self.dir, "_optimizer.pt"), [400])

    def test_the_resume_targets_state_survives_even_out_of_step_order(self):
        trainer = FakeTrainer(self.dir)
        write_set(self.dir, trainer.run_name, 100)
        write_set(self.dir, trainer.run_name, 200)
        # A newer optimizer state with no weights beside it (an emergency save
        # whose checkpoint write failed).
        (self.dir / f"{trainer.run_name}_step_000300_optimizer.pt").write_bytes(b"o")

        bt.BaseTrainer._cleanup_old_optimizer_states(trainer, 1)

        self.assertEqual(steps_on_disk(self.dir, "_optimizer.pt"), [200, 300])

    def test_a_stale_higher_step_stump_does_not_protect_itself(self):
        trainer = FakeTrainer(self.dir)
        write_set(self.dir, trainer.run_name, 29332, weight_bytes=64, optimizer_bytes=64)
        (self.dir / f"{trainer.run_name}_step_039672.safetensors").write_bytes(b"w" * 64)
        (self.dir / f"{trainer.run_name}_step_039672_optimizer.pt").write_bytes(b"o")

        bt.BaseTrainer._cleanup_old_optimizer_states(trainer, 1, current_step=29332)

        self.assertEqual(steps_on_disk(self.dir, "_optimizer.pt"), [29332])

    def test_a_periodic_save_never_prunes_its_own_optimizer_state(self):
        trainer = FakeTrainer(self.dir)
        patch_free(self, 10 * GIB)
        write_set(self.dir, trainer.run_name, 29332, weight_bytes=64, optimizer_bytes=64)
        (self.dir / f"{trainer.run_name}_step_039672.safetensors").write_bytes(b"w" * 64)
        (self.dir / f"{trainer.run_name}_step_039672_optimizer.pt").write_bytes(b"o")

        trainer._periodic_save_with_space_guard(
            step=29500, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=2, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )

        self.assertEqual(steps_on_disk(self.dir, "_optimizer.pt"), [29500])
        self.assertIn(29500, steps_on_disk(self.dir))

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



class PeriodicSaveSpaceGuardTest(TempDirCase):
    def _trainer(self, free, fail_saves=0):
        trainer = FakeTrainer(self.dir, fail_saves=fail_saves)
        patch_free(self, free)
        return trainer

    def test_ample_space_saves_then_prunes(self):
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
        trainer = self._trainer(free=1)
        write_set(self.dir, trainer.run_name, 100)
        trainer._periodic_save_with_space_guard(
            step=200, epoch=0, batch_idx=1, multi_noise_timesteps=1,
            max_step_saves_to_keep=5, max_optimizer_saves_to_keep=1,
            save_every_n_steps=100,
        )
        self.assertIn(100, steps_on_disk(self.dir))

    def test_reduced_retention_reaches_the_warning_channel(self):
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
        self.assertIn(400, steps_on_disk(self.dir))
        self.assertEqual(
            (self.dir / f"{trainer.run_name}_step_000400.safetensors").stat().st_size,
            _STAGE_FULL_BYTES["weights"])
        self.assertTrue((self.dir / f"{trainer.run_name}_step_000400_state.json").exists())
        # The stump, and only the stump, is gone.
        self.assertNotIn(400, steps_on_disk(self.dir, "_optimizer.pt"))

    def test_a_truncated_weights_write_is_still_cleared(self):
        """The other side of the stage rule: when the WEIGHTS stage is what
        failed, what it left is half a checkpoint and must not be kept."""
        trainer = self._trainer(free=40, fail_saves=5)
        trainer.fail_stage = "weights"
        for step in (100, 200, 300):
            write_set(self.dir, trainer.run_name, step)
        with self.assertRaises(CheckpointSaveSpaceError):
            trainer._periodic_save_with_space_guard(
                step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
                max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
                save_every_n_steps=100,
            )
        self.assertNotIn(400, steps_on_disk(self.dir))
        self.assertNotIn(400, steps_on_disk(self.dir, "_optimizer.pt"))

    def test_a_retry_that_truncates_the_weights_does_not_keep_the_first_attempt(self):
        trainer = self._trainer(free=40, fail_saves=2)
        for step in (100, 200, 300):
            write_set(self.dir, trainer.run_name, step)

        original_write = trainer._write_stage

        def write_stage(step, stage):
            if trainer.fail_saves == 0:      # the retry attempt
                trainer.fail_stage = "weights"
            return original_write(step, stage)

        trainer._write_stage = write_stage
        with self.assertRaises(CheckpointSaveSpaceError):
            trainer._periodic_save_with_space_guard(
                step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
                max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
                save_every_n_steps=100,
            )
        self.assertNotIn(400, steps_on_disk(self.dir))

    def test_a_disk_full_is_catchable_where_the_periodic_save_is_called(self):
        self.assertTrue(issubclass(CheckpointSaveSpaceError, OSError))
        trainer = self._trainer(free=40, fail_saves=5)
        for step in (100, 200, 300):
            write_set(self.dir, trainer.run_name, step)
        try:
            trainer._periodic_save_with_space_guard(
                step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
                max_step_saves_to_keep=3, max_optimizer_saves_to_keep=1,
                save_every_n_steps=100,
            )
        except (PermissionError, OSError) as caught:   # the real except clause
            self.assertIsInstance(caught, CheckpointSaveSpaceError)
        else:
            self.fail("expected the guard to raise")

    def test_a_prune_failure_after_a_good_save_does_not_arm_the_emergency_delete(self):
        trainer = self._trainer(free=10 * GIB)
        write_set(self.dir, trainer.run_name, 100)

        def exploding_cleanup(keep, global_step=0, save_every_n_steps=0):
            if trainer.saved_steps:
                raise OSError(13, "Access is denied")
            bt.BaseTrainer._run_checkpoint_cleanup(trainer, keep, global_step,
                                                   save_every_n_steps)

        trainer._run_checkpoint_cleanup = exploding_cleanup
        with self.assertRaises(OSError):
            trainer._periodic_save_with_space_guard(
                step=400, epoch=0, batch_idx=1, multi_noise_timesteps=1,
                max_step_saves_to_keep=2, max_optimizer_saves_to_keep=1,
                save_every_n_steps=100,
            )
        # train() swallows that OSError; the emergency handler then runs with
        # _delete_partial_step_artifacts(global_step) as its only guard.
        trainer._delete_partial_step_artifacts(400)
        self.assertIn(400, steps_on_disk(self.dir))
        self.assertIn(400, steps_on_disk(self.dir, "_optimizer.pt"))

    def test_a_non_space_failure_is_not_swallowed(self):
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


class MarkerPlacementTest(unittest.TestCase):
    def test_the_completed_save_marker_is_set_inside_the_bundle(self):
        import ast

        source = (BACKEND / "core" / "training" / "base_trainer.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        bundle = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_save_checkpoint_bundle")
        body = ast.unparse(bundle)
        self.assertIn("self._last_periodic_checkpoint_step = step", body)
        self.assertLess(body.index("self.save_checkpoint("),
                        body.index("self._last_periodic_checkpoint_step = step"))
        self.assertLess(body.index("self._last_periodic_checkpoint_step = step"),
                        body.index("self.save_optimizer_state("))
        self.assertEqual(source.count("self._last_periodic_checkpoint_step = step"), 1)
        self.assertNotIn("self._last_periodic_checkpoint_step = global_step", source)


class PartialArtefactCleanupTest(TempDirCase):
    def test_a_completed_periodic_save_is_not_mistaken_for_a_partial_one(self):
        trainer = FakeTrainer(self.dir)
        write_set(self.dir, trainer.run_name, 400)
        trainer._last_periodic_checkpoint_step = 400

        trainer._delete_partial_step_artifacts(400)

        self.assertEqual(steps_on_disk(self.dir), [400])


class SaveOptimizerStatePartialTest(TempDirCase):
    def test_a_failed_torch_save_deletes_its_own_output(self):
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
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_a_failed_rewrite_does_not_destroy_the_previous_bytes(self):
        import torch

        trainer = FakeTrainer(self.dir)
        trainer.save_optimizer_state = bt.BaseTrainer.save_optimizer_state.__get__(trainer)
        target = self.dir / f"{trainer.run_name}_step_000400_optimizer.pt"
        target.write_bytes(b"previous state")

        class _Optimizer:
            def state_dict(self):
                return {"state": {}}

        original_save, original_all = torch.save, bt.all_optimizers

        def truncating_save(payload, path, *args, **kwargs):
            Path(path).write_bytes(b"partial")
            raise RuntimeError(TORCH_SHORT_WRITE)

        torch.save = truncating_save
        bt.all_optimizers = lambda trainer_: [_Optimizer()]
        self.addCleanup(lambda: setattr(torch, "save", original_save))
        self.addCleanup(lambda: setattr(bt, "all_optimizers", original_all))

        with self.assertRaises(RuntimeError):
            trainer.save_optimizer_state(step=400)
        self.assertEqual(target.read_bytes(), b"previous state")
        self.assertEqual(list(self.dir.glob("*.tmp")), [])


class UndeletableStumpTest(TempDirCase):
    """Real files, real unlink semantics: _safe_unlink logs and swallows, so a
    locked or read-only leftover SURVIVES a cleanup that assumed it went."""

    def test_a_stump_that_cannot_be_deleted_still_does_not_shadow_the_good_state(self):
        import stat as _stat
        import time as _time

        original_sleep = _time.sleep
        _time.sleep = lambda *_a, **_kw: None    # _safe_unlink retries with backoff
        self.addCleanup(lambda: setattr(_time, "sleep", original_sleep))

        trainer = FakeTrainer(self.dir)
        # Resumed from 029332, wrote 029500; 039672's leftovers are still there.
        write_set(self.dir, trainer.run_name, 29332, weight_bytes=64, optimizer_bytes=64)
        write_set(self.dir, trainer.run_name, 29500, weight_bytes=64, optimizer_bytes=64)
        (self.dir / f"{trainer.run_name}_step_039672.safetensors").write_bytes(b"w" * 64)
        stump = self.dir / f"{trainer.run_name}_step_039672_optimizer.pt"
        stump.write_bytes(b"o")
        stump.chmod(_stat.S_IREAD)
        self.addCleanup(lambda: stump.chmod(_stat.S_IWRITE | _stat.S_IREAD)
                        if stump.exists() else None)

        bt.BaseTrainer._cleanup_old_optimizer_states(trainer, 1, current_step=29500)

        # An undeletable stump keeps its higher step number on disk for the NEXT
        # cleanup to rank -- so it must not be able to protect itself then either.
        self.assertIn(29500, steps_on_disk(self.dir, "_optimizer.pt"))
        self.assertEqual(
            (self.dir / f"{trainer.run_name}_step_029500_optimizer.pt").stat().st_size, 64)



class ParameterRoundTripTest(unittest.TestCase):
    def test_it_survives_a_config_panel_edit(self):
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
