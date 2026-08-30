"""``TrainingCheckpoint`` DB registration on periodic saves.

Run 121 (SenseNova full fine-tune) has real checkpoint files on disk but zero
rows in ``training_checkpoints``. The Training Monitor's checkpoint list
(``GET /training/runs/{id}/checkpoints``) reads the DB table exclusively --
it does not fall back to scanning the filesystem -- so those checkpoints are
invisible in the UI despite existing on disk.

Root cause: DB registration for this table was implemented once, in the
pre-refactor ``lora_trainer.py`` (commit 6b128594), and lost when the
architecture refactor rebuilt checkpoint saving around
``BaseTrainer._save_checkpoint_bundle``. Only ``vae_trainer.py`` (a separate,
non-``BaseTrainer`` code path) still writes this table.

This test exercises the fix -- ``BaseTrainer._record_checkpoint_db_row`` /
``_delete_checkpoint_db_row``, wired into ``_save_checkpoint_bundle`` and
``_cleanup_old_checkpoints`` -- against every checkpoint LAYOUT actually used
by a trainer in this repo: a single safetensors file, a sharded
index+shards save, and a ControlNet "standard" directory save. All with an
isolated in-memory training DB; the real ``training.db`` is never touched.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/checkpoint_db_registration_test.py -q
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import core.training.base_trainer as bt  # noqa: E402
import core.training.controlnet_trainer as ct  # noqa: E402
import database as db_module  # noqa: E402
from database.models import TrainingBase, TrainingCheckpoint  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


class FakeTrainer:
    """The shipped checkpoint-bundle methods, with only the writers faked."""

    _safe_unlink = bt.BaseTrainer._safe_unlink
    _begin_checkpoint_bundle = bt.BaseTrainer._begin_checkpoint_bundle
    _note_checkpoint_bundle_stage = bt.BaseTrainer._note_checkpoint_bundle_stage
    _completed_checkpoint_bundle_stages = bt.BaseTrainer._completed_checkpoint_bundle_stages
    _record_checkpoint_db_row = bt.BaseTrainer._record_checkpoint_db_row
    _delete_checkpoint_db_row = bt.BaseTrainer._delete_checkpoint_db_row
    _cleanup_old_checkpoints = bt.BaseTrainer._cleanup_old_checkpoints

    def __init__(self, output_dir: Path, run_id=1, layout: str = "single_file"):
        self.output_dir = Path(output_dir)
        self.run_name = "run121"
        self.log_prefix = "[Test]"
        self.run_id = run_id
        self.layout = layout
        self.events = []

    def save_checkpoint(self, step: int, epoch: int) -> None:
        self.events.append(("save", step))
        base = f"{self.run_name}_step_{step:06d}"
        if self.layout == "single_file":
            (self.output_dir / f"{base}.safetensors").write_bytes(b"w" * 16)
        elif self.layout == "sharded":
            index = self.output_dir / f"{base}.safetensors.index.json"
            shard1 = f"{base}-00001-of-00002.safetensors"
            shard2 = f"{base}-00002-of-00002.safetensors"
            (self.output_dir / shard1).write_bytes(b"a" * 10)
            (self.output_dir / shard2).write_bytes(b"b" * 6)
            index.write_text(
                '{"weight_map": {"w1": "%s", "w2": "%s"}}' % (shard1, shard2)
            )
        elif self.layout == "controlnet_dir":
            ckpt_dir = self.output_dir / f"{self.run_name}_controlnet_step_{step:06d}"
            ckpt_dir.mkdir()
            (ckpt_dir / "config.json").write_text("{}")
            (ckpt_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 20)
        else:
            raise AssertionError(f"unknown layout {self.layout}")


class ControlNetFakeTrainer(FakeTrainer):
    """Exercises ControlNetTrainer's own ``_cleanup_old_checkpoints`` override."""

    _cleanup_old_checkpoints = ct.ControlNetTrainer._cleanup_old_checkpoints

    def __init__(self, output_dir: Path, run_id=1):
        super().__init__(output_dir, run_id=run_id, layout="controlnet_dir")
        self.controlnet_type = "standard"


class TrainingDbCase(unittest.TestCase):
    """Isolated in-memory training DB -- the real training.db is never opened."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

        self.engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        TrainingBase.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        original = db_module.get_training_db

        def fake_get_training_db():
            session = self.SessionLocal()
            try:
                yield session
            finally:
                session.close()

        db_module.get_training_db = fake_get_training_db
        self.addCleanup(lambda: setattr(db_module, "get_training_db", original))

    def all_checkpoints(self):
        session = self.SessionLocal()
        try:
            return session.query(TrainingCheckpoint).order_by(TrainingCheckpoint.step).all()
        finally:
            session.close()


# ---------------------------------------------------------------------------
# 1. Registration across every on-disk layout a trainer actually writes
# ---------------------------------------------------------------------------

class RegistrationLayoutTest(TrainingDbCase):
    def test_single_file_checkpoint_is_registered(self):
        trainer = FakeTrainer(self.dir, layout="single_file")
        before = set(self.dir.iterdir())
        trainer.save_checkpoint(step=100, epoch=2)
        trainer._record_checkpoint_db_row(step=100, epoch=2, before_entries=before)

        rows = self.all_checkpoints()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.run_id, 1)
        self.assertEqual(row.step, 100)
        self.assertEqual(row.epoch, 2)
        self.assertEqual(row.checkpoint_name, "run121_step_000100.safetensors")
        self.assertEqual(row.file_size, 16)

    def test_sharded_checkpoint_uses_index_as_entry_and_sums_shard_bytes(self):
        trainer = FakeTrainer(self.dir, layout="sharded")
        before = set(self.dir.iterdir())
        trainer.save_checkpoint(step=200, epoch=0)
        trainer._record_checkpoint_db_row(step=200, epoch=0, before_entries=before)

        rows = self.all_checkpoints()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row.checkpoint_name.endswith(".safetensors.index.json"))
        # 10 (shard1) + 6 (shard2) + the index.json's own text bytes.
        index_path = self.dir / "run121_step_000200.safetensors.index.json"
        self.assertEqual(row.file_size, 10 + 6 + index_path.stat().st_size)

    def test_controlnet_standard_directory_checkpoint_is_registered(self):
        """MUTANT: a naming-convention-based finder. ``_list_checkpoint_entries``
        globs ``*_step_*.safetensors[.index.json]`` and never matches a
        ControlNet "standard" directory save -- the before/after diff must."""
        trainer = FakeTrainer(self.dir, layout="controlnet_dir")
        before = set(self.dir.iterdir())
        trainer.save_checkpoint(step=300, epoch=1)
        trainer._record_checkpoint_db_row(step=300, epoch=1, before_entries=before)

        rows = self.all_checkpoints()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.checkpoint_name, "run121_controlnet_step_000300")
        self.assertEqual(row.file_size, len("{}") + 20)


# ---------------------------------------------------------------------------
# 2. Upsert behavior and failure isolation
# ---------------------------------------------------------------------------

class RegistrationBehaviorTest(TrainingDbCase):
    def test_resaving_the_same_step_upserts_not_duplicates(self):
        trainer = FakeTrainer(self.dir, layout="single_file")
        before = set(self.dir.iterdir())
        trainer.save_checkpoint(step=50, epoch=0)
        trainer._record_checkpoint_db_row(step=50, epoch=0, before_entries=before)

        # A retry overwrites the same file with different content/size.
        (self.dir / "run121_step_000050.safetensors").write_bytes(b"y" * 40)
        trainer._record_checkpoint_db_row(step=50, epoch=1, before_entries=set())

        rows = self.all_checkpoints()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].epoch, 1)
        self.assertEqual(rows[0].file_size, 40)

    def test_run_id_none_is_a_silent_noop(self):
        trainer = FakeTrainer(self.dir, run_id=None, layout="single_file")
        before = set(self.dir.iterdir())
        trainer.save_checkpoint(step=10, epoch=0)
        trainer._record_checkpoint_db_row(step=10, epoch=0, before_entries=before)
        self.assertEqual(self.all_checkpoints(), [])

    def test_no_new_entry_found_is_a_silent_noop(self):
        """A step where save_checkpoint() somehow wrote nothing matching the
        marker (e.g. an unrecognized future layout) must not raise or insert
        a bogus row."""
        trainer = FakeTrainer(self.dir, layout="single_file")
        before = set(self.dir.iterdir())
        # Do NOT call save_checkpoint(): nothing new on disk for step 999.
        trainer._record_checkpoint_db_row(step=999, epoch=0, before_entries=before)
        self.assertEqual(self.all_checkpoints(), [])

    def test_db_failure_during_registration_does_not_raise(self):
        """Training must survive a DB hiccup (locked file, disk full, etc.)."""
        def raising_get_training_db():
            raise RuntimeError("simulated DB failure")
            yield  # pragma: no cover - generator shape only

        db_module.get_training_db = raising_get_training_db

        trainer = FakeTrainer(self.dir, layout="single_file")
        before = set(self.dir.iterdir())
        trainer.save_checkpoint(step=77, epoch=0)
        # Must not raise.
        trainer._record_checkpoint_db_row(step=77, epoch=0, before_entries=before)


# ---------------------------------------------------------------------------
# 3. Full bundle wiring + pruning deletes the row
# ---------------------------------------------------------------------------

class BundleAndPruningTest(TrainingDbCase):
    def test_save_checkpoint_bundle_registers_the_row(self):
        trainer = FakeTrainer(self.dir, layout="single_file")

        def save_training_state(step, epoch, batch_idx, multi_noise_timesteps):
            (self.dir / f"run121_step_{step:06d}_state.json").write_text("{}")

        def save_optimizer_state(step):
            (self.dir / f"run121_step_{step:06d}_optimizer.pt").write_bytes(b"o")

        trainer.save_training_state = save_training_state
        trainer.save_optimizer_state = save_optimizer_state
        trainer.save_ema_state = lambda step: None
        trainer._save_ema_checkpoint = lambda step, epoch: None
        trainer._log_metrics_to_db = lambda step, force_flush=True: None
        trainer._save_checkpoint_bundle = bt.BaseTrainer._save_checkpoint_bundle.__get__(trainer)

        trainer._save_checkpoint_bundle(step=500, epoch=3, batch_idx=0, multi_noise_timesteps=1)

        rows = self.all_checkpoints()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].step, 500)
        self.assertEqual(rows[0].epoch, 3)
        # Sidecars written by later stages must not be folded into the
        # weights-stage entry's registered size.
        self.assertEqual(rows[0].file_size, 16)

    def test_pruning_an_old_checkpoint_deletes_its_db_row(self):
        trainer = FakeTrainer(self.dir, layout="single_file")
        for step in (100, 200, 300):
            before = set(self.dir.iterdir())
            trainer.save_checkpoint(step=step, epoch=0)
            trainer._record_checkpoint_db_row(step=step, epoch=0, before_entries=before)

        self.assertEqual(len(self.all_checkpoints()), 3)

        trainer._cleanup_old_checkpoints(max_step_saves_to_keep=2)

        rows = self.all_checkpoints()
        remaining_steps = {r.step for r in rows}
        self.assertEqual(remaining_steps, {200, 300})

    def test_controlnet_cleanup_override_also_deletes_the_db_row(self):
        trainer = ControlNetFakeTrainer(self.dir)
        for step in (10, 20, 30):
            before = set(self.dir.iterdir())
            trainer.save_checkpoint(step=step, epoch=0)
            trainer._record_checkpoint_db_row(step=step, epoch=0, before_entries=before)

        self.assertEqual(len(self.all_checkpoints()), 3)

        trainer._cleanup_old_checkpoints(max_step_saves_to_keep=1)

        rows = self.all_checkpoints()
        self.assertEqual({r.step for r in rows}, {30})


if __name__ == "__main__":
    unittest.main()
