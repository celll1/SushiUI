"""Guard: mid-epoch resume must reproduce the interrupted epoch's shuffle order.

Why this file exists
--------------------
``save_training_state`` used to save ``random.getstate()`` at checkpoint time --
the LIVE state, already advanced past the shuffle(s) that built the epoch's
batch list and past every per-step ``random`` draw taken since. Restoring that
state and re-shuffling on resume therefore produced a DIFFERENT permutation
than the interrupted epoch used, so ``batches[resume_batch_idx:]`` skipped an
arbitrary slice of the new (wrong) order instead of the already-trained
prefix of the original order.

The fix: ``train()``'s epoch loop snapshots ``random.getstate()`` into
``self._epoch_batch_rng_state`` immediately before building this epoch's
batch list (after the mid-epoch-resume restore, so a resumed epoch's
snapshot equals the restored state). ``save_training_state`` persists that
snapshot instead of the live state.

Mirrors the harness style of ``resume_epoch_bookkeeping_test.py``: drives the
real ``BaseTrainer`` state helpers plus a miniature of the epoch loop's
snapshot-then-shuffle sequence (the exact code path the defect lived in),
with no model/dataset/GPU.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.base_trainer import BaseTrainer


class StateHarness:
    """Minimal stand-in exposing only the BaseTrainer methods under test."""

    save_training_state = BaseTrainer.save_training_state
    load_training_state = BaseTrainer.load_training_state

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.run_name = "20260101_000000_deadbeef"
        self.log_prefix = "[Test]"
        self._dataset_fingerprint = {"total_item_count": 400, "image_paths_hash": "x",
                                     "dataset_ids": ["a"]}
        self._batches_per_epoch = 100
        self._crop_plan_fingerprint = None
        # Deliberately absent until a "snapshot" is taken, mirroring a
        # trainer object that hasn't entered the epoch loop yet.


def _snapshot_then_shuffle(trainer, items):
    """Mirrors train()'s epoch-loop sequence: snapshot RNG state, THEN shuffle.

    This is the exact ordering added at the fix site (base_trainer.py, right
    before "# Create batches"): the snapshot captures the state that
    reproduces the permutation the shuffle is about to produce.
    """
    trainer._epoch_batch_rng_state = random.getstate()
    shuffled = list(items)
    random.shuffle(shuffled)
    return shuffled


def _legacy_checkpoint_random_state():
    """Pre-fix save_training_state read: the LIVE state at checkpoint time."""
    return random.getstate()


# ---------------------------------------------------------------------------
# (a) resume reproduces the interrupted epoch's permutation -- bucketed and
#     non-bucketed both funnel through plain random.shuffle(), so one harness
#     covers both real code paths.
# ---------------------------------------------------------------------------

def test_resume_reproduces_original_epoch_shuffle_order():
    items = list(range(50))
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp)

        random.seed(12345)
        full_order = _snapshot_then_shuffle(trainer, items)

        # Checkpoint mid-epoch, after some batches were consumed (each batch's
        # training step also draws from `random`, e.g. augmentation -- advance
        # the live stream to prove the fix does NOT rely on it).
        resume_batch_idx = 17
        for _ in range(resume_batch_idx):
            random.random()
        trainer.save_training_state(step=1000, epoch=0, batch_idx=resume_batch_idx)

        # Simulate a fresh process: unrelated RNG consumption before resume.
        random.seed(999)
        for _ in range(30):
            random.random()

        loaded = trainer.load_training_state(1000)
        random.setstate(loaded["random_state"])
        rebuilt_order = _snapshot_then_shuffle(trainer, items)

        assert rebuilt_order == full_order
        assert rebuilt_order[resume_batch_idx:] == full_order[resume_batch_idx:]


def test_legacy_live_state_does_not_reproduce_the_order():
    """Proves the test above is not vacuous: the pre-fix read really breaks it."""
    items = list(range(50))
    random.seed(12345)
    full_order = _snapshot_then_shuffle(StateHarness.__new__(StateHarness), items)

    resume_batch_idx = 17
    for _ in range(resume_batch_idx):
        random.random()
    legacy_state = _legacy_checkpoint_random_state()  # LIVE state, post-shuffle + post-draws

    random.seed(999)
    for _ in range(30):
        random.random()

    random.setstate(legacy_state)
    rebuilt_order = _snapshot_then_shuffle(StateHarness.__new__(StateHarness), items)

    assert rebuilt_order != full_order


# ---------------------------------------------------------------------------
# (b) old-format state (no snapshot ever taken) does not raise -- falls back
#     to the live state.
# ---------------------------------------------------------------------------

def test_missing_snapshot_attribute_falls_back_without_raising():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp)
        assert not hasattr(trainer, "_epoch_batch_rng_state")

        random.seed(1)
        live_state_before_save = random.getstate()
        trainer.save_training_state(step=1, epoch=0, batch_idx=0)
        loaded = trainer.load_training_state(1)

        assert loaded["random_state"] == live_state_before_save


# ---------------------------------------------------------------------------
# (c) after a mid-epoch resume, the NEXT checkpoint in that same (resumed)
#     epoch still saves the epoch's reproducing state, not one advanced by
#     further per-step draws.
# ---------------------------------------------------------------------------

def test_resumed_epoch_next_checkpoint_saves_the_epoch_snapshot():
    items = list(range(50))
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp)

        random.seed(42)
        full_order = _snapshot_then_shuffle(trainer, items)
        resume_batch_idx = 10
        trainer.save_training_state(step=500, epoch=0, batch_idx=resume_batch_idx)
        first_saved = trainer.load_training_state(500)

        # New session resumes: restore, THEN snapshot+rebuild (mirrors train()'s
        # restore-before-batch-build, capture-right-before-shuffle ordering).
        random.seed(7777)
        random.setstate(first_saved["random_state"])
        resumed_order = _snapshot_then_shuffle(trainer, items)
        assert resumed_order == full_order

        # More per-step random draws happen during this resumed epoch's training.
        for _ in range(23):
            random.random()

        # The next checkpoint, still inside the resumed epoch, must save the
        # state captured at the top of THIS epoch iteration (== the restored
        # state), not the live state advanced by the draws above.
        trainer.save_training_state(step=523, epoch=0, batch_idx=33)
        second_saved = trainer.load_training_state(523)

        assert second_saved["random_state"] == first_saved["random_state"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
