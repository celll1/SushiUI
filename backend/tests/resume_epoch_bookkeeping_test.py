"""Guard: epoch/batch bookkeeping must survive repeated mid-epoch resumes.

Why this file exists
--------------------
``training_metrics.epoch`` for run 112 stayed at 0 for ~70k steps across seven
resumes. Two mechanisms, only one of them a defect:

1. Not a defect: at resume_seq 8 the dataset changed (fingerprint + 28812 ->
   954880 batches/epoch), so the resume structure-change guard restarted epoch
   bookkeeping from 0 by design, and one epoch of the new dataset is 954,880
   batches -- 147k steps really is still inside epoch 0.

2. A defect: ``save_training_state(batch_idx=batch_idx + 1)`` saved an index
   into the batch list *as sliced for that session*. After a mid-epoch resume
   that list starts at ``resume_batch_idx``, so the saved position was relative
   to the resume point and the epoch's data cursor rewound on every restart --
   run 112's own state files go 17790 (@106000) -> 7349 (@113348) -> 16657
   (@130004). An epoch longer than a single session then never completes, so
   ``epoch`` can never advance and the same batches are retrained while others
   are skipped.

Everything here drives the real ``BaseTrainer`` state helpers (no model, no
dataset, no GPU) plus a miniature of the training loop's epoch/batch
bookkeeping: slice the batch list, ``enumerate`` it, save through
``_epoch_batch_position``. That is exactly the code path the defect lived in.
"""

from __future__ import annotations

import os
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
    _epoch_batch_position = BaseTrainer._epoch_batch_position
    _resolve_start_epoch = BaseTrainer._resolve_start_epoch

    def __init__(self, output_dir, batches_per_epoch=100):
        self.output_dir = Path(output_dir)
        self.run_name = "20260101_000000_deadbeef"
        self.log_prefix = "[Test]"
        self._dataset_fingerprint = {"total_item_count": 400, "image_paths_hash": "x",
                                     "dataset_ids": ["a"]}
        self._batches_per_epoch = batches_per_epoch
        self._crop_plan_fingerprint = None
        self._epoch_batch_offset = 0
        self.crop_planner = None
        self._crop_step_offsets = None


def _legacy_position(trainer, batch_idx):
    """The pre-fix formula, kept so the regression tests can prove they bite."""
    return batch_idx + 1


def _run_session(trainer, *, start_epoch, resume_batch_idx, global_step, has_state,
                 num_epochs, batches_per_epoch, session_steps, position_fn):
    """Miniature of BaseTrainer.train()'s epoch/batch bookkeeping.

    Returns the state dict saved when the session stops (as
    ``save_training_state`` would write it), or None if training finished.
    """
    steps_this_session = 0
    for epoch in range(start_epoch, num_epochs):
        trainer._epoch_batch_offset = 0
        batches = list(range(batches_per_epoch))
        if epoch == start_epoch and has_state:
            batches = batches[resume_batch_idx:]
            trainer._epoch_batch_offset = resume_batch_idx
            has_state = False
        for batch_idx, _ in enumerate(batches):
            global_step += 1
            steps_this_session += 1
            if steps_this_session >= session_steps:
                trainer.save_training_state(
                    step=global_step, epoch=epoch,
                    batch_idx=position_fn(trainer, batch_idx))
                return {"global_step": global_step, "epoch": epoch,
                        "batch_idx": position_fn(trainer, batch_idx)}
    return None


def _resume_chain(position_fn, *, batches_per_epoch=100, session_steps=30,
                  sessions=5, num_epochs=3):
    """Stop/resume ``sessions`` times, returning the saved state after each."""
    saved = []
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp, batches_per_epoch=batches_per_epoch)
        start_epoch, resume_batch_idx, global_step, has_state = 0, 0, 0, False
        for _ in range(sessions):
            state = _run_session(
                trainer, start_epoch=start_epoch, resume_batch_idx=resume_batch_idx,
                global_step=global_step, has_state=has_state, num_epochs=num_epochs,
                batches_per_epoch=batches_per_epoch, session_steps=session_steps,
                position_fn=position_fn)
            if state is None:
                break
            saved.append(state)
            reloaded = trainer.load_training_state(state["global_step"])
            start_epoch = trainer._resolve_start_epoch(
                reloaded, reloaded["global_step"], batches_per_epoch)
            resume_batch_idx = reloaded["batch_idx"]
            global_step = reloaded["global_step"]
            has_state = True
    return saved


# ---------------------------------------------------------------------------
# start_epoch restoration
# ---------------------------------------------------------------------------

def test_start_epoch_restored_from_training_state():
    """Epoch-count run: resuming a state saved in epoch N restarts at epoch N."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp)
        trainer.save_training_state(step=1234, epoch=7, batch_idx=42)
        state = trainer.load_training_state(1234)
        assert state["epoch"] == 7
        assert trainer._resolve_start_epoch(state, 1234, steps_per_epoch=100) == 7


def test_start_epoch_derived_from_global_step_without_state():
    """No state sidecar: fall back to epoch-level resume from global_step."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp)
        assert trainer.load_training_state(999) is None
        assert trainer._resolve_start_epoch(None, 350, steps_per_epoch=100) == 3
        # steps_per_epoch is never trusted to be non-zero here.
        assert trainer._resolve_start_epoch(None, 350, steps_per_epoch=0) == 350


def test_start_epoch_uses_crop_planner_when_steps_per_epoch_is_variable():
    class _Planner:
        def epoch_for_step(self, step, mnt):
            return 11

    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp)
        trainer.crop_planner = _Planner()
        trainer._crop_step_offsets = [0, 10]
        assert trainer._resolve_start_epoch(None, 350, steps_per_epoch=100) == 11
        # An explicit state still wins over the derivation.
        assert trainer._resolve_start_epoch({"epoch": 4, "batch_idx": 0}, 350, 100) == 4


def test_step_specified_run_records_an_integer_epoch():
    """A ``steps``-only run still has a well-defined dataset-pass counter.

    num_epochs = ceil(total_steps / steps_per_epoch), so epoch is the pass index
    and 0 is the truthful value inside the first pass -- it is not a placeholder
    for "unknown" and must not become NULL.
    """
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp, batches_per_epoch=954880)
        trainer.save_training_state(step=147641, epoch=0, batch_idx=17638)
        state = trainer.load_training_state(147641)
        assert state["epoch"] == 0
        assert isinstance(state["epoch"], int)
        assert trainer._resolve_start_epoch(state, 147641, 954880) == 0


# ---------------------------------------------------------------------------
# epoch position must not rewind across resumes
# ---------------------------------------------------------------------------

def test_epoch_position_does_not_rewind_across_resumes():
    saved = _resume_chain(StateHarness._epoch_batch_position)
    positions = [(s["epoch"], s["batch_idx"]) for s in saved]
    assert positions == sorted(positions), positions
    assert positions[:3] == [(0, 30), (0, 60), (0, 90)]


def test_legacy_position_rewinds_the_epoch_forever():
    """The pre-fix formula: proves the tests above are not vacuous."""
    saved = _resume_chain(_legacy_position)
    positions = [(s["epoch"], s["batch_idx"]) for s in saved]
    assert positions[:3] == [(0, 30), (0, 30), (0, 30)]
    assert max(e for e, _ in positions) == 0


def test_epoch_advances_when_the_resumed_epoch_completes():
    saved = _resume_chain(StateHarness._epoch_batch_position,
                          batches_per_epoch=100, session_steps=40, sessions=4)
    assert [s["epoch"] for s in saved] == [0, 0, 1, 1]
    # global_step is driven purely by batches consumed -- unchanged by the fix.
    assert [s["global_step"] for s in saved] == [40, 80, 120, 160]


def test_offset_applies_only_to_the_resumed_epoch():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp)
        trainer._epoch_batch_offset = 90
        assert trainer._epoch_batch_position(4) == 95
        trainer._epoch_batch_offset = 0  # next epoch's batch list is not sliced
        assert trainer._epoch_batch_position(4) == 5


def test_run112_state_sequence_is_no_longer_reproducible():
    """Reproduces the run-112 numbers: 17790 -> 7349 was a 10441-batch rewind."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = StateHarness(tmp, batches_per_epoch=954880)
        trainer._epoch_batch_offset = 17790     # resumed at batch 17790 (step 106000)
        batch_idx = 7348                        # batches consumed by that session
        assert _legacy_position(trainer, batch_idx) == 7349    # observed @113348
        assert trainer._epoch_batch_position(batch_idx) == 25139


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
