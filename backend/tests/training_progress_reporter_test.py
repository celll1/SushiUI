from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.train_runner import (  # noqa: E402
    TrainingProgressReporter,
    _apply_training_progress,
    make_training_progress_callback,
)


def test_apply_training_progress_preserves_training_fields():
    run = SimpleNamespace(loss=None, learning_rate=None)

    _apply_training_progress(run, "training", 4, 10, 2, 0.25, 1e-4, None)

    assert run.phase == "training"
    assert run.phase_progress == 40.0
    assert run.phase_detail == "Epoch 2, Step 4/10"
    assert run.current_step == 4
    assert run.loss == 0.25
    assert run.learning_rate == 1e-4
    assert run.progress == 40.0


def test_reporter_coalesces_pending_snapshots_and_flushes_latest():
    run = SimpleNamespace(loss=None, learning_rate=None)
    first_commit_started = threading.Event()
    release_first_commit = threading.Event()
    commits = []

    class Session:
        def get(self, _model, _run_id):
            return run

        def commit(self):
            commits.append(run.current_step)
            if len(commits) == 1:
                first_commit_started.set()
                assert release_first_commit.wait(timeout=2)

        def rollback(self):
            raise AssertionError("rollback should not be needed")

        def close(self):
            pass

    reporter = TrainingProgressReporter(7, session_factory=Session)
    try:
        reporter.publish("training", 1, 10)
        assert first_commit_started.wait(timeout=2)
        reporter.publish("training", 2, 10)
        reporter.publish("training", 3, 10)
        release_first_commit.set()
        reporter.flush()
        assert commits == [1, 3]
        assert run.current_step == 3
    finally:
        release_first_commit.set()
        reporter.close()


def test_callback_publishes_without_touching_a_database():
    calls = []
    reporter = SimpleNamespace(publish=lambda *args: calls.append(args))
    trainer = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": 2e-4}]))
    callback = make_training_progress_callback(reporter, trainer)

    callback("training", 5, 20, epoch=1, loss=0.5, detail="ignored")

    assert calls == [("training", 5, 20, 1, 0.5, 2e-4, "ignored")]
