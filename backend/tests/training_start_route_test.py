"""Route-level contract of POST /training/runs/{id}/start and /stop.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_start_route_test.py -q

The handler is called directly with a fake DB session — no TestClient, no
lifespan, no model. Covers the three things the release commit added at the
route layer and that no test reached before: the pre-training VRAM release
happening AT ALL and under the lifecycle gate, the live-child refusal, and the
stop escape hatch the refusal points the user at.
"""

import asyncio
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import api.routes as routes
from core.model_state_coordinator import ModelStateBusyError, model_state_coordinator
from core.training.training_process import TrainingProcessManager
from fastapi import HTTPException


class _FakeQuery:
    def __init__(self, run):
        self._run = run

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._run


class _FakeDb:
    def __init__(self, run):
        self.run = run
        self.commits = 0

    def query(self, _model):
        return _FakeQuery(self.run)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _make_run(tmp_path, status="created"):
    output_dir = str(tmp_path)
    run = types.SimpleNamespace(
        id=1,
        run_name="run",
        status=status,
        error_message=None,
        started_at=None,
        last_resumed_at=None,
        resumed_from_step=None,
        current_step=0,
        warnings=[],
        output_dir=output_dir,
        config_yaml="",
        dataset_configs=[],
        dataset_id=None,
    )
    run.to_dict = lambda: {"id": run.id, "status": run.status, "warnings": run.warnings}
    with open(os.path.join(output_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        f.write("config: {}\n")
    return run


class _FakeProcess:
    def __init__(self):
        self.process = None
        self.is_running = False
        self.started = False

    async def start(self, **kwargs):
        self.started = True
        self.process = types.SimpleNamespace(returncode=None, pid=1234)
        self.is_running = True

    async def stop(self):
        self.is_running = False


def _patch_common(monkeypatch, tmp_path, release=None):
    """Isolate the handler: fresh process registry, no WebSocket, recorded release."""
    manager = TrainingProcessManager()
    created = {}

    def _create_process(run_id, config_path, output_dir):
        if manager.is_live(run_id):
            raise RuntimeError("live")
        proc = _FakeProcess()
        manager.processes[run_id] = proc
        created["process"] = proc
        return proc

    monkeypatch.setattr(manager, "create_process", _create_process)
    monkeypatch.setattr(routes, "training_process_manager", manager)
    monkeypatch.setattr(routes.manager, "send_training_log", lambda **kwargs: None)

    calls = []

    def _release(reason=""):
        if release is not None:
            return release(reason)
        calls.append({"reason": reason,
                      "mutation": model_state_coordinator.snapshot()["mutation"],
                      "started": created.get("process") is not None
                      and created["process"].started})
        return {"components": ["unet"], "freed_bytes": 1024,
                "keep_hot_cleared": [], "auxiliary": []}

    fake_pipeline_module = types.SimpleNamespace(
        pipeline_manager=types.SimpleNamespace(release_gpu_memory=_release))
    monkeypatch.setitem(sys.modules, "core.pipeline", fake_pipeline_module)
    return manager, calls, created


def test_start_releases_backend_vram_before_spawning(monkeypatch, tmp_path):
    """MUTANT: deleting the release block from start_training_run. The backend is
    the process that holds the generation VRAM; the trainer child cannot free it,
    so nothing else in the system does."""
    manager, calls, created = _patch_common(monkeypatch, tmp_path)
    run = _make_run(tmp_path)

    result = asyncio.run(routes.start_training_run(run_id=1, db=_FakeDb(run)))

    assert len(calls) == 1
    assert "training run 1 start" in calls[0]["reason"]
    # Released BEFORE the child was spawned, not after.
    assert calls[0]["started"] is False
    assert created["process"].started is True
    assert result["message"] == "Training started"


def test_the_lifecycle_gate_is_released_after_the_start(monkeypatch, tmp_path):
    """A held gate would refuse every later generation and model load."""
    _patch_common(monkeypatch, tmp_path)
    run = _make_run(tmp_path)

    asyncio.run(routes.start_training_run(run_id=1, db=_FakeDb(run)))

    assert model_state_coordinator.snapshot()["mutation"] is None


def test_release_holds_the_gate_while_it_runs(monkeypatch, tmp_path):
    """MUTANT: calling release_gpu_memory() with no model-state gate. Moving the
    U-Net to CPU under a running denoise kills that generation; a generation that
    survives re-stages and re-marks keep-hot in its own finally, putting the
    freed GiB straight back while the release log claims success."""
    seen = {}

    def _release(reason=""):
        seen["mutation"] = model_state_coordinator.snapshot()["mutation"]
        return {"components": [], "freed_bytes": 0, "keep_hot_cleared": [], "auxiliary": []}

    _patch_common(monkeypatch, tmp_path, release=_release)
    run = _make_run(tmp_path)
    asyncio.run(routes.start_training_run(run_id=1, db=_FakeDb(run)))
    assert seen["mutation"] == "training run 1 start"


def test_a_blocked_gate_warns_but_still_starts_the_run(monkeypatch, tmp_path):
    """MUTANT: letting ModelStateBusyError abort the start (a 409). The release
    is a VRAM optimization, not a precondition — the user asked for the run to
    start. It must surface through the SAME warning path as any other release
    failure."""
    def _release(reason=""):
        raise ModelStateBusyError("Cannot start x; blocked by: 1 generation request.")

    manager, _calls, created = _patch_common(monkeypatch, tmp_path, release=_release)
    logged = []
    monkeypatch.setattr(routes.manager, "send_training_log",
                        lambda **kwargs: logged.append(kwargs))
    run = _make_run(tmp_path)

    result = asyncio.run(routes.start_training_run(run_id=1, db=_FakeDb(run)))

    assert result["message"] == "Training started"
    assert created["process"].started is True
    assert any(w.get("code") == "pre_training_vram_release_failed" for w in run.warnings)
    assert any(entry.get("code") == "pre_training_vram_release_failed" for entry in logged)


def test_start_refuses_a_run_with_a_live_child(monkeypatch, tmp_path):
    """MUTANT: deleting the 409/reap block. The DB status is not liveness; a
    second spawn overwrites the registry entry and orphans the first child."""
    manager, _calls, _created = _patch_common(monkeypatch, tmp_path)
    manager.processes[1] = types.SimpleNamespace(
        process=types.SimpleNamespace(returncode=None, pid=9), is_running=True)
    run = _make_run(tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(routes.start_training_run(run_id=1, db=_FakeDb(run)))
    assert excinfo.value.status_code == 409


def test_start_refuses_a_registered_but_unspawned_entry(monkeypatch, tmp_path):
    """MUTANT: `is_live` returning False for an entry registered but not yet
    spawned. create_process and the spawn are seconds apart (pre-flight, the VRAM
    release); request B arriving in that window used to reap request A's entry
    and spawn its own — two trainers on one GPU, A's child unstoppable."""
    manager, _calls, _created = _patch_common(monkeypatch, tmp_path)
    manager.processes[1] = types.SimpleNamespace(process=None, is_running=False)
    run = _make_run(tmp_path, status="starting")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(routes.start_training_run(run_id=1, db=_FakeDb(run)))
    assert excinfo.value.status_code == 409
    # The other request's entry must still be there, NOT reaped.
    assert 1 in manager.processes


def test_a_failed_start_removes_its_own_unspawned_entry(monkeypatch, tmp_path):
    """Corollary of the rule above: an entry that reads as live forever would
    make the run permanently unstartable, which is worse than what it replaced.
    The request that registered it owns its removal."""
    manager, _calls, created = _patch_common(monkeypatch, tmp_path)

    async def _boom(**kwargs):
        raise RuntimeError("spawn failed")

    run = _make_run(tmp_path)
    db = _FakeDb(run)

    original = manager.create_process

    def _create_and_break(run_id, config_path, output_dir):
        proc = original(run_id, config_path, output_dir)
        proc.start = _boom
        return proc

    monkeypatch.setattr(manager, "create_process", _create_and_break)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(routes.start_training_run(run_id=1, db=db))
    assert excinfo.value.status_code == 500
    assert 1 not in manager.processes
    assert manager.is_live(1) is False


# ------------------------------------------------------------------ stop


def test_stop_is_allowed_whenever_a_child_is_live(monkeypatch, tmp_path):
    """MUTANT: gating stop on run.status alone. /start's 409 tells the user to
    stop the run first; in exactly that state (status 'failed' with a live child)
    the old status check answered 400, leaving the run unstartable AND
    unstoppable until a backend restart."""
    manager, _calls, _created = _patch_common(monkeypatch, tmp_path)
    child = _FakeProcess()
    child.process = types.SimpleNamespace(returncode=None, pid=11)
    child.is_running = True
    manager.processes[1] = child
    run = _make_run(tmp_path, status="failed")

    result = asyncio.run(routes.stop_training_run(run_id=1, db=_FakeDb(run)))

    assert result["message"] == "Training stopped"
    assert run.status == "stopped"
    assert 1 not in manager.processes


def test_stop_still_refuses_a_dead_run(monkeypatch, tmp_path):
    manager, _calls, _created = _patch_common(monkeypatch, tmp_path)
    run = _make_run(tmp_path, status="failed")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(routes.stop_training_run(run_id=1, db=_FakeDb(run)))
    assert excinfo.value.status_code == 400


# ---------------------------------------------------- double-start refusal


class _FakeChild:
    def __init__(self, returncode=None, pid=4242):
        self.returncode = returncode
        self.pid = pid


def test_create_process_refuses_to_overwrite_a_live_process():
    """MUTANT: `self.processes[run_id] = process` unconditionally. Two
    train_runner children for one run orphan the first -- the registry entry
    that could stop it is gone."""
    manager = TrainingProcessManager()
    existing = types.SimpleNamespace(process=_FakeChild(returncode=None), is_running=True)
    manager.processes[7] = existing

    assert manager.is_live(7) is True
    with pytest.raises(RuntimeError) as excinfo:
        manager.create_process(run_id=7, config_path="c.yaml", output_dir="out")
    assert "already has a live training process" in str(excinfo.value)
    assert manager.processes[7] is existing


def test_is_live_is_false_only_for_an_exited_or_absent_process():
    """MUTANT: reading `is_running` instead of the child's returncode. The flag
    is cleared only once the monitor task observes the exit, so it is stale
    exactly during the window a restart is attempted."""
    manager = TrainingProcessManager()
    manager.processes[1] = types.SimpleNamespace(process=_FakeChild(returncode=0), is_running=True)
    assert manager.is_live(1) is False
    assert manager.is_live(99) is False


def test_a_registered_but_unspawned_entry_counts_as_live():
    """MUTANT: `process.process is None -> not live` (what this file used to
    assert as intended). create_process and the spawn are seconds apart --
    pre-flight rescan, the pre-training VRAM release -- and in that window the
    route's reap branch deleted the OTHER request's entry and spawned a second
    trainer on the same GPU, orphaning the first child with no registry entry
    left to stop it: verbatim the failure the double-start guard prevents."""
    manager = TrainingProcessManager()
    manager.processes[2] = types.SimpleNamespace(process=None, is_running=False)
    assert manager.is_live(2) is True
    with pytest.raises(RuntimeError):
        manager.create_process(run_id=2, config_path="c.yaml", output_dir="out")
