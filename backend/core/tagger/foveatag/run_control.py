"""Thread-safe lifecycle controls shared by FoveaTag training stages."""

from __future__ import annotations

import threading


class FoveaRunControl:
    """Request a checkpointed stop from a backend-owned training thread."""

    def __init__(self, *, run_id: str | None = None, output_dir: str | None = None) -> None:
        if (run_id is None) != (output_dir is None):
            raise ValueError("run_id and output_dir must be supplied together")
        self._stop = threading.Event()
        self.run_id = run_id
        self.output_dir = output_dir
        self._handle = None

    def request_stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def attach(self, model, optimizer, criterion) -> None:
        if self.run_id is None:
            return
        if self._handle is not None:
            raise RuntimeError("training handle is already attached")
        from core.gpu_coordinator import gpu_coordinator
        from core.tagger.tagger_offload import TaggerTrainerHandle
        handle = TaggerTrainerHandle(self.run_id, self.output_dir,
            threading.Event(), threading.Event(), threading.Event())
        handle.attach(model, optimizer, criterion)
        gpu_coordinator.register_trainer(handle)
        self._handle = handle

    def batch_boundary(self) -> None:
        wait_for_gpu_release(self._handle)

    def detach(self) -> None:
        if self._handle is None:
            return
        from core.gpu_coordinator import gpu_coordinator
        gpu_coordinator.unregister_trainer(self._handle)
        self._handle.detach()
        self._handle = None


def wait_for_gpu_release(handle) -> None:
    if handle is None or not handle.pause_event.is_set():
        return
    decision = handle.pending_decision
    if decision is not None:
        handle.offload(decision)
    handle.resumed_event.set()
    # A stop must not restore optimizer VRAM while generation owns it.
    pause_wait = threading.Event()
    while handle.pause_event.is_set():
        pause_wait.wait(0.5)
    if decision is not None:
        handle.restore()
    handle.pending_decision = None
    # The coordinator clears acknowledgements at the next pause cycle.
    handle.restored_event.set()
