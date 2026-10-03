"""Pause acknowledgements outlive the trainer's boundary call."""

from types import SimpleNamespace
import threading

import pytest

from core.tagger.foveatag.run_control import FoveaRunControl


def test_catalog_identity_requires_output_directory():
    with pytest.raises(ValueError, match="together"):
        FoveaRunControl(run_id="test")


@pytest.mark.parametrize("stop_before_resume", [False, True])
def test_pause_restore_waits_for_gpu_release_and_retains_ack(stop_before_resume):
    control = FoveaRunControl()
    pause = threading.Event()
    pause.set()
    restored = threading.Event()
    calls = []

    def restore():
        assert not pause.is_set()
        calls.append("restore")

    handle = SimpleNamespace(
        pause_event=pause, resumed_event=threading.Event(), restored_event=restored,
        pending_decision="test", offload=lambda decision: calls.append("offload"),
        restore=restore,
    )
    control._handle = handle
    worker = threading.Thread(target=control.batch_boundary)
    worker.start()
    try:
        assert handle.resumed_event.wait(2)
        if stop_before_resume:
            control.request_stop()
        assert not restored.is_set()
        assert calls == ["offload"]
        pause.clear()
        assert restored.wait(2)
    finally:
        pause.clear()
        worker.join(2)
    assert not worker.is_alive()
    assert handle.resumed_event.is_set()
    assert restored.is_set()
    assert handle.pending_decision is None
    assert calls == ["offload", "restore"]
