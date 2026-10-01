"""TaggerTrainerHandle.service_pause: the batch-boundary pause protocol shared by the tagger and style trainers."""

from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.gpu_coordinator import OffloadDecision
from core.tagger.tagger_offload import TaggerTrainerHandle


def _handle():
    handle = TaggerTrainerHandle(run_id="run", output_dir=".", pause_event=threading.Event(),
                                 resumed_event=threading.Event(), restored_event=threading.Event())
    calls = []
    handle.offload = lambda decision: calls.append(("offload", decision.mode))
    handle.restore = lambda: calls.append(("restore",))
    return handle, calls


def _serve(handle, stop_event, stop_flag):
    thread = threading.Thread(target=handle.service_pause, args=(stop_event, lambda: stop_flag[0]))
    thread.start()
    deadline = time.monotonic() + 5
    while not handle.resumed_event.is_set():
        assert time.monotonic() < deadline and thread.is_alive()
        time.sleep(0.01)
    return thread


def test_no_pause_returns_at_once():
    handle, calls = _handle()
    handle.service_pause(threading.Event(), lambda: False)
    assert calls == [] and not handle.resumed_event.is_set() and not handle.restored_event.is_set()


def test_decision_is_offloaded_and_restored_when_the_pause_ends():
    handle, calls = _handle()
    handle.pending_decision = OffloadDecision("none")
    handle.pause_event.set()
    thread = _serve(handle, threading.Event(), [False])
    assert calls == [("offload", "none")] and thread.is_alive()
    handle.pause_event.clear()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert calls == [("offload", "none"), ("restore",)]
    assert not handle.resumed_event.is_set() and handle.restored_event.is_set()
    assert handle.restored_event.wait(timeout=0)
    assert handle.pending_decision is None


def test_without_a_decision_only_the_events_toggle():
    handle, calls = _handle()
    handle.pause_event.set()
    thread = _serve(handle, threading.Event(), [False])
    handle.pause_event.clear()
    thread.join(timeout=5)
    assert calls == [] and not handle.resumed_event.is_set()


def test_stop_during_the_wait_still_restores():
    handle, calls = _handle()
    handle.pending_decision = OffloadDecision("dram")
    handle.pause_event.set()
    stop_event, stop_flag = threading.Event(), [False]
    thread = _serve(handle, stop_event, stop_flag)
    stop_flag[0] = True
    stop_event.set()
    thread.join(timeout=5)
    assert not thread.is_alive() and handle.pause_event.is_set()
    assert calls == [("offload", "dram"), ("restore",)] and handle.pending_decision is None
