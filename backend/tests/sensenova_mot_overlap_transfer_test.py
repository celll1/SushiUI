"""``sensenova_mot_overlap_transfer``: run a phase swap's two directions
concurrently on two CUDA streams (see ``sensenova_phase_eviction``'s OVERLAPPED
TRANSFER note). No CUDA is touched anywhere in this file -- the streams and
events are injected fakes, exactly as ``sensenova_mot_pageable_staging_test``
patches the two transfer primitives.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_mot_overlap_transfer_test.py -v
"""

import contextlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.arch_capabilities import (
    TRAINING_DECLARED_ARCHS,
    TRAINING_FEATURE_PARAMS,
    training_feature_unsupported_reason,
)
from api.param_defaults import TRAINING_DEFAULTS
from core.models.sensenova.mot_cpu_staging import _stage_tensor
from core.training import sensenova_phase_eviction
from core.training.sensenova_phase_eviction import (
    _OVERLAP_WINDOW_PAIRS,
    SenseNovaTrainingPhaseEvictor,
    _TransferStreams,
    _move_modules_to_device,
    install_training_phase_eviction,
)
from core.training.train_runner import _apply_sensenova_training_contract

from sensenova_training_phase_eviction_test import transformer  # noqa: E402


# ---------------------------------------------------------------------------
# fakes: streams, events, and the two transfer primitives
# ---------------------------------------------------------------------------


class _Source:
    """Stands in for a CUDA tensor the d2h copy reads from."""

    is_cuda = True

    def __init__(self, log):
        self._log = log
        self.recorded = []

    def record_stream(self, stream):
        self.recorded.append(stream)
        self._log.append(("record", stream.name))


class _Event:
    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def synchronize(self):
        self._log.append(("retire", self._stream.name))

    def elapsed_time(self, other):
        del other
        return 2.0   # ms


class _Stream:
    def __init__(self, name):
        self.name = name


class _Streams:
    def __init__(self, log):
        self._log = log
        self.d2h = _Stream("d2h")
        self.h2d = _Stream("h2d")

    def stream_for(self, operation):
        return self.d2h if operation == "d2h" else self.h2d

    def stream_context(self, stream):
        @contextlib.contextmanager
        def _entered():
            self._log.append(("enter", stream.name))
            yield
            self._log.append(("exit", stream.name))

        return _entered()

    def record_event(self, stream):
        return _Event(stream, self._log)

    def join(self):
        self._log.append(("join", None))


class _Harness:
    """Records what the evictor issues, without moving a byte."""

    def __init__(self, *, fail_pin_at=None):
        self.log = []
        self.calls = []
        self.streams = _Streams(self.log)
        self._fail_pin_at = fail_pin_at
        self.d2h_sources = []
        self.h2d_sources = []
        self.h2d_seams = []

    def factory(self, device):
        del device
        return self.streams

    def d2h(self, modules, *, warn_once, pageable=False, non_blocking=False,
            sources=None):
        del modules
        self.calls.append(("d2h", non_blocking, pageable))
        self.log.append(("issue", "d2h"))
        if non_blocking:
            source = _Source(self.log)
            self.d2h_sources.append(source)
            sources.append(source)
        if self._fail_pin_at is not None and len(self.calls) >= self._fail_pin_at:
            warn_once["pin_failed"] = True

    def h2d(self, modules, device, *, non_blocking=False, sources=None,
            streams=None, stream=None):
        del modules, device
        self.calls.append(("h2d", non_blocking, None))
        self.log.append(("issue", "h2d"))
        self.h2d_seams.append((streams, stream))
        if non_blocking:
            source = object()
            self.h2d_sources.append(source)
            sources.append(source)

    def patched(self):
        return patch.multiple(
            sensenova_phase_eviction,
            _move_modules_to_cpu=self.d2h,
            _move_modules_to_device=self.h2d,
        )


def _evictor(harness=None, **kwargs):
    return SenseNovaTrainingPhaseEvictor(
        transformer(), "meta", overlap_transfer=True,
        streams_factory=(harness.factory if harness else None), **kwargs
    )


# ---------------------------------------------------------------------------
# the regression guard: off, and on a device that cannot overlap
# ---------------------------------------------------------------------------


def _order(evictor):
    events = []
    gen_ids = {id(module) for module in evictor._gen_modules}

    def d2h(modules, *, warn_once, pageable=False):
        del warn_once, pageable
        events.append(("d2h", "gen" if id(tuple(modules)[0]) in gen_ids else "und"))

    def h2d(modules, device):
        del device
        events.append(("h2d", "gen" if id(tuple(modules)[0]) in gen_ids else "und"))

    with patch.multiple(
        sensenova_phase_eviction,
        _move_modules_to_cpu=d2h, _move_modules_to_device=h2d,
    ):
        evictor.enter_prefix()
        evictor.enter_denoise()
        evictor.enter_prefix()
    return events


_SHIPPED_ORDER = (
    [("d2h", "gen")] * 42
    + [("d2h", "und"), ("h2d", "gen")] * 42
    + [("d2h", "gen"), ("h2d", "und")] * 42
)


def test_overlap_off_reproduces_the_shipped_operation_order_exactly():
    """MUTANT: routing the default path through the overlap runner (or changing
    the serial loop's signature) shows up here as a different op sequence."""
    assert _order(SenseNovaTrainingPhaseEvictor(transformer(), "meta")) == _SHIPPED_ORDER


def test_overlap_on_a_non_cuda_device_falls_back_to_the_same_order(capsys):
    """MUTANT: a flag that assumes CUDA makes a CPU/meta evictor crash rather
    than no-op.

    The real ``_make_transfer_streams`` runs here, but it returns None on a
    non-CUDA device BEFORE constructing a ``_TransferStreams``, so this covers
    that guard and not the stream object -- see
    ``test_the_transfer_streams_object_routes_and_joins_both_streams`` for that.

    MUTANT: latching the downgrade silently (which is what shipped) leaves a
    user who ticked the box with no line saying it did nothing.
    """
    evictor = SenseNovaTrainingPhaseEvictor(
        transformer(), "meta", overlap_transfer=True
    )
    assert _order(evictor) == _SHIPPED_ORDER
    assert evictor.overlap_active is False
    assert evictor._streams is None
    # Said once, though three transitions asked for the streams.
    assert capsys.readouterr().out.count("inert on this run") == 1


# ---------------------------------------------------------------------------
# mutual exclusion with pageable staging: refused, not degraded
# ---------------------------------------------------------------------------


def test_install_refuses_overlap_together_with_pageable_staging():
    """MUTANT: silently dropping one of the two flags. Both names must appear,
    since either one is a legitimate thing for the caller to turn off."""
    trainer = type("T", (), {})()
    trainer.transformer = transformer()
    trainer.device = "meta"
    trainer.sensenova_four_phase_eviction = False
    trainer.config = {
        "sensenova_mot_pageable_staging": True,
        "sensenova_mot_overlap_transfer": True,
    }

    with pytest.raises(ValueError) as excinfo:
        install_training_phase_eviction(trainer)
    message = str(excinfo.value)
    assert "sensenova_mot_overlap_transfer" in message
    assert "sensenova_mot_pageable_staging" in message
    assert "bounce buffer" in message


def test_the_constructor_backstops_the_same_pair():
    with pytest.raises(ValueError, match="sensenova_mot_pageable_staging"):
        SenseNovaTrainingPhaseEvictor(
            transformer(), "meta", pageable_staging=True, overlap_transfer=True
        )


def test_the_staging_primitive_refuses_pageable_plus_non_blocking():
    """The reason for the refusal, at the layer that would suffer it: an async
    copy out of pageable memory is bounce-buffered and host-synchronous."""
    with pytest.raises(ValueError, match="host-synchronous"):
        _stage_tensor(torch.ones(4), {}, "unused", pageable=True, non_blocking=True)


def test_install_reads_the_flag_off_trainer_config():
    trainer = type("T", (), {})()
    trainer.transformer = transformer()
    trainer.device = "meta"
    trainer.sensenova_four_phase_eviction = False
    trainer.config = {"sensenova_mot_overlap_transfer": True}

    assert install_training_phase_eviction(trainer)._overlap is True


def test_install_defaults_to_false_with_no_config_attribute_at_all():
    trainer = type("T", (), {})()
    trainer.transformer = transformer()
    trainer.device = "meta"
    trainer.sensenova_four_phase_eviction = False

    assert install_training_phase_eviction(trainer)._overlap is False


# ---------------------------------------------------------------------------
# the overlapped path itself
# ---------------------------------------------------------------------------


def test_at_most_k_incoming_modules_are_admitted_ahead_of_their_twins():
    """The whole cost of the relaxation. An incoming module is "ahead" from the
    moment its h2d is issued until its outgoing twin's d2h has been WAITED on
    (a queued copy has not freed anything).

    MUTANT: dropping the window cap -- issuing the whole plan and joining once
    at the end -- drives this to one full half.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()
        evictor.enter_denoise()

    ahead = peak = 0
    for kind, direction in harness.log:
        if kind == "join":
            ahead = 0        # the window is drained; nothing is in flight
        elif kind == "issue" and direction == "h2d":
            ahead += 1
        elif kind == "retire" and direction == "d2h":
            ahead -= 1
        peak = max(peak, ahead)

    assert peak == _OVERLAP_WINDOW_PAIRS == 4


def test_both_streams_are_joined_before_the_transition_returns():
    """``assert_generation_resident`` checks DEVICE PLACEMENT only, so it would
    pass on a queued-but-unfinished copy.

    MUTANT: returning without the join, or without draining the window, leaves
    issues unmatched by retires and no join at the tail.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()
        assert harness.log[-1] == ("join", None)
        first = list(harness.log)
        evictor.enter_denoise()

    for log in (first, harness.log):
        assert log[-1] == ("join", None)
        assert sum(1 for kind, _ in log if kind == "issue") == \
            sum(1 for kind, _ in log if kind == "retire")


def test_every_d2h_source_is_record_streamed_before_the_next_leg_is_issued():
    """``stage_modules_to_pinned_cpu`` reassigns ``parameter.data`` the moment
    the copy is issued, so the model's reference is already gone; what keeps the
    block alive across that is the ``sources`` list, and ``record_stream`` has to
    land before that list is released and before any later leg can be handed the
    block.

    MUTANT: omitting ``record_stream`` lets the caching allocator hand that
    block to the concurrent h2d destination -- silent corruption.
    MUTANT: hoisting the loop to the end of the plan (record everything once,
    after the whole swap) breaks the adjacency below while still populating
    ``recorded``.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert harness.d2h_sources
    assert all(source.recorded == [harness.streams.d2h]
               for source in harness.d2h_sources)
    # Each d2h leg: enter the stream, issue, leave the stream, record. Nothing
    # else may come between the issue and its record.
    issues = [i for i, entry in enumerate(harness.log) if entry == ("issue", "d2h")]
    assert issues
    for index in issues:
        assert harness.log[index + 1] == ("exit", "d2h")
        assert harness.log[index + 2] == ("record", "d2h")


def test_the_h2d_leg_is_handed_the_stream_seam_instead_of_being_wrapped_in_it():
    """The destination is DEVICE memory, and torch's caching allocator
    partitions free blocks by owning stream: a destination allocated inside the
    side stream's context could never be handed a block the outgoing half just
    freed on the default stream, so every incoming module would cudaMalloc fresh
    and the window bound would be a whole half rather than four modules.
    ``_move_modules_to_device`` therefore allocates outside and enters the
    context for the copies alone.

    MUTANT: wrapping the h2d leg in ``streams.stream_context`` at the runner (as
    shipped) puts an enter/exit around the issue here.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert ("enter", "h2d") not in harness.log
    assert harness.h2d_seams
    assert all(streams is harness.streams and stream is harness.streams.h2d
               for streams, stream in harness.h2d_seams)


def test_the_h2d_pinned_source_is_held_until_its_own_event_is_waited_on():
    """Symmetric hazard: ``_move_modules_to_device`` drops the pinned source
    into torch's caching HOST allocator, which the next ``_stage_tensor`` may
    re-hand out. Torch may event-guard this; that is not verified here.

    MUTANT: dropping the keepalive at issue time empties every h2d entry below.
    MUTANT: releasing it and only then waiting -- the named property is UNTIL
    the event is waited on, so ``held_at_sync`` is measured inside
    ``synchronize`` rather than around ``_retire``.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    seen = []
    original = SenseNovaTrainingPhaseEvictor._retire

    def spy(self, entry):
        held_at_sync = []
        inner = entry.end.synchronize

        def synchronize():
            held_at_sync.append(len(entry.keepalive))
            inner()

        entry.end.synchronize = synchronize
        result = original(self, entry)
        seen.append((entry.operation, held_at_sync))
        return result

    with harness.patched(), patch.object(
        SenseNovaTrainingPhaseEvictor, "_retire", spy
    ):
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert seen
    assert [held for operation, held in seen if operation == "h2d"]
    assert all(held == [1] for operation, held in seen if operation == "h2d")
    # The d2h leg keeps nothing: record_stream is what protects it, and holding
    # the source would defeat the point by delaying the free.
    assert all(held == [0] for operation, held in seen if operation == "d2h")


def test_the_seconds_are_measured_with_events_not_a_perf_counter_sandwich():
    """Under overlap a ``perf_counter`` around a non-blocking issue measures the
    launch, not the copy.

    MUTANT: keeping the host sandwich makes these totals ~0 rather than the
    fake event's 2 ms per operation.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()           # one-sided: 42 d2h
        assert evictor.d2h_seconds == pytest.approx(42 * 0.002)
        assert evictor.h2d_seconds == 0.0
        evictor.enter_denoise()          # 42 d2h + 42 h2d

    assert evictor.d2h_seconds == pytest.approx(84 * 0.002)
    assert evictor.h2d_seconds == pytest.approx(42 * 0.002)
    assert evictor.overlap_active is True


def test_the_byte_counters_are_unchanged_by_the_transfer_mode():
    """Bytes are a property of the plan, not of how it is issued."""
    harness = _Harness()
    overlapped = _evictor(harness)
    with harness.patched():
        overlapped.enter_prefix()
        overlapped.enter_denoise()

    serial = SenseNovaTrainingPhaseEvictor(transformer(), "meta")

    def d2h(modules, *, warn_once, pageable=False):
        del modules, warn_once, pageable

    def h2d(modules, device):
        del modules, device

    with patch.multiple(
        sensenova_phase_eviction, _move_modules_to_cpu=d2h, _move_modules_to_device=h2d
    ):
        serial.enter_prefix()
        serial.enter_denoise()

    assert (overlapped.d2h_bytes, overlapped.h2d_bytes) == \
        (serial.d2h_bytes, serial.h2d_bytes)


def test_the_leading_barrier_precedes_every_side_stream_issue():
    """Under overlap the leading ``_sync`` in ``_transition`` is a CORRECTNESS
    barrier, not the timing convenience 8.6 introduced it as: the side streams
    read -- and free -- weights the preceding phase's still-queued compute may
    still be writing, and ``join()`` at the tail only makes the DEFAULT stream
    wait on the side streams, never the reverse.

    MUTANT: deleting it as redundant (the serial copies block the host anyway)
    leaves the first side-stream issue ahead of any barrier.
    """
    harness = _Harness()
    evictor = _evictor(harness)

    def sync(self):
        harness.log.append(("sync", None))

    with harness.patched(), patch.object(
        SenseNovaTrainingPhaseEvictor, "_sync", sync
    ):
        evictor.enter_prefix()
        assert harness.log[0] == ("sync", None)
        boundary = len(harness.log)
        evictor.enter_denoise()

    assert harness.log[boundary] == ("sync", None)
    # One per transition, at its head -- not one per operation.
    assert sum(1 for entry in harness.log if entry == ("sync", None)) == 2


# ---------------------------------------------------------------------------
# the two pieces the evictor's own fakes replace: _TransferStreams and the
# device-side move
# ---------------------------------------------------------------------------


class _FakeCudaStream:
    def __init__(self, name):
        self.name = name
        self.waited = []

    def wait_stream(self, other):
        self.waited.append(other)


class _FakeCudaEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream


class _FakeCuda:
    """Injected INTO ``_TransferStreams`` -- the real routing and join wiring
    runs, only the driver objects are fake."""

    def __init__(self):
        self.current = _FakeCudaStream("default")
        self.entered = []

    def Stream(self, device=None):
        return _FakeCudaStream(f"side-{device}")

    def Event(self, enable_timing=False):
        return _FakeCudaEvent(enable_timing=enable_timing)

    def stream(self, stream):
        self.entered.append(stream)
        return contextlib.nullcontext()

    def current_stream(self, device=None):
        del device
        return self.current


def test_the_transfer_streams_object_routes_and_joins_both_streams():
    """``_TransferStreams`` itself was never constructed by any test: the
    evictor injects a whole fake in its place, and ``_make_transfer_streams``
    returns None before building one on a non-CUDA device.

    MUTANT: routing both directions to one stream (serializing the swap while
    still paying every correctness cost), or joining only one of them, which
    leaves ``assert_generation_resident`` passing on a queued copy.
    MUTANT: ``stream_context`` ignoring its argument (``self._cuda.stream(
    self.d2h)``) -- entered with both streams below, since entering with only
    the d2h one cannot tell the two apart, and the evictor-level test runs
    against the injected fake rather than this class.
    """
    cuda = _FakeCuda()
    streams = _TransferStreams("cuda:0", cuda=cuda)

    assert streams.d2h is not streams.h2d
    assert streams.stream_for("d2h") is streams.d2h
    assert streams.stream_for("h2d") is streams.h2d

    event = streams.record_event(streams.h2d)
    assert event.enable_timing is True          # elapsed_time needs it
    assert event.recorded_on is streams.h2d

    with streams.stream_context(streams.d2h):
        pass
    with streams.stream_context(streams.h2d):
        pass
    assert cuda.entered == [streams.d2h, streams.h2d]

    streams.join()
    assert cuda.current.waited == [streams.d2h, streams.h2d]


def test_the_h2d_destination_is_allocated_outside_the_side_streams_context():
    """The Finding-1 property, at the function that owns it. Torch's device
    caching allocator partitions free blocks by OWNING STREAM, so a destination
    requested inside the side stream's context can never be handed the block the
    paired d2h just freed on the default stream: it cudaMallocs instead, and the
    transient extra residency becomes a whole half rather than
    ``_OVERLAP_WINDOW_PAIRS`` modules.

    MUTANT: allocating with ``.to(device)`` inside the context (as shipped) puts
    the allocation between the enter and the exit below.
    MUTANT: dropping ``record_stream`` on the destination lets the default
    stream free a block the side stream is still writing.
    MUTANT: hoisting ``destination.copy_`` above the ``with`` -- only the record
    needs the context -- runs every transfer on the DEFAULT stream, so the flag
    becomes pure overhead at the full correctness cost and zero overlap. The
    copy is logged for that reason: with only the record logged, that mutant
    produces an identical sequence.
    """
    log = []

    class _Dest(torch.Tensor):
        def copy_(self, source, non_blocking=False):
            log.append(("copy", tuple(source.shape)))
            return torch.Tensor.copy_(self, source, non_blocking=non_blocking)

        def record_stream(self, stream):
            log.append(("record", stream))

    real_empty_like = torch.empty_like

    def empty_like(tensor, **kwargs):
        log.append(("alloc", tuple(tensor.shape), tensor.dtype))
        return real_empty_like(tensor, **kwargs).as_subclass(_Dest)

    class _ContextSpy:
        def stream_context(self, stream):
            @contextlib.contextmanager
            def _entered():
                log.append(("enter", stream))
                yield
                log.append(("exit", stream))

            return _entered()

    module = nn.Module()
    contiguous = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    transposed = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    module.register_buffer("weight", contiguous)
    module.register_buffer("scale", transposed)
    sources = []

    with patch("torch.empty_like", empty_like):
        _move_modules_to_device(
            (module,), "cpu", non_blocking=True, sources=sources,
            streams=_ContextSpy(), stream="h2d-stream",
        )

    assert [entry[0] for entry in log] == [
        "alloc", "alloc", "enter", "copy", "record", "copy", "record", "exit"
    ]
    assert sources[0] is contiguous and sources[1] is transposed
    for name, original in (("weight", contiguous), ("scale", transposed)):
        destination = module._buffers[name]
        assert isinstance(destination, _Dest)
        assert destination.dtype == original.dtype
        assert destination.shape == original.shape
        # ``empty_like`` preserves the source's layout, so the non-contiguous
        # buffer keeps its stride rather than being silently made contiguous.
        assert destination.stride() == original.stride()
        assert torch.equal(destination.float(), original.float())


def test_an_already_resident_tensor_is_not_reallocated_by_the_split_path():
    """``Tensor.to`` short-circuits an already-resident tensor and the split
    path has to keep that: re-allocating one would copy a whole half that never
    left the device.

    MUTANT: taking the empty_like path unconditionally.
    """
    module = nn.Module()
    resident = torch.ones(3, device="meta")
    module.register_buffer("weight", resident)

    calls = []
    real_empty_like = torch.empty_like

    def empty_like(tensor, **kwargs):
        calls.append(tensor)
        return real_empty_like(tensor, **kwargs)

    with patch("torch.empty_like", empty_like):
        _move_modules_to_device(
            (module,), "meta", non_blocking=True, streams=object(), stream=None
        )

    assert calls == []
    assert module._buffers["weight"] is resident


# ---------------------------------------------------------------------------
# failure and recovery
# ---------------------------------------------------------------------------


def test_a_pin_failure_drops_to_the_serial_path_for_the_rest_of_the_run(capsys):
    """``_stage_tensor`` falls back to a pageable destination per tensor, warned
    once. Continuing to issue async copies against it would be exactly the
    combination the installer refuses, arrived at silently.

    MUTANT: ignoring ``warn_once`` leaves every later call non_blocking=True.
    """
    harness = _Harness(fail_pin_at=10)
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()
        evictor.enter_denoise()

    non_blocking = [flag for _op, flag, _pageable in harness.calls]
    assert non_blocking[:10] == [True] * 10
    assert not any(non_blocking[10:])
    assert evictor.overlap_active is False   # the next transition ran serially
    assert capsys.readouterr().out.count("MoT overlapped transfer disabled") == 1


def test_best_effort_cpu_synchronizes_before_its_first_staged_check():
    """``_module_already_staged_cpu`` checks device and pin flag, never content:
    a pinned buffer whose d2h has not landed reads as already staged and is
    skipped, yielding a corrupt CPU half.

    MUTANT: moving the barrier below the loop (or after the first predicate
    call) puts a 'staged' answer before the join.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    evictor._streams = harness.streams
    order = []

    def staged(module, *, pageable=False):
        del module, pageable
        order.append("staged-check")
        return True

    harness.log.clear()
    with patch.object(sensenova_phase_eviction, "_module_already_staged_cpu", staged):
        original_join = harness.streams.join

        def join():
            original_join()
            order.append("join")

        harness.streams.join = join
        assert evictor._best_effort_cpu() is None

    assert order[0] == "join"
    assert order.count("staged-check") == 84


def test_the_straddling_transition_is_charted_as_serial_not_overlapped():
    """``sn_swap_overlap`` is the UNIT LABEL for sn_d2h_s/sn_h2d_s, so the
    transition that begins overlapped and finishes on the serial path -- part
    CUDA event milliseconds, part host wall seconds -- must not claim either.

    MUTANT: setting the flag from whether streams EXIST, at the top of
    ``_transition`` and before the downgrade can fire (as shipped), reports 1 for
    a transition whose seconds are a mixture.
    """
    harness = _Harness(fail_pin_at=10)
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()
        assert evictor.overlap_active is False        # downgraded mid-plan
        assert evictor.drain_transfer_stats()["overlap_active"] is False
        evictor.enter_denoise()

    assert evictor.drain_transfer_stats()["overlap_active"] is False


def test_a_fully_overlapped_step_ands_to_true_and_the_drain_resets_it():
    """The flag travels with the totals it labels; reading it off the evictor
    after the drain would label a step by the NEXT step's mode."""
    harness = _Harness()
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()
        evictor.enter_denoise()
        assert evictor.drain_transfer_stats()["overlap_active"] is True
        # Nothing has run since; the label is not carried forward.
        assert evictor.drain_transfer_stats()["overlap_active"] is False


def test_a_failed_transition_drains_the_window_before_it_unwinds():
    """The in-flight deque and every pinned keepalive in it are released as
    ``_run_overlapped``'s frame unwinds, handing those blocks back to the
    caching HOST allocator while their copies may still be reading them.

    MUTANT: leaving the drain to ``_best_effort_cpu`` (as shipped) puts the
    first join AFTER the unwind rather than before it.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    calls = []

    def explode(modules, *, warn_once, pageable=False, non_blocking=False, sources=None):
        del modules, warn_once, pageable, sources
        calls.append(non_blocking)
        if len(calls) > 6:
            raise RuntimeError("copy failed")
        harness.log.append(("issue", "d2h"))

    original = SenseNovaTrainingPhaseEvictor._best_effort_cpu

    def spy(self):
        harness.log.append(("best-effort", None))
        return original(self)

    with harness.patched(), patch.object(
        SenseNovaTrainingPhaseEvictor, "_best_effort_cpu", spy
    ):
        with patch.object(sensenova_phase_eviction, "_move_modules_to_cpu", explode):
            with pytest.raises(RuntimeError, match="copy failed"):
                evictor.enter_prefix()

    assert evictor.state == "failed"
    joins = [i for i, entry in enumerate(harness.log) if entry == ("join", None)]
    recovery = harness.log.index(("best-effort", None))
    assert joins and joins[0] < recovery
    # ...and copies were outstanding when that first join happened.
    assert any(entry == ("issue", "d2h") for entry in harness.log[: joins[0]])


def test_the_drain_synchronizes_the_device_after_joining_the_streams():
    """``_sync_transfers`` is a join AND a device sync, in that order. Every
    other test of it runs on a ``"meta"`` evictor, where ``_sync_device`` is
    None and ``_sync`` is a silent no-op, so they observe only the join.

    MUTANT: reducing ``_sync_transfers`` to ``self._streams.join()``.
    ``join()`` only makes the DEFAULT STREAM wait on the side streams -- the
    HOST does not -- so ``_run_overlapped``'s frame unwinds and hands pinned
    blocks back to the caching host allocator while their d2h copies are still
    reading them, which is the corruption this path exists to prevent.
    """
    harness = _Harness()
    calls = []

    def explode(modules, *, warn_once, pageable=False, non_blocking=False, sources=None):
        del modules, warn_once, pageable, sources, non_blocking
        calls.append(1)
        if len(calls) > 6:
            raise RuntimeError("copy failed")
        harness.log.append(("issue", "d2h"))

    original = SenseNovaTrainingPhaseEvictor._best_effort_cpu

    def spy(self):
        harness.log.append(("best-effort", None))
        return original(self)

    def synchronize(device=None):
        harness.log.append(("sync", device))

    # No CUDA is touched: only the two calls `_sync_device` and `_sync` make are
    # patched, so a "cuda:0" evictor resolves its sync device off-hardware.
    with patch("torch.cuda.is_available", lambda: True), patch(
        "torch.cuda.synchronize", synchronize
    ):
        evictor = SenseNovaTrainingPhaseEvictor(
            transformer(), "cuda:0", overlap_transfer=True,
            streams_factory=harness.factory,
        )
        with harness.patched(), patch.object(
            SenseNovaTrainingPhaseEvictor, "_best_effort_cpu", spy
        ):
            with patch.object(sensenova_phase_eviction, "_move_modules_to_cpu", explode):
                with pytest.raises(RuntimeError, match="copy failed"):
                    evictor.enter_prefix()

    device = torch.device("cuda:0")
    assert harness.log[0] == ("sync", device)           # the leading barrier
    joins = [i for i, entry in enumerate(harness.log) if entry == ("join", None)]
    recovery = harness.log.index(("best-effort", None))
    assert joins and joins[0] < recovery
    # The unwinding drain: join the side streams, THEN block the host on them.
    assert harness.log[joins[0] + 1] == ("sync", device)
    assert joins[0] + 1 < recovery


def test_a_failed_transition_still_normalizes_and_locks_the_evictor():
    harness = _Harness()
    evictor = _evictor(harness)

    def explode(modules, *, warn_once, pageable=False, non_blocking=False, sources=None):
        raise RuntimeError("copy failed")

    with harness.patched():
        with patch.object(sensenova_phase_eviction, "_move_modules_to_cpu", explode):
            with pytest.raises(RuntimeError, match="copy failed"):
                evictor.enter_prefix()
    assert evictor.state == "failed"


# ---------------------------------------------------------------------------
# the contract: refused without eviction, refused with pageable staging
# ---------------------------------------------------------------------------


def _sensenova():
    from core.model_loader import ModelLoader
    return patch.object(ModelLoader, "detect_model_type", return_value="sensenova")


def _config(**overrides):
    config = {
        "batch_size": 1,
        "optimizer": "adafactor",
        "gradient_accumulation_steps": 1,
        "use_ema": False,
        "num_optimizer_groups": 0,
        "blocks_to_swap": 0,
        "block_swap_h2d_only": False,
        "train_unet": True,
        "train_text_encoder": False,
        "sensenova_mot_phase_eviction": False,
        "sensenova_four_phase_eviction": False,
    }
    config.update(overrides)
    return config


def test_overlap_is_refused_without_eviction():
    with _sensenova():
        with pytest.raises(ValueError, match="requires sensenova_mot_phase_eviction"):
            _apply_sensenova_training_contract(
                "model", "lora",
                _config(sensenova_mot_overlap_transfer=True), {"sample": {}})


def test_overlap_plus_pageable_is_refused_before_the_checkpoint_loads():
    with _sensenova():
        with pytest.raises(ValueError) as excinfo:
            _apply_sensenova_training_contract(
                "model", "lora",
                _config(sensenova_mot_phase_eviction=True,
                        sensenova_mot_overlap_transfer=True,
                        sensenova_mot_pageable_staging=True), {"sample": {}})
    message = str(excinfo.value)
    assert "sensenova_mot_overlap_transfer" in message
    assert "sensenova_mot_pageable_staging" in message


def test_overlap_is_accepted_alongside_eviction():
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora",
            _config(sensenova_mot_phase_eviction=True,
                    sensenova_mot_overlap_transfer=True), {"sample": {}})


def test_overlap_defaults_off_and_does_not_arm_the_refusal():
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora", _config(), {"sample": {}})


# ---------------------------------------------------------------------------
# surface parity with sensenova_mot_pageable_staging
# ---------------------------------------------------------------------------


def test_the_flag_is_declared_as_its_own_feature():
    assert TRAINING_FEATURE_PARAMS["sensenova_mot_eviction"] == [
        "sensenova_mot_phase_eviction", "sensenova_four_phase_eviction",
        "sensenova_four_phase_shared_prefix",
        "sensenova_four_phase_grad_reduction"]
    assert TRAINING_FEATURE_PARAMS["sensenova_mot_overlap_transfer"] == [
        "sensenova_mot_overlap_transfer"]


def test_the_mechanism_is_declared_absent_everywhere_but_sensenova():
    for arch in sorted(TRAINING_DECLARED_ARCHS - {"sensenova"}):
        assert training_feature_unsupported_reason(
            arch, "sensenova_mot_overlap_transfer"), arch
    assert training_feature_unsupported_reason(
        "sensenova", "sensenova_mot_overlap_transfer") is None


def test_the_openapi_entry_matches_the_arch_capabilities_claim():
    import yaml

    repo = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((repo / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    description = props["sensenova_mot_overlap_transfer"]["description"]
    assert "Accepted and warned by every other architecture" in description
    # No speed is claimed: the ceiling is arithmetic and the realized figure is
    # explicitly unmeasured.
    assert "UNMEASURED" in description
    assert props["sensenova_mot_overlap_transfer"]["default"] is False


def test_the_default_matches_across_param_defaults_and_pydantic():
    from api.routes import TrainingRunCreateRequest

    assert TRAINING_DEFAULTS["sensenova_mot_overlap_transfer"] is False
    assert (
        TrainingRunCreateRequest.model_fields["sensenova_mot_overlap_transfer"].default
        is False
    )


def test_the_overlap_marker_is_registered_for_the_chart():
    """Without it a reader cannot tell which unit sn_d2h_s/sn_h2d_s are in."""
    from core.training.metric_registry import EXTRA_METRIC_DEFS

    assert EXTRA_METRIC_DEFS["sn_swap_overlap"]["axis"] == "right"
