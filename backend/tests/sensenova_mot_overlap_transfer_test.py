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

    def __init__(self):
        self.recorded = []

    def record_stream(self, stream):
        self.recorded.append(stream)


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
        del stream
        return contextlib.nullcontext()

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

    def factory(self, device):
        del device
        return self.streams

    def d2h(self, modules, *, warn_once, pageable=False, non_blocking=False,
            sources=None):
        del modules
        self.calls.append(("d2h", non_blocking, pageable))
        self.log.append(("issue", "d2h"))
        if non_blocking:
            source = _Source()
            self.d2h_sources.append(source)
            sources.append(source)
        if self._fail_pin_at is not None and len(self.calls) >= self._fail_pin_at:
            warn_once["pin_failed"] = True

    def h2d(self, modules, device, *, non_blocking=False, sources=None):
        del modules, device
        self.calls.append(("h2d", non_blocking, None))
        self.log.append(("issue", "h2d"))
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


def test_overlap_on_a_non_cuda_device_falls_back_to_the_same_order():
    """MUTANT: a flag that assumes CUDA makes a CPU/meta evictor crash rather
    than no-op. The real stream factory is used here, not a fake."""
    evictor = SenseNovaTrainingPhaseEvictor(
        transformer(), "meta", overlap_transfer=True
    )
    assert _order(evictor) == _SHIPPED_ORDER
    assert evictor.overlap_active is False
    assert evictor._streams is None


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


def test_every_d2h_source_is_record_streamed_before_the_model_drops_it():
    """``stage_modules_to_pinned_cpu`` reassigns ``parameter.data``, dropping
    the last reference to the CUDA source while the copy is still in flight.

    MUTANT: omitting ``record_stream`` lets the caching allocator hand that
    block to the concurrent h2d destination -- silent corruption, and nothing
    else in this file would notice.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    with harness.patched():
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert harness.d2h_sources
    assert all(source.recorded == [harness.streams.d2h]
               for source in harness.d2h_sources)


def test_the_h2d_pinned_source_is_held_until_its_own_event_is_waited_on():
    """Symmetric hazard: ``_move_modules_to_device`` drops the pinned source
    into torch's caching HOST allocator, which the next ``_stage_tensor`` may
    re-hand out. Torch may event-guard this; that is not verified here.

    MUTANT: dropping the keepalive at issue time empties every h2d entry below.
    """
    harness = _Harness()
    evictor = _evictor(harness)
    seen = []
    original = SenseNovaTrainingPhaseEvictor._retire

    def spy(self, entry):
        seen.append((entry.operation, len(entry.keepalive)))
        return original(self, entry)

    with harness.patched(), patch.object(
        SenseNovaTrainingPhaseEvictor, "_retire", spy
    ):
        evictor.enter_prefix()
        evictor.enter_denoise()

    assert seen
    assert all(held > 0 for operation, held in seen if operation == "h2d")
    # The d2h leg keeps nothing: record_stream is what protects it, and holding
    # the source would defeat the point by delaying the free.
    assert all(held == 0 for operation, held in seen if operation == "d2h")


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
