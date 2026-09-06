"""Guard: the runtime LR timeline (P1 of docs/guides/LR_SCHEDULER_DESIGN.md).

P0's ``ScheduleTimeline`` was a stub holding one ``total_steps(at=0)`` event.
This file fixes what the real one owes:

* events at the SAME step are ACCEPTED and applied in ``seq`` order (§17.3) --
  refusing them would drop a resume-time recomputation or a cancellation, and
  "last one wins" would silently reorder a decay/cancel/decay burst;
* ``request_id`` makes re-delivery idempotent, including for a refused request;
* the BASE/DECAYING/FLOOR/RECOVERING machine of §5.3 with §17.3's formulas:
  no ``ramp`` factor inside a decay, a start refused during warmup, a
  non-positive denominator refused, and a cancel that also voids the
  config-declared WSD decay;
* the state is per SPEC over one shared event list, so P6's per-group
  schedules diverge without a second timeline;
* the ``total_steps`` warp of §7.2: the curve BEFORE the anchor is
  bit-identical, continuous at it, and still reaches the end at the new total;
* the state.json round trip, truncated to ``at <= step``.

Purity (§4.3) is re-checked WITH events present: the fast-forward, the
re-assertion and the composed re-warmup evaluate the same lambda out of order.

CPU-only and hermetic: one 4-element parameter, no model, no dataset, no GPU.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/lr_schedule_timeline_test.py -v
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.lr_schedules import (  # noqa: E402
    STATE_BASE,
    STATE_DECAYING,
    STATE_FLOOR,
    STATE_RECOVERING,
    ScheduleTimeline,
    build_lr_scheduler,
    make_lambda,
    resolve_spec,
)

BASE_LR = 1e-4


def _optimizer(lr: float = BASE_LR, groups: int = 1):
    params = [torch.nn.Parameter(torch.zeros(4)) for _ in range(groups)]
    return torch.optim.AdamW([{"params": [p]} for p in params], lr=lr)


def _spec(name: str, W: int, T: int, config=None):
    return resolve_spec(config or {}, warmup_steps=W, total_steps=T, name=name)


def _run(name: str, W: int, T: int, config=None):
    """(spec, timeline, lambda) for one schedule, as a trainer builds it."""
    spec = _spec(name, W, T, config)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    scheduler = build_lr_scheduler(_optimizer(), spec, timeline)
    return spec, timeline, scheduler.lr_lambdas[0]


def _cos(q: float) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * q))


PLATEAU = {"lr_decay_start_ratio": 0.85, "lr_floor_ratio": 0.25}


# ---------------------------------------------------------------------------
# Ordering and idempotency (§17.3)
# ---------------------------------------------------------------------------

def test_two_events_at_the_same_step_are_accepted_in_seq_order():
    spec, timeline, fn = _run("constant", 50, 1000)
    assert timeline.add("decay", at=100) == "applied"
    assert timeline.add("cancel", at=100) == "applied"
    # Both landed, and the cancel saw the decay: a run that ordered "decay,
    # then no, cancel" inside one poll gets a recovery, not a decay.
    assert len([e for e in timeline.events if e["kind"] != "total_steps"]) == 2
    assert timeline.state_at(spec, 100).code == STATE_RECOVERING
    assert fn(100) == pytest.approx(1.0)


def test_the_same_two_events_in_the_other_order_give_the_other_state():
    spec, timeline, _ = _run("constant", 50, 1000)
    assert timeline.add("cancel", at=100) == "ignored_no_active_decay"
    assert timeline.add("decay", at=100) == "applied"
    assert timeline.state_at(spec, 100).code == STATE_DECAYING


def test_seq_and_not_list_position_decides_the_order():
    """load() sorts by (at, seq), so a state file whose list arrived out of
    order still replays in the order the commands were issued."""
    spec = _spec("constant", 0, 1000)
    timeline = ScheduleTimeline()
    timeline.load([
        {"kind": "total_steps", "at": 0, "value": 1000, "seq": 0},
        {"kind": "cancel", "at": 100, "length": 0, "seq": 2},
        {"kind": "decay", "at": 100, "length": None, "shape": "cosine", "seq": 1},
    ])
    assert [e["seq"] for e in timeline.events] == [0, 1, 2]
    assert timeline.state_at(spec, 100).code == STATE_BASE  # decay then cancel, R=0


def test_re_delivering_a_request_id_neither_re_applies_nor_re_answers():
    _, timeline, _ = _run("constant", 0, 1000)
    first = timeline.add("decay", at=100, request_id="req-1")
    before = len(timeline.events)
    assert timeline.add("decay", at=700, request_id="req-1") == first == "applied"
    assert len(timeline.events) == before


def test_a_refused_request_is_idempotent_too():
    """The refusal is recorded as a `noop`, so the same request_id answers the
    same thing however late it is re-delivered -- and can never take effect."""
    _, timeline, _ = _run("constant", 100, 1000)
    assert timeline.add("decay", at=50, request_id="req-2") == "rejected_during_warmup"
    assert timeline.add("decay", at=500, request_id="req-2") == "rejected_during_warmup"
    kinds = [e["kind"] for e in timeline.events]
    assert kinds.count("noop") == 1 and "decay" not in kinds


# ---------------------------------------------------------------------------
# Refusals (§5.3 / §17.3)
# ---------------------------------------------------------------------------

def test_a_decay_during_warmup_is_refused():
    spec, timeline, fn = _run("constant", 100, 1000)
    assert timeline.add("decay", at=99) == "rejected_during_warmup"
    assert fn(99) == pytest.approx(0.99)          # still the ramp
    assert timeline.state_at(spec, 99).code == STATE_BASE
    assert timeline.add("decay", at=100) == "applied"


@pytest.mark.parametrize("length", [0, -1, -900])
def test_a_non_positive_explicit_length_is_refused(length):
    _, timeline, _ = _run("constant", 0, 1000)
    assert timeline.add("decay", at=100, length=length) == "rejected_zero_length"


def test_a_start_with_no_room_left_before_the_nominal_end_is_refused():
    """length=None divides by T_nominal - tau(at); at the end that is zero."""
    _, timeline, _ = _run("constant", 0, 1000)
    assert timeline.add("decay", at=1000) == "rejected_zero_length"
    assert timeline.add("decay", at=1500) == "rejected_zero_length"
    assert timeline.add("decay", at=999) == "applied"


def test_a_second_decay_and_a_second_cancel_are_ignored():
    spec, timeline, _ = _run("constant", 50, 1000)
    assert timeline.add("decay", at=100) == "applied"
    assert timeline.add("decay", at=200) == "ignored_already_decaying"
    assert timeline.add("cancel", at=300) == "applied"
    assert timeline.add("cancel", at=310) == "ignored_already_recovering"
    assert timeline.state_at(spec, 310).code == STATE_RECOVERING


# ---------------------------------------------------------------------------
# The state machine (§5.3 with §17.3's formulas)
# ---------------------------------------------------------------------------

def test_a_decay_starts_at_the_current_multiplier_with_no_ramp_factor():
    """§17.3: m(s) = F + (m_start - F)*k(q). The ramp(s) factor §5 wrote is
    gone -- it cannot apply, because a start during warmup is refused."""
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 1000, PLATEAU)
    timeline.add("decay", at=100)
    assert fn(100) == pytest.approx(1.0)
    for step in (200, 500, 900):
        q = (step - 100) / 900.0
        assert fn(step) == pytest.approx(0.25 + 0.75 * _cos(q))


def test_q_reaching_one_is_the_floor_and_stays_there():
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 1000, PLATEAU)
    timeline.add("decay", at=100, length=100)
    assert timeline.state_at(spec, 199).code == STATE_DECAYING
    assert timeline.state_at(spec, 200).code == STATE_FLOOR
    for step in (200, 500, 1000, 5000):
        assert fn(step) == 0.25, step


def test_cancel_recovers_linearly_to_the_base_curve():
    spec, timeline, fn = _run("constant", 50, 1000)
    timeline.add("decay", at=100)
    m_c = fn(200)
    assert timeline.add("cancel", at=200) == "applied"
    assert fn(200) == pytest.approx(m_c)            # continuous
    assert fn(225) == pytest.approx(m_c + (1.0 - m_c) * 0.5)
    assert fn(250) == pytest.approx(1.0)
    assert timeline.state_at(spec, 250).code == STATE_BASE


def test_a_zero_warmup_run_recovers_discontinuously():
    """§5.4: R = W, and a run with no warmup asked for no ramp."""
    spec, timeline, fn = _run("constant", 0, 1000)
    timeline.add("decay", at=100)
    assert fn(200) < 1.0
    timeline.add("cancel", at=200)
    assert fn(200) == pytest.approx(1.0)
    assert timeline.state_at(spec, 200).code == STATE_BASE


def test_decay_cancel_decay_is_evaluated_in_arrival_order():
    """Not "last one wins": the second decay starts from the value the
    recovery had reached, not from the base curve."""
    spec, timeline, fn = _run("constant", 50, 1000)
    timeline.add("decay", at=100)
    timeline.add("cancel", at=200)
    recovering_at_210 = fn(210)
    assert 0 < recovering_at_210 < 1.0
    assert timeline.state_at(spec, 205).code == STATE_RECOVERING
    assert timeline.add("decay", at=210) == "applied"

    assert fn(210) == pytest.approx(recovering_at_210)   # continuous
    assert timeline.state_at(spec, 215).code == STATE_DECAYING
    assert fn(215) == pytest.approx(recovering_at_210 * _cos(5 / 790.0))
    # A "last one wins" reading would have started this decay from 1.0.
    assert fn(215) < _cos(5 / 790.0)


def test_a_command_is_scored_against_the_state_time_has_reached():
    """A recovery that finished, or a decay that reached its floor, must not
    still be the state a later command is judged against."""
    spec, timeline, _ = _run("constant", 50, 1000)
    timeline.add("decay", at=100)
    timeline.add("cancel", at=200)          # recovers over 50 steps
    assert timeline.state_at(spec, 300).code == STATE_BASE
    assert timeline.add("cancel", at=300) == "ignored_no_active_decay"
    assert timeline.add("decay", at=400) == "applied"


def test_the_recovery_length_is_baked_into_the_cancel():
    """§5.4: a later config edit cannot reshape a cancel that happened."""
    _, timeline, fn = _run("constant", 50, 1000)
    timeline.add("decay", at=100)
    timeline.add("cancel", at=200, length=10)
    assert fn(210) == pytest.approx(1.0)
    assert timeline.events[-1]["length"] == 10


# ---------------------------------------------------------------------------
# The cancel voids the config-declared decay (§17.3)
# ---------------------------------------------------------------------------

def test_cancel_disarms_a_scheduled_config_decay():
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 1000, PLATEAU)
    assert fn(900) == pytest.approx(0.25 + 0.75 * _cos(50 / 150.0))
    assert timeline.add("cancel", at=100) == "disarmed_scheduled_decay"
    assert timeline.state_at(spec, 900).code == STATE_BASE
    for step in (100, 850, 900, 1000, 2000):
        assert fn(step) == 1.0, step


def test_cancel_during_a_config_decay_recovers_from_it():
    spec, timeline, fn = _run("plateau_cosine_floor", 40, 1000, PLATEAU)
    m_c = fn(900)
    assert m_c < 1.0
    assert timeline.add("cancel", at=900) == "applied"
    assert fn(900) == pytest.approx(m_c)
    assert fn(940) == pytest.approx(1.0)     # recovered onto the DISARMED base
    assert fn(999) == 1.0


def test_a_later_decay_starts_from_a_new_event_not_the_disarmed_config_one():
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 1000, PLATEAU)
    timeline.add("cancel", at=100)
    assert fn(900) == 1.0
    assert timeline.add("decay", at=900) == "applied"
    assert fn(950) == pytest.approx(0.25 + 0.75 * _cos(50 / 100.0))


def test_cancel_on_a_curve_with_no_config_decay_is_ignored():
    """cosine's decay IS the base curve; §5.3 recovers TO the base curve, so
    there is nothing here to cancel."""
    spec, timeline, fn = _run("cosine", 0, 1000)
    assert timeline.add("cancel", at=500) == "ignored_no_active_decay"
    assert fn(500) == pytest.approx(_cos(0.5))
    assert timeline.state_at(spec, 500).code == STATE_BASE


# ---------------------------------------------------------------------------
# Per-group state over one shared event list (§17.3)
# ---------------------------------------------------------------------------

def test_one_event_list_two_specs_two_states():
    plateau = _spec("plateau_cosine_floor", 40, 1000, PLATEAU)
    cosine = _spec("cosine", 40, 1000)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(1000)
    plateau_fn = make_lambda(plateau, timeline)
    cosine_fn = make_lambda(cosine, timeline)

    assert timeline.add("cancel", at=900, spec=plateau) == "applied"

    assert timeline.state_at(plateau, 900).code == STATE_RECOVERING
    assert timeline.state_at(cosine, 900).code == STATE_BASE
    assert plateau_fn(900) == pytest.approx(0.25 + 0.75 * _cos(50 / 150.0))
    assert plateau_fn(940) == pytest.approx(1.0)
    assert cosine_fn(900) == pytest.approx(_cos(860 / 960.0))
    # The cancel that recovered the plateau group left the cosine group alone.
    assert cosine_fn(950) == pytest.approx(_cos(910 / 960.0))


def test_the_start_multiplier_comes_from_the_group_s_own_curve():
    linear = _spec("linear", 0, 1000)
    constant = _spec("constant", 0, 1000)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(1000)
    linear_fn = make_lambda(linear, timeline)
    constant_fn = make_lambda(constant, timeline)

    timeline.add("decay", at=500, spec=constant)
    assert linear_fn(500) == pytest.approx(0.5)
    assert constant_fn(500) == pytest.approx(1.0)
    assert linear_fn(750) == pytest.approx(0.5 * _cos(0.5))
    assert constant_fn(750) == pytest.approx(1.0 * _cos(0.5))


# ---------------------------------------------------------------------------
# The total_steps warp (§7.2 / §7.3)
# ---------------------------------------------------------------------------

def test_an_extension_leaves_the_past_bit_identical_and_still_lands_on_the_floor():
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 10000, PLATEAU)
    before = [fn(s) for s in range(0, 9001)]

    assert timeline.add("total_steps", at=9000, value=20000) == "applied"

    assert [fn(s) for s in range(0, 9001)] == before, "the past moved"
    assert fn(9001) == pytest.approx(before[-1], abs=1e-3), "discontinuous at the anchor"
    assert fn(20000) == 0.25
    # It does NOT go back to the plateau, which is what the old rebuild did.
    assert fn(15000) < 1.0
    assert 0.25 < fn(15000) < before[-1]


def test_a_second_extension_composes():
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 10000, PLATEAU)
    timeline.add("total_steps", at=9000, value=20000)
    before = [fn(s) for s in range(0, 15001, 25)]

    timeline.add("total_steps", at=15000, value=30000)

    assert [fn(s) for s in range(0, 15001, 25)] == before
    assert timeline.clock(30000) == pytest.approx(10000)
    assert fn(30000) == 0.25


def test_a_shrink_reaches_the_floor_early_and_holds_it():
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 10000, PLATEAU)
    timeline.add("total_steps", at=9000, value=9500)
    assert timeline.clock(9500) == pytest.approx(10000)
    assert fn(9500) == 0.25
    assert fn(20000) == 0.25


def test_a_shrink_to_the_anchor_holds_the_old_end():
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 10000, PLATEAU)
    timeline.add("total_steps", at=9000, value=8000)
    assert timeline.clock(9000) == pytest.approx(10000)
    assert fn(9000) == 0.25


def test_the_alias_start_resolves_from_the_nominal_total():
    """§17.2: the new T a resume was built with must not be re-injected into
    an alias's derived D. Built with 20000, resumed onto a nominal 10000, the
    plateau still ends at 8500 -- not at 17000."""
    spec = _spec("plateau_cosine_floor", 0, 20000, PLATEAU)
    timeline = ScheduleTimeline()
    timeline.load([{"kind": "total_steps", "at": 0, "value": 10000, "seq": 0}])
    fn = make_lambda(spec, timeline)
    assert fn(8499) == 1.0
    assert fn(8501) < 1.0
    assert fn(10000) == 0.25


def test_the_nominal_total_stays_the_first_event_and_current_the_last():
    _, timeline, _ = _run("cosine", 0, 10000)
    timeline.add("total_steps", at=9000, value=20000)
    assert timeline.nominal_total(0) == 10000
    assert timeline.current_total(0) == 20000
    assert timeline.add("total_steps", at=9500, value=20000) == "ignored_unchanged"


def test_an_extension_stretches_a_decay_that_was_already_running():
    spec, timeline, fn = _run("constant", 0, 10000)
    timeline.add("decay", at=8000)
    running = fn(9000)
    timeline.add("total_steps", at=9000, value=20000)
    assert fn(9000) == pytest.approx(running)
    assert fn(20000) == pytest.approx(0.0, abs=1e-12)
    assert timeline.state_at(spec, 20000).code == STATE_FLOOR


def test_an_explicit_length_is_a_real_axis_length_an_extension_does_not_stretch():
    """§7.3: "you said how long", so the extension runs on the floor."""
    spec, timeline, fn = _run("plateau_cosine_floor", 0, 10000, PLATEAU)
    timeline.add("decay", at=8000, length=500)
    timeline.add("total_steps", at=9000, value=20000)
    assert timeline.state_at(spec, 8500).code == STATE_FLOOR
    assert fn(8500) == 0.25
    assert fn(19000) == 0.25


# ---------------------------------------------------------------------------
# Purity, WITH events present (§4.3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["constant", "cosine", "linear",
                                  "plateau_cosine_floor"])
def test_evaluation_order_does_not_change_any_value(name):
    spec, timeline, fn = _run(name, 37, 1000, PLATEAU)
    timeline.add("decay", at=100)
    timeline.add("cancel", at=200)
    timeline.add("decay", at=200)
    timeline.add("total_steps", at=400, value=2000)
    timeline.add("cancel", at=600)

    steps = list(range(0, 2200))
    ascending = [fn(s) for s in steps]
    descending = {s: fn(s) for s in reversed(steps)}
    shuffled_steps = steps[:]
    random.Random(0).shuffle(shuffled_steps)
    shuffled = {s: fn(s) for s in shuffled_steps}
    repeated = [fn(s) for s in steps]

    for step, value in zip(steps, ascending):
        assert descending[step] == value, step
        assert shuffled[step] == value, step
    assert repeated == ascending
    # ...and evaluating never wrote to the timeline.
    assert len(timeline.events) == 6


def test_every_scheduler_of_a_fused_run_sees_the_same_timeline():
    spec = _spec("plateau_cosine_floor", 0, 1000, PLATEAU)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    schedulers = [build_lr_scheduler(_optimizer(), spec, timeline)
                  for _ in range(3)]
    timeline.add("decay", at=100)
    timeline.add("total_steps", at=200, value=2000)
    for step in range(0, 2001, 7):
        values = {s.lr_lambdas[0](step) for s in schedulers}
        assert len(values) == 1, (step, values)


# ---------------------------------------------------------------------------
# Persistence (§5.5 / D4)
# ---------------------------------------------------------------------------

def test_dump_truncates_to_the_checkpoint_position():
    _, timeline, _ = _run("constant", 0, 10000)
    timeline.add("decay", at=9137)
    kinds = [e["kind"] for e in timeline.dump(9000)]
    assert kinds == ["total_steps"]
    assert [e["kind"] for e in timeline.dump(9137)] == ["total_steps", "decay"]


def test_load_truncates_too_so_rewinding_drops_later_commands():
    spec, timeline, fn = _run("constant", 0, 10000)
    timeline.add("decay", at=9137)
    saved = timeline.dump(20000)

    rewound = ScheduleTimeline()
    rewound.load(saved, upto_step=9000)
    assert [e["kind"] for e in rewound.events] == ["total_steps"]
    assert rewound.state_at(spec, 9500).code == STATE_BASE

    kept = ScheduleTimeline()
    kept.load(saved, upto_step=9500)
    assert kept.state_at(spec, 9500).code == STATE_DECAYING


def test_a_reloaded_timeline_reproduces_the_curve_exactly():
    spec, timeline, fn = _run("plateau_cosine_floor", 20, 10000, PLATEAU)
    timeline.add("decay", at=1000)
    timeline.add("cancel", at=2000)
    timeline.add("total_steps", at=3000, value=20000)
    expected = [fn(s) for s in range(0, 20001, 13)]

    restored = ScheduleTimeline()
    restored.load(json.loads(json.dumps(timeline.dump(20000))))
    restored_fn = make_lambda(spec, restored)
    assert [restored_fn(s) for s in range(0, 20001, 13)] == expected


# ---------------------------------------------------------------------------
# The trainer seams
# ---------------------------------------------------------------------------

from core.training.base_trainer import (  # noqa: E402
    dump_lr_schedule_events,
    install_lr_schedule_events,
)


class _ResumeProbe:
    """What the resume seam reads off a trainer."""

    def __init__(self, name, W, T, config=None, gas=1, saved_events=None,
                 scheduler_step=None):
        self.log_prefix = "[Test]"
        self._grad_accum_steps = gas
        self._resume_scheduler_step = scheduler_step
        self._resume_scheduler_interval = gas
        self._resume_lr_schedule_events = saved_events
        self.lr_schedule_spec = _spec(name, W, T, config)
        self.lr_timeline = ScheduleTimeline()
        self.lr_timeline.set_total_steps(self.lr_schedule_spec.total_steps)
        self.lr_scheduler = build_lr_scheduler(
            _optimizer(), self.lr_schedule_spec, self.lr_timeline)

    @property
    def fn(self):
        return self.lr_scheduler.lr_lambdas[0]


@pytest.mark.parametrize("name", ["linear", "polynomial", "cosine",
                                  "cosine_with_restarts"])
def test_extension_during_warmup_keeps_the_peak_and_bounds(name):
    spec, timeline, fn = _run(name, 100, 1000, {"lr_floor_ratio": 0.0})
    timeline.add("total_steps", at=50, value=2000)
    assert fn(99) == pytest.approx(0.99)
    assert fn(100) == pytest.approx(1.0)
    values = [fn(s) for s in range(100, 2001)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert fn(2000) == pytest.approx(0.0)


@pytest.mark.parametrize("old_gas,new_gas,global_step,saved_position", [
    (1, 4, 500, 500),
    (4, 1, 500, 125),
    (4, 4, 503, 120),  # skipped advances and an incomplete window
])
def test_resume_total_counts_only_remaining_update_boundaries(
        old_gas, new_gas, global_step, saved_position):
    first = _ResumeProbe("cosine", 0, 1000 // old_gas)
    before = first.fn(saved_position)
    probe = _ResumeProbe("cosine", 0, 1000 // new_gas, gas=new_gas,
                         saved_events=first.lr_timeline.dump(saved_position),
                         scheduler_step=saved_position)
    probe._resume_scheduler_interval = old_gas
    install_lr_schedule_events(probe, global_step)
    end = saved_position + 1000 // new_gas - global_step // new_gas
    assert probe.lr_timeline.current_total(0) == end
    assert probe.fn(saved_position) == pytest.approx(before)
    assert probe.fn(end) == pytest.approx(0.0)
    # A later resume uses the same offset, not a second warp.
    later = _ResumeProbe("cosine", 0, 1000 // new_gas, gas=new_gas,
                         saved_events=probe.lr_timeline.dump(saved_position + 1),
                         scheduler_step=saved_position + 1)
    install_lr_schedule_events(later, (global_step // new_gas + 1) * new_gas)
    assert later.lr_timeline.events == probe.lr_timeline.events


def test_the_resume_seam_keeps_the_old_nominal_axis_and_warps(capsys):
    first = _ResumeProbe("plateau_cosine_floor", 0, 10000, PLATEAU)
    before = [first.fn(s) for s in range(0, 9001)]
    saved = first.lr_timeline.dump(9000)

    # Same run, resumed at 9000 after the owner raised total_steps to 20000.
    second = _ResumeProbe("plateau_cosine_floor", 0, 20000, PLATEAU,
                          saved_events=saved, scheduler_step=9000)
    install_lr_schedule_events(second, 9000)

    assert [second.fn(s) for s in range(0, 9001)] == before
    assert second.fn(20000) == 0.25
    assert "lr_schedule_total_steps_changed" in capsys.readouterr().out


def test_the_resume_seam_says_so_when_the_checkpoint_has_no_timeline(capsys):
    probe = _ResumeProbe("plateau_cosine_floor", 0, 20000, PLATEAU,
                         saved_events=None, scheduler_step=9000)
    install_lr_schedule_events(probe, 9000)
    out = capsys.readouterr().out
    assert "lr_schedule_state_missing" in out
    # The current total becomes the nominal axis: no warp, no second event.
    assert "lr_schedule_total_steps_changed" not in out
    assert [e["kind"] for e in probe.lr_timeline.events] == ["total_steps"]
    assert probe.lr_timeline.nominal_total(0) == 20000


def test_an_unchanged_total_steps_adds_no_event(capsys):
    first = _ResumeProbe("cosine", 0, 10000)
    second = _ResumeProbe("cosine", 0, 10000,
                          saved_events=first.lr_timeline.dump(5000),
                          scheduler_step=5000)
    install_lr_schedule_events(second, 5000)
    assert [e["kind"] for e in second.lr_timeline.events] == ["total_steps"]
    assert "lr_schedule_total_steps_changed" not in capsys.readouterr().out


def test_the_seam_drops_commands_issued_after_the_checkpoint(capsys):
    first = _ResumeProbe("constant", 0, 10000)
    first.lr_timeline.add("decay", at=9137)
    saved = first.lr_timeline.dump(20000)      # everything, including the decay

    second = _ResumeProbe("constant", 0, 10000, saved_events=saved,
                          scheduler_step=9000)
    install_lr_schedule_events(second, 9000)
    assert [e["kind"] for e in second.lr_timeline.events] == ["total_steps"]
    assert second.fn(9500) == 1.0


def test_dump_from_a_trainer_uses_the_live_scheduler_position():
    probe = _ResumeProbe("constant", 0, 10000)
    probe.lr_timeline.add("decay", at=5)
    for _ in range(4):
        probe.lr_scheduler.step()
    assert [e["kind"] for e in dump_lr_schedule_events(probe)] == ["total_steps"]
    for _ in range(2):
        probe.lr_scheduler.step()
    assert [e["kind"] for e in dump_lr_schedule_events(probe)] == \
        ["total_steps", "decay"]


def test_a_trainer_without_a_timeline_dumps_nothing():
    assert dump_lr_schedule_events(object()) == []


class _MntProbe(_ResumeProbe):
    """The MNT recomputation hook, which replaced three warnings that said the
    decay curve was now wrong."""

    from core.training.base_trainer import BaseTrainer as _B

    _reanchor_lr_schedule_total = _B._reanchor_lr_schedule_total
    del _B


def test_the_mnt_hook_reanchors_and_applies_the_new_lr_at_once(capsys):
    probe = _MntProbe("plateau_cosine_floor", 0, 10000, PLATEAU)
    probe.lr_scheduler.last_epoch = 9000
    at_9000 = probe.fn(9000)

    probe._reanchor_lr_schedule_total(20000)

    assert [e["kind"] for e in probe.lr_timeline.events] == \
        ["total_steps", "total_steps"]
    assert probe.fn(9000) == pytest.approx(at_9000)
    assert probe.fn(20000) == 0.25
    # §5.6: without the write, the next optimizer step runs on the old value.
    assert probe.lr_scheduler.optimizer.param_groups[0]["lr"] == \
        pytest.approx(BASE_LR * at_9000)
    assert "lr_schedule_total_steps_changed" in capsys.readouterr().out


def test_the_mnt_hook_is_silent_when_the_total_did_not_move(capsys):
    probe = _MntProbe("plateau_cosine_floor", 0, 10000, PLATEAU)
    probe.lr_scheduler.last_epoch = 9000
    probe._reanchor_lr_schedule_total(10000)
    assert [e["kind"] for e in probe.lr_timeline.events] == ["total_steps"]
    assert "lr_schedule_total_steps_changed" not in capsys.readouterr().out


def test_mnt_recomputation_keeps_the_saved_axis_offset():
    probe = _MntProbe("cosine", 0, 250, gas=4,
                      saved_events=[{"kind": "total_steps", "at": 0,
                                     "seq": 0, "value": 1000}],
                      scheduler_step=500)
    install_lr_schedule_events(probe, 500)
    probe.lr_scheduler.last_epoch = 500
    before = probe.fn(500)
    probe._reanchor_lr_schedule_total(1200, global_step=500)
    assert probe.lr_timeline.current_total(0) == 675
    assert probe.fn(500) == pytest.approx(before)
    assert probe.fn(675) == pytest.approx(0.0)


class _StateHarness:
    """The real state file round trip, with a timeline attached."""

    from core.training.base_trainer import BaseTrainer as _B

    save_training_state = _B.save_training_state
    load_training_state = _B.load_training_state
    del _B

    def __init__(self, output_dir, T=1000):
        self.output_dir = Path(output_dir)
        self.run_name = "20260101_000000_deadbeef"
        self.log_prefix = "[Test]"
        self._grad_accum_steps = 1
        self._dataset_fingerprint = None
        self._batches_per_epoch = 10
        self._crop_plan_fingerprint = None
        self.lr_schedule_spec = _spec("constant", 0, T)
        self.lr_timeline = ScheduleTimeline()
        self.lr_timeline.set_total_steps(T)
        self.lr_scheduler = build_lr_scheduler(
            _optimizer(), self.lr_schedule_spec, self.lr_timeline)


def test_the_events_round_trip_through_state_json(tmp_path):
    saver = _StateHarness(tmp_path)
    for _ in range(400):
        saver.lr_scheduler.step()
    saver.lr_timeline.add("decay", at=300)
    saver.save_training_state(step=400, epoch=0, batch_idx=3)

    loader = _StateHarness(tmp_path)
    state = loader.load_training_state(400)
    assert [e["kind"] for e in state["lr_schedule_events"]] == \
        ["total_steps", "decay"]
    assert loader._resume_lr_schedule_events == state["lr_schedule_events"]


def test_a_state_file_without_the_key_reads_as_no_events(tmp_path):
    harness = _StateHarness(tmp_path)
    harness.save_training_state(step=400, epoch=0, batch_idx=3)
    path = next(Path(tmp_path).glob("*_state.json"))
    body = json.loads(path.read_text())
    del body["lr_schedule_events"]
    path.write_text(json.dumps(body))

    loader = _StateHarness(tmp_path)
    loader.load_training_state(400)
    assert loader._resume_lr_schedule_events is None
