"""Guard: the runtime `retarget` event (R1 of docs/guides/LR_SCHEDULER_DESIGN.md).

§19 widens the timeline from "decay now / cancel" to "replace the schedule
itself from step S". This file fixes what R1 owes:

* ``ScheduleSpec.to_dict``/``from_dict``: versioned, tolerant of unknown keys,
  and carrying no absolute step (§19.5, invariant 15);
* the blend of §19.2 under both anchors, with ``L = 0`` as an instant switch,
  the chain rule when a retarget lands mid-blend, and D30's bound
  ``m(s) <= max(m_old(s), m_new(s))``;
* every row of §19.3's interaction table, the axis rebase included: a
  ``warmup_steps > 0`` retarget starts its new curve from 0, and a later decay
  command is scored against ``at - S``, not against ``at``;
* the eight refusals of §19.4, each recorded as a ``noop`` with a
  ``refused_kind`` and changing no multiplier;
* ``issued``-based truncation (invariant 6 as §19.5 revised it): a reservation
  for a future step survives every save between the order and its effect.

Determinism (invariant 2) is re-checked with blends in flight: the chain is
re-derived from the event list on every evaluation, so out-of-order evaluation
must return the same numbers.

CPU-only and hermetic: no optimizer, no model, no GPU.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/lr_schedule_retarget_test.py -v
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.lr_schedules import (  # noqa: E402
    BLEND_SHAPE_NAMES,
    RETARGET_ANCHORS,
    SPEC_VERSION,
    STATE_BASE,
    STATE_DECAYING,
    STATE_FLOOR,
    STATE_RECOVERING,
    WARN_SELECTOR_ON_UNGROUPED_RUN,
    ScheduleSpec,
    ScheduleTimeline,
    blend_length_on_scheduler_axis,
    make_lambda,
    resolve_spec,
)

TOTAL = 10000
PLATEAU = {"lr_decay_start_ratio": 0.85, "lr_floor_ratio": 0.25}


def _spec(name: str, W: int = 0, T: int = TOTAL, config=None) -> ScheduleSpec:
    return resolve_spec(config or {}, warmup_steps=W, total_steps=T, name=name)


def _run(name: str = "cosine", W: int = 0, T: int = TOTAL, config=None):
    """(spec, timeline) as the trainer builds them, with the spec bound."""
    spec = _spec(name, W, T, config)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    timeline.bind_spec(spec)
    return spec, timeline


def _events(timeline: ScheduleTimeline):
    return [e["kind"] for e in timeline.dump(10 ** 9)]


def _curve(spec, timeline, steps):
    return [timeline.multiplier(spec, s) for s in steps]


def _reloaded(timeline: ScheduleTimeline, upto: int = 10 ** 9,
              load_upto=None) -> ScheduleTimeline:
    """The timeline through a real JSON round trip."""
    restored = ScheduleTimeline()
    restored.load(json.loads(json.dumps(timeline.dump(upto))),
                  upto_step=load_upto)
    return restored


# ---------------------------------------------------------------------------
# Serialization (§19.5)
# ---------------------------------------------------------------------------

ROUND_TRIP_SPECS = [
    ("cosine", None),
    ("plateau_cosine_floor", PLATEAU),
    ("wsd", {"lr_decay_start_step": 400, "lr_decay_steps": 300,
             "lr_decay_shape": "rex"}),
    ("cosine_with_restarts", {"lr_cycle_steps": 500,
                              "lr_cycle_peak_decay": 0.8}),
]


@pytest.mark.parametrize("name,config", ROUND_TRIP_SPECS)
def test_a_spec_round_trips_through_json(name, config):
    spec = _spec(name, 100, TOTAL, config)
    payload = json.loads(json.dumps(spec.to_dict()))
    assert payload["v"] == SPEC_VERSION
    # An alias's derived start does not travel (invariant 15); the ratio it is
    # re-derived from does, so the restored spec evaluates identically. The
    # run's own start_decay parameters do not travel either (D42) and come back
    # at the dataclass defaults, which is what `_retarget` re-stamps over.
    expected = replace(spec, command_decay_length=None,
                       command_decay_shape="cosine")
    if spec.decay_start_ratio is not None:
        expected = replace(expected, decay_start_step=None)
    assert ScheduleSpec.from_dict(payload) == expected


def test_only_a_real_axis_decay_start_survives_serialization():
    alias = _spec("plateau_cosine_floor", 0, TOTAL, PLATEAU)
    assert alias.decay_start_step == 8500          # derived from the seed total
    assert "decay_start_step" not in alias.to_dict()
    assert alias.to_dict()["decay_start_ratio"] == 0.85

    configured = _spec("wsd", 0, TOTAL, {"lr_decay_start_step": 400})
    assert configured.to_dict()["decay_start_step"] == 400
    assert configured.to_dict()["decay_start_ratio"] is None


def test_from_dict_ignores_a_key_it_does_not_know():
    """A state file written by a newer build still has to resume."""
    payload = _spec("cosine", 10).to_dict()
    payload["some_future_key"] = {"nested": [1, 2]}
    assert ScheduleSpec.from_dict(payload) == _spec("cosine", 10)


def test_from_dict_restores_ints_that_json_widened_to_floats():
    payload = _spec("cosine", 10).to_dict()
    payload["warmup_steps"] = 10.0
    payload["total_steps"] = float(TOTAL)
    restored = ScheduleSpec.from_dict(payload)
    assert isinstance(restored.warmup_steps, int)
    assert isinstance(restored.total_steps, int)


def test_from_dict_refuses_a_payload_missing_a_required_key():
    payload = _spec("cosine").to_dict()
    del payload["name"]
    with pytest.raises(ValueError, match="missing required key"):
        ScheduleSpec.from_dict(payload)


def test_the_event_stores_the_spec_in_its_serialized_form():
    spec, timeline = _run()
    assert timeline.add("retarget", at=5000, new_spec=_spec("constant"),
                        length=250, shape="cosine", groups=["unet"],
                        gain=1.5, issued=4800) == "applied"
    event = timeline.dump(TOTAL)[-1]
    assert event["kind"] == "retarget"
    assert event["at"] == 5000 and event["issued"] == 4800
    assert event["anchor"] == "restart" and event["gain"] == 1.5
    assert event["length"] == 250 and event["shape"] == "cosine"
    assert event["groups"] == ["unet"]
    assert event["spec"]["v"] == SPEC_VERSION
    json.dumps(event)  # the state file has to be able to hold it


# ---------------------------------------------------------------------------
# §19.3 row 1: a retarget in BASE
# ---------------------------------------------------------------------------

def test_a_retarget_in_base_starts_from_the_current_base_value():
    spec, timeline = _run("cosine")
    before = timeline.multiplier(spec, 5000)
    timeline.add("retarget", at=5000, new_spec=_spec("constant"), length=400)
    assert timeline.multiplier(spec, 5000) == pytest.approx(before)
    # constant, restart-anchored: the value at S is held to the end.
    assert timeline.multiplier(spec, 9999) == pytest.approx(before)
    assert timeline.active_spec(spec, 5400).name == "constant"


def test_the_new_curve_replaces_the_old_one_for_good():
    spec, timeline = _run("cosine")
    plain = make_lambda(spec, ScheduleTimeline([
        {"kind": "total_steps", "at": 0, "value": TOTAL, "seq": 0}]))
    timeline.add("retarget", at=4000, new_spec=_spec("constant"), length=0)
    assert timeline.multiplier(spec, 3999) == pytest.approx(plain(3999))
    assert timeline.multiplier(spec, 4001) != pytest.approx(plain(4001))


# ---------------------------------------------------------------------------
# §19.3 row 2: DECAYING / FLOOR / RECOVERING are ABSORBED (D26)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state_code,at", [
    (STATE_DECAYING, 4200),
    (STATE_FLOOR, 5000),
    (STATE_RECOVERING, 4200),
])
def test_a_retarget_absorbs_a_running_overlay(state_code, at):
    spec, timeline = _run("constant", 0, TOTAL, {"lr_floor_ratio": 0.1})
    timeline.add("decay", at=4000, length=500)
    if state_code == STATE_RECOVERING:
        timeline.add("cancel", at=4100, length=2000)
    assert timeline.state_at(spec, at).code == state_code

    realized = timeline.multiplier(spec, at)
    assert timeline.add("retarget", at=at, new_spec=_spec("constant"),
                        length=0) == "applied"
    # Back to BASE under the new spec, and continuous from the realized value.
    assert timeline.state_at(spec, at).code == STATE_BASE
    assert timeline.multiplier(spec, at) == pytest.approx(realized)
    assert timeline.multiplier(spec, at + 3000) == pytest.approx(realized)


def test_the_absorbed_overlay_does_not_disarm_the_new_config_decay():
    """D26: `decay_disarmed` is not carried over -- the new spec's own decay is
    armed again."""
    spec, timeline = _run("plateau_cosine_floor", 0, TOTAL, PLATEAU)
    timeline.add("decay", at=1000, length=500)
    timeline.add("cancel", at=1200, length=0)
    assert timeline.state_at(spec, 1500).decay_disarmed is True

    new = _spec("plateau_cosine_floor", 0, TOTAL, PLATEAU)
    timeline.add("retarget", at=2000, new_spec=new, length=0)
    assert timeline.state_at(spec, 3000).decay_disarmed is False
    # The new spec's plateau ends at 85% of its own remaining span.
    at_start = timeline.multiplier(spec, 2000 + round(0.85 * 8000) - 1)
    at_end = timeline.multiplier(spec, TOTAL)
    assert at_start > at_end
    assert at_end == pytest.approx(0.25 * at_start, rel=1e-9)


# ---------------------------------------------------------------------------
# §19.3 row 3: a retarget during warmup
# ---------------------------------------------------------------------------

def test_a_retarget_during_warmup_is_allowed_and_stays_in_base():
    """The blend must not be read as "a decay that raises the LR": the state
    stays BASE, so `lr_decay_state` keeps reporting 0."""
    spec, timeline = _run("cosine", 1000, TOTAL)
    assert timeline.add("retarget", at=400, new_spec=_spec("cosine", 200),
                        length=300) == "applied"
    for step in (400, 500, 700, 1200):
        assert timeline.state_at(spec, step).code == STATE_BASE


# ---------------------------------------------------------------------------
# §19.3 row 4: a total_steps change after a retarget
# ---------------------------------------------------------------------------

def test_an_extension_after_a_retarget_moves_the_new_curve_s_end():
    """D22: the restart-anchored span is derived at evaluation time, so the
    curve before the anchor is untouched and the end follows the new total."""
    spec, timeline = _run("constant")
    timeline.add("retarget", at=2000, new_spec=_spec("cosine"), length=0)
    before = _curve(spec, timeline, range(0, 5001, 250))
    at_old_end = timeline.multiplier(spec, TOTAL)

    timeline.add("total_steps", at=5000, value=2 * TOTAL)
    assert _curve(spec, timeline, range(0, 5001, 250)) == before
    assert timeline.multiplier(spec, TOTAL) > at_old_end
    assert timeline.multiplier(spec, 2 * TOTAL) == pytest.approx(at_old_end)


@pytest.mark.parametrize("name,config", ROUND_TRIP_SPECS)
def test_the_stored_spec_is_never_re_derived(name, config):
    """Invariant 15: the payload carries no step derived from the seed axis,
    and an extension does not rewrite what was written once -- the span is
    derived at evaluation time instead (D22).

    The one step-valued field that may survive is `wsd`'s configured `D`, a
    real-axis quantity (D8) read from the retarget's own origin, which
    `test_a_configured_decay_start_is_a_length_from_the_retarget_step` pins.
    """
    spec, timeline = _run("constant")
    timeline.add("retarget", at=2000, new_spec=_spec(name, 100, TOTAL, config),
                 length=0)
    stored = json.loads(json.dumps(timeline.dump(TOTAL)[-1]["spec"]))
    assert stored["total_steps"] == TOTAL   # the seed, not the derived span
    if stored["decay_start_ratio"] is not None:
        assert "decay_start_step" not in stored

    timeline.add("total_steps", at=3000, value=3 * TOTAL)
    assert timeline.dump(3 * TOTAL)[-2]["spec"] == stored


def test_an_alias_s_derived_decay_start_is_re_derived_on_the_local_span():
    """The alias bakes 8500 for a T=10000 seed; retargeted at 2000 its plateau
    must end at 85% of the REMAINING span (local 6800 = step 8800), not there."""
    spec, timeline = _run("constant")
    timeline.add("retarget", at=2000,
                 new_spec=_spec("plateau_cosine_floor", 0, TOTAL, PLATEAU),
                 length=0)
    assert "decay_start_step" not in timeline.dump(TOTAL)[-1]["spec"]
    assert timeline.multiplier(spec, 8799) == pytest.approx(1.0)
    assert timeline.multiplier(spec, 8801) < 1.0
    assert timeline.multiplier(spec, TOTAL) == pytest.approx(0.25)


def test_a_configured_decay_start_is_a_length_from_the_retarget_step():
    """`wsd`'s D is a real-axis position (D8); under `anchor=restart` that axis
    starts at S, so D=400 means 400 steps after the retarget."""
    spec, timeline = _run("constant")
    timeline.add("retarget", at=2000, length=0, new_spec=_spec(
        "wsd", 0, TOTAL, {"lr_decay_start_step": 400, "lr_decay_steps": 300}))
    assert timeline.multiplier(spec, 2399) == pytest.approx(1.0)
    assert timeline.multiplier(spec, 2401) < 1.0
    assert timeline.multiplier(spec, 2700) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# §19.3 row 5: decay / cancel after a retarget -- the AXIS REBASE
# ---------------------------------------------------------------------------

def test_a_decay_after_a_warming_retarget_is_accepted():
    """The rebase compares `at - S` with the new W. A command past the new
    warmup is accepted and starts the decay from the realized value."""
    spec, timeline = _run("cosine")
    timeline.add("retarget", at=5000, new_spec=_spec("cosine", 1000), length=0)
    assert timeline.add("decay", at=6500, length=1000) == "applied"
    assert timeline.state_at(spec, 6500).code == STATE_DECAYING
    assert timeline.state_at(spec, 7500).code == STATE_FLOOR


def test_a_decay_inside_the_new_warmup_is_refused_on_the_rebased_axis():
    """Unrebased, `at=5500 >= W_new=1000` would ACCEPT a decay that starts on a
    rising ramp -- exactly what `rejected_during_warmup` exists to stop."""
    spec, timeline = _run("cosine")
    timeline.add("retarget", at=5000, new_spec=_spec("cosine", 1000), length=0)
    assert timeline.add("decay", at=5500) == "rejected_during_warmup"
    assert timeline.state_at(spec, 5500).code == STATE_BASE


def test_the_new_curve_starts_from_zero_when_the_retarget_warms_up():
    """§19.2: g(0) = 1 holds only for W = 0. A warming retarget is a deliberate
    "warm up again from here", so it descends to 0 and climbs back."""
    spec, timeline = _run("cosine")
    peak = timeline.multiplier(spec, 5000)
    timeline.add("retarget", at=5000, new_spec=_spec("constant", 1000),
                 length=0)
    assert timeline.multiplier(spec, 5000) == 0.0
    assert timeline.multiplier(spec, 5500) == pytest.approx(0.5 * peak)
    assert timeline.multiplier(spec, 6000) == pytest.approx(peak)


def test_a_decay_after_a_scaled_retarget_starts_from_the_unscaled_curve():
    """`gain` multiplies the whole new curve, floor included, so the decay's
    seed is the curve's OWN value: seeding from the realized (already scaled)
    multiplier squares the gain and makes the LR jump at the decay step."""
    spec, timeline = _run("constant")
    new = _spec("constant", 0, TOTAL, {"lr_floor_ratio": 0.3})
    timeline.add("retarget", at=3000, new_spec=new, length=0, gain=2.0)
    assert timeline.add("decay", at=5000, length=1000) == "applied"

    assert timeline.multiplier(spec, 4999) == pytest.approx(2.0)
    assert timeline.multiplier(spec, 5000) == pytest.approx(2.0)
    assert timeline.multiplier(spec, 6000) == pytest.approx(2.0 * 0.3)
    assert timeline.multiplier(spec, 9000) == pytest.approx(2.0 * 0.3)


def test_a_cancel_after_a_retarget_recovers_over_the_new_warmup_length():
    spec, timeline = _run("cosine")
    timeline.add("retarget", at=3000, new_spec=_spec("constant", 200),
                 length=0)
    timeline.add("decay", at=4000, length=500)
    assert timeline.add("cancel", at=4200) == "applied"
    state = timeline.state_at(spec, 4200)
    assert state.code == STATE_RECOVERING and state.length == 200
    assert timeline.state_at(spec, 4400).code == STATE_BASE


# ---------------------------------------------------------------------------
# §19.3 rows 6-7: cycle restarts, and the group selector with D16 off
# ---------------------------------------------------------------------------

def test_a_relora_restart_event_is_not_a_retarget():
    """Two different things that share a word (§19.3): a ReLoRA merge event
    neither replaces the spec nor disturbs a blend."""
    spec, timeline = _run("constant")
    timeline.add("retarget", at=2000, new_spec=_spec("cosine"), length=1000)
    expected = _curve(spec, timeline, range(0, TOTAL + 1, 250))
    assert timeline.add("restart", at=2500) == "applied"
    assert timeline.active_spec(spec, 3000).name == "cosine"
    assert _curve(spec, timeline, range(0, TOTAL + 1, 250)) == expected


def test_a_group_selector_is_accepted_and_applies_to_every_spec():
    """§19.3's last row: with `lr_group_schedules` off there is one spec, so a
    `groups` selector is recorded and not refused. R1 applies a retarget to
    EVERY spec sharing the timeline; R2 owns the filtering."""
    spec_a, timeline = _run("constant")
    spec_b = _spec("linear")
    assert timeline.add("retarget", at=2000, new_spec=_spec("cosine"),
                        length=0, groups=["unet"],
                        known_groups=("unet", "text_encoder")) == "applied"
    assert timeline.dump(TOTAL)[-1]["groups"] == ["unet"]
    assert "known_groups" not in timeline.dump(TOTAL)[-1]
    for spec in (spec_a, spec_b):
        assert timeline.active_spec(spec, 3000).name == "cosine"


# ---------------------------------------------------------------------------
# §19.2: blending, both anchors, L = 0 and L > 0
# ---------------------------------------------------------------------------

def test_length_zero_switches_instantly():
    spec, timeline = _run("constant")
    new = _spec("cosine")
    timeline.add("retarget", at=4000, new_spec=new, length=0)
    lone = ScheduleTimeline([{"kind": "total_steps", "at": 0,
                              "value": TOTAL - 4000, "seq": 0}])
    for step in range(4000, TOTAL, 137):
        assert timeline.multiplier(spec, step) == pytest.approx(
            lone.multiplier(new, step - 4000))


@pytest.mark.parametrize("anchor", RETARGET_ANCHORS)
@pytest.mark.parametrize("shape", BLEND_SHAPE_NAMES)
def test_a_blend_starts_on_the_old_curve_and_ends_on_the_new_one(shape, anchor):
    spec, timeline = _run("constant")
    old = timeline.multiplier(spec, 4000)
    timeline.add("retarget", at=4000, new_spec=_spec("cosine", 0), length=800,
                 shape=shape, anchor=anchor)
    instant = _reloaded(timeline)
    instant.events[-1]["length"] = 0

    assert timeline.multiplier(spec, 4000) == pytest.approx(old)
    # Mid-blend the old curve (a flat 1.0) still has weight, so the value sits
    # above the new curve; by `at + length` the weight is exactly 1.
    for step in (4200, 4400, 4600):
        assert timeline.multiplier(spec, step) > instant.multiplier(spec, step)
    for step in (4800, 5200, 9000):
        assert timeline.multiplier(spec, step) == pytest.approx(
            instant.multiplier(spec, step))


def test_the_continue_anchor_evaluates_the_new_spec_on_the_global_axis():
    spec, timeline = _run("constant")
    new = _spec("cosine")
    timeline.add("retarget", at=4000, new_spec=new, anchor="continue",
                 length=0)
    plain = ScheduleTimeline([{"kind": "total_steps", "at": 0,
                               "value": TOTAL, "seq": 0}])
    for step in range(4000, TOTAL + 1, 271):
        assert timeline.multiplier(spec, step) == pytest.approx(
            plain.multiplier(new, step))


def test_the_restart_anchor_is_not_the_continue_anchor():
    """D23's reason for the default: `continue` jumps to the new curve's
    mid-run value, `restart` starts from where the run actually is."""
    spec, timeline = _run("constant")
    other = ScheduleTimeline(timeline.dump(TOTAL))
    timeline.add("retarget", at=8000, new_spec=_spec("cosine"), length=0)
    other.add("retarget", at=8000, new_spec=_spec("cosine"),
              anchor="continue", length=0)
    assert timeline.multiplier(spec, 8000) == pytest.approx(1.0)
    assert other.multiplier(spec, 8000) < 0.1


@pytest.mark.parametrize("anchor", ["restart", "continue"])
def test_gain_scales_the_new_curve(anchor):
    spec, timeline = _run("constant")
    doubled = ScheduleTimeline(timeline.dump(TOTAL))
    timeline.add("retarget", at=3000, new_spec=_spec("cosine"), anchor=anchor,
                 length=0)
    doubled.add("retarget", at=3000, new_spec=_spec("cosine"), anchor=anchor,
                length=0, gain=2.0)
    for step in range(3000, TOTAL + 1, 311):
        assert doubled.multiplier(spec, step) == pytest.approx(
            2.0 * timeline.multiplier(spec, step))


# ---------------------------------------------------------------------------
# §19.2: the chain rule
# ---------------------------------------------------------------------------

def test_a_retarget_mid_blend_starts_from_the_realized_value():
    spec, timeline = _run("constant")
    timeline.add("retarget", at=2000, new_spec=_spec("cosine"), length=2000)
    realized = timeline.multiplier(spec, 3000)
    timeline.add("retarget", at=3000, new_spec=_spec("constant"), length=0)
    assert timeline.multiplier(spec, 3000) == pytest.approx(realized)
    assert timeline.multiplier(spec, 9000) == pytest.approx(realized)
    # The first blend was still running, so the value it handed over is
    # neither of the two curves it was mixing.
    assert realized != pytest.approx(1.0)


def test_a_completed_link_contributes_nothing():
    """w = 1 past `at + length`, so the chain walk stops there: a finished
    blend cannot leak into a later one."""
    spec, timeline = _run("constant")
    timeline.add("retarget", at=1000, new_spec=_spec("cosine"), length=500)
    timeline.add("retarget", at=6000, new_spec=_spec("constant"), length=400)

    handover = timeline.multiplier(spec, 6000)
    for step in range(6400, TOTAL + 1, 200):
        assert timeline.multiplier(spec, step) == pytest.approx(handover)


def test_the_blend_never_exceeds_either_curve():
    """D30, checked against the two curves it mixes: the timeline WITHOUT the
    retarget is m_old, and the same retarget with L = 0 is m_new."""
    spec, timeline = _run("cosine", 100, TOTAL)
    timeline.add("decay", at=2000, length=4000)
    timeline.add("retarget", at=3000, new_spec=_spec("constant", 500),
                 length=1500, shape="cosine", gain=1.4)

    saved = timeline.dump(TOTAL)
    old = ScheduleTimeline([e for e in saved if e["kind"] != "retarget"])
    new = ScheduleTimeline(saved)
    new.events[-1]["length"] = 0

    for step in range(3000, 4501):
        blended = timeline.multiplier(spec, step)
        lo = min(old.multiplier(spec, step), new.multiplier(spec, step))
        hi = max(old.multiplier(spec, step), new.multiplier(spec, step))
        assert lo - 1e-12 <= blended <= hi + 1e-12, step


# ---------------------------------------------------------------------------
# §19.4: the eight refusals
# ---------------------------------------------------------------------------

def _assert_refused(timeline, spec, result, expected):
    assert result == expected
    event = timeline.dump(10 ** 9)[-1]
    assert event["kind"] == "noop" and event["refused_kind"] == "retarget"
    assert timeline.active_spec(spec, 10 ** 6) is spec


def test_rule_1_a_backdated_retarget_is_refused():
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=4000, issued=5000, new_spec=_spec("constant")),
        "rejected_backdated")


@pytest.mark.parametrize("at,expected", [
    (5000, "applied"),          # at == issued
    (4999, "rejected_backdated"),
])
def test_rule_1_s_boundary_is_at_equals_issued(at, expected):
    """One step either side: this is the guard against the retroactive rewrite
    §17.3 records for ReLoRA, so it is pinned as tightly as rule 5's."""
    _, timeline = _run()
    assert timeline.add("retarget", at=at, issued=5000, length=0,
                        new_spec=_spec("constant")) == expected


def test_rule_1_a_future_dated_retarget_is_accepted():
    spec, timeline = _run()
    assert timeline.add("retarget", at=6000, issued=5000,
                        new_spec=_spec("constant"), length=0) == "applied"
    assert timeline.multiplier(spec, 5999) != timeline.multiplier(spec, 6000)


@pytest.mark.parametrize("bad", ["relora", "not_a_schedule"])
def test_rule_2_a_schedule_outside_the_vocabulary_is_refused(bad):
    """`relora` resolves but is not selectable: it is shaped by restart events,
    so a run could otherwise ask for a curve with no restarts in it."""
    spec, timeline = _run()
    broken = replace(_spec("constant"), name=bad)
    _assert_refused(timeline, spec,
                    timeline.add("retarget", at=4000, new_spec=broken),
                    "rejected_unknown_scheduler")


def test_rule_2_an_unknown_curve_is_refused_too():
    spec, timeline = _run()
    broken = replace(_spec("constant"), curve="parabola")
    _assert_refused(timeline, spec,
                    timeline.add("retarget", at=4000, new_spec=broken),
                    "rejected_unknown_scheduler")


@pytest.mark.parametrize("curve", ["relora", "wsd"])
def test_rule_2_refuses_a_curve_that_does_not_belong_to_the_name(curve):
    """A serialized pair is not evidence: `relora` is a real curve, so
    name="cosine" with curve="relora" would otherwise install the segmented
    ReLoRA curve on a run with no merges."""
    spec, timeline = _run()
    broken = replace(_spec("cosine"), curve=curve)
    _assert_refused(timeline, spec,
                    timeline.add("retarget", at=4000, new_spec=broken),
                    "rejected_unknown_scheduler")


def test_rule_3_a_restart_with_no_remaining_span_is_refused():
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=TOTAL, new_spec=_spec("constant")),
        "rejected_no_remaining_span")


def test_rule_3_does_not_apply_to_the_continue_anchor():
    """`continue` has no span of its own to derive, so the same step is fine."""
    _, timeline = _run()
    assert timeline.add("retarget", at=TOTAL, new_spec=_spec("constant"),
                        anchor="continue") == "applied"


def test_rule_4_a_negative_blend_length_is_refused():
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=4000, new_spec=_spec("constant"), length=-1),
        "rejected_negative_length")


def test_rule_4_a_length_that_rounds_to_zero_is_refused_not_clamped():
    """§18.5's sentinel: 0 means "instant" here, so a blend of 2 global steps
    under gas=4 must not become a hard switch behind the caller's back."""
    assert blend_length_on_scheduler_axis(2, 4) == -1
    assert blend_length_on_scheduler_axis(8, 4) == 2
    assert blend_length_on_scheduler_axis(0, 4) == 0
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=4000, new_spec=_spec("constant"),
        length=blend_length_on_scheduler_axis(2, 4)),
        "rejected_negative_length")


def test_rule_5_a_warmup_longer_than_the_remaining_span_is_refused():
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=9500, new_spec=_spec("constant", 2000)),
        "rejected_warmup_exceeds_span")


def test_rule_5_measures_the_span_from_the_retarget_step():
    _, timeline = _run()
    assert timeline.add("retarget", at=7000, new_spec=_spec("constant", 2000),
                        length=0) == "applied"


def test_rule_5_refuses_a_warmup_exactly_as_long_as_the_span():
    """§17.2's contract is 0 <= W < T_sched, so the boundary is >=: a warmup
    ending exactly at the run's end leaves no step at peak."""
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=8000, new_spec=_spec("constant", 2000)),
        "rejected_warmup_exceeds_span")


def test_rule_5_measures_the_continue_span_as_the_whole_total():
    """A `continue` warmup is a position from 0, not a length from S, so a
    long-finished warmup must not refuse a retarget late in the run."""
    _, timeline = _run()
    assert timeline.add("retarget", at=9500, new_spec=_spec("constant", 2000),
                        anchor="continue", length=0) == "applied"


def test_rule_6_a_floor_outside_zero_to_one_is_refused():
    spec, timeline = _run()
    broken = replace(_spec("constant"), floor_ratio=1.5)
    _assert_refused(timeline, spec,
                    timeline.add("retarget", at=4000, new_spec=broken),
                    "rejected_floor_out_of_range")


def test_rule_7_an_unknown_group_name_is_refused():
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=4000, new_spec=_spec("constant"),
        groups=["unet", "typo_encoder"], known_groups=("unet", "text_encoder")),
        "rejected_unknown_group")


def test_rule_7_cannot_be_checked_without_the_component_list():
    """No `known_groups` means the caller has not said what exists; refusing
    every name would make the selector unusable from a context that does not
    know the run's components."""
    _, timeline = _run()
    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        groups=["whatever"], length=0) == "applied"


@pytest.mark.parametrize("gain", [0.0, -1.0])
def test_rule_8_a_non_positive_gain_is_refused(gain):
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=4000, new_spec=_spec("constant"), gain=gain),
        "rejected_non_positive_gain")


def test_rule_8_a_gain_above_one_is_allowed():
    """It raises the LR, which is a legitimate instruction (D29 shows it in the
    preview rather than refusing it)."""
    spec, timeline = _run("constant")
    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        gain=3.0, length=0) == "applied"
    assert timeline.multiplier(spec, 5000) == pytest.approx(3.0)


def test_a_refused_retarget_is_idempotent_by_request_id():
    _, timeline = _run()
    first = timeline.add("retarget", at=1000, issued=2000,
                         new_spec=_spec("constant"), request_id="req-9")
    assert timeline.add("retarget", at=9000, new_spec=_spec("linear"),
                        request_id="req-9") == first == "rejected_backdated"
    assert _events(timeline).count("noop") == 1


@pytest.mark.parametrize("bad_kwargs", [
    {"anchor": "sideways"},
    {"shape": "sigmoid"},
    {"shape": "exp"},          # D22: the blend adds no shape of its own
])
def test_a_misspelled_vocabulary_word_raises_at_the_seam(bad_kwargs):
    """As `decay` does with its shape: a word outside the vocabulary is a
    caller bug, not a request the run refused."""
    _, timeline = _run()
    with pytest.raises(ValueError):
        timeline.add("retarget", at=100, new_spec=_spec("constant"),
                     **bad_kwargs)


def test_a_retarget_needs_its_new_spec():
    _, timeline = _run()
    with pytest.raises(ValueError, match="new_spec"):
        timeline.add("retarget", at=100)


# ---------------------------------------------------------------------------
# §19.5: persistence by `issued`
# ---------------------------------------------------------------------------

def test_a_future_reservation_survives_a_save_at_an_earlier_step():
    spec, timeline = _run("constant")
    timeline.add("decay", at=9000, length=100)          # issued = at = 9000
    timeline.add("retarget", at=9500, issued=9000, new_spec=_spec("cosine"),
                 length=0)
    timeline.add("retarget", at=9800, issued=9700, new_spec=_spec("linear"),
                 length=0)

    saved = timeline.dump(9000)
    assert [e["kind"] for e in saved] == ["total_steps", "decay", "retarget"]
    assert saved[-1]["at"] == 9500

    restored = ScheduleTimeline()
    restored.load(json.loads(json.dumps(saved)), upto_step=9000)
    assert restored.active_spec(spec, 9400).name == "constant"
    assert restored.active_spec(spec, 9600).name == "cosine"


def test_an_event_with_no_issued_still_truncates_by_its_step():
    """Every event written before R1 -- the read-back must not change them."""
    spec, timeline = _run("constant")
    timeline.add("decay", at=9137)
    assert [e["kind"] for e in timeline.dump(9000)] == ["total_steps"]
    assert "issued" not in timeline.dump(9137)[-1]


def test_a_reloaded_timeline_with_a_retarget_reproduces_the_curve_exactly():
    spec, timeline = _run("plateau_cosine_floor", 20, TOTAL, PLATEAU)
    timeline.add("decay", at=1000, length=800)
    timeline.add("retarget", at=1400, new_spec=_spec("wsd", 100, TOTAL,
                                                     {"lr_floor_ratio": 0.1}),
                 length=600, shape="cosine", gain=0.9)
    timeline.add("cancel", at=3000)
    timeline.add("total_steps", at=4000, value=2 * TOTAL)
    steps = range(0, 2 * TOTAL + 1, 97)
    expected = _curve(spec, timeline, steps)

    restored = _reloaded(timeline)
    assert _curve(spec, restored, steps) == expected


# ---------------------------------------------------------------------------
# Invariant 2: the multiplier is a pure function of (step, events)
# ---------------------------------------------------------------------------

def test_evaluation_order_does_not_change_any_value():
    spec, timeline = _run("cosine", 200, TOTAL)
    timeline.add("retarget", at=2000, new_spec=_spec("constant", 300),
                 length=1200, shape="cosine")
    timeline.add("retarget", at=2600, new_spec=_spec("cosine"), length=900,
                 shape="rex", gain=1.2)
    timeline.add("decay", at=6000, length=1000)

    steps = list(range(0, TOTAL + 1, 41))
    sequential = _curve(spec, timeline, steps)

    for seed in (0, 1):
        shuffled = list(steps)
        random.Random(seed).shuffle(shuffled)
        replayed = {s: timeline.multiplier(spec, s) for s in shuffled}
        assert [replayed[s] for s in steps] == sequential


def test_two_timelines_folding_the_same_events_agree_bit_for_bit():
    spec, timeline = _run("cosine", 200, TOTAL)
    timeline.add("retarget", at=2000, new_spec=_spec("wsd", 100, TOTAL,
                                                     {"lr_decay_start_step": 3000}),
                 length=700, shape="rex")
    timeline.add("decay", at=5000, length=500)
    twin = ScheduleTimeline(json.loads(json.dumps(timeline.dump(TOTAL))))
    steps = list(range(0, TOTAL + 1, 29))
    assert _curve(spec, twin, steps) == _curve(spec, timeline, steps)


def _plateau(step: float) -> float:
    """§4.2's plateau curve for W=100, T=10000, ratio 0.85, F=0.25, written
    out. Deliberately does not call the module under test: comparing the lambda
    with `timeline.multiplier` cannot fail, since the lambda IS that call."""
    if step < 100:
        return step / 100.0
    if step <= 8500:
        return 1.0
    q = min(1.0, (step - 8500) / 1500.0)
    return 0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * q))


def test_a_run_with_no_retarget_follows_the_closed_form_curve():
    """The R0/R1 regression condition: with no `retarget` in the list, the
    multiplier is still §4.2's formula, warp included."""
    spec, timeline = _run("plateau_cosine_floor", 100, TOTAL, PLATEAU)
    fn = make_lambda(spec, timeline)
    for step in (0, 50, 99, 100, 2000, 8499, 8500, 9250, 9999, TOTAL):
        assert fn(step) == pytest.approx(_plateau(step)), step

    timeline.add("total_steps", at=4000, value=2 * TOTAL)
    for step in (2000, 4000, 12000, 16000, 18000, 2 * TOTAL):
        # tau(s) = 4000 + (s - 4000) * 6000/16000 past the anchor (§7.2).
        clock = step if step <= 4000 else 4000 + (step - 4000) * 0.375
        assert fn(step) == pytest.approx(_plateau(clock)), step


def test_a_decay_and_a_cancel_with_no_retarget_follow_the_closed_form():
    spec, timeline = _run("plateau_cosine_floor", 100, TOTAL, PLATEAU)
    timeline.add("decay", at=2000, length=500)
    fn = make_lambda(spec, timeline)

    def decaying(step):                      # F + (m_start - F) * k(q), m_start = 1
        q = min(1.0, (step - 2000) / 500.0)
        return 0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * q))

    for step in (2000, 2125, 2250, 2400, 2500, 3000):
        assert fn(step) == pytest.approx(decaying(step)), step

    m_c = decaying(2300)
    timeline.add("cancel", at=2300)          # R = W = 100, linear back to base
    for step in (2300, 2350, 2400, 2500):
        ratio = min(1.0, (step - 2300) / 100.0)
        assert fn(step) == pytest.approx(m_c + (1.0 - m_c) * ratio), step

# ---------------------------------------------------------------------------
# R2: the `groups` selector (D24, §19.3's last row)
# ---------------------------------------------------------------------------

def _grouped(names, name: str = "cosine", W: int = 0, T: int = TOTAL,
             config=None):
    """One spec per named param group off one run spec, sharing a timeline."""
    run, timeline = _run(name, W, T, config)
    specs = {group: run.for_group(group) for group in names}
    # What build_lr_scheduler binds: with identities installed, D37's warning
    # about a selector on a one-spec run does not apply.
    timeline.bind_spec(run, group_specs=list(specs.values()))
    return specs, timeline


def test_a_scoped_retarget_moves_only_the_named_group():
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    untouched, pristine = _grouped(("text_encoder_1",))
    steps = list(range(0, TOTAL + 1, 137))
    expected = _curve(untouched["text_encoder_1"], pristine, steps)

    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        length=0, groups=["unet"]) == "applied"
    assert timeline.active_spec(specs["unet"], 5000).name == "constant"
    assert timeline.active_spec(specs["text_encoder_1"], 5000).name == "cosine"
    assert _curve(specs["unet"], timeline, steps) != expected
    # Bit-identical, not approx: the fold has to skip the event, not re-derive
    # a curve that happens to agree.
    assert _curve(specs["text_encoder_1"], timeline, steps) == expected


def test_a_null_selector_still_reaches_every_group():
    """R1's meaning, which every event written before R2 also has."""
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        length=0, groups=None) == "applied"
    for spec in specs.values():
        assert timeline.active_spec(spec, 5000).name == "constant"
        assert timeline.multiplier(spec, 6000) == pytest.approx(
            timeline.multiplier(spec, 4000))


def test_an_omitted_selector_is_the_same_as_null():
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    timeline.add("retarget", at=4000, new_spec=_spec("constant"), length=0)
    assert timeline.dump(TOTAL)[-1]["groups"] is None
    for spec in specs.values():
        assert timeline.active_spec(spec, 5000).name == "constant"


def test_a_scoped_retarget_applies_to_all_when_the_specs_have_no_identity():
    """§19.3's last row: `lr_group_schedules` off means one spec on every
    group, so the selector is honoured by applying it rather than refused."""
    spec_a, timeline = _run("constant")
    spec_b = _spec("linear")
    assert timeline.add("retarget", at=2000, new_spec=_spec("cosine"),
                        length=0, groups=["unet"],
                        known_groups=("unet", "text_encoder_1")) == "applied"
    for spec in (spec_a, spec_b):
        assert timeline.active_spec(spec, 3000).name == "cosine"
        assert timeline.multiplier(spec, 3000) != pytest.approx(
            timeline.multiplier(spec, 8000))


def test_two_scoped_retargets_on_different_groups_coexist():
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    solo_specs, solo = _grouped(("unet",))
    steps = list(range(0, TOTAL + 1, 137))

    timeline.add("retarget", at=3000, new_spec=_spec("constant"), length=0,
                 groups=["unet"])
    solo.add("retarget", at=3000, new_spec=_spec("constant"), length=0,
             groups=["unet"])
    timeline.add("retarget", at=5000, new_spec=_spec("linear"), length=0,
                 groups=["text_encoder_1"])

    assert timeline.active_spec(specs["unet"], 6000).name == "constant"
    assert timeline.active_spec(specs["text_encoder_1"], 6000).name == "linear"
    # The second event leaves the first group's chain exactly where it was.
    assert (_curve(specs["unet"], timeline, steps)
            == _curve(solo_specs["unet"], solo, steps))


def test_one_selector_naming_both_groups_reaches_both():
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    timeline.add("retarget", at=4000, new_spec=_spec("constant"), length=0,
                 groups=["text_encoder_1", "unet"])
    for spec in specs.values():
        assert timeline.active_spec(spec, 5000).name == "constant"


def test_a_second_retarget_on_the_same_group_still_knows_the_group():
    """The identity has to survive the replacement: the spec an event carries
    is serialized without one (§19.5), so the fold re-stamps it from the curve
    it replaces. Otherwise the second event would read as unscoped."""
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    untouched, pristine = _grouped(("text_encoder_1",))
    steps = list(range(0, TOTAL + 1, 137))
    expected = _curve(untouched["text_encoder_1"], pristine, steps)

    timeline.add("retarget", at=3000, new_spec=_spec("constant"), length=0,
                 groups=["unet"])
    timeline.add("retarget", at=6000, new_spec=_spec("linear"), length=0,
                 groups=["unet"])
    assert timeline.active_spec(specs["unet"], 7000).name == "linear"
    assert timeline.active_spec(specs["text_encoder_1"], 7000).name == "cosine"
    assert _curve(specs["text_encoder_1"], timeline, steps) == expected


def test_an_unscoped_retarget_after_a_scoped_one_reaches_both_chains():
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    timeline.add("retarget", at=3000, new_spec=_spec("constant"), length=0,
                 groups=["unet"])
    timeline.add("retarget", at=6000, new_spec=_spec("linear"), length=0)
    for spec in specs.values():
        assert timeline.active_spec(spec, 7000).name == "linear"


def test_a_group_name_matches_case_insensitively():
    """`lr_group_schedules` resolves its component names case-folded, so a
    selector that passed rule 7 cannot then miss the group it named."""
    specs, timeline = _grouped(("Text_Encoder_1",))
    timeline.add("retarget", at=4000, new_spec=_spec("constant"), length=0,
                 groups=["text_encoder_1"])
    assert timeline.active_spec(specs["Text_Encoder_1"], 5000).name == "constant"


def test_a_scoped_retarget_blends_only_the_named_group():
    """The blend, not just the spec swap: a length > 0 must leave the other
    group's realized multiplier untouched for the whole blend."""
    specs, timeline = _grouped(("unet", "text_encoder_1"), "constant")
    untouched, pristine = _grouped(("text_encoder_1",), "constant")
    timeline.add("retarget", at=2000, new_spec=_spec("cosine"), length=1000,
                 gain=0.5, groups=["unet"])
    for step in range(2100, 3200, 100):   # 2000 itself is w = 0, i.e. m_old
        assert timeline.multiplier(specs["unet"], step) < 1.0
        assert (timeline.multiplier(specs["text_encoder_1"], step)
                == pristine.multiplier(untouched["text_encoder_1"], step))


def test_rule_7_still_refuses_an_unknown_group_and_moves_no_curve():
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    steps = list(range(0, TOTAL + 1, 137))
    before = {name: _curve(spec, timeline, steps)
              for name, spec in specs.items()}
    assert timeline.add(
        "retarget", at=4000, new_spec=_spec("constant"),
        groups=["unet", "typo_encoder"],
        known_groups=("unet", "text_encoder_1")) == "rejected_unknown_group"
    assert timeline.dump(TOTAL)[-1]["kind"] == "noop"
    for name, spec in specs.items():
        assert _curve(spec, timeline, steps) == before[name]


def test_the_serialized_spec_carries_no_group_identity():
    """Invariant 15's neighbour: the payload is the SHAPE. Who a retarget
    reaches is the event's `groups`, so an identity inside the spec could claim
    a group the spec was never installed on."""
    assert "group" not in _spec("cosine").for_group("unet").to_dict()
    specs, timeline = _grouped(("unet",))
    timeline.add("retarget", at=4000, new_spec=specs["unet"], length=0,
                 groups=["unet"])
    assert "group" not in timeline.dump(TOTAL)[-1]["spec"]


def test_a_stale_identity_in_a_payload_does_not_redirect_the_event():
    """A hand-written (or newer-build) payload carrying a `group` key is folded
    onto the curve the selector chose, not onto the one the payload names."""
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    payload = _spec("constant").to_dict()
    payload["group"] = "text_encoder_1"
    timeline.add("retarget", at=4000, new_spec=payload, length=0,
                 groups=["unet"])
    assert timeline.active_spec(specs["unet"], 5000).group == "unet"
    assert timeline.active_spec(specs["text_encoder_1"], 5000).name == "cosine"


def test_a_scoped_retarget_survives_a_save_and_reload_per_group():
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    timeline.add("retarget", at=3000, new_spec=_spec("constant"), length=800,
                 groups=["unet"])
    twin = ScheduleTimeline(json.loads(json.dumps(timeline.dump(TOTAL))))
    steps = list(range(0, TOTAL + 1, 61))
    for spec in specs.values():
        assert _curve(spec, twin, steps) == _curve(spec, timeline, steps)


def test_a_decay_command_still_reaches_a_group_a_retarget_skipped():
    """§17.3: the event list is shared and `decay` has no selector, so scoping
    a retarget must not scope the overlay."""
    specs, timeline = _grouped(("unet", "text_encoder_1"), "constant")
    timeline.add("retarget", at=2000, new_spec=_spec("linear"), length=0,
                 groups=["unet"])
    timeline.add("decay", at=4000, length=500, spec=specs["text_encoder_1"])
    assert timeline.multiplier(specs["text_encoder_1"], 4500) == pytest.approx(0.0)

# ---------------------------------------------------------------------------
# D34: an empty selector names no group
# ---------------------------------------------------------------------------

def test_an_empty_selector_is_refused():
    """`null` is the only spelling of "every group". An array the UI sent with
    nothing checked, read as "all", is the silent global replacement D24 is
    there to prevent."""
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    steps = list(range(0, TOTAL + 1, 137))
    before = {name: _curve(spec, timeline, steps)
              for name, spec in specs.items()}
    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        length=0, groups=[]) == "rejected_empty_group_selector"
    event = timeline.dump(TOTAL)[-1]
    assert event["kind"] == "noop" and event["refused_kind"] == "retarget"
    for name, spec in specs.items():
        assert _curve(spec, timeline, steps) == before[name]


def test_an_empty_selector_is_refused_before_the_name_check():
    """It is refused whether or not the caller said what components exist:
    there is no name in it to check against `known_groups`."""
    _, timeline = _run()
    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        groups=[], known_groups=("unet",)
                        ) == "rejected_empty_group_selector"


def test_an_empty_selector_stored_before_d34_reaches_no_group():
    """A refused event never reaches the fold, so this can only arrive from an
    older state file. "Every group" is the one reading D34 rules out."""
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    steps = list(range(0, TOTAL + 1, 137))
    before = {name: _curve(spec, timeline, steps)
              for name, spec in specs.items()}
    legacy = dict(timeline.dump(TOTAL)[0])          # the total_steps anchor
    twin = ScheduleTimeline([legacy, {
        "kind": "retarget", "at": 4000, "issued": 4000, "seq": 1,
        "spec": _spec("constant").to_dict(), "anchor": "restart",
        "gain": 1.0, "length": 0, "shape": "linear", "groups": []}])
    for name, spec in specs.items():
        assert _curve(spec, twin, steps) == before[name]


# ---------------------------------------------------------------------------
# D35: one spelling of a component name
# ---------------------------------------------------------------------------

def test_rule_7_accepts_a_group_name_in_another_case():
    """§10.1 resolves the mapping case-folded, so an exact-match acceptance
    could admit a name that then addresses nothing."""
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        length=0, groups=["UNet"],
                        known_groups=("unet", "text_encoder_1")) == "applied"
    assert timeline.active_spec(specs["unet"], 5000).name == "constant"
    assert timeline.active_spec(specs["text_encoder_1"], 5000).name == "cosine"


def test_rule_7_case_folds_the_component_list_too():
    specs, timeline = _grouped(("Unet",))
    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        length=0, groups=["unet"],
                        known_groups=("UNET",)) == "applied"
    assert timeline.active_spec(specs["Unet"], 5000).name == "constant"


def test_rule_7_still_refuses_a_name_that_is_not_a_case_variant():
    spec, timeline = _run()
    _assert_refused(timeline, spec, timeline.add(
        "retarget", at=4000, new_spec=_spec("constant"), groups=["Typo_Encoder"],
        known_groups=("unet", "text_encoder_1")), "rejected_unknown_group")


# ---------------------------------------------------------------------------
# D37: a selector on a run whose groups all share one spec
# ---------------------------------------------------------------------------

def test_a_selector_on_an_ungrouped_run_is_accepted_with_a_warning(capsys):
    spec, timeline = _run("constant")
    assert timeline.add("retarget", at=2000, new_spec=_spec("cosine"),
                        length=0, groups=["unet"]) == "applied"
    assert (timeline.dump(TOTAL)[-1]["warning"]
            == WARN_SELECTOR_ON_UNGROUPED_RUN)
    out = capsys.readouterr().out
    assert WARN_SELECTOR_ON_UNGROUPED_RUN in out and "unet" in out
    assert timeline.active_spec(spec, 3000).name == "cosine"


def test_the_warning_is_decided_at_acceptance_not_at_evaluation(capsys):
    """D37/D25: a reservation for a future step must not change meaning if the
    config is edited between the order and its effect."""
    _, timeline = _run("constant")
    timeline.add("retarget", at=9000, issued=1000, new_spec=_spec("cosine"),
                 length=0, groups=["unet"])
    assert WARN_SELECTOR_ON_UNGROUPED_RUN in capsys.readouterr().out
    assert (timeline.dump(1000)[-1]["warning"]
            == WARN_SELECTOR_ON_UNGROUPED_RUN), "and it is saved with the event"


def test_a_grouped_run_takes_a_selector_without_the_warning(capsys):
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    assert timeline.add("retarget", at=4000, new_spec=_spec("constant"),
                        length=0, groups=["unet"]) == "applied"
    assert "warning" not in timeline.dump(TOTAL)[-1]
    assert WARN_SELECTOR_ON_UNGROUPED_RUN not in capsys.readouterr().out


def test_an_unscoped_retarget_never_warns(capsys):
    _, timeline = _run("constant")
    timeline.add("retarget", at=2000, new_spec=_spec("cosine"), length=0)
    assert "warning" not in timeline.dump(TOTAL)[-1]
    assert WARN_SELECTOR_ON_UNGROUPED_RUN not in capsys.readouterr().out


def test_a_refused_selector_does_not_also_warn(capsys):
    _, timeline = _run("constant")
    timeline.add("retarget", at=2000, new_spec=_spec("constant"), groups=[])
    assert "warning" not in timeline.dump(TOTAL)[-1]
    assert WARN_SELECTOR_ON_UNGROUPED_RUN not in capsys.readouterr().out

# ---------------------------------------------------------------------------
# D38: a decay/cancel is scored per group, not on the representative spec
# ---------------------------------------------------------------------------

SHORT = 1000


def test_a_group_the_retarget_skipped_still_gets_the_decay():
    """The representative spec is UNSTAMPED, so it absorbs a scoped retarget no
    param group received. Scoring the command on it alone refused the decay
    inside the retarget's new warmup and stored a `noop`, which deleted the
    decay for the group that was never named."""
    specs, timeline = _grouped(("unet", "text_encoder_1"), "constant", 0, SHORT)
    plain_specs, plain = _grouped(("text_encoder_1",), "constant", 0, SHORT)

    timeline.add("retarget", at=400, length=0, groups=["unet"],
                 new_spec=_spec("constant", 200, SHORT))
    assert timeline.add("decay", at=450) == "applied"
    assert timeline.dump(SHORT)[-1]["kind"] == "decay"
    plain.add("decay", at=450)

    te = specs["text_encoder_1"]
    steps = (450, 600, 800, 999)
    assert [round(timeline.multiplier(te, s), 4) for s in steps] == [
        1.0, 0.8274, 0.2923, 0.0]
    assert ([timeline.multiplier(te, s) for s in range(0, SHORT + 1, 7)]
            == [plain.multiplier(plain_specs["text_encoder_1"], s)
                for s in range(0, SHORT + 1, 7)])
    # And the group the command WAS refused for keeps climbing its new ramp.
    assert timeline.multiplier(specs["unet"], 450) == pytest.approx(0.25)
    assert timeline.multiplier(specs["unet"], 800) == 1.0


def test_a_cancel_takes_its_recovery_length_from_each_group_s_own_spec():
    """The recovery length was baked from the representative spec, so a scoped
    retarget onto a longer warmup stretched every group's recovery."""
    specs, timeline = _grouped(("unet", "text_encoder_1"), "cosine", 50, SHORT)
    plain_specs, plain = _grouped(("text_encoder_1",), "cosine", 50, SHORT)

    for line in (timeline, plain):
        line.add("decay", at=300, length=200)
    timeline.add("retarget", at=310, length=0, groups=["unet"],
                 new_spec=_spec("cosine", 400, SHORT))
    assert timeline.add("cancel", at=500) == "applied"
    plain.add("cancel", at=500)
    assert "length" not in timeline.dump(SHORT)[-1], "nothing to bake: R = W"

    te = specs["text_encoder_1"]
    assert [round(timeline.multiplier(te, s), 4) for s in (520, 550, 600)] == [
        0.2033, 0.4587, 0.3773]
    assert ([timeline.multiplier(te, s) for s in range(0, SHORT + 1, 7)]
            == [plain.multiplier(plain_specs["text_encoder_1"], s)
                for s in range(0, SHORT + 1, 7)])


def test_a_command_every_group_refuses_is_still_a_noop():
    """The `noop` conversion is not gone, only narrowed to unanimity."""
    specs, timeline = _grouped(("unet", "text_encoder_1"), "constant", 200,
                               SHORT)
    assert timeline.add("decay", at=100) == "rejected_during_warmup"
    event = timeline.dump(SHORT)[-1]
    assert event["kind"] == "noop" and event["refused_kind"] == "decay"
    for spec in specs.values():
        assert timeline.state_at(spec, 300).code == STATE_BASE


def test_a_caller_supplied_length_and_shape_are_still_baked():
    """§5.4 for what the caller fixed: the RPC path passes both from config."""
    _, timeline = _grouped(("unet",), "constant", 0, SHORT)
    timeline.add("decay", at=300, length=200, shape="linear")
    event = timeline.dump(SHORT)[-1]
    assert event["length"] == 200 and event["shape"] == "linear"


# ---------------------------------------------------------------------------
# D39: the reach a scoped event was accepted with is frozen on the event
# ---------------------------------------------------------------------------

LONG = 10000


def test_a_future_dated_selector_keeps_its_reach_across_a_grouped_resume():
    """D39: accepted on a run with one schedule, warned as "every group". If
    adding `lr_group_schedules` before it fires narrowed it to `unet`, the
    warning saved beside it would be false about what the run did."""
    run_spec, timeline = _run("cosine", 0, LONG)
    assert timeline.add("retarget", at=9000, issued=1000, length=0,
                        new_spec=_spec("constant", 0, LONG),
                        groups=["unet"]) == "applied"
    event = timeline.dump(1000)[-1]
    assert event["scope"] == "all" and event["groups"] == ["unet"]

    # The operator adds lr_group_schedules and resumes: same events, but the
    # specs now carry identities.
    resumed = ScheduleTimeline(json.loads(json.dumps(timeline.dump(1000))))
    specs = {name: run_spec.for_group(name)
             for name in ("unet", "text_encoder_1")}
    resumed.bind_spec(run_spec, group_specs=list(specs.values()))
    assert resumed.multiplier(specs["unet"], 8999) == pytest.approx(0.024520,
                                                                   abs=5e-6)
    for spec in specs.values():
        assert resumed.multiplier(spec, 9500) == pytest.approx(0.024472,
                                                               abs=5e-6)


def test_a_scoped_event_on_a_grouped_run_carries_no_frozen_scope():
    """Only the decided-all case is frozen: with identities present the
    selector itself is the record, and it is honoured group by group."""
    specs, timeline = _grouped(("unet", "text_encoder_1"))
    timeline.add("retarget", at=4000, length=0, groups=["unet"],
                 new_spec=_spec("constant"))
    assert "scope" not in timeline.dump(TOTAL)[-1]
    assert timeline.active_spec(specs["text_encoder_1"], 5000).name == "cosine"


# ---------------------------------------------------------------------------
# D41: bake the fallback when the bound groups agree
# ---------------------------------------------------------------------------

def test_the_recovery_length_is_baked_when_every_group_agrees():
    """§10.1 keeps the numeric parameters run-wide, so agreement is the normal
    case and §5.4's baking survives D38's per-group scoring."""
    specs, timeline = _grouped(("unet", "text_encoder_1"), "cosine", 50, SHORT)
    timeline.add("decay", at=300, length=200)
    assert timeline.add("cancel", at=500) == "applied"
    assert timeline.dump(SHORT)[-1]["length"] == 50
    for spec in specs.values():
        assert timeline.state_at(spec, 549).code == STATE_RECOVERING
        assert timeline.state_at(spec, 550).code == STATE_BASE


def test_the_decay_shape_is_baked_when_every_group_agrees():
    specs, timeline = _grouped(("unet", "text_encoder_1"), "wsd", 0, SHORT,
                               {"lr_decay_shape": "linear"})
    assert timeline.add("decay", at=300) == "applied"
    assert timeline.dump(SHORT)[-1]["shape"] == "linear"


def test_a_warmup_edit_across_a_resume_does_not_reshape_a_past_cancel():
    """The guarantee D38's narrowing had lost: the recovery length went into
    the event, so re-resolving the spec with a different lr_warmup_steps cannot
    stretch a cancel that already happened."""
    spec, timeline = _run("cosine", 50, SHORT)
    timeline.add("decay", at=300, length=200)
    assert timeline.add("cancel", at=500) == "applied"
    assert timeline.dump(SHORT)[-1]["length"] == 50

    # Edited to 200, not past the decay's step: a warmup that swallowed the
    # decay would refuse it at fold time and prove nothing about the recovery.
    edited = _spec("cosine", 200, SHORT)
    resumed = ScheduleTimeline(json.loads(json.dumps(timeline.dump(SHORT))))
    resumed.bind_spec(edited)
    assert resumed.state_at(edited, 549).code == STATE_RECOVERING
    assert resumed.state_at(edited, 550).code == STATE_BASE


# ---------------------------------------------------------------------------
# D43: the reported result does not depend on param-group order
# ---------------------------------------------------------------------------

def _bound(order, name="constant", W=0, T=1000, config=None):
    _, timeline = _run(name, W, T, config)
    timeline.bind_spec(order[0], group_specs=list(order))
    return timeline


def test_a_mixed_outcome_reports_the_effect_not_whichever_group_sorts_first():
    """`unet` has nothing to cancel and `text_encoder_1` has a scheduled decay
    to disarm. Ranking by list position told the operator "nothing happened"
    while a decay was permanently disarmed -- and `poll_lr_schedule_commands`
    does not count that string as applied."""
    config = {"lr_decay_start_step": 700, "lr_floor_ratio": 0.0}
    unet = _spec("constant", 0, SHORT, config).for_group("unet")
    te = _spec("wsd", 0, SHORT, config).for_group("text_encoder_1")
    for order in ((unet, te), (te, unet)):
        timeline = _bound(order, "constant", 0, SHORT, config)
        assert timeline.add("cancel", at=300) == "disarmed_scheduled_decay"
        assert timeline.state_at(te, 400).decay_disarmed is True
        assert timeline.state_at(unet, 400).code == STATE_BASE


def test_a_command_one_group_refuses_reports_applied_in_either_order():
    ramping = _spec("constant", 200, SHORT).for_group("unet")
    ready = _spec("constant", 0, SHORT).for_group("text_encoder_1")
    for order in ((ramping, ready), (ready, ramping)):
        timeline = _bound(order, "constant", 0, SHORT)
        assert timeline.add("decay", at=100) == "applied"
        assert timeline.state_at(ready, 100).code == STATE_DECAYING
        assert timeline.state_at(ramping, 100).code == STATE_BASE


def test_a_command_every_group_refuses_reports_the_same_refusal_either_way():
    a = _spec("constant", 200, SHORT).for_group("unet")
    b = _spec("constant", 300, SHORT).for_group("text_encoder_1")
    for order in ((a, b), (b, a)):
        timeline = _bound(order, "constant", 200, SHORT)
        assert timeline.add("decay", at=100) == "rejected_during_warmup"
        assert timeline.dump(SHORT)[-1]["kind"] == "noop"


# ---------------------------------------------------------------------------
# D42: the run's start_decay parameters are not part of a retarget's payload
# ---------------------------------------------------------------------------

COMMAND_CONFIG = {"lr_decay_steps": 200, "lr_decay_shape": "linear",
                  "lr_floor_ratio": 0.0}


def test_a_payload_cannot_carry_the_runs_decay_command_parameters():
    spec = _spec("wsd", 0, SHORT, COMMAND_CONFIG)
    assert (spec.command_decay_length, spec.command_decay_shape) == (
        200, "linear")
    payload = spec.to_dict()
    assert "command_decay_length" not in payload
    assert "command_decay_shape" not in payload


def test_a_scoped_retarget_does_not_hand_its_decay_parameters_to_other_groups():
    """The RPC path reads `length`/`shape` off the spec in force
    (`base_trainer.poll_lr_schedule_commands`), and the representative spec
    absorbs every scoped retarget. If the payload carried them, a retarget
    scoped to `unet` would decay `text_encoder_1` with 321 steps of `rex`."""
    specs, timeline = _grouped(("unet", "text_encoder_1"), "constant", 0, SHORT,
                               COMMAND_CONFIG)
    run_spec = _spec("constant", 0, SHORT, COMMAND_CONFIG)
    timeline.add("retarget", at=300, length=0, groups=["unet"],
                 new_spec=_spec("constant", 0, SHORT,
                                {"lr_decay_steps": 321,
                                 "lr_decay_shape": "rex",
                                 "lr_floor_ratio": 0.0}))

    active = timeline.active_spec(run_spec, 400)
    assert (active.command_decay_length, active.command_decay_shape) == (
        200, "linear")
    timeline.add("decay", at=400, length=active.command_decay_length,
                 shape=active.command_decay_shape)
    te = specs["text_encoder_1"]
    assert timeline.multiplier(te, 500) == pytest.approx(0.5)
    assert timeline.multiplier(te, 600) == pytest.approx(0.0)
