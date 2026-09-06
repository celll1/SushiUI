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
    # re-derived from does, so the restored spec evaluates identically.
    expected = (replace(spec, decay_start_step=None)
                if spec.decay_start_ratio is not None else spec)
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
