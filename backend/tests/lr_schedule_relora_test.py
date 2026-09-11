"""Guard: ReLoRA's restart schedule is a registry curve, and a resume no longer
reshapes the past.

P4 of docs/guides/LR_SCHEDULER_DESIGN.md. ReLoRA was the last scheduler in this
project that was not a ``LambdaLR``: it installed
``relora_scheduler.CosineWithMultipleWarmups``, an ``_LRScheduler`` subclass
with its own step counter and its own restart list. Three generic mechanisms
degraded for exactly those runs -- the resume fast-forward replayed
``scheduler.step()`` ``global_step`` times, the post-reset re-warmup skipped it,
and ``lr_utils.reassert_config_lr`` wrote the base LR with a multiplier of 1.0.

What this file pins down:

* the shipped ``relora`` curve is BIT-IDENTICAL to the deleted class, which is
  transcribed below rather than imported, with ``min_lr_ratio = 0`` reading as
  ``lr_floor_ratio = 0``. Restarts are delivered in ARRIVAL order -- the list
  the old scheduler actually held while running, one entry per merge it had
  already reached;
* the correctness change of §17.3: the old ``get_lr`` took each segment's
  terminus from the whole registered list INCLUDING restarts in the future, so
  a resume that re-registered every past merge at once retroactively shortened
  cosine segments the run had already trained through. Both halves are tested:
  that the old code really does that, and that the new one does not;
* a restart placed by an epoch-unit merge survives a resume. The old
  re-registration could only recompute step-unit merge positions and said so;
* the three fallbacks are unreachable for ReLoRA now, and still work as guards.

CPU-only and hermetic: real ``torch.optim`` objects over 4-element parameters,
no model, no dataset, no GPU.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/lr_schedule_relora_test.py -v
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.base_trainer import (
    BaseTrainer,
    install_lr_schedule_events,
    resolve_lr_schedule_spec,
)
from core.training.lr_schedules import (
    INTERNAL_SCHEDULER_NAMES,
    LR_SCHEDULER_NAMES,
    ScheduleTimeline,
    build_lr_scheduler,
    make_lambda,
    resolve_spec,
)
from core.training.lr_utils import reassert_config_lr
from core.training.relora_trainer import ReLoRATrainer

BASE_LR = 1e-4



class LegacyCosineWithMultipleWarmups:
    """A copy of ``relora_scheduler.py``'s multiplier, module and all deleted.

    Transcribed rather than imported so that a silent change to the shipped
    curve fails here instead of being followed.
    """

    def __init__(self, total_steps, initial_warmup_steps=0,
                 restart_warmup_steps=100, min_lr_ratio=0.0):
        self.total_steps = total_steps
        self.initial_warmup_steps = initial_warmup_steps
        self.restart_warmup_steps = restart_warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.restart_steps = []

    def add_restart(self, step):
        self.restart_steps.append(step)
        self.restart_steps.sort()

    def multiplier(self, step):
        cycle_start = 0
        warmup_steps = self.initial_warmup_steps

        for restart_step in self.restart_steps:
            if step >= restart_step:
                cycle_start = restart_step
                warmup_steps = self.restart_warmup_steps
            else:
                break

        steps_in_cycle = step - cycle_start

        next_restart = self.total_steps
        for restart_step in self.restart_steps:
            if restart_step > cycle_start:
                next_restart = restart_step
                break

        cycle_length = next_restart - cycle_start

        if steps_in_cycle < warmup_steps:
            if warmup_steps > 0:
                lr_mult = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (
                    steps_in_cycle / warmup_steps)
            else:
                lr_mult = 1.0
        else:
            decay_steps = cycle_length - warmup_steps
            if decay_steps > 0:
                progress = (steps_in_cycle - warmup_steps) / decay_steps
                progress = min(progress, 1.0)
                lr_mult = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * progress))
            else:
                lr_mult = 1.0

        return lr_mult


def _legacy_arrival_order(step, *, T, W, Wr, restarts, min_lr_ratio=0.0):
    """The old multiplier at ``step`` with the list it held AT that step."""
    legacy = LegacyCosineWithMultipleWarmups(
        total_steps=T, initial_warmup_steps=W, restart_warmup_steps=Wr,
        min_lr_ratio=min_lr_ratio)
    for restart in restarts:
        if restart <= step:
            legacy.add_restart(restart)
    return legacy.multiplier(step)


def _legacy_all_registered(T, W, Wr, restarts, min_lr_ratio=0.0):
    """The old multiplier after a resume re-registered the whole list."""
    legacy = LegacyCosineWithMultipleWarmups(
        total_steps=T, initial_warmup_steps=W, restart_warmup_steps=Wr,
        min_lr_ratio=min_lr_ratio)
    for restart in restarts:
        legacy.add_restart(restart)
    return legacy.multiplier



def _optimizer(lr=BASE_LR, groups=1):
    params = [torch.nn.Parameter(torch.zeros(4)) for _ in range(groups)]
    return torch.optim.AdamW([{"params": [p]} for p in params], lr=lr)


def _spec(W, T, Wr, floor=0.0):
    return resolve_spec({"lr_floor_ratio": floor}, warmup_steps=W,
                        total_steps=T, name="relora", restart_warmup_steps=Wr)


def _ours(W, T, Wr, restarts=(), floor=0.0, upto=None):
    """The shipping lambda, with ``restarts`` already on its timeline."""
    spec = _spec(W, T, Wr, floor)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    for restart in restarts:
        if upto is None or restart <= upto:
            timeline.add("restart", at=restart)
    return make_lambda(spec, timeline)


# (W, Wr, T, restarts) -- a step-unit run, an epoch-unit one with irregular
# merges, a restart during the initial warmup, and no restarts at all.
_CASES = [
    (0, 100, 2000, (500, 1000, 1500)),
    (200, 100, 2000, (500, 1000, 1500)),
    (200, 100, 2000, (137, 289, 631, 1902)),
    (200, 100, 2000, (50, 500)),
    (200, 0, 2000, (500, 1000)),
    (200, 100, 2000, ()),
    (0, 1, 37, (5, 6, 30)),
]


@pytest.mark.parametrize("W,Wr,T,restarts", _CASES)
def test_bit_identical_to_the_deleted_scheduler_in_arrival_order(W, Wr, T, restarts):
    ours = _ours(W, T, Wr, restarts)
    for step in range(0, T + 200):
        got = ours(step)
        want = _legacy_arrival_order(step, T=T, W=W, Wr=Wr, restarts=restarts)
        assert got == want, f"step {step}: {got!r} != {want!r}"


def test_a_live_restart_is_the_same_curve_whenever_it_is_delivered():
    """Arrival order is not an artefact of the test: our lambda reads only
    restarts at or before the step it is asked about, so pre-loading the whole
    list (what a resume does) cannot change any value."""
    W, Wr, T, restarts = 200, 100, 2000, (500, 1000, 1500)
    full = _ours(W, T, Wr, restarts)
    for step in range(0, T + 200):
        live = _ours(W, T, Wr, restarts, upto=step)
        assert full(step) == live(step), step



def test_the_old_scheduler_really_did_shorten_a_segment_on_resume():
    """The defect this phase fixes, demonstrated against the transcription.

    While running, the segment that began at 500 decayed towards the run's END
    (2000). After a resume re-registered 1000 and 1500, the same position reads
    as a segment ending at 1000 -- a cosine four times shorter than the one the
    run trained through, and already near its floor.
    """
    W, Wr, T, restarts = 200, 100, 2000, (500, 1000, 1500)
    after_resume = _legacy_all_registered(T, W, Wr, restarts)

    live = _legacy_arrival_order(900, T=T, W=W, Wr=Wr, restarts=restarts)
    resumed = after_resume(900)
    # 300 steps into a 1400-step cosine, against 300 into a 400-step one.
    assert live == pytest.approx(0.8909, abs=1e-4)
    assert resumed == pytest.approx(0.1464, abs=1e-4)

    differing = [s for s in range(0, T)
                 if _legacy_arrival_order(s, T=T, W=W, Wr=Wr, restarts=restarts)
                 != after_resume(s)]
    assert len(differing) > T // 4


def test_a_resume_does_not_reshape_a_segment_the_run_already_trained_through():
    W, Wr, T, restarts = 200, 100, 2000, (500, 1000, 1500)
    ours = _ours(W, T, Wr, restarts)
    for step in range(0, T + 200):
        assert ours(step) == _legacy_arrival_order(
            step, T=T, W=W, Wr=Wr, restarts=restarts), step


def test_a_segment_decays_towards_the_run_end_not_towards_the_next_merge():
    W, Wr, T = 200, 100, 2000
    ours = _ours(W, T, Wr, (500,))
    # Half way from the end of the 500-merge's re-warmup (600) to the run's end.
    midpoint = 600 + (T - 600) // 2
    assert ours(midpoint) == pytest.approx(0.5, abs=1e-9)



def test_restarts_survive_a_dump_and_load_and_a_later_one_is_dropped():
    """``dump(upto)`` truncates like every other event: rewinding to an earlier
    checkpoint un-does the merges that came after it."""
    W, Wr, T = 200, 100, 2000
    spec = _spec(W, T, Wr)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    for at in (137, 289, 631):
        assert timeline.add("restart", at=at) == "applied"

    saved = timeline.dump(upto_step=400)
    assert [e["at"] for e in saved if e["kind"] == "restart"] == [137, 289]

    restored = ScheduleTimeline()
    restored.load(saved, upto_step=400)
    assert restored.restarts() == [137, 289]

    truncated = _ours(W, T, Wr, (137, 289))
    for step in range(0, 401):
        assert restored.multiplier(spec, step) == truncated(step), step


def test_a_restart_is_recorded_once_per_position():
    """Re-registering a legacy checkpoint's merges must be idempotent: a
    duplicate would be indistinguishable from a second merge."""
    timeline = ScheduleTimeline()
    timeline.set_total_steps(1000)
    assert timeline.add("restart", at=500) == "applied"
    assert timeline.add("restart", at=500) == "ignored_duplicate_restart"
    assert timeline.restarts() == [500]


def test_a_restart_does_not_disturb_the_decay_overlay():
    """The state machine reads decay/cancel only; the base curve reads restarts.
    A merge must not be scored as, or cancel, a runtime decay."""
    W, Wr, T = 100, 50, 1000
    spec = _spec(W, T, Wr)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    timeline.add("restart", at=200, spec=spec)
    assert timeline.state_at(spec, 300).code == 0
    assert timeline.add("decay", at=300, spec=spec, length=100) == "applied"
    timeline.add("restart", at=350, spec=spec)
    assert timeline.state_at(spec, 360).code == 1
    assert timeline.multiplier(spec, 400) == pytest.approx(spec.floor_ratio)


# ---------------------------------------------------------------------------
# An epoch-unit merge survives a resume (the old path lost it)
# ---------------------------------------------------------------------------

class _ResumeHarness:
    """Enough of a trainer for the resume seam and the legacy restart path."""

    _restore_legacy_lr_restarts = ReLoRATrainer._restore_legacy_lr_restarts
    # @staticmethod on BaseTrainer: reading it off the class yields a plain
    # function, which would be rebound as an instance method here.
    _compose_warmup_lambda = staticmethod(BaseTrainer._compose_warmup_lambda)

    def __init__(self, spec, *, saved_events=None, scheduler_step=None,
                 merge_count=0, merge_unit="steps", merge_every=500, gas=1):
        self.log_prefix = "[Test]"
        self.config = {}
        self._grad_accum_steps = gas
        self.lr_schedule_spec = spec
        self.lr_timeline = ScheduleTimeline()
        self.lr_timeline.set_total_steps(spec.total_steps)
        self._resume_lr_schedule_events = saved_events
        self._resume_scheduler_step = scheduler_step
        self._resume_scheduler_interval = gas
        self.merge_count = merge_count
        self.relora_merge_unit = merge_unit
        self.relora_merge_every = merge_every
        self.optimizer = _optimizer()
        self.lr_scheduler = build_lr_scheduler(self.optimizer, spec,
                                               self.lr_timeline)
        self.fused_optimizer_groups = None


def test_epoch_unit_restarts_survive_a_resume():
    """The old ``_restore_scheduler_restarts`` could only recompute
    ``i * merge_every``, so an epoch-unit run resumed with NO restarts at all
    and its LR jumped back onto the un-restarted cosine."""
    W, Wr, T = 200, 100, 2000
    spec = _spec(W, T, Wr)
    live = ScheduleTimeline()
    live.set_total_steps(spec.total_steps)
    for at in (137, 289, 631):     # first batch of epochs 1, 2, 3
        live.add("restart", at=at)
    saved = live.dump(upto_step=700)

    harness = _ResumeHarness(spec, saved_events=saved, scheduler_step=700,
                             merge_count=3, merge_unit="epochs")
    install_lr_schedule_events(harness, 700)
    assert harness.lr_timeline.restarts() == [137, 289, 631]

    resumed = make_lambda(spec, harness.lr_timeline)
    for step in range(0, T + 100):
        assert resumed(step) == _legacy_arrival_order(
            step, T=T, W=W, Wr=Wr, restarts=(137, 289, 631)), step


@pytest.mark.parametrize("saved_events", [
    None,                                                  # no timeline at all
    [{"kind": "total_steps", "at": 0, "value": 2000}],      # P1..P3 ReLoRA run
])
def test_a_pre_p4_checkpoint_rebuilds_step_unit_restarts_from_merge_count(saved_events):
    W, Wr, T = 200, 100, 2000
    spec = _spec(W, T, Wr)
    harness = _ResumeHarness(spec, saved_events=saved_events, scheduler_step=1600,
                             merge_count=3, merge_unit="steps", merge_every=500)
    install_lr_schedule_events(harness, 1600)
    assert harness.lr_timeline.restarts() == [500, 1000, 1500]


def test_a_pre_p4_epoch_checkpoint_cannot_place_its_merges(capsys):
    """Nothing records where an epoch-unit merge fell, so the honest answer is
    to say so rather than to invent positions."""
    spec = _spec(200, 2000, 100)
    harness = _ResumeHarness(spec, saved_events=None, scheduler_step=1600,
                             merge_count=3, merge_unit="epochs")
    install_lr_schedule_events(harness, 1600)
    assert harness.lr_timeline.restarts() == []
    assert "merges by epoch" in capsys.readouterr().out


def test_the_legacy_rebuild_does_not_run_when_events_carry_the_restarts():
    """Both paths at once would double-count a merge as two."""
    spec = _spec(200, 2000, 100)
    live = ScheduleTimeline()
    live.set_total_steps(spec.total_steps)
    live.add("restart", at=500)
    harness = _ResumeHarness(spec, saved_events=live.dump(1000),
                             scheduler_step=1000, merge_count=1,
                             merge_unit="steps", merge_every=500)
    install_lr_schedule_events(harness, 1000)
    assert harness.lr_timeline.restarts() == [500]



def test_relora_builds_a_lambdalr_with_one_lambda_per_group():
    spec = _spec(200, 2000, 100)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    optimizer = _optimizer(groups=3)
    scheduler = build_lr_scheduler(optimizer, spec, timeline)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert len(scheduler.lr_lambdas) == len(optimizer.param_groups) == 3


def test_the_fast_forward_does_not_replay_for_relora():
    """``_fast_forward_one_lr_scheduler``'s O(step) replay is the branch for a
    scheduler that is not a LambdaLR. ReLoRA no longer reaches it."""
    spec = _spec(200, 2000, 100)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    timeline.add("restart", at=500)
    optimizer = _optimizer()
    scheduler = build_lr_scheduler(optimizer, spec, timeline)

    def _explode():
        raise AssertionError("the replay fallback ran for a LambdaLR")

    scheduler.step = _explode
    BaseTrainer._fast_forward_one_lr_scheduler(scheduler, 900)
    assert scheduler.last_epoch == 900
    assert optimizer.param_groups[0]["lr"] == BASE_LR * make_lambda(
        spec, timeline)(900)


def test_the_post_reset_rewarmup_arms_a_relora_schedule():
    """It used to count ReLoRA as a "non-LambdaLR skipped" and resume a zeroed
    optimizer at the full scheduled LR."""
    spec = _spec(200, 2000, 100)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    timeline.add("restart", at=500)

    harness = _ResumeHarness(spec, scheduler_step=900)
    harness.config = {"rewarmup_on_optimizer_reset": True}
    harness.optimizer_warmup_steps = 200
    harness.lr_timeline = timeline
    harness.lr_scheduler = build_lr_scheduler(harness.optimizer, spec, timeline)
    harness._rearm_warmup_after_optimizer_reset = (
        BaseTrainer._rearm_warmup_after_optimizer_reset.__get__(harness))
    harness._optimizer_state_partially_fresh = False

    bare = make_lambda(spec, timeline)
    assert harness._rearm_warmup_after_optimizer_reset(900) is True
    armed = harness.lr_scheduler.lr_lambdas[0]
    assert armed(900) == 0.0
    assert armed(1000) == pytest.approx(bare(1000) * 0.5)
    assert armed(1100) == bare(1100)


def test_reassert_config_lr_applies_the_relora_multiplier():
    """With no ``lr_lambdas`` the multiplier was 1.0, so a resume mid-re-warmup
    wrote the full base LR into every group."""
    spec = _spec(200, 2000, 100)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    timeline.add("restart", at=500)
    optimizer = _optimizer()
    scheduler = build_lr_scheduler(optimizer, spec, timeline)
    BaseTrainer._fast_forward_one_lr_scheduler(scheduler, 550)

    reassert_config_lr(optimizer, scheduler, 2e-4, verbose=False)
    expected = 2e-4 * make_lambda(spec, timeline)(550)
    assert optimizer.param_groups[0]["lr"] == expected
    assert expected < 2e-4 * 0.6, "550 is inside the restart's re-warmup"


# ---------------------------------------------------------------------------
# Axes and the shape of the segment
# ---------------------------------------------------------------------------

class _AxisProbe:
    def __init__(self, gas, warmup, restart_warmup):
        self.log_prefix = "[Test]"
        self.config = {}
        self._grad_accum_steps = gas
        self.optimizer_warmup_steps = warmup
        self.restart_warmup_steps = restart_warmup


def test_gas_1_leaves_every_relora_length_as_configured():
    spec = resolve_lr_schedule_spec(_AxisProbe(1, 400, 100), "relora", 4000)
    assert (spec.warmup_steps, spec.total_steps,
            spec.relora_restart_warmup_steps) == (400, 4000, 100)


@pytest.mark.parametrize("gas", [2, 4])
def test_every_relora_length_lands_on_the_scheduler_axis(gas):
    """The class this replaces passed total_steps and both warmups as global
    steps while its counter advanced once per optimizer update, so at gas>1 the
    curve was defined over `gas` times the positions the run ever reached."""
    spec = resolve_lr_schedule_spec(_AxisProbe(gas, 400, 100), "relora", 4000)
    assert spec.total_steps == 4000 // gas
    assert spec.warmup_steps == 400 // gas
    assert spec.relora_restart_warmup_steps == 100 // gas

    # The same fractions of the schedule, whatever the accumulation width.
    at_gas_1 = _ours(400, 4000, 100, (1000, 2000))
    scaled = _ours(spec.warmup_steps, spec.total_steps,
                   spec.relora_restart_warmup_steps,
                   (1000 // gas, 2000 // gas))
    for step in range(0, 4000, gas * 13):
        assert scaled(step // gas) == pytest.approx(at_gas_1(step), abs=1e-9)


def test_a_restart_warmup_shorter_than_one_accumulation_window_is_no_ramp():
    spec = resolve_lr_schedule_spec(_AxisProbe(8, 400, 4), "relora", 4000)
    assert spec.relora_restart_warmup_steps == 0
    curve = _ours(spec.warmup_steps, spec.total_steps, 0, (100,))
    assert curve(100) == pytest.approx(curve(101), abs=0.01), "straight to decay"


def test_the_floor_replaces_the_hardcoded_min_lr_ratio():
    """min_lr_ratio was fixed at 0.0; D10 makes it lr_floor_ratio, and a YAML
    with no floor key still reads as 0.0 (§12.2)."""
    W, Wr, T = 200, 100, 2000
    absent = resolve_spec({}, warmup_steps=W, total_steps=T, name="relora",
                          restart_warmup_steps=Wr)
    assert absent.floor_ratio == 0.0 and absent.floor_defaulted

    floored = _ours(W, T, Wr, (500,), floor=0.25)
    assert floored(T) == pytest.approx(0.25)
    # A re-warmup climbs from the floor; the run's FIRST warmup from 0 (§17.3).
    assert floored(500) == pytest.approx(0.25)
    assert floored(550) == pytest.approx(0.25 + 0.75 * 0.5)
    assert floored(600) == pytest.approx(1.0)
    assert floored(0) == 0.0
    assert floored(100) == pytest.approx(0.5)


def test_the_rewarmup_is_not_multiplied_by_the_shared_ramp():
    """§17.3: composing the segment's ramp with the common warmup ramp would
    square it while the two overlap."""
    curve = _ours(1000, 4000, 200, (400,))
    assert curve(500) == pytest.approx(0.5)      # not 0.5 * (500/1000)


def test_evaluation_order_does_not_change_any_value():
    ours = _ours(200, 2000, 100, (500, 1000, 1500), floor=0.2)
    steps = list(range(0, 2100))
    ascending = [ours(s) for s in steps]
    descending = {s: ours(s) for s in reversed(steps)}
    shuffled_steps = steps[:]
    random.Random(4).shuffle(shuffled_steps)
    shuffled = {s: ours(s) for s in shuffled_steps}
    for step, value in zip(steps, ascending):
        assert descending[step] == value, step
        assert shuffled[step] == value, step



def test_relora_is_resolvable_but_not_selectable():
    assert "relora" in INTERNAL_SCHEDULER_NAMES
    assert "relora" not in LR_SCHEDULER_NAMES
    assert _spec(0, 100, 10).curve == "relora"


def test_the_ui_offers_the_floor_to_a_relora_run():
    """D10's floor replaces a hardcoded 0.0 for ReLoRA, so it now shapes the
    curve -- but a ReLoRA run's lr_scheduler is `constant` by default, and that
    is the value the control is otherwise hidden by."""
    panel = (Path(__file__).resolve().parents[2]
             / "frontend/src/components/training/TrainingConfig.tsx"
             ).read_text(encoding="utf-8")
    before = panel[:panel.index('updateParam("lr_floor_ratio"')]
    condition = before[before.rindex("{/* Every schedule but Constant"):]
    assert 'trainingMethod === "relora"' in condition


def test_setup_optimizer_asks_the_registry_for_relora(capsys):
    """The override is gone: the parent builds the schedule, under the one name
    that carries the restart events."""
    trainer = ReLoRATrainer.__new__(ReLoRATrainer)
    trainer.log_prefix = "[Test]"
    trainer.optimizer_warmup_steps = 200
    trainer.restart_warmup_steps = 100

    seen = {}
    original = ReLoRATrainer.__mro__[1].setup_optimizer

    def _record(self, optimizer_type, lr_scheduler_type, total_steps):
        seen.update(optimizer_type=optimizer_type, name=lr_scheduler_type,
                    total_steps=total_steps)

    ReLoRATrainer.__mro__[1].setup_optimizer = _record
    try:
        trainer.setup_optimizer("adamw8bit", "plateau_cosine_floor", 2000)
    finally:
        ReLoRATrainer.__mro__[1].setup_optimizer = original

    assert seen == {"optimizer_type": "adamw8bit", "name": "relora",
                    "total_steps": 2000}
    assert "is ignored" in capsys.readouterr().out


def test_a_merge_records_one_restart_and_applies_it_immediately():
    """The merge hook runs after ``scheduler.step()``, so without rewriting the
    param groups the reinitialized adapter would take one more step at the
    pre-restart LR."""
    spec = _spec(200, 2000, 100)
    trainer = _ResumeHarness(spec)
    trainer._add_lr_restart = ReLoRATrainer._add_lr_restart.__get__(trainer)
    trainer.log_prefix = "[Test]"

    for _ in range(500):
        trainer.lr_scheduler.step()
    assert trainer.lr_scheduler.last_epoch == 500
    before = trainer.optimizer.param_groups[0]["lr"]

    trainer._add_lr_restart(global_step=500)
    assert trainer.lr_timeline.restarts() == [500]
    after = trainer.optimizer.param_groups[0]["lr"]
    assert after == 0.0 and before > 0.9 * BASE_LR


def test_a_full_run_matches_the_deleted_scheduler_step_for_step():
    """The whole loop: step the schedule, merge where the old trainer merged,
    and compare the LR the optimizer actually carries."""
    W, Wr, T, every = 100, 50, 1200, 400
    spec = _spec(W, T, Wr)
    trainer = _ResumeHarness(spec, merge_every=every)
    trainer._add_lr_restart = ReLoRATrainer._add_lr_restart.__get__(trainer)

    legacy = LegacyCosineWithMultipleWarmups(
        total_steps=T, initial_warmup_steps=W, restart_warmup_steps=Wr,
        min_lr_ratio=0.0)

    for step in range(1, T + 1):
        trainer.lr_scheduler.step()
        if step % every == 0:
            trainer._add_lr_restart(global_step=step)
            legacy.add_restart(step)
        # The old TRAINER carried the pre-restart value at the merge step (its
        # get_lr ran inside scheduler.step(), before the merge hook), so this
        # equality at `step == merge` is what the immediate rewrite buys: the
        # LR in the group is the curve at the position, always.
        expected = legacy.multiplier(step)
        got = trainer.optimizer.param_groups[0]["lr"] / BASE_LR
        assert got == pytest.approx(expected, abs=1e-12), step
