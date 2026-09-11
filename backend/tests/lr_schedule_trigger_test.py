"""Guard: conditional LR-schedule triggers (R6, §20 of docs/guides/LR_SCHEDULER_DESIGN.md).

R0-R5 gave the operator every control; R6 gives the run a finger to press them
with. What has to hold, and what a plausible simplification breaks:

* the signal comes from an in-memory RING fed at the metrics site and read at
  seam (c) -- never a database query in the training loop (D45);
* ONE OBSERVATION IS A MEAN over `interval` steps, and `patience` counts
  observations. Evaluating the raw per-step loss instead fires on the timestep
  noise that dominates a diffusion run's step-to-step variance (D47), which is
  what `test_smoothing_suppresses_a_noisy_signal...` pins;
* a firing materialises an ORDINARY event through `add()` (D50), so §19's
  refusals apply -- and a refused firing is NOT a firing: no `max_fires` spent,
  no cooldown started, a warning raised (D51);
* the CONDITION is never an event. It lives in state.json, is restored on
  resume, and no multiplier reads it (D28/D49, invariants 2, 18, 19);
* NO NUMERIC DEFAULTS. `interval`, `patience`, `min_delta`, `threshold` are
  required and refused by name when missing (D46, invariant 20).

CPU-only and hermetic: one 4-element parameter, no model, no dataset, no GPU,
no database.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/lr_schedule_trigger_test.py -v
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training import training_control_rpc as control_rpc  # noqa: E402
from core.training.base_trainer import (  # noqa: E402
    BaseTrainer,
    dump_lr_triggers,
    install_lr_schedule_triggers,
    lr_schedule_status,
    poll_lr_schedule_commands,
)
from core.training.lr_schedules import (  # noqa: E402
    STATE_BASE,
    STATE_DECAYING,
    ScheduleTimeline,
    build_lr_scheduler,
    is_refused_result,
    resolve_spec,
)
from core.training.lr_triggers import (  # noqa: E402
    MAX_TRIGGERS,
    TRIGGER_PREDICATES,
    TRIGGER_SIGNALS,
    Trigger,
    TriggerSet,
    validate_trigger,
)
from core.training.training_events import TRAINING_EVENT_SENTINEL  # noqa: E402
from api.param_defaults import LR_TRIGGER_DEFAULTS  # noqa: E402

BASE_LR = 1e-4
RUN_ID = 11
# A plateau whose configured decay sits at the very end, so a trigger is the
# only thing that starts one inside the range these tests walk.
LATE_PLATEAU = {"lr_decay_start_ratio": 1.0, "lr_floor_ratio": 0.25}

PLATEAU = {"signal": "loss", "predicate": "plateau", "interval": 10,
           "patience": 3, "min_delta": 0.01,
           "action": {"command": "start_decay"}}


class FakeTrainer:
    """The attributes the poll, the metrics seam and the status file read."""

    # The real seam, so `log()` below exercises the production call, not a copy.
    _feed_lr_trigger_signals = BaseTrainer._feed_lr_trigger_signals

    def __init__(self, output_dir, name="constant", W=0, T=1000, config=None,
                 gas=1, run_id=RUN_ID):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.log_prefix = "[Test]"
        self.fused_optimizer_groups = None
        self._grad_accum_steps = gas
        param = torch.nn.Parameter(torch.zeros(4))
        self.optimizer = torch.optim.SGD([{"params": [param]}], lr=BASE_LR)
        self.lr_schedule_spec = resolve_spec(
            config or {}, warmup_steps=W, total_steps=T, name=name)
        self.lr_timeline = ScheduleTimeline()
        self.lr_timeline.set_total_steps(self.lr_schedule_spec.total_steps)
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer, self.lr_schedule_spec, self.lr_timeline)
        self.lr_triggers = TriggerSet()
        self._resume_lr_triggers = None
        # What _log_metrics_to_db needs to buffer a row without flushing.
        self._metrics_buffer = []
        self._metrics_flush_interval = 10 ** 9
        self._extra_metrics = {}
        self._current_epoch = 0
        self.resume_seq = 0
        self._db_futures = []
        self._db_executor = None

    def seek(self, step: int) -> None:
        BaseTrainer._fast_forward_one_lr_scheduler(self.lr_scheduler, step)

    def arm(self, **overrides) -> Trigger:
        record = validate_trigger({**PLATEAU, **overrides})
        assert self.lr_triggers.register(record, 0) == "registered"
        return self.lr_triggers.get(record["id"])

    def log(self, step, loss=None, grad_norm=None, extra=None):
        """One pass through the real metrics site."""
        for name, value in (extra or {}).items():
            BaseTrainer.log_extra_metric(self, name, value)
        BaseTrainer._log_metrics_to_db(self, step=step, loss=loss,
                                       grad_norm=grad_norm)

    def state(self, step: int) -> int:
        return self.lr_timeline.state_at(self.lr_schedule_spec, step).code

    def multiplier(self, step: int) -> float:
        return self.lr_timeline.multiplier(self.lr_schedule_spec, step)

    @property
    def events(self):
        return self.lr_timeline.events


def poll(trainer, global_step: int = 0, capture=False):
    """The batch-head seam, with its console output swallowed (or returned)."""
    out = io.StringIO()
    with redirect_stdout(out):
        applied = poll_lr_schedule_commands(trainer, global_step)
    return (applied, out.getvalue()) if capture else applied


def feed(trainer, values, start=0, signal="loss", stride=1, poll_every=None):
    """Log `values` one per step and poll, the way the loop interleaves them."""
    console = []
    for i, value in enumerate(values):
        step = start + i * stride
        if signal == "loss":
            trainer.log(step, loss=value)
        elif signal == "grad_norm":
            trainer.log(step, grad_norm=value)
        else:
            trainer.log(step, loss=1.0, extra={signal.split(":", 1)[1]: value})
        if poll_every is None or step % poll_every == 0:
            console.append(poll(trainer, global_step=step, capture=True)[1])
    return "".join(console)


# ---------------------------------------------------------------------------
# D46 / invariant 20: the shape is ours, the numbers are the operator's
# ---------------------------------------------------------------------------

def test_the_defaults_table_holds_only_what_genuinely_has_a_default():
    # Every number a predicate reads is absent on purpose: a usable min_delta
    # is a property of the run's loss scale, and §16-12 forbids writing a
    # number nobody measured.
    assert set(LR_TRIGGER_DEFAULTS) == {"max_fires"}
    assert LR_TRIGGER_DEFAULTS["max_fires"] == 1


@pytest.mark.parametrize("missing,predicate", [
    ("interval", "plateau"),
    ("patience", "plateau"),
    ("min_delta", "plateau"),
    ("interval", "below"),
    ("threshold", "below"),
    ("interval", "above"),
    ("threshold", "above"),
])
def test_a_trigger_without_a_required_number_is_refused_by_name(missing, predicate):
    payload = {"signal": "loss", "predicate": predicate, "interval": 10,
               "action": {"command": "start_decay"}}
    payload.update({"patience": 3, "min_delta": 0.01} if predicate == "plateau"
                   else {"threshold": 0.5})
    payload.pop(missing)
    with pytest.raises(ValueError) as e:
        validate_trigger(payload)
    assert missing in str(e.value)
    # And it says WHY there is no default, so the answer is not "pick 0".
    assert "no default" in str(e.value)


def test_a_field_the_predicate_does_not_read_is_refused_not_ignored():
    """`patience: 3` on a `below` trigger looks like a debounce. It is not one,
    so accepting it silently would describe a condition nothing evaluates."""
    with pytest.raises(ValueError) as e:
        validate_trigger({"signal": "loss", "predicate": "below",
                          "interval": 10, "threshold": 0.5, "patience": 3,
                          "action": {"command": "start_decay"}})
    assert "patience" in str(e.value)
    with pytest.raises(ValueError):
        validate_trigger({**PLATEAU, "threshold": 0.5})


def test_interval_is_required_under_every_predicate():
    # Not just plateau: a threshold on a raw per-step diffusion loss is as noisy
    # as a plateau on one, so the smoothing is not optional anywhere (D47).
    for predicate in TRIGGER_PREDICATES:
        payload = {"signal": "loss", "predicate": predicate,
                   "action": {"command": "start_decay"}}
        payload.update({"patience": 3, "min_delta": 0.01}
                       if predicate == "plateau" else {"threshold": 0.5})
        with pytest.raises(ValueError) as e:
            validate_trigger(payload)
        assert "interval" in str(e.value)


def test_max_fires_is_the_only_field_with_a_default():
    record = validate_trigger(PLATEAU)
    assert record["max_fires"] == LR_TRIGGER_DEFAULTS["max_fires"]
    assert "cooldown" not in record


def test_cooldown_is_required_exactly_when_more_than_one_fire_is_allowed():
    with pytest.raises(ValueError) as e:
        validate_trigger({**PLATEAU, "max_fires": 3})
    assert "cooldown" in str(e.value)
    # And is meaningless with one fire: there is no second firing to space out.
    with pytest.raises(ValueError):
        validate_trigger({**PLATEAU, "cooldown": 2})
    assert validate_trigger({**PLATEAU, "max_fires": 3,
                             "cooldown": 2})["cooldown"] == 2


# ---------------------------------------------------------------------------
# D45: what may be watched
# ---------------------------------------------------------------------------

def test_learning_rate_is_not_a_signal_and_says_why():
    with pytest.raises(ValueError) as e:
        validate_trigger({**PLATEAU, "signal": "learning_rate"})
    message = str(e.value)
    assert "feedback loop" in message
    # The alternatives, so the refusal is actionable.
    assert "loss" in message and "grad_norm" in message


@pytest.mark.parametrize("signal", ["extra:lr", "extra:lr_unet",
                                    "extra:lr_decay_state"])
def test_the_learning_rate_cannot_be_watched_through_the_extra_channel(signal):
    """The trainer publishes its own LR through log_extra_metric, so D45's
    refusal has to cover the back door as well as the front one."""
    with pytest.raises(ValueError):
        validate_trigger({**PLATEAU, "signal": signal})


def test_an_unknown_signal_is_refused_and_extra_needs_a_name():
    with pytest.raises(ValueError):
        validate_trigger({**PLATEAU, "signal": "vram"})
    with pytest.raises(ValueError):
        validate_trigger({**PLATEAU, "signal": "extra:"})
    assert validate_trigger({**PLATEAU, "signal": "extra:known_loss"})[
        "signal"] == "extra:known_loss"
    for signal in TRIGGER_SIGNALS:
        assert validate_trigger({**PLATEAU, "signal": signal})["signal"] == signal


def test_an_action_cannot_pin_a_step():
    """D50: a firing takes effect where it fires. A fixed `at` would be in the
    past for every firing after the first, and refused as backdated."""
    with pytest.raises(ValueError) as e:
        validate_trigger({**PLATEAU, "action": {"op": "hold", "at": 500}})
    assert "at" in str(e.value)


# ---------------------------------------------------------------------------
# D47: one observation is a mean over `interval` steps
# ---------------------------------------------------------------------------

def test_a_plateau_fires_after_exactly_patience_observations_and_not_before():
    trigger = Trigger(**validate_trigger({**PLATEAU, "interval": 10,
                                          "patience": 3, "min_delta": 0.01}))
    fired = []
    # Ten windows of a flat signal. The first closed observation sets `best`;
    # the three after it are the misses that fire.
    for step in range(0, 100):
        closed = trigger.sample(step, 1.0)
        if closed is not None:
            hit = trigger.observe(closed[1])
            fired.append((closed[0], hit))
            if hit:
                trigger.record_fire(refused=False)
    assert [f for _, f in fired] == [False, False, False, True, False,
                                     False, False, False, False]
    # Fired on the 4th observation: one to set the baseline, then `patience`
    # without improvement. Not the 3rd, and not the 2nd.
    assert fired[3][0] == 39
    assert trigger.observations == len(fired)


def test_smoothing_suppresses_a_noisy_signal_that_would_fire_on_raw_values():
    """D47's whole reason. The signal improves steadily on average, but its
    per-step values wobble by far more than `min_delta` -- which is what a
    diffusion loss does, because the timestep is drawn at random every step.

    Evaluated per step, three consecutive up-ticks arrive almost immediately and
    the trigger fires on noise. Evaluated on the mean of each window, no
    observation fails to improve and it never fires.
    """
    # 60 steps: the trend falls by 0.02 a step, under a wobble of +-0.5 whose
    # period (5) divides the window (10), so every window mean carries the same
    # offset and the means fall monotonically.
    wobble = [-0.5, 0.4, 0.45, 0.5, 0.35]
    values = [2.0 - 0.02 * i + wobble[i % 5] for i in range(60)]
    spec = {**PLATEAU, "interval": 10, "patience": 3, "min_delta": 0.01}

    smoothed = Trigger(**validate_trigger(spec))
    fires = 0
    for step, value in enumerate(values):
        closed = smoothed.sample(step, value)
        if closed is not None and smoothed.observe(closed[1]):
            fires += 1
    assert fires == 0, "a falling trend must not read as a plateau"
    assert smoothed.observations == 5

    # The same values with no window at all (interval 1 = one sample per
    # observation) is the unsmoothed evaluation, and it fires: four steps pass
    # between the dips that beat the running best, which is one more than
    # `patience`.
    raw = Trigger(**validate_trigger({**spec, "interval": 1}))
    raw_fired_at = []
    for step, value in enumerate(values):
        closed = raw.sample(step, value)
        if closed is not None and raw.observe(closed[1]):
            raw_fired_at.append(closed[0])
            raw.record_fire(refused=False)
    # Four steps in: one to set the baseline, three up-ticks of pure noise.
    assert raw_fired_at == [3]


def test_a_window_with_no_sample_produces_no_observation():
    """`grad_norm` only exists on update boundaries, so under gradient
    accumulation most short windows hold fewer samples than steps -- and an
    EMPTY one must not spend patience on a value nobody measured."""
    trigger = Trigger(**validate_trigger({**PLATEAU, "signal": "grad_norm",
                                          "interval": 2, "patience": 2}))
    # gas = 4: a sample every 4th step, so every other 2-step window is empty.
    closed = [trigger.sample(step, 1.0) for step in range(0, 40, 4)]
    for c in closed:
        if c is not None:
            trigger.observe(c[1])
    assert sum(1 for c in closed if c is not None) == 9
    assert trigger.observations == 9
    # Nine closed observations out of twenty windows: the ten empty ones did
    # not close, and neither did the one still open.


def test_the_observation_is_the_mean_of_the_window():
    trigger = Trigger(**validate_trigger({**PLATEAU, "interval": 4}))
    for step, value in enumerate([1.0, 2.0, 3.0, 4.0]):
        assert trigger.sample(step, value) is None
    closed = trigger.sample(4, 9.0)
    assert closed == (3, 2.5)


def test_the_window_grid_is_absolute_so_a_resume_cannot_shift_it():
    a = Trigger(**validate_trigger({**PLATEAU, "interval": 10}))
    b = Trigger(**validate_trigger({**PLATEAU, "interval": 10}))
    closed_a = [a.sample(step, 1.0) for step in range(0, 30)]
    a_closed = [c for c in closed_a if c is not None]
    assert a_closed == [(9, 1.0), (19, 1.0)]

    # b joins mid-window; its first CLOSED observation is still the one ending
    # at 19, matching a's second observation boundary, not "10 steps from where I started".
    b.created_step = 7
    closed_b = [b.sample(step, 1.0) for step in range(7, 30)]
    b_closed = [c for c in closed_b if c is not None]
    assert b_closed == [(19, 1.0)]
    assert b_closed[0] == a_closed[1]


# ---------------------------------------------------------------------------
# below / above
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("predicate,threshold,values,expected", [
    ("below", 0.5, [1.0, 0.9, 0.4], [False, False, True]),
    ("above", 2.0, [1.0, 1.9, 2.1], [False, False, True]),
])
def test_below_and_above_fire_on_the_observation_that_crosses(
        predicate, threshold, values, expected):
    record = validate_trigger({"signal": "grad_norm", "predicate": predicate,
                               "interval": 1, "threshold": threshold,
                               "max_fires": 1,
                               "action": {"command": "cancel_decay"}})
    trigger = Trigger(**record)
    fired = []
    for step, value in enumerate(values + [values[-1]]):
        closed = trigger.sample(step, value)
        if closed is not None:
            hit = trigger.observe(closed[1])
            fired.append(hit)
            if hit:
                trigger.record_fire(refused=False)
    assert fired == expected
    # Strict comparisons: a value EQUAL to the threshold is neither below nor
    # above it.
    edge = Trigger(**record)
    edge.sample(0, threshold)
    assert edge.observe(threshold) is False


# ---------------------------------------------------------------------------
# D48 / D61 / D62: max_fires and cooldown
# ---------------------------------------------------------------------------

def test_a_trigger_disarms_after_max_fires_and_stays_visible():
    trigger = Trigger(**validate_trigger(
        {"signal": "loss", "predicate": "below", "interval": 1,
         "threshold": 1.0, "max_fires": 2, "cooldown": 1,
         "action": {"op": "scale", "gain": 0.5}}))
    fired = []
    for _ in range(4):
        hit = trigger.observe(0.5)
        fired.append(hit)
        if hit:
            trigger.record_fire(refused=False)
    assert fired == [True, False, True, False]
    assert trigger.fires == 2
    assert trigger.armed is False
    assert trigger.status()["fires_left"] == 0
    # Disarmed, not deleted: what it did is still readable (§20.4).
    assert trigger.status()["fires"] == 2


def test_a_cooldown_skips_observations_between_firings():
    trigger = Trigger(**validate_trigger(
        {"signal": "loss", "predicate": "below", "interval": 1,
         "threshold": 1.0, "max_fires": 3, "cooldown": 2,
         "action": {"op": "scale", "gain": 0.5}}))
    fired = []
    for _ in range(9):
        hit = trigger.observe(0.5)
        fired.append(hit)
        if hit:
            trigger.record_fire(refused=False)
    # Fire, skip 2, fire, skip 2, fire -- and then nothing, max_fires spent.
    assert fired == [True, False, False, True, False, False, True, False, False]


def test_the_plateau_baseline_is_dropped_after_a_firing():
    """The best-so-far belonged to the LR the run had. Keeping it would compare
    the post-change loss against a pre-change record it may never beat, firing
    every `patience` observations regardless of what the change did."""
    trigger = Trigger(**validate_trigger(
        {**PLATEAU, "max_fires": 2, "cooldown": 1}))
    trigger.best = 0.5
    trigger.misses = 2
    trigger.record_fire(refused=False)
    assert trigger.best is None and trigger.misses == 0


# ---------------------------------------------------------------------------
# D50 / D51: firing through the ordinary add(), and what a refusal costs
# ---------------------------------------------------------------------------

def test_a_firing_materialises_an_ordinary_event_at_the_current_step(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=1000,
                          config=LATE_PLATEAU)
    trainer.arm(interval=10, patience=1, min_delta=0.01)
    trainer.seek(50)
    feed(trainer, [1.0] * 30, poll_every=10)

    assert trainer.state(50) == STATE_DECAYING
    decays = [e for e in trainer.events if e["kind"] == "decay"]
    assert len(decays) == 1
    # D50: at = issued = the scheduler step it fired at, so it is
    # indistinguishable from an event the operator queued.
    assert decays[0]["at"] == decays[0]["issued"] == 50
    assert decays[0]["result"] == "applied"


def test_a_fired_retarget_needs_no_branch_of_its_own(tmp_path):
    """§19.5.3's payload vocabulary is the action vocabulary, so `scale` /
    `hold` / `undo` fire through the same code an operator's request does."""
    for op, extra in [("scale", {"gain": 0.5}), ("hold", {})]:
        trainer = FakeTrainer(tmp_path / op, name="cosine", T=1000)
        trainer.arm(interval=5, patience=1, min_delta=0.01,
                    action={"op": op, **extra})
        trainer.seek(100)
        before = trainer.multiplier(100)
        feed(trainer, [1.0] * 15, poll_every=5)
        retargets = [e for e in trainer.events if e["kind"] == "retarget"]
        assert len(retargets) == 1, op
        if op == "hold":
            # Frozen at the value it was at, from here to the end.
            assert trainer.multiplier(100) == pytest.approx(before)
            assert trainer.multiplier(900) == pytest.approx(before)
        else:
            # x0.5 from here, blended over the run's warmup (0 here).
            assert trainer.multiplier(100) == pytest.approx(before * 0.5)


def test_a_refused_firing_spends_no_fire_and_starts_no_cooldown(tmp_path):
    """D51. The refusal here is §5.3's: a decay cannot start during the warmup.
    A once-only trigger that lost its one chance to it would be unusable."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=400, T=1000,
                          config=LATE_PLATEAU)
    trigger = trainer.arm(interval=10, patience=1, min_delta=0.01,
                          max_fires=2, cooldown=5)
    trainer.seek(100)                       # inside the warmup
    console = feed(trainer, [1.0] * 30, poll_every=10)

    assert trainer.state(100) == STATE_BASE
    assert trigger.fires == 0
    assert trigger.armed is True
    assert trigger.cooldown_left == 0
    # It is recorded as a refusal, and the operator is told (§13).
    noops = [e for e in trainer.events if e["kind"] == "noop"]
    assert [e["result"] for e in noops] == ["rejected_during_warmup"] * len(noops)
    assert "lr_trigger_fire_refused" in console
    assert TRAINING_EVENT_SENTINEL in console


def test_a_refused_firing_does_not_retry_on_every_observation(tmp_path):
    """The condition is still true, so without restarting the patience counter
    the trigger would warn once per observation for the rest of the run."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=400, T=1000,
                          config=LATE_PLATEAU)
    trigger = trainer.arm(interval=10, patience=3, min_delta=0.01)
    trainer.seek(100)
    feed(trainer, [1.0] * 100, poll_every=10)
    assert trigger.observations == 9
    # One attempt per `patience` observations, not one per observation.
    refusals = [e for e in trainer.events if e["kind"] == "noop"]
    assert len(refusals) == 2


def test_an_ignored_firing_still_counts(tmp_path):
    """`ignored_*` is not a refusal: the event was recorded and folded, it just
    changed nothing given the state. D51 exempts refusals only."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=1000,
                          config=LATE_PLATEAU)
    trainer.lr_timeline.add("decay", at=0, issued=0)   # already decaying
    trigger = trainer.arm(interval=10, patience=1, min_delta=0.01)
    trainer.seek(50)
    feed(trainer, [1.0] * 30, poll_every=10)
    assert trigger.fires == 1
    assert not is_refused_result("ignored_already_decaying")


# ---------------------------------------------------------------------------
# D45: the ring, fed at the metrics site and read at seam (c)
# ---------------------------------------------------------------------------

def test_the_metrics_site_feeds_the_ring_and_the_poll_drains_it(tmp_path):
    trainer = FakeTrainer(tmp_path, T=1000)
    trainer.arm(interval=5, patience=1, min_delta=0.01,
                action={"command": "cancel_decay"})
    trainer.log(0, loss=1.0)
    trainer.log(1, loss=2.0)
    assert len(trainer.lr_triggers._ring) == 2
    poll(trainer, global_step=1)
    assert len(trainer.lr_triggers._ring) == 0
    # The row still reached the metrics buffer: the seam observes, it does not
    # consume.
    assert [e["step"] for e in trainer._metrics_buffer] == [0, 1]


def test_the_learning_rate_is_never_pushed_onto_the_ring(tmp_path):
    trainer = FakeTrainer(tmp_path)
    trainer.arm()
    BaseTrainer._log_metrics_to_db(trainer, step=0, loss=1.0,
                                   learning_rate=3e-4, grad_norm=0.7)
    assert sorted(name for _, name, _ in trainer.lr_triggers._ring) == [
        "grad_norm", "loss"]


def test_an_extra_metric_reaches_a_trigger_with_the_value_the_chart_gets(tmp_path):
    trainer = FakeTrainer(tmp_path)
    trainer.arm(signal="extra:known_loss")
    trainer.log(0, loss=1.0, extra={"known_loss": 0.25})
    assert (0, "extra:known_loss", 0.25) in list(trainer.lr_triggers._ring)
    assert trainer._metrics_buffer[0]["extra"] == {"known_loss": 0.25}


def test_the_seam_never_raises_into_the_training_loop(tmp_path):
    """`poll_lr_schedule_commands`'s existing contract, which R6 shares."""
    trainer = FakeTrainer(tmp_path, T=1000)
    trigger = trainer.arm(interval=1, patience=1, min_delta=0.01)
    trigger.action = {"command": "retarget", "op": "explode"}
    trainer.log(0, loss=1.0)
    trainer.log(1, loss=1.0)
    trainer.log(2, loss=1.0)
    applied, console = poll(trainer, global_step=2, capture=True)
    assert applied == 0
    assert "LR trigger evaluation failed" in console


# ---------------------------------------------------------------------------
# D49 / invariants 18-19: state.json, not the event list
# ---------------------------------------------------------------------------

def test_the_condition_never_appears_in_the_event_list(tmp_path):
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=1000,
                          config=LATE_PLATEAU)
    trainer.arm(interval=10, patience=1, min_delta=0.01)
    trainer.seek(50)
    feed(trainer, [1.0] * 30, poll_every=10)

    dumped = json.dumps(trainer.lr_timeline.dump(1000))
    for word in ("plateau", "patience", "min_delta", "signal", "predicate",
                 "interval", "trigger"):
        assert word not in dumped, word
    # What IS there is the ordinary event the firing materialised.
    assert [e["kind"] for e in trainer.lr_timeline.dump(1000)] == [
        "total_steps", "decay"]
    with redirect_stdout(io.StringIO()):
        status = lr_schedule_status(trainer, global_step=50)
    assert "trigger" not in json.dumps(status["events"])
    # And the multiplier is still a pure function of (step, events): a fresh
    # timeline holding only the events reproduces it, with no trigger at all.
    replay = ScheduleTimeline(status["events"])
    assert replay.multiplier(trainer.lr_schedule_spec, 500) == pytest.approx(
        trainer.multiplier(500))


def test_trigger_state_survives_a_state_json_round_trip(tmp_path):
    trainer = FakeTrainer(tmp_path, T=1000)
    trigger = trainer.arm(interval=10, patience=3, min_delta=0.01)
    trainer.log(0, loss=1.0)
    trainer.log(10, loss=1.0)
    trainer.log(20, loss=1.0)
    poll(trainer, global_step=20)
    assert (trigger.observations, trigger.misses) == (2, 1)

    saved = json.loads(json.dumps({"lr_schedule_triggers":
                                   dump_lr_triggers(trainer)}))
    fresh = FakeTrainer(tmp_path / "resumed", T=1000)
    fresh._resume_lr_triggers = saved["lr_schedule_triggers"]
    install_lr_schedule_triggers(fresh)

    restored = fresh.lr_triggers.get(trigger.id)
    assert restored.to_dict() == trigger.to_dict()
    # Including the window in progress, so the resume continues the partial
    # observation rather than restarting it.
    assert (restored.window, restored.window_count) == (2, 1)


def test_resume_preserves_samples_pending_at_checkpoint(tmp_path):
    trainer = FakeTrainer(tmp_path, T=1000)
    trainer.arm(predicate="below", interval=2, threshold=0.75,
                patience=None, min_delta=None)
    trainer.log(0, loss=1.0)
    poll(trainer, global_step=0)
    trainer.log(1, loss=0.0)
    saved = json.loads(json.dumps({
        "triggers": dump_lr_triggers(trainer),
        "signals": trainer.lr_triggers.dump_signals(),
    }))
    resumed = FakeTrainer(tmp_path / "resumed", T=1000)
    resumed._resume_lr_triggers = saved["triggers"]
    resumed._resume_lr_trigger_signals = saved["signals"]
    install_lr_schedule_triggers(resumed)
    for run in (trainer, resumed):
        run.log(2, loss=1.0)
        poll(run, global_step=2)
    assert resumed.lr_triggers.dump() == trainer.lr_triggers.dump()
    assert trainer.lr_triggers.triggers[0].fires == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("predicate,field", [("plateau", "min_delta"),
                                              ("below", "threshold")])
def test_trigger_predicate_values_must_be_finite(value, predicate, field):
    payload = {"signal": "loss", "predicate": predicate, "interval": 2,
               "action": {"command": "cancel_decay"}, field: value}
    if predicate == "plateau":
        payload["patience"] = 1
    with pytest.raises(ValueError, match="finite"):
        validate_trigger(payload)


def test_restored_trigger_spec_uses_validated_numeric_types():
    record = validate_trigger(PLATEAU)
    saved = Trigger.from_dict(record).to_dict()
    saved.update(interval="10", max_fires="1")
    restored = TriggerSet()
    restored.load([saved])
    assert restored.status()[0]["armed"]
    assert restored.triggers[0].interval == 10


@pytest.mark.parametrize("key,value", [("window_count", "1"),
                                       ("window_sum", float("nan")),
                                       ("best", "1.0"), ("window", -1),
                                       ("fires", 0.5)])
def test_corrupt_observation_state_is_dropped(key, value):
    saved = Trigger.from_dict(validate_trigger(PLATEAU)).to_dict()
    saved[key] = value
    restored, warnings = TriggerSet(), []
    restored.load([saved], warn=warnings.append)
    assert not restored.triggers
    assert len(warnings) == 1


def test_a_resume_from_before_a_firing_drops_the_event_and_rearms(tmp_path):
    """§20.5. The event is truncated by `issued`, the trigger state goes back
    to the checkpoint's -- so the same condition fires again, which is what
    "the condition is still live" means."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=1000,
                          config=LATE_PLATEAU)
    trigger = trainer.arm(interval=10, patience=2, min_delta=0.01)
    trainer.seek(20)
    # Two observations: a baseline and one miss. Checkpoint here.
    feed(trainer, [1.0] * 30, poll_every=10)
    assert trigger.fires == 0 and trigger.misses == 1
    checkpoint = json.loads(json.dumps({
        "scheduler_step": 20,
        "lr_schedule_events": trainer.lr_timeline.dump(20),
        "lr_schedule_triggers": dump_lr_triggers(trainer),
    }))

    # Keep going: the next observation is the second miss, and it fires.
    trainer.seek(60)
    feed(trainer, [1.0] * 20, start=30, poll_every=10)
    assert trigger.fires == 1
    assert any(e["kind"] == "decay" for e in trainer.events)

    resumed = FakeTrainer(tmp_path / "resumed", name="plateau_cosine_floor",
                          T=1000, config=LATE_PLATEAU)
    resumed.lr_timeline.load(checkpoint["lr_schedule_events"], upto_step=20)
    resumed._resume_lr_triggers = checkpoint["lr_schedule_triggers"]
    install_lr_schedule_triggers(resumed)

    assert not any(e["kind"] == "decay" for e in resumed.events)
    rearmed = resumed.lr_triggers.get(trigger.id)
    assert rearmed.fires == 0 and rearmed.armed and rearmed.misses == 1
    resumed.seek(20)
    feed(resumed, [1.0] * 20, start=30, poll_every=10)
    assert rearmed.fires == 1
    assert resumed.state(20) == STATE_DECAYING


# ---------------------------------------------------------------------------
# Registration through the control queue (§20.6/D53) and the published state
# ---------------------------------------------------------------------------

def test_registration_and_cancellation_ride_the_queue_and_make_no_event(tmp_path):
    trainer = FakeTrainer(tmp_path, T=1000)
    record = validate_trigger({**PLATEAU, "id": "halve"})
    control_rpc.queue_request(tmp_path, command="add_trigger", run_id=RUN_ID,
                              extra={"trigger": record})
    before = list(trainer.events)
    poll(trainer, global_step=5)
    assert [t.id for t in trainer.lr_triggers] == ["halve"]
    assert trainer.lr_triggers.get("halve").created_step == 5
    assert trainer.events == before

    results = {r["request_id"]: r for r in control_rpc.list_results(tmp_path)}
    assert [r["result"] for r in results.values()] == ["registered"]
    assert all(r["trigger_id"] == "halve" for r in results.values())

    control_rpc.queue_request(tmp_path, command="remove_trigger", run_id=RUN_ID,
                              extra={"trigger_id": "halve"})
    poll(trainer, global_step=6)
    assert list(trainer.lr_triggers) == []
    assert trainer.events == before
    assert all(r["trigger_id"] == "halve"
               for r in control_rpc.list_results(tmp_path))


@pytest.mark.parametrize("command,extra", [
    ("add_trigger", {}),
    ("remove_trigger", {}),
])
def test_a_trigger_command_without_its_payload_is_refused_before_it_is_written(
        tmp_path, command, extra):
    with pytest.raises(ValueError):
        control_rpc.queue_request(tmp_path, command=command, run_id=RUN_ID,
                                  extra=extra)
    assert control_rpc.list_pending_requests(tmp_path) == []


def test_a_trigger_command_is_not_a_timeline_event_kind():
    for command in control_rpc.TRIGGER_COMMANDS:
        assert command not in control_rpc.COMMAND_EVENT_KINDS
        assert command in control_rpc.ALL_COMMANDS


def test_duplicate_ids_and_the_cap_are_the_trainers_answer(tmp_path):
    trainer = FakeTrainer(tmp_path)
    record = validate_trigger({**PLATEAU, "id": "one"})
    assert trainer.lr_triggers.register(record, 0) == "registered"
    assert trainer.lr_triggers.register(record, 0) == "rejected_duplicate_trigger_id"
    for i in range(MAX_TRIGGERS - 1):
        trainer.lr_triggers.register(validate_trigger({**PLATEAU, "id": f"t{i}"}), 0)
    assert trainer.lr_triggers.register(
        validate_trigger({**PLATEAU, "id": "over"}), 0) == "rejected_trigger_limit"
    assert trainer.lr_triggers.remove("nope") == "rejected_unknown_trigger"


def test_an_unarmable_record_from_a_state_file_is_named_not_a_traceback(tmp_path):
    trainer = FakeTrainer(tmp_path, T=1000)
    control_rpc.queue_request(tmp_path, command="add_trigger", run_id=RUN_ID,
                              extra={"trigger": {"signal": "loss",
                                                 "predicate": "plateau"}})
    poll(trainer, global_step=1)
    result = control_rpc.list_results(tmp_path)[0]
    assert result["result"] == "rejected_invalid_trigger"
    assert "interval" in result["error"]


def test_the_display_file_publishes_what_each_trigger_is_watching(tmp_path):
    """D52: an automation nobody can see coming is worse than watching."""
    trainer = FakeTrainer(tmp_path, T=1000)
    trigger = trainer.arm(interval=10, patience=3, min_delta=0.01)
    feed(trainer, [1.0] * 30, poll_every=10)

    published = control_rpc.read_status(tmp_path)["triggers"]
    assert len(published) == 1
    state = published[0]
    assert state["id"] == trigger.id
    assert state["signal"] == "loss" and state["predicate"] == "plateau"
    assert state["armed"] is True and state["fires_left"] == 1
    assert state["observation"] == pytest.approx(1.0)
    assert state["observation_step"] == 19
    assert state["patience_used"] == 1
    assert state["observations_to_fire"] == 2


def test_the_display_file_is_rewritten_when_an_observation_moves_it(tmp_path):
    trainer = FakeTrainer(tmp_path, T=1000)
    trainer.arm(interval=10, patience=5, min_delta=0.01)
    feed(trainer, [1.0] * 25, poll_every=None)
    first = control_rpc.read_status(tmp_path)["triggers"][0]["observations"]
    feed(trainer, [1.0] * 25, start=25, poll_every=None)
    assert control_rpc.read_status(tmp_path)["triggers"][0]["observations"] > first


# ---------------------------------------------------------------------------
# The axis: `interval` counts signal, not schedule position (§18's lesson)
# ---------------------------------------------------------------------------

def test_the_interval_is_on_the_global_axis_and_gas_does_not_divide_it(tmp_path):
    """§18 records three defects from confusing the two axes. `interval` counts
    SIGNAL, which arrives once per global step; the EVENT a firing makes is
    dated on the scheduler axis, like every other event (invariant 4)."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", T=1000,
                          config=LATE_PLATEAU, gas=4)
    trainer.arm(interval=10, patience=1, min_delta=0.01)
    trainer.seek(25)                       # 25 scheduler advances = 100 global
    feed(trainer, [1.0] * 30, start=100, poll_every=10)

    trigger = next(iter(trainer.lr_triggers))
    # 30 global steps at interval 10 = 3 windows, 2 of them closed. Not
    # 30 // (10 // 4) = 12 observations, which flooring the interval onto the
    # scheduler axis would have produced -- and an interval below gas would
    # have floored to 0.
    assert trigger.observations == 2
    decay = [e for e in trainer.events if e["kind"] == "decay"][0]
    assert decay["at"] == 25


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def routes():
    import api.routes as module
    return module


class _FakeProc:
    def __init__(self, output_dir, running=True):
        self.output_dir = str(output_dir)
        self.is_running = running


class _FakeDb:
    def __init__(self, row):
        self._row = row

    def query(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._row


def call_add(routes, tmp_path, body=None, run=object(), proc=True, **overrides):
    import asyncio
    from core.training.training_process import training_process_manager

    payload = dict(body if body is not None else PLATEAU)
    payload.update(overrides)
    processes = training_process_manager.processes
    if proc:
        processes[RUN_ID] = _FakeProc(tmp_path, running=proc != "stopped")
    else:
        processes.pop(RUN_ID, None)
    try:
        return asyncio.run(routes.queue_lr_schedule_trigger(
            RUN_ID, routes.LrScheduleTriggerRequest(**payload),
            db=_FakeDb(run)))
    finally:
        processes.pop(RUN_ID, None)


def status_of(excinfo):
    return excinfo.value.status_code


def test_the_endpoint_queues_a_trigger_the_trainer_arms(routes, tmp_path):
    accepted = call_add(routes, tmp_path)
    assert accepted["command"] == "add_trigger"
    assert accepted["trigger"]["id"]           # generated when not named
    assert accepted["trigger"]["max_fires"] == 1
    assert "cooldown" not in accepted["trigger"]

    trainer = FakeTrainer(tmp_path, T=1000)
    poll(trainer, global_step=3)
    assert control_rpc.list_results(tmp_path)[0]["result"] == "registered"
    assert [t.id for t in trainer.lr_triggers] == [accepted["trigger"]["id"]]


@pytest.mark.parametrize("bad,message", [
    ({"signal": "learning_rate"}, "feedback loop"),
    ({"signal": "vram"}, "Unknown signal"),
    ({"predicate": "wobbles"}, "Unknown predicate"),
    ({"interval": None}, "interval"),
    ({"patience": None}, "patience"),
    ({"min_delta": None}, "min_delta"),
    ({"max_fires": 2}, "cooldown"),
    ({"cooldown": 3}, "cooldown"),
    ({"action": {"op": "scale"}}, "gain"),
    ({"action": {"op": "retarget", "lr_scheduler": "nope"}}, "lr_scheduler"),
    ({"action": {"op": "hold", "at": 10}}, "at"),
    ({"action": {"command": "start_decay", "gain": 2.0}}, "no parameters"),
])
def test_the_endpoint_refuses_what_it_can_decide_without_the_run(
        routes, tmp_path, bad, message):
    with pytest.raises(routes.HTTPException) as e:
        call_add(routes, tmp_path, **bad)
    assert status_of(e) == 400
    assert message in e.value.detail
    assert control_rpc.list_pending_requests(tmp_path) == []


def test_the_endpoint_rejects_an_unknown_key_rather_than_ignoring_it(routes):
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        routes.LrScheduleTriggerRequest(signal="loss", pateince=3)


def test_a_missing_run_is_a_404_and_a_stopped_one_is_a_409(routes, tmp_path):
    with pytest.raises(routes.HTTPException) as e:
        call_add(routes, tmp_path, run=None)
    assert status_of(e) == 404
    for proc in (False, "stopped"):
        with pytest.raises(routes.HTTPException) as e:
            call_add(routes, tmp_path, proc=proc)
        assert status_of(e) == 409
    assert control_rpc.list_pending_requests(tmp_path) == []


def test_a_full_queue_is_a_429(routes, tmp_path):
    for _ in range(control_rpc.MAX_PENDING_REQUESTS):
        control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    with pytest.raises(routes.HTTPException) as e:
        call_add(routes, tmp_path)
    assert status_of(e) == 429


def test_the_get_and_the_delete(routes, tmp_path):
    import asyncio
    from core.training.training_process import training_process_manager

    trainer = FakeTrainer(tmp_path, T=1000)
    trigger = trainer.arm(interval=10, patience=3, min_delta=0.01)
    feed(trainer, [1.0] * 25, poll_every=10)

    processes = training_process_manager.processes
    processes[RUN_ID] = _FakeProc(tmp_path)
    try:
        listed = asyncio.run(routes.get_lr_schedule_triggers(
            RUN_ID, db=_FakeDb(object())))
        assert [t["id"] for t in listed["triggers"]] == [trigger.id]
        # Both resource bounds. Without the second one a UI can say "1 of 20
        # registered" but has to copy the fire ceiling out of the spec.
        assert listed["max_triggers"] == MAX_TRIGGERS
        assert listed["max_trigger_fires"] == control_rpc.MAX_TRIGGER_FIRES

        accepted = asyncio.run(routes.cancel_lr_schedule_trigger(
            RUN_ID, trigger.id, db=_FakeDb(object())))
        assert accepted["command"] == "remove_trigger"
        assert accepted["trigger_id"] == trigger.id

        pending = asyncio.run(routes.get_lr_schedule_triggers(
            RUN_ID, db=_FakeDb(object())))["pending"]
        assert [p["trigger_id"] for p in pending] == [trigger.id]
    finally:
        processes.pop(RUN_ID, None)

    poll(trainer, global_step=30)
    assert list(trainer.lr_triggers) == []


def test_an_unknown_id_is_the_trainers_answer_not_a_404(routes, tmp_path):
    """The published list is display state and can be a batch old, so the
    endpoint queues and the trainer names it."""
    import asyncio
    from core.training.training_process import training_process_manager

    processes = training_process_manager.processes
    processes[RUN_ID] = _FakeProc(tmp_path)
    try:
        asyncio.run(routes.cancel_lr_schedule_trigger(
            RUN_ID, "ghost", db=_FakeDb(object())))
    finally:
        processes.pop(RUN_ID, None)
    trainer = FakeTrainer(tmp_path, T=1000)
    poll(trainer, global_step=1)
    assert control_rpc.list_results(tmp_path)[0]["result"] == \
        "rejected_unknown_trigger"


def test_the_endpoints_are_documented_in_openapi():
    import yaml
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    path = spec["paths"]["/training/runs/{run_id}/lr-schedule/triggers"]
    assert set(path) == {"get", "post"}
    assert set(path["post"]["responses"]) == {"202", "400", "404", "409",
                                              "429", "500"}
    assert set(path["get"]["responses"]) == {"200", "404"}
    body = path["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body["$ref"].endswith("/LrScheduleTriggerRequest")

    delete = spec["paths"][
        "/training/runs/{run_id}/lr-schedule/triggers/{trigger_id}"]["delete"]
    assert set(delete["responses"]) == {"202", "400", "404", "409", "429", "500"}

    schema = spec["components"]["schemas"]["LrScheduleTriggerRequest"]
    assert schema["properties"]["predicate"]["enum"] == list(TRIGGER_PREDICATES)
    # D46: nothing is `required`. Which fields a body needs depends on its
    # predicate, and the contract is "send what you have and be refused by
    # name" -- a required list makes a generated client block the request
    # locally, and the named refusal never happens.
    assert "required" not in schema
    # Invariant 20: the documented body carries a default for max_fires and for
    # nothing else. A documented `default: 0` would be the number D46 refuses.
    defaulted = {k for k, v in schema["properties"].items() if "default" in v}
    assert defaulted == {"max_fires"}

    documented = set(spec["components"]["schemas"]["LrScheduleCommandResult"]
                     ["properties"]["result"]["enum"])
    assert {"registered", "removed", "rejected_unknown_trigger",
            "rejected_duplicate_trigger_id", "rejected_trigger_limit",
            "rejected_invalid_trigger"} <= documented
    queued = set(spec["components"]["schemas"]["LrScheduleQueueItem"]
                 ["properties"]["command"]["enum"])
    assert set(control_rpc.ALL_COMMANDS) == queued


def test_the_documented_defaults_are_the_ones_the_endpoint_uses():
    """Invariant 7: one defaults table, and openapi says what it says."""
    import yaml
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["LrScheduleTriggerRequest"]["properties"]
    for key, value in LR_TRIGGER_DEFAULTS.items():
        assert props[key]["default"] == value, key
    schema = spec["components"]["schemas"]["LrTriggerDefaults"]
    assert set(schema["properties"]) == set(LR_TRIGGER_DEFAULTS)


def test_the_schema_endpoint_returns_the_table(routes):
    import asyncio
    assert asyncio.run(routes.get_lr_trigger_defaults()) == LR_TRIGGER_DEFAULTS


def test_the_published_state_is_documented():
    import yaml
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    state = spec["components"]["schemas"]["LrScheduleState"]
    assert state["properties"]["triggers"]["items"]["$ref"].endswith(
        "/LrScheduleTriggerState")
    schema = spec["components"]["schemas"]["LrScheduleTriggerState"]
    trigger = Trigger(**validate_trigger(PLATEAU))
    assert set(trigger.status()) == set(schema["properties"])
    # Against what `status()` returns, not against the property list: every
    # key is always there, and nullable is not the same as absent (19.5.4-24).
    assert set(schema["required"]) == set(trigger.status())


# ---------------------------------------------------------------------------
# D61 / D62 / D63 / D64: audit fixes
# ---------------------------------------------------------------------------

def test_non_finite_signals_are_silently_dropped_by_the_ring():
    ts = TriggerSet()
    record = validate_trigger(PLATEAU)
    ts.register(record, step=0)
    ts.push(0, "loss", float("nan"))
    ts.push(1, "loss", float("inf"))
    ts.push(2, "loss", float("-inf"))
    assert len(ts.drain()) == 0


def test_max_fires_is_capped_at_max_trigger_fires():
    from core.training.training_control_rpc import MAX_TRIGGER_FIRES
    with pytest.raises(ValueError) as e:
        validate_trigger({**PLATEAU, "max_fires": MAX_TRIGGER_FIRES + 1,
                          "cooldown": 1})
    assert f"<= {MAX_TRIGGER_FIRES}" in str(e.value)


def test_cooldown_minimum_is_one_when_max_fires_is_above_one():
    with pytest.raises(ValueError) as e:
        validate_trigger({**PLATEAU, "max_fires": 2, "cooldown": 0})
    assert ">= 1" in str(e.value)


def test_trigger_set_load_discards_corrupt_records_with_warning():
    """D63: A corrupt record in state.json must not kill a resume.

    Missing required fields, negative counters, or invalid types are dropped
    with a warning, while healthy records survive intact.
    """
    ts = TriggerSet()
    good = validate_trigger({**PLATEAU, "id": "healthy"})
    bad_missing_key = {"id": "broken_no_signal", "predicate": "plateau"}
    bad_negative = {**good, "id": "broken_neg", "fires": -1}
    bad_interval = {**good, "id": "broken_zero_interval", "interval": 0}
    bad_non_dict = "not_a_dict"

    warnings = []
    ts.load([good, bad_missing_key, bad_negative, bad_interval, bad_non_dict],
            warn=warnings.append)

    assert len(ts.triggers) == 1
    assert ts.triggers[0].id == "healthy"
    assert len(warnings) == 4
    assert any("broken_no_signal" in w for w in warnings)
    assert any("broken_neg" in w for w in warnings)
    assert any("broken_zero_interval" in w for w in warnings)


def test_trigger_set_evaluate_isolates_failing_triggers():
    """D63: An exception in one trigger during evaluate must not starve healthy ones."""
    ts = TriggerSet()
    ts.register(validate_trigger({**PLATEAU, "id": "healthy", "interval": 1,
                                 "patience": 1, "min_delta": 0.0}), step=0)
    ts.register(validate_trigger({**PLATEAU, "id": "broken", "interval": 1,
                                 "patience": 1, "min_delta": 0.0}), step=0)

    # Monkeypatch the broken trigger's sample method to raise
    broken_trigger = ts.get("broken")
    def faulty_sample(*args, **kwargs):
        raise RuntimeError("Simulated corruption inside evaluate")
    broken_trigger.sample = faulty_sample

    errors = []
    ts.push(0, "loss", 10.0)
    ts.push(1, "loss", 10.0)
    ts.push(2, "loss", 10.0)  # plateau triggers fire on 2nd observation

    fired_triggers = []
    def fire_cb(trigger, observed, step):
        fired_triggers.append(trigger.id)
        return "applied"

    outcomes = ts.evaluate(fire_cb, on_error=lambda trig, err: errors.append((trig.id, err)))

    assert len(errors) == 3  # error caught for all 3 samples
    assert errors[0][0] == "broken"
    # Healthy trigger still evaluated and fired!
    assert "healthy" in fired_triggers
    assert any(t.id == "healthy" for t, _ in outcomes)


def test_install_lr_schedule_triggers_survives_corrupt_records(tmp_path):
    """D63: Seam (b) install_lr_schedule_triggers must not raise when state.json
    has corrupt trigger records."""
    trainer = FakeTrainer(tmp_path, T=1000)
    trainer._resume_lr_triggers = [
        {"id": "corrupt_missing_keys"},
        validate_trigger(PLATEAU),
    ]
    # Seam (b) must complete cleanly and emit a warning
    out = io.StringIO()
    with redirect_stdout(out):
        install_lr_schedule_triggers(trainer)
    console = out.getvalue()
    assert len(trainer.lr_triggers.triggers) == 1
    assert "lr_trigger_corrupt_record" in console


def test_a_refused_threshold_firing_does_not_retry_on_every_observation(tmp_path):
    """D64: below/above have no patience counter to restart. Without debouncing
    on refusal, they would retry on every single observation while the refusal holds."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=400, T=1000,
                          config=LATE_PLATEAU)
    # below 2.0, loss stays 1.0, inside warmup so start_decay is refused.
    record = validate_trigger({
        "signal": "loss", "predicate": "below", "interval": 10,
        "threshold": 2.0, "max_fires": 1,
        "action": {"command": "start_decay"},
    })
    trainer.lr_triggers.register(record, 0)
    trigger = trainer.lr_triggers.get(record["id"])
    trainer.seek(100)
    feed(trainer, [1.0] * 100, poll_every=10)
    assert trigger.observations == 9
    refusals = [e for e in trainer.events if e["kind"] == "noop"]
    # With debounce (cooldown_left = 1 on refusal), it skips every alternate observation,
    # rather than firing all 9 times.
    assert len(refusals) < 9
    assert len(refusals) == 5


def test_openapi_bounds_are_consistent_with_rpc():
    import yaml
    from core.training.training_control_rpc import MAX_TRIGGER_FIRES
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["LrScheduleTriggerRequest"]["properties"]
    assert props["max_fires"]["maximum"] == MAX_TRIGGER_FIRES
    assert props["cooldown"]["minimum"] == 1


def test_a_refused_stand_off_is_readable_as_one(tmp_path):
    """D52: `cooldown_left > 0` has two causes and the number cannot tell them
    apart. Here nothing is configured to space out -- the hold is D64's
    debounce after a refusal, and the published record has to say so."""
    trainer = FakeTrainer(tmp_path, name="plateau_cosine_floor", W=400, T=1000,
                          config=LATE_PLATEAU)
    record = validate_trigger({
        "signal": "loss", "predicate": "below", "interval": 10,
        "threshold": 2.0, "max_fires": 1,
        "action": {"command": "start_decay"},
    })
    trainer.lr_triggers.register(record, 12)
    trainer.seek(100)
    feed(trainer, [1.0] * 100, poll_every=10)

    published = trainer.lr_triggers.status()[0]
    assert published["created_step"] == 12
    assert published["fires"] == 0
    assert published["cooldown"] is None
    # One fewer than the window count: registering at 12 discards the window
    # already in progress rather than averaging a fraction of it.
    assert published["refusals"] == 4
    assert published["last_refusal"] == "rejected_during_warmup"
    assert published["last_refusal_step"] == published["observation_step"]
    assert published["cooldown_left"] > 0
    assert published["cooldown_reason"] == "refusal"


def test_the_refusal_flag_is_set_on_every_path_not_only_where_it_is_read():
    """`cooldown_from_refusal` is read only while `cooldown_left > 0`, so a
    path that leaves it stale is invisible through `status()`. It is persisted
    state (D49), so a resume can hand a plateau trigger a True that its own
    refusal -- which holds nothing -- has to clear rather than inherit."""
    triggers = TriggerSet()
    triggers.load([{**validate_trigger(PLATEAU), "cooldown_from_refusal": True}])
    plateau = triggers.triggers[0]
    assert plateau.cooldown_from_refusal is True      # restored, not defaulted

    plateau.record_fire(True, result="rejected_during_warmup", step=100)
    assert plateau.cooldown_left == 0
    assert plateau.cooldown_from_refusal is False
    plateau.record_fire(False, result="applied", step=200)
    assert plateau.cooldown_from_refusal is False

    below = Trigger(**validate_trigger({
        "signal": "loss", "predicate": "below", "interval": 10,
        "threshold": 2.0, "max_fires": 2, "cooldown": 2,
        "action": {"command": "start_decay"}}))
    below.record_fire(True, result="rejected_during_warmup", step=100)
    assert below.cooldown_left == 2 and below.cooldown_from_refusal is True
    # Expiry ends the stand-off, so it ends the attribution too: the flag says
    # why the trigger is waiting, and after this it is not waiting.
    assert below.observe(9.0) is False
    assert below.cooldown_left == 1 and below.cooldown_from_refusal is True
    assert below.observe(9.0) is False
    assert below.cooldown_left == 0 and below.cooldown_from_refusal is False

    below.record_fire(False, result="applied", step=200)
    assert below.cooldown_left == 2 and below.cooldown_from_refusal is False
