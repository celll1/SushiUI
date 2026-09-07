"""Conditional LR-schedule triggers: a registered condition that presses a button.

§20 of ``docs/guides/LR_SCHEDULER_DESIGN.md`` (D45-D53, phase R6). A trigger is
not a new way to change the learning rate -- it is an automatic finger on the
controls §19 already built. Firing materialises an ordinary ``retarget`` /
``decay`` / ``cancel`` through ``ScheduleTimeline.add()``, so every §19 rule
(refusals, truncation, groups, blending) applies to it unchanged and the event
list stays free of conditions (invariant 18, D28).

Three things this module owns and nothing else does:

* the SIGNAL RING (D45). Values are pushed at the trainer's metrics site and
  drained at §5.2's seam (c). Nothing here reads the database: a per-step query
  in the training loop is the cost D45 exists to refuse.
* the OBSERVATION (D47). One observation is the mean of the samples that landed
  in one window of ``interval`` steps, and ``patience`` counts observations, not
  steps. Diffusion training draws a random timestep per step, so the
  step-to-step variance of the loss dominates its trend -- a plateau detector on
  raw per-step values fires on noise. That is structural, which is why there is
  no un-smoothed mode.
* the STATE (D49). Armed, best-so-far, the partial window, the patience counter,
  the fire count and the cooldown live in ``state.json`` and are
  restored on resume. They are never events, and no multiplier reads them
  (invariants 2 and 19).

``interval`` is on the GLOBAL step axis, the axis the signal is produced on and
the axis every other API-facing step count uses (§19.5.3-1). It is neither a
position on the curve nor a length along it, so it is never mapped through
``to_scheduler_axis`` -- flooring it would turn any interval below
``gradient_accumulation_steps`` into 0, which is §18.5's defect with no sentinel
to catch it. The EVENT a firing materialises is still dated on the scheduler
axis, like every other event (invariant 4).

Numbers: there are none. ``interval``, ``patience``, ``min_delta`` and
``threshold`` are required, because a usable value for any of them depends on
the run's own loss scale (D46, invariant 20). ``max_fires`` defaults to 1 and
lives in ``api/param_defaults.LR_TRIGGER_DEFAULTS``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple

from api.param_defaults import LR_TRIGGER_DEFAULTS
from core.training.lr_schedules import is_refused_result
from core.training.training_control_rpc import (
    COMMANDS as _BUTTON_COMMANDS,
    MAX_TRIGGER_FIRES,
    MAX_TRIGGERS,
    RETARGET_COMMAND,
)
from core.training.training_file_rpc import make_request_id

__all__ = [
    "EXTRA_SIGNAL_PREFIX",
    "INFO_TRIGGER_FIRED",
    "MAX_TRIGGER_FIRES",
    "MAX_TRIGGERS",
    "RESULT_DUPLICATE_TRIGGER",
    "RESULT_TRIGGER_LIMIT",
    "RESULT_TRIGGER_REGISTERED",
    "RESULT_TRIGGER_REMOVED",
    "RESULT_UNKNOWN_TRIGGER",
    "TRIGGER_ACTION_COMMANDS",
    "TRIGGER_PREDICATES",
    "TRIGGER_SIGNALS",
    "Trigger",
    "TriggerSet",
    "WARN_CORRUPT_TRIGGER_RECORD",
    "WARN_TRIGGER_REFUSED",
    "validate_trigger",
]

# D45: what the trainer already computes every step. `extra:<name>` reaches any
# key a trainer routes through `log_extra_metric`.
TRIGGER_SIGNALS = ("loss", "grad_norm")
EXTRA_SIGNAL_PREFIX = "extra:"

# D46. `plateau` reads `patience`/`min_delta`; `below`/`above` read `threshold`.
TRIGGER_PREDICATES = ("plateau", "below", "above")

# Required per predicate, with no default for any of them (D46). `interval` is
# required under every predicate: a threshold on a raw per-step diffusion loss
# is as noisy as a plateau on one.
_REQUIRED_FIELDS = {
    "plateau": ("interval", "patience", "min_delta"),
    "below": ("interval", "threshold"),
    "above": ("interval", "threshold"),
}
_PREDICATE_FIELDS = ("patience", "min_delta", "threshold")

# What a firing may issue: the same commands the operator has (§20.1). A
# `retarget` action is the retarget endpoint's own payload, so `op: scale` /
# `hold` / `undo` need no branch of their own here (§19.5.3).
TRIGGER_ACTION_COMMANDS = _BUTTON_COMMANDS + (RETARGET_COMMAND,)

# Resource bounds (MAX_TRIGGERS, MAX_TRIGGER_FIRES) live in
# training_control_rpc.py beside MAX_PENDING_REQUESTS (D60, D61).
RING_CAPACITY = 4096

RESULT_TRIGGER_REGISTERED = "registered"
RESULT_TRIGGER_REMOVED = "removed"
RESULT_UNKNOWN_TRIGGER = "rejected_unknown_trigger"
RESULT_DUPLICATE_TRIGGER = "rejected_duplicate_trigger_id"
RESULT_TRIGGER_LIMIT = "rejected_trigger_limit"

# §13. A firing that `add()` refused is not a firing (D51); the operator has to
# be told, because the trigger is still armed and still holds its `max_fires`.
WARN_TRIGGER_REFUSED = "lr_trigger_fire_refused"
WARN_CORRUPT_TRIGGER_RECORD = "lr_trigger_corrupt_record"
INFO_TRIGGER_FIRED = "lr_trigger_fired"


def _positive_int(payload: Mapping[str, Any], key: str, *, minimum: int) -> int:
    try:
        value = int(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"`{key}` must be a whole number of steps, "
                         f"got {payload.get(key)!r}.")
    if value < minimum:
        raise ValueError(f"`{key}` must be >= {minimum}, got {value}.")
    return value


def _float(payload: Mapping[str, Any], key: str,
           *, minimum: Optional[float] = None) -> float:
    try:
        value = float(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"`{key}` must be a number, got {payload.get(key)!r}.")
    if not math.isfinite(value):
        raise ValueError(f"`{key}` must be finite, got {value}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"`{key}` must be >= {minimum}, got {value}.")
    return value


def _validated_signal(signal: str) -> str:
    if not signal:
        raise ValueError(
            "A trigger needs a `signal` to watch: 'loss', 'grad_norm', or "
            "'extra:<name>' for any metric the run logs on the loss chart.")
    if signal == "learning_rate" or signal == EXTRA_SIGNAL_PREFIX + "lr":
        raise ValueError(
            "`learning_rate` is not a trigger signal (D45). A trigger CHANGES "
            "the learning rate, so watching it is a feedback loop whose "
            "semantics are not defined: the firing moves the value the "
            "condition reads, and a repeating trigger would then respond to "
            "its own effect. Watch what the change is meant to act on -- "
            "'loss', 'grad_norm', or 'extra:<name>'.")
    if signal in TRIGGER_SIGNALS:
        return signal
    if not signal.startswith(EXTRA_SIGNAL_PREFIX):
        raise ValueError(
            f"Unknown signal '{signal}'. Supported: "
            f"{', '.join(TRIGGER_SIGNALS)}, or 'extra:<name>' for a metric the "
            f"run logs through log_extra_metric.")
    name = signal[len(EXTRA_SIGNAL_PREFIX):]
    if not name:
        raise ValueError(
            "'extra:' needs the metric's name after the colon, e.g. "
            "'extra:known_loss'.")
    if name == "lr" or name.startswith("lr_"):
        # The trainer publishes the learning rate itself through the same extra
        # channel (`lr`, `lr_<component>`, `lr_decay_state`), so D45's refusal
        # has to cover the back door as well as the front one.
        raise ValueError(
            f"'{signal}' is the run's own learning-rate reporting, which D45 "
            f"excludes for the same reason as `learning_rate`: a trigger that "
            f"watches the value it changes has no defined semantics.")
    return signal


def _validated_action(action: Any) -> Dict[str, Any]:
    if not isinstance(action, Mapping) or not action:
        raise ValueError(
            "A trigger needs an `action`: the command to issue when it fires. "
            "It is a retarget payload -- the same body "
            "POST /training/runs/{id}/lr-schedule/retarget takes, including "
            "op 'scale' / 'hold' / 'undo' -- or "
            "{\"command\": \"start_decay\"} / {\"command\": \"cancel_decay\"}.")
    out = dict(action)
    command = str(out.pop("command", RETARGET_COMMAND) or RETARGET_COMMAND)
    if command not in TRIGGER_ACTION_COMMANDS:
        raise ValueError(
            f"Unknown action command '{command}'. Supported: "
            f"{', '.join(TRIGGER_ACTION_COMMANDS)}.")
    if command != RETARGET_COMMAND and out:
        raise ValueError(
            f"The '{command}' command takes no parameters; it uses the run's "
            f"own decay length and shape. Remove {', '.join(sorted(out))}.")
    if "at" in out:
        raise ValueError(
            "A trigger's action cannot name `at`: a firing takes effect at the "
            "step it fires at (D50). A fixed step would be in the past for "
            "every firing after the first, and refused as backdated.")
    return {"command": command, **out}


def validate_trigger(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The registration record, or a ValueError naming what is wrong.

    Everything decidable without the run, which for a trigger is everything but
    the duplicate-id and cap checks the trainer makes when it registers one.
    The caller validates the `action`'s retarget payload with the retarget
    endpoint's own validator, so an action that queues is an action that draws.
    """
    payload = dict(payload or {})
    signal = _validated_signal(str(payload.get("signal") or "").strip())

    predicate = str(payload.get("predicate") or "").strip()
    if predicate not in TRIGGER_PREDICATES:
        raise ValueError(
            f"Unknown predicate '{payload.get('predicate')}'. Supported: "
            f"{', '.join(TRIGGER_PREDICATES)}.")

    required = _REQUIRED_FIELDS[predicate]
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(
            f"A '{predicate}' trigger requires {', '.join(missing)}, and this "
            f"build supplies no default for any of them (D46): a usable "
            f"threshold, min_delta, patience or observation interval depends "
            f"entirely on this model's and dataset's loss scale, and a default "
            f"would be a number nobody measured. Read the run's loss chart and "
            f"choose them.")
    unused = [key for key in _PREDICATE_FIELDS
              if key not in required and payload.get(key) is not None]
    if unused:
        raise ValueError(
            f"A '{predicate}' trigger does not read {', '.join(unused)}, so "
            f"setting them would describe a condition it does not evaluate. "
            f"'plateau' reads patience and min_delta; 'below'/'above' read "
            f"threshold.")

    max_fires_value = payload.get("max_fires")
    if max_fires_value is None:
        max_fires_value = LR_TRIGGER_DEFAULTS["max_fires"]
    max_fires = _positive_int({"max_fires": max_fires_value},
                              "max_fires", minimum=1)
    if max_fires > MAX_TRIGGER_FIRES:
        raise ValueError(
            f"`max_fires` must be <= {MAX_TRIGGER_FIRES} (D61), got {max_fires}: "
            f"each firing anchors a blend on the timeline, and an unbounded "
            f"chain makes scheduler evaluation walk every past anchor.")
    cooldown = payload.get("cooldown")
    if max_fires > 1:
        if cooldown is None:
            raise ValueError(
                "A trigger with max_fires > 1 requires a `cooldown`, in "
                "observations (D48): without one the same plateau fires it again "
                "on the very next observation, spending every fire at once.")
        cooldown_val = _positive_int(payload, "cooldown", minimum=1)
    elif cooldown is not None:
        raise ValueError(
            "`cooldown` has nothing to space out at max_fires = 1: there is no "
            "second firing to delay. Raise max_fires, or drop the cooldown.")
    else:
        cooldown_val = None

    record: Dict[str, Any] = {
        "id": str(payload.get("id") or "").strip() or make_request_id(),
        "signal": signal,
        "predicate": predicate,
        "interval": _positive_int(payload, "interval", minimum=1),
        "max_fires": max_fires,
        "action": _validated_action(payload.get("action")),
    }
    if predicate == "plateau":
        record["patience"] = _positive_int(payload, "patience", minimum=1)
        record["min_delta"] = _float(payload, "min_delta", minimum=0.0)
    else:
        record["threshold"] = _float(payload, "threshold")
    if cooldown_val is not None:
        record["cooldown"] = cooldown_val
    return record


@dataclass
class Trigger:
    """One registered condition and everything the resume has to restore.

    The spec fields come from `validate_trigger`; the rest is the state D49
    puts in state.json. `window*` is the observation being accumulated, so a
    resume continues a partial window instead of restarting it.
    """

    id: str
    signal: str
    predicate: str
    interval: int
    max_fires: int
    action: Dict[str, Any] = field(default_factory=dict)
    patience: Optional[int] = None
    min_delta: Optional[float] = None
    threshold: Optional[float] = None
    cooldown: Optional[int] = None
    # The step registration happened at. The window in progress then is
    # discarded rather than averaged over a fraction of its steps.
    created_step: int = 0

    fires: int = 0
    misses: int = 0
    cooldown_left: int = 0
    best: Optional[float] = None
    observations: int = 0
    last_value: Optional[float] = None
    last_step: Optional[int] = None
    window: Optional[int] = None
    window_sum: float = 0.0
    window_count: int = 0

    @property
    def armed(self) -> bool:
        """D48: a trigger that has spent its fires stays, disarmed, so what it
        did is still readable."""
        return self.fires < self.max_fires

    def sample(self, step: int, value: float) -> Optional[Tuple[int, float]]:
        """Take one raw signal value; return an observation when one closes.

        The window is `step // interval`, an absolute grid rather than a count
        since the last one, so where a run started or resumed cannot shift the
        boundaries. A window with no sample in it produces no observation --
        `grad_norm` only exists on update boundaries, so under
        gradient accumulation most windows of a short interval hold fewer
        samples than steps, and an empty one must not spend patience.
        """
        step = int(step)
        if step < self.created_step:
            return None
        window = step // self.interval
        closed = None
        if self.window is not None and window != self.window:
            closed = self._close()
        if self.window != window:
            self.window = window
            self.window_sum = 0.0
            self.window_count = 0
        self.window_sum += float(value)
        self.window_count += 1
        return closed

    def _close(self) -> Optional[Tuple[int, float]]:
        if self.window is None or self.window_count <= 0:
            return None
        start = self.window * self.interval
        if start < self.created_step:
            return None
        self.last_step = start + self.interval - 1
        return self.last_step, self.window_sum / self.window_count

    def observe(self, value: float) -> bool:
        """Score one observation; True means fire.

        Improvement is a DECREASE: every signal D45 admits (`loss`,
        `grad_norm`, and the extra metrics, which are losses) is minimised, and
        §20.2's record has no field to say otherwise.
        """
        self.observations += 1
        self.last_value = float(value)
        if not self.armed:
            return False
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return False
        if self.predicate == "plateau":
            if self.best is None or value < self.best - float(self.min_delta):
                self.best = float(value)
                self.misses = 0
                return False
            self.misses += 1
            return self.misses >= int(self.patience)
        if self.predicate == "below":
            return value < float(self.threshold)
        return value > float(self.threshold)

    def record_fire(self, refused: bool) -> None:
        """Book a firing, or unbook a refused one (D51).

        A refusal costs neither a fire nor a cooldown -- a once-only trigger
        must not lose its one chance to a decay refused during a warmup. The
        patience counter restarts, so the next attempt is another `patience`
        observations away rather than on every observation from here.

        Threshold predicates (below/above) have no patience counter, so a
        refusal debounces by one observation (or cooldown if configured) to
        prevent a warning storm on every single observation (D64).
        """
        self.misses = 0
        if refused:
            if self.predicate != "plateau":
                self.cooldown_left = int(self.cooldown or 1)
            return
        self.fires += 1
        # The best-so-far belongs to the LR the run had; after a change it is
        # not a baseline any more.
        self.best = None
        self.cooldown_left = int(self.cooldown or 0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trigger":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(data or {}).items() if k in known})

    def status(self) -> Dict[str, Any]:
        """D52: what it watches, where it is, and how much is left.

        `patience_used` / `observations_to_fire` are the two numbers "how much
        longer" needs; without them an automation is worse than watching.
        """
        remaining = None
        if self.predicate == "plateau" and self.patience:
            remaining = max(0, int(self.patience) - self.misses)
        return {
            "id": self.id,
            "signal": self.signal,
            "predicate": self.predicate,
            "interval": self.interval,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "threshold": self.threshold,
            "max_fires": self.max_fires,
            "cooldown": self.cooldown,
            "action": dict(self.action),
            "armed": self.armed,
            "fires": self.fires,
            "fires_left": max(0, self.max_fires - self.fires),
            "cooldown_left": self.cooldown_left,
            "observations": self.observations,
            "observation": self.last_value,
            "observation_step": self.last_step,
            "best": self.best,
            "patience_used": self.misses if self.predicate == "plateau" else None,
            "observations_to_fire": remaining,
        }


class TriggerSet:
    """The run's triggers and the in-memory signal ring they read (D45).

    Fed at the trainer's metrics site, drained at seam (c). Both are the
    training thread; the ring exists so the evaluation is not a database query
    per step, and its bound only matters if a poll stops happening.
    """

    def __init__(self) -> None:
        self.triggers: List[Trigger] = []
        self._ring: Deque[Tuple[int, str, float]] = deque(maxlen=RING_CAPACITY)

    def __len__(self) -> int:
        return len(self.triggers)

    def __iter__(self):
        return iter(self.triggers)

    def push(self, step: int, signal: str, value: Any) -> None:
        """Record one raw sample. Cheap enough to call unconditionally."""
        if not self.triggers:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        self._ring.append((int(step), str(signal), v))

    def drain(self) -> List[Tuple[int, str, float]]:
        samples = list(self._ring)
        self._ring.clear()
        return samples

    def get(self, trigger_id: str) -> Optional[Trigger]:
        for trigger in self.triggers:
            if trigger.id == str(trigger_id):
                return trigger
        return None

    def register(self, record: Mapping[str, Any], step: int) -> str:
        """Install a validated record. Refusals the API process cannot make."""
        trigger = Trigger.from_dict({**dict(record), "created_step": int(step)})
        if self.get(trigger.id) is not None:
            return RESULT_DUPLICATE_TRIGGER
        if len(self.triggers) >= MAX_TRIGGERS:
            return RESULT_TRIGGER_LIMIT
        self.triggers.append(trigger)
        return RESULT_TRIGGER_REGISTERED

    def remove(self, trigger_id: str) -> str:
        trigger = self.get(trigger_id)
        if trigger is None:
            return RESULT_UNKNOWN_TRIGGER
        self.triggers.remove(trigger)
        return RESULT_TRIGGER_REMOVED

    def evaluate(self, fire: Callable[[Trigger, float, int], str],
                 on_error: Optional[Callable[[Trigger, Exception], None]] = None
                 ) -> List[Tuple[Trigger, str]]:
        """Drain the ring, close observations, fire what is due.

        ``fire`` materialises the trigger's action and returns the timeline's
        result code; a `rejected_*` one is not a firing (D51). Returns one
        entry per attempt, refusals included, for the caller to report.

        Each trigger is evaluated inside its own try/except block so one broken
        trigger cannot starve healthy ones on the same run (D63).
        """
        outcomes: List[Tuple[Trigger, str]] = []
        if not self.triggers:
            self._ring.clear()
            return outcomes
        for step, signal, value in self.drain():
            for trigger in list(self.triggers):
                try:
                    if trigger.signal != signal:
                        continue
                    closed = trigger.sample(step, value)
                    if closed is None:
                        continue
                    observed_step, observed = closed
                    if not trigger.observe(observed):
                        continue
                    result = fire(trigger, observed, observed_step)
                    trigger.record_fire(is_refused_result(result))
                    outcomes.append((trigger, result))
                except Exception as e:
                    if on_error:
                        on_error(trigger, e)
                    else:
                        print(f"WARNING: trigger '{trigger.id}' evaluation failed: {e}")
        return outcomes

    def dump(self) -> List[Dict[str, Any]]:
        """D49: state.json's `lr_schedule_triggers`. Never the event list."""
        return [t.to_dict() for t in self.triggers]

    def dump_signals(self) -> List[Tuple[int, str, float]]:
        """Preserve samples awaiting the next batch-head evaluation."""
        return list(self._ring)

    def load(self, records: Optional[List[Mapping[str, Any]]],
             warn: Optional[Callable[[str], None]] = None,
             signals: Optional[List[Tuple[int, str, float]]] = None) -> None:
        """D63: restore triggers from state.json, validating each record.

        Invalid records are dropped with a warning rather than raising into
        seam (b), so a bad entry in a checkpoint cannot kill a resume.
        """
        loaded: List[Trigger] = []
        for r in (records or []):
            if not isinstance(r, Mapping):
                if warn:
                    warn(f"Discarding invalid non-dict trigger record: {r!r}")
                continue
            try:
                trigger = Trigger.from_dict({**r, **validate_trigger(r)})
                for key in ("created_step", "fires", "misses", "cooldown_left",
                            "observations", "window_count", "window", "last_step"):
                    value = getattr(trigger, key)
                    if value is None and key in ("window", "last_step"):
                        continue
                    if type(value) is not int or value < 0:
                        raise ValueError(f"Invalid trigger state {key}: {value!r}")
                for key in ("best", "last_value", "window_sum"):
                    value = getattr(trigger, key)
                    if value is None and key != "window_sum":
                        continue
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError(f"Invalid trigger state {key}: {value!r}")
                if len(loaded) >= MAX_TRIGGERS or any(t.id == trigger.id for t in loaded):
                    raise ValueError("Duplicate trigger id or trigger limit exceeded")
                loaded.append(trigger)
            except Exception as e:
                msg = f"Discarding corrupt trigger record {r.get('id') or '<unknown>'}: {e}"
                if warn:
                    warn(msg)
                else:
                    print(f"WARNING: {msg}")
        self.triggers = loaded
        self._ring.clear()
        for sample in (signals or []):
            try:
                step, signal, value = sample
                self.push(step, signal, value)
            except (TypeError, ValueError, OverflowError):
                if warn:
                    warn(f"Discarding corrupt pending trigger sample: {sample!r}")

    def status(self) -> List[Dict[str, Any]]:
        return [t.status() for t in self.triggers]

    def signature(self) -> Tuple[Any, ...]:
        """What has to change for `.lr_schedule.json` to be rewritten.

        Ticks once per observation rather than once per step, which is what
        keeps a published patience counter from costing an atomic write a batch.
        """
        return tuple((t.id, t.observations, t.fires, t.misses, t.cooldown_left)
                     for t in self.triggers)
