"""LR schedule definitions, resolution, runtime timeline and construction.

Phases P0 and P1 of ``docs/guides/LR_SCHEDULER_DESIGN.md``: the one place a
learning-rate schedule is built for the diffusion trainers. Everything it
returns is a ``torch.optim.lr_scheduler.LambdaLR`` whose multiplier is a PURE
function of the step and of the timeline's events -- the invariant
``lr_utils.reassert_config_lr`` and ``BaseTrainer._fast_forward_one_lr_scheduler``
both evaluate lambdas out of order and rely on.

Step axis (D9/§17.1): one unit per ``scheduler.step()``, i.e. per update
boundary, NOT per ``global_step``. ``BaseTrainer`` divides by the effective
advance interval before calling ``resolve_spec``.

P0 ported the six diffusers schedules and the in-house ``plateau_cosine_floor``
so that the multiplier is bit-identical for ``0 <= s <= total_steps``. Three
deliberate exceptions, listed in the design's §18: ``constant`` now warms up
when ``lr_warmup_steps > 0``, ``cosine`` holds its terminal value past
``total_steps`` instead of rising again, and the gas axis fix moves where a
resume lands.

P1 adds the runtime timeline: the ``total_steps`` warp of §7.2 and the
BASE/DECAYING/FLOOR/RECOVERING overlay of §5.3 as §17.3 restates it. The
overlay state is derived per SPEC from one shared event list, so P6's per-group
schedules can diverge without a second timeline. P2 feeds it commands through
``training_control_rpc``; the ReLoRA ``restart`` event is P4, and the
generalized floor with the ``wsd``/``rex`` names is P3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from torch.optim.lr_scheduler import LambdaLR

# SSoT: api/param_defaults.TRAINING_DEFAULTS.
from api.param_defaults import TRAINING_DEFAULTS as _TRAINING_DEFAULTS

__all__ = [
    "LR_SCHEDULER_NAMES",
    "OverlayState",
    "STATE_BASE",
    "STATE_DECAYING",
    "STATE_FLOOR",
    "STATE_NAMES",
    "STATE_RECOVERING",
    "ScheduleSpec",
    "ScheduleTimeline",
    "base_multiplier",
    "build_lr_scheduler",
    "describe_spec",
    "make_lambda",
    "resolve_spec",
]

# The vocabulary P0 implements. D18's validator/enum, and the `wsd` / `rex`
# names, are P3.
LR_SCHEDULER_NAMES = (
    "constant",
    "constant_with_warmup",
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "plateau_cosine_floor",
)

# diffusers' get_polynomial_decay_schedule_with_warmup defaults, which neither
# construction site ever overrode. Its floor is lr_end/lr_init -- NOT 0
# (optimization.py:226, :252-270). P3 replaces it with an explicit floor.
_POLYNOMIAL_LR_END = 1e-7
_POLYNOMIAL_POWER = 1.0
# get_scheduler's num_cycles default, likewise never passed: today's
# `cosine_with_restarts` is a single cosine (§1-1).
_RESTART_CYCLES = 1.0

# Overlay states (§5.3). The integers D19 emits as `lr_decay_state` in P2.
STATE_BASE = 0
STATE_DECAYING = 1
STATE_FLOOR = 2
STATE_RECOVERING = 3
# The strings `.lr_schedule.json` and the API report; the integers are what the
# `lr_decay_state` metric carries.
STATE_NAMES = {
    STATE_BASE: "base",
    STATE_DECAYING: "decaying",
    STATE_FLOOR: "floor",
    STATE_RECOVERING: "recovering",
}

_CURVES = ("constant", "linear", "cosine", "cosine_with_restarts",
           "polynomial", "wsd")

# `add` results meaning "recorded, but the state machine must never fold it":
# the request was refused, for every group, before it became an event.
_REFUSED = ("rejected_during_warmup", "rejected_zero_length")


@dataclass(frozen=True)
class ScheduleSpec:
    """What config resolves to. Immutable for the lifetime of the scheduler.

    Axes (D8/§17.2): ``warmup_steps`` and ``decay_length`` are REAL scheduler
    steps; ``total_steps`` and a ``decay_start_axis="nominal"`` start are read
    through the timeline's clock, which P1 warps when ``total_steps`` changes.
    ``decay_start_step=None`` means "no configured decay" (the external ``D=0``
    P3 exposes as manual WSD), which is distinct from a start of 0.
    """

    name: str
    curve: str
    warmup_steps: int
    total_steps: int
    floor_ratio: float = 0.0
    decay_start_step: Optional[int] = None
    # §17.2: an alias's derived start resolves from the NOMINAL total, so a
    # resume with a new total_steps does not move it. Equal to
    # decay_start_step until the timeline warps.
    decay_start_ratio: Optional[float] = None
    decay_start_axis: str = "nominal"
    decay_length: Optional[int] = None
    decay_end_kind: str = "nominal_total"
    decay_shape: str = "cosine"
    poly_lr_end: float = _POLYNOMIAL_LR_END
    # Bound to the optimizer by build_lr_scheduler: diffusers' polynomial
    # multiplier is expressed relative to optimizer.defaults["lr"].
    poly_lr_init: Optional[float] = None


@dataclass(frozen=True)
class OverlayState:
    """The runtime overlay at one step, for one spec (§5.3).

    ``start_multiplier`` is the value the phase began at: ``m_start`` while
    DECAYING, ``m_c`` while RECOVERING. ``length`` is the decay's ``L`` (None =
    "to the nominal end") or the recovery's ``R``. ``decay_disarmed`` records
    that a cancel voided the config-declared WSD decay (§17.3).
    """

    code: int = STATE_BASE
    at: int = 0
    start_multiplier: float = 1.0
    length: Optional[int] = None
    shape: str = "cosine"
    decay_disarmed: bool = False


class ScheduleTimeline:
    """The runtime events every scheduler of one run shares.

    Ordered by the persisted ``(at, seq)``. Events at the SAME step are
    ACCEPTED: refusing them would drop a resume-time recomputation or a
    cancellation (§17.3). ``request_id`` makes re-delivery idempotent.

    Mutates only at the four seams of §5.2 -- build, the resume load before the
    fast-forward, the command poll, ReLoRA's merge. Every other method here is
    a pure read, evaluable out of order and repeatedly; the lambda closes over
    this OBJECT, so a resume can install saved events after construction.
    """

    def __init__(self, events: Optional[Sequence[Mapping[str, Any]]] = None):
        self.events: List[Dict[str, Any]] = []
        self._next_seq = 0
        self.spec: Optional[ScheduleSpec] = None
        if events:
            self.load(events)

    # -- seams -----------------------------------------------------------

    def set_total_steps(self, value: int) -> None:
        """Seam (a), the construction event. Re-anchoring is a different seam
        with a different anchor: ``add("total_steps", at=<resume step>, ...)``.
        """
        if any(e.get("kind") == "total_steps" for e in self.events):
            raise NotImplementedError(
                "re-anchoring total_steps is the extension warp: "
                "add('total_steps', at=<resume step>, value=...)")
        self._append({"kind": "total_steps", "at": 0, "value": int(value)})

    def add(self, kind: str, at: int, *, spec: Optional[ScheduleSpec] = None,
            request_id: Optional[str] = None, **payload: Any) -> str:
        """Record one event, returning its §5.3 result code.

        The code is scored against the REPRESENTATIVE spec (bound at build, or
        passed as ``spec=``). The per-group state machine re-derives every
        event's effect from that group's own curve, so a stored ``result`` is a
        record of what the requester was told -- never an input to `state_at`.
        """
        at = int(at)
        if request_id is not None:
            for event in self.events:
                if event.get("request_id") == request_id:
                    return str(event.get("result", "applied"))

        if kind == "total_steps":
            value = int(payload["value"])
            # Totals are >= 1, so 0 stands for "no anchor recorded yet".
            if value == self.current_total(0):
                return "ignored_unchanged"
            self._append({"kind": kind, "at": at, "value": value,
                          "request_id": request_id, "result": "applied"})
            return "applied"

        if kind not in ("decay", "cancel"):
            raise ValueError("Unknown timeline event kind: " + repr(kind))

        resolved = spec or self.spec
        if resolved is None:
            raise ValueError(
                "decay/cancel need a spec: build_lr_scheduler binds the "
                "representative one, or pass spec= explicitly")

        event: Dict[str, Any] = {"kind": kind, "at": at,
                                 "request_id": request_id}
        length = payload.get("length")
        if kind == "decay":
            event["length"] = None if length is None else int(length)
            event["shape"] = str(payload.get("shape") or resolved.decay_shape)
            _decay_shape(event["shape"])  # refuse a P3 shape at the seam
        else:
            # §5.4: the recovery length is baked in, so a later config edit
            # cannot reshape a cancel that already happened.
            event["length"] = int(resolved.warmup_steps if length is None
                                  else length)

        state = self._fold(resolved, at)
        if kind == "decay":
            _, result = self._apply_decay(resolved, state, event)
        else:
            _, result = self._apply_cancel(resolved, state, event)
        if result in _REFUSED:
            # Kept so re-delivering the request_id answers the same thing.
            # `noop` is a kind the state machine does not know, so a refused
            # command can never take effect for any group.
            event["refused_kind"] = event["kind"]
            event["kind"] = "noop"
        event["result"] = result
        self._append(event)
        return result

    def load(self, events: Optional[Sequence[Mapping[str, Any]]],
             upto_step: Optional[int] = None) -> None:
        """Seam (b): install a saved event list, BEFORE the fast-forward.

        ``upto_step`` drops later commands, the same semantics as
        ``_cleanup_future_metrics``: rewinding to an earlier checkpoint un-does
        what was ordered after it.
        """
        self.events = []
        self._next_seq = 0
        for index, event in enumerate(events or []):
            record = dict(event)
            if upto_step is not None and int(record.get("at", 0)) > int(upto_step):
                continue
            record.setdefault("at", 0)
            record.setdefault("seq", index)
            self.events.append(record)
            self._next_seq = max(self._next_seq, int(record["seq"]) + 1)
        self.events.sort(key=_order)

    def dump(self, upto_step: int) -> List[Dict[str, Any]]:
        """JSON-ready events with ``at <= upto_step``, in application order."""
        return [dict(e) for e in self._sorted()
                if int(e.get("at", 0)) <= int(upto_step)]

    def bind_spec(self, spec: ScheduleSpec) -> None:
        """Name the representative spec `add` scores result codes against."""
        if self.spec is None:
            self.spec = spec

    # -- pure reads ------------------------------------------------------

    def nominal_total(self, default: int) -> int:
        """``T_nominal``: the FIRST total_steps event, the axis §7.2 warps onto."""
        for event in self._sorted():
            if event.get("kind") == "total_steps":
                return int(event["value"])
        return int(default)

    def current_total(self, default: int) -> int:
        """The LAST total_steps event: this session's real-axis end."""
        value = None
        for event in self._sorted():
            if event.get("kind") == "total_steps":
                value = int(event["value"])
        return int(default) if value is None else value

    def clock(self, step: int) -> float:
        """Real scheduler step -> nominal axis (§7.2).

        ``tau(s) = tau_1(tau_2(...tau_n(s)))``: the latest anchor is applied
        first, so a step before an anchor keeps the mapping it already had and
        the curve before a resume stays bit-identical.
        """
        anchors = [e for e in self._sorted() if e.get("kind") == "total_steps"]
        value = float(step)
        for i in range(len(anchors) - 1, 0, -1):
            at = float(anchors[i]["at"])
            if value < at:
                continue
            previous = float(anchors[i - 1]["value"])
            current = float(anchors[i]["value"])
            if current > at:
                value = at + (value - at) * (previous - at) / (current - at)
            else:
                # Shrunk to at or below the anchor: already at the old end.
                value = previous
        return value

    def state_at(self, spec: ScheduleSpec, step: int) -> OverlayState:
        """The overlay state at ``step`` for ONE spec/group (§17.3)."""
        return self._fold(spec, int(step))

    def multiplier(self, spec: ScheduleSpec, step: int) -> float:
        """The LR multiplier: §4's base curve with the overlay on top."""
        step = int(step)
        return self._value(spec, self._fold(spec, step), step)

    # -- internals -------------------------------------------------------

    def _append(self, event: Dict[str, Any]) -> None:
        event["seq"] = self._next_seq
        self._next_seq += 1
        self.events.append(event)
        self.events.sort(key=_order)

    def _sorted(self) -> List[Dict[str, Any]]:
        return sorted(self.events, key=_order)

    def _fold(self, spec: ScheduleSpec, step: int) -> OverlayState:
        """Apply every event at or before ``step``, in arrival order.

        This is also what an event arriving AT ``step`` sees: earlier same-step
        events carry a smaller ``seq``, so they are already folded in, and the
        trailing advance is the same one the replay does before applying it.
        """
        state = OverlayState()
        for event in self._sorted():
            at = int(event.get("at", 0))
            if at > step:
                break
            kind = event.get("kind")
            if kind == "decay":
                state = self._advance(spec, state, at)
                state, _ = self._apply_decay(spec, state, event)
            elif kind == "cancel":
                state = self._advance(spec, state, at)
                state, _ = self._apply_cancel(spec, state, event)
        return self._advance(spec, state, step)

    def _apply_decay(self, spec: ScheduleSpec, state: OverlayState,
                     event: Mapping[str, Any]) -> Tuple[OverlayState, str]:
        at = int(event.get("at", 0))
        if at < spec.warmup_steps:
            # Also what keeps a ramp from being read as "a decay that raises
            # the LR", and q's denominator away from zero.
            return state, "rejected_during_warmup"
        if state.code in (STATE_DECAYING, STATE_FLOOR):
            return state, "ignored_already_decaying"
        length = event.get("length")
        if length is None:
            if self.nominal_total(spec.total_steps) - self.clock(at) <= 0:
                return state, "rejected_zero_length"
        elif int(length) <= 0:
            return state, "rejected_zero_length"
        return OverlayState(
            code=STATE_DECAYING, at=at,
            start_multiplier=self._value(spec, state, at),
            length=None if length is None else int(length),
            shape=str(event.get("shape") or spec.decay_shape),
            decay_disarmed=state.decay_disarmed,
        ), "applied"

    def _apply_cancel(self, spec: ScheduleSpec, state: OverlayState,
                      event: Mapping[str, Any]) -> Tuple[OverlayState, str]:
        at = int(event.get("at", 0))
        length = event.get("length")
        recovery = int(spec.warmup_steps if length is None else length)
        if state.code == STATE_RECOVERING:
            return state, "ignored_already_recovering"
        if state.code in (STATE_DECAYING, STATE_FLOOR):
            return OverlayState(
                code=STATE_RECOVERING, at=at,
                start_multiplier=self._value(spec, state, at),
                length=recovery, shape=state.shape, decay_disarmed=True,
            ), "applied"
        if _has_config_decay(spec) and not state.decay_disarmed:
            # §17.3: a cancel voids the config-declared WSD decay too. Already
            # past its start, that is a recovery; before it, a disarm.
            if self._config_decay_started(spec, at):
                return OverlayState(
                    code=STATE_RECOVERING, at=at,
                    start_multiplier=self._base(spec, at, False),
                    length=recovery, decay_disarmed=True,
                ), "applied"
            return replace(state, decay_disarmed=True), "disarmed_scheduled_decay"
        return state, "ignored_no_active_decay"

    def _advance(self, spec: ScheduleSpec, state: OverlayState,
                 step: int) -> OverlayState:
        """The transitions time makes on its own (§5.3's last two rows)."""
        if state.code == STATE_DECAYING and self._q(spec, state, step) >= 1.0:
            return replace(state, code=STATE_FLOOR)
        if (state.code == STATE_RECOVERING
                and step >= state.at + int(state.length or 0)):
            return OverlayState(decay_disarmed=state.decay_disarmed)
        return state

    def _q(self, spec: ScheduleSpec, state: OverlayState, step: int) -> float:
        if state.length is None:
            span = self.nominal_total(spec.total_steps) - self.clock(state.at)
            elapsed = self.clock(step) - self.clock(state.at)
        else:
            span = float(state.length)
            elapsed = float(step - state.at)
        if span <= 0:
            return 1.0
        return min(1.0, max(0.0, elapsed / span))

    def _config_decay_started(self, spec: ScheduleSpec, step: int) -> bool:
        start = _config_decay_start(spec, self.nominal_total(spec.total_steps))
        if start is None:
            return False
        position = (self.clock(step) if spec.decay_start_axis == "nominal"
                    else float(step))
        return position >= float(start)

    def _value(self, spec: ScheduleSpec, state: OverlayState,
               step: int) -> float:
        if state.code == STATE_DECAYING:
            floor = spec.floor_ratio
            shape = _decay_shape(state.shape)
            return floor + (state.start_multiplier - floor) * shape(
                self._q(spec, state, step))
        if state.code == STATE_FLOOR:
            return spec.floor_ratio
        if state.code == STATE_RECOVERING:
            base = self._base(spec, step, state.decay_disarmed)
            recovery = int(state.length or 0)
            if recovery <= 0:
                return base
            ratio = min(1.0, max(0.0, (step - state.at) / float(recovery)))
            return state.start_multiplier + (base - state.start_multiplier) * ratio
        return self._base(spec, step, state.decay_disarmed)

    def _base(self, spec: ScheduleSpec, step: int, disarmed: bool) -> float:
        return base_multiplier(spec, self, step, decay_disarmed=disarmed)


def _order(event: Mapping[str, Any]) -> Tuple[int, int]:
    return (int(event.get("at", 0)), int(event.get("seq", 0)))


def _config_decay_start(spec: ScheduleSpec,
                        nominal_total: int) -> Optional[int]:
    """``D`` on the nominal axis. An alias re-derives it from ``T_nominal``
    every time (§17.2), so an extension does not drag the plateau along."""
    if spec.decay_start_ratio is not None:
        T = int(nominal_total)
        return max(spec.warmup_steps,
                   min(round(spec.decay_start_ratio * T), T))
    return spec.decay_start_step


def _has_config_decay(spec: ScheduleSpec) -> bool:
    return spec.curve == "wsd" and (spec.decay_start_ratio is not None
                                    or spec.decay_start_step is not None)


def _lookup(config: Optional[Mapping[str, Any]], key: str) -> Any:
    value = (config or {}).get(key, _TRAINING_DEFAULTS[key])
    return _TRAINING_DEFAULTS[key] if value is None else value


def resolve_spec(
    config: Optional[Mapping[str, Any]],
    *,
    warmup_steps: int,
    total_steps: int,
    name: str,
) -> ScheduleSpec:
    """Resolve the run's config into an immutable spec.

    ``total_steps`` must already be on the scheduler axis (``T_sched``).
    """
    key = str(name).strip().lower()
    if key not in LR_SCHEDULER_NAMES:
        raise ValueError(
            f"Unknown lr_scheduler '{name}'. Supported: "
            f"{', '.join(LR_SCHEDULER_NAMES)}"
        )

    W = max(0, int(warmup_steps or 0))
    T = max(1, int(total_steps))

    if key == "plateau_cosine_floor":
        # D11: an alias for wsd -- decay start on the NOMINAL axis, length "to
        # the nominal end", never stored as an explicit real length (§17.2).
        ratio = float(_lookup(config, "lr_decay_start_ratio"))
        floor = float(_lookup(config, "lr_floor_ratio"))
        start = max(W, min(round(ratio * T), T))
        return ScheduleSpec(
            name=key, curve="wsd", warmup_steps=W, total_steps=T,
            floor_ratio=floor, decay_start_step=start, decay_start_ratio=ratio,
            decay_start_axis="nominal", decay_length=None,
            decay_end_kind="nominal_total", decay_shape="cosine",
        )

    # The floor is per-curve in P0 (polynomial's is diffusers' own; the others
    # have none). Applying lr_floor_ratio to every curve is D10, in P3.
    curve = "constant" if key == "constant_with_warmup" else key
    return ScheduleSpec(name=key, curve=curve, warmup_steps=W, total_steps=T)


def _decay_shape(name: str) -> Callable[[float], float]:
    if name == "cosine":
        return lambda q: 0.5 * (1.0 + math.cos(math.pi * q))
    raise NotImplementedError(f"decay_shape '{name}' ships in P3")


def base_multiplier(spec: ScheduleSpec, timeline: ScheduleTimeline, step: int,
                    *, decay_disarmed: bool = False) -> float:
    """``m_base(s)``: §4's curve, with no runtime overlay on it.

    Reads ``timeline`` only through ``nominal_total``/``clock``, so it stays a
    pure function of ``(step, timeline.events)``.
    """
    W = spec.warmup_steps
    if W > 0 and step < W:
        return step / float(W)

    curve = spec.curve
    if curve == "constant":
        # constant_with_warmup's shape. diffusers' bare `constant` ignored
        # num_warmup_steps entirely (optimization.py:323-324); §4.2 makes the
        # two one curve.
        return 1.0

    T = timeline.nominal_total(spec.total_steps)

    if curve == "linear":
        return max(0.0, float(T - timeline.clock(step)) / float(max(1, T - W)))

    if curve == "cosine":
        progress = float(timeline.clock(step) - W) / float(max(1, T - W))
        # Clamped, where diffusers lets the cosine rise again past T.
        if progress > 1.0:
            progress = 1.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    if curve == "cosine_with_restarts":
        progress = float(timeline.clock(step) - W) / float(max(1, T - W))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(
            math.pi * ((_RESTART_CYCLES * progress) % 1.0))))

    if curve == "polynomial":
        lr_init = spec.poly_lr_init
        lr_end = spec.poly_lr_end
        s = timeline.clock(step)
        if s > T:
            return lr_end / lr_init
        pct_remaining = 1 - (s - W) / (T - W)
        decay = (lr_init - lr_end) * pct_remaining ** _POLYNOMIAL_POWER + lr_end
        return decay / lr_init

    if curve == "wsd":
        D = _config_decay_start(spec, T)
        if D is None or decay_disarmed:
            # Manual WSD not started, or a cancel voided the config decay
            # (§17.3). Holds 1 past T as well.
            return 1.0
        F = spec.floor_ratio
        position = (timeline.clock(step)
                    if spec.decay_start_axis == "nominal" else float(step))
        # Tested before the q form so that D == T holds the floor rather than
        # restarting the decay (the old lambda's max(1, T - D)).
        if position >= T:
            return F
        if position < D:
            return 1.0
        q = (position - D) / float(max(1, T - D))
        return F + (1.0 - F) * _decay_shape(spec.decay_shape)(q)

    raise ValueError(f"Unknown schedule curve '{curve}'")


def make_lambda(spec: ScheduleSpec, timeline: ScheduleTimeline) -> Callable[[int], float]:
    """The multiplier as a pure function of ``(step, timeline.events)``."""
    if spec.curve not in _CURVES:
        raise ValueError(f"Unknown schedule curve '{spec.curve}'")
    if spec.curve == "polynomial" and spec.poly_lr_init is None:
        raise ValueError(
            "polynomial needs poly_lr_init; build it through build_lr_scheduler()")
    if spec.curve == "wsd":
        _decay_shape(spec.decay_shape)
        if spec.decay_end_kind != "nominal_total" or spec.decay_length is not None:
            raise NotImplementedError("explicit config decay lengths ship in P3")

    def lr_lambda(step: int) -> float:
        return timeline.multiplier(spec, step)

    return lr_lambda


def describe_spec(spec: ScheduleSpec) -> str:
    """One line for the startup log (§13)."""
    parts = [f"warmup={spec.warmup_steps}", f"total={spec.total_steps}"]
    if spec.curve == "wsd":
        parts.append(f"decay_start={spec.decay_start_step}")
        parts.append(f"decay_shape={spec.decay_shape}")
        parts.append(f"floor_ratio={spec.floor_ratio}")
    if spec.curve == "polynomial":
        parts.append(f"lr_end={spec.poly_lr_end}")
    return f"{spec.name} ({', '.join(parts)}) [scheduler steps]"


def build_lr_scheduler(
    optimizer,
    spec: ScheduleSpec,
    timeline: ScheduleTimeline,
    group_names: Optional[Sequence[str]] = None,
    group_schedules: Optional[Mapping[str, str]] = None,
) -> LambdaLR:
    """The only place this project constructs an LR scheduler.

    Always a ``LambdaLR`` carrying a LIST of lambdas, one per param group, so
    ``lr_utils.reassert_config_lr``'s ``len(lambdas) == n_groups`` test and the
    fast-forward's zip hold without a special case.
    """
    if group_schedules:
        raise NotImplementedError("lr_group_schedules ships in P6")
    del group_names  # P6

    if spec.curve == "polynomial":
        lr_init = float(optimizer.defaults["lr"])
        if not lr_init > spec.poly_lr_end:
            raise ValueError(
                f"lr_end ({spec.poly_lr_end}) must be smaller than initial lr "
                f"({lr_init})")
        spec = replace(spec, poly_lr_init=lr_init)

    timeline.bind_spec(spec)
    lr_lambda = make_lambda(spec, timeline)
    return LambdaLR(optimizer, lr_lambda=[lr_lambda] * len(optimizer.param_groups))
