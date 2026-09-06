"""LR schedule definitions, resolution, runtime timeline and construction.

Phases P0 and P1 of ``docs/guides/LR_SCHEDULER_DESIGN.md``: the one place a
learning-rate schedule is built for the diffusion trainers. Everything it
returns is a ``torch.optim.lr_scheduler.LambdaLR`` whose multiplier is a PURE
function of the step and of the timeline's events -- the invariant
``lr_utils.reassert_config_lr`` and ``BaseTrainer._fast_forward_one_lr_scheduler``
both evaluate lambdas out of order and rely on.

Step axis (D9/§17.1): one unit per ``scheduler.step()``, i.e. per update
boundary, NOT per ``global_step``. ``BaseTrainer`` puts BOTH the warmup and the
total on that axis (``to_scheduler_axis``) before calling ``resolve_spec``,
which converts the config's own step- and length-valued keys itself from
``advance_interval``.

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
``training_control_rpc``; the ReLoRA ``restart`` event is P4.

P4 brings ReLoRA in: ``relora`` is a curve here rather than a scheduler class
of its own, and a merge is a ``restart`` event on the timeline. It is the one
name outside ``LR_SCHEDULER_NAMES`` -- the restart list shapes it, so it is not
something a run can select -- and the one curve that returns its own multiplier
instead of composing with the shared ramp (§17.3).

P3 opens the vocabulary: ``wsd`` and ``rex`` (both aliases of one curve),
``cosine_with_restarts`` with a real-axis cycle length and per-cycle peak
annealing, and D10's floor -- ``m = ramp * (F + (1 - F) * shape)`` for EVERY
curve, which is bit-identical to P0 wherever ``F == 0``. ``polynomial`` joins
that form, so its floor is now ``lr_floor_ratio`` and no longer diffusers'
``1e-7 / lr`` (§17.2, recorded in §18.3). A YAML with no floor key at all reads
as 0.25 for ``plateau_cosine_floor`` and 0.0 everywhere else (§12.2).

R1 adds §19's ``retarget``: one more event kind, carrying a serialized
``ScheduleSpec``, that replaces the curve mid-run and blends from the old one
over ``length`` steps. The fold therefore returns a CHAIN of curves rather than
one spec, and an ``anchor="restart"`` link is evaluated on its own axis whose
origin is the event's step -- which is why every comparison between an event's
``at`` and a spec's ``warmup_steps`` rebases (§19.3's last row).
"""

from __future__ import annotations

import math
from dataclasses import MISSING, asdict, dataclass, fields, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from torch.optim.lr_scheduler import LambdaLR

# SSoT: api/param_defaults.TRAINING_DEFAULTS.
from api.param_defaults import TRAINING_DEFAULTS as _TRAINING_DEFAULTS

__all__ = [
    "BLEND_SHAPE_NAMES",
    "DECAY_SHAPE_NAMES",
    "INTERNAL_SCHEDULER_NAMES",
    "LR_SCHEDULER_NAMES",
    "OverlayState",
    "RETARGET_ANCHORS",
    "SPEC_VERSION",
    "STATE_BASE",
    "STATE_DECAYING",
    "STATE_FLOOR",
    "STATE_NAMES",
    "STATE_RECOVERING",
    "ScheduleSpec",
    "ScheduleTimeline",
    "apply_layer_decay",
    "base_multiplier",
    "blend_length_on_scheduler_axis",
    "build_depth_map",
    "build_lr_scheduler",
    "describe_spec",
    "make_lambda",
    "resolve_spec",
    "sample_curve",
    "to_scheduler_axis",
]

# The canonical vocabulary (D18). routes.py validates against it and
# openapi.yaml's enum mirrors it. The VAE trainer's own list is P7.
LR_SCHEDULER_NAMES = (
    "constant",
    "constant_with_warmup",
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "plateau_cosine_floor",
    "wsd",
    "rex",
)

# Resolvable but NOT selectable: ReLoRA's curve is chosen by
# training_method='relora' and shaped by its restart events, so offering it as
# an lr_scheduler value would let a non-ReLoRA run ask for a curve with no
# restarts in it. Kept out of the API validator and the openapi enum.
INTERNAL_SCHEDULER_NAMES = ("relora",)
_RESOLVABLE_NAMES = LR_SCHEDULER_NAMES + INTERNAL_SCHEDULER_NAMES

# `wsd`'s decay shape k(q), k(0)=1, k(1)=0 (§4.2). `rex` is a shape, not a
# separate curve: no exponent on a cosine reaches it, because cosine enters the
# decay with slope 0 and REX with -1/2 (§8).
DECAY_SHAPE_NAMES = ("cosine", "linear", "rex")

# §19.2's blend weight w(u), w(0)=0, w(1)=1: the decay vocabulary read as
# w = 1 - k(u). D22 adds no shape of its own.
BLEND_SHAPE_NAMES = DECAY_SHAPE_NAMES

# §19.1. `restart` re-anchors the new curve at the event (D23); `continue`
# evaluates it on the global axis.
RETARGET_ANCHORS = ("restart", "continue")

# ScheduleSpec.to_dict's `v` (§19.5). Bump only for a change a reader cannot
# absorb by ignoring unknown keys.
SPEC_VERSION = 1

# diffusers' polynomial exponent, which neither construction site ever
# overrode. Its floor used to be lr_end/lr_init = 1e-7/lr; D10 replaced that
# with lr_floor_ratio, so the two now differ for a run that sets no floor.
_POLYNOMIAL_POWER = 1.0
# §12.2: what a floor-less YAML means for every name but plateau_cosine_floor.
# Not an API default (that is param_defaults' 0.25) -- it is how the schedules
# that predate D10 actually behaved.
_LEGACY_ABSENT_FLOOR = 0.0

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
           "polynomial", "wsd", "relora")

# `add` results meaning "recorded, but the state machine must never fold it":
# the request was refused, for every group, before it became an event.
# The eight `retarget` ones are §19.4, in the order `_refuse_retarget` tests them.
_REFUSED = ("rejected_during_warmup", "rejected_zero_length",
            "rejected_backdated", "rejected_unknown_scheduler",
            "rejected_non_positive_gain", "rejected_negative_length",
            "rejected_floor_out_of_range", "rejected_no_remaining_span",
            "rejected_warmup_exceeds_span", "rejected_unknown_group")


@dataclass(frozen=True)
class ScheduleSpec:
    """What config resolves to. Immutable for the lifetime of the scheduler.

    Axes (D8/§17.2): ``warmup_steps`` and ``decay_length`` are REAL scheduler
    steps; ``total_steps`` and a ``decay_start_axis="nominal"`` start are read
    through the timeline's clock, which P1 warps when ``total_steps`` changes.
    That real/nominal CLOCK is independent of the UNIT: every count in here is
    scheduler advances, converted from the config's global steps by
    ``resolve_spec`` (§17.1).
    ``decay_start_step=None`` means "no configured decay" (the external ``D=0``
    P3 exposes as manual WSD), which is distinct from a start of 0.
    """

    name: str
    curve: str
    warmup_steps: int
    total_steps: int
    floor_ratio: float = 0.0
    # True when the YAML carried no lr_floor_ratio and §12.2's rule supplied
    # one. The trainer warns on the one name where that changes the curve.
    floor_defaulted: bool = False
    decay_start_step: Optional[int] = None
    # §17.2: an alias's derived start resolves from the NOMINAL total, so a
    # resume with a new total_steps does not move it. Equal to
    # decay_start_step until the timeline warps.
    decay_start_ratio: Optional[float] = None
    decay_start_axis: str = "nominal"
    decay_length: Optional[int] = None
    decay_end_kind: str = "nominal_total"
    decay_shape: str = "cosine"
    # §12.1: lr_decay_steps / lr_decay_shape also give a runtime `start_decay`
    # its length and shape, under EVERY name. The base curve reads
    # decay_length/decay_shape instead, and the two aliases fix those to their
    # own definition (cosine to the nominal end / rex from warmup).
    command_decay_length: Optional[int] = None
    command_decay_shape: str = "cosine"
    # cosine_with_restarts (§9.1). 0 = one cycle over the whole nominal run,
    # which is the single cosine every existing YAML already got.
    cycle_steps: int = 0
    cycle_peak_decay: float = 1.0
    # `relora` only: the ramp length after a merge. The run's first warmup is
    # `warmup_steps`; every later segment uses this one (§4.2).
    relora_restart_warmup_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """The JSON form a ``retarget`` event carries (§19.5).

        Invariant 15: no absolute step travels in the payload. An alias's
        ``decay_start_step`` is DROPPED, because it was derived from the seed
        total and names a position on an axis the reader no longer has --
        ``decay_start_ratio`` is its only source. ``wsd``'s configured ``D``
        (no ratio) stays: D8 makes it a real-axis quantity, read from the
        retarget's own origin.
        """
        payload: Dict[str, Any] = {"v": SPEC_VERSION}
        payload.update(asdict(self))
        if self.decay_start_ratio is not None:
            payload.pop("decay_start_step")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleSpec":
        """Rebuild a spec, IGNORING keys this build does not know (§19.5).

        A state file written by a newer build has to keep resuming, so an
        unknown key is dropped rather than refused; a missing required one is
        an error, since guessing a name or a total is not recoverable.
        """
        known = {f.name: f for f in fields(cls)}
        kwargs = {name: _coerce_field(known[name].type, value)
                  for name, value in dict(data or {}).items()
                  if name in known}
        missing = [name for name, f in known.items()
                   if name not in kwargs
                   and f.default is MISSING and f.default_factory is MISSING]
        if missing:
            raise ValueError(
                "ScheduleSpec.from_dict is missing required key(s): "
                + ", ".join(sorted(missing)))
        return cls(**kwargs)


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


@dataclass(frozen=True)
class _Blend:
    """How one curve was entered from the one before it (§19.2).

    ``prev`` is the curve the retarget replaced, still evolving: ``m_old(s)``
    is what it WOULD have produced, so a decay it was in keeps decaying for the
    length of the blend. Past ``at + length`` the weight is 1 and ``prev`` is
    never evaluated again, which is what bounds the chain walk to the number of
    blends in progress rather than to the number of events.
    """

    prev: "_Curve"
    at: int
    length: int
    shape: str


@dataclass(frozen=True)
class _Curve:
    """One link of the fold: a spec, its overlay, and the axis it reads on.

    ``origin`` is 0 for the config curve and ``S`` for an ``anchor="restart"``
    retarget; ``scale`` is ``m_at_S * gain`` there and ``gain`` on the global
    axis. ``view`` maps the timeline's clock and totals onto that axis. Every
    field is rebuilt by each fold -- nothing here is cached on the timeline,
    because the lambda must stay a pure function of ``(step, events)``.
    """

    spec: ScheduleSpec
    state: OverlayState
    view: Any
    origin: int = 0
    scale: float = 1.0
    link: Optional[_Blend] = None


class _LocalAxis:
    """The timeline seen from step ``origin`` (D22).

    Duck-types the four reads ``base_multiplier`` makes. The span is derived
    here, at evaluation time, from the run's own totals -- never stored on the
    spec -- so an extension moves a restart-anchored curve's end with it, and
    composing with the warp keeps the curve continuous at the anchor.
    """

    def __init__(self, timeline: "ScheduleTimeline", origin: int):
        self._timeline = timeline
        self._origin = int(origin)
        self._clock_origin = timeline.clock(self._origin)

    def nominal_total(self, default: int) -> int:
        span = self._timeline.nominal_total(default) - self._clock_origin
        return max(1, int(round(span)))

    def current_total(self, default: int) -> int:
        return max(1, self._timeline.current_total(default) - self._origin)

    def clock(self, step: float) -> float:
        return self._timeline.clock(self._origin + step) - self._clock_origin

    def restarts(self, upto: Optional[int] = None) -> List[int]:
        limit = None if upto is None else self._origin + int(upto)
        return [at - self._origin
                for at in self._timeline.restarts(upto=limit)
                if at >= self._origin]


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

        if kind == "restart":
            # Seam (d): a ReLoRA merge. The BASE curve reads these; the overlay
            # state machine below does not, so a restart can never disturb a
            # decay or a cancellation. Deduplicated by position so that
            # re-registering a legacy checkpoint's merges is idempotent.
            if any(e.get("kind") == "restart" and int(e.get("at", 0)) == at
                   for e in self.events):
                return "ignored_duplicate_restart"
            self._append({"kind": kind, "at": at, "request_id": request_id,
                          "result": "applied"})
            return "applied"

        if kind == "retarget":
            return self._add_retarget(at, spec, request_id, payload)

        if kind not in ("decay", "cancel"):
            raise ValueError("Unknown timeline event kind: " + repr(kind))

        resolved = spec or self.spec
        if resolved is None:
            raise ValueError(
                "decay/cancel need a spec: build_lr_scheduler binds the "
                "representative one, or pass spec= explicitly")

        # What gets baked in comes from the spec in force AT `at`, which the
        # fold reports; `resolved` only seeds it.
        curve = self._fold_curve(resolved, at)
        active = curve.spec

        event: Dict[str, Any] = {"kind": kind, "at": at,
                                 "request_id": request_id}
        issued = payload.get("issued")
        if issued is not None:
            event["issued"] = int(issued)
        length = payload.get("length")
        if kind == "decay":
            event["length"] = None if length is None else int(length)
            event["shape"] = str(payload.get("shape") or active.decay_shape)
            _decay_shape(event["shape"])  # refuse a P3 shape at the seam
        else:
            # §5.4: the recovery length is baked in, so a later config edit
            # cannot reshape a cancel that already happened.
            event["length"] = int(active.warmup_steps if length is None
                                  else length)

        if kind == "decay":
            _, result = self._apply_decay(curve, event)
        else:
            _, result = self._apply_cancel(curve, event)
        if result in _REFUSED:
            # Kept so re-delivering the request_id answers the same thing.
            # `noop` is a kind the state machine does not know, so a refused
            # command can never take effect for any group.
            event["refused_kind"] = event["kind"]
            event["kind"] = "noop"
        event["result"] = result
        self._append(event)
        return result

    def _add_retarget(self, at: int, seed: Optional[ScheduleSpec],
                      request_id: Optional[str],
                      payload: Mapping[str, Any]) -> str:
        """§19.1's event: replace the curve from ``at`` on, blending from the old.

        ``new_spec=`` is the retarget's own schedule (a ``ScheduleSpec`` or its
        ``to_dict()``); the ``spec=`` argument of ``add`` keeps its old meaning,
        the representative curve the result code is scored against.
        ``known_groups=`` is the component list §19.4's rule 7 validates
        against; without it the names cannot be checked and are accepted.

        An unknown ``anchor`` or blend ``shape`` RAISES rather than refusing --
        the same seam-time check `decay` makes on its shape, since a
        misspelled vocabulary word is a caller bug, not a refused request.
        """
        raw = payload.get("new_spec")
        if raw is None:
            raise ValueError(
                "retarget needs new_spec=<ScheduleSpec or its to_dict()>")
        new_spec = (raw if isinstance(raw, ScheduleSpec)
                    else ScheduleSpec.from_dict(raw))

        anchor = str(payload.get("anchor") or "restart").strip().lower()
        if anchor not in RETARGET_ANCHORS:
            raise ValueError(
                f"Unknown retarget anchor '{anchor}'. Supported: "
                f"{', '.join(RETARGET_ANCHORS)}")
        shape = str(payload.get("shape") or "linear").strip().lower()
        _blend_weight(shape)

        gain = payload.get("gain")
        gain = 1.0 if gain is None else float(gain)
        length = payload.get("length")
        length = 0 if length is None else int(length)
        issued = payload.get("issued")
        issued = at if issued is None else int(issued)
        groups = payload.get("groups")
        groups = None if groups is None else [str(g) for g in groups]

        event: Dict[str, Any] = {
            "kind": "retarget", "at": at, "issued": issued,
            "spec": new_spec.to_dict(), "anchor": anchor, "gain": gain,
            "length": length, "shape": shape, "groups": groups,
            "request_id": request_id,
        }
        result = self._refuse_retarget(
            at, issued, new_spec, anchor, gain, length, groups,
            payload.get("known_groups"), seed) or "applied"
        if result in _REFUSED:
            event["refused_kind"] = event["kind"]
            event["kind"] = "noop"
        event["result"] = result
        self._append(event)
        return result

    def _refuse_retarget(self, at: int, issued: int, new_spec: ScheduleSpec,
                         anchor: str, gain: float, length: int,
                         groups: Optional[Sequence[str]],
                         known_groups: Optional[Sequence[str]],
                         seed: Optional[ScheduleSpec]) -> Optional[str]:
        """§19.4's eight rules, in the order they are tested. None = accept."""
        if at < issued:                                             # 1 (D25)
            return "rejected_backdated"
        if new_spec.name not in LR_SCHEDULER_NAMES:                 # 2
            return "rejected_unknown_scheduler"
        # Never trust a serialized (name, curve) pair: R3 takes this dict from
        # an endpoint, and name="cosine" with curve="relora" would install the
        # segmented ReLoRA curve on a run that has no merges. Resolved from the
        # name, not compared against a second table that could drift.
        if new_spec.curve != resolve_spec(
                {}, warmup_steps=0, total_steps=1, name=new_spec.name).curve:
            return "rejected_unknown_scheduler"
        if gain <= 0.0:                                             # 8
            return "rejected_non_positive_gain"
        if length < 0:                                              # 4
            return "rejected_negative_length"
        if not 0.0 <= new_spec.floor_ratio <= 1.0:                  # 6
            return "rejected_floor_out_of_range"
        total = self.current_total(
            (seed or self.spec or new_spec).total_steps)
        span = total - at if anchor == "restart" else total
        if anchor == "restart" and span <= 0:                       # 3
            return "rejected_no_remaining_span"
        # >=, not >: §17.2's construction contract is 0 <= W < T_sched, and a
        # warmup that ends exactly at the run's end leaves no step at peak.
        if new_spec.warmup_steps >= span:                           # 5
            return "rejected_warmup_exceeds_span"
        if groups and known_groups is not None:                     # 7
            allowed = {str(g) for g in known_groups}
            if any(g not in allowed for g in groups):
                return "rejected_unknown_group"
        return None

    def load(self, events: Optional[Sequence[Mapping[str, Any]]],
             upto_step: Optional[int] = None) -> None:
        """Seam (b): install a saved event list, BEFORE the fast-forward.

        ``upto_step`` drops later commands, the same semantics as
        ``_cleanup_future_metrics``: rewinding to an earlier checkpoint un-does
        what was ordered after it. Cut by ``issued``, like ``dump`` (§19.5): a
        reservation for a future step was ordered BEFORE the checkpoint, and
        cutting it by ``at`` would delete it on the way back in.
        """
        self.events = []
        self._next_seq = 0
        for index, event in enumerate(events or []):
            record = dict(event)
            if upto_step is not None and _issued(record) > int(upto_step):
                continue
            record.setdefault("at", 0)
            record.setdefault("seq", index)
            self.events.append(record)
            self._next_seq = max(self._next_seq, int(record["seq"]) + 1)
        self.events.sort(key=_order)

    def dump(self, upto_step: int) -> List[Dict[str, Any]]:
        """JSON-ready events with ``issued <= upto_step``, in application order.

        Invariant 6 as §19.5 revised it: the cut is by ``issued``, so a future
        reservation (``at > upto_step``) survives every save between its order
        and its effect. An event with no ``issued`` reads as ``issued = at``,
        which is what every event written before R1 was.
        """
        return [dict(e) for e in self._sorted()
                if _issued(e) <= int(upto_step)]

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

    def restarts(self, upto: Optional[int] = None) -> List[int]:
        """ReLoRA restart positions, ascending, none of them after ``upto``.

        §17.3: reading a restart the run has not reached yet is what let a
        resume shorten a cosine segment retroactively.
        """
        return [int(e["at"]) for e in self._sorted()
                if e.get("kind") == "restart"
                and (upto is None or int(e["at"]) <= int(upto))]

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
        return self._fold(spec, int(step))[1]

    def active_spec(self, spec: ScheduleSpec, step: int) -> ScheduleSpec:
        """The spec in force at ``step``: ``spec`` itself, or the last
        ``retarget``'s. Read it wherever a caller reports or bakes a spec field
        beside a timeline state, so that replacement reaches it. Under
        ``anchor="restart"`` its step-valued fields are lengths from the
        retarget's step, not positions on the run's axis."""
        return self._fold(spec, int(step))[0]

    def multiplier(self, spec: ScheduleSpec, step: int) -> float:
        """The LR multiplier: §4's base curve with the overlay on top, blended
        with whatever a ``retarget`` replaced (§19.2)."""
        step = int(step)
        return self._chain_value(self._fold_curve(spec, step), step)

    # -- internals -------------------------------------------------------

    def _append(self, event: Dict[str, Any]) -> None:
        event["seq"] = self._next_seq
        self._next_seq += 1
        self.events.append(event)
        self.events.sort(key=_order)

    def _sorted(self) -> List[Dict[str, Any]]:
        return sorted(self.events, key=_order)

    def _fold(self, spec: ScheduleSpec,
              step: int) -> Tuple[ScheduleSpec, OverlayState]:
        """The spec in force at ``step`` and the overlay on it."""
        curve = self._fold_curve(spec, int(step))
        return curve.spec, curve.state

    def _fold_curve(self, spec: ScheduleSpec, step: int) -> _Curve:
        """Apply every event at or before ``step``, in arrival order.

        The spec is folded rather than fixed because a ``retarget`` replaces it
        mid-run; every event is then applied over the curve that was active
        when it arrived, not over the one config resolved to. A retarget also
        keeps a reference to what it replaced, for the blend of §19.2.

        This is also what an event arriving AT ``step`` sees: earlier same-step
        events carry a smaller ``seq``, so they are already folded in, and the
        trailing advance is the same one the replay does before applying it.
        """
        curve = _Curve(spec=spec, state=OverlayState(), view=self)
        for event in self._sorted():
            at = int(event.get("at", 0))
            if at > step:
                break
            kind = event.get("kind")
            if kind == "decay":
                curve = self._advanced(curve, at)
                curve = replace(curve,
                                state=self._apply_decay(curve, event)[0])
            elif kind == "cancel":
                curve = self._advanced(curve, at)
                curve = replace(curve,
                                state=self._apply_cancel(curve, event)[0])
            elif kind == "retarget":
                curve = self._retarget(self._advanced(curve, at), event)
        return self._advanced(curve, step)

    def _retarget(self, curve: _Curve, event: Mapping[str, Any]) -> _Curve:
        """Enter the event's spec, keeping ``curve`` as the blend's ``m_old``.

        D26: whatever overlay was running is ABSORBED -- the new curve starts
        in BASE, and the value it starts from is the realized multiplier, decay
        or recovery included. ``decay_disarmed`` does not carry over: the new
        spec's own configured decay is armed.

        R2 owns ``groups``: the event stores the selector, but a retarget here
        replaces the curve of EVERY spec. Filtering needs a group identifier on
        the spec (D24) -- without one, matching by name would apply the
        retarget to whichever group happens to share a schedule.
        """
        at = int(event.get("at", 0))
        spec = ScheduleSpec.from_dict(event["spec"])
        anchor = str(event.get("anchor") or "restart")
        gain = event.get("gain")
        gain = 1.0 if gain is None else float(gain)
        link = _Blend(prev=curve, at=at, length=int(event.get("length") or 0),
                      shape=str(event.get("shape") or "linear"))
        if anchor == "continue":
            return _Curve(spec=spec, state=OverlayState(), view=self,
                          scale=gain, link=link)
        # anchor=restart: m_new(s) = m_at_S * gain * g(s - S). The registry's
        # curves all peak at 1 after their warmup, so g is the curve itself on
        # the local axis -- nothing is divided by g(0), which is 0 whenever the
        # new spec warms up (§19.2).
        return _Curve(spec=spec, state=OverlayState(),
                      view=_LocalAxis(self, at), origin=at,
                      scale=self._chain_value(curve, at) * gain, link=link)

    def _apply_decay(self, curve: _Curve,
                     event: Mapping[str, Any]) -> Tuple[OverlayState, str]:
        state = curve.state
        at = int(event.get("at", 0))
        # §19.3's last row: `at` is an absolute scheduler step and warmup_steps
        # is a length from this curve's origin. Unrebased, a command inside the
        # new warmup but past step W_new would be ACCEPTED, starting a decay on
        # a rising ramp -- the design states this failure with the sign the
        # other way round.
        if at - curve.origin < curve.spec.warmup_steps:
            # Also what keeps a ramp from being read as "a decay that raises
            # the LR", and q's denominator away from zero.
            return state, "rejected_during_warmup"
        if state.code in (STATE_DECAYING, STATE_FLOOR):
            return state, "ignored_already_decaying"
        length = event.get("length")
        if length is None:
            if (curve.view.nominal_total(curve.spec.total_steps)
                    - curve.view.clock(at - curve.origin)) <= 0:
                return state, "rejected_zero_length"
        elif int(length) <= 0:
            return state, "rejected_zero_length"
        return OverlayState(
            code=STATE_DECAYING, at=at,
            start_multiplier=self._own_value(curve, at),
            length=None if length is None else int(length),
            shape=str(event.get("shape") or curve.spec.decay_shape),
            decay_disarmed=state.decay_disarmed,
        ), "applied"

    def _apply_cancel(self, curve: _Curve,
                      event: Mapping[str, Any]) -> Tuple[OverlayState, str]:
        state = curve.state
        at = int(event.get("at", 0))
        length = event.get("length")
        # A LENGTH, so no rebase: R = W is the same number on either axis.
        recovery = int(curve.spec.warmup_steps if length is None else length)
        if state.code == STATE_RECOVERING:
            return state, "ignored_already_recovering"
        if state.code in (STATE_DECAYING, STATE_FLOOR):
            return OverlayState(
                code=STATE_RECOVERING, at=at,
                start_multiplier=self._own_value(curve, at),
                length=recovery, shape=state.shape, decay_disarmed=True,
            ), "applied"
        if _has_config_decay(curve.spec) and not state.decay_disarmed:
            # §17.3: a cancel voids the config-declared WSD decay too. Already
            # past its start, that is a recovery; before it, a disarm.
            if self._config_decay_started(curve, at):
                return OverlayState(
                    code=STATE_RECOVERING, at=at,
                    start_multiplier=self._own_base(curve, at, False),
                    length=recovery, decay_disarmed=True,
                ), "applied"
            return replace(state, decay_disarmed=True), "disarmed_scheduled_decay"
        return state, "ignored_no_active_decay"

    def _advanced(self, curve: _Curve, step: int) -> _Curve:
        """The transitions time makes on its own (§5.3's last two rows).

        Idempotent, so the chain walk can re-apply it to a curve the fold has
        already advanced.
        """
        state = curve.state
        if state.code == STATE_DECAYING and self._q(curve, step) >= 1.0:
            return replace(curve, state=replace(state, code=STATE_FLOOR))
        if (state.code == STATE_RECOVERING
                and step >= state.at + int(state.length or 0)):
            return replace(curve, state=OverlayState(
                decay_disarmed=state.decay_disarmed))
        return curve

    def _q(self, curve: _Curve, step: int) -> float:
        state = curve.state
        if state.length is None:
            view, origin = curve.view, curve.origin
            span = (view.nominal_total(curve.spec.total_steps)
                    - view.clock(state.at - origin))
            elapsed = view.clock(step - origin) - view.clock(state.at - origin)
        else:
            span = float(state.length)
            elapsed = float(step - state.at)
        if span <= 0:
            return 1.0
        return min(1.0, max(0.0, elapsed / span))

    def _config_decay_started(self, curve: _Curve, step: int) -> bool:
        spec, view = curve.spec, curve.view
        start = _config_decay_start(spec, view.nominal_total(spec.total_steps))
        if start is None:
            return False
        local = step - curve.origin
        position = (view.clock(local) if spec.decay_start_axis == "nominal"
                    else float(local))
        return position >= float(start)

    def _chain_value(self, curve: _Curve, step: int) -> float:
        """The realized multiplier: this curve, blended with what it replaced.

        Walks back one link per blend still in progress; a finished one returns
        before recursing, so the depth is the number of overlapping blends and
        not the number of events (§19.2).
        """
        curve = self._advanced(curve, step)
        value = curve.scale * self._own_value(curve, step)
        link = curve.link
        if link is None:
            return value
        u = (1.0 if link.length <= 0
             else min(1.0, max(0.0, (step - link.at) / float(link.length))))
        w = _blend_weight(link.shape)(u)
        if w >= 1.0:
            return value
        # A convex combination with w in [0, 1]: D30's bound
        # m(s) <= max(m_old(s), m_new(s)) is the clamp above, not the shape.
        return (1.0 - w) * self._chain_value(link.prev, step) + w * value

    def _own_value(self, curve: _Curve, step: int) -> float:
        """This curve's own multiplier, UNSCALED: the overlay on its base.

        Unscaled because ``scale`` multiplies the whole curve, floor included,
        so a decay started after a restart-anchored retarget runs between this
        curve's ``F`` and its own realized value rather than between two
        already-scaled numbers.
        """
        spec, state = curve.spec, curve.state
        if state.code == STATE_DECAYING:
            floor = spec.floor_ratio
            shape = _decay_shape(state.shape)
            return floor + (state.start_multiplier - floor) * shape(
                self._q(curve, step))
        if state.code == STATE_FLOOR:
            return spec.floor_ratio
        if state.code == STATE_RECOVERING:
            base = self._own_base(curve, step, state.decay_disarmed)
            recovery = int(state.length or 0)
            if recovery <= 0:
                return base
            ratio = min(1.0, max(0.0, (step - state.at) / float(recovery)))
            return state.start_multiplier + (base - state.start_multiplier) * ratio
        return self._own_base(curve, step, state.decay_disarmed)

    def _own_base(self, curve: _Curve, step: int, disarmed: bool) -> float:
        return base_multiplier(curve.spec, curve.view, step - curve.origin,
                               decay_disarmed=disarmed)


def _order(event: Mapping[str, Any]) -> Tuple[int, int]:
    return (int(event.get("at", 0)), int(event.get("seq", 0)))


def _issued(event: Mapping[str, Any]) -> int:
    """When the event was ACCEPTED (§19.5). Absent means it was accepted at
    its own ``at``, which every event written before R1 was."""
    issued = event.get("issued")
    at = int(event.get("at", 0))
    return at if issued is None else int(issued)


def _coerce_field(annotation: Any, value: Any) -> Any:
    """Read one JSON value back into a ScheduleSpec field's type.

    ``from __future__ import annotations`` leaves the annotation a string, and
    JSON does not distinguish 10 from 10.0, so a round trip through a state
    file would otherwise hand the curve a float where it counts steps.
    """
    if value is None:
        return None
    text = str(annotation)
    if "bool" in text:
        return bool(value)
    if "float" in text:
        return float(value)
    if "int" in text:
        return int(value)
    if "str" in text:
        return str(value)
    return value


def _blend_weight(name: str) -> Callable[[float], float]:
    """``w(u)`` of §19.2: w(0)=0, w(1)=1, non-decreasing on [0, 1]."""
    if name == "linear":
        return lambda u: u
    if name == "cosine":
        return lambda u: 0.5 * (1.0 - math.cos(math.pi * u))
    if name == "rex":
        return lambda u: 1.0 - (1.0 - u) / (1.0 - u / 2.0)
    raise ValueError(
        f"Unknown blend shape '{name}'. Supported: {', '.join(BLEND_SHAPE_NAMES)}")


def blend_length_on_scheduler_axis(steps: Optional[int],
                                   interval: int) -> Optional[int]:
    """A retarget's ``L`` from global steps onto the scheduler axis.

    0 means "switch instantly" here, so a positive request that floors to 0
    returns -1 -- which ``add`` refuses (§19.4 rule 4) rather than silently
    turning a requested blend into a hard switch. §18.5's sentinel, refusing
    where the config path clamps, because there is no reason to guess.
    """
    if steps is None:
        return None
    requested = int(steps)
    if requested <= 0:
        return requested
    return to_scheduler_axis(requested, interval) or -1


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


def _resolve_floor(config: Optional[Mapping[str, Any]],
                   name: str) -> Tuple[float, bool]:
    """``F`` and whether §12.2's compatibility rule supplied it.

    The rule keys off the YAML, never off the Pydantic default: a run written
    before D10 has no key at all, and only ``plateau_cosine_floor`` had a floor
    then.
    """
    raw = (config or {}).get("lr_floor_ratio")
    if raw is None:
        if name == "plateau_cosine_floor":
            return float(_TRAINING_DEFAULTS["lr_floor_ratio"]), True
        return _LEGACY_ABSENT_FLOOR, True
    floor = float(raw)
    if not 0.0 <= floor <= 1.0:
        raise ValueError(
            f"lr_floor_ratio must be between 0 and 1 (got {floor}): it is a "
            f"fraction of the base learning rate.")
    return floor, False


def _positive_or_none(config: Optional[Mapping[str, Any]], key: str) -> Optional[int]:
    """A step count where the API spells "unset" as 0 (§12.1)."""
    value = int(_lookup(config, key))
    if value < 0:
        raise ValueError(f"{key} must be >= 0 (got {value})")
    return value or None


def to_scheduler_axis(steps: int, interval: int) -> int:
    """A global-step count -> scheduler advances (D9/§17.1).

    Floor, because a trailing partial accumulation window is never flushed.
    Every step count ``resolve_spec`` receives OR reads out of the config goes
    through here: converting ``T`` but not ``W`` would make the warmup occupy
    ``interval`` times the fraction of the schedule the configured number asks
    for, and the same is true of a decay start or a cycle length.
    """
    return max(0, int(steps)) // max(1, int(interval))


def _length_on_scheduler_axis(steps: Optional[int],
                              interval: int) -> Optional[int]:
    """A configured LENGTH onto the scheduler axis, never floored to 0.

    0 is the "unset" sentinel for both length keys -- a decay that runs to the
    nominal end, one cycle over the whole run -- and it is also the divisor in
    those two branches, so a length shorter than one accumulation window
    becomes the shortest representable one rather than a different curve.
    """
    if steps is None:
        return None
    return max(1, to_scheduler_axis(steps, interval))


def resolve_spec(
    config: Optional[Mapping[str, Any]],
    *,
    warmup_steps: int,
    total_steps: int,
    name: str,
    advance_interval: int = 1,
    restart_warmup_steps: Optional[int] = None,
) -> ScheduleSpec:
    """Resolve the run's config into an immutable spec.

    ``warmup_steps`` and ``total_steps`` must both already be on the scheduler
    axis (``to_scheduler_axis``). ``advance_interval`` is the run's
    ``gradient_accumulation_steps``: the config's step- and length-valued keys
    are entered on the global_step axis, like those two, and are converted
    HERE, at the one seam. 1 = nothing to convert.

    Two independent axes meet in this function: the UNIT of a count
    (global_step vs scheduler advance, §17.1) and the CLOCK a position is read
    on (real vs nominal, §17.2). Converting the unit never changes the clock.

    ``restart_warmup_steps`` is ReLoRA's post-merge ramp. It arrives as an
    argument rather than out of ``config`` because it lives in the YAML's
    ``network.relora`` section, which the trainer's ``config`` (the ``train``
    section) does not carry.
    """
    key = str(name).strip().lower()
    if key not in _RESOLVABLE_NAMES:
        raise ValueError(
            f"Unknown lr_scheduler '{name}'. Supported: "
            f"{', '.join(LR_SCHEDULER_NAMES)}"
        )

    W = max(0, int(warmup_steps or 0))
    T = max(1, int(total_steps))
    A = max(1, int(advance_interval))
    floor, floor_defaulted = _resolve_floor(config, key)
    # Available to a runtime `start_decay` under every name; the base curve
    # reads decay_length/decay_shape, which only `wsd` takes from these. The
    # sentinel is resolved on the CONFIGURED number, so no accumulation width
    # can turn a length a user asked for into "unset".
    command_length = _length_on_scheduler_axis(
        _positive_or_none(config, "lr_decay_steps"), A)
    command_shape = str(_lookup(config, "lr_decay_shape")).strip().lower()
    if command_shape not in DECAY_SHAPE_NAMES:
        raise ValueError(
            f"Unknown lr_decay_shape '{command_shape}'. Supported: "
            f"{', '.join(DECAY_SHAPE_NAMES)}")
    common = dict(warmup_steps=W, total_steps=T, floor_ratio=floor,
                  floor_defaulted=floor_defaulted,
                  command_decay_length=command_length,
                  command_decay_shape=command_shape)

    if key == "plateau_cosine_floor":
        # D11: an alias for wsd -- decay start on the NOMINAL axis, length "to
        # the nominal end", never stored as an explicit real length (§17.2).
        # A ratio has no unit, and T is already T_sched, so nothing converts.
        ratio = float(_lookup(config, "lr_decay_start_ratio"))
        start = max(W, min(round(ratio * T), T))
        return ScheduleSpec(
            name=key, curve="wsd", decay_start_step=start,
            decay_start_ratio=ratio, decay_start_axis="nominal",
            decay_length=None, decay_end_kind="nominal_total",
            decay_shape="cosine", **common)

    if key == "rex":
        # §8/D14: WSD with no plateau. D = W is a REAL step, already on the
        # scheduler axis, so W = 0 starts the decay at 0 -- distinct from
        # `wsd`'s external D = 0, which means "manual" (§17.2).
        return ScheduleSpec(
            name=key, curve="wsd", decay_start_step=W, decay_start_axis="real",
            decay_length=None, decay_end_kind="nominal_total",
            decay_shape="rex", **common)

    if key == "wsd":
        # A REAL-axis POSITION (the timeline's warp never moves it, §17.2) that
        # is nonetheless configured in global steps: only the unit converts. 0
        # already means "manual" here, so a start inside the first accumulation
        # window lands at scheduler step 0 and is not read as unset.
        start = _positive_or_none(config, "lr_decay_start_step")
        return ScheduleSpec(
            name=key, curve="wsd",
            decay_start_step=(None if start is None
                              else to_scheduler_axis(start, A)),
            decay_start_axis="real", decay_length=command_length,
            decay_end_kind="length" if command_length else "nominal_total",
            decay_shape=command_shape, **common)

    if key == "relora":
        # §4.2's relora row. The old scheduler's hardcoded min_lr_ratio=0.0 is
        # now F, so a YAML written before D10 (no floor key) reads as 0.0.
        # W_r is a LENGTH in global steps like W and floors the same way: one
        # shorter than an accumulation window is 0 LR updates, not 1.
        restart_warmup = (_TRAINING_DEFAULTS["restart_warmup_steps"]
                          if restart_warmup_steps is None
                          else restart_warmup_steps)
        return ScheduleSpec(
            name=key, curve=key,
            relora_restart_warmup_steps=to_scheduler_axis(restart_warmup, A),
            **common)

    if key == "cosine_with_restarts":
        peak = float(_lookup(config, "lr_cycle_peak_decay"))
        if not 0.0 < peak <= 1.0:
            raise ValueError(
                f"lr_cycle_peak_decay must be in (0, 1] (got {peak}): it is the "
                f"factor each restart's peak is multiplied by.")
        return ScheduleSpec(
            name=key, curve=key,
            cycle_steps=int(_length_on_scheduler_axis(
                _positive_or_none(config, "lr_cycle_steps"), A) or 0),
            cycle_peak_decay=peak, **common)

    curve = "constant" if key == "constant_with_warmup" else key
    return ScheduleSpec(name=key, curve=curve, **common)


def _decay_shape(name: str) -> Callable[[float], float]:
    if name == "cosine":
        return lambda q: 0.5 * (1.0 + math.cos(math.pi * q))
    if name == "linear":
        return lambda q: 1.0 - q
    if name == "rex":
        return lambda q: (1.0 - q) / (1.0 - q / 2.0)
    raise ValueError(
        f"Unknown decay shape '{name}'. Supported: {', '.join(DECAY_SHAPE_NAMES)}")


def base_multiplier(spec: ScheduleSpec, timeline: ScheduleTimeline, step: int,
                    *, decay_disarmed: bool = False) -> float:
    """``m_base(s)``: §4's curve, with no runtime overlay on it.

    ``m = ramp(s) * (F + (1 - F) * shape(s))`` (D10). With ``F == 0`` that
    composition is bit-identical to the bare shape -- ``1 - 0 == 1``,
    ``1 * x == x`` and ``0 + x == x`` are all exact -- which is what keeps the
    ported curves equal to what diffusers produced.

    Reads ``timeline`` only through ``nominal_total``/``clock``/``restarts``, so
    it stays a pure function of ``(step, timeline.events)``.
    """
    if spec.curve == "relora":
        # Returns its segment's multiplier directly, floor included: a
        # re-warmup composed with the shared ramp would be squared (§17.3).
        return _relora_multiplier(spec, timeline, step)

    W = spec.warmup_steps
    if W > 0 and step < W:
        # The ramp is OUTSIDE the floor: warmup climbs from 0, as the plateau
        # schedule it replaces did.
        return step / float(W)

    F = spec.floor_ratio
    return F + (1.0 - F) * _shape(spec, timeline, step, decay_disarmed)


def _relora_multiplier(spec: ScheduleSpec, timeline: ScheduleTimeline,
                       step: int) -> float:
    """ReLoRA's segmented curve (§4.2's ``relora`` row).

    A merge starts a segment: a linear re-warmup of
    ``relora_restart_warmup_steps``, then a cosine to the run's end.

    §17.3, and the reason this is not a transcription of the class it replaced:
    only restarts at or before ``step`` are read, and the terminus is the run's
    TOTAL, never the next restart. The old ``get_lr`` took its terminus from the
    whole registered list, so a resume that re-registered every past merge at
    once retroactively shortened the cosine the run had already trained through.
    """
    start, warmup, restarted = 0, spec.warmup_steps, False
    for at in timeline.restarts(upto=step):
        start, warmup, restarted = at, spec.relora_restart_warmup_steps, True

    F = spec.floor_ratio
    elapsed = step - start
    if warmup > 0 and elapsed < warmup:
        ramp = elapsed / float(warmup)
        # §17.3: the run's FIRST warmup climbs from 0 (D10 puts the shared ramp
        # outside the floor); a re-warmup climbs from the floor it fell to.
        return (F + (1.0 - F) * ramp) if restarted else ramp

    T = timeline.nominal_total(spec.total_steps)
    decay_from = timeline.clock(start + warmup)
    span = T - decay_from
    if span <= 0:
        # The replaced scheduler's `decay_steps <= 0` branch: hold the peak.
        return 1.0
    q = min(1.0, max(0.0, (timeline.clock(step) - decay_from) / span))
    return F + (1.0 - F) * 0.5 * (1.0 + math.cos(math.pi * q))


def _shape(spec: ScheduleSpec, timeline: ScheduleTimeline, step: int,
           decay_disarmed: bool) -> float:
    """``shape(s)`` of §4.2: in [0, 1], with the floor applied by the caller."""
    W = spec.warmup_steps
    curve = spec.curve
    if curve == "constant":
        # constant_with_warmup's shape. diffusers' bare `constant` ignored
        # num_warmup_steps entirely (optimization.py:323-324); §4.2 makes the
        # two one curve.
        return 1.0

    T = timeline.nominal_total(spec.total_steps)
    warmup_end = timeline.clock(W)

    if curve == "linear":
        return min(1.0, max(0.0, float(T - timeline.clock(step))
                            / float(max(1, T - warmup_end))))

    if curve in ("cosine", "polynomial") or (
            curve == "cosine_with_restarts" and spec.cycle_steps <= 0):
        progress = max(0.0, float(timeline.clock(step) - warmup_end)
                       / float(max(1, T - warmup_end)))
        # Clamped, where diffusers lets the cosine rise again past T.
        if progress > 1.0:
            progress = 1.0
        if curve == "polynomial":
            return max(0.0, (1.0 - progress) ** _POLYNOMIAL_POWER)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    if curve == "cosine_with_restarts":
        # C > 0 is an ABSOLUTE cycle length: real-axis, so an extension adds
        # cycles instead of stretching them, and nothing here reads T.
        C = float(spec.cycle_steps)
        elapsed = max(0.0, float(step - W))
        index = int(elapsed // C)
        q = (elapsed - index * C) / C
        peak = spec.cycle_peak_decay ** index
        return max(0.0, peak * 0.5 * (1.0 + math.cos(math.pi * q)))

    if curve == "wsd":
        D = _config_decay_start(spec, T)
        if D is None or decay_disarmed:
            # Manual WSD not started, or a cancel voided the config decay
            # (§17.3). Holds 1 past T as well.
            return 1.0
        k = _decay_shape(spec.decay_shape)
        if spec.decay_length is not None:
            # An explicit length is REAL steps (D8): an extension runs the
            # extra steps at the floor rather than stretching the decay.
            if step < D:
                return 1.0
            return k(min(1.0, (step - D) / float(spec.decay_length)))
        start = (float(D) if spec.decay_start_axis == "nominal"
                 else timeline.clock(D))
        position = timeline.clock(step)
        # Tested before the q form so that D == T holds the floor rather than
        # restarting the decay (the old lambda's max(1, T - D)).
        if position >= T:
            return 0.0
        if position < start:
            return 1.0
        return k((position - start) / float(max(1, T - start)))

    raise ValueError(f"Unknown schedule curve '{curve}'")


def make_lambda(spec: ScheduleSpec, timeline: ScheduleTimeline) -> Callable[[int], float]:
    """The multiplier as a pure function of ``(step, timeline.events)``."""
    if spec.curve not in _CURVES:
        raise ValueError(f"Unknown schedule curve '{spec.curve}'")
    if spec.curve == "wsd":
        _decay_shape(spec.decay_shape)

    def lr_lambda(step: int) -> float:
        return timeline.multiplier(spec, step)

    return lr_lambda


def describe_spec(spec: ScheduleSpec) -> str:
    """One line for the startup log (§13)."""
    parts = [f"warmup={spec.warmup_steps}", f"total={spec.total_steps}",
             f"floor_ratio={spec.floor_ratio}"
             + (" (default)" if spec.floor_defaulted else "")]
    if spec.curve == "wsd":
        parts.append("decay_start=" + (
            "manual" if spec.decay_start_step is None
            else f"{spec.decay_start_step} ({spec.decay_start_axis})"))
        parts.append("decay_length=" + (
            "to_end" if spec.decay_length is None else str(spec.decay_length)))
        parts.append(f"decay_shape={spec.decay_shape}")
    if spec.curve == "relora":
        parts.append(f"restart_warmup={spec.relora_restart_warmup_steps}")
    if spec.curve == "cosine_with_restarts":
        parts.append("cycle_steps=" + (
            "whole_run" if spec.cycle_steps <= 0 else str(spec.cycle_steps)))
        parts.append(f"cycle_peak_decay={spec.cycle_peak_decay}")
    return f"{spec.name} ({', '.join(parts)}) [scheduler steps]"


def sample_curve(spec: ScheduleSpec, timeline: ScheduleTimeline,
                 n_points: int = 256) -> List[Tuple[int, float]]:
    """``n_points`` evenly spaced ``(step, multiplier)`` samples (D20).

    The preview endpoint's only source, so the UI never re-implements a
    schedule in TypeScript. Endpoints included; a discontinuity BETWEEN two
    samples (a hard restart, a zero-length cancel) is not resolved by them.
    """
    total = timeline.current_total(spec.total_steps)
    points = max(2, min(512, int(n_points)))
    last = max(1, int(total))
    steps = sorted({round(i * last / (points - 1)) for i in range(points)})
    return [(step, timeline.multiplier(spec, step)) for step in steps]


def build_lr_scheduler(
    optimizer,
    spec: ScheduleSpec,
    timeline: ScheduleTimeline,
    group_specs: Optional[Sequence[ScheduleSpec]] = None,
) -> LambdaLR:
    """The only place this project constructs an LR scheduler.

    Always a ``LambdaLR`` carrying a LIST of lambdas, one per param group, so
    ``lr_utils.reassert_config_lr``'s ``len(lambdas) == n_groups`` test and the
    fast-forward's zip hold without a special case.

    ``group_specs`` is D16's per-component schedule, already resolved: one spec
    per param group, in group order. None (the default) puts the run's single
    spec on every group, which is what every run does unless
    ``lr_group_schedules`` is set. The specs share ONE timeline -- only the
    event list is shared, the overlay state is derived per spec (§17.3).
    """
    timeline.bind_spec(spec)
    groups = list(optimizer.param_groups)
    if group_specs is None:
        lr_lambda = make_lambda(spec, timeline)
        return LambdaLR(optimizer, lr_lambda=[lr_lambda] * len(groups))

    specs = list(group_specs)
    if len(specs) != len(groups):
        raise ValueError(
            f"group_specs describes {len(specs)} param group(s) but the "
            f"optimizer has {len(groups)}: the lambdas are applied BY INDEX, so "
            f"a mismatch would give some group another group's schedule.")
    return LambdaLR(optimizer,
                    lr_lambda=[make_lambda(s, timeline) for s in specs])


def build_depth_map(blocks: Optional[Sequence[Any]]
                    ) -> Tuple[Dict[int, int], int]:
    """``id(param) -> depth`` over an architecture's forward-ordered blocks.

    An entry is one block, or several blocks that SHARE a depth: Ideogram 4
    runs a conditional and an unconditional copy of one stack, whose layer j is
    the same depth in both. LoRA parameters are covered without being mentioned
    -- an adapter replaces the target Linear in its parent's module tree, so
    ``block.parameters()`` already yields them.
    """
    depth_of: Dict[int, int] = {}
    count = 0
    for depth, entry in enumerate(blocks or []):
        modules = entry if isinstance(entry, (list, tuple)) else [entry]
        for module in modules:
            for param in module.parameters():
                depth_of[id(param)] = depth
        count = depth + 1
    return depth_of, count


def apply_layer_decay(groups: Sequence[Mapping[str, Any]],
                      depth_of: Mapping[int, int], n_depths: int,
                      factor: float) -> List[Dict[str, Any]]:
    """Split each optimizer group by block depth and scale its LR (D17/§11.2).

    ``lr * factor ** (n - 1 - depth)``: the deepest block keeps the group's own
    rate and every earlier one is scaled down. A parameter in no block (an
    embedder, the final layer, a text encoder) is treated as the last depth,
    i.e. left at 1.0.

    Group order is preserved and depths ascend within a group, because a resume
    writes the recorded per-group base LRs back BY INDEX. The split keeps
    ``component`` and puts the depth only in ``name`` (§17.3), so a
    ``lr_group_schedules`` mapping still resolves after the split.
    """
    factor = float(factor)
    if n_depths <= 0 or factor == 1.0:
        return [dict(group) for group in groups]

    out: List[Dict[str, Any]] = []
    for index, group in enumerate(groups):
        name = str(group.get("name") or f"group{index}")
        buckets: Dict[int, List[Any]] = {}
        for param in group.get("params", []):
            buckets.setdefault(depth_of.get(id(param), n_depths - 1),
                               []).append(param)
        base_lr = group.get("lr")
        for depth in sorted(buckets):
            split = dict(group)
            split["params"] = buckets[depth]
            split["name"] = f"{name}.d{depth:02d}"
            split.setdefault("component", name)
            if base_lr is not None:
                split["lr"] = float(base_lr) * factor ** (n_depths - 1 - depth)
            out.append(split)
    return out
