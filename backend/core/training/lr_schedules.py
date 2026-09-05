"""LR schedule definitions, resolution and construction.

Phase P0 of ``docs/guides/LR_SCHEDULER_DESIGN.md``: the one place a learning-rate
schedule is built for the diffusion trainers. Everything it returns is a
``torch.optim.lr_scheduler.LambdaLR`` whose multiplier is a PURE function of the
step and of the timeline's events -- the invariant ``lr_utils.reassert_config_lr``
and ``BaseTrainer._fast_forward_one_lr_scheduler`` both evaluate lambdas out of
order and rely on.

Step axis (D9/§17.1): one unit per ``scheduler.step()``, i.e. per update
boundary, NOT per ``global_step``. ``BaseTrainer`` divides by the effective
advance interval before calling ``resolve_spec``.

P0 ports the six diffusers schedules and the in-house ``plateau_cosine_floor``
so that the multiplier is bit-identical for ``0 <= s <= total_steps``. Three
deliberate exceptions, listed in the design's §18: ``constant`` now warms up
when ``lr_warmup_steps > 0``, ``cosine`` holds its terminal value past
``total_steps`` instead of rising again, and the gas axis fix moves where a
resume lands. The runtime timeline (P1/P2), the generalized floor and the
``wsd``/``rex``/in-house restart curves (P3) land later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from torch.optim.lr_scheduler import LambdaLR

# SSoT: api/param_defaults.TRAINING_DEFAULTS.
from api.param_defaults import TRAINING_DEFAULTS as _TRAINING_DEFAULTS

__all__ = [
    "LR_SCHEDULER_NAMES",
    "ScheduleSpec",
    "ScheduleTimeline",
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
    decay_start_axis: str = "nominal"
    decay_length: Optional[int] = None
    decay_end_kind: str = "nominal_total"
    decay_shape: str = "cosine"
    poly_lr_end: float = _POLYNOMIAL_LR_END
    # Bound to the optimizer by build_lr_scheduler: diffusers' polynomial
    # multiplier is expressed relative to optimizer.defaults["lr"].
    poly_lr_init: Optional[float] = None


class ScheduleTimeline:
    """Runtime events shared by every scheduler of one run.

    P0 stub: the only event is the construction-time ``total_steps(at=0)``, so
    :meth:`clock` is the identity and the nominal total never moves. The lambda
    closes over this OBJECT rather than over its values, so P1's resume seam can
    install a saved event list after construction and every later evaluation
    sees it.
    """

    def __init__(self, events: Optional[Sequence[Mapping[str, Any]]] = None):
        self.events: List[Dict[str, Any]] = [dict(e) for e in (events or [])]

    def set_total_steps(self, value: int) -> None:
        if any(e.get("kind") == "total_steps" for e in self.events):
            raise NotImplementedError(
                "re-anchoring total_steps is the extension warp (P1)")
        self.events.append({"kind": "total_steps", "at": 0, "value": int(value)})

    def nominal_total(self, default: int) -> int:
        for event in self.events:
            if event.get("kind") == "total_steps":
                return int(event["value"])
        return int(default)

    def clock(self, step: int) -> float:
        """Real scheduler step -> nominal axis. Identity until P1's warp."""
        return float(step)


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
            floor_ratio=floor, decay_start_step=start,
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


def make_lambda(spec: ScheduleSpec, timeline: ScheduleTimeline) -> Callable[[int], float]:
    """The multiplier as a pure function of ``(step, timeline.events)``."""
    W = spec.warmup_steps
    T_spec = spec.total_steps
    curve = spec.curve

    def ramp(step: int) -> Optional[float]:
        if W > 0 and step < W:
            return step / float(W)
        return None

    if curve == "constant":
        # constant_with_warmup's shape. diffusers' bare `constant` ignored
        # num_warmup_steps entirely (optimization.py:323-324); §4.2 makes the
        # two one curve.
        def constant_lambda(step: int) -> float:
            warm = ramp(step)
            return 1.0 if warm is None else warm

        return constant_lambda

    if curve == "linear":
        def linear_lambda(step: int) -> float:
            warm = ramp(step)
            if warm is not None:
                return warm
            T = timeline.nominal_total(T_spec)
            s = timeline.clock(step)
            return max(0.0, float(T - s) / float(max(1, T - W)))

        return linear_lambda

    if curve == "cosine":
        def cosine_lambda(step: int) -> float:
            warm = ramp(step)
            if warm is not None:
                return warm
            T = timeline.nominal_total(T_spec)
            progress = float(timeline.clock(step) - W) / float(max(1, T - W))
            # Clamped, where diffusers lets the cosine rise again past T.
            if progress > 1.0:
                progress = 1.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return cosine_lambda

    if curve == "cosine_with_restarts":
        def restarts_lambda(step: int) -> float:
            warm = ramp(step)
            if warm is not None:
                return warm
            T = timeline.nominal_total(T_spec)
            progress = float(timeline.clock(step) - W) / float(max(1, T - W))
            if progress >= 1.0:
                return 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(
                math.pi * ((_RESTART_CYCLES * progress) % 1.0))))

        return restarts_lambda

    if curve == "polynomial":
        lr_init = spec.poly_lr_init
        lr_end = spec.poly_lr_end
        if lr_init is None:
            raise ValueError(
                "polynomial needs poly_lr_init; build it through build_lr_scheduler()")

        def polynomial_lambda(step: int) -> float:
            warm = ramp(step)
            if warm is not None:
                return warm
            T = timeline.nominal_total(T_spec)
            s = timeline.clock(step)
            if s > T:
                return lr_end / lr_init
            lr_range = lr_init - lr_end
            decay_steps = T - W
            pct_remaining = 1 - (s - W) / decay_steps
            decay = lr_range * pct_remaining ** _POLYNOMIAL_POWER + lr_end
            return decay / lr_init

        return polynomial_lambda

    if curve == "wsd":
        F = spec.floor_ratio
        D = spec.decay_start_step
        shape = _decay_shape(spec.decay_shape)
        nominal_start = spec.decay_start_axis == "nominal"
        if spec.decay_end_kind != "nominal_total" or spec.decay_length is not None:
            raise NotImplementedError("explicit decay lengths ship in P3")

        def wsd_lambda(step: int) -> float:
            warm = ramp(step)
            if warm is not None:
                return warm
            if D is None:
                return 1.0  # manual WSD, not started: holds 1 past T as well
            T = timeline.nominal_total(T_spec)
            position = timeline.clock(step) if nominal_start else float(step)
            # Tested before the q form so that D == T holds the floor rather
            # than restarting the decay (the old lambda's max(1, T - D)).
            if position >= T:
                return F
            if position < D:
                return 1.0
            q = (position - D) / float(max(1, T - D))
            return F + (1.0 - F) * shape(q)

        return wsd_lambda

    raise ValueError(f"Unknown schedule curve '{curve}'")


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

    lr_lambda = make_lambda(spec, timeline)
    return LambdaLR(optimizer, lr_lambda=[lr_lambda] * len(optimizer.param_groups))
