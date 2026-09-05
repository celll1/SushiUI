"""Guard: the in-house LR schedules reproduce what they replaced.

``core/training/lr_schedules.py`` (P0 of docs/guides/LR_SCHEDULER_DESIGN.md)
took over from ``diffusers.optimization.get_scheduler`` and from
``BaseTrainer._build_plateau_cosine_floor_scheduler``. A run resumed across that
change must land on the same curve, so the bar here is bit-identity -- ``abs
diff == 0``, not "close" -- against:

* ``get_scheduler(name, ...).lr_lambdas[0]`` for the compatible cases: the six
  ported names with their own floors, ``0 <= W < T`` and ``0 <= s <= T``
  (§17.2). Outside that window the deliberate changes below apply and
  equivalence is NOT claimed.
* a LITERAL COPY of the old plateau lambda, transcribed below rather than
  imported, so that a silent change to the shape fails here instead of being
  followed.

The deliberate P0 changes get their own tests rather than being held to
bit-identity (§14, §18):

* ``constant`` + ``lr_warmup_steps > 0`` now warms up (diffusers dropped the
  warmup for that one name);
* ``cosine`` holds its terminal value past ``T`` instead of rising again;
* ``polynomial``'s floor is ``1e-7 / optimizer.defaults['lr']``, ported as-is;
* the scheduler axis is ``total_steps // gradient_accumulation_steps``.

Purity (§4.3) is checked by evaluating every step ascending, descending and
shuffled: the fast-forward, the re-assertion and the composed re-warmup all
evaluate the same lambda out of order.

CPU-only and hermetic: one 4-element parameter, no model, no dataset, no GPU.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/lr_schedules_test.py -v
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

from core.training.lr_schedules import (
    LR_SCHEDULER_NAMES,
    ScheduleTimeline,
    build_lr_scheduler,
    make_lambda,
    resolve_spec,
)

BASE_LR = 1e-4


def _optimizer(lr: float = BASE_LR, groups: int = 1):
    params = [torch.nn.Parameter(torch.zeros(4)) for _ in range(groups)]
    return torch.optim.AdamW([{"params": [p]} for p in params], lr=lr)


def _ours(name: str, W: int, T: int, config=None, lr: float = BASE_LR):
    """The shipping lambda for one schedule."""
    spec = resolve_spec(config or {}, warmup_steps=W, total_steps=T, name=name)
    scheduler = build_lr_scheduler(_optimizer(lr), spec, _timeline(spec))
    return scheduler.lr_lambdas[0]


def _timeline(spec):
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    return timeline


def _diffusers(name: str, W: int, T: int, lr: float = BASE_LR):
    from diffusers.optimization import get_scheduler

    return get_scheduler(
        name, optimizer=_optimizer(lr), num_warmup_steps=W, num_training_steps=T
    ).lr_lambdas[0]


# ---------------------------------------------------------------------------
# The old plateau_cosine_floor lambda, transcribed (deleted from base_trainer)
# ---------------------------------------------------------------------------

def _legacy_plateau(W: int, T: int, decay_start_ratio: float, floor_ratio: float):
    W = max(0, int(W))
    T = max(1, int(T))
    D = round(decay_start_ratio * T)
    D = max(W, min(D, T))

    def lr_lambda(step: int) -> float:
        if W > 0 and step < W:
            return step / float(W)
        if step < D:
            return 1.0
        if step < T:
            span = max(1, T - D)
            progress = (step - D) / float(span)
            return floor_ratio + 0.5 * (1.0 - floor_ratio) * (1.0 + math.cos(math.pi * progress))
        return floor_ratio

    return lr_lambda


# ---------------------------------------------------------------------------
# Bit-identity against diffusers, over the compatible window
# ---------------------------------------------------------------------------

_COMPATIBLE = ("constant_with_warmup", "linear", "cosine",
               "cosine_with_restarts", "polynomial")
_SHAPES = [(0, 1000), (1, 1000), (100, 1000), (999, 1000), (0, 7), (3, 7)]


@pytest.mark.parametrize("name", _COMPATIBLE)
@pytest.mark.parametrize("W,T", _SHAPES)
def test_bit_identical_to_diffusers(name, W, T):
    ours, theirs = _ours(name, W, T), _diffusers(name, W, T)
    for step in range(0, T + 1):
        got, want = ours(step), theirs(step)
        assert got == want, f"{name} step {step}: {got!r} != {want!r}"


def test_constant_without_warmup_is_bit_identical():
    """`constant` ignores num_warmup_steps in diffusers, so W=0 is the only
    window where the two agree at all."""
    ours, theirs = _ours("constant", 0, 1000), _diffusers("constant", 0, 1000)
    for step in range(0, 1001):
        assert ours(step) == theirs(step), step


def test_polynomial_floor_is_the_ported_diffusers_one():
    """Not 0: lr_end / lr_init, with lr_end = 1e-7 (optimization.py:226)."""
    for lr in (1e-4, 1e-6, 2.5e-6):
        ours = _ours("polynomial", 10, 100, lr=lr)
        assert ours(100) == pytest.approx(1e-7 / lr, rel=1e-12)
        assert ours(100) == _diffusers("polynomial", 10, 100, lr=lr)(100)
        assert ours(500) == 1e-7 / lr, "past T it holds lr_end/lr_init"


def test_polynomial_refuses_an_lr_below_its_floor():
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="polynomial")
    with pytest.raises(ValueError):
        build_lr_scheduler(_optimizer(lr=1e-8), spec, _timeline(spec))


# ---------------------------------------------------------------------------
# plateau_cosine_floor: bit-identical to the deleted implementation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("W,T", _SHAPES + [(0, 10000), (500, 10000)])
@pytest.mark.parametrize("ratio,floor", [(0.85, 0.25), (0.0, 0.0), (1.0, 0.5),
                                         (0.5, 0.1), (0.999, 0.25)])
def test_plateau_cosine_floor_matches_the_old_lambda(W, T, ratio, floor):
    config = {"lr_decay_start_ratio": ratio, "lr_floor_ratio": floor}
    ours = _ours("plateau_cosine_floor", W, T, config)
    legacy = _legacy_plateau(W, T, ratio, floor)
    for step in range(0, T + T // 2 + 2):   # past T too: both hold the floor
        assert ours(step) == legacy(step), f"step {step}"


def test_plateau_defaults_come_from_param_defaults():
    from api.param_defaults import TRAINING_DEFAULTS

    ours = _ours("plateau_cosine_floor", 0, 1000, {})
    legacy = _legacy_plateau(0, 1000, TRAINING_DEFAULTS["lr_decay_start_ratio"],
                             TRAINING_DEFAULTS["lr_floor_ratio"])
    for step in range(0, 1001):
        assert ours(step) == legacy(step), step


def test_plateau_decay_start_equals_total():
    """ratio=1.0 puts D at T. The old lambda's max(1, T-D) never divided by
    zero there because the D<=step<T branch was empty -- the floor is held."""
    ours = _ours("plateau_cosine_floor", 0, 100,
                 {"lr_decay_start_ratio": 1.0, "lr_floor_ratio": 0.25})
    assert ours(99) == 1.0
    assert ours(100) == 0.25
    assert ours(200) == 0.25


# ---------------------------------------------------------------------------
# The deliberate changes
# ---------------------------------------------------------------------------

def test_constant_now_warms_up():
    ours = _ours("constant", 100, 1000)
    assert ours(0) == 0.0
    assert ours(50) == 0.5
    assert ours(100) == 1.0
    assert ours(5000) == 1.0
    # diffusers' `constant` never ramped: this is the change, not a port.
    assert _diffusers("constant", 100, 1000)(50) == 1
    # and it is exactly `constant_with_warmup`, which did.
    warmed = _diffusers("constant_with_warmup", 100, 1000)
    for step in range(0, 1001):
        assert ours(step) == warmed(step), step


def test_cosine_holds_its_terminal_value_past_total():
    ours, theirs = _ours("cosine", 0, 100), _diffusers("cosine", 0, 100)
    assert ours(100) == theirs(100) == 0.0
    for step in (101, 150, 200, 1000):
        assert ours(step) == 0.0, step
    # Precondition: diffusers' cosine rises again -- a full period past T it is
    # back at the un-decayed LR.
    assert theirs(200) == pytest.approx(1.0)


def test_cosine_with_restarts_is_a_single_cosine_today():
    """num_cycles is never passed, so the name is a single cosine (§1-1)."""
    single, restarts = _ours("cosine", 10, 500), _ours("cosine_with_restarts", 10, 500)
    for step in range(0, 501):
        assert single(step) == restarts(step), step


# ---------------------------------------------------------------------------
# Purity (§4.3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", LR_SCHEDULER_NAMES)
def test_evaluation_order_does_not_change_any_value(name):
    ours = _ours(name, 37, 400, {"lr_decay_start_ratio": 0.7, "lr_floor_ratio": 0.3})
    steps = list(range(0, 500))
    ascending = [ours(s) for s in steps]

    descending = {s: ours(s) for s in reversed(steps)}
    shuffled_steps = steps[:]
    random.Random(0).shuffle(shuffled_steps)
    shuffled = {s: ours(s) for s in shuffled_steps}
    repeated = [ours(s) for s in steps]

    for step, value in zip(steps, ascending):
        assert descending[step] == value, step
        assert shuffled[step] == value, step
    assert repeated == ascending


# ---------------------------------------------------------------------------
# Construction contract
# ---------------------------------------------------------------------------

def test_one_lambda_per_param_group():
    """lr_utils.reassert_config_lr applies the multiplier only when
    len(lr_lambdas) == len(param_groups)."""
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="cosine")
    optimizer = _optimizer(groups=3)
    scheduler = build_lr_scheduler(optimizer, spec, _timeline(spec))
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert len(scheduler.lr_lambdas) == len(optimizer.param_groups) == 3


def test_unknown_name_is_refused_by_name():
    with pytest.raises(ValueError, match="piecewise_constant"):
        resolve_spec({}, warmup_steps=0, total_steps=100, name="piecewise_constant")


def test_the_stub_timeline_refuses_a_second_total_steps():
    """P1 owns re-anchoring; a silent second event would move the curve."""
    timeline = ScheduleTimeline()
    timeline.set_total_steps(100)
    with pytest.raises(NotImplementedError):
        timeline.set_total_steps(200)


def test_the_lambda_reads_the_timeline_object_not_a_snapshot():
    """P1 installs the saved events after construction; the already-built
    lambda has to see them."""
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="cosine")
    timeline = ScheduleTimeline()
    fn = make_lambda(spec, timeline)
    assert fn(100) == 0.0
    timeline.events.append({"kind": "total_steps", "at": 0, "value": 200})
    assert fn(100) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The scheduler axis (D9 / §17.1)
# ---------------------------------------------------------------------------

from core.training.base_trainer import (  # noqa: E402
    resume_scheduler_position,
    scheduler_total_steps,
)


class _AxisProbe:
    """What the axis helpers read off a trainer."""

    def __init__(self, gas=1):
        self.log_prefix = "[Test]"
        self._grad_accum_steps = gas


@pytest.mark.parametrize("T,gas,expected", [(1000, 1, 1000), (1000, 4, 250),
                                            (10, 4, 2), (10, 3, 3), (4, 4, 1)])
def test_total_steps_floors_onto_the_scheduler_axis(T, gas, expected):
    assert scheduler_total_steps(_AxisProbe(gas), T) == expected


def test_a_run_that_could_never_step_is_refused():
    with pytest.raises(ValueError, match="never step"):
        scheduler_total_steps(_AxisProbe(gas=4), 3)


def test_resume_prefers_the_saved_scheduler_step():
    probe = _AxisProbe(gas=4)
    probe._resume_scheduler_step = 1234
    probe._resume_scheduler_interval = 4
    assert resume_scheduler_position(probe, 9999) == 1234


def test_resume_without_a_saved_step_divides_and_says_so(capsys):
    probe = _AxisProbe(gas=4)
    assert resume_scheduler_position(probe, 9000) == 2250
    assert "lr_schedule_position_estimated" in capsys.readouterr().out


def test_resume_at_gas_1_is_the_step_itself(capsys):
    probe = _AxisProbe(gas=1)
    assert resume_scheduler_position(probe, 9000) == 9000
    assert "lr_schedule_position_estimated" not in capsys.readouterr().out


class _StateHarness:
    """The real state file round trip, with nothing else attached."""

    from core.training.base_trainer import BaseTrainer as _B

    save_training_state = _B.save_training_state
    load_training_state = _B.load_training_state
    del _B

    def __init__(self, output_dir, gas=4):
        self.output_dir = Path(output_dir)
        self.run_name = "20260101_000000_deadbeef"
        self.log_prefix = "[Test]"
        self._grad_accum_steps = gas
        self._dataset_fingerprint = None
        self._batches_per_epoch = 10
        self._crop_plan_fingerprint = None
        spec = resolve_spec({}, warmup_steps=0, total_steps=1000, name="cosine")
        self.lr_scheduler = build_lr_scheduler(_optimizer(), spec, _timeline(spec))


def test_the_scheduler_position_round_trips_through_state_json(tmp_path):
    saver = _StateHarness(tmp_path)
    for _ in range(7):
        saver.lr_scheduler.step()
    saver.save_training_state(step=400, epoch=0, batch_idx=3)

    loader = _StateHarness(tmp_path)
    state = loader.load_training_state(400)
    assert state["scheduler_step"] == 7
    assert state["lr_scheduler_advance_interval"] == 4
    assert state["lr_schedule_version"] == 1
    # The resume reads 7, not 400 // 4 = 100.
    assert resume_scheduler_position(loader, 400) == 7


def test_a_state_file_without_the_key_falls_back_to_the_estimate(tmp_path, capsys):
    harness = _StateHarness(tmp_path)
    harness.save_training_state(step=400, epoch=0, batch_idx=3)
    path = next(Path(tmp_path).glob("*_state.json"))
    body = json.loads(path.read_text())
    del body["scheduler_step"], body["lr_scheduler_advance_interval"]
    path.write_text(json.dumps(body))

    loader = _StateHarness(tmp_path)
    loader.load_training_state(400)
    assert resume_scheduler_position(loader, 400) == 100
    assert "lr_schedule_position_estimated" in capsys.readouterr().out


def test_an_accumulation_change_across_a_resume_is_reported(capsys):
    probe = _AxisProbe(gas=8)
    probe._resume_scheduler_step = 500
    probe._resume_scheduler_interval = 4
    assert resume_scheduler_position(probe, 4000) == 500
    assert "lr_schedule_accumulation_changed" in capsys.readouterr().out
