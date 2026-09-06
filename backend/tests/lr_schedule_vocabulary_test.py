"""Gate: P3's schedule vocabulary is reachable, restorable and self-consistent.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/lr_schedule_vocabulary_test.py -v

P3 of docs/guides/LR_SCHEDULER_DESIGN.md opened the registry (`wsd`, `rex`,
`cosine_with_restarts` with real cycles) and generalized the floor (D10). Five
new config keys had to land on every layer of §12.3's checklist, and the one
that bites silently is `PARAM_KEYS`: a key the request model has but that list
does not is dropped on every edit-save, so the run quietly reverts to the
default. `training_edit_restore_coverage_test.py` cannot see that for a
pass-through key -- removing the entry removes it from BOTH sides of its
comparison -- so the coverage is asserted here, against the Pydantic model.

What is checked:

* every `lr_*` request field is in `PARAM_KEYS` and lands in the YAML `train`
  section, and the create -> YAML -> read-back round trip returns it;
* the new keys are written UNCONDITIONALLY (a conditional write hands the
  read-back a Pydantic default, which is only harmless while the default is
  inert -- the floor's is not);
* §12.2's compatibility rule for a YAML with no floor key;
* every name the registry offers builds, and the UI's `<select>`, the openapi
  enum and `LR_SCHEDULER_NAMES` are the same vocabulary;
* §8's REX numbers, which are the reason `rex` is a shape and not "cosine with
  a stronger exponent";
* the preview endpoint samples the same lambda the trainer runs.

CPU-only and hermetic: no model, no dataset, no GPU.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest
import torch
import yaml

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from api.routes import (  # noqa: E402
    TrainingRunCreateRequest,
    _extract_request_params_from_yaml,
)
from core.training.lr_schedules import (  # noqa: E402
    DECAY_SHAPE_NAMES,
    LR_SCHEDULER_NAMES,
    ScheduleTimeline,
    build_lr_scheduler,
    make_lambda,
    resolve_spec,
    sample_curve,
)
from core.training.training_config import TrainingConfigGenerator  # noqa: E402

_PANEL = REPO / "frontend/src/components/training/TrainingConfig.tsx"
_PARAMS_TS = REPO / "frontend/src/components/training/trainingParams.ts"

# The five keys P3 adds, and a value for each that is NOT the default -- a
# round trip that returns the default proves nothing.
NEW_KEYS = {
    "lr_decay_start_step": 700,
    "lr_decay_steps": 250,
    "lr_decay_shape": "linear",
    "lr_cycle_steps": 400,
    "lr_cycle_peak_decay": 0.8,
}


def _optimizer(lr: float = 1e-4, groups: int = 1):
    params = [torch.nn.Parameter(torch.zeros(4)) for _ in range(groups)]
    return torch.optim.AdamW([{"params": [p]} for p in params], lr=lr)


def _timeline(spec):
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    return timeline


def _lambda(name: str, W: int, T: int, config=None):
    spec = resolve_spec(config or {}, warmup_steps=W, total_steps=T, name=name)
    return make_lambda(spec, _timeline(spec))


def _train_section(**overrides) -> dict:
    params = {
        "learning_rate": 1e-4, "batch_size": 1, "optimizer": "adamw8bit",
        "train_unet": True, "train_text_encoder": False, "total_steps": 1000,
        **overrides,
    }
    config = TrainingConfigGenerator().generate_lora_config(
        params,
        run_name="lr_vocab_test",
        base_model_path="/models/does-not-exist.safetensors",
        output_dir="/tmp/lr_vocab_test",
        dataset_configs=[{"dataset_id": 1, "path": "/data/ds"}],
        sample_prompts=[],
    )
    return yaml.safe_load(config)["config"]["process"][0]["train"]


def _param_keys() -> list:
    source = _PARAMS_TS.read_text(encoding="utf-8")
    start = source.index("export const PARAM_KEYS: (keyof TrainingRunCreateRequest)[] = [")
    body = re.sub(r"//[^\n]*", "", source[start:source.index("\n];", start)])
    return re.findall(r'"([A-Za-z0-9_]+)"', body)


# ---------------------------------------------------------------------------
# §12.3's checklist, per key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(NEW_KEYS))
def test_the_request_model_declares_it_with_the_shared_default(key):
    field = TrainingRunCreateRequest.model_fields[key]
    assert field.get_default(call_default_factory=True) == TRAINING_DEFAULTS[key]


@pytest.mark.parametrize("key", sorted(NEW_KEYS))
def test_the_form_restores_it(key):
    """Missing here, an edit-save silently resets the run to the default."""
    assert key in _param_keys()


def test_every_lr_request_field_is_restorable():
    """The general form of the test above: no lr_* field may be unlisted."""
    keys = set(_param_keys())
    missing = sorted(f for f in TrainingRunCreateRequest.model_fields
                     if f.startswith("lr_") and f not in keys)
    assert missing == []


@pytest.mark.parametrize("key,value", sorted(NEW_KEYS.items()))
def test_it_round_trips_request_to_yaml_to_request(key, value):
    train = _train_section(**{key: value})
    assert train[key] == value
    back = _extract_request_params_from_yaml({"train": train}, job="lora")
    assert back[key] == value
    # And the read-back still validates as the request the edit form PUTs.
    restored = TrainingRunCreateRequest(
        training_method="lora", base_model_path="x",
        **{k: v for k, v in back.items() if k in TrainingRunCreateRequest.model_fields
           and k not in ("training_method", "base_model_path")})
    assert getattr(restored, key) == value


@pytest.mark.parametrize("scheduler", ["constant", "cosine", "wsd",
                                       "plateau_cosine_floor"])
def test_the_new_keys_are_written_whatever_the_scheduler_is(scheduler):
    """Unconditionally, the way rewarmup_on_optimizer_reset is. A conditional
    write makes the read-back return the Pydantic default instead."""
    train = _train_section(lr_scheduler=scheduler)
    for key in list(NEW_KEYS) + ["lr_floor_ratio"]:
        assert train[key] == TRAINING_DEFAULTS[key], key


def test_the_plateau_ratio_stays_conditional():
    """§12.1 keeps it on its own scheduler: no other name reads it."""
    assert "lr_decay_start_ratio" not in _train_section(lr_scheduler="cosine")
    assert "lr_decay_start_ratio" in _train_section(
        lr_scheduler="plateau_cosine_floor")


# ---------------------------------------------------------------------------
# §12.2: the floor of a YAML that predates D10
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", LR_SCHEDULER_NAMES)
def test_a_yaml_without_a_floor_key_reads_as_the_old_behaviour(name):
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name=name)
    expected = 0.25 if name == "plateau_cosine_floor" else 0.0
    assert spec.floor_ratio == expected
    assert spec.floor_defaulted is True


@pytest.mark.parametrize("name", LR_SCHEDULER_NAMES)
def test_an_explicit_floor_is_used_and_marked_explicit(name):
    spec = resolve_spec({"lr_floor_ratio": 0.4}, warmup_steps=0,
                        total_steps=100, name=name)
    assert spec.floor_ratio == 0.4
    assert spec.floor_defaulted is False


def test_the_rule_reads_the_yaml_not_the_pydantic_default():
    """The request model defaults lr_floor_ratio to 0.25 for NEW runs; that
    must not leak into how an old run's YAML is read."""
    assert TRAINING_DEFAULTS["lr_floor_ratio"] == 0.25
    assert resolve_spec({}, warmup_steps=0, total_steps=100,
                        name="cosine").floor_ratio == 0.0


@pytest.mark.parametrize("name", ["linear", "cosine", "polynomial", "rex",
                                  "wsd", "cosine_with_restarts"])
def test_the_floor_is_where_every_decaying_curve_ends(name):
    fn = _lambda(name, 0, 100, {"lr_floor_ratio": 0.2, "lr_decay_start_step": 1})
    assert fn(100) == pytest.approx(0.2)
    assert fn(400) == pytest.approx(0.2), "and it is held past the end"


def test_constant_ignores_the_floor():
    """D10: there is no decay to put a floor under."""
    fn = _lambda("constant", 10, 100, {"lr_floor_ratio": 0.5})
    assert fn(10) == 1.0 and fn(100) == 1.0 and fn(500) == 1.0


# ---------------------------------------------------------------------------
# The vocabulary is one vocabulary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", LR_SCHEDULER_NAMES)
def test_every_registry_name_builds(name):
    spec = resolve_spec({}, warmup_steps=10, total_steps=100, name=name)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    scheduler = build_lr_scheduler(_optimizer(), spec, timeline)
    values = [scheduler.lr_lambdas[0](s) for s in range(0, 121)]
    assert all(0.0 <= v <= 1.0 for v in values), name


def test_the_ui_offers_exactly_the_registry():
    """The `<select>` mirrors LR_SCHEDULER_NAMES minus constant_with_warmup,
    which is accepted but not offered (§12.5: it is constant's curve)."""
    source = _PANEL.read_text(encoding="utf-8")
    start = source.index("const LR_SCHEDULER_OPTIONS")
    body = source[start:source.index("\n];", start)]
    offered = re.findall(r'value: "([a-z_]+)"', body)
    assert offered == [n for n in LR_SCHEDULER_NAMES if n != "constant_with_warmup"]
    # Still selectable when a run already stored it, or it would read as blank.
    assert 'lrScheduler === "constant_with_warmup"' in source


def test_the_openapi_enum_is_the_registry():
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    assert props["lr_scheduler"]["enum"] == list(LR_SCHEDULER_NAMES)
    assert props["lr_decay_shape"]["enum"] == list(DECAY_SHAPE_NAMES)
    for key in NEW_KEYS:
        assert props[key]["default"] == TRAINING_DEFAULTS[key], key


def test_an_unknown_name_is_refused_at_the_api_not_at_the_optimizer():
    with pytest.raises(ValueError, match="piecewise_constant"):
        TrainingRunCreateRequest(training_method="lora", base_model_path="x",
                                 lr_scheduler="piecewise_constant")


def test_the_ui_decay_shapes_are_the_registry_shapes():
    source = _PANEL.read_text(encoding="utf-8")
    block = source[source.index('updateParam("lr_decay_shape"'):]
    block = block[:block.index("</select>")]
    assert re.findall(r'<option value="([a-z]+)">', block) == list(DECAY_SHAPE_NAMES)


# ---------------------------------------------------------------------------
# The curves themselves
# ---------------------------------------------------------------------------

def test_rex_is_not_a_steeper_cosine():
    """§8: the shapes differ at the midpoint and enter the decay at different
    slopes, which is why lr_decay_shape exists instead of an exponent."""
    rex = _lambda("rex", 0, 1000)
    cosine = _lambda("cosine", 0, 1000)
    assert rex(500) == pytest.approx(2 / 3, rel=1e-9)
    assert cosine(500) == pytest.approx(0.5, rel=1e-9)
    # k'(0): -1/2 for REX, 0 for cosine -- so over the first step REX has
    # already fallen two orders of magnitude further.
    assert (rex(1) - rex(0)) * 1000 == pytest.approx(-0.5, abs=1e-3)
    assert abs(cosine(1) - cosine(0)) < abs(rex(1) - rex(0)) / 100
    assert rex(1000) == 0.0 and cosine(1000) == 0.0


def test_rex_starts_at_the_end_of_warmup_on_the_real_axis():
    fn = _lambda("rex", 100, 1000)
    assert fn(50) == 0.5
    assert fn(100) == 1.0
    assert fn(550) == pytest.approx(2 / 3, rel=1e-9)


def test_wsd_holds_until_its_start_then_takes_the_configured_shape():
    fn = _lambda("wsd", 0, 1000, {"lr_decay_start_step": 600,
                                  "lr_decay_shape": "linear",
                                  "lr_floor_ratio": 0.1})
    assert fn(599) == 1.0
    assert fn(800) == pytest.approx(0.1 + 0.9 * 0.5)
    assert fn(1000) == pytest.approx(0.1)


def test_wsd_with_no_start_step_never_decays_from_config():
    """External D = 0 is "manual": the command endpoint starts it (§17.2)."""
    spec = resolve_spec({}, warmup_steps=0, total_steps=1000, name="wsd")
    assert spec.decay_start_step is None
    fn = _lambda("wsd", 0, 1000, {"lr_floor_ratio": 0.1})
    assert [fn(s) for s in (0, 500, 1000, 5000)] == [1.0, 1.0, 1.0, 1.0]


def test_an_explicit_decay_length_is_real_steps():
    fn = _lambda("wsd", 0, 1000, {"lr_decay_start_step": 400,
                                  "lr_decay_steps": 200,
                                  "lr_floor_ratio": 0.25})
    assert fn(400) == 1.0
    assert fn(500) == pytest.approx(0.25 + 0.75 * 0.5)
    assert fn(600) == pytest.approx(0.25)
    assert fn(900) == pytest.approx(0.25), "the rest of the run is at the floor"


def test_a_zero_cycle_length_is_still_a_single_cosine():
    """§9.1: existing YAMLs resume on the same curve, bit for bit."""
    cosine = _lambda("cosine", 10, 500)
    restarts = _lambda("cosine_with_restarts", 10, 500)
    for step in range(0, 600):
        assert cosine(step) == restarts(step), step


def test_cycles_restart_and_anneal_on_the_real_axis():
    fn = _lambda("cosine_with_restarts", 0, 1000,
                 {"lr_cycle_steps": 250, "lr_cycle_peak_decay": 0.5})
    assert fn(0) == 1.0
    assert fn(125) == pytest.approx(0.5)
    assert fn(250) == pytest.approx(0.5), "hard restart to this cycle's peak"
    assert fn(375) == pytest.approx(0.25)
    assert fn(500) == pytest.approx(0.25)
    # Absolute length: the cycle structure does not depend on the total.
    longer = _lambda("cosine_with_restarts", 0, 4000,
                     {"lr_cycle_steps": 250, "lr_cycle_peak_decay": 0.5})
    for step in range(0, 1001):
        assert longer(step) == fn(step), step


def test_annealed_cycles_still_respect_the_floor():
    fn = _lambda("cosine_with_restarts", 0, 1000,
                 {"lr_cycle_steps": 100, "lr_cycle_peak_decay": 0.5,
                  "lr_floor_ratio": 0.2})
    values = [fn(s) for s in range(0, 1001)]
    # The trough sits between two integer steps, so the floor is approached
    # rather than sampled; what matters is that nothing goes under it.
    assert min(values) >= 0.2
    assert min(values) == pytest.approx(0.2, abs=1e-4)


@pytest.mark.parametrize("name", LR_SCHEDULER_NAMES)
def test_evaluation_order_still_does_not_matter(name):
    """§4.3, extended to the P3 names: the fast-forward and the re-assertion
    both evaluate the lambda out of order."""
    fn = _lambda(name, 13, 400, {"lr_decay_start_step": 200, "lr_decay_steps": 90,
                                 "lr_decay_shape": "rex", "lr_cycle_steps": 60,
                                 "lr_cycle_peak_decay": 0.9,
                                 "lr_floor_ratio": 0.3})
    ascending = [fn(s) for s in range(0, 500)]
    descending = {s: fn(s) for s in reversed(range(0, 500))}
    assert [descending[s] for s in range(0, 500)] == ascending


# ---------------------------------------------------------------------------
# The preview endpoint (D20)
# ---------------------------------------------------------------------------

def _preview(**kwargs):
    import asyncio

    from api.routes import preview_lr_schedule

    return asyncio.run(preview_lr_schedule(**kwargs))


def test_the_preview_samples_the_same_lambda_the_trainer_runs():
    body = _preview(lr_scheduler="wsd", total_steps=1000, lr_warmup_steps=50,
                    lr_decay_start_step=600, lr_decay_shape="rex",
                    lr_floor_ratio=0.1, n_points=64)
    fn = _lambda("wsd", 50, 1000, {"lr_decay_start_step": 600,
                                   "lr_decay_shape": "rex",
                                   "lr_floor_ratio": 0.1})
    assert len(body["points"]) == body["n_points"] > 1
    for step, value in body["points"]:
        assert value == fn(step), step


def test_the_preview_is_on_the_optimizer_step_axis():
    body = _preview(lr_scheduler="cosine", total_steps=1000,
                    gradient_accumulation_steps=4, n_points=8)
    assert body["scheduler_total_steps"] == 250
    assert body["points"][-1][0] == 250


def test_the_preview_puts_the_config_step_keys_on_the_trainer_s_axis():
    """The preview resolves the same spec the trainer would, at gas > 1 too."""
    from core.training.base_trainer import resolve_lr_schedule_spec

    class _Probe:
        log_prefix = "[Test]"
        _grad_accum_steps = 4
        optimizer_warmup_steps = 200
        config = {"lr_decay_start_step": 600, "lr_decay_steps": 200,
                  "lr_floor_ratio": 0.1, "lr_decay_shape": "cosine"}

    spec = resolve_lr_schedule_spec(_Probe(), "wsd", 1000)
    body = _preview(lr_scheduler="wsd", total_steps=1000,
                    gradient_accumulation_steps=4, lr_warmup_steps=200,
                    lr_decay_start_step=600, lr_decay_steps=200,
                    lr_floor_ratio=0.1, n_points=64)

    assert (spec.decay_start_step, spec.decay_length) == (150, 50)
    fn = make_lambda(spec, _timeline(spec))
    assert body["scheduler_total_steps"] == spec.total_steps
    assert body["warmup_steps"] == spec.warmup_steps
    for step, value in body["points"]:
        assert value == fn(step), step


def test_the_preview_reports_the_compatibility_floor_as_defaulted():
    body = _preview(lr_scheduler="polynomial", total_steps=100, n_points=4)
    assert body["floor_ratio"] == 0.0 and body["floor_defaulted"] is True
    body = _preview(lr_scheduler="plateau_cosine_floor", total_steps=100,
                    n_points=4)
    assert body["floor_ratio"] == 0.25 and body["floor_defaulted"] is True


def test_the_preview_refuses_what_the_trainer_would_refuse():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as bad_name:
        _preview(lr_scheduler="nope", total_steps=100)
    assert bad_name.value.status_code == 400
    with pytest.raises(HTTPException) as never_steps:
        _preview(lr_scheduler="cosine", total_steps=3,
                 gradient_accumulation_steps=4)
    assert never_steps.value.status_code == 400


def test_sample_curve_is_clamped_and_includes_both_ends():
    spec = resolve_spec({}, warmup_steps=0, total_steps=1000, name="cosine")
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    points = sample_curve(spec, timeline, n_points=10_000)
    assert len(points) <= 512
    assert points[0][0] == 0 and points[-1][0] == 1000
    assert points == sorted(points)


# ---------------------------------------------------------------------------
# The two warnings P3 turns on (§13)
# ---------------------------------------------------------------------------

class _WarnTrainer:
    log_prefix = "[Test]"
    _grad_accum_steps = 1

    def __init__(self, config):
        self.config = config
        self.optimizer_warmup_steps = 0
        self.lr_scheduler = None


def _resolve_loudly(config, name, total):
    import io
    from contextlib import redirect_stdout

    from core.training.base_trainer import resolve_lr_schedule_spec

    trainer = _WarnTrainer(config)
    out = io.StringIO()
    with redirect_stdout(out):
        trainer.lr_schedule_spec = resolve_lr_schedule_spec(trainer, name, total)
    return trainer, out.getvalue()


def test_a_polynomial_run_with_no_floor_key_is_told_its_floor_moved():
    """The one name whose absent-floor reading changed with D10."""
    _, log = _resolve_loudly({}, "polynomial", 1000)
    assert "lr_schedule_polynomial_floor_changed" in log
    _, explicit = _resolve_loudly({"lr_floor_ratio": 0.1}, "polynomial", 1000)
    assert "lr_schedule_polynomial_floor_changed" not in explicit
    _, other = _resolve_loudly({}, "cosine", 1000)
    assert "lr_schedule_polynomial_floor_changed" not in other


def _resume_with_a_raised_total(config, old_total, new_total):
    import io
    from contextlib import redirect_stdout

    from core.training.base_trainer import install_lr_schedule_events

    trainer, _ = _resolve_loudly(config, "wsd", new_total)
    trainer.lr_timeline = ScheduleTimeline()
    trainer.lr_timeline.set_total_steps(old_total)
    trainer._resume_lr_schedule_events = list(trainer.lr_timeline.events)
    trainer._resume_scheduler_step = old_total // 2
    trainer._resume_scheduler_interval = 1
    out = io.StringIO()
    with redirect_stdout(out):
        install_lr_schedule_events(trainer, old_total // 2)
    return out.getvalue()


def test_an_extension_over_an_explicit_decay_length_says_so():
    """§7.3's warning, which P1 and P2 could not fire: neither could create a
    real-axis length. lr_decay_steps can."""
    log = _resume_with_a_raised_total(
        {"lr_decay_start_step": 100, "lr_decay_steps": 200}, 1000, 2000)
    assert "lr_schedule_total_steps_changed" in log
    assert "lr_schedule_extension_on_floor" in log


def test_a_decay_that_runs_to_the_end_is_stretched_and_says_nothing_extra():
    log = _resume_with_a_raised_total({"lr_decay_start_step": 100}, 1000, 2000)
    assert "lr_schedule_total_steps_changed" in log
    assert "lr_schedule_extension_on_floor" not in log


# ---------------------------------------------------------------------------
# What a runtime command inherits from the config (§12.1)
# ---------------------------------------------------------------------------

def test_a_start_decay_command_takes_its_length_and_shape_from_the_config(tmp_path):
    """The POST body carries only `command`; both come from the run's spec."""
    import io
    from contextlib import redirect_stdout

    from core.training import training_control_rpc as control_rpc
    from core.training.base_trainer import poll_lr_schedule_commands
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lr_schedule_control_rpc_test import RUN_ID, FakeTrainer

    trainer = FakeTrainer(tmp_path, name="constant", T=1000,
                          config={"lr_decay_steps": 200,
                                  "lr_decay_shape": "linear",
                                  "lr_floor_ratio": 0.2})
    trainer.seek(400)
    control_rpc.queue_request(tmp_path, command="start_decay", run_id=RUN_ID)
    with redirect_stdout(io.StringIO()):
        assert poll_lr_schedule_commands(trainer, 400) == 1

    event = [e for e in trainer.lr_timeline.events if e.get("kind") == "decay"][0]
    assert event["length"] == 200 and event["shape"] == "linear"
    # Linear from 1.0 at 400 to the floor at 600, then held.
    assert trainer.multiplier(500) == pytest.approx(0.2 + 0.8 * 0.5)
    assert trainer.multiplier(600) == pytest.approx(0.2)
    assert trainer.multiplier(900) == pytest.approx(0.2)


def test_the_registry_shapes_are_the_documented_ones():
    """k(0) = 1, k(1) = 0 for each, and the midpoints of §8's table."""
    mid = {}
    for shape in DECAY_SHAPE_NAMES:
        fn = _lambda("wsd", 0, 1000, {"lr_decay_start_step": 500,
                                      "lr_decay_shape": shape})
        assert fn(500) == 1.0 and fn(1000) == 0.0, shape
        mid[shape] = fn(750)
    assert mid["cosine"] == pytest.approx(0.5)
    assert mid["linear"] == pytest.approx(0.5)
    assert mid["rex"] == pytest.approx(2 / 3)
    assert math.isclose(mid["rex"], 2 / 3, rel_tol=1e-9)
