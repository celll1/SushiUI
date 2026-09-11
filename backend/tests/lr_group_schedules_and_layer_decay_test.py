"""Gate: per-component LR schedules (D16) and layer-wise LR decay (D17).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/lr_group_schedules_and_layer_decay_test.py -v

Both features are OFF by default, which is the first thing checked here and the
strongest claim in the file: for every registry name, and for a config that
sets every numeric schedule key, the multiplier a default run gets is
BIT-IDENTICAL to the one the previous commit produced. The comparison loads
that commit's own ``lr_schedules.py`` out of git rather than restating its
formulas.

What else is checked:

* a per-component schedule actually diverges between two param groups, while
  one shared event list still reaches both (§17.3);
* both features refuse fused optimizer groups, which rebuild the optimizer from
  a flat parameter list and lose the component boundaries (§10.4/§11.4);
* a resume whose group structure changed refuses the state restore and takes
  the existing fresh-optimizer path instead of writing one depth's moments onto
  another's parameters (§17.3);
* the depth factors, and that the component name survives the ``.dNN`` the
  split appends to the group NAME (§17.3);
* the architectures that decline ``depth_blocks`` are the ones the capability
  table refuses ``lr_layer_decay`` for, and every other declared one implements
  it;
* §12.3's API, YAML, generator, and OpenAPI contracts for the two new keys.

CPU-only and hermetic: no model, no dataset, no GPU.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.arch_capabilities import (  # noqa: E402
    TRAINING_DECLARED_ARCHS,
    TRAINING_FEATURE_PARAMS,
    TRAINING_FEATURE_UNSUPPORTED,
)
from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from api.routes import (  # noqa: E402
    TrainingRunCreateRequest,
    _extract_request_params_from_yaml,
)
from core.training.arch import ARCH_REGISTRY  # noqa: E402
from core.training.arch.base_arch import ArchHandler  # noqa: E402
from core.training.base_trainer import (  # noqa: E402
    BaseTrainer,
    apply_layer_lr_decay,
    depth_split_structure_changed,
    lr_schedule_group_states,
    resolve_lr_group_specs,
)
from core.training.lr_schedules import (  # noqa: E402
    LR_SCHEDULER_NAMES,
    WARN_SELECTOR_ON_UNGROUPED_RUN,
    ScheduleTimeline,
    apply_layer_decay,
    build_depth_map,
    build_lr_scheduler,
    make_lambda,
    resolve_spec,
)
from core.training.training_config import (  # noqa: E402
    TrainingConfigGenerator,
    train_section_key_vocabulary,
)

# The two keys, and a value for each that is NOT the default.
NEW_KEYS = {
    "lr_layer_decay": 0.9,
    "lr_group_schedules": {"unet": "wsd", "text_encoder_1": "constant"},
}

# Every numeric schedule key set away from its default, so the equivalence
# below is not tested on a spec where most fields are inert.
BUSY_CONFIG = {
    "lr_floor_ratio": 0.15,
    "lr_decay_start_ratio": 0.7,
    "lr_decay_start_step": 240,
    "lr_decay_steps": 130,
    "lr_decay_shape": "rex",
    "lr_cycle_steps": 90,
    "lr_cycle_peak_decay": 0.85,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _blocks(n=6, width=3):
    return nn.ModuleList([nn.Linear(width, width) for _ in range(n)])


def _groups_from(blocks, extra=0, name="unet", lr=1e-4):
    """One optimizer group over every block parameter, plus ``extra`` loose ones
    that belong to no block (an embedder, a final layer)."""
    params = [p for block in blocks for p in block.parameters()]
    params += [nn.Parameter(torch.zeros(2)) for _ in range(extra)]
    return [{"params": params, "lr": lr, "name": name, "component": name}]


class _Probe:
    """A trainer stand-in for the module-level helpers under test."""

    _build_component_lr_list = BaseTrainer._build_component_lr_list
    _record_configured_group_lrs = BaseTrainer._record_configured_group_lrs
    _name_configured_groups = BaseTrainer._name_configured_groups
    _per_group_lr_metric_labels = BaseTrainer._per_group_lr_metric_labels
    _load_one_optimizer_state = BaseTrainer._load_one_optimizer_state
    _remap_optimizer_state_by_group_prefix = (
        BaseTrainer._remap_optimizer_state_by_group_prefix)
    _optimizer_state_entry_fits_param = staticmethod(
        BaseTrainer._optimizer_state_entry_fits_param)
    _lr_metric_labels = None
    _configured_group_lrs = None
    _configured_group_names = None
    lr_group_specs = None

    def __init__(self, config=None, blocks=None, **kwargs):
        self.log_prefix = "[test]"
        self.device = torch.device("cpu")
        self.learning_rate = 1e-4
        self.unet_lr = 1e-4
        self.text_encoder_lr = 1e-4
        self.text_encoder_1_lr = 1e-4
        self.text_encoder_2_lr = 1e-4
        self.unet = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.controlnet = None
        self.vision_encoder = None
        self.is_sdxl = False
        self.is_sensenova = False
        self.train_unet = True
        self.train_text_encoder = False
        self._train_vision_encoder = False
        self._grad_accum_steps = 1
        self.optimizer_warmup_steps = 0
        self.blocks_to_swap = 0
        self.num_optimizer_groups = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_schedulers = []
        self.fused_optimizer_groups = None
        self.config = dict(config or {})
        self.arch = SimpleNamespace(depth_blocks=lambda trainer: blocks)
        self.__dict__.update(kwargs)


def _quiet(fn, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


def _train_section(**overrides) -> dict:
    params = {
        "learning_rate": 1e-4, "batch_size": 1, "optimizer": "adamw8bit",
        "train_unet": True, "train_text_encoder": False, "total_steps": 1000,
        **overrides,
    }
    config = TrainingConfigGenerator().generate_lora_config(
        params,
        run_name="lr_p56_test",
        base_model_path="/models/does-not-exist.safetensors",
        output_dir="/tmp/lr_p56_test",
        dataset_configs=[{"dataset_id": 1, "path": "/data/ds"}],
        sample_prompts=[],
    )
    return yaml.safe_load(config)["config"]["process"][0]["train"]


# ---------------------------------------------------------------------------
# Default off == the previous commit, bit for bit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def previous_lr_schedules():
    """``lr_schedules.py`` as of the commit before this change, imported.

    Loaded out of git rather than re-derived, so "unchanged" means the same
    floating-point result from the same source, not the same intent.
    """
    source = subprocess.run(
        ["git", "show", "HEAD:backend/core/training/lr_schedules.py"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    path = Path(REPO / "backend" / "tests" / "_previous_lr_schedules_tmp.py")
    path.write_text(source, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(
            "_previous_lr_schedules", path)
        module = importlib.util.module_from_spec(spec)
        # @dataclass resolves its annotations through sys.modules[__module__].
        sys.modules["_previous_lr_schedules"] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("_previous_lr_schedules", None)
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("name", LR_SCHEDULER_NAMES + ("relora",))
@pytest.mark.parametrize("gas", [1, 4])
def test_a_default_run_is_bit_identical_to_the_previous_commit(
        previous_lr_schedules, name, gas):
    """Neither feature is on unless asked for, so no curve moves."""
    now = resolve_spec(BUSY_CONFIG, warmup_steps=20, total_steps=500, name=name,
                       advance_interval=gas)
    before = previous_lr_schedules.resolve_spec(
        BUSY_CONFIG, warmup_steps=20, total_steps=500, name=name,
        advance_interval=gas)
    # Two classes, so compare the fields rather than the dataclasses. R2's
    # group identity is unset here and is what makes it inert, so it is
    # asserted rather than compared -- and dropped from BOTH sides, because
    # HEAD moves and the field is only absent from a build that predates it.
    fields_now = dataclasses.asdict(now)
    fields_before = dataclasses.asdict(before)
    assert fields_now.pop("group") is None
    fields_before.pop("group", None)
    assert fields_now == fields_before

    timeline, old_timeline = ScheduleTimeline(), previous_lr_schedules.ScheduleTimeline()
    timeline.set_total_steps(now.total_steps)
    old_timeline.set_total_steps(before.total_steps)
    if name == "relora":
        for at in (120, 260):
            timeline.add("restart", at=at)
            old_timeline.add("restart", at=at)

    fn = make_lambda(now, timeline)
    old_fn = previous_lr_schedules.make_lambda(before, old_timeline)
    for step in range(0, 620):
        assert fn(step) == old_fn(step), (name, gas, step)


def test_a_default_run_still_gets_one_lambda_per_group():
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="cosine")
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    optimizer = torch.optim.AdamW(
        [{"params": [nn.Parameter(torch.zeros(2))]} for _ in range(3)], lr=1e-4)
    scheduler = build_lr_scheduler(optimizer, spec, timeline)
    assert len(scheduler.lr_lambdas) == 3
    assert len({fn(50) for fn in scheduler.lr_lambdas}) == 1


def test_the_defaults_are_off():
    assert TRAINING_DEFAULTS["lr_layer_decay"] == 1.0
    assert TRAINING_DEFAULTS["lr_group_schedules"] is None


def test_a_factor_of_one_returns_the_groups_untouched():
    blocks = _blocks()
    groups = _groups_from(blocks)
    depth_of, n = build_depth_map(blocks)
    out = apply_layer_decay(groups, depth_of, n, 1.0)
    assert [g["lr"] for g in out] == [g["lr"] for g in groups]
    assert [g["name"] for g in out] == ["unet"]


# ---------------------------------------------------------------------------
# D16: a per-component schedule diverges
# ---------------------------------------------------------------------------

def _two_group_probe(mapping, run_schedule="cosine"):
    probe = _Probe(config={"lr_group_schedules": mapping,
                           "lr_floor_ratio": 0.0})
    probe.optimizer = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4,
         "name": "unet", "component": "unet"},
        {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-5,
         "name": "text_encoder_1", "component": "text_encoder_1"},
    ])
    spec = resolve_spec(probe.config, warmup_steps=0, total_steps=100,
                        name=run_schedule)
    specs, _ = _quiet(resolve_lr_group_specs, probe, spec, 100)
    probe.lr_group_specs = specs
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    scheduler = build_lr_scheduler(probe.optimizer, spec, timeline,
                                   group_specs=specs)
    probe.lr_scheduler = scheduler
    return probe, spec, timeline, scheduler


def test_a_per_component_schedule_diverges_between_groups():
    probe, spec, timeline, scheduler = _two_group_probe(
        {"text_encoder_1": "constant"})
    assert [s.name for s in probe.lr_group_specs] == ["cosine", "constant"]
    unet, te = scheduler.lr_lambdas
    assert unet(0) == te(0) == 1.0
    assert unet(50) == pytest.approx(0.5)
    assert te(50) == 1.0, "the text encoder holds its rate while the DiT decays"
    assert unet(100) == 0.0 and te(100) == 1.0


def test_the_group_that_is_not_named_keeps_the_run_s_schedule():
    probe, spec, timeline, scheduler = _two_group_probe({"unet": "linear"})
    assert [s.name for s in probe.lr_group_specs] == ["linear", "cosine"]
    unet, te = scheduler.lr_lambdas
    assert unet(50) == pytest.approx(0.5)
    assert te(50) == pytest.approx(0.5)   # cosine's midpoint coincides here
    assert unet(25) == pytest.approx(0.75)
    assert te(25) != pytest.approx(0.75)


def test_one_event_list_still_reaches_every_group():
    """§17.3: the timeline is shared, the STATE is per spec. A decay command
    reaches the constant group too."""
    probe, spec, timeline, scheduler = _two_group_probe(
        {"text_encoder_1": "constant"})
    unet, te = scheduler.lr_lambdas
    assert te(60) == 1.0
    timeline.add("decay", at=50, spec=probe.lr_group_specs[1])
    assert te(50) == 1.0
    assert te(75) == pytest.approx(0.5)
    assert te(100) == pytest.approx(0.0)
    # And the group already decaying on its own curve is not disturbed.
    assert unet(50) == pytest.approx(0.5)


def test_the_group_states_report_each_group_s_own_schedule():
    probe, spec, timeline, scheduler = _two_group_probe(
        {"text_encoder_1": "constant"})
    states = lr_schedule_group_states(probe, spec, timeline, 50)
    assert [s["name"] for s in states] == ["unet", "text_encoder_1"]
    assert [s["schedule"] for s in states] == ["cosine", "constant"]
    assert states[0]["multiplier"] == pytest.approx(0.5)
    assert states[1]["multiplier"] == 1.0


def test_an_alias_is_re_resolved_rather_than_name_substituted():
    """§17.3: `rex` fixes its own start axis; substituting the name into the
    run's spec would leave `cosine`'s."""
    probe, spec, timeline, scheduler = _two_group_probe({"text_encoder_1": "rex"})
    rex = probe.lr_group_specs[1]
    assert (rex.curve, rex.decay_shape, rex.decay_start_axis) == (
        "wsd", "rex", "real")
    assert scheduler.lr_lambdas[1](50) == pytest.approx(2 / 3, rel=1e-9)


def test_a_component_no_group_carries_is_reported_not_applied():
    probe, _ = _quiet(lambda: None), None
    probe = _Probe(config={"lr_group_schedules": {"text_encoder_2": "linear"}})
    probe.optimizer = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4,
         "name": "unet", "component": "unet"}])
    spec = resolve_spec(probe.config, warmup_steps=0, total_steps=100,
                        name="cosine")
    specs, output = _quiet(resolve_lr_group_specs, probe, spec, 100)
    assert specs == [spec.for_group("unet")]
    assert "text_encoder_2" in output and "lr_group_schedules" in output


def test_unnamed_groups_are_refused_the_mapping_not_guessed_at():
    """§10.3: never quietly apply a component's schedule to another group."""
    probe = _Probe(config={"lr_group_schedules": {"unet": "linear"}})
    probe.optimizer = torch.optim.AdamW(
        [{"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4},
         {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-5}])
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="cosine")
    specs, output = _quiet(resolve_lr_group_specs, probe, spec, 100)
    assert specs is None
    assert "lr_group_schedules_unnamed_groups" in output


def test_an_unknown_schedule_name_is_refused_at_setup():
    probe = _Probe(config={"lr_group_schedules": {"unet": "piecewise_constant"}})
    probe.optimizer = torch.optim.AdamW(
        [{"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4,
          "name": "unet", "component": "unet"}])
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="cosine")
    with pytest.raises(ValueError, match="piecewise_constant"):
        resolve_lr_group_specs(probe, spec, 100)


def test_relora_ignores_the_mapping_and_says_so():
    probe = _Probe(config={"lr_group_schedules": {"unet": "linear"}})
    probe.optimizer = torch.optim.AdamW(
        [{"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4,
          "name": "unet", "component": "unet"}])
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="relora")
    specs, output = _quiet(resolve_lr_group_specs, probe, spec, 100)
    assert specs is None
    assert "ReLoRA" in output


# ---------------------------------------------------------------------------
# R2: a `groups`-scoped retarget through the lambdas the trainer builds
# ---------------------------------------------------------------------------

def test_the_resolved_group_specs_carry_their_component_name():
    """D24's identity comes from the optimizer group's `component`, which is
    what `lr_group_schedules` itself resolves against."""
    probe, spec, timeline, scheduler = _two_group_probe(
        {"text_encoder_1": "constant"})
    assert [s.group for s in probe.lr_group_specs] == ["unet", "text_encoder_1"]


def test_a_scoped_retarget_reaches_only_that_component_s_lambda():
    # `linear`, not `constant`: a group held at 1.0 would read the same whether
    # the retarget skipped it or replaced it with a curve anchored at 1.0.
    probe, spec, timeline, scheduler = _two_group_probe(
        {"text_encoder_1": "linear"})
    unet, te = scheduler.lr_lambdas
    before = [te(step) for step in range(0, 101, 5)]

    assert timeline.add("retarget", at=50, new_spec=resolve_spec(
        {}, warmup_steps=0, total_steps=100, name="constant"),
        length=0, groups=["unet"]) == "applied"
    # anchor=restart: the DiT holds the rate it had at 50, cosine(50) = 0.5.
    assert unet(75) == pytest.approx(0.5)
    assert [te(step) for step in range(0, 101, 5)] == before


def test_with_group_schedules_off_a_scoped_retarget_reaches_every_lambda(capsys):
    """§19.3's last row end to end: no mapping means no identities, so the
    selector applies to all rather than being refused -- and D37 says so at
    acceptance time."""
    probe = _Probe(config={"lr_floor_ratio": 0.0})
    probe.optimizer = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4,
         "name": "unet", "component": "unet"},
        {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-5,
         "name": "text_encoder_1", "component": "text_encoder_1"},
    ])
    spec = resolve_spec(probe.config, warmup_steps=0, total_steps=100,
                        name="cosine")
    specs = resolve_lr_group_specs(probe, spec, 100)
    assert specs is None
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    scheduler = build_lr_scheduler(probe.optimizer, spec, timeline,
                                   group_specs=specs)
    assert timeline.grouped_specs is False
    timeline.add("retarget", at=50, new_spec=resolve_spec(
        {}, warmup_steps=0, total_steps=100, name="constant"),
        length=0, groups=["unet"])
    for lambda_ in scheduler.lr_lambdas:
        assert lambda_(75) == pytest.approx(lambda_(50))
    assert WARN_SELECTOR_ON_UNGROUPED_RUN in capsys.readouterr().out


def test_the_build_tells_the_timeline_the_groups_carry_identities(capsys):
    """D37's condition comes from what `build_lr_scheduler` installed, not
    from the event: a run WITH per-component schedules takes a selector
    silently."""
    probe, spec, timeline, scheduler = _two_group_probe(
        {"text_encoder_1": "linear"})
    assert timeline.grouped_specs is True
    timeline.add("retarget", at=50, new_spec=resolve_spec(
        {}, warmup_steps=0, total_steps=100, name="constant"),
        length=0, groups=["unet"])
    assert WARN_SELECTOR_ON_UNGROUPED_RUN not in capsys.readouterr().out


def test_a_depth_split_group_keeps_its_component_s_identity():
    """§17.3: LLRD puts the depth in the group NAME and keeps `component`, so
    a `groups: ["unet"]` retarget still reaches every depth of the DiT."""
    probe = _Probe(config={"lr_group_schedules": {"text_encoder_1": "linear"},
                           "lr_floor_ratio": 0.0})
    probe.optimizer = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4,
         "name": "unet.d00", "component": "unet"},
        {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4,
         "name": "unet.d01", "component": "unet"},
        {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-5,
         "name": "text_encoder_1", "component": "text_encoder_1"},
    ])
    spec = resolve_spec(probe.config, warmup_steps=0, total_steps=100,
                        name="cosine")
    specs, _ = _quiet(resolve_lr_group_specs, probe, spec, 100)
    assert [s.group for s in specs] == ["unet", "unet", "text_encoder_1"]

    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    scheduler = build_lr_scheduler(probe.optimizer, spec, timeline,
                                   group_specs=specs)
    timeline.add("retarget", at=50, new_spec=resolve_spec(
        {}, warmup_steps=0, total_steps=100, name="constant"),
        length=0, groups=["unet"])
    d00, d01, te = scheduler.lr_lambdas
    assert d00(75) == d01(75) == pytest.approx(0.5)
    assert te(75) == pytest.approx(0.25)   # still its own linear curve


def test_the_group_states_report_a_scoped_retarget_only_where_it_landed():
    probe, spec, timeline, scheduler = _two_group_probe(
        {"text_encoder_1": "constant"})
    timeline.add("retarget", at=25, new_spec=resolve_spec(
        {}, warmup_steps=0, total_steps=100, name="linear"),
        length=0, groups=["text_encoder_1"])
    states = lr_schedule_group_states(probe, spec, timeline, 50)
    assert [s["schedule"] for s in states] == ["cosine", "linear"]


# ---------------------------------------------------------------------------
# D40: the warning says why the groups share one schedule
# ---------------------------------------------------------------------------

def _one_group_probe(mapping, name="unet", component="unet", schedule="cosine"):
    config = {"lr_floor_ratio": 0.0}
    if mapping is not None:
        config["lr_group_schedules"] = mapping
    probe = _Probe(config=config)
    group = {"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4}
    if name:
        group["name"] = name
        group["component"] = component
    probe.optimizer = torch.optim.AdamW([group])
    spec = resolve_spec(probe.config, warmup_steps=0, total_steps=100,
                        name=schedule)
    specs, _ = _quiet(resolve_lr_group_specs, probe, spec, 100)
    return probe, spec, specs


@pytest.mark.parametrize("mapping,name,schedule,expected", [
    (None, "unet", "cosine", "lr_group_schedules is not set"),
    ({"unet": "linear"}, "unet", "relora", "ignored on a ReLoRA run"),
    ({"unet": "linear"}, None, "cosine", "carry no component name"),
])
def test_every_ungrouped_path_records_why(mapping, name, schedule, expected):
    probe, _, specs = _one_group_probe(mapping, name=name, schedule=schedule)
    assert specs is None
    assert expected in probe.lr_group_specs_ignored_reason


def test_the_warning_gives_the_reason_instead_of_a_missing_key(capsys):
    """Two of the three paths reach here with `lr_group_schedules` SET; telling
    that operator the key is missing sends them after a bug that is not there."""
    probe, spec, specs = _one_group_probe({"unet": "linear"}, name=None)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    build_lr_scheduler(probe.optimizer, spec, timeline, group_specs=specs,
                       ungrouped_reason=probe.lr_group_specs_ignored_reason)
    timeline.add("retarget", at=50, length=0, groups=["unet"],
                 new_spec=resolve_spec({}, warmup_steps=0, total_steps=100,
                                       name="constant"))
    out = capsys.readouterr().out
    assert "carry no component name" in out
    assert "has no lr_group_schedules" not in out
    assert "share one LR schedule" in out


def test_a_run_that_never_set_the_key_still_says_so(capsys):
    probe, spec, specs = _one_group_probe(None)
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    build_lr_scheduler(probe.optimizer, spec, timeline, group_specs=specs,
                       ungrouped_reason=probe.lr_group_specs_ignored_reason)
    timeline.add("retarget", at=50, length=0, groups=["unet"],
                 new_spec=resolve_spec({}, warmup_steps=0, total_steps=100,
                                       name="constant"))
    assert "lr_group_schedules is not set" in capsys.readouterr().out


def test_a_spec_list_of_the_wrong_length_is_refused():
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="cosine")
    timeline = ScheduleTimeline()
    timeline.set_total_steps(spec.total_steps)
    optimizer = torch.optim.AdamW(
        [{"params": [nn.Parameter(torch.zeros(2))]} for _ in range(3)], lr=1e-4)
    with pytest.raises(ValueError, match="BY INDEX"):
        build_lr_scheduler(optimizer, spec, timeline, group_specs=[spec, spec])


# ---------------------------------------------------------------------------
# D17: the depth factors
# ---------------------------------------------------------------------------

def test_the_factors_are_geometric_in_depth():
    blocks = _blocks(n=5)
    groups = _groups_from(blocks, lr=1e-4)
    depth_of, n = build_depth_map(blocks)
    assert n == 5
    out = apply_layer_decay(groups, depth_of, n, 0.5)
    assert [g["lr"] for g in out] == [
        pytest.approx(1e-4 * 0.5 ** 4), pytest.approx(1e-4 * 0.5 ** 3),
        pytest.approx(1e-4 * 0.5 ** 2), pytest.approx(1e-4 * 0.5),
        pytest.approx(1e-4)]


def test_the_component_survives_the_depth_suffix():
    """§17.3: `.dNN` goes on the NAME; lr_group_schedules maps `component`."""
    blocks = _blocks(n=3)
    depth_of, n = build_depth_map(blocks)
    out = apply_layer_decay(_groups_from(blocks), depth_of, n, 0.8)
    assert [g["name"] for g in out] == ["unet.d00", "unet.d01", "unet.d02"]
    assert {g["component"] for g in out} == {"unet"}


def test_every_trainable_parameter_lands_in_exactly_one_split():
    blocks = _blocks(n=4)
    groups = _groups_from(blocks, extra=3)
    depth_of, n = build_depth_map(blocks)
    out = apply_layer_decay(groups, depth_of, n, 0.7)
    before = [id(p) for p in groups[0]["params"]]
    after = [id(p) for g in out for p in g["params"]]
    assert sorted(after) == sorted(before)
    assert len(after) == len(set(after))


def test_a_parameter_in_no_block_keeps_the_group_s_own_rate():
    blocks = _blocks(n=4)
    groups = _groups_from(blocks, extra=2, lr=3e-4)
    depth_of, n = build_depth_map(blocks)
    out = apply_layer_decay(groups, depth_of, n, 0.5)
    deepest = out[-1]
    assert deepest["lr"] == pytest.approx(3e-4)
    loose = [p for p in groups[0]["params"] if id(p) not in depth_of]
    assert all(any(p is q for q in deepest["params"]) for p in loose)


def test_parallel_stacks_share_a_depth():
    """Ideogram 4's cond/uncond copies: layer j is depth j in both."""
    cond, uncond = _blocks(n=3), _blocks(n=3)
    entries = [[cond[j], uncond[j]] for j in range(3)]
    depth_of, n = build_depth_map(entries)
    assert n == 3
    for j in range(3):
        for module in (cond[j], uncond[j]):
            assert {depth_of[id(p)] for p in module.parameters()} == {j}


def test_the_split_groups_reach_the_optimizer_in_order():
    blocks = _blocks(n=4)
    probe = _Probe(config={"lr_layer_decay": 0.5}, blocks=blocks)
    groups, output = _quiet(apply_layer_lr_decay, probe, _groups_from(blocks))
    assert "Layer-wise LR decay 0.5" in output
    optimizer = torch.optim.AdamW(groups)
    assert [g["name"] for g in optimizer.param_groups] == [
        "unet.d00", "unet.d01", "unet.d02", "unet.d03"]
    probe.optimizer = optimizer
    probe._record_configured_group_lrs(None)
    assert probe._configured_group_names == [
        "unet.d00", "unet.d01", "unet.d02", "unet.d03"]


def test_only_the_deepest_split_carries_the_component_s_metric_series():
    """§11.4: 30 depths must not replace `lr_unet` with `lr_unetd00`.."""
    blocks = _blocks(n=4)
    probe = _Probe(config={"lr_layer_decay": 0.5}, blocks=blocks)
    groups, _ = _quiet(apply_layer_lr_decay, probe, _groups_from(blocks))
    probe.optimizer = torch.optim.AdamW(groups)
    probe._record_configured_group_lrs(None)
    assert probe._per_group_lr_metric_labels() == [None, None, None, "unet"]


def test_an_off_run_s_metric_labels_are_what_they_always_were():
    probe = _Probe()
    probe.unet = nn.Linear(2, 2)
    probe.text_encoder = nn.Linear(2, 2)
    probe.train_text_encoder = True
    probe.optimizer = torch.optim.AdamW(
        [{"params": list(probe.unet.parameters()), "lr": 1e-4},
         {"params": list(probe.text_encoder.parameters()), "lr": 1e-5}])
    probe._record_configured_group_lrs(None)
    assert probe._per_group_lr_metric_labels() == ["unet", "te1"]


def test_an_out_of_range_factor_is_refused():
    probe = _Probe(config={"lr_layer_decay": 1.5}, blocks=_blocks())
    with pytest.raises(ValueError, match="lr_layer_decay"):
        apply_layer_lr_decay(probe, _groups_from(_blocks()))


def test_an_architecture_without_blocks_refuses_rather_than_no_ops():
    probe = _Probe(config={"lr_layer_decay": 0.9}, blocks=None)
    with pytest.raises(ValueError, match="no ordered block stack"):
        apply_layer_lr_decay(probe, _groups_from(_blocks()))


# ---------------------------------------------------------------------------
# Fused optimizer groups: refused, not worked around
# ---------------------------------------------------------------------------

def test_layer_decay_refuses_fused_optimizer_groups():
    blocks = _blocks()
    probe = _Probe(config={"lr_layer_decay": 0.9}, blocks=blocks,
                   blocks_to_swap=8, num_optimizer_groups=4)
    with pytest.raises(ValueError, match="num_optimizer_groups"):
        apply_layer_lr_decay(probe, _groups_from(blocks))


def test_group_schedules_refuse_fused_optimizer_groups():
    probe = _Probe(config={"lr_group_schedules": {"unet": "linear"}},
                   blocks_to_swap=8, num_optimizer_groups=4)
    probe.optimizer = torch.optim.AdamW(
        [{"params": [nn.Parameter(torch.zeros(2))], "lr": 1e-4,
          "name": "unet", "component": "unet"}])
    spec = resolve_spec({}, warmup_steps=0, total_steps=100, name="cosine")
    with pytest.raises(ValueError, match="num_optimizer_groups"):
        resolve_lr_group_specs(probe, spec, 100)


def test_fused_groups_without_block_swap_do_not_refuse():
    """`num_optimizer_groups` is only read inside setup_optimizer's
    `blocks_to_swap > 0` arm, so on its own it rebuilds nothing."""
    blocks = _blocks()
    probe = _Probe(config={"lr_layer_decay": 0.5}, blocks=blocks,
                   blocks_to_swap=0, num_optimizer_groups=4)
    groups, _ = _quiet(apply_layer_lr_decay, probe, _groups_from(blocks))
    assert len(groups) == len(blocks)


# ---------------------------------------------------------------------------
# §17.3: a changed group structure takes the reset path
# ---------------------------------------------------------------------------

def _split_optimizer(n_blocks=3, factor=0.5, name="unet"):
    blocks = _blocks(n=n_blocks)
    depth_of, n = build_depth_map(blocks)
    groups = apply_layer_decay(_groups_from(blocks, name=name), depth_of, n, factor)
    return torch.optim.AdamW(groups)


def _with_moments(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = torch.ones_like(p)
    optimizer.step()
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = None
    return optimizer


def test_the_same_split_structure_still_restores():
    saved = _with_moments(_split_optimizer()).state_dict()
    live = _split_optimizer()
    assert depth_split_structure_changed(live, saved) is None
    probe = _Probe()
    probe.optimizer = live
    ok, _ = _quiet(probe._load_one_optimizer_state, live, saved, "ckpt.pt")
    assert ok is True
    assert any(live.state.get(p, {}).get("exp_avg") is not None
               for g in live.param_groups for p in g["params"])


@pytest.mark.parametrize("live_blocks,live_factor", [(4, 0.5), (3, 0.5)])
def test_a_changed_split_structure_refuses_and_starts_fresh(live_blocks,
                                                            live_factor):
    """Depth changed, or the split was turned off: restoring by index would
    give one depth's moments to another's parameters."""
    saved = _with_moments(_split_optimizer(n_blocks=3)).state_dict()
    live = (_split_optimizer(n_blocks=live_blocks, factor=live_factor)
            if live_blocks != 3 else
            torch.optim.AdamW(_groups_from(_blocks(n=3))))
    assert depth_split_structure_changed(live, saved) is not None
    probe = _Probe()
    probe.optimizer = live
    ok, output = _quiet(probe._load_one_optimizer_state, live, saved, "ckpt.pt")
    assert ok is False, "the reset path, not a by-index restore"
    assert "fresh optimizer state" in output
    assert all(live.state.get(p, {}).get("exp_avg") is None
               for g in live.param_groups for p in g["params"])


def test_a_run_without_layer_decay_resumes_exactly_as_before():
    """The guard claims nothing where no depth split is present, including when
    the group count legitimately changed (a REPA projector appended)."""
    saved = _with_moments(torch.optim.AdamW(
        [{"params": [nn.Parameter(torch.zeros(4))], "lr": 1e-4,
          "name": "unet"}])).state_dict()
    live = torch.optim.AdamW([
        {"params": [nn.Parameter(torch.zeros(4))], "lr": 1e-4, "name": "unet"},
        {"params": [nn.Parameter(torch.zeros(4))], "lr": 1e-4,
         "name": "repa_projector"}])
    assert depth_split_structure_changed(live, saved) is None


# ---------------------------------------------------------------------------
# Which architectures expose a depth axis
# ---------------------------------------------------------------------------

_DECLINES_DEPTH = {"sd15", "sdxl"}


@pytest.mark.parametrize("arch", sorted(TRAINING_DECLARED_ARCHS))
def test_depth_blocks_and_the_capability_table_say_the_same_thing(arch):
    handler = ARCH_REGISTRY[arch]
    declines = handler.depth_blocks is ArchHandler.depth_blocks
    refused = "lr_layer_decay" in TRAINING_FEATURE_UNSUPPORTED.get(arch, {})
    assert declines == refused, (
        f"{arch}: depth_blocks {'declines' if declines else 'is implemented'} "
        f"but the capability table {'refuses' if refused else 'offers'} "
        f"lr_layer_decay")
    assert declines == (arch in _DECLINES_DEPTH)


def test_the_refusal_names_the_structural_reason():
    reason = TRAINING_FEATURE_UNSUPPORTED["sd15"]["lr_layer_decay"]["reason"]
    assert "skip connection" in reason and "depth_blocks" in reason


def test_the_feature_arms_its_own_key():
    assert TRAINING_FEATURE_PARAMS["lr_layer_decay"] == ["lr_layer_decay"]


def test_no_declared_arch_returns_a_single_block():
    """A one-entry stack would make every factor 1.0 -- silently inert. The
    refusal path (`n_depths < 2`) covers it, and this pins that no handler
    returns one from a real model by construction."""
    probe = _Probe(config={"lr_layer_decay": 0.5}, blocks=_blocks(n=1))
    with pytest.raises(ValueError, match="no ordered block stack"):
        apply_layer_lr_decay(probe, _groups_from(_blocks(n=1)))


# ---------------------------------------------------------------------------
# §12.3's checklist, per key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(NEW_KEYS))
def test_the_request_model_declares_it_with_the_shared_default(key):
    field = TrainingRunCreateRequest.model_fields[key]
    assert field.get_default(call_default_factory=True) == TRAINING_DEFAULTS[key]


@pytest.mark.parametrize("key,value", sorted(NEW_KEYS.items()))
def test_it_round_trips_request_to_yaml_to_request(key, value):
    train = _train_section(**{key: value})
    assert train[key] == value
    back = _extract_request_params_from_yaml({"train": train}, job="lora")
    assert back[key] == value
    restored = TrainingRunCreateRequest(
        training_method="lora", base_model_path="x",
        **{k: v for k, v in back.items()
           if k in TrainingRunCreateRequest.model_fields
           and k not in ("training_method", "base_model_path")})
    assert getattr(restored, key) == value


@pytest.mark.parametrize("name,expected", [
    ("cosine", 0.0), ("linear", 0.0),
    ("plateau_cosine_floor", TRAINING_DEFAULTS["lr_floor_ratio"]),
])
@pytest.mark.parametrize("floor", [None, 0.0, 0.4])
def test_editing_legacy_run_preserves_floor(name, expected, floor):
    train = {"lr_scheduler": name, "steps": 1000}
    if floor is not None:
        train["lr_floor_ratio"] = floor
        expected = floor
    params = _extract_request_params_from_yaml({"train": train}, job="lora")
    params["total_steps"] = 2000
    rewritten = _train_section(**params)
    assert rewritten["lr_floor_ratio"] == expected
    spec = resolve_spec(rewritten, warmup_steps=0, total_steps=2000, name=name)
    assert spec.floor_ratio == expected


def test_new_run_keeps_new_floor_default():
    assert _train_section(lr_scheduler="cosine")["lr_floor_ratio"] == TRAINING_DEFAULTS["lr_floor_ratio"]


@pytest.mark.parametrize("floor", [None, 0.4])
def test_relora_edit_uses_its_effective_schedule_for_legacy_floor(floor):
    train = {"lr_scheduler": "plateau_cosine_floor", "steps": 1000}
    if floor is not None:
        train["lr_floor_ratio"] = floor
    params = _extract_request_params_from_yaml({"train": train}, job="relora")
    rewritten = _train_section(**params)
    spec = resolve_spec(rewritten, warmup_steps=0, total_steps=2000, name="relora")
    assert spec.floor_ratio == (0.0 if floor is None else floor)


def test_group_plateau_retains_start_ratio_through_edit():
    train = _train_section(lr_scheduler="cosine", lr_decay_start_ratio=0.5,
                           lr_group_schedules={"unet": "plateau_cosine_floor"})
    params = _extract_request_params_from_yaml({"train": train}, job="lora")
    rewritten = _train_section(**params)
    assert rewritten["lr_decay_start_ratio"] == 0.5
    spec = resolve_spec(rewritten, warmup_steps=0, total_steps=1000,
                        name=rewritten["lr_group_schedules"]["unet"])
    assert spec.decay_start_step == 500


def test_the_layer_decay_is_written_unconditionally():
    for scheduler in ("constant", "cosine", "wsd"):
        train = _train_section(lr_scheduler=scheduler)
        assert train["lr_layer_decay"] == TRAINING_DEFAULTS["lr_layer_decay"]


def test_the_group_mapping_is_absent_when_it_is_off():
    """null and absent mean the same thing, so an off run writes no key."""
    assert "lr_group_schedules" not in _train_section()
    assert "lr_group_schedules" in _train_section(
        lr_group_schedules={"unet": "wsd"})


@pytest.mark.parametrize("key", sorted(NEW_KEYS))
def test_the_generator_can_emit_it_by_name(key):
    """A key written through a loop is invisible to
    `train_section_key_vocabulary`, which `preserve_unmodelled_train_keys`
    reads to tell a config-only key from a deliberately-off one."""
    assert key in train_section_key_vocabulary()


def test_the_openapi_schema_matches_the_defaults():
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    assert props["lr_layer_decay"]["default"] == TRAINING_DEFAULTS["lr_layer_decay"]
    assert props["lr_group_schedules"]["default"] is None
    assert props["lr_group_schedules"]["nullable"] is True
    assert (props["lr_group_schedules"]["additionalProperties"]["enum"]
            == list(LR_SCHEDULER_NAMES))


def test_the_request_validator_refuses_an_unknown_group_schedule():
    with pytest.raises(ValueError, match="piecewise_constant"):
        TrainingRunCreateRequest(
            training_method="lora", base_model_path="x",
            lr_group_schedules={"unet": "piecewise_constant"})


def test_the_request_validator_normalizes_and_empties():
    request = TrainingRunCreateRequest(
        training_method="lora", base_model_path="x",
        lr_group_schedules={"UNet": "WSD"})
    assert request.lr_group_schedules == {"unet": "wsd"}
    assert TrainingRunCreateRequest(
        training_method="lora", base_model_path="x",
        lr_group_schedules={}).lr_group_schedules is None


def test_the_request_validator_bounds_the_decay_factor():
    with pytest.raises(ValueError):
        TrainingRunCreateRequest(training_method="lora", base_model_path="x",
                                 lr_layer_decay=0.0)
    with pytest.raises(ValueError):
        TrainingRunCreateRequest(training_method="lora", base_model_path="x",
                                 lr_layer_decay=1.5)
