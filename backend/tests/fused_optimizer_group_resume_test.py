"""What a resume restores when ``num_optimizer_groups > 0``.

``_setup_fused_optimizer_groups`` builds N optimizers over a flat parameter list
and leaves ``self.optimizer`` pointing at ``optimizers[0]``. The shipped
``save_optimizer_state`` / ``load_optimizer_state`` / ``_reassert_config_lr_on_resume``
and the scheduler fast-forward all read ``self.optimizer`` (or
``self.lr_scheduler``) alone, so a resumed run kept 1/N of its optimizer state
and 1/N of its schedule position.

The NEGATIVE CONTROLS are ``test_negative_control_*``: they run the exact
shipped statements against the same fixture and report what was lost.

CPU only; no model is loaded.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/fused_optimizer_group_resume_test.py -v
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.base_trainer import (
    FUSED_GROUP_STATES_KEY,
    BaseTrainer,
    all_lr_schedulers,
    all_optimizers,
)
from core.training.optimizers.fused_optimizer_groups import (
    FusedOptimizerGroups,
    create_optimizer_groups,
)
from core.training.training_events import TRAINING_EVENT_SENTINEL

CKPT_LR = 9.876e-06
N_PARAMS = 6


class _Probe:
    save_optimizer_state = BaseTrainer.save_optimizer_state
    load_optimizer_state = BaseTrainer.load_optimizer_state
    _split_saved_optimizer_states = staticmethod(BaseTrainer._split_saved_optimizer_states)
    _repartition_optimizer_states = BaseTrainer._repartition_optimizer_states
    _load_one_optimizer_state = BaseTrainer._load_one_optimizer_state
    _optimizer_state_param_count = staticmethod(BaseTrainer._optimizer_state_param_count)
    _fast_forward_lr_schedulers = BaseTrainer._fast_forward_lr_schedulers
    _fast_forward_one_lr_scheduler = staticmethod(BaseTrainer._fast_forward_one_lr_scheduler)
    _build_component_lr_list = BaseTrainer._build_component_lr_list
    _record_configured_group_lrs = BaseTrainer._record_configured_group_lrs
    _name_configured_groups = BaseTrainer._name_configured_groups
    _configured_component_lr_description = BaseTrainer._configured_component_lr_description
    _reassert_config_lr_on_resume = BaseTrainer._reassert_config_lr_on_resume

    def __init__(self, tmp_path, learning_rate=1e-4):
        self.log_prefix = "[test]"
        self.device = torch.device("cpu")
        self.output_dir = Path(tmp_path)
        self.run_name = "20260101_000000_abcdef"
        self.learning_rate = learning_rate
        self.unet_lr = learning_rate
        self.text_encoder_lr = learning_rate
        self.text_encoder_1_lr = learning_rate
        self.text_encoder_2_lr = learning_rate
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
        self.config = {}
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_schedulers = []
        self.fused_optimizer_groups = None
        self.num_optimizer_groups = 0
        self.params = []


def _make_params(n=N_PARAMS, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.rand(4, generator=generator)) for _ in range(n)]


def _fused_probe(tmp_path, num_groups=3, n_params=N_PARAMS, optimizer_type="adamw",
                 lr_lambda=None, seed=0):
    """A trainer stand-in wired the way ``_setup_fused_optimizer_groups`` wires one."""
    probe = _Probe(tmp_path)
    probe.params = _make_params(n_params, seed=seed)
    probe.num_optimizer_groups = num_groups
    with redirect_stdout(io.StringIO()):
        optimizers = create_optimizer_groups(
            params=probe.params, optimizer_type=optimizer_type,
            num_groups=num_groups, learning_rate=probe.learning_rate)
    probe.optimizer = optimizers[0]
    if lr_lambda is not None:
        probe.lr_schedulers = [LambdaLR(o, lr_lambda=lr_lambda) for o in optimizers]
        probe.lr_scheduler = probe.lr_schedulers[0]
    probe.fused_optimizer_groups = FusedOptimizerGroups(optimizers=optimizers)
    probe._record_configured_group_lrs(None)
    return probe


def _single_probe(tmp_path, n_params=N_PARAMS, lr_lambda=None, seed=0):
    probe = _Probe(tmp_path)
    probe.params = _make_params(n_params, seed=seed)
    probe.optimizer = torch.optim.AdamW(
        [{"params": probe.params, "lr": probe.learning_rate}])
    if lr_lambda is not None:
        probe.lr_scheduler = LambdaLR(probe.optimizer, lr_lambda=lr_lambda)
    probe._record_configured_group_lrs(None)
    return probe


def _train(probe, steps=3, seed=1234):
    """Give every parameter a real gradient history, so moments are non-trivial."""
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for p in probe.params:
            p.grad = torch.rand(p.shape, generator=generator)
        for optimizer in all_optimizers(probe):
            optimizer.step()
    for p in probe.params:
        p.grad = None


def _moments(probe):
    """``exp_avg`` per parameter, in global parameter order (None = no state)."""
    out = []
    for optimizer in all_optimizers(probe):
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                out.append(state.get("exp_avg"))
    return out


def _moments_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None or y is None:
            if x is not y:
                return False
            continue
        if not torch.equal(x, y):
            return False
    return True


def _n_restored(probe):
    return sum(1 for m in _moments(probe) if m is not None)


def _group_lrs(probe):
    return [float(g["lr"]) for optimizer in all_optimizers(probe)
            for g in optimizer.param_groups]


def _simulate_checkpoint_lr(probe, lr=CKPT_LR):
    for optimizer in all_optimizers(probe):
        for group in optimizer.param_groups:
            group["lr"] = lr


def _quiet(fn, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


def _events(text):
    return [json.loads(line.split(TRAINING_EVENT_SENTINEL, 1)[1])
            for line in text.splitlines() if TRAINING_EVENT_SENTINEL in line]


# ---------------------------------------------------------------------------
# The shipped statements, verbatim, for the negative controls
# ---------------------------------------------------------------------------

def _shipped_save(probe, step):
    """HEAD~: ``self.optimizer`` alone."""
    opt_state = probe.optimizer.state_dict()
    opt_state["_sushi_opt_class"] = type(probe.optimizer).__name__
    torch.save(opt_state,
               probe.output_dir / f"{probe.run_name}_step_{step:06d}_optimizer.pt")


def _shipped_load(probe, step):
    state = torch.load(
        probe.output_dir / f"{probe.run_name}_step_{step:06d}_optimizer.pt",
        map_location="cpu", weights_only=False)
    probe.optimizer.load_state_dict(state)


def _shipped_fast_forward(probe, global_step):
    for _ in range(global_step):
        probe.lr_scheduler.step()


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL: the shipped resume loses N-1 optimizers' moments
# ---------------------------------------------------------------------------

def test_negative_control_shipped_resume_keeps_only_the_first_optimizers_moments(tmp_path):
    trained = _fused_probe(tmp_path, num_groups=3)
    _train(trained)
    before = _moments(trained)
    assert all(m is not None for m in before), "degenerate fixture: no moments"
    assert len(before) == N_PARAMS

    _shipped_save(trained, step=10)

    resumed = _fused_probe(tmp_path, num_groups=3)
    assert _n_restored(resumed) == 0
    _shipped_load(resumed, step=10)

    after = _moments(resumed)
    # BEFORE: 6 of 6 parameters carry moments. AFTER the shipped resume: 2 of 6
    # -- one group of ceil(6/3) -- and the other 4 restart from zero.
    assert _n_restored(trained) == 6
    assert _n_restored(resumed) == 2
    assert [m is not None for m in after] == [True, True, False, False, False, False]
    assert _moments_equal(after[:2], before[:2])
    assert not _moments_equal(after, before)


def test_fused_resume_restores_every_optimizer_group(tmp_path):
    trained = _fused_probe(tmp_path, num_groups=3)
    _train(trained)
    before = _moments(trained)
    _quiet(trained.save_optimizer_state, 10)

    resumed = _fused_probe(tmp_path, num_groups=3)
    ok, _ = _quiet(resumed.load_optimizer_state, 10)

    assert ok is True
    assert _n_restored(resumed) == 6
    assert _moments_equal(_moments(resumed), before)


@pytest.mark.parametrize("num_groups", [2, 3, 4, 6])
def test_every_group_count_round_trips(tmp_path, num_groups):
    trained = _fused_probe(tmp_path / f"g{num_groups}", num_groups=num_groups)
    (tmp_path / f"g{num_groups}").mkdir(exist_ok=True)
    _train(trained)
    before = _moments(trained)
    _quiet(trained.save_optimizer_state, 5)

    resumed = _fused_probe(tmp_path / f"g{num_groups}", num_groups=num_groups)
    ok, _ = _quiet(resumed.load_optimizer_state, 5)
    assert ok is True
    assert _moments_equal(_moments(resumed), before)


def test_the_saved_file_names_every_group(tmp_path):
    probe = _fused_probe(tmp_path, num_groups=3)
    _train(probe)
    _quiet(probe.save_optimizer_state, 7)
    payload = torch.load(tmp_path / f"{probe.run_name}_step_000007_optimizer.pt",
                         map_location="cpu", weights_only=False)
    assert isinstance(payload[FUSED_GROUP_STATES_KEY], list)
    assert len(payload[FUSED_GROUP_STATES_KEY]) == 3
    assert [BaseTrainer._optimizer_state_param_count(s)
            for s in payload[FUSED_GROUP_STATES_KEY]] == [2, 2, 2]


# ---------------------------------------------------------------------------
# Runs that do NOT use fused groups are untouched
# ---------------------------------------------------------------------------

def test_single_optimizer_file_format_is_unchanged(tmp_path):
    probe = _single_probe(tmp_path)
    _train(probe)
    _quiet(probe.save_optimizer_state, 3)
    payload = torch.load(tmp_path / f"{probe.run_name}_step_000003_optimizer.pt",
                         map_location="cpu", weights_only=False)
    assert FUSED_GROUP_STATES_KEY not in payload
    assert set(payload) == {"state", "param_groups", "_sushi_opt_class"}
    assert payload["_sushi_opt_class"] == "AdamW"


def test_single_optimizer_resume_is_unchanged(tmp_path):
    trained = _single_probe(tmp_path)
    _train(trained)
    before = _moments(trained)
    _quiet(trained.save_optimizer_state, 3)

    resumed = _single_probe(tmp_path)
    ok, _ = _quiet(resumed.load_optimizer_state, 3)
    assert ok is True
    assert _moments_equal(_moments(resumed), before)

    # ...and a file written by the shipped single-optimizer save loads identically.
    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    trained.output_dir = shipped_dir
    _shipped_save(trained, step=3)
    legacy = _single_probe(shipped_dir)
    ok, _ = _quiet(legacy.load_optimizer_state, 3)
    assert ok is True
    assert _moments_equal(_moments(legacy), before)


def test_missing_file_still_returns_false(tmp_path):
    probe = _single_probe(tmp_path)
    ok, text = _quiet(probe.load_optimizer_state, 99)
    assert ok is False
    assert "No optimizer state file found" in text


def test_repa_style_trailing_group_partial_load_still_works(tmp_path):
    """A group appended since the checkpoint: the prefix's moments survive."""
    probe = _single_probe(tmp_path)
    _train(probe)
    before = _moments(probe)
    _quiet(probe.save_optimizer_state, 4)

    wider = _Probe(tmp_path)
    wider.params = _make_params(N_PARAMS)
    projector = _make_params(1, seed=7)
    wider.optimizer = torch.optim.AdamW([
        {"params": wider.params, "lr": wider.learning_rate},
        {"params": projector, "lr": wider.learning_rate},
    ])
    ok, text = _quiet(wider.load_optimizer_state, 4)
    assert ok is True
    assert "Partial optimizer state load OK" in text
    assert _moments_equal(_moments(wider)[:N_PARAMS], before)


# ---------------------------------------------------------------------------
# Changing num_optimizer_groups between runs
# ---------------------------------------------------------------------------

def test_a_pre_fix_file_restores_group_zero_and_says_so(tmp_path):
    """Files written before the fix hold only optimizer 0; nothing can invent
    the rest, so the resume restores what exists and reports the shortfall."""
    trained = _fused_probe(tmp_path, num_groups=3)
    _train(trained)
    before = _moments(trained)
    _shipped_save(trained, step=10)

    resumed = _fused_probe(tmp_path, num_groups=3)
    ok, text = _quiet(resumed.load_optimizer_state, 10)
    assert ok is True
    assert _n_restored(resumed) == 2
    assert _moments_equal(_moments(resumed)[:2], before[:2])
    assert [e["code"] for e in _events(text)] == ["optimizer_state_partial_fused_resume"]


@pytest.mark.parametrize("saved,live", [(3, 2), (2, 3), (3, 6), (6, 3)])
def test_a_changed_group_count_reslices_every_moment(tmp_path, saved, live):
    directory = tmp_path / f"{saved}to{live}"
    directory.mkdir()
    trained = _fused_probe(directory, num_groups=saved)
    _train(trained)
    before = _moments(trained)
    _quiet(trained.save_optimizer_state, 8)

    resumed = _fused_probe(directory, num_groups=live)
    ok, _ = _quiet(resumed.load_optimizer_state, 8)
    assert ok is True
    assert _moments_equal(_moments(resumed), before)


def test_turning_fused_groups_off_keeps_every_moment(tmp_path):
    """``num_optimizer_groups: 3`` -> ``0``: one optimizer over the same params."""
    trained = _fused_probe(tmp_path, num_groups=3)
    _train(trained)
    before = _moments(trained)
    _quiet(trained.save_optimizer_state, 9)

    resumed = _single_probe(tmp_path)
    ok, _ = _quiet(resumed.load_optimizer_state, 9)
    assert ok is True
    assert _moments_equal(_moments(resumed), before)


def test_turning_fused_groups_on_keeps_every_moment(tmp_path):
    trained = _single_probe(tmp_path)
    _train(trained)
    before = _moments(trained)
    _quiet(trained.save_optimizer_state, 9)

    resumed = _fused_probe(tmp_path, num_groups=3)
    ok, _ = _quiet(resumed.load_optimizer_state, 9)
    assert ok is True
    assert _moments_equal(_moments(resumed), before)


def test_a_different_parameter_set_resets_and_says_so(tmp_path):
    trained = _fused_probe(tmp_path, num_groups=3, n_params=6)
    _train(trained)
    _quiet(trained.save_optimizer_state, 11)

    resumed = _fused_probe(tmp_path, num_groups=3, n_params=9)
    ok, text = _quiet(resumed.load_optimizer_state, 11)
    assert ok is False
    assert _n_restored(resumed) == 0
    events = _events(text)
    assert [e["code"] for e in events] == ["optimizer_state_not_restored"]
    # D2: honest about WHY nothing was restored -- a mode limitation of grouped
    # optimizers (no per-component boundary to validate a prefix salvage
    # against), not proof the parameter set changed.
    assert "may not have changed at all" in events[0]["message"]
    assert "grouped-optimizer resumes reset every group" in events[0]["message"]


def test_pre_fix_file_resumed_under_a_different_group_count_says_predates_not_changed(tmp_path):
    """D1: a pre-fix fused file (only optimizer 0 ever written) resumed with a
    DIFFERENT num_optimizer_groups falls through the group-0-size match at
    :3291 (saved holds one group's worth under the OLD count, not the new
    one) and must still name the pre-fix cause, not claim the parameter set
    changed -- it did not."""
    trained = _fused_probe(tmp_path, num_groups=3, n_params=6)
    _train(trained)
    _shipped_save(trained, step=10)

    resumed = _fused_probe(tmp_path, num_groups=2, n_params=6)
    ok, text = _quiet(resumed.load_optimizer_state, 10)
    assert ok is False
    assert _n_restored(resumed) == 0
    events = _events(text)
    assert [e["code"] for e in events] == ["optimizer_state_not_restored"]
    assert "may predate fused-optimizer-group state saving" in events[0]["message"]
    assert "trainable parameter set" not in events[0]["message"]


class _FakeOptimizer:
    """Enough surface for load_optimizer_state's counting/branching logic
    without torch's refusal to construct an optimizer over zero parameters."""

    def __init__(self, n_params):
        self.param_groups = [{"params": list(range(n_params)), "lr": 1e-4}]
        self.state = {}

    def state_dict(self):
        return {"state": {}, "param_groups": [dict(self.param_groups[0])]}

    def load_state_dict(self, state_dict):
        pass


def test_d6_group_zero_size_match_alone_is_not_enough(tmp_path):
    """A saved single-group file whose size matches live group 0 must ALSO
    have a smaller total than the live run to be read as a pre-fix fused
    partial (:3287-3291) -- size-matching group 0 alone is not proof. Here the
    live run's second group happens to be empty (unreachable via
    create_optimizer_groups today, but not a documented invariant of this
    function), so the saved total equals the live total exactly: this is a
    complete match, not a partial one, and must fall through to the ordinary
    exact-total reslice instead of claiming a pre-fix file with lost state."""
    probe = _Probe(tmp_path)
    probe.fused_optimizer_groups = SimpleNamespace(
        optimizers=[_FakeOptimizer(2), _FakeOptimizer(0)])
    probe.optimizer = probe.fused_optimizer_groups.optimizers[0]

    payload = {"state": {}, "param_groups": [{"params": [0, 1], "lr": 1e-4}],
               "_sushi_opt_class": "AdamW"}
    optimizer_file = probe.output_dir / f"{probe.run_name}_step_000005_optimizer.pt"
    torch.save(payload, optimizer_file)

    ok, text = _quiet(probe.load_optimizer_state, 5)
    assert ok is True
    assert _events(text) == []
    assert "re-slicing" in text


# ---------------------------------------------------------------------------
# The learning rate and the schedule position, for every group
# ---------------------------------------------------------------------------

def test_negative_control_shipped_lr_reassert_reaches_only_the_first_optimizer(tmp_path):
    from core.training.lr_utils import reassert_config_lr

    probe = _fused_probe(tmp_path, num_groups=3)
    assert probe._configured_group_lrs == [1e-4, 1e-4, 1e-4]
    _simulate_checkpoint_lr(probe)

    # HEAD~: reassert_config_lr(self.optimizer, ...) -- one of three.
    with redirect_stdout(io.StringIO()):
        reassert_config_lr(probe.optimizer, probe.lr_scheduler, [1e-4],
                           log_prefix="[test]", fallback_lr=1e-4, verbose=False)
    assert _group_lrs(probe) == [1e-4, CKPT_LR, CKPT_LR]


def test_fixed_lr_reassert_reaches_every_optimizer(tmp_path):
    probe = _fused_probe(tmp_path, num_groups=3)
    _simulate_checkpoint_lr(probe)
    _quiet(probe._reassert_config_lr_on_resume)
    assert _group_lrs(probe) == [1e-4, 1e-4, 1e-4]


def test_negative_control_shipped_fast_forward_advances_one_schedule(tmp_path):
    warmup = lambda step: min(1.0, step / 1000.0)
    probe = _fused_probe(tmp_path, num_groups=3, lr_lambda=warmup)

    _shipped_fast_forward(probe, 500)
    assert [s.last_epoch for s in probe.lr_schedulers] == [500, 0, 0]

    _simulate_checkpoint_lr(probe)
    _quiet(probe._reassert_config_lr_on_resume)
    # Groups 1 and 2 resume at the START of warmup: 0.0, not half the base rate.
    assert _group_lrs(probe) == [0.5e-4, 0.0, 0.0]


def test_fixed_fast_forward_advances_every_schedule(tmp_path):
    warmup = lambda step: min(1.0, step / 1000.0)
    probe = _fused_probe(tmp_path, num_groups=3, lr_lambda=warmup)

    probe._fast_forward_lr_schedulers(500)
    assert [s.last_epoch for s in probe.lr_schedulers] == [500, 500, 500]

    _simulate_checkpoint_lr(probe)
    _quiet(probe._reassert_config_lr_on_resume)
    assert _group_lrs(probe) == pytest.approx([0.5e-4] * 3)


def test_lambda_fast_forward_evaluates_only_the_resumed_step(tmp_path):
    calls = []

    def schedule(step):
        calls.append(step)
        return min(1.0, step / 1000.0)

    probe = _single_probe(tmp_path, lr_lambda=schedule)
    calls.clear()  # LambdaLR evaluates step zero during construction.
    probe._fast_forward_lr_schedulers(99_180)

    assert calls == [99_180]
    assert probe.lr_scheduler.last_epoch == 99_180
    assert probe.lr_scheduler._step_count == 99_181
    assert probe.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_fast_forward_is_unchanged_without_fused_groups(tmp_path):
    warmup = lambda step: min(1.0, step / 1000.0)
    probe = _single_probe(tmp_path, lr_lambda=warmup)
    probe._fast_forward_lr_schedulers(500)
    assert probe.lr_scheduler.last_epoch == 500
    assert all_lr_schedulers(probe) == [probe.lr_scheduler]


def test_no_optimizer_is_a_no_op(tmp_path):
    probe = _Probe(tmp_path)
    assert all_optimizers(probe) == []
    ok, _ = _quiet(probe.load_optimizer_state, 1)
    assert ok is False
    probe.save_optimizer_state(1)              # must not raise, must not write
    assert list(Path(tmp_path).glob("*_optimizer.pt")) == []
    probe._reassert_config_lr_on_resume()      # must not raise


# ---------------------------------------------------------------------------
# Architecture independence: the same flat list, whatever built it
# ---------------------------------------------------------------------------

_LORA_ARCHS = ["acestep", "anima", "ideogram4", "krea2", "lens", "ltx2", "minimax_h3"]


def _lora_groups(name, probe):
    from core.training.adapters import (
        AceStepLoRAAdapter, AnimaLoRAAdapter, Ideogram4LoRAAdapter, Krea2LoRAAdapter,
        LensLoRAAdapter, Ltx2LoRAAdapter, MiniMaxH3LoRAAdapter,
    )
    adapters = {
        "acestep": AceStepLoRAAdapter, "anima": AnimaLoRAAdapter,
        "ideogram4": Ideogram4LoRAAdapter, "krea2": Krea2LoRAAdapter,
        "lens": LensLoRAAdapter, "ltx2": Ltx2LoRAAdapter,
        "minimax_h3": MiniMaxH3LoRAAdapter,
    }
    layers = {}
    for i in range(3):
        layer = nn.Module()
        layer.lora_down = nn.Linear(2, 2, bias=False)
        layer.lora_up = nn.Linear(2, 2, bias=False)
        layers[f"lora_unet_x{i}"] = layer
    return adapters[name](probe, lora_rank=2, lora_alpha=2).setup_trainable_parameters(layers)


@pytest.mark.parametrize("name", _LORA_ARCHS)
def test_arch_adapter_groups_round_trip_under_fused_groups(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()

    def build():
        probe = _Probe(directory)
        probe.unet_lr = 2e-5
        groups = _lora_groups(name, probe)
        probe.params = [p for g in groups for p in g["params"]]
        probe.num_optimizer_groups = 3
        with redirect_stdout(io.StringIO()):
            optimizers = create_optimizer_groups(
                params=probe.params, optimizer_type="adamw", num_groups=3,
                learning_rate=probe.learning_rate)
        probe.optimizer = optimizers[0]
        probe.fused_optimizer_groups = FusedOptimizerGroups(optimizers=optimizers)
        probe._record_configured_group_lrs(None)
        return probe

    trained = build()
    assert len(all_optimizers(trained)) == 3
    _train(trained)
    before = _moments(trained)
    assert all(m is not None for m in before)
    _quiet(trained.save_optimizer_state, 2)

    resumed = build()
    ok, _ = _quiet(resumed.load_optimizer_state, 2)
    assert ok is True
    assert _moments_equal(_moments(resumed), before)


# ---------------------------------------------------------------------------
# Wiring, in the shipping source
# ---------------------------------------------------------------------------

def _source():
    return Path(sys.modules[BaseTrainer.__module__].__file__).read_text(encoding="utf-8")


def test_both_resume_branches_fast_forward_every_scheduler():
    source = _source()
    body = source[source.index("    def train("):]
    assert body.count("self._fast_forward_lr_schedulers(global_step)") == 2
    assert "for _ in range(global_step):\n                        self.lr_scheduler.step()" not in body


def test_the_state_paths_no_longer_read_self_optimizer_alone():
    source = _source()
    for name in ("save_optimizer_state", "load_optimizer_state"):
        body = source[source.index(f"    def {name}(self"):]
        body = body[:body.index("\n    def ", 10)]
        assert "all_optimizers(self)" in body
        assert "self.optimizer." not in body
