"""How a configured learning rate reaches an optimizer param group.

The centre of this file is the NEGATIVE CONTROL: the shipped
``getattr(trainer, "unet_lr", None) or <literal>`` idiom replaces a configured
0.0 with a hardcoded rate, and the literal differed per adapter (1e-4, 1e-5,
1e-6) so the same config trained at different rates depending on the
architecture. The control reproduces both, with the numbers.

It also pins the two facts that decide how far that reached: ``BaseTrainer``
derives every component LR from ``learning_rate`` when the config does not set
one, and ``train_runner`` passes the config's ``unet_lr`` through unchanged. Both
are asserted against the shipping source, because they are what makes
"no component LR in the YAML" resolve to the user's ``lr`` rather than to a
literal.
"""

import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.adapters.base_adapter import resolve_component_lr
from core.training.base_trainer import BaseTrainer
from core.training.training_events import TRAINING_EVENT_SENTINEL

ADAPTERS_DIR = Path(__file__).resolve().parents[1] / "core" / "training" / "adapters"

# The literals the shipped idiom substituted, per adapter.
SHIPPED_LITERALS = {
    "sensenova_adapter.py": (1e-4, 1e-6),   # LoRA, full parameter
    "krea2_adapter.py": (1e-4, 1e-5),
    "ideogram4_adapter.py": (1e-4, None),
    "minit2i_adapter.py": (1e-4, 1e-5),
    "acestep_adapter.py": (1e-4, 1e-5),
    "anima_adapter.py": (1e-4, 1e-5),
    "ltx2_adapter.py": (1e-4, 1e-5),
    "lens_adapter.py": (1e-4, 1e-5),
    "minimax_h3_adapter.py": (1e-4, None),
}


def _shipped_unet_lr(trainer, literal):
    """The expression every affected adapter used before this change."""
    return getattr(trainer, "unet_lr", None) or literal


def _trainer(**attrs):
    """A trainer namespace with the LR attributes BaseTrainer.__init__ derives."""
    lr = attrs.pop("learning_rate", 5e-6)
    unet_lr = attrs.pop("unet_lr", None)
    te_lr = attrs.pop("text_encoder_lr", None)
    te1_lr = attrs.pop("text_encoder_1_lr", None)
    return SimpleNamespace(
        learning_rate=lr,
        unet_lr=unet_lr if unet_lr is not None else lr,
        text_encoder_lr=te_lr if te_lr is not None else lr,
        text_encoder_1_lr=te1_lr if te1_lr is not None else (te_lr if te_lr is not None else lr),
        config={},
        **attrs,
    )


# ---------------------------------------------------------------------------
# The negative control
# ---------------------------------------------------------------------------

def test_negative_control_shipped_idiom_substitutes_a_literal_for_a_configured_zero():
    """``lr: 0.0`` (or ``unet_lr: 0.0``) trained at a hardcoded rate.

    Both YAML keys are creatable through the API: ``learning_rate`` is a bare
    ``float`` on ``TrainingRunCreateRequest`` and ``unet_lr`` an
    ``Optional[float]``, neither with a lower bound. The config, the
    ``training_runs`` row and the UI all say 0.0 while the optimizer group runs
    at the adapter's literal -- and which literal depends on the architecture.
    """
    zero_run = _trainer(learning_rate=0.0)
    assert zero_run.unet_lr == 0.0 and zero_run.text_encoder_1_lr == 0.0

    # Shipped: the requested 0.0 becomes a different nonzero rate per adapter.
    assert _shipped_unet_lr(zero_run, 1e-6) == 1e-6      # SenseNova full FT
    assert _shipped_unet_lr(zero_run, 1e-5) == 1e-5      # Krea2/Anima/LTX2/Lens/... full FT
    assert _shipped_unet_lr(zero_run, 1e-4) == 1e-4      # every LoRA adapter

    # Fixed: what was configured is what trains.
    assert resolve_component_lr(zero_run, "unet_lr") == 0.0
    assert resolve_component_lr(
        zero_run, "text_encoder_1_lr", "text_encoder_lr", "unet_lr") == 0.0

    # Same substitution for a per-component zero on an otherwise normal run.
    frozen_half = _trainer(learning_rate=5e-6, unet_lr=0.0)
    assert _shipped_unet_lr(frozen_half, 1e-6) == 1e-6
    assert resolve_component_lr(frozen_half, "unet_lr") == 0.0


def test_absent_component_lr_resolves_to_learning_rate_before_and_after():
    """The wider claim -- "a full FT with only ``lr`` set trains at 1e-6" -- does
    not hold, and this records why: ``BaseTrainer.__init__`` fills ``unet_lr``
    from ``learning_rate``, so the literal was only reachable through a falsy LR.
    """
    run = _trainer(learning_rate=5e-6)  # no component LRs, as the YAML omits them
    assert _shipped_unet_lr(run, 1e-6) == 5e-6
    assert resolve_component_lr(run, "unet_lr") == 5e-6


def test_base_trainer_derives_component_lrs_from_learning_rate():
    """Pins the normalization the test above depends on, in the real source."""
    source = Path(BaseTrainer.__module__ and
                  sys.modules[BaseTrainer.__module__].__file__).read_text(encoding="utf-8")
    assert re.search(
        r"self\.unet_lr = unet_lr if unet_lr is not None else learning_rate", source)
    assert re.search(
        r"self\.text_encoder_1_lr = text_encoder_1_lr if text_encoder_1_lr is not None", source)


def test_train_runner_passes_the_configs_component_lrs_through():
    """The full-FT branch reads ``lr`` and ``unet_lr`` off the YAML unchanged."""
    runner = (Path(__file__).resolve().parents[1] / "core" / "training" /
              "train_runner.py").read_text(encoding="utf-8")
    block = runner[runner.index("trainer = FullParameterTrainer(") - 4000:
                   runner.index("trainer = FullParameterTrainer(") + 3000]
    assert "unet_lr = train_config.get('unet_lr')" in block
    assert "learning_rate=train_config.get('lr', 1e-4)" in block
    assert "unet_lr=unet_lr," in block


# ---------------------------------------------------------------------------
# The resolver's rule
# ---------------------------------------------------------------------------

def test_resolver_precedence_first_configured_then_learning_rate():
    run = _trainer(learning_rate=1e-4, text_encoder_lr=3e-5, text_encoder_1_lr=7e-5)
    assert resolve_component_lr(run, "text_encoder_1_lr", "text_encoder_lr") == 7e-5
    run.text_encoder_1_lr = None
    assert resolve_component_lr(run, "text_encoder_1_lr", "text_encoder_lr") == 3e-5
    run.text_encoder_lr = None
    assert resolve_component_lr(run, "text_encoder_1_lr", "text_encoder_lr") == 1e-4


def test_resolver_refuses_rather_than_inventing_a_rate():
    with pytest.raises(ValueError, match="Cannot resolve a learning rate"):
        resolve_component_lr(SimpleNamespace(), "unet_lr", label="nothing configured")


# ---------------------------------------------------------------------------
# Census: no adapter carries a literal LR fallback any more
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", sorted(SHIPPED_LITERALS))
def test_adapter_has_no_literal_lr_fallback(filename):
    source = (ADAPTERS_DIR / filename).read_text(encoding="utf-8")
    offenders = [
        line.strip() for line in source.splitlines()
        if re.search(r"_lr\"?,?\s*(None\)|1e-\d)\s*(\)|or\s*1e-\d)", line)
        and "resolve_component_lr" not in line
    ]
    assert offenders == [], offenders


def test_every_adapter_lr_site_goes_through_the_resolver():
    """No ``x or <literal>`` LR idiom survives anywhere under adapters/."""
    pattern = re.compile(r'getattr\([^)]*_lr"[^)]*\)\s*or\s')
    hits = {}
    for path in sorted(ADAPTERS_DIR.glob("*.py")):
        if path.name == "base_adapter.py":
            continue  # the resolver's own docstring quotes the idiom it replaced
        found = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                 if pattern.search(l)]
        if found:
            hits[path.name] = found
    assert hits == {}


# ---------------------------------------------------------------------------
# Per-adapter behaviour with a real fake trainer
# ---------------------------------------------------------------------------

def _lora_layers():
    """The real branch class, not a stub: it owns the tensor protocol the
    adapters collect parameters through."""
    from core.adapters import LoRALinearLayer

    return {"lora_unet_x": LoRALinearLayer(nn.Linear(4, 4, bias=False), 2, 2,
                                           "lora_unet_x")}


def _lora_adapters():
    from core.training.adapters import (
        AceStepLoRAAdapter, AnimaLoRAAdapter, Ideogram4LoRAAdapter, Krea2LoRAAdapter,
        LensLoRAAdapter, Ltx2LoRAAdapter, MiniMaxH3LoRAAdapter, SenseNovaLoRAAdapter,
    )
    return {
        "acestep": AceStepLoRAAdapter,
        "anima": AnimaLoRAAdapter,
        "ideogram4": Ideogram4LoRAAdapter,
        "krea2": Krea2LoRAAdapter,
        "lens": LensLoRAAdapter,
        "ltx2": Ltx2LoRAAdapter,
        "minimax_h3": MiniMaxH3LoRAAdapter,
        "sensenova": SenseNovaLoRAAdapter,
    }


@pytest.mark.parametrize("name", sorted(_lora_adapters()))
def test_lora_adapter_group_uses_the_configured_lr(name):
    adapter_cls = _lora_adapters()[name]
    run = _trainer(learning_rate=5e-6)
    adapter = adapter_cls(run, lora_rank=4, lora_alpha=4)
    groups = adapter.setup_trainable_parameters(_lora_layers())
    assert [g["lr"] for g in groups] == [5e-6]

    zero = _trainer(learning_rate=0.0)
    adapter = adapter_cls(zero, lora_rank=4, lora_alpha=4)
    assert [g["lr"] for g in adapter.setup_trainable_parameters(_lora_layers())] == [0.0]


def _full_adapters():
    from core.training.adapters import (
        AceStepFullParameterAdapter, Krea2FullParameterAdapter, Ltx2FullParameterAdapter,
    )
    return {
        "acestep": AceStepFullParameterAdapter,
        "krea2": Krea2FullParameterAdapter,
        "ltx2": Ltx2FullParameterAdapter,
    }


@pytest.mark.parametrize("name", sorted(_full_adapters()))
def test_full_parameter_adapter_group_uses_the_configured_lr(name):
    adapter_cls = _full_adapters()[name]
    transformer = nn.Sequential(nn.Linear(4, 4))
    run = _trainer(learning_rate=5e-6, transformer=transformer, text_encoder=None)
    groups = adapter_cls(run).setup_trainable_parameters()
    assert [g["lr"] for g in groups] == [5e-6]

    zero = _trainer(learning_rate=0.0, transformer=transformer, text_encoder=None)
    assert [g["lr"] for g in adapter_cls(zero).setup_trainable_parameters()] == [0.0]


# ---------------------------------------------------------------------------
# SenseNova: the two MoT halves
# ---------------------------------------------------------------------------

def _sensenova_full_adapter(**lrs):
    from sensenova_int8_materialize_test import _Decoder
    from core.models.sensenova.loader import materialize_int8_decoder_linears
    from core.training.adapters import SenseNovaFullParameterAdapter

    transformer = _Decoder()
    materialize_int8_decoder_linears(transformer, branch="both")
    trainer = _trainer(
        transformer=transformer, unet=None, text_encoder=None, text_encoder_2=None,
        is_sensenova=True, train_unet=True, train_text_encoder=True, **lrs,
    )
    trainer.config = {"optimizer": "adafactor"}
    adapter = SenseNovaFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    return trainer, adapter


def test_sensenova_full_ft_trains_both_halves_at_the_configured_rate():
    _, adapter = _sensenova_full_adapter(learning_rate=5e-6)
    assert [g["lr"] for g in adapter.setup_trainable_parameters()] == [5e-6, 5e-6]


def test_sensenova_full_ft_honours_explicit_component_lrs():
    _, adapter = _sensenova_full_adapter(
        learning_rate=5e-6, unet_lr=2e-6, text_encoder_lr=3e-6)
    assert [g["lr"] for g in adapter.setup_trainable_parameters()] == [2e-6, 3e-6]


def test_sensenova_full_ft_run_121_configuration_is_unchanged():
    """Run 121 sets ``lr``, ``unet_lr`` and ``text_encoder_lr`` all at 1e-6."""
    _, adapter = _sensenova_full_adapter(
        learning_rate=1e-6, unet_lr=1e-6, text_encoder_lr=1e-6)
    groups = adapter.setup_trainable_parameters()
    assert [g["lr"] for g in groups] == [1e-6, 1e-6]
    # And the shipped expression produced the same two numbers for it.
    run = _trainer(learning_rate=1e-6, unet_lr=1e-6, text_encoder_lr=1e-6)
    assert _shipped_unet_lr(run, 1e-6) == 1e-6
    assert (getattr(run, "text_encoder_1_lr", None)
            or getattr(run, "text_encoder_lr", None)
            or _shipped_unet_lr(run, 1e-6)) == 1e-6


# ---------------------------------------------------------------------------
# The effective LR is reported, and a surprise is warned about
# ---------------------------------------------------------------------------

def _optimizer(group_lrs):
    return SimpleNamespace(param_groups=[
        {"lr": lr, "params": [torch.zeros(1)]} for lr in group_lrs
    ])


class _LRProbe:
    """Just enough of a trainer to run the two reporting methods."""

    _build_component_lr_list = BaseTrainer._build_component_lr_list
    _report_effective_component_lrs = BaseTrainer._report_effective_component_lrs

    def __init__(self, group_lrs, **attrs):
        self.log_prefix = "[test]"
        self.learning_rate = attrs.pop("learning_rate", 5e-6)
        self.unet_lr = attrs.pop("unet_lr", self.learning_rate)
        self.text_encoder_lr = attrs.pop("text_encoder_lr", self.learning_rate)
        self.text_encoder_1_lr = attrs.pop("text_encoder_1_lr", self.text_encoder_lr)
        self.is_sensenova = True
        self.train_unet = True
        self.train_text_encoder = True
        self.unet = None
        self.text_encoder = None
        self.controlnet = None
        self.num_optimizer_groups = attrs.pop("num_optimizer_groups", 0)
        self.fused_optimizer_groups = None
        for key, value in attrs.items():
            setattr(self, key, value)
        self.optimizer = _optimizer(group_lrs)


def _report(probe, requested=None):
    """(events, console text) from one report call."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        probe._report_effective_component_lrs(requested)
    text = buffer.getvalue()
    events = [json.loads(line.split(TRAINING_EVENT_SENTINEL, 1)[1])
              for line in text.splitlines() if TRAINING_EVENT_SENTINEL in line]
    return events, text


def test_matching_group_lrs_emit_no_warning():
    events, text = _report(_LRProbe([5e-6, 5e-6], learning_rate=5e-6))
    assert events == []
    assert "did not run" not in text


def test_group_lr_that_is_not_the_configured_one_is_warned_about():
    events, _ = _report(_LRProbe([1e-6, 5e-6], learning_rate=5e-6))
    assert [e["code"] for e in events] == ["component_lr_mismatch"]
    assert "MoT-Generation" in events[0]["message"]
    assert "1e-06" in events[0]["message"] and "5e-06" in events[0]["message"]


def test_zero_lr_group_is_warned_about():
    events, _ = _report(_LRProbe([0.0, 0.0], learning_rate=0.0))
    assert [e["code"] for e in events] == ["component_lr_zero"] * 2
    assert "MoT-Generation" in events[0]["message"]
    assert "will not change" in events[0]["message"]


def test_the_check_announces_when_it_cannot_run():
    """A DiT architecture: ``unet`` is None and no SenseNova/ControlNet/VE
    branch fires, so ``_build_component_lr_list`` is empty and the index-wise
    comparison has nothing to compare. It must say so.
    """
    probe = _LRProbe([2e-5, 4e-5], learning_rate=1e-4, is_sensenova=False,
                     train_text_encoder=False)
    assert probe._build_component_lr_list() == ([], [])
    events, text = _report(probe)
    assert events == []
    assert "per-component LR verification did not run" in text
    assert "describes 0 group(s) [] and the optimizer has 2" in text


def test_the_check_announces_when_the_component_list_raises():
    probe = _LRProbe([1e-4])
    del probe.learning_rate  # _build_component_lr_list cannot resolve anything
    events, text = _report(probe)
    assert "per-component LR verification did not run" in text
    assert [e["code"] for e in events] == []


# ---------------------------------------------------------------------------
# The path that actually destroys per-component LRs
# ---------------------------------------------------------------------------

def test_fused_optimizer_groups_flattening_is_reported():
    """``_setup_fused_optimizer_groups`` rebuilds N optimizers from a flat
    parameter list at ``learning_rate``; the adapter's per-component rates are
    gone by then. Reported because the report runs after that rebuild.
    """
    probe = _LRProbe([1e-4], learning_rate=1e-4, num_optimizer_groups=3)
    probe.fused_optimizer_groups = SimpleNamespace(
        optimizers=[_optimizer([1e-4]), _optimizer([1e-4]), _optimizer([1e-4])])
    events, _ = _report(probe, requested=[2e-5, 4e-5, 1e-5])
    assert events[0]["code"] == "component_lr_flattened"
    assert "[2e-05, 4e-05, 1e-05]" in events[0]["message"]
    assert "num_optimizer_groups=3" in events[0]["message"]


def test_fused_optimizer_groups_with_one_requested_rate_is_not_reported():
    probe = _LRProbe([1e-4], learning_rate=1e-4, num_optimizer_groups=2)
    probe.fused_optimizer_groups = SimpleNamespace(
        optimizers=[_optimizer([1e-4]), _optimizer([1e-4])])
    events, _ = _report(probe, requested=[1e-4, 1e-4])
    assert [e["code"] for e in events] == []


def test_the_report_runs_after_the_fused_setup():
    """Placement, in the shipping source: the only call site is below
    ``_setup_fused_optimizer_groups`` / ``_setup_fused_backward_pass``.
    """
    source = Path(sys.modules[BaseTrainer.__module__].__file__).read_text(encoding="utf-8")
    body = source[source.index("    def setup_optimizer("):]
    body = body[:body.index("\n    def ", 10)]
    assert body.count("_report_effective_component_lrs(") == 1
    assert (body.index("_report_effective_component_lrs(")
            > body.index("_setup_fused_optimizer_groups("))
    assert (body.index("_report_effective_component_lrs(")
            > body.rindex("_setup_fused_backward_pass("))
    assert (body.index("_report_effective_component_lrs(")
            > body.index("_attach_stochastic_rounding("))
