"""What a resume writes into each optimizer param group, per architecture.

The NEGATIVE CONTROL is ``test_negative_control_*``: the shipped
``_reassert_config_lr_on_resume`` passed
``component_lrs if component_lrs else self.learning_rate`` to
``reassert_config_lr``, and a SCALAR there is broadcast over every group
(``lr_utils.resolve_group_lrs``). ``_build_component_lr_list`` is empty on every
DiT architecture (``self.unet is None``), so on an Anima full fine-tune with
``lr: 1e-4``, ``unet_lr: 2e-5``, ``anima_attn_mlp_lr_factor: 2.0``,
``anima_mod_lr_factor: 0.5``:

    groups at setup:  [2e-05, 4e-05, 1e-05]
    resume writes:    [0.0001, 0.0001, 0.0001]

Two misalignment modes, both of which write by index and are therefore both
wrong:

1. the list is EMPTY -> a scalar is broadcast (every DiT architecture);
2. the list is NON-EMPTY but does not correspond index-for-index -- a
   Flux2/MiniT2I/Z-Image run with ``train_text_encoder`` describes only ``TE1``
   while group 0 is the transformer, and an SDXL run with a custom text encoder
   describes ``[U-Net, TE1, TE2]`` for groups ``[U-Net, bridge adapters, TE
   body]``.

The param groups here come from the REAL adapters wherever building one is
cheap (tiny CPU ``nn.Linear`` stand-ins), so the group ORDER under test is the
shipping order rather than a restatement of it.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/component_lr_resume_alignment_test.py -v
"""

from __future__ import annotations

import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.base_trainer import BaseTrainer
from core.training.lr_utils import reassert_config_lr
from core.training.training_events import TRAINING_EVENT_SENTINEL

CKPT_LR = 9.876e-06   # what a checkpoint's optimizer state carries back in


# ---------------------------------------------------------------------------
# A trainer stand-in carrying the real methods
# ---------------------------------------------------------------------------

class _Probe:
    _build_component_lr_list = BaseTrainer._build_component_lr_list
    _record_configured_group_lrs = BaseTrainer._record_configured_group_lrs
    _name_configured_groups = BaseTrainer._name_configured_groups
    _configured_component_lr_description = BaseTrainer._configured_component_lr_description
    _reassert_config_lr_on_resume = BaseTrainer._reassert_config_lr_on_resume
    _report_effective_component_lrs = BaseTrainer._report_effective_component_lrs

    def __init__(self, learning_rate=1e-4, **attrs):
        self.log_prefix = "[test]"
        self.learning_rate = learning_rate
        self.unet_lr = attrs.pop("unet_lr", learning_rate)
        self.text_encoder_lr = attrs.pop("text_encoder_lr", learning_rate)
        self.text_encoder_1_lr = attrs.pop("text_encoder_1_lr", self.text_encoder_lr)
        self.text_encoder_2_lr = attrs.pop("text_encoder_2_lr", self.text_encoder_lr)
        self.unet = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.controlnet = None
        self.is_sdxl = False
        self.is_sensenova = False
        self.train_unet = True
        self.train_text_encoder = False
        self._train_vision_encoder = False
        self.vision_encoder = None
        self.num_optimizer_groups = 0
        self.fused_optimizer_groups = None
        self.config = {}
        self.optimizer = None
        self.lr_scheduler = None
        for key, value in attrs.items():
            setattr(self, key, value)


def _attach_optimizer(probe, param_groups, lr_lambda=None):
    """Build a real optimizer (+ optional real scheduler) and take the snapshot."""
    probe.optimizer = torch.optim.AdamW(
        [dict(g) for g in param_groups], lr=probe.learning_rate)
    requested = [g.get("lr") for g in probe.optimizer.param_groups]
    if lr_lambda is not None:
        probe.lr_scheduler = LambdaLR(probe.optimizer, lr_lambda=lr_lambda)
    probe._record_configured_group_lrs(requested)
    return requested


def _simulate_checkpoint_load(probe, lr=CKPT_LR):
    """What ``Optimizer.load_state_dict`` does to every group's lr."""
    for group in probe.optimizer.param_groups:
        group["lr"] = lr


def _group_lrs(probe):
    return [float(g["lr"]) for g in probe.optimizer.param_groups]


def _events(fn, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn(*args, **kwargs)
    text = buffer.getvalue()
    events = [json.loads(line.split(TRAINING_EVENT_SENTINEL, 1)[1])
              for line in text.splitlines() if TRAINING_EVENT_SENTINEL in line]
    return events, text


def _shipped_reassert(probe):
    """The exact statement HEAD~ shipped, for the negative control."""
    component_lrs, component_names = probe._build_component_lr_list()
    reassert_config_lr(
        probe.optimizer,
        probe.lr_scheduler,
        component_lrs if component_lrs else probe.learning_rate,
        log_prefix=probe.log_prefix,
        component_names=component_names,
        fallback_lr=probe.learning_rate,
        verbose=False,
    )


def _resume(probe):
    with redirect_stdout(io.StringIO()):
        probe._reassert_config_lr_on_resume()
    return _group_lrs(probe)


def _params(n=1):
    return [nn.Parameter(torch.ones(2)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Real adapter param groups
# ---------------------------------------------------------------------------

class _AnimaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Linear(2, 2, bias=False)
        self.adaln_modulation_1 = nn.Linear(2, 2, bias=False)


class _AnimaLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_AnimaBlock()])
        self.final_layer = nn.Linear(2, 2, bias=False)


class _LensAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_qkv = nn.Linear(2, 2, bias=False)
        self.txt_qkv = nn.Linear(2, 2, bias=False)


class _LensBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _LensAttn()


class _LensLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_LensBlock()])
        self.final_layer = nn.Linear(2, 2, bias=False)


def _anima_full_groups(probe):
    from core.training.adapters import AnimaFullParameterAdapter
    probe.transformer = _AnimaLike()
    return AnimaFullParameterAdapter(probe).setup_trainable_parameters()


def _lens_full_groups(probe):
    from core.training.adapters import LensFullParameterAdapter
    probe.transformer = _LensLike()
    return LensFullParameterAdapter(probe).setup_trainable_parameters()


def _flux2_full_groups(probe):
    from core.training.adapters import FLUX2FullParameterAdapter
    probe.transformer = nn.Linear(2, 2)
    probe.text_encoder = nn.Linear(2, 2)
    probe.train_text_encoder = True
    return FLUX2FullParameterAdapter(probe).setup_trainable_parameters()


def _minit2i_full_groups(probe):
    from core.training.adapters import MiniT2IFullParameterAdapter
    probe.transformer = nn.Linear(2, 2)
    probe.text_encoder = nn.Linear(2, 2)
    probe.train_text_encoder = True
    probe.repa_enable = False
    return MiniT2IFullParameterAdapter(probe).setup_trainable_parameters()


def _zimage_full_groups(probe):
    from core.training.adapters import ZImageFullParameterAdapter
    probe.transformer = nn.Linear(2, 2)
    probe.text_encoder = nn.Linear(2, 2)
    probe.train_text_encoder = True
    return ZImageFullParameterAdapter(probe).setup_trainable_parameters()


def _sdxl_custom_te_groups(probe):
    from core.training.adapters import SDXLFullParameterAdapter
    probe.is_sdxl = True
    probe.unet = nn.Linear(2, 2)
    probe.text_encoder = nn.Linear(2, 2)
    probe.text_encoder_2 = nn.Linear(2, 2)
    probe.train_text_encoder = True
    probe.sdxl_te_type = "t5"
    probe.sdxl_te_train_encoder = True
    probe.te_adapters = nn.Linear(2, 2)
    probe.te_custom = nn.Linear(2, 2)
    return SDXLFullParameterAdapter(probe).setup_trainable_parameters()


# ---------------------------------------------------------------------------
# The negative control: the shipped broadcast, with the auditor's numbers
# ---------------------------------------------------------------------------

def _anima_probe():
    """lr 1e-4, unet_lr 2e-5, attn_mlp x2.0, mod x0.5 -- the audited run."""
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    probe.config = {"anima_attn_mlp_lr_factor": 2.0, "anima_mod_lr_factor": 0.5}
    groups = _anima_full_groups(probe)
    _attach_optimizer(probe, groups)
    return probe


def test_negative_control_shipped_resume_broadcasts_over_every_anima_group():
    probe = _anima_probe()
    assert probe._build_component_lr_list() == ([], [])
    assert _group_lrs(probe) == [2e-05, 4e-05, 1e-05]

    _simulate_checkpoint_load(probe)
    _shipped_reassert(probe)

    assert _group_lrs(probe) == [1e-04, 1e-04, 1e-04]
    # 5x, 2.5x and 10x wrong respectively.
    assert [round(got / want, 6) for got, want in
            zip(_group_lrs(probe), [2e-05, 4e-05, 1e-05])] == [5.0, 2.5, 10.0]


def test_fixed_resume_restores_each_anima_group_to_its_own_rate():
    probe = _anima_probe()
    setup = _group_lrs(probe)
    _simulate_checkpoint_load(probe)
    assert _resume(probe) == setup == [2e-05, 4e-05, 1e-05]


def test_negative_control_single_group_dit_gets_learning_rate_not_unet_lr():
    """One group is not immune: the broadcast writes ``lr``, not ``unet_lr``."""
    from core.training.adapters import Krea2FullParameterAdapter

    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    probe.transformer = nn.Linear(2, 2)
    groups = Krea2FullParameterAdapter(probe).setup_trainable_parameters()
    _attach_optimizer(probe, groups)
    assert _group_lrs(probe) == [2e-05]

    _simulate_checkpoint_load(probe)
    _shipped_reassert(probe)
    assert _group_lrs(probe) == [1e-04]           # shipped: the run's lr

    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    probe.transformer = nn.Linear(2, 2)
    _attach_optimizer(probe, Krea2FullParameterAdapter(probe).setup_trainable_parameters())
    _simulate_checkpoint_load(probe)
    assert _resume(probe) == [2e-05]              # fixed: the configured unet_lr


# ---------------------------------------------------------------------------
# Mode 1: the component list is EMPTY (every DiT architecture)
# ---------------------------------------------------------------------------

_EMPTY_LIST_ARCHS = {
    "anima_full": _anima_full_groups,
    "lens_full": _lens_full_groups,
}


@pytest.mark.parametrize("name", sorted(_EMPTY_LIST_ARCHS))
def test_multi_group_dit_full_ft_survives_a_resume(name):
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    probe.config = {"anima_attn_mlp_lr_factor": 2.0, "anima_mod_lr_factor": 0.5,
                    "lens_img_lr_factor": 3.0, "lens_txt_lr_factor": 0.25}
    groups = _EMPTY_LIST_ARCHS[name](probe)
    setup = [g["lr"] for g in groups]
    assert len(set(setup)) > 1, "degenerate fixture: the groups share one rate"
    _attach_optimizer(probe, groups)

    assert probe._build_component_lr_list() == ([], [])   # mode 1
    _simulate_checkpoint_load(probe)
    assert _resume(probe) == setup


_LORA_ARCHS = [
    "acestep", "anima", "ideogram4", "krea2", "lens", "ltx2", "minimax_h3",
]


@pytest.mark.parametrize("name", _LORA_ARCHS)
def test_single_group_dit_lora_resumes_at_the_configured_unet_lr(name):
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
    from core.adapters import LoRALinearLayer

    layer = LoRALinearLayer(nn.Linear(2, 2, bias=False), 2, 2, "lora_unet_x")

    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    groups = adapters[name](probe, lora_rank=2, lora_alpha=2).setup_trainable_parameters(
        {"lora_unet_x": layer})
    _attach_optimizer(probe, groups)
    assert _group_lrs(probe) == [2e-05]

    _simulate_checkpoint_load(probe)
    assert _resume(probe) == [2e-05]


# ---------------------------------------------------------------------------
# Mode 2: the component list is NON-EMPTY but does not line up
# ---------------------------------------------------------------------------

_MISALIGNED_ARCHS = {
    "flux2_full_te": _flux2_full_groups,
    "minit2i_full_te": _minit2i_full_groups,
    "zimage_full_te": _zimage_full_groups,
}


@pytest.mark.parametrize("name", sorted(_MISALIGNED_ARCHS))
def test_dit_with_a_trained_text_encoder_is_described_short_not_empty(name):
    """``train_text_encoder`` makes the list ``[TE1]`` while group 0 is the DiT."""
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5, text_encoder_lr=3e-6)
    probe.config = {"minit2i_lr_factor": 1.0}
    groups = _MISALIGNED_ARCHS[name](probe)
    setup = [g["lr"] for g in groups]
    assert setup == [2e-05, 3e-06]

    legacy_lrs, legacy_names = probe._build_component_lr_list()
    assert legacy_names == ["TE1"] and legacy_lrs == [3e-06]   # non-empty, shorter

    _attach_optimizer(probe, groups)
    _simulate_checkpoint_load(probe)

    # Shipped: the TE rate lands on the transformer and the TE takes the run's lr.
    shipped = _Probe(learning_rate=1e-4, unet_lr=2e-5, text_encoder_lr=3e-6)
    shipped.config = {"minit2i_lr_factor": 1.0}
    _attach_optimizer(shipped, _MISALIGNED_ARCHS[name](shipped))
    _simulate_checkpoint_load(shipped)
    _shipped_reassert(shipped)
    assert _group_lrs(shipped) == [3e-06, 1e-04]

    assert _resume(probe) == setup


def test_sdxl_custom_te_groups_are_described_by_the_adapter_not_by_te1_te2():
    """Same length, different meaning: [U-Net, bridge, TE body] vs [U-Net, TE1, TE2]."""
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5, text_encoder_lr=3e-6,
                   text_encoder_1_lr=4e-6, text_encoder_2_lr=5e-6)
    groups = _sdxl_custom_te_groups(probe)
    setup = [g["lr"] for g in groups]
    assert setup == [2e-05, 3e-06, 4e-06]   # unet, bridge adapters, custom TE body

    legacy_lrs, legacy_names = probe._build_component_lr_list()
    assert legacy_names == ["U-Net", "TE1", "TE2"]
    assert legacy_lrs == [2e-05, 4e-06, 5e-06]   # same length, wrong components

    _attach_optimizer(probe, groups)
    _simulate_checkpoint_load(probe)
    assert _resume(probe) == setup

    shipped = _Probe(learning_rate=1e-4, unet_lr=2e-5, text_encoder_lr=3e-6,
                     text_encoder_1_lr=4e-6, text_encoder_2_lr=5e-6)
    _attach_optimizer(shipped, _sdxl_custom_te_groups(shipped))
    _simulate_checkpoint_load(shipped)
    _shipped_reassert(shipped)
    assert _group_lrs(shipped) == [2e-05, 4e-06, 5e-06]   # bridge/TE body swapped


# ---------------------------------------------------------------------------
# The aligned architectures are unchanged
# ---------------------------------------------------------------------------

def test_sdxl_unet_te1_te2_resume_is_unchanged():
    probe = _Probe(learning_rate=1e-4, unet_lr=1e-5,
                   text_encoder_1_lr=5e-6, text_encoder_2_lr=2e-6)
    probe.is_sdxl = True
    probe.unet = nn.Linear(2, 2)
    probe.text_encoder = nn.Linear(2, 2)
    probe.text_encoder_2 = nn.Linear(2, 2)
    probe.train_text_encoder = True
    groups = [{"params": _params(), "lr": 1e-5},
              {"params": _params(), "lr": 5e-6},
              {"params": _params(), "lr": 2e-6}]
    _attach_optimizer(probe, groups)

    assert probe._build_component_lr_list() == ([1e-5, 5e-6, 2e-6], ["U-Net", "TE1", "TE2"])
    _simulate_checkpoint_load(probe)
    assert _resume(probe) == [1e-5, 5e-6, 2e-6]

    # ...and the shipped path already produced the same three numbers here.
    shipped = _Probe(learning_rate=1e-4, unet_lr=1e-5,
                     text_encoder_1_lr=5e-6, text_encoder_2_lr=2e-6)
    shipped.is_sdxl = True
    shipped.unet = nn.Linear(2, 2)
    shipped.text_encoder = nn.Linear(2, 2)
    shipped.text_encoder_2 = nn.Linear(2, 2)
    shipped.train_text_encoder = True
    _attach_optimizer(shipped, [{"params": _params(), "lr": lr}
                                for lr in (1e-5, 5e-6, 2e-6)])
    _simulate_checkpoint_load(shipped)
    _shipped_reassert(shipped)
    assert _group_lrs(shipped) == [1e-5, 5e-6, 2e-6]


def _sensenova_probe(unet_lr, und_lr, learning_rate):
    probe = _Probe(learning_rate=learning_rate, unet_lr=unet_lr,
                   text_encoder_lr=und_lr, text_encoder_1_lr=und_lr)
    probe.is_sensenova = True
    probe.train_unet = True
    probe.train_text_encoder = True
    _attach_optimizer(probe, [{"params": _params(), "lr": unet_lr},
                              {"params": _params(), "lr": und_lr}])
    return probe


def test_sensenova_two_mot_halves_are_unchanged():
    probe = _sensenova_probe(2e-6, 3e-6, 5e-6)
    lrs, names = probe._build_component_lr_list()
    assert names == ["MoT-Generation", "MoT-Understanding"] and lrs == [2e-6, 3e-6]
    _simulate_checkpoint_load(probe)
    assert _resume(probe) == [2e-6, 3e-6]

    shipped = _sensenova_probe(2e-6, 3e-6, 5e-6)
    _simulate_checkpoint_load(shipped)
    _shipped_reassert(shipped)
    assert _group_lrs(shipped) == [2e-6, 3e-6]   # aligned: never broke here


def test_run_121_shape_is_unaffected():
    """``sensenova_both_fullft_1024_v1``: lr = unet_lr = text_encoder_lr = 1e-6,
    both MoT halves trained, lr_scheduler constant, num_optimizer_groups 0."""
    for build in (lambda: _sensenova_probe(1e-6, 1e-6, 1e-6),):
        probe = build()
        assert probe._configured_group_lrs == [1e-6, 1e-6]
        _simulate_checkpoint_load(probe)
        assert _resume(probe) == [1e-6, 1e-6]

        shipped = build()
        _simulate_checkpoint_load(shipped)
        _shipped_reassert(shipped)
        assert _group_lrs(shipped) == [1e-6, 1e-6]


def test_controlnet_single_group_is_unchanged():
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    probe.train_unet = False
    probe.controlnet = nn.Linear(2, 2)
    _attach_optimizer(probe, [{"params": _params(), "lr": 2e-5}])
    assert probe._build_component_lr_list() == ([2e-5], ["ControlNet"])
    _simulate_checkpoint_load(probe)
    assert _resume(probe) == [2e-5]


# ---------------------------------------------------------------------------
# What happens when no description can be built
# ---------------------------------------------------------------------------

def test_resume_refuses_to_write_when_the_description_cannot_be_built():
    probe = _anima_probe()
    probe._configured_group_lrs = None          # no setup_optimizer ran
    probe._configured_group_names = None
    _simulate_checkpoint_load(probe)

    events, text = _events(probe._reassert_config_lr_on_resume)

    assert _group_lrs(probe) == [CKPT_LR] * 3, "a refusal must not write anything"
    assert [e["code"] for e in events] == ["component_lr_resume_unavailable"]
    assert "NOT re-asserted" in events[0]["message"]
    assert "3 live param group(s)" in events[0]["message"]


def test_a_single_group_without_a_snapshot_still_takes_the_run_learning_rate():
    """One group, one rate: no index to get wrong, so the write is unambiguous."""
    probe = _Probe(learning_rate=2.5e-6)
    _attach_optimizer(probe, [{"params": _params(), "lr": 2.5e-6}])
    probe._configured_group_lrs = None
    probe._configured_group_names = None
    _simulate_checkpoint_load(probe)
    events, _ = _events(probe._reassert_config_lr_on_resume)
    assert _group_lrs(probe) == [2.5e-6]
    assert events == []


def test_no_param_groups_is_a_no_op():
    probe = _Probe()
    probe.optimizer = SimpleNamespace(param_groups=[])
    probe._reassert_config_lr_on_resume()       # must not raise


# ---------------------------------------------------------------------------
# The snapshot's other properties
# ---------------------------------------------------------------------------

def test_snapshot_is_the_base_rate_not_the_warmup_scaled_one():
    """The scheduler exists by the time the snapshot is taken; at step 0 a
    warmup lambda has already written 0.0 into every group."""
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    probe.config = {"anima_attn_mlp_lr_factor": 2.0, "anima_mod_lr_factor": 0.5}
    groups = _anima_full_groups(probe)
    _attach_optimizer(probe, groups, lr_lambda=lambda step: min(1.0, step / 1000.0))

    assert _group_lrs(probe) == [0.0, 0.0, 0.0]          # warmup step 0
    assert probe._configured_group_lrs == [2e-05, 4e-05, 1e-05]


def test_resume_applies_the_schedule_multiplier_to_each_snapshot_rate():
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    probe.config = {"anima_attn_mlp_lr_factor": 2.0, "anima_mod_lr_factor": 0.5}
    _attach_optimizer(probe, _anima_full_groups(probe),
                      lr_lambda=lambda step: min(1.0, step / 1000.0))
    for _ in range(500):
        probe.lr_scheduler.step()
    _simulate_checkpoint_load(probe)

    assert _resume(probe) == pytest.approx([2e-05 * 0.5, 4e-05 * 0.5, 1e-05 * 0.5])
    assert probe.lr_scheduler.base_lrs == [2e-05, 4e-05, 1e-05]


def test_lens_group_names_reach_the_log():
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5)
    probe.config = {"lens_img_lr_factor": 3.0, "lens_txt_lr_factor": 0.25}
    _attach_optimizer(probe, _lens_full_groups(probe))
    assert probe._configured_group_names == ["img_stream", "txt_stream", "other"]
    _simulate_checkpoint_load(probe)
    _, text = _events(probe._reassert_config_lr_on_resume)
    assert "LR img_stream:" in text and "LR txt_stream:" in text


def test_fused_optimizer_groups_snapshot_the_flattened_rate():
    """``_setup_fused_optimizer_groups`` rebuilds every optimizer at the run's
    base LR; the snapshot must describe THAT, not the adapter's discarded rates
    -- and must cover all N optimizers, not just ``optimizers[0]``.
    """
    probe = _Probe(learning_rate=1e-4, unet_lr=2e-5, num_optimizer_groups=2)
    probe.optimizer = torch.optim.AdamW([{"params": _params(), "lr": 1e-4}])
    probe.fused_optimizer_groups = SimpleNamespace(
        optimizers=[probe.optimizer,
                    torch.optim.AdamW([{"params": _params(), "lr": 1e-4}])])
    probe._record_configured_group_lrs([2e-5])   # what the adapter had asked for
    assert probe._configured_group_lrs == [1e-4, 1e-4]


# ---------------------------------------------------------------------------
# The setup-time report
# ---------------------------------------------------------------------------

def test_warmup_does_not_look_like_a_dead_group_at_setup():
    """Pre-fix, the report compared the schedule-scaled ``lr``: with any warmup
    every aligned architecture emitted a spurious mismatch AND zero warning at
    step 0.
    """
    probe = _Probe(learning_rate=1e-5, unet_lr=1e-5, text_encoder_1_lr=5e-6)
    probe.unet = nn.Linear(2, 2)
    probe.text_encoder = nn.Linear(2, 2)
    probe.train_text_encoder = True
    _attach_optimizer(probe, [{"params": _params(), "lr": 1e-5},
                              {"params": _params(), "lr": 5e-6}],
                      lr_lambda=lambda step: min(1.0, step / 1000.0))
    assert _group_lrs(probe) == [0.0, 0.0]

    events, _ = _events(probe._report_effective_component_lrs, [1e-5, 5e-6])
    assert [e["code"] for e in events] == []

    # Negative control: dropping initial_lr leaves only the schedule-scaled lr,
    # which is what the pre-fix comparison read.
    for group in probe.optimizer.param_groups:
        group.pop("initial_lr")
    events, _ = _events(probe._report_effective_component_lrs, [1e-5, 5e-6])
    assert sorted({e["code"] for e in events}) == [
        "component_lr_mismatch", "component_lr_zero"]


def test_a_genuinely_zero_base_rate_is_still_reported():
    probe = _Probe(learning_rate=0.0, unet_lr=0.0)
    probe.unet = nn.Linear(2, 2)
    _attach_optimizer(probe, [{"params": _params(), "lr": 0.0}])
    events, _ = _events(probe._report_effective_component_lrs, [0.0])
    assert [e["code"] for e in events] == ["component_lr_zero"]


# ---------------------------------------------------------------------------
# Wiring, in the shipping source
# ---------------------------------------------------------------------------

def _setup_optimizer_body():
    source = Path(sys.modules[BaseTrainer.__module__].__file__).read_text(encoding="utf-8")
    body = source[source.index("    def setup_optimizer("):]
    return body[:body.index("\n    def ", 10)]


def test_the_snapshot_is_taken_after_every_path_that_can_replace_the_optimizer():
    body = _setup_optimizer_body()
    assert body.count("_record_configured_group_lrs(") == 1
    for earlier in ("_setup_fused_optimizer_groups(", "_attach_stochastic_rounding("):
        assert body.index("_record_configured_group_lrs(") > body.index(earlier)
    assert body.rindex("_setup_fused_backward_pass(") < body.index(
        "_record_configured_group_lrs(")
    assert (body.index("_record_configured_group_lrs(")
            < body.index("_report_effective_component_lrs("))


def test_the_resume_no_longer_passes_a_scalar_to_the_helper():
    """The broadcast, at source level: the exact fallback that caused it."""
    source = Path(sys.modules[BaseTrainer.__module__].__file__).read_text(encoding="utf-8")
    body = source[source.index("    def _reassert_config_lr_on_resume("):]
    body = body[:body.index("\n    def ", 10)]
    assert "component_lrs if component_lrs else self.learning_rate" not in body
    assert "_configured_component_lr_description(" in body
