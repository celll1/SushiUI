"""Per-MNT-iteration redraw of the aligned-CFG-null label.

Without this, one Bernoulli draw is shared by every MNT transform of a batch,
so at multi_noise_timesteps=N a single null draw trains N CONSECUTIVE
optimizer steps null on the same image (the clustering the chart showed as
16 consecutive `Loss (null)` points at N=8). `cfg_drop_mask_for_mnt` redraws
the label independently per iteration instead -- same expected rate, no
clustering. CPU-only, no CUDA, no checkpoints, no model load.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/cfg_null_per_mnt_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.base_trainer import BaseTrainer  # noqa: E402


class _Trainer:
    cfg_drop_mask_for_mnt = BaseTrainer.cfg_drop_mask_for_mnt
    cfg_null_drop_rate = BaseTrainer.cfg_null_drop_rate
    log_prefix = "[cfg-null-per-mnt-test]"
    sensenova_four_phase = None

    def __init__(self, rate, per_mnt=True):
        self._cfg_null_drop_rate_resolved = rate
        self.config = {"cfg_uncond_drop_per_mnt": per_mnt}


# ---------------------------------------------------------------------------
# mnt_index == 0 always keeps the assembly draw
# ---------------------------------------------------------------------------

def test_mnt_index_zero_returns_the_batch_mask_unchanged():
    trainer = _Trainer(rate=0.5)
    batch_mask = torch.tensor([True, False, True])
    result = trainer.cfg_drop_mask_for_mnt(batch_mask, 0, 3)
    assert result is batch_mask


def test_mnt_index_zero_is_unchanged_even_with_per_mnt_off():
    trainer = _Trainer(rate=0.5, per_mnt=False)
    batch_mask = torch.tensor([True])
    assert trainer.cfg_drop_mask_for_mnt(batch_mask, 0, 1) is batch_mask


# ---------------------------------------------------------------------------
# rate falsy / no label at all
# ---------------------------------------------------------------------------

def test_a_none_mask_stays_none_regardless_of_mnt_index():
    """A falsy rate never draws a batch mask in the first place
    (sample_cfg_drop_mask returns None); this is that None flowing through
    every MNT iteration with nothing to redraw."""
    for rate in (None, 0.0):
        trainer = _Trainer(rate=rate)
        for mnt_index in (0, 1, 4, 7):
            assert trainer.cfg_drop_mask_for_mnt(None, mnt_index, 4) is None


def test_a_zero_rate_never_redraws_even_if_a_mask_were_passed():
    """Defensive: cfg_null_drop_rate() resolving to 0 after a mask somehow
    exists must not fabricate a new draw -- the mechanism is off."""
    trainer = _Trainer(rate=0.0)
    batch_mask = torch.tensor([True, True])
    result = trainer.cfg_drop_mask_for_mnt(batch_mask, 1, 2)
    assert result is batch_mask


# ---------------------------------------------------------------------------
# per_mnt=False reproduces the shared-window behaviour exactly
# ---------------------------------------------------------------------------

def test_disabled_reproduces_the_old_shared_window_behaviour():
    torch.manual_seed(0)
    trainer = _Trainer(rate=0.5, per_mnt=False)
    batch_mask = torch.tensor([True, False, True, False])
    draws = [trainer.cfg_drop_mask_for_mnt(batch_mask, i, 4) for i in range(8)]
    assert all(d is batch_mask for d in draws)


# ---------------------------------------------------------------------------
# per_mnt=True: independent draws, not a shared window
# ---------------------------------------------------------------------------

def test_the_eight_mnt_iterations_are_not_all_identical():
    torch.manual_seed(1234)
    trainer = _Trainer(rate=0.5)
    batch_mask = torch.tensor([True])

    found_mixed_window = False
    for _ in range(500):
        draws = [
            bool(trainer.cfg_drop_mask_for_mnt(batch_mask, i, 1).item())
            for i in range(8)
        ]
        if len(set(draws)) > 1:
            found_mixed_window = True
            break
    assert found_mixed_window, "500 windows of 8 draws at rate=0.5 were all uniform"


def test_a_disabled_batch_size_one_run_never_redraws():
    """Same setup as above but per_mnt=False: the negative control that the
    mixing above is this setting's doing, not batch_size=1 noise."""
    torch.manual_seed(1234)
    trainer = _Trainer(rate=0.5, per_mnt=False)
    batch_mask = torch.tensor([True])
    for _ in range(500):
        draws = [
            bool(trainer.cfg_drop_mask_for_mnt(batch_mask, i, 1).item())
            for i in range(8)
        ]
        assert len(set(draws)) == 1


# ---------------------------------------------------------------------------
# Long-run frequency: same nominal rate, just no longer window-clustered
# ---------------------------------------------------------------------------

def test_the_empirical_null_fraction_converges_to_the_configured_rate():
    torch.manual_seed(7)
    rate = 0.3
    trainer = _Trainer(rate=rate)
    batch_mask = torch.tensor([False])  # mnt_index 0's label, irrelevant here

    n_draws = 20000
    hits = sum(
        bool(trainer.cfg_drop_mask_for_mnt(batch_mask, 1, 1).item())
        for _ in range(n_draws)
    )
    empirical = hits / n_draws
    assert abs(empirical - rate) < 0.02, (empirical, rate)


# ---------------------------------------------------------------------------
# SenseNova's four-phase shared-window route: forced off, warned once
# ---------------------------------------------------------------------------

def test_the_shared_window_route_keeps_the_batch_label_regardless_of_the_setting(capsys):
    trainer = _Trainer(rate=0.5)
    trainer.sensenova_four_phase = SimpleNamespace(shared_window=True)
    batch_mask = torch.tensor([True])

    draws = [trainer.cfg_drop_mask_for_mnt(batch_mask, i, 1) for i in range(4)]
    assert all(d is batch_mask for d in draws)
    out = capsys.readouterr().out
    assert "four-phase shared-window route" in out
    assert '"code": "cfg_null_per_mnt_shared_window"' in out


def test_the_shared_window_warning_fires_once_per_trainer_not_once_per_call(capsys):
    trainer = _Trainer(rate=0.5)
    trainer.sensenova_four_phase = SimpleNamespace(shared_window=True)
    batch_mask = torch.tensor([True])

    for i in range(1, 6):
        trainer.cfg_drop_mask_for_mnt(batch_mask, i, 1)
    out = capsys.readouterr().out
    assert out.count('"code": "cfg_null_per_mnt_shared_window"') == 1
    assert trainer._cfg_null_per_mnt_shared_window_warned is True

    # A different trainer instance is a different run and must warn again --
    # the flag lives on the instance, not shared trainer-class state.
    trainer2 = _Trainer(rate=0.5)
    trainer2.sensenova_four_phase = SimpleNamespace(shared_window=True)
    trainer2.cfg_drop_mask_for_mnt(batch_mask, 1, 1)
    out2 = capsys.readouterr().out
    assert out2.count('"code": "cfg_null_per_mnt_shared_window"') == 1


def test_a_non_shared_four_phase_route_still_redraws():
    """Only shared_window forces the old behaviour; a plain per-phase
    SenseNova run (shared_window=False) redraws like every other arch."""
    torch.manual_seed(42)
    trainer = _Trainer(rate=0.5)
    trainer.sensenova_four_phase = SimpleNamespace(shared_window=False)
    batch_mask = torch.tensor([True])

    found_mixed_window = False
    for _ in range(500):
        draws = [
            bool(trainer.cfg_drop_mask_for_mnt(batch_mask, i, 1).item())
            for i in range(8)
        ]
        if len(set(draws)) > 1:
            found_mixed_window = True
            break
    assert found_mixed_window


# ---------------------------------------------------------------------------
# FIX 5: a stale/never-set _sensenova_prefix_cfg_null raises, not defaults
# ---------------------------------------------------------------------------

def test_a_none_prefix_cfg_null_raises_rather_than_defaulting():
    """A None _sensenova_prefix_cfg_null at mnt_index > 0 must not be read as
    False -- that would silently pick the wrong prefix while the loss is
    logged under the other label."""
    class _SensenovaFrozenTrainer:
        train_text_encoder = False
        sensenova_four_phase = None
        _sensenova_mnt_conditioning = BaseTrainer._sensenova_mnt_conditioning

        def encode_caption(self, caption, requires_grad=False, cfg_null=False):
            raise AssertionError("must not reach an encode call")

    trainer = _SensenovaFrozenTrainer()
    with pytest.raises(RuntimeError, match="_sensenova_prefix_cfg_null"):
        trainer._sensenova_mnt_conditioning(
            "assembly", captions=["a cat"], mnt_index=1, cfg_null=True)


def test_a_set_prefix_cfg_null_of_false_does_not_raise():
    """False is a legitimate label, not a missing one -- only None is."""
    class _SensenovaFrozenTrainer:
        train_text_encoder = False
        sensenova_four_phase = None
        _sensenova_mnt_conditioning = BaseTrainer._sensenova_mnt_conditioning
        _sensenova_prefix_cfg_null = False
        _sensenova_alt_cfg_null_prefix = None

        def encode_caption(self, caption, requires_grad=False, cfg_null=False):
            return "rebuilt", None

    trainer = _SensenovaFrozenTrainer()
    result = trainer._sensenova_mnt_conditioning(
        "assembly", captions=["a cat"], mnt_index=1, cfg_null=False)
    assert result[3] == "assembly"


# ---------------------------------------------------------------------------
# FIX 4b: the phase-eviction disclosure, gated on all three conditions
# ---------------------------------------------------------------------------

class _StubSenseNovaArch:
    name = "sensenova"
    cfg_null_stage = "encode"


class _SenseNovaTrainer:
    cfg_null_drop_rate = BaseTrainer.cfg_null_drop_rate
    log_prefix = "[cfg-null-per-mnt-test]"

    def __init__(self, rate, per_mnt=True, mnt=3, phase_evictor=None):
        self.config = {
            "cfg_uncond_drop_rate": rate,
            "minit2i_label_drop_rate": None,
            "danbooru_aug_enable": False,
            "danbooru_aug_caption_dropout_rate": 0.0,
            "multi_noise_timesteps": mnt,
            "cfg_uncond_drop_per_mnt": per_mnt,
        }
        self.arch = _StubSenseNovaArch()
        self.sensenova_phase_evictor = phase_evictor


def test_the_phase_eviction_disclosure_fires_once_when_all_conditions_hold(capsys):
    trainer = _SenseNovaTrainer(rate=0.1, per_mnt=True, mnt=3, phase_evictor=object())
    trainer.cfg_null_drop_rate()
    out = capsys.readouterr().out
    assert out.count('"code": "cfg_null_per_mnt_phase_eviction"') == 1
    assert trainer._cfg_null_per_mnt_phase_eviction_warned is True

    # cfg_null_drop_rate() memoizes its resolution, so a repeat call must not
    # emit the disclosure again.
    trainer.cfg_null_drop_rate()
    assert capsys.readouterr().out == ""


def test_the_phase_eviction_disclosure_is_silent_without_a_phase_evictor():
    trainer = _SenseNovaTrainer(rate=0.1, per_mnt=True, mnt=3, phase_evictor=None)
    trainer.cfg_null_drop_rate()
    assert not hasattr(trainer, "_cfg_null_per_mnt_phase_eviction_warned")


def test_the_phase_eviction_disclosure_is_silent_with_per_mnt_off():
    trainer = _SenseNovaTrainer(rate=0.1, per_mnt=False, mnt=3, phase_evictor=object())
    trainer.cfg_null_drop_rate()
    assert not hasattr(trainer, "_cfg_null_per_mnt_phase_eviction_warned")


def test_the_phase_eviction_disclosure_is_silent_at_zero_rate():
    trainer = _SenseNovaTrainer(rate=0.0, per_mnt=True, mnt=3, phase_evictor=object())
    trainer.cfg_null_drop_rate()
    assert not hasattr(trainer, "_cfg_null_per_mnt_phase_eviction_warned")


def test_the_phase_eviction_disclosure_is_silent_for_another_architecture():
    trainer = _SenseNovaTrainer(rate=0.1, per_mnt=True, mnt=3, phase_evictor=object())
    trainer.arch = SimpleNamespace(name="minit2i", cfg_null_stage="collated")
    trainer.cfg_null_drop_rate()
    assert not hasattr(trainer, "_cfg_null_per_mnt_phase_eviction_warned")
