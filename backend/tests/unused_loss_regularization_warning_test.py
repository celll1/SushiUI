"""Coverage for BaseTrainer._warn_unused_loss_regularization_keys.

min_snr_gamma / snr_regularization_* / energy_regularization_* can be set in
the UI and training config for any architecture, but only a subset of the
per-architecture op modules ever read them (verified against
ops/sd_sdxl_ops.py, ops/flux2_ops.py, ops/zimage_ops.py, and every other
ops/*_ops.py, which have zero references). This warns once, in a single
block, when a configured key will have no effect -- and must never change
what the loss computes.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.base_trainer import BaseTrainer


def _fake_trainer(**overrides):
    base = dict(
        min_snr_gamma=0.0,
        snr_regularization_loss=None,
        energy_regularization_loss=None,
        use_condition_images=False,
        prediction_target="epsilon",
        log_prefix="[test]",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _warn(trainer, arch_name):
    BaseTrainer._warn_unused_loss_regularization_keys(trainer, arch_name)


# ---------------------------------------------------------------------------
# (a) SenseNova + min_snr_gamma: must warn (sensenova_ops.py never reads it).
# ---------------------------------------------------------------------------

def test_sensenova_min_snr_gamma_warns(capsys):
    trainer = _fake_trainer(min_snr_gamma=5.0)
    _warn(trainer, "sensenova")
    out = capsys.readouterr().out
    assert "min_snr_gamma=5.0" in out
    assert "sensenova" in out


def test_sensenova_snr_regularization_warns(capsys):
    trainer = _fake_trainer(snr_regularization_loss=object())
    _warn(trainer, "sensenova")
    out = capsys.readouterr().out
    assert "snr_regularization_*" in out


def test_sensenova_energy_regularization_warns(capsys):
    trainer = _fake_trainer(energy_regularization_loss=object())
    _warn(trainer, "sensenova")
    out = capsys.readouterr().out
    assert "energy_regularization_*" in out


def test_other_non_consuming_archs_warn_too(capsys):
    for arch in ("acestep", "anima", "ideogram4", "krea2", "lens", "ltx2",
                 "minimax_h3", "minit2i"):
        trainer = _fake_trainer(min_snr_gamma=5.0, snr_regularization_loss=object(),
                                 energy_regularization_loss=object())
        _warn(trainer, arch)
        out = capsys.readouterr().out
        assert "min_snr_gamma=5.0" in out, arch
        assert "snr_regularization_*" in out, arch
        assert "energy_regularization_*" in out, arch


# ---------------------------------------------------------------------------
# (b) Actually-consumed combinations must NOT warn.
# ---------------------------------------------------------------------------

def test_sd15_min_snr_gamma_epsilon_does_not_warn(capsys):
    trainer = _fake_trainer(min_snr_gamma=5.0, prediction_target="epsilon")
    _warn(trainer, "sd15")
    assert capsys.readouterr().out == ""


def test_sdxl_min_snr_gamma_epsilon_does_not_warn(capsys):
    trainer = _fake_trainer(min_snr_gamma=5.0, prediction_target="epsilon")
    _warn(trainer, "sdxl")
    assert capsys.readouterr().out == ""


def test_sd15_min_snr_gamma_velocity_warns(capsys):
    """min_snr_gamma is gated on prediction_target=='epsilon' even for sd15."""
    trainer = _fake_trainer(min_snr_gamma=5.0, prediction_target="velocity")
    _warn(trainer, "sd15")
    out = capsys.readouterr().out
    assert "prediction_target='epsilon'" in out
    assert "prediction_target='velocity'" in out


def test_sd_sdxl_flux2_zimage_regularization_does_not_warn(capsys):
    for arch in ("sd15", "sdxl", "flux2", "zimage"):
        trainer = _fake_trainer(snr_regularization_loss=object(), energy_regularization_loss=object())
        _warn(trainer, arch)
        assert capsys.readouterr().out == "", arch


def test_controlnet_min_snr_gamma_epsilon_applies_to_zimage_and_flux2(capsys):
    for arch in ("sd15", "sdxl", "zimage", "flux2"):
        trainer = _fake_trainer(min_snr_gamma=5.0, prediction_target="epsilon",
                                 use_condition_images=True)
        _warn(trainer, arch)
        assert capsys.readouterr().out == "", arch


def test_controlnet_regularization_never_applies(capsys):
    """train_step_controlnet never reads snr/energy_regularization_loss,
    regardless of architecture."""
    for arch in ("sd15", "sdxl", "zimage", "flux2"):
        trainer = _fake_trainer(snr_regularization_loss=object(),
                                 energy_regularization_loss=object(),
                                 use_condition_images=True)
        _warn(trainer, arch)
        out = capsys.readouterr().out
        assert "snr_regularization_*" in out, arch
        assert "energy_regularization_*" in out, arch
        assert "ControlNet training" in out, arch


def test_disabled_keys_never_warn(capsys):
    trainer = _fake_trainer()  # all defaults: 0.0 / None / None
    for arch in ("sensenova", "sd15", "sdxl", "flux2", "zimage", "acestep"):
        _warn(trainer, arch)
        assert capsys.readouterr().out == "", arch


# ---------------------------------------------------------------------------
# (c) The warning is print-only: it must not touch the trainer's loss state.
# ---------------------------------------------------------------------------

def test_warning_does_not_mutate_trainer_state(capsys):
    snr_module = object()
    energy_module = object()
    trainer = _fake_trainer(
        min_snr_gamma=5.0,
        snr_regularization_loss=snr_module,
        energy_regularization_loss=energy_module,
        prediction_target="velocity",
    )
    before = dict(vars(trainer))
    _warn(trainer, "sensenova")
    capsys.readouterr()
    after = dict(vars(trainer))
    assert before == after
    assert trainer.min_snr_gamma == 5.0
    assert trainer.snr_regularization_loss is snr_module
    assert trainer.energy_regularization_loss is energy_module
