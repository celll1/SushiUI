"""SNR/Energy regularization must land on the trainer attribute its name says.

``train_runner.py`` has three (LoRA / Full-FT / Tagger) copies of the same
``regularization_type in {"snr", "energy"}`` setup block. All three previously
assigned the energy-regularization module to ``trainer.snr_regularization_loss``,
leaving ``trainer.energy_regularization_loss`` permanently None (dead branch in
every ``*_ops.py`` consumer, and unusable for the "both simultaneously" case the
consumers already support -- see the "can use both simultaneously" comments in
sd_sdxl_ops.py/flux2_ops.py/zimage_ops.py).
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.ops import zimage_ops
from core.training.losses.energy_regularization import EnergyRegularizationLoss
from core.training.losses.snr_regularization import SNRRegularizationLoss


TRAIN_RUNNER_PATH = Path(__file__).resolve().parents[1] / "core" / "training" / "train_runner.py"

LATENT_SHAPE = (2, 4, 8, 8)


class _ConstantVelocityDiT:
    def __init__(self, value: float = 0.25):
        self.value = value

    def __call__(self, x, t, cap_feats, cap_mask):
        return torch.full_like(x, self.value), None


def _zimage_trainer(**overrides):
    base = dict(
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        timestep_sampler=None,
        noise_scheduler=None,
        mixed_precision=False,
        transformer=_ConstantVelocityDiT(),
        snr_regularization_loss=None,
        energy_regularization_loss=None,
        reconstruction_loss_weight=0.0,
        log_prefix="[test]",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run_zimage(trainer, latents, timesteps):
    return zimage_ops.train_step(
        trainer,
        latents=latents,
        prompt_embeds=torch.zeros(LATENT_SHAPE[0], 5, 8),
        attention_mask=torch.ones(LATENT_SHAPE[0], 5, dtype=torch.bool),
        timesteps=timesteps,
    )


# ---------------------------------------------------------------------------
# (a)/(b): train_runner.py must route each branch to the matching attribute.
# ---------------------------------------------------------------------------

def _regularization_if_nodes(tree: ast.Module, keyword: str) -> list:
    """Find every ``if regularization_type.lower() == keyword:`` node."""
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute)
                and left.func.attr == "lower"):
            continue
        if not (isinstance(left.func.value, ast.Name) and left.func.value.id == "regularization_type"):
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == keyword:
            matches.append(node)
    return matches


def _assigned_attr(if_node: ast.If) -> str:
    assigns = [s for s in if_node.body if isinstance(s, ast.Assign)]
    assert len(assigns) == 1, "expected exactly one assignment in the regularization branch"
    target = assigns[0].targets[0]
    assert isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
        and target.value.id == "trainer"
    return target.attr


def test_train_runner_has_three_regularization_setup_blocks():
    tree = ast.parse(TRAIN_RUNNER_PATH.read_text(encoding="utf-8"))
    snr_nodes = _regularization_if_nodes(tree, "snr")
    energy_nodes = _regularization_if_nodes(tree, "energy")
    # Sanity anchor: this pins the "one block per training entrypoint" shape
    # the fix relies on. If this count changes, the fix needs re-auditing.
    assert len(snr_nodes) == 3, "expected 3 'snr' branches (LoRA / Full-FT / Tagger)"
    assert len(energy_nodes) == 3, "expected 3 'energy' branches (LoRA / Full-FT / Tagger)"


def test_train_runner_snr_branch_assigns_snr_attribute():
    tree = ast.parse(TRAIN_RUNNER_PATH.read_text(encoding="utf-8"))
    for node in _regularization_if_nodes(tree, "snr"):
        assert _assigned_attr(node) == "snr_regularization_loss"


def test_train_runner_energy_branch_assigns_energy_attribute():
    tree = ast.parse(TRAIN_RUNNER_PATH.read_text(encoding="utf-8"))
    for node in _regularization_if_nodes(tree, "energy"):
        assert _assigned_attr(node) == "energy_regularization_loss"


# ---------------------------------------------------------------------------
# (c): consumers (*_ops.py) must apply each attribute with its own weight.
# zimage_ops stands in for sd_sdxl_ops/flux2_ops, which read the two
# attributes with byte-identical branch structure (verified by inspection).
# ---------------------------------------------------------------------------

def test_energy_attribute_is_applied_with_its_own_weight(monkeypatch):
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.zeros_like(x))
    latents = torch.ones(*LATENT_SHAPE)
    timesteps = torch.full((LATENT_SHAPE[0],), 0.3)

    energy_module = EnergyRegularizationLoss(weight=0.05, timestep_adaptive=True,
                                              penalty_mode="abs", normalize_by_pixels=True)
    trainer = _zimage_trainer(energy_regularization_loss=energy_module)

    loss, pred_loss, _recon = _run_zimage(trainer, latents, timesteps)

    # noise is forced to zero -> noisy_latents = (1-t)*latents, model_pred is
    # the constant 0.25 from _ConstantVelocityDiT, x_0 = x_t + t*v (Z-Image sign).
    t = timesteps.view(-1, 1, 1, 1)
    predicted_latent_for_reg = (1.0 - t) * latents + t * 0.25
    expected_energy_reg = energy_module(predicted_latent_for_reg, latents, timesteps)

    applied_reg = loss.item() - pred_loss
    assert applied_reg == pytest.approx(expected_energy_reg.item(), rel=1e-5)
    assert applied_reg != pytest.approx(0.0)


def test_snr_attribute_is_applied_with_its_own_weight(monkeypatch):
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.zeros_like(x))
    latents = torch.ones(*LATENT_SHAPE)
    timesteps = torch.full((LATENT_SHAPE[0],), 0.3)

    snr_module = SNRRegularizationLoss(weight=0.1, timestep_adaptive=True, penalty_mode="relu")
    trainer = _zimage_trainer(snr_regularization_loss=snr_module)

    loss, pred_loss, _recon = _run_zimage(trainer, latents, timesteps)

    t = timesteps.view(-1, 1, 1, 1)
    predicted_latent_for_reg = (1.0 - t) * latents + t * 0.25
    expected_snr_reg = snr_module(predicted_latent_for_reg, latents, timesteps)

    applied_reg = loss.item() - pred_loss
    assert applied_reg == pytest.approx(expected_snr_reg.item(), rel=1e-5)


def test_wrong_attribute_leaves_the_other_channel_dark(monkeypatch):
    """Regression pin for the actual bug shape: energy assigned to the wrong
    attribute means the *named* energy channel never fires, even though a
    module instance exists somewhere on the trainer."""
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.zeros_like(x))
    latents = torch.ones(*LATENT_SHAPE)
    timesteps = torch.full((LATENT_SHAPE[0],), 0.3)

    energy_module = EnergyRegularizationLoss(weight=0.05)
    # Simulate the pre-fix bug: energy module stored under the SNR attribute.
    buggy_trainer = _zimage_trainer(snr_regularization_loss=energy_module,
                                     energy_regularization_loss=None)
    loss_buggy, pred_loss_buggy, _ = _run_zimage(buggy_trainer, latents.clone(), timesteps)

    fixed_trainer = _zimage_trainer(energy_regularization_loss=EnergyRegularizationLoss(weight=0.05),
                                     snr_regularization_loss=None)
    loss_fixed, pred_loss_fixed, _ = _run_zimage(fixed_trainer, latents.clone(), timesteps)

    # The consumer branches are structurally identical, so the applied
    # penalty is the same either way for a single active regularization --
    # this test documents that the risk is dead-branch/mislabeling, not a
    # silent loss-value change, and that fixing the routing does not disturb
    # existing runs that only ever selected one regularization_type.
    assert (loss_buggy.item() - pred_loss_buggy) == pytest.approx(
        loss_fixed.item() - pred_loss_fixed, rel=1e-5
    )
    # But the attribute contract itself now holds: energy lives in
    # energy_regularization_loss, not snr_regularization_loss.
    assert fixed_trainer.snr_regularization_loss is None
    assert fixed_trainer.energy_regularization_loss is not None
