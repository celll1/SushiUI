"""``debug_latents`` dumps must never take a training run down.

The dump is a diagnostic that only runs every ``debug_latents_every`` steps, so a
full disk / permission error / unexpected shape inside it must not kill a run
that has been going for hours. Z-Image stands in for the image archs
behaviourally; the AST check pins the invariant for all nine ``train_step``s,
including the ones with no cheap stand-in harness.
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


OPS_DIR = Path(__file__).resolve().parents[1] / "core" / "training" / "ops"

DUMPING_OPS = (
    "sd_sdxl_ops.py", "zimage_ops.py", "flux2_ops.py", "minit2i_ops.py",
    "sensenova_ops.py", "anima_ops.py", "ltx2_ops.py", "minimax_h3_ops.py",
    "acestep_ops.py",
)

LATENT_SHAPE = (2, 4, 8, 8)


class _ConstantVelocityDiT:
    def __init__(self, value: float = 0.25):
        self.value = value

    def __call__(self, x, t, cap_feats, cap_mask):
        return torch.full_like(x, self.value), None


def _zimage_trainer():
    return SimpleNamespace(
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


def _run_zimage(out, sigma=0.3):
    latents = torch.randn(*LATENT_SHAPE)
    return zimage_ops.train_step(
        _zimage_trainer(),
        latents=latents,
        prompt_embeds=torch.zeros(LATENT_SHAPE[0], 5, 8),
        attention_mask=torch.ones(LATENT_SHAPE[0], 5, dtype=torch.bool),
        timesteps=torch.full((LATENT_SHAPE[0],), sigma),
        debug_save_path=out,
    )


def test_zimage_dump_is_written(tmp_path):
    """Sanity anchor: without it the failure test below could pass vacuously."""
    out = tmp_path / "step_000010"
    loss, pred_loss, _recon = _run_zimage(out)
    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert data["scheduler_type"] == "FlowMatching"
    assert torch.isfinite(loss)


def test_zimage_dump_failure_does_not_break_train_step(tmp_path, monkeypatch, capsys):
    def _boom(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(torch, "save", _boom)
    out = tmp_path / "step_000010"

    loss, pred_loss, recon_loss = _run_zimage(out)

    assert torch.isfinite(loss)
    assert isinstance(pred_loss, float) and isinstance(recon_loss, float)
    assert not list(out.glob("latents_t*.pt"))

    # Never silent: an empty debug dir with no explanation is the failure mode
    # this guard is here to avoid.
    err = capsys.readouterr().out
    assert "[debug_latents] save failed" in err
    assert "No space left on device" in err


def test_dump_failure_does_not_swallow_keyboard_interrupt(tmp_path, monkeypatch):
    """A run interrupt outranks the dump: ``except Exception`` must let it out."""
    def _interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run_zimage(tmp_path / "step_000010")


@pytest.mark.parametrize("filename", DUMPING_OPS)
def test_every_dump_block_is_exception_guarded(filename):
    """``if debug_save_path is not None:`` must open directly onto a ``try``.

    Structural rather than behavioural so the archs without a cheap stand-in
    harness (sd_sdxl, flux2, minit2i, sensenova) are covered too.
    """
    tree = ast.parse((OPS_DIR / filename).read_text(encoding="utf-8"))

    blocks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "debug_save_path"
        and isinstance(node.test.ops[0], ast.IsNot)
    ]
    assert blocks, f"{filename}: no debug_save_path dump block found"

    for block in blocks:
        assert isinstance(block.body[0], ast.Try), (
            f"{filename}: dump block at line {block.lineno} is not exception-guarded"
        )
        handlers = block.body[0].handlers
        assert len(handlers) == 1
        # Exception, not BaseException: KeyboardInterrupt/SystemExit must pass.
        assert isinstance(handlers[0].type, ast.Name)
        assert handlers[0].type.id == "Exception"
        # The handler must report; a bare pass is the silent-failure pattern.
        assert any(isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                   and getattr(stmt.value.func, "id", None) == "print"
                   for stmt in handlers[0].body), (
            f"{filename}: dump guard at line {block.lineno} does not warn"
        )
