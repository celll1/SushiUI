"""Dual reconstruction loss must reach the weights, not only the log.

Anima / ACE-Step / LTX-2.3 computed ``recon_loss`` under ``no_grad`` right after
gating on ``reconstruction_loss_weight > 0``, then added it to the backward
loss -- a constant, so the dual-loss half of the objective never trained
anything. SD/SDXL and Z-Image gate the same way without ``no_grad`` and are the
shape these three now follow. At weight == 0 these three compute no
reconstruction term at all, and that path is asserted to stay bit-identical.

The pre-fix modules are loaded out of git and asserted to show the defect, so
the red-before/green-after pair stays executable after the fix is committed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.ops import acestep_ops, anima_ops, ltx2_ops

REPO = Path(__file__).resolve().parents[2]
PRE_FIX_COMMIT = "095206c0"  # the ops sources as they were before this fix

RECON_WEIGHT = 0.3


def _load_prefix(rel_path: str, mod_name: str, tmp_dir: Path):
    source = subprocess.run(
        ["git", "show", f"{PRE_FIX_COMMIT}:{rel_path}"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    path = tmp_dir / f"{mod_name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    # The ops modules use relative imports, which resolve through __package__.
    module.__package__ = "core.training.ops"
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prefix_ops(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("prefix_dual_ops")
    names = ("anima", "acestep", "ltx2")
    mods = {n: _load_prefix(f"backend/core/training/ops/{n}_ops.py",
                            f"_prefix_{n}_ops", tmp_dir) for n in names}
    yield mods
    for n in names:
        sys.modules.pop(f"_prefix_{n}_ops", None)


# ---------------------------------------------------------------------------
# Stand-in networks: one trainable projection is enough
# ---------------------------------------------------------------------------

class _AnimaDiT(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, timesteps, context, padding_mask, target_input_ids,
                target_attention_mask, source_attention_mask):
        return self.conv(x.squeeze(2)).unsqueeze(2)


class _AceDecoder(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.proj = nn.Linear(channels, channels)

    def forward(self, hidden_states, **kwargs):
        return (self.proj(hidden_states),)


class _AceDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _AceDecoder()

    def prepare_condition(self, hidden_states, **kwargs):
        b = hidden_states.shape[0]
        return (torch.zeros(b, 4, 8), torch.ones(b, 4, dtype=torch.bool),
                hidden_states.detach())


class _Rope:
    def prepare_video_coords(self, b, t, h, w, device, fps=1.0):
        return torch.zeros(b, 3, t * h * w, 2, device=device)


class _Ltx2DiT(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.proj = nn.Linear(channels, channels)
        self.rope = _Rope()

    def forward(self, hidden_states, audio_hidden_states=None, **kwargs):
        return self.proj(hidden_states), audio_hidden_states


def _trainer(net, recon_weight: float, **extra):
    return SimpleNamespace(
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        timestep_sampler=None,
        noise_scheduler=None,
        mixed_precision=False,
        transformer=net,
        reconstruction_loss_weight=recon_weight,
        log_prefix="[test]",
        **extra,
    )


# ---------------------------------------------------------------------------
# One runner per architecture
# ---------------------------------------------------------------------------

SEED = 4321


def _run_anima(module, recon_weight):
    net = _AnimaDiT(16)
    trainer = _trainer(net, recon_weight)
    torch.manual_seed(SEED)
    latents = torch.randn(1, 16, 16, 16)
    out = module.train_step(
        trainer,
        latents=latents,
        prompt_embeds=torch.zeros(1, 4, 8),
        anima_aux={
            "source_mask": torch.ones(1, 4, dtype=torch.bool),
            "t5_input_ids": torch.zeros(1, 4, dtype=torch.long),
            "t5_attn_mask": torch.ones(1, 4, dtype=torch.bool),
        },
    )
    return out, net


def _run_acestep(module, recon_weight):
    net = _AceDiT()
    trainer = _trainer(net, recon_weight,
                       acestep_silence_latent=torch.zeros(1, 750, 64))
    torch.manual_seed(SEED)
    latents = torch.randn(1, 32, 64)
    out = module.train_step(
        trainer,
        latents=latents,
        text_embeddings=torch.zeros(1, 4, 8),
        aux={
            "text_attention_mask": torch.ones(1, 4, dtype=torch.bool),
            "lyric_hidden_states": torch.zeros(1, 4, 8),
            "lyric_attention_mask": torch.ones(1, 4, dtype=torch.bool),
        },
    )
    return out, net


def _run_ltx2(module, recon_weight):
    net = _Ltx2DiT(128)
    trainer = _trainer(net, recon_weight)
    torch.manual_seed(SEED)
    latents = torch.randn(1, 128, 2, 4, 4)
    out = module.train_step(
        trainer,
        latents=latents,
        prompt_embeds=torch.zeros(1, 4, 8),
        ltx2_aux={
            "audio_text_embedding": torch.zeros(1, 4, 8),
            "mask": torch.ones(1, 4, dtype=torch.bool),
        },
    )
    return out, net


RUNNERS = {"anima": _run_anima, "acestep": _run_acestep, "ltx2": _run_ltx2}
LIVE = {"anima": anima_ops, "acestep": acestep_ops, "ltx2": ltx2_ops}


def _step(module, arch: str, recon_weight: float):
    torch.manual_seed(11)
    (loss, pred_value, recon_value), net = RUNNERS[arch](module, recon_weight)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in net.parameters()])
    return loss.item(), pred_value, recon_value, grad


@pytest.mark.parametrize("arch", list(RUNNERS))
def test_dual_reconstruction_loss_changes_the_gradient(arch):
    loss_dual, _, recon_value, g_dual = _step(LIVE[arch], arch, RECON_WEIGHT)
    loss_plain, _, _, g_plain = _step(LIVE[arch], arch, 0.0)

    assert recon_value > 0.0, "the reconstruction value must still be reported"
    assert loss_dual != loss_plain
    assert not torch.allclose(g_dual, g_plain), (
        "the reconstruction term added a constant: the network gradient is "
        "identical to a run with the dual loss disabled")
    assert torch.isfinite(g_dual).all()


@pytest.mark.parametrize("arch", list(RUNNERS))
def test_the_pre_fix_module_shows_the_defect(prefix_ops, arch):
    """Red-before, kept executable: the shipped-then source added a constant."""
    loss_dual, _, recon_value, g_dual = _step(prefix_ops[arch], arch, RECON_WEIGHT)
    loss_plain, _, _, g_plain = _step(prefix_ops[arch], arch, 0.0)

    assert recon_value > 0.0 and loss_dual != loss_plain, "it did reach the log"
    assert torch.equal(g_dual, g_plain), "expected the pre-fix constant-add"


@pytest.mark.parametrize("arch", list(RUNNERS))
@pytest.mark.parametrize("recon_weight", [0.0, RECON_WEIGHT])
def test_reported_values_are_unchanged_from_the_pre_fix_module(
        prefix_ops, arch, recon_weight):
    now = _step(LIVE[arch], arch, recon_weight)
    before = _step(prefix_ops[arch], arch, recon_weight)
    assert now[:3] == before[:3]


@pytest.mark.parametrize("arch", list(RUNNERS))
def test_the_monitoring_path_is_untouched(prefix_ops, arch):
    """weight == 0 keeps its pre-fix gradient, bit for bit."""
    _, _, _, g_now = _step(LIVE[arch], arch, 0.0)
    _, _, _, g_before = _step(prefix_ops[arch], arch, 0.0)
    assert torch.equal(g_now, g_before)
