"""The crop decode auxiliary loss must reach the weights, not only the log.

SD1.5 / SDXL / Z-Image built the x_0 the aux loss decodes inside ``no_grad``
whenever no regularization and no dual reconstruction loss was on -- the one
configuration where the crop decode loss is the *only* extra term. ``loss +
aux_loss`` then added a constant, and the op's own grad-ratio probe raised
inside its bare ``except``, so the run silently lost the metric too.

Each test runs the architecture's ``train_step`` twice, once with the aux loss
on and once off, and compares the gradient the fake network receives. The
pre-fix modules are loaded out of git and asserted to show the defect, so the
red-before/green-after pair stays executable after the fix is committed.
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.ops import sd_sdxl_ops, zimage_ops
from core.training.losses.snr_regularization import SNRRegularizationLoss

REPO = Path(__file__).resolve().parents[2]
PRE_FIX_COMMIT = "095206c0"  # the ops sources as they were before this fix



def _load_prefix(rel_path: str, mod_name: str, tmp_dir: Path):
    source = subprocess.run(
        ["git", "show", f"{PRE_FIX_COMMIT}:{rel_path}"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    path = tmp_dir / f"{mod_name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    # The ops modules use relative imports (``from .training_method import ...``),
    # which resolve through __package__, not the module's own name.
    module.__package__ = "core.training.ops"
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prefix_ops(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("prefix_ops")
    mods = {
        "zimage": _load_prefix("backend/core/training/ops/zimage_ops.py",
                               "_prefix_zimage_ops", tmp_dir),
        "sd_sdxl": _load_prefix("backend/core/training/ops/sd_sdxl_ops.py",
                                "_prefix_sd_sdxl_ops", tmp_dir),
    }
    yield mods
    for name in ("_prefix_zimage_ops", "_prefix_sd_sdxl_ops"):
        sys.modules.pop(name, None)



class _MockVAE(nn.Module):
    def __init__(self, latent_channels: int, scale: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(latent_channels, 3, kernel_size=3, padding=1)
        self.scale = scale
        self.config = SimpleNamespace(
            latent_channels=latent_channels,
            scaling_factor=0.18215,
            shift_factor=0.0,
            # len - 1 upsample stages == self.scale (spatial_compression_of)
            block_out_channels=[64, 128, 256, 512],
        )

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        out = self.conv(F.interpolate(z, scale_factor=float(self.scale), mode="nearest"))
        return (out,) if not return_dict else SimpleNamespace(sample=out)


class _ZImageDiT(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, t, cap_feats, cap_mask):
        return self.conv(x.squeeze(2)).unsqueeze(2), None


class _UNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None):
        return SimpleNamespace(sample=self.conv(sample))


def _trainer(net_attr: str, net, vae, *, crop_on: bool, recon_weight: float = 0.0,
             snr_reg=None, **extra):
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        vae_dtype=torch.float32,
        vae=vae,
        timestep_sampler=None,
        mixed_precision=False,
        snr_regularization_loss=snr_reg,
        energy_regularization_loss=None,
        reconstruction_loss_weight=recon_weight,
        crop_decode_loss_enable=crop_on,
        crop_decode_loss_weight=0.5 if crop_on else 0.0,
        crop_decode_loss_metric="mse",
        crop_decode_loss_margin_cells=4,
        crop_decode_loss_out_cells=8,
        crop_decode_loss_snr_range="",
        log_prefix="[test]",
        metrics={},
        **extra,
    )
    setattr(trainer, net_attr, net)
    trainer.log_extra_metric = lambda k, v: trainer.metrics.__setitem__(k, v)
    return trainer



SEED = 1234


def _run_zimage(module, net, vae, **cfg):
    trainer = _trainer("transformer", net, vae, noise_scheduler=None, **cfg)
    torch.manual_seed(SEED)
    latents = torch.randn(1, 16, 16, 16)
    out = module.train_step(
        trainer,
        latents=latents,
        prompt_embeds=torch.zeros(1, 4, 8),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    return out[0], trainer


def _sd_runner(is_sdxl: bool):
    def _run(module, net, vae, **cfg):
        from diffusers import DDPMScheduler
        trainer = _trainer(
            "unet", net, vae,
            noise_scheduler=DDPMScheduler(num_train_timesteps=1000),
            is_sdxl=is_sdxl, min_snr_gamma=0.0, **cfg)
        torch.manual_seed(SEED)
        latents = torch.randn(1, 4, 16, 16)
        kwargs = {}
        if is_sdxl:
            kwargs["pooled_embeddings"] = torch.zeros(1, 1280)
        out = module.train_step(
            trainer,
            latents=latents,
            text_embeddings=torch.zeros(1, 4, 2048 if is_sdxl else 768),
            **kwargs,
        )
        return out[0], trainer
    return _run


ARCHS = {
    "zimage": ("zimage", lambda: _ZImageDiT(16), lambda: _MockVAE(16), _run_zimage),
    "sd15": ("sd_sdxl", lambda: _UNet(4), lambda: _MockVAE(4), _sd_runner(False)),
    "sdxl": ("sd_sdxl", lambda: _UNet(4), lambda: _MockVAE(4), _sd_runner(True)),
}


def _grads(module, arch: str, **cfg):
    """Run one train_step and return (flat grad on the network, trainer)."""
    _mod_key, make_net, make_vae, run = ARCHS[arch]
    torch.manual_seed(7)
    net, vae = make_net(), make_vae()
    loss, trainer = run(module, net, vae, **cfg)
    loss.backward()
    flat = torch.cat([p.grad.flatten() for p in net.parameters()])
    return loss.item(), flat, trainer


# ---------------------------------------------------------------------------
# The invariant: an enabled aux term with a positive weight moves the gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", list(ARCHS))
def test_crop_decode_loss_changes_the_gradient(arch):
    loss_on, g_on, tr_on = _grads(zimage_ops if arch == "zimage" else sd_sdxl_ops,
                                  arch, crop_on=True)
    loss_off, g_off, _ = _grads(zimage_ops if arch == "zimage" else sd_sdxl_ops,
                                arch, crop_on=False)

    assert loss_on != loss_off, "aux loss must be in the reported total"
    assert not torch.allclose(g_on, g_off), (
        "the crop decode loss added a constant: the network gradient is "
        "identical to a run with the aux loss disabled")
    assert torch.isfinite(g_on).all()
    assert "crop_decode_loss" in tr_on.metrics
    # The probe differentiates the aux loss w.r.t. model_pred; it can only
    # succeed if the aux loss depends on the prediction at all.
    assert "crop_decode_grad_norm_ratio" in tr_on.metrics
    assert tr_on.metrics["crop_decode_grad_norm_ratio"] > 0.0


@pytest.mark.parametrize("arch", list(ARCHS))
def test_the_pre_fix_module_shows_the_defect(prefix_ops, arch):
    """Red-before, kept executable: the shipped-then source added a constant."""
    module = prefix_ops["zimage" if arch == "zimage" else "sd_sdxl"]
    loss_on, g_on, tr_on = _grads(module, arch, crop_on=True)
    loss_off, g_off, _ = _grads(module, arch, crop_on=False)

    assert loss_on != loss_off, "the aux value did reach the log"
    assert torch.equal(g_on, g_off), "expected the pre-fix constant-add"
    assert "crop_decode_loss" in tr_on.metrics
    assert "crop_decode_grad_norm_ratio" not in tr_on.metrics



INVARIANT_CONFIGS = {
    # crop decode off: the fix must not be observable at all
    "plain": dict(crop_on=False),
    "dual_recon": dict(crop_on=False, recon_weight=0.3),
    "snr_reg": dict(crop_on=False),
    # crop decode on: same number as before, only the graph is new
    "crop_only": dict(crop_on=True),
    "crop_and_recon": dict(crop_on=True, recon_weight=0.3),
    "crop_and_reg": dict(crop_on=True),
}


@pytest.mark.parametrize("arch", list(ARCHS))
@pytest.mark.parametrize("config", list(INVARIANT_CONFIGS))
def test_loss_value_is_unchanged_from_the_pre_fix_module(prefix_ops, arch, config):
    cfg = dict(INVARIANT_CONFIGS[config])
    if config in ("snr_reg", "crop_and_reg"):
        cfg["snr_reg"] = SNRRegularizationLoss(weight=0.1, timestep_adaptive=True,
                                               penalty_mode="relu")
    now = _grads(zimage_ops if arch == "zimage" else sd_sdxl_ops, arch, **cfg)[0]
    if config in ("snr_reg", "crop_and_reg"):
        cfg["snr_reg"] = SNRRegularizationLoss(weight=0.1, timestep_adaptive=True,
                                               penalty_mode="relu")
    before = _grads(prefix_ops["zimage" if arch == "zimage" else "sd_sdxl"],
                    arch, **cfg)[0]
    assert now == before


@pytest.mark.parametrize("arch", list(ARCHS))
@pytest.mark.parametrize("config", ["plain", "dual_recon"])
def test_gradients_are_unchanged_where_the_aux_loss_is_off(prefix_ops, arch, config):
    cfg = dict(INVARIANT_CONFIGS[config])
    _, g_now, _ = _grads(zimage_ops if arch == "zimage" else sd_sdxl_ops, arch, **cfg)
    _, g_before, _ = _grads(prefix_ops["zimage" if arch == "zimage" else "sd_sdxl"],
                            arch, **cfg)
    assert torch.equal(g_now, g_before)
