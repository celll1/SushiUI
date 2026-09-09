"""SenseNova crop decode auxiliary loss wiring.

SenseNova is the one architecture whose training tensors are in TOKEN space
(1 token = ``gen_patch`` x ``gen_patch`` latent cells, 8x8 = 64px on the swapped
SDXL VAE run) and whose network emits x0 directly under a t=1-is-clean
convention. These tests pin the four things a plausible copy of another arch's
wiring would get wrong: the guard, the token->2D restoration, the gradient path,
and the denormalisation into the decoder's own domain.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.training.ops import sensenova_ops

SCALING_FACTOR = 0.13025
PATCH = 8
CHANNELS = 4


class _MockDecoder(nn.Module):
    def __init__(self, in_channels: int, scale: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=float(self.scale), mode="nearest"))


class _MockVAE(nn.Module):
    """4-channel, 8x-compressing stand-in with the SDXL VAE's scaling factor."""

    def __init__(self, in_channels: int = CHANNELS):
        super().__init__()
        self.decoder = _MockDecoder(in_channels, scale=8)
        self.last_tile = None
        self.config = SimpleNamespace(
            latent_channels=in_channels,
            scaling_factor=SCALING_FACTOR,
            shift_factor=None,
            # 4 entries => spatial_compression_of() == 8, matching the decoder.
            block_out_channels=[128, 256, 512, 512],
        )

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        self.last_tile = z
        out = self.decoder(z)
        return (out,) if not return_dict else SimpleNamespace(sample=out)


def _unpatchify(x: torch.Tensor, patch: int, height: int, width: int) -> torch.Tensor:
    from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel

    # `unpatchify(sle, ...)` never touches its self argument.
    return NEOChatModel.unpatchify(None, x, patch, height, width)


def _make_trainer(vae, *, out_cells: int = 16, margin_cells: int = 4):
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        vae_dtype=torch.float32,
        vae=vae,
        wiring=SimpleNamespace(vae_norm="shift_scale", vae_norm_pack=1),
        crop_decode_loss_enable=True,
        crop_decode_loss_weight=0.5,
        crop_decode_loss_metric="mse",
        crop_decode_loss_margin_cells=margin_cells,
        crop_decode_loss_out_cells=out_cells,
        crop_decode_loss_snr_range="",
        log_prefix="[SenseNovaTest]",
        metrics={},
    )
    trainer.log_extra_metric = lambda k, v: trainer.metrics.__setitem__(k, v)
    trainer.transformer = SimpleNamespace(unpatchify=_unpatchify)
    return trainer


def _call(trainer, x0_pred, x0, z_image, t, main_loss, *, lat: int = 16):
    return sensenova_ops._crop_decode_aux_loss(
        trainer,
        transformer=trainer.transformer,
        x0_pred=x0_pred,
        x0=x0,
        z_image=z_image,
        t=t,
        patch=PATCH,
        height=lat,
        width=lat,
        main_loss=main_loss,
    )


# --- the guard ---------------------------------------------------------------


def test_train_step_call_site_is_guarded_by_both_keys():
    """Disabled runs must not reach the aux path at all."""
    source = Path(sensenova_ops.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    train_step = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_step"
    )

    calls = [
        node for node in ast.walk(train_step)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_crop_decode_aux_loss"
    ]
    assert len(calls) == 1

    guards = [
        node for node in ast.walk(train_step)
        if isinstance(node, ast.If) and calls[0] in list(ast.walk(node))
    ]
    assert guards, "the aux call is unguarded"
    # The two keys moved into the shared predicate when the convergence
    # diagnostics' x0 capture joined the same gate; what must hold is that a run
    # asking for neither still never reaches the aux path.
    assert any("crop_decode_or_x0_capture_needed" in ast.unparse(guard.test)
               for guard in guards)
    from core.training.ops.crop_decode_loss import crop_decode_or_x0_capture_needed
    for weight in (0.0, 0.5):
        for enable in (False, True):
            off = SimpleNamespace(crop_decode_loss_enable=enable,
                                  crop_decode_loss_weight=weight,
                                  convergence_diagnostics_enable=False)
            assert crop_decode_or_x0_capture_needed(off) is (enable and weight > 0)


def test_pixel_space_run_is_a_noop(capsys):
    """No VAE (native pixel-space base) means nothing to decode through."""
    trainer = _make_trainer(vae=None)
    t = torch.tensor([0.4])
    tokens = torch.zeros(1, 4, PATCH * PATCH * CHANNELS)
    grid = torch.zeros(1, CHANNELS, 16, 16)

    assert _call(trainer, tokens, grid, grid, t, torch.zeros(())) is None
    assert _call(trainer, tokens, grid, grid, t, torch.zeros(())) is None
    assert capsys.readouterr().out.count("[crop_decode_loss]") == 1


# --- token space -> 2D -------------------------------------------------------


def test_unpatchify_inverts_the_train_step_token_layout():
    """train_step's x0_pred construction IS patchify, so unpatchify undoes it."""
    from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel

    batch, token_h, token_w = 2, 3, 2
    decoded = torch.randn(batch, CHANNELS, token_h * PATCH, token_w * PATCH)

    # Verbatim from ops/sensenova_ops.train_step.
    x0_pred = (
        decoded.view(batch, CHANNELS, token_h, PATCH, token_w, PATCH)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(batch, token_h * token_w, PATCH * PATCH * CHANNELS)
    )

    assert torch.equal(x0_pred, NEOChatModel.patchify(None, decoded, PATCH))
    restored = _unpatchify(x0_pred, PATCH, token_h * PATCH, token_w * PATCH)
    assert restored.shape == decoded.shape
    assert torch.equal(restored, decoded)


# --- gradient and decoder domain ---------------------------------------------


def test_gradient_reaches_the_token_prediction():
    torch.manual_seed(0)
    lat = 16
    head = nn.Linear(PATCH * PATCH * CHANNELS, PATCH * PATCH * CHANNELS)
    tokens_in = torch.randn(1, (lat // PATCH) ** 2, PATCH * PATCH * CHANNELS)
    x0_pred = head(tokens_in)
    x0 = torch.randn(1, CHANNELS, lat, lat)
    z_image = torch.randn(1, CHANNELS, lat, lat)
    t = torch.tensor([0.7])
    main_loss = F.mse_loss(x0_pred, torch.zeros_like(x0_pred))

    trainer = _make_trainer(_MockVAE())
    aux_loss = _call(trainer, x0_pred, x0, z_image, t, main_loss, lat=lat)

    assert aux_loss is not None
    assert torch.isfinite(aux_loss)
    assert trainer.metrics["crop_decode_loss"] > 0.0
    # The main loss is differentiable w.r.t. the same token tensor, so the
    # op's ||grad_aux||/||grad_main|| probe resolves instead of reading None.
    assert trainer.metrics["crop_decode_grad_norm_ratio"] > 0.0

    aux_loss.backward()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.abs().sum() > 0.0


def test_weight_scales_the_returned_loss():
    torch.manual_seed(0)
    lat = 16
    x0_pred = torch.randn(1, (lat // PATCH) ** 2, PATCH * PATCH * CHANNELS, requires_grad=True)
    x0 = torch.randn(1, CHANNELS, lat, lat)
    t = torch.tensor([0.5])

    trainer = _make_trainer(_MockVAE())
    aux_loss = _call(trainer, x0_pred, x0, x0, t, None, lat=lat)
    assert float(aux_loss.detach()) == pytest.approx(
        trainer.metrics["crop_decode_loss"] * trainer.crop_decode_loss_weight, rel=1e-5)


def test_latents_are_denormalised_before_the_decoder():
    """The decoder must see raw-domain latents, not the normalised training ones."""
    torch.manual_seed(0)
    lat = 16
    vae = _MockVAE()
    trainer = _make_trainer(vae, out_cells=lat, margin_cells=4)

    x0_pred = torch.zeros(1, (lat // PATCH) ** 2, PATCH * PATCH * CHANNELS, requires_grad=True)
    x0 = torch.randn(1, CHANNELS, lat, lat)
    t = torch.tensor([0.5])

    _call(trainer, x0_pred, x0, x0, t, None, lat=lat)

    # out_cells == the whole grid, so the last (GT) tile is the whole latent.
    assert vae.last_tile.shape == x0.shape
    assert torch.allclose(vae.last_tile, x0 / SCALING_FACTOR, atol=1e-5)


def test_out_cells_off_the_token_grid_warns_once(capsys):
    torch.manual_seed(0)
    lat = 16
    trainer = _make_trainer(_MockVAE(), out_cells=12, margin_cells=2)
    x0_pred = torch.zeros(1, (lat // PATCH) ** 2, PATCH * PATCH * CHANNELS, requires_grad=True)
    x0 = torch.randn(1, CHANNELS, lat, lat)
    t = torch.tensor([0.5])

    for _ in range(2):
        _call(trainer, x0_pred, x0, x0, t, None, lat=lat)
    out = capsys.readouterr().out
    assert out.count("is not a multiple of this run's 8-cell token") == 1
