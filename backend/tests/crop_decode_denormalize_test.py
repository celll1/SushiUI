"""The decoder's operating point in the crop decode auxiliary loss.

Training latents are NORMALISED (`core.models.components.vae_registry`, §8.4),
so the shared op owns the inverse for all eight architectures: what reaches
`vae.decode` must be the same raw-domain tensor `latent_space.decode` passes,
built in the same order (dtype cast, then denormalise).

Every normalisation method is covered, because the amount the operating point
moves by is method-dependent and only `shift_scale` is a plain scalar.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.components.vae_registry import denormalize
from core.training.ops.crop_decode_loss import compute_crop_decode_loss

CHANNELS = 4
LAT = 16


class _RecordingVAE(nn.Module):
    """8x-upsampling decoder that keeps every tile it was handed."""

    def __init__(self, channels: int = CHANNELS, **config):
        super().__init__()
        self.conv = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        self.spatial_compression_ratio = 8
        self.tiles: list[torch.Tensor] = []
        self.config = SimpleNamespace(latent_channels=channels, **config)

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        self.tiles.append(z.detach().clone())
        # Functional, in fp32: the parameters keep the dtype the tile is judged
        # against, and CPU fp16 convolution stays out of it.
        out = F.conv2d(
            F.interpolate(z.float(), scale_factor=8.0, mode="nearest"),
            self.conv.weight.float(), self.conv.bias.float(), padding=1)
        return (out,) if not return_dict else SimpleNamespace(sample=out)


def _shift_scale_vae(scaling: float = 0.3611, shift=0.1159) -> _RecordingVAE:
    return _RecordingVAE(scaling_factor=scaling, shift_factor=shift)


def _per_channel_vae() -> _RecordingVAE:
    return _RecordingVAE(
        scaling_factor=1.3,
        latents_mean=[0.1, -0.2, 0.35, 0.0],
        latents_std=[0.9, 1.4, 0.7, 1.1],
    )


def _batchnorm_vae(channels: int = CHANNELS) -> _RecordingVAE:
    vae = _RecordingVAE(channels=CHANNELS, batch_norm_eps=1e-5)
    bn = nn.BatchNorm2d(channels)
    with torch.no_grad():
        bn.running_mean.copy_(torch.linspace(-0.3, 0.4, channels))
        bn.running_var.copy_(torch.linspace(0.5, 2.0, channels))
    vae.bn = bn
    return vae


def _trainer(vae, *, wiring=None, out_cells: int = LAT, margin_cells: int = 4):
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        vae_dtype=torch.float32,
        vae=vae,
        crop_decode_loss_enable=True,
        crop_decode_loss_weight=0.5,
        crop_decode_loss_metric="mse",
        crop_decode_loss_margin_cells=margin_cells,
        crop_decode_loss_out_cells=out_cells,
        crop_decode_loss_snr_range="",
        noise_scheduler=None,
        log_prefix="[CropDecodeDenormTest]",
        metrics={},
    )
    if wiring is not None:
        trainer.wiring = wiring
    trainer.log_extra_metric = lambda k, v: trainer.metrics.__setitem__(k, v)
    return trainer


def _run(trainer, pred_x0, clean, *, channels: int = CHANNELS, lat: int = LAT):
    """One aux-loss call whose crop covers the whole latent (out_cells == lat)."""
    noisy = torch.zeros(1, channels, lat, lat)
    return compute_crop_decode_loss(
        trainer=trainer,
        model_pred=pred_x0,
        noisy_latents=noisy,
        timesteps=torch.tensor([0.5]),
        clean_latents=clean,
        noise_process="flow",
        prediction_target="velocity",
        noise_scheduler=None,
        velocity_sign="eps_minus_x0",
        predicted_latent=pred_x0,
    )


def test_shift_scale_latents_reach_the_decoder_denormalised():
    """SDXL's method: the decoder sees `x / scale + shift`, not the training x."""
    torch.manual_seed(0)
    vae = _shift_scale_vae()
    trainer = _trainer(vae, wiring=SimpleNamespace(vae_norm="shift_scale", vae_norm_pack=1))

    pred = torch.randn(1, CHANNELS, LAT, LAT, requires_grad=True)
    clean = torch.randn(1, CHANNELS, LAT, LAT)
    aux, _ = _run(trainer, pred, clean)

    assert aux is not None
    assert len(vae.tiles) == 2  # prediction, then ground truth
    pred_tile, gt_tile = vae.tiles
    assert pred_tile.shape == pred.shape
    assert torch.allclose(pred_tile, pred.detach() / 0.3611 + 0.1159, atol=1e-5)
    assert torch.allclose(gt_tile, clean / 0.3611 + 0.1159, atol=1e-5)


def test_per_channel_latents_reach_the_decoder_denormalised():
    """Anima/Krea2/LTX-2.3's method: per-channel mean/std, with a scaling factor."""
    torch.manual_seed(1)
    vae = _per_channel_vae()
    spec = SimpleNamespace(vae_norm="per_channel", vae_norm_pack=1)
    trainer = _trainer(vae, wiring=spec)

    pred = torch.randn(1, CHANNELS, LAT, LAT, requires_grad=True)
    clean = torch.randn(1, CHANNELS, LAT, LAT)
    _run(trainer, pred, clean)

    pred_tile, gt_tile = vae.tiles
    assert torch.allclose(pred_tile, denormalize(pred.detach(), vae, spec), atol=1e-6)
    assert torch.allclose(gt_tile, denormalize(clean, vae, spec), atol=1e-6)
    # Not a no-op: a per-channel mean this far from zero cannot be missed.
    assert not torch.allclose(gt_tile, clean, atol=1e-3)


def test_batchnorm_latents_reach_the_decoder_denormalised():
    """Lens/FLUX.2's method on unpacked channels (`vae_norm_pack=1`)."""
    torch.manual_seed(2)
    vae = _batchnorm_vae()
    spec = SimpleNamespace(vae_norm="batchnorm", vae_norm_pack=1)
    trainer = _trainer(vae, wiring=spec)

    pred = torch.randn(1, CHANNELS, LAT, LAT, requires_grad=True)
    clean = torch.randn(1, CHANNELS, LAT, LAT)
    _run(trainer, pred, clean)

    pred_tile, gt_tile = vae.tiles
    assert torch.allclose(pred_tile, denormalize(pred.detach(), vae, spec), atol=1e-6)
    assert torch.allclose(gt_tile, denormalize(clean, vae, spec), atol=1e-6)
    assert not torch.allclose(gt_tile, clean, atol=1e-3)


def test_batchnorm_2x2_packed_domain_reaches_the_decoder_denormalised():
    """FLUX.2/Lens run their BatchNorm on the 2x2-packed 4C domain."""
    torch.manual_seed(3)
    vae = _batchnorm_vae(channels=CHANNELS * 4)
    spec = SimpleNamespace(vae_norm="batchnorm", vae_norm_pack=2)
    trainer = _trainer(vae, wiring=spec)

    pred = torch.randn(1, CHANNELS, LAT, LAT, requires_grad=True)
    clean = torch.randn(1, CHANNELS, LAT, LAT)
    _run(trainer, pred, clean)

    pred_tile, gt_tile = vae.tiles
    # Shape is unchanged: the pack/unpack lives inside the normalisation.
    assert gt_tile.shape == clean.shape
    assert torch.allclose(pred_tile, denormalize(pred.detach(), vae, spec), atol=1e-6)
    assert torch.allclose(gt_tile, denormalize(clean, vae, spec), atol=1e-6)


def test_no_spec_falls_back_to_the_vae_s_own_observation():
    """A trainer with no `wiring` (Z-Image's encode side) still denormalises."""
    torch.manual_seed(4)
    vae = _shift_scale_vae(scaling=0.18215, shift=None)
    trainer = _trainer(vae)
    assert not hasattr(trainer, "wiring")

    pred = torch.randn(1, CHANNELS, LAT, LAT, requires_grad=True)
    clean = torch.randn(1, CHANNELS, LAT, LAT)
    _run(trainer, pred, clean)

    assert torch.allclose(vae.tiles[1], clean / 0.18215, atol=1e-4)


def test_the_cast_happens_before_the_denormalisation():
    """`latent_space.decode`'s order: into the VAE's dtype, then denormalise."""
    torch.manual_seed(5)
    vae = _shift_scale_vae().half()
    spec = SimpleNamespace(vae_norm="shift_scale", vae_norm_pack=1)
    trainer = _trainer(vae, wiring=spec)
    trainer.vae_dtype = torch.float16

    pred = torch.randn(1, CHANNELS, LAT, LAT, requires_grad=True)
    clean = torch.randn(1, CHANNELS, LAT, LAT)
    _run(trainer, pred, clean)

    gt_tile = vae.tiles[1]
    assert gt_tile.dtype == torch.float16
    assert torch.equal(gt_tile, denormalize(clean.half(), vae, spec))


def test_disabled_runs_never_reach_the_decoder():
    """Invariant: `crop_decode_loss_enable=False` leaves the step untouched."""
    vae = _shift_scale_vae()
    trainer = _trainer(vae)
    trainer.crop_decode_loss_enable = False

    pred = torch.randn(1, CHANNELS, LAT, LAT, requires_grad=True)
    aux, raw = _run(trainer, pred, torch.randn(1, CHANNELS, LAT, LAT))

    assert aux is None
    assert raw == pytest.approx(0.0)
    assert vae.tiles == []
    assert not hasattr(trainer, "_crop_decode_loss_module")
