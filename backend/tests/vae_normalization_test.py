"""Shared latent-normalisation tests -- CPU, model-free, ~5s (design §8.4, P5).

The five architectures that used to spell the formula inline now call
``components/vae_registry.normalize`` / ``denormalize``. The bar is BIT
identity against the formula each one carried, in the dtypes production runs
in (fp32 AND fp16/bf16 -- an fp32-only check has passed here before while the
fp16 path was broken). The old formulas are transcribed below because they no
longer exist in the tree; each one cites where it came from.

The one deviation is lens's ENCODE, which downcast ``running_var`` before the
sqrt while lens's own decode, the lens vendor pipeline and every FLUX.2 site
do not. No single function can be bit-identical to both orders in fp16/bf16
(they disagree on 12-18% of channels), so the shared layer keeps the
higher-precision one; the bound on how far lens's encode moves is asserted,
not assumed, below.

Run with (cwd backend/):
    ../venv/Scripts/python.exe -m pytest tests/vae_normalization_test.py -v
"""

from types import SimpleNamespace

import pytest
import torch

from core.models.components.vae_registry import (  # noqa: E402
    _pack_2x2, _unpack_2x2, denormalize, normalize,
)
from core.models.components.wiring import (  # noqa: E402
    ANIMA_WIRING, FLUX2_WIRING, KREA2_WIRING, LENS_WIRING, LTX2_WIRING,
    SDXL_WIRING,
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# The Qwen-Image VAE's own vectors (M:/model/krea2/vae/config.json), shared by
# anima and krea2.
QWEN_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
             0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
QWEN_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]


def _seeded(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _bn_vae(channels=32, pack=2, seed=7):
    """An AutoencoderKLFlux2 stand-in: BatchNorm over ``pack**2 * C`` channels,
    fp32 buffers (what ``from_pretrained`` gives, whatever the module dtype)."""
    g = torch.Generator().manual_seed(seed)
    n = channels * pack * pack
    bn = SimpleNamespace(running_mean=torch.randn(n, generator=g),
                         running_var=torch.rand(n, generator=g) * 4.0 + 0.05)
    return SimpleNamespace(
        bn=bn,
        config=SimpleNamespace(batch_norm_eps=1e-4, latent_channels=channels))


def _qwen_vae():
    """AutoencoderKLQwenImage stand-in: config lists, no scaling_factor."""
    return SimpleNamespace(config=SimpleNamespace(
        z_dim=16, latents_mean=list(QWEN_MEAN), latents_std=list(QWEN_STD)))


def _ltx_vae(channels=8, scaling_factor=1.0, seed=3):
    """LTX-2.3 stand-in: registered fp32 buffers plus a scalar scaling factor."""
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        latents_mean=torch.randn(channels, generator=g),
        latents_std=torch.rand(channels, generator=g) * 2.0 + 0.1,
        config=SimpleNamespace(scaling_factor=scaling_factor))


# --- the formulas P5 replaces (verbatim, with their provenance) -------------

def _old_flux2_bn(x, vae):
    """flux2_ops.vae_encode / pipeline_backends/flux2.py, on PACKED latents."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(x.device, x.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1)
                     + vae.config.batch_norm_eps).to(x.device, x.dtype)
    return (x - mean) / std


def _old_flux2_bn_inverse(x, vae):
    """flux2 decode sites: ``latents * std + mean`` on PACKED latents."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(x.device, x.dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1)
                     + vae.config.batch_norm_eps).to(x.device, x.dtype)
    return x * std + mean


def _old_lens_bn(x, vae):
    """lens_pipeline_ops._bn_normalize: casts var BEFORE the sqrt."""
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    var = vae.bn.running_var.view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    std = torch.sqrt(var + vae.config.batch_norm_eps)
    return (x - mean) / std


def _old_anima_per_channel(x, vae):
    """anima_pipeline_ops._get_qwen_vae_normalization + ``(x - mean) / std``."""
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, dtype=x.dtype,
                        device=x.device).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=x.dtype,
                       device=x.device).view(1, z, 1, 1, 1)
    return (x - mean) / std


def _old_krea2_per_channel(x, vae):
    """krea2_pipeline_ops._vae_norm_stats: fp32 tensor, then cast."""
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(
        device=x.device, dtype=x.dtype)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(
        device=x.device, dtype=x.dtype)
    return (x - mean) / std


def _old_krea2_per_channel_inverse(x, vae):
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(
        device=x.device, dtype=x.dtype)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(
        device=x.device, dtype=x.dtype)
    return x * std + mean


def _old_ltx2_per_channel(x, vae):
    """ltx2_ops._normalize_ltx_latents (== LTX2Pipeline._normalize_latents)."""
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(x.device, x.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(x.device, x.dtype)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    return (x - mean) * scaling_factor / std


# --- the pack domain --------------------------------------------------------

def test_the_pack_domain_is_the_arch_pack_function_and_is_lossless():
    from core.models.lens.lens_pipeline_ops import _patchify, _unpatchify

    x = _seeded(2, 32, 8, 6)
    assert torch.equal(_pack_2x2(x), _patchify(x))
    assert torch.equal(_unpack_2x2(_pack_2x2(x)), x)
    assert torch.equal(_unpatchify(_pack_2x2(x)), x)


def test_the_pack_domain_matches_the_flux2_trainer_patchify():
    from core.training.base_trainer import BaseTrainer

    x = _seeded(1, 32, 6, 4)
    assert torch.equal(
        _pack_2x2(x), BaseTrainer._flux2_patchify_latents_for_training(None, x))
    assert torch.equal(
        _unpack_2x2(_pack_2x2(x)),
        BaseTrainer._flux2_unpatchify_latents(None, _pack_2x2(x)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_flux2_generation_reordering_is_value_preserving(dtype):
    """The generation path became "normalise, then patchify" (and "unpatchify,
    then denormalise"), against the pipeline's OWN patchify -- imported, not
    reimplemented, because a shared mistake would pass either way."""
    from core.pipeline_backends.flux2 import Flux2Mixin

    patchify = lambda t: Flux2Mixin._flux2_patchify_latents(None, t)      # noqa: E731
    unpatchify = lambda t: Flux2Mixin._flux2_unpatchify_latents(None, t)  # noqa: E731
    vae = _bn_vae()
    raw = _seeded(1, 32, 8, 6, seed=11).to(dtype)

    assert torch.equal(patchify(normalize(raw, vae)),
                       _old_flux2_bn(patchify(raw), vae))
    packed = _seeded(1, 128, 4, 3, seed=12).to(dtype)
    assert torch.equal(denormalize(unpatchify(packed), vae),
                       unpatchify(_old_flux2_bn_inverse(packed, vae)))


# --- bit identity, per arch, per dtype --------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_flux2_training_encode_is_bit_identical(dtype):
    vae = _bn_vae()
    raw = _seeded(2, 32, 8, 6, seed=1).to(dtype)

    old = _old_flux2_bn(_pack_2x2(raw), vae)                 # normalise, then pack
    new = _pack_2x2(normalize(raw, vae, FLUX2_WIRING))       # pack inside, then pack
    assert new.dtype == old.dtype
    assert torch.equal(new, old)
    assert torch.equal(_pack_2x2(normalize(raw, vae)), old)  # and with no spec


@pytest.mark.parametrize("dtype", DTYPES)
def test_flux2_decode_is_bit_identical(dtype):
    vae = _bn_vae()
    packed = _seeded(1, 128, 4, 3, seed=2).to(dtype)

    old = _unpack_2x2(_old_flux2_bn_inverse(packed, vae))
    new = denormalize(_unpack_2x2(packed), vae, FLUX2_WIRING)
    assert new.dtype == old.dtype
    assert torch.equal(new, old)


@pytest.mark.parametrize("dtype", DTYPES)
def test_anima_is_bit_identical(dtype):
    vae = _qwen_vae()
    raw = _seeded(2, 16, 1, 8, 6, seed=4).to(dtype)

    assert torch.equal(normalize(raw, vae, ANIMA_WIRING), _old_anima_per_channel(raw, vae))
    assert torch.equal(normalize(raw, vae), _old_anima_per_channel(raw, vae))
    assert torch.equal(denormalize(raw, vae, ANIMA_WIRING),
                       _old_krea2_per_channel_inverse(raw, vae))


@pytest.mark.parametrize("dtype", DTYPES)
def test_krea2_is_bit_identical(dtype):
    vae = _qwen_vae()
    raw = _seeded(1, 16, 1, 6, 4, seed=5).to(dtype)

    assert torch.equal(normalize(raw, vae, KREA2_WIRING), _old_krea2_per_channel(raw, vae))
    assert torch.equal(denormalize(raw, vae, KREA2_WIRING),
                       _old_krea2_per_channel_inverse(raw, vae))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scaling_factor", [1.0, 0.7])
def test_ltx2_is_bit_identical(dtype, scaling_factor):
    vae = _ltx_vae(scaling_factor=scaling_factor)
    raw = _seeded(1, 8, 3, 4, 4, seed=6).to(dtype)

    assert torch.equal(normalize(raw, vae, LTX2_WIRING), _old_ltx2_per_channel(raw, vae))
    assert torch.equal(normalize(raw, vae), _old_ltx2_per_channel(raw, vae))


@pytest.mark.parametrize("dtype", DTYPES)
def test_lens_encode_keeps_the_higher_precision_of_the_two_sqrt_orders(dtype):
    """fp32: identical. fp16/bf16: at most one ulp of the std, and equal to what
    lens's own decode side (and its vendor pipeline) already computed."""
    vae = _bn_vae()
    raw = _seeded(1, 32, 8, 6, seed=8).to(dtype)

    old = _unpack_2x2(_old_lens_bn(_pack_2x2(raw), vae))
    new = normalize(raw, vae, LENS_WIRING)
    vendor_order = _unpack_2x2(_old_flux2_bn(_pack_2x2(raw), vae))

    assert torch.equal(new, vendor_order)
    if dtype is torch.float32:
        assert torch.equal(new, old)
    else:
        # The old order rounds three extra times (cast var, add eps, sqrt).
        # Measured maxima: 0.13% of the value in fp16, 1.0% in bf16.
        bound = 3 * torch.finfo(dtype).eps
        rel = ((new.float() - old.float()).abs()
               / old.float().abs().clamp_min(1e-6))
        assert rel.max() <= bound


@pytest.mark.parametrize("dtype", DTYPES)
def test_sd_sdxl_encode_is_unchanged_for_a_native_vae(dtype):
    """``sd_sdxl_ops.vae_encode`` now goes through the spec-aware entry point;
    with no swap it must still be exactly ``(sample - shift) * scale``."""
    from core.models.components.vae_registry import normalize_latent

    vae = SimpleNamespace(config=SimpleNamespace(scaling_factor=0.13025,
                                                 shift_factor=None))
    raw = _seeded(1, 4, 8, 8, seed=13).to(dtype)
    assert torch.equal(normalize(raw, vae, None), normalize_latent(raw, vae))

    flux1 = SimpleNamespace(config=SimpleNamespace(scaling_factor=0.3611,
                                                   shift_factor=0.1159))
    assert torch.equal(normalize(raw, flux1, None), normalize_latent(raw, flux1))


# --- crossing the normalisation domain (what P5 unblocks) -------------------

def test_a_batchnorm_vae_normalises_into_an_arch_that_does_not_pack():
    """§11's P5 acceptance case: a 32ch FLUX.2-family VAE under sdxl's wiring.
    The caller sees a raw 32ch latent; the statistics were still applied on the
    packed 4C domain."""
    vae = _bn_vae()
    spec = SDXL_WIRING.replace(latent_channels=32, vae_norm="batchnorm",
                               vae_norm_pack=2)
    raw = _seeded(1, 32, 8, 6, seed=9)

    out = normalize(raw, vae, spec)
    assert out.shape == raw.shape
    assert torch.equal(out, _unpack_2x2(_old_flux2_bn(_pack_2x2(raw), vae)))
    assert torch.allclose(denormalize(out, vae, spec), raw, atol=1e-5)


def test_a_shift_scale_vae_in_a_packing_arch_does_not_pack():
    """The other direction: flux2 wearing a plain scaling-factor VAE. No pack,
    so an odd latent height is fine here where the BatchNorm domain refuses it."""
    vae = SimpleNamespace(config=SimpleNamespace(scaling_factor=0.13025,
                                                 shift_factor=None))
    spec = FLUX2_WIRING.replace(vae_norm="shift_scale", vae_norm_pack=1)
    raw = _seeded(1, 4, 5, 3, seed=10)

    assert torch.equal(normalize(raw, vae, spec), raw * 0.13025)
    assert torch.allclose(denormalize(normalize(raw, vae, spec), vae, spec), raw)


def test_a_packed_domain_refuses_an_odd_latent_grid():
    vae = _bn_vae(channels=4)
    spec = SDXL_WIRING.replace(vae_norm="batchnorm", vae_norm_pack=2)
    with pytest.raises(ValueError, match="even latent"):
        normalize(_seeded(1, 4, 5, 6), vae, spec)


def test_a_batchnorm_that_does_not_pack_squarely_is_refused():
    vae = _bn_vae(channels=32)
    vae.bn.running_mean = vae.bn.running_mean[:96]
    vae.bn.running_var = vae.bn.running_var[:96]
    with pytest.raises(ValueError, match="square"):
        normalize(_seeded(1, 32, 4, 4), vae)


def test_the_normalisation_domains_may_now_be_crossed():
    from core.models.common import vae_source as vs

    bn = {"latent_channels": 32, "scale_factor": 8, "scale_temporal": 1,
          "ndim": 4, "norm": "batchnorm"}
    assert vs.check_vae_compatibility(bn, "sdxl") == (True, None)
    assert vs.check_vae_compatibility(dict(bn, norm="shift_scale"), "flux2") == (True, None)
    assert vs.check_vae_compatibility(bn, "flux2") == (True, None)


# --- "no scaling factor" means unknown, never 1.0 ---------------------------

def test_a_missing_scaling_factor_is_refused_rather_than_read_as_one():
    vae = SimpleNamespace(config=SimpleNamespace(scaling_factor=None))
    spec = SDXL_WIRING.replace(vae_norm="shift_scale")
    with pytest.raises(ValueError, match="cannot be determined"):
        normalize(_seeded(1, 4, 4, 4), vae, spec)
    with pytest.raises(ValueError, match="cannot be determined"):
        denormalize(_seeded(1, 4, 4, 4), vae, spec)


def test_a_batchnorm_arch_refuses_a_vae_without_one():
    vae = _qwen_vae()
    with pytest.raises(ValueError, match="no `bn` module"):
        normalize(_seeded(1, 16, 4, 4), vae, FLUX2_WIRING)


# --- the declarations the shared layer now reads ----------------------------

def test_each_wiring_declares_what_its_vae_actually_does():
    assert (LENS_WIRING.vae_norm, LENS_WIRING.vae_norm_pack) == ("batchnorm", 2)
    assert (FLUX2_WIRING.vae_norm, FLUX2_WIRING.vae_norm_pack) == ("batchnorm", 2)
    for spec in (ANIMA_WIRING, KREA2_WIRING, LTX2_WIRING):
        assert (spec.vae_norm, spec.vae_norm_pack) == ("per_channel", 1)
