"""The sd15/sdxl sampler's latent normalisation (design §8.4, phase P7).

``core/inference/custom_sampling.py`` used to divide by a scalar
``vae.config.scaling_factor`` in eleven places, which is why an sd15/sdxl full
fine-tune against a BatchNorm VAE was refused at preflight even though it
trained correctly. Every site now goes through
``components/vae_registry.normalize`` / ``denormalize``.

Two bars, both asserted here:

  1. a native model is BIT identical to the formula the file used to spell
     inline, in fp32 AND fp16/bf16 (an fp32-only check has passed in this repo
     while the fp16 production path was broken), and
  2. a pipeline carrying a BatchNorm or per-channel VAE encodes and decodes
     without raising, with the right numbers -- three of these call sites
     swallow exceptions and return ``None``/``applied=False``, so "did not
     raise" alone would pass vacuously.

CPU, model-free, no GPU. Run with (cwd backend/):
    ../venv/Scripts/python.exe -m pytest tests/vae_swap_sampler_normalization_test.py -v
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.inference import custom_sampling as cs
from core.models.components.vae_registry import denormalize, normalize

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

SDXL_SCALING = 0.13025
FLUX1_SCALING, FLUX1_SHIFT = 0.3611, 0.1159


# --- the formulas P7 replaces (verbatim, from the pre-P7 file) ---------------

def _old_encode(sample, vae):
    return (sample - (getattr(vae.config, "shift_factor", None) or 0.0)) * vae.config.scaling_factor


def _old_decode(latent, vae):
    return latent / vae.config.scaling_factor + (
        getattr(vae.config, "shift_factor", None) or 0.0)


# --- fakes ------------------------------------------------------------------

class _Dist:
    def __init__(self, tensor):
        self._t = tensor

    def sample(self, generator=None):
        return self._t

    def mode(self):
        return self._t


class _FakeVae:
    """A ``pipeline.vae`` stand-in that records what it was handed.

    ``latent`` is what ``encode`` returns (the RAW sample, pre-normalisation)
    and ``image`` what ``decode`` returns.
    """

    def __init__(self, config, *, channels=4, dtype=torch.float32,
                 size=8, bn=None, latent_seed=5):
        self.config = config
        self.bn = bn                       # absent on a shift_scale/per-channel VAE
        if bn is None:
            del self.bn
        self._dtype = dtype
        self._param = torch.zeros(1, dtype=dtype)
        g = torch.Generator().manual_seed(latent_seed)
        self.latent = torch.randn(1, channels, size, size, generator=g).to(dtype)
        self.image = torch.randn(1, 3, size * 8, size * 8, generator=g).to(dtype)
        self.encoded = None                # last encode() input
        self.decoded = None                # last decode() input

    # VAE-like surface the sampler touches
    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._param.device

    def parameters(self):
        return iter([self._param])

    def to(self, *a, **k):
        return self

    def encode(self, x):
        self.encoded = x
        return SimpleNamespace(latent_dist=_Dist(self.latent))

    def decode(self, latent, return_dict=True):
        self.decoded = latent
        return SimpleNamespace(sample=self.image)


def _native_sdxl_vae(dtype=torch.float32, **kw):
    # latents_mean/std are real AutoencoderKL config keys that default to None;
    # spelling them out proves a native config is still observed as shift_scale.
    config = SimpleNamespace(scaling_factor=SDXL_SCALING, latent_channels=4,
                             latents_mean=None, latents_std=None)
    return _FakeVae(config, channels=4, dtype=dtype, **kw)


def _shifted_vae(dtype=torch.float32, **kw):
    config = SimpleNamespace(scaling_factor=FLUX1_SCALING, shift_factor=FLUX1_SHIFT,
                             latent_channels=16, latents_mean=None, latents_std=None)
    return _FakeVae(config, channels=16, dtype=dtype, **kw)


def _batchnorm_vae(dtype=torch.float32, channels=32, **kw):
    """AutoencoderKLFlux2 stand-in: statistics over the 2x2-packed 4C channels."""
    g = torch.Generator().manual_seed(21)
    n = channels * 4
    bn = SimpleNamespace(running_mean=torch.randn(n, generator=g),
                         running_var=torch.rand(n, generator=g) * 4.0 + 0.05)
    config = SimpleNamespace(batch_norm_eps=1e-4, latent_channels=channels)
    return _FakeVae(config, channels=channels, dtype=dtype, bn=bn, **kw)


def _per_channel_vae(dtype=torch.float32, channels=16, **kw):
    """AutoencoderKLQwenImage stand-in: config vectors, no scaling factor."""
    g = torch.Generator().manual_seed(22)
    config = SimpleNamespace(
        latent_channels=channels,
        latents_mean=torch.randn(channels, generator=g).tolist(),
        latents_std=(torch.rand(channels, generator=g) * 2 + 0.5).tolist())
    return _FakeVae(config, channels=channels, dtype=dtype, **kw)


SWAPPED = {"batchnorm": _batchnorm_vae, "per_channel": _per_channel_vae}


def _pipeline(vae):
    return SimpleNamespace(vae=vae)


def _no_vram_moves(monkeypatch):
    import core.vram_optimization as vram

    monkeypatch.setattr(vram, "move_vae_to_gpu", lambda *a, **k: None)
    monkeypatch.setattr(vram, "move_vae_to_cpu", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 1. A native model is bit-identical, in every production dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("build", [_native_sdxl_vae, _shifted_vae])
def test_the_shared_layer_is_bit_identical_to_the_old_inline_formula(dtype, build):
    vae = build(dtype=dtype)
    raw = vae.latent
    assert torch.equal(normalize(raw, vae), _old_encode(raw, vae))
    assert torch.equal(denormalize(raw, vae), _old_decode(raw, vae))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_reference_guide_encode_is_bit_identical_for_a_native_model(dtype):
    vae = _native_sdxl_vae(dtype=dtype)
    guides = cs.prepare_reference_guide_latents(
        [{"image": Image.new("RGB", (64, 64), (30, 60, 90)), "strength": 0.4}],
        _pipeline(vae), 64, 64, "cpu", torch.float32, None)
    assert torch.equal(guides[0]["clean_latent"],
                       _old_encode(vae.latent, vae).to(torch.float32))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_style_reference_encode_is_bit_identical_for_a_native_model(dtype):
    vae = _native_sdxl_vae(dtype=dtype)
    ref_x0, eps_ref = cs.prepare_style_reference_latent(
        Image.new("RGB", (64, 64), (10, 20, 30)), _pipeline(vae),
        64, 64, "cpu", torch.float32, seed=7)
    assert torch.equal(ref_x0, _old_encode(vae.latent, vae).to(torch.float32))
    assert eps_ref.shape == ref_x0.shape


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_reference_decodes_are_bit_identical_for_a_native_model(dtype):
    vae = _native_sdxl_vae(dtype=dtype)
    latents = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(3))
    expected = _old_decode(latents, vae).to(dtype)

    bias = cs.compute_vae_dc_bias(_pipeline(vae), latents,
                                  torch.zeros(1, 3, 1, 1))
    assert bias is not None and torch.equal(vae.decoded, expected)

    vae.decoded = None
    roundtrip = cs.compute_outpaint_hf_roundtrip(_pipeline(vae), latents)
    assert roundtrip is not None and torch.equal(vae.decoded, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_in_loop_flatten_roundtrip_is_bit_identical_for_a_native_model(
        dtype, monkeypatch):
    _no_vram_moves(monkeypatch)
    import core.inference.inloop_flatten as flat

    monkeypatch.setattr(flat, "hard_flatten", lambda arr, min_region_frac: (arr, True))
    vae = _native_sdxl_vae(dtype=dtype)
    x0 = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(4))
    latents = torch.zeros_like(x0)

    out, applied = cs.inloop_hard_flatten_step(_pipeline(vae), latents, x0, 0.02)

    assert applied is True
    assert torch.equal(vae.decoded, _old_decode(x0, vae).to(dtype))
    # delta == old-formula encode of the re-encoded image, minus x0
    expected = _old_encode(vae.latent, vae) - x0.to(vae.latent.dtype)
    assert torch.equal(out, expected.to(latents.dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("wrapped", [False, True])
def test_the_real_vae_objects_are_observed_as_shift_scale(dtype, wrapped):
    """The two shapes a native sd15/sdxl ``pipeline.vae`` actually has. A real
    ``AutoencoderKL`` answers ``latents_mean`` through a deprecation shim rather
    than raising, and the PiD override is a wrapper with no ``bn`` and a
    delegated config -- both must still observe shift_scale."""
    from diffusers import AutoencoderKL

    from core.models.components.vae_registry import _observe_norm

    vae = AutoencoderKL(block_out_channels=[32], latent_channels=4,
                        down_block_types=["DownEncoderBlock2D"],
                        up_block_types=["UpDecoderBlock2D"], layers_per_block=1,
                        scaling_factor=SDXL_SCALING)
    if wrapped:
        from core.models.pid.pid_vae_wrapper import PidVaeWrapper
        vae = PidVaeWrapper(real_vae=vae, pid_pth_path="unused.pth")

    x = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(9)).to(dtype)
    assert _observe_norm(vae, x) == ("shift_scale", 1)
    assert torch.equal(normalize(x, vae), _old_encode(x, vae))
    assert torch.equal(denormalize(x, vae), _old_decode(x, vae))


# ---------------------------------------------------------------------------
# 2. A swapped VAE reaches the same call sites without raising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", sorted(SWAPPED))
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_swapped_vae_encodes_through_the_reference_and_style_sites(norm, dtype):
    vae = SWAPPED[norm](dtype=dtype)
    pipeline = _pipeline(vae)
    expected = normalize(vae.latent, vae).to(torch.float32)

    guides = cs.prepare_reference_guide_latents(
        [{"image": Image.new("RGB", (64, 64), (5, 5, 5)), "strength": 0.4}],
        pipeline, 64, 64, "cpu", torch.float32, None)
    assert torch.equal(guides[0]["clean_latent"], expected)

    ref_x0, _ = cs.prepare_style_reference_latent(
        Image.new("RGB", (64, 64), (9, 9, 9)), pipeline, 64, 64, "cpu",
        torch.float32, seed=1)
    assert torch.equal(ref_x0, expected)


@pytest.mark.parametrize("norm", sorted(SWAPPED))
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_swapped_vae_decodes_through_the_reference_sites(norm, dtype):
    vae = SWAPPED[norm](dtype=dtype)
    latents = torch.randn(vae.latent.shape,
                          generator=torch.Generator().manual_seed(6))
    expected = denormalize(latents, vae).to(dtype)

    # None here would mean the site raised and was swallowed (the pre-P7 result).
    bias = cs.compute_vae_dc_bias(_pipeline(vae), latents, torch.zeros(1, 3, 1, 1))
    assert bias is not None and torch.equal(vae.decoded, expected)

    vae.decoded = None
    roundtrip = cs.compute_outpaint_hf_roundtrip(_pipeline(vae), latents)
    assert roundtrip is not None and torch.equal(vae.decoded, expected)


@pytest.mark.parametrize("norm", sorted(SWAPPED))
def test_a_swapped_vae_survives_the_in_loop_flatten_roundtrip(norm, monkeypatch):
    _no_vram_moves(monkeypatch)
    import core.inference.inloop_flatten as flat

    monkeypatch.setattr(flat, "hard_flatten", lambda arr, min_region_frac: (arr, True))
    vae = SWAPPED[norm]()
    x0 = torch.randn(vae.latent.shape, generator=torch.Generator().manual_seed(8))

    out, applied = cs.inloop_hard_flatten_step(
        _pipeline(vae), torch.zeros_like(x0), x0, 0.02)

    assert applied is True
    assert torch.equal(vae.decoded, denormalize(x0, vae))
    assert torch.equal(out, normalize(vae.latent, vae) - x0)


def test_the_batchnorm_statistics_stay_on_their_own_packed_domain():
    """The sampler hands raw [B, C, H, W] latents; the 2x2 pack that the
    BatchNorm's 4C statistics live on happens inside the shared layer, and the
    sd15/sdxl U-Net never sees a packed tensor."""
    vae = _batchnorm_vae(channels=32, size=8)
    raw = vae.latent
    assert vae.bn.running_mean.numel() == 4 * raw.shape[1]
    out = normalize(raw, vae)
    assert out.shape == raw.shape
    assert torch.allclose(denormalize(out, vae), raw, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. No site was left behind
# ---------------------------------------------------------------------------

_SOURCE = Path(cs.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def test_no_call_site_reads_a_scalar_scaling_factor_any_more():
    reads = [node for node in ast.walk(_TREE)
             if isinstance(node, ast.Attribute) and node.attr == "scaling_factor"]
    assert reads == [], f"scaling_factor still read on lines " \
                        f"{[n.lineno for n in reads]}"


@pytest.mark.parametrize("loop", ["custom_sampling_loop",
                                  "custom_img2img_sampling_loop",
                                  "custom_inpaint_sampling_loop"])
def test_every_sampling_loop_denormalises_through_the_shared_layer(loop):
    fn = next(n for n in _TREE.body
              if isinstance(n, ast.FunctionDef) and n.name == loop)
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "vae_denormalize" in called
