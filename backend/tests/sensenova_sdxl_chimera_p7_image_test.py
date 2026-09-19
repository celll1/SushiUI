"""Chimera SDEdit, RePaint, and image-endpoint routing contracts."""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from core.models.sensenova_sdxl_chimera.attention_processor import ChimeraAttnProcessor
from core.models.sensenova_sdxl_chimera.pipeline_ops import (
    ChimeraConditioning,
    decode_latents,
    encode_image_latents,
    sample_img2img_latents,
)


class _ZeroUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.attn_processors = {"down.attn2.processor": ChimeraAttnProcessor()}
        self.calls = 0

    def set_attn_processor(self, processors):
        self.attn_processors = processors

    def forward(self, sample, timestep, **kwargs):
        self.calls += 1
        return (torch.zeros_like(sample),)


class _DtypeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float16))
        self.config = SimpleNamespace(scaling_factor=0.13025, shift_factor=None)
        self.seen_encode_dtype = None
        self.seen_decode_dtype = None

    def encode(self, pixels):
        self.seen_encode_dtype = pixels.dtype
        latents = torch.zeros(
            pixels.shape[0], 4, pixels.shape[2] // 8, pixels.shape[3] // 8,
            device=pixels.device, dtype=pixels.dtype,
        )
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latents))

    def decode(self, latents, return_dict=False):
        self.seen_decode_dtype = latents.dtype
        pixels = torch.zeros(
            latents.shape[0], 3, latents.shape[2] * 8, latents.shape[3] * 8,
            device=latents.device, dtype=latents.dtype,
        )
        return (pixels,)


def test_vae_boundary_uses_vae_dtype_and_returns_unet_dtype():
    vae = _DtypeVAE()
    image = Image.new("RGB", (16, 16), (10, 20, 30))

    latents = encode_image_latents(
        vae, image, height=16, width=16, device="cpu", dtype=torch.bfloat16
    )
    decoded = decode_latents(vae, latents)

    assert vae.seen_encode_dtype == torch.float16
    assert latents.dtype == torch.bfloat16
    assert vae.seen_decode_dtype == torch.float16
    assert decoded.size == (16, 16)


def _conditioning() -> ChimeraConditioning:
    return ChimeraConditioning(
        encoder_hidden_states=torch.zeros(1, 3, 8),
        pooled_text_embeds=torch.zeros(1, 4),
        context_positions=torch.zeros(1, 3, 3),
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        fingerprint="test",
    )


def _sample(unet, source, *, strength, mask=None, seed=17):
    return sample_img2img_latents(
        unet,
        _conditioning(),
        None,
        source,
        steps=4,
        denoising_strength=strength,
        cfg_scale=1.0,
        seed=seed,
        generate_mask=mask,
    )


def test_zero_strength_returns_source_without_running_unet():
    source = torch.randn(1, 4, 2, 3)
    unet = _ZeroUNet()
    result = _sample(unet, source, strength=0.0)
    assert torch.equal(result, source)
    assert unet.calls == 0


def test_sdedit_is_seed_deterministic_and_strength_controls_start_step():
    source = torch.zeros(1, 4, 2, 3)
    first_unet = _ZeroUNet()
    second_unet = _ZeroUNet()
    first = _sample(first_unet, source, strength=0.5, seed=29)
    second = _sample(second_unet, source, strength=0.5, seed=29)
    assert torch.equal(first, second)
    assert first_unet.calls == second_unet.calls == 2


def test_full_strength_starts_from_seeded_noise_and_runs_every_step():
    source = torch.zeros(1, 4, 2, 3)
    first_unet = _ZeroUNet()
    second_unet = _ZeroUNet()
    first = _sample(first_unet, source, strength=1.0, seed=41)
    second = _sample(second_unet, source, strength=1.0, seed=41)
    assert torch.equal(first, second)
    assert not torch.equal(first, source)
    assert first_unet.calls == second_unet.calls == 3


def test_repaint_pins_preserve_cells_to_clean_source_at_final_time():
    source = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3)
    generate = torch.zeros(1, 1, 2, 3)
    generate[..., 1] = 1.0
    result = _sample(_ZeroUNet(), source, strength=1.0, mask=generate)
    preserve = (1.0 - generate).bool().expand_as(source)
    assert torch.equal(result[preserve], source[preserve])
    assert not torch.equal(result[~preserve], source[~preserve])


def _manager_for_dispatch(method_name, method):
    values = {
        name: False
        for name in (
            "is_zimage_model", "is_flux2_model", "is_anima_model", "is_lens_model",
            "is_ideogram4_model", "is_minit2i_model", "is_krea2_model", "is_ltx2_model",
            "is_sensenova_model", "is_minimax_h3_model", "is_minimax_music3_model",
            "is_acestep_model", "is_yue2_model",
        )
    }
    values.update(
        is_sensenova_sdxl_chimera_model=True,
        extensions=[],
        **{method_name: method},
    )
    return SimpleNamespace(**values)


def test_img2img_dispatches_to_chimera_backend():
    from core.pipeline import DiffusionPipelineManager

    sentinel = ("image", 3, 0)
    manager = _manager_for_dispatch(
        "_generate_img2img_sensenova_sdxl_chimera", lambda *args: sentinel
    )
    assert DiffusionPipelineManager.generate_img2img(manager, {}, object()) == sentinel


def test_inpaint_dispatches_to_chimera_backend():
    from core.pipeline import DiffusionPipelineManager

    sentinel = ("image", 5, 0)
    manager = _manager_for_dispatch(
        "_generate_inpaint_sensenova_sdxl_chimera", lambda *args: sentinel
    )
    assert DiffusionPipelineManager.generate_inpaint(
        manager, {}, object(), object()
    ) == sentinel


def test_outpaint_delegates_to_inpaint_and_restores_placed_pixels_exactly():
    from core.pipeline import DiffusionPipelineManager

    source_array = np.zeros((16, 16, 3), dtype=np.uint8)
    source_array[..., 0] = np.arange(16, dtype=np.uint8)[None, :]
    source_array[..., 1] = np.arange(16, dtype=np.uint8)[:, None]
    source = Image.fromarray(source_array)
    seen = {}

    def inpaint(params, canvas, mask, progress_callback=None, step_callback=None):
        seen.update(params=dict(params), canvas=canvas.copy(), mask=mask.copy())
        return Image.new("RGB", canvas.size, (200, 100, 50)), 73, 0

    manager = SimpleNamespace(generate_inpaint=inpaint)
    params = {
        "canvas_width": 48,
        "canvas_height": 32,
        "place_x": 16,
        "place_y": 8,
        "place_width": 16,
        "place_height": 16,
        "mask_blur": 0,
        "denoising_strength": 1.0,
        "outpaint_seam_fix": False,
    }
    result, seed, ancestral_seed = DiffusionPipelineManager.generate_outpaint(
        manager, params, source
    )
    assert (seed, ancestral_seed) == (73, 0)
    assert result.size == (48, 32)
    assert np.array_equal(np.asarray(result)[8:24, 16:32], source_array)
    assert seen["params"]["width"] == 48
    assert seen["params"]["height"] == 32
    assert np.asarray(seen["mask"])[8:24, 16:32].max() == 0
