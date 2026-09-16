"""Chimera SDEdit, RePaint, and image-endpoint routing contracts."""

from types import SimpleNamespace

import torch

from core.models.sensenova_sdxl_chimera.attention_processor import ChimeraAttnProcessor
from core.models.sensenova_sdxl_chimera.pipeline_ops import (
    ChimeraConditioning,
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


def _conditioning() -> ChimeraConditioning:
    return ChimeraConditioning(
        encoder_hidden_states=torch.zeros(1, 3, 8),
        pooled_text_embeds=torch.zeros(1, 4),
        context_positions=torch.zeros(1, 3, 3),
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
