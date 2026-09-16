from __future__ import annotations

import pytest
import torch
from diffusers.models.attention_processor import Attention

from core.models.sensenova_sdxl_chimera.attention_processor import (
    ChimeraAttentionContext,
    ChimeraAttnProcessor,
)
from core.models.sensenova_sdxl_chimera.pipeline_ops import (
    ChimeraConditioning,
    sample_txt2img_latents,
)


def _conditioning(value: float, fingerprint: str) -> ChimeraConditioning:
    return ChimeraConditioning(
        encoder_hidden_states=torch.full((1, 5, 32), value),
        pooled_text_embeds=torch.full((1, 8), value),
        context_positions=torch.zeros(1, 5, 3),
        fingerprint=fingerprint,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_cross_attention_cached_and_uncached_are_equal_and_metadata_invalidates(dtype):
    torch.manual_seed(3)
    attention = Attention(
        query_dim=32, cross_attention_dim=32, heads=4, dim_head=8
    ).to(dtype=dtype)
    processor = ChimeraAttnProcessor()
    hidden = torch.randn(1, 16, 32, dtype=dtype)
    encoder = torch.randn(1, 5, 32, dtype=dtype)
    context = ChimeraAttentionContext(
        context_positions=torch.zeros(1, 5, 3),
        target_height=32,
        target_width=32,
        cache_key=("bridge", 32, 32, 0, 0, "text-only-v1"),
    )
    processor.set_context(context)
    uncached = processor(attention, hidden, encoder)
    cached = processor(attention, hidden, encoder)
    assert torch.equal(uncached, cached)
    assert len(processor._cross_kv_cache) == 1
    processor.set_context(
        ChimeraAttentionContext(
            context_positions=context.context_positions,
            target_height=64,
            target_width=32,
            cache_key=("bridge", 64, 32, 0, 0, "text-only-v1"),
        )
    )
    processor(attention, hidden, encoder)
    assert len(processor._cross_kv_cache) == 2
    processor.clear_cache()
    assert not processor._cross_kv_cache


class _FakeUNet(torch.nn.Module):
    def __init__(self, *, fail: bool = False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.attn_processors = {"down.attn2.processor": ChimeraAttnProcessor()}
        self.fail = fail
        self.conditioning_means = []

    def set_attn_processor(self, processors):
        self.attn_processors = processors

    def forward(
        self, sample, timestep, *, encoder_hidden_states, added_cond_kwargs, return_dict
    ):
        if self.fail:
            raise RuntimeError("injected failure")
        means = encoder_hidden_states.mean(dim=(1, 2))
        self.conditioning_means.extend(means.detach().cpu().tolist())
        velocity = sample * 0.05 + means[:, None, None, None] * 0.01
        return (velocity,)


def _sample(unet, mode: str, seed: int = 11):
    return sample_txt2img_latents(
        unet,
        _conditioning(2.0, "positive"),
        _conditioning(-1.0, "negative"),
        height=64,
        width=64,
        steps=3,
        cfg_scale=4.0,
        seed=seed,
        cfg_mode=mode,
    )


def test_sequential_and_batched_cfg_match_and_branches_remain_separate():
    sequential_unet = _FakeUNet()
    batched_unet = _FakeUNet()
    sequential = _sample(sequential_unet, "sequential")
    batched = _sample(batched_unet, "batched")
    assert torch.allclose(sequential, batched, atol=1e-6, rtol=1e-6)
    assert set(sequential_unet.conditioning_means) == {-1.0, 2.0}
    assert batched_unet.conditioning_means == [-1.0, 2.0] * 3


def test_sampling_is_deterministic_and_cleans_cache_on_success_and_error():
    first_unet = _FakeUNet()
    second_unet = _FakeUNet()
    first = _sample(first_unet, "sequential", seed=29)
    second = _sample(second_unet, "sequential", seed=29)
    assert torch.equal(first, second)
    assert all(
        not processor._cross_kv_cache and processor.context is None
        for processor in first_unet.attn_processors.values()
    )

    failing = _FakeUNet(fail=True)
    with pytest.raises(RuntimeError, match="injected failure"):
        _sample(failing, "sequential")
    assert all(
        not processor._cross_kv_cache and processor.context is None
        for processor in failing.attn_processors.values()
    )
