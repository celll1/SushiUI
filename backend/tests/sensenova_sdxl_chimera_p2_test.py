from __future__ import annotations

import pytest
import torch
from diffusers.models.attention_processor import Attention

from core.attention import AttentionMode
from core.models.sensenova_sdxl_chimera.attention_processor import (
    ChimeraAttentionContext,
    ChimeraAttnProcessor,
)
from core.models.sensenova_sdxl_chimera.pipeline_ops import (
    ChimeraConditioning,
    sample_txt2img_latents,
)


def _conditioning(value: float, fingerprint: str, length: int = 5) -> ChimeraConditioning:
    return ChimeraConditioning(
        encoder_hidden_states=torch.full((1, length, 32), value),
        pooled_text_embeds=torch.full((1, 8), value),
        context_positions=torch.zeros(1, length, 3),
        attention_mask=torch.ones(1, length, dtype=torch.bool),
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


def test_cross_attention_padding_mask_matches_unpadded_context():
    torch.manual_seed(4)
    attention = Attention(query_dim=32, cross_attention_dim=32, heads=4, dim_head=8)
    hidden = torch.randn(2, 16, 32)
    short = torch.randn(1, 3, 32)
    long = torch.randn(1, 7, 32)
    padded = torch.cat((short, torch.randn(1, 4, 32)), dim=1)
    encoder = torch.cat((padded, long), dim=0)
    positions = torch.zeros(2, 7, 3)
    processor = ChimeraAttnProcessor()
    processor.set_context(ChimeraAttentionContext(
        context_positions=positions, target_height=32, target_width=32,
    ))
    additive_mask = torch.tensor(
        [[[0.0, 0.0, 0.0, -10000.0, -10000.0, -10000.0, -10000.0]],
         [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    )
    batched = processor(attention, hidden, encoder, attention_mask=additive_mask)

    singles = []
    for item_hidden, item_encoder in ((hidden[:1], short), (hidden[1:], long)):
        single = ChimeraAttnProcessor()
        single.set_context(ChimeraAttentionContext(
            context_positions=torch.zeros(1, item_encoder.shape[1], 3),
            target_height=32, target_width=32,
        ))
        singles.append(single(attention, item_hidden, item_encoder))
    assert torch.allclose(batched, torch.cat(singles), atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_varlen_cross_attention_forward_backward_on_cuda():
    pytest.importorskip("flash_attn")
    torch.manual_seed(5)
    device = torch.device("cuda")
    attention = Attention(
        query_dim=256, cross_attention_dim=256, heads=4, dim_head=64
    ).to(device=device, dtype=torch.float16)
    hidden = torch.randn(
        2, 16, 256, device=device, dtype=torch.float16, requires_grad=True
    )
    encoder = torch.randn(
        2, 7, 256, device=device, dtype=torch.float16, requires_grad=True
    )
    processor = ChimeraAttnProcessor(backend="flash", mode=AttentionMode.TRAINING)
    processor.set_context(ChimeraAttentionContext(
        context_positions=torch.zeros(2, 7, 3, device=device),
        target_height=32,
        target_width=32,
    ))
    mask = torch.tensor(
        [[True, True, True, False, False, False, False],
         [True, True, True, True, True, True, True]],
        device=device,
    )
    output = processor(attention, hidden, encoder, attention_mask=mask)
    output.float().square().mean().backward()
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert encoder.grad is not None and torch.isfinite(encoder.grad).all()


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
        self, sample, timestep, *, encoder_hidden_states, encoder_attention_mask,
        added_cond_kwargs, return_dict
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


def test_batched_cfg_pads_different_prefix_lengths_and_masks_padding():
    class MaskAwareUNet(_FakeUNet):
        def forward(self, sample, timestep, *, encoder_hidden_states,
                    encoder_attention_mask, added_cond_kwargs, return_dict):
            mask = encoder_attention_mask.to(encoder_hidden_states).unsqueeze(-1)
            means = (encoder_hidden_states * mask).sum(dim=(1, 2)) / (
                mask.sum(dim=(1, 2)) * encoder_hidden_states.shape[-1]
            )
            self.conditioning_means.extend(means.detach().cpu().tolist())
            return (sample * 0.05 + means[:, None, None, None] * 0.01,)

    def run(mode):
        return sample_txt2img_latents(
            MaskAwareUNet(),
            _conditioning(2.0, "positive-long", length=9),
            _conditioning(-1.0, "negative-short", length=2),
            height=64, width=64, steps=2, cfg_scale=4.0, seed=7, cfg_mode=mode,
        )

    assert torch.allclose(run("sequential"), run("batched"), atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sampling_moves_cpu_conditioning_to_the_unet_device():
    class DeviceCheckingUNet(_FakeUNet):
        def forward(self, sample, timestep, *, encoder_hidden_states,
                    encoder_attention_mask, added_cond_kwargs, return_dict):
            tensors = (
                timestep,
                encoder_hidden_states,
                encoder_attention_mask,
                added_cond_kwargs["text_embeds"],
                added_cond_kwargs["time_ids"],
            )
            assert all(tensor.device == sample.device for tensor in tensors)
            assert encoder_hidden_states.dtype == sample.dtype
            assert added_cond_kwargs["text_embeds"].dtype == sample.dtype
            assert added_cond_kwargs["time_ids"].dtype == sample.dtype
            return super().forward(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=return_dict,
            )

    unet = DeviceCheckingUNet().to(device="cuda", dtype=torch.float16)
    result = sample_txt2img_latents(
        unet,
        _conditioning(2.0, "positive"),
        _conditioning(-1.0, "negative"),
        height=64,
        width=64,
        steps=2,
        cfg_scale=4.0,
        seed=17,
        cfg_mode="sequential",
    )
    assert result.device.type == "cuda"
    assert result.dtype == torch.float16


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
