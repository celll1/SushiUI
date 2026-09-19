from __future__ import annotations

import sys
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers.models.attention_processor import Attention

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.attention import AttentionMode
from core.models.sensenova_sdxl_chimera.attention_processor import (
    ChimeraAttentionContext,
    ChimeraAttnProcessor,
)
from core.models.sensenova_sdxl_chimera.pipeline_ops import (
    ChimeraConditioning,
    _combine_cfg_velocity,
    decode_latents,
    endpoint_observable_logsnr_timesteps,
    sample_txt2img_latents,
    shifted_timesteps,
)
from core.models.sensenova_sdxl_chimera.artifact import prediction_contract
from core.models.sensenova_sdxl_chimera.flow import (
    FLOW_V2_PREDICTION,
    FLOW_V2_VELOCITY_PREDICTION,
    FLOW_V3_PREDICTION,
)
from core.models.sensenova_sdxl_chimera.unet import install_polar_radial_head
from core.training.arch.base_arch import SampleContext
from core.training.arch.sensenova_sdxl_chimera import (
    SenseNovaSDXLChimeraArchHandler,
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


class _FakePolarUNet(_FakeUNet):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(block_out_channels=(4,))
        self.mid_block = torch.nn.Identity()
        install_polar_radial_head(self)

    def forward(
        self, sample, timestep, *, encoder_hidden_states, encoder_attention_mask,
        added_cond_kwargs, return_dict
    ):
        means = encoder_hidden_states.mean(dim=(1, 2))
        self.conditioning_means.extend(means.detach().cpu().tolist())
        conditioned = sample + means[:, None, None, None] * 0.01
        mid = self.mid_block(conditioned)
        return (mid * 0.05,)


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
    assert batched_unet.conditioning_means == [-1.0, 2.0] * 2


def test_pure_noise_first_step_is_analytic_and_skips_unet():
    unet = _FakeUNet()
    previews = []
    result = sample_txt2img_latents(
        unet,
        _conditioning(2.0, "positive"),
        None,
        height=64,
        width=64,
        steps=2,
        cfg_scale=1.0,
        seed=23,
        progress_callback=lambda _step, _total, sample: previews.append(sample.clone()),
    )

    generator = torch.Generator(device="cpu").manual_seed(23)
    initial = torch.randn((1, 4, 8, 8), generator=generator)
    assert torch.allclose(previews[0], initial * 0.5)
    assert len(unet.conditioning_means) == 1
    assert result.shape == initial.shape


def _v2_prediction(prediction_type):
    return prediction_contract(
        prediction_type,
        latent_mean=[0.1, -0.2, 0.3, -0.4],
        latent_centered_second_moment=1.25,
    )


def test_v2_residual_skips_first_unet_but_direct_velocity_evaluates_it():
    residual_unet = _FakeUNet()
    direct_unet = _FakeUNet()
    common = dict(
        positive=_conditioning(2.0, "positive"),
        negative=None,
        height=64,
        width=64,
        steps=2,
        cfg_scale=1.0,
        seed=29,
    )
    sample_txt2img_latents(
        residual_unet,
        prediction=_v2_prediction(FLOW_V2_PREDICTION),
        **common,
    )
    sample_txt2img_latents(
        direct_unet,
        prediction=_v2_prediction(FLOW_V2_VELOCITY_PREDICTION),
        **common,
    )
    assert len(residual_unet.conditioning_means) == 1
    assert len(direct_unet.conditioning_means) == 2


def test_v2_residual_cfg_probe_exposes_zero_first_endpoint_delta():
    records = []
    sample_txt2img_latents(
        _FakeUNet(),
        _conditioning(2.0, "positive"),
        _conditioning(-1.0, "negative"),
        height=64,
        width=64,
        steps=3,
        cfg_scale=7.0,
        seed=31,
        prediction=_v2_prediction(FLOW_V2_PREDICTION),
        cfg_probe_callback=records.append,
    )
    assert records[0]["bypassed_unet"] == 1
    assert records[0]["prediction_delta_rms"] == 0.0
    assert records[0]["analytic_velocity_rms"] > 0.0
    assert all(record["bypassed_unet"] == 0 for record in records[1:])


def test_v3_evaluates_noise_endpoint_and_matches_cfg_execution_modes():
    torch.manual_seed(37)
    sequential_unet = _FakePolarUNet()
    batched_unet = _FakePolarUNet()
    batched_unet.load_state_dict(sequential_unet.state_dict())
    prediction = prediction_contract(
        FLOW_V3_PREDICTION,
        latent_mean=[0.0, 0.0, 0.0, 0.0],
        latent_centered_second_moment=1.0,
    )
    records = []
    common = dict(
        positive=_conditioning(2.0, "positive"),
        negative=_conditioning(-1.0, "negative"),
        height=64,
        width=64,
        steps=3,
        cfg_scale=4.0,
        seed=41,
        prediction=prediction,
    )
    sequential = sample_txt2img_latents(
        sequential_unet,
        cfg_mode="sequential",
        cfg_probe_callback=records.append,
        **common,
    )
    batched = sample_txt2img_latents(
        batched_unet,
        cfg_mode="batched",
        **common,
    )

    assert torch.allclose(sequential, batched, atol=2e-6, rtol=2e-6)
    assert len(sequential_unet.conditioning_means) == 6
    assert batched_unet.conditioning_means == [-1.0, 2.0] * 3
    assert len(records) == 3
    assert records[0]["timestep"] == 0.0
    assert all(record["bypassed_unet"] == 0 for record in records)
    assert all(record["radial_cfg_delta"] == 0.0 for record in records)
    assert all(
        record["tangent_orthogonality_abs_max"] <= 1e-6
        for record in records
    )


def test_cfg_norm_caps_global_and_per_channel_overshoot():
    conditional = torch.tensor([[[[1.0, 0.0]], [[0.0, 2.0]]]])
    unconditional = -conditional
    raw = _combine_cfg_velocity(conditional, unconditional, 7.0, "none")
    global_normed = _combine_cfg_velocity(conditional, unconditional, 7.0, "global")
    channel_normed = _combine_cfg_velocity(conditional, unconditional, 7.0, "channel")

    assert torch.linalg.vector_norm(raw) > torch.linalg.vector_norm(conditional)
    assert torch.allclose(
        torch.linalg.vector_norm(global_normed),
        torch.linalg.vector_norm(conditional),
    )
    assert torch.allclose(
        torch.linalg.vector_norm(channel_normed, dim=(2, 3)),
        torch.linalg.vector_norm(conditional, dim=(2, 3)),
    )


@pytest.mark.parametrize("mode", ("sequential", "batched"))
def test_cfg_probe_records_one_finite_scalar_payload_per_euler_step(mode):
    records = []
    sample_txt2img_latents(
        _FakeUNet(),
        _conditioning(2.0, "positive"),
        _conditioning(-1.0, "negative"),
        height=64,
        width=64,
        steps=4,
        cfg_scale=7.0,
        cfg_mode=mode,
        cfg_norm="global",
        seed=13,
        cfg_probe_callback=records.append,
    )

    assert len(records) == 4
    assert [record["step"] for record in records] == [1, 2, 3, 4]
    assert [record["total_steps"] for record in records] == [4] * 4
    assert [record["timestep"] for record in records] == sorted(
        record["timestep"] for record in records
    )
    for record in records:
        assert all(math.isfinite(value) for value in record.values())
        assert record["delta_t"] > 0
        assert record["clamp_norm_ratio"] <= 1.0 + 1e-6
        assert record["post_cond_norm_ratio"] <= 1.0 + 1e-6
        assert record["raw_cond_norm_ratio"] + 1e-6 >= record["post_cond_norm_ratio"]


@pytest.mark.parametrize("mode", ("sequential", "batched"))
def test_cfg_probe_reports_noise_to_clean_schedule(mode):
    records = []
    sample_txt2img_latents(
        _FakeUNet(),
        _conditioning(2.0, "positive"),
        _conditioning(-1.0, "negative"),
        height=64,
        width=64,
        steps=4,
        cfg_scale=7.0,
        cfg_schedule_type="linear",
        cfg_schedule_min=1.0,
        cfg_schedule_max=7.0,
        cfg_mode=mode,
        seed=17,
        cfg_probe_callback=records.append,
    )

    scales = [record["cfg_scale"] for record in records]
    assert scales[0] == pytest.approx(1.0)
    assert scales == sorted(scales)
    assert scales[-1] > scales[0]


def test_cfg_probe_refuses_sampling_without_an_unconditional_branch():
    with pytest.raises(ValueError, match="requires a negative branch"):
        sample_txt2img_latents(
            _FakeUNet(),
            _conditioning(2.0, "positive"),
            None,
            height=64,
            width=64,
            steps=2,
            cfg_scale=1.0,
            seed=13,
            cfg_probe_callback=lambda _record: None,
        )


def test_shift_three_allocates_more_steps_near_noise_than_shift_one():
    neutral = shifted_timesteps(4, 1.0, device="cpu")
    shifted = shifted_timesteps(4, 3.0, device="cpu")
    assert shifted[0] == neutral[0] == 0
    assert shifted[-1] == neutral[-1] == 1
    assert torch.all(shifted[1:-1] < neutral[1:-1])


def test_v2_bounded_logsnr_grid_is_monotonic_and_equal_spaced_inside():
    times = endpoint_observable_logsnr_timesteps(
        6, 1.25, log_snr_min=-8.0, log_snr_max=8.0, device="cpu"
    ).double()
    assert times[0] == 0.0 and times[-1] == 1.0
    assert torch.all(times[1:] > times[:-1])
    interior = times[1:-1]
    alpha = 2.0 * interior.square() - interior.pow(3)
    sigma = 1.0 - interior - interior.square() + interior.pow(3)
    log_snr = math.log(1.25) + 2.0 * (alpha.log() - sigma.log())
    assert torch.allclose(
        log_snr[1:] - log_snr[:-1],
        torch.full_like(log_snr[1:], 16.0 / 6.0),
        atol=2e-6,
        rtol=2e-6,
    )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_training_preview_stages_fp16_vae_on_cuda_and_restores_cpu():
    class TinyVAE(torch.nn.Module):
        config = type("Config", (), {"scaling_factor": 1.0, "shift_factor": None})()

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones((), dtype=torch.float16))

        def decode(self, latents, return_dict=False):
            assert next(self.parameters()).device.type == "cuda"
            assert latents.device.type == "cuda"
            assert latents.dtype == torch.float16
            return (latents[:, :3],)

    vae = TinyVAE()
    image = decode_latents(
        vae,
        torch.zeros(1, 4, 8, 8, device="cuda", dtype=torch.bfloat16),
        device="cuda",
        restore_device=True,
    )
    assert image.size == (8, 8)
    assert next(vae.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_training_preview_stages_unet_on_cuda_and_restores_cpu():
    handler = SenseNovaSDXLChimeraArchHandler()
    unet = _FakeUNet().to(dtype=torch.float16).train()
    trainer = SimpleNamespace(unet=unet, vae=object(), device=torch.device("cuda"))
    hidden = torch.zeros(1, 3, 32)
    aux = {
        "pooled_text_embeds": torch.zeros(1, 8),
        "context_positions": torch.zeros(1, 3, 3),
        "context_attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    def fake_sample(active_unet, *_args, **_kwargs):
        assert next(active_unet.parameters()).device.type == "cuda"
        assert not active_unet.training
        assert _kwargs["timestep_shift"] == 3.0
        assert _kwargs["cfg_norm"] == "global"
        return torch.zeros(1, 4, 8, 8, device="cuda", dtype=torch.float16)

    def fake_decode(_vae, latents, **kwargs):
        assert next(unet.parameters()).device.type == "cpu"
        assert unet.training
        assert latents.device.type == "cuda"
        assert kwargs == {"device": torch.device("cuda"), "restore_device": True}
        return "preview"

    with (
        patch.object(handler, "encode_prompt", return_value=(hidden, aux)),
        patch(
            "core.models.sensenova_sdxl_chimera.pipeline_ops.sample_txt2img_latents",
            side_effect=fake_sample,
        ),
        patch(
            "core.models.sensenova_sdxl_chimera.pipeline_ops.decode_latents",
            side_effect=fake_decode,
        ),
    ):
        result = handler.sample(
            trainer,
            SampleContext(
                prompt="test", negative_prompt="", width=64, height=64,
                num_inference_steps=2, guidance_scale=1.0, seed=1,
            ),
        )

    assert result == "preview"


def test_training_preview_forwards_flow_sampler_controls_on_cpu():
    handler = SenseNovaSDXLChimeraArchHandler()
    unet = _FakeUNet().train()
    trainer = SimpleNamespace(unet=unet, vae=object(), device=torch.device("cpu"))
    hidden = torch.zeros(1, 3, 32)
    aux = {
        "pooled_text_embeds": torch.zeros(1, 8),
        "context_positions": torch.zeros(1, 3, 3),
        "context_attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    forwarded = {}
    probe_callback = lambda _record: None

    def fake_sample(_unet, *_args, **kwargs):
        forwarded.update(kwargs)
        return torch.zeros(1, 4, 8, 8)

    with (
        patch.object(handler, "encode_prompt", return_value=(hidden, aux)),
        patch(
            "core.models.sensenova_sdxl_chimera.pipeline_ops.sample_txt2img_latents",
            side_effect=fake_sample,
        ),
        patch(
            "core.models.sensenova_sdxl_chimera.pipeline_ops.decode_latents",
            return_value="preview",
        ),
    ):
        result = handler.sample(
            trainer,
            SampleContext(
                prompt="test", negative_prompt="", width=64, height=64,
                num_inference_steps=2, guidance_scale=7.0, seed=1,
                sensenova_timestep_shift=2.5, sensenova_cfg_norm="global",
                cfg_probe_callback=probe_callback,
            ),
        )

    assert result == "preview"
    assert forwarded["timestep_shift"] == 2.5
    assert forwarded["cfg_norm"] == "global"
    assert forwarded["cfg_probe_callback"] is probe_callback


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
