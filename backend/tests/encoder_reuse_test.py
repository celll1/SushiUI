"""Synthetic encoder-call counts, posterior RNG and bounded CPU ownership."""
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from core.training.encoder_reuse import (
    EncoderReuseCache, reuse_vae_posterior, reuse_caption_encoding, reuse_caption_batch,
)


class Codec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.calls = 0
        self.eval()

    def encode(self, pixels):
        self.calls += 1
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(
            torch.cat((pixels * self.weight, torch.zeros_like(pixels)), dim=1)))

    def decode(self, latents, return_dict=True):
        return SimpleNamespace(sample=latents)


def test_posterior_hits_keep_sampling_rng_and_invalidate_changed_pixels_weights():
    vae = Codec()
    owner = SimpleNamespace(_latent_encoding_mode="swap_onthefly", _encoder_reuse_cache=EncoderReuseCache(8192))
    pixels = torch.zeros(1, 1, 4, 4)
    generator = torch.Generator().manual_seed(7)
    original_generator = torch.Generator().manual_seed(7)
    outputs = []
    for _ in range(3):
        with reuse_vae_posterior(owner, vae):
            outputs.append(vae.encode(pixels).latent_dist.sample(generator=generator))
        expected = DiagonalGaussianDistribution(torch.cat((pixels, pixels), 1)).sample(generator=original_generator)
        torch.testing.assert_close(outputs[-1], expected, rtol=0, atol=0)
    assert vae.calls == 1
    assert not torch.equal(outputs[0], outputs[1])
    assert "encode" not in vae.__dict__
    with reuse_vae_posterior(owner, vae):
        vae.encode(pixels + 1)
    with torch.no_grad():
        vae.weight.add_(1)
    with reuse_vae_posterior(owner, vae):
        vae.encode(pixels)
    assert vae.calls == 3


def test_posterior_wrapper_restores_on_failure_and_trainable_codec_bypasses():
    vae = Codec()
    owner = SimpleNamespace(_latent_encoding_mode="swap_onthefly", _encoder_reuse_cache=EncoderReuseCache(8192))
    with pytest.raises(RuntimeError):
        with reuse_vae_posterior(owner, vae):
            raise RuntimeError("cancel")
    assert "encode" not in vae.__dict__
    vae.weight.requires_grad_(True)
    for _ in range(2):
        with reuse_vae_posterior(owner, vae):
            vae.encode(torch.zeros(1, 1, 4, 4))
    assert vae.calls == 2 and not owner._encoder_reuse_cache.entries


class TextOwner:
    def __init__(self):
        self.arch = SimpleNamespace(name="anima")
        self.text_encoder = Codec()
        self._text_encoding_mode = "swap_onthefly"
        self._encoder_reuse_cache = EncoderReuseCache(8192)
        self.calls = []

    @reuse_caption_encoding
    def encode_caption(self, caption, requires_grad=False, lyrics="", reference_image_paths=None, cfg_null=False):
        self.calls.append((caption, lyrics, cfg_null))
        return torch.tensor([len(caption)], dtype=torch.float32), {"mask": torch.ones(1)}

    @reuse_caption_batch
    def encode_captions_batched(self, captions, requires_grad=False, lyrics=None):
        self.calls.append(tuple(captions))
        return [(torch.tensor([len(caption)], dtype=torch.float32), None) for caption in captions]


def test_text_reuses_actual_caption_and_keeps_null_lyrics_and_grad_separate():
    owner = TextOwner()
    first = owner.encode_caption("same")
    first[0].zero_()
    assert owner.encode_caption("same")[0].item() == 4
    owner.encode_caption("changed")
    owner.encode_caption("same", lyrics="new")
    owner.encode_caption("same", cfg_null=True)
    owner.encode_caption("same", requires_grad=True)
    owner.encode_caption("same", reference_image_paths=["not-read"])
    assert len(owner.calls) == 6
    with torch.no_grad():
        owner.text_encoder.weight.add_(1)
    owner.encode_caption("same")
    assert len(owner.calls) == 7


def test_batched_text_deduplicates_misses_and_preserves_order_and_independent_payloads():
    owner = TextOwner()
    result = owner.encode_captions_batched(["a", "b", "a"])
    assert owner.calls == [("a", "b")]
    result[0][0].zero_()
    assert result[2][0].item() == 1
    result = owner.encode_captions_batched(["b", "new", "a"])
    assert owner.calls[-1] == ("new",)
    assert [item[0].item() for item in result] == [1, 3, 1]


def test_lru_budget_evicts_and_rejects_oversized_entries():
    cache = EncoderReuseCache(16)
    cache.put("a", torch.ones(2))
    cache.put("b", torch.ones(2))
    cache.get("a")
    cache.put("c", torch.ones(2))
    assert set(cache.entries) == {"a", "c"}
    cache.put("large", torch.ones(5))
    assert cache.bytes == 16 and cache.evictions == 1
    assert all(payload.device.type == "cpu" for payload, _ in cache.entries.values())


def test_single_and_batched_caption_formats_do_not_share_entries():
    owner = TextOwner()
    assert owner.encode_caption("a")[1] is not None
    assert owner.encode_captions_batched(["a"])[0][1] is None
    assert len(owner.calls) == 2


def test_small_scalar_buffers_invalidate_and_training_mode_bypasses():
    from core.training.encoder_reuse import producer_identity
    owner = TextOwner()
    owner.text_encoder.register_buffer("scale", torch.tensor(1.))
    identity = producer_identity(owner.text_encoder)
    owner.encode_caption("a")
    owner.text_encoder.register_buffer("scale", torch.tensor(2.))
    assert producer_identity(owner.text_encoder) != identity
    owner.encode_caption("a")
    owner.text_encoder.train()
    owner.encode_caption("a")
    owner.encode_caption("a")
    assert len(owner.calls) == 4


def test_entry_limit_bounds_zero_tensor_payloads():
    cache = EncoderReuseCache(16, max_entries=2)
    for key in range(3):
        cache.put(key, {"value": None})
    assert len(cache.entries) == 2 and cache.bytes == 0 and cache.evictions == 1


def test_precision_context_and_tokenizer_settings_invalidate():
    owner = TextOwner()
    owner.tokenizer = SimpleNamespace(padding_side="left")
    owner.encode_caption("a")
    owner.tokenizer.padding_side = "right"
    owner.encode_caption("a")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        owner.encode_caption("a")
    assert len(owner.calls) == 3


def test_603_exposures_of_187_inputs_only_run_187_vae_forwards():
    vae = Codec()
    owner = SimpleNamespace(_latent_encoding_mode="swap_onthefly",
        _encoder_reuse_cache=EncoderReuseCache(1024*1024))
    for index in range(603):
        with reuse_vae_posterior(owner, vae):
            vae.encode(torch.full((1, 1, 4, 4), float(index % 187))).latent_dist.sample()
    assert vae.calls == 187
    assert owner._encoder_reuse_cache.hits == 416


def test_normalized_codec_reuses_raw_posterior_and_preserves_normalized_samples():
    from core.models.components.latent_adapter import NormalizedImageVAE
    from diffusers.configuration_utils import FrozenDict
    from backend.tests.latent_adapter_contract_test import declaration
    vae = Codec()
    vae.config = FrozenDict(scaling_factor=2., shift_factor=.1)
    wrapper = NormalizedImageVAE(vae, declaration(3, 3, scale=8))
    wrapper.eval()
    owner = SimpleNamespace(_latent_encoding_mode="swap_onthefly",
        _encoder_reuse_cache=EncoderReuseCache(8192))
    pixels = torch.zeros(1, 3, 4, 4)
    first_generator, second_generator = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    for _ in range(2):
        expected = wrapper.encode(pixels).latent_dist.sample(generator=first_generator)
        with reuse_vae_posterior(owner, wrapper):
            actual = wrapper.encode(pixels).latent_dist.sample(generator=second_generator)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert owner._encoder_reuse_cache.hits == 1


def test_ming_conditioning_uses_processed_vision_inputs_and_frozen_producer():
    from core.models.ming_image_01.conditioning import MingConditioning
    from core.training.encoder_reuse import reuse_ming_conditioning
    producer = Codec()
    producer.cacheable = True
    calls = []
    def encode():
        calls.append(1)
        return MingConditioning(torch.ones(1, 2, 3), torch.ones(1, 1, 3))
    prepared = SimpleNamespace(width=128, height=128,
        model_inputs={"input_ids": torch.ones(1, 3, dtype=torch.long), "pixel_values": torch.zeros(1, 3, 4, 4)})
    source = SimpleNamespace(conditioner=producer, prepared=prepared, for_step=encode)
    owner = SimpleNamespace(config={}, _encoder_reuse_cache=EncoderReuseCache(8192))
    first = reuse_ming_conditioning(owner, source)
    first.queries.zero_()
    assert reuse_ming_conditioning(owner, source).queries.sum() > 0
    assert len(calls) == 1
    prepared.model_inputs["pixel_values"].add_(1)
    reuse_ming_conditioning(owner, source)
    assert len(calls) == 2


def test_anima_sampling_restores_frozen_and_mixed_child_modes(monkeypatch):
    import core.training.ops.anima_ops as ops
    transformer = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
    transformer.patch_spatial, transformer.in_channels = 2, 16
    transformer.train()
    transformer[1].eval()
    vae, te = Codec(), Codec()
    modes = [module.training for root in (transformer, vae, te) for module in root.modules()]
    def fail():
        raise RuntimeError("test cancellation")
    trainer = SimpleNamespace(transformer=transformer, vae=vae, text_encoder=te,
        device=torch.device("cpu"), log_prefix="[Test]", move_main_model_to_cpu=fail)
    with pytest.raises(RuntimeError, match="test cancellation"):
        ops.generate_sample(trainer, "synthetic", height=128, width=128)
    assert modes == [module.training for root in (transformer, vae, te) for module in root.modules()]


@pytest.mark.parametrize("architecture", ["sd_sdxl", "lens"])
def test_other_sample_paths_restore_frozen_encoder_modes(monkeypatch, architecture):
    import importlib
    ops = importlib.import_module(f"core.training.ops.{architecture}_ops")
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
    model[1].eval()
    vae, te, te2 = Codec(), Codec(), Codec()
    roots = (model, vae, te, te2)
    modes = [module.training for root in roots for module in root.modules()]
    def fail():
        raise RuntimeError("test cancellation")
    trainer = SimpleNamespace(transformer=model, unet=model, vae=vae, text_encoder=te,
        text_encoder_2=te2, device=torch.device("cpu"), log_prefix="[Test]",
        move_main_model_to_cpu=fail, move_main_model_to_gpu=lambda: None,
        _arch_pixel_align=lambda: 16, is_sdxl=True)
    if architecture == "lens":
        import core.models.lens.lens_pipeline_ops as pipeline
        monkeypatch.setattr(pipeline, "token_stride", lambda vae: 8)
        with pytest.raises(RuntimeError, match="test cancellation"):
            ops.generate_sample(trainer, "synthetic", height=128, width=128)
    else:
        ops.generate_sample(trainer, "synthetic", height=128, width=128)
    assert modes == [module.training for root in roots for module in root.modules()]
