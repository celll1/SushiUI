import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.arch.sensenova import SenseNovaArchHandler
from core.training.ops.sensenova_ops import (
    SenseNovaTrainingPrefix,
    _assert_plain_int8_training_base,
    encode_prompt,
    load_components,
    setup_attention_backend,
    train_step,
    vae_encode,
)


class _CacheLayer:
    def __init__(self):
        self.keys = torch.ones(1, 1, 2, 1)
        self.values = torch.ones(1, 1, 2, 1)
        self.flash_k_cache = None
        self.flash_v_cache = None


class _Cache:
    def __init__(self):
        self.layers = [_CacheLayer()]
        self._kv_cache_streamer = None
        self._kv_cache_streamer_branch = None


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, **kwargs):
        assert kwargs["update_cache"] is False
        assert kwargs["use_cache"] is False
        return hidden_states * self.scale


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden):
        return hidden[:, :1].expand(-1, 3, 32, 32) * self.scale


class _Transformer(nn.Module):
    patch_size = 16
    downsample_ratio = 0.5
    config = SimpleNamespace(t_eps=0.05)

    def __init__(self):
        super().__init__()
        model = nn.Module()
        model.layers = nn.ModuleList([_Layer()])
        model.norm_mot_gen = nn.Identity()
        self.language_model = SimpleNamespace(model=model)
        self.fm_modules = nn.ModuleDict({"fm_head": _Head()})

    def _build_t2i_image_indexes(self, token_h, token_w, text_length, device):
        self.index_call = (token_h, token_w, text_length, device)
        return torch.zeros(3, token_h * token_w, dtype=torch.long, device=device)

    def patchify(self, image, patch, channel_first=False):
        assert patch == 32
        return image.permute(0, 2, 3, 1).reshape(1, 1, -1)


def _attach_plain_int8(transformer, count=588):
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    transformer.quantized_linears = nn.ModuleList(
        [Int8Linear(1, 1, False, torch.bfloat16) for _ in range(count)]
    )
    return transformer


def test_load_components_keeps_training_runtime_minimal():
    transformer = _attach_plain_int8(_Transformer())
    trainer = SimpleNamespace(
        model_path="checkpoint.safetensors",
        weight_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        attention_backend="native",
    )
    components = {
        "transformer": transformer,
        "tokenizer": object(),
        "config": object(),
    }
    with patch(
        "core.models.sensenova.loader.load_sensenova_from_path",
        return_value=components,
    ), patch("core.training.ops.sensenova_ops.setup_attention_backend") as setup:
        load_components(trainer)

    assert trainer.transformer is transformer and transformer.training
    assert trainer.transformer_uncond is None
    assert trainer.vae is None and trainer.noise_scheduler is None
    setup.assert_called_once_with(trainer, "native")


def test_plain_int8_training_base_refuses_convrot_and_incomplete_census():
    incomplete = _attach_plain_int8(_Transformer(), count=587)
    with pytest.raises(RuntimeError, match="plain Int8Linear=587"):
        _assert_plain_int8_training_base(incomplete)

    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    mixed = _attach_plain_int8(_Transformer(), count=587)
    mixed.quantized_linears.append(
        ConvRotInt8Linear(
            256,
            1,
            False,
            torch.bfloat16,
            convrot_groupsize=256,
            marker_numel=1,
        )
    )
    with pytest.raises(RuntimeError, match="ConvRotInt8Linear=1"):
        _assert_plain_int8_training_base(mixed)


def test_attention_setup_stamps_training_mode_and_checks_layer_count():
    from core.attention import AttentionMode

    transformer = _Transformer()
    trainer = SimpleNamespace(
        transformer=transformer,
        _resolve_training_backend=lambda backend: f"resolved-{backend}",
    )
    with patch(
        "core.models.sensenova.sensenova_pipeline_ops.set_attention_backend",
        return_value=1,
    ) as setter:
        setup_attention_backend(trainer, "native")
    setter.assert_called_once_with(
        transformer, "resolved-native", AttentionMode.TRAINING
    )

    with patch(
        "core.models.sensenova.sensenova_pipeline_ops.set_attention_backend",
        return_value=0,
    ), pytest.raises(RuntimeError, match="configured 0 attention"):
        setup_attention_backend(trainer, "native")


def test_encode_prompt_builds_detached_training_prefix():
    cache = _Cache()
    transformer = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        _build_t2i_query=lambda prompt, **kwargs: prompt,
        _build_t2i_text_inputs=lambda tokenizer, query: (
            torch.ones(1, 3, dtype=torch.long),
            torch.zeros(3, 3, dtype=torch.long),
            {"full_attention": None},
        ),
        _t2i_prefix_forward=lambda *args: (cache, torch.zeros(1, 3, 1)),
    )
    trainer = SimpleNamespace(transformer=transformer, tokenizer=object())

    prefix = encode_prompt(trainer, "caption")

    assert prefix.cache is cache
    assert prefix.text_length == 3
    assert not cache.layers[0].keys.requires_grad


def test_encode_prompt_phase_retry_and_success_transition():
    cache = _Cache()
    calls = []

    class Evictor:
        def enter_prefix(self):
            calls.append("prefix")

        def enter_denoise(self):
            calls.append("denoise")

        def assert_generation_resident(self):
            calls.append("resident")

    attempts = iter([RuntimeError("prefix failed"), (cache, torch.zeros(1, 3, 1))])

    def prefix_forward(*args):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    transformer = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        _build_t2i_query=lambda prompt, **kwargs: prompt,
        _build_t2i_text_inputs=lambda tokenizer, query: (
            torch.ones(1, 3, dtype=torch.long),
            torch.zeros(3, 3, dtype=torch.long),
            {"full_attention": None},
        ),
        _t2i_prefix_forward=prefix_forward,
    )
    trainer = SimpleNamespace(
        transformer=transformer,
        tokenizer=object(),
        sensenova_phase_evictor=Evictor(),
    )
    with pytest.raises(RuntimeError, match="prefix failed"):
        encode_prompt(trainer, "caption")
    encode_prompt(trainer, "caption")
    assert calls == ["prefix", "prefix", "denoise", "resident"]


def test_pixel_encode_is_cpu_and_requires_32_alignment():
    trainer = SimpleNamespace(training_dtype=torch.float32)
    encoded = vae_encode(trainer, torch.zeros(1, 3, 32, 64))
    assert encoded.device.type == "cpu"
    assert encoded.shape == (1, 3, 32, 64)

    try:
        vae_encode(trainer, torch.zeros(1, 3, 32, 48))
    except ValueError as exc:
        assert "divisible by 32" in str(exc)
    else:
        raise AssertionError("unaligned image was accepted")


def test_flow_step_matches_vendor_noising_conditioning_and_velocity_math():
    transformer = _Transformer()
    trainer = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=True,
    )
    cache = _Cache()
    prefix = SenseNovaTrainingPrefix(cache, text_length=3)
    key = cache.layers[0].keys

    context_call = {}

    def build_context(model, shape, image, timestep, noise_scale):
        context_call.update(
            image=image.detach().clone(),
            timestep=timestep.detach().clone(),
            noise_scale=noise_scale,
            shape=shape,
        )
        return (
            model.patchify(image, 32),
            torch.full((1, 1, 1), 2.0),
            torch.ones(1, 1, 1),
        )

    images = torch.ones(1, 3, 32, 32)
    with patch(
        "core.models.sensenova.sensenova_pipeline_ops.compute_noise_scale",
        return_value=2.0,
    ), patch(
        "core.models.sensenova.sensenova_pipeline_ops._build_step_context",
        side_effect=build_context,
    ), patch(
        "torch.randn_like",
        return_value=torch.full_like(images, 0.2),
    ):
        loss, pred_loss, recon_loss = train_step(
            trainer,
            images=images,
            prefix=prefix,
            timesteps=torch.tensor([0.25]),
        )
    loss.backward()

    torch.testing.assert_close(context_call["image"], torch.full_like(images, 0.55))
    torch.testing.assert_close(context_call["timestep"], torch.tensor(0.25))
    assert context_call["noise_scale"] == 2.0
    assert (
        context_call["shape"].grid_h,
        context_call["shape"].grid_w,
        context_call["shape"].token_h,
        context_call["shape"].token_w,
    ) == (2, 2, 1, 1)
    assert transformer.index_call == (1, 1, 3, torch.device("cpu"))
    torch.testing.assert_close(loss, torch.tensor(16.0 / 9.0))
    assert pred_loss == pytest.approx(16.0 / 9.0)
    assert recon_loss == pytest.approx(1.0)
    assert transformer.language_model.model.layers[0].scale.grad is not None
    assert prefix.cache is cache and cache.layers[0].keys is key


def test_handler_is_registered_and_declares_pixel_grid():
    handler = SenseNovaArchHandler()
    assert handler.name == "sensenova"
    assert handler.pixel_align == 32
    from core.training.arch import ARCH_REGISTRY

    assert ARCH_REGISTRY["sensenova"] is SenseNovaArchHandler
