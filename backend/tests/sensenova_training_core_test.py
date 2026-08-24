import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.arch.base_arch import SampleContext, TrainStepContext
from core.training.arch.sensenova import SenseNovaArchHandler
from core.training.ops.sensenova_ops import (
    SenseNovaTrainingPrefix,
    _assert_supported_quantized_training_base,
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

    def __init__(self, use_pixel_head=True, use_deep_fm_head=False):
        super().__init__()
        self.use_pixel_head = use_pixel_head
        self.use_deep_fm_head = use_deep_fm_head
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

    def unpatchify(self, x, patch, h=None, w=None):
        return x.reshape(1, h // patch, w // patch, patch, patch, 3).permute(
            0, 5, 1, 3, 2, 4
        ).reshape(1, 3, h, w)


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


def _convrot_linear():
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    return ConvRotInt8Linear(
        256, 1, False, torch.bfloat16, convrot_groupsize=256, marker_numel=1
    )


def _attach_convrot_int8(transformer, count=588):
    transformer.quantized_linears = nn.ModuleList(
        [_convrot_linear() for _ in range(count)]
    )
    return transformer


def test_quantized_training_base_accepts_either_pure_flavour():
    _assert_supported_quantized_training_base(_attach_plain_int8(_Transformer()))
    _assert_supported_quantized_training_base(_attach_convrot_int8(_Transformer()))


def test_quantized_training_base_refuses_mixed_and_off_count_census():
    incomplete = _attach_plain_int8(_Transformer(), count=587)
    with pytest.raises(RuntimeError, match="Int8Linear=587"):
        _assert_supported_quantized_training_base(incomplete)

    short_convrot = _attach_convrot_int8(_Transformer(), count=587)
    with pytest.raises(RuntimeError, match="ConvRotInt8Linear=587"):
        _assert_supported_quantized_training_base(short_convrot)

    mixed = _attach_plain_int8(_Transformer(), count=587)
    mixed.quantized_linears.append(_convrot_linear())
    with pytest.raises(RuntimeError, match="ConvRotInt8Linear=1"):
        _assert_supported_quantized_training_base(mixed)

    over = _attach_convrot_int8(_Transformer(), count=589)
    with pytest.raises(RuntimeError, match="ConvRotInt8Linear=589"):
        _assert_supported_quantized_training_base(over)

    with pytest.raises(RuntimeError, match="bf16 base"):
        _assert_supported_quantized_training_base(_Transformer())


def test_quantized_training_base_refuses_unknown_quantized_subclass():
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    class _FutureConvRot(ConvRotInt8Linear):
        pass

    transformer = _Transformer()
    transformer.quantized_linears = nn.ModuleList(
        [
            _FutureConvRot(
                256, 1, False, torch.bfloat16, convrot_groupsize=256, marker_numel=1
            )
            for _ in range(588)
        ]
    )
    with pytest.raises(RuntimeError, match="_FutureConvRot=588"):
        _assert_supported_quantized_training_base(transformer)


def test_quantized_training_base_refuses_untested_pure_fp8_flavour():
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear

    transformer = _Transformer()
    transformer.quantized_linears = nn.ModuleList(
        [Fp8Linear(256, 1, False, torch.bfloat16) for _ in range(588)]
    )
    with pytest.raises(RuntimeError, match="Fp8Linear=588"):
        _assert_supported_quantized_training_base(transformer)


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


def _text_only_inputs(tokenizer, query, length=3):
    """Vendor `_build_t2i_text_inputs` shape: t is arange, h/w are zero."""
    t_idx = torch.arange(length, dtype=torch.long)
    zeros = torch.zeros_like(t_idx)
    return (
        torch.ones(1, length, dtype=torch.long),
        torch.stack([t_idx, zeros, zeros], dim=0),
        {"full_attention": None},
    )


def test_encode_prompt_builds_detached_training_prefix():
    cache = _Cache()
    transformer = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        _build_t2i_query=lambda prompt, **kwargs: prompt,
        _build_t2i_text_inputs=_text_only_inputs,
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
        _build_t2i_text_inputs=_text_only_inputs,
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


def test_flow_step_conditions_on_unquantized_timestep_under_bf16():
    """bf16 training must still condition on the fp32 t inference uses.

    bf16 carries ~2e-3 resolution near t=0.31, so casting t to training_dtype
    would feed timestep_embedder a different value than any inference step.
    """
    transformer = _Transformer()
    trainer = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        training_dtype=torch.bfloat16,
        gradient_checkpointing=False,
    )
    context_call = {}

    def build_context(model, shape, image, timestep, noise_scale):
        context_call.update(image=image.detach().clone(), timestep=timestep.detach().clone())
        return (
            model.patchify(image, 32),
            torch.full((1, 1, 1), 2.0, dtype=torch.bfloat16),
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
        )

    sampled = torch.tensor([0.3123456])
    assert sampled.to(torch.bfloat16).float() != sampled
    images = torch.ones(1, 3, 32, 32)
    with patch(
        "core.models.sensenova.sensenova_pipeline_ops.compute_noise_scale",
        return_value=2.0,
    ), patch(
        "core.models.sensenova.sensenova_pipeline_ops._build_step_context",
        side_effect=build_context,
    ), patch("torch.randn_like", return_value=torch.full_like(images, 0.2, dtype=torch.bfloat16)):
        loss, _, _ = train_step(
            trainer, images=images, prefix=SenseNovaTrainingPrefix(_Cache(), text_length=3),
            timesteps=sampled,
        )

    assert context_call["timestep"].dtype == torch.float32
    torch.testing.assert_close(context_call["timestep"], sampled[0])
    # The noised map keeps training_dtype: the gen-branch ViT runs outside autocast.
    assert context_call["image"].dtype == torch.bfloat16
    expected = sampled.to(torch.bfloat16) + (1 - sampled).to(torch.bfloat16) * torch.tensor(
        0.4, dtype=torch.bfloat16
    )
    torch.testing.assert_close(context_call["image"], torch.full_like(context_call["image"], expected.item()))
    assert loss.dtype == torch.float32


def _run_train_step(transformer):
    trainer = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=False,
    )
    return train_step(
        trainer,
        images=torch.ones(1, 3, 32, 32),
        prefix=SenseNovaTrainingPrefix(_Cache(), text_length=3),
        timesteps=torch.tensor([0.25]),
    )


def test_train_step_refuses_non_pixel_head_fm_decoders():
    with pytest.raises(RuntimeError, match="plain fm_head branch"):
        _run_train_step(_Transformer(use_pixel_head=False))

    with pytest.raises(RuntimeError, match="use_deep_fm_head branch"):
        _run_train_step(_Transformer(use_deep_fm_head=True))

    unknown = _Transformer()
    del unknown.use_pixel_head
    del unknown.use_deep_fm_head
    with pytest.raises(RuntimeError, match="use_pixel_head, use_deep_fm_head"):
        _run_train_step(unknown)


def test_load_components_refuses_non_pixel_head_fm_decoders():
    transformer = _attach_plain_int8(_Transformer(use_deep_fm_head=True))
    trainer = SimpleNamespace(
        model_path="checkpoint.safetensors",
        weight_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        attention_backend="native",
    )
    components = {"transformer": transformer, "tokenizer": object(), "config": object()}
    with patch(
        "core.models.sensenova.loader.load_sensenova_from_path",
        return_value=components,
    ), patch("core.training.ops.sensenova_ops.setup_attention_backend"):
        with pytest.raises(RuntimeError, match="use_deep_fm_head branch"):
            load_components(trainer)


def test_pixel_debug_dump_writes_previews_and_scalar_metrics(tmp_path):
    """The pixel-space equivalent of the latent archs' debug latents."""
    handler = SenseNovaArchHandler()
    transformer = _Transformer()
    trainer = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=False,
        log_prefix="[test]",
        debug_vram=False,
    )
    debug_dir = tmp_path / "step_000100"
    ctx = TrainStepContext(
        latents=torch.zeros(1, 3, 32, 32),
        sensenova_prefix=SenseNovaTrainingPrefix(_Cache(), text_length=3),
        timesteps=torch.tensor([0.25]),
        debug_save_path=debug_dir,
        debug_captions=["a caption"],
        debug_reference_image_paths=[None],
    )
    def build_context(model, shape, image, timestep, noise_scale):
        return model.patchify(image, 32), torch.full((1, 1, 1), 2.0), torch.ones(1, 1, 1)

    with patch(
        "core.models.sensenova.sensenova_pipeline_ops.compute_noise_scale",
        return_value=2.0,
    ), patch(
        "core.models.sensenova.sensenova_pipeline_ops._build_step_context",
        side_effect=build_context,
    ):
        handler.train_step(trainer, ctx)

    saved = torch.load(debug_dir / "latents_t0.2500.pt", map_location="cpu")
    assert saved["model_type"] == "sensenova" and saved["is_latent"] is False
    assert saved["caption"] == "a caption"
    assert saved["noise_scale"] == 2.0 and saved["loss"] > 0
    # Filenames the visualize endpoint derives from the .pt name.
    for name in ("noisy", "target", "pred_x0"):
        assert (debug_dir / f"decode_t0.2500_{name}.webp").exists()


class _SampleEvictor:
    def __init__(self, calls):
        self.calls = calls
        self.state = "full"

    def enter_prefix(self):
        self.calls.append("enter_prefix")
        self.state = "prefix"

    def enter_denoise(self):
        self.calls.append("enter_denoise")
        self.state = "denoise"

    def assert_generation_resident(self):
        self.calls.append("assert_resident")


def _sample_trainer(transformer, evictor=None):
    return SimpleNamespace(
        transformer=transformer,
        tokenizer=object(),
        attention_backend="native",
        log_prefix="[test]",
        _resolve_training_backend=lambda backend: f"resolved-{backend}",
        move_main_model_to_gpu=lambda: None,
        sensenova_phase_evictor=evictor,
    )


def _patched_sample(trainer, calls, *, denoise=None, **overrides):
    from core.attention import AttentionMode

    ops = "core.models.sensenova.sensenova_pipeline_ops"

    def _set_backend(model, backend, mode=None):
        calls.append(("attention", backend, mode))
        return 42

    def _encode(model, tokenizer, prompt, height, width, cfg_scale, **kwargs):
        calls.append(("encode", prompt, height, width, cfg_scale, kwargs.get("negative_prompt")))
        return SimpleNamespace(consumed=False)

    def _denoise(model, prefix, **kwargs):
        calls.append(("denoise", kwargs, torch.is_grad_enabled(), model.training))
        if denoise is not None:
            return denoise()
        return torch.zeros(1, 3, 8, 8)

    kwargs = dict(
        prompt="a prompt",
        height=860,
        width=1536,
        num_inference_steps=8,
        guidance_scale=4.0,
        seed=7,
        negative_prompt="bad",
    )
    kwargs.update(overrides)
    with patch(f"{ops}.set_attention_backend", side_effect=_set_backend), patch(
        f"{ops}.encode_prompt", side_effect=_encode
    ), patch(f"{ops}.denoise_loop", side_effect=_denoise), patch(
        f"{ops}.clear_prefix_caches", side_effect=lambda p: calls.append(("cleared",))
    ):
        image = SenseNovaArchHandler().sample(
            trainer,
            SampleContext(
                prompt=kwargs["prompt"],
                width=kwargs["width"],
                height=kwargs["height"],
                num_inference_steps=kwargs["num_inference_steps"],
                guidance_scale=kwargs["guidance_scale"],
                seed=kwargs["seed"],
                negative_prompt=kwargs["negative_prompt"],
            ),
        )
    return image, AttentionMode


def test_sample_runs_inference_mode_and_restores_training_state():
    transformer = _Transformer()
    transformer.train()
    calls = []
    image, AttentionMode = _patched_sample(_sample_trainer(transformer), calls)

    assert image.size == (8, 8)
    attention = [c for c in calls if c[0] == "attention"]
    # Mode is passed EXPLICITLY both ways: the inference stamp must not be left
    # behind, and set_attention_backend would otherwise infer it from grad state.
    assert attention == [
        ("attention", "resolved-native", AttentionMode.INFERENCE),
        ("attention", "resolved-native", AttentionMode.TRAINING),
    ]
    assert transformer.training

    encode = next(c for c in calls if c[0] == "encode")
    # 860 snapped up to the 32px token grid, 1536 already aligned.
    assert encode[1:] == ("a prompt", 864, 1536, 4.0, "bad")
    denoise = next(c for c in calls if c[0] == "denoise")
    assert denoise[1]["seed"] == 7 and denoise[1]["num_inference_steps"] == 8
    assert denoise[1]["cfg_scale"] == 4.0
    assert denoise[2] is False  # no_grad
    assert denoise[3] is False  # eval() during generation
    assert ("cleared",) in calls


def test_sample_random_seed_and_defaults_come_from_param_defaults():
    from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS

    calls = []
    _patched_sample(_sample_trainer(_Transformer()), calls, seed=-1)
    denoise = next(c for c in calls if c[0] == "denoise")
    assert denoise[1]["seed"] is None
    assert denoise[1]["timestep_shift"] == SENSENOVA_GENERATION_DEFAULTS["timestep_shift"]
    assert denoise[1]["cfg_norm"] == SENSENOVA_GENERATION_DEFAULTS["cfg_norm"]


def test_sample_failure_returns_none_and_still_restores_training_mode():
    transformer = _Transformer()
    transformer.train()
    calls = []

    def _boom():
        raise RuntimeError("denoise exploded")

    image, AttentionMode = _patched_sample(
        _sample_trainer(transformer), calls, denoise=_boom
    )

    assert image is None
    assert transformer.training
    assert calls[-2] == ("cleared",)
    assert calls[-1] == ("attention", "resolved-native", AttentionMode.TRAINING)


def test_sample_drives_the_phase_evictor_state_machine():
    calls = []
    evictor = _SampleEvictor(calls)
    _patched_sample(_sample_trainer(_Transformer(), evictor), calls)

    ordered = [c if isinstance(c, str) else c[0] for c in calls]
    assert ordered.index("enter_prefix") < ordered.index("encode")
    assert ordered.index("encode") < ordered.index("enter_denoise")
    assert ordered.index("assert_resident") < ordered.index("denoise")
    # A completed sample leaves the machine where a training step expects it.
    assert evictor.state == "denoise"


def test_sample_failure_leaves_the_evictor_in_a_recoverable_state():
    calls = []
    evictor = _SampleEvictor(calls)

    def _boom():
        raise RuntimeError("denoise exploded")

    image, _ = _patched_sample(
        _sample_trainer(_Transformer(), evictor), calls, denoise=_boom
    )
    assert image is None
    # "denoise" and "prefix" are both states encode_prompt can transition out of.
    assert evictor.state in ("prefix", "denoise")


def test_handler_is_registered_and_declares_pixel_grid():
    handler = SenseNovaArchHandler()
    assert handler.name == "sensenova"
    assert handler.pixel_align == 32
    from core.training.arch import ARCH_REGISTRY

    assert ARCH_REGISTRY["sensenova"] is SenseNovaArchHandler
