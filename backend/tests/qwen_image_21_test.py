import json
import asyncio
import sys
from pathlib import Path

import torch
import pytest
from PIL import Image
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.model_loader import ModelLoader
from core.models.qwen_image_21.artifact import FORMAT_VERSION, MODEL_TYPE, load_manifest
from core.models.qwen_image_21.vendor import QwenImage21Transformer2DModel
from core.models.qwen_image_21.lora import (
    build_lora_branch,
    iter_lora_slots,
    normalise_lora_state_dict,
)
from core.extensions.lora_manager import classify_lora_keys
from core.training.arch.qwen_image_21 import QwenImage21ArchHandler
from core.training.adapters.qwen_image_21_adapter import (
    QwenImage21FullParameterAdapter,
    QwenImage21LoRAAdapter,
)
from core.training.qwen_partition import (
    PartitionBox,
    QwenPartitionGlobalAdapter,
    build_fixed_partition_plan,
    flatten_region,
    full_canvas_position_ids,
)
from core.training.ops import qwen_image_21_ops
from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin
from api import routes
from api.schema_routes import get_arch_capabilities, get_generation_defaults


class _DebugTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])

    def forward(self, *, hidden_states, **_kwargs):
        return (hidden_states * self.scale,)


class _GuidanceTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])

    def forward(self, *, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, img_mask, **_kwargs):
        assert img_mask.shape[1] == encoder_hidden_states.shape[1] + hidden_states.shape[1] // 4
        assert encoder_hidden_states_mask.shape[1] == encoder_hidden_states.shape[1]
        return (hidden_states * self.scale + encoder_hidden_states.mean(dim=(1, 2), keepdim=True),)


def test_qwen_partition_retains_trainable_encoder_graph_until_last_region():
    transformer = _GuidanceTransformer()
    encoder = torch.nn.Linear(4, 4, bias=False)
    features = encoder(torch.ones(1, 2, 4))
    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, use_grad_scaler=False, transformer=transformer,
        arch=QwenImage21ArchHandler(),
        config={"dit_partition_fixed_count": 2},
        _active_mnt_noise=torch.ones(1, 32, 4),
        reconstruction_loss_weight=0.0,
        log_extra_metric=lambda *_args: None,
    )
    qwen_image_21_ops.train_step_partitioned_backward(
        trainer, latents=torch.zeros(1, 32, 4), encoder_features=features,
        encoder_mask=torch.ones(1, 2, dtype=torch.bool),
        timesteps=torch.tensor([0.5]), latent_h=4, latent_w=8,
        backward_scale=1.0,
    )
    assert encoder.weight.grad is not None
    assert encoder.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("partitioned", [False, True])
def test_qwen_guidance_loss_mixes_targets_and_detaches_null_forward(partitioned):
    transformer = _GuidanceTransformer()
    clean = torch.zeros(1, 32, 4)
    noise = torch.ones_like(clean)
    metrics = {}
    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, use_grad_scaler=False, transformer=transformer,
        arch=QwenImage21ArchHandler(),
        config={"dit_partition_fixed_count": 2, "qwen_guidance_loss_weight": 0.25,
                "qwen_guidance_loss_scale": 3.0, "qwen_guidance_loss_schedule": "sigma",
                "qwen_guidance_loss_mix_mode": "blend"},
        _active_mnt_noise=noise,
        _qwen_guidance_blank_encoding=(torch.zeros(1, 3, 4), torch.ones(3, dtype=torch.bool)),
        log_extra_metric=lambda key, value: metrics.__setitem__(key, value),
    )
    inputs = dict(
        latents=clean, encoder_features=torch.ones(1, 2, 4),
        encoder_mask=torch.ones(1, 2, dtype=torch.bool),
        timesteps=torch.tensor([0.5]), latent_h=4, latent_w=8,
        cfg_drop_mask=torch.tensor([False]),
    )
    if partitioned:
        loss_value, _, _ = qwen_image_21_ops.train_step_partitioned_backward(
            trainer, **inputs, backward_scale=1.0
        )
    else:
        loss, _, _ = qwen_image_21_ops.train_step(trainer, **inputs)
        loss_value = float(loss.detach())
        loss.backward()

    conditional = 1.25
    unconditional = 0.25
    guided_target = unconditional + 2.0 * (1.0 - unconditional)
    expected_normal = (conditional - 1.0) ** 2
    expected_guided = (conditional - guided_target) ** 2
    assert loss_value == pytest.approx(0.75 * expected_normal + 0.25 * expected_guided)
    assert metrics["qwen_guidance_loss_normal"] == pytest.approx(expected_normal)
    assert metrics["qwen_guidance_loss_guided"] == pytest.approx(expected_guided)
    assert metrics["qwen_guidance_loss_mix_weight"] == pytest.approx(0.25)
    # The no-gradient null prediction is a fixed target, not a second gradient path.
    expected_grad = 0.75 * 2 * (conditional - 1.0) * 0.5 + 0.25 * 2 * (conditional - guided_target) * 0.5
    assert float(transformer.scale.grad) == pytest.approx(expected_grad)


def test_qwen_guidance_loss_skips_cfg_null_items():
    transformer = _GuidanceTransformer()
    metrics = {}
    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, transformer=transformer,
        config={"qwen_guidance_loss_weight": 0.5},
        _active_mnt_noise=torch.ones(1, 32, 4),
        log_extra_metric=lambda key, value: metrics.__setitem__(key, value),
    )
    loss, _, _ = qwen_image_21_ops.train_step(
        trainer, latents=torch.zeros(1, 32, 4),
        encoder_features=torch.zeros(1, 3, 4),
        encoder_mask=torch.ones(1, 3, dtype=torch.bool),
        timesteps=torch.tensor([0.5]), latent_h=4, latent_w=8,
        cfg_drop_mask=torch.tensor([True]),
    )
    assert float(loss) == pytest.approx(0.75 ** 2)
    assert metrics == {}


@pytest.mark.parametrize("partitioned", [False, True])
def test_qwen_stochastic_selection_is_per_image_and_fixed_across_tiles(monkeypatch, partitioned):
    class CountingTransformer(_GuidanceTransformer):
        def __init__(self):
            super().__init__()
            self.forward_batches = []

        def forward(self, **kwargs):
            self.forward_batches.append(kwargs["hidden_states"].shape[0])
            return super().forward(**kwargs)

    monkeypatch.setattr(torch, "rand_like", lambda value: value.new_tensor([0.1, 0.9, 0.1]))
    transformer = CountingTransformer()
    metrics = {}
    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, use_grad_scaler=False, transformer=transformer,
        arch=QwenImage21ArchHandler(),
        config={"dit_partition_fixed_count": 2, "qwen_guidance_loss_weight": 0.5,
                "qwen_guidance_loss_scale": 3.0, "qwen_guidance_loss_schedule": "sigma",
                "qwen_guidance_loss_mix_mode": "stochastic"},
        _active_mnt_noise=torch.ones(3, 32, 4),
        _qwen_guidance_blank_encoding=(torch.zeros(1, 3, 4), torch.ones(3, dtype=torch.bool)),
        log_extra_metric=lambda key, value: metrics.__setitem__(key, value),
    )
    inputs = dict(
        latents=torch.zeros(3, 32, 4), encoder_features=torch.ones(3, 2, 4),
        encoder_mask=torch.ones(3, 2, dtype=torch.bool),
        timesteps=torch.full((3,), 0.5), latent_h=4, latent_w=8,
        cfg_drop_mask=torch.tensor([False, False, True]),
    )
    if partitioned:
        loss_value, _, _ = qwen_image_21_ops.train_step_partitioned_backward(
            trainer, **inputs, backward_scale=1.0
        )
    else:
        loss, _, _ = qwen_image_21_ops.train_step(trainer, **inputs)
        loss_value = float(loss.detach())
        loss.backward()
    guided_target = 0.25 + 2.0 * (1.0 - 0.25)
    expected = ((1.25 - guided_target) ** 2 + (1.25 - 1.0) ** 2 +
                (1.25 - 1.0) ** 2) / 3
    assert loss_value == pytest.approx(expected)
    assert float(transformer.scale.grad) == pytest.approx(
        (2 * (1.25 - guided_target) + 4 * (1.25 - 1.0)) * 0.5 / 3,
        abs=1e-7,
    )
    assert transformer.forward_batches == ([1, 3] * 2 if partitioned else [1, 3])
    assert metrics["qwen_guidance_loss_mix_weight"] == pytest.approx(1 / 3)
    assert metrics["qwen_guidance_loss_selected_fraction"] == pytest.approx(1 / 3)


def test_qwen_stochastic_probability_follows_sigma_schedule(monkeypatch):
    monkeypatch.setattr(torch, "rand_like", lambda value: value.new_tensor([0.3, 0.6, 0.99]))
    trainer = SimpleNamespace(config={"qwen_guidance_loss_mix_mode": "stochastic"})
    condition = (None, None, torch.tensor([True, True, True]),
                 0.25, 3.0, "sigma", "high_noise_smoothstep", 1.0, 0.5, 0.8)
    selection, guided = qwen_image_21_ops._guidance_selection(
        trainer, torch.tensor([0.2, 0.65, 1.0]), condition
    )
    assert guided.tolist() == [False, True, True]
    assert selection.tolist() == [0.0, 1.0, 1.0]


def test_qwen_stochastic_selection_reuses_recovery_draw(monkeypatch):
    monkeypatch.setattr(torch, "rand_like", lambda _value: pytest.fail("redrew loss choice"))
    condition = (None, None, torch.tensor([True, True]), 0.5, 3.0, "sigma")
    trainer = SimpleNamespace(
        config={"qwen_guidance_loss_mix_mode": "stochastic"},
        _qwen_guidance_draw=torch.tensor([0.2, 0.8]),
    )
    first = qwen_image_21_ops._guidance_selection(
        trainer, torch.tensor([0.5, 0.5]), condition
    )[1]
    second = qwen_image_21_ops._guidance_selection(
        trainer, torch.tensor([0.5, 0.5]), condition
    )[1]
    assert first.tolist() == second.tolist() == [True, False]


@pytest.mark.parametrize("partitioned", [False, True])
def test_qwen_stochastic_uncond_keeps_transformer_output_dtype(partitioned):
    class FloatOutputTransformer(_GuidanceTransformer):
        def forward(self, **kwargs):
            return (super().forward(**kwargs)[0].float(),)

    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.bfloat16,
        mixed_precision=False, use_grad_scaler=False,
        transformer=FloatOutputTransformer(), arch=QwenImage21ArchHandler(),
        config={"dit_partition_fixed_count": 2,
                "qwen_guidance_loss_weight": 1.0,
                "qwen_guidance_loss_mix_mode": "stochastic"},
        _active_mnt_noise=torch.ones(1, 32, 4, dtype=torch.bfloat16),
        _qwen_guidance_blank_encoding=(torch.zeros(1, 3, 4), torch.ones(3, dtype=torch.bool)),
        log_extra_metric=lambda *_args: None,
    )
    inputs = dict(
        latents=torch.zeros(1, 32, 4, dtype=torch.bfloat16),
        encoder_features=torch.ones(1, 2, 4, dtype=torch.bfloat16),
        encoder_mask=torch.ones(1, 2, dtype=torch.bool),
        timesteps=torch.tensor([0.5]), latent_h=4, latent_w=8,
    )
    if partitioned:
        loss, _, _ = qwen_image_21_ops.train_step_partitioned_backward(
            trainer, **inputs, backward_scale=1.0
        )
    else:
        value, _, _ = qwen_image_21_ops.train_step(trainer, **inputs)
        value.backward()
        loss = float(value.detach())
    assert torch.isfinite(torch.tensor(loss))
    assert trainer.transformer.scale.grad is not None


def test_qwen_sigma_null_drop_is_nested_under_ordinary_selection(monkeypatch):
    monkeypatch.setattr(torch, "rand_like", lambda value: value.new_tensor([0.1, 0.28, 0.5]))
    config = {
        "qwen_guidance_loss_weight": 0.25,
        "qwen_guidance_loss_weight_schedule": "constant",
    }
    null_mask, draw = qwen_image_21_ops.sigma_null_drop_mask(
        config, torch.tensor([0.2, 0.2, 0.2]), 0.1
    )
    assert null_mask.tolist() == [False, True, False]

    condition = (
        None, None, torch.ones(3, dtype=torch.bool), 0.25, 3.0, "sigma",
    )
    trainer = SimpleNamespace(
        config={"qwen_guidance_loss_mix_mode": "stochastic"},
        _qwen_guidance_draw=draw,
    )
    _, guided = qwen_image_21_ops._guidance_selection(
        trainer, torch.tensor([0.2, 0.2, 0.2]), condition
    )
    assert guided.tolist() == [True, False, False]


def test_qwen_sigma_null_replaces_only_selected_caption_rows():
    features = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    mask = torch.ones(3, 2, dtype=torch.bool)
    blank = (torch.full((1, 3, 4), 7.0), torch.tensor([True, True, False]))
    replaced, replaced_mask = qwen_image_21_ops.apply_sigma_null_condition(
        features, mask, torch.tensor([False, True, False]), blank
    )
    assert replaced.shape == (3, 3, 4)
    assert torch.equal(replaced[1], blank[0][0])
    assert replaced_mask[1].tolist() == [True, True, False]
    assert torch.equal(replaced[0, :2], features[0])
    assert torch.equal(replaced[2, :2], features[2])
    assert features.shape == (3, 2, 4)


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("sigma, weight", [(0.2, 0.25), (0.5, 0.25), (0.65, 0.625),
                                            (0.8, 1.0), (1.0, 1.0)])
def test_qwen_high_noise_mix_schedule_matches_full_and_partitioned_gradients(partitioned, sigma, weight):
    transformer = _GuidanceTransformer()
    metrics = {}
    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, use_grad_scaler=False, transformer=transformer,
        arch=QwenImage21ArchHandler(),
        config={
            "dit_partition_fixed_count": 2,
            "qwen_guidance_loss_weight": 0.25,
            "qwen_guidance_loss_mix_mode": "blend",
            "qwen_guidance_loss_scale": 3.0,
            "qwen_guidance_loss_schedule": "sigma",
            "qwen_guidance_loss_weight_schedule": "high_noise_smoothstep",
            "qwen_guidance_loss_high_noise_weight": 1.0,
            "qwen_guidance_loss_ramp_start": 0.5,
            "qwen_guidance_loss_ramp_end": 0.8,
        },
        _active_mnt_noise=torch.ones(1, 32, 4),
        _qwen_guidance_blank_encoding=(torch.zeros(1, 3, 4), torch.ones(3, dtype=torch.bool)),
        log_extra_metric=lambda key, value: metrics.__setitem__(key, value),
    )
    inputs = dict(
        latents=torch.zeros(1, 32, 4), encoder_features=torch.ones(1, 2, 4),
        encoder_mask=torch.ones(1, 2, dtype=torch.bool),
        timesteps=torch.tensor([sigma]), latent_h=4, latent_w=8,
    )
    if partitioned:
        loss_value, _, _ = qwen_image_21_ops.train_step_partitioned_backward(
            trainer, **inputs, backward_scale=1.0
        )
    else:
        loss, _, _ = qwen_image_21_ops.train_step(trainer, **inputs)
        loss_value = float(loss.detach())
        loss.backward()
    conditional = 1.0 + 0.5 * sigma
    unconditional = 0.5 * sigma
    target = 1.0
    guided = unconditional + (1 + 2 * sigma) * (target - unconditional)
    expected_loss = (1 - weight) * (conditional - target) ** 2 + weight * (conditional - guided) ** 2
    expected_grad = 2 * sigma * ((1 - weight) * (conditional - target) + weight * (conditional - guided))
    assert loss_value == pytest.approx(expected_loss, abs=2e-6)
    assert float(transformer.scale.grad) == pytest.approx(expected_grad, abs=2e-6)
    assert metrics["qwen_guidance_loss_mix_weight"] == pytest.approx(weight, abs=1e-6)


def test_qwen_guidance_api_defaults_match_contract():
    from api.param_defaults import TRAINING_DEFAULTS

    fields = routes.TrainingRunCreateRequest.model_fields
    for name in ("qwen_guidance_loss_weight", "qwen_guidance_loss_mix_mode",
                 "qwen_cfg_null_sigma_schedule", "qwen_guidance_loss_scale",
                 "qwen_guidance_loss_schedule", "qwen_guidance_loss_weight_schedule",
                 "qwen_guidance_loss_high_noise_weight", "qwen_guidance_loss_ramp_start",
                 "qwen_guidance_loss_ramp_end"):
        assert fields[name].default == TRAINING_DEFAULTS[name]


def test_qwen_legacy_guidance_config_keeps_blend_on_resume_and_edit():
    from api.routes import _extract_request_params_from_yaml

    params = _extract_request_params_from_yaml(
        {"train": {"qwen_guidance_loss_weight": 0.25}}, "lora"
    )
    assert params["qwen_guidance_loss_mix_mode"] == "blend"
    condition = (None, None, torch.tensor([True]), 0.25, 3.0, "sigma")
    selection, guided = qwen_image_21_ops._guidance_selection(
        SimpleNamespace(config={}), torch.tensor([0.5]), condition
    )
    assert selection.tolist() == [0.25]
    assert guided.tolist() == [True]


@pytest.mark.parametrize("overrides", [
    {"qwen_guidance_loss_ramp_start": 0.8, "qwen_guidance_loss_ramp_end": 0.8},
    {"qwen_guidance_loss_weight": 0.5,
     "qwen_guidance_loss_weight_schedule": "high_noise_smoothstep",
     "qwen_guidance_loss_high_noise_weight": 0.25},
])
def test_qwen_guidance_ramp_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        routes.TrainingRunCreateRequest(
            base_model_path="unused", training_method="lora", **overrides
        )


def test_qwen_guidance_sigma_schedule_returns_to_ordinary_at_clean_end():
    pred = torch.tensor([[[0.25]]])
    target = torch.tensor([[[1.0]]])
    uncond = torch.tensor([[[0.5]]])
    condition = (None, None, torch.tensor([True]), 1.0, 3.0, "sigma")
    loss, normal, guided = qwen_image_21_ops._guidance_loss(
        pred, target, uncond, torch.tensor([0.0]), condition
    )
    assert float(loss) == pytest.approx(float(normal))
    assert float(guided) == pytest.approx(float(normal))


@pytest.mark.parametrize("partitioned", [False, True])
def test_qwen_reconstruction_loss_is_reported_and_weighted(partitioned):
    transformer = _DebugTransformer()
    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, use_grad_scaler=False, transformer=transformer,
        arch=QwenImage21ArchHandler(), config={"dit_partition_fixed_count": 2},
        reconstruction_loss_weight=0.2,
        _active_mnt_noise=torch.ones(1, 32, 4),
        log_extra_metric=lambda *_args: None,
    )
    inputs = dict(
        latents=torch.zeros(1, 32, 4),
        encoder_features=torch.zeros(1, 2, 4),
        encoder_mask=torch.ones(1, 2, dtype=torch.bool),
        timesteps=torch.tensor([0.5]), latent_h=4, latent_w=8,
    )
    if partitioned:
        total, pred, recon = qwen_image_21_ops.train_step_partitioned_backward(
            trainer, **inputs, backward_scale=1.0
        )
    else:
        total, pred, recon = qwen_image_21_ops.train_step(trainer, **inputs)
    assert float(pred) == pytest.approx(0.5625)
    assert float(recon) == pytest.approx(0.140625)
    assert float(total) == pytest.approx(0.8 * float(pred) + 0.2 * float(recon))


def test_qwen_hybrid_and_debug_view_survive_config_generation():
    from core.training.training_config import _build_train_section, train_section_key_vocabulary

    request = routes.TrainingRunCreateRequest(
        base_model_path="unused", training_method="lora",
        qwen_guidance_loss_weight=0.25, qwen_guidance_loss_scale=3.0,
        qwen_cfg_null_sigma_schedule=True,
        qwen_guidance_loss_schedule="sigma", qwen_debug_latent_view="pixel",
        qwen_guidance_loss_weight_schedule="high_noise_smoothstep",
        qwen_guidance_loss_high_noise_weight=1.0,
        qwen_guidance_loss_ramp_start=0.5, qwen_guidance_loss_ramp_end=0.8,
    )
    train = _build_train_section(
        request.model_dump(), total_steps=20, epochs=None,
        train_unet=True, train_text_encoder=False, arch="qwen_image_21",
    )
    for key, expected in (
        ("qwen_guidance_loss_weight", 0.25),
        ("qwen_cfg_null_sigma_schedule", True),
        ("qwen_guidance_loss_scale", 3.0),
        ("qwen_guidance_loss_schedule", "sigma"),
        ("qwen_guidance_loss_weight_schedule", "high_noise_smoothstep"),
        ("qwen_guidance_loss_high_noise_weight", 1.0),
        ("qwen_guidance_loss_ramp_start", 0.5),
        ("qwen_guidance_loss_ramp_end", 0.8),
        ("qwen_debug_latent_view", "pixel"),
    ):
        assert train[key] == expected
        assert key in train_section_key_vocabulary()


def test_qwen_sigma_null_schedule_rejects_blend_mode():
    with pytest.raises(ValueError, match="requires stochastic guidance loss selection"):
        routes.TrainingRunCreateRequest(
            base_model_path="unused",
            training_method="lora",
            qwen_guidance_loss_weight=0.25,
            qwen_guidance_loss_mix_mode="blend",
            qwen_cfg_null_sigma_schedule=True,
        )


def test_qwen_pixel_debug_decodes_after_forward(tmp_path):
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(latents_mean=[0] * 4, latents_std=[1] * 4)

        def decode(self, latents, return_dict=False):
            return (latents[:, :3],)

    class Processor:
        def postprocess(self, image, output_type):
            assert output_type == "pil"
            return [Image.new("RGB", (8, 4), "pink")]

    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        vae_dtype=torch.float32, mixed_precision=False,
        transformer=_DebugTransformer(), vae=VAE(),
        qwen_image_21_pipeline=SimpleNamespace(image_processor=Processor()),
        config={"qwen_debug_latent_view": "pixel"},
        _active_mnt_noise=torch.ones(1, 32, 4),
        log_extra_metric=lambda *_args: None,
    )
    qwen_image_21_ops.train_step(
        trainer, latents=torch.zeros(1, 32, 4),
        encoder_features=torch.zeros(1, 2, 4),
        encoder_mask=torch.ones(1, 2, dtype=torch.bool),
        timesteps=torch.tensor([0.5]), latent_h=4, latent_w=8,
        debug_save_path=tmp_path,
    )
    assert list(tmp_path.glob("decode_*.webp")) == []
    qwen_image_21_ops.flush_pending_debug_previews(trainer)
    assert sorted(p.name for p in tmp_path.glob("decode_*.webp")) == [
        "decode_t0.5000_noisy.webp", "decode_t0.5000_pred_x0.webp",
        "decode_t0.5000_target.webp",
    ]
    saved = torch.load(next(tmp_path.glob("latents_t*.pt")), weights_only=False)
    assert saved["recon_loss"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("partitioned", [False, True])
def test_qwen_guidance_loss_bf16_cuda_smoke(partitioned):
    transformer = _GuidanceTransformer().cuda()
    trainer = SimpleNamespace(
        device=torch.device("cuda"), training_dtype=torch.bfloat16,
        mixed_precision=True, use_grad_scaler=False, transformer=transformer,
        arch=QwenImage21ArchHandler(),
        config={"dit_partition_fixed_count": 2, "qwen_guidance_loss_weight": 0.25},
        _active_mnt_noise=torch.ones(1, 32, 4, device="cuda", dtype=torch.bfloat16),
        _qwen_guidance_blank_encoding=(torch.zeros(1, 3, 4), torch.ones(3, dtype=torch.bool)),
        log_extra_metric=lambda *_args: None,
    )
    inputs = dict(
        latents=torch.zeros(1, 32, 4, device="cuda", dtype=torch.bfloat16),
        encoder_features=torch.ones(1, 2, 4, device="cuda", dtype=torch.bfloat16),
        encoder_mask=torch.ones(1, 2, device="cuda", dtype=torch.bool),
        timesteps=torch.tensor([0.5], device="cuda"), latent_h=4, latent_w=8,
    )
    if partitioned:
        qwen_image_21_ops.train_step_partitioned_backward(
            trainer, **inputs, backward_scale=1.0
        )
    else:
        loss, _, _ = qwen_image_21_ops.train_step(trainer, **inputs)
        loss.backward()
    assert torch.isfinite(transformer.scale.grad)


@pytest.mark.parametrize("partitioned", [False, True])
def test_qwen_debug_latents_save_full_canvas(tmp_path, partitioned):
    height, width, channels = 4, 8, 4
    clean = torch.arange(height * width * channels, dtype=torch.float32).reshape(
        1, height * width, channels
    ) / 100
    noise = torch.ones_like(clean)
    sigma = torch.tensor([0.25])
    transformer = _DebugTransformer()
    trainer = SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, use_grad_scaler=False, transformer=transformer,
        arch=QwenImage21ArchHandler(), config={"dit_partition_fixed_count": 2},
        _active_mnt_noise=noise, log_prefix="[test]",
        log_extra_metric=lambda *_args: None,
    )
    debug_path = tmp_path / "step_000200"
    inputs = dict(
        latents=clean, encoder_features=torch.zeros(1, 2, channels),
        encoder_mask=torch.ones(1, 2), timesteps=sigma,
        latent_h=height, latent_w=width, debug_save_path=debug_path,
        debug_captions=["caption"],
    )
    if partitioned:
        qwen_image_21_ops.train_step_partitioned_backward(
            trainer, **inputs, backward_scale=1.0
        )
    else:
        qwen_image_21_ops.train_step(trainer, **inputs)

    saved = torch.load(next(debug_path.glob("latents_t*.pt")), weights_only=False)
    noisy = (1 - sigma) * clean + sigma * noise
    expected_prediction = noisy * transformer.scale.detach()
    as_grid = lambda packed: packed.transpose(1, 2).reshape(1, channels, height, width)
    torch.testing.assert_close(saved["latents"], as_grid(clean))
    torch.testing.assert_close(saved["noisy_latents"], as_grid(noisy))
    torch.testing.assert_close(saved["predicted_velocity"], as_grid(expected_prediction))
    torch.testing.assert_close(
        saved["predicted_latent"], as_grid(noisy - sigma * expected_prediction)
    )
    assert saved["partition_count"] == (2 if partitioned else 1)
    assert saved["caption"] == "caption"


def test_manifest_detection(tmp_path):
    (tmp_path / "processor").mkdir()
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "te.json").write_text("{}", encoding="utf-8")
    manifest = {
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "components": {
            "transformer": "dit.safetensors",
            "text_encoder": "te.safetensors",
            "vae": "vae.safetensors",
            "processor": "processor",
            "scheduler": "scheduler",
            "text_encoder_config": "te.json",
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert ModelLoader.detect_model_type(str(tmp_path)) == MODEL_TYPE
    parsed = load_manifest(str(tmp_path))
    assert parsed.transformer.endswith("dit.safetensors")


def test_model_selector_expands_prepared_variants(tmp_path):
    root = tmp_path / "qwen21"
    for variant in ("original", "int8_convrot"):
        variant_dir = root / variant
        variant_dir.mkdir(parents=True)
        (variant_dir / "dit.safetensors").write_bytes(b"dit")
        (variant_dir / "te.safetensors").write_bytes(b"te")
        (variant_dir / "vae.safetensors").write_bytes(b"vae")
        (variant_dir / "manifest.json").write_text(json.dumps({
            "model_type": MODEL_TYPE,
            "format_version": FORMAT_VERSION,
            "variant": variant,
            "components": {
                "transformer": "dit.safetensors",
                "text_encoder": "te.safetensors",
                "vae": "vae.safetensors",
                "processor": "processor",
                "text_encoder_config": "te.json",
            },
        }), encoding="utf-8")
    entries = routes._expand_qwen_image_21_tree(str(root), "qwen21", str(tmp_path))
    assert [(entry["name"], entry["architecture"], entry["variant"]) for entry in entries] == [
        ("qwen21/int8_convrot", MODEL_TYPE, "int8_convrot"),
        ("qwen21/original", MODEL_TYPE, "original"),
    ]


class _ModelDirectoryDB:
    def __init__(self, model_dirs):
        self._record = SimpleNamespace(model_dirs=model_dirs)

    def query(self, *_args, **_kwargs):
        return SimpleNamespace(first=lambda: self._record)


def _scan_models(model_dirs, monkeypatch, tmp_path):
    monkeypatch.setattr(routes.settings, "models_dir", str(tmp_path / "empty_default"))
    monkeypatch.setattr(routes, "_models_cache", None)
    monkeypatch.setattr(routes, "_models_cache_timestamp", None)
    return routes.get_models(
        db=_ModelDirectoryDB(model_dirs), force_rescan=True
    )["models"]


def test_model_selector_scans_qwen_parent_and_family_roots(tmp_path, monkeypatch):
    family_root = tmp_path / "models" / "qwen21"
    for variant in ("original", "int8_convrot"):
        variant_dir = family_root / variant
        variant_dir.mkdir(parents=True)
        (variant_dir / "dit.safetensors").write_bytes(b"dit")
        (variant_dir / "manifest.json").write_text(json.dumps({
            "model_type": MODEL_TYPE,
            "format_version": FORMAT_VERSION,
            "variant": variant,
            "components": {"transformer": "dit.safetensors"},
        }), encoding="utf-8")

    for configured_dir in (family_root.parent, family_root):
        entries = _scan_models([str(configured_dir)], monkeypatch, tmp_path)
        qwen_entries = [entry for entry in entries if entry.get("architecture") == MODEL_TYPE]
        assert {entry["variant"] for entry in qwen_entries} == {"original", "int8_convrot"}
        assert all(entry["path"] != str(family_root) for entry in qwen_entries)


def test_tiny_transformer_flow_training_backward():
    model = QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=32,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    clean = torch.randn(1, 4, 8)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.4])
    noisy = (1 - sigma[:, None, None]) * clean + sigma[:, None, None] * noise
    text = torch.randn(1, 3, 32)
    text_mask = torch.ones(1, 3, dtype=torch.long)
    img_mask = torch.tensor([[False, False, False, True]])
    prediction = model(
        hidden_states=noisy,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=text_mask,
        timestep=sigma,
        img_shapes=[[(1, 2, 2)]],
        img_mask=img_mask,
        return_dict=False,
    )[0][:, -clean.shape[1]:]
    target = noise - clean
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    assert prediction.shape == clean.shape
    assert model.transformer_blocks[0].attn.to_q.weight.grad is not None


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("halo", [0, 2])
def test_qwen_fixed_partition_covers_full_canvas_and_preserves_global_positions(count, halo):
    plan = build_fixed_partition_plan(
        8, 12, count=count, halo=halo, seed=17, epoch=3, occurrence=5
    )
    owner = torch.zeros(8, 12, dtype=torch.int32)
    for region in plan.regions:
        owner[
            region.core.top : region.core.bottom,
            region.core.left : region.core.right,
        ] += 1
        positions = full_canvas_position_ids(8, 12, region.input)
        assert positions.shape == (region.input.tokens, 2)
        assert region.input.tokens % 4 == 0
    assert torch.all(owner == 1)


def _tiny_partition_loss(model, clean, noise, sigma, text, text_mask, plan, *, backward_each):
    noisy = (1 - sigma[:, None, None]) * clean + sigma[:, None, None] * noise
    target = noise - clean
    noisy_grid = noisy.reshape(1, plan.full_height, plan.full_width, clean.shape[-1])
    target_grid = target.reshape_as(noisy_grid)
    total = torch.zeros((), dtype=torch.float32)
    for region in plan.regions:
        tile = flatten_region(noisy_grid, region.input)
        tile_target = flatten_region(target_grid, region.input)
        tokens = region.input.tokens
        image_mask = torch.cat(
            [
                torch.zeros(1, text.shape[1], dtype=torch.bool),
                torch.ones(1, tokens // 4, dtype=torch.bool),
            ],
            dim=1,
        )
        prediction = model(
            hidden_states=tile,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=text_mask,
            timestep=sigma,
            img_shapes=[[(1, region.input.height, region.input.width)]],
            img_mask=image_mask,
            target_spatial_position_ids=full_canvas_position_ids(
                plan.full_height, plan.full_width, region.input
            ),
            return_dict=False,
        )[0][:, -tokens:]
        local = region.core_in_input
        prediction_grid = prediction.reshape(
            1, region.input.height, region.input.width, clean.shape[-1]
        )
        target_tile_grid = tile_target.reshape_as(prediction_grid)
        core_prediction = flatten_region(prediction_grid, local)
        core_target = flatten_region(target_tile_grid, local)
        weighted = (
            region.core.tokens / plan.full_tokens
        ) * torch.nn.functional.mse_loss(core_prediction.float(), core_target.float())
        if backward_each:
            weighted.backward()
            total = total + weighted.detach()
        else:
            total = total + weighted
    if not backward_each:
        total.backward()
    return total.detach()


def test_qwen_sequential_partition_backward_matches_summed_partition_graph():
    torch.manual_seed(41)
    first = QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=32,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    second = QwenImage21Transformer2DModel.from_config(first.config)
    second.load_state_dict(first.state_dict())
    clean = torch.randn(1, 16, 8)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.35])
    text = torch.randn(1, 3, 32)
    text_mask = torch.ones(1, 3, dtype=torch.long)
    plan = build_fixed_partition_plan(4, 4, count=2, seed=9, epoch=2)

    sequential_loss = _tiny_partition_loss(
        first, clean, noise, sigma, text, text_mask, plan, backward_each=True
    )
    summed_loss = _tiny_partition_loss(
        second, clean, noise, sigma, text, text_mask, plan, backward_each=False
    )
    torch.testing.assert_close(sequential_loss, summed_loss, atol=1e-6, rtol=1e-6)
    for left, right in zip(first.parameters(), second.parameters()):
        if left.grad is None or right.grad is None:
            assert left.grad is right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)


def test_qwen_explicit_full_canvas_positions_preserve_full_forward():
    torch.manual_seed(43)
    model = QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=32,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    ).eval()
    hidden = torch.randn(1, 16, 8)
    text = torch.randn(1, 3, 32)
    text_mask = torch.ones(1, 3, dtype=torch.long)
    image_mask = torch.cat(
        [torch.zeros(1, 3, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)], dim=1
    )
    kwargs = dict(
        hidden_states=hidden,
        timestep=torch.tensor([0.4]),
        encoder_hidden_states=text,
        encoder_hidden_states_mask=text_mask,
        img_shapes=[[(1, 4, 4)]],
        img_mask=image_mask,
        return_dict=False,
    )
    implicit = model(**kwargs)[0]
    explicit = model(
        **kwargs,
        target_spatial_position_ids=full_canvas_position_ids(
            4, 4, PartitionBox(0, 0, 4, 4)
        ),
    )[0]
    torch.testing.assert_close(implicit, explicit, atol=0, rtol=0)


def test_qwen_full_kv_query_chunk_preserves_prediction_and_gradients():
    torch.manual_seed(47)
    reference = QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    chunked = QwenImage21Transformer2DModel.from_config(reference.config)
    chunked.load_state_dict(reference.state_dict())
    chunked.transformer_blocks[0].attn.processor._target_query_chunk_tokens = 5
    hidden = torch.randn(1, 16, 8)
    text = torch.randn(1, 3, 32)
    kwargs = dict(
        hidden_states=hidden,
        timestep=torch.tensor([0.4]),
        encoder_hidden_states=text,
        encoder_hidden_states_mask=torch.ones(1, 3, dtype=torch.long),
        img_shapes=[[(1, 4, 4)]],
        img_mask=torch.cat(
            [torch.zeros(1, 3, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)], dim=1
        ),
        return_dict=False,
    )
    expected = reference(**kwargs)[0]
    actual = chunked(**kwargs)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    for left, right in zip(reference.parameters(), chunked.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=3e-6, rtol=3e-5)


def test_qwen_partition_global_adapter_is_zero_init_and_uses_remote_content():
    torch.manual_seed(53)
    adapter = QwenPartitionGlobalAdapter(8, 32, rank=4, summary_tokens=3)
    grid = torch.randn(1, 4, 4, 8)
    box = PartitionBox(0, 0, 2, 4)
    initial = adapter(grid, box)
    torch.testing.assert_close(initial, torch.zeros_like(initial), atol=0, rtol=0)
    with torch.no_grad():
        adapter.out.weight.normal_()
    changed = grid.clone()
    changed[:, 2:] += 3
    assert not torch.allclose(adapter(grid, box), adapter(changed, box))
    adapter(grid, box).square().mean().backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())
    clone = QwenPartitionGlobalAdapter(8, 32, rank=4, summary_tokens=3)
    clone.load_tensors(adapter.export_tensors())
    torch.testing.assert_close(clone(grid, box), adapter(grid, box))


def test_qwen_partition_global_adapter_accepts_bf16_activations_with_fp32_masters():
    adapter = QwenPartitionGlobalAdapter(8, 32, rank=4, summary_tokens=3)
    grid = torch.randn(1, 4, 4, 8, dtype=torch.bfloat16)
    output = adapter(grid, PartitionBox(0, 0, 2, 4))
    assert output.dtype == torch.float32
    output.sum().backward()
    assert adapter.out.weight.grad is not None


def test_qwen_partition_checkpoint_auto_preserves_base_memory_policy():
    plan = build_fixed_partition_plan(64, 64, count=4, seed=2)
    trainer = SimpleNamespace(
        config={"qwen_partition_gradient_checkpointing_blocks": None},
        transformer=SimpleNamespace(transformer_blocks=[object()] * 32),
    )
    assert qwen_image_21_ops._resolve_partition_checkpoint_blocks(
        trainer, plan, 16
    ) == 16
    trainer.config["qwen_partition_gradient_checkpointing_blocks"] = 11
    assert qwen_image_21_ops._resolve_partition_checkpoint_blocks(
        trainer, plan, 16
    ) == 11


def test_qwen_checkpoint_controls_prefer_canonical_names_and_reject_conflicts():
    plan = build_fixed_partition_plan(64, 64, count=2, seed=3)
    trainer = SimpleNamespace(
        config={
            "dit_partition_gradient_checkpointing_blocks": 24,
            "qwen_partition_gradient_checkpointing_blocks": None,
        },
        transformer=SimpleNamespace(transformer_blocks=[object()] * 32),
    )
    assert qwen_image_21_ops._resolve_partition_checkpoint_blocks(
        trainer, plan, 16
    ) == 24
    trainer.config["qwen_partition_gradient_checkpointing_blocks"] = 12
    with pytest.raises(ValueError, match="conflicts"):
        qwen_image_21_ops._resolve_partition_checkpoint_blocks(trainer, plan, 16)


def test_qwen_checkpoint_alias_conflict_is_rejected_by_api():
    with pytest.raises(ValueError, match="conflicts"):
        routes.TrainingRunCreateRequest(
            training_method="lora",
            base_model_path="unused",
            dit_partition_gradient_checkpointing_blocks=24,
            qwen_partition_gradient_checkpointing_blocks=12,
        )


def test_qwen_declares_partial_checkpoint_and_partition_capabilities():
    from core.training.arch.qwen_image_21 import QwenImage21ArchHandler

    assert QwenImage21ArchHandler.dit_checkpoint_block_count == 32
    assert QwenImage21ArchHandler.supports_dit_partition_training


def test_dit_checkpoint_contract_refuses_unsupported_arch_and_depth(monkeypatch):
    from core.model_loader import ModelLoader
    from core.training.train_runner import _apply_dit_checkpoint_contract

    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda _path: "sdxl")
    with pytest.raises(ValueError, match="unsupported for sdxl"):
        _apply_dit_checkpoint_contract(
            "unused", {"dit_gradient_checkpointing_blocks": 1}
        )
    with pytest.raises(ValueError, match="partition training is unsupported for sdxl"):
        _apply_dit_checkpoint_contract(
            "unused", {"dit_partition_training_enabled": True}
        )

    monkeypatch.setattr(
        ModelLoader, "detect_model_type", lambda _path: "qwen_image_21"
    )
    with pytest.raises(ValueError, match="between 0 and 32"):
        _apply_dit_checkpoint_contract(
            "unused", {"dit_gradient_checkpointing_blocks": 33}
        )


def test_qwen_auto_uses_measured_bounded_convrot_backward_cache():
    trainer = SimpleNamespace(
        config={"qwen_partition_training_enabled": True},
        qwen_image_21_transformer_variant="int8_convrot",
        lora_rank=128,
        training_dtype=torch.bfloat16,
    )
    assert qwen_image_21_ops._resolve_convrot_training_forward(
        trainer, "auto"
    ) == ("prefetch_bf16", "measured bounded-cache policy")
    assert qwen_image_21_ops._resolve_convrot_training_forward(
        trainer, "prefetch_bf16"
    ) == ("prefetch_bf16", "explicit")

    trainer.config["qwen_partition_training_enabled"] = False
    assert qwen_image_21_ops._resolve_convrot_training_forward(
        trainer, "auto"
    ) == ("prefetch_bf16", "measured bounded-cache policy")


def test_qwen_transient_convrot_training_uses_same_artifact_contract():
    trainer = SimpleNamespace(
        qwen_convrot_training_forward="transient_bf16",
        learning_rate=1e-4,
        unet_lr=None,
    )
    adapter = QwenImage21LoRAAdapter(
        trainer, lora_rank=2, lora_alpha=2, lora_dtype=torch.float32
    )
    metadata = adapter.checkpoint_metadata({}, step=1, epoch=0)
    assert metadata["qwen_base_forward"] == "convrot_int8_bf16_backward_v1"


def test_qwen_prefetch_convrot_training_uses_same_artifact_contract():
    trainer = SimpleNamespace(
        qwen_convrot_training_forward="prefetch_bf16",
        learning_rate=1e-4,
        unet_lr=None,
        transformer=None,
    )
    adapter = QwenImage21LoRAAdapter(
        trainer, lora_rank=2, lora_alpha=2, lora_dtype=torch.float32
    )
    metadata = adapter.checkpoint_metadata({}, step=1, epoch=0)
    assert metadata["qwen_base_forward"] == "convrot_int8_bf16_backward_v1"


def test_qwen_prefetch_depth_must_be_smaller_than_cache_blocks():
    with pytest.raises(ValueError, match="smaller than"):
        routes.TrainingRunCreateRequest(
            training_method="lora",
            base_model_path="unused",
            qwen_convrot_training_forward="prefetch_bf16",
            qwen_convrot_backward_cache_blocks=2,
            qwen_convrot_backward_prefetch_depth=2,
        )


def test_qwen_partition_api_refuses_odd_halo():
    with pytest.raises(ValueError, match="dit_partition_halo_tokens must be even"):
        routes.TrainingRunCreateRequest(qwen_partition_halo_tokens=3)


def test_qwen_partition_legacy_request_populates_canonical_fields():
    request = routes.TrainingRunCreateRequest(
        training_method="lora",
        base_model_path="unused",
        qwen_partition_training_enabled=True,
        qwen_partition_fixed_count=4,
        qwen_partition_global_adapter_enabled=True,
    )
    assert request.dit_partition_training_enabled is True
    assert request.dit_partition_fixed_count == 4
    assert request.dit_partition_global_adapter_enabled is True


def test_qwen_partition_conflicting_aliases_are_rejected():
    with pytest.raises(ValueError, match="dit_partition_fixed_count conflicts"):
        routes.TrainingRunCreateRequest(
            training_method="lora",
            base_model_path="unused",
            dit_partition_fixed_count=2,
            qwen_partition_fixed_count=4,
        )


def test_qwen_partition_global_adapter_registers_and_exports():
    model = QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    trainer = SimpleNamespace(
        transformer=model,
        learning_rate=1e-4,
        unet_lr=None,
        config={
            "qwen_partition_training_enabled": True,
            "qwen_partition_global_adapter_enabled": True,
            "qwen_partition_global_rank": 4,
            "qwen_partition_global_tokens": 3,
        },
    )
    adapter = QwenImage21LoRAAdapter(
        trainer, lora_rank=2, lora_alpha=2, lora_dtype=torch.float32
    )
    layers = {}
    assert adapter.apply_lora_to_unet(layers) == 4
    assert "qwen_partition_global_adapter" in layers
    state = adapter.export_state_dict(layers)
    assert "qwen_partition_global_adapter.summary_queries" in state
    metadata = adapter.checkpoint_metadata(layers, step=1, epoch=0)
    assert metadata["qwen_partition_global_adapter"] == "latent_summary_v1"
    assert metadata["qwen_partition_global_rank"] == "4"


def test_handler_is_concrete_and_tiny_lora_inventory_is_complete():
    trainer = SimpleNamespace()
    handler = QwenImage21ArchHandler(trainer)
    model = QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=32,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    assert handler.name == MODEL_TYPE
    assert [path for _parent, _slot, path in iter_lora_slots(model)] == [
        "txt_in.in_layer",
        "txt_in.out_layer",
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
    ]


def test_qwen_txt_in_lora_is_opt_in_and_has_independent_lr():
    model = QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    trainer = SimpleNamespace(
        transformer=model, learning_rate=1e-4, unet_lr=1e-4,
        config={"train_adapter": True, "adapter_lr": 2e-5},
    )
    adapter = QwenImage21LoRAAdapter(
        trainer, lora_rank=2, lora_alpha=2, lora_dtype=torch.float32
    )
    layers = {}
    assert adapter.apply_lora_to_unet(layers) == 6
    assert "lora_unet_txt_in__in_layer" in layers
    assert "lora_unet_txt_in__out_layer" in layers
    groups = adapter.arch_param_groups(layers)
    assert [(group["component"], group["lr"]) for group in groups] == [
        ("unet", 1e-4), ("adapter", 2e-5)
    ]
    state = adapter.export_state_dict(layers)
    grouped = normalise_lora_state_dict(state)
    assert "txt_in.in_layer" in grouped and "txt_in.out_layer" in grouped

    adapter_only_model = QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    trainer.transformer = adapter_only_model
    trainer.train_unet = False
    adapter_only_layers = {}
    assert adapter.apply_lora_to_unet(adapter_only_layers) == 2
    assert [g["component"] for g in adapter.arch_param_groups(adapter_only_layers)] == ["adapter"]


def test_adapter_controls_reach_training_yaml_and_validate_lr():
    from core.training.training_config import _build_train_section

    request = routes.TrainingRunCreateRequest(
        training_method="lora", base_model_path="unused",
        train_adapter=True, adapter_lr=2e-5,
    )
    train = _build_train_section(
        request.model_dump(), total_steps=1, epochs=None,
        train_unet=True, train_text_encoder=False, train_image_encoder=False,
    )
    assert train["train_adapter"] is True
    assert train["adapter_lr"] == 2e-5
    with pytest.raises(ValueError):
        routes.TrainingRunCreateRequest(adapter_lr=-1)


def test_qwen_full_txt_in_lr_and_freeze_are_independent():
    model = QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    trainer = SimpleNamespace(
        transformer=model, text_encoder=torch.nn.Linear(1, 1),
        vae=torch.nn.Linear(1, 1), learning_rate=1e-4, unet_lr=1e-4,
        train_text_encoder=False,
        config={"train_adapter": True, "adapter_lr": 2e-5},
    )
    adapter = QwenImage21FullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    groups = adapter.arch_param_groups()
    assert [(group["component"], group["lr"]) for group in groups] == [
        ("unet", 1e-4), ("adapter", 2e-5)
    ]
    assert not ({id(p) for p in groups[0]["params"]}
                & {id(p) for p in groups[1]["params"]})
    trainer.config["train_adapter"] = False
    adapter.prepare_models_for_training()
    assert all(not p.requires_grad for p in model.txt_in.parameters())
    assert len(adapter.arch_param_groups()) == 1
    trainer.config.update(train_adapter=None, adapter_lr=None)
    adapter.prepare_models_for_training()
    assert all(p.requires_grad for p in model.txt_in.parameters())
    assert len(adapter.arch_param_groups()) == 1
    trainer.config.update(train_adapter=True, adapter_lr=2e-5)
    trainer.train_unet = False
    adapter.prepare_models_for_training()
    assert all(p.requires_grad for p in model.txt_in.parameters())
    assert all(not p.requires_grad for p in model.transformer_blocks.parameters())
    assert [g["component"] for g in adapter.arch_param_groups()] == ["adapter"]

    trainer.train_text_encoder = True
    trainer.text_encoder_lr = 3e-5
    adapter.prepare_models_for_training()
    groups = adapter.arch_param_groups()
    assert [g["component"] for g in groups] == ["adapter", "text_encoder_1"]
    assert groups[1]["lr"] == 3e-5
    assert all(p.requires_grad for p in trainer.text_encoder.parameters())


def test_qwen_trainable_prompt_encoding_preserves_text_encoder_gradient():
    encoder = torch.nn.Linear(4, 4)

    class Pipe:
        def encode_prompt(self, *, prompt, device):
            source = torch.ones(1, 2, 4, device=device)
            return encoder(source), None, None

    trainer = SimpleNamespace(text_encoder=encoder, qwen_image_21_pipeline=Pipe())
    embeds, mask = qwen_image_21_ops.encode_prompt(
        trainer, "new character", requires_grad=True
    )
    embeds.square().mean().backward()
    assert encoder.weight.grad is not None
    assert bool(torch.isfinite(encoder.weight.grad).all())
    assert bool(mask.all())


def test_qwen_full_te_bundle_round_trip(tmp_path, monkeypatch):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from core.models.qwen_image_21 import loader

    dit = QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    te_config = Qwen3VLConfig(
        text_config={
            "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 4, "head_dim": 8,
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 4, "out_hidden_size": 32,
            "num_position_embeddings": 16, "deepstack_visual_indexes": [],
        },
    )
    te = Qwen3VLForConditionalGeneration(te_config)
    companion = tmp_path / "companion"
    companion.mkdir()
    (companion / "manifest.json").write_text("{}", encoding="utf-8")
    config_path = companion / "text_encoder_config.json"
    te_config.to_json_file(str(config_path))
    trainer = SimpleNamespace(
        transformer=dit, text_encoder=te, model_path=str(companion),
        qwen_image_21_companion_path=str(companion), train_text_encoder=True,
    )
    checkpoint = QwenImage21FullParameterAdapter(trainer).write_checkpoint(
        1, 0, tmp_path / "run_step_000001"
    )
    monkeypatch.setattr(loader, "_artifact_components", lambda *_args, **_kwargs: {
        "transformer": None, "text_encoder": None,
        "vae": torch.nn.Linear(1, 1), "processor": object(),
        "scheduler": object(),
        "manifest": SimpleNamespace(text_encoder_config=str(config_path)),
    })
    loaded = loader.load_qwen_image_21_components(
        str(checkpoint), torch_dtype=torch.float32
    )
    assert loaded["text_encoder_variant"] == "bf16"
    assert torch.equal(
        loaded["text_encoder"].model.language_model.embed_tokens.weight,
        te.model.language_model.embed_tokens.weight,
    )
    assert torch.equal(
        loaded["transformer"].txt_in.in_layer.weight,
        dit.txt_in.in_layer.weight,
    )


def test_convrot_materialization_releases_packed_weight_and_preserves_gradient():
    from comfy_kitchen.backends.eager.quantization import quantize_int8_convrot_weight
    from core.models.common.convrot_int8_linear import (
        ConvRotInt8Linear, materialize_convrot_linears,
    )

    reference = torch.randn(32, 256, dtype=torch.bfloat16)
    packed, scale = quantize_int8_convrot_weight(
        reference, 256, stochastic_rounding=0
    )
    layer = ConvRotInt8Linear(
        256, 32, False, torch.bfloat16,
        convrot_groupsize=256, marker_numel=1, device="cpu",
    )
    layer.weight.copy_(packed)
    layer.weight_scale.copy_(scale.reshape(-1))
    module = torch.nn.Sequential(layer)
    assert materialize_convrot_linears(module, torch.bfloat16) == 1
    assert isinstance(module[0], torch.nn.Linear)
    assert not hasattr(module[0], "weight_scale")
    x = torch.randn(2, 256, dtype=torch.bfloat16, requires_grad=True)
    module(x).float().square().mean().backward()
    assert module[0].weight.grad is not None
    assert x.grad is not None


def test_lora_save_classify_and_rebuild_round_trip():
    model = QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=32,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    trainer = SimpleNamespace(transformer=model, learning_rate=1e-4, unet_lr=None)
    adapter = QwenImage21LoRAAdapter(trainer, lora_rank=2, lora_alpha=4, lora_dtype=torch.float32)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) == 4
    first_stem, trained_branch = next(iter(layers.items()))
    with torch.no_grad():
        trained_branch.lora_down.weight.fill_(0.25)
        trained_branch.lora_up.weight.fill_(0.5)

    state = adapter.export_state_dict(layers)
    metadata = adapter.checkpoint_metadata(layers, step=3, epoch=1)
    assert classify_lora_keys(state, metadata)["arch"] == MODEL_TYPE
    grouped = normalise_lora_state_dict(state)
    module_path = first_stem.removeprefix("lora_unet_").replace("__", ".")
    fresh_base = torch.nn.Linear(
        trained_branch.original_module.in_features,
        trained_branch.original_module.out_features,
        bias=False,
    )
    rebuilt = build_lora_branch(fresh_base, grouped[module_path], module_path)
    x = torch.randn(2, fresh_base.in_features)
    torch.testing.assert_close(
        rebuilt.reference_delta(x), trained_branch.reference_delta(x), rtol=0, atol=0
    )


def test_convrot_training_metadata_and_generation_base_gate():
    trainer = SimpleNamespace(
        qwen_convrot_training_forward="cached_bf16",
        learning_rate=1e-4,
        unet_lr=None,
    )
    adapter = QwenImage21LoRAAdapter(
        trainer, lora_rank=2, lora_alpha=2, lora_dtype=torch.float32
    )
    metadata = adapter.checkpoint_metadata({}, step=1, epoch=0)
    assert metadata["qwen_base_variant"] == "int8_convrot"
    assert metadata["qwen_base_forward"] == "convrot_int8_bf16_backward_v1"

    file = SimpleNamespace(name="trained.safetensors", metadata=metadata, tensors={})
    compatible = QwenImage21Mixin()
    compatible.qwen_image_21_components = {"transformer_variant": "int8_convrot"}
    assert compatible._qwen21_prepare_lora_file(file) == {}

    incompatible = QwenImage21Mixin()
    incompatible.qwen_image_21_components = {"transformer_variant": "bf16"}
    with pytest.raises(ValueError, match="requires an int8_convrot model"):
        incompatible._qwen21_prepare_lora_file(file)


def test_api_defaults_and_capabilities_expose_qwen_controls():
    defaults = asyncio.run(get_generation_defaults())
    assert defaults["image_arch_overlays"][MODEL_TYPE] == {
        "steps": 40,
        "cfg_scale": 1.0,
        "qwen_image_21_kv_cache": True,
    }
    assert defaults["txt2img"]["qwen_image_21_kv_cache"] is True
    capabilities = asyncio.run(get_arch_capabilities())
    unsupported = capabilities["unsupported"][MODEL_TYPE]
    assert "controlnets" in unsupported
    assert "style_transfer" not in unsupported
    assert "qwen_image_21_kv_cache" not in unsupported

    from api.arch_capabilities import check_arch_capabilities
    assert check_arch_capabilities({
        "controlnets": [{"is_reference_guide": True}]
    }, MODEL_TYPE) == []
    assert check_arch_capabilities({
        "controlnets": [{"is_style_transfer": True}]
    }, MODEL_TYPE) == []
    warnings = check_arch_capabilities({
        "controlnets": [{"model_path": "controlnet.safetensors"}]
    }, MODEL_TYPE)
    assert any("ControlNet" in warning["message"] for warning in warnings)


def test_generation_callback_uses_shared_progress_contract():
    latents = torch.randn(1, 4, 8)
    pred_x0 = torch.randn(1, 4, 8)
    progress_calls = []
    step_calls = []

    class _Pipe:
        _interrupt = False

        def __call__(self, **kwargs):
            assert kwargs["callback_on_step_end_tensor_inputs"] == [
                "latents", "pred_original_sample"]
            assert callable(kwargs["before_step_callback"])
            callback_kwargs = {
                "latents": latents, "pred_original_sample": pred_x0}
            returned = kwargs["callback_on_step_end"](
                self, 0, torch.tensor(1.0), callback_kwargs
            )
            assert returned is callback_kwargs
            return SimpleNamespace(images=[object()])

    class _Harness(QwenImage21Mixin):
        cancel_requested = False

        def _qwen_image_21_pipe(self):
            return _Pipe()

        def _load_lora_qwen21(self, _configs):
            return 0

        def _unload_lora_qwen21(self):
            return 0

    image, seed, ancestral_seed = _Harness()._qwen_image_21_run(
        {"prompt": "test", "steps": 2, "seed": 7},
        progress_callback=lambda step, total, current, metrics, predicted: progress_calls.append(
            (step, total, current, metrics, predicted)
        ),
        step_callback=lambda step, timestep, current: step_calls.append(
            (step, timestep, current)
        ),
    )

    assert image is not None
    assert seed == ancestral_seed == 7
    assert len(progress_calls) == 1
    assert progress_calls[0][:2] == (0, 2)
    assert progress_calls[0][2] is latents
    assert progress_calls[0][4] is pred_x0
    assert step_calls[0][0] == 0
    assert step_calls[0][2] is latents


def test_qwen_reference_controls_reach_pipeline_and_real_controlnet_refuses():
    reference_image = Image.new("RGB", (32, 32), "red")
    captured = {}

    class _Pipe:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[reference_image])

    class _Session:
        def set_step(self, _step, _total):
            pass

    class _Harness(QwenImage21Mixin):
        cancel_requested = False
        _qwen21_lora_session_instance = _Session()

        def _qwen_image_21_pipe(self):
            return _Pipe()

        def _load_lora_qwen21(self, _configs):
            return 0

        def _unload_lora_qwen21(self):
            return 0

    reference_guide = {
        "image": reference_image, "strength": 0.4,
        "start_step": 100, "end_step": 800,
        "is_reference_guide": True,
    }
    style = {"image": reference_image, "ref_k_strength": 0.7}
    params = {
        "prompt": "test", "steps": 2, "seed": 8,
        "controlnets": [{"is_reference_guide": True}, {"is_style_transfer": True}],
        "controlnet_images": [reference_guide],
        "style_transfers": [style],
        "style_combine_mode": "common_concept",
    }
    _Harness()._qwen_image_21_run(params)
    assert captured["reference_guides"] == [reference_guide]
    assert captured["style_transfers"] == [style]
    assert captured["style_combine_mode"] == "common_concept"

    with pytest.raises(Exception, match="ControlNet is not available"):
        _Harness()._qwen_image_21_run({
            "prompt": "test", "steps": 1, "seed": 9,
            "controlnets": [{"model_path": "sdxl-controlnet.safetensors"}],
        })


def test_qwen_controlnet_route_preflight_distinguishes_native_entries(monkeypatch):
    monkeypatch.setattr(routes.pipeline_manager, "is_qwen_image_21_model", True)
    routes._reject_if_qwen21_controlnet(json.dumps([
        {"is_reference_guide": True}, {"is_style_transfer": True},
    ]))
    with pytest.raises(Exception, match="ControlNet is not available"):
        routes._reject_if_qwen21_controlnet(json.dumps([
            {"model_path": "sdxl-controlnet.safetensors"},
        ]))


def test_qwen_segmented_attention_captures_and_injects_style_kv():
    from core.inference.reference_style import StyleContext, StyleTransferConfig
    from core.models.qwen_image_21.vendor.transformer import QwenImage21Attention

    torch.manual_seed(123)
    attention = QwenImage21Attention(dim=16, heads=2, dim_head=8)
    attention.block_idx = 0
    hidden = torch.randn(1, 6, 16)
    segments = [(0, 2, True)]
    config = StyleTransferConfig(
        ref_k_strength=0.7, adain_strength=0.0,
        axes_dims=(2, 2, 4), block_range=(0, 0),
    )

    capture = StyleContext(mode="capture", config=config)
    capture.img_start, capture.img_end = 2, 6
    attention._style_ctx = capture
    attention(hidden, segments=segments)
    assert capture.store[0][1].shape[1] == 4

    attention._style_ctx = None
    baseline = attention(hidden, segments=segments)
    inject = StyleContext(mode="inject", config=config, store=capture.store)
    inject.img_start, inject.img_end = 2, 6
    attention._style_ctx = inject
    styled = attention(hidden, segments=segments)
    attention._style_ctx = None
    assert styled.shape == baseline.shape
    assert not torch.equal(styled, baseline)


def test_qwen_live_preview_routes_to_64_channel_projection():
    from api.generation_utils import preview_arch_kwargs
    from core.utils.taesd import TAESDManager

    manager = SimpleNamespace(
        current_model_info={"type": "qwen_image_21", "latent_channels": 64},
        minit2i_components=None,
    )
    kwargs = preview_arch_kwargs(manager, SimpleNamespace())
    assert kwargs["is_qwen_image_21"] is True
    assert kwargs["preview_predicted_x0"] is True

    preview = TAESDManager().decode_latent(
        torch.zeros(1, 8 * 6, 64),
        is_qwen_image_21=True,
        image_width=128,
        image_height=96,
    )
    assert preview is not None
    assert preview.size == (128, 96)


def test_static_openapi_exposes_qwen_model_and_kv_cache():
    import yaml

    spec = yaml.safe_load((BACKEND.parent / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["GenerationParams"]["properties"]
    assert props["qwen_image_21_kv_cache"]["default"] is True
    assert MODEL_TYPE in str(spec)


def test_full_checkpoint_records_companion_and_loader_replaces_transformer(tmp_path, monkeypatch):
    from safetensors import safe_open
    import core.models.qwen_image_21.loader as loader

    model = QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=32,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    companion = tmp_path / "companion"
    companion.mkdir()
    (companion / "manifest.json").write_text("{}", encoding="utf-8")
    trainer = SimpleNamespace(
        transformer=model,
        model_path=str(companion),
        qwen_image_21_companion_path=str(companion),
    )
    checkpoint = QwenImage21FullParameterAdapter(trainer).write_checkpoint(
        7, 2, tmp_path / "trained"
    )
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert metadata["companion_path"] == str(companion)
    assert metadata["component"] == "transformer"

    base_transformer = torch.nn.Linear(1, 1)
    trained_transformer = torch.nn.Linear(2, 2)
    monkeypatch.setattr(loader, "_artifact_components", lambda *_args, **_kwargs: {
        "transformer": base_transformer,
        "text_encoder": torch.nn.Linear(1, 1),
        "vae": torch.nn.Linear(1, 1),
        "processor": object(),
        "scheduler": object(),
        "transformer_variant": "bf16",
        "text_encoder_variant": "bf16",
    })
    monkeypatch.setattr(
        loader, "load_transformer_artifact", lambda *_args, **_kwargs: (trained_transformer, "bf16")
    )
    loaded = loader.load_qwen_image_21_components(str(checkpoint))
    assert loaded["transformer"] is trained_transformer
    assert loaded["checkpoint_path"] == str(checkpoint)
    assert loaded["companion_path"] == str(companion.resolve())

    trainer.qwen_image_21_companion_path = str(checkpoint)
    resumed_checkpoint = QwenImage21FullParameterAdapter(trainer).write_checkpoint(
        8, 2, tmp_path / "resumed"
    )
    resumed = loader.load_qwen_image_21_components(str(resumed_checkpoint))
    assert resumed["checkpoint_path"] == str(resumed_checkpoint)
    assert resumed["companion_path"] == str(companion.resolve())
