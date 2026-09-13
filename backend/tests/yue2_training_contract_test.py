from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from core.adapters import LoRALinearLayer
from core.models.common.convrot_int8_linear import ConvRotInt8Linear
from core.models.yue2.vendor.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from core.models.yue2.vendor.protocol import ABC_END, ABC_START
from core.models.yue2.artifacts import load_yue2_training_artifacts, write_yue2_sidecar
from core.models.yue2.yue2_lora import iter_yue2_lora_targets, normalize_yue2_stages, stage_is_active
from core.training.adapters.yue2_adapter import YuE2FullParameterAdapter, YuE2LoRAAdapter
from core.training.ops.yue2_ops import (build_abc_ar_example, collate_abc_ar,
                                       prepare_abc_items, train_abc_ar_loop,
                                       train_abc_ar_step)
from core.training.arch import ARCH_REGISTRY, resolve_arch_name
from core.training.lora_trainer import LoRATrainer
from core.training.base_trainer import BaseTrainer
from core.training.train_runner import _apply_yue2_training_contract
from core.pipeline_backends.yue2 import YuE2Mixin
from core.adapters import CompositeAdapterLayer
from core.extensions.lora_manager import classify_lora_keys


class TinyTokenizer:
    def encode(self, text):
        return [10 + (ord(char) % 97) for char in text]


def tiny_model(layers=2):
    config = YuE2Config(vocab_size=184704, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=layers, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                        max_latent_frames=64, latent_dim=8)
    return YuE2ForCausalLM(config)


def test_abc_example_masks_prefix_and_padding():
    first = build_abc_ar_example(TinyTokenizer(), style="pop", lyrics="[Verse]\nhello", abc="X:1\nCDEF")
    second = build_abc_ar_example(TinyTokenizer(), style="jazz", lyrics="[Verse]\nhi", abc="X:1\nC")
    start = first.input_ids.tolist().index(ABC_START)
    assert torch.all(first.labels[:start + 1] == -100)
    assert first.labels[-1].item() == ABC_END
    batch = collate_abc_ar([first, second])
    assert torch.all(batch["labels"][1, second.input_ids.numel():] == -100)
    assert torch.all(batch["attention_mask"][1, second.input_ids.numel():] == 0)


def test_abc_example_refuses_silent_truncation():
    with pytest.raises(ValueError, match="leaves no room|exceeds context"):
        build_abc_ar_example(TinyTokenizer(), style="pop", lyrics="words", abc="A" * 50, context=90)


def test_target_counts_and_half_separation():
    model = tiny_model(layers=2)
    ar = list(iter_yue2_lora_targets(model, half="ar", scope={"attention": True, "mlp": True}))
    nar = list(iter_yue2_lora_targets(model, half="nar", scope={"attention": True, "mlp": True}))
    assert len(ar) == len(nar) == 14
    assert all("nar_" not in target.path for target in ar)
    assert all("nar_" in target.path for target in nar)


def test_production_layer_count_contract():
    model = tiny_model(layers=28)
    attention = list(iter_yue2_lora_targets(model, half="ar"))
    with_mlp = list(iter_yue2_lora_targets(
        model, half="ar", scope={"attention": True, "mlp": True}
    ))
    assert len(attention) == 112
    assert len(with_mlp) - len(attention) == 84


def test_stage_contract():
    assert normalize_yue2_stages("abc") == ("abc",)
    assert stage_is_active(("ar",), "abc")
    assert stage_is_active(("ar",), "semantic")
    assert not stage_is_active(("abc",), "semantic")
    with pytest.raises(ValueError):
        normalize_yue2_stages("ar,abc")


def test_lora_step_and_gradient_checkpointing_are_differentiable():
    model = tiny_model(layers=2)
    trainer = SimpleNamespace(transformer=model, learning_rate=1e-4, unet_lr=None, arch=None,
                              adapter_algorithm="lora", weight_decompose=False, adapter_config={})
    adapter = YuE2LoRAAdapter(trainer, 4, 4, objective="abc_ar")
    layers = {}
    assert adapter.apply_lora_to_unet(layers) == 8
    model.gradient_checkpointing_enable()
    model.train()
    example = build_abc_ar_example(TinyTokenizer(), style="pop", lyrics="[Verse]\nhello", abc="X:1\nC")
    loss = train_abc_ar_step(model, collate_abc_ar([example]))
    loss.backward()
    grads = [parameter.grad for layer in layers.values() for parameter in layer.parameters()
             if parameter.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)
    metadata = adapter.checkpoint_metadata(layers, 3, 1)
    assert metadata["yue2_apply_stages"] == "abc"
    assert metadata["yue2_lora_half"] == "ar"
    assert metadata["yue2_training_protocol"] == "yue2-abc-ar-v1"
    assert metadata["modelspec.license"] == "CC-BY-NC-4.0"
    assert metadata["modelspec.source"].endswith("/m-a-p/YuE2-3B")


def test_generated_sidecar_is_a_hash_checked_training_input(tmp_path):
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"lossless-audio")
    result = SimpleNamespace(
        abc_ids=[7, 8], semantic_tokens=[1, 2, 3],
        latents=torch.zeros(3, 64), seed=1, sample_rate=48000,
        abc_text="X:1\nC", truncated={"abc": False, "semantic": False},
        effective_config={}, timings={}, model_identity={"checkpoint": "base.safetensors"},
    )
    sidecar = Path(write_yue2_sidecar(audio, result))
    loaded = load_yue2_training_artifacts(sidecar, require_abc=True,
                                          require_semantic=True, require_latents=True)
    assert loaded.latents.shape == (3, 64)
    arrays = audio.with_suffix(".yue2.npz")
    arrays.write_bytes(arrays.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_yue2_training_artifacts(sidecar)


def test_convrot_frozen_base_lora_backward_and_checkpoint(tmp_path):
    base = ConvRotInt8Linear(256, 256, False, torch.float32,
                             convrot_groupsize=256, marker_numel=1)
    base.weight.copy_(torch.randint(-8, 8, base.weight.shape, dtype=torch.int8))
    base.weight_scale.fill_(0.01)
    base.comfy_quant.zero_()
    branch = LoRALinearLayer(base, 4, 4, "lora_unet_probe")
    inputs = torch.randn(2, 256, requires_grad=True)
    branch(inputs).square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in branch.parameters() if parameter.requires_grad)

    trainer = SimpleNamespace(transformer=None, learning_rate=1e-4, unet_lr=None, arch=None,
                              adapter_algorithm="lora", weight_decompose=False, adapter_config={})
    adapter = YuE2LoRAAdapter(trainer, 4, 4, objective="abc_ar")
    path = tmp_path / "planner.safetensors"
    adapter.save_checkpoint({"lora_unet_probe": branch}, 2, 1, path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        assert handle.metadata()["yue2_apply_stages"] == "abc"
        assert set(handle.keys()) == {
            "lora_unet_probe.alpha", "lora_unet_probe.lora_down.weight",
            "lora_unet_probe.lora_up.weight",
        }


def test_yue2_is_a_registered_training_architecture():
    assert "yue2" in ARCH_REGISTRY
    assert resolve_arch_name(SimpleNamespace(is_yue2=True)) == "yue2"


def test_yue2_component_lr_fallback_matches_its_single_adapter_group():
    trainer = SimpleNamespace(
        is_yue2=True, is_sensenova=False, train_unet=True,
        unet=None, controlnet=None, train_text_encoder=False,
        _train_vision_encoder=False, learning_rate=1e-4, unet_lr=2e-5,
    )
    assert BaseTrainer._build_component_lr_list(trainer) == ([2e-5], ["YuE2-AR"])


def test_phase_a_preflight_refuses_inapplicable_training_controls(monkeypatch):
    from core.model_loader import ModelLoader
    monkeypatch.setattr(ModelLoader, "detect_model_type", staticmethod(lambda _path: "yue2"))
    accepted = {"train_unet": True, "train_text_encoder": False,
                "yue2_lora_scope": "attention,mlp", "yue2_abc_mode": "melody"}
    assert _apply_yue2_training_contract("model.safetensors", "lora", accepted)
    for key, value, message in (
        ("blocks_to_swap", 1, "block swap"),
        ("use_ema", True, "EMA"),
        ("repa_enable", True, "REPA"),
        ("yue2_training_objective", "semantic_ar", "abc_ar"),
    ):
        config = {"train_unet": True, "train_text_encoder": False, key: value}
        with pytest.raises(ValueError, match=message):
            _apply_yue2_training_contract("model.safetensors", "lora", config)


def test_full_finetune_contract_requires_dense_bf16_and_safe_settings(monkeypatch):
    from core.model_loader import ModelLoader
    from core.models.yue2 import loader
    from api.param_defaults import full_finetune_forces_stochastic_rounding

    monkeypatch.setattr(ModelLoader, "detect_model_type", staticmethod(lambda _path: "yue2"))
    monkeypatch.setattr(loader, "preflight_yue2", lambda _path: {"quantized": {}})
    accepted = {
        "train_unet": True,
        "train_text_encoder": False,
        "gradient_checkpointing": True,
        "batch_size": 1,
        "optimizer": "adamw8bit_ringbuffer",
        "optimizer_stochastic_rounding": None,
    }
    assert _apply_yue2_training_contract("dense.safetensors", "full_finetune", accepted)
    assert accepted["optimizer_stochastic_rounding"] is True
    assert full_finetune_forces_stochastic_rounding("yue2") is True

    monkeypatch.setattr(loader, "preflight_yue2", lambda _path: {"quantized": {"layer": {}}})
    with pytest.raises(ValueError, match="ConvRot INT8"):
        _apply_yue2_training_contract("int8.safetensors", "full_finetune", dict(accepted))


def test_full_parameter_adapter_trains_only_ar_and_writes_complete_file(tmp_path):
    from core.models.yue2.vendor.modeling_yue2 import RMSNorm

    model = tiny_model(layers=2)
    model.nar_norm = RMSNorm(model.config.hidden_size, model.config.rms_norm_eps)
    trainer = SimpleNamespace(
        transformer=model,
        gradient_checkpointing=True,
        optimizer_stochastic_rounding=True,
        learning_rate=1e-5,
        unet_lr=None,
        yue2_frozen_vae=torch.nn.Linear(2, 2),
        tokenizer=SimpleNamespace(payload='{"version":"1.0"}'),
        yue2_model_identity={"checkpoint": "dense.safetensors"},
    )
    adapter = YuE2FullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    groups = adapter.arch_param_groups()
    selected = {id(parameter) for parameter in groups[0]["params"]}
    assert selected
    assert id(model.model.layers[0].self_attn.q_proj.weight) in selected
    assert id(model.model.layers[0].nar_self_attn.q_proj.weight) not in selected
    assert not model.model.layers[0].nar_self_attn.q_proj.weight.requires_grad
    example = build_abc_ar_example(
        TinyTokenizer(), style="pop", lyrics="[Verse]\nhello", abc="X:1\nC"
    )
    train_abc_ar_step(model, collate_abc_ar([example])).backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.model.layers[0].nar_self_attn.q_proj.weight.grad is None
    path = adapter.write_checkpoint(3, 1, tmp_path / "planner")
    with safe_open(path, framework="pt", device="cpu") as handle:
        assert handle.metadata()["yue2_training_scope"] == "abc_ar_full"
        assert handle.metadata()["step"] == "3"
        assert handle.metadata()["yue2_weight_storage"] == "dense_bf16"
        assert "vae.weight" in handle.keys()


def test_production_full_scope_parameter_count_is_pinned():
    from core.models.yue2.loader import build_empty_models
    from core.models.yue2.pipeline import ar_modules

    model, _ = build_empty_models()
    seen = {}
    for module in ar_modules(model):
        for parameter in module.parameters():
            seen[id(parameter)] = parameter
    assert sum(parameter.numel() for parameter in seen.values()) == 2_165_957_632


def test_full_resume_uses_weights_already_loaded_as_base(tmp_path):
    from safetensors.torch import save_file

    checkpoint = tmp_path / "run_step_000003.safetensors"
    save_file({"probe": torch.zeros(1)}, checkpoint, metadata={
        "yue2_training_scope": "abc_ar_full",
        "yue2_training_protocol": "yue2-abc-ar-v1",
    })
    audio = tmp_path / "example.flac"
    audio.write_bytes(b"unused")
    audio.with_suffix(".abc").write_text("X:1\nK:C\nCDEF", encoding="utf-8")
    datasets = [SimpleNamespace(items=[{
        "image_path": str(audio), "audio_path": str(audio), "caption": "pop",
        "lyrics": "[Verse]\nhello", "width": None, "height": None,
    }])]
    model = torch.nn.Linear(1, 1)
    trainer = SimpleNamespace(
        trains_base_weights=True, _loaded_checkpoint_path=str(checkpoint),
        config={"yue2_training_objective": "abc_ar"}, tokenizer=TinyTokenizer(),
        transformer=model, device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, run_id=None, _metrics_buffer=[], output_dir=tmp_path,
        run_name="run", scaler=None, log_prefix="[test]",
    )
    trainer.setup_optimizer = lambda *_args: (
        setattr(trainer, "optimizer", torch.optim.SGD(model.parameters(), lr=0.1)),
        setattr(trainer, "lr_scheduler", torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lambda _step: 1.0)),
    )
    trainer._check_stop_requested = lambda: None
    trainer.load_checkpoint = lambda _path: (_ for _ in ()).throw(
        AssertionError("full resume must not call FullParameterTrainer.load_checkpoint")
    )
    trainer.load_training_state = lambda _step: None
    trainer._fast_forward_lr_schedulers = lambda _step: None
    trainer.load_optimizer_state = lambda _step: True
    trainer._reassert_config_lr_on_resume = lambda: None
    assert train_abc_ar_loop(
        trainer, datasets=datasets, total_steps=3, optimizer_type="adamw8bit",
        resume_from_checkpoint="latest",
    ) is False


def test_training_api_and_openapi_share_yue2_defaults():
    import yaml
    from api.param_defaults import TRAINING_DEFAULTS
    from api.routes import TrainingRunCreateRequest

    keys = ("yue2_training_objective", "yue2_lora_scope", "yue2_abc_mode",
            "yue2_allow_truncated_targets")
    request = TrainingRunCreateRequest(training_method="lora", base_model_path="x")
    spec = yaml.safe_load((Path(__file__).parents[2] / "openapi.yaml").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"]
    for key in keys:
        assert getattr(request, key) == TRAINING_DEFAULTS[key]
        assert props[key]["default"] == TRAINING_DEFAULTS[key]


def test_resume_refuses_a_different_yue2_objective(tmp_path):
    trainer = SimpleNamespace(transformer=None, learning_rate=1e-4, unet_lr=None, arch=None,
                              adapter_algorithm="lora", weight_decompose=False, adapter_config={})
    semantic = YuE2LoRAAdapter(trainer, 4, 4, objective="semantic_ar")
    base = torch.nn.Linear(8, 8, bias=False)
    branch = LoRALinearLayer(base, 4, 4, "lora_unet_probe")
    path = tmp_path / "semantic.safetensors"
    semantic.save_checkpoint({"lora_unet_probe": branch}, 2, 0, path)

    resume = LoRATrainer.__new__(LoRATrainer)
    resume.is_yue2 = True
    resume.adapter = YuE2LoRAAdapter(trainer, 4, 4, objective="abc_ar")
    resume.lora_rank = 4
    resume.lora_alpha = 4
    resume.lora_layers = {"lora_unet_probe": branch}
    resume.log_prefix = "[test]"
    with pytest.raises(ValueError, match="yue2_objective"):
        resume.load_checkpoint(str(path))


def test_lora_listing_classifies_yue2_before_sd_catchall():
    result = classify_lora_keys([
        "lora_unet_model_layers_3_self_attn_q_proj.lora_down.weight",
        "lora_unet_model_layers_3_self_attn_q_proj.lora_up.weight",
    ])
    assert result == {"arch": "yue2", "blocks": ["AR03"]}


def test_token_native_loop_skips_image_and_vae_paths(tmp_path):
    audio = tmp_path / "example.flac"
    audio.write_bytes(b"unused")
    audio.with_suffix(".abc").write_text("X:1\nK:C\nCDEF", encoding="utf-8")
    datasets = [SimpleNamespace(items=[{
        "image_path": str(audio), "audio_path": str(audio), "caption": "pop",
        "lyrics": "[Verse]\nhello", "width": None, "height": None,
    }])]

    class ToyAR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, *, labels, **kwargs):
            assert (labels != -100).any()
            return SimpleNamespace(loss=self.weight.square())

    model = ToyAR()
    saved = []
    trainer = SimpleNamespace(
        config={"yue2_training_objective": "abc_ar"}, tokenizer=TinyTokenizer(),
        transformer=model, device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, run_id=None, _metrics_buffer=[], output_dir=tmp_path,
        run_name="test", scaler=None,
    )
    trainer.setup_optimizer = lambda _kind, _schedule, _steps: (
        setattr(trainer, "optimizer", torch.optim.SGD(model.parameters(), lr=0.1)),
        setattr(trainer, "lr_scheduler", torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lambda _step: 1.0)),
    )
    trainer._check_stop_requested = lambda: None
    trainer.save_checkpoint = lambda step, epoch: saved.append(("weights", step, epoch))
    trainer.save_optimizer_state = lambda step: saved.append(("optimizer", step))
    trainer.save_training_state = lambda step, epoch, batch: saved.append(("state", step, epoch, batch))
    trainer.load_checkpoint = lambda _path: 0
    trainer.load_training_state = lambda _step: None
    trainer.load_optimizer_state = lambda _step: False

    assert prepare_abc_items(trainer, datasets)[0].target_tokens > 1
    stopped = train_abc_ar_loop(trainer, datasets=datasets, total_steps=1,
                                batch_size=1, save_every_n_steps=0)
    assert stopped is False
    assert model.weight.item() < 1.0
    assert [entry[0] for entry in saved] == ["weights", "optimizer", "state"]


def test_token_native_loop_saves_completed_step_on_stop(tmp_path):
    items = []
    for index in range(2):
        audio = tmp_path / f"example-{index}.flac"
        audio.write_bytes(b"unused")
        audio.with_suffix(".abc").write_text("X:1\nK:C\nCDEF", encoding="utf-8")
        items.append({
            "image_path": str(audio), "audio_path": str(audio), "caption": "pop",
            "lyrics": "[Verse]\nhello", "width": None, "height": None,
        })
    datasets = [SimpleNamespace(items=items)]

    class ToyAR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, **_kwargs):
            return SimpleNamespace(loss=self.weight.square())

    model = ToyAR()
    saved = []
    checks = 0
    trainer = SimpleNamespace(
        config={"yue2_training_objective": "abc_ar"}, tokenizer=TinyTokenizer(),
        transformer=model, device=torch.device("cpu"), training_dtype=torch.float32,
        mixed_precision=False, run_id=None, _metrics_buffer=[], output_dir=tmp_path,
        run_name="test", scaler=None, log_prefix="[test]",
    )
    trainer.setup_optimizer = lambda _kind, _schedule, _steps: (
        setattr(trainer, "optimizer", torch.optim.SGD(model.parameters(), lr=0.1)),
        setattr(trainer, "lr_scheduler", torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lambda _step: 1.0)),
    )

    def stop_after_first_update():
        nonlocal checks
        checks += 1
        if checks == 4:  # two preparation checks, then the second training batch
            raise KeyboardInterrupt("stop")

    trainer._check_stop_requested = stop_after_first_update
    trainer.save_checkpoint = lambda step, epoch: saved.append(("weights", step, epoch))
    trainer.save_optimizer_state = lambda step: saved.append(("optimizer", step))
    trainer.save_training_state = lambda step, epoch, batch: saved.append(
        ("state", step, epoch, batch)
    )

    with pytest.raises(KeyboardInterrupt, match="stop"):
        train_abc_ar_loop(trainer, datasets=datasets, total_steps=2,
                          batch_size=1, save_every_n_steps=0)
    assert saved == [
        ("weights", 1, 0), ("optimizer", 1), ("state", 1, 0, 1),
    ]


def test_token_native_loop_autocasts_bf16_base_with_fp32_lora_on_cpu(tmp_path):
    audio = tmp_path / "example.flac"
    audio.write_bytes(b"unused")
    audio.with_suffix(".abc").write_text("X:1\nK:C\nCDEF", encoding="utf-8")
    datasets = [SimpleNamespace(items=[{
        "image_path": str(audio), "audio_path": str(audio), "caption": "pop",
        "lyrics": "[Verse]\nhello", "width": None, "height": None,
    }])]

    class MixedDtypeAR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            base.requires_grad_(False)
            self.projection = LoRALinearLayer(
                base, 1, 1, "lora_unet_probe", torch.float32
            )

        def forward(self, *, input_ids, **_kwargs):
            hidden = torch.ones(
                input_ids.shape[0], 4, dtype=torch.bfloat16,
                device=input_ids.device,
            )
            return SimpleNamespace(loss=self.projection(hidden).float().square().mean())

    model = MixedDtypeAR()
    trainer = SimpleNamespace(
        config={"yue2_training_objective": "abc_ar"}, tokenizer=TinyTokenizer(),
        transformer=model, device=torch.device("cpu"), training_dtype=torch.bfloat16,
        mixed_precision=True, run_id=None, _metrics_buffer=[], output_dir=tmp_path,
        run_name="test", scaler=None, log_prefix="[test]",
    )
    trainer.setup_optimizer = lambda _kind, _schedule, _steps: (
        setattr(trainer, "optimizer", torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )),
        setattr(trainer, "lr_scheduler", torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lambda _step: 1.0)),
    )
    trainer._check_stop_requested = lambda: None
    trainer.save_checkpoint = lambda *_args: None
    trainer.save_optimizer_state = lambda *_args: None
    trainer.save_training_state = lambda *_args: None

    assert train_abc_ar_loop(
        trainer, datasets=datasets, total_steps=1, batch_size=1,
        save_every_n_steps=0,
    ) is False


def test_planner_lora_is_active_only_during_abc_stage(tmp_path):
    trained = tiny_model(layers=2)
    trainer = SimpleNamespace(transformer=trained, learning_rate=1e-4, unet_lr=None,
                              arch=None, adapter_algorithm="lora", weight_decompose=False,
                              adapter_config={})
    adapter = YuE2LoRAAdapter(trainer, 4, 4, objective="abc_ar")
    layers = {}
    adapter.apply_lora_to_unet(layers)
    path = tmp_path / "planner.safetensors"
    adapter.save_checkpoint(layers, 1, 0, path)

    runtime = tiny_model(layers=2)
    backend = YuE2Mixin()
    backend.yue2_components = {"transformer": runtime}
    backend._yue2_resolve_lora_path = lambda raw: str(raw) if Path(raw).is_file() else None
    files = backend._prepare_yue2_loras([{"path": str(path), "strength": 0.75}])
    assert backend._set_yue2_adapter_stage(files, "abc") == 8
    assert isinstance(runtime.model.layers[0].self_attn.q_proj, CompositeAdapterLayer)
    assert backend._set_yue2_adapter_stage(files, "semantic") == 0
    assert not isinstance(runtime.model.layers[0].self_attn.q_proj, CompositeAdapterLayer)
    assert backend._set_yue2_adapter_stage(files, None) == 0
