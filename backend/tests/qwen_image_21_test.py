import json
import asyncio
import sys
from pathlib import Path

import torch
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
from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin
from api import routes
from api.schema_routes import get_arch_capabilities, get_generation_defaults


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
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
    ]


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
    assert "qwen_image_21_kv_cache" not in unsupported


def test_generation_callback_uses_shared_progress_contract():
    latents = torch.randn(1, 4, 8)
    progress_calls = []
    step_calls = []

    class _Pipe:
        _interrupt = False

        def __call__(self, **kwargs):
            callback_kwargs = {"latents": latents}
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
        progress_callback=lambda step, total, current: progress_calls.append(
            (step, total, current)
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
    assert step_calls[0][0] == 0
    assert step_calls[0][2] is latents


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
        loader, "load_transformer", lambda *_args, **_kwargs: (trained_transformer, "bf16")
    )
    loaded = loader.load_qwen_image_21_components(str(checkpoint))
    assert loaded["transformer"] is trained_transformer
    assert loaded["checkpoint_path"] == str(checkpoint)
