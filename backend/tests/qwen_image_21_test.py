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
    build_fixed_partition_plan,
    flatten_region,
    full_canvas_position_ids,
)
from core.training.ops import qwen_image_21_ops
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


def test_qwen_partition_checkpoint_auto_policy_keeps_measured_floor():
    plan = build_fixed_partition_plan(64, 64, count=4, seed=2)
    trainer = SimpleNamespace(
        config={"qwen_partition_gradient_checkpointing_blocks": None},
        transformer=SimpleNamespace(transformer_blocks=[object()] * 32),
    )
    assert qwen_image_21_ops._resolve_partition_checkpoint_blocks(
        trainer, plan, 16
    ) == 8
    trainer.config["qwen_partition_gradient_checkpointing_blocks"] = 11
    assert qwen_image_21_ops._resolve_partition_checkpoint_blocks(
        trainer, plan, 16
    ) == 11


def test_qwen_partition_api_refuses_odd_halo():
    with pytest.raises(ValueError, match="qwen_partition_halo_tokens must be even"):
        routes.TrainingRunCreateRequest(qwen_partition_halo_tokens=3)


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
        loader, "load_transformer", lambda *_args, **_kwargs: (trained_transformer, "bf16")
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
