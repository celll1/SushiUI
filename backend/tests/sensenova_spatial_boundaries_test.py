"""Synthetic tests of trained patch surgery and external-grid diffusion boundaries."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensenova_latent_migration_test import _PixelTree, _LatentTree
from sensenova_training_core_test import _Cache
from core.models.components.latent_adapter import SpatialAlignmentPair
from core.models.components.spatial_swap import artifact_metadata, read_declaration, install, validate_loaded
from core.models.sensenova.vae_alignment import migrate_generation_io, input_state, output_x0
from core.models.sensenova.latent_space import gen_geometry
from core.models.sensenova.sensenova_pipeline_ops import _build_step_context
from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel
from core.training.ops.sensenova_ops import train_step, SenseNovaTrainingPrefix


def _pair(old, new, dtype=torch.float32):
    declaration = dict(version=1, method="paired_spatial", token_policy="target_native",
        source="file:synthetic", source_identity="pixel", target_identity="synthetic",
        old_channels=old, new_channels=new, old_scale=8, new_scale=16,
        ndim=4, norm="identity", norm_pack=1, input_channels=3)
    pair = SpatialAlignmentPair(declaration).to(dtype=dtype)
    for bridge in (pair.input, pair.output):
        weight = torch.zeros(bridge.source_channels * 9, bridge.target_channels)
        for channel in range(min(bridge.source_channels, bridge.target_channels)):
            weight[channel * 9 + 4, channel] = 1
        bridge.initialize(weight, torch.arange(bridge.target_channels).float() / 20)
    return pair


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("patch", [4, 8, 16])
def test_trained_pixel_io_keeps_constant_fields_and_block_averaged_head(dtype, patch):
    torch.manual_seed(27)
    model = _PixelTree().to(dtype=dtype)
    incoming = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding
    head = model.fm_modules["fm_head"]
    hidden = torch.randn(1, 64, 2, 3).to(dtype)
    native_output = head(hidden)
    native_input = torch.randn(1, 3, 4, 6).to(dtype)
    old_response = incoming(native_input.repeat_interleave(16, -2).repeat_interleave(16, -1))
    untouched = {name: value.clone() for name, value in model.state_dict().items()
        if "patch_embedding" not in name and "fm_head.conv2" not in name}
    migrate_generation_io(model, channels=7, patch=patch, scale=16)
    new_input = native_input.repeat_interleave(patch // 2, -2).repeat_interleave(patch // 2, -1)
    new_input = torch.cat((new_input, torch.randn(1, 4, *new_input.shape[-2:]).to(dtype)), 1)
    new_response = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding(new_input)
    tolerance = .05 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(new_response, old_response, atol=tolerance, rtol=tolerance)
    new_output = head(hidden)
    torch.testing.assert_close(new_output[:, :3], F.avg_pool2d(native_output.float(), 32 // patch).to(dtype),
                               atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(new_output[:, 3:]) == 0
    for name, value in untouched.items():
        assert torch.equal(value, model.state_dict()[name]), name
    assert gen_geometry(model).token_pixel_width == patch * 16
    assert model.config.gen_in_channels == 7
    assert model.fm_modules["vision_model_mot_gen"].embeddings.config.patch_size == patch // 2
    assert model.noise_scale == 1.0


def test_invalid_patch_ratio_fails_before_io_replacement():
    model = _PixelTree()
    incoming = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding
    outgoing = model.fm_modules["fm_head"].conv2
    with pytest.raises(ValueError, match="integer patch ratios"):
        migrate_generation_io(model, channels=7, patch=12, scale=16)
    assert model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding is incoming
    assert model.fm_modules["fm_head"].conv2 is outgoing
    assert gen_geometry(model).patch == 32


@pytest.mark.parametrize("external,refiner", [(4, False), (24, False), (4, True)])
def test_real_training_boundary_keeps_external_noise_and_trains_both_bridges(external, refiner):
    torch.manual_seed(28)
    internal = max(16, external)
    model = _LatentTree(channels=internal)
    if refiner:
        from core.models.sensenova.latent_refiner import LatentRefiner
        model.fm_modules["fm_refiner"] = LatentRefiner(internal, 16, 1)
    nn.init.normal_(model.fm_modules["fm_head"].conv2.weight, std=.03)
    model.vae_latent_bridge = _pair(16, external)
    image = torch.randn(1, external, 4, 4)
    shape = SimpleNamespace(batch_size=1, grid_h=2, grid_w=2, token_h=1, token_w=1)
    z, _, _ = _build_step_context(model, shape, image, torch.tensor(.25), 1., enable_grad=True)
    assert torch.equal(z, model.patchify(image, 4))
    mapped = input_state(model, image, torch.tensor([.25]))
    torch.testing.assert_close(mapped[:, :min(external, internal)],
        image[:, :min(external, internal)] + .25 * model.vae_latent_bridge.input.projection.bias[:min(external, internal)][None, :, None, None])
    metrics = []
    trainer = SimpleNamespace(transformer=model, device=torch.device("cpu"),
        training_dtype=torch.float32, gradient_checkpointing=False,
        defer_extra_metric=lambda name, value: metrics.append((name, value)))
    loss, _, _ = train_step(trainer, images=image,
        prefix=SenseNovaTrainingPrefix(_Cache(), text_length=3), timesteps=torch.tensor([.25]))
    loss.backward()
    assert torch.isfinite(loss)
    if refiner:
        assert any(name == "sn_refiner_delta_rel" for name, value in metrics)
    for bridge in (model.vae_latent_bridge.input, model.vae_latent_bridge.output):
        assert bridge.projection.weight.grad is not None
        assert torch.isfinite(bridge.projection.weight.grad).all()
        assert torch.count_nonzero(bridge.projection.weight.grad) > 0
    assert gen_geometry(model).channels == external
    assert gen_geometry(model).token_pixel_width == 64
    restored = _LatentTree(channels=internal)
    if refiner:
        restored.fm_modules["fm_refiner"] = LatentRefiner(internal, 16, 1)
    install(restored, read_declaration(artifact_metadata(model)))
    restored.load_state_dict(model.state_dict(), strict=True)
    validate_loaded(restored)
    torch.testing.assert_close(output_x0(restored, torch.ones(1, internal, 4, 4)),
                               output_x0(model, torch.ones(1, internal, 4, 4)))


@pytest.mark.parametrize("refiner", [False, True])
def test_vendor_inference_maps_x0_before_external_velocity(refiner):
    model = _LatentTree(channels=16)
    if refiner:
        from core.models.sensenova.latent_refiner import LatentRefiner
        model.fm_modules["fm_refiner"] = LatentRefiner(16, 16, 1)
    model.vae_latent_bridge = _pair(16, 4)
    model.language_model.model.forward = lambda **kwargs: SimpleNamespace(last_hidden_state=torch.ones(1, 1, 64))
    # The model's tiny decoder inherits nn.Module, so the real vendor head path runs.
    z = torch.randn(1, 1, 4 * 4 * 4)
    t = torch.tensor(.25)
    velocity = NEOChatModel._t2i_predict_v(model, torch.ones(1, 1, 64), None, None, None,
        t, z, 1, token_hw=(1, 1), noise_scale=1.)
    expected_x0 = model.patchify(output_x0(model, model.fm_modules["fm_head"](torch.ones(1, 64, 1, 1))), 4)
    torch.testing.assert_close(velocity, (expected_x0 - z) / (1 - t))
    assert velocity.shape == z.shape


def test_patch_migration_preserves_unet_regions_and_updates_native_identity():
    from core.models.sensenova.vendor.configuration_neo_chat import NEOChatConfig
    from core.models.sensenova.vendor.modeling_qwen3 import Qwen3Model
    from core.models.sensenova.residual_unet import native_config, restore_topology, validate_owner_topology
    from core.rewiring.spec import build_spec, OPERATORS
    model = _PixelTree()
    model.config = NEOChatConfig(vision_config=dict(patch_size=16), use_pixel_head=True,
        llm_config=dict(architectures=["Qwen3ForCausalLM"], vocab_size=32, hidden_size=64,
            intermediate_size=80, num_hidden_layers=7, num_attention_heads=2,
            num_key_value_heads=1, head_dim=32, attention_dropout=0.0))
    model.config.llm_config._attn_implementation = "eager"
    model.language_model = nn.Module()
    model.language_model.model = Qwen3Model(model.config.llm_config)
    regions = [dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
        down_factor=[2, 2], **OPERATORS), dict(id="inner", stream="main", parent="outer",
        start_block=2, end_block=5, down_factor=[2, 2], **OPERATORS)]
    old_spec = build_spec("sensenova", [{"name":"main","block_count":7}], regions, native_config(model.config))
    restore_topology(model, old_spec)
    native_weights = {name: value.clone() for name, value in model.language_model.state_dict().items()}
    migrate_generation_io(model, channels=7, patch=4, scale=16)
    updated = model._residual_unet_spec
    assert updated["regions"] == old_spec["regions"]
    assert updated != old_spec
    assert model.language_model.model._residual_unet_spec == updated
    validate_owner_topology(updated, model)
    for name, value in native_weights.items():
        assert torch.equal(value, model.language_model.state_dict()[name])
