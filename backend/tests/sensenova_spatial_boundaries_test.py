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


def test_invalid_patch_fails_before_io_replacement():
    model = _PixelTree()
    incoming = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding
    outgoing = model.fm_modules["fm_head"].conv2
    with pytest.raises(ValueError, match="multiple of 4"):
        migrate_generation_io(model, channels=7, patch=10, scale=16)
    assert model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding is incoming
    assert model.fm_modules["fm_head"].conv2 is outgoing
    assert gen_geometry(model).patch == 32


@pytest.mark.parametrize("old,size", [(16,6), (6,16), (8,3), (3,8)])
def test_fractional_patch_resampling_preserves_integral_and_constant_fields(old,size):
    from core.models.sensenova.vae_alignment import _spatial_weights
    torch.manual_seed(29)
    value = torch.randn(2,3,old,old)
    averaged = _spatial_weights(value,size)
    torch.testing.assert_close(averaged.mean((-1,-2)),value.mean((-1,-2)))
    summed = _spatial_weights(value,size,sum_preserving=True)
    torch.testing.assert_close(summed.sum((-1,-2)),value.sum((-1,-2)),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(_spatial_weights(torch.ones_like(value),size),torch.ones(2,3,size,size))


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
def test_fractional_patch_migration_preserves_input_response_and_per_token_output_mean(dtype):
    torch.manual_seed(30)
    model = _PixelTree().to(dtype=dtype)
    head = model.fm_modules["fm_head"]
    hidden = torch.randn(1,64,2,3).to(dtype)
    old_output = head(hidden).float().reshape(1,3,2,32,3,32).mean((3,5))
    old_response = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding(
        torch.ones(1,3,16,16).to(dtype))
    migrate_generation_io(model,channels=7,patch=12,scale=16)
    image = torch.cat((torch.ones(1,3,6,6),torch.randn(1,4,6,6)),1).to(dtype)
    new_response = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding(image)
    new_output = head(hidden).float().reshape(1,7,2,12,3,12)
    tolerance = .05 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(new_response,old_response,atol=tolerance,rtol=tolerance)
    torch.testing.assert_close(new_output[:,:3].mean((3,5)),old_output,atol=tolerance,rtol=tolerance)
    assert torch.count_nonzero(new_output[:,3:]) == 0
    assert gen_geometry(model).patch == 12


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


def test_shared_calibration_and_native_codec_full_export_round_trip(tmp_path):
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL
    from core.models.common.vae_source import resolve_vae_source, load_declared_latent_io
    from core.models.common.single_file_format import read_state_dict, strip_prefix, TRANSFORMER_PREFIX, VAE_PREFIX
    from core.models.components.spatial_swap import restored_image_codec
    from core.models.sensenova.vae_alignment import initialize_swap, calibrate
    from core.models.sensenova.loader import save_sensenova_full_finetune_checkpoint, install_sensenova_state_dict, _assert_declared_latent_geometry
    from core.training.arch.sensenova import SenseNovaArchHandler
    from core.training.vae_swap import swap_metadata
    from sensenova_full_finetune_save_test import _trained_tree, _Decoder, _plain
    target = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=2,
        block_out_channels=(4,) * 5, down_block_types=("DownEncoderBlock2D",) * 5,
        up_block_types=("UpDecoderBlock2D",) * 5, norm_num_groups=1, sample_size=128,
        scaling_factor=1.)
    target.save_pretrained(tmp_path / "target")
    resolved = resolve_vae_source(f"file:{tmp_path / 'target'}", arch="sensenova")
    model = _PixelTree()
    trainer = SimpleNamespace(transformer=model, vae=None, config={"training_method":"full_finetune",
        "vae_swap_calibration_samples":2,"vae_swap_calibration_resolution":128},
        device=torch.device("cpu"), weight_dtype=torch.float32, vae_dtype=torch.float32,
        text_encoder=None, text_encoder_2=None, _check_stop_requested=lambda:None,
        sensenova_train_fm_modules=True)
    handler = SenseNovaArchHandler(trainer)
    trainer.arch = handler
    trainer._main_model_module = lambda: model
    trainer.move_main_model_to_cpu = lambda: model.to("cpu")
    trainer.move_main_model_to_gpu = lambda: model.to("cpu")
    trainer.move_vae_to_cpu = lambda: trainer.vae.to("cpu")
    initialize_swap(trainer, resolved, handler=handler)
    assert model.gen_in_channels == 3 and trainer.wiring.latent_channels == 2
    assert trainer.wiring.vae_scale_factor == 16 and trainer.wiring.vae_norm == "identity"
    assert model.vae_latent_bridge.declaration["old_scale"] == 1
    assert not model.vae_latent_bridge.ready
    paths = []
    for index in range(2):
        path = tmp_path / f"synthetic{index}.png"
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(path)
        paths.append({"image_path":str(path)})
    calibrate(trainer, [SimpleNamespace(items=paths)])
    assert model.vae_latent_bridge.ready
    assert not hasattr(trainer, "_sensenova_vae_calibration_source")
    from core.training.adapters.sensenova_adapter import SenseNovaFullParameterAdapter
    adapter = SenseNovaFullParameterAdapter(trainer)
    parameters = adapter._fm_parameters("gen")
    assert {id(p) for p in model.vae_latent_bridge.parameters()} <= {id(p) for p in parameters}
    saved_pair = model.vae_latent_bridge
    handler.apply_vae_swap(trainer, resolved, module=trainer.vae)
    assert model.vae_latent_bridge is saved_pair
    _, _, metadata = swap_metadata(trainer)
    exported = _trained_tree("gen")
    exported.vae_latent_bridge = model.vae_latent_bridge
    path, _ = save_sensenova_full_finetune_checkpoint(exported, str(tmp_path / "full"),
        branch="gen", save_format="mixed", config={"downsample_ratio":.5},
        extra_metadata=metadata, vae=trainer.vae, max_shard_bytes=4096)
    state, metadata = read_state_dict(path)
    declaration = read_declaration(metadata)
    assert declaration == model.vae_latent_bridge.declaration
    assert not any(key.startswith("vae.inner.") for key in state)
    restored = _Decoder(factory=_plain)
    install(restored, declaration)
    install_sensenova_state_dict(restored, strip_prefix(state, TRANSFORMER_PREFIX), {},
        torch.float32, require_complete=True)
    validate_loaded(restored)
    for name, value in exported.vae_latent_bridge.state_dict().items():
        assert torch.equal(value, restored.vae_latent_bridge.state_dict()[name])
    saved_vae = load_declared_latent_io(path, arch="sensenova")
    facade, saved_vae = restored_image_codec(saved_vae.load_module(torch_dtype=torch.float32), saved_vae, declaration)
    for name, value in target.state_dict().items():
        assert torch.equal(value, facade.inner.state_dict()[name]), name
    config = SimpleNamespace(gen_in_channels=3, gen_patch_size=4)
    _assert_declared_latent_geometry(config, saved_vae, path=path, alignment=declaration)
    with pytest.raises(ValueError, match="generation grid"):
        _assert_declared_latent_geometry(SimpleNamespace(gen_in_channels=2, gen_patch_size=4),
            saved_vae, path=path, alignment=declaration)
    with pytest.raises(ValueError, match="declared VAE"):
        _assert_declared_latent_geometry(config, None, path=path, alignment=declaration)
    pixels = torch.randn(1, 3, 64, 64)
    assert facade.encode(pixels).latent_dist.mode().shape == (1, 2, 4, 4)
    assert facade.decode(torch.randn(1, 2, 4, 4)).sample.shape == pixels.shape
    broken = strip_prefix(state, TRANSFORMER_PREFIX)
    broken.pop("vae_latent_bridge.input.projection.weight")
    incomplete = _Decoder(factory=_plain)
    install(incomplete, declaration)
    with pytest.raises(RuntimeError, match="incomplete"):
        install_sensenova_state_dict(incomplete, broken, {}, torch.float32, require_complete=True)


@pytest.mark.parametrize("rewired,refiner,patch", [
    (False,False,4), (True,False,4), (False,True,4), (True,True,4),
    (False,False,12), (True,False,12)])
def test_actual_composite_model_public_loader_training_and_sampling(tmp_path, monkeypatch, rewired, refiner, patch):
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL, AutoencoderKLQwenImage
    from core.models.common.vae_source import resolve_vae_source
    from core.models.sensenova.vendor.configuration_neo_chat import NEOChatConfig
    from core.models.sensenova.vendor.modeling_qwen3 import create_block_causal_mask
    from core.models.sensenova.vae_alignment import initialize_swap, calibrate
    from core.models.sensenova.residual_unet import native_config, restore_topology, inspect_artifact
    from core.models.sensenova.loader import save_sensenova_full_finetune_checkpoint, load_sensenova_from_path, SENSENOVA_BRANCH_LINEAR_COUNTS
    from core.models.sensenova.sensenova_pipeline_ops import SenseNovaPrefix, denoise_loop
    from core.training.arch.sensenova import SenseNovaArchHandler
    from core.training.ops.sensenova_ops import forward_und_prefix_layers
    from core.training.vae_swap import swap_metadata, apply_latent_space, preflight_vae_swap
    from core.rewiring.spec import OPERATORS, build_spec
    from dit_rewiring_sensenova_test import SenseNovaRuntimeContracts
    raw_config = SenseNovaRuntimeContracts.full_native_config()
    raw_config.update(vision_config=dict(patch_size=16, hidden_size=32, llm_hidden_size=64, downsample_ratio=.5),
        llm_config=dict(architectures=["Qwen3ForCausalLM"], vocab_size=32, hidden_size=64,
            intermediate_size=80, num_hidden_layers=7, num_attention_heads=2,
            num_key_value_heads=1, head_dim=32, attention_dropout=0.))
    old_vae = old_identity = None
    channels = 8 if refiner else 2
    if refiner:
        from core.models.sensenova.latent_refiner import REFINER_VERSION, REFINER_INPUTS, REFINER_NORM
        raw_config.update(gen_in_channels=3, gen_patch_size=4, gen_refiner=dict(
            version=REFINER_VERSION, width=16, depth=1, inputs=list(REFINER_INPUTS), norm=REFINER_NORM,
            detach_anchor_step=None, detach_steps=None, detach_accum=None))
        old_vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=3,
            block_out_channels=(4,)*4, down_block_types=("DownEncoderBlock2D",)*4,
            up_block_types=("UpDecoderBlock2D",)*4, norm_num_groups=1, sample_size=128, scaling_factor=1.)
        old_vae.save_pretrained(tmp_path / "source")
        old_identity = resolve_vae_source(f"file:{tmp_path / 'source'}", arch="sensenova")
    config = NEOChatConfig(**raw_config)
    model = NEOChatModel(config).float().eval()
    if refiner:
        model.gen_vae_scale_factor = 8
        nn.init.normal_(model.fm_modules["fm_refiner"].out.weight, std=.01)
        model.fm_modules["fm_refiner"].gate.fill_(.75)
    if rewired:
        restore_topology(model, build_spec("sensenova", [{"name":"main","block_count":7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
                 down_factor=[2,2], **OPERATORS),
            dict(id="inner", stream="main", parent="outer", start_block=2, end_block=5,
                 down_factor=[2,2], **OPERATORS)], native_config(config)))
    target = (AutoencoderKLQwenImage(base_dim=4, z_dim=channels, dim_mult=[1]*5, num_res_blocks=1,
        temperal_downsample=[False]*4, latents_mean=[0.]*channels, latents_std=[1.]*channels)
        if rewired else AutoencoderKL(in_channels=3, out_channels=3, latent_channels=channels,
            block_out_channels=(4,) * 5, down_block_types=("DownEncoderBlock2D",) * 5,
            up_block_types=("UpDecoderBlock2D",) * 5, norm_num_groups=1, sample_size=128, scaling_factor=1.))
    target.save_pretrained(tmp_path / "target")
    resolved = resolve_vae_source(f"file:{tmp_path / 'target'}", arch="sensenova")
    trainer = SimpleNamespace(transformer=model, vae=old_vae, is_sensenova=True, config={"training_method":"full_finetune",
        "vae_swap_source":f"file:{tmp_path / 'target'}",
        "sensenova_gen_patch":patch,
        "vae_swap_calibration_samples":2, "vae_swap_calibration_resolution":128},
        sensenova_train_fm_modules=True, device=torch.device("cpu"), weight_dtype=torch.float32,
        training_dtype=torch.float32, vae_dtype=torch.float32, gradient_checkpointing=False,
        text_encoder=None, text_encoder_2=None, _check_stop_requested=lambda:None,
        sensenova_config_dict=raw_config)
    handler = SenseNovaArchHandler(trainer)
    trainer.arch = handler
    trainer._main_model_module = lambda: trainer.transformer
    trainer.move_main_model_to_cpu = lambda: trainer.transformer.to("cpu")
    trainer.move_main_model_to_gpu = lambda: trainer.transformer.to("cpu")
    trainer.move_vae_to_cpu = lambda: trainer.vae.to("cpu")
    preflight_vae_swap(trainer.config, arch="sensenova", method="full_finetune", bundle_vae_explicit_false=False)
    apply_latent_space(trainer, old_identity)
    if refiner:
        assert model.fm_modules["fm_refiner"].in_channels == channels
    paths = []
    for i in range(2):
        image = tmp_path / f"synthetic{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (128,128,3), dtype=np.uint8)).save(image)
        paths.append({"image_path":str(image)})
    calibrate(trainer, [SimpleNamespace(items=paths)])
    _, _, metadata = swap_metadata(trainer)
    # Only the export census is narrowed; construction and weight loading stay native.
    monkeypatch.setitem(SENSENOVA_BRANCH_LINEAR_COUNTS, "gen", 49)
    monkeypatch.setitem(SENSENOVA_BRANCH_LINEAR_COUNTS, "und", 49)
    path, _ = save_sensenova_full_finetune_checkpoint(model, str(tmp_path / "actual"),
        branch="both", save_format="bf16", config=model.config,
        raw_config=trainer.sensenova_config_dict,
        extra_metadata=metadata, vae=trainer.vae, max_shard_bytes=100000 if rewired else None)
    monkeypatch.setattr("core.models.sensenova.loader._load_sensenova_tokenizer", lambda directory: object())
    components = load_sensenova_from_path(path, torch_dtype=torch.float32)
    inspected, topology, stride = inspect_artifact(path)
    assert inspected["gen_in_channels"] == max(3,channels) and stride == patch * 16
    assert topology == getattr(model, "_residual_unet_spec", None)
    restored = components["transformer"]
    assert restored.vae_latent_bridge.ready
    assert components["wiring"].latent_channels == channels
    assert components["wiring"].vae_scale_factor == 16
    assert getattr(restored, "_residual_unet_spec", None) == getattr(model, "_residual_unet_spec", None)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    for name, value in target.state_dict().items():
        assert torch.equal(value, components["vae"].inner.state_dict()[name]), name

    def prefix_cache(owner, generation=False):
        indexes = torch.stack((torch.arange(4), torch.zeros(4), torch.zeros(4))).long()
        if generation:
            return owner.language_model.model(input_ids=torch.tensor([[2,5,7,3]]), indexes=indexes,
                image_gen_indicators=torch.zeros(1,4,dtype=torch.bool),
                attention_mask={"full_attention":create_block_causal_mask(indexes[0])},
                use_cache=True).past_key_values
        with torch.no_grad():
            return forward_und_prefix_layers(owner.language_model.model, torch.tensor([[2,5,7,3]]),
                indexes, {"full_attention":create_block_causal_mask(indexes[0])}, checkpoint_layers=False)

    def sample(owner):
        cache = prefix_cache(owner, generation=True)
        prefix = SenseNovaPrefix(device=torch.device("cpu"), dtype=torch.float32, batch_size=1,
            token_h=4, token_w=4, grid_h=8, grid_w=8, merge_size=2, image_size=(patch*64,patch*64),
            gen_size=(patch*4,patch*4), cond_past_key_values=cache,
            cond_indexes_image=owner._build_t2i_image_indexes(4,4,4,device="cpu"),
            cond_attention_mask={"full_attention":None})
        result = denoise_loop(owner, prefix, cfg_scale=1., timestep_shift=1., num_inference_steps=2, seed=25)
        assert prefix.consumed
        assert result.shape == (1,channels,patch*4,patch*4) and torch.isfinite(result).all()
        return result

    torch.testing.assert_close(sample(model), sample(restored))
    restored.requires_grad_(True)
    owner = SimpleNamespace(transformer=restored, device=torch.device("cpu"),
        training_dtype=torch.float32, gradient_checkpointing=False, defer_extra_metric=lambda *args:None)
    loss, _, _ = train_step(owner, images=torch.randn(1,channels,patch*4,patch*4),
        prefix=SenseNovaTrainingPrefix(prefix_cache(restored), text_length=4), timesteps=torch.tensor([.25]))
    loss.backward()
    assert torch.isfinite(loss)
    for bridge in (restored.vae_latent_bridge.input, restored.vae_latent_bridge.output):
        assert bridge.projection.weight.grad is not None and torch.isfinite(bridge.projection.weight.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_expanded_refiner_preserves_each_input_half_and_learned_residual(dtype):
    from core.models.sensenova.latent_refiner import LatentRefiner
    from core.models.sensenova.vae_alignment import _expanded_refiner_io
    torch.manual_seed(83)
    refiner = LatentRefiner(3,16,2).to(dtype=dtype)
    refiner.gate.data = torch.tensor(.75, dtype=torch.float32)
    nn.init.normal_(refiner.out.weight, std=.02)
    gate = refiner.gate
    preserved = {name:value.clone() for name,value in refiner.state_dict().items()
        if not name.startswith(("stem.","out."))}
    head, noisy = torch.randn(2,7,8,12).to(dtype), torch.randn(2,7,8,12).to(dtype)
    t = torch.tensor([.25,.75])
    original = refiner(head[:,:3], noisy[:,:3], t, 1.2)
    old_stem = refiner.stem.weight.clone()
    refiner.stem, refiner.out = _expanded_refiner_io(refiner, 7)
    refiner.in_channels = 7
    assert torch.equal(refiner.stem.weight[:,:3],old_stem[:,:3])
    assert torch.equal(refiner.stem.weight[:,7:10],old_stem[:,3:])
    for name,value in preserved.items():
        assert torch.equal(value,refiner.state_dict()[name])
    assert refiner.gate is gate and gate.dtype == torch.float32
    expanded = refiner(head,noisy,t,1.2)
    tolerance = .01 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(expanded[:,:3],original,atol=tolerance,rtol=tolerance)
    torch.testing.assert_close(expanded[:,3:],head[:,3:],atol=0,rtol=0)
    (expanded.float()-torch.randn_like(expanded)).square().mean().backward()
    assert refiner.stem.weight.grad[:,3:7].abs().sum() > 0
    assert refiner.out.weight.grad[3:].abs().sum() > 0
