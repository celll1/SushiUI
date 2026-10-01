"""Native conditioner gradient/lifecycle tests run entirely on CPU."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def forbidden_cuda(*args, **kwargs):
    raise AssertionError("Conditioner CPU test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.conditioning import MingConditioner
from core.models.ming_image_01.vendor.configuration_bailingmm2 import BailingMM2Config
from core.models.ming_image_01.vendor.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from core.models.ming_image_01.loader import _strict_loading_info


def make_conditioner():
    config = BailingMM2Config(
        llm_config={"vocab_size": 64, "hidden_size": 24, "intermediate_size": 48,
                    "num_hidden_layers": 2, "num_attention_heads": 3, "num_key_value_heads": 3,
                    "head_dim": 8, "pad_token_id": 0, "num_experts": 4,
                    "num_experts_per_tok": 2, "num_shared_experts": 1, "n_group": 1, "topk_group": 1,
                    "moe_intermediate_size": 16, "first_k_dense_replace": 1,
                    "image_patch_token": 40, "image_start_token": 41, "image_end_token": 42,
                    "video_start_token": 43, "video_patch_token": 44,
                    "rope_scaling": {"type": "default", "mrope_section": [1, 0, 1]},
                    "embedding_dropout": 0.0, "attention_dropout": 0.0,
                    "output_dropout": 0.0, "_attn_implementation": "eager"},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "patch_size": 2, "temporal_patch_size": 1,
                       "spatial_merge_size": 2, "out_hidden_size": 8192,
                       "window_size": 8, "fullatt_block_indexes": [0],
                       "_attn_implementation": "eager"})
    config._attn_implementation = "eager"
    native = BailingMM2NativeForConditionalGeneration(config)
    connector_config = Qwen2Config(vocab_size=64, hidden_size=12, intermediate_size=24,
                                  num_hidden_layers=1, num_attention_heads=3,
                                  num_key_value_heads=3, attention_dropout=0.0)
    connector_config._attn_implementation = "eager"
    connector = Qwen2ForCausalLM(connector_config)
    projection_config = {"img_gen_scales": [2], "connector_norm": False,
                         "use_vlm_directvlm_condition": True, "use_learnable_token_condition": True,
                         "selected_hidden_states_layers": [1, 2], "diffusion_inner_dim": 24,
                         "diffusion_c_input_dim": 8}
    result = MingConditioner(native, connector, projection_config)
    torch.nn.init.normal_(native.query_tokens_dict["2x2"], std=0.02)
    return result


def inputs():
    return {"input_ids": torch.tensor([[1, 3, 5, 6]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}


def test_frozen_conditioning_is_cacheable_and_zero_branch_is_zero():
    model = make_conditioner().eval()
    model.set_trainable(set())
    result = model(**inputs())
    assert result.queries.shape == (1, 4, 8)
    assert result.direct.shape == (1, 4, 24)
    assert model.cacheable
    assert not result.queries.requires_grad and not result.direct.requires_grad
    assert torch.count_nonzero(result.zero().queries) == 0
    assert torch.count_nonzero(result.zero().direct) == 0
    assert not model.native.connector.model.layers[0].self_attn.is_causal
    assert not torch.cuda.is_initialized()


def test_full_language_training_preserves_direct_stream_gradient():
    model = make_conditioner()
    model.set_trainable({"language", "projections", "connector", "queries"})
    result = model(**inputs())
    result.direct.square().mean().backward()
    embedding = model.native.model.get_input_embeddings().weight
    assert embedding.grad is not None and embedding.grad.abs().sum() > 0
    assert model.native.proj_directvlm[1].weight.grad.abs().sum() > 0
    assert not model.cacheable
    assert all(p.grad is None and not p.requires_grad for p in model.native.vision.parameters())
    assert not torch.cuda.is_initialized()


def test_query_stream_trains_connector_queries_and_language():
    model = make_conditioner()
    model.set_trainable({"language", "projections", "connector", "queries"})
    model(**inputs()).queries.square().mean().backward()
    assert model.native.query_tokens_dict["2x2"].grad.abs().sum() > 0
    assert model.native.proj_in.weight.grad.abs().sum() > 0
    assert model.native.connector.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.native.model.get_input_embeddings().weight.grad.abs().sum() > 0


def test_projection_only_scope_and_checkpoint_round_trip():
    model = make_conditioner()
    state = model.projection_state_dict()
    clone = make_conditioner()
    clone.load_projection_state(state)
    assert set(clone.projection_state_dict()) == set(state)
    assert all(torch.equal(value, clone.projection_state_dict()[key]) for key, value in state.items())
    clone.set_trainable({"projections"})
    clone(**inputs()).direct.square().mean().backward()
    assert clone.native.proj_directvlm[1].weight.grad is not None
    assert clone.native.model.get_input_embeddings().weight.grad is None
    assert not clone.native._ming_live_conditioning
    with pytest.raises(ValueError, match="Unknown"):
        clone.set_trainable({"foreign"})
    assert not clone.cacheable


def test_projection_state_shape_and_key_mismatch_refused():
    model = make_conditioner()
    state = model.projection_state_dict()
    with pytest.raises(ValueError, match="state mismatch"):
        model.load_projection_state({})
    state["proj_in.weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="shape"):
        model.load_projection_state(state)


def test_loader_refuses_incomplete_native_state():
    _strict_loading_info({"missing_keys": [], "unexpected_keys": []}, "synthetic")
    with pytest.raises(ValueError, match="state mismatch"):
        _strict_loading_info({"missing_keys": ["vision.weight"]}, "mllm")


def test_native_package_loader_round_trip(tmp_path, monkeypatch):
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from safetensors.torch import save_file
    from diffusers import FlowMatchEulerDiscreteScheduler
    from core.models.ming_image_01.model_definition import create_transformer
    from core.models.ming_image_01.vendor.diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    from core.models.ming_image_01.vendor.tokenization_bailing import BailingTokenizer
    from core.models.ming_image_01.vendor.image_processing_bailingmm2 import BailingMM2ImageProcessor
    from core.models.ming_image_01.loader import load_components

    original = make_conditioner().eval()
    original.set_trainable(set())
    excluded = ("connector.", "query_tokens_dict.", "proj_in.", "proj_out.", "proj_directvlm.")
    mllm_state = {key: value for key, value in original.native.state_dict().items() if not key.startswith(excluded)}
    original.native.save_pretrained(tmp_path / "mllm", state_dict=mllm_state)
    original.native.connector.save_pretrained(tmp_path / "connector")
    vocabulary = {f"word{i}": i for i in range(64)}
    tokenizer = BailingTokenizer(tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="word3")),
                                bos_token="word1", eos_token="word2", pad_token="word0",
                                gmask_token="word3", cls_token="word4")
    tokenizer.save_pretrained(tmp_path / "mllm")
    BailingMM2ImageProcessor(patch_size=2, temporal_patch_size=1, merge_size=2).save_pretrained(tmp_path / "mllm")
    projection_dir = tmp_path / "mlp"
    projection_dir.mkdir()
    config = {"use_identity_mlp": True, "text_encoder_norm": False, "img_gen_scales": [2], "connector_norm": False,
              "use_vlm_directvlm_condition": True, "use_learnable_token_condition": True,
              "selected_hidden_states_layers": [1, 2], "diffusion_inner_dim": 24,
              "diffusion_c_input_dim": 8}
    (projection_dir / "config.json").write_text(json.dumps(config))
    save_file(original.projection_state_dict(), projection_dir / "model.safetensors")
    transformer_config = {"dim": 24, "n_layers": 6, "n_refiner_layers": 1, "n_heads": 3,
                          "n_kv_heads": 3, "cap_feat_dim": 8, "in_channels": 16,
                          "axes_dims": [2, 2, 4], "axes_lens": [128, 32, 32],
                          "alignment_padding_mode": "zero_masked", "multi_frame_output": False}
    transformer = create_transformer(transformer_config, device="cpu", dtype=torch.float32)
    from core.models.ming_image_01.residual_unet import native_config, restore_topology, model_topology_metadata
    from core.rewiring.spec import OPERATORS, build_spec
    topology = build_spec("ming_image_01", [{"name": "main", "block_count": 6}], [
        dict(id="r0", stream="main", parent=None, start_block=1, end_block=5, down_factor=[2, 2], **OPERATORS)
    ], native_config(transformer))
    restore_topology(transformer, topology)
    transformer.save_pretrained(tmp_path / "transformer")
    save_file(transformer.state_dict(), tmp_path / "transformer" / "diffusion_pytorch_model.safetensors",
              metadata=model_topology_metadata(transformer))
    vae = AutoencoderKLQwenImage(base_dim=32, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                                input_channels=4, z_dim=16, latents_mean=None, latents_std=None)
    vae.register_to_config(scaling_factor=8.0064, shift_factor=0.0)
    vae.save_pretrained(tmp_path / "vae")
    FlowMatchEulerDiscreteScheduler(shift=6.0).save_pretrained(tmp_path / "scheduler")
    restored = load_components(tmp_path, dtype=torch.float32, conditioner_attention="eager")
    differences = [key for key, value in original.native.state_dict().items()
                   if not torch.equal(value, restored.conditioner.native.state_dict()[key])]
    assert not differences, differences
    original_modes = {name: module.training for name, module in original.named_modules()}
    restored_modes = {name: module.training for name, module in restored.conditioner.named_modules()}
    assert original_modes == restored_modes
    original_rope = original.native.model.model.rotary_emb.inv_freq
    restored_rope = restored.conditioner.native.model.model.rotary_emb.inv_freq
    assert torch.equal(original_rope, restored_rope), (original_rope, restored_rope)
    expected = original(**inputs())
    actual = restored.conditioner(**inputs())
    torch.testing.assert_close(actual.queries, expected.queries, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.direct, expected.direct, rtol=1e-5, atol=1e-6)
    assert restored.conditioner.cacheable
    assert restored.vae.config.scaling_factor == 8.0064
    assert restored.vae.config.shift_factor == 0.0
    assert all(torch.equal(value, restored.conditioner.native.state_dict()[key]) for key, value in original.native.state_dict().items())
    assert restored.artifact.profile == "generation_edit"
    assert restored.transformer.config.n_layers == 6
    assert restored.transformer._residual_unet_spec == topology
    from core.models.ming_image_01.inspection import loaded_model_info
    info = loaded_model_info(restored)
    json.dumps(info, allow_nan=False)
    assert info["total_loaded_parameters"] == sum(p.numel() for m in
        (restored.transformer, restored.conditioner, restored.vae) for p in m.parameters())
    assert info["trainable_parameters"] == 0 and info["conditioner_cacheable"]
    assert info["main_blocks"] == 6 and info["noise_refiner_blocks"] == info["context_refiner_blocks"] == 1
    assert info["component_parameter_counts"]["dit"]["parameters"] == info["dit_rewiring_parameter_counts"]["total"]
    restored.conditioner.set_trainable({"vision", "projections"})
    trainable = loaded_model_info(restored)
    assert trainable["trainable_parameters"] == sum(p.numel() for p in restored.conditioner.parameters() if p.requires_grad)
    assert trainable["component_parameter_counts"]["vision"]["trainable_parameters"] > 0
    assert trainable["component_parameter_counts"]["language"]["trainable_parameters"] == 0
    assert not trainable["conditioner_cacheable"]
    restored.conditioner.set_trainable(set())
    from core.rewiring.inspect import inspect_model
    from core.models.ming_image_01.artifact import write_manifest
    inspected = inspect_model(tmp_path)
    assert inspected["architecture"] == "ming_image_01"
    assert inspected["topology"] == topology
    assert inspected["parameter_counts"]["total"] == sum(p.numel() for p in transformer.parameters())
    manifest = write_manifest(tmp_path)
    assert inspect_model(manifest)["topology"] == topology
    assert all(torch.equal(value, restored.transformer.state_dict()[key]) for key, value in transformer.state_dict().items())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from core.model_loader import ModelLoader
    from core.pipeline import DiffusionPipelineManager
    assert ModelLoader.detect_model_type(str(tmp_path)) == "ming_image_01"
    assert ModelLoader.detect_model_type(str(manifest)) == "ming_image_01"
    manager = DiffusionPipelineManager()
    manager.device = "cpu"
    monkeypatch.setattr(manager, "_save_last_model", lambda *args: None)
    manager.load_model("diffusers", str(tmp_path), force_reload=True)
    assert manager.current_pipeline_kind == "ming_image_01"
    assert manager.current_model_info["profile"] == "generation_edit"
    assert manager.current_model_info["dit_rewiring_spec"] == topology
    assert not manager.current_model_info["is_video"]
    assert manager.current_model_info["latent_channels"] == 16
    assert manager._sushi_wiring.te_out_dim == 8
    from core.models.component_registry import scan_model
    scanned = scan_model(str(tmp_path), "diffusers")
    assert scanned["arch"] == "ming_image_01" and scanned["latent_ndim"] == 5
    assert scanned["is_video"] is False
    assert manager.current_model_info["generation_available"] is True
    loaded_refs = list(manager._iter_loaded_components())
    assert any(value is manager.ming_image_01_components["text_encoder"] for _, value in loaded_refs)
    manager._cleanup_component_architectures()
    assert manager.ming_image_01_components is None and not manager.is_ming_image_01_model
    from core.models.ming_image_01.checkpoint import save_bundle
    destination = tmp_path.parent / (tmp_path.name + "-export")
    exported_manifest = save_bundle(restored, destination, max_shard_bytes=262144)
    assert (destination / "transformer" / "diffusion_pytorch_model.safetensors.index.json").is_file()
    exported = load_components(exported_manifest, dtype=torch.float32, conditioner_attention="eager")
    assert exported.transformer._residual_unet_spec == topology
    for old, new in ((restored.transformer, exported.transformer),
                     (restored.conditioner, exported.conditioner), (restored.vae, exported.vae)):
        assert set(old.state_dict()) == set(new.state_dict())
        assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    output = exported.conditioner(**inputs())
    torch.testing.assert_close(output.queries, actual.queries, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.direct, actual.direct, rtol=1e-5, atol=1e-6)
    with pytest.raises(FileExistsError):
        save_bundle(restored, destination)
    failed = tmp_path.parent / (tmp_path.name + "-cancelled")
    calls = []
    def cancel_after_component():
        calls.append(True)
        return len(calls) > 1
    with pytest.raises(InterruptedError):
        save_bundle(restored, failed, cancelled=cancel_after_component)
    assert not failed.exists()
    assert not list(failed.parent.glob(f".{failed.name}.ming-*"))
    assert not torch.cuda.is_initialized()
