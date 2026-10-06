"""Official package shards must load into the training-capable native model."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def no_cuda():
    with patch.object(torch.cuda, "get_device_capability", return_value=(8,9)), \
         patch.object(torch.cuda, "_lazy_init", return_value=None), \
         patch.object(torch._C, "_cuda_init", return_value=None), \
         patch.object(torch.cuda, "is_available", return_value=False):
        yield


@pytest.fixture
def package(tmp_path, no_cuda):
    with patch.object(torch.cuda, "get_device_capability", return_value=(8,9)), \
         patch.object(torch.cuda, "_lazy_init", return_value=None), \
         patch.object(torch._C, "_cuda_init", return_value=None):
        from core.models.zimage_transformer import ZImageTransformer2DModel
        from core.models.zimage_residual_unet import native_config
        from core.model_loader import ModelLoader
    model = ZImageTransformer2DModel(dim=24, n_layers=2, n_refiner_layers=1,
        n_heads=2, n_kv_heads=2, cap_feat_dim=12, axes_dims=[4,4,4], axes_lens=[128,64,64])
    config = dict(native_config(model), _class_name="ZImageTransformer2DModel")
    (tmp_path / "transformer").mkdir()
    for component in ("vae", "text_encoder", "tokenizer", "scheduler"):
        (tmp_path / component).mkdir()
    (tmp_path / "model_index.json").write_text(json.dumps(dict(_class_name="ZImagePipeline",
        transformer=["diffusers", "ZImageTransformer2DModel"])))
    (tmp_path / "transformer/config.json").write_text(json.dumps(config))
    state = model.state_dict()
    keys = list(state); chunks = (keys[::2], keys[1::2]); weight_map = {}
    for i, chunk in enumerate(chunks):
        name = f"weights-{i}.safetensors"
        save_file({key: state[key].clone() for key in chunk}, str(tmp_path / "transformer" / name))
        weight_map.update({key: name for key in chunk})
    (tmp_path / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(dict(metadata={}, weight_map=weight_map)))
    return tmp_path, state, ModelLoader


def load(package, *, channels=16, hidden=12):
    from core.models.zimage_directory import load_directory
    path, _, loader = package
    vae = torch.nn.Linear(1,1); vae.config = SimpleNamespace(latent_channels=channels)
    encoder = torch.nn.Linear(1,1); encoder.config = SimpleNamespace(hidden_size=hidden)
    with patch("diffusers.AutoencoderKL.from_pretrained", return_value=vae) as v, \
         patch("transformers.AutoModel.from_pretrained", return_value=encoder) as te, \
         patch("transformers.AutoTokenizer.from_pretrained", return_value="tokenizer"), \
         patch("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained", return_value="scheduler"):
        result = load_directory(path, device="cpu", torch_dtype=torch.bfloat16,
            build_transformer=loader._build_zimage_transformer_from_state)
        assert v.call_args.kwargs["local_files_only"] and te.call_args.kwargs["local_files_only"]
    return result


def test_sharded_official_weights_and_component_sources(package):
    result = load(package)
    model = result["transformer"]
    assert not model.training and model.n_layers == 2
    assert result["vae_path"] == str(package[0] / "vae")
    assert result["declared_vae"] is None and result["vae_type"] == "flux"
    for key, value in package[1].items():
        assert torch.equal(model.state_dict()[key], value.to(torch.bfloat16))


@pytest.mark.parametrize("channels,hidden,message", [(4,12,"latent channels"),(16,24,"caption feature")])
def test_inconsistent_components_refused(package, channels, hidden, message):
    with pytest.raises(ValueError, match=message):
        load(package, channels=channels, hidden=hidden)


def test_missing_shard_refused(package):
    (package[0] / "transformer/weights-1.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        load(package)


def test_unknown_pipeline_refused_before_component_load(package):
    (package[0] / "model_index.json").write_text('{"_class_name":"OtherPipeline"}')
    with pytest.raises(ValueError, match="official ZImagePipeline"):
        load(package)


def test_saved_foundation_package_preserves_scheduler_without_turbo_fallback(package):
    from diffusers import AutoencoderKL
    from core.models.zimage_directory import validate_component_directory
    root, state, loader = package
    assert validate_component_directory(str(root)) == root
    vae = AutoencoderKL(block_out_channels=(32,), norm_num_groups=8, latent_channels=16)
    vae.save_pretrained(root / "vae")
    (root / "scheduler/scheduler_config.json").write_text(json.dumps(dict(num_train_timesteps=1000,
        shift=6.0, use_dynamic_shifting=False)))
    checkpoint = root / "full.safetensors"
    config = json.loads((root / "transformer/config.json").read_text())
    save_file({"model.diffusion_model."+key:value.clone() for key,value in state.items()}, str(checkpoint),
        metadata=dict(model_type="zimage", transformer_config=json.dumps(config), zimage_base_directory=str(root)))
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Unexpected Turbo fallback")), \
         patch("transformers.AutoModel.from_pretrained", return_value=torch.nn.Linear(1,1)) as te, \
         patch("transformers.AutoTokenizer.from_pretrained", return_value="tokenizer"):
        result = loader.load_zimage_from_comfy_safetensors(str(checkpoint), device="cpu", torch_dtype=torch.bfloat16)
    assert result["scheduler"].config.shift == 6.0
    assert te.call_args.args[0] == str(root / "text_encoder")
    assert result["base_component_directory"] == str(root)


def test_saved_component_directory_must_exist(package):
    from core.models.zimage_directory import validate_component_directory
    (package[0] / "text_encoder").rmdir()
    with pytest.raises(ValueError, match="missing text_encoder"):
        validate_component_directory(str(package[0]))


def test_component_binding_keeps_provenance_on_next_full_save(package, tmp_path):
    from core.training.ops.zimage_ops import bind_components
    from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
    from core.training.adapters.zimage_adapter import ZImageFullParameterAdapter
    from core.models.common.single_file_format import read_metadata
    components = load(package)
    trainer = SimpleNamespace(train_unet=True, train_text_encoder=False, config={}, bundle_vae=False)
    bind_components(trainer, components)
    assert trainer._zimage_base_component_directory == str(package[0])
    trainer.transformer = BatchedZImageWrapperOptimized(trainer.transformer_original)
    with patch("core.training.vae_swap.swap_metadata", return_value=(None, False, {})):
        checkpoint = ZImageFullParameterAdapter(trainer).write_checkpoint(2,0,tmp_path/'resaved.safetensors')
    assert read_metadata(str(checkpoint))["zimage_base_directory"] == str(package[0])
