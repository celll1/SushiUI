"""Synthetic Z-Image swap: native/dense/U-Net boundaries and strict artifacts."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from core.models.components.latent_adapter import NormalizedImageVAE
from core.models.components.spatial_swap import install, artifact_metadata, read_declaration
from backend.tests.latent_adapter_contract_test import declaration, identity, ToyCodec


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda, "get_device_capability", return_value=(8,9)), \
         patch.object(torch.cuda, "_lazy_init", return_value=None), \
         patch.object(torch._C, "_cuda_init", return_value=None):
        yield


def model(old, new, rewired=False):
    from core.models.zimage_transformer import ZImageTransformer2DModel
    from core.models.zimage_residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec, OPERATORS
    value = ZImageTransformer2DModel(in_channels=max(old,new), dim=24, n_layers=7,
        n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=12,
        axes_dims=[4,4,4], axes_lens=[128,64,64])
    with torch.no_grad():
        for parameter in value.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0,.02)
    install(value, declaration(old,new))
    pair = value.vae_latent_bridge
    identity(pair.input, .2); identity(pair.output, .3)
    if rewired:
        restore_topology(value, build_spec("zimage", [{"name":"main","block_count":7}], [
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)
        ], native_config(value)))
    return value


@pytest.mark.parametrize("old,new", [(4,8),(8,4)])
@pytest.mark.parametrize("rewired", [False,True])
def test_training_bridge_matches_explicit_noise_phase_and_preserves_grid(old,new,rewired):
    from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
    value = model(old,new,rewired)
    wrapper = BatchedZImageWrapperOptimized(value)
    x = torch.randn(2,new,1,16,24,requires_grad=True)
    sigma = torch.tensor([.2,.7])
    caption = torch.randn(2,3,12)
    mask = torch.ones(2,3,dtype=torch.bool)
    actual = wrapper(x,sigma,caption,mask)[0]
    assert actual.shape == x.shape
    pair = value.vae_latent_bridge
    del value.vae_latent_bridge
    try:
        native = wrapper(pair.input.flow_state(x,sigma),sigma,caption,mask)[0]
        expected = pair.output.flow_velocity(native,convention="x0_minus_eps")
    finally:
        value.add_module("vae_latent_bridge",pair)
    torch.testing.assert_close(actual,expected)
    actual.square().mean().backward()
    assert pair.input.projection.weight.grad.abs().sum() > 0
    assert pair.output.projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_inference_uses_reversed_native_time_and_strict_checkpoint_restores(tmp_path):
    from core.model_loader import ModelLoader
    from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
    from core.models.zimage_residual_unet import native_config
    from core.models.common.single_file_format import read_state_dict
    from core.training.adapters.zimage_adapter import ZImageFullParameterAdapter
    value = model(8,4,True)
    x = torch.randn(1,4,1,16,24)
    sigma = torch.tensor([.25]); time = 1-sigma
    caption = [torch.randn(3,12)]
    expected = torch.stack(value(list(x.unbind()),time,caption)[0])
    explicit = torch.stack(value(list(x.unbind()),time,caption,alignment_sigma=sigma)[0])
    torch.testing.assert_close(expected,explicit)
    codec = NormalizedImageVAE(ToyCodec(4), declaration(8,4))
    trainer = SimpleNamespace(transformer=BatchedZImageWrapperOptimized(value),train_unet=True,
        train_text_encoder=False,text_encoder=None,vae=codec,config={},bundle_vae=True)
    path = tmp_path/"aligned.safetensors"
    with patch("core.training.vae_swap.swap_metadata",return_value=(SimpleNamespace(provenance="synthetic"),True,{})):
        ZImageFullParameterAdapter(trainer).write_checkpoint(3,0,path)
    raw, md = read_state_dict(str(path))
    assert "vae.weight" in raw and "vae.inner.weight" not in raw
    assert read_declaration(md) == value.vae_latent_bridge.declaration
    state, _, _, layout = ModelLoader._normalize_zimage_state_dict(raw)
    config = native_config(value)
    restored = ModelLoader._build_zimage_transformer_from_state(config,7,8,state,layout,torch.float32,
        path=str(path),alignment_declaration=read_declaration(md))
    actual = torch.stack(restored(list(x.unbind()),time,caption)[0])
    torch.testing.assert_close(actual,expected)
    assert restored.vae_latent_bridge.ready
    broken = dict(state); broken.pop("vae_latent_bridge.input.calibrated")
    with pytest.raises(RuntimeError):
        ModelLoader._build_zimage_transformer_from_state(config,7,8,broken,layout,torch.float32,
            alignment_declaration=read_declaration(md))


@pytest.mark.parametrize("rank", [4,5])
def test_image_codec_normalization_posterior_and_scale(rank):
    codec = NormalizedImageVAE(ToyCodec(rank),declaration(3,3,rank=rank))
    pixels = torch.randn(2,3,8,12)
    dist = codec.encode(pixels).latent_dist
    torch.testing.assert_close(dist.mode(),(pixels-.5)*2)
    torch.testing.assert_close(dist.mean,dist.mode())
    g1,g2 = torch.Generator().manual_seed(72),torch.Generator().manual_seed(72)
    torch.testing.assert_close(dist.sample(g1),dist.sample(g2))
    torch.testing.assert_close(codec.decode(dist.mode()).sample,pixels)
    assert 2**(len(codec.config.block_out_channels)-1) == 16
    assert dict(codec.inner.config)["scaling_factor"] == 2


def test_metadata_refuses_unfitted_bridge():
    value = torch.nn.Module(); install(value,declaration())
    with pytest.raises(ValueError,match="uncalibrated"):
        artifact_metadata(value)
    with pytest.raises(ValueError):
        read_declaration({"component.vae.alignment":"{}"})


def test_real_image_codec_calibration_save_restore_and_resume_keeps_bridge(tmp_path):
    from diffusers import AutoencoderKL
    from PIL import Image
    import numpy as np
    from core.models.common.vae_source import resolve_vae_source, load_declared_latent_io
    from core.models.components.spatial_swap import restored_image_codec
    from core.models.common.single_file_format import read_state_dict
    from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
    from core.training.arch.zimage import ZImageArchHandler
    from core.training.vae_swap import apply_latent_space, validate_latent_io
    from core.training.adapters.zimage_adapter import ZImageFullParameterAdapter
    from core.models.components.wiring import ZIMAGE_WIRING
    from core.model_loader import ModelLoader
    from core.models.zimage_residual_unet import native_config
    def codec(channels, blocks):
        return AutoencoderKL(in_channels=3,out_channels=3,latent_channels=channels,
            down_block_types=("DownEncoderBlock2D",)*blocks,up_block_types=("UpDecoderBlock2D",)*blocks,
            block_out_channels=(4,)*blocks,norm_num_groups=4,scaling_factor=.7,shift_factor=.2)
    source = codec(4,4); target = codec(8,5)
    directory = tmp_path/"target"; target.save_pretrained(directory)
    resolved = resolve_vae_source(f"file:{directory}",arch="zimage")
    value = model(4,4,True); del value.vae_latent_bridge
    wrapper = BatchedZImageWrapperOptimized(value)
    trainer = SimpleNamespace(transformer_original=value,transformer=wrapper,vae=source,
        config={"training_method":"full_finetune","vae_swap_source":resolved.source,
            "vae_swap_calibration_samples":2,"vae_swap_calibration_resolution":128},
        weight_dtype=torch.float32,vae_dtype=torch.float32,device="cpu",train_unet=True,
        train_text_encoder=False,text_encoder=torch.nn.Linear(2,2),text_encoder_2=None,
        wiring=ZIMAGE_WIRING.replace(latent_channels=4),bundle_vae=True,
        is_zimage=True,log_prefix="[SyntheticSwap]",_check_stop_requested=lambda:None)
    trainer.arch = ZImageArchHandler(trainer)
    trainer._main_model_module = lambda:value
    trainer.move_main_model_to_cpu = lambda:value.to("cpu")
    trainer.move_main_model_to_gpu = lambda:value.to("cpu")
    trainer.move_vae_to_cpu = lambda:trainer.vae.to("cpu")
    events=[]; trainer._vae_alignment_progress_callback=lambda **kwargs:events.append(kwargs)
    report = trainer.arch.apply_vae_swap(trainer,resolved)
    assert report.new_channels == 8 and trainer.wiring.vae_scale_factor == 16
    from core.training.base_trainer import BaseTrainer
    assert BaseTrainer._arch_pixel_align(trainer) == 128
    assert wrapper.in_channels == wrapper.out_channels == 8
    images=[]
    for i in range(2):
        pixels = (np.random.default_rng(i).uniform(0,255,(128,128,3))).astype(np.uint8)
        path = tmp_path/f"synthetic{i}.png"; Image.fromarray(pixels).save(path)
        images.append({"image_path":str(path)})
    trainer.arch.calibrate_before_training(trainer,[SimpleNamespace(items=images)],None)
    assert trainer.transformer_original.vae_latent_bridge.ready
    assert events[-1]["step"] == events[-1]["total"] == 4
    assert not hasattr(trainer,"_zimage_vae_calibration_source")
    assert validate_latent_io(trainer) == []
    with torch.no_grad():
        value.vae_latent_bridge.input.projection.bias.add_(.1)
    path = tmp_path/"full.safetensors"
    ZImageFullParameterAdapter(trainer).write_checkpoint(9,0,path)
    raw,md = read_state_dict(str(path)); d = read_declaration(md)
    saved = load_declared_latent_io(str(path),arch="zimage")
    assert saved.scale_factor == 16 and saved.config["scaling_factor"] == .7
    assert saved.config["block_out_channels"] == [4]*5
    facade,saved = restored_image_codec(saved.load_module(torch_dtype=torch.float32),saved,d)
    state,_,_,layout = ModelLoader._normalize_zimage_state_dict(raw)
    restored = ModelLoader._build_zimage_transformer_from_state(native_config(value),7,8,state,layout,
        torch.float32,path=str(path),alignment_declaration=d)
    trainer.transformer_original = restored
    trainer.transformer = BatchedZImageWrapperOptimized(restored)
    trainer.vae = facade
    with patch("core.training.vae_swap.apply_configured_vae_swap",side_effect=AssertionError("must not recalibrate")):
        apply_latent_space(trainer,saved)
    torch.testing.assert_close(restored.vae_latent_bridge.input.projection.bias,
        value.vae_latent_bridge.input.projection.bias)
    assert trainer.vae_identity.state_dict is None
    assert trainer.wiring.vae_scale_factor == 16
    assert not torch.cuda.is_initialized()
