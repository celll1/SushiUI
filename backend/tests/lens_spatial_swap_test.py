"""Lens target-native geometry, sigma algebra and codec checkpoint contracts."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from core.models.components.spatial_swap import install,map_packed_image
from backend.tests.latent_adapter_contract_test import declaration,identity


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
         patch.object(torch.cuda,"_lazy_init",return_value=None), \
         patch.object(torch._C,"_cuda_init",return_value=None):
        yield


def model(channels,rewired=False):
    from core.models.lens.vendor.transformer import LensTransformer2DModel
    from core.models.lens.residual_unet import native_config,restore_topology
    from core.rewiring.spec import build_spec,OPERATORS
    value=LensTransformer2DModel(in_channels=channels*4,out_channels=channels,num_layers=7,
        attention_head_dim=12,num_attention_heads=2,enc_hidden_dim=12,
        axes_dims_rope=(4,4,4),selected_layer_index=(5,11))
    if rewired:
        restore_topology(value,build_spec("lens",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)
        ],native_config(dict(value.config))))
    return value


@pytest.mark.parametrize("old,new",[(4,8),(8,4)])
@pytest.mark.parametrize("rewired",[False,True])
@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_dense_and_unet_keep_target_grid_and_actual_sigma(old,new,rewired,dtype):
    value=model(max(old,new),rewired)
    install(value,declaration(old,new))
    pair=value.vae_latent_bridge
    identity(pair.input,.2); identity(pair.output,.3)
    value.to(dtype)
    x=torch.randn(2,96,new*4,dtype=dtype,requires_grad=True)
    features=[torch.randn(2,5,12,dtype=dtype) for _ in range(2)]
    mask=torch.ones(2,5,dtype=torch.bool)
    times=torch.tensor([200.,800.],dtype=dtype)
    actual=value(x,features,mask,times,[(1,8,12)])
    assert actual.shape==x.shape
    native=value._forward_native(map_packed_image(pair,x,(8,12),sigma=times/1000.),
        features,mask,times,[(1,8,12)])
    torch.testing.assert_close(actual,map_packed_image(pair,native,(8,12),output=True))
    actual.float().square().mean().backward()
    for grad in (x.grad,pair.input.projection.weight.grad,pair.output.projection.weight.grad,value.img_in.weight.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
    with pytest.raises(ValueError,match="grid"):
        value(x,features,mask,times,[(1,8,11)])


@pytest.mark.parametrize("sharded",[False,True])
def test_batchnorm_teacher_calibration_codec_loader_and_resume(tmp_path,sharded):
    from diffusers import AutoencoderKL,AutoencoderKLFlux2
    from PIL import Image
    import numpy as np
    from core.models.common.vae_source import resolve_vae_source
    from core.models.lens import lens_loader as loader
    from core.models.lens.lens_pipeline_ops import vae_encode,vae_decode,prepare_latents,token_stride
    from core.models.components.wiring import LENS_WIRING
    from core.training.arch.lens import LensArchHandler
    from core.training.vae_swap import apply_latent_space,validate_latent_io
    from core.training.adapters.lens_adapter import LensFullParameterAdapter
    from core.training.base_trainer import BaseTrainer
    old=AutoencoderKLFlux2(latent_channels=4,block_out_channels=(4,)*4,norm_num_groups=4,
        layers_per_block=1).eval()
    with torch.no_grad():
        old.bn.running_mean.copy_(torch.linspace(-.3,.3,16))
        old.bn.running_var.copy_(torch.linspace(.7,1.3,16))
    target=AutoencoderKL(latent_channels=8,block_out_channels=(4,)*5,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*5,up_block_types=("UpDecoderBlock2D",)*5,
        scaling_factor=.7,shift_factor=.2)
    directory=tmp_path/"target"; target.save_pretrained(directory)
    resolved=resolve_vae_source(f"file:{directory}",arch="lens")
    value=model(4,sharded)
    trainer=SimpleNamespace(transformer=value,transformer_original=value,vae=old,
        config={"training_method":"full_finetune","vae_swap_source":resolved.source,
            "vae_swap_calibration_samples":2,"vae_swap_calibration_resolution":128},
        weight_dtype=torch.float32,vae_dtype=torch.float32,device="cpu",train_unet=True,
        train_text_encoder=False,text_encoder=None,text_encoder_2=None,
        wiring=LENS_WIRING.replace(latent_channels=4),bundle_vae=True,is_lens=True,
        model_path=str(tmp_path/"base"),lens_base_dir=str(tmp_path/"base"),
        _check_stop_requested=lambda:None,log_prefix="[SyntheticLens]")
    trainer.arch=LensArchHandler(trainer)
    trainer._main_model_module=lambda:value
    trainer.move_main_model_to_cpu=lambda:value.to("cpu")
    trainer.move_main_model_to_gpu=lambda:value.to("cpu")
    trainer.move_vae_to_cpu=lambda:trainer.vae.to("cpu")
    trainer.move_text_encoder_to_cpu=lambda:None
    trainer.arch.apply_vae_swap(trainer,resolved)
    assert trainer.wiring.vae_scale_factor==16 and trainer.wiring.vae_norm=="identity"
    assert BaseTrainer._arch_pixel_align(trainer)==(128 if sharded else 32)
    images=[]
    for i in range(2):
        pixels=np.random.default_rng(i).integers(0,256,(128,128,3),dtype=np.uint8)
        path=tmp_path/f"synthetic{i}.png"; Image.fromarray(pixels).save(path)
        images.append({"image_path":str(path)})
    trainer.arch.calibrate_before_training(trainer,[SimpleNamespace(items=images)],None)
    assert value.vae_latent_bridge.ready and validate_latent_io(trainer)==[]
    image=Image.new("RGB",(128,192),(80,140,40))
    latent=vae_encode(trainer.vae,image,192,128,"cpu",torch.float32)
    assert latent.shape==(1,24,32) and token_stride(trainer.vae)==32
    assert vae_decode(trainer.vae,latent,6,4).size==(128,192)
    assert prepare_latents(192,128,torch.float32,"cpu",channels=8,vae=trainer.vae).shape==latent.shape
    path=tmp_path/"full.safetensors"
    if sharded:
        from core.models.common.single_file_format import save_single_file_state
        with patch("core.models.common.single_file_format.save_single_file_state",
                side_effect=lambda state,metadata,path:save_single_file_state(state,metadata,path,max_shard_bytes=32768)):
            path=LensFullParameterAdapter(trainer).write_checkpoint(7,0,path)
        assert str(path).endswith(".index.json")
    else:
        path=LensFullParameterAdapter(trainer).write_checkpoint(7,0,path)
    with patch.object(loader,"_resolve_lens_base_dir",return_value=str(tmp_path/"base")), \
         patch.object(loader,"load_lens_components",side_effect=lambda **kw:{"transformer":kw["transformer_override"]}):
        components=loader.load_lens_single_file(str(path),torch_dtype=torch.float32)
    restored=components["transformer"]
    assert restored.vae_latent_bridge.ready and components["vae"].spatial_scale_factor==16
    assert components["vae"].inner.config.scaling_factor==.7
    trainer.transformer=trainer.transformer_original=restored
    trainer.vae=components["vae"]
    with patch("core.training.vae_swap.apply_configured_vae_swap",side_effect=AssertionError("must not recalibrate")):
        apply_latent_space(trainer,components["declared_vae"])
    torch.testing.assert_close(restored.vae_latent_bridge.input.projection.weight,value.vae_latent_bridge.input.projection.weight)
    from core.models.common.single_file_format import read_state_dict
    from safetensors.torch import save_file
    state,metadata=read_state_dict(str(path))
    del state["net.vae_latent_bridge.input.projection.weight"]
    broken=tmp_path/"missing_bridge.safetensors"
    save_file(state,str(broken),metadata={key:str(value) for key,value in metadata.items()})
    with patch.object(loader,"_resolve_lens_base_dir",return_value=str(tmp_path/"base")), \
         pytest.raises(ValueError,match="weights do not match"):
        loader.load_lens_single_file(str(broken),torch_dtype=torch.float32)
    assert not torch.cuda.is_initialized()
