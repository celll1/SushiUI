"""Synthetic Anima codec boundaries across native and residual execution."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from core.models.components.spatial_swap import install
from core.models.components.vae_alignment import SpatialLatentBridge
from backend.tests.latent_adapter_contract_test import declaration,identity


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
         patch.object(torch.cuda,"_lazy_init",return_value=None), \
         patch.object(torch._C,"_cuda_init",return_value=None):
        yield


@pytest.mark.parametrize("rewired",[False,True])
@pytest.mark.parametrize("old,new",[(4,8),(8,4)])
def test_native_grid_sigma_mask_and_gradients(old,new,rewired):
    from core.models.anima.anima_models import Anima
    from core.models.anima.residual_unet import native_config,restore_topology
    from core.rewiring.spec import build_spec,OPERATORS
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import ANIMA_LATENT_IO
    model=Anima(max_img_h=64,max_img_w=64,max_frames=8,in_channels=old,out_channels=old,
        model_channels=64,num_blocks=3,num_heads=4,crossattn_emb_channels=32,
        adaln_lora_dim=16,use_llm_adapter=False)
    resize_latent_io(model,ANIMA_LATENT_IO,max(old,new))
    if rewired:
        restore_topology(model,build_spec("anima",[{"name":"main","block_count":3}],
            [dict(id="s",stream="main",parent=None,start_block=1,end_block=2,
                down_factor=[2,2],**OPERATORS)],native_config(model)))
    original=deepcopy(model)
    install(model,declaration(old,new))
    pair=model.vae_latent_bridge
    pair.initialize_from_pair(identity(SpatialLatentBridge(new,old),.2),
        identity(SpatialLatentBridge(old,new),.3))
    x=torch.randn(2,new,1,8,12,requires_grad=True)
    sigma=torch.tensor([.2,.8])
    context=torch.randn(2,5,32)
    mask=torch.randint(0,2,(2,1,8,12)).float()
    incoming=pair.input.flow_state(x.squeeze(2),sigma).unsqueeze(2)
    expected=pair.output.flow_velocity(original(incoming,sigma,context,padding_mask=mask).squeeze(2),
        convention="eps_minus_x0").unsqueeze(2)
    result=model(x,sigma,context,padding_mask=mask)
    assert result.shape==x.shape
    torch.testing.assert_close(result,expected)
    torch.testing.assert_close(model(x,sigma[:,None],context,padding_mask=mask),result)
    result.square().mean().backward()
    for bridge in (pair.input,pair.output):
        assert bridge.projection.weight.grad is not None
        assert torch.isfinite(bridge.projection.weight.grad).all()
        assert bridge.projection.weight.grad.abs().sum()>0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError,match="single image frame"):
        model(x.expand(-1,-1,2,-1,-1),sigma,context,padding_mask=mask)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("old_c,new_c",[(4,8),(8,4)])
def test_native_teacher_calibration_and_coarse_training_encode(tmp_path,old_c,new_c):
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL,AutoencoderKLQwenImage
    from core.models.anima.anima_models import Anima
    from core.models.anima.vae_alignment import initialize_swap
    from core.models.anima.residual_unet import native_config,restore_topology
    from core.models.common.vae_source import resolve_vae_source
    from core.models.components.wiring import ANIMA_WIRING
    from core.training.arch.anima import AnimaArchHandler
    from core.training.ops.anima_ops import vae_encode
    from core.training.vae_swap import validate_latent_io
    from core.training.base_trainer import BaseTrainer
    from core.rewiring.spec import build_spec,OPERATORS
    teacher=AutoencoderKLQwenImage(base_dim=4,z_dim=old_c,dim_mult=[1,1,1,1],num_res_blocks=1,
        temperal_downsample=[False,False,False],latents_mean=[0.]*old_c,latents_std=[1.]*old_c)
    target=AutoencoderKL(latent_channels=new_c,block_out_channels=(4,)*5,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*5,up_block_types=("UpDecoderBlock2D",)*5,
        scaling_factor=.7,shift_factor=.2)
    directory=tmp_path/"target"; target.save_pretrained(directory)
    resolved=resolve_vae_source(f"file:{directory}",arch="anima")
    value=Anima(max_img_h=64,max_img_w=64,max_frames=8,in_channels=old_c,out_channels=old_c,
        model_channels=64,num_blocks=3,num_heads=4,crossattn_emb_channels=32,
        adaln_lora_dim=16,use_llm_adapter=False)
    restore_topology(value,build_spec("anima",[{"name":"main","block_count":3}],
        [dict(id="s",stream="main",parent=None,start_block=1,end_block=2,
            down_factor=[2,2],**OPERATORS)],native_config(value)))
    trainer=SimpleNamespace(transformer=value,vae=teacher,
        config={"training_method":"full_finetune","vae_swap_calibration_samples":2,
            "vae_swap_calibration_resolution":128},device="cpu",weight_dtype=torch.float32,
        vae_dtype=torch.float32,wiring=ANIMA_WIRING.replace(latent_channels=old_c),
        text_encoder=None,text_encoder_2=None,log_prefix="[SyntheticAnima]",is_anima=True,
        _check_stop_requested=lambda:None)
    trainer.arch=AnimaArchHandler(trainer)
    trainer._main_model_module=lambda:value
    trainer.move_main_model_to_cpu=lambda:value.to("cpu")
    trainer.move_main_model_to_gpu=lambda:value.to("cpu")
    trainer.move_vae_to_cpu=lambda:trainer.vae.to("cpu")
    initialize_swap(trainer,resolved,handler=trainer.arch)
    assert trainer.wiring.vae_scale_factor==16 and trainer.wiring.latent_channels==new_c
    assert BaseTrainer._arch_pixel_align(trainer)==64
    assert value._residual_unet_spec["regions"][0]["id"]=="s"
    assert validate_latent_io(trainer)==[]
    images=[]
    for i in range(2):
        path=tmp_path/f"synthetic{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0,256,(128,128,3),dtype=np.uint8)).save(path)
        images.append({"image_path":str(path)})
    trainer.arch.calibrate_before_training(trainer,[SimpleNamespace(items=images)],None)
    assert value.vae_latent_bridge.ready
    latent=vae_encode(trainer,torch.randn(1,3,128,192))
    assert latent.shape==(1,new_c,8,12)
    assert trainer.vae.decode(latent).sample.shape==(1,3,128,192)
    output=value(latent.unsqueeze(2),torch.tensor([.6]),torch.randn(1,5,32),
        padding_mask=torch.zeros(1,1,8,12))
    assert output.shape==(1,new_c,1,8,12)
    output.square().mean().backward()
    assert value.vae_latent_bridge.input.projection.weight.grad.abs().sum()>0
    assert not hasattr(trainer,"_anima_vae_calibration_source")
    assert not torch.cuda.is_initialized()
