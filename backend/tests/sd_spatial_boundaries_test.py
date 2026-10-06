"""VP/flow image boundaries around real SD U-Nets without codec-grid resizing."""
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


@pytest.mark.parametrize("xl",[False,True])
@pytest.mark.parametrize("old,new",[(4,8),(8,4)])
@pytest.mark.parametrize("noise_process,target",[("ddpm","epsilon"),("ddpm","velocity"),("ddpm","sample"),
    ("flow","epsilon"),("flow","velocity"),("flow","sample")])
def test_prediction_coefficients_native_conditioning_and_gradients(xl,old,new,noise_process,target):
    from diffusers import UNet2DConditionModel,DDPMScheduler
    from core.models.components.sd_vae_alignment import training_coefficients,map_input,map_prediction
    options=dict(addition_embed_type="text_time",addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=32) if xl else {}
    model=UNet2DConditionModel(in_channels=max(old,new),out_channels=max(old,new),
        block_out_channels=(8,16),norm_num_groups=4,layers_per_block=1,cross_attention_dim=8,
        down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","CrossAttnUpBlock2D"),**options)
    install(model,declaration(old,new))
    pair=model.vae_latent_bridge
    pair.initialize_from_pair(identity(SpatialLatentBridge(new,old),.2),identity(SpatialLatentBridge(old,new),.3))
    scheduler=DDPMScheduler(num_train_timesteps=20)
    t=torch.tensor([2,17]) if noise_process=="ddpm" else torch.tensor([.2,.8])
    x=torch.randn(2,new,8,12,requires_grad=True)
    alpha,sigma=training_coefficients(model,scheduler,t,noise_process=noise_process,reference=x)
    expected_alpha=scheduler.alphas_cumprod[t].sqrt() if noise_process=="ddpm" else 1-t
    expected_sigma=(1-scheduler.alphas_cumprod[t]).sqrt() if noise_process=="ddpm" else t
    torch.testing.assert_close(alpha,expected_alpha)
    torch.testing.assert_close(sigma,expected_sigma)
    cached=training_coefficients(model,scheduler,t,noise_process=noise_process,reference=x,
        alphas_cumprod=scheduler.alphas_cumprod)
    torch.testing.assert_close(cached,(alpha,sigma))
    incoming=map_input(model,x,alpha=alpha)
    assert incoming.shape==(2,max(old,new),8,12)
    expected=pair.input._project(x,bias=False)+alpha[:,None,None,None]*pair.input._bias_like(x)
    torch.testing.assert_close(incoming,expected)
    cond={"text_embeds":torch.randn(2,8),"time_ids":torch.randn(2,6)} if xl else None
    native=model(incoming,t,torch.randn(2,3,8),added_cond_kwargs=cond).sample
    result=map_prediction(model,native,noise_process=noise_process,prediction_target=target,sigma=sigma)
    coefficient={"epsilon":0.,"sample":1.,"velocity":-sigma[:,None,None,None] if noise_process=="ddpm" else -1.}[target]
    expected=pair.output._project(native,bias=False)+coefficient*pair.output._bias_like(native)
    torch.testing.assert_close(result,expected)
    assert result.shape==x.shape
    result.square().mean().backward()
    for bridge in (pair.input,pair.output):
        assert torch.isfinite(bridge.projection.weight.grad).all()
        assert bridge.projection.weight.grad.abs().sum()>0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("arch",["sd15","sdxl"])
@pytest.mark.parametrize("old,new",[(4,8),(8,4)])
def test_native_codec_calibration_coarse_grid_and_training(tmp_path,arch,old,new):
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL,UNet2DConditionModel
    from core.models.components.sd_vae_alignment import initialize_swap
    from core.models.components.vae_wrapper import SDXLVAEWrapper
    from core.models.common.vae_source import resolve_vae_source
    from core.training.arch import get_arch_handler
    from core.training.ops.sd_sdxl_ops import vae_encode,train_step
    from core.training.vae_swap import validate_latent_io
    from core.training.base_trainer import BaseTrainer
    from backend.tests.repa_sd_unet_tap_test import _tiny_unet,_train_step_trainer,CROSS_DIM,POOLED_DIM
    model=UNet2DConditionModel.from_config(_tiny_unet(arch).config,in_channels=old,out_channels=old)
    teacher=AutoencoderKL(latent_channels=old,block_out_channels=(4,)*4,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*4,up_block_types=("UpDecoderBlock2D",)*4,
        scaling_factor=.18)
    target=AutoencoderKL(latent_channels=new,block_out_channels=(4,)*5,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*5,up_block_types=("UpDecoderBlock2D",)*5,
        scaling_factor=.7,shift_factor=.2)
    directory=tmp_path/"target"; target.save_pretrained(directory)
    resolved=resolve_vae_source(f"file:{directory}",arch=arch)
    trainer=_train_step_trainer(arch,model,[])
    if arch=="sdxl":
        with patch.object(AutoencoderKL,"from_pretrained",return_value=teacher):
            trainer.vae=SDXLVAEWrapper(device="cpu",dtype=torch.float32)
    else:trainer.vae=teacher
    trainer.vae_dtype=trainer.weight_dtype=torch.float32
    trainer.text_encoder=trainer.text_encoder_2=None
    trainer.config.update(vae_swap_calibration_samples=2,vae_swap_calibration_resolution=128)
    trainer.arch=get_arch_handler(trainer)
    trainer.wiring=trainer.arch.wiring.replace(latent_channels=old)
    trainer._main_model_module=lambda:model
    trainer.move_main_model_to_cpu=lambda:model.to("cpu")
    trainer.move_main_model_to_gpu=lambda:model.to("cpu")
    trainer.move_vae_to_cpu=lambda:trainer.vae.to("cpu")
    trainer._check_stop_requested=lambda:None
    initialize_swap(trainer,resolved,handler=trainer.arch)
    assert model.config.in_channels==model.config.out_channels==max(old,new)
    assert trainer.wiring.latent_channels==new and trainer.wiring.vae_scale_factor==16
    assert BaseTrainer._arch_pixel_align(trainer)==16
    assert validate_latent_io(trainer)==[]
    images=[]
    for i in range(2):
        path=tmp_path/f"synthetic{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0,256,(128,128,3),dtype=np.uint8)).save(path)
        images.append({"image_path":str(path)})
    trainer.arch.calibrate_before_training(trainer,[SimpleNamespace(items=images)],None)
    assert model.vae_latent_bridge.ready
    with torch.no_grad():latent=vae_encode(trainer,torch.randn(2,3,128,192))
    assert latent.shape==(2,new,8,12)
    assert trainer.vae.decode(latent).sample.shape==(2,3,128,192)
    loss,reported,_=train_step(trainer,latents=latent,text_embeddings=torch.randn(2,5,CROSS_DIM[arch]),
        pooled_embeddings=torch.randn(2,POOLED_DIM) if arch=="sdxl" else None)
    assert torch.isfinite(loss) and reported>=0
    loss.backward()
    assert model.vae_latent_bridge.input.projection.weight.grad.abs().sum()>0
    assert model.vae_latent_bridge.output.projection.weight.grad.abs().sum()>0
    assert not hasattr(trainer,"_sd_vae_calibration_source")
    from core.training.adapters.sd15_adapter import SD15FullParameterAdapter
    from core.training.adapters.sdxl_adapter import SDXLFullParameterAdapter
    from core.models.components.sd_vae_alignment import restore_alignment
    from core.models.components.spatial_swap import read_declaration,restored_image_codec
    from core.models.common.single_file_format import read_state_dict,save_single_file_state
    from core.models.common.vae_source import load_declared_latent_io
    trainer.train_unet=True
    trainer.train_text_encoder=trainer.train_text_encoder_1=trainer.train_text_encoder_2=False
    trainer.bundle_vae=True
    trainer.config["vae_swap_source"]=resolved.source
    adapter=SDXLFullParameterAdapter(trainer) if arch=="sdxl" else SD15FullParameterAdapter(trainer)
    path=adapter.write_checkpoint(7,0,tmp_path/"aligned.safetensors")
    state,metadata=read_state_dict(str(path))
    assert metadata["modelspec.prediction_type"]=="epsilon"
    assert "component.vae.alignment" in metadata and "unet_config" in metadata
    assert not any(key.startswith("vae.inner.") for key in state)
    rebuilt=UNet2DConditionModel.from_config(model.config)
    rebuilt.load_state_dict({key:tensor for key,tensor in model.state_dict().items()
        if not key.startswith("vae_latent_bridge.")},strict=True)
    restore_alignment(rebuilt,metadata,state)
    assert rebuilt.vae_latent_bridge.ready
    for key,tensor in model.vae_latent_bridge.state_dict().items():
        torch.testing.assert_close(rebuilt.vae_latent_bridge.state_dict()[key],tensor,rtol=0,atol=0)
    declared=load_declared_latent_io(str(path),arch=arch)
    codec,_=restored_image_codec(declared.load_module(torch_dtype=torch.float32),declared,read_declaration(metadata))
    assert codec.spatial_scale_factor==16 and codec.config.latent_channels==new
    for key,tensor in trainer.vae.inner.state_dict().items():
        torch.testing.assert_close(codec.inner.state_dict()[key],tensor,rtol=0,atol=0)
    sharded=save_single_file_state(state,metadata,str(tmp_path/"sharded.safetensors"),max_shard_bytes=32768)
    shard_state,shard_md=read_state_dict(str(sharded))
    restore_alignment(rebuilt,shard_md,shard_state)
    assert rebuilt.vae_latent_bridge.ready
    del shard_state["model.diffusion_model.vae_latent_bridge.input.projection.weight"]
    with pytest.raises(RuntimeError,match="Missing key"):
        restore_alignment(rebuilt,shard_md,shard_state)
    with patch("core.training.vae_swap.swap_metadata",return_value=(None,False,{})), \
         pytest.raises(ValueError,match="codec declaration"):
        adapter.write_checkpoint(8,0,tmp_path/"no_codec.safetensors")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("arch",["sd15","sdxl"])
@pytest.mark.parametrize("mixed",[False,True])
@pytest.mark.parametrize("process",["ddpm","flow"])
def test_real_train_step_preserves_external_loss_space_and_autocast(arch,mixed,process):
    from diffusers import UNet2DConditionModel
    from backend.tests.repa_sd_unet_tap_test import _tiny_unet,_train_step_trainer,_train_step
    model=UNet2DConditionModel.from_config(_tiny_unet(arch).config,in_channels=8,out_channels=8)
    install(model,declaration(8,4))
    pair=model.vae_latent_bridge
    pair.initialize_from_pair(identity(SpatialLatentBridge(4,8),.2),identity(SpatialLatentBridge(8,4),.3))
    trainer=_train_step_trainer(arch,model,[])
    trainer.mixed_precision=mixed
    trainer.training_dtype=torch.bfloat16 if mixed else torch.float32
    trainer.noise_process=process
    trainer.prediction_target="velocity"
    loss,reported,_=_train_step(trainer,arch,repa_pixels=None)
    assert torch.isfinite(loss) and reported>=0
    loss.backward()
    for bridge in (pair.input,pair.output):
        assert torch.isfinite(bridge.projection.weight.grad).all()
        assert bridge.projection.weight.grad.abs().sum()>0
    assert not torch.cuda.is_initialized()


def test_identical_sdxl_codec_keeps_existing_wrapper(tmp_path):
    from types import SimpleNamespace
    from diffusers import AutoencoderKL
    from core.models.components.sd_vae_alignment import initialize_swap
    from core.models.components.vae_wrapper import SDXLVAEWrapper
    from core.models.common.vae_source import resolve_vae_source
    from core.training.arch.sdxl import SDXLArchHandler
    from backend.tests.repa_sd_unet_tap_test import _tiny_unet
    native=AutoencoderKL(latent_channels=4,block_out_channels=(4,)*4,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*4,up_block_types=("UpDecoderBlock2D",)*4)
    directory=tmp_path/"same"; native.save_pretrained(directory)
    resolved=resolve_vae_source(f"file:{directory}",arch="sdxl")
    with patch.object(AutoencoderKL,"from_pretrained",return_value=native):
        wrapper=SDXLVAEWrapper(device="cpu",dtype=torch.float32)
    trainer=SimpleNamespace(unet=_tiny_unet("sdxl"),vae=wrapper,vae_dtype=torch.float32,
        weight_dtype=torch.float32,config={})
    trainer.arch=SDXLArchHandler(trainer)
    trainer.wiring=trainer.arch.wiring
    conv=trainer.unet.conv_in
    report=initialize_swap(trainer,resolved,handler=trainer.arch)
    assert report.replaced==() and trainer.unet.conv_in is conv
    assert trainer.vae is wrapper
    assert not hasattr(trainer.unet,"vae_latent_bridge")
    assert not hasattr(trainer,"_sd_vae_calibration_source")
