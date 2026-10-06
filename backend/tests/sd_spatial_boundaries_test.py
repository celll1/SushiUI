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
