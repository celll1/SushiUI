"""Real samplers and U-Nets retain external codec geometry through sampling."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from core.inference.sd_codec_boundary import coefficients,sampling_boundary,aligned_sampling
from core.models.components.spatial_swap import install
from core.models.components.vae_alignment import SpatialLatentBridge
from backend.tests.latent_adapter_contract_test import declaration,identity


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
         patch.object(torch.cuda,"_lazy_init",return_value=None), \
         patch.object(torch._C,"_cuda_init",return_value=None):
        yield


@pytest.mark.parametrize("name",["DDIMScheduler","DDPMScheduler","PNDMScheduler",
    "EulerDiscreteScheduler","EulerAncestralDiscreteScheduler","LMSDiscreteScheduler",
    "HeunDiscreteScheduler","KDPM2DiscreteScheduler","KDPM2AncestralDiscreteScheduler",
    "DPMSolverMultistepScheduler","DPMSolverSinglestepScheduler","UniPCMultistepScheduler"])
@pytest.mark.parametrize("prediction",["epsilon","v_prediction"])
def test_real_sampler_forward_external_grid_and_cleanup(name,prediction):
    import diffusers
    from diffusers import UNet2DConditionModel
    model=UNet2DConditionModel(in_channels=8,out_channels=8,block_out_channels=(8,16),
        norm_num_groups=4,layers_per_block=1,cross_attention_dim=8,
        down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","CrossAttnUpBlock2D"))
    install(model,declaration(8,4))
    pair=model.vae_latent_bridge
    pair.initialize_from_pair(identity(SpatialLatentBridge(4,8),.2),identity(SpatialLatentBridge(8,4),.3))
    cls=getattr(diffusers,name)
    options={"prediction_type":prediction,"num_train_timesteps":100}
    if name=="PNDMScheduler":options["skip_prk_steps"]=True
    if "use_karras_sigmas" in __import__("inspect").signature(cls.__init__).parameters:
        options["use_karras_sigmas"]=True
    scheduler=cls(**options)
    scheduler.set_timesteps(4)
    x=torch.randn(1,4,8,12)*scheduler.init_noise_sigma
    cond=torch.randn(1,3,8)
    hooks=(len(model._forward_pre_hooks),len(model._forward_hooks))
    with torch.no_grad(),sampling_boundary(model,scheduler):
        for t in scheduler.timesteps:
            incoming=scheduler.scale_model_input(x,t)
            alpha,sigma,_=coefficients(scheduler,t,incoming)
            if not name.startswith("KDPM2") or scheduler.state_in_first_order:
                clean=scheduler.add_noise(torch.ones_like(x),torch.zeros_like(x),t.reshape(1))
                clean=scheduler.scale_model_input(clean,t)
                torch.testing.assert_close(clean,torch.ones_like(x)*alpha,rtol=1e-4,atol=1e-5)
            # Observe the actual tensor reaching conv_in, beyond the pre-hook.
            seen=[]
            handle=model.conv_in.register_forward_pre_hook(lambda module,args:seen.append(args[0].clone()))
            try:pred=model(incoming,t,cond).sample
            finally:handle.remove()
            expected=pair.input._project(incoming,bias=False)+alpha*pair.input._bias_like(incoming)
            torch.testing.assert_close(seen[0],expected)
            assert pred.shape==x.shape
            # Run the same native model outside the boundary to check output algebra.
            with sampling_boundary_disabled(model):
                native=model(expected,t,cond).sample
            expected_pred=pair.output._project(native,bias=False)
            if prediction=="v_prediction":expected_pred=expected_pred-sigma*pair.output._bias_like(native)
            torch.testing.assert_close(pred,expected_pred)
            x=scheduler.step(pred,t,x).prev_sample
            assert torch.isfinite(x).all()
    assert hooks==(len(model._forward_pre_hooks),len(model._forward_hooks))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("name",["FlowMatchEulerDiscreteScheduler","DPMSolverMultistepScheduler","UniPCMultistepScheduler"])
def test_flow_sigma_schedule_and_prediction_bias(name):
    import diffusers
    from diffusers import UNet2DConditionModel
    options={} if name=="FlowMatchEulerDiscreteScheduler" else {"use_flow_sigmas":True,"prediction_type":"flow_prediction"}
    scheduler=getattr(diffusers,name)(**options)
    scheduler.set_timesteps(4)
    model=UNet2DConditionModel(in_channels=8,out_channels=8,block_out_channels=(8,),
        norm_num_groups=4,layers_per_block=1,cross_attention_dim=8,
        down_block_types=("DownBlock2D",),up_block_types=("UpBlock2D",))
    install(model,declaration(8,4))
    pair=model.vae_latent_bridge
    pair.initialize_from_pair(identity(SpatialLatentBridge(4,8),.2),identity(SpatialLatentBridge(8,4),.3))
    x=torch.randn(1,4,8,12)
    cond=torch.randn(1,3,8)
    with torch.no_grad(),sampling_boundary(model,scheduler):
        for index,t in enumerate(scheduler.timesteps):
            alpha,sigma,process=coefficients(scheduler,t,x)
            assert process=="flow"
            torch.testing.assert_close(alpha+sigma,torch.ones_like(sigma))
            torch.testing.assert_close(sigma,scheduler.sigmas[index])
            incoming=pair.input._project(x,bias=False)+alpha*pair.input._bias_like(x)
            with sampling_boundary_disabled(model):native=model(incoming,t,cond).sample
            prediction=model(x,t,cond,return_dict=False)[0]
            torch.testing.assert_close(prediction,pair.output._project(native,bias=False)-pair.output._bias_like(native))
            x=scheduler.step(prediction,t,x).prev_sample
            assert torch.isfinite(x).all()
    assert not model._forward_pre_hooks and not model._forward_hooks


from contextlib import contextmanager

@contextmanager
def sampling_boundary_disabled(model):
    # Preserve existing hooks while obtaining a native numerical oracle.
    pre,post=model._forward_pre_hooks,model._forward_hooks
    model._forward_pre_hooks={};model._forward_hooks={}
    try:yield
    finally:model._forward_pre_hooks=pre;model._forward_hooks=post


def test_decorator_removes_hooks_after_failure_and_native_noop():
    from diffusers import DDPMScheduler,UNet2DConditionModel
    model=UNet2DConditionModel(in_channels=8,out_channels=8,block_out_channels=(8,),
        norm_num_groups=4,layers_per_block=1,cross_attention_dim=8,
        down_block_types=("DownBlock2D",),up_block_types=("UpBlock2D",))
    pipeline=SimpleNamespace(unet=model,scheduler=DDPMScheduler(num_train_timesteps=100))
    @aligned_sampling
    def fail(pipeline):
        raise RuntimeError("cancelled")
    for aligned in (False,True):
        if aligned:
            install(model,declaration(8,4))
            model.vae_latent_bridge.initialize_from_pair(identity(SpatialLatentBridge(4,8),0),identity(SpatialLatentBridge(8,4),0))
        with pytest.raises(RuntimeError,match="cancelled"):
            fail(pipeline=pipeline)
        assert not model._forward_pre_hooks and not model._forward_hooks


@pytest.mark.parametrize("arch",["sd15","sdxl"])
@pytest.mark.parametrize("mode",["txt2img","img2img","inpaint"])
@pytest.mark.parametrize("gaussian",[False,True])
@pytest.mark.parametrize("sampler",["EulerDiscreteScheduler","EulerAncestralDiscreteScheduler"])
def test_public_sampling_loops_use_external_channels_and_native_codec_grid(arch,mode,gaussian,sampler,feature="base"):
    import numpy as np
    from PIL import Image
    import diffusers
    from diffusers import AutoencoderKL,UNet2DConditionModel
    from core.models.components.gaussian_alignment import POLICY
    from backend.tests.repa_sd_unet_tap_test import _tiny_unet,CROSS_DIM,POOLED_DIM
    from core.models.components.latent_adapter import NormalizedImageVAE
    from core.inference import custom_sampling as sampling
    model=UNet2DConditionModel.from_config(_tiny_unet(arch).config,in_channels=8,out_channels=8)
    old,new=(4,8) if gaussian else (8,4)
    decl=declaration(old,new)
    if gaussian:decl=dict(decl,noise_policy=POLICY)
    install(model,decl)
    model.vae_latent_bridge.initialize_from_pair(identity(SpatialLatentBridge(new,old),.2),identity(SpatialLatentBridge(old,new),.3))
    vae=AutoencoderKL(latent_channels=new,block_out_channels=(4,)*5,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*5,up_block_types=("UpDecoderBlock2D",)*5)
    codec=NormalizedImageVAE(vae,decl)
    pipeline=SimpleNamespace(unet=model,vae=codec,vae_scale_factor=8,
        scheduler=getattr(diffusers,sampler)(num_train_timesteps=100),
        text_encoder_2=object() if arch=="sdxl" else None)
    observed=[]
    def progress(step,total,latents,**kwargs):observed.append(tuple(latents.shape))
    kwargs=dict(pipeline=pipeline,prompt_embeds=torch.randn(1,3,CROSS_DIM[arch]),
        negative_prompt_embeds=torch.randn(1,3,CROSS_DIM[arch]),num_inference_steps=2,
        width=192,height=128,guidance_scale=2.,generator=torch.Generator().manual_seed(13),
        progress_callback=progress)
    if arch=="sdxl":
        kwargs.update(pooled_prompt_embeds=torch.randn(1,POOLED_DIM),
            negative_pooled_prompt_embeds=torch.randn(1,POOLED_DIM))
    if mode!="txt2img":
        kwargs["init_image"]=Image.fromarray(np.zeros((128,192,3),dtype=np.uint8))
        kwargs["strength"]=1.
    if mode=="inpaint":
        kwargs["mask_image"]=Image.new("L",(192,128),255)
        if gaussian:kwargs["mask_image"].paste(0,(0,0,96,128))
    if feature in ("style","multi_style"):
        from core.inference.reference_style import StyleTransferConfig
        refs=[(StyleTransferConfig(block_range=(0,100)),torch.randn(1,new,8,12),
            torch.randn(1,new,8,12)) for _ in range(2 if feature=="multi_style" else 1)]
        if feature=="style":
            kwargs.update(style_cfg=refs[0][0],style_ref_x0=refs[0][1],style_eps_ref=refs[0][2])
        else:kwargs["style_refs"]=refs
    elif feature=="nag":
        kwargs.update(nag_enable=True,nag_negative_prompt_embeds=torch.randn_like(kwargs["prompt_embeds"]))
    elif feature=="guide":
        kwargs["ref_guide_configs"]=[dict(image=Image.new("RGB",(192,128),"blue"),strength=.3)]
    function={"txt2img":sampling.custom_sampling_loop,"img2img":sampling.custom_img2img_sampling_loop,
        "inpaint":sampling.custom_inpaint_sampling_loop}[mode]
    with patch("core.vram_optimization.move_vae_to_gpu"), \
         patch("core.vram_optimization.move_vae_to_cpu"), \
         patch("core.vram_optimization.log_device_status"), \
         patch.object(sampling,"_loaded_wiring",return_value=SimpleNamespace(vae_scale_factor=8,latent_channels=8)), \
         patch.object(sampling,"get_inpaint_use_dedicated_model_setting",return_value=False),torch.no_grad():
        image=function(**kwargs)
    assert image.size==(192,128)
    assert observed and all(shape==(1,new,8,12) for shape in observed)
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert not hasattr(model,"_sushi_sd_noise_session")


    assert not torch.cuda.is_initialized()
    def cancelled(*args,**kwargs):raise RuntimeError("cancelled at step")
    kwargs["progress_callback"]=cancelled
    with patch("core.vram_optimization.move_vae_to_gpu"), \
         patch("core.vram_optimization.move_vae_to_cpu"), \
         patch("core.vram_optimization.log_device_status"), \
         patch.object(sampling,"get_inpaint_use_dedicated_model_setting",return_value=False),torch.no_grad(), \
         pytest.raises(RuntimeError,match="cancelled at step"):
        function(**kwargs)
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert not hasattr(model,"_sushi_sd_noise_session")


@pytest.mark.parametrize("arch",["sd15","sdxl"])
@pytest.mark.parametrize("mode",["txt2img","img2img","inpaint"])
@pytest.mark.parametrize("feature",["style","multi_style","nag","guide"])
def test_gaussian_public_advanced_paths(arch,mode,feature):
    test_public_sampling_loops_use_external_channels_and_native_codec_grid(
        arch,mode,True,"EulerAncestralDiscreteScheduler",feature)
