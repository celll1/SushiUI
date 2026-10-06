"""Synthetic MiniT2I coarse-grid boundaries across dense and U-net execution."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from core.models.components.spatial_swap import install
from core.models.components.vae_alignment import SpatialLatentBridge
from backend.tests.latent_adapter_contract_test import declaration, identity


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
         patch.object(torch.cuda, "_lazy_init", return_value=None), \
         patch.object(torch._C, "_cuda_init", return_value=None):
        yield


def model_for(channels, *, patch_size=2, vae_type="sdxl"):
    from core.models.minit2i.vendor.transformer import MiniT2IMMJiTModel
    model = MiniT2IMMJiTModel(image_size=32, patch_size=patch_size, in_channels=channels,
        txt_input_size=16, hidden_size=32, txt_hidden_size=32, cond_vec_size=32,
        depth_double=3, txt_preamble_depth=1, num_heads=2, head_dim=16,
        pca_channels=8, prompt_length=5, vae_type=vae_type, noise_scale=2. if vae_type=="none" else 1.)
    torch.nn.init.normal_(model.model.net.final_layer.linear.weight, std=.02)
    return model


@pytest.mark.parametrize("rewired", [False, True])
@pytest.mark.parametrize("old,new", [(4,8), (8,4)])
@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
def test_x0_timestep_native_grid_gradients_and_restore(old, new, rewired, dtype):
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import MINIT2I_LATENT_IO
    from core.models.minit2i.residual_unet import native_config, restore_topology
    from core.models.minit2i.vae_alignment import latent_geometry, generation_token_stride
    from core.rewiring.spec import build_spec, OPERATORS
    model = model_for(old)
    resize_latent_io(model, MINIT2I_LATENT_IO, max(old,new))
    model.to(dtype)
    if rewired:
        restore_topology(model, build_spec("minit2i", [{"name":"main", "block_count":3}],
            [dict(id="s", stream="main", parent=None, start_block=1, end_block=2,
                down_factor=[2,2], **OPERATORS)], native_config(model)))
    original = deepcopy(model)
    install(model, declaration(old,new))
    pair = model.vae_latent_bridge
    pair.initialize_from_pair(identity(SpatialLatentBridge(new,old), .2),
        identity(SpatialLatentBridge(old,new), .3))
    pair.to(dtype)
    image = torch.randn(2,new,8,12, dtype=dtype, requires_grad=True)
    t = torch.tensor([.0, .8],dtype=dtype)
    context, mask = torch.randn(2,5,16,dtype=dtype), torch.ones(2,5)
    incoming = pair.input.diffusion_state(image, alpha=t)
    torch.testing.assert_close(incoming[0], pair.input.epsilon(image)[0])
    torch.testing.assert_close(incoming[1], pair.input.epsilon(image)[1] +
        .8 * pair.input.projection.bias[:,None,None])
    expected = pair.output(original(incoming,t,context,mask))
    grids = []
    hook = model.model.net.img_embedder.register_forward_hook(
        lambda module, args, out: grids.append(out[1:]))
    try:
        result = model(image,t,context,mask)
    finally:
        hook.remove()
    assert grids == [(4,6)]
    assert result.shape == image.shape
    torch.testing.assert_close(result,expected)
    torch.testing.assert_close(model.pred_velocity(image,t,context,mask),
        (expected-image)/torch.clamp(1-t[:,None,None,None],min=.05))
    assert latent_geometry(model) == (16,new)
    assert generation_token_stride(model) == 32
    result.square().mean().backward()
    for bridge in (pair.input,pair.output):
        grad = bridge.projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    assert image.grad is not None and torch.isfinite(image.grad).all()
    restored = deepcopy(original)
    install(restored, declaration(old,new))
    restored.to(dtype)
    restored.load_state_dict(model.state_dict(),strict=True)
    assert restored.vae_latent_bridge.ready
    torch.testing.assert_close(restored(image,t,context,mask), result)
    with pytest.raises(ValueError, match="external BCHW"):
        model(torch.randn(2,new+1,8,12),t,context,mask)
    with pytest.raises(ValueError, match="one timestep"):
        model(image,t[:,None],context,mask)
    with pytest.raises(ValueError, match="complete native patches"):
        model(image[:,:,:7],t,context,mask)
    assert not torch.cuda.is_initialized()


def test_native_boundary_is_identity_and_uncalibrated_swap_fails():
    from core.models.minit2i.vae_alignment import latent_geometry, generation_token_stride
    model = model_for(4)
    image, t = torch.randn(1,4,8,12), torch.tensor([.3])
    context, mask = torch.randn(1,5,16), torch.ones(1,5)
    torch.testing.assert_close(model(image,t,context,mask),model.model.net(image,t,context,mask))
    torch.testing.assert_close(model.pred_velocity(image,t,context,mask),
        model.model.pred_velocity(image,t,context,mask))
    assert latent_geometry(model)==(8,4) and generation_token_stride(model)==16
    install(model, declaration(4,4))
    with pytest.raises(RuntimeError,match="no paired calibration"):
        model(image,t,context,mask)


@pytest.mark.parametrize("old,new", [(4,8), (8,4)])
@pytest.mark.parametrize("rewired", [False,True])
@pytest.mark.parametrize("pixel", [False,True])
def test_calibration_training_export_loader_and_native_sampling(tmp_path, old, new, rewired, pixel):
    from types import SimpleNamespace
    import json
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL, AutoencoderKLQwenImage
    from core.models.minit2i.vendor.single_file import save_single_file, load_single_file
    from core.models.minit2i.minit2i_loader import load_minit2i_components
    from core.models.minit2i.residual_unet import restore_topology, native_config
    from core.models.minit2i.vae_alignment import latent_geometry
    from core.models.components.wiring import MINIT2I_WIRING
    from core.training.arch.minit2i import MiniT2IArchHandler
    from core.training.vae_swap import apply_latent_space, preflight_vae_swap, validate_latent_io, swap_metadata
    from core.training.base_trainer import BaseTrainer
    from core.training.ops.minit2i_ops import vae_encode, train_step
    from core.rewiring.spec import build_spec, OPERATORS
    from core.models.minit2i.minit2i_pipeline_ops import (
        vae_encode_image, vae_decode_latent, denoise_loop, denoise_loop_img2img, denoise_loop_inpaint)
    teacher = AutoencoderKL(latent_channels=old,block_out_channels=(4,)*4,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*4,up_block_types=("UpDecoderBlock2D",)*4,
        scaling_factor=.6,shift_factor=.2)
    if pixel:
        teacher=None
    target = AutoencoderKL(latent_channels=new,block_out_channels=(4,)*5,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*5,up_block_types=("UpDecoderBlock2D",)*5,
        scaling_factor=.7,shift_factor=.1,sample_size=128)
    if rewired:
        target=AutoencoderKLQwenImage(base_dim=4,z_dim=new,dim_mult=[1]*5,num_res_blocks=1,
            temperal_downsample=[False]*4,latents_mean=[0.]*new,latents_std=[1.]*new)
    directory=tmp_path/"target"; target.save_pretrained(directory)
    source=f"file:{directory}"
    model=model_for(3,patch_size=16,vae_type="none") if pixel else model_for(old)
    preserved={key:value.clone() for key,value in model.state_dict().items()
        if "img_embedder.proj1" not in key and "final_layer.linear" not in key}
    if rewired:
        restore_topology(model,build_spec("minit2i",[{"name":"main","block_count":3}],
            [dict(id="s",stream="main",parent=None,start_block=1,end_block=2,
                down_factor=[2,2],**OPERATORS)],native_config(model)))
    base=save_single_file(str(tmp_path/"base.safetensors"),model,"b16",
        extra_metadata={"mmjit_config":json.dumps(native_config(model))})
    preflight_vae_swap({"vae_swap_source":source},arch="minit2i",method="full_finetune",
        bundle_vae_explicit_false=False,base_model_path=base)
    trainer=SimpleNamespace(transformer=model,vae=teacher,
        config={"training_method":"full_finetune","vae_swap_source":source,
            "vae_swap_calibration_samples":2,"vae_swap_calibration_resolution":128},
        device=torch.device("cpu"),weight_dtype=torch.float32,training_dtype=torch.float32,
        vae_dtype=torch.float32,wiring=MINIT2I_WIRING,is_minit2i=True,
        text_encoder=None,text_encoder_2=None,log_prefix="[SyntheticMiniT2I]",
        _check_stop_requested=lambda:None,minit2i_latent=True,minit2i_noise_scale=1.,
        mixed_precision=False,stash_cfg_null_per_sample_loss=lambda *args:None)
    trainer.arch=MiniT2IArchHandler(trainer)
    trainer._main_model_module=lambda:model
    trainer.move_main_model_to_cpu=lambda:model.to("cpu")
    trainer.move_main_model_to_gpu=lambda:model.to("cpu")
    trainer.move_vae_to_cpu=lambda:trainer.vae.to("cpu")
    apply_latent_space(trainer,None)
    trainer.minit2i_noise_scale=float(model.mmjit_config.noise_scale)
    assert model.mmjit_config.patch_size==2
    assert model.mmjit_config.noise_scale==(2. if pixel else 1.)
    for key,value in preserved.items():
        torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    trainer.arch=MiniT2IArchHandler(trainer)
    assert trainer.wiring.vae_scale_factor==16 and trainer.wiring.latent_channels==new
    assert model.vae_latent_bridge.declaration["old_scale"]==(1 if pixel else 8)
    assert model.vae_latent_bridge.declaration["old_channels"]==(3 if pixel else old)
    assert BaseTrainer._arch_pixel_align(trainer)==(64 if rewired else 32)
    assert validate_latent_io(trainer)==[]
    images=[]
    for i in range(2):
        path=tmp_path/f"synthetic{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0,256,(128,128,3),dtype=np.uint8)).save(path)
        images.append({"image_path":str(path)})
    trainer.arch.calibrate_before_training(trainer,[SimpleNamespace(items=images)],None)
    assert model.vae_latent_bridge.ready and not hasattr(trainer,"_minit2i_vae_calibration_source")
    if pixel:
        from core.training.ops.minit2i_ops import load_components
        fresh=SimpleNamespace(**vars(trainer))
        fresh.model_path=base;fresh.resume_from_checkpoint=None
        fresh.gradient_checkpointing=False;fresh.blocks_to_swap=0;fresh.use_flash_attention=False
        fresh._main_model_module=lambda:fresh.transformer
        fresh.move_main_model_to_cpu=lambda:fresh.transformer.to("cpu")
        fresh.move_main_model_to_gpu=lambda:fresh.transformer.to("cpu")
        fresh.move_vae_to_cpu=lambda:fresh.vae.to("cpu")
        with patch("core.models.minit2i.minit2i_loader._resolve_flan_t5",return_value="synthetic"), \
             patch("core.models.minit2i.minit2i_loader._load_flan_t5",return_value=(object(),torch.nn.Linear(1,1))):
            load_components(fresh)
        fresh.arch=MiniT2IArchHandler(fresh)
        assert fresh.minit2i_latent and fresh.minit2i_vae_scale_factor==16
        assert fresh.transformer.mmjit_config.patch_size==2
        assert fresh.minit2i_noise_scale==2.
        assert not fresh.transformer.vae_latent_bridge.ready
        fresh.arch.calibrate_before_training(fresh,[SimpleNamespace(items=images)],None)
        assert fresh.transformer.vae_latent_bridge.ready
        assert not hasattr(fresh,"_minit2i_vae_calibration_source")
        for key,value in preserved.items():
            torch.testing.assert_close(fresh.transformer.state_dict()[key],value,rtol=0,atol=0)
    latent=vae_encode(trainer,torch.randn(1,3,128,192))
    assert latent.shape==(1,new,8,12)
    context,mask=torch.randn(1,5,16),torch.ones(1,5)
    loss,*_=train_step(trainer,latent,context,mask,timesteps=torch.tensor([.4]))
    assert torch.isfinite(loss)
    loss.backward()
    for bridge in (model.vae_latent_bridge.input,model.vae_latent_bridge.output):
        assert bridge.projection.weight.grad is not None
        assert torch.isfinite(bridge.projection.weight.grad).all()
    _,bundled,metadata=swap_metadata(trainer)
    assert bundled
    written=save_single_file(str(tmp_path/"aligned.safetensors"),model,"b16",
        vae=trainer.vae,extra_metadata=metadata,max_shard_bytes=4000 if rewired else 10**9)
    restored=load_single_file(written,torch_dtype=torch.float32)["transformer"]
    assert restored.vae_latent_bridge.ready
    for key,value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
    assert (restored._residual_unet_spec["regions"][0]["id"]=="s") if rewired else not hasattr(restored,"_residual_unet_spec")
    with patch("core.models.minit2i.minit2i_loader._resolve_flan_t5",return_value="synthetic"), \
         patch("core.models.minit2i.minit2i_loader._load_flan_t5",return_value=(object(),torch.nn.Linear(1,1))):
        components=load_minit2i_components(written,torch_dtype=torch.float32,vae_dtype=torch.float32)
    restored,codec=components["transformer"],components["vae"]
    assert components["vae_scale_factor"]==16 and latent_geometry(restored)==(16,new)
    for key,value in target.state_dict().items():
        torch.testing.assert_close(codec.inner.state_dict()[key],value,rtol=0,atol=0)
    resume=SimpleNamespace(**vars(trainer)); resume.transformer=restored;resume.vae=codec
    resume.arch=MiniT2IArchHandler(resume)
    apply_latent_space(resume,components["declared_vae"])
    assert validate_latent_io(resume)==[] and not hasattr(resume,"_minit2i_vae_calibration_source")
    from core.training.ops.minit2i_ops import load_components, generate_sample
    resume.model_path=written
    resume.resume_from_checkpoint=written
    resume.gradient_checkpointing=False
    resume.blocks_to_swap=0
    resume.use_flash_attention=False
    with patch("core.models.minit2i.minit2i_loader._resolve_flan_t5",return_value="synthetic"), \
         patch("core.models.minit2i.minit2i_loader._load_flan_t5",return_value=(object(),torch.nn.Linear(1,1))):
        load_components(resume)
    assert resume.minit2i_vae_scale_factor==16 and resume.minit2i_latent
    assert resume.vae.inner.use_tiling and resume.vae.inner.use_slicing
    assert validate_latent_io(resume)==[] and not hasattr(resume,"_minit2i_vae_calibration_source")
    from core.training.adapters.minit2i_adapter import MiniT2IFullParameterAdapter
    adapter=MiniT2IFullParameterAdapter(resume)
    adapter.prepare_models_for_training()
    assert all(p.requires_grad for p in resume.transformer.vae_latent_bridge.parameters())
    saved_again=adapter.write_checkpoint(output_path=tmp_path/"saved_again.safetensors",step=1,epoch=0)
    from core.models.common.single_file_format import read_state_dict
    _,first_md=read_state_dict(written)
    _,next_md=read_state_dict(str(saved_again))
    assert first_md["component.vae.alignment"]==next_md["component.vae.alignment"]
    assert not any(key.startswith("vae.inner.") for key in read_state_dict(str(saved_again))[0])
    resume.optimizer=torch.optim.Adam(resume.transformer.parameters(),lr=1e-4)
    with patch("core.models.minit2i.minit2i_pipeline_ops.encode_prompt",return_value=(context,mask)):
        periodic=generate_sample(resume,"synthetic",height=128,width=192,
            num_inference_steps=2,guidance_scale=2.,seed=3)
    assert periodic.size==(192,128)
    image=Image.new("RGB",(192,128),(40,120,80))
    encoded=vae_encode_image(codec,image,128,192,"cpu",torch.float32)
    assert encoded.shape==latent.shape and vae_decode_latent(codec,encoded).size==image.size
    common=dict(transformer=restored,text=context,mask=mask,num_inference_steps=2,
        cfg_scale=2.,cfg_interval=(0.,1.),device="cpu",dtype=torch.float32,
        channels=new,noise_scale=1.,seed=3,clamp_output=False)
    sampled=denoise_loop(height=8,width=12,**common)
    assert sampled.shape==encoded.shape and torch.isfinite(sampled).all()
    common.pop("channels")
    edited=denoise_loop_img2img(init_image=encoded,denoising_strength=.8,**common)
    assert edited.shape==encoded.shape and torch.isfinite(edited).all()
    painted=denoise_loop_inpaint(init_image=encoded,mask_latent=torch.ones(1,1,8,12),
        denoising_strength=.8,**common)
    assert painted.shape==encoded.shape and torch.isfinite(painted).all()
    from core.pipeline_backends.minit2i import MiniT2IMixin
    owner=SimpleNamespace(minit2i_components=components)
    params=MiniT2IMixin._minit2i_common_params(owner,{"width":192,"height":128,"seed":3},192,128)
    assert (params["channels"],params["vae_scale_factor"],params["latent_h"],params["latent_w"])==(new,16,8,12)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("patch_size", [16,20])
def test_pixel_patch_migration_preserves_coarse_input_output_and_body(dtype,patch_size):
    from core.models.minit2i.vae_alignment import migrate_pixel_patches
    from torch.nn import functional as F
    model=model_for(3,patch_size=patch_size,vae_type="none").to(dtype)
    net=model.model.net
    before=deepcopy(model)
    new_patch=migrate_pixel_patches(model)
    assert new_patch==(2 if patch_size==16 else 4)
    factor=patch_size//new_patch
    coarse=torch.randn(1,3,new_patch,new_patch,dtype=dtype)
    pixels=coarse.repeat_interleave(factor,-1).repeat_interleave(factor,-2)
    torch.testing.assert_close(net.img_embedder.proj1(coarse),before.model.net.img_embedder.proj1(pixels),
        rtol=.03 if dtype==torch.bfloat16 else 1e-5,atol=.003 if dtype==torch.bfloat16 else 1e-6)
    hidden=torch.randn(2,5,32,dtype=dtype)
    original=before.model.net.final_layer.linear(hidden).reshape(10,patch_size,patch_size,3).permute(0,3,1,2)
    expected=F.avg_pool2d(original.float(),factor).to(dtype).permute(0,2,3,1).reshape(2,5,new_patch*new_patch*3)
    torch.testing.assert_close(net.final_layer.linear(hidden),expected,
        rtol=.03 if dtype==torch.bfloat16 else 1e-5,atol=.003 if dtype==torch.bfloat16 else 1e-6)
    for key,value in before.state_dict().items():
        if "img_embedder.proj1" not in key and "final_layer.linear" not in key:
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    assert model.config.patch_size==net.cfg.patch_size==net.img_embedder.patch_size==net.final_layer.patch_size==new_patch
    from core.models.components.pixel_codec import RGBPixelCodec
    teacher=RGBPixelCodec()
    torch.testing.assert_close(teacher.encode(pixels).latent_dist.mode(),pixels)
    torch.testing.assert_close(teacher.encode(pixels).latent_dist.sample(),pixels)


def test_public_codec_candidates_and_legacy_zero_contract():
    from core.training.arch.minit2i import MiniT2IArchHandler
    from core.training.vae_swap import initialization_mode
    model=model_for(4)
    from types import SimpleNamespace
    trainer=SimpleNamespace(transformer=model)
    handler=MiniT2IArchHandler(trainer)
    assert initialization_mode({},"minit2i")=="paired_spatial"
    for rank in (4,5):
        ok,reason=handler.check_vae_compatibility(dict(ndim=rank,scale_factor=16),trainer=trainer)
        assert ok,reason
    for scale in (None,0,12,True):
        ok,reason=handler.check_vae_compatibility(dict(ndim=4,scale_factor=scale),trainer=trainer)
        assert not ok and reason


@pytest.mark.parametrize("damage", ["missing_bridge","wrong_width","missing_config","missing_alignment"])
def test_alignment_artifact_rejects_incomplete_identity(tmp_path,damage):
    import json
    from core.models.minit2i.vendor.single_file import save_single_file, load_single_file
    from core.models.common.single_file_format import read_state_dict, save_single_file_state
    model=model_for(4)
    install(model,declaration(4,4))
    model.vae_latent_bridge.initialize_from_pair(identity(SpatialLatentBridge(4,4)),
        identity(SpatialLatentBridge(4,4)))
    good=save_single_file(str(tmp_path/"good.safetensors"),model,"b16")
    state,metadata=read_state_dict(good)
    if damage=="missing_bridge":
        del state["transformer.vae_latent_bridge.input.projection.weight"]
        message="missing transformer keys"
    elif damage=="wrong_width":
        config=json.loads(metadata["mmjit_config"]);config["in_channels"]=8
        metadata["mmjit_config"]=json.dumps(config)
        message="disagrees with native config"
    elif damage=="missing_config":
        del metadata["mmjit_config"]
        message="bounded native mmjit_config"
    else:
        del metadata["component.vae.alignment"]
        message="unexpected transformer keys"
    damaged=save_single_file_state(state,metadata,str(tmp_path/"damaged.safetensors"))
    with pytest.raises((ValueError,RuntimeError),match=message):
        load_single_file(damaged,torch_dtype=torch.float32)
