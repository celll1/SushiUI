"""Synthetic Anima codec boundaries across native and residual execution."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from core.models.components.spatial_swap import install
from core.models.components.vae_alignment import SpatialLatentBridge
from core.models.anima.vae_alignment import noise_kwargs
from backend.tests.latent_adapter_contract_test import declaration,identity


def test_recreated_handler_uses_swapped_codec_for_nested_unet_buckets():
    from types import SimpleNamespace
    from core.training.base_trainer import BaseTrainer
    from core.training.bucketing import get_bucket_sizes
    from core.rewiring.spec import build_spec, OPERATORS, preview_plan
    spec = build_spec("anima", [{"name": "main", "block_count": 28}], [
        dict(id="r0", stream="main", parent=None, start_block=4, end_block=24,
             down_factor=[2, 2], **OPERATORS),
        dict(id="r1", stream="main", parent="r0", start_block=12, end_block=16,
             down_factor=[2, 2], **OPERATORS),
    ], {})
    model = SimpleNamespace(patch_spatial=2, _residual_unet_spec=spec,
        vae_latent_bridge=SimpleNamespace(declaration={"new_scale": 16, "new_channels": 64}))
    trainer = SimpleNamespace(is_anima=True, transformer=model, arch=SimpleNamespace(pixel_align=16))
    align = BaseTrainer._arch_pixel_align(trainer)
    assert align == 128
    with pytest.raises(ValueError, match="r1 grid"):
        preview_plan(spec, 832//32, 1216//32)
    for bucket in get_bucket_sizes(1024, align):
        preview_plan(spec, bucket.height//32, bucket.width//32)


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
@pytest.mark.parametrize("explicit_topology",[False,True])
def test_native_teacher_calibration_and_coarse_training_encode(tmp_path,old_c,new_c,explicit_topology):
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL,AutoencoderKLQwenImage
    from core.models.anima.anima_models import Anima
    from core.models.anima.residual_unet import native_config,restore_topology
    from core.models.common.vae_source import resolve_vae_source
    from core.models.components.wiring import ANIMA_WIRING
    from core.training.arch.anima import AnimaArchHandler
    from core.training.ops.anima_ops import vae_encode
    from core.training.vae_swap import validate_latent_io,apply_latent_space,preflight_vae_swap,initialization_mode
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
            "vae_swap_calibration_resolution":128,"vae_swap_source":resolved.source},device="cpu",weight_dtype=torch.float32,
        vae_dtype=torch.float32,wiring=ANIMA_WIRING.replace(latent_channels=old_c),
        text_encoder=None,text_encoder_2=None,log_prefix="[SyntheticAnima]",is_anima=True,
        _check_stop_requested=lambda:None)
    trainer.arch=AnimaArchHandler(trainer)
    if explicit_topology:
        trainer.config.update(dit_rewiring_mode="configure",
            dit_rewiring_spec=deepcopy(value._residual_unet_spec))
        trainer.config["dit_rewiring_spec"]["regions"][0]["down"]="rms_pool"
    original_requested=deepcopy(trainer.config.get("dit_rewiring_spec"))
    trainer._main_model_module=lambda:value
    trainer.move_main_model_to_cpu=lambda:value.to("cpu")
    trainer.move_main_model_to_gpu=lambda:value.to("cpu")
    trainer.move_vae_to_cpu=lambda:trainer.vae.to("cpu")
    assert initialization_mode(trainer.config,"anima")=="paired_spatial"
    preflight_vae_swap(trainer.config,arch="anima",method="full_finetune",bundle_vae_explicit_false=False)
    apply_latent_space(trainer,None)
    from core.models.anima.residual_unet import configure_training_topology
    configured=configure_training_topology(value,trainer.config)
    if explicit_topology:
        assert configured["regions"][0]["down"]=="rms_pool"
        resume_config=dict(trainer.config,dit_rewiring_spec=original_requested)
        assert configure_training_topology(value,resume_config,resuming=True)==configured
        edited=deepcopy(original_requested)
        edited["regions"][0]["down"]="avg_pool"
        with pytest.raises(ValueError):
            configure_training_topology(value,dict(resume_config,dit_rewiring_spec=edited),resuming=True)
    assert trainer.wiring.vae_scale_factor==16 and trainer.wiring.latent_channels==new_c
    assert BaseTrainer._arch_pixel_align(trainer)==64
    trainer.arch=AnimaArchHandler(trainer)
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
        padding_mask=torch.zeros(1,1,8,12),**noise_kwargs(value,torch.randn_like(latent)))
    assert output.shape==(1,new_c,1,8,12)
    output.square().mean().backward()
    assert value.vae_latent_bridge.input.projection.weight.grad.abs().sum()>0
    assert not hasattr(trainer,"_anima_vae_calibration_source")
    from core.models.anima.anima_pipeline_ops import (
        vae_encode_image,vae_decode_latents,prepare_style_reference,
        sample_txt2img,sample_img2img,sample_inpaint,make_mask_latents)
    from core.models.anima.anima_scheduler import AnimaFlowMatchScheduler
    from core.models.anima.vae_alignment import generation_token_stride
    assert generation_token_stride(value)==32
    image=Image.new("RGB",(192,128),(80,140,40))
    encoded=vae_encode_image(trainer.vae,image,"cpu",torch.float32)
    assert encoded.shape==(1,new_c,1,8,12)
    assert vae_decode_latents(trainer.vae,encoded)[0].size==image.size
    ref,noise=prepare_style_reference(trainer.vae,image,128,192,"cpu",torch.float32,seed=42)
    assert ref.shape==noise.shape==encoded.shape
    value.eval()
    cond=dict(prompt_embeds=torch.randn(1,5,32),t5_input_ids=None,t5_attn_mask=None,source_mask=None)
    common=dict(transformer=value,scheduler=AnimaFlowMatchScheduler(),cond_embeds=cond,
        uncond_embeds=cond,num_inference_steps=2,guidance_scale=2.,
        generator=torch.Generator().manual_seed(3),device="cpu",dtype=torch.float32,spectrum_params={})
    sampled=sample_txt2img(height=128,width=192,**common)
    assert sampled.shape==encoded.shape and torch.isfinite(sampled).all()
    edited=sample_img2img(init_latents=encoded,denoising_strength=.8,**common)
    assert edited.shape==encoded.shape and torch.isfinite(edited).all()
    mask=make_mask_latents(Image.new("L",image.size,255),8,12,"cpu",torch.float32)
    painted=sample_inpaint(init_latents=encoded,mask_latents=mask,denoising_strength=.8,**common)
    assert painted.shape==encoded.shape and torch.isfinite(painted).all()
    assert vae_decode_latents(trainer.vae,sampled)[0].size==image.size
    from core.training.adapters.anima_adapter import AnimaFullParameterAdapter
    from core.models.anima import anima_loader
    if new_c>old_c:
        before={k:v.clone() for k,v in value.state_dict().items()}
        trainer.unet_lr=2e-6
        trainer.config.update(vae_swap_io_lr_factor=5.,vae_swap_bridge_lr_factor=3.)
        groups={g["name"]:g for g in AnimaFullParameterAdapter(trainer).arch_param_groups()}
        assert list(value.state_dict())==list(before)
        for key,tensor in before.items():
            assert torch.equal(value.state_dict()[key],tensor)
        embed,head=value.x_embedder.proj[1],value.final_layer.linear
        # The split selects exactly the zero-initialized target-channel columns and rows.
        assert embed.weight_fresh.shape[1]==head.weight_fresh.shape[0]==new_c*4
        assert not embed.weight_fresh.any() and not head.weight_fresh.any()
        assert embed.weight_kept.abs().sum(0).all() and head.weight_kept.abs().sum(1).all()
        assert groups["vae_swap_io"]["lr"]==pytest.approx(1e-5)
        assert {id(p) for p in groups["vae_swap_io"]["params"]}=={id(embed.weight_fresh),id(head.weight_fresh)}
        assert groups["vae_swap_bridge"]["lr"]==pytest.approx(6e-6)
        assert {id(p) for p in groups["vae_swap_bridge"]["params"]}=={
            id(p) for p in value.vae_latent_bridge.parameters()}
    trainer.bundle_vae=True
    trainer.config["vae_swap_source"]=resolved.source
    path=AnimaFullParameterAdapter(trainer).write_checkpoint(7,0,tmp_path/"full.safetensors")
    restored=anima_loader.load_anima_dit(str(path),dtype=torch.float32,latent_channels=new_c)
    if hasattr(value.vae_latent_bridge,"noise_alignment"):
        loaded_bf16=anima_loader.load_anima_dit(str(path),dtype=torch.bfloat16,latent_channels=new_c)
        for key,tensor in value.vae_latent_bridge.noise_alignment.state_dict().items():
            assert torch.equal(tensor,loaded_bf16.vae_latent_bridge.noise_alignment.state_dict()[key])
        del loaded_bf16
    assert restored.vae_latent_bridge.ready
    assert restored.in_channels==(old_c+new_c if new_c>old_c else old_c)
    assert (value.vae_latent_bridge.declaration.get("passthrough")=="full")==(new_c>old_c)
    assert restored._residual_unet_spec==value._residual_unet_spec
    for key,tensor in value.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],tensor,rtol=0,atol=0)
    assert restored(latent.unsqueeze(2),torch.tensor([.6]),torch.randn(1,5,32),
        padding_mask=torch.zeros(1,1,8,12),**noise_kwargs(restored,torch.randn_like(latent))).shape==output.shape
    from core.models.common.single_file_format import read_state_dict,save_single_file_state
    state,metadata=read_state_dict(str(path))
    sharded=save_single_file_state(state,metadata,str(tmp_path/"sharded.safetensors"),max_shard_bytes=32768)
    restored_shards=anima_loader.load_anima_dit(str(sharded),dtype=torch.float32,latent_channels=new_c)
    assert restored_shards.vae_latent_bridge.ready
    torch.testing.assert_close(restored_shards.vae_latent_bridge.input.projection.weight,
        value.vae_latent_bridge.input.projection.weight)
    with patch.object(anima_loader,"discover_anima_components",return_value={"dit":str(path),"vae":None,"text_encoder":"stub"}), \
         patch.object(anima_loader,"load_qwen3_text_encoder",return_value=(torch.nn.Linear(2,2),object())), \
         patch.object(anima_loader,"load_t5_tokenizer",return_value=object()):
        components=anima_loader.load_anima_components(str(path),dit_dtype=torch.float32,vae_dtype=torch.float32)
    assert components["vae_scale_factor"]==16 and components["latent_channels"]==new_c
    assert components["vae"].spatial_scale_factor==16
    for key,tensor in trainer.vae.inner.state_dict().items():
        torch.testing.assert_close(components["vae"].inner.state_dict()[key],tensor,rtol=0,atol=0)
    trainer.transformer=components["transformer"]
    trainer.vae=components["vae"]
    if explicit_topology:
        restored_config=dict(trainer.config,dit_rewiring_spec=original_requested)
        assert configure_training_topology(trainer.transformer,restored_config,resuming=True)==configured
    from core.training.vae_swap import apply_latent_space
    with patch("core.training.vae_swap.apply_configured_vae_swap",side_effect=AssertionError("must not recalibrate")):
        apply_latent_space(trainer,components["declared_vae"])
    assert trainer.wiring.latent_channels==new_c and trainer.wiring.vae_scale_factor==16
    assert not hasattr(trainer,"_anima_vae_calibration_source")
    del state["net.vae_latent_bridge.input.projection.weight"]
    invalid=save_single_file_state(state,metadata,str(tmp_path/"missing_bridge.safetensors"))
    with pytest.raises(ValueError,match="does not match"):
        anima_loader.load_anima_dit(str(invalid),dtype=torch.float32,latent_channels=new_c)
    absent=tmp_path/"no_codec.safetensors"
    with patch("core.training.vae_swap.swap_metadata",return_value=(None,False,{})), \
         pytest.raises(ValueError,match="codec declaration"):
        AnimaFullParameterAdapter(trainer).write_checkpoint(8,0,absent)
    assert not absent.exists()
    assert not torch.cuda.is_initialized()
