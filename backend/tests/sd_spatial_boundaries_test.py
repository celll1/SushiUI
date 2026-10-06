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


@pytest.mark.parametrize("arch",["sd15","sdxl"])
def test_public_swap_modes_and_codec_candidate_contract(arch):
    from core.training.arch import ARCH_REGISTRY
    from core.training.vae_swap import initialization_mode,preflight_vae_swap
    handler=ARCH_REGISTRY[arch]()
    assert initialization_mode({},arch)=="paired_spatial"
    for rank in (4,5):
        for scale in (4,8,16,32):
            assert handler.check_vae_compatibility({"ndim":rank,"scale_factor":scale})==(True,None)
    for rank,scale in ((3,8),(4,12),(4,None),(4,True)):
        assert not handler.check_vae_compatibility({"ndim":rank,"scale_factor":scale})[0]
    assert preflight_vae_swap({"vae_swap_source":"registry:qwen_image"},arch=arch,
        method="full_finetune",bundle_vae_explicit_false=False)=="registry:qwen_image"
    with pytest.raises(ValueError,match="Legacy zero SD"):
        preflight_vae_swap({"vae_swap_source":"registry:qwen_image","vae_swap_new_channel_init":"zero"},
            arch=arch,method="full_finetune",bundle_vae_explicit_false=False)


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
    trainer.arch.apply_vae_swap(trainer,resolved)
    assert model.config.in_channels==model.config.out_channels==max(old,new)
    assert trainer.wiring.latent_channels==new and trainer.wiring.vae_scale_factor==16
    assert BaseTrainer._arch_pixel_align(trainer)==16
    trainer.arch=get_arch_handler(trainer)
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
    from core.models.components.sd_vae_alignment import restore_alignment,load_unet_artifact
    from core.models.components.spatial_swap import read_declaration,restored_image_codec
    from core.models.common.single_file_format import read_state_dict,save_single_file_state
    from core.models.common.vae_source import load_declared_latent_io
    trainer.train_unet=True
    trainer.trains_base_weights=True
    trainer.train_text_encoder=trainer.train_text_encoder_1=trainer.train_text_encoder_2=False
    import json
    from transformers import CLIPTextConfig,CLIPTextModel,CLIPTextModelWithProjection,CLIPTokenizer
    tokenizer_dir=tmp_path/"tokenizer";tokenizer_dir.mkdir()
    (tokenizer_dir/"vocab.json").write_text(json.dumps({"!":0,"<|startoftext|>":1,"<|endoftext|>":2,"a</w>":3}),encoding="utf-8")
    (tokenizer_dir/"merges.txt").write_text("#version: 0.2\n",encoding="utf-8")
    trainer.tokenizer=CLIPTokenizer.from_pretrained(str(tokenizer_dir),pad_token="!",model_max_length=77)
    def clip_config(width):
        return CLIPTextConfig(vocab_size=4,hidden_size=width,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=1,projection_dim=POOLED_DIM,
            bos_token_id=1,eos_token_id=2,pad_token_id=0)
    trainer.text_encoder=CLIPTextModel(clip_config(CROSS_DIM[arch]-(POOLED_DIM if arch=="sdxl" else 0)))
    trainer.text_encoder_2=CLIPTextModelWithProjection(clip_config(POOLED_DIM)) if arch=="sdxl" else None
    trainer.tokenizer_2=trainer.tokenizer if arch=="sdxl" else None
    original_te=trainer.text_encoder
    original_te2=trainer.text_encoder_2
    original_te.eval()
    if original_te2 is not None:original_te2.eval()
    trainer.bundle_vae=True
    trainer.config["vae_swap_source"]=resolved.source
    adapter=SDXLFullParameterAdapter(trainer) if arch=="sdxl" else SD15FullParameterAdapter(trainer)
    path=adapter.write_checkpoint(7,0,tmp_path/"aligned.safetensors")
    state,metadata=read_state_dict(str(path))
    assert metadata["modelspec.prediction_type"]=="epsilon"
    assert "component.vae.alignment" in metadata and "unet_config" in metadata
    assert not any(key.startswith("vae.inner.") for key in state)
    assert "text_encoder_config" in metadata and "scheduler_config" in metadata
    rebuilt=UNet2DConditionModel.from_config(model.config)
    rebuilt.load_state_dict({key:tensor for key,tensor in model.state_dict().items()
        if not key.startswith("vae_latent_bridge.")},strict=True)
    restore_alignment(rebuilt,metadata,state)
    assert rebuilt.vae_latent_bridge.ready
    loaded=load_unet_artifact(str(path),dtype=torch.float32)
    for key,tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key],tensor,rtol=0,atol=0)
    for key,tensor in model.vae_latent_bridge.state_dict().items():
        torch.testing.assert_close(rebuilt.vae_latent_bridge.state_dict()[key],tensor,rtol=0,atol=0)
    declared=load_declared_latent_io(str(path),arch=arch)
    codec,_=restored_image_codec(declared.load_module(torch_dtype=torch.float32),declared,read_declaration(metadata))
    assert codec.spatial_scale_factor==16 and codec.config.latent_channels==new
    for key,tensor in trainer.vae.inner.state_dict().items():
        torch.testing.assert_close(codec.inner.state_dict()[key],tensor,rtol=0,atol=0)
    sharded=save_single_file_state(state,metadata,str(tmp_path/"sharded.safetensors"),max_shard_bytes=32768)
    shard_state,shard_md=read_state_dict(str(sharded))
    loaded_shards=load_unet_artifact(str(sharded),dtype=torch.float32)
    for key,tensor in model.state_dict().items():
        torch.testing.assert_close(loaded_shards.state_dict()[key],tensor,rtol=0,atol=0)
    restore_alignment(rebuilt,shard_md,shard_state)
    assert rebuilt.vae_latent_bridge.ready
    from core.model_loader import ModelLoader
    for artifact in (path,sharded,str(sharded).removesuffix(".index.json")):
        with patch("core.model_loader.StableDiffusionXLPipeline.from_single_file",side_effect=AssertionError("no partial shard load")), \
             patch("core.model_loader.StableDiffusionPipeline.from_single_file",side_effect=AssertionError("no partial shard load")):
            pipeline=ModelLoader.reconstruct_sd_sdxl_pipeline(str(artifact),arch,torch.float32,"cpu")
        for key,tensor in original_te.state_dict().items():
            torch.testing.assert_close(pipeline.text_encoder.state_dict()[key],tensor,rtol=0,atol=0)
        assert pipeline.tokenizer.special_tokens_map==trainer.tokenizer.special_tokens_map
        assert pipeline.tokenizer.padding_side==trainer.tokenizer.padding_side
        tokens=pipeline.tokenizer("a",padding="max_length",max_length=77,return_tensors="pt").input_ids
        torch.testing.assert_close(pipeline.text_encoder(tokens).last_hidden_state,
            original_te(tokens).last_hidden_state,rtol=1e-5,atol=2e-6)
        pipeline.encode_prompt("a",device="cpu",num_images_per_prompt=1,do_classifier_free_guidance=True,
            negative_prompt="")
        if arch=="sdxl":
            for key,tensor in original_te2.state_dict().items():
                torch.testing.assert_close(pipeline.text_encoder_2.state_dict()[key],tensor,rtol=0,atol=0)
    assert pipeline.unet.vae_latent_bridge.ready
    assert pipeline.vae.spatial_scale_factor==16
    for key,tensor in model.state_dict().items():
        torch.testing.assert_close(pipeline.unet.state_dict()[key],tensor,rtol=0,atol=0)
    trainer.unet=pipeline.unet
    trainer.vae=pipeline.vae
    from core.training.vae_swap import apply_latent_space
    with patch("core.training.vae_swap.apply_configured_vae_swap",side_effect=AssertionError("must not recalibrate")):
        apply_latent_space(trainer,declared)
    assert trainer.unet.config.in_channels==max(old,new)
    assert trainer.wiring.latent_channels==new and trainer.wiring.vae_scale_factor==16
    from core.training.ops.sd_sdxl_ops import load_components
    trainer.model_path=str(sharded)
    trainer.dtype=torch.float32
    trainer.use_flash_attention=trainer.gradient_checkpointing=trainer.debug_vram=False
    with patch("core.training.ops.sd_sdxl_ops.StableDiffusionXLPipeline.from_single_file",side_effect=AssertionError("no partial shard load")), \
         patch("core.training.ops.sd_sdxl_ops.StableDiffusionPipeline.from_single_file",side_effect=AssertionError("no partial shard load")), \
         patch("core.training.vae_swap.apply_configured_vae_swap",side_effect=AssertionError("must not recalibrate")):
        load_components(trainer)
    assert trainer.is_sdxl==(arch=="sdxl")
    assert trainer.vae.config.latent_channels==trainer.vae_latent_channels==new
    assert trainer.wiring.vae_scale_factor==16
    for key,tensor in model.state_dict().items():
        torch.testing.assert_close(trainer.unet.state_dict()[key],tensor,rtol=0,atol=0)
    from core.models.components.sd_vae_alignment import load_pipeline_artifact
    no_te=dict(state)
    te_prefix="conditioner.embedders.0.transformer." if arch=="sdxl" else "cond_stage_model.transformer."
    embedding_key=next(key for key in original_te.state_dict() if key.endswith("embeddings.token_embedding.weight"))
    del no_te[te_prefix+embedding_key]
    incomplete_te=save_single_file_state(no_te,metadata,str(tmp_path/"incomplete_te.safetensors"),max_shard_bytes=32768)
    with pytest.raises(RuntimeError,match="Missing key"):
        load_pipeline_artifact(str(incomplete_te),arch=arch,dtype=torch.float32)
    no_alignment=dict(metadata)
    del no_alignment["component.vae.alignment"]
    undeclared=save_single_file_state(state,no_alignment,str(tmp_path/"undeclared.safetensors"))
    with pytest.raises(ValueError,match="alignment declaration"):
        load_unet_artifact(str(undeclared),dtype=torch.float32)
    no_body=dict(state)
    del no_body["model.diffusion_model.input_blocks.0.0.weight"]
    incomplete=save_single_file_state(no_body,metadata,str(tmp_path/"incomplete.safetensors"))
    with pytest.raises(RuntimeError,match="Missing key"):
        load_unet_artifact(str(incomplete),dtype=torch.float32)
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
