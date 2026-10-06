"""Synthetic packed-image alignment, native codecs and checkpoint continuation."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from core.models.components.spatial_swap import install, map_packed_image
from backend.tests.latent_adapter_contract_test import declaration, identity


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
         patch.object(torch.cuda,"_lazy_init",return_value=None), \
         patch.object(torch._C,"_cuda_init",return_value=None):
        yield


def model(channels, rewired=False):
    from core.models.krea2.vendor.transformer import Krea2Transformer2DModel
    from core.models.krea2.residual_unet import native_config,restore_topology
    from core.rewiring.spec import build_spec,OPERATORS
    value = Krea2Transformer2DModel(in_channels=channels*4,num_layers=7,
        attention_head_dim=12,num_attention_heads=2,num_key_value_heads=1,
        intermediate_size=32,timestep_embed_dim=8,text_hidden_dim=12,
        num_text_layers=2,text_num_attention_heads=2,text_num_key_value_heads=1,
        text_intermediate_size=16,num_layerwise_text_blocks=1,num_refiner_text_blocks=1,
        axes_dims_rope=(4,4,4))
    with torch.no_grad():
        for p in value.parameters():
            if p.ndim >= 2:
                p.normal_(0,.02)
    if rewired:
        restore_topology(value,build_spec("krea2",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
            dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)
        ],native_config(dict(value.config))))
    return value


@pytest.mark.parametrize("rewired", [False,True])
def test_gaussian_packed_model_gradients_and_sampler_noise(rewired):
    from core.models.components.gaussian_alignment import POLICY
    from core.models.krea2.krea2_pipeline_ops import prepare_position_ids, denoise_loop, denoise_loop_img2img, denoise_loop_inpaint
    from diffusers import FlowMatchEulerDiscreteScheduler
    value=model(8,rewired);install(value,dict(declaration(4,8),noise_policy=POLICY))
    pair=value.vae_latent_bridge
    identity(pair.input,.2);identity(pair.output,.3)
    pair.noise_alignment.capture(pair.input,pair.output)
    from core.models.krea2.vendor.single_file import build_krea2_transformer
    expected_noise={k:v.clone() for k,v in pair.noise_alignment.state_dict().items()}
    loaded=build_krea2_transformer(value.state_dict(),dict(value.config),torch.bfloat16,
                                   alignment_declaration=pair.declaration)
    for key,tensor in expected_noise.items():
        assert torch.equal(tensor,loaded.vae_latent_bridge.noise_alignment.state_dict()[key])
    del loaded
    x=torch.randn(1,64,32,requires_grad=True);noise=torch.randn_like(x)
    caption=torch.randn(1,3,2,12);mask=torch.ones(1,3,dtype=torch.bool)
    sigma=torch.tensor([.6]);positions=prepare_position_ids(3,8,8,"cpu")
    output=value(x,caption,sigma,positions,alignment_noise=noise,return_dict=False)[0]
    del value.vae_latent_bridge
    try:
        native=value(map_packed_image(pair,x,(8,8),sigma=sigma,noise=noise),caption,sigma,positions,return_dict=False)[0]
        expected=map_packed_image(pair,native,(8,8),output=True,noise=noise)
    finally:value.add_module("vae_latent_bridge",pair)
    torch.testing.assert_close(output,expected)
    output.square().mean().backward()
    assert pair.input.projection.weight.grad.abs().sum()>0
    assert pair.output.projection.weight.grad.abs().sum()>0
    assert all(b.grad is None for b in pair.noise_alignment.buffers())
    recorded=[];original=value.forward
    def forward(*a,**kw):
        recorded.append((kw['hidden_states'].detach().clone(),kw['alignment_noise'].detach().clone()))
        return original(*a,**kw)
    value.forward=forward
    common=dict(transformer=value,scheduler=FlowMatchEulerDiscreteScheduler(),prompt_embeds=caption,
        prompt_embeds_mask=mask,neg_prompt_embeds=caption,neg_prompt_embeds_mask=mask,guidance=2.,
        num_inference_steps=3,grid_h=8,grid_w=8,patch_size=2,is_distilled=False,device="cpu")
    result=denoise_loop(latents=noise.clone(),**common)
    assert torch.isfinite(result).all() and len(recorded)==6
    assert all(torch.equal(n,noise) for _,n in recorded)
    for inpaint in (False,True):
        recorded.clear();common['scheduler']=FlowMatchEulerDiscreteScheduler()
        fn=denoise_loop_inpaint if inpaint else denoise_loop_img2img
        extra=dict(mask_latent=torch.ones(1,64,1)) if inpaint else {}
        result=fn(init_latents=x.detach(),denoising_strength=.8,seed=63,**extra,**common)
        actual_noise=recorded[0][1]
        first_sigma=float(common['scheduler'].timesteps[1])/1000
        torch.testing.assert_close(recorded[0][0],(1-first_sigma)*x.detach()+first_sigma*actual_noise)
        assert all(torch.equal(n,actual_noise) for _,n in recorded)
        assert torch.isfinite(result).all()


@pytest.mark.parametrize("old,new",[(4,8),(8,4)])
@pytest.mark.parametrize("rewired",[False,True])
@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_packed_bridge_preserves_native_grid_bias_and_differentiation(old,new,rewired,dtype):
    from core.models.krea2.krea2_pipeline_ops import prepare_position_ids
    value = model(max(old,new),rewired)
    install(value,declaration(old,new))
    pair = value.vae_latent_bridge
    identity(pair.input,.2); identity(pair.output,.3)
    value.to(dtype)
    x = torch.randn(1,8*12,new*4,dtype=dtype,requires_grad=True)
    caption = torch.randn(1,3,2,12,dtype=dtype)
    sigma = torch.tensor([.7],dtype=dtype)
    positions = prepare_position_ids(3,8,12,"cpu")
    from core.models.krea2.vae_alignment import image_grid
    with patch("core.models.krea2.vae_alignment.torch.equal",side_effect=AssertionError("native grid must not synchronize")):
        assert image_grid(positions,3,96) == (8,12)
    actual = value(x,caption,sigma,positions,return_dict=False)[0]
    assert actual.shape == x.shape
    del value.vae_latent_bridge
    try:
        native = value(map_packed_image(pair,x,(8,12),sigma=sigma),caption,sigma,positions,return_dict=False)[0]
        expected = map_packed_image(pair,native,(8,12),output=True)
    finally:
        value.add_module("vae_latent_bridge",pair)
    torch.testing.assert_close(actual,expected)
    actual.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert pair.input.projection.weight.grad.abs().sum() > 0
    assert pair.output.projection.weight.grad.abs().sum() > 0
    assert value.img_in.weight.grad.abs().sum() > 0


def test_packing_bias_is_per_pixel_and_wrong_positions_are_refused():
    from core.models.krea2.krea2_pipeline_ops import prepare_position_ids
    value = model(4); install(value,declaration(4,4))
    pair = value.vae_latent_bridge
    identity(pair.input,.2); identity(pair.output,.3)
    x = torch.randn(2,6,16)
    sigma = torch.tensor([.2,.8])
    torch.testing.assert_close(map_packed_image(pair,x,(2,3),sigma=sigma),x+.2*(1-sigma[:,None,None]))
    torch.testing.assert_close(map_packed_image(pair,x,(2,3),output=True),x-.3)
    ids = prepare_position_ids(3,2,3,"cpu"); ids[-1,1] = 0
    with pytest.raises(ValueError,match="row-major"):
        value(x,torch.randn(2,3,2,12),sigma,ids)


def test_real_causal_source_codec_calibration_public_loader_and_resume(tmp_path):
    from diffusers import AutoencoderKL,AutoencoderKLQwenImage
    from PIL import Image
    import numpy as np
    from core.models.common.vae_source import resolve_vae_source
    from core.models.krea2 import krea2_loader as loader
    from core.models.krea2.vendor.single_file import load_single_file,save_single_file
    from core.models.krea2.krea2_pipeline_ops import vae_encode,vae_decode
    from core.models.components.latent_adapter import NormalizedImageVAE
    from core.models.components.wiring import KREA2_WIRING
    from core.training.arch.krea2 import Krea2ArchHandler
    from core.training.vae_swap import apply_latent_space,validate_latent_io,swap_metadata
    from core.training.adapters.krea2_adapter import Krea2FullParameterAdapter
    from core.training.base_trainer import BaseTrainer
    old = AutoencoderKLQwenImage(base_dim=4,z_dim=4,dim_mult=[1,1,1,1],num_res_blocks=1,
        temperal_downsample=[False,False,False],latents_mean=[0.]*4,latents_std=[1.]*4)
    target = AutoencoderKL(in_channels=3,out_channels=3,latent_channels=8,
        down_block_types=("DownEncoderBlock2D",)*5,up_block_types=("UpDecoderBlock2D",)*5,
        block_out_channels=(4,)*5,norm_num_groups=4,scaling_factor=.7,shift_factor=.2)
    directory = tmp_path/"target"; target.save_pretrained(directory)
    resolved = resolve_vae_source(f"file:{directory}",arch="krea2")
    value = model(4,True)
    trainer = SimpleNamespace(transformer=value,transformer_original=value,vae=old,
        config={"training_method":"full_finetune","vae_swap_source":resolved.source,
            "vae_swap_calibration_samples":2,"vae_swap_calibration_resolution":128},
        weight_dtype=torch.float32,vae_dtype=torch.float32,device="cpu",train_unet=True,
        train_text_encoder=False,text_encoder=torch.nn.Linear(2,2),text_encoder_2=None,
        wiring=KREA2_WIRING.replace(latent_channels=4),bundle_vae=True,is_krea2=True,
        _check_stop_requested=lambda:None,log_prefix="[SyntheticKrea]")
    trainer.arch = Krea2ArchHandler(trainer)
    trainer._main_model_module=lambda:value
    trainer.move_main_model_to_cpu=lambda:value.to("cpu")
    trainer.move_main_model_to_gpu=lambda:value.to("cpu")
    trainer.move_vae_to_cpu=lambda:trainer.vae.to("cpu")
    trainer.arch.apply_vae_swap(trainer,resolved)
    assert isinstance(trainer.vae,NormalizedImageVAE)
    assert trainer.wiring.vae_scale_factor == 16
    assert BaseTrainer._arch_pixel_align(trainer) == 128
    images=[]
    for i in range(2):
        pixels=np.random.default_rng(i).integers(0,256,(128,128,3),dtype=np.uint8)
        path=tmp_path/f"synthetic{i}.png"; Image.fromarray(pixels).save(path)
        images.append({"image_path":str(path)})
    trainer.arch.calibrate_before_training(trainer,[SimpleNamespace(items=images)],None)
    assert value.vae_latent_bridge.ready
    assert validate_latent_io(trainer) == []
    image=Image.new("RGB",(128,128),(80,140,40))
    latent=vae_encode(trainer.vae,image,128,128,2,"cpu",torch.float32)
    assert latent.shape == (1,16,32)
    assert vae_decode(trainer.vae,latent,4,4,2).size == (128,128)
    with torch.no_grad(): value.vae_latent_bridge.input.projection.bias.add_(.1)
    path=tmp_path/"full.safetensors"
    Krea2FullParameterAdapter(trainer).write_checkpoint(7,0,path)
    companion=tmp_path/"original_model"
    for name in ("text_encoder","vae"):
        folder=companion/name;folder.mkdir(parents=True,exist_ok=True)
        (folder/"config.json").write_text("{}",encoding="utf-8")
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler=FlowMatchEulerDiscreteScheduler(base_shift=.73,max_shift=1.31,use_dynamic_shifting=True)
    with patch.object(loader,"_load_scheduler",return_value=scheduler) as resolve_scheduler, \
         patch.object(loader,"_resolve_te_dir",wraps=loader._resolve_te_dir) as resolve_te, \
         patch.object(loader,"_resolve_vae_dir",wraps=loader._resolve_vae_dir) as resolve_vae:
        components=loader.load_krea2_components(str(path),torch_dtype=torch.float32,
            load_text_encoder=False,companion_path=str(companion))
    resolve_te.assert_called_once_with(str(companion),None)
    resolve_scheduler.assert_called_once_with(str(companion))
    resolve_vae.assert_called_once_with(str(companion),None)
    assert loader._resolve_te_dir(str(companion),None) == str(companion/"text_encoder")
    restored=components["transformer"]
    assert components["vae_scale_factor"] == 16
    assert components["declared_vae"].config["scaling_factor"] == .7
    assert restored.vae_latent_bridge.ready
    trainer.transformer = trainer.transformer_original = restored
    trainer.vae=components["vae"]
    with patch("core.training.vae_swap.apply_configured_vae_swap",side_effect=AssertionError("must not recalibrate")):
        apply_latent_space(trainer,components["declared_vae"])
    torch.testing.assert_close(restored.vae_latent_bridge.input.projection.bias,value.vae_latent_bridge.input.projection.bias)
    _,_,metadata=swap_metadata(trainer)
    invalid=tmp_path/"missing_codec.safetensors"
    with pytest.raises(ValueError,match="codec declaration"):
        save_single_file(str(invalid),restored,False)
    assert not invalid.exists()
    sharded=tmp_path/"shards.safetensors"
    save_single_file(str(sharded),restored,False,vae=trainer.vae.inner,
        extra_metadata=metadata,max_shard_bytes=32768)
    result=load_single_file(str(sharded),torch_dtype=torch.float32)
    assert result["transformer"].vae_latent_bridge.ready
    assert not torch.cuda.is_initialized()

    tokenizer_dir=companion/"tokenizer";tokenizer_dir.mkdir()
    (tokenizer_dir/"tokenizer_config.json").write_text("{}",encoding="utf-8")
    references=components["companion_references"]
    references["tokenizer"]={"kind":"local","path":str(tokenizer_dir)}
    trainer.krea2_companion_references=references
    persisted=tmp_path/"persisted.safetensors"
    Krea2FullParameterAdapter(trainer).write_checkpoint(8,0,persisted)
    import json
    from core.models.common.single_file_format import read_metadata
    assert json.loads(read_metadata(str(persisted))[loader.COMPANION_METADATA_KEY]) == references
    for checkpoint in (persisted,tmp_path/"persisted_shards.safetensors.index.json"):
        if checkpoint != persisted:
            save_single_file(str(tmp_path/"persisted_shards.safetensors"),restored,False,
                vae=trainer.vae.inner,extra_metadata={**metadata,
                    loader.COMPANION_METADATA_KEY:json.dumps(references)},max_shard_bytes=32768)
        with patch.object(loader,"_load_scheduler",side_effect=AssertionError("saved scheduler required")), \
             patch.object(loader,"_resolve_te_dir",side_effect=AssertionError("saved TE required")), \
             patch.object(loader,"_load_qwen3vl_text_encoder",return_value=trainer.text_encoder) as load_te, \
             patch("transformers.AutoTokenizer.from_pretrained",return_value=object()) as load_tokenizer:
            saved=loader.load_krea2_components(str(checkpoint),torch_dtype=torch.float32)
        load_te.assert_called_once_with(str(companion/"text_encoder"),torch.float32)
        load_tokenizer.assert_called_once_with(str(tokenizer_dir),trust_remote_code=True)
        assert saved["companion_references"] == references
        assert dict(saved["scheduler"].config) == dict(scheduler.config)
        assert saved["transformer"].vae_latent_bridge.ready
    (companion/"text_encoder"/"config.json").unlink()
    with patch.object(loader,"load_single_file",side_effect=AssertionError("must preflight first")):
        with pytest.raises(ValueError,match="Saved Krea companion is missing"):
            loader.load_krea2_components(str(persisted))


@pytest.mark.parametrize("references", [
    {"version":2,"scheduler_config":{}},
    {"version":1,"scheduler_config":{}},
    {"version":1,"scheduler_config":{"base_shift":.5},"text_encoder":{"kind":"local","path":""}},
])
def test_invalid_companion_metadata_rejected_before_weights(tmp_path,references):
    import json
    from safetensors.torch import save_file
    from core.models.krea2 import krea2_loader as loader
    path=tmp_path/"invalid.safetensors"
    save_file({"synthetic":torch.zeros(1)},str(path),
        metadata={loader.COMPANION_METADATA_KEY:json.dumps(references)})
    with patch.object(loader,"load_single_file",side_effect=AssertionError("must preflight first")):
        with pytest.raises(ValueError,match="Invalid Krea"):
            loader.load_krea2_components(str(path))


@pytest.mark.parametrize("location", ["text_encoder", "tokenizer", None])
def test_tokenizer_prefers_local_companions_before_hub(tmp_path,location):
    from core.models.krea2 import krea2_loader as loader
    if location:
        folder=tmp_path/location;folder.mkdir()
        (folder/"tokenizer_config.json").write_text("{}",encoding="utf-8")
    with patch("transformers.AutoTokenizer.from_pretrained",return_value=object()) as load:
        loader._load_tokenizer(str(tmp_path/"text_encoder"))
    source=str(tmp_path/location) if location else loader.TE_HUB_ID
    load.assert_called_once_with(source,trust_remote_code=True)
