"""Target-only noise and clean reference boundaries across real Flux2 forwards."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from backend.tests.flux2_spatial_boundaries_test import ids, model, no_cuda
from backend.tests.gaussian_alignment_test import pair
from core.models.components.spatial_swap import map_packed_image
from core.models.flux2.vae_alignment import forward_with_noise, _contexts, validate_noise_contract


def aligned(dtype=torch.float32):
    value=model(3,5)
    value.add_module("vae_latent_bridge",pair())
    return value.to(dtype)


@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
@pytest.mark.parametrize("driver_kind",["native","fast_wrapper","swapped","rewired"])
def test_actual_target_noise_and_clean_reference_projection_oracle(dtype,driver_kind):
    from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
    value=aligned(dtype)
    if driver_kind=="rewired":
        from core.models.flux2.residual_unet import restore_topology,native_config
        from core.rewiring.spec import build_spec,OPERATORS
        restore_topology(value,build_spec("flux2",[{"name":"double","block_count":3},{"name":"single","block_count":3}],
            [dict(id="s",stream="single",parent=None,start_block=1,end_block=2,down_factor=[2,2],**OPERATORS)],native_config(value)))
    offloader=SimpleNamespace(blocks_to_swap=1,num_dual_blocks=3,
        wait_for_block=lambda i:None,submit_move_blocks_forward=lambda i:None)
    driver=value if driver_kind=="native" else Flux2BlockSwapWrapper(value,offloader if driver_kind=="swapped" else None)
    noise=torch.randn(2,24,20,dtype=dtype)
    has_reference=driver_kind!="rewired"
    states=torch.randn(2,28 if has_reference else 24,20,dtype=dtype,requires_grad=True)
    times=torch.tensor([.2,.8],dtype=dtype)
    inputs=[];head=[]
    h1=value.x_embedder.register_forward_pre_hook(lambda m,args:inputs.append(args[0].clone()))
    h2=value.proj_out.register_forward_pre_hook(lambda m,args:head.append(args[0].clone()))
    options=dict(hidden_states=states,encoder_hidden_states=torch.randn(2,3,12,dtype=dtype),
        timestep=times,img_ids=torch.cat([ids(4,6),ids(2,2,10)],1) if has_reference else ids(4,6)[0],
        txt_ids=ids(1,3) if has_reference else ids(1,3)[0],
        guidance=torch.ones(2,dtype=dtype),return_dict=False)
    result=forward_with_noise(driver,noise,**options)[0]
    h1.remove();h2.remove()
    bridge=value.vae_latent_bridge
    zero=noise.new_zeros(2,4,20)
    expected=map_packed_image(bridge,states[:,:24],(4,6),sigma=times,noise=noise)
    if has_reference:
        expected=torch.cat([expected,map_packed_image(bridge,states[:,24:],(2,2),sigma=0,noise=zero)],1)
    torch.testing.assert_close(inputs[0],expected)
    raw=torch.nn.functional.linear(head[0],value.proj_out.weight,value.proj_out.bias)
    expected_output=map_packed_image(bridge,raw[:,:24],(4,6),output=True,noise=noise)
    if has_reference:
        expected_output=torch.cat([expected_output,map_packed_image(bridge,raw[:,24:],(2,2),output=True,noise=zero)],1)
    torch.testing.assert_close(result,expected_output)
    result.float().square().mean().backward()
    assert bridge.input.projection.weight.grad.abs().sum()>0
    assert bridge.output.projection.weight.grad.abs().sum()>0
    assert _contexts.get()==()


def test_kv_extract_cached_target_and_failure_cleanup():
    value=aligned()
    noise=torch.randn(1,6,20)
    options=dict(hidden_states=torch.randn(1,10,20),encoder_hidden_states=torch.randn(1,3,12),
        timestep=torch.tensor([.5]),img_ids=torch.cat([ids(2,2,10),ids(2,3)],1),txt_ids=ids(1,3),guidance=torch.ones(1))
    result=forward_with_noise(value,noise,**options,kv_cache_mode="extract",num_ref_tokens=4)
    assert result.sample.shape==(1,6,20)
    cached=dict(options,hidden_states=options["hidden_states"][:,4:],img_ids=ids(2,3))
    assert forward_with_noise(value,noise,**cached,kv_cache_mode="cached",kv_cache=result.kv_cache).sample.shape==(1,6,20)
    with pytest.raises(ValueError,match="clean"):
        forward_with_noise(value,noise,**options,ref_fixed_timestep=.1)
    assert _contexts.get()==()
    with pytest.raises(ValueError,match="target image"):
        forward_with_noise(value,torch.randn(1,10,20),**options)
    assert _contexts.get()==()
    with pytest.raises(ValueError,match="noise"):
        value(**options)
    assert _contexts.get()==()


def test_direct_keyword_and_ema_copy_use_their_own_bridge_without_noise_leak():
    import copy
    value=aligned()
    noise=torch.randn(1,6,20)
    options=dict(hidden_states=torch.randn_like(noise),encoder_hidden_states=torch.randn(1,3,12),
        timestep=torch.tensor([.4]),img_ids=ids(2,3),txt_ids=ids(1,3),guidance=torch.ones(1))
    expected=forward_with_noise(value,noise,**options).sample
    torch.testing.assert_close(value(**options,alignment_noise=noise).sample,expected)
    copied=copy.deepcopy(value)
    with torch.no_grad():copied.vae_latent_bridge.output.projection.bias.add_(.1)
    torch.testing.assert_close(forward_with_noise(copied,noise,**options).sample,expected-.1)
    assert _contexts.get()==()


@pytest.mark.parametrize("cfg",[False,True])
@pytest.mark.parametrize("gaussian",[False,True])
def test_nag_and_cfg_expand_the_same_target_noise(cfg,gaussian):
    from core.inference.nag_flux2 import Flux2NAGWrapper
    value=aligned() if gaussian else model(3,5)
    driver=Flux2NAGWrapper(value)
    batch=2 if cfg else 1
    text_batch=3 if cfg else 2
    noise=torch.randn(1,6,20)
    result=forward_with_noise(driver,noise,hidden_states=torch.randn(batch,6,20),
        encoder_hidden_states=torch.randn(text_batch,3,12),timestep=torch.full((batch,),.4),
        img_ids=ids(2,3),txt_ids=ids(1,3).expand(text_batch,-1,-1),guidance=torch.ones(batch),return_dict=False)[0]
    assert result.shape==(batch,6,20) and torch.isfinite(result).all()
    driver.restore()
    assert _contexts.get()==()


def test_gaussian_contract_rejects_untracked_processes_without_changing_legacy():
    value=aligned()
    with pytest.raises(ValueError,match="flow velocity"):
        validate_noise_contract(value,noise_process="vp")
    with pytest.raises(ValueError,match="deterministic"):
        validate_noise_contract(value,scheduler=SimpleNamespace(config=SimpleNamespace(stochastic_sampling=True)))
    validate_noise_contract(model(4,8),noise_process="vp",scheduler=SimpleNamespace(config=SimpleNamespace(stochastic_sampling=True)))


@pytest.mark.parametrize("cfg",[1.,2.])
def test_training_sample_keeps_reference_clean_and_decodes_only_target(tmp_path,cfg):
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    from core.training.ops import flux2_ops as ops
    from core.models.components.wiring import FLUX2_WIRING
    value=aligned(torch.bfloat16)
    reference=torch.full((1,5,4,6),.4)
    class VAE(torch.nn.Linear):
        dtype=torch.float32
        def encode(self,pixels):
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda:reference))
    vae=VAE(1,1)
    def patchify(x):
        return x.reshape(1,5,2,2,3,2).permute(0,1,3,5,2,4).reshape(1,20,2,3)
    trainer=SimpleNamespace(transformer=value,vae=vae,text_encoder=torch.nn.Linear(12,12),
        optimizer=torch.optim.SGD(value.parameters(),lr=.01),scheduler=FlowMatchEulerDiscreteScheduler(),
        device=torch.device("cpu"),vae_dtype=torch.float32,log_prefix="[SyntheticFlux]",
        wiring=FLUX2_WIRING.replace(latent_channels=5,vae_scale_factor=16),
        _flux2_patchify_latents_for_training=patchify)
    path=tmp_path/"synthetic.png";Image.new("RGB",(96,64),"blue").save(path)
    received=[];decoded=[]
    hook=value.register_forward_pre_hook(lambda m,args,kw:received.append(kw["hidden_states"].clone()),with_kwargs=True)
    def decode(t,latents,image_ids,*shape):
        decoded.append((latents.clone(),image_ids.clone()));return "synthetic_result"
    with patch.object(ops,"_flux2_encode_prompt_for_sample",return_value=(torch.randn(1,3,12,dtype=torch.bfloat16),ids(1,3))), \
         patch("core.models.components.vae_registry.normalize",side_effect=lambda x,*a:x), \
         patch.object(ops,"_decode_flux2_latents",side_effect=decode):
        result=ops.generate_sample(trainer,"synthetic",height=64,width=96,num_inference_steps=3,
            guidance_scale=cfg,seed=63,reference_image_path=str(path))
    hook.remove()
    assert result=="synthetic_result" and len(received)==3
    expected=ops._flux2_pack_latents_for_sample(patchify(reference)).to(torch.bfloat16)
    for state in received:
        torch.testing.assert_close(state[:,6:],expected.expand(state.shape[0],-1,-1))
    assert decoded[0][0].shape==(1,6,20) and decoded[0][1].shape[1]==6
    assert _contexts.get()==()


@pytest.mark.parametrize("mixed",[False,True])
@pytest.mark.parametrize("swapped",[False,True])
def test_training_ops_forward_backward_with_per_item_sigma_and_clean_references(mixed,swapped):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from core.training.ops import flux2_ops as ops
    from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
    dtype=torch.bfloat16 if mixed else torch.float32
    value=aligned(dtype)
    clean=torch.randn(2,6,20,dtype=dtype)
    noise=torch.randn_like(clean)
    times=torch.tensor([.2,.8])
    refs=[torch.randn(1,20,2,2,dtype=dtype) for _ in range(2)]
    trainer=SimpleNamespace(transformer=value,device=torch.device("cpu"),training_dtype=dtype,mixed_precision=mixed,
        noise_scheduler=FlowMatchEulerDiscreteScheduler(),noise_process="flow",prediction_target="velocity",
        snr_regularization_loss=None,energy_regularization_loss=None,reconstruction_loss_weight=0,
        _flux2_pack_latents=ops._flux2_pack_latents_for_sample,
        _flux2_prepare_latent_ids=ops._flux2_prepare_latent_ids_for_sample,log_prefix="[SyntheticFlux]")
    if swapped:
        offloader=SimpleNamespace(blocks_to_swap=1,num_dual_blocks=3,
            wait_for_block=lambda i:None,submit_move_blocks_forward=lambda i:None)
        trainer.flux2_transformer_wrapper=Flux2BlockSwapWrapper(value,offloader)
    with patch("core.training.mnt.training_noise_like",return_value=noise), \
         patch("core.training.ops.crop_decode_loss.crop_decode_or_x0_capture_needed",return_value=False):
        loss,*_=ops.train_step(trainer,clean,torch.randn(2,3,12,dtype=dtype),ids(2,3).expand(2,-1,-1).clone(),
            ids(1,3).expand(2,-1,-1).clone(),timesteps=times,reference_latents_nested=[[ref] for ref in refs])
    assert torch.isfinite(loss)
    loss.backward()
    assert value.vae_latent_bridge.input.projection.weight.grad.abs().sum()>0
    assert value.vae_latent_bridge.output.projection.weight.grad.abs().sum()>0
    assert _contexts.get()==()
