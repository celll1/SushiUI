"""Noise endpoints, frozen checkpoint anchors and live bridge gradients."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from core.models.components.gaussian_alignment import POLICY
from core.models.components.latent_adapter import SpatialAlignmentPair, validate_declaration
from core.models.components.vae_alignment import SpatialLatentBridge
from backend.tests.latent_adapter_contract_test import declaration


def pair(dtype=torch.float32):
    torch.manual_seed(218)
    spec = dict(declaration(3, 5), noise_policy=POLICY)
    value = SpatialAlignmentPair(spec).to(dtype=dtype)
    incoming = SpatialLatentBridge(5, 3).to(dtype=dtype)
    outgoing = SpatialLatentBridge(3, 5).to(dtype=dtype)
    for bridge in (incoming, outgoing):
        bridge.initialize(torch.randn(bridge.source_channels*9, bridge.target_channels)*.1,
                          torch.randn(bridge.target_channels)*.1)
    value.initialize_from_pair(incoming, outgoing)
    return value


@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clean_and_noise_endpoints_and_gradients(rank, dtype):
    value = pair(dtype)
    shape = (2, 5, 12, 16) if rank == 4 else (2, 5, 2, 12, 16)
    noise = torch.randn(shape, dtype=dtype)
    clean = torch.randn(shape, dtype=dtype, requires_grad=True)
    sigma = torch.tensor([.2, .8]);fraction = sigma.reshape(2, *([1]*(rank-1)))
    states = ((1-fraction)*clean + fraction*noise).to(dtype)
    actual = value.flow_state(states, sigma, noise=noise)
    white = F.conv2d(noise.float() if rank==4 else noise[:,:,0].float(), value.noise_alignment.white)
    endpoint = value.flow_state(noise, torch.ones(2), noise=noise)
    reference = endpoint[:,:3] if rank==4 else endpoint[:,:3,0]
    torch.testing.assert_close(reference.float(), white, atol=.035 if dtype==torch.bfloat16 else 1e-6, rtol=.035 if dtype==torch.bfloat16 else 1e-5)
    torch.testing.assert_close(value.flow_state(clean, torch.zeros(2), noise=noise), value.input(clean),
        atol=.002 if dtype==torch.bfloat16 else 1e-6, rtol=.02 if dtype==torch.bfloat16 else 1e-5)
    loss = value.flow_velocity(actual, convention='x0_minus_eps', noise=noise).float().square().mean()
    loss.backward()
    assert torch.isfinite(clean.grad).all()
    assert value.input.projection.weight.grad.abs().sum()>0
    assert value.output.projection.weight.grad.abs().sum()>0
    assert list(value.noise_alignment.parameters())==[]
    assert all(not b.requires_grad for b in value.noise_alignment.buffers())


@pytest.mark.parametrize("convention,sign", [('eps_minus_x0',1),('x0_minus_eps',-1)])
def test_frozen_original_head_reconstructs_target_noise_endpoint(convention, sign):
    value=pair();noise=torch.randn(2,5,12,16)
    white=F.conv2d(noise,value.noise_alignment.white)
    native=torch.zeros_like(noise);native[:,:3]=sign*white
    actual=value.flow_velocity(native,convention=convention,noise=noise)
    bias=value.output.projection.bias[None,:,None,None]
    torch.testing.assert_close(actual,sign*(noise-bias),atol=1e-6,rtol=1e-5)


def test_checkpoint_buffers_and_fp32_noise_definition_survive_dtype_changes():
    value=pair();original={k:v.clone() for k,v in value.noise_alignment.state_dict().items()}
    value.bfloat16().float()
    for key,tensor in original.items(): assert torch.equal(tensor,value.noise_alignment.state_dict()[key])
    restored=SpatialAlignmentPair(value.declaration);restored.load_state_dict(value.state_dict(),strict=True)
    assert restored.ready
    noise=torch.randn(2,5,8,8);sigma=torch.tensor([.1,.7])
    torch.testing.assert_close(restored.flow_state(noise,sigma,noise=noise),value.flow_state(noise,sigma,noise=noise))
    broken=deepcopy(value.state_dict());broken['noise_alignment.identity'][0]^=1
    with pytest.raises(ValueError,match='identity'):restored.load_state_dict(broken)
    broken=deepcopy(value.state_dict());broken.pop('noise_alignment.white')
    with pytest.raises(RuntimeError):restored.load_state_dict(broken,strict=True)


def test_noise_required_and_legacy_pair_remains_explicitly_uncorrected():
    value=pair();noise=torch.randn(2,5,8,8)
    with pytest.raises(ValueError,match='exposure'):value.flow_state(noise,.5)
    with pytest.raises(ValueError):value.flow_state(noise,.5,noise=noise[:1])
    legacy=SpatialAlignmentPair(declaration(3,5))
    assert not hasattr(legacy,'noise_alignment')
    assert not any('noise_alignment' in key for key in legacy.state_dict())
    with pytest.raises(ValueError,match='noise policy'):validate_declaration(dict(value.declaration,noise_policy='unknown'))
    with pytest.raises(ValueError,match='target channels'):SpatialAlignmentPair(dict(declaration(5,3),noise_policy=POLICY))


@pytest.mark.parametrize("order", ["outer","inner"])
@pytest.mark.parametrize("output", [False,True])
def test_packed_gaussian_noise_uses_the_same_pixel_order(order, output):
    from core.models.components.spatial_swap import map_packed_image
    value=pair();noise=torch.randn(2,5,8,12);image=torch.randn_like(noise)
    def pack(x):
        x=x.reshape(2,x.shape[1],4,2,6,2).permute(0,2,4,1,3,5)
        if order=='inner':x=x.permute(0,1,2,4,5,3)
        return x.reshape(2,24,-1)
    sigma=torch.tensor([.2,.8])
    actual=map_packed_image(value,pack(image),(4,6),sigma=sigma,output=output,
                            noise=pack(noise),channel_order=order)
    expected=(value.flow_velocity(image,convention='eps_minus_x0',noise=noise) if output
              else value.flow_state(image,sigma,noise=noise))
    torch.testing.assert_close(actual,pack(expected))
    with pytest.raises(ValueError,match='external image grid'):
        map_packed_image(value,pack(image),(4,6),sigma=sigma,noise=pack(noise)[:,:23])


def test_calibration_cannot_capture_a_singular_center_projection():
    value=pair()
    with torch.no_grad():value.input.projection.weight[:3,:,1,1].zero_()
    with pytest.raises(ValueError,match='full-row-rank'):value.noise_alignment.capture(value.input,value.output)


@pytest.mark.parametrize('rewired',[False,True])
def test_zimage_training_uses_inference_time_and_each_exposure_noise(rewired):
    from unittest.mock import patch
    with patch.object(torch.cuda,'get_device_capability',return_value=(8,9)), \
         patch.object(torch.cuda,'_lazy_init',return_value=None), \
         patch.object(torch._C,'_cuda_init',return_value=None):
        from backend.tests.zimage_spatial_swap_test import model
        from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
        native=model(3,5,rewired);native.vae_latent_bridge=pair()
        wrapper=BatchedZImageWrapperOptimized(native)
        noise=torch.randn(2,5,1,16,24);clean=torch.randn_like(noise)
        sigma=torch.tensor([.17,.81]);fraction=sigma[:,None,None,None,None]
        states=(1-fraction)*clean+fraction*noise
        captions=torch.randn(2,3,12);mask=torch.ones(2,3,dtype=torch.bool)
        actual=wrapper(states,sigma,captions,mask,alignment_noise=noise)[0]
        expected=torch.stack(native(list(states.unbind()),1-sigma,list(captions.unbind()),
            alignment_noise=list(noise.unbind()))[0])
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        explicit=torch.stack(native(list(states.unbind()),1-sigma,list(captions.unbind()),
            alignment_sigma=sigma,alignment_noise=list(noise.unbind()))[0])
        torch.testing.assert_close(actual,explicit,rtol=0,atol=0)
        actual.square().mean().backward()
        assert native.vae_latent_bridge.input.projection.weight.grad.abs().sum()>0
        assert native.vae_latent_bridge.output.projection.weight.grad.abs().sum()>0
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mode',['txt2img','img2img','inpaint'])
def test_zimage_sampler_supplies_cfg_noise_and_updates_inpaint_anchor(mode):
    from unittest.mock import patch
    from types import SimpleNamespace
    with patch.object(torch.cuda,'get_device_capability',return_value=(8,9)), \
         patch.object(torch.cuda,'_lazy_init',return_value=None), \
         patch.object(torch._C,'_cuda_init',return_value=None):
        from backend.tests.zimage_spatial_swap_test import model
        from core.pipeline_backends.zimage import ZImageMixin
        from diffusers import FlowMatchEulerDiscreteScheduler
        native=model(3,5);native.vae_latent_bridge=pair()
        manager=ZImageMixin();manager.device='cpu';manager.cancel_requested=False
        manager.zimage_components={'vae':SimpleNamespace(spatial_scale_factor=16,config=SimpleNamespace(latent_channels=5))}
        initial=torch.randn(1,5,8,8);seen=[]
        def capture(module,args,kwargs):seen.append(torch.stack(kwargs['alignment_noise']).detach().clone())
        handle=native.register_forward_pre_hook(capture,with_kwargs=True)
        kwargs={}
        if mode!='txt2img':kwargs.update(init_latents=torch.randn_like(initial),init_noise=initial)
        if mode=='inpaint':
            mask=torch.ones_like(initial);mask[:,:,:,:4]=0
            kwargs.update(mask_latent=mask,original_latents=torch.zeros_like(initial))
        output=manager._zimage_denoising_loop(native,FlowMatchEulerDiscreteScheduler(shift=6),
            [torch.randn(3,12)],[torch.randn(3,12)],128,128,3,4,True,
            torch.Generator().manual_seed(217),None,None,**kwargs)
        handle.remove()
        assert torch.isfinite(output).all() and len(seen)==2
        noise=(torch.randn(initial.shape,generator=torch.Generator().manual_seed(217)) if mode=='txt2img' else initial).unsqueeze(2)
        torch.testing.assert_close(seen[0],noise.repeat(2,1,1,1,1),rtol=0,atol=0)
        if mode=='inpaint':
            torch.testing.assert_close(seen[1][:,:,:,:,4:],seen[0][:,:,:,:,4:],rtol=0,atol=0)
            assert not torch.equal(seen[1][:,:,:,:,:4],seen[0][:,:,:,:,:4])
        else:torch.testing.assert_close(seen[1],seen[0],rtol=0,atol=0)
        assert not torch.cuda.is_initialized()
