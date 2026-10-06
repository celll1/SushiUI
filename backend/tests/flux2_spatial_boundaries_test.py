"""Native Flux2 geometry, clean references and shared projection boundaries."""
from unittest.mock import patch

import pytest
import torch

from backend.tests.latent_adapter_contract_test import declaration,identity
from core.models.components.spatial_swap import install,map_packed_image


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
         patch.object(torch.cuda,"_lazy_init",return_value=None), \
         patch.object(torch._C,"_cuda_init",return_value=None):
        yield


def ids(h,w,time=0):
    value=torch.cartesian_prod(torch.tensor([time]),torch.arange(h),torch.arange(w),torch.tensor([0]))
    return value[None]


def model(old,new):
    from diffusers import Flux2Transformer2DModel
    from core.models.flux2.vae_alignment import bind_boundaries
    value=Flux2Transformer2DModel(num_layers=3,num_single_layers=3,in_channels=max(old,new)*4,
        num_attention_heads=2,attention_head_dim=8,axes_dims_rope=(2,2,2,2),
        joint_attention_dim=12,timestep_guidance_channels=8,mlp_ratio=1,guidance_embeds=True)
    install(value,declaration(old,new))
    identity(value.vae_latent_bridge.input,.2); identity(value.vae_latent_bridge.output,.3)
    bind_boundaries(value)
    return value


@pytest.mark.parametrize("old,new",[(4,8),(8,4)])
@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_native_target_and_reference_use_distinct_sigma(old,new,dtype):
    value=model(old,new).to(dtype)
    x=torch.randn(2,10,new*4,dtype=dtype,requires_grad=True)
    image_ids=torch.cat((ids(2,3),ids(2,2,10)),dim=1)
    sigma=torch.tensor([.2,.8],dtype=dtype)
    captured=[]
    handle=value.x_embedder.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
    result=value(x,torch.randn(2,3,12,dtype=dtype),sigma,image_ids,ids(1,3),torch.ones(2,dtype=dtype)).sample
    handle.remove()
    pair=value.vae_latent_bridge
    expected=torch.cat((map_packed_image(pair,x[:,:6],(2,3),sigma=sigma),
        map_packed_image(pair,x[:,6:],(2,2),sigma=0)),dim=1)
    torch.testing.assert_close(captured[0],expected)
    assert result.shape==x.shape
    result.float().square().mean().backward()
    for grad in (x.grad,pair.input.projection.weight.grad,pair.output.projection.weight.grad,value.x_embedder.weight.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
    from core.models.flux2.vae_alignment import _contexts
    assert _contexts.get()==()


def test_kv_extract_returns_target_grid_and_context_is_cleaned_after_failure():
    value=model(4,8)
    x=torch.randn(1,10,32)
    image_ids=torch.cat((ids(2,2,10),ids(2,3)),dim=1)
    text=torch.randn(1,3,12); times=torch.tensor([.7]); guidance=torch.ones(1)
    output=value(x,text,times,image_ids,ids(1,3),guidance,kv_cache_mode="extract",num_ref_tokens=4)
    assert output.sample.shape==(1,6,32) and output.kv_cache is not None
    cached=value(x[:,4:],text,times,ids(2,3),ids(1,3),guidance,
        kv_cache_mode="cached",kv_cache=output.kv_cache)
    assert cached.sample.shape==(1,6,32)
    with pytest.raises(ValueError,match="rectangular"):
        bad=ids(2,3);bad[0,0,1]=1
        value(x[:,4:],text,times,bad,ids(1,3),guidance)
    from core.models.flux2.vae_alignment import _contexts
    assert _contexts.get()==()


def test_rewired_block_swap_direct_projection_uses_same_boundary_context():
    from core.models.flux2.residual_unet import restore_topology,native_config
    from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
    from core.rewiring.spec import build_spec,OPERATORS
    value=model(4,8)
    spec=build_spec("flux2",[{"name":"double","block_count":3},{"name":"single","block_count":3}],
        [dict(id="s",stream="single",parent=None,start_block=1,end_block=2,down_factor=[2,2],**OPERATORS)],native_config(value))
    restore_topology(value,spec)
    wrapper=Flux2BlockSwapWrapper(value,None)
    x=torch.randn(1,24,32,requires_grad=True)
    options=dict(hidden_states=x,encoder_hidden_states=torch.randn(1,3,12),timestep=torch.tensor([.4]),
        img_ids=ids(4,6),txt_ids=ids(1,3),guidance=torch.ones(1),return_dict=False)
    expected=value(**options)[0]
    actual=wrapper(**options)[0]
    torch.testing.assert_close(actual,expected)
    actual.square().mean().backward()
    assert value.vae_latent_bridge.input.projection.weight.grad.abs().sum()>0
    assert not torch.cuda.is_initialized()


def test_ema_copy_and_projection_replacement_keep_one_bridge():
    import copy
    value=model(4,8)
    options=dict(hidden_states=torch.randn(1,6,32),encoder_hidden_states=torch.randn(1,3,12),
        timestep=torch.tensor([.5]),img_ids=ids(2,3),txt_ids=ids(1,3),guidance=torch.ones(1))
    expected=value(**options).sample
    copied=copy.deepcopy(value)
    with torch.no_grad():copied.vae_latent_bridge.output.projection.bias.add_(.1)
    torch.testing.assert_close(copied(**options).sample,expected-.1)
    replacement=torch.nn.Linear(value.x_embedder.in_features,value.x_embedder.out_features,bias=False)
    replacement.load_state_dict(value.x_embedder.state_dict())
    value.x_embedder=replacement
    torch.testing.assert_close(value(**options).sample,expected)
    torch.testing.assert_close(value(**options).sample,expected)
    assert len(value.x_embedder._forward_pre_hooks)==1
    from core.models.flux2.vae_alignment import _contexts
    assert _contexts.get()==()


@pytest.mark.parametrize("batch",[1,2])
@pytest.mark.parametrize("checkpoint",[False,True])
def test_dense_block_swap_uses_projection_boundaries_without_native_forward(batch,checkpoint):
    from types import SimpleNamespace
    from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
    value=model(4,8)
    if checkpoint:value.enable_gradient_checkpointing()
    offloader=SimpleNamespace(blocks_to_swap=1,num_dual_blocks=3,
        wait_for_block=lambda index:None,submit_move_blocks_forward=lambda index:None)
    wrapper=Flux2BlockSwapWrapper(value,offloader)
    options=dict(hidden_states=torch.randn(batch,6,32,requires_grad=True),encoder_hidden_states=torch.randn(batch,3,12),
        timestep=torch.linspace(.2,.8,batch),img_ids=ids(2,3),txt_ids=ids(1,3),guidance=torch.ones(batch),return_dict=False)
    actual=wrapper(**options)[0]
    torch.testing.assert_close(actual,value(**options)[0])
    torch.testing.assert_close(Flux2BlockSwapWrapper(value)(**options)[0],actual)
    actual.square().mean().backward()
    assert torch.isfinite(value.vae_latent_bridge.input.projection.weight.grad).all()
