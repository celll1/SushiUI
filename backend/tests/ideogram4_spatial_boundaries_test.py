"""Inner-packed geometry, normalization and dual native diffusion boundaries."""
from types import SimpleNamespace

import pytest
import torch

from core.models.components.spatial_swap import install,map_packed_image
from latent_adapter_contract_test import declaration,identity


def _pack(image):
    b,c,h,w=image.shape
    return image.reshape(b,c,h//2,2,w//2,2).permute(0,2,4,3,5,1).reshape(b,h*w//4,c*4)


def _unpack(value,h,w):
    b,n,c=value.shape
    return value.reshape(b,h,w,2,2,c//4).permute(0,5,1,3,2,4).reshape(b,c//4,h*2,w*2)


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
def test_inner_packing_uses_channel_bias_and_correct_flow_sign(dtype):
    holder=torch.nn.Module()
    install(holder,declaration(3,5))
    pair=holder.vae_latent_bridge
    identity(pair.input,.2);identity(pair.output,.3)
    pair.to(dtype)
    image=torch.arange(2*5*8*12).reshape(2,5,8,12).to(dtype)
    sigma=torch.tensor([.25,.75],dtype=dtype)
    packed=map_packed_image(pair,_pack(image),(4,6),sigma=sigma,channel_order="inner")
    torch.testing.assert_close(_unpack(packed,4,6),pair.input.flow_state(image,sigma))
    output=map_packed_image(pair,packed,(4,6),output=True,channel_order="inner",convention="x0_minus_eps")
    torch.testing.assert_close(_unpack(output,4,6),pair.output.flow_velocity(_unpack(packed,4,6),convention="x0_minus_eps"))


@pytest.mark.parametrize("dtype", [torch.float32,torch.float16,torch.bfloat16])
def test_native_inner_batchnorm_statistics_match_shared_teacher_normalization(dtype):
    from core.models.components.vae_registry import normalize,denormalize
    from core.models.components.wiring import IDEOGRAM4_WIRING
    vae=SimpleNamespace(bn=SimpleNamespace(running_mean=torch.arange(12).float()/7,
        running_var=torch.arange(12).float()/5+1),config=SimpleNamespace(batch_norm_eps=1e-5))
    image=torch.randn(2,3,8,12).to(dtype)
    mean=vae.bn.running_mean.to(dtype).reshape(1,1,12)
    std=torch.sqrt(vae.bn.running_var+1e-5).to(dtype).reshape(1,1,12)
    expected=_unpack((_pack(image)-mean)/std,4,6)
    actual=normalize(image,vae,IDEOGRAM4_WIRING)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(denormalize(actual,vae,IDEOGRAM4_WIRING),
        _unpack(_pack(actual)*std+mean,4,6),atol=0,rtol=0)


@pytest.mark.parametrize("old,new", [(3,5),(5,3)])
@pytest.mark.parametrize("rewired", [False,True])
def test_dual_native_forward_keeps_target_grid_and_receives_extra_channel_gradients(old,new,rewired):
    from core.models.ideogram4.vendor.transformer import Ideogram4Transformer2DModel
    from core.models.ideogram4.ideogram4_pipeline_ops import _prepare_ids
    from core.models.ideogram4.vae_alignment import resize_pair,image_grid
    from core.models.ideogram4.residual_unet import native_config,restore_topology,pair_topology_metadata
    from core.rewiring.spec import build_spec,OPERATORS
    def model():
        value=Ideogram4Transformer2DModel(in_channels=old*4,num_layers=7,attention_head_dim=12,
            num_attention_heads=2,intermediate_size=32,adaln_dim=8,llm_features_dim=12,mrope_section=(2,2,2))
        if rewired:
            restore_topology(value,build_spec("ideogram4",[{"name":"main","block_count":7}],[
                dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
                dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)
            ],native_config(value)))
        return value
    cond,uncond=model(),model()
    resize_pair(cond,uncond,max(old,new))
    for owner in (cond,uncond):
        install(owner,declaration(old,new));identity(owner.vae_latent_bridge.input,.2);identity(owner.vae_latent_bridge.output,.3)
    assert bool(pair_topology_metadata(cond,uncond)) == rewired
    positions,segments,indicator=_prepare_ids([2,3],4,8,3,torch.device("cpu"))
    assert image_grid(positions,indicator)==(3,(4,8))
    image=torch.randn(2,35,new*4)
    caption=torch.randn(2,35,12)
    for owner,prefix in ((cond,3),(uncond,0)):
        cut=0 if prefix else 3
        output=owner(hidden_states=image[:,cut:],timestep=torch.tensor([.25,.75]),
            encoder_hidden_states=caption[:,cut:],position_ids=positions[:,cut:],
            segment_ids=segments[:,cut:],indicator=indicator[:,cut:],return_dict=False)[0]
        assert output.shape==image[:,cut:].shape
        (output-torch.randn_like(output)).square().mean().backward()
        assert owner.vae_latent_bridge.input.projection.weight.grad.abs().sum()>0
        assert owner.vae_latent_bridge.output.projection.weight.grad.abs().sum()>0
        if new>old:
            assert owner.input_proj.weight.grad.reshape(owner.hidden_size,4,new)[:,:,old:].abs().sum()>0
            assert owner.final_layer.linear.weight.grad.reshape(4,new,owner.hidden_size)[:,old:].abs().sum()>0


def test_cached_grid_is_revalidated_after_native_position_or_indicator_mutation():
    from core.models.ideogram4.ideogram4_pipeline_ops import _prepare_ids
    from core.models.ideogram4.vae_alignment import image_grid
    positions,segments,indicator=_prepare_ids([2],2,3,2,torch.device("cpu"))
    assert image_grid(positions,indicator)==(2,(2,3))
    positions[0,-1,1]+=1
    with pytest.raises(ValueError,match="rectangular"):
        image_grid(positions,indicator)


def test_dual_io_resize_checks_the_other_branch_before_replacing_any_weight():
    from ideogram4_full_training_test import _trainer
    from core.models.ideogram4.vae_alignment import resize_pair
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    trainer=_trainer()
    original=trainer.transformer.input_proj
    trainer.transformer_uncond.input_proj=Fp8Linear(8,24,bias=True,compute_dtype=torch.float32)
    with pytest.raises(NotImplementedError,match="weight-only quantized"):
        resize_pair(trainer.transformer,trainer.transformer_uncond,5)
    assert trainer.transformer.input_proj is original
    assert trainer.transformer.in_channels == 8
