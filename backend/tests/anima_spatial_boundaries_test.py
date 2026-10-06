"""Synthetic Anima codec boundaries across native and residual execution."""
from copy import deepcopy
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
