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


def model_for(channels):
    from core.models.minit2i.vendor.transformer import MiniT2IMMJiTModel
    model = MiniT2IMMJiTModel(image_size=32, patch_size=2, in_channels=channels,
        txt_input_size=16, hidden_size=32, txt_hidden_size=32, cond_vec_size=32,
        depth_double=3, txt_preamble_depth=1, num_heads=2, head_dim=16,
        pca_channels=8, prompt_length=5, vae_type="sdxl", noise_scale=1.)
    torch.nn.init.normal_(model.model.net.final_layer.linear.weight, std=.02)
    return model


@pytest.mark.parametrize("rewired", [False, True])
@pytest.mark.parametrize("old,new", [(4,8), (8,4)])
def test_x0_timestep_native_grid_gradients_and_restore(old, new, rewired):
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import MINIT2I_LATENT_IO
    from core.models.minit2i.residual_unet import native_config, restore_topology
    from core.models.minit2i.vae_alignment import latent_geometry, generation_token_stride
    from core.rewiring.spec import build_spec, OPERATORS
    model = model_for(old)
    resize_latent_io(model, MINIT2I_LATENT_IO, max(old,new))
    if rewired:
        restore_topology(model, build_spec("minit2i", [{"name":"main", "block_count":3}],
            [dict(id="s", stream="main", parent=None, start_block=1, end_block=2,
                down_factor=[2,2], **OPERATORS)], native_config(model)))
    original = deepcopy(model)
    install(model, declaration(old,new))
    pair = model.vae_latent_bridge
    pair.initialize_from_pair(identity(SpatialLatentBridge(new,old), .2),
        identity(SpatialLatentBridge(old,new), .3))
    image = torch.randn(2,new,8,12, requires_grad=True)
    t = torch.tensor([.0, .8])
    context, mask = torch.randn(2,5,16), torch.ones(2,5)
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
