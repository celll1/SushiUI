"""Chimera codec boundaries against path derivatives and residual reconstruction."""
from types import SimpleNamespace
import pytest
import torch
from core.models.components.vae_alignment import SpatialLatentBridge


def _bridge():
    bridge=SpatialLatentBridge(3,5).double()
    generator=torch.Generator().manual_seed(21)
    bridge.initialize(torch.randn(27,5,generator=generator,dtype=torch.float64)/9,
        torch.tensor([.1,-.2,.3,.4,-.5],dtype=torch.float64))
    return bridge


@pytest.mark.parametrize("frames",[False,True])
@pytest.mark.parametrize("kind",["flow_velocity","endpoint_observable_velocity"])
def test_velocity_is_derivative_of_affinely_mapped_native_grid_path(frames,kind):
    from core.models.sensenova_sdxl_chimera.flow import endpoint_observable_coefficients
    from core.models.sensenova_sdxl_chimera.vae_alignment import map_velocity
    bridge=_bridge()
    shape=(2,3,2,8,12) if frames else (2,3,8,12)
    clean=torch.randn(shape,dtype=torch.float64);noise=torch.randn_like(clean)
    time=torch.tensor([.2,.8],dtype=torch.float64)
    def coefficients(t):
        if kind=="flow_velocity":
            a=t.reshape(2,*([1]*(clean.ndim-1)))
            return a,1-a,torch.ones_like(a),-torch.ones_like(a)
        return endpoint_observable_coefficients(t,clean)
    def path(t):
        a,b,_,_=coefficients(t)
        return a*bridge(clean)+b*bridge.epsilon(noise)
    a,b,ap,bp=coefficients(time)
    torch.testing.assert_close(bridge.diffusion_state(a*clean+b*noise,alpha=a.reshape(2)),path(time))
    actual=map_velocity(SimpleNamespace(vae_latent_bridge=SimpleNamespace(output=bridge)),
        ap*clean+bp*noise,time,prediction={"type":kind})
    derivative=(path(time+1e-5)-path(time-1e-5))/(2e-5)
    torch.testing.assert_close(actual,derivative,atol=1e-8,rtol=1e-8)
    assert actual.shape[-2:]==(8,12) and actual.shape[1]==5
    actual.square().mean().backward()
    assert bridge.projection.weight.grad is not None and bridge.projection.bias.grad is not None


def test_residual_transfer_reconstructs_velocity_in_the_target_contract():
    from core.models.sensenova_sdxl_chimera.flow import endpoint_observable_reconstruct_velocity
    from core.models.sensenova_sdxl_chimera.vae_alignment import map_residual,map_velocity
    bridge=_bridge();owner=SimpleNamespace(vae_latent_bridge=SimpleNamespace(output=bridge))
    sample=torch.randn(2,3,8,12,dtype=torch.float64);residual=torch.randn_like(sample)
    target=torch.randn(2,5,8,12,dtype=torch.float64);time=torch.tensor([.3,.9],dtype=torch.float64)
    source={"type":"endpoint_observable_residual","latent_mean":[.1,.2,-.1],"latent_centered_second_moment":.4}
    external={"type":"endpoint_observable_residual","latent_mean":[0.,.2,.3,-.1,.4],"latent_centered_second_moment":.7}
    mapped=map_residual(owner,sample,residual,target,time,source_prediction=source,target_prediction=external)
    recovered=endpoint_observable_reconstruct_velocity(target,mapped,time,
        latent_mean=external["latent_mean"],latent_centered_second_moment=external["latent_centered_second_moment"])
    velocity=endpoint_observable_reconstruct_velocity(sample,residual,time,
        latent_mean=source["latent_mean"],latent_centered_second_moment=source["latent_centered_second_moment"])
    torch.testing.assert_close(recovered,map_velocity(owner,velocity,time,prediction=external))


def test_polar_prediction_cannot_silently_enter_the_linear_boundary():
    from core.models.sensenova_sdxl_chimera.vae_alignment import path_coefficients
    with pytest.raises(ValueError,match="direction/radius"):
        path_coefficients({"type":"polar_tangent_flow"},.5,torch.ones(1,4,8,8))


@pytest.mark.parametrize("channels", [2, 6])
@pytest.mark.parametrize("kind", ["flow_velocity", "endpoint_observable_velocity", "endpoint_observable_residual"])
def test_native_unet_boundary_keeps_target_grid_and_trains_both_bridges(channels, kind):
    from diffusers import UNet2DConditionModel
    from core.models.components.latent_adapter import SpatialAlignmentPair
    from core.models.sensenova_sdxl_chimera.vae_alignment import unet_forward
    from core.models.sensenova_sdxl_chimera.flow import (
        endpoint_observable_coefficients, endpoint_observable_preconditioning,
    )
    torch.manual_seed(91)
    width = max(4, channels)
    unet = UNet2DConditionModel(
        in_channels=width, out_channels=width, layers_per_block=1,
        block_out_channels=(8,), down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",), cross_attention_dim=6,
        attention_head_dim=2, norm_num_groups=4,
    )
    sample = torch.randn(2, channels, 8, 12)
    timestep = torch.tensor([.2, .8])
    context = torch.randn(2, 3, 6)
    pair = SpatialAlignmentPair(dict(
        version=1, method="paired_spatial", token_policy="target_native",
        old_channels=4, new_channels=channels, old_scale=8, new_scale=16,
        source="synthetic", ndim=4, norm="identity", norm_pack=1, input_channels=3,
    ))
    unet.vae_latent_bridge = pair
    for bridge in (pair.input, pair.output):
        bridge.initialize(torch.randn(bridge.source_channels*9, bridge.target_channels)/9,
                          torch.randn(bridge.target_channels)/10)
    prediction = dict(type=kind, latent_mean=[.1]*channels, latent_centered_second_moment=.7)
    source = dict(type=kind, latent_mean=[-.2]*width, latent_centered_second_moment=.4)
    unet._chimera_source_prediction = source
    if kind == "flow_velocity":
        alpha = timestep[:, None, None, None]
        derivative = torch.ones_like(alpha)
    else:
        alpha, _, derivative, _ = endpoint_observable_coefficients(timestep, sample)
    incoming = pair.input.projection
    internal = torch.nn.functional.conv2d(sample, incoming.weight, padding=1) + alpha*incoming.bias[None, :, None, None]
    raw = unet(internal, timestep, encoder_hidden_states=context, return_dict=False)[0]
    if kind == "endpoint_observable_residual":
        analytic, _ = endpoint_observable_preconditioning(internal, timestep,
            latent_mean=source["latent_mean"], latent_centered_second_moment=.4)
        raw = raw + analytic
    outgoing = pair.output.projection
    expected = torch.nn.functional.conv2d(raw, outgoing.weight, padding=1) + derivative*outgoing.bias[None, :, None, None]
    if kind == "endpoint_observable_residual":
        analytic, _ = endpoint_observable_preconditioning(sample, timestep,
            latent_mean=prediction["latent_mean"], latent_centered_second_moment=.7)
        expected = expected - analytic
    actual = unet_forward(unet, sample, timestep, prediction=prediction, encoder_hidden_states=context)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == sample.shape
    optimizer = torch.optim.SGD(unet.parameters(), lr=.01)
    before = outgoing.weight.detach().clone()
    actual.square().mean().backward()
    for bridge in (pair.input, pair.output):
        for parameter in bridge.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, outgoing.weight)


def test_unaligned_native_unet_is_unchanged_and_residual_requires_source_contract():
    from diffusers import UNet2DConditionModel
    from core.models.sensenova_sdxl_chimera.vae_alignment import unet_forward
    unet = UNet2DConditionModel(
        in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(8,), down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",), cross_attention_dim=6, norm_num_groups=4,
    )
    sample = torch.randn(1, 4, 8, 12)
    timestep = torch.tensor([.4])
    context = torch.randn(1, 3, 6)
    expected = unet(sample, timestep, encoder_hidden_states=context, return_dict=False)[0]
    actual = unet_forward(unet, sample, timestep,
        prediction={"type":"flow_velocity"}, encoder_hidden_states=context)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    unet.vae_latent_bridge = SimpleNamespace(input=SimpleNamespace(diffusion_state=lambda sample, **_: sample))
    with pytest.raises(ValueError, match="saved source contract"):
        unet_forward(unet, sample, timestep, prediction={"type":"endpoint_observable_residual"})


@pytest.mark.parametrize("mode", ["no_cfg", "sequential", "batched"])
def test_sampling_cfg_paths_integrate_the_cubic_boundary_bias(mode):
    from core.models.components.latent_adapter import SpatialAlignmentPair
    from core.models.sensenova_sdxl_chimera.pipeline_ops import sample_img2img_latents
    from sensenova_sdxl_chimera_p7_image_test import _ZeroUNet, _conditioning
    unet = _ZeroUNet()
    pair = SpatialAlignmentPair(dict(
        version=1, method="paired_spatial", token_policy="target_native",
        old_channels=4, new_channels=2, old_scale=8, new_scale=16,
        source="synthetic", ndim=4, norm="identity", norm_pack=1, input_channels=3,
    ))
    pair.input.initialize(torch.zeros(18, 4), torch.zeros(4))
    bias = torch.tensor([.1, -.2])
    pair.output.initialize(torch.zeros(36, 2), bias)
    unet.vae_latent_bridge = pair
    source = torch.zeros(1, 2, 8, 12)
    actual = sample_img2img_latents(
        unet, _conditioning(), None if mode == "no_cfg" else _conditioning(), source,
        steps=4, denoising_strength=1., cfg_scale=3., seed=17,
        cfg_mode="sequential" if mode == "no_cfg" else mode,
        prediction=dict(type="endpoint_observable_velocity", latent_mean=[0., 0.],
                        latent_centered_second_moment=1.),
    )
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(17))
    # alpha=t²(2-t): four Euler steps integrate alpha'=4t-3t² to 27/32.
    expected = noise + (27/32)*bias[None, :, None, None]
    torch.testing.assert_close(actual, expected)
