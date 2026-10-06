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
