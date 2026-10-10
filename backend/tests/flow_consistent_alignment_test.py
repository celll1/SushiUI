"""Flow closure, boundary behavior, precision and restoration of constrained maps."""
import pytest
import torch
from torch.nn import functional as F

from core.experiments.flow_consistent_alignment import AffineFlowCalibrator
from core.models.components.latent_adapter import SpatialAlignmentPair
from core.models.components.gaussian_alignment import POLICY
from latent_adapter_contract_test import declaration


def fitted(input_mode, output_kernel_size):
    generator = torch.Generator().manual_seed(218)
    target = torch.randn(4, 5, 12, 16, generator=generator)+.4
    matrix = torch.randn(3, 5, 1, 1, generator=generator)
    native = F.conv2d(target, matrix)+torch.tensor([.2, -.3, .5])[None, :, None, None]
    fitter = AffineFlowCalibrator(3, 5, ridge=1e-4, output_kernel_size=output_kernel_size)
    fitter.add(native[:2], target[:2]); fitter.add(native[2:], target[2:])
    before = torch.get_rng_state().clone()
    incoming, outgoing = fitter.finish(orthogonal_input=input_mode!='fitted',
                                       match_native_covariance=input_mode=='moment')
    assert torch.equal(before, torch.get_rng_state())
    assert fitter.metrics['composition_max_error'] < 1e-12
    assert fitter.metrics['bias_closure_max_error'] < 1e-12
    if input_mode=='orthogonal':
        a = incoming.projection.weight[:,:,1,1]
        torch.testing.assert_close(a@a.T,torch.eye(3),atol=1e-6,rtol=1e-6)
        assert abs(fitter.metrics['input_condition_number']-1) < 1e-12
    if input_mode=='moment':
        assert fitter.metrics['covariance_max_error'] < 1e-10
        predicted=incoming(target).movedim(1,-1).reshape(-1,3)
        source=native.movedim(1,-1).reshape(-1,3)
        torch.testing.assert_close(predicted.mean(0),source.mean(0),atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(torch.cov(predicted.T),torch.cov(source.T),atol=2e-5,rtol=2e-5)
    return incoming, outgoing


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('frames', [False, True])
@pytest.mark.parametrize('input_mode', ['fitted', 'orthogonal', 'moment'])
@pytest.mark.parametrize('output_kernel_size', [1, 3])
def test_actual_pair_flow_derivative_and_reload(dtype, frames, input_mode, output_kernel_size):
    incoming, outgoing = fitted(input_mode, output_kernel_size)
    pair = SpatialAlignmentPair(dict(declaration(3, 5), noise_policy=POLICY)).to(dtype=dtype)
    pair.initialize_from_pair(incoming, outgoing)
    shape = (2,5,2,8,10) if frames else (2,5,8,10)
    generator = torch.Generator().manual_seed(521)
    value = torch.randn(shape, generator=generator).to(dtype)
    noise = torch.randn(shape, generator=generator).to(dtype)
    native_velocity = torch.randn(shape, generator=generator).to(dtype)
    native_velocity[:, 3:] = 0
    target_velocity = pair.flow_velocity(native_velocity, convention='eps_minus_x0', noise=noise)
    # Differentiate affine state exactly, including the sigma-dependent bias/noise terms.
    derivative = pair.input.epsilon(target_velocity).float()
    derivative -= pair.input._bias_like(target_velocity).float()
    derivative[:, :3] += pair.noise_alignment.input_delta(noise)
    torch.testing.assert_close(derivative[:, :3], native_velocity[:, :3].float(),
                               atol=.05 if dtype==torch.bfloat16 else 2e-5,
                               rtol=.05 if dtype==torch.bfloat16 else 2e-5)
    restored = SpatialAlignmentPair(pair.declaration).to(dtype=dtype)
    restored.load_state_dict(pair.state_dict(), strict=True)
    assert restored.ready
    sigma = torch.tensor([.2,.8])
    torch.testing.assert_close(restored.flow_state(value,sigma,noise=noise),pair.flow_state(value,sigma,noise=noise))
    pair.requires_grad_(True)
    output = pair.flow_velocity(pair.flow_state(value,sigma,noise=noise),convention='eps_minus_x0',noise=noise)
    output.float().square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in pair.parameters())
    assert not list(pair.noise_alignment.parameters())


def test_rejects_degenerate_or_incompatible_pairs():
    for channels in ((5,3),(0,5),(True,5)):
        with pytest.raises(ValueError):AffineFlowCalibrator(*channels,ridge=1e-4)
    for ridge in (0,float('nan'),True):
        with pytest.raises(ValueError):AffineFlowCalibrator(3,5,ridge=ridge)
    for size in (0,2,True):
        with pytest.raises(ValueError,match='kernel'):AffineFlowCalibrator(3,5,ridge=1e-4,output_kernel_size=size)
    fitter = AffineFlowCalibrator(3,5,ridge=1e-4)
    with pytest.raises(ValueError,match='boolean'):fitter.finish(orthogonal_input=1)
    with pytest.raises(ValueError,match='boolean'):fitter.finish(match_native_covariance=1)
    with pytest.raises(ValueError,match='requires orthogonal'):fitter.finish(match_native_covariance=True)
    with pytest.raises(ValueError,match='insufficient'):fitter.finish()
    with pytest.raises(ValueError,match='equal'):fitter.add(torch.zeros(1,3,8,8),torch.zeros(1,5,4,4))
    with pytest.raises(ValueError,match='nonfinite'):fitter.add(torch.zeros(1,3,8,8),torch.full((1,5,8,8),float('nan')))
    fitter.add(torch.zeros(1,3,8,8),torch.randn(1,5,8,8))
    with pytest.raises(ValueError,match='full-row-rank'):fitter.finish()


@pytest.mark.parametrize('frames', [False, True])
def test_spatial_output_recovers_neighbor_features_without_flow_leakage(frames):
    generator = torch.Generator().manual_seed(473)
    native = torch.randn(4,3,12,16,generator=generator)
    shifted = F.pad(native[:,:2,:,:-1],(1,0))
    target = torch.cat((native,shifted),dim=1)
    if frames:
        native = torch.stack((native,2*native),dim=2)
        target = torch.stack((target,2*target),dim=2)
    spatial = AffineFlowCalibrator(3,5,ridge=1e-6,output_kernel_size=3)
    point = AffineFlowCalibrator(3,5,ridge=1e-6)
    for fitter in (spatial,point):fitter.add(native,target)
    incoming,outgoing = spatial.finish()
    _,point_out = point.finish()
    image = native if not frames else native[:,:,0]
    expected = target if not frames else target[:,:,0]
    error = (outgoing(image)-expected).square().mean()
    baseline = (point_out(image)-expected).square().mean()
    assert error < 1e-8 and baseline > .1
    off_center = outgoing.projection.weight.clone()
    off_center[:,:,1,1] = 0
    assert off_center.abs().max() > .9
    torch.testing.assert_close(incoming(outgoing(image)),image,atol=1e-5,rtol=1e-5)
