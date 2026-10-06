"""Architecture-neutral codec geometry, diffusion conventions and posterior behavior."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.models.components.latent_adapter import (
    ImageCodecContract, SpatialSwapPlan, SpatialAlignmentPair, NormalizedFrameVAE,
    PairedSpatialCalibration, SpatialFlowNoiseSnapshot, calibration_grid,
)
from core.models.components.vae_alignment import SpatialLatentBridge


def declaration(old=4, new=8, scale=16, rank=4):
    return dict(version=1, method="paired_spatial", token_policy="target_native", source="registry:test",
                old_channels=old, new_channels=new, old_scale=8, new_scale=scale, input_channels=3,
                ndim=rank, norm="shift_scale", norm_pack=1)


def identity(bridge, bias=.3):
    w = torch.zeros(bridge.projection.in_channels * 9, bridge.projection.out_channels)
    for c in range(min(bridge.source_channels, bridge.target_channels)):
        w[c*9+4, c] = 1
    return bridge.initialize(w, torch.full((bridge.target_channels,), bias))


@pytest.mark.parametrize("old,new,scale", [(4,16,8), (16,64,16), (64,16,8), (16,32,32)])
def test_plan_keeps_target_grid_and_independent_channel_capacity(old, new, scale):
    plan = SpatialSwapPlan.build(declaration(old, new, scale), patch_size=2, downsample_factors=(2,4))
    assert plan.external_channels == new and plan.backbone_channels == max(old,new)
    assert plan.spatial_scale == scale and plan.pixel_alignment == scale*8
    pair = SpatialAlignmentPair(declaration(old,new,scale))
    pair.initialize_from_pair(identity(SpatialLatentBridge(new,old)), identity(SpatialLatentBridge(old,new)))
    value = torch.randn(2,new,8,12,requires_grad=True)
    mapped = pair.output(pair.input(value))
    assert mapped.shape == value.shape
    mapped.square().mean().backward()
    assert pair.input.projection.weight.grad.abs().sum() > 0
    restored = SpatialAlignmentPair(declaration(old,new,scale))
    restored.load_state_dict(pair.state_dict(), strict=True)
    assert restored.ready
    torch.testing.assert_close(restored.output(restored.input(value)), mapped)


@pytest.mark.parametrize("kind", ["epsilon", "x0", "v", "eps_minus_x0", "x0_minus_eps"])
def test_prediction_bias_matches_diffusion_convention(kind):
    bridge = identity(SpatialLatentBridge(3,3))
    x = torch.randn(2,3,5,7)
    sigma = torch.tensor([.2,.7])
    expected = {"epsilon":0., "x0":1., "v":-sigma[:,None,None,None],
                "eps_minus_x0":-1., "x0_minus_eps":1.}[kind]
    torch.testing.assert_close(bridge.prediction(x,kind=kind,sigma=sigma), x+.3*expected)
    alpha = (1-sigma.square()).sqrt()
    torch.testing.assert_close(bridge.diffusion_state(x,alpha=alpha), x+.3*alpha[:,None,None,None])


def test_prediction_and_alpha_reject_ambiguous_contracts():
    bridge = identity(SpatialLatentBridge(3,3))
    x = torch.randn(2,3,5,7)
    with pytest.raises(ValueError): bridge.prediction(x,kind="v")
    with pytest.raises(ValueError): bridge.diffusion_state(x,alpha=torch.ones(2,1,1,1))


@pytest.mark.parametrize("old,new", [(4,8), (8,4), (4,4)])
@pytest.mark.parametrize("rank", [4,5])
@pytest.mark.parametrize("convention", ["eps_minus_x0", "x0_minus_eps"])
@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
def test_flow_noise_complement_restores_external_clean_endpoint(old,new,rank,convention,dtype):
    pair = SpatialAlignmentPair(declaration(old,new,rank=rank))
    pair.initialize_from_pair(identity(SpatialLatentBridge(new,old),.2),
                              identity(SpatialLatentBridge(old,new),.3))
    pair.to(dtype=dtype)
    shape = (2,new,5,7) if rank==4 else (2,new,3,5,7)
    noise = torch.randn(shape,dtype=dtype,requires_grad=True)
    clean = pair.input(torch.randn(shape,dtype=dtype)).detach()
    native_noise = pair.input.epsilon(noise).detach()
    clean[:,old:] = native_noise[:,old:] = 0
    native_velocity = native_noise-clean if convention=="eps_minus_x0" else clean-native_noise
    corrected = pair.output.flow_velocity(native_velocity,convention=convention) + \
        pair.untrained_flow_noise_residual(noise,convention=convention)
    expected_clean = pair.output(clean)
    expected = noise.float()-expected_clean.float() if convention=="eps_minus_x0" else expected_clean.float()-noise.float()
    tolerance = .03 if dtype==torch.bfloat16 else 1e-5
    torch.testing.assert_close(corrected,expected,atol=tolerance,rtol=tolerance)
    assert not pair.untrained_flow_noise_residual(noise,convention=convention).requires_grad


def test_flow_noise_complement_refuses_unfitted_or_wrong_geometry():
    pair = SpatialAlignmentPair(declaration())
    with pytest.raises(ValueError,match="convention"):
        pair.untrained_flow_noise_residual(torch.randn(1,8,3,5),convention="epsilon")
    with pytest.raises(RuntimeError,match="calibration"):
        pair.untrained_flow_noise_residual(torch.randn(1,8,3,5),convention="eps_minus_x0")
    identity(pair.input); identity(pair.output)
    with pytest.raises(ValueError,match="declared channels"):
        pair.untrained_flow_noise_residual(torch.randn(1,4,3,5),convention="eps_minus_x0")


@pytest.mark.parametrize("old,new,rank", [(4,8,4),(8,4,5)])
def test_noise_snapshot_survives_bridge_training_and_checkpoint_restore(old,new,rank):
    import io
    pair=SpatialAlignmentPair(declaration(old,new,rank=rank))
    pair.initialize_from_pair(identity(SpatialLatentBridge(new,old)),
                              identity(SpatialLatentBridge(old,new)))
    noise=torch.randn((2,new,5,7) if rank==4 else (2,new,3,5,7),requires_grad=True)
    rng=torch.get_rng_state().clone()
    snapshot=SpatialFlowNoiseSnapshot.capture(pair)
    assert torch.equal(torch.get_rng_state(),rng)
    assert list(snapshot.parameters())==[]
    expected=pair.untrained_flow_noise_residual(noise,convention="eps_minus_x0")
    torch.testing.assert_close(snapshot.velocity_residual(noise,convention="eps_minus_x0"),expected)
    with torch.no_grad():
        pair.input.projection.weight.add_(.1)
        pair.output.projection.weight.mul_(2)
    stream=io.BytesIO();torch.save(snapshot.state_dict(),stream);stream.seek(0)
    restored=SpatialFlowNoiseSnapshot(snapshot.declaration)
    restored.load_state_dict(torch.load(stream,weights_only=True))
    actual=restored.velocity_residual(noise,convention="eps_minus_x0")
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(restored.velocity_residual(noise,convention="x0_minus_eps"),-expected)
    assert not actual.requires_grad


def test_noise_snapshot_rejects_other_codec_and_unfitted_state():
    pair=SpatialAlignmentPair(declaration())
    with pytest.raises(ValueError,match="calibration"):
        SpatialFlowNoiseSnapshot.capture(pair)
    identity(pair.input);identity(pair.output)
    snapshot=SpatialFlowNoiseSnapshot.capture(pair)
    other=SpatialFlowNoiseSnapshot(dict(pair.declaration,source="registry:other"))
    with pytest.raises(ValueError,match="identity"):
        other.load_state_dict(snapshot.state_dict())
    blank=SpatialFlowNoiseSnapshot(pair.declaration)
    with pytest.raises(RuntimeError,match="calibration"):
        blank.velocity_residual(torch.randn(1,8,3,5),convention="eps_minus_x0")


class ToyCodec(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        from diffusers.configuration_utils import FrozenDict
        self.config = FrozenDict(in_channels=3, scaling_factor=2.,shift_factor=.5)
        self.rank = rank

    def encode(self, pixels):
        assert pixels.ndim == self.rank
        return SimpleNamespace(latent_dist=SimpleNamespace(
            mode=lambda:pixels,
            sample=lambda generator=None:pixels+torch.randn(pixels.shape,generator=generator)))

    def decode(self, z, return_dict=False):
        return (z,)


@pytest.mark.parametrize("rank", [4,5])
def test_generic_codec_preserves_mode_sample_generator_and_alpha(rank):
    codec = NormalizedFrameVAE(ToyCodec(rank), declaration(3,3,rank=rank))
    pixels = torch.randn(2,4,2,8,12)
    posterior = codec.encode(pixels).latent_dist
    z = posterior.mode()
    torch.testing.assert_close(z, (pixels[:,:3]-.5)*2)
    a = posterior.sample(generator=torch.Generator().manual_seed(15))
    b = posterior.sample(generator=torch.Generator().manual_seed(15))
    torch.testing.assert_close(a,b)
    assert not torch.equal(a,z)
    result = codec.decode(z).sample
    torch.testing.assert_close(result[:,:3],pixels[:,:3])
    torch.testing.assert_close(result[:,3],torch.ones_like(result[:,3]))
    assert codec.config.latents_mean == [0.]*3
    assert codec.config.latents_std == [1.]*3


@pytest.mark.parametrize("rank", [4,5])
def test_streamed_calibration_resamples_teacher_only(rank):
    torch.manual_seed(284)
    pair = SpatialAlignmentPair(declaration(3,3))
    fit = PairedSpatialCalibration(pair,ridge=1e-5)
    coarse = torch.randn(2,3,8,12)
    source = torch.nn.functional.interpolate(coarse,size=(16,24),mode="nearest")
    if rank == 5: source,coarse = source.unsqueeze(2),coarse.unsqueeze(2)
    target = coarse*1.2+.1
    fit.add(source,target)
    fit.finish()
    assert pair.ready
    torch.testing.assert_close(pair.input(target), calibration_grid(source,(8,12)),atol=.001,rtol=.001)


def test_codec_contract_does_not_depend_on_parent_architecture():
    resolved = SimpleNamespace(latent_channels=32,scale_factor=16,ndim=5,norm="per_channel",norm_pack=1,
                               config=dict(input_channels=4))
    codec = ImageCodecContract.from_resolved(resolved)
    d = codec.declaration(source="model:arbitrary",source_identity="old",target_identity="new",old_channels=16,old_scale=8)
    assert d["new_channels"] == 32 and d["input_channels"] == 4
    resolved.ndim = 3
    with pytest.raises(ValueError,match="individual RGB/RGBA"): ImageCodecContract.from_resolved(resolved)


def test_pair_refuses_unfitted_or_repacked_calibration():
    pair = SpatialAlignmentPair(declaration())
    with pytest.raises(ValueError,match="fitted"):
        pair.initialize_from_pair(SpatialLatentBridge(8,4),SpatialLatentBridge(4,8))
    a,b = identity(SpatialLatentBridge(8,4)),identity(SpatialLatentBridge(4,8))
    a.source_pack = 2
    with pytest.raises(ValueError,match="repack"): pair.initialize_from_pair(a,b)
