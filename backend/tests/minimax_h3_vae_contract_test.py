"""Real small H3 codec geometry and normalization on synthetic RGB clips."""

from dataclasses import asdict

import pytest
import torch

from core.models.components.latent_adapter import ImageCodecContract
from core.models.components.video_latent_adapter import VideoCodecContract, NormalizedVideoVAE
from core.models.common import vae_source as vs


def small_codec():
    from core.models.minimax_h3.vendor.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    codec = AutoencoderKLMiniMaxH3(latent_channels=2, block_out_channels=(4,4,4), layers_per_block=1,
        spatial_downsample_factors=(2,2,2), temporal_downsample_factors=(1,2,2), norm_num_groups=1,
        decoder_num_layers=1, decoder_num_attention_heads=1, decoder_attention_head_dim=8,
        decoder_num_register_tokens=1, latents_mean=(.2,.3), latents_std=(.8,1.2))
    codec.disable_tiling()
    return codec.eval()


def contract():
    return VideoCodecContract(ImageCodecContract(2,8,5,"per_channel",1,3),4,"minimax_h3_chunks_v1")


@pytest.mark.parametrize("frames,latent_frames", [(1,1),(22,7),(39,12)])
@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
def test_real_chunk_geometry_and_pixel_latent_normalization(frames, latent_frames, dtype):
    codec = small_codec().to(dtype=dtype)
    c = contract()
    assert c.latent_frames(frames) == latent_frames and c.pixel_frames(latent_frames) == frames
    assert VideoCodecContract(ImageCodecContract(**asdict(c.image)),c.temporal_scale,c.temporal_policy) == c
    facade = NormalizedVideoVAE(codec,c)
    pixels = torch.rand(1,3,frames,32,32,generator=torch.Generator().manual_seed(53)).to(dtype)*2-1
    mean = pixels.new_tensor([.485,.456,.406]).view(1,3,1,1,1)
    std = pixels.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
    with torch.no_grad():
        raw = codec.encode(((pixels+1)*.5-mean)/std).latent_dist.mode()
        expected = (raw-raw.new_tensor([.2,.3]).view(1,2,1,1,1))/raw.new_tensor([.8,1.2]).view(1,2,1,1,1)
        actual = facade.encode(pixels).latent_dist.mode()
        assert actual.shape == (1,2,latent_frames,4,4)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        # Compare to the same denormalized latent, including BF16 rounding.
        reconstructed = actual*actual.new_tensor([.8,1.2]).view(1,2,1,1,1)+actual.new_tensor([.2,.3]).view(1,2,1,1,1)
        expected_pixels = (codec.decode(reconstructed).sample*std+mean)*2-1
        decoded = facade.decode(actual).sample
        assert decoded.shape == pixels.shape
        torch.testing.assert_close(decoded,expected_pixels,rtol=0,atol=0)


def test_chunk_contract_rejects_bad_geometry_before_codec_call():
    c = contract()
    for frames in (0,2,5,17,21,23,38,40):
        with pytest.raises(ValueError):
            c.latent_frames(frames)
    for frames in (0,2,5,6,8,11,13):
        with pytest.raises(ValueError):
            c.pixel_frames(frames)
    codec = small_codec()
    codec.register_to_config(token_drop=2)
    with pytest.raises(ValueError,match="config disagrees"):
        NormalizedVideoVAE(codec,c)


def test_real_codec_source_resolution_and_restore(tmp_path, monkeypatch):
    codec = small_codec()
    codec.save_pretrained(tmp_path)
    resolved = vs.resolve_vae_source(f"file:{tmp_path}",load_weights=True)
    assert (resolved.latent_channels,resolved.scale_factor,resolved.scale_temporal,resolved.ndim) == (2,8,4,5)
    assert resolved.facts()["frame_policy"] == "minimax_h3_chunks_v1"
    restored = resolved.load_module(torch.float32)
    from core.training.vae_swap import _materialized_config
    assert vs.latent_space_hash(resolved.content_hash,_materialized_config(restored,resolved.config)) == resolved.latent_hash
    for key,value in codec.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
    config = dict(codec.config,_class_name="AutoencoderKLMiniMaxH3")
    assert vs.latent_space_hash("weights",config) != vs.latent_space_hash("weights",dict(config,token_drop=2))
    monkeypatch.setattr(type(codec),"from_pretrained",lambda *a,**k:pytest.fail("inspection allocated codec"))
    assert vs.resolve_vae_source(f"file:{tmp_path}",load_weights=False).state_dict is None


@pytest.mark.parametrize("change", [{"token_drop":2},{"latent_channels":3},{"latents_std":[0.,1.]},
    {"temporal_downsample_factors":[1,2,1]},{"spatial_downsample_factors":[2,2,1]},
    {"decoder_attention_head_dim":16},{"scaling_factor":2.}])
def test_source_header_and_normalization_refusals(change):
    codec = small_codec()
    config = dict(codec.config,_class_name="AutoencoderKLMiniMaxH3",**change)
    with pytest.raises(vs.VaeSourceError):
        vs._resolve_facts(source="synthetic",config=config,family="custom",declared={},
            shapes={k:tuple(v.shape) for k,v in codec.state_dict().items()})


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
def test_independent_image_facade_matches_native_still_codec(dtype):
    from core.models.components.latent_adapter import NormalizedImageVAE
    codec = small_codec().to(dtype=dtype)
    d = contract().image.declaration(source="synthetic",source_identity="old",target_identity="new",
        old_channels=4,old_scale=8)
    image_vae = NormalizedImageVAE(codec,d)
    video_vae = NormalizedVideoVAE(codec,contract())
    pixels = torch.rand(2,3,32,32,generator=torch.Generator().manual_seed(87)).to(dtype)*2-1
    with torch.no_grad():
        actual = image_vae.encode(pixels).latent_dist.mode()
        expected = video_vae.encode(pixels.unsqueeze(2)).latent_dist.mode().squeeze(2)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        decoded = image_vae.decode(actual).sample
        expected_pixels = video_vae.decode(expected.unsqueeze(2)).sample.squeeze(2)
        torch.testing.assert_close(decoded,expected_pixels,rtol=0,atol=0)
