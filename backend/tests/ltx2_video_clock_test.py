"""Target codec timestamps remain physical across LTX spatial rewiring."""

from types import SimpleNamespace

import pytest
import torch

from core.models.components.latent_adapter import ImageCodecContract
from core.models.components.video_latent_adapter import VideoCodecContract
from core.models.ltx2.video_clock import calibration_clip_frames, prepare_video_coords, video_pixel_frames


def codec(kind):
    h3 = kind == "minimax_h3_chunks_v1"
    return VideoCodecContract(ImageCodecContract(4,16,5,"per_channel" if h3 else "identity",1,3),
                              4 if h3 else (2 if kind == "causal_first_frame_v1" else 1),kind)


@pytest.mark.parametrize("kind,frames",[("causal_first_frame_v1",3),("independent_frames_v1",5),
                                       ("minimax_h3_chunks_v1",7),("minimax_h3_chunks_v1",12)])
def test_actual_rope_uses_codec_spans_and_per_sample_fps(kind,frames):
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed
    target = codec(kind)
    rope = LTX2AudioVideoRotaryPosEmbed(dim=32,scale_factors=(target.temporal_scale,16,16))
    native = rope.prepare_video_coords(2,frames,2,3,"cpu",fps=1)
    coords = prepare_video_coords(rope,target,2,frames,2,3,"cpu",fps=torch.tensor([24.,30.]))
    bounds = torch.tensor(target.frame_spans(frames),dtype=torch.float32).repeat_interleave(6,0)
    torch.testing.assert_close(coords[0,0],bounds/24,rtol=0,atol=0)
    torch.testing.assert_close(coords[1,0],bounds/30,rtol=0,atol=0)
    torch.testing.assert_close(coords[:,1:],native[:,1:],rtol=0,atol=0)
    assert video_pixel_frames(SimpleNamespace(contract=target),frames) == int(bounds[-1,-1])
    if kind != "minimax_h3_chunks_v1":
        torch.testing.assert_close(coords[0],rope.prepare_video_coords(1,frames,2,3,"cpu",fps=24)[0],rtol=0,atol=0)


@pytest.mark.parametrize("kind,expected",[("causal_first_frame_v1",9),("independent_frames_v1",9),
                                          ("minimax_h3_chunks_v1",73)])
def test_calibration_clip_satisfies_both_lattices(kind,expected):
    target = codec(kind)
    frames = calibration_clip_frames(8,target)
    assert frames == expected and (frames-1)%8 == 0
    assert target.pixel_frames(target.latent_frames(frames)) == frames


def test_h3_policy_is_resolved_by_official_class():
    from core.models.ltx2.vae_training import temporal_policy
    assert temporal_policy(5,"AutoencoderKLMiniMaxH3") == "minimax_h3_chunks_v1"


def test_native_frame_count_is_unchanged():
    assert video_pixel_frames(SimpleNamespace(temporal_compression_ratio=8),3) == 17


def test_training_buckets_use_nonuniform_target_lattice():
    from core.models.components.video_latent_adapter import NormalizedVideoVAE
    from core.training.arch.ltx2 import Ltx2ArchHandler
    module = torch.nn.Module()
    module.config = {"clip_length":17,"token_drop":3}
    module.temporal_compression_ratio = 4
    vae = NormalizedVideoVAE(module,codec("minimax_h3_chunks_v1"))
    trainer = SimpleNamespace(vae=vae,transformer=SimpleNamespace(
        config=SimpleNamespace(vae_scale_factors=(4,16,16),in_channels=4)))
    handler = Ltx2ArchHandler(trainer)
    assert handler.temporal.frame_multiple == 17 and handler.temporal.frame_offset == 5
    assert handler.temporal.min_decodable_frames == 22
    assert handler.temporal.latent_frames(39) == 12
    assert all(vae.contract.latent_frames(n) > 1 for n in handler.temporal.default_clip_lengths)


@pytest.mark.parametrize("fps",[0.,float("nan"),[24.,0.],[24.]])
def test_invalid_fps_is_refused(fps):
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed
    with pytest.raises(ValueError,match="FPS"):
        prepare_video_coords(LTX2AudioVideoRotaryPosEmbed(dim=32),codec("causal_first_frame_v1"),2,3,2,3,"cpu",fps=fps)

