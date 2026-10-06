"""Physical frame spans for swapped codecs and H3 editing planners."""

import numpy as np
import pytest

from core.models.components.latent_adapter import ImageCodecContract
from core.models.components.video_latent_adapter import VideoCodecContract,temporal_spec_for_codec


def codec(policy):
    return VideoCodecContract(ImageCodecContract(2,32,5,"per_channel" if policy == "minimax_h3_chunks_v1" else "identity",1,3),
        4 if policy == "minimax_h3_chunks_v1" else (2 if policy == "causal_first_frame_v1" else 1),policy)


@pytest.mark.parametrize("policy",["minimax_h3_chunks_v1","causal_first_frame_v1","independent_frames_v1"])
def test_spans_cover_pixels_and_overlap_boundaries_are_exact(policy):
    from api.generation_utils import latent_frame_spans,video_continuation_overlap_lengths
    from core.models.components.wiring import temporal_spec_for_arch
    c = codec(policy)
    frames = 39 if policy == "minimax_h3_chunks_v1" else 25
    count = c.latent_frames(frames)
    spans = c.frame_spans(count)
    assert spans[0] == (0,1) and spans[-1][1] == frames
    assert all(left[1] == right[0] for left,right in zip(spans,spans[1:]))
    native = temporal_spec_for_arch("minimax_h3")
    resolved = temporal_spec_for_codec(native,c)
    assert resolved.min_frames == native.min_frames and resolved.trained_max_frames == native.trained_max_frames
    assert resolved.fps_fixed == native.fps_fixed and resolved.outpaint_placements == native.outpaint_placements
    assert resolved.pixel_align == 64
    assert latent_frame_spans(native,count,video_codec_contract=c) == spans
    valid = video_continuation_overlap_lengths(native,17,video_codec_contract=c)
    expected = (1,5,9,13,17) if policy == "minimax_h3_chunks_v1" else (
        tuple(range(1,18,2)) if policy == "causal_first_frame_v1" else tuple(range(1,18)))
    assert valid == expected
    if policy == "minimax_h3_chunks_v1":
        assert spans == latent_frame_spans(native,count)


def test_inpaint_and_outpaint_use_target_lattice_and_preserve_native_envelope(monkeypatch):
    from api.generation_utils import plan_video_inpaint_span,plan_video_outpaint_placement
    from api.error_handlers import ValidationError
    monkeypatch.setenv("SUSHI_TEMPORAL_SMOKE","1")
    c = codec("causal_first_frame_v1")
    plan = plan_video_inpaint_span(dict(regenerate_start_frame=2,regenerate_end_frame=4),"minimax_h3",
        clip_frames=25,video_codec_contract=c)
    assert plan["latent_frames"] == 13
    assert (plan["start_frame"],plan["end_frame"]) == (1,5)
    assert plan["regenerate_latent_frames"] == (1,2)
    assert 0 in plan["pinned_latent_frames"] and 12 in plan["pinned_latent_frames"]
    with pytest.raises(ValidationError):
        plan_video_inpaint_span(dict(regenerate_start_frame=2,regenerate_end_frame=4),"minimax_h3",
            clip_frames=24,video_codec_contract=c)
    out = plan_video_outpaint_placement(dict(total_frames=24),"minimax_h3",head_frames=3,video_codec_contract=c)
    assert out["generated_frames"] == 23 and out["total_frames"] == 25
    monkeypatch.delenv("SUSHI_TEMPORAL_SMOKE")
    with pytest.raises(ValidationError):
        plan_video_inpaint_span(dict(regenerate_start_frame=2,regenerate_end_frame=4),"minimax_h3",
            clip_frames=25,video_codec_contract=c)
    with pytest.raises(ValidationError,match="cannot reach"):
        plan_video_inpaint_span(dict(regenerate_start_frame=2,regenerate_end_frame=4),"minimax_h3",
            clip_frames=124,video_codec_contract=c)


def test_h3_pinned_tail_uses_target_spans_without_mutating_original_pixels(monkeypatch):
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin
    from core.models.minimax_h3 import generation_codec
    c = codec("causal_first_frame_v1")
    monkeypatch.setattr(generation_codec,"generation_contract",lambda components:c)
    runner = MiniMaxH3Mixin()
    runner.minimax_h3_components = {}
    head = np.arange(10,dtype=np.uint8)[:,None,None,None].repeat(3,axis=3)
    pinned,source = runner._minimax_h3_pinned_tail_video(head,"minimax_h3",generated_frames=25,overlap_frames=5)
    assert pinned == (0,1,2)
    np.testing.assert_array_equal(source[:5],head[-5:])
    np.testing.assert_array_equal(source[5:],np.repeat(head[-1:],20,axis=0))
    assert head.shape[0] == 10
    from api.error_handlers import ValidationError
    with pytest.raises(ValidationError):
        runner._minimax_h3_pinned_tail_video(head,"minimax_h3",generated_frames=25,overlap_frames=4)
