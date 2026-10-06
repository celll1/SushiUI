"""Target codec clocks, permuted media segments, and real H3 execution."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from core.models.components.latent_adapter import ImageCodecContract
from core.models.components.video_latent_adapter import VideoCodecContract
from core.models.minimax_h3 import h3_pipeline_ops as ops
from core.models.minimax_h3.video_alignment import VIDEO_POSITION_POLICY,VIDEO_POSITION_POLICY_KEY
from minimax_h3_video_alignment_test import setup


@pytest.fixture(autouse=True)
def suppress_import_cuda_context(monkeypatch):
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda *a,**k:(8,9))
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda *a,**k:None)
    monkeypatch.setattr(torch._C,"_cuda_init",lambda *a,**k:None)


def codec(policy):
    rank,norm,stride = (4,"shift_scale",1) if policy == "independent_frames_v1" else (
        (5,"identity",3) if policy == "causal_first_frame_v1" else (5,"per_channel",4))
    return VideoCodecContract(ImageCodecContract(4,32,rank,norm,1,3),stride,policy)


@pytest.mark.parametrize("policy",["independent_frames_v1","causal_first_frame_v1","minimax_h3_chunks_v1"])
def test_clock_matches_pixel_coordinates_and_preserves_audio_grid(policy):
    contract = codec(policy)
    frames = contract.latent_frames(22)
    layout = ops.build_packed_layout(3,frames,8,12,37,keyframe_anchors=("first","last",10),
                                     video_codec_contract=contract)
    start = layout["target_video_grid"]["start_row"]
    time = layout["position_ids"][start:,0].reshape(frames,24)
    offsets = ([0,1,5,9,13,17,18] if policy == "minimax_h3_chunks_v1" else
               [i*contract.temporal_scale for i in range(frames)])
    expected = torch.tensor([3+f*5/3 for f in offsets],dtype=torch.float32)
    torch.testing.assert_close(time,expected[:,None].expand_as(time),rtol=0,atol=0)
    anchor = layout["position_ids"][3:75,0].reshape(3,24)[:,0]
    torch.testing.assert_close(anchor,torch.tensor([3,38,3+10*5/3]),rtol=0,atol=0)
    audio = layout["position_ids"][layout["audio_indices"],0].reshape(2,37)
    torch.testing.assert_close(audio,torch.arange(3,40).float()[None].expand(2,37),rtol=0,atol=0)
    assert layout["video_codec_contract"] == asdict(contract)
    assert layout["video_position_policy"] == VIDEO_POSITION_POLICY
    for group in layout["video_codec_grids"]:
        selected = layout["video_indices"][group["rows"]]
        assert torch.equal(selected,selected.sort().values)


def test_native_codec_layout_is_bit_identical_and_keeps_legacy_float_arithmetic():
    contract = codec("minimax_h3_chunks_v1")
    for frames in (1,7,12,37):
        for builder,args in ((ops.build_packed_layout,(3,frames,8,12,37)),
                             (ops.build_ref2va_packed_layout,([1,1,1],[("video",True),("image",False)],
                                                            [(7,4,8),(1,8,12)],[74],frames,8,12,37))):
            native = builder(*args,keyframe_anchors=("first","last"))
            aligned = builder(*args,keyframe_anchors=("first","last"),video_codec_contract=contract)
            for key,value in native.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(aligned[key],value,rtol=0,atol=0)
                else:
                    assert aligned[key] == value


@pytest.mark.parametrize("spatial_pin",[False,True])
def test_codec_segments_restore_frame_major_order_after_target_pins(spatial_pin):
    contract = codec("independent_frames_v1")
    flags = dict(pinned_video_row_indices=(1,26,51)) if spatial_pin else dict(pinned_video_frames=(1,))
    layout = ops.build_packed_layout(3,3,8,12,3,video_codec_contract=contract,**flags)
    group, = layout["video_codec_grids"]
    start = layout["target_video_grid"]["start_row"]
    assert torch.equal(layout["video_indices"][group["rows"]],torch.arange(start,start+72))
    assert not torch.equal(group["rows"],torch.arange(72))
    assert layout["num_pinned_video_rows"] == (3 if spatial_pin else 24)


def test_reference_clock_and_segment_boundaries_survive_target_video_and_audio_pins():
    contract = codec("causal_first_frame_v1")
    layout = ops.build_ref2va_packed_layout([1,1,1],[("video",True),("image",False),("audio",False)],
        [(8,4,8),(1,8,12)],[74,4],8,8,12,37,video_codec_contract=contract,
        pinned_video_frames=(2,),pinned_audio_latents=(4,))
    # Reference video covers 22 pixel frames; its audio occupies 37 rotary units.
    origin = 3+37+1+2
    start = layout["target_video_grid"]["start_row"]
    torch.testing.assert_close(layout["position_ids"][start,0],torch.tensor(float(origin)),rtol=0,atol=0)
    assert layout["num_pinned_audio_rows"] == 2
    groups = layout["video_codec_grids"]
    assert [(g["frames"],g["height"],g["width"]) for g in groups] == [(8,4,8),(1,8,12),(8,8,12)]
    all_rows = torch.cat([g["rows"] for g in groups])
    assert torch.equal(all_rows.sort().values,torch.arange(layout["video_indices"].numel()))
    for group in groups:
        sequence = layout["video_indices"][group["rows"]]
        assert torch.equal(sequence,sequence.sort().values)
    anchored = ops.build_ref2va_packed_layout([1,1,1],[("video",True),("image",False),("audio",False)],
        [(8,4,8),(1,8,12)],[74,4],8,8,12,37,video_codec_contract=contract,keyframe_anchors=("last",))
    anchor = anchored["video_codec_grids"][2]
    anchor_start = int(anchored["video_indices"][anchor["rows"]][0])
    assert float(anchored["position_ids"][anchor_start,0]) == origin+35


@pytest.mark.parametrize("rewired",[False,True])
@pytest.mark.parametrize("policy",["independent_frames_v1","causal_first_frame_v1","minimax_h3_chunks_v1"])
def test_actual_sampler_preserves_reference_and_pinned_rows_on_target_codec(rewired,policy):
    from core.models.minimax_h3.vendor import MiniMaxH3Scheduler
    contract = codec(policy)
    wrapper,pair,_,_ = setup(4,rewired,torch.float32,contract)
    frames = contract.latent_frames(22)
    layout = ops.build_ref2va_packed_layout([1,1,1],[("image",False)],[(1,4,8)],[],
        frames,8,12,37,pinned_video_frames=(1,),pinned_audio_latents=(1,),video_codec_contract=contract)
    rows = layout["video_indices"].numel()
    video,audio = torch.randn(rows,16),torch.randn(74,6)
    held_video = video[:layout["num_condition_video_rows"]].clone()
    held_audio = audio[:2].clone()
    seen = []
    hook = wrapper.register_forward_pre_hook(lambda owner,args,kwargs:seen.append(kwargs),with_kwargs=True)
    try:
        with torch.no_grad():
            result = ops.denoise(wrapper,MiniMaxH3Scheduler(),MiniMaxH3Scheduler(),prompt_embeds=torch.randn(1,3,10),
                layout=layout,video_rows=video,audio_rows=audio,num_inference_steps=4,device="cpu")
    finally:
        hook.remove()
    assert len(seen) == 3
    assert rows == frames*24+8
    assert all(call["hidden_states"].shape[1] == rows and "video_codec_grids" in call for call in seen)
    torch.testing.assert_close(result[0][:held_video.shape[0]],held_video,rtol=0,atol=0)
    torch.testing.assert_close(result[1][:2],held_audio,rtol=0,atol=0)
    assert all(value.isfinite().all() for value in result)


def test_training_step_uses_target_clock_and_layout_cache_contract():
    from core.training.ops.minimax_h3_ops import train_step,_layout_for,_pixel_frames_for
    wrapper,pair,_,_ = setup(4,True,torch.float32)
    trainer = SimpleNamespace(device=torch.device("cpu"),training_dtype=torch.float32,transformer=wrapper,
                              audio_loss_weight=.7,reconstruction_loss_weight=0.,log_extra_metric=lambda *a:None)
    contract = wrapper.transformer._video_codec_contract
    layout = _layout_for(trainer,3,3,8,12,3)
    assert layout is _layout_for(trainer,3,3,8,12,3)
    assert _pixel_frames_for(trainer,3) == 3
    assert _pixel_frames_for(trainer,1) == 0
    assert layout["video_codec_contract"] == asdict(contract)
    loss,_,_ = train_step(trainer,torch.randn(1,4,3,8,12),torch.randn(1,3,10),
        dict(audio_latents=torch.randn(1,6,6),audio_present=torch.tensor([True])),timesteps=torch.tensor([.4]))
    loss.backward()
    assert torch.isfinite(loss) and pair.input.projection.weight.grad.abs().sum() > 0
    wrapper.transformer._video_codec_contract = codec("causal_first_frame_v1")
    changed = _layout_for(trainer,3,3,8,12,3)
    assert changed is not layout
    assert _pixel_frames_for(trainer,3) == 7


def test_invalid_codec_clock_refuses_before_scheduler_mutation():
    from core.models.minimax_h3.vendor import MiniMaxH3Scheduler
    from core.models.minimax_h3.residual_unet import native_config
    wrapper,_,_,_ = setup(4,True,torch.float32)
    layout = ops.build_packed_layout(3,3,8,12,3,video_codec_contract=wrapper.transformer._video_codec_contract)
    scheduler = MiniMaxH3Scheduler()
    with patch.object(scheduler,"set_shift",side_effect=AssertionError("scheduler mutated")):
        for bad,reason in ((dict(layout,video_position_policy="latent_index"),"position contract"),
                           (dict(layout,video_codec_grids=None),"explicit grids"),
                           (dict(layout,video_codec_grids=layout["video_codec_grids"]*2),"partition")):
            with pytest.raises(ValueError,match=reason):
                ops.denoise(wrapper,scheduler,MiniMaxH3Scheduler(),prompt_embeds=torch.randn(1,3,10),
                    layout=bad,video_rows=torch.randn(72,16),audio_rows=torch.randn(6,6),
                    num_inference_steps=4,device="cpu")
    config = dict(wrapper.transformer.config)
    config.pop(VIDEO_POSITION_POLICY_KEY)
    with pytest.raises(ValueError,match="position policy"):
        native_config(config)
    with pytest.raises(ValueError,match="5n"):
        ops.build_packed_layout(3,3,8,12,3,video_codec_contract=codec("minimax_h3_chunks_v1"))
