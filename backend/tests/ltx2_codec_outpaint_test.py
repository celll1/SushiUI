"""Codec-aware placement through actual LTX denoising and H3 decoding."""

import numpy as np
import pytest
import torch

from ltx2_codec_pipeline_test import pipeline


@pytest.mark.parametrize("moving,index,start",[(True,5,17),(False,6,18)])
def test_offset_uses_actual_group_starts_and_compatible_chunk_phase(moving,index,start):
    from core.pipeline_backends.ltx2 import _snap_offset_to_latent_index,_trim_conditioning_sequence_frames
    pipe = pipeline()
    c = pipe.vae.contract
    assert _snap_offset_to_latent_index(18,4,12,codec_contract=c,moving=moving) == (index,start)
    assert _trim_conditioning_sequence_frames(17,24,39,4,codec_contract=c) == 22
    with pytest.raises(ValueError,match="cannot hold"):
        _trim_conditioning_sequence_frames(17,12,39,4,codec_contract=c)


def test_uniform_codec_placement_preserves_native_tie_breaking():
    from core.pipeline_backends.ltx2 import _snap_offset_to_latent_index
    from core.models.components.video_latent_adapter import VideoCodecContract
    from core.models.components.latent_adapter import ImageCodecContract
    c = VideoCodecContract(ImageCodecContract(2,16,5,"identity",1,3),8,"causal_first_frame_v1")
    for offset in range(0,34):
        assert _snap_offset_to_latent_index(offset,8,5,codec_contract=c) == _snap_offset_to_latent_index(offset,8,5)


def test_actual_outpaint_decodes_and_pastes_the_physical_preserved_span(monkeypatch):
    from core.pipeline_backends.ltx2 import LTX2Mixin
    base = pipeline(rewired=True)
    base.audio_vae.decode = lambda value,return_dict=False:(torch.zeros(1,1,value.shape[2],1),)
    class Vocoder(torch.nn.Module):
        def forward(self,value):
            return torch.zeros(value.shape[0],1,value.shape[2])
    base.register_modules(vocoder=Vocoder())
    class Owner(LTX2Mixin):
        pass
    owner = Owner()
    owner.ltx2_components = {key:getattr(base,key) for key in ("scheduler","vae","audio_vae","text_encoder",
        "tokenizer","connectors","transformer","vocoder")}
    owner.ltx2_components["pipeline"] = base
    monkeypatch.setattr(owner,"_ltx2_build_fbcache",lambda *a:None)
    monkeypatch.setattr(owner,"_ltx2_build_spectrum",lambda *a:(None,None))
    monkeypatch.setattr(owner,"_ltx2_rewiring_preflight",lambda *a:([],None))
    monkeypatch.setattr(owner,"_ltx2_runtime_int8",lambda *a,**k:None)
    monkeypatch.setattr(owner,"_ensure_ltx2_swap_and_offload",lambda *a,**k:base)
    for name in ("_load_lora_ltx2","_unload_lora_ltx2","_ltx2_begin_rewiring_metadata","_ltx2_capture_rewiring_metadata"):
        monkeypatch.setattr(owner,name,lambda *a,**k:None)
    cond = owner._ensure_ltx2_condition_pipeline()
    monkeypatch.setattr(cond,"encode_prompt",lambda **kw:(torch.ones(1,4,32),torch.ones(1,4),None,None))
    source = np.random.default_rng(4).integers(0,256,(22,32,32,3),dtype=np.uint8)
    frames,audio,rate,seed = owner._generate_vidoutpaint_ltx2(dict(prompt="synthetic",width=32,height=32,
        total_frames=39,input_offset_frames=18,num_inference_steps=2,guidance_scale=1,frame_rate=24,seed=7,
        outpaint_video_audio_mode="regenerate"),source,fps=24,input_audio=None)
    assert frames.shape == (39,32,32,3) and seed == 7
    np.testing.assert_array_equal(frames[17:39],source)
    assert audio is not None
