"""Actual target-clock denoising on a tiny LTX model, with synthetic conditioning."""

from types import SimpleNamespace

import pytest
import torch


class Connector(torch.nn.Module):
    device = torch.device("cpu")
    dtype = torch.float32
    def forward(self, embeds, mask, padding_side="left"):
        return embeds, embeds[..., :16], mask


def pipeline(rewired=False):
    from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
    from core.models.components.latent_adapter import ImageCodecContract, SpatialAlignmentPair
    from core.models.components.video_latent_adapter import VideoCodecContract, NormalizedVideoVAE
    from core.models.ltx2.video_alignment import install_video_alignment
    from core.models.ltx2.codec_pipelines.geometry import install_pipeline_clock
    model = LTX2VideoTransformer3DModel(in_channels=4,out_channels=4,num_attention_heads=2,
        attention_head_dim=16,cross_attention_dim=32,audio_in_channels=4,audio_out_channels=4,
        audio_num_attention_heads=2,audio_attention_head_dim=8,audio_cross_attention_dim=16,
        num_layers=4,use_prompt_embeddings=False,cross_attn_mod=True,audio_cross_attn_mod=True)
    if rewired:
        from core.models.ltx2.residual_unet import native_config,wrap_transformer
        from core.rewiring.spec import build_spec,OPERATORS
        model = wrap_transformer(model,build_spec("ltx2",[{"name":"main","block_count":4}],
            [dict(id="region",stream="main",parent=None,start_block=1,end_block=3,down_factor=[2,2],**OPERATORS)],native_config(model.config)))
    image = ImageCodecContract(2,8,5,"per_channel",1,3)
    contract = VideoCodecContract(image,4,"minimax_h3_chunks_v1")
    pair = SpatialAlignmentPair(image.declaration(source="synthetic",source_identity="old",target_identity="new",old_channels=4,old_scale=32))
    for bridge in (pair.input,pair.output):
        bridge.initialize(torch.randn(bridge.source_channels*9,bridge.target_channels)/9,torch.zeros(bridge.target_channels))
    model = install_video_alignment(model,pair,contract)
    from minimax_h3_vae_contract_test import small_codec
    video = small_codec()
    vae = NormalizedVideoVAE(video,contract)
    audio = torch.nn.Module()
    audio.device, audio.dtype = torch.device("cpu"), torch.float32
    audio.latents_mean, audio.latents_std = torch.zeros(4), torch.ones(4)
    audio.config = SimpleNamespace(sample_rate=24,mel_hop_length=1,mel_bins=1,latent_channels=4)
    audio.mel_compression_ratio = audio.temporal_compression_ratio = 1
    pipe = LTX2Pipeline(scheduler=FlowMatchEulerDiscreteScheduler(),vae=vae,audio_vae=audio,
        text_encoder=None,tokenizer=None,connectors=Connector(),transformer=model,vocoder=None)
    return install_pipeline_clock(pipe)


@pytest.mark.parametrize("rewired",[False,True])
@pytest.mark.parametrize("mode",["txt","image","condition"])
def test_actual_txt_denoising_retains_h3_grid_and_physical_audio_duration(rewired,mode):
    pipe = pipeline(rewired)
    extra = {}
    if mode != "txt":
        from diffusers import LTX2ImageToVideoPipeline,LTX2ConditionPipeline
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from core.models.ltx2.codec_pipelines.geometry import pipeline_class
        native = LTX2ImageToVideoPipeline if mode == "image" else LTX2ConditionPipeline
        cls = pipeline_class(native,pipe.vae)
        pipe = cls(**{key:getattr(pipe,key) for key in ("scheduler","vae","audio_vae","text_encoder",
            "tokenizer","connectors","transformer","vocoder")})
        if mode == "image":
            extra = {"image":torch.zeros(3,32,32)}
        else:
            extra = {"conditions":[LTX2VideoCondition(frames=torch.zeros(3,32,32),index=0,strength=1.)]}
    coords = []
    hook = pipe.transformer.register_forward_pre_hook(lambda m,args,kw:coords.append(kw["video_coords"].clone()),with_kwargs=True)
    try:
        video,audio = pipe(prompt_embeds=torch.randn(1,4,32),prompt_attention_mask=torch.ones(1,4),
            height=32,width=32,num_frames=22,frame_rate=24,num_inference_steps=2,
            guidance_scale=1,stg_scale=0,modality_scale=1,output_type="latent",return_dict=False,
            generator=torch.Generator().manual_seed(2),**extra)
    finally:
        hook.remove()
    assert video.shape == (1,2,7,4,4)
    assert audio.shape == (1,4,22,1)
    expected = torch.tensor(pipe.vae.contract.frame_spans(7),dtype=torch.float32).repeat_interleave(16,0)/24
    assert len(coords) == 2
    for value in coords:
        torch.testing.assert_close(value[0,0],expected,rtol=0,atol=0)
    if mode != "txt":
        with torch.no_grad():
            held = pipe.vae.encode(torch.full((1,3,1,32,32),-1.)).latent_dist.mode()
        torch.testing.assert_close(video[:,:,:1],held,rtol=0,atol=0)


def test_native_pipeline_selection_is_unchanged():
    from diffusers import LTX2Pipeline
    from core.models.ltx2.codec_pipelines.geometry import pipeline_class
    assert pipeline_class(LTX2Pipeline,None) is LTX2Pipeline


def test_generated_methods_have_only_reviewed_geometry_changes():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("generate_ltx2",root/"tools/generate_ltx2_codec_pipelines.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    for module,cls in generator.MODULES.items():
        expected = generator.generate(module,cls,write=False)
        actual = (generator.TARGET/(module+".py")).read_text(encoding="utf-8")
        assert actual == expected


def test_nonuniform_video_condition_clip_uses_physical_start_and_retains_latents():
    from diffusers import LTX2ConditionPipeline
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
    from core.models.ltx2.codec_pipelines.geometry import pipeline_class
    base = pipeline()
    cls = pipeline_class(LTX2ConditionPipeline,base.vae)
    pipe = cls(**{key:getattr(base,key) for key in ("scheduler","vae","audio_vae","text_encoder",
        "tokenizer","connectors","transformer","vocoder")})
    condition = LTX2VideoCondition(frames=torch.zeros(22,3,32,32),index=5,strength=1.)
    with torch.no_grad():
        latents,mask,clean = pipe.prepare_latents(conditions=[condition],batch_size=1,num_channels_latents=2,
            height=32,width=32,num_frames=39,noise_scale=1,dtype=torch.float32,device="cpu",
            generator=torch.Generator().manual_seed(7))
        held = pipe.vae.encode(torch.full((1,3,22,32,32),-1.)).latent_dist.mode()
    assert latents.shape == (1,12*16,2)
    assert not mask[:,:5*16].any() and (mask[:,5*16:] == 1).all()
    torch.testing.assert_close(clean[:,5*16:],held.flatten(2).transpose(1,2),rtol=0,atol=0)
    torch.testing.assert_close(latents[:,5*16:],clean[:,5*16:],rtol=0,atol=0)
    with pytest.raises(ValueError,match="chunk phase"):
        pipe.preprocess_conditions([LTX2VideoCondition(frames=condition.frames,index=1,strength=1.)],32,32,39)
    with pytest.raises(ValueError,match="cannot hold"):
        pipe.preprocess_conditions([LTX2VideoCondition(frames=condition.frames,index=10,strength=1.)],32,32,39)
