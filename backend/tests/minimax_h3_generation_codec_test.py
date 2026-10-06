"""Synthetic target-native H3 generation through the shipped backend."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from minimax_h3_video_artifact_test import fixture, suppress_import_cuda_context


def components(trainer):
    model,vae = trainer.transformer,trainer.vae
    c = vae.contract
    return dict(transformer=model,transformer_config=dict(model.transformer.config),vae=vae,
        image_vae=object(),audio_vae=None,text_encoder=object(),tokenizer=object(),
        latent_channels=c.image.latent_channels,vae_scale_factor_spatial=c.image.spatial_scale,
        vae_scale_factor_temporal=c.temporal_scale,audio_latent_channels=6,
        latents_mean=[999.],latents_std=[999.],pixel_mean=[999.],pixel_std=[999.],variant="ref2va")


def test_codec_geometry_and_double_normalization_are_rejected(tmp_path):
    from core.models.minimax_h3.generation_codec import generation_contract,target_grid
    trainer = fixture(tmp_path,4)
    value = components(trainer)
    contract = generation_contract(value)
    assert target_grid(contract,3,16,24) == (3,8,12)
    for changes in (dict(vae=trainer.vae.inner),dict(latent_channels=24),dict(vae_scale_factor_spatial=16)):
        with pytest.raises(ValueError):
            generation_contract({**value,**changes})
    with pytest.raises(ValueError):
        target_grid(contract,3,18,24)
    with pytest.raises(ValueError,match="down_factor"):
        target_grid(contract,3,16,20,topology=trainer.transformer.transformer._residual_unet_spec)
    with pytest.raises(ValueError):
        generation_contract(dict(vae=trainer.vae,transformer=SimpleNamespace()))
    assert generation_contract(dict(vae=object(),transformer=SimpleNamespace())) is None


@pytest.mark.parametrize("new",[1,4])
def test_real_codec_condition_seed_and_decode_match_single_facade_normalization(tmp_path,new):
    from core.models.minimax_h3.h3_pipeline_ops import encode_visual_condition,decode_video
    trainer = fixture(tmp_path,new)
    vae = trainer.vae
    frames = np.random.default_rng(94).integers(0,256,size=(3,16,24,3),dtype=np.uint8)
    poison = dict(latents_mean=[999.],latents_std=[999.],pixel_mean=[999.],pixel_std=[999.],device="cpu")
    canonical = torch.from_numpy(frames).permute(3,0,1,2)[None].float()/127.5-1
    expected = vae.encode(canonical).latent_dist.sample(generator=torch.Generator().manual_seed(42)).float()
    actual = encode_visual_condition(vae,frames,**poison)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert not torch.equal(actual,encode_visual_condition(vae,frames,encode_seed=43,**poison))
    decoded = vae.decode(actual,return_dict=False)[0]
    oracle = ((decoded.float()*.5+.5).clamp(0,1)[0].permute(1,2,3,0).numpy()*255).round().astype(np.uint8)
    np.testing.assert_array_equal(decode_video(vae,actual,**poison),oracle)


@pytest.mark.parametrize("workflow",["text","keyframe","reference"])
@pytest.mark.parametrize("rewired",[False,True])
@pytest.mark.parametrize("new",[1,4])
@pytest.mark.parametrize("kind",["image","causal","strong_spatial"])
def test_real_backend_target_grid_conditions_sampler_preview_and_decode(tmp_path,monkeypatch,workflow,rewired,new,kind):
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin
    from core.models.minimax_h3 import h3_pipeline_ops as ops,h3_references as refs,prompt_cache
    from core.models.minimax_h3.vendor import MiniMaxH3Scheduler
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    if kind == "image":
        trainer = fixture(tmp_path,new,rewired=rewired)
    elif kind == "causal":
        from ltx2_video_artifact_test import fixture as ltx_fixture
        from minimax_h3_video_alignment_test import setup
        trainer,_ = ltx_fixture(tmp_path,new,codec_kind="video")
        trainer.transformer = setup(new,rewired,torch.float32,trainer.vae.contract)[0]
    else:
        from diffusers import AutoencoderKL
        from minimax_h3_video_alignment_test import setup
        from core.models.components.latent_adapter import ImageCodecContract
        from core.models.components.video_latent_adapter import VideoCodecContract,NormalizedVideoVAE
        raw = AutoencoderKL(in_channels=3,out_channels=3,latent_channels=new,block_out_channels=(4,)*6,
            down_block_types=("DownEncoderBlock2D",)*6,up_block_types=("UpDecoderBlock2D",)*6,
            norm_num_groups=1,scaling_factor=2.,shift_factor=.25).eval().requires_grad_(False)
        contract = VideoCodecContract(ImageCodecContract(new,32,4,"shift_scale",1,3),1,"independent_frames_v1")
        trainer = SimpleNamespace(transformer=setup(new,rewired,torch.float32,contract)[0],
                                  vae=NormalizedVideoVAE(raw,contract))
    value = components(trainer)
    value.update(scheduler=MiniMaxH3Scheduler(shift=12),audio_scheduler=MiniMaxH3Scheduler(shift=3))
    events,previews,layouts = [],[],[]
    embeds = torch.randn(1,3,10)
    monkeypatch.setattr(ops,"encode_prompt",lambda *a,**k:(embeds,3))
    monkeypatch.setattr(prompt_cache,"get_or_encode_prompt",lambda a,b,c,d,encode:(encode(),False))
    monkeypatch.setattr(refs,"normalize_references",lambda items,**kwargs:list(items))
    monkeypatch.setattr(refs,"build_ref2va_presentation",lambda *a,**k:([1,2,3],[ops.TEXT_TAG]*3,{}))
    monkeypatch.setattr(ops,"encode_presentation",lambda *a,**k:embeds)
    original = ops.denoise
    def denoise(*args,**kwargs):
        layouts.append(kwargs["layout"])
        return original(*args,**kwargs)
    monkeypatch.setattr(ops,"denoise",denoise)

    class Runner(MiniMaxH3Mixin):
        device = "cpu"
        minimax_h3_components = value
        def _minimax_h3_project_prompt_embeds(self,x,*a,**k): return x
        def _minimax_h3_move(self,name,device):
            events.append((name,str(device)))
            value[name].to(device)
        def _minimax_h3_empty_cache(self): pass
        def _minimax_h3_reset_peak_vram(self): pass
        def _minimax_h3_vram_stats(self): return (0.,0.,0.)
        def _minimax_h3_peak_vram(self): return 0.
        def _minimax_h3_assert_components_off_cuda(self,*names):
            for name in names:
                item = value.get(name)
                if isinstance(item,torch.nn.Module):
                    assert all(p.device.type == "cpu" for p in item.parameters())
        def _ensure_minimax_h3_swap_and_offload(self,*a):
            events.append(("transformer","stage"))
            return super()._ensure_minimax_h3_swap_and_offload(*a)
        def _minimax_h3_apply_attention_backend(self,*a): pass
        def _unload_lora_minimax_h3(self): pass
        def _unstage_minimax_h3_transformer(self,*a):
            super()._unstage_minimax_h3_transformer(*a)
            events.append(("transformer","unstage"))
        def _minimax_h3_dump_residual_probe(self,*a,**k): pass
    pixel_frames = 5 if kind == "causal" else 3
    height,width = (128,256) if kind == "strong_spatial" else (16,24)
    grid = (4,8) if kind == "strong_spatial" else (8,12)
    params = dict(prompt="synthetic CPU verification",width=width,height=height,num_frames=pixel_frames,
        num_inference_steps=2,audio_enable=False,seed=94)
    image = Image.fromarray(np.zeros((height,width,3),dtype=np.uint8))
    extra = dict(keyframes=(("last",image),)) if workflow == "keyframe" else (
        dict(references=(refs.MiniMaxH3Reference(kind="image",image=image),)) if workflow == "reference" else {})
    result = Runner()._generate_minimax_h3(params,**extra,
        step_callback=lambda *args:previews.append(args[2].shape))
    frames,audio,rate,seed = result
    assert frames.shape == (pixel_frames,height,width,3) and frames.dtype == np.uint8
    assert audio is None and rate is None and seed == 94
    assert previews == [torch.Size((1,new,3,*grid))]
    assert layouts[0]["video_codec_contract"]["temporal_policy"] == trainer.vae.contract.temporal_policy
    assert layouts[0]["video_position_policy"] == "pixel_frame_v1"
    if kind == "strong_spatial":
        assert ops.packed_row_counts(layouts[0])["target_video"] == 3*4*8//4
    assert events[-3:] == [("transformer","unstage"),("vae","cpu"),("vae","cpu")]
    assert value["transformer"].config.in_channels == new
    again = Runner()._generate_minimax_h3(dict(params),**extra)
    np.testing.assert_array_equal(again[0],frames)


def test_original_h3_condition_rounding_and_decode_pixels_are_unchanged():
    from minimax_h3_vae_contract_test import small_codec
    from core.models.minimax_h3.h3_pipeline_ops import encode_visual_condition,decode_video
    vae = small_codec()
    frames = np.random.default_rng(35).integers(0,256,size=(22,32,32,3),dtype=np.uint8)
    pixel_mean,pixel_std = (.485,.456,.406),(.229,.224,.225)
    mean,std = torch.tensor([.2,.3]).view(1,2,1,1,1),torch.tensor([.8,1.2]).view(1,2,1,1,1)
    pm,ps = torch.tensor(pixel_mean).view(1,3,1,1,1),torch.tensor(pixel_std).view(1,3,1,1,1)
    pixels = torch.from_numpy(frames).permute(3,0,1,2)[None].float()/255.
    with torch.no_grad():
        raw = vae.encode((pixels-pm)/ps).latent_dist.sample(generator=torch.Generator().manual_seed(42))
        expected = (raw.half().float()-mean)/std
        options = dict(latents_mean=[.2,.3],latents_std=[.8,1.2],pixel_mean=pixel_mean,pixel_std=pixel_std,device="cpu")
        actual = encode_visual_condition(vae,frames,**options)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        decoded = vae.decode(actual*std+mean,return_dict=False)[0]
        oracle = ((decoded.float()*ps+pm).clamp(0,1)[0].permute(1,2,3,0).numpy()*255).round().astype(np.uint8)
        np.testing.assert_array_equal(decode_video(vae,actual,**options),oracle)


def test_target_reference_video_snapping_and_encode_use_target_clock(tmp_path):
    from core.models.minimax_h3 import h3_references as refs
    from core.models.minimax_h3.h3_pipeline_ops import encode_visual_condition
    from core.models.components.latent_adapter import ImageCodecContract
    from core.models.components.video_latent_adapter import VideoCodecContract
    trainer = fixture(tmp_path,4)
    contract = trainer.vae.contract
    pixels = np.zeros((15,16,24,3),dtype=np.uint8)
    normalized = refs.normalize_reference_video(pixels,24.,15,target_fps=24.,multiple=4,
        canvas=(16,24),video_codec_contract=contract)
    assert normalized.shape[0] == 15
    reference = refs.MiniMaxH3Reference(kind="video",frames=normalized,fps=24.)
    poison = dict(latents_mean=[999.],latents_std=[999.],pixel_mean=[999.],pixel_std=[999.],device="cpu")
    actual, = refs.encode_reference_visuals(trainer.vae,[reference],**poison)
    torch.testing.assert_close(actual,encode_visual_condition(trainer.vae,normalized,**poison),rtol=0,atol=0)
    causal = VideoCodecContract(ImageCodecContract(4,2,5,"identity",1,3),2,"causal_first_frame_v1")
    assert refs.snap_reference_video_frames(16,causal) == 15
    with pytest.raises(ValueError,match="13"):
        refs.snap_reference_video_frames(12,causal)
