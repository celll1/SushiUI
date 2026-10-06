"""Loaded codec geometry reaches API preflight without reading source weights."""

import ast
import inspect
from types import SimpleNamespace

import pytest

from api.generation_utils import loaded_video_codec_contract,validate_video_geometry
from api.error_handlers import ValidationError
from core.models.components.latent_adapter import ImageCodecContract
from core.models.components.video_latent_adapter import VideoCodecContract,temporal_spec_for_codec
from core.models.components.wiring import temporal_spec_for_arch
from minimax_h3_video_artifact_test import fixture,suppress_import_cuda_context


def manager(trainer,arch):
    c = trainer.vae.contract
    components = dict(transformer=trainer.transformer,vae=trainer.vae,
        latent_channels=c.image.latent_channels,vae_scale_factor_spatial=c.image.spatial_scale,
        vae_scale_factor_temporal=c.temporal_scale)
    return SimpleNamespace(**{arch+"_components":components}),components


@pytest.mark.parametrize("channels",[1,4])
def test_real_loaded_h3_pair_replaces_native_lattice(tmp_path,channels,monkeypatch):
    monkeypatch.delenv("SUSHI_TEMPORAL_SMOKE",raising=False)
    owner,components = manager(fixture(tmp_path,channels),"minimax_h3")
    c = loaded_video_codec_contract(owner,"minimax_h3")
    assert c is components["vae"].contract
    params = dict(width=64,height=64,num_frames=126,frame_rate=30.)
    warnings = validate_video_geometry(params,"minimax_h3",video_codec_contract=c)
    assert params["num_frames"] == 126 and params["frame_rate"] == 24.
    assert len(warnings) == 1
    params["num_frames"] = 123
    validate_video_geometry(params,"minimax_h3",video_codec_contract=c)
    assert params["num_frames"] == 124
    params["num_frames"] = 363
    warnings = validate_video_geometry(params,"minimax_h3",video_codec_contract=c)
    assert params["num_frames"] == 363 and any("trained range" in value for value in warnings)
    # Model/codec component disagreement is a preflight error, not a fallback.
    components["latent_channels"] += 1
    with pytest.raises(ValidationError,match="contract is invalid"):
        loaded_video_codec_contract(owner,"minimax_h3")


@pytest.mark.parametrize("kind",["image","causal"])
def test_real_loaded_ltx_pair_uses_codec_lattice(tmp_path,kind):
    from ltx2_video_artifact_test import fixture as ltx_fixture
    trainer,_ = ltx_fixture(tmp_path,4,kind)
    owner,_ = manager(trainer,"ltx2")
    c = loaded_video_codec_contract(owner,"ltx2")
    params = dict(width=64,height=64,num_frames=11,frame_rate=24.)
    assert validate_video_geometry(params,"ltx2",video_codec_contract=c) == []
    assert params["num_frames"] == 11
    if kind == "causal":
        params["num_frames"] = 10
        with pytest.raises(ValidationError):
            validate_video_geometry(params,"ltx2",video_codec_contract=c)


def test_stronger_spatial_scale_obeys_architecture_patch_and_keeps_model_envelope():
    c = VideoCodecContract(ImageCodecContract(4,64,5,"identity",1,3),2,"causal_first_frame_v1")
    for arch,alignment in (("minimax_h3",128),("ltx2",64)):
        native = temporal_spec_for_arch(arch)
        spec = temporal_spec_for_codec(native,c)
        assert spec.pixel_align == alignment
        assert spec.min_frames == native.min_frames and spec.max_pixel_hw == native.max_pixel_hw
        params = dict(width=192,height=192,num_frames=125,frame_rate=24.)
        if arch == "minimax_h3":
            with pytest.raises(ValidationError,match="divisible by 128"):
                validate_video_geometry(params,arch,video_codec_contract=c)
        else:
            assert validate_video_geometry(params,arch,video_codec_contract=c) == []


def test_causal_h3_snaps_to_target_grid_above_production_floor(monkeypatch):
    monkeypatch.delenv("SUSHI_TEMPORAL_SMOKE",raising=False)
    c = VideoCodecContract(ImageCodecContract(4,16,5,"identity",1,3),2,"causal_first_frame_v1")
    params = dict(width=64,height=64,num_frames=124,frame_rate=24.)
    warnings = validate_video_geometry(params,"minimax_h3",video_codec_contract=c)
    assert params["num_frames"] == 125 and len(warnings) == 1
    params["num_frames"] = 1
    validate_video_geometry(params,"minimax_h3",video_codec_contract=c)
    assert params["num_frames"] == 1


def test_native_and_nonvideo_managers_keep_existing_geometry():
    owner = SimpleNamespace(minimax_h3_components={"vae":object()},ltx2_components={"vae":object()})
    assert loaded_video_codec_contract(owner,"minimax_h3") is None
    assert loaded_video_codec_contract(owner,"ltx2") is None
    assert loaded_video_codec_contract(owner,"sdxl") is None
    params = dict(width=64,height=64,num_frames=126,frame_rate=24.)
    validate_video_geometry(params,"minimax_h3")
    assert params["num_frames"] == 141


def test_ltx_does_not_advertise_unwired_chunk_clock():
    from core.models.components.video_latent_adapter import NormalizedVideoVAE
    c = VideoCodecContract(ImageCodecContract(4,16,5,"per_channel",1,3),4,"minimax_h3_chunks_v1")
    vae = object.__new__(NormalizedVideoVAE)
    object.__setattr__(vae,"contract",c)
    inner = SimpleNamespace(_video_codec_contract=c,_video_latent_alignment=SimpleNamespace(ready=True))
    owner = SimpleNamespace(ltx2_components=dict(transformer=SimpleNamespace(transformer=inner),vae=vae))
    with pytest.raises(ValidationError,match="contract is invalid") as error:
        loaded_video_codec_contract(owner,"ltx2")
    assert "chunk frame clock" in str(error.value.detail)


def test_all_public_video_consumers_receive_loaded_contract():
    import api.routes as routes
    names = ("generate_txt2vid","generate_img2vid","generate_ref2vid",
             "generate_outpaint_video","generate_inpaint_video")
    found = set()
    for name in names:
        handler = getattr(routes,name)
        source = inspect.getsource(handler)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in (
                    "validate_video_geometry","plan_video_inpaint_span","plan_video_outpaint_placement",
                    "plan_video_continuation_context"):
                assert any(kw.arg == "video_codec_contract" for kw in node.keywords)
                found.add(node.func.id)
    assert found == {"validate_video_geometry","plan_video_inpaint_span","plan_video_outpaint_placement",
                     "plan_video_continuation_context"}
