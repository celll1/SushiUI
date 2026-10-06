"""Current-model presentation and chain plans share loaded codec geometry."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from api.arch_capabilities import video_constraints_payload
from api.generation_utils import plan_video_outpaint_placement
from minimax_h3_video_artifact_test import fixture,suppress_import_cuda_context
from loaded_video_codec_geometry_test import manager


def loaded_h3(tmp_path,kind):
    if kind == "image":
        trainer = fixture(tmp_path,4)
    else:
        from ltx2_video_artifact_test import fixture as ltx_fixture
        from minimax_h3_video_alignment_test import setup
        trainer,_ = ltx_fixture(tmp_path,4,codec_kind="video")
        trainer.transformer = setup(4,True,torch.float32,trainer.vae.contract)[0]
    owner,components = manager(trainer,"minimax_h3")
    owner.current_model_info = dict(type="minimax_h3",variant="fl2va",source="synthetic",latent_channels=24)
    return owner,components,trainer.vae.contract


@pytest.mark.parametrize("kind",["image","causal"])
def test_current_model_and_chain_plans_follow_real_loaded_codec(tmp_path,monkeypatch,kind):
    import api.routes as routes
    from core.inference.video_chain_context import build_segment_spans
    owner,components,c = loaded_h3(tmp_path,kind)
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    baseline = video_constraints_payload()
    response = asyncio.run(routes.get_current_model())
    geometry = response["model_info"]["loaded_video_constraints"]
    from pathlib import Path
    import yaml
    spec = yaml.safe_load((Path(__file__).resolve().parents[2]/"openapi.yaml").read_text(encoding="utf-8"))
    schema = spec["components"]["schemas"]["LoadedVideoConstraints"]
    types = {"integer":int,"number":(int,float),"boolean":bool,"string":str,"array":list}
    assert set(schema["required"]) <= geometry.keys()
    assert geometry.keys() <= schema["properties"].keys()
    for key,value in geometry.items():
        field = schema["properties"][key]
        if value is None:
            assert field.get("nullable") is True
            continue
        assert isinstance(value,types[field["type"]])
        assert not isinstance(value,bool) or field["type"] == "boolean"
        if "minimum" in field:
            assert value >= field["minimum"]
        if "enum" in field:
            assert value in field["enum"]
        if field["type"] == "array":
            assert len(value) >= field.get("minItems",0) and len(value) <= field.get("maxItems",len(value))
            assert all(isinstance(item,types[field["items"]["type"]]) for item in value)
    assert response["loaded"] and response["model_info"]["latent_channels"] == 4
    assert owner.current_model_info["latent_channels"] == 24
    assert "loaded_video_constraints" not in owner.current_model_info
    assert geometry["frame_multiple"] == c.temporal_scale
    assert geometry["frame_offset"] == 1
    assert geometry["latent_frame_policy"] == c.temporal_policy
    assert geometry["latent_chunk_pattern"] == ([1] if kind == "image" else [])
    assert geometry["fps_fixed"] == 24.
    assert geometry["min_frames"] == 124 and geometry["trained_max_frames"] == 362
    assert all((frames-1) % c.temporal_scale == 0 for frames in geometry["suggested_frames"])
    assert video_constraints_payload() == baseline
    grid = routes._video_chain_grid("minimax_h3")
    assert grid.frame_multiple == c.temporal_scale
    overlap,cap = (6,126) if kind == "image" else (7,125)
    request = SimpleNamespace(continuation_mode="pinned_tail",requested_overlap_frames=overlap,
                            requested_anchor_count=0,variant="fl2va")
    continuation = routes._video_chain_plan_continuation(request,"minimax_h3")
    assert continuation["overlap_frames"] == overlap
    spans = build_segment_spans(grid,500,cap,None,overlap)
    accumulated = spans[0].owned_end_frame
    for span in spans[1:]:
        placement = plan_video_outpaint_placement(dict(total_frames=span.requested_total_frames,input_offset_frames=0),
            "minimax_h3",head_frames=accumulated,overlap_frames=overlap,video_codec_contract=c)
        assert placement["generated_frames"] == span.generated_span_frames
        assert placement["total_frames"] == span.owned_end_frame
        accumulated = span.owned_end_frame
    # Other architectures and unloaded state retain their own base constraints.
    assert routes._video_chain_grid("ltx2").frame_multiple == 8
    owner.current_model_info = None
    assert asyncio.run(routes.get_current_model()) == {"loaded":False}
    assert routes._video_chain_grid("minimax_h3").frame_multiple == 17


def test_codec_payload_describes_all_frame_policies_without_mutating_defaults():
    from video_codec_temporal_plan_test import codec
    baseline = video_constraints_payload()
    for policy in ("independent_frames_v1","causal_first_frame_v1","minimax_h3_chunks_v1"):
        c = codec(policy)
        payload = video_constraints_payload(codec_contracts={"minimax_h3":c})
        assert payload["ltx2"] == baseline["ltx2"]
        item = payload["minimax_h3"]
        assert item["pixel_align"] == 64 and item["latent_frame_policy"] == policy
        assert item["latent_chunk_pattern"] == {
            "independent_frames_v1":[1],"causal_first_frame_v1":[],
            "minimax_h3_chunks_v1":[1,4,4,4,4]}[policy]
    assert video_constraints_payload() == baseline


def test_native_current_model_omits_loaded_overlay(monkeypatch):
    import api.routes as routes
    owner = SimpleNamespace(current_model_info={"type":"minimax_h3","latent_channels":24},minimax_h3_components={})
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    assert "loaded_video_constraints" not in asyncio.run(routes.get_current_model())["model_info"]


def test_unwired_ltx_clock_remains_inspectable_but_generation_is_unavailable(monkeypatch):
    import api.routes as routes
    from video_codec_temporal_plan_test import codec
    from core.models.components.video_latent_adapter import NormalizedVideoVAE
    c = codec("minimax_h3_chunks_v1")
    vae = object.__new__(NormalizedVideoVAE)
    object.__setattr__(vae,"contract",c)
    model = SimpleNamespace(config=SimpleNamespace(in_channels=2,vae_scale_factors=(4,32,32)),
        transformer=SimpleNamespace(_video_codec_contract=c,_video_latent_alignment=SimpleNamespace(ready=True)))
    owner = SimpleNamespace(current_model_info={"type":"ltx2"},ltx2_components=dict(
        vae=vae,transformer=model,latent_channels=2,vae_scale_factor_spatial=32,vae_scale_factor_temporal=4))
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    info = asyncio.run(routes.get_current_model())["model_info"]
    assert info["generation_available"] is False
    assert info["loaded_video_constraints"]["latent_frame_policy"] == "minimax_h3_chunks_v1"
