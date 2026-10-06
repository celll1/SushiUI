"""Frozen chain geometry is checked over real HTTP routes after codec changes."""

import copy

import pytest

from loaded_video_codec_presentation_test import loaded_h3,suppress_import_cuda_context
from video_chain_motion_preroll_plan_test import _post


def request(kind,mode):
    return dict(architecture="minimax_h3",variant="fl2va",root_prompt="a geometric object moves",
        target_frames=500,fps=24.,requested_segment_frames=126 if kind == "image" else 125,
        context_mode="legacy_repeat",continuation_mode=mode,
        requested_overlap_frames=(6 if kind == "image" else 7) if mode == "pinned_tail" else (9 if mode == "motion_preroll" else 0),
        requested_anchor_count=3 if mode == "motion_preroll" else 0,root_seed=17,seed_policy="derived")


@pytest.mark.parametrize("kind",["image","causal"])
@pytest.mark.parametrize("mode",["boundary_frame","pinned_tail","motion_preroll"])
def test_valid_loaded_chain_survives_http_validation_with_same_hash(tmp_path,monkeypatch,kind,mode):
    import api.routes as routes
    owner,_,_ = loaded_h3(tmp_path,kind)
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    status,body = _post("/video-chain/plan",request(kind,mode))
    assert status == 200 and body["success"],body
    manifest = body["manifest"]
    snapshot = copy.deepcopy(manifest)
    status,validated = _post("/video-chain/validate",dict(manifest=manifest,recompute_plan_hash=True))
    assert status == 200 and validated["valid"],validated
    assert validated["plan_hash"] == manifest["plan_hash"]
    assert manifest == snapshot
    assert [row["seed"] for row in validated["manifest"]["segments"]] == [row["seed"] for row in manifest["segments"]]


@pytest.mark.parametrize("field,value,index",[
    ("generated_span_frames",126,0),("requested_total_frames",126,0),
    ("anchor_global_frame",117,1),("effective_overlap_frames",6,1),
    ("requested_overlap_frames",6,1),("owned_start_frame",124,1),
])
def test_edited_geometry_is_invalid_without_rehash_or_resnap(tmp_path,monkeypatch,field,value,index):
    import api.routes as routes
    owner,_,_ = loaded_h3(tmp_path,"causal")
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    _,body = _post("/video-chain/plan",request("causal","pinned_tail"))
    assert body["success"],body
    manifest = copy.deepcopy(body["manifest"])
    manifest["segments"][index][field] = value
    snapshot = copy.deepcopy(manifest)
    status,result = _post("/video-chain/validate",dict(manifest=manifest,recompute_plan_hash=True))
    assert status == 200 and not result["valid"],result
    assert result["plan_hash"] is None and result["manifest"] is None
    assert any(row["code"] == "loaded_codec_segment_mismatch" and row["segment_index"] == index for row in result["errors"])
    assert manifest == snapshot


def test_saved_plan_is_refused_after_change_to_causal_codec(tmp_path,monkeypatch):
    import api.routes as routes
    source = tmp_path/"source"
    source.mkdir()
    owner,_,_ = loaded_h3(source,"image")
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    _,body = _post("/video-chain/plan",request("image","pinned_tail"))
    assert body["success"],body
    target = tmp_path/"target"
    target.mkdir()
    replacement,_,_ = loaded_h3(target,"causal")
    monkeypatch.setattr(routes,"pipeline_manager",replacement)
    _,result = _post("/video-chain/validate",dict(manifest=body["manifest"],recompute_plan_hash=False))
    assert not result["valid"] and result["plan_hash"] is None and result["manifest"] is None
    assert any(row["code"] == "loaded_codec_segment_mismatch" for row in result["errors"])


@pytest.mark.parametrize("field,value",[("fps",30.),("expected_final_frames",1)])
def test_global_clock_and_length_mismatches_are_reported(tmp_path,monkeypatch,field,value):
    import api.routes as routes
    owner,_,_ = loaded_h3(tmp_path,"causal")
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    _,body = _post("/video-chain/plan",request("causal","boundary_frame"))
    manifest = body["manifest"]
    manifest[field] = value
    _,result = _post("/video-chain/validate",dict(manifest=manifest))
    assert not result["valid"]
    code = "loaded_codec_fps_mismatch" if field == "fps" else "loaded_codec_final_length_mismatch"
    assert any(row["code"] == code for row in result["errors"])


def test_off_grid_first_segment_is_not_marked_runnable(tmp_path,monkeypatch):
    import api.routes as routes
    owner,_,_ = loaded_h3(tmp_path,"causal")
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    payload = request("causal","boundary_frame")
    payload["requested_segment_frames"] = 126
    status,body = _post("/video-chain/plan",payload)
    assert status == 200 and not body["success"]
    assert any(row["code"] == "loaded_codec_segment_mismatch" for row in body["errors"])


@pytest.mark.parametrize("field,value",[("anchor_count",2),("anchor_local_frames",[])])
def test_motion_anchor_metadata_must_match_actual_queued_conditioning(tmp_path,monkeypatch,field,value):
    import api.routes as routes
    owner,_,_ = loaded_h3(tmp_path,"causal")
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    _,body = _post("/video-chain/plan",request("causal","motion_preroll"))
    assert body["success"],body
    manifest = copy.deepcopy(body["manifest"])
    manifest["segments"][1]["visual_context"][field] = value
    _,result = _post("/video-chain/validate",dict(manifest=manifest))
    assert not result["valid"] and result["plan_hash"] is None
    assert any(row["code"] == "loaded_codec_segment_mismatch" and row["segment_index"] == 1 for row in result["errors"])


@pytest.mark.parametrize("kind",["image","video"])
def test_ltx_free_placement_validates_the_whole_output_timeline(tmp_path,monkeypatch,kind):
    from ltx2_video_artifact_test import fixture
    from loaded_video_codec_geometry_test import manager
    import api.routes as routes
    trainer,_ = fixture(tmp_path,4,kind)
    owner,_ = manager(trainer,"ltx2")
    owner.current_model_info = {"type":"ltx2"}
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    cap = 126 if kind == "image" else 125
    payload = request("image","boundary_frame")
    payload.update(architecture="ltx2",variant="",requested_segment_frames=cap,target_frames=cap+3*(cap-1))
    status,body = _post("/video-chain/plan",payload)
    assert status == 200 and body["success"],body
    status,valid = _post("/video-chain/validate",dict(manifest=body["manifest"]))
    assert status == 200 and valid["valid"],valid
    assert valid["plan_hash"] == body["manifest"]["plan_hash"]
    edited = copy.deepcopy(body["manifest"])
    edited["segments"][1]["requested_total_frames"] += 1
    _,invalid = _post("/video-chain/validate",dict(manifest=edited))
    assert not invalid["valid"] and any(row["code"] == "loaded_codec_segment_mismatch" for row in invalid["errors"])
