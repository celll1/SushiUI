"""Free-placement chains use the target timeline grid, not standalone span grids."""

import copy
from types import SimpleNamespace

import pytest

from ltx2_codec_pipeline_test import pipeline
from loaded_video_codec_geometry_test import manager
from video_chain_motion_preroll_plan_test import _post


def loaded_owner():
    base = pipeline(rewired=True)
    owner,components = manager(SimpleNamespace(transformer=base.transformer,vae=base.vae),"ltx2")
    components["pipeline"] = base
    owner.current_model_info = {"type":"ltx2"}
    return owner,base


@pytest.mark.parametrize("cap,target",[(22,100),(39,100),(124,500)])
def test_real_loaded_codec_http_plan_validate_and_placement_agree(monkeypatch,cap,target):
    import api.routes as routes
    from api.generation_utils import loaded_video_codec_contract,plan_video_outpaint_placement
    owner,base = loaded_owner()
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    codec = loaded_video_codec_contract(owner,"ltx2")
    status,body = _post("/video-chain/plan",dict(architecture="ltx2",root_prompt="a geometric object moves",
        target_frames=target,fps=24,requested_segment_frames=cap,context_mode="legacy_repeat",
        continuation_mode="boundary_frame",root_seed=7,seed_policy="derived"))
    assert status == 200 and body["success"],body
    manifest = body["manifest"]
    assert manifest["expected_final_frames"] >= target
    for row in manifest["segments"][1:]:
        assert row["generated_span_frames"] <= cap
        assert codec.pixel_frames(codec.latent_frames(row["requested_total_frames"])) == row["requested_total_frames"]
        placement = plan_video_outpaint_placement(dict(total_frames=row["requested_total_frames"],input_offset_frames=0),
            "ltx2",head_frames=row["owned_start_frame"],overlap_frames=1,video_codec_contract=codec)
        assert placement == {"placement":"free"}
        assert row["owned_end_frame"] == row["requested_total_frames"]
    _,validated = _post("/video-chain/validate",dict(manifest=manifest))
    assert validated["valid"] and validated["plan_hash"] == manifest["plan_hash"]
    edited = copy.deepcopy(manifest)
    edited["segments"][1]["requested_total_frames"] += 1
    _,invalid = _post("/video-chain/validate",dict(manifest=edited))
    assert not invalid["valid"] and invalid["plan_hash"] is None


def test_model_info_allows_installed_clock_but_refuses_stale_cached_pipeline(monkeypatch):
    import asyncio
    import api.routes as routes
    from api.error_handlers import ValidationError
    from api.generation_utils import loaded_video_codec_contract
    owner,base = loaded_owner()
    monkeypatch.setattr(routes,"pipeline_manager",owner)
    info = asyncio.run(routes.get_current_model())["model_info"]
    assert info.get("generation_available",True)
    assert info["loaded_video_constraints"]["latent_frame_policy"] == "minimax_h3_chunks_v1"
    owner.ltx2_components["cond_pipeline"] = SimpleNamespace(vae=base.vae,transformer=base.transformer)
    with pytest.raises(ValidationError,match="contract is invalid"):
        loaded_video_codec_contract(owner,"ltx2")
    assert asyncio.run(routes.get_current_model())["model_info"]["generation_available"] is False


def test_shot_aligned_free_timeline_keeps_each_cut_on_target_grid():
    from core.inference.video_chain_context import VideoGridSpec,plan_shot_aligned_spans
    spec = VideoGridSpec(17,5,1,22,None,("free",))
    spans = plan_shot_aligned_spans(spec,100,[0,39,73],39)
    assert spans is not None
    assert all(spec.is_on_grid(row.owned_end_frame) for row in spans)
    assert all(row.generated_span_frames <= 39 for row in spans)
    assert spans[-1].owned_end_frame >= 100


def test_dense_alignment_survives_block_swap_enable_disable(monkeypatch):
    from core.pipeline_backends.ltx2 import LTX2Mixin
    from core.models.ltx2.video_alignment import AlignedLtx2BlockLoopWrapper
    import core.memory_management as memory
    base = pipeline(rewired=False)
    class FakeOffloader:
        def __init__(self,**kwargs):
            self.blocks_to_swap = kwargs["blocks_to_swap"]
        def prepare_block_devices_before_forward(self):
            pass
        def cleanup(self):
            pass
    monkeypatch.setattr(memory,"TransformerBlockOffloader",FakeOffloader)
    owner = LTX2Mixin()
    owner.ltx2_components = {"pipeline":base,"transformer":base.transformer}
    pair = base.transformer.transformer._video_latent_alignment
    for count in (0,2,2,0,0):
        owner._ensure_ltx2_block_swap_wrapper(count)
        assert isinstance(base.transformer,AlignedLtx2BlockLoopWrapper)
        assert base.transformer.config.in_channels == 2
        assert base.transformer.transformer._video_latent_alignment is pair
        assert owner.ltx2_components["transformer"] is base.transformer
        assert (base.transformer._block_offloader is not None) == bool(count)
