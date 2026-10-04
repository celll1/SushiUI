"""Fixed-budget unpadded views and true router diagnostics for owner review."""
import json
from dataclasses import replace

import pytest
import torch
from PIL import Image

from api.param_defaults import TAGGER_JOINT_REVIEW_DEFAULTS
from core.tagger.foveatag.artifact import FoveaArtifactConfig
from core.tagger.foveatag.views import ViewConfig, build_views
from core.tagger.foveatag.router import fit_candidate_boxes


@pytest.mark.parametrize("size", [(1200,300), (300,1200), (81,37)])
def test_fixed_overview_and_local_views_have_no_invented_bands(size):
    cfg = ViewConfig(overview_size=32, local_size=16, local_views=12,
        overview_resize_mode="stretch", local_border_mode="shift_inside")
    views = build_views(Image.new("RGB", size, (31,71,113)), cfg, seed=5, training=False)
    assert views.overview.size == (32,32)
    assert views.overview.getextrema() == ((31,31),(71,71),(113,113))
    for view, spec in zip(views.local_views, views.local_specs):
        assert view.getextrema() == ((31,31),(71,71),(113,113))
        box = spec.source_box
        assert box.width == box.height
        assert 0 <= box.x <= size[0] - box.width
        assert 0 <= box.y <= size[1] - box.height
    spec = views.overview_spec
    assert spec.source_to_canvas(*size) == pytest.approx((32,32))
    assert spec.canvas_to_source(32,32) == pytest.approx(size)


def test_legacy_view_serialization_remains_unchanged():
    assert "overview_resize_mode" not in ViewConfig().to_dict()
    assert "local_border_mode" not in FoveaArtifactConfig(num_tags=2).to_dict()["views"]
    cfg = replace(FoveaArtifactConfig(num_tags=2), views=ViewConfig(overview_resize_mode="stretch", local_border_mode="shift_inside"))
    assert FoveaArtifactConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_dense_candidate_fit_matches_source_square_constraints():
    boxes = torch.tensor([[[0.,0.,3.,4.], [1.,1.,.1,.2]]])
    fitted = fit_candidate_boxes(boxes, 300, 100)
    assert torch.allclose(fitted[...,2] * 300, fitted[...,3] * 100)
    assert ((fitted[...,:2] - fitted[...,2:] / 2) >= -1e-7).all()
    assert ((fitted[...,:2] + fitted[...,2:] / 2) <= 1 + 1e-7).all()


def test_request_defaults_and_required_learned_slot():
    from api.routes import FoveaJointReviewRequest
    from pydantic import ValidationError
    request = FoveaJointReviewRequest(source_run_id="synthetic", output_dir="synthetic")
    assert {key: request.model_dump()[key] for key in TAGGER_JOINT_REVIEW_DEFAULTS} == TAGGER_JOINT_REVIEW_DEFAULTS
    with pytest.raises(ValidationError):
        FoveaJointReviewRequest(source_run_id="synthetic", output_dir="synthetic", selection_budget=4, random_count=4)


def test_direct_joint_does_not_consult_abc_quality_gate():
    import inspect
    from core.tagger.foveatag.direct_joint import run_direct_joint
    code = inspect.getsource(run_direct_joint)
    assert "phase3_gate" not in code
    assert "FoveaJointTrainer" in code
    assert "load_joint_checkpoint" in code
