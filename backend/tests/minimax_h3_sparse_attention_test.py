"""CPU contract tests for MiniMax-H3 sparse connectivity."""

import os
import sys

import pytest
import torch

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.attention import AttentionMode, dispatch_planned_attention
from core.models.minimax_h3.sparse_attention import (
    H3SolAttentionPlan,
    H3VideoWindowPlan,
    build_h3_attention_plan,
)


def _plan():
    positions = torch.zeros(12, 3)
    positions[:, 0] = torch.arange(12)
    target = torch.tensor([False, False, False, True] + [True] * 8)
    return H3VideoWindowPlan(
        positions, target, temporal_radius=1.0, spatial_radius=0.0, block_size=4
    )


def test_only_pure_target_video_block_pairs_are_sparse():
    plan = _plan()

    assert plan.allowed(torch.tensor(0), torch.tensor(11))
    assert plan.allowed(torch.tensor(11), torch.tensor(0))
    assert plan.allowed(torch.tensor(3), torch.tensor(11))
    assert plan.allowed(torch.tensor(4), torch.tensor(5))
    assert not plan.allowed(torch.tensor(4), torch.tensor(11))


def test_layout_conditioning_prefix_remains_dense():
    layout = {
        "position_ids": torch.zeros(12, 3),
        "video_indices": torch.tensor([1, 2, 4, 5, 6, 7, 8, 9, 10, 11]),
        "num_condition_video_rows": 2,
    }
    plan = H3VideoWindowPlan.from_layout(
        layout, temporal_radius=1.0, spatial_radius=1.0, block_size=4
    )

    assert not plan.target_video_rows[1]
    assert not plan.target_video_rows[2]
    assert plan.target_video_rows[4:].all()
    assert plan.position_ids.dtype == torch.float32


def test_dense_method_builds_no_plan():
    assert build_h3_attention_plan(
        "dense", {}, temporal_radius=1.0, spatial_radius=1.0
    ) is None


def test_sparse_training_is_refused_before_flex_compilation():
    q = torch.randn(1, 12, 2, 8)
    with pytest.raises(RuntimeError, match="inference-only"):
        dispatch_planned_attention(q, q, q, _plan(), mode=AttentionMode.TRAINING)


def _sol_plan(**overrides):
    values = {
        "position_ids": torch.zeros(12, 3),
        "target_video_rows": torch.tensor([False] * 4 + [True] * 8),
        "tau": 1.0,
        "threshold_type": "diag",
        "dense_steps": 1,
        "dense_layers": 2,
    }
    values.update(overrides)
    return H3SolAttentionPlan(**values)


def test_sol_plan_requires_one_contiguous_target_video_suffix():
    with pytest.raises(ValueError, match="contiguous suffix"):
        _sol_plan(
            target_video_rows=torch.tensor(
                [False, False, False, False, True, False, True, True, True, True, True, True]
            )
        )


def test_sol_plan_dense_warmup_is_step_and_layer_stable():
    plan = _sol_plan()

    plan.begin_forward()
    assert plan.use_dense(49)
    plan.begin_forward()
    assert plan.use_dense(0)
    assert plan.use_dense(1)
    assert not plan.use_dense(2)


def test_sol_plan_builds_from_h3_layout():
    layout = {
        "position_ids": torch.zeros(12, 3),
        "video_indices": torch.tensor([1, 2, 4, 5, 6, 7, 8, 9, 10, 11]),
        "num_condition_video_rows": 2,
    }
    plan = build_h3_attention_plan(
        "h3_sol_attn",
        layout,
        temporal_radius=1.0,
        spatial_radius=1.0,
        sol_tau=1.25,
        sol_threshold_type="exact",
        sol_dense_steps=0,
        sol_dense_layers=0,
    )

    assert isinstance(plan, H3SolAttentionPlan)
    assert plan.prefix_tokens == 4
    assert plan.tau == 1.25
    assert plan.threshold_type == "exact"


def test_sol_sparse_path_refuses_cpu_before_loading_optional_kernel():
    plan = _sol_plan(dense_steps=0, dense_layers=0)
    plan.begin_forward()
    q = torch.randn(1, 12, 2, 8)

    with torch.no_grad(), pytest.raises(RuntimeError, match="requires a CUDA device"):
        dispatch_planned_attention(q, q, q, plan, layer_index=0)
