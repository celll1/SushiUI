"""CPU oracle and adapter tests for the optional official Sol-Attn kernel."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.attention import planned, sol
from core.attention.sol_reference import sol_attention_reference
from core.models.minimax_h3.sparse_attention import H3SolAttentionPlan


@pytest.mark.parametrize("tokens", [7, 8, 9, 17])
@pytest.mark.parametrize("threshold_type", ["diag", "exact"])
def test_full_sink_reference_matches_dense_attention(tokens, threshold_type):
    torch.manual_seed(tokens)
    q, k, v = (torch.randn(1, tokens, 2, 4) for _ in range(3))

    actual = sol_attention_reference(
        q,
        k,
        v,
        threshold_type=threshold_type,
        block_size=8,
        sink_tokens=tokens,
    ).output
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_reference_routes_neighbors_and_sink_blocks_exactly():
    torch.manual_seed(31)
    q, k, v = (torch.randn(1, 25, 1, 4) for _ in range(3))
    result = sol_attention_reference(
        q,
        k,
        v,
        tau=100.0,
        block_size=8,
        sink_start=16,
        sink_tokens=1,
    )

    expected = torch.tensor(
        [[True, True, True, False], [True, True, True, False],
         [False, True, True, True], [False, False, True, True]]
    )
    assert torch.equal(result.routes[0, 0], expected)


def test_adapter_replaces_prefix_queries_with_dense_result(monkeypatch):
    q, k, v = (torch.randn(1, 6, 2, 4) for _ in range(3))
    plan = H3SolAttentionPlan(
        position_ids=torch.zeros(6, 3),
        target_video_rows=torch.tensor([False, False, True, True, True, True]),
        dense_steps=0,
        dense_layers=0,
    )
    plan.begin_forward()

    monkeypatch.setattr(sol, "_validate_sol_qkv", lambda *args: None)
    monkeypatch.setattr(sol, "_load_sol_attention", lambda: (lambda *args, **kwargs: torch.zeros_like(args[0])))
    monkeypatch.setattr(sol, "note_backend", lambda backend: None)

    with torch.no_grad():
        actual = planned.dispatch_planned_attention(
            q, k, v, plan, dense_backend="native", layer_index=0
        )
    expected_prefix = F.scaled_dot_product_attention(
        q[:, :2].transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)

    torch.testing.assert_close(actual[:, :2], expected_prefix)
    assert torch.count_nonzero(actual[:, 2:]) == 0


def test_explicit_missing_kernel_is_not_silently_dense(monkeypatch):
    sol._load_sol_attention.cache_clear()
    monkeypatch.setattr(sol, "sol_attention_available", lambda: False)
    with pytest.raises(RuntimeError, match="optional official sol-attn"):
        sol._load_sol_attention()
    sol._load_sol_attention.cache_clear()
