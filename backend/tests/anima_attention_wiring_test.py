import os
import sys
from unittest.mock import patch

import torch

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.attention import AttentionMode
from core.models.anima import anima_attention
from core.models.anima.anima_models import LLMAdapterAttention


class _AnimaStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mode = "native"
        self.adapter = LLMAdapterAttention(16, 16, 2, 8)


def test_backend_selection_is_model_local_and_includes_llm_adapter():
    first = _AnimaStub()
    second = _AnimaStub()

    anima_attention.set_attention_backend(first, "sage", AttentionMode.INFERENCE)
    anima_attention.set_attention_backend(second, "flash", AttentionMode.TRAINING)

    assert first.attn_mode == "sage"
    assert first.adapter._attn_backend == "sage"
    assert first.adapter._attn_mode == AttentionMode.INFERENCE
    assert second.attn_mode == "flash"
    assert second.adapter._attn_backend == "flash"
    assert second.adapter._attn_mode == AttentionMode.TRAINING


def test_llm_adapter_routes_through_the_conduit_with_its_stamped_mode():
    module = LLMAdapterAttention(16, 16, 2, 8)
    module._attn_backend = "flash"
    module._attn_mode = AttentionMode.TRAINING
    captured = {}

    def _dispatch(q, k, v, **kwargs):
        captured.update(kwargs)
        return q

    with patch.object(anima_attention, "dispatch_attention", side_effect=_dispatch):
        out = module(torch.randn(1, 3, 16))

    assert out.shape == (1, 3, 16)
    assert captured == {
        "attn_mask": None,
        "backend": "flash",
        "mode": AttentionMode.TRAINING,
        "layout": "BHSD",
    }


def test_main_attention_propagates_training_mode_to_strict_dispatch():
    params = anima_attention.AttentionParams.create_attention_params(
        "flash", False, AttentionMode.TRAINING
    )
    q = torch.randn(1, 3, 2, 8)
    captured = {}

    def _dispatch(query, key, value, **kwargs):
        captured.update(kwargs)
        return query

    with patch.object(anima_attention, "dispatch_attention", side_effect=_dispatch):
        out = anima_attention.attention(q, q, q, params)

    assert out.shape == (1, 3, 16)
    assert captured["backend"] == "flash"
    assert captured["mode"] == AttentionMode.TRAINING
