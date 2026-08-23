from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.ops.sensenova_ops import forward_gen_decoder_layers


class _RecordingLayer(nn.Module):
    def __init__(self, width: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.proj = nn.Linear(width, width, bias=False)
        self.gradient_checkpointing = True
        self.forward_calls = []
        self.calls = []

    def __call__(self, *args, **kwargs):
        raise AssertionError("layer.__call__ must be bypassed explicitly")

    def forward(
        self,
        hidden_states,
        image_gen_indicators,
        exist_non_image_gen_tokens,
        exist_image_gen_tokens,
        indexes=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        self.forward_calls.append(
            {
                "image_gen_indicators": image_gen_indicators.detach().clone(),
                "exist_non_image_gen_tokens": exist_non_image_gen_tokens,
                "exist_image_gen_tokens": exist_image_gen_tokens,
                "indexes": indexes,
                "attention_mask": attention_mask,
                "cache": past_key_values,
                "use_cache": use_cache,
                "update_cache": kwargs.get("update_cache"),
                "cache_position": cache_position,
            }
        )
        return self.forward_gen(
            hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    def forward_gen(
        self,
        hidden_states,
        image_gen_indicators,
        exist_non_image_gen_tokens,
        exist_image_gen_tokens,
        indexes,
        attention_mask,
        past_key_values,
        use_cache,
        cache_position=None,
        **kwargs,
    ):
        cache = past_key_values
        cache_layer = cache.layers[self.layer_idx]
        self.calls.append(
            {
                "cache": cache,
                "image_gen_indicators": image_gen_indicators.detach().clone(),
                "exist_non_image_gen_tokens": exist_non_image_gen_tokens,
                "exist_image_gen_tokens": exist_image_gen_tokens,
                "indexes": indexes,
                "attention_mask": attention_mask,
                "update_cache": kwargs.get("update_cache"),
                "use_cache": use_cache,
                "key_id": id(cache_layer.keys),
                "key_len": cache_layer.keys.shape[-2],
            }
        )
        prefix_value = cache_layer.keys.mean().to(hidden_states.dtype)
        return self.proj(hidden_states) + hidden_states + prefix_value


class _TinyGenModel(nn.Module):
    def __init__(self, width: int = 4, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_RecordingLayer(width, i) for i in range(layers)])
        self.norm_mot_gen = nn.LayerNorm(width)


def _prefix_cache(layers: int = 2):
    return SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.randn(1, 2, 3 + i, 2),
                values=torch.randn(1, 2, 3 + i, 2),
            )
            for i in range(layers)
        ]
    )


def _cache_snapshot(cache):
    return [
        (
            id(layer.keys),
            id(layer.values),
            layer.keys.shape,
            layer.values.shape,
            layer.keys.clone(),
            layer.values.clone(),
        )
        for layer in cache.layers
    ]


def _assert_cache_snapshot_unchanged(before, cache):
    for old, current in zip(before, cache.layers):
        key_id, value_id, key_shape, value_shape, keys, values = old
        assert id(current.keys) == key_id
        assert id(current.values) == value_id
        assert current.keys.shape == key_shape
        assert current.values.shape == value_shape
        torch.testing.assert_close(current.keys, keys)
        torch.testing.assert_close(current.values, values)


def _assert_forward_contract(layer, *, indexes, attention_mask, cache, hidden_shape):
    assert layer.forward_calls
    for call in layer.forward_calls:
        assert call["image_gen_indicators"].shape == hidden_shape[:2]
        assert bool(call["image_gen_indicators"].all())
        assert call["exist_non_image_gen_tokens"] is False
        assert call["exist_image_gen_tokens"] is True
        assert call["indexes"] is indexes
        assert call["attention_mask"] is attention_mask
        assert call["cache"] is cache
        assert call["update_cache"] is False
        assert call["use_cache"] is False


def test_gen_decoder_keeps_prefix_cache_immutable_and_passes_training_flags():
    model = _TinyGenModel()
    cache = _prefix_cache()
    before = _cache_snapshot(cache)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    indexes = torch.arange(9, dtype=torch.long).reshape(3, 3)
    attention_mask = torch.randn(1, 1, 3, 5)

    output = forward_gen_decoder_layers(
        model,
        hidden,
        indexes=indexes,
        prefix_cache=cache,
        attention_mask=attention_mask,
    )
    output.square().mean().backward()

    for layer in model.layers:
        assert len(layer.calls) == 1
        _assert_forward_contract(
            layer,
            indexes=indexes,
            attention_mask=attention_mask,
            cache=cache,
            hidden_shape=hidden.shape,
        )
    _assert_cache_snapshot_unchanged(before, cache)


def test_checkpointed_and_plain_gen_decoder_outputs_and_gradients_match():
    torch.manual_seed(7)
    plain_model = _TinyGenModel()
    checkpointed_model = copy.deepcopy(plain_model)
    plain_cache = _prefix_cache()
    checkpointed_cache = copy.deepcopy(plain_cache)
    indexes = torch.zeros(3, 3, dtype=torch.long)
    attention_mask = torch.randn(1, 1, 3, 5)
    plain_hidden = torch.randn(1, 3, 4, requires_grad=True)
    checkpointed_hidden = plain_hidden.detach().clone().requires_grad_(True)
    checkpointed_cache_before = _cache_snapshot(checkpointed_cache)

    hook_events = {"pre": [], "post": [], "backward": []}
    hook_handles = []
    for index, layer in enumerate(checkpointed_model.layers):
        hook_handles.extend(
            [
                layer.register_forward_pre_hook(
                    lambda module, inputs, index=index: hook_events["pre"].append(index)
                ),
                layer.register_forward_hook(
                    lambda module, inputs, output, index=index: hook_events["post"].append(index)
                ),
                layer.register_full_backward_hook(
                    lambda module, grad_input, grad_output, index=index: hook_events[
                        "backward"
                    ].append(index)
                ),
            ]
        )

    plain_output = forward_gen_decoder_layers(
        plain_model,
        plain_hidden,
        indexes=indexes,
        prefix_cache=plain_cache,
        attention_mask=attention_mask,
        checkpoint_layers=False,
    )
    checkpointed_output = forward_gen_decoder_layers(
        checkpointed_model,
        checkpointed_hidden,
        indexes=indexes,
        prefix_cache=checkpointed_cache,
        attention_mask=attention_mask,
        checkpoint_layers=True,
    )
    torch.testing.assert_close(checkpointed_output, plain_output)

    plain_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    for handle in hook_handles:
        handle.remove()
    torch.testing.assert_close(checkpointed_hidden.grad, plain_hidden.grad)
    for plain_param, checkpointed_param in zip(
        plain_model.parameters(), checkpointed_model.parameters()
    ):
        torch.testing.assert_close(checkpointed_param.grad, plain_param.grad)
    _assert_cache_snapshot_unchanged(checkpointed_cache_before, checkpointed_cache)

    # Recompute still reaches forward_gen through the normal all-gen forward.
    assert all(len(layer.calls) == 2 for layer in checkpointed_model.layers)
    assert all(len(layer.forward_calls) == 2 for layer in checkpointed_model.layers)
    for layer in checkpointed_model.layers:
        _assert_forward_contract(
            layer,
            indexes=indexes,
            attention_mask=attention_mask,
            cache=checkpointed_cache,
            hidden_shape=checkpointed_hidden.shape,
        )
    checkpointed_calls = [
        call for layer in checkpointed_model.layers for call in layer.calls
    ]
    assert all(call["cache"] is checkpointed_cache for call in checkpointed_calls)
    assert all(call["update_cache"] is False for call in checkpointed_calls)
    assert all(hook_events["pre"].count(index) == 2 for index in range(2))
    assert all(hook_events["post"].count(index) >= 1 for index in range(2))
    assert all(hook_events["backward"].count(index) == 1 for index in range(2))


def test_gen_decoder_refuses_inference_cache_buffers():
    model = _TinyGenModel(layers=1)
    cache = _prefix_cache(layers=1)
    cache.layers[0].flash_k_cache = torch.empty(1)

    try:
        forward_gen_decoder_layers(
            model,
            torch.randn(1, 2, 4),
            indexes=torch.zeros(3, 2, dtype=torch.long),
            prefix_cache=cache,
        )
    except ValueError as exc:
        assert "inference flash KV buffers" in str(exc)
    else:
        raise AssertionError("prepared inference cache was accepted")


@pytest.mark.parametrize(
    "cache, message",
    [
        (SimpleNamespace(), "non-empty"),
        (SimpleNamespace(layers=[]), "non-empty"),
        (_prefix_cache(layers=1), "expected 2"),
    ],
)
def test_gen_decoder_refuses_missing_or_wrong_prefix_layers(cache, message):
    with pytest.raises(ValueError, match=message):
        forward_gen_decoder_layers(
            _TinyGenModel(),
            torch.randn(1, 2, 4),
            indexes=torch.zeros(3, 2, dtype=torch.long),
            prefix_cache=cache,
        )


@pytest.mark.parametrize("field", ["keys", "values"])
def test_gen_decoder_refuses_missing_or_attached_prefix_tensors(field):
    model = _TinyGenModel()
    cache = _prefix_cache()
    setattr(cache.layers[0], field, None)
    with pytest.raises(ValueError, match=f"missing non-empty {field}"):
        forward_gen_decoder_layers(
            model,
            torch.randn(1, 2, 4),
            indexes=torch.zeros(3, 2, dtype=torch.long),
            prefix_cache=cache,
        )

    cache = _prefix_cache()
    getattr(cache.layers[0], field).requires_grad_(True)
    with pytest.raises(ValueError, match=f"{field} tensors must be detached"):
        forward_gen_decoder_layers(
            model,
            torch.randn(1, 2, 4),
            indexes=torch.zeros(3, 2, dtype=torch.long),
            prefix_cache=cache,
        )


def test_gen_decoder_refuses_streamer_branch_marker():
    model = _TinyGenModel()
    cache = _prefix_cache()
    cache._kv_cache_streamer_branch = "cond"
    with pytest.raises(ValueError, match="streamer"):
        forward_gen_decoder_layers(
            model,
            torch.randn(1, 2, 4),
            indexes=torch.zeros(3, 2, dtype=torch.long),
            prefix_cache=cache,
        )
