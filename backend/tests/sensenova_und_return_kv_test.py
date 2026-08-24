"""Pin the vendor ``return_kv`` seam (Phase U-0) and the training prefix loop.

Two properties, both CPU-sized:

* the seam is opt-in -- every ``return_kv`` parameter defaults to False, so a
  caller that does not ask for it keeps the exact pre-seam return arity;
* a per-layer checkpointed prefix loop that carries K/V out as checkpoint
  OUTPUTS reproduces the vendor ``DynamicCache`` contents BITWISE. That parity
  is U-0's first exit criterion, verified there against the real 42-layer
  checkpoint and pinned here against a 2-layer toy.
"""

import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.sensenova.vendor.configuration_neo_chat import NEOLLMConfig  # noqa: E402
from core.models.sensenova.vendor.modeling_qwen3 import (  # noqa: E402
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3Model,
    create_block_causal_mask,
)
from core.training.probes.sensenova_und_prefix import training_prefix_forward  # noqa: E402


SEAM_METHODS = (
    Qwen3Attention.forward_und,
    Qwen3Attention.forward,
    Qwen3DecoderLayer.forward_und,
    Qwen3DecoderLayer.forward,
)


def _toy_config() -> NEOLLMConfig:
    config = NEOLLMConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        max_position_embeddings_hw=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    return config


def _toy_model() -> Qwen3Model:
    torch.manual_seed(0)
    model = Qwen3Model(_toy_config())
    model.eval()
    return model


def _toy_inputs(model: Qwen3Model):
    torch.manual_seed(1)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 7))
    t_idx = torch.arange(input_ids.shape[1], dtype=torch.long)
    indexes = torch.stack([t_idx, torch.zeros_like(t_idx), torch.zeros_like(t_idx)], dim=0)
    return input_ids, indexes, {"full_attention": create_block_causal_mask(indexes[0])}


@pytest.mark.parametrize("method", SEAM_METHODS, ids=lambda m: m.__qualname__)
def test_return_kv_is_opt_in(method):
    parameter = inspect.signature(method).parameters["return_kv"]
    assert parameter.default is False
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_return_kv_default_keeps_the_two_tuple():
    model = _toy_model()
    input_ids, indexes, mask = _toy_inputs(model)
    layer = model.layers[0]
    hidden = model.embed_tokens(input_ids)
    with torch.no_grad():
        output = layer.forward_und(
            hidden, None, True, False, indexes, mask["full_attention"]
        )
    assert isinstance(output, torch.Tensor)


def test_return_kv_refused_on_the_generation_branch():
    model = _toy_model()
    input_ids, indexes, mask = _toy_inputs(model)
    hidden = model.embed_tokens(input_ids)
    indicators = torch.ones(hidden.shape[:2], dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="understanding branch"):
        model.layers[0].forward_und.__self__.forward(
            hidden, indicators, False, True, indexes, None, return_kv=True
        )


@pytest.mark.parametrize("checkpoint_from", [0, None, 1])
def test_training_prefix_loop_matches_the_vendor_cache_bitwise(checkpoint_from):
    model = _toy_model()
    input_ids, indexes, mask = _toy_inputs(model)
    with torch.no_grad():
        vendor = model(
            input_ids=input_ids, indexes=indexes, attention_mask=mask, use_cache=True
        )
        hidden, cache = training_prefix_forward(
            model, input_ids, indexes, mask, checkpoint_from=checkpoint_from
        )
        hidden = model.norm(hidden)

    assert len(cache.layers) == model.config.num_hidden_layers
    for index, layer in enumerate(cache.layers):
        assert torch.equal(layer.keys, vendor.past_key_values.layers[index].keys)
        assert torch.equal(layer.values, vendor.past_key_values.layers[index].values)
    assert torch.equal(hidden, vendor.last_hidden_state)


def test_prefix_kv_carries_a_gradient_path():
    """The property §13.1 infers structurally: no ``no_grad``, so K/V have a graph."""
    model = _toy_model()
    input_ids, indexes, mask = _toy_inputs(model)
    _hidden, cache = training_prefix_forward(model, input_ids, indexes, mask)
    assert all(
        layer.keys.grad_fn is not None and layer.values.grad_fn is not None
        for layer in cache.layers
    )
    cache.layers[0].keys.sum().backward()
    assert model.layers[0].self_attn.k_proj.weight.grad is not None
