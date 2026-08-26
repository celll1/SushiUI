"""``causal_fastpath`` coverage: the fast K/V-conduit path in ``forward_und``.

Nothing exercises ``causal_fastpath=True`` elsewhere: ``sensenova_und_lora_test.py``
and ``sensenova_und_reference_test.py`` stub ``_Layer.forward`` behind
``**kwargs``, which swallows the argument without running it. This file runs
the real ``Qwen3Attention``/``Qwen3Model`` against it, on the toy geometry used
throughout the sibling test files (4 q heads / 2 kv heads, head_dim 8).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.attention import AttentionMode  # noqa: E402
from core.models.sensenova.vendor.configuration_neo_chat import NEOLLMConfig  # noqa: E402
from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel  # noqa: E402
from core.models.sensenova.vendor import modeling_qwen3 as mq  # noqa: E402
from core.models.sensenova.vendor.modeling_qwen3 import (  # noqa: E402
    Qwen3Attention,
    Qwen3Model,
    create_block_causal_mask,
    is_plain_causal_thw_index,
)
from core.training.ops import sensenova_ops as ops  # noqa: E402


def _toy_config(layers: int = 4) -> NEOLLMConfig:
    config = NEOLLMConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        max_position_embeddings_hw=128,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.layer_types = ["full_attention"] * layers
    return config


def _toy_model(layers: int = 4) -> Qwen3Model:
    torch.manual_seed(0)
    model = Qwen3Model(_toy_config(layers))
    model.eval()
    return model


def _toy_inputs(model: Qwen3Model, seq_len: int = 11):
    torch.manual_seed(1)
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
    t_idx = torch.arange(seq_len, dtype=torch.long)
    indexes = torch.stack([t_idx, torch.zeros_like(t_idx), torch.zeros_like(t_idx)], dim=0)
    mask = {"full_attention": create_block_causal_mask(indexes[0])}
    return input_ids, indexes, mask


# ---------------------------------------------------------------------------
# (a) fast-vs-eager K/V parity, real 4-layer Qwen3Model
# ---------------------------------------------------------------------------


def test_fast_vs_eager_kv_parity():
    """Layer 0 bitwise; later layers within fp32 accumulation-order tolerance.

    The classifier is forced True (fast) then False (eager) around the same
    weights/inputs, isolating the algorithm from the classifier decision.
    """
    model = _toy_model(layers=4)
    input_ids, indexes, mask = _toy_inputs(model)

    fast_cache = ops.forward_und_prefix_layers(model, input_ids, indexes, mask, checkpoint_layers=False)

    orig = mq.is_plain_causal_thw_index
    mq.is_plain_causal_thw_index = lambda idx: False
    try:
        eager_cache = ops.forward_und_prefix_layers(model, input_ids, indexes, mask, checkpoint_layers=False)
    finally:
        mq.is_plain_causal_thw_index = orig

    assert torch.equal(fast_cache.layers[0].keys, eager_cache.layers[0].keys)
    assert torch.equal(fast_cache.layers[0].values, eager_cache.layers[0].values)

    for i in range(1, len(fast_cache.layers)):
        dk = (fast_cache.layers[i].keys - eager_cache.layers[i].keys).abs().max().item()
        dv = (fast_cache.layers[i].values - eager_cache.layers[i].values).abs().max().item()
        assert dk < 1e-5, f"layer{i} K maxabs {dk:.3e}"
        assert dv < 1e-5, f"layer{i} V maxabs {dv:.3e}"


# ---------------------------------------------------------------------------
# (b) classifier gate over the real get_thw_indexes, 0/1/2 reference images
# ---------------------------------------------------------------------------


class _IndexHost:
    """Exposes exactly the attributes ``NEOChatModel.get_thw_indexes`` reads.

    ``t_indexes`` (what the classifier consumes) does not depend on
    ``grid_hw``, so this only needs the id sequence -- no ViT, no tokenizer.
    """

    img_start_token_id = 900
    img_context_token_id = 901
    downsample_ratio = 0.5

    get_thw_indexes = NEOChatModel.get_thw_indexes


def _ids_for_reference_images(n_images: int, ctx_per_image: int = 4) -> torch.Tensor:
    ids = [1, 1]
    for _ in range(n_images):
        ids.append(_IndexHost.img_start_token_id)
        ids.extend([_IndexHost.img_context_token_id] * ctx_per_image)
        ids.append(1)
    ids.extend([1, 1])
    return torch.tensor(ids, dtype=torch.long)


@pytest.mark.parametrize("n_images", [0, 1, 2, 3])
def test_classifier_matches_real_mask_for_n_reference_images(n_images):
    ids = _ids_for_reference_images(n_images)
    host = _IndexHost()
    indexes = host.get_thw_indexes(ids, grid_hw=None)
    t_index = indexes[0]

    mask = create_block_causal_mask(t_index)
    L = t_index.numel()
    plain_causal = torch.where(
        torch.arange(L).unsqueeze(0) <= torch.arange(L).unsqueeze(1),
        torch.tensor(0.0),
        torch.tensor(float("-inf")),
    )
    actually_plain_causal = torch.equal(mask[0, 0], plain_causal)

    assert is_plain_causal_thw_index(t_index) == actually_plain_causal
    if n_images == 0:
        assert is_plain_causal_thw_index(t_index) is True
    else:
        assert is_plain_causal_thw_index(t_index) is False


# ---------------------------------------------------------------------------
# (c) dispatch_attention call-count/reach, both checkpoint_layers values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_dispatch_attention_reached_once_per_layer(checkpoint_layers):
    model = _toy_model(layers=4)
    input_ids, indexes, mask = _toy_inputs(model)

    calls = []
    real = mq.dispatch_attention

    def counting(q, k, v, **kw):
        calls.append(kw.get("is_causal"))
        return real(q, k, v, **kw)

    mq.dispatch_attention = counting
    try:
        ops.forward_und_prefix_layers(model, input_ids, indexes, mask, checkpoint_layers=checkpoint_layers)
    finally:
        mq.dispatch_attention = real

    assert len(calls) == len(model.layers)
    assert all(c is True for c in calls)


# ---------------------------------------------------------------------------
# D1: the conduit must not be handed enable_gqa=True bait when it resolves
# native (the slow SDPA path); it should stay unexpanded when flash is what
# will actually run (expanding there would only cost memory for nothing).
# ---------------------------------------------------------------------------


def _single_attn():
    attn = Qwen3Attention(_toy_config(layers=1), layer_idx=0)
    attn.eval()
    return attn


def _und_inputs(seq_len=6):
    hidden = torch.randn(1, seq_len, 32)
    idx = torch.arange(seq_len)
    indexes = torch.stack([idx, torch.zeros_like(idx), torch.zeros_like(idx)])
    return hidden, indexes


def test_causal_fastpath_expands_kv_when_resolved_backend_is_native():
    attn = _single_attn()
    attn._attn_backend = "native"
    attn._attn_mode = AttentionMode.INFERENCE
    hidden, indexes = _und_inputs()

    calls = []
    real = mq.dispatch_attention

    def spy(q, k, v, **kw):
        calls.append((q.shape[1], k.shape[1]))
        return real(q, k, v, **kw)

    mq.dispatch_attention = spy
    try:
        attn.forward_und(hidden, indexes, None, causal_fastpath=True)
    finally:
        mq.dispatch_attention = real

    assert len(calls) == 1
    q_heads, k_heads = calls[0]
    assert q_heads == attn.config.num_attention_heads == 4
    # Pre-expanded to match q heads -- dispatch_attention's own GQA check
    # (k.shape[2] != q.shape[2] in its canonical BSHD view) then sees equal
    # heads and never sets enable_gqa=True for the native SDPA call.
    assert k_heads == q_heads


def test_causal_fastpath_leaves_kv_unexpanded_when_resolved_backend_is_flash():
    attn = _single_attn()
    attn._attn_backend = "flash"
    attn._attn_mode = AttentionMode.INFERENCE
    hidden, indexes = _und_inputs()

    calls = []
    real = mq.dispatch_attention

    def spy(q, k, v, **kw):
        calls.append((q.shape[1], k.shape[1]))
        return real(q, k, v, **kw)

    mq.dispatch_attention = spy
    try:
        attn.forward_und(hidden, indexes, None, causal_fastpath=True)
    finally:
        mq.dispatch_attention = real

    assert len(calls) == 1
    q_heads, k_heads = calls[0]
    # Flash broadcasts GQA natively; expanding here would only cost memory.
    assert k_heads == attn.config.num_key_value_heads == 2
    assert q_heads == 4


# ---------------------------------------------------------------------------
# D3: is_causal=True is top-left-aligned on native SDPA, bottom-right on
# FlashAttention, once K is longer than Q (a KV cache). causal_fastpath must
# refuse rather than silently pick one.
# ---------------------------------------------------------------------------


def test_causal_fastpath_refuses_when_key_longer_than_query():
    from transformers.cache_utils import DynamicCache

    attn = _single_attn()
    attn._attn_backend = "native"
    attn._attn_mode = AttentionMode.INFERENCE

    cache = DynamicCache()
    hidden0, indexes0 = _und_inputs(seq_len=2)
    mask0 = {"full_attention": create_block_causal_mask(indexes0[0])}
    attn.forward_und(hidden0, indexes0, mask0["full_attention"], past_key_values=cache, causal_fastpath=False)

    hidden1, indexes1 = _und_inputs(seq_len=5)
    with pytest.raises(RuntimeError, match="causal_fastpath requires query/key sequence lengths to match"):
        attn.forward_und(hidden1, indexes1, None, past_key_values=cache, causal_fastpath=True)
