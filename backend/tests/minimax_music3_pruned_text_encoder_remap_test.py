"""``core.models.minimax_music3.pruned_text_encoder_remap`` -- design doc phase 10.

Pure-function / small-tensor coverage, mirroring ``minimax_music3_flat_remap_test.py``'s
shape: no multi-GB weight file needed. The REAL-checkpoint numeric proof (that the split
q/k/v/gate/up land bit-identically to ``official/``'s own separate Linears, that
``lm_head_pruned`` row 0 is end-of-audio, etc.) was run manually against
``<MODEL_ROOT>/minimax-music3`` while writing this module and is recorded in its module
docstring and in the MiniMax Music 3 loader/remap contract --
this file proves the remap's LOGIC is correct and TOTAL for any input shaped like the real
files; ``minimax_music3_loader_test.py``'s pruned round-trip test proves an actual tiny
round-trip through a real (patched) ``Qwen3ForCausalLM``.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_pruned_text_encoder_remap_test.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3 import pruned_text_encoder_remap as pv  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_eoa_row_is_zero_and_head_vocab_is_semantic_plus_one():
    """Pinned against the real checkpoint (see module docstring): row 0 of
    `lm_head_pruned` is end-of-audio, rows 1..16384 are semantic codes 0..16383."""
    assert pv.EOA_LM_HEAD_ROW == 0
    assert pv.AUDIO_HEAD_VOCAB_SIZE == pv.SEMANTIC_VOCAB_SIZE + 1 == 16385


# ---------------------------------------------------------------------------
# lm_qkv_split_sizes
# ---------------------------------------------------------------------------

def test_lm_qkv_split_sizes_gqa_explicit():
    sizes = pv.lm_qkv_split_sizes({
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096,
    })
    assert sizes == (4096, 1024, 1024)


def test_lm_qkv_split_sizes_falls_back_to_plain_mha_when_fields_absent():
    sizes = pv.lm_qkv_split_sizes({"num_attention_heads": 2, "hidden_size": 8})
    assert sizes == (8, 8, 8)  # head_dim = 8//2 = 4, kv_heads defaults to num_attention_heads=2 -> kv_dim=8


# ---------------------------------------------------------------------------
# Plan: structural coverage over a tiny fused key set
# ---------------------------------------------------------------------------

_TINY_LM_CONFIG = {"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4, "hidden_size": 8}
# q_dim=8, kv_dim=4 -> qkv_proj is [16, 8]


def _tiny_pruned_keys(num_layers=2, num_depth_layers=1):
    keys = [
        "model.embed_tokens_prefill.weight",
        "model.embed_tokens_audio.weight",
        "model.lm_head_pruned.weight",
        "model.norm.weight",
        "model.audio_extra_embedding.weight",
        "model.audio_decoder.norm.weight",
        "model.audio_decoder.pos_embedding.weight",
        "model.audio_decoder.projection.weight",
        "tokenizer_json",
    ]
    for i in range(7):
        keys.append(f"model.audio_decoder.audio_heads.{i}.weight")
    for i in range(num_layers):
        keys += [
            f"model.layers.{i}.input_layernorm.weight",
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"model.layers.{i}.self_attn.qkv_proj.weight",
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.self_attn.q_norm.weight",
            f"model.layers.{i}.self_attn.k_norm.weight",
            f"model.layers.{i}.mlp.gate_up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight",
        ]
    for i in range(num_depth_layers):
        keys += [
            f"model.audio_decoder.layers.{i}.input_layernorm.weight",
            f"model.audio_decoder.layers.{i}.post_attention_layernorm.weight",
            f"model.audio_decoder.layers.{i}.self_attn.qkv_proj.weight",
            f"model.audio_decoder.layers.{i}.self_attn.o_proj.weight",
            f"model.audio_decoder.layers.{i}.mlp.gate_up_proj.weight",
            f"model.audio_decoder.layers.{i}.mlp.down_proj.weight",
        ]
    return keys


def test_plan_has_no_unrecognized_keys_and_drops_only_tokenizer_json():
    plan = pv.plan_pruned_text_encoder_keys(_tiny_pruned_keys(), _TINY_LM_CONFIG)
    assert plan.unrecognized == []
    assert list(plan.dropped.keys()) == ["tokenizer_json"]


def test_plan_vocab_table_renames():
    plan = pv.plan_pruned_text_encoder_keys(_tiny_pruned_keys(num_layers=0, num_depth_layers=0), _TINY_LM_CONFIG)
    lm = plan.renames[pv.LANGUAGE_MODEL_COMPONENT]
    assert lm["model.embed_tokens_prefill.weight"] == "model.embed_tokens.weight"
    assert lm["model.embed_tokens_audio.weight"] == "model.embed_tokens_audio.weight"
    assert lm["model.lm_head_pruned.weight"] == "lm_head_pruned.weight"
    assert lm["model.norm.weight"] == "model.norm.weight"


def test_plan_audio_extra_embedding_is_cross_component_rename():
    plan = pv.plan_pruned_text_encoder_keys(_tiny_pruned_keys(num_layers=0, num_depth_layers=0), _TINY_LM_CONFIG)
    assert "model.audio_extra_embedding.weight" not in plan.renames[pv.LANGUAGE_MODEL_COMPONENT]
    assert plan.renames[pv.RVQ_DEPTH_DECODER_COMPONENT]["model.audio_extra_embedding.weight"] == "audio_embeddings.weight"


def test_plan_lm_qkv_split_uses_config_derived_uneven_sizes():
    plan = pv.plan_pruned_text_encoder_keys(_tiny_pruned_keys(num_layers=1, num_depth_layers=0), _TINY_LM_CONFIG)
    splits = plan.splits[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.self_attn.qkv_proj.weight"]
    assert splits == (
        ("model.layers.0.self_attn.q_proj.weight", 8),
        ("model.layers.0.self_attn.k_proj.weight", 4),
        ("model.layers.0.self_attn.v_proj.weight", 4),
    )


def test_plan_lm_gate_up_split_is_equal_halves_with_no_config_dependence():
    plan = pv.plan_pruned_text_encoder_keys(_tiny_pruned_keys(num_layers=1, num_depth_layers=0), _TINY_LM_CONFIG)
    splits = plan.splits[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.mlp.gate_up_proj.weight"]
    dest_keys = [d for d, _size in splits]
    assert dest_keys == ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"]
    assert all(size == -1 for _d, size in splits)  # equal-split sentinel, resolved from the tensor at apply time


def test_plan_depth_decoder_qkv_split_is_equal_thirds():
    plan = pv.plan_pruned_text_encoder_keys(_tiny_pruned_keys(num_layers=0, num_depth_layers=1), _TINY_LM_CONFIG)
    splits = plan.splits[pv.RVQ_DEPTH_DECODER_COMPONENT]["model.audio_decoder.layers.0.self_attn.qkv_proj.weight"]
    dest_keys = [d for d, _size in splits]
    assert dest_keys == ["layers.0.attn.to_q.weight", "layers.0.attn.to_k.weight", "layers.0.attn.to_v.weight"]


def test_plan_flags_a_genuinely_unknown_lm_layer_suffix():
    keys = _tiny_pruned_keys(num_layers=0, num_depth_layers=0) + ["model.layers.0.something_new.weight"]
    plan = pv.plan_pruned_text_encoder_keys(keys, _TINY_LM_CONFIG)
    assert plan.unrecognized == ["model.layers.0.something_new.weight"]


def test_plan_flags_a_genuinely_unknown_depth_layer_suffix():
    keys = _tiny_pruned_keys(num_layers=0, num_depth_layers=0) + ["model.audio_decoder.layers.0.something_new.weight"]
    plan = pv.plan_pruned_text_encoder_keys(keys, _TINY_LM_CONFIG)
    assert plan.unrecognized == ["model.audio_decoder.layers.0.something_new.weight"]


def test_plan_still_accepts_an_already_unfused_lm_suffix():
    """Not present in any real pruned file (always fused), but the plan reuses
    `flat_remap._LM_LAYER_SUFFIX_WHITELIST` unconditionally, so an already-separate
    q_proj must still be accepted -- proves the reuse is wired, not copy-pasted."""
    keys = ["model.layers.0.self_attn.q_proj.weight"]
    plan = pv.plan_pruned_text_encoder_keys(keys, _TINY_LM_CONFIG)
    assert plan.unrecognized == []
    assert plan.renames[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.self_attn.q_proj.weight"] == \
        "model.layers.0.self_attn.q_proj.weight"


# ---------------------------------------------------------------------------
# Apply: real tensors, splits verified against the fused source
# ---------------------------------------------------------------------------

def _tiny_pruned_state_dict(num_layers=1, num_depth_layers=1, dim=8):
    generator = torch.Generator().manual_seed(42)
    keys = _tiny_pruned_keys(num_layers=num_layers, num_depth_layers=num_depth_layers)
    out = {}
    for key in keys:
        if key == "model.embed_tokens_prefill.weight":
            out[key] = torch.randn(20, dim, generator=generator)
        elif key == "model.embed_tokens_audio.weight":
            out[key] = torch.randn(6, dim, generator=generator)
        elif key == "model.lm_head_pruned.weight":
            out[key] = torch.randn(7, dim, generator=generator)
        elif key == "model.audio_extra_embedding.weight":
            out[key] = torch.randn(4, dim, generator=generator)
        elif key == "tokenizer_json":
            out[key] = torch.zeros(4, dtype=torch.uint8)
        elif key.endswith("self_attn.qkv_proj.weight") and "audio_decoder" not in key:
            out[key] = torch.randn(16, dim, generator=generator)  # q=8,k=4,v=4
        elif key.endswith("self_attn.qkv_proj.weight"):  # depth decoder: equal thirds
            out[key] = torch.randn(3 * dim, dim, generator=generator)
        elif key.endswith("mlp.gate_up_proj.weight"):
            out[key] = torch.randn(4 * dim, dim, generator=generator)  # intermediate=2*dim -> gate_up=4*dim
        else:
            out[key] = torch.randn(dim, dim, generator=generator)
    return out


def test_apply_splits_lm_qkv_in_order_with_config_derived_sizes():
    sd = _tiny_pruned_state_dict(num_layers=1, num_depth_layers=0, dim=8)
    fused = sd["model.layers.0.self_attn.qkv_proj.weight"]
    remapped = pv.apply_pruned_text_encoder_state_dict(sd, _TINY_LM_CONFIG)
    q = remapped[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.self_attn.q_proj.weight"]
    k = remapped[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.self_attn.k_proj.weight"]
    v = remapped[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.self_attn.v_proj.weight"]
    assert torch.equal(q, fused[0:8])
    assert torch.equal(k, fused[8:12])
    assert torch.equal(v, fused[12:16])


def test_apply_splits_lm_gate_up_evenly():
    sd = _tiny_pruned_state_dict(num_layers=1, num_depth_layers=0, dim=8)
    fused = sd["model.layers.0.mlp.gate_up_proj.weight"]
    remapped = pv.apply_pruned_text_encoder_state_dict(sd, _TINY_LM_CONFIG)
    gate = remapped[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.mlp.gate_proj.weight"]
    up = remapped[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.mlp.up_proj.weight"]
    half = fused.shape[0] // 2
    assert torch.equal(gate, fused[:half])
    assert torch.equal(up, fused[half:])


def test_apply_splits_depth_decoder_qkv_in_equal_thirds():
    sd = _tiny_pruned_state_dict(num_layers=0, num_depth_layers=1, dim=8)
    fused = sd["model.audio_decoder.layers.0.self_attn.qkv_proj.weight"]
    remapped = pv.apply_pruned_text_encoder_state_dict(sd, _TINY_LM_CONFIG)
    third = fused.shape[0] // 3
    q = remapped[pv.RVQ_DEPTH_DECODER_COMPONENT]["layers.0.attn.to_q.weight"]
    k = remapped[pv.RVQ_DEPTH_DECODER_COMPONENT]["layers.0.attn.to_k.weight"]
    v = remapped[pv.RVQ_DEPTH_DECODER_COMPONENT]["layers.0.attn.to_v.weight"]
    assert torch.equal(q, fused[0:third])
    assert torch.equal(k, fused[third:2 * third])
    assert torch.equal(v, fused[2 * third:3 * third])


def test_apply_splits_are_not_aliased_views_of_the_fused_source():
    """Mirrors flat_remap's own `.contiguous().clone()` regression concern: a split result
    sharing storage with the fused tensor would break a future `safetensors.save_file`."""
    sd = _tiny_pruned_state_dict(num_layers=1, num_depth_layers=0, dim=8)
    remapped = pv.apply_pruned_text_encoder_state_dict(sd, _TINY_LM_CONFIG)
    q = remapped[pv.LANGUAGE_MODEL_COMPONENT]["model.layers.0.self_attn.q_proj.weight"]
    assert q.is_contiguous()
    fused = sd["model.layers.0.self_attn.qkv_proj.weight"]
    assert q.data_ptr() != fused.data_ptr()


def test_apply_raises_on_unrecognized_key():
    sd = _tiny_pruned_state_dict(num_layers=0, num_depth_layers=0, dim=8)
    sd["totally_unexpected_key"] = torch.randn(1)
    with pytest.raises(ValueError, match="unrecognized|matched no known rule"):
        pv.apply_pruned_text_encoder_state_dict(sd, _TINY_LM_CONFIG)


def test_apply_raises_when_lm_qkv_row_count_does_not_match_config():
    sd = _tiny_pruned_state_dict(num_layers=1, num_depth_layers=0, dim=8)
    sd["model.layers.0.self_attn.qkv_proj.weight"] = torch.randn(15, 8)  # expected 16 (8+4+4)
    with pytest.raises(ValueError, match="rows"):
        pv.apply_pruned_text_encoder_state_dict(sd, _TINY_LM_CONFIG)


def test_apply_raises_when_depth_qkv_not_divisible_by_three():
    sd = _tiny_pruned_state_dict(num_layers=0, num_depth_layers=1, dim=8)
    sd["model.audio_decoder.layers.0.self_attn.qkv_proj.weight"] = torch.randn(23, 8)
    with pytest.raises(ValueError, match="rows"):
        pv.apply_pruned_text_encoder_state_dict(sd, _TINY_LM_CONFIG)


def test_apply_produces_no_key_for_tokenizer_json():
    sd = _tiny_pruned_state_dict(num_layers=1, num_depth_layers=1, dim=8)
    remapped = pv.apply_pruned_text_encoder_state_dict(sd, _TINY_LM_CONFIG)
    all_values = set(remapped[pv.LANGUAGE_MODEL_COMPONENT]) | set(remapped[pv.RVQ_DEPTH_DECODER_COMPONENT])
    assert "tokenizer_json" not in all_values
