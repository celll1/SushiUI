"""``core.models.minimax_music3.flat_remap`` -- design doc phase 9.

Pure-function coverage: every test here works on plain tensors and key
lists, no model snapshot needed. The REAL-checkpoint numeric proof (that the
QKV split lands in the same rows the official fp32 transformer's separate
q/k/v Linears hold, etc.) was run manually against
``M:/model/minimax-music3`` while writing this module and is not repeated
here -- this file proves the remap's LOGIC is correct and TOTAL for any
input shaped like the real files; ``minimax_music3_loader_test.py``'s
``test_flat_dit_with_official_present_is_now_loadable`` proves an actual
tiny round-trip through the vendored classes.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_flat_remap_test.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3 import flat_remap as fr  # noqa: E402


# ---------------------------------------------------------------------------
# DiT: structural plan
# ---------------------------------------------------------------------------

def _tiny_flat_dit_keys(num_layers=2):
    keys = [
        "diffusion_transformer.preprocess_conv.weight",
        "diffusion_transformer.postprocess_conv.weight",
        "diffusion_transformer.timestep_features.weight",
        "diffusion_transformer.to_timestep_embed.0.weight",
        "diffusion_transformer.to_timestep_embed.0.bias",
        "diffusion_transformer.to_timestep_embed.2.weight",
        "diffusion_transformer.to_timestep_embed.2.bias",
        "diffusion_transformer.transformer.project_in.weight",
        "diffusion_transformer.transformer.project_out.weight",
        "diffusion_transformer.transformer.rotary_pos_emb.inv_freq",
        "cond_layer_logits",
        "cond_layer_scale",
        "latent_conditioners.0.weight",
        "latent_conditioners.0.bias",
    ]
    for i in range(num_layers):
        keys += [
            f"diffusion_transformer.transformer.layers.{i}.pre_norm.gamma",
            f"diffusion_transformer.transformer.layers.{i}.pre_norm.beta",
            f"diffusion_transformer.transformer.layers.{i}.ff_norm.gamma",
            f"diffusion_transformer.transformer.layers.{i}.ff_norm.beta",
            f"diffusion_transformer.transformer.layers.{i}.self_attn.to_qkv.weight",
            f"diffusion_transformer.transformer.layers.{i}.self_attn.to_out.weight",
            f"diffusion_transformer.transformer.layers.{i}.ff.ff.0.proj.weight",
            f"diffusion_transformer.transformer.layers.{i}.ff.ff.0.proj.bias",
            f"diffusion_transformer.transformer.layers.{i}.ff.ff.2.weight",
            f"diffusion_transformer.transformer.layers.{i}.ff.ff.2.bias",
        ]
    return keys


def test_dit_plan_has_no_unrecognized_keys_and_drops_only_rotary():
    plan = fr.plan_flat_dit_keys(_tiny_flat_dit_keys())
    assert plan.unrecognized == []
    assert list(plan.dropped.keys()) == ["diffusion_transformer.transformer.rotary_pos_emb.inv_freq"]


def test_dit_plan_produces_12_tensors_per_layer_from_10():
    plan = fr.plan_flat_dit_keys(_tiny_flat_dit_keys(num_layers=1))
    produced = plan.produced_keys(fr.TRANSFORMER_COMPONENT)
    layer0 = {k for k in produced if k.startswith("transformer_blocks.0.")}
    assert layer0 == {
        "transformer_blocks.0.norm1.weight",
        "transformer_blocks.0.norm1.bias",
        "transformer_blocks.0.norm2.weight",
        "transformer_blocks.0.norm2.bias",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.attn.to_out.0.weight",
        "transformer_blocks.0.ff_in.weight",
        "transformer_blocks.0.ff_in.bias",
        "transformer_blocks.0.ff_out.weight",
        "transformer_blocks.0.ff_out.bias",
    }


def test_dit_plan_condition_encoder_renames():
    plan = fr.plan_flat_dit_keys(_tiny_flat_dit_keys(num_layers=0))
    assert plan.renames[fr.CONDITION_ENCODER_COMPONENT] == {
        "cond_layer_logits": "layer_weight_logits",
        "cond_layer_scale": "layer_scale",
        "latent_conditioners.0.weight": "proj.weight",
        "latent_conditioners.0.bias": "proj.bias",
    }


def test_dit_plan_flags_a_genuinely_unknown_key():
    plan = fr.plan_flat_dit_keys(_tiny_flat_dit_keys(num_layers=0) + ["diffusion_transformer.some_new_thing.weight"])
    assert plan.unrecognized == ["diffusion_transformer.some_new_thing.weight"]


def test_dit_plan_flags_an_unknown_per_layer_suffix():
    keys = _tiny_flat_dit_keys(num_layers=0) + ["diffusion_transformer.transformer.layers.0.some_new_thing.weight"]
    plan = fr.plan_flat_dit_keys(keys)
    assert plan.unrecognized == ["diffusion_transformer.transformer.layers.0.some_new_thing.weight"]


# ---------------------------------------------------------------------------
# DiT: applying the plan to real tensors -- the QKV split
# ---------------------------------------------------------------------------

def _tiny_flat_dit_state_dict(num_layers=1, dim=8):
    keys = _tiny_flat_dit_keys(num_layers=num_layers)
    out = {}
    for key in keys:
        if key.endswith("self_attn.to_qkv.weight"):
            out[key] = torch.randn(3 * dim, dim)
        elif key.endswith((".gamma", ".beta")) or "layer_weight_logits" in key or key == "cond_layer_logits":
            out[key] = torch.randn(dim)
        elif key in ("cond_layer_scale",):
            out[key] = torch.randn(1)
        elif key.endswith("to_out.weight"):
            out[key] = torch.randn(dim, dim)
        elif "ff.ff.0.proj.weight" in key:
            out[key] = torch.randn(2 * dim, dim)
        elif "ff.ff.0.proj.bias" in key:
            out[key] = torch.randn(2 * dim)
        elif "ff.ff.2.weight" in key:
            out[key] = torch.randn(dim, 2 * dim)
        elif "ff.ff.2.bias" in key:
            out[key] = torch.randn(dim)
        elif key.endswith("preprocess_conv.weight") or key.endswith("postprocess_conv.weight"):
            out[key] = torch.randn(dim, dim, 1)
        elif key.endswith("latent_conditioners.0.weight"):
            out[key] = torch.randn(dim, dim, 3)
        elif key.endswith("latent_conditioners.0.bias"):
            out[key] = torch.randn(dim)
        elif key.endswith("rotary_pos_emb.inv_freq"):
            out[key] = torch.randn(4)
        elif key.endswith("timestep_features.weight"):
            out[key] = torch.randn(4, 1)
        else:
            out[key] = torch.randn(dim, dim)
    return out


def test_apply_dit_splits_qkv_contiguously_in_q_k_v_order():
    sd = _tiny_flat_dit_state_dict(num_layers=1, dim=8)
    fused = sd["diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight"]

    remapped = fr.apply_flat_dit_state_dict(sd)
    q = remapped["transformer"]["transformer_blocks.0.attn.to_q.weight"]
    k = remapped["transformer"]["transformer_blocks.0.attn.to_k.weight"]
    v = remapped["transformer"]["transformer_blocks.0.attn.to_v.weight"]

    assert torch.equal(q, fused[0:8])
    assert torch.equal(k, fused[8:16])
    assert torch.equal(v, fused[16:24])


def test_apply_dit_drops_rotary_and_produces_no_key_for_it():
    sd = _tiny_flat_dit_state_dict(num_layers=1, dim=8)
    remapped = fr.apply_flat_dit_state_dict(sd)
    all_keys = set(remapped["transformer"]) | set(remapped["condition_encoder"])
    assert not any("rotary" in k or "inv_freq" in k for k in all_keys)


def test_apply_dit_raises_on_an_unrecognized_key():
    sd = _tiny_flat_dit_state_dict(num_layers=0, dim=8)
    sd["totally_unexpected_key"] = torch.randn(1)
    with pytest.raises(ValueError, match="unrecognized|matched no known rule"):
        fr.apply_flat_dit_state_dict(sd)


def test_apply_dit_raises_on_a_qkv_row_count_not_divisible_by_three():
    sd = _tiny_flat_dit_state_dict(num_layers=1, dim=8)
    sd["diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight"] = torch.randn(23, 8)
    with pytest.raises(ValueError, match="rows"):
        fr.apply_flat_dit_state_dict(sd)


# ---------------------------------------------------------------------------
# Text encoder: pruned detection and refusal
# ---------------------------------------------------------------------------

def test_is_pruned_detects_any_single_tell():
    assert fr.is_pruned_flat_text_encoder(["model.embed_tokens_prefill.weight"])
    assert fr.is_pruned_flat_text_encoder(["model.embed_tokens_audio.weight"])
    assert fr.is_pruned_flat_text_encoder(["model.lm_head_pruned.weight"])
    assert not fr.is_pruned_flat_text_encoder(["model.embed_tokens.weight", "model.lm_head.weight"])


def test_plan_text_encoder_raises_pruned_not_supported():
    with pytest.raises(fr.PrunedTextEncoderNotSupported, match="phase-plan item 10"):
        fr.plan_flat_text_encoder_keys(["model.embed_tokens_prefill.weight", "model.lm_head_pruned.weight"])


def test_apply_text_encoder_raises_pruned_not_supported_before_touching_tensors():
    sd = {"model.embed_tokens_prefill.weight": torch.empty(0)}
    with pytest.raises(fr.PrunedTextEncoderNotSupported):
        fr.apply_flat_text_encoder_state_dict(sd)


def test_raise_if_pruned_is_a_no_op_for_the_non_pruned_layout():
    fr.raise_if_pruned_flat_text_encoder(["model.embed_tokens.weight", "model.lm_head.weight"])  # must not raise


def test_lm_layer_whitelist_flags_a_genuinely_foreign_suffix():
    """F1 regression: `model.layers.N.<anything>` must NOT be accepted
    unconditionally -- only the whitelisted Qwen3 suffixes."""
    plan = fr.plan_flat_text_encoder_keys(["model.layers.0.brand_new_thing.weight"])
    assert plan.unrecognized == ["model.layers.0.brand_new_thing.weight"]


def test_lm_layer_whitelist_flags_the_pruned_variants_fused_qkv_and_gate_up():
    """F1 regression: even with the three `_PRUNED_TELLS` stripped out (as if
    a future repack folded a fused-QKV layout into a file that no longer
    carries the vocab-split tells), the fused per-layer keys must still be
    refused rather than silently renamed straight through."""
    keys = [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",  # this ONE suffix is shared/valid
    ]
    plan = fr.plan_flat_text_encoder_keys(keys)
    assert set(plan.unrecognized) == {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
    }
    assert plan.renames[fr.LANGUAGE_MODEL_COMPONENT] == {
        "model.layers.0.self_attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
    }


# ---------------------------------------------------------------------------
# Text encoder: structural plan (non-pruned)
# ---------------------------------------------------------------------------

def _tiny_flat_text_encoder_keys(num_layers=2, num_depth_layers=1):
    keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.lm_head.weight",
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
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"model.layers.{i}.mlp.up_proj.weight",
            f"model.layers.{i}.mlp.down_proj.weight",
            f"model.layers.{i}.self_attn.q_proj.weight",
            f"model.layers.{i}.self_attn.k_proj.weight",
            f"model.layers.{i}.self_attn.v_proj.weight",
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"model.layers.{i}.self_attn.q_norm.weight",
            f"model.layers.{i}.self_attn.k_norm.weight",
        ]
    for i in range(num_depth_layers):
        keys += [
            f"model.audio_decoder.layers.{i}.input_layernorm.weight",
            f"model.audio_decoder.layers.{i}.post_attention_layernorm.weight",
            f"model.audio_decoder.layers.{i}.mlp.gate_proj.weight",
            f"model.audio_decoder.layers.{i}.mlp.up_proj.weight",
            f"model.audio_decoder.layers.{i}.mlp.down_proj.weight",
            f"model.audio_decoder.layers.{i}.self_attn.q_proj.weight",
            f"model.audio_decoder.layers.{i}.self_attn.k_proj.weight",
            f"model.audio_decoder.layers.{i}.self_attn.v_proj.weight",
            f"model.audio_decoder.layers.{i}.self_attn.o_proj.weight",
        ]
    return keys


def test_text_encoder_plan_has_no_unrecognized_keys_and_drops_only_tokenizer_json():
    plan = fr.plan_flat_text_encoder_keys(_tiny_flat_text_encoder_keys())
    assert plan.unrecognized == []
    assert list(plan.dropped.keys()) == ["tokenizer_json"]


def test_text_encoder_plan_lm_layers_are_identity():
    plan = fr.plan_flat_text_encoder_keys(_tiny_flat_text_encoder_keys(num_layers=1, num_depth_layers=0))
    lm_renames = plan.renames[fr.LANGUAGE_MODEL_COMPONENT]
    for key in lm_renames:
        if key.startswith("model.layers."):
            assert lm_renames[key] == key  # unchanged


def test_text_encoder_plan_lm_head_moves_out_of_the_model_namespace():
    plan = fr.plan_flat_text_encoder_keys(_tiny_flat_text_encoder_keys(num_layers=0, num_depth_layers=0))
    assert plan.renames[fr.LANGUAGE_MODEL_COMPONENT]["model.lm_head.weight"] == "lm_head.weight"
    assert plan.renames[fr.LANGUAGE_MODEL_COMPONENT]["model.embed_tokens.weight"] == "model.embed_tokens.weight"
    assert plan.renames[fr.LANGUAGE_MODEL_COMPONENT]["model.norm.weight"] == "model.norm.weight"


def test_text_encoder_plan_audio_extra_embedding_is_a_cross_component_rename_to_depth_decoder():
    plan = fr.plan_flat_text_encoder_keys(_tiny_flat_text_encoder_keys(num_layers=0, num_depth_layers=0))
    assert "model.audio_extra_embedding.weight" not in plan.renames[fr.LANGUAGE_MODEL_COMPONENT]
    assert plan.renames[fr.RVQ_DEPTH_DECODER_COMPONENT]["model.audio_extra_embedding.weight"] == "audio_embeddings.weight"


def test_text_encoder_plan_depth_decoder_layer_renames():
    plan = fr.plan_flat_text_encoder_keys(_tiny_flat_text_encoder_keys(num_layers=0, num_depth_layers=1))
    depth = plan.renames[fr.RVQ_DEPTH_DECODER_COMPONENT]
    assert depth["model.audio_decoder.layers.0.mlp.down_proj.weight"] == "layers.0.down_proj.weight"
    assert depth["model.audio_decoder.layers.0.mlp.gate_proj.weight"] == "layers.0.gate_proj.weight"
    assert depth["model.audio_decoder.layers.0.mlp.up_proj.weight"] == "layers.0.up_proj.weight"
    assert depth["model.audio_decoder.layers.0.self_attn.q_proj.weight"] == "layers.0.attn.to_q.weight"
    assert depth["model.audio_decoder.layers.0.self_attn.k_proj.weight"] == "layers.0.attn.to_k.weight"
    assert depth["model.audio_decoder.layers.0.self_attn.v_proj.weight"] == "layers.0.attn.to_v.weight"
    assert depth["model.audio_decoder.layers.0.self_attn.o_proj.weight"] == "layers.0.attn.to_out.weight"
    assert depth["model.audio_decoder.layers.0.input_layernorm.weight"] == "layers.0.input_layernorm.weight"


def test_text_encoder_plan_audio_heads_renamed_by_index():
    plan = fr.plan_flat_text_encoder_keys(_tiny_flat_text_encoder_keys(num_layers=0, num_depth_layers=0))
    depth = plan.renames[fr.RVQ_DEPTH_DECODER_COMPONENT]
    for i in range(7):
        assert depth[f"model.audio_decoder.audio_heads.{i}.weight"] == f"audio_heads.{i}.weight"


def test_apply_text_encoder_drops_tokenizer_json_and_produces_no_key_for_it():
    keys = _tiny_flat_text_encoder_keys(num_layers=1, num_depth_layers=1)
    sd = {k: torch.randn(2, 2) if not k.endswith("_json") else torch.zeros(4, dtype=torch.uint8) for k in keys}
    remapped = fr.apply_flat_text_encoder_state_dict(sd)
    all_values = set(remapped[fr.LANGUAGE_MODEL_COMPONENT]) | set(remapped[fr.RVQ_DEPTH_DECODER_COMPONENT])
    assert "tokenizer_json" not in all_values
    assert sum(v.numel() for v in remapped[fr.LANGUAGE_MODEL_COMPONENT].values()) + \
        sum(v.numel() for v in remapped[fr.RVQ_DEPTH_DECODER_COMPONENT].values()) == \
        sum(v.numel() for k, v in sd.items() if k != "tokenizer_json")


# ---------------------------------------------------------------------------
# Totality assertion
# ---------------------------------------------------------------------------

def test_assert_totality_passes_on_matching_sets():
    fr.assert_state_dict_matches_module_keys(["a", "b"], ["b", "a"], component="x")  # must not raise


def test_assert_totality_raises_on_missing_key():
    with pytest.raises(ValueError, match="NOT produced"):
        fr.assert_state_dict_matches_module_keys(["a"], ["a", "b"], component="x")


def test_assert_totality_raises_on_extra_key():
    with pytest.raises(ValueError, match="does not expect"):
        fr.assert_state_dict_matches_module_keys(["a", "b"], ["a"], component="x")


def test_expected_module_state_dict_keys_reads_a_real_module():
    module = torch.nn.Linear(4, 4, bias=True)
    assert fr.expected_module_state_dict_keys(module) == {"weight", "bias"}
