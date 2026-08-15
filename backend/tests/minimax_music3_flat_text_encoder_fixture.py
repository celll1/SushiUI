"""A tiny, REAL flat text encoder + matching official/ config pair, for
``minimax_music3_loader_test.py``'s
``build_language_model_and_depth_decoder_from_flat_text_encoder`` round-trip
test (F2 in the phase-9 audit: that builder had zero callers and zero tests).

Mirrors ``minimax_music3_flat_dit_fixture.py``'s shape: tiny dimensions, real
(non-zero-byte) tensors, deterministic seeded random values -- not a
scaled-down copy of the real checkpoint's numbers.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

# Tiny Qwen3 geometry -- no GQA (num_key_value_heads == num_attention_heads),
# so q/k/v/o are all [hidden_size, hidden_size]. Verified against a real tiny
# `Qwen3Config` + `AutoModelForCausalLM.from_config` on the installed
# transformers 5.1.0: this shape produces the expected 14 state_dict keys
# (embed_tokens, norm, lm_head, and 11 per-layer keys) with no
# `rotary_emb.inv_freq` entry (it is a non-persistent buffer).
VOCAB_SIZE = 32
HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 16
NUM_HIDDEN_LAYERS = 1
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 4
MAX_POSITION_EMBEDDINGS = 32
ROPE_THETA = 1_000_000.0

# Tiny RVQ depth decoder geometry.
DEPTH_HIDDEN_SIZE = 8
DEPTH_NUM_LAYERS = 1
DEPTH_NUM_HEADS = 2
DEPTH_INTERMEDIATE_SIZE = 16
AUDIO_VOCAB_SIZE = 4
NUM_CODEBOOKS = 3  # -> 2 audio_heads, audio_embeddings rows = 4 * (3-1) = 8
DEPTH_MAX_POSITION_EMBEDDINGS = 8


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def write_tiny_flat_text_encoder_and_official_tree(tmp_path) -> dict:
    """Write ``<tmp_path>/official/{language_model,rvq_depth_decoder}`` (config
    only -- this builder never reads official/'s WEIGHTS for either
    component) and ``<tmp_path>/text_encoders/`` (the flat file, real tiny
    weights).

    Returns ``{official, text_encoder_path, expected_lm_head_weight,
    expected_audio_embeddings_weight}``.
    """
    root = str(tmp_path)
    official = os.path.join(root, "official")
    generator = torch.Generator().manual_seed(5678)

    lm_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "rope_parameters": {"rope_theta": ROPE_THETA, "rope_type": "default"},
    }
    _write_json(os.path.join(official, "language_model", "config.json"), lm_config)

    depth_config = {
        "_class_name": "MiniMaxMusic3RVQDepthDecoder",
        "hidden_size": DEPTH_HIDDEN_SIZE,
        "num_layers": DEPTH_NUM_LAYERS,
        "num_attention_heads": DEPTH_NUM_HEADS,
        "intermediate_size": DEPTH_INTERMEDIATE_SIZE,
        "audio_vocab_size": AUDIO_VOCAB_SIZE,
        "num_codebooks": NUM_CODEBOOKS,
        "max_position_embeddings": DEPTH_MAX_POSITION_EMBEDDINGS,
    }
    _write_json(os.path.join(official, "rvq_depth_decoder", "config.json"), depth_config)

    lm_head = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=generator)
    audio_embeddings_rows = AUDIO_VOCAB_SIZE * (NUM_CODEBOOKS - 1)
    audio_extra_embedding = torch.randn(audio_embeddings_rows, DEPTH_HIDDEN_SIZE, generator=generator)

    flat_state_dict = {
        "model.embed_tokens.weight": torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=generator),
        "model.norm.weight": torch.randn(HIDDEN_SIZE, generator=generator),
        "model.lm_head.weight": lm_head,
        "model.audio_extra_embedding.weight": audio_extra_embedding,
        "model.audio_decoder.norm.weight": torch.randn(DEPTH_HIDDEN_SIZE, generator=generator),
        "model.audio_decoder.pos_embedding.weight": torch.randn(DEPTH_MAX_POSITION_EMBEDDINGS, DEPTH_HIDDEN_SIZE, generator=generator),
        "model.audio_decoder.projection.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
        "tokenizer_json": torch.zeros(4, dtype=torch.uint8),
    }
    for i in range(NUM_CODEBOOKS - 1):
        flat_state_dict[f"model.audio_decoder.audio_heads.{i}.weight"] = torch.randn(
            AUDIO_VOCAB_SIZE, DEPTH_HIDDEN_SIZE, generator=generator,
        )
    for layer in range(NUM_HIDDEN_LAYERS):
        prefix = f"model.layers.{layer}."
        flat_state_dict.update({
            prefix + "input_layernorm.weight": torch.randn(HIDDEN_SIZE, generator=generator),
            prefix + "post_attention_layernorm.weight": torch.randn(HIDDEN_SIZE, generator=generator),
            prefix + "mlp.gate_proj.weight": torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, generator=generator),
            prefix + "mlp.up_proj.weight": torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, generator=generator),
            prefix + "mlp.down_proj.weight": torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, generator=generator),
            prefix + "self_attn.q_proj.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.k_proj.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.v_proj.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.o_proj.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.q_norm.weight": torch.randn(HEAD_DIM, generator=generator),
            prefix + "self_attn.k_norm.weight": torch.randn(HEAD_DIM, generator=generator),
        })
    for depth_layer in range(DEPTH_NUM_LAYERS):
        prefix = f"model.audio_decoder.layers.{depth_layer}."
        flat_state_dict.update({
            prefix + "input_layernorm.weight": torch.randn(DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "post_attention_layernorm.weight": torch.randn(DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "mlp.gate_proj.weight": torch.randn(DEPTH_INTERMEDIATE_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "mlp.up_proj.weight": torch.randn(DEPTH_INTERMEDIATE_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "mlp.down_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_INTERMEDIATE_SIZE, generator=generator),
            prefix + "self_attn.q_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.k_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.v_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.o_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
        })

    text_encoder_path = os.path.join(root, "text_encoders", "minimax_music3_text_encoder_bf16.safetensors")
    os.makedirs(os.path.dirname(text_encoder_path), exist_ok=True)
    save_file(flat_state_dict, text_encoder_path)

    return {
        "official": official,
        "text_encoder_path": text_encoder_path,
        "expected_lm_head_weight": lm_head,
        "expected_audio_embeddings_weight": audio_extra_embedding,
    }
