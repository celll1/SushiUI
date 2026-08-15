"""A tiny, REAL pruned-vocabulary flat text encoder + matching official/ config pair,
for ``minimax_music3_loader_test.py``'s
``build_language_model_and_depth_decoder_from_pruned_flat_text_encoder`` round-trip test
(design doc phase 10).

Mirrors ``minimax_music3_flat_text_encoder_fixture.py``'s shape (tiny hidden dims, real
non-zero-byte tensors, deterministic seeded random values), with two differences that ARE
the substance of this phase: fused ``self_attn.qkv_proj`` / ``mlp.gate_up_proj`` per layer
(GQA-uneven for the language model: ``num_key_value_heads < num_attention_heads``, so the
split sizes are NOT equal thirds -- exercises the one config-dependent split this phase adds),
and the three-way vocabulary split (``embed_tokens_prefill`` / ``embed_tokens_audio`` /
``lm_head_pruned``).

The vocabulary tables are sized at the REAL checkpoint's row counts
(``AUDIO_CODE_OFFSET`` = 151,675 text rows, ``SEMANTIC_VOCAB_SIZE`` = 16,384 audio rows,
``+1`` for end-of-audio) rather than tiny placeholder counts: the loader cross-checks these
counts against those two constants (design doc phase 10, "verify from evidence") and would
otherwise refuse a fixture shaped any other way -- see
``build_language_model_and_depth_decoder_from_pruned_flat_text_encoder``'s docstring. This
stays cheap because only the ROW COUNT is real; ``HIDDEN_SIZE`` (the column count) is tiny,
so the biggest tensor here is ``151675 * 8`` float32 elements (~4.9 MB), not multi-GB.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

from core.models.minimax_music3.defaults import AUDIO_CODE_OFFSET
from core.models.minimax_music3.pruned_text_encoder_remap import AUDIO_HEAD_VOCAB_SIZE, SEMANTIC_VOCAB_SIZE

# Tiny Qwen3 geometry -- GQA (num_key_value_heads < num_attention_heads), so the qkv split
# is uneven: q_dim=8, k_dim=v_dim=4, qkv_proj is [16, 8]. Verified against a real tiny
# `Qwen3Config` + `AutoModelForCausalLM.from_config` on the installed transformers 5.1.0,
# the same way the non-pruned fixture verifies its own tiny geometry.
HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 16  # gate_up_proj is [32, 8]
NUM_HIDDEN_LAYERS = 1
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 1
HEAD_DIM = 4
MAX_POSITION_EMBEDDINGS = 32
ROPE_THETA = 1_000_000.0

Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM  # 8
KV_DIM = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 4

# Tiny RVQ depth decoder geometry -- plain MHA (no GQA field), so its qkv_proj splits evenly
# in three: [3 * DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE].
DEPTH_HIDDEN_SIZE = 8
DEPTH_NUM_LAYERS = 1
DEPTH_NUM_HEADS = 2
DEPTH_INTERMEDIATE_SIZE = 16  # gate_up_proj is [32, 8]
AUDIO_VOCAB_SIZE = 4
NUM_CODEBOOKS = 3  # -> 2 audio_heads, audio_embeddings rows = 4 * (3-1) = 8
DEPTH_MAX_POSITION_EMBEDDINGS = 8


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def write_tiny_pruned_text_encoder_and_official_tree(tmp_path) -> dict:
    """Write ``<tmp_path>/official/{language_model,rvq_depth_decoder}`` (config only) and
    ``<tmp_path>/text_encoders/`` (the pruned flat file, real tiny-hidden-dim weights).

    Returns ``{official, text_encoder_path, expected_lm_head_pruned_weight,
    expected_embed_tokens_audio_weight, expected_audio_embeddings_weight, expected_fused_qkv,
    expected_fused_gate_up}``.
    """
    root = str(tmp_path)
    official = os.path.join(root, "official")
    generator = torch.Generator().manual_seed(9012)

    lm_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 200_000,  # official/'s OWN (merged) vocab_size -- the loader overrides this
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

    lm_head_pruned = torch.randn(AUDIO_HEAD_VOCAB_SIZE, HIDDEN_SIZE, generator=generator)
    embed_tokens_audio = torch.randn(SEMANTIC_VOCAB_SIZE, HIDDEN_SIZE, generator=generator)
    audio_embeddings_rows = AUDIO_VOCAB_SIZE * (NUM_CODEBOOKS - 1)
    audio_extra_embedding = torch.randn(audio_embeddings_rows, DEPTH_HIDDEN_SIZE, generator=generator)
    fused_qkv = torch.randn(Q_DIM + 2 * KV_DIM, HIDDEN_SIZE, generator=generator)
    fused_gate_up = torch.randn(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, generator=generator)

    flat_state_dict = {
        "model.embed_tokens_prefill.weight": torch.randn(AUDIO_CODE_OFFSET, HIDDEN_SIZE, generator=generator),
        "model.embed_tokens_audio.weight": embed_tokens_audio,
        "model.lm_head_pruned.weight": lm_head_pruned,
        "model.norm.weight": torch.randn(HIDDEN_SIZE, generator=generator),
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
            prefix + "self_attn.qkv_proj.weight": fused_qkv,
            prefix + "self_attn.o_proj.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.q_norm.weight": torch.randn(HEAD_DIM, generator=generator),
            prefix + "self_attn.k_norm.weight": torch.randn(HEAD_DIM, generator=generator),
            prefix + "mlp.gate_up_proj.weight": fused_gate_up,
            prefix + "mlp.down_proj.weight": torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, generator=generator),
        })
    depth_fused_qkv_by_layer = {}
    depth_fused_gate_up_by_layer = {}
    for depth_layer in range(DEPTH_NUM_LAYERS):
        prefix = f"model.audio_decoder.layers.{depth_layer}."
        depth_qkv = torch.randn(3 * DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator)
        depth_gate_up = torch.randn(2 * DEPTH_INTERMEDIATE_SIZE, DEPTH_HIDDEN_SIZE, generator=generator)
        depth_fused_qkv_by_layer[depth_layer] = depth_qkv
        depth_fused_gate_up_by_layer[depth_layer] = depth_gate_up
        flat_state_dict.update({
            prefix + "input_layernorm.weight": torch.randn(DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "post_attention_layernorm.weight": torch.randn(DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.qkv_proj.weight": depth_qkv,
            prefix + "self_attn.o_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "mlp.gate_up_proj.weight": depth_gate_up,
            prefix + "mlp.down_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_INTERMEDIATE_SIZE, generator=generator),
        })

    text_encoder_path = os.path.join(root, "text_encoders", "minimax_music3_text_encoder_pruned_bf16.safetensors")
    os.makedirs(os.path.dirname(text_encoder_path), exist_ok=True)
    save_file(flat_state_dict, text_encoder_path)

    return {
        "official": official,
        "text_encoder_path": text_encoder_path,
        "expected_lm_head_pruned_weight": lm_head_pruned,
        "expected_embed_tokens_audio_weight": embed_tokens_audio,
        "expected_audio_embeddings_weight": audio_extra_embedding,
        "expected_fused_qkv": fused_qkv,
        "expected_fused_gate_up": fused_gate_up,
        "expected_depth_fused_qkv_by_layer": depth_fused_qkv_by_layer,
        "expected_depth_fused_gate_up_by_layer": depth_fused_gate_up_by_layer,
    }
