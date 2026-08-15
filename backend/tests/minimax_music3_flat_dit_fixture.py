"""A tiny, REAL (non-zero-byte) flat DiT + matching official/ config-and-weight
pair, for ``minimax_music3_loader_test.py``'s round-trip test.

Every shape below is derived the same way the production shapes were (see
``core.models.minimax_music3.flat_remap``'s module docstring): from
``MiniMaxMusic3Transformer1DModel.__init__`` and
``MiniMaxMusic3ConditionEncoder.__init__``'s own arithmetic, just with tiny
dimensions instead of the checkpoint's real ones. It is NOT a scaled-down copy
of the real checkpoint's numbers -- the values are arbitrary (deterministic,
seeded) and exist only to prove the remap moves the RIGHT tensor to the RIGHT
destination with the RIGHT split, not to approximate real audio quality.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file

# Tiny transformer geometry.
IN_CHANNELS = 4
CONDITION_DIM = 6
NUM_LAYERS = 1
NUM_HEADS = 2
HEAD_DIM = 4
FF_INNER_DIM = 8
ROTARY_DIM = 4
FOURIER_EMBEDDING_DIM = 8

INNER_DIM = NUM_HEADS * HEAD_DIM  # 8
CONCAT_CHANNELS = 2 * IN_CHANNELS + CONDITION_DIM  # 14

# Tiny condition-encoder geometry (its out_dim MUST equal the transformer's
# condition_dim -- that is a real cross-component invariant, not a fixture
# convenience).
CONDITION_HIDDEN_DIM = 6
NUM_CONDITION_LAYERS = 2


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def write_tiny_flat_dit_and_official_tree(tmp_path) -> dict:
    """Write ``<tmp_path>/official/{transformer,condition_encoder}`` (config +
    real tiny weights) and ``<tmp_path>/diffusion_models/`` (the flat DiT,
    real tiny weights, DIFFERENT values from official/'s so a substitution
    bug is visible).

    Returns a dict with ``official`` (the official/ dir), ``dit_path`` (the
    flat file), ``expected_proj_in_weight`` (the tensor the flat file's
    ``proj_in`` should end up holding), and
    ``official_placeholder_proj_in_weight`` (official/'s OWN placeholder for
    the same slot -- deliberately different, for the negative assertion).
    """
    root = str(tmp_path)
    official = os.path.join(root, "official")
    generator = torch.Generator().manual_seed(1234)

    transformer_config = {
        "_class_name": "MiniMaxMusic3Transformer1DModel",
        "in_channels": IN_CHANNELS,
        "condition_dim": CONDITION_DIM,
        "num_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "attention_head_dim": HEAD_DIM,
        "ff_inner_dim": FF_INNER_DIM,
        "rotary_dim": ROTARY_DIM,
        "fourier_embedding_dim": FOURIER_EMBEDDING_DIM,
    }
    _write_json(os.path.join(official, "transformer", "config.json"), transformer_config)

    condition_encoder_config = {
        "_class_name": "MiniMaxMusic3ConditionEncoder",
        "condition_hidden_dim": CONDITION_HIDDEN_DIM,
        "num_condition_layers": NUM_CONDITION_LAYERS,
        "out_dim": CONDITION_DIM,
        "input_sampling_rate": 24000,
        "input_hop_length": 960,
        "output_sampling_rate": 44100,
        "output_hop_length": 512,
    }
    _write_json(os.path.join(official, "condition_encoder", "config.json"), condition_encoder_config)

    # official/'s OWN placeholder weight (not read by the flat path, but
    # written so the negative assertion has something concrete to differ
    # from -- and so this fixture would also serve a non-flat / official-only
    # test unmodified).
    official_proj_in = torch.randn(INNER_DIM, CONCAT_CHANNELS, generator=generator)
    official_transformer_sd = {
        "proj_in.weight": official_proj_in,
        "proj_out.weight": torch.randn(IN_CHANNELS, INNER_DIM, generator=generator),
        "preprocess_conv.weight": torch.randn(CONCAT_CHANNELS, CONCAT_CHANNELS, 1, generator=generator),
        "postprocess_conv.weight": torch.randn(IN_CHANNELS, IN_CHANNELS, 1, generator=generator),
        "time_proj.weight": torch.randn(FOURIER_EMBEDDING_DIM // 2, 1, generator=generator),
        "time_embed.linear_1.weight": torch.randn(INNER_DIM, FOURIER_EMBEDDING_DIM, generator=generator),
        "time_embed.linear_1.bias": torch.randn(INNER_DIM, generator=generator),
        "time_embed.linear_2.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator),
        "time_embed.linear_2.bias": torch.randn(INNER_DIM, generator=generator),
        "transformer_blocks.0.norm1.weight": torch.randn(INNER_DIM, generator=generator),
        "transformer_blocks.0.norm1.bias": torch.randn(INNER_DIM, generator=generator),
        "transformer_blocks.0.norm2.weight": torch.randn(INNER_DIM, generator=generator),
        "transformer_blocks.0.norm2.bias": torch.randn(INNER_DIM, generator=generator),
        "transformer_blocks.0.attn.to_q.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator),
        "transformer_blocks.0.attn.to_k.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator),
        "transformer_blocks.0.attn.to_v.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator),
        "transformer_blocks.0.attn.to_out.0.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator),
        "transformer_blocks.0.ff_in.weight": torch.randn(FF_INNER_DIM * 2, INNER_DIM, generator=generator),
        "transformer_blocks.0.ff_in.bias": torch.randn(FF_INNER_DIM * 2, generator=generator),
        "transformer_blocks.0.ff_out.weight": torch.randn(INNER_DIM, FF_INNER_DIM, generator=generator),
        "transformer_blocks.0.ff_out.bias": torch.randn(INNER_DIM, generator=generator),
    }
    save_file(official_transformer_sd, os.path.join(official, "transformer", "diffusion_pytorch_model.safetensors"))

    official_condition_encoder_sd = {
        "layer_weight_logits": torch.randn(NUM_CONDITION_LAYERS, generator=generator),
        "layer_scale": torch.randn(1, generator=generator),
        "proj.weight": torch.randn(CONDITION_DIM, CONDITION_HIDDEN_DIM, 3, generator=generator),
        "proj.bias": torch.randn(CONDITION_DIM, generator=generator),
    }
    save_file(official_condition_encoder_sd, os.path.join(official, "condition_encoder", "diffusion_pytorch_model.safetensors"))

    # The FLAT file -- DIFFERENT values from official/'s, deliberately, using
    # the flat repack's OWN naming (fused qkv, .gamma/.beta, GEGLU ff naming,
    # the folded-in condition encoder, and the rotary inv_freq that has no
    # destination -- see flat_remap.py).
    flat_proj_in = torch.randn(INNER_DIM, CONCAT_CHANNELS, generator=generator) + 100.0  # visibly different range
    fused_qkv = torch.randn(3 * INNER_DIM, INNER_DIM, generator=generator)
    flat_state_dict = {
        "diffusion_transformer.transformer.project_in.weight": flat_proj_in,
        "diffusion_transformer.transformer.project_out.weight": torch.randn(IN_CHANNELS, INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": torch.randn(ROTARY_DIM // 2, generator=generator),
        "diffusion_transformer.preprocess_conv.weight": torch.randn(CONCAT_CHANNELS, CONCAT_CHANNELS, 1, generator=generator),
        "diffusion_transformer.postprocess_conv.weight": torch.randn(IN_CHANNELS, IN_CHANNELS, 1, generator=generator),
        "diffusion_transformer.timestep_features.weight": torch.randn(FOURIER_EMBEDDING_DIM // 2, 1, generator=generator),
        "diffusion_transformer.to_timestep_embed.0.weight": torch.randn(INNER_DIM, FOURIER_EMBEDDING_DIM, generator=generator),
        "diffusion_transformer.to_timestep_embed.0.bias": torch.randn(INNER_DIM, generator=generator),
        "diffusion_transformer.to_timestep_embed.2.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator),
        "diffusion_transformer.to_timestep_embed.2.bias": torch.randn(INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.layers.0.pre_norm.gamma": torch.randn(INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.layers.0.pre_norm.beta": torch.randn(INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.layers.0.ff_norm.gamma": torch.randn(INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.layers.0.ff_norm.beta": torch.randn(INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight": fused_qkv,
        "diffusion_transformer.transformer.layers.0.self_attn.to_out.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.layers.0.ff.ff.0.proj.weight": torch.randn(FF_INNER_DIM * 2, INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.layers.0.ff.ff.0.proj.bias": torch.randn(FF_INNER_DIM * 2, generator=generator),
        "diffusion_transformer.transformer.layers.0.ff.ff.2.weight": torch.randn(INNER_DIM, FF_INNER_DIM, generator=generator),
        "diffusion_transformer.transformer.layers.0.ff.ff.2.bias": torch.randn(INNER_DIM, generator=generator),
        "cond_layer_logits": torch.randn(NUM_CONDITION_LAYERS, generator=generator),
        "cond_layer_scale": torch.randn(1, generator=generator),
        "latent_conditioners.0.weight": torch.randn(CONDITION_DIM, CONDITION_HIDDEN_DIM, 3, generator=generator),
        "latent_conditioners.0.bias": torch.randn(CONDITION_DIM, generator=generator),
    }
    dit_path = os.path.join(root, "diffusion_models", "minimax_music3_dit_fp16.safetensors")
    os.makedirs(os.path.dirname(dit_path), exist_ok=True)
    save_file(flat_state_dict, dit_path)

    return {
        "official": official,
        "dit_path": dit_path,
        "expected_proj_in_weight": flat_proj_in,
        "official_placeholder_proj_in_weight": official_proj_in,
        "expected_fused_qkv": fused_qkv,
    }
