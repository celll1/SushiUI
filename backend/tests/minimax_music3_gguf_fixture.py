"""Tiny, REAL (non-placeholder) GGUF v3 files for the phase-11 container-reader
and MiniMax-Music3-loader tests.

This module is the WRITER side; production code (``core.models.common.
gguf_container``) is read-only by design (see that module's docstring) --
nothing here is imported by any non-test module. The byte layout matches
``gguf_container.parse_gguf_header``'s reader exactly (same magic/version,
same metadata value-type ids, same dim-order convention: ``ne[]`` on disk is
``reversed(torch_tensor.shape)``, same default 32-byte alignment applied both
to the tensor-info-to-data-section boundary and between individual tensors),
so a round trip through these fixtures is a real test of the reader, not of
some more permissive writer.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import save_file

# GGUF metadata value type ids -- see gguf_container.py's own table (all 13:
# 11 scalar kinds + STRING + ARRAY).
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_STRUCT_BY_TYPE = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}

_DTYPE_TO_GGML_TYPE_ID = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 30}


@dataclass
class GGUFValue:
    """An explicitly-typed scalar/string metadata value -- for a type the
    plain-Python shorthand in ``_encode_metadata_value`` cannot express
    unambiguously (e.g. a Python ``int`` is one of 8 possible GGUF integer
    types)."""

    type_id: int
    value: Any


@dataclass
class GGUFArrayValue:
    """A metadata ARRAY value: ``elem_type_id`` plus its elements. An element
    may itself be a ``GGUFArrayValue`` (nested array), a raw scalar, or a
    ``str`` (for an array of strings)."""

    elem_type_id: int
    values: "List[Any]"


def _encode_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _encode_scalar(type_id: int, value: Any) -> bytes:
    if type_id == T_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if type_id == T_STRING:
        return _encode_string(value)
    if type_id == T_ARRAY:
        assert isinstance(value, GGUFArrayValue)
        return _encode_array(value)
    fmt = _SCALAR_STRUCT_BY_TYPE.get(type_id)
    if fmt is None:
        raise ValueError(f"unsupported GGUF scalar type id {type_id}")
    return struct.pack(fmt, value)


def _encode_array(array: "GGUFArrayValue") -> bytes:
    out = struct.pack("<I", array.elem_type_id) + struct.pack("<Q", len(array.values))
    for item in array.values:
        out += _encode_scalar(array.elem_type_id, item)
    return out


def _encode_metadata_value(value: Any) -> bytes:
    """``value`` -> ``type_id (uint32) + encoded value`` -- the full
    ``key -> value`` tail this fixture's ``write_gguf`` appends after each
    key string. Supports every one of GGUF's 13 metadata value types, either
    via the plain-Python shorthand (bool/str/int -> BOOL/STRING/UINT32, the
    common case every other fixture in this module uses) or explicitly via
    ``GGUFValue``/``GGUFArrayValue`` for a specific type or nesting."""
    if isinstance(value, GGUFValue):
        return struct.pack("<I", value.type_id) + _encode_scalar(value.type_id, value.value)
    if isinstance(value, GGUFArrayValue):
        return struct.pack("<I", T_ARRAY) + _encode_array(value)
    if isinstance(value, bool):
        return struct.pack("<I", T_BOOL) + struct.pack("<B", int(value))
    if isinstance(value, str):
        return struct.pack("<I", T_STRING) + _encode_string(value)
    if isinstance(value, int):
        return struct.pack("<I", T_UINT32) + struct.pack("<I", value)
    raise TypeError(f"unsupported fixture metadata value type: {type(value)}")


def _tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        # Bit-reinterpret as int16 -- numpy has no native bf16 dtype.
        return tensor.view(torch.int16).numpy().tobytes()
    return tensor.numpy().tobytes()


def write_gguf(
    path: str,
    tensors: "Dict[str, torch.Tensor]",
    metadata: "Optional[Dict[str, Any]]" = None,
    *,
    alignment: int = 32,
    extra_raw_tensors: "Optional[Dict[str, bytes]]" = None,
    extra_raw_ggml_type_id: int = 8,  # Q8_0, for the refusal-path fixtures.
) -> str:
    """Write a real GGUF v3 file at ``path``.

    ``tensors``: name -> a float32/float16/bfloat16 torch tensor, written with
    its GGML type inferred from ``tensor.dtype``.

    ``extra_raw_tensors``: name -> ALREADY-ENCODED raw bytes, declared as
    ``extra_raw_ggml_type_id`` (default Q8_0) regardless of their actual
    content -- for the "this reader must refuse a Q8_0 tensor" fixtures,
    where the byte CONTENT is irrelevant (the refusal is header-only and
    never reads it) but the file must still be big enough for
    ``parse_gguf_header``'s data-range validation to accept it. The declared
    shape is inferred as a single 1-D tensor of ``len(bytes) // 34 * 32``
    Q8_0-equivalent elements (34 bytes/block, 32 elements/block) -- exact
    numeric shape does not matter for these fixtures, only that it round-trips
    through the block-size arithmetic cleanly.
    """
    metadata = dict(metadata or {})

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    total_tensor_count = len(tensors) + len(extra_raw_tensors or {})
    header += struct.pack("<Q", total_tensor_count)
    header += struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        header += _encode_string(key)
        header += _encode_metadata_value(value)

    # Compute each tensor's payload bytes and its (alignment-padded) relative
    # offset within the data section, in the SAME order tensor-info records
    # are about to be written.
    infos = []  # (name, gguf_dims, ggml_type_id, rel_offset, payload_bytes)
    running = 0

    def _place(name: str, dims, ggml_type_id: int, payload: bytes) -> None:
        nonlocal running
        pad = (-running) % alignment
        running += pad
        rel_offset = running
        infos.append((name, dims, ggml_type_id, rel_offset, payload))
        running += len(payload)

    for name, tensor in tensors.items():
        ggml_type_id = _DTYPE_TO_GGML_TYPE_ID[tensor.dtype]
        dims = tuple(reversed(tensor.shape))  # ne[0] fastest -- see module docstring.
        _place(name, dims, ggml_type_id, _tensor_payload_bytes(tensor))

    for name, raw in (extra_raw_tensors or {}).items():
        block_bytes = 34 if extra_raw_ggml_type_id == 8 else None
        if block_bytes is None:
            raise ValueError("this fixture writer only knows Q8_0's block layout")
        if len(raw) % block_bytes != 0:
            raise ValueError(f"{name}: {len(raw)} byte(s) is not a multiple of Q8_0's 34-byte block")
        n_elements = (len(raw) // block_bytes) * 32
        _place(name, (n_elements,), extra_raw_ggml_type_id, raw)

    for name, dims, ggml_type_id, rel_offset, _payload in infos:
        header += _encode_string(name)
        header += struct.pack("<I", len(dims))
        for d in dims:
            header += struct.pack("<Q", int(d))
        header += struct.pack("<I", ggml_type_id)
        header += struct.pack("<Q", rel_offset)

    pad_to_data = (-len(header)) % alignment
    header += b"\x00" * pad_to_data

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes(header))
        prev_end = 0
        for _name, _dims, _ggml_type_id, rel_offset, payload in infos:
            fh.write(b"\x00" * (rel_offset - prev_end))
            fh.write(payload)
            prev_end = rel_offset + len(payload)
    return path


# ---------------------------------------------------------------------------
# DiT fixture: the SAME tiny geometry as
# `minimax_music3_flat_dit_fixture.write_tiny_flat_dit_and_official_tree`,
# written as GGUF instead of safetensors. Two independently-real files (one
# safetensors, one GGUF) of the same tensors, so a bug specific to either
# format's read path would show up as a mismatch between the two tests, not
# be hidden by a shared fixture.
# ---------------------------------------------------------------------------

from tests.minimax_music3_flat_dit_fixture import (  # noqa: E402
    CONDITION_DIM,
    CONDITION_HIDDEN_DIM,
    FF_INNER_DIM,
    FOURIER_EMBEDDING_DIM,
    HEAD_DIM,
    IN_CHANNELS,
    INNER_DIM,
    NUM_CONDITION_LAYERS,
    NUM_HEADS,
    NUM_LAYERS,
    ROTARY_DIM,
)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def write_tiny_gguf_dit_and_official_tree(tmp_path) -> dict:
    """GGUF analog of ``write_tiny_flat_dit_and_official_tree``: same tiny
    transformer/condition-encoder geometry, real (non-placeholder) tensors,
    ``general.architecture = "minimax_music3"`` metadata. Values are
    DIFFERENT from official/'s own placeholders (same substitution-bug guard
    the safetensors fixture uses) and different from the safetensors DiT
    fixture's own values too (seeded independently), so a format mix-up would
    also be visible.

    Returns the same key shape as the safetensors fixture, plus ``dit_path``
    pointing at the ``.gguf`` file.
    """
    root = str(tmp_path)
    official = os.path.join(root, "official")
    generator = torch.Generator().manual_seed(4321)

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

    CONCAT_CHANNELS = 2 * IN_CHANNELS + CONDITION_DIM

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

    # The GGUF file -- different values from official/'s, deliberately (a
    # visible substitution-bug guard), using the flat repack's own naming
    # (fused qkv, .gamma/.beta, GEGLU ff naming, folded-in condition encoder,
    # the rotary inv_freq with no destination) -- IDENTICAL tensor names to
    # the safetensors DiT fixture, per design doc "GGUF weights". Mixed
    # F32/F16/BF16 dtypes across keys, matching the real staged file's own
    # "F32 + F16 on disk" mix (not a single uniform dtype), so the reader's
    # per-tensor dtype resolution is exercised, not just one code path.
    flat_proj_in = (torch.randn(INNER_DIM, CONCAT_CHANNELS, generator=generator) + 100.0).to(torch.float32)
    fused_qkv = torch.randn(3 * INNER_DIM, INNER_DIM, generator=generator).to(torch.float16)
    tensors = {
        "diffusion_transformer.transformer.project_in.weight": flat_proj_in,
        "diffusion_transformer.transformer.project_out.weight": torch.randn(IN_CHANNELS, INNER_DIM, generator=generator).to(torch.bfloat16),
        "diffusion_transformer.preprocess_conv.weight": torch.randn(CONCAT_CHANNELS, CONCAT_CHANNELS, 1, generator=generator).to(torch.float32),
        "diffusion_transformer.postprocess_conv.weight": torch.randn(IN_CHANNELS, IN_CHANNELS, 1, generator=generator).to(torch.float32),
        "diffusion_transformer.timestep_features.weight": torch.randn(FOURIER_EMBEDDING_DIM // 2, 1, generator=generator).to(torch.float32),
        "diffusion_transformer.to_timestep_embed.0.weight": torch.randn(INNER_DIM, FOURIER_EMBEDDING_DIM, generator=generator).to(torch.float16),
        "diffusion_transformer.to_timestep_embed.0.bias": torch.randn(INNER_DIM, generator=generator).to(torch.float32),
        "diffusion_transformer.to_timestep_embed.2.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator).to(torch.float16),
        "diffusion_transformer.to_timestep_embed.2.bias": torch.randn(INNER_DIM, generator=generator).to(torch.float32),
        "diffusion_transformer.transformer.layers.0.pre_norm.gamma": torch.randn(INNER_DIM, generator=generator).to(torch.bfloat16),
        "diffusion_transformer.transformer.layers.0.pre_norm.beta": torch.randn(INNER_DIM, generator=generator).to(torch.bfloat16),
        "diffusion_transformer.transformer.layers.0.ff_norm.gamma": torch.randn(INNER_DIM, generator=generator).to(torch.bfloat16),
        "diffusion_transformer.transformer.layers.0.ff_norm.beta": torch.randn(INNER_DIM, generator=generator).to(torch.bfloat16),
        "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight": fused_qkv,
        "diffusion_transformer.transformer.layers.0.self_attn.to_out.weight": torch.randn(INNER_DIM, INNER_DIM, generator=generator).to(torch.float16),
        "diffusion_transformer.transformer.layers.0.ff.ff.0.proj.weight": torch.randn(FF_INNER_DIM * 2, INNER_DIM, generator=generator).to(torch.float16),
        "diffusion_transformer.transformer.layers.0.ff.ff.0.proj.bias": torch.randn(FF_INNER_DIM * 2, generator=generator).to(torch.float32),
        "diffusion_transformer.transformer.layers.0.ff.ff.2.weight": torch.randn(INNER_DIM, FF_INNER_DIM, generator=generator).to(torch.float16),
        "diffusion_transformer.transformer.layers.0.ff.ff.2.bias": torch.randn(INNER_DIM, generator=generator).to(torch.float32),
        "cond_layer_logits": torch.randn(NUM_CONDITION_LAYERS, generator=generator).to(torch.float32),
        "cond_layer_scale": torch.randn(1, generator=generator).to(torch.float32),
        "latent_conditioners.0.weight": torch.randn(CONDITION_DIM, CONDITION_HIDDEN_DIM, 3, generator=generator).to(torch.float32),
        "latent_conditioners.0.bias": torch.randn(CONDITION_DIM, generator=generator).to(torch.float32),
        # No destination in the vendored module (see flat_remap.py) --
        # present here to prove the GGUF path drops it the same way the
        # safetensors path does, not merely that the remap function does.
        "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": torch.randn(ROTARY_DIM // 2, generator=generator).to(torch.float32),
    }
    dit_path = os.path.join(root, "diffusion_models", "minimax_music3_dit_BF16.gguf")
    write_gguf(dit_path, tensors, {"general.architecture": "minimax_music3", "general.file_type": 1})

    return {
        "official": official,
        "dit_path": dit_path,
        "expected_proj_in_weight": flat_proj_in,
        "official_placeholder_proj_in_weight": official_proj_in,
        "expected_fused_qkv": fused_qkv,
    }


def write_gguf_dit_with_q8_0_tensor(tmp_path) -> str:
    """A GGUF DiT-shaped file (passes the tensor-name + architecture
    detection) that ALSO carries one Q8_0 tensor -- the header-only refusal
    fixture. The Q8_0 tensor's bytes are zero-filled: irrelevant, since the
    refusal never reads them."""
    path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_q8_0.gguf")
    tensors = {
        "diffusion_transformer.transformer.project_in.weight": torch.randn(2, 2),
        "cond_layer_logits": torch.randn(2),
        "latent_conditioners.0.weight": torch.randn(2, 2),
    }
    # 34 bytes = one Q8_0 block (32 int8 codes + 1 fp16 scale); content unused.
    extra_raw_tensors = {"diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight": b"\x00" * 34}
    write_gguf(
        path, tensors, {"general.architecture": "minimax_music3"},
        extra_raw_tensors=extra_raw_tensors, extra_raw_ggml_type_id=8,
    )
    return path


# ---------------------------------------------------------------------------
# Pruned text-encoder fixture: mirrors
# `minimax_music3_pruned_text_encoder_fixture.write_tiny_pruned_text_encoder_
# and_official_tree`'s geometry, written as GGUF (all F32 -- no Q8_0), so the
# pruned GGUF builder's remap path (not just its refusal path) is exercised.
# ---------------------------------------------------------------------------

from tests.minimax_music3_pruned_text_encoder_fixture import (  # noqa: E402
    AUDIO_VOCAB_SIZE,
    DEPTH_HIDDEN_SIZE,
    DEPTH_INTERMEDIATE_SIZE,
    DEPTH_MAX_POSITION_EMBEDDINGS,
    DEPTH_NUM_LAYERS,
    HEAD_DIM as TE_HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    KV_DIM,
    MAX_POSITION_EMBEDDINGS,
    NUM_ATTENTION_HEADS,
    NUM_CODEBOOKS,
    NUM_HIDDEN_LAYERS,
    NUM_KEY_VALUE_HEADS,
    Q_DIM,
    ROPE_THETA,
)
from core.models.minimax_music3.defaults import AUDIO_CODE_OFFSET  # noqa: E402
from core.models.minimax_music3.pruned_text_encoder_remap import (  # noqa: E402
    AUDIO_HEAD_VOCAB_SIZE,
    SEMANTIC_VOCAB_SIZE,
)


def write_tiny_pruned_gguf_text_encoder_and_official_tree(tmp_path) -> dict:
    """GGUF analog of ``write_tiny_pruned_text_encoder_and_official_tree``:
    same tiny (GQA-uneven) geometry and real row counts for the three vocab
    tables, written as an all-F32 GGUF file -- no Q8_0, so
    ``build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder``
    proceeds PAST its header-only refusal gate and exercises the real remap
    (``pruned_text_encoder_remap.apply_pruned_text_encoder_state_dict``)
    against a lazy ``GGUFStateDict`` source instead of an eager safetensors
    one. ``tokenizer_json`` is omitted (dropped by the remap either way; see
    ``flat_remap._TEXT_ENCODER_DROP``, and this reader does not materialize
    a uint8 byte-blob tensor type regardless)."""
    root = str(tmp_path)
    official = os.path.join(root, "official")
    generator = torch.Generator().manual_seed(6789)

    lm_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 200_000,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "head_dim": TE_HEAD_DIM,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "rope_parameters": {"rope_theta": ROPE_THETA, "rope_type": "default"},
    }
    _write_json(os.path.join(official, "language_model", "config.json"), lm_config)

    depth_config = {
        "_class_name": "MiniMaxMusic3RVQDepthDecoder",
        "hidden_size": DEPTH_HIDDEN_SIZE,
        "num_layers": DEPTH_NUM_LAYERS,
        "num_attention_heads": 2,
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

    tensors = {
        "model.embed_tokens_prefill.weight": torch.randn(AUDIO_CODE_OFFSET, HIDDEN_SIZE, generator=generator),
        "model.embed_tokens_audio.weight": embed_tokens_audio,
        "model.lm_head_pruned.weight": lm_head_pruned,
        "model.norm.weight": torch.randn(HIDDEN_SIZE, generator=generator),
        "model.audio_extra_embedding.weight": audio_extra_embedding,
        "model.audio_decoder.norm.weight": torch.randn(DEPTH_HIDDEN_SIZE, generator=generator),
        "model.audio_decoder.pos_embedding.weight": torch.randn(DEPTH_MAX_POSITION_EMBEDDINGS, DEPTH_HIDDEN_SIZE, generator=generator),
        "model.audio_decoder.projection.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
    }
    for i in range(NUM_CODEBOOKS - 1):
        tensors[f"model.audio_decoder.audio_heads.{i}.weight"] = torch.randn(
            AUDIO_VOCAB_SIZE, DEPTH_HIDDEN_SIZE, generator=generator,
        )
    for layer in range(NUM_HIDDEN_LAYERS):
        prefix = f"model.layers.{layer}."
        tensors.update({
            prefix + "input_layernorm.weight": torch.randn(HIDDEN_SIZE, generator=generator),
            prefix + "post_attention_layernorm.weight": torch.randn(HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.qkv_proj.weight": fused_qkv,
            prefix + "self_attn.o_proj.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.q_norm.weight": torch.randn(TE_HEAD_DIM, generator=generator),
            prefix + "self_attn.k_norm.weight": torch.randn(TE_HEAD_DIM, generator=generator),
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
        tensors.update({
            prefix + "input_layernorm.weight": torch.randn(DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "post_attention_layernorm.weight": torch.randn(DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "self_attn.qkv_proj.weight": depth_qkv,
            prefix + "self_attn.o_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_HIDDEN_SIZE, generator=generator),
            prefix + "mlp.gate_up_proj.weight": depth_gate_up,
            prefix + "mlp.down_proj.weight": torch.randn(DEPTH_HIDDEN_SIZE, DEPTH_INTERMEDIATE_SIZE, generator=generator),
        })

    text_encoder_path = os.path.join(root, "text_encoders", "minimax_music3_text_encoder_pruned_f32.gguf")
    write_gguf(text_encoder_path, tensors, {"general.architecture": "minimax_music3"})

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


def write_pruned_gguf_text_encoder_with_q8_0_tensor(tmp_path) -> str:
    """Header-only refusal fixture: a pruned-vocabulary-tell-bearing GGUF
    text encoder (so it passes the pruned-layout check) that also carries one
    Q8_0 tensor."""
    path = os.path.join(str(tmp_path), "text_encoders", "minimax_music3_text_encoder_pruned_q8_0.gguf")
    tensors = {
        "model.embed_tokens_prefill.weight": torch.randn(4, 4),
        "model.embed_tokens_audio.weight": torch.randn(4, 4),
        "model.lm_head_pruned.weight": torch.randn(4, 4),
    }
    extra_raw_tensors = {"model.layers.0.self_attn.qkv_proj.weight": b"\x00" * 34}
    write_gguf(
        path, tensors, {"general.architecture": "minimax_music3"},
        extra_raw_tensors=extra_raw_tensors, extra_raw_ggml_type_id=8,
    )
    return path
