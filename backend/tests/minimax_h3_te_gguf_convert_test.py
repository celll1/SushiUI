"""MiniMax-H3: Qwen3-VL GGUF -> truncated bf16 text encoder.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_gguf_convert_test.py -v

Covers the converter's contract on a SYNTHETIC 2-block GGUF (32-wide, a few
hundred KB) rather than the real 4 GB files: the GGUF->HF name map, the
truncation, the dropped `output_norm`/`output`, the `minimax_h3_te` metadata
the loader and P2 read, that an unmapped tensor fails loudly, and that
row-chunked dequantization is bit-identical to one-shot `gguf.dequantize`.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from model_root import model_path  # noqa: E402

from core.models.minimax_h3.loader import MINIMAX_H3_TE_PATTERNS  # noqa: E402
from core.models.minimax_h3.te_gguf_convert import (  # noqa: E402
    ConversionError,
    convert,
    map_name,
    sha256_file,
)

# Small but Q8_0-legal: every quantized row length is a multiple of 32.
HIDDEN = 32
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 16
INTER = 64
VOCAB = 64
BLOCKS = 2
SECTIONS = [4, 2, 2, 0]  # llama.cpp pads the 4th slot; mrope_section is [4, 2, 2]
EPS = 1e-6
THETA = 5_000_000.0


def _write_gguf(path, extra_tensors=None):
    from gguf import GGUFWriter, quants
    from gguf.constants import GGMLQuantizationType

    rng = np.random.default_rng(0)
    writer = GGUFWriter(str(path), "qwen3vl")
    writer.add_name("Synthetic Qwen3 Vl Tiny")
    writer.add_basename("qwen3-vl")
    writer.add_finetune("synthetic")
    writer.add_size_label("Tiny")
    writer.add_block_count(BLOCKS)
    writer.add_embedding_length(HIDDEN)
    writer.add_feed_forward_length(INTER)
    writer.add_head_count(HEADS)
    writer.add_head_count_kv(KV_HEADS)
    writer.add_key_length(HEAD_DIM)
    writer.add_value_length(HEAD_DIM)
    writer.add_layer_norm_rms_eps(EPS)
    writer.add_rope_freq_base(THETA)
    writer.add_rope_dimension_sections(SECTIONS)

    def quantized(name, shape):
        data = rng.standard_normal(shape).astype(np.float32)
        # uint8 payload: the writer derives the logical shape from the byte shape.
        writer.add_tensor(name, quants.quantize(data, GGMLQuantizationType.Q8_0),
                          raw_dtype=GGMLQuantizationType.Q8_0)

    def f32(name, shape):
        writer.add_tensor(name, rng.standard_normal(shape).astype(np.float32))

    quantized("token_embd.weight", (VOCAB, HIDDEN))
    for layer in range(BLOCKS):
        f32(f"blk.{layer}.attn_norm.weight", (HIDDEN,))
        f32(f"blk.{layer}.ffn_norm.weight", (HIDDEN,))
        f32(f"blk.{layer}.attn_q_norm.weight", (HEAD_DIM,))
        f32(f"blk.{layer}.attn_k_norm.weight", (HEAD_DIM,))
        quantized(f"blk.{layer}.attn_q.weight", (HEADS * HEAD_DIM, HIDDEN))
        quantized(f"blk.{layer}.attn_k.weight", (KV_HEADS * HEAD_DIM, HIDDEN))
        quantized(f"blk.{layer}.attn_v.weight", (KV_HEADS * HEAD_DIM, HIDDEN))
        quantized(f"blk.{layer}.attn_output.weight", (HIDDEN, HEADS * HEAD_DIM))
        quantized(f"blk.{layer}.ffn_gate.weight", (INTER, HIDDEN))
        quantized(f"blk.{layer}.ffn_up.weight", (INTER, HIDDEN))
        quantized(f"blk.{layer}.ffn_down.weight", (HIDDEN, INTER))
    f32("output_norm.weight", (HIDDEN,))
    quantized("output.weight", (VOCAB, HIDDEN))
    for name, shape in (extra_tensors or {}).items():
        f32(name, shape)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.fixture(scope="module")
def synthetic_gguf(tmp_path_factory):
    return _write_gguf(tmp_path_factory.mktemp("gguf") / "tiny.gguf")


@pytest.fixture(scope="module")
def converted(synthetic_gguf, tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "tiny_tap1_bf16.safetensors"
    summary = convert(synthetic_gguf, out, tap=1, quiet=True)
    return out, summary


def _header(path):
    import struct

    with open(path, "rb") as fh:
        length = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(length))


# ---------------------------------------------------------------------------
# Name map and truncation
# ---------------------------------------------------------------------------

def test_output_keys_are_exactly_the_kept_blocks_in_flat_hf_naming(converted):
    path, summary = converted
    keys = set(_header(path)) - {"__metadata__"}
    expected = {"model.embed_tokens.weight"} | {
        f"model.layers.0.{suffix}"
        for suffix in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        )
    }
    assert keys == expected
    assert summary["tensors_written"] == len(expected)
    # blk.1's 11 tensors + output_norm + output
    assert summary["dropped_dead"] == 13


def test_no_lm_head_and_no_final_norm(converted):
    """The two keys `loader._TE_EXPECTED_MISSING` allows to be absent."""
    keys = set(_header(converted[0]))
    assert "lm_head.weight" not in keys
    assert "model.norm.weight" not in keys
    assert "output_norm.weight" not in keys


def test_shapes_follow_the_declared_dims_untransposed(converted):
    header = _header(converted[0])
    shapes = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    assert shapes["model.embed_tokens.weight"] == (VOCAB, HIDDEN)
    assert shapes["model.layers.0.self_attn.q_proj.weight"] == (HEADS * HEAD_DIM, HIDDEN)
    assert shapes["model.layers.0.self_attn.k_proj.weight"] == (KV_HEADS * HEAD_DIM, HIDDEN)
    assert shapes["model.layers.0.self_attn.o_proj.weight"] == (HIDDEN, HEADS * HEAD_DIM)
    assert shapes["model.layers.0.mlp.down_proj.weight"] == (HIDDEN, INTER)
    assert shapes["model.layers.0.self_attn.q_norm.weight"] == (HEAD_DIM,)
    assert {v["dtype"] for k, v in header.items() if k != "__metadata__"} == {"BF16"}


def test_flat_names_survive_the_loader_rewrite(converted):
    from core.models.minimax_h3.loader import _rewrite_te_key

    keys = set(_header(converted[0])) - {"__metadata__"}
    rewritten = {_rewrite_te_key(k) for k in keys}
    assert "model.language_model.embed_tokens.weight" in rewritten
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in rewritten
    assert all(k.startswith("model.language_model.") for k in rewritten)


# ---------------------------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------------------------

def test_metadata_carries_the_full_text_config_and_provenance(converted, synthetic_gguf):
    path, _ = converted
    declared = json.loads(_header(path)["__metadata__"]["minimax_h3_te"])
    assert declared["num_hidden_layers"] == 1
    assert declared["hidden_size"] == HIDDEN
    assert declared["num_attention_heads"] == HEADS
    assert declared["num_key_value_heads"] == KV_HEADS
    assert declared["head_dim"] == HEAD_DIM
    assert declared["intermediate_size"] == INTER
    assert declared["vocab_size"] == VOCAB
    assert declared["rms_norm_eps"] == pytest.approx(EPS)
    assert declared["rope_theta"] == pytest.approx(THETA)
    assert declared["mrope_section"] == SECTIONS[:-1]  # trailing llama.cpp pad stripped
    assert declared["output"] == "unnormalized_hidden_after_layer_1"
    assert declared["modalities"] == "text"
    assert declared["tap"] == 1
    assert declared["source_gguf"] == os.path.basename(str(synthetic_gguf))
    assert declared["source_gguf_sha256"] == sha256_file(synthetic_gguf)
    assert declared["source_arch"] == "qwen3vl"
    assert declared["source_finetune"] == "synthetic"
    assert declared["source_block_count"] == BLOCKS
    assert declared["converter"] == "minimax_h3_te_gguf_convert"
    assert declared["converter_version"]


def test_metadata_matches_the_shipped_files_own_key_and_wording(converted):
    """The loader reads `minimax_h3_te` -> `num_hidden_layers`; the shipped 32B
    file declares `unnormalized_hidden_after_layer_50` under the same key."""
    metadata = _header(converted[0])["__metadata__"]
    assert set(metadata) == {"minimax_h3_te"}
    assert isinstance(metadata["minimax_h3_te"], str)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def test_row_chunked_dequant_is_bit_identical_to_one_shot(converted):
    assert converted[1]["chunked_dequant_max_abs_diff"] == 0.0


def test_fp32_output_keeps_the_same_key_contract(synthetic_gguf, tmp_path):
    out = tmp_path / "tiny_tap1_fp32.safetensors"
    summary = convert(synthetic_gguf, out, tap=1, dtype="fp32", quiet=True)
    header = _header(out)
    assert {v["dtype"] for k, v in header.items() if k != "__metadata__"} == {"F32"}
    assert summary["tensors_written"] == 12


def test_values_round_trip_through_q8_0_within_bf16(converted, synthetic_gguf):
    """The written weights are the GGUF's own dequantized values, bf16-rounded."""
    from gguf import GGUFReader, dequantize
    from safetensors.torch import load_file

    reader = GGUFReader(str(synthetic_gguf))
    source = next(t for t in reader.tensors if t.name == "blk.0.attn_q.weight")
    reference = torch.from_numpy(dequantize(source.data, source.tensor_type).copy())
    written = load_file(str(converted[0]))["model.layers.0.self_attn.q_proj.weight"]
    assert written.dtype == torch.bfloat16
    assert torch.equal(written, reference.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------

def test_unmapped_tensor_fails_loudly_naming_it(tmp_path):
    path = _write_gguf(tmp_path / "extra.gguf", extra_tensors={"v.patch_embed.weight": (HIDDEN,)})
    with pytest.raises(ConversionError) as excinfo:
        convert(path, tmp_path / "out.safetensors", tap=1, quiet=True)
    assert "v.patch_embed.weight" in str(excinfo.value)
    assert not (tmp_path / "out.safetensors").exists()


def test_map_name_raises_for_an_unknown_block_suffix():
    with pytest.raises(ConversionError):
        map_name("blk.0.attn_rot_embd.weight", tap=24)


def test_tap_beyond_the_gguf_depth_is_refused(synthetic_gguf, tmp_path):
    with pytest.raises(ConversionError):
        convert(synthetic_gguf, tmp_path / "deep.safetensors", tap=BLOCKS + 1, quiet=True)


def test_shipped_32b_filenames_are_refused(synthetic_gguf, tmp_path):
    """A degraded encoder must never land on a name `_find_first` auto-selects."""
    for name in MINIMAX_H3_TE_PATTERNS:
        with pytest.raises(ConversionError):
            convert(synthetic_gguf, tmp_path / name, tap=1, quiet=True)


# ---------------------------------------------------------------------------
# Real tree (header-only, skipped when absent)
# ---------------------------------------------------------------------------

def test_real_converted_encoders_are_not_auto_selectable():
    root = model_path("minimax_h3")
    if not os.path.isdir(root):
        pytest.skip(f"{root} not present on this machine")
    directory = os.path.join(root, "text_encoders")
    converted = [
        name for name in os.listdir(directory)
        if name.endswith(".safetensors") and "tap" in name
    ]
    if not converted:
        pytest.skip("no converted GGUF-derived text encoder on this machine")
    for name in converted:
        assert name not in MINIMAX_H3_TE_PATTERNS
        declared = json.loads(
            _header(os.path.join(directory, name))["__metadata__"]["minimax_h3_te"])
        assert declared["modalities"] == "text"
        assert declared["num_hidden_layers"] == declared["tap"]
        assert declared["output"] == f"unnormalized_hidden_after_layer_{declared['tap']}"
