"""MiniMax-H3: loading a Qwen3-VL Q8_0 GGUF as the text encoder, unconverted.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_gguf_native_test.py -v

``te_gguf_native`` maps the GGUF itself instead of ``te_gguf_convert``'s bf16
re-export. The things that can load clean and be wrong here are:

* the Q8_0 block layout -- 34 bytes per 32 values, so a row's packed width is
  not its value count;
* the depth -- an unconverted file carries EVERY block, so the trained
  projection's ``tap`` decides how many are mapped, and blocks at or beyond it
  must never be touched;
* the type coverage -- anything but Q8_0/F32 has no dequantizer here and must be
  refused by name rather than half-loaded;
* selection -- a small stand-in encoder must stay unreachable except through an
  explicit override.

Everything is built at toy dims (hidden 64, 4 blocks, vocab 32), so the real
writer, the real reader and the real ``Qwen3VLForConditionalGeneration`` build
all run on a few hundred kilobytes.
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from gguf import GGMLQuantizationType, GGUFWriter  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.models.minimax_h3 import loader  # noqa: E402
from core.models.minimax_h3.te_gguf_native import (  # noqa: E402
    GgufQ8Embedding, GgufQ8Linear, GgufTextEncoderError, dequantize_q8_0, open_gguf,
    plan_gguf_text_encoder, q8_0_row_bytes,
)

HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
FFN = 128
VOCAB = 32
BLOCKS = 4
TAP = 2
TEXT_DIM = 128
GGUF_NAME = "qwen3-vl-4b-heretic-Q8_0.gguf"


# ---------------------------------------------------------------------------
# Synthetic files
# ---------------------------------------------------------------------------

def _pack_q8_0(values: np.ndarray) -> np.ndarray:
    """``[rows, cols]`` float -> ``[rows, blocks*34]`` uint8, llama.cpp's layout."""
    rows, cols = values.shape
    blocks = cols // 32
    x = values.reshape(rows, blocks, 32).astype(np.float32)
    scale = np.abs(x).max(axis=-1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
    codes = np.rint(x / scale.astype(np.float32)).clip(-127, 127).astype(np.int8)
    packed = np.empty((rows, blocks, 34), dtype=np.uint8)
    packed[:, :, :2] = scale.view(np.uint8).reshape(rows, blocks, 2)
    packed[:, :, 2:] = codes.view(np.uint8)
    return packed.reshape(rows, blocks * 34)


def _weights(seed: int = 0):
    """Every tensor a ``BLOCKS``-block toy Qwen3-VL GGUF carries, as float32."""
    rng = np.random.default_rng(seed)
    q, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM

    def dense(shape):
        return rng.standard_normal(shape).astype(np.float32)

    tensors = {"token_embd.weight": dense((VOCAB, HIDDEN))}
    for block in range(BLOCKS):
        for suffix, shape in {
            "attn_norm.weight": (HIDDEN,),
            "ffn_norm.weight": (HIDDEN,),
            "attn_q.weight": (q, HIDDEN),
            "attn_k.weight": (kv, HIDDEN),
            "attn_v.weight": (kv, HIDDEN),
            "attn_output.weight": (HIDDEN, q),
            "attn_q_norm.weight": (HEAD_DIM,),
            "attn_k_norm.weight": (HEAD_DIM,),
            "ffn_gate.weight": (FFN, HIDDEN),
            "ffn_up.weight": (FFN, HIDDEN),
            "ffn_down.weight": (HIDDEN, FFN),
        }.items():
            tensors[f"blk.{block}.{suffix}"] = dense(shape)
    tensors["output_norm.weight"] = dense((HIDDEN,))
    return tensors


def _write_gguf(path, *, tensors=None, unsupported=(), blocks=BLOCKS):
    """A qwen3vl GGUF: 1-D tensors F32, 2-D Q8_0, ``unsupported`` names as Q4_K."""
    tensors = _weights() if tensors is None else tensors
    writer = GGUFWriter(str(path), "qwen3vl")
    writer.add_uint32("qwen3vl.embedding_length", HIDDEN)
    writer.add_uint32("qwen3vl.block_count", blocks)
    writer.add_uint32("qwen3vl.attention.head_count", HEADS)
    writer.add_uint32("qwen3vl.attention.head_count_kv", KV_HEADS)
    writer.add_uint32("qwen3vl.attention.key_length", HEAD_DIM)
    writer.add_uint32("qwen3vl.attention.value_length", HEAD_DIM)
    writer.add_uint32("qwen3vl.feed_forward_length", FFN)
    writer.add_float32("qwen3vl.attention.layer_norm_rms_epsilon", 1e-06)
    writer.add_float32("qwen3vl.rope.freq_base", 5000000.0)
    writer.add_array("qwen3vl.rope.dimension_sections", [4, 2, 2, 0])
    writer.add_string("general.size_label", "4B")
    for name, value in tensors.items():
        if name in unsupported:
            # A payload no dequantizer here understands; the type tag is the
            # point. 144 bytes is one Q4_K block, which the writer requires the
            # byte row to be a multiple of.
            writer.add_tensor(name, np.zeros((value.shape[0], 144), dtype=np.uint8),
                              raw_shape=(value.shape[0], 144),
                              raw_dtype=GGMLQuantizationType.Q4_K)
        elif value.ndim == 1:
            writer.add_tensor(name, value)
        else:
            packed = _pack_q8_0(value)
            writer.add_tensor(name, packed, raw_shape=packed.shape,
                              raw_dtype=GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


def _header_only(path, keys, metadata=None):
    header = dict(keys)
    header["__metadata__"] = metadata or {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
    return str(path)


def _projection(path, *, d_in=HIDDEN, d_out=TEXT_DIM, tap=TAP, hidden=4):
    generator = torch.Generator().manual_seed(5)
    tensors = {
        "W": torch.randn(d_in, d_out, generator=generator),
        "mean_in": torch.randn(d_in, generator=generator),
        "std_in": torch.rand(d_in, generator=generator) + 0.5,
        "mean_out": torch.randn(d_out, generator=generator),
        "std_out": torch.rand(d_out, generator=generator) + 0.5,
        "sink_out": torch.randn(d_out, generator=generator),
        "mlp.0.weight": torch.randn(hidden, d_in, generator=generator),
        "mlp.0.bias": torch.randn(hidden, generator=generator),
        "mlp.2.weight": torch.randn(d_out, hidden, generator=generator),
        "mlp.2.bias": torch.randn(d_out, generator=generator),
    }
    save_file(tensors, str(path), metadata={
        "d_in": str(d_in), "d_out": str(d_out), "tap": str(tap),
        "mlp_hidden": str(hidden), "mlp_depth": "1"})
    return str(path)


def _official_tree(tmp_path):
    official = tmp_path / "official"
    (official / "text_encoder").mkdir(parents=True)
    config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "image_token_id": 5, "video_token_id": 6,
        "vision_start_token_id": 7, "vision_end_token_id": 8,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "attention_bias": False, "hidden_act": "silu",
            # Deliberately not the GGUF's dims, so "which config was used" shows.
            "hidden_size": 128, "intermediate_size": 256,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
            "num_hidden_layers": 2, "vocab_size": 32, "rms_norm_eps": 1e-06,
            "rope_theta": 5000000, "max_position_embeddings": 4096,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [4, 2, 2],
                             "rope_type": "default"},
        },
        "vision_config": {
            "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
            "intermediate_size": 32, "num_heads": 2, "in_channels": 3,
            "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
            "num_position_embeddings": 64, "out_hidden_size": 128,
            "deepstack_visual_indexes": [0], "hidden_act": "gelu_pytorch_tanh",
        },
    }
    (official / "text_encoder" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(official)


def _tree(tmp_path, *, te_files=(GGUF_NAME,)):
    """A MiniMax-H3 tree whose text_encoders/ holds only what is asked for."""
    root = tmp_path / "tree"
    (root / "diffusion_models").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "text_encoders").mkdir()
    (root / "clip_projections").mkdir()
    _header_only(root / "diffusion_models" / "minimax_h3_fl2va_pruned_bf16.safetensors", {
        "adaln_t_table": {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]},
        "condition_proj.weight": {"dtype": "BF16", "shape": [256, TEXT_DIM],
                                  "data_offsets": [0, 0]},
        "token_refiner.0.weight": {"dtype": "BF16", "shape": [1, 1], "data_offsets": [0, 0]},
    })
    for name in ("minimax_h3_video_vae_fp16.safetensors", "minimax_h3_audio_vae_fp32.safetensors"):
        _header_only(root / "vae" / name,
                     {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]}})
    for name in te_files:
        if name.endswith(".gguf"):
            _write_gguf(root / "text_encoders" / name)
        else:
            _header_only(root / "text_encoders" / name, {
                "model.layers.0.self_attn.q_proj.weight": {
                    "dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 0]}})
    _projection(root / "clip_projections" / "mmh3-4b-clipproj-celeb-mlp.safetensors")
    return root


def _drop(model):
    import gc

    del model
    gc.collect()


# ---------------------------------------------------------------------------
# Q8_0
# ---------------------------------------------------------------------------

def test_q8_0_unpack_matches_a_hand_built_block():
    """One block, built byte by byte: ``x[i] = d * qs[i]``."""
    codes = np.arange(-16, 16, dtype=np.int8)
    scale = np.float16(0.25)
    block = np.concatenate([
        np.frombuffer(scale.tobytes(), dtype=np.uint8),
        codes.view(np.uint8),
    ])
    assert block.size == 34

    got = dequantize_q8_0(torch.from_numpy(block.copy()).reshape(1, 34), 32)

    assert got.shape == (1, 32)
    assert torch.equal(got[0], torch.from_numpy(codes.astype(np.float32)) * 0.25)


def test_a_rows_packed_width_is_not_its_value_count():
    """The real 4B width: 2560 values occupy 2720 bytes."""
    assert q8_0_row_bytes(2560) == 2720
    assert q8_0_row_bytes(9728) == 10336

    values = np.random.default_rng(1).standard_normal((3, 2560)).astype(np.float32)
    packed = torch.from_numpy(_pack_q8_0(values))
    assert packed.shape == (3, 2720)

    got = dequantize_q8_0(packed, 2560)
    assert got.shape == (3, 2560)
    # Q8_0 is 127 levels per block, so agreement is a quantization step, not zero.
    assert float((got - torch.from_numpy(values)).abs().max()) < 0.02


def test_a_packed_width_that_is_not_a_whole_number_of_blocks_is_refused():
    with pytest.raises(GgufTextEncoderError, match=r"2719 packed byte\(s\).*2560"):
        dequantize_q8_0(torch.zeros(2, 2719, dtype=torch.uint8), 2560)


def test_a_packed_width_from_the_wrong_row_is_refused():
    """2720 bytes hold 2560 values, not 2688; the count is checked, not inferred."""
    with pytest.raises(GgufTextEncoderError, match=r"2720 packed byte\(s\) per row is not 2688"):
        dequantize_q8_0(torch.zeros(2, 2720, dtype=torch.uint8), 2688)


# ---------------------------------------------------------------------------
# Type coverage
# ---------------------------------------------------------------------------

def test_an_unsupported_ggml_type_is_refused_by_name(tmp_path):
    path = _write_gguf(tmp_path / "q4.gguf",
                       unsupported=("blk.0.attn_q.weight", "blk.1.ffn_down.weight"))
    reader = open_gguf(path)

    with pytest.raises(GgufTextEncoderError) as excinfo:
        plan_gguf_text_encoder(reader, TAP, path, loader._rewrite_te_key)

    message = str(excinfo.value)
    assert "Q4_K" in message
    assert "blk.0.attn_q.weight" in message
    assert "te_gguf_convert" in message


def test_an_unsupported_type_beyond_the_tap_does_not_refuse_the_file(tmp_path):
    """Blocks the tap excludes are never mapped, so their type never matters."""
    path = _write_gguf(tmp_path / "q4_deep.gguf", unsupported=("blk.3.ffn_down.weight",))
    reader = open_gguf(path)

    plan = plan_gguf_text_encoder(reader, TAP, path, loader._rewrite_te_key)

    assert plan["linear_configs"]


def test_the_listing_reports_an_unsupported_type_instead_of_offering_the_file(tmp_path):
    path = _write_gguf(tmp_path / "q4.gguf", unsupported=("blk.0.attn_q.weight",))

    entry = loader.inspect_minimax_h3_text_encoder_candidate(path)

    assert entry["compatible"] is False
    assert entry["variant"] == "gguf_unsupported"
    assert "Q4_K" in entry["reason"] and "te_gguf_convert" in entry["reason"]


# ---------------------------------------------------------------------------
# The tap rule
# ---------------------------------------------------------------------------

def test_only_blocks_below_the_tap_are_mapped(tmp_path):
    path = _write_gguf(tmp_path / GGUF_NAME)
    reader = open_gguf(path)

    plan = plan_gguf_text_encoder(reader, TAP, path, loader._rewrite_te_key)

    mapped = sorted(plan["state_dict"])
    layers = {int(key.split(".")[3]) for key in mapped if ".layers." in key}
    assert layers == set(range(TAP))
    assert not [key for key in mapped if "language_model.norm" in key or "lm_head" in key]
    # 11 tensors per mapped block + the embedding; the rest is deliberately dead.
    assert len(mapped) == TAP * 11 + 1
    assert plan["skipped"] == (BLOCKS - TAP) * 11 + 1


def test_a_tap_beyond_the_files_blocks_names_both_numbers(tmp_path):
    path = _write_gguf(tmp_path / GGUF_NAME)
    reader = open_gguf(path)

    with pytest.raises(GgufTextEncoderError) as excinfo:
        plan_gguf_text_encoder(reader, BLOCKS + 1, path, loader._rewrite_te_key)

    message = str(excinfo.value)
    assert f"tap={BLOCKS + 1}" in message and f"{BLOCKS} block(s)" in message


def test_the_projection_gate_bounds_the_tap_instead_of_matching_it(tmp_path):
    """A GGUF is not truncated, so a projection only has to FIT its block count."""
    from core.models.minimax_h3.te_projection import resolve_te_projection

    path = _write_gguf(tmp_path / GGUF_NAME)
    fitting = _projection(tmp_path / "fits.safetensors", tap=TAP)
    too_deep = _projection(tmp_path / "too_deep.safetensors", tap=BLOCKS + 1)

    spec = resolve_te_projection(
        root=None, te_path=path, hidden_size=HIDDEN, num_hidden_layers=0,
        text_dim=TEXT_DIM, override=fitting, available_blocks=BLOCKS)
    assert spec["tap"] == TAP

    with pytest.raises(ValueError) as excinfo:
        resolve_te_projection(
            root=None, te_path=path, hidden_size=HIDDEN, num_hidden_layers=0,
            text_dim=TEXT_DIM, override=too_deep, available_blocks=BLOCKS)
    assert f"tap={BLOCKS + 1}" in str(excinfo.value)
    assert f"{BLOCKS} block(s)" in str(excinfo.value)


def test_a_converted_file_keeps_the_equality_gate(tmp_path):
    """The bound is only for unconverted files; a declared depth must still match."""
    from core.models.minimax_h3.te_projection import resolve_te_projection

    projection = _projection(tmp_path / "tap2.safetensors", tap=TAP)

    with pytest.raises(ValueError, match=r"num_hidden_layers=3"):
        resolve_te_projection(
            root=None, te_path="converted.safetensors", hidden_size=HIDDEN,
            num_hidden_layers=3, text_dim=TEXT_DIM, override=projection)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_the_glob_fallback_never_selects_a_gguf(tmp_path):
    root = _tree(tmp_path)

    layout = loader.detect_minimax_h3_layout(str(root))

    assert layout is not None
    assert layout["text_encoder"] is None
    reason = layout["text_encoder_reason"]
    assert GGUF_NAME in reason and "explicit override" in reason


def test_a_released_file_still_wins_over_a_gguf(tmp_path):
    released = loader.MINIMAX_H3_TE_PATTERNS[1]
    root = _tree(tmp_path, te_files=(GGUF_NAME, released))

    layout = loader.detect_minimax_h3_layout(str(root))

    assert layout["text_encoder"] == os.path.join(str(root), "text_encoders", released)


def test_an_explicit_override_reaches_a_gguf(tmp_path):
    root = _tree(tmp_path)
    override = os.path.join(str(root), "text_encoders", GGUF_NAME)

    layout = loader.detect_minimax_h3_layout(str(root), te_override=override)

    assert layout["text_encoder"] == override
    assert layout["text_encoder_reason"] == "explicit override"


def test_the_capability_predicate_rejects_a_gguf_directly(tmp_path):
    path = tmp_path / GGUF_NAME
    _write_gguf(path)

    assert loader._te_capability_accept(path) is False


def test_the_listing_offers_a_gguf_with_its_own_variant(tmp_path):
    root = _tree(tmp_path)

    entries = loader.list_minimax_h3_text_encoder_candidates(str(root))

    entry = next(item for item in entries if item["path"].endswith(GGUF_NAME))
    assert entry["compatible"] is True
    assert entry["variant"] == "gguf_q8_0"
    assert entry["hidden_size"] == HIDDEN
    assert entry["block_count"] == BLOCKS
    assert entry["num_hidden_layers"] is None
    assert "Never the architecture default" in entry["reason"]
    assert "clip_projections/" in entry["reason"]


def test_the_choices_pair_a_gguf_with_the_trained_projection(tmp_path):
    root = _tree(tmp_path)

    choices = loader.describe_minimax_h3_text_encoder_choices(str(root))

    entry = next(item for item in choices["text_encoders"] if item["path"].endswith(GGUF_NAME))
    assert entry["requires_projection"] is True
    assert entry["projection"].endswith("mmh3-4b-clipproj-celeb-mlp.safetensors")
    # Keyed by basename, and no measurement was taken on THIS file.
    assert entry["agreement"] is None


# ---------------------------------------------------------------------------
# The modules, and the streaming trip the encode path makes
# ---------------------------------------------------------------------------

def test_packed_buffers_survive_the_gpu_module_params_trip():
    """``_gpu_module_params`` must hand the uint8 codes over unwidened."""
    from core.models.minimax_h3.h3_pipeline_ops import _gpu_module_params

    values = np.random.default_rng(3).standard_normal((8, 64)).astype(np.float32)
    module = GgufQ8Linear(64, 8)
    module.q_packed = torch.from_numpy(_pack_q8_0(values))

    params = _gpu_module_params(module, torch.device("cpu"))
    assert params["q_packed"].dtype is torch.uint8

    x = torch.randn(2, 5, 64)
    got = torch.func.functional_call(module, params, args=(x,))

    reference = torch.nn.functional.linear(x, dequantize_q8_0(module.q_packed, 64))
    assert torch.allclose(got, reference)


def test_the_embedding_dequantizes_only_the_rows_it_gathers():
    values = np.random.default_rng(4).standard_normal((VOCAB, HIDDEN)).astype(np.float32)
    module = GgufQ8Embedding(VOCAB, HIDDEN, torch.bfloat16)
    module.q_packed = torch.from_numpy(_pack_q8_0(values))

    got = module(torch.tensor([[3, 7, 3]]))

    assert got.shape == (1, 3, HIDDEN)
    assert got.dtype is torch.bfloat16
    whole = dequantize_q8_0(module.q_packed, HIDDEN).to(torch.bfloat16)
    assert torch.equal(got[0, 0], whole[3])
    assert torch.equal(got[0, 1], whole[7])


# ---------------------------------------------------------------------------
# The real build
# ---------------------------------------------------------------------------

def test_the_module_is_built_from_the_gguf_metadata_at_the_projections_tap(tmp_path, capsys):
    official = _official_tree(tmp_path)
    path = _write_gguf(tmp_path / GGUF_NAME)

    model, config = loader._build_text_encoder(path, official, tap=TAP)
    try:
        assert config.text_config.hidden_size == HIDDEN
        assert config.text_config.intermediate_size == FFN
        assert config.text_config.num_hidden_layers == TAP
        layers = model.model.language_model.layers
        assert len(layers) == TAP
        assert isinstance(layers[0].self_attn.q_proj, GgufQ8Linear)
        assert isinstance(model.model.language_model.embed_tokens, GgufQ8Embedding)
        assert layers[0].self_attn.q_proj.q_packed.dtype is torch.uint8
        out = capsys.readouterr().out
        assert "geometry from the GGUF's own KV metadata" in out
        assert f"decoder layer {TAP} of the GGUF's {BLOCKS}" in out
    finally:
        _drop(model)


def test_a_build_without_a_resolved_tap_is_refused(tmp_path):
    official = _official_tree(tmp_path)
    path = _write_gguf(tmp_path / GGUF_NAME)

    with pytest.raises(ValueError, match=r"trained projection's tap"):
        loader._build_text_encoder(path, official)


def test_the_two_value_entry_point_refuses_a_gguf(tmp_path):
    path = _write_gguf(tmp_path / GGUF_NAME)

    with pytest.raises(ValueError, match=r"small stand-in text encoder"):
        loader.build_minimax_h3_text_encoder(path, None)


def test_the_built_encoder_reproduces_the_files_own_weights(tmp_path):
    """The installed buffers dequantize back to what was written."""
    official = _official_tree(tmp_path)
    tensors = _weights(seed=9)
    path = _write_gguf(tmp_path / GGUF_NAME, tensors=tensors)

    model, _config = loader._build_text_encoder(path, official, tap=TAP)
    try:
        q_proj = model.model.language_model.layers[1].self_attn.q_proj
        got = dequantize_q8_0(q_proj.q_packed, HIDDEN)
        want = torch.from_numpy(tensors["blk.1.attn_q.weight"])
        assert float((got - want).abs().max()) < 0.05
        norm = model.model.language_model.layers[0].input_layernorm.weight
        assert torch.equal(norm.detach(), torch.from_numpy(tensors["blk.0.attn_norm.weight"]))
    finally:
        _drop(model)


def test_dropping_the_encoder_releases_the_gguf_mapping(tmp_path):
    """``assert_no_live_text_encoder`` must cover a GGUF-backed encoder's buffers.

    Deleting the file is the second half of the proof: on Windows an open
    mapping refuses that outright, so a successful ``os.remove`` says the mmap
    really went away with the encoder.
    """
    import gc

    official = _official_tree(tmp_path)
    path = _write_gguf(tmp_path / GGUF_NAME)

    model, _config = loader._build_text_encoder(path, official, tap=TAP)
    held = model.model.language_model.layers[0].self_attn.q_proj.q_packed
    with pytest.raises(RuntimeError, match=r"mapped tensors"):
        loader.assert_no_live_text_encoder()

    del held, model
    gc.collect()
    loader.assert_no_live_text_encoder()
    os.remove(path)
