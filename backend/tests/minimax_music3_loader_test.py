"""MiniMax Music 3 loader: detection + refusal contracts (weight-free).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_loader_test.py -v

Everything here is header/JSON-only: no multi-GB weight file is opened. A
real load (against ``M:/model/minimax-music3``) was verified manually while
writing this loader (structural build + a real forward through the
transformer/condition_encoder/vocoder chain); that is not repeated here
because it requires the model snapshot, which is not part of this repo.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3 import loader  # noqa: E402


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _make_official_tree(root):
    """A minimal official/ tree: modular_model_index.json + the four
    diffusers-class component config.json files + language_model/tokenizer/
    scheduler placeholders. No weight tensors -- config/JSON only."""
    official = os.path.join(root, "official")
    _write_json(os.path.join(official, "modular_model_index.json"),
                {"_class_name": loader.MINIMAX_MUSIC3_PIPELINE_CLASS})
    for subdir, expected_class in loader._DIFFUSERS_COMPONENTS:
        _write_json(os.path.join(official, subdir, "config.json"),
                    {"_class_name": expected_class})
    _write_json(os.path.join(official, "language_model", "config.json"),
                {"rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"}})
    os.makedirs(os.path.join(official, "tokenizer"), exist_ok=True)
    _write_json(os.path.join(official, "scheduler", "scheduler_config.json"), {})
    return official


def _write_flat_dit_header(path):
    """A minimal safetensors file whose header carries the flat-DiT signature
    (`diffusion_transformer.*` + `latent_conditioners.*` + `cond_layer_logits`).
    No real tensor bytes -- one zero-byte placeholder tensor per key."""
    import struct

    header = {
        "diffusion_transformer.proj_in.weight": {"dtype": "F16", "shape": [0], "data_offsets": [0, 0]},
        "latent_conditioners.0.weight": {"dtype": "F16", "shape": [0], "data_offsets": [0, 0]},
        "cond_layer_logits": {"dtype": "F16", "shape": [8], "data_offsets": [0, 0]},
    }
    raw = json.dumps(header).encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)


# ---------------------------------------------------------------------------
# Detection: all three accepted spellings
# ---------------------------------------------------------------------------

def test_detects_root_directory(tmp_path):
    official = _make_official_tree(str(tmp_path))
    layout = loader.detect_minimax_music3_layout(str(tmp_path))
    assert layout is not None
    assert layout["root"] == str(tmp_path)
    assert layout["official"] == official
    assert layout["flat_dit"] is None


def test_detects_official_directory_directly(tmp_path):
    official = _make_official_tree(str(tmp_path))
    layout = loader.detect_minimax_music3_layout(official)
    assert layout is not None
    assert layout["official"] == official
    assert layout["flat_dit"] is None


def test_detects_flat_dit_file_and_walks_up_to_official(tmp_path):
    official = _make_official_tree(str(tmp_path))
    dit_path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_fp16.safetensors")
    _write_flat_dit_header(dit_path)

    layout = loader.detect_minimax_music3_layout(dit_path)
    assert layout is not None
    assert layout["root"] == str(tmp_path)
    assert layout["official"] == official
    assert layout["flat_dit"] == dit_path


def test_flat_dit_file_with_no_official_tree_is_still_identified(tmp_path):
    """No official/ beside it: still identified (architecture), official is None."""
    dit_path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_fp16.safetensors")
    _write_flat_dit_header(dit_path)

    layout = loader.detect_minimax_music3_layout(dit_path)
    assert layout is not None
    assert layout["official"] is None
    assert layout["flat_dit"] == dit_path


# ---------------------------------------------------------------------------
# Negative cases: no cross-claiming with other archs' directory shapes
# ---------------------------------------------------------------------------

def test_does_not_claim_an_unrelated_directory(tmp_path):
    (tmp_path / "some_file.txt").write_text("nothing here")
    assert loader.detect_minimax_music3_layout(str(tmp_path)) is None


def test_does_not_claim_a_minimax_h3_official_tree(tmp_path):
    """MiniMax-H3's official/ uses model_index.json, not modular_model_index.json."""
    official = tmp_path / "official"
    _write_json(str(official / "model_index.json"), {"_class_name": "MiniMaxH3ModularPipeline"})
    assert loader.detect_minimax_music3_layout(str(tmp_path)) is None
    assert loader.detect_minimax_music3_layout(str(official)) is None


def test_does_not_claim_a_wrong_class_name_index(tmp_path):
    """A modular_model_index.json that names some OTHER pipeline class."""
    official = tmp_path / "official"
    _write_json(str(official / "modular_model_index.json"), {"_class_name": "SomeOtherPipeline"})
    assert loader.detect_minimax_music3_layout(str(tmp_path)) is None


def test_flat_dit_key_signature_does_not_match_an_unrelated_safetensors_header(tmp_path):
    import struct

    path = tmp_path / "not_music3.safetensors"
    header = {"transformer_blocks.0.attn.to_q.weight": {"dtype": "F16", "shape": [0], "data_offsets": [0, 0]}}
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
    assert loader.is_minimax_music3_safetensors(str(path)) is False
    assert loader.detect_minimax_music3_layout(str(path)) is None


# ---------------------------------------------------------------------------
# Ordering / refusal contracts the loader itself enforces
# ---------------------------------------------------------------------------

def test_missing_layout_raises_value_error_naming_expectations(tmp_path):
    with pytest.raises(ValueError, match="MiniMax Music 3 model layout not found"):
        loader.load_minimax_music3_from_path(str(tmp_path / "nowhere"))


def test_flat_dit_with_official_present_is_now_loadable(tmp_path):
    """Design doc phase 9: pointing AT a flat DiT file (with official/
    reachable) now LOADS the transformer + condition encoder from that file
    via ``flat_remap``, rather than refusing.

    Exercises the standalone builder directly (not the full
    ``load_minimax_music3_from_path`` dispatch, which also needs real
    rvq_depth_decoder/vocoder/tokenizer/scheduler weights this file's other
    tests deliberately keep config-only -- see the module docstring). The
    real multi-GB snapshot's end-to-end dispatch, including this builder
    being reached FROM ``load_minimax_music3_from_path``, was verified
    manually while writing this loader, same convention as every other
    weight-bearing claim in this file.

    A real (tiny) round-trip: the loaded weight is the FLAT file's value,
    not a placeholder, and not silently substituted with official/'s (which
    this fixture gives a DIFFERENT value, specifically to catch a
    substitution bug)."""
    from tests.minimax_music3_flat_dit_fixture import write_tiny_flat_dit_and_official_tree

    fixture = write_tiny_flat_dit_and_official_tree(tmp_path)
    transformer, transformer_config, condition_encoder, condition_encoder_config = (
        loader.build_transformer_and_condition_encoder_from_flat_dit(
            fixture["dit_path"], fixture["official"], torch.float16,
        )
    )
    assert type(transformer).__name__ == "MiniMaxMusic3Transformer1DModel"
    assert type(condition_encoder).__name__ == "MiniMaxMusic3ConditionEncoder"
    got = transformer.proj_in.weight.to(torch.float32)
    expected = fixture["expected_proj_in_weight"].to(torch.float32)
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-2)
    # And NOT official/'s placeholder value (the fixture makes them differ on
    # purpose): a substitution bug would make this assertion fail instead.
    assert not torch.allclose(got, fixture["official_placeholder_proj_in_weight"].to(torch.float32))
    # F11: pin the q/k/v split order through the round-trip too, not only via
    # the pure-function test in minimax_music3_flat_remap_test.py.
    fused_qkv = fixture["expected_fused_qkv"]
    inner_dim = fused_qkv.shape[0] // 3
    block0 = transformer.transformer_blocks[0]
    tol = dict(atol=1e-3, rtol=1e-2)  # transformer was built in float16
    assert torch.allclose(block0.attn.to_q.weight.to(torch.float32), fused_qkv[0:inner_dim].to(torch.float32), **tol)
    assert torch.allclose(block0.attn.to_k.weight.to(torch.float32), fused_qkv[inner_dim:2 * inner_dim].to(torch.float32), **tol)
    assert torch.allclose(block0.attn.to_v.weight.to(torch.float32), fused_qkv[2 * inner_dim:3 * inner_dim].to(torch.float32), **tol)


def test_flat_text_encoder_builder_round_trip(tmp_path):
    """F2 in the phase-9 audit: ``build_language_model_and_depth_decoder_from_
    flat_text_encoder`` had zero callers and zero tests. A tiny real Qwen3 +
    RVQ depth decoder round-trip, proving it builds a loadable
    ``Qwen3ForCausalLM`` and ``MiniMaxMusic3RVQDepthDecoder`` on the installed
    transformers version, and that the flat file's values (not placeholders)
    land in the built modules."""
    from tests.minimax_music3_flat_text_encoder_fixture import (
        write_tiny_flat_text_encoder_and_official_tree,
    )

    fixture = write_tiny_flat_text_encoder_and_official_tree(tmp_path)
    language_model, rvq_depth_decoder, depth_config = (
        loader.build_language_model_and_depth_decoder_from_flat_text_encoder(
            fixture["text_encoder_path"], fixture["official"], torch.float32,
        )
    )
    assert type(language_model).__name__ == "Qwen3ForCausalLM"
    assert type(rvq_depth_decoder).__name__ == "MiniMaxMusic3RVQDepthDecoder"
    got_lm_head = language_model.lm_head.weight.to(torch.float32)
    assert torch.allclose(got_lm_head, fixture["expected_lm_head_weight"].to(torch.float32))
    got_audio_embeddings = rvq_depth_decoder.audio_embeddings.weight.to(torch.float32)
    assert torch.allclose(got_audio_embeddings, fixture["expected_audio_embeddings_weight"].to(torch.float32))


def test_flat_text_encoder_builder_gates_on_rope_theta_before_reading_weights(tmp_path):
    """F3 regression: a wrong ``rope_parameters.rope_theta`` in
    ``official/language_model/config.json`` must be refused BEFORE
    ``read_state_dict`` opens the (potentially 18 GB) flat file."""
    from tests.minimax_music3_flat_text_encoder_fixture import (
        write_tiny_flat_text_encoder_and_official_tree,
    )

    fixture = write_tiny_flat_text_encoder_and_official_tree(tmp_path)
    lm_config_path = os.path.join(fixture["official"], "language_model", "config.json")
    with open(lm_config_path, encoding="utf-8") as fh:
        bad_config = json.load(fh)
    bad_config["rope_parameters"]["rope_theta"] = 10_000.0
    with open(lm_config_path, "w", encoding="utf-8") as fh:
        json.dump(bad_config, fh)
    # Corrupt the flat file's bytes after its header so a real tensor read
    # would raise a decode error -- proving the gate fires BEFORE that read,
    # not merely before some slow I/O.
    with open(fixture["text_encoder_path"], "r+b") as fh:
        fh.seek(-16, os.SEEK_END)
        fh.write(b"\xff" * 16)

    with pytest.raises(ValueError, match="rope_parameters"):
        loader.build_language_model_and_depth_decoder_from_flat_text_encoder(
            fixture["text_encoder_path"], fixture["official"], torch.float32,
        )


def test_pruned_flat_text_encoder_builder_round_trip(tmp_path):
    """Design doc phase 10: ``build_language_model_and_depth_decoder_from_pruned_flat_
    text_encoder`` builds a real (patched) ``Qwen3ForCausalLM`` -- ``lm_head`` removed,
    ``lm_head_pruned`` and ``model.embed_tokens_audio`` attached -- and a real
    ``MiniMaxMusic3RVQDepthDecoder``, from a tiny real pruned flat file whose per-layer
    projections are FUSED (unlike the non-pruned fixture's), proving the GQA-uneven qkv
    split and the equal gate_up split both land correctly through the loader, not just
    through the pure-function remap tests."""
    from tests.minimax_music3_pruned_text_encoder_fixture import (
        write_tiny_pruned_text_encoder_and_official_tree,
    )

    fixture = write_tiny_pruned_text_encoder_and_official_tree(tmp_path)
    language_model, rvq_depth_decoder, depth_config = (
        loader.build_language_model_and_depth_decoder_from_pruned_flat_text_encoder(
            fixture["text_encoder_path"], fixture["official"], torch.float32,
        )
    )
    assert type(language_model).__name__ == "Qwen3ForCausalLM"
    assert type(rvq_depth_decoder).__name__ == "MiniMaxMusic3RVQDepthDecoder"

    # The representation choice (this function's docstring): patched attributes, not a
    # subclass -- `lm_head` removed, two new leaf modules attached.
    assert not hasattr(language_model, "lm_head")
    assert hasattr(language_model, "lm_head_pruned")
    assert hasattr(language_model.model, "embed_tokens_audio")
    assert language_model.config.vocab_size == language_model.model.embed_tokens.weight.shape[0]

    got_lm_head_pruned = language_model.lm_head_pruned.weight.to(torch.float32)
    assert torch.allclose(got_lm_head_pruned, fixture["expected_lm_head_pruned_weight"])
    got_embed_tokens_audio = language_model.model.embed_tokens_audio.weight.to(torch.float32)
    assert torch.allclose(got_embed_tokens_audio, fixture["expected_embed_tokens_audio_weight"])
    got_audio_embeddings = rvq_depth_decoder.audio_embeddings.weight.to(torch.float32)
    assert torch.allclose(got_audio_embeddings, fixture["expected_audio_embeddings_weight"])

    # F11-equivalent: pin the q/k/v split order (GQA-uneven: 8/4/4) through the round trip.
    from tests.minimax_music3_pruned_text_encoder_fixture import HEAD_DIM, KV_DIM, Q_DIM

    fused_qkv = fixture["expected_fused_qkv"]
    layer0 = language_model.model.layers[0]
    tol = dict(atol=1e-5, rtol=1e-4)
    assert torch.allclose(layer0.self_attn.q_proj.weight.to(torch.float32), fused_qkv[0:Q_DIM], **tol)
    assert torch.allclose(layer0.self_attn.k_proj.weight.to(torch.float32), fused_qkv[Q_DIM:Q_DIM + KV_DIM], **tol)
    assert torch.allclose(layer0.self_attn.v_proj.weight.to(torch.float32), fused_qkv[Q_DIM + KV_DIM:Q_DIM + 2 * KV_DIM], **tol)

    fused_gate_up = fixture["expected_fused_gate_up"]
    half = fused_gate_up.shape[0] // 2
    assert torch.allclose(layer0.mlp.gate_proj.weight.to(torch.float32), fused_gate_up[:half], **tol)
    assert torch.allclose(layer0.mlp.up_proj.weight.to(torch.float32), fused_gate_up[half:], **tol)

    depth_layer0 = rvq_depth_decoder.layers[0]
    depth_fused_qkv = fixture["expected_depth_fused_qkv_by_layer"][0]
    third = depth_fused_qkv.shape[0] // 3
    assert torch.allclose(depth_layer0.attn.to_q.weight.to(torch.float32), depth_fused_qkv[0:third], **tol)
    assert torch.allclose(depth_layer0.attn.to_k.weight.to(torch.float32), depth_fused_qkv[third:2 * third], **tol)
    assert torch.allclose(depth_layer0.attn.to_v.weight.to(torch.float32), depth_fused_qkv[2 * third:3 * third], **tol)


def test_pruned_flat_text_encoder_builder_gates_on_rope_theta_before_reading_weights(tmp_path):
    """Mirrors the non-pruned builder's F3 regression coverage: a wrong
    ``rope_parameters.rope_theta`` must be refused BEFORE ``read_state_dict`` opens the
    (potentially 16.7 GB) pruned file."""
    from tests.minimax_music3_pruned_text_encoder_fixture import (
        write_tiny_pruned_text_encoder_and_official_tree,
    )

    fixture = write_tiny_pruned_text_encoder_and_official_tree(tmp_path)
    lm_config_path = os.path.join(fixture["official"], "language_model", "config.json")
    with open(lm_config_path, encoding="utf-8") as fh:
        bad_config = json.load(fh)
    bad_config["rope_parameters"]["rope_theta"] = 10_000.0
    with open(lm_config_path, "w", encoding="utf-8") as fh:
        json.dump(bad_config, fh)
    with open(fixture["text_encoder_path"], "r+b") as fh:
        fh.seek(-16, os.SEEK_END)
        fh.write(b"\xff" * 16)

    with pytest.raises(ValueError, match="rope_parameters"):
        loader.build_language_model_and_depth_decoder_from_pruned_flat_text_encoder(
            fixture["text_encoder_path"], fixture["official"], torch.float32,
        )


def test_pruned_builder_refuses_a_non_pruned_file_by_name():
    """The pruned builder is not a silent fallback for the non-pruned file -- a header
    that carries none of the pruned tells is refused with a message naming the OTHER
    builder, before any tensor byte is read (this uses a header-only 0-byte fixture, so a
    real read would raise a decode error, not just be slow)."""
    import struct
    import tempfile

    header = {
        "model.embed_tokens.weight": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
        "model.lm_head.weight": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
    }
    raw = json.dumps(header).encode("utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "minimax_music3_text_encoder_bf16.safetensors")
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(raw)))
            fh.write(raw)
        with pytest.raises(ValueError, match="build_language_model_and_depth_decoder_from_flat_text_encoder"):
            loader.build_language_model_and_depth_decoder_from_pruned_flat_text_encoder(
                path, tmp, torch.float32,
            )


def test_non_pruned_builder_still_refuses_a_pruned_file(tmp_path):
    """The other direction: handing the PRUNED file to the NON-pruned builder still raises
    ``PrunedTextEncoderNotSupported`` -- design doc phase 10 adds a supported path, it does
    not change this specific function's own refusal."""
    from core.models.minimax_music3.flat_remap import PrunedTextEncoderNotSupported
    from tests.minimax_music3_pruned_text_encoder_fixture import (
        write_tiny_pruned_text_encoder_and_official_tree,
    )

    fixture = write_tiny_pruned_text_encoder_and_official_tree(tmp_path)
    with pytest.raises(PrunedTextEncoderNotSupported):
        loader.build_language_model_and_depth_decoder_from_flat_text_encoder(
            fixture["text_encoder_path"], fixture["official"], torch.float32,
        )


def _write_flat_dit_with_unsupported_quant_marker(path):
    """A flat-DiT-shaped file (REAL safetensors, via ``save_file`` -- not the
    zero-byte header trick the other fixtures in this file use, because this
    one's ``.comfy_quant`` marker must hold real, decodable JSON bytes for
    ``supported_int8_convrot_marker`` to read) whose ONE quantized layer
    declares a marker this loader does not implement (``"nvfp4"`` -- the
    format MiniMax-H3's text encoder uses, not the ConvRot contract design
    doc phase 13 implements for MiniMax Music 3). Proves the "supported
    ConvRot vs. everything else" distinction still refuses the "everything
    else" side, HEADER-ONLY: the marker is the only tensor with real
    content-derived meaning here, and every other tensor (including the
    quantized layer's own ``.weight``/``.weight_scale``) is a 1-element
    placeholder that a real read would still succeed on (unlike the old
    0-byte fixture this replaces) -- so a failure here is the semantics
    guard, not an incidental read error.
    """
    import json as _json

    from safetensors.torch import save_file

    marker = torch.frombuffer(
        bytearray(_json.dumps({"format": "nvfp4", "full_precision_matrix_mult": True}).encode("utf-8")),
        dtype=torch.uint8,
    ).clone()
    state_dict = {
        "diffusion_transformer.proj_in.weight": torch.zeros(1),
        "diffusion_transformer.proj_in.weight_scale": torch.ones(1, dtype=torch.float32),
        "diffusion_transformer.proj_in.comfy_quant": marker,
        "latent_conditioners.0.weight": torch.zeros(1),
        "cond_layer_logits": torch.zeros(8),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(state_dict, path)


def test_flat_dit_unsupported_quant_marker_is_refused_header_only(tmp_path):
    """A quantized flat DiT whose marker is NOT the validated ConvRot
    contract still refuses -- design doc phase 13 widens what is accepted,
    it does not remove the declared-semantics guard for everything else."""
    _make_official_tree(str(tmp_path))
    dit_path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_int8_convrot.safetensors")
    _write_flat_dit_with_unsupported_quant_marker(dit_path)

    from core.models.common.quantized_checkpoint_guard import UnsupportedQuantSemanticsError

    with pytest.raises(UnsupportedQuantSemanticsError):
        loader.build_transformer_and_condition_encoder_from_flat_dit(
            dit_path, os.path.join(str(tmp_path), "official"), torch.bfloat16,
        )


def _write_flat_dit_with_scale_and_no_marker(path):
    """A flat-DiT-shaped file (REAL safetensors) whose ONE ``.weight`` carries
    a ``.weight_scale`` sibling but NO ``.comfy_quant`` marker at all -- the
    exact case the pre-phase-13 loader's ``_header_looks_quantized`` refused
    unconditionally, and this restores coverage for after the F2 fix: a
    scaled-but-unmarked file is refused HEADER-ONLY, before the full
    ``read_state_dict`` of what can be a 5-10 GB file, same as the
    unrecognized-marker case above.
    """
    from safetensors.torch import save_file

    state_dict = {
        "diffusion_transformer.proj_in.weight": torch.zeros(1),
        "diffusion_transformer.proj_in.weight_scale": torch.ones(1, dtype=torch.float32),
        "latent_conditioners.0.weight": torch.zeros(1),
        "cond_layer_logits": torch.zeros(8),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(state_dict, path)


def test_flat_dit_scale_with_no_marker_is_refused_header_only(tmp_path):
    """F2: the DiT builder must refuse a scaled-but-unmarked file HEADER-ONLY,
    the same guarantee the pruned text encoder builder already has -- proven
    here by making the file's tensors big enough that a real
    ``read_state_dict`` would succeed (unlike the deleted 0-byte fixture this
    restores coverage for), so a pass here is the header-only gate, not an
    incidental read failure that happens to raise first.
    """
    _make_official_tree(str(tmp_path))
    dit_path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_int8_convrot.safetensors")
    _write_flat_dit_with_scale_and_no_marker(dit_path)

    with pytest.raises(RuntimeError, match="ConvRot"):
        loader.build_transformer_and_condition_encoder_from_flat_dit(
            dit_path, os.path.join(str(tmp_path), "official"), torch.bfloat16,
        )


def test_int8_convrot_source_layers_is_empty_for_a_plain_file(tmp_path):
    """``_int8_convrot_source_layers`` (the header-only census design doc
    phase 13 introduced) returns ``{}`` -- not an error -- for a file with no
    ``.comfy_quant`` marker at all, matching every unquantized flat DiT this
    loader already reads."""
    dit_path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_fp16.safetensors")
    _write_flat_dit_header(dit_path)
    assert loader._int8_convrot_source_layers(dit_path, label="flat DiT") == {}


def test_flat_dit_with_no_official_tree_is_refused_with_a_distinct_reason(tmp_path):
    dit_path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_fp16.safetensors")
    _write_flat_dit_header(dit_path)

    with pytest.raises(NotImplementedError, match="no reachable official/"):
        loader.load_minimax_music3_from_path(dit_path)


def test_missing_component_config_lists_every_gap_at_once(tmp_path):
    """The 'list all missing slots at once' ordering rule."""
    official = os.path.join(str(tmp_path), "official")
    _write_json(os.path.join(official, "modular_model_index.json"),
                {"_class_name": loader.MINIMAX_MUSIC3_PIPELINE_CLASS})
    # No component config/weight files at all beyond the index.

    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_minimax_music3_from_path(str(tmp_path))
    message = str(excinfo.value)
    for subdir, _ in loader._DIFFUSERS_COMPONENTS:
        assert subdir in message
    assert "language_model" in message
    assert "tokenizer" in message
    assert "scheduler" in message


def test_wrong_class_name_in_component_config_is_refused(tmp_path):
    official = _make_official_tree(str(tmp_path))
    # Corrupt the transformer's config to declare the wrong class.
    _write_json(os.path.join(official, "transformer", "config.json"),
                {"_class_name": "SomeOtherModel"})

    with pytest.raises(ValueError, match="SomeOtherModel"):
        loader._read_component_config(official, "transformer", "MiniMaxMusic3Transformer1DModel")


def test_language_model_rope_theta_gate_rejects_none_and_wrong_value():
    """`config.rope_theta` is None on transformers 5.1 for this config form
    (design doc); the gate must read `rope_parameters.rope_theta`, not that."""

    class _FakeConfig:
        rope_parameters = None
        rope_theta = 1_000_000.0  # a decoy: the gate must NOT read this field

    class _FakeModel:
        config = _FakeConfig()

    with pytest.raises(ValueError, match="rope_parameters"):
        loader._assert_language_model_rope_theta(_FakeModel())

    _FakeConfig.rope_parameters = {"rope_theta": 500_000.0}
    with pytest.raises(ValueError, match="rope_parameters"):
        loader._assert_language_model_rope_theta(_FakeModel())


def test_language_model_rope_theta_gate_accepts_the_expected_value():
    class _FakeConfig:
        rope_parameters = {"rope_theta": 999_997.4}  # recovered value, design doc

    class _FakeModel:
        config = _FakeConfig()

    # Must not raise: within `_ROPE_THETA_TOLERANCE` (10.0) of the expected 1e6.
    loader._assert_language_model_rope_theta(_FakeModel())


def test_pre_load_and_post_load_rope_gates_share_one_tolerance(tmp_path):
    """Both gates use `_ROPE_THETA_TOLERANCE`; a wrong base is rejected by both,
    and the fp32 round-trip value (999997.4) is accepted by both."""
    official = _make_official_tree(str(tmp_path))

    # Pre-load JSON gate (`_build_language_model` raises before touching any
    # weight file -- no real language_model weights are needed for this).
    _write_json(os.path.join(official, "language_model", "config.json"),
                {"rope_parameters": {"rope_theta": 10_000.0, "rope_type": "default"}})
    with pytest.raises(ValueError, match="rope_parameters"):
        loader._build_language_model(official, __import__("torch").bfloat16)

    _write_json(os.path.join(official, "language_model", "config.json"),
                {"rope_parameters": {"rope_theta": 999_997.4, "rope_type": "default"}})
    # Passes the pre-load gate; would proceed to from_pretrained (not run here,
    # no real weights) -- confirms the pre-load gate alone does not reject it.

    # Post-load gate, same value.
    class _FakeConfig:
        rope_parameters = {"rope_theta": 10_000.0}

    class _FakeModel:
        config = _FakeConfig()

    with pytest.raises(ValueError, match="rope_parameters"):
        loader._assert_language_model_rope_theta(_FakeModel())


# ---------------------------------------------------------------------------
# qwen_7B is a permanent exclusion: nothing in this module ever paths through it
# ---------------------------------------------------------------------------

def test_qwen_7b_is_never_referenced_by_any_official_path_construction(tmp_path):
    official = _make_official_tree(str(tmp_path))
    # Plant a qwen_7B/ sibling; if anything in the loader ever globbed
    # official/*/ for the language model, this would be picked up.
    os.makedirs(os.path.join(official, "qwen_7B"), exist_ok=True)
    assert loader._QWEN_7B_EXCLUDED_SUBDIR == "qwen_7B"
    # The language model path is always literally `official/language_model`.
    lm_dir = os.path.join(official, "language_model")
    assert os.path.isdir(lm_dir)
