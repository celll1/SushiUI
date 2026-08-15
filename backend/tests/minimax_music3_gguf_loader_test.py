"""MiniMax Music 3 GGUF wiring -- design doc phase 11.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_gguf_loader_test.py -v

Weight-bearing round trips use the tiny, real, hand-built GGUF fixtures in
``tests.minimax_music3_gguf_fixture`` -- no multi-GB file is opened. A real
load against the staged snapshot (header census, dim-order proof against the
installed ``gguf`` package, bit-exactness checks against ``official/``, and a
full DiT-through-remap-into-vendored-modules load) was verified manually
while writing this loader, same convention every other weight-bearing claim
in ``minimax_music3_loader_test.py`` follows -- not repeated here because it
requires the model snapshot, which is not part of this repo.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.common import gguf_container as g  # noqa: E402
from core.models.minimax_music3 import loader  # noqa: E402


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_is_minimax_music3_gguf_dit_true_for_the_dit_signature(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_tiny_gguf_dit_and_official_tree

    fixture = write_tiny_gguf_dit_and_official_tree(tmp_path)
    assert loader.is_minimax_music3_gguf_dit(fixture["dit_path"]) is True


def test_is_minimax_music3_gguf_dit_false_for_wrong_architecture_metadata(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_gguf

    path = os.path.join(str(tmp_path), "foreign.gguf")
    write_gguf(
        path,
        {"diffusion_transformer.transformer.project_in.weight": torch.randn(2, 2),
         "cond_layer_logits": torch.randn(2), "latent_conditioners.0.weight": torch.randn(2, 2)},
        {"general.architecture": "some_other_model"},
    )
    assert loader.is_minimax_music3_gguf_dit(path) is False


def test_is_minimax_music3_gguf_dit_false_for_a_music3_gguf_that_is_not_the_dit(tmp_path):
    """`general.architecture` matches, but the tensor-name signature does not
    -- e.g. the text-encoder GGUF must not be mis-claimed as the DiT."""
    from tests.minimax_music3_gguf_fixture import write_tiny_pruned_gguf_text_encoder_and_official_tree

    fixture = write_tiny_pruned_gguf_text_encoder_and_official_tree(tmp_path)
    assert loader.is_minimax_music3_gguf_dit(fixture["text_encoder_path"]) is False


def test_is_minimax_music3_gguf_dit_never_raises_on_a_foreign_file(tmp_path):
    path = os.path.join(str(tmp_path), "not_gguf.bin")
    with open(path, "wb") as fh:
        fh.write(b"not a gguf file at all")
    assert loader.is_minimax_music3_gguf_dit(path) is False


def test_detect_layout_walks_up_from_a_gguf_dit_file_to_official(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_tiny_gguf_dit_and_official_tree

    fixture = write_tiny_gguf_dit_and_official_tree(tmp_path)
    layout = loader.detect_minimax_music3_layout(fixture["dit_path"])
    assert layout is not None
    assert layout["flat_dit"] == fixture["dit_path"]
    # (the fixture's official/ has no modular_model_index.json -- same
    # convention `minimax_music3_flat_dit_fixture` follows; the builder round
    # trip below is exercised directly, matching
    # `test_flat_dit_with_official_present_is_now_loadable`'s own note.)


def test_detect_layout_identifies_a_lone_gguf_dit_with_no_official_tree(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_gguf

    dit_path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_lonely.gguf")
    write_gguf(
        dit_path,
        {"diffusion_transformer.transformer.project_in.weight": torch.randn(2, 2),
         "cond_layer_logits": torch.randn(2), "latent_conditioners.0.weight": torch.randn(2, 2)},
        {"general.architecture": "minimax_music3"},
    )
    layout = loader.detect_minimax_music3_layout(dit_path)
    assert layout is not None
    assert layout["official"] is None
    assert layout["flat_dit"] == dit_path


def test_detect_layout_does_not_claim_a_gguf_with_no_music3_signature(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_gguf

    path = os.path.join(str(tmp_path), "diffusion_models", "unrelated.gguf")
    write_gguf(path, {"some.weight": torch.randn(2, 2)}, {})
    assert loader.detect_minimax_music3_layout(path) is None


# ---------------------------------------------------------------------------
# DiT round trip: real values land in the right place, through GGUF instead
# of safetensors, via the SAME `flat_remap` used by the safetensors path.
# ---------------------------------------------------------------------------

def test_gguf_dit_builder_round_trip(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_tiny_gguf_dit_and_official_tree

    fixture = write_tiny_gguf_dit_and_official_tree(tmp_path)
    transformer, transformer_config, condition_encoder, condition_encoder_config = (
        loader.build_transformer_and_condition_encoder_from_gguf_dit(
            fixture["dit_path"], fixture["official"], torch.float16,
        )
    )
    assert type(transformer).__name__ == "MiniMaxMusic3Transformer1DModel"
    assert type(condition_encoder).__name__ == "MiniMaxMusic3ConditionEncoder"

    got = transformer.proj_in.weight.to(torch.float32)
    expected = fixture["expected_proj_in_weight"].to(torch.float32)
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-2)
    # And NOT official/'s placeholder value -- catches a source-substitution bug.
    assert not torch.allclose(got, fixture["official_placeholder_proj_in_weight"].to(torch.float32))

    # Pin the q/k/v split order through the GGUF round trip too.
    fused_qkv = fixture["expected_fused_qkv"]
    inner_dim = fused_qkv.shape[0] // 3
    block0 = transformer.transformer_blocks[0]
    tol = dict(atol=1e-3, rtol=1e-2)
    assert torch.allclose(block0.attn.to_q.weight.to(torch.float32), fused_qkv[0:inner_dim].to(torch.float32), **tol)
    assert torch.allclose(block0.attn.to_k.weight.to(torch.float32), fused_qkv[inner_dim:2 * inner_dim].to(torch.float32), **tol)
    assert torch.allclose(block0.attn.to_v.weight.to(torch.float32), fused_qkv[2 * inner_dim:3 * inner_dim].to(torch.float32), **tol)


def test_gguf_dit_builder_matches_the_safetensors_builder_on_identical_tensors(tmp_path):
    """The SAME fixture geometry, written twice -- once as safetensors (the
    existing phase-9 fixture), once as GGUF -- with different underlying
    random values (independently seeded) but structurally identical shapes.
    Both builders must produce a transformer whose GEOMETRY (module tree,
    parameter shapes) agrees; this is the closest a synthetic fixture can get
    to "the two container formats feed the same remap identically" without
    the real multi-GB files."""
    from tests.minimax_music3_flat_dit_fixture import write_tiny_flat_dit_and_official_tree
    from tests.minimax_music3_gguf_fixture import write_tiny_gguf_dit_and_official_tree

    st_fixture = write_tiny_flat_dit_and_official_tree(tmp_path / "st")
    gguf_fixture = write_tiny_gguf_dit_and_official_tree(tmp_path / "gguf")

    st_transformer, _, st_cond, _ = loader.build_transformer_and_condition_encoder_from_flat_dit(
        st_fixture["dit_path"], st_fixture["official"], torch.float32,
    )
    gguf_transformer, _, gguf_cond, _ = loader.build_transformer_and_condition_encoder_from_gguf_dit(
        gguf_fixture["dit_path"], gguf_fixture["official"], torch.float32,
    )
    assert st_transformer.proj_in.weight.shape == gguf_transformer.proj_in.weight.shape
    assert st_cond.proj.weight.shape == gguf_cond.proj.weight.shape
    assert set(dict(st_transformer.named_parameters()).keys()) == set(dict(gguf_transformer.named_parameters()).keys())


# ---------------------------------------------------------------------------
# Q8_0 refusal (DiT): header-only, before any tensor byte is read.
# ---------------------------------------------------------------------------

def test_gguf_dit_with_q8_0_tensor_is_refused_header_only(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_gguf_dit_with_q8_0_tensor

    dit_path = write_gguf_dit_with_q8_0_tensor(tmp_path)
    official = os.path.join(str(tmp_path), "official")
    os.makedirs(official, exist_ok=True)
    with pytest.raises(g.GGUFUnsupportedTensorTypeError, match="phase 12"):
        loader.build_transformer_and_condition_encoder_from_gguf_dit(dit_path, official, torch.bfloat16)


def test_gguf_dit_q8_0_refusal_fires_before_official_configs_are_even_needed(tmp_path):
    """`official` does not need to exist at all for the refusal to fire --
    proves the gate runs before `_read_component_config` (config.json reads),
    let alone before any tensor byte."""
    from tests.minimax_music3_gguf_fixture import write_gguf_dit_with_q8_0_tensor

    dit_path = write_gguf_dit_with_q8_0_tensor(tmp_path)
    with pytest.raises(g.GGUFUnsupportedTensorTypeError, match="phase 12"):
        loader.build_transformer_and_condition_encoder_from_gguf_dit(
            dit_path, os.path.join(str(tmp_path), "nowhere"), torch.bfloat16,
        )


# ---------------------------------------------------------------------------
# Pruned text-encoder GGUF: round trip on an all-unquantized fixture, Q8_0
# refusal on a Q8_0-bearing one (mirrors the real staged file's situation).
# ---------------------------------------------------------------------------

def test_pruned_gguf_text_encoder_builder_round_trip(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_tiny_pruned_gguf_text_encoder_and_official_tree

    fixture = write_tiny_pruned_gguf_text_encoder_and_official_tree(tmp_path)
    language_model, rvq_depth_decoder, depth_config = (
        loader.build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder(
            fixture["text_encoder_path"], fixture["official"], torch.float32,
        )
    )
    assert type(language_model).__name__ == "Qwen3ForCausalLM"
    assert type(rvq_depth_decoder).__name__ == "MiniMaxMusic3RVQDepthDecoder"
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

    # Pin the GQA-uneven qkv split through the GGUF round trip too (mirrors
    # the safetensors pruned test's own check).
    from tests.minimax_music3_pruned_text_encoder_fixture import KV_DIM, Q_DIM

    fused_qkv = fixture["expected_fused_qkv"]
    layer0 = language_model.model.layers[0]
    tol = dict(atol=1e-5, rtol=1e-4)
    assert torch.allclose(layer0.self_attn.q_proj.weight.to(torch.float32), fused_qkv[0:Q_DIM], **tol)
    assert torch.allclose(layer0.self_attn.k_proj.weight.to(torch.float32), fused_qkv[Q_DIM:Q_DIM + KV_DIM], **tol)
    assert torch.allclose(layer0.self_attn.v_proj.weight.to(torch.float32), fused_qkv[Q_DIM + KV_DIM:Q_DIM + 2 * KV_DIM], **tol)


def test_pruned_gguf_text_encoder_with_q8_0_tensor_is_refused_header_only(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_pruned_gguf_text_encoder_with_q8_0_tensor

    path = write_pruned_gguf_text_encoder_with_q8_0_tensor(tmp_path)
    official = os.path.join(str(tmp_path), "official")
    os.makedirs(official, exist_ok=True)
    with pytest.raises(g.GGUFUnsupportedTensorTypeError, match="phase 12"):
        loader.build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder(path, official, torch.float32)


def test_pruned_gguf_text_encoder_q8_0_refusal_fires_before_rope_theta_config_read(tmp_path):
    """`official/language_model/config.json` does not need to exist -- the
    Q8_0 gate runs before that read (and long before the ~9.6 GB Q8_0-heavy
    real file's data section would be opened)."""
    from tests.minimax_music3_gguf_fixture import write_pruned_gguf_text_encoder_with_q8_0_tensor

    path = write_pruned_gguf_text_encoder_with_q8_0_tensor(tmp_path)
    with pytest.raises(g.GGUFUnsupportedTensorTypeError, match="phase 12"):
        loader.build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder(
            path, os.path.join(str(tmp_path), "nowhere"), torch.float32,
        )


def test_pruned_gguf_text_encoder_builder_refuses_wrong_architecture_metadata(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_gguf

    path = os.path.join(str(tmp_path), "text_encoders", "foreign.gguf")
    write_gguf(
        path,
        {"model.embed_tokens_prefill.weight": torch.randn(2, 2),
         "model.embed_tokens_audio.weight": torch.randn(2, 2),
         "model.lm_head_pruned.weight": torch.randn(2, 2)},
        {"general.architecture": "some_other_model"},
    )
    with pytest.raises(ValueError, match="general.architecture"):
        loader.build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder(
            path, os.path.join(str(tmp_path), "official"), torch.float32,
        )


def test_pruned_gguf_text_encoder_builder_refuses_a_non_pruned_signature(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_gguf

    path = os.path.join(str(tmp_path), "text_encoders", "non_pruned.gguf")
    write_gguf(
        path,
        {"model.embed_tokens.weight": torch.randn(2, 2), "model.lm_head.weight": torch.randn(2, 2)},
        {"general.architecture": "minimax_music3"},
    )
    with pytest.raises(ValueError, match="pruned-vocabulary"):
        loader.build_language_model_and_depth_decoder_from_pruned_gguf_text_encoder(
            path, os.path.join(str(tmp_path), "official"), torch.float32,
        )
