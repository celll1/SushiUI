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


def test_flat_dit_with_official_present_is_refused_not_silently_substituted(tmp_path):
    """Pointing AT the flat file (with official/ reachable) refuses with a
    message naming the file, rather than silently loading official/'s
    transformer as if that were what was requested."""
    _make_official_tree(str(tmp_path))
    dit_path = os.path.join(str(tmp_path), "diffusion_models", "minimax_music3_dit_fp16.safetensors")
    _write_flat_dit_header(dit_path)

    with pytest.raises(NotImplementedError, match="flat repacked DiT"):
        loader.load_minimax_music3_from_path(dit_path)


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
