"""MiniMax Music 3: which text-encoder builder a `text_encoder_file` reaches,
and the same-model-reload gate that decides whether naming one forces a
reload.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_text_encoder_choice_test.py -v

WHY THIS FILE EXISTS
--------------------
`core.models.minimax_music3.loader.detect_minimax_music3_text_encoder_source`
and `core.pipeline.DiffusionPipelineManager._minimax_music3_te_selection_
differs` are the two pieces of code that make the four (previously
unreachable) text-encoder builders reachable from a real `POST /models/load`.
Neither had a dedicated test before this file: the detector was covered only
implicitly by the four builders' own round-trip tests (which always pass the
right file to the right function directly), and the gate had no test at all
-- an early version of it fast-returned on `text_encoder_file is None` alone,
with no architecture guard, and silently misfired when an H3 model was
loaded and a Music3-unrelated `text_encoder_file` request reached it (and the
symmetric bug existed on `_minimax_h3_te_selection_differs` itself, fixed
alongside this file). Entirely header-only / fixture-only: no real checkpoint
under `M:/model/minimax-music3` is opened.
"""

import json
import os
import struct
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from core.models.minimax_music3 import loader  # noqa: E402
from core.models.minimax_music3.loader import (  # noqa: E402
    MiniMaxMusic3TextEncoderRefusal,
    detect_minimax_music3_text_encoder_source,
)

# ---------------------------------------------------------------------------
# Detector matrix -- one real (tiny) fixture per source kind.
# ---------------------------------------------------------------------------

def test_flat_non_pruned_safetensors_detected(tmp_path):
    from tests.minimax_music3_flat_text_encoder_fixture import (
        write_tiny_flat_text_encoder_and_official_tree,
    )

    fx = write_tiny_flat_text_encoder_and_official_tree(tmp_path)
    assert detect_minimax_music3_text_encoder_source(fx["text_encoder_path"]) == "flat_non_pruned"


def test_flat_pruned_safetensors_detected(tmp_path):
    from tests.minimax_music3_pruned_text_encoder_fixture import (
        write_tiny_pruned_text_encoder_and_official_tree,
    )

    fx = write_tiny_pruned_text_encoder_and_official_tree(tmp_path)
    assert detect_minimax_music3_text_encoder_source(fx["text_encoder_path"]) == "flat_pruned"


def test_gguf_pruned_dense_detected(tmp_path):
    from tests.minimax_music3_gguf_fixture import (
        write_tiny_pruned_gguf_text_encoder_and_official_tree,
    )

    fx = write_tiny_pruned_gguf_text_encoder_and_official_tree(tmp_path)
    assert detect_minimax_music3_text_encoder_source(fx["text_encoder_path"]) == "gguf_pruned_dense"


def test_gguf_pruned_q8_0_detected(tmp_path):
    from tests.minimax_music3_gguf_fixture import (
        write_tiny_pruned_gguf_q8_0_text_encoder_and_official_tree,
    )

    fx = write_tiny_pruned_gguf_q8_0_text_encoder_and_official_tree(tmp_path)
    assert detect_minimax_music3_text_encoder_source(fx["text_encoder_path"]) == "gguf_pruned_q8_0"


# ---------------------------------------------------------------------------
# Refusals -- every one HEADER-ONLY, none a bare unlabelled exception.
# ---------------------------------------------------------------------------

def test_flat_dit_safetensors_refused_as_text_encoder(tmp_path):
    from tests.minimax_music3_flat_dit_fixture import write_tiny_flat_dit_and_official_tree

    fx = write_tiny_flat_dit_and_official_tree(tmp_path)
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal, match="DiT"):
        detect_minimax_music3_text_encoder_source(fx["dit_path"])


def test_gguf_dit_refused_as_text_encoder(tmp_path):
    from tests.minimax_music3_gguf_fixture import write_tiny_gguf_dit_and_official_tree

    fx = write_tiny_gguf_dit_and_official_tree(tmp_path)
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal, match="DiT"):
        detect_minimax_music3_text_encoder_source(fx["dit_path"])


def test_foreign_safetensors_refused_header_only_not_read_as_state_dict(tmp_path, monkeypatch):
    """F4: a file that is neither the DiT signature nor a pruned/non-pruned
    text-encoder key plan (an SDXL checkpoint, MiniMax-H3's own encoder, any
    unrelated safetensors file) must be refused from the HEADER alone -- not
    after `read_state_dict` has pulled real tensor bytes into RAM. Proven by
    making `read_state_dict` explode if it is ever called."""
    import core.models.common.single_file_format as single_file_format

    def _explode(*_a, **_kw):
        raise AssertionError("read_state_dict was called -- detection was not header-only")

    monkeypatch.setattr(single_file_format, "read_state_dict", _explode)

    path = tmp_path / "not_a_music3_text_encoder.safetensors"
    save_file(
        {
            "unet.conv_in.weight": torch.zeros(4, 4, 3, 3),
            "text_model.embeddings.token_embedding.weight": torch.zeros(8, 8),
        },
        str(path),
    )
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal, match="key signature"):
        detect_minimax_music3_text_encoder_source(str(path))


def test_truncated_safetensors_names_the_path_not_a_bare_json_error(tmp_path):
    """A truncated/corrupt safetensors header must surface as a refusal
    naming the file, not a bare `json.JSONDecodeError` with no path."""
    path = tmp_path / "truncated.safetensors"
    # A plausible header LENGTH (10) followed by 10 bytes that are not valid
    # JSON at all -- exercises the json.loads failure specifically, not the
    # "implausible length" guard `read_safetensors_header` already had.
    path.write_bytes(struct.pack("<Q", 10) + b"not-json.." )
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal, match=r"truncated.safetensors") as exc_info:
        detect_minimax_music3_text_encoder_source(str(path))
    assert "could not be read" in str(exc_info.value)


def test_truncated_gguf_names_the_path(tmp_path):
    path = tmp_path / "truncated.gguf"
    path.write_bytes(b"not a gguf file at all, no magic bytes here")
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal, match=r"truncated.gguf"):
        detect_minimax_music3_text_encoder_source(str(path))


def test_gguf_wrong_architecture_tag_refused(tmp_path):
    """A `llama`-architecture GGUF (or anything not declaring
    general.architecture=minimax_music3) is refused by the tag, never by
    guessing from tensor names."""
    from tests.minimax_music3_gguf_fixture import write_gguf

    path = tmp_path / "llama.gguf"
    write_gguf(
        str(path),
        metadata={"general.architecture": "llama"},
        tensors={"token_embd.weight": torch.zeros(4, 4)},
    )
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal, match="general.architecture"):
        detect_minimax_music3_text_encoder_source(str(path))


def test_gguf_non_pruned_text_encoder_refused_no_builder_exists(tmp_path):
    """A music3-tagged GGUF text encoder without the pruned-vocabulary tells
    has no builder at all (design doc phase 11 covers only the pruned
    distribution) and must say so, not silently route to a dense builder
    that would misread it."""
    from tests.minimax_music3_gguf_fixture import write_gguf

    path = tmp_path / "non_pruned.gguf"
    write_gguf(
        str(path),
        metadata={"general.architecture": "minimax_music3"},
        tensors={"model.embed_tokens.weight": torch.zeros(4, 4)},
    )
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal, match="No non-pruned GGUF"):
        detect_minimax_music3_text_encoder_source(str(path))


def test_nonexistent_path_raises_file_not_found_unwrapped(tmp_path):
    """`FileNotFoundError` for a path that does not exist at all propagates
    exactly as every neighbouring header read in this module does -- it is
    not this function's job to relabel "the file is not there"."""
    with pytest.raises(FileNotFoundError):
        detect_minimax_music3_text_encoder_source(str(tmp_path / "nowhere.safetensors"))


def test_wrong_suffix_refused(tmp_path):
    path = tmp_path / "encoder.bin"
    path.write_bytes(b"whatever")
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal, match="neither a .safetensors nor a .gguf"):
        detect_minimax_music3_text_encoder_source(str(path))


def test_directory_path_refused_not_a_crash(tmp_path):
    d = tmp_path / "some_dir.safetensors"
    d.mkdir()
    with pytest.raises((MiniMaxMusic3TextEncoderRefusal, IsADirectoryError, OSError)):
        detect_minimax_music3_text_encoder_source(str(d))


# ---------------------------------------------------------------------------
# Gate matrix -- `_minimax_music3_te_selection_differs`, real implementation.
# ---------------------------------------------------------------------------

def _manager(**overrides):
    from core import pipeline as pipeline_module

    defaults = dict(is_minimax_music3_model=True, minimax_music3_components={})
    defaults.update(overrides)
    ns = SimpleNamespace(**defaults)
    return ns, pipeline_module.DiffusionPipelineManager._minimax_music3_te_selection_differs


def test_gate_false_when_no_music3_model_loaded_regardless_of_request():
    """F1's twin: the guard on THIS side too. A non-None text_encoder_file
    evaluated while some OTHER architecture is loaded (is_minimax_music3_model
    is False) must not read as 'the music3 selection differs' -- there is no
    music3 selection to differ from."""
    manager, method = _manager(
        is_minimax_music3_model=False,
        minimax_music3_components={"text_encoder_path": "/some/other/file.gguf"},
    )
    assert method(manager, "/completely/unrelated/h3_encoder.safetensors") is False
    assert method(manager, None) is False


def test_gate_false_for_default_load_with_no_prior_override():
    manager, method = _manager(minimax_music3_components={
        "text_encoder_path": "/model/official/language_model",
    })
    assert method(manager, None) is False


def test_gate_false_when_renaming_the_same_loaded_file():
    manager, method = _manager(minimax_music3_components={
        "text_encoder_path": "/model/text_encoders/pruned_q8_0.gguf",
    })
    assert method(manager, "/model/text_encoders/pruned_q8_0.gguf") is False


def test_gate_true_when_a_different_file_is_named():
    manager, method = _manager(minimax_music3_components={
        "text_encoder_path": "/model/text_encoders/pruned_q8_0.gguf",
    })
    assert method(manager, "/model/text_encoders/pruned_bf16.safetensors") is True


def test_gate_true_dropping_back_to_default_from_an_override():
    """The inverse direction the design doc's F2 fix depends on: a loaded
    override (text_encoder_origin: selected_external) must reload when the
    caller asks for None (the official/ default) again."""
    manager, method = _manager(minimax_music3_components={
        "text_encoder_path": "/model/text_encoders/pruned_q8_0.gguf",
        "text_encoder_origin": "selected_external",
    })
    assert method(manager, None) is True


# ---------------------------------------------------------------------------
# Gate matrix -- `_minimax_h3_te_selection_differs`'s NEW music3-side guard
# (audit F1). Pinned with the REAL bound method, not a stub.
# ---------------------------------------------------------------------------

def test_h3_gate_false_when_no_h3_model_loaded_even_with_fields_set():
    from core import pipeline as pipeline_module

    manager = SimpleNamespace(is_minimax_h3_model=False, minimax_h3_components={
        "text_encoder_path": "/some/h3/file.safetensors",
    })
    method = pipeline_module.DiffusionPipelineManager._minimax_h3_te_selection_differs
    # Before the F1 fix this returned True purely because
    # `minimax_h3_components` (real H3 state) happened to be non-empty here
    # despite `is_minimax_h3_model` being False -- the guard must short
    # circuit on the flag, not on whether the dict happens to be populated.
    assert method(manager, "/a/different/music3/text_encoder.gguf", None) is False
    assert method(manager, None, "/some/projection.safetensors") is False


# ---------------------------------------------------------------------------
# Persistence across a restart (audit F2). Mirrors minimax_h3_te_selection_
# api_test.py's own "Persistence across a restart" section exactly, one
# architecture over.
# ---------------------------------------------------------------------------

@pytest.fixture
def music3_manager(tmp_path, monkeypatch):
    """A real `DiffusionPipelineManager`, `ModelLoader.load_model` replaced
    so `_load_model_locked` reaches the real minimax_music3 branch without
    touching a checkpoint on disk."""
    from core import pipeline as pipeline_module

    manager = pipeline_module.DiffusionPipelineManager()
    monkeypatch.setattr(pipeline_module, "LAST_MODEL_CONFIG_FILE", tmp_path / "last_model.json")
    monkeypatch.setattr(
        pipeline_module.ModelLoader, "load_model",
        staticmethod(lambda **kwargs: {
            "type": "minimax_music3",
            "text_encoder_path": kwargs.get("text_encoder_file") or "/model/official/language_model",
        }))
    return manager


def test_the_chosen_text_encoder_is_persisted_and_replayed(music3_manager, monkeypatch):
    """The bug this pins: before the F2 fix, the music3 branch's
    `_save_last_model` call carried no encoder fields at all, so a restart
    silently rebuilt from `official/language_model` -- different weights and
    (for a pruned source) a different vocabulary view -- with no warning."""
    from core import pipeline as pipeline_module

    encoder = "/model/text_encoders/pruned_q8_0.gguf"
    music3_manager._load_model_locked(
        "diffusers", "/model/root", text_encoder_file=encoder)

    with open(pipeline_module.LAST_MODEL_CONFIG_FILE, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["text_encoder_file"] == encoder

    replayed = {}
    monkeypatch.setattr(music3_manager, "load_model", lambda **kwargs: replayed.update(kwargs))
    music3_manager._auto_load_last_model()
    assert replayed["text_encoder_file"] == encoder


def test_a_default_load_persists_no_text_encoder_file(music3_manager, monkeypatch):
    """The complement: a load with no override must not write a stale
    `text_encoder_file` that a later restart would incorrectly replay."""
    from core import pipeline as pipeline_module

    music3_manager._load_model_locked("diffusers", "/model/root")

    with open(pipeline_module.LAST_MODEL_CONFIG_FILE, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert "text_encoder_file" not in saved

    replayed = {}
    monkeypatch.setattr(music3_manager, "load_model", lambda **kwargs: replayed.update(kwargs))
    music3_manager._auto_load_last_model()
    assert replayed["text_encoder_file"] is None


# ---------------------------------------------------------------------------
# Preflight before teardown (audit F3). A REAL directory, so
# `ModelLoader.detect_model_type` genuinely resolves it as minimax_music3
# (not mocked) -- the point is proving the header-only refusal in
# `_load_model_locked` runs BEFORE `ModelLoader.load_model` is ever reached,
# so the live model survives a bad `text_encoder_file`.
# ---------------------------------------------------------------------------

def _minimal_music3_root(tmp_path):
    root = tmp_path / "music3_root"
    (root / "official").mkdir(parents=True)
    (root / "official" / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxMusic3ModularPipeline"}), encoding="utf-8")
    return root


def test_bad_text_encoder_file_is_refused_before_the_live_model_is_torn_down(tmp_path):
    from core import pipeline as pipeline_module

    root = _minimal_music3_root(tmp_path)
    manager = pipeline_module.DiffusionPipelineManager()
    manager.is_minimax_music3_model = True
    manager.current_model = f"diffusers:{root}"
    manager.current_model_info = {"source": str(root), "type": "minimax_music3"}
    manager.minimax_music3_components = {
        "text_encoder_path": str(root / "official" / "language_model"),
    }

    bad_path = tmp_path / "typo.safetensors"  # does not exist at all
    with pytest.raises(FileNotFoundError):
        manager._load_model_locked("diffusers", str(root), text_encoder_file=str(bad_path))

    # The live model must be untouched -- proof this ran BEFORE Step 1's
    # cleanup (which clears current_model/current_model_info first thing).
    assert manager.current_model == f"diffusers:{root}"
    assert manager.current_model_info == {"source": str(root), "type": "minimax_music3"}
    assert manager.is_minimax_music3_model is True


def test_wrong_extension_text_encoder_file_is_refused_before_teardown_too(tmp_path):
    """The same proof for a MiniMaxMusic3TextEncoderRefusal (content-based
    refusal), not just a FileNotFoundError."""
    from core import pipeline as pipeline_module

    root = _minimal_music3_root(tmp_path)
    manager = pipeline_module.DiffusionPipelineManager()
    manager.is_minimax_music3_model = True
    manager.current_model = f"diffusers:{root}"
    manager.current_model_info = {"source": str(root), "type": "minimax_music3"}
    manager.minimax_music3_components = {}

    bad_path = tmp_path / "encoder.bin"
    bad_path.write_bytes(b"not a real container")
    with pytest.raises(MiniMaxMusic3TextEncoderRefusal):
        manager._load_model_locked("diffusers", str(root), text_encoder_file=str(bad_path))

    assert manager.current_model == f"diffusers:{root}"
    assert manager.current_model_info == {"source": str(root), "type": "minimax_music3"}
