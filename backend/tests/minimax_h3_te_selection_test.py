"""MiniMax-H3: which text encoder file gets loaded, and why.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_te_selection_test.py -v

WHY THIS FILE EXISTS
--------------------
Three text encoder files are co-distributed for MiniMax-H3
(``qwen3vl_32b_minimax_h3_int8_convrot.safetensors``, ``..._bf16...``,
``..._nvfp4_awq...``), but ``_build_text_encoder`` has no quantized-Linear
swap at all: it calls ``refuse_quantized_state_dict`` /
``refuse_unsupported_quant_semantics`` unconditionally, so today it can only
install the bf16 one. A preference list that names the int8 file FIRST does
nothing on its own if the resolver just hands back whatever the list's first
entry is and lets ``_build_text_encoder`` blow up on it -- the fallback has to
be real, decided from what a candidate's HEADER actually declares, not from
its filename. This file pins that mechanism, entirely with synthetic
header-only ``.safetensors`` files (no real tensor bytes, no GPU, no large
files on disk), plus the ``te_override`` escape hatch and the glob fallback.

Only the SELECTION layer is covered here -- ``detect_minimax_h3_layout`` /
``_layout_from_root`` / ``_te_capability_accept`` -- not the actual quantized
decode path, which does not exist yet for this component.
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402

from core.models.minimax_h3 import loader  # noqa: E402
from core.models.minimax_h3.loader import (  # noqa: E402
    MINIMAX_H3_TE_PATTERNS,
    detect_minimax_h3_layout,
)


def _write_header(path: str, keys: dict) -> None:
    """A safetensors file carrying only the JSON header ``keys``, zero tensor bytes."""
    header = dict(keys)
    header["__metadata__"] = {"format": "pt"}
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)


def _write_fake_h3_dit(path: str) -> None:
    """Minimal header carrying the H3 DiT key signature (mirrors the load-dispatch test)."""
    _write_header(path, {
        "token_refiner.0.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 0]},
        "adaln_t_table": {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]},
    })


# A header with NO quantization evidence at all -- what a plain bf16 file (or,
# for these tests, a stand-in for "this build can currently read it") declares.
_PLAIN_TE_HEADER = {
    "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 0],
    },
}

# A header carrying positive quantization evidence via each of the three
# suffixes ``_te_capability_accept`` looks for. Any one is sufficient on its
# own; this uses all three so a predicate that only checks one still rejects.
_QUANTIZED_TE_HEADER = {
    "model.layers.0.self_attn.q_proj.weight": {
        "dtype": "I8", "shape": [8, 8], "data_offsets": [0, 0],
    },
    "model.layers.0.self_attn.q_proj.weight_scale": {
        "dtype": "F32", "shape": [8, 1], "data_offsets": [0, 0],
    },
    "model.layers.0.self_attn.q_proj.comfy_quant": {
        "dtype": "U8", "shape": [4], "data_offsets": [0, 0],
    },
}


def _build_root(tmp_path, te_files: dict) -> str:
    """A tree with a valid DiT + empty vae/ + the given ``{filename: header}`` TE files."""
    root = tmp_path / "minimax_h3"
    dit_dir = root / "diffusion_models"
    dit_dir.mkdir(parents=True)
    _write_fake_h3_dit(str(dit_dir / "minimax_h3_fl2va_pruned_bf16.safetensors"))
    (root / "vae").mkdir()
    te_dir = root / "text_encoders"
    te_dir.mkdir()
    for name, header in te_files.items():
        _write_header(str(te_dir / name), header)
    return str(root)


INT8_NAME, BF16_NAME, NVFP4_NAME = MINIMAX_H3_TE_PATTERNS


def test_preference_order_is_int8_then_bf16_then_nvfp4():
    """The list itself, unconditionally -- everything below depends on this order."""
    assert MINIMAX_H3_TE_PATTERNS == [
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "qwen3vl_32b_minimax_h3_bf16.safetensors",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ]


def test_capability_set_now_includes_int8_tensorwise_and_nvfp4():
    """Both the ConvRot INT8 and the NVFP4/AWQ TE decode paths exist now --
    the single named place says so."""
    assert loader.MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS == frozenset({"int8_tensorwise", "nvfp4"})


def test_capability_accept_rejects_quant_evidence_when_capability_set_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS", frozenset())
    quantized = tmp_path / "q.safetensors"
    _write_header(str(quantized), _QUANTIZED_TE_HEADER)
    assert loader._te_capability_accept(quantized) is False


def test_capability_accept_accepts_quant_evidence_with_the_real_default(tmp_path):
    """The header-only predicate cannot decode WHICH format a marker declares
    (that JSON lives in the tensor body); it only asks whether the capability
    set is non-empty. With the real (non-empty) default, header evidence of
    quantization is accepted -- the int8_convrot file's own marker is decoded
    for real, and refused if it does not match, inside `_build_text_encoder`."""
    quantized = tmp_path / "q.safetensors"
    _write_header(str(quantized), _QUANTIZED_TE_HEADER)
    assert loader._te_capability_accept(quantized) is True


def test_capability_accept_accepts_a_plain_header(tmp_path):
    plain = tmp_path / "p.safetensors"
    _write_header(str(plain), _PLAIN_TE_HEADER)
    assert loader._te_capability_accept(plain) is True


def test_capability_accept_reads_only_the_named_capability_set(tmp_path, monkeypatch):
    """The ONE place a later decode step extends is actually consulted here.

    A quantized-looking header that was rejected against the empty set is
    accepted the moment ``MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS`` is populated,
    with no other code path touched -- proving the predicate is wired to that
    set rather than hardcoding "always reject a marker".
    """
    quantized = tmp_path / "q.safetensors"
    _write_header(str(quantized), _QUANTIZED_TE_HEADER)
    monkeypatch.setattr(loader, "MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS", frozenset())
    assert loader._te_capability_accept(quantized) is False

    monkeypatch.setattr(loader, "MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS", frozenset({"int8_tensorwise"}))
    assert loader._te_capability_accept(quantized) is True


def test_preference_picks_int8_when_it_is_the_loadable_candidate(tmp_path):
    """When the preferred file's header is loadable, it wins -- not filename order alone."""
    root = _build_root(tmp_path, {
        INT8_NAME: _PLAIN_TE_HEADER,
        BF16_NAME: _PLAIN_TE_HEADER,
        NVFP4_NAME: _PLAIN_TE_HEADER,
    })
    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    assert layout["text_encoder"] == os.path.join(root, "text_encoders", INT8_NAME)
    assert layout["text_encoder_reason"] == "preferred"


def test_falls_through_to_bf16_when_int8_is_present_but_not_loadable(tmp_path, monkeypatch):
    """The general fallback mechanism, independent of today's real capability
    set: with NOTHING declared loadable, int8_convrot and nvfp4_awq both
    declare quantization this build cannot install; only bf16 survives."""
    monkeypatch.setattr(loader, "MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS", frozenset())
    root = _build_root(tmp_path, {
        INT8_NAME: _QUANTIZED_TE_HEADER,
        BF16_NAME: _PLAIN_TE_HEADER,
        NVFP4_NAME: _QUANTIZED_TE_HEADER,
    })
    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    assert layout["text_encoder"] == os.path.join(root, "text_encoders", BF16_NAME)
    reason = layout["text_encoder_reason"]
    assert "fell back past" in reason
    assert INT8_NAME in reason
    assert "not loadable by this build" in reason


def test_falls_through_to_nvfp4_only_when_the_other_two_are_absent(tmp_path):
    """Not merely rejected -- genuinely absent from disk."""
    root = _build_root(tmp_path, {NVFP4_NAME: _PLAIN_TE_HEADER})
    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    assert layout["text_encoder"] == os.path.join(root, "text_encoders", NVFP4_NAME)
    # Not "fell back past X" -- nothing more-preferred was ever on disk to skip.
    assert layout["text_encoder_reason"] == "preferred candidate(s) not present"


def test_no_loadable_candidate_leaves_text_encoder_unresolved(tmp_path, monkeypatch):
    """Every candidate present, every one rejected -- ``text_encoder`` is None, not a guess."""
    monkeypatch.setattr(loader, "MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS", frozenset())
    root = _build_root(tmp_path, {
        INT8_NAME: _QUANTIZED_TE_HEADER,
        BF16_NAME: _QUANTIZED_TE_HEADER,
        NVFP4_NAME: _QUANTIZED_TE_HEADER,
    })
    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    assert layout["text_encoder"] is None
    assert layout["text_encoder_reason"] == "no text encoder file found"


def test_glob_fallback_still_finds_an_unlisted_but_loadable_file(tmp_path):
    """A re-exported filename that matches no literal pattern, only the glob."""
    root = _build_root(tmp_path, {
        "qwen3vl_32b_minimax_h3_custom_reexport.safetensors": _PLAIN_TE_HEADER,
    })
    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    assert layout["text_encoder"] == os.path.join(
        root, "text_encoders", "qwen3vl_32b_minimax_h3_custom_reexport.safetensors")
    assert layout["text_encoder_reason"] == "resolved via glob fallback, no listed filename matched"


def test_te_override_wins_over_everything_including_a_rejecting_header(tmp_path):
    """An explicit override bypasses the preference list AND the predicate entirely."""
    root = _build_root(tmp_path, {
        BF16_NAME: _PLAIN_TE_HEADER,
    })
    override_dir = tmp_path / "elsewhere"
    override_dir.mkdir()
    override_path = override_dir / "hand_picked_int8.safetensors"
    _write_header(str(override_path), _QUANTIZED_TE_HEADER)  # would be REJECTED by the predicate

    layout = detect_minimax_h3_layout(root, te_override=str(override_path))
    assert layout is not None
    assert layout["text_encoder"] == str(override_path)
    assert layout["text_encoder_reason"] == "explicit override"
    # bf16 was on disk and would ordinarily win by default -- the override still wins.
    assert layout["text_encoder"] != os.path.join(root, "text_encoders", BF16_NAME)


def test_missing_override_path_errors_clearly(tmp_path):
    root = _build_root(tmp_path, {BF16_NAME: _PLAIN_TE_HEADER})
    bad_path = str(tmp_path / "does_not_exist.safetensors")

    with pytest.raises(FileNotFoundError, match=r"does_not_exist\.safetensors"):
        detect_minimax_h3_layout(root, te_override=bad_path)


def test_override_must_be_a_safetensors_file(tmp_path):
    """A real file that is not ``.safetensors`` is refused the same way, not silently used."""
    root = _build_root(tmp_path, {BF16_NAME: _PLAIN_TE_HEADER})
    wrong_ext = tmp_path / "not_safetensors.bin"
    wrong_ext.write_bytes(b"\x00")

    with pytest.raises(FileNotFoundError, match=r"not_safetensors\.bin"):
        detect_minimax_h3_layout(root, te_override=str(wrong_ext))


def test_missing_component_message_describes_the_search_not_one_filename(tmp_path):
    """Deliverable 4: the message must not assert only ``MINIMAX_H3_TE_PATTERNS[0]``."""
    root = _build_root(tmp_path, {})  # no text encoder candidate at all
    with pytest.raises(ValueError) as excinfo:
        loader.load_minimax_h3_from_path(root, load_text_encoder=True)
    message = str(excinfo.value)
    for pattern in MINIMAX_H3_TE_PATTERNS:
        assert pattern in message, f"{pattern} missing from: {message}"


def test_selection_log_names_the_file_and_the_reason(tmp_path, monkeypatch, capsys):
    """Deliverable 5: a user expecting int8 and getting bf16 can see why in the log."""
    official = tmp_path / "official"
    for component in ("text_encoder", "vae", "audio_vae"):
        directory = official / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")

    layout = {
        "root": str(tmp_path),
        "dit": str(tmp_path / "dit.safetensors"),
        "vae": str(tmp_path / "video_vae.safetensors"),
        "audio_vae": str(tmp_path / "audio_vae.safetensors"),
        "text_encoder": str(tmp_path / "text_encoders" / BF16_NAME),
        "text_encoder_reason": (
            f"fell back past {INT8_NAME} (present but not loadable by this build -- "
            f"see MINIMAX_H3_TE_LOADABLE_QUANT_FORMATS)"
        ),
        "official": str(official),
        "variant": "ref2va",
    }

    monkeypatch.setattr(loader, "detect_minimax_h3_layout", lambda _path: layout)
    monkeypatch.setattr(loader, "_build_text_encoder", lambda *_a: (object(), object()))
    monkeypatch.setattr(loader, "_build_transformer", lambda *_a: (object(), object()))
    monkeypatch.setattr(loader, "_build_video_vae", lambda *_a: (object(), {}))
    monkeypatch.setattr(loader, "_build_audio_vae", lambda *_a: (object(), {}))
    monkeypatch.setattr(loader, "_load_tokenizer_and_processor", lambda *_a: (None, None))
    monkeypatch.setattr(loader, "_load_schedulers", lambda *_a: (object(), object()))
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    loader.load_minimax_h3_from_path(layout["dit"])

    out = capsys.readouterr().out
    assert layout["text_encoder"] in out
    assert "fell back past" in out
    assert INT8_NAME in out
