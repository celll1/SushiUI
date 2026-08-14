"""MiniMax Music 3 frame-code sidecar: write/read round-trip (weight-free).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_frame_codes_test.py -v

Covers the dtype contract the module docstring calls out specifically: the
on-disk representation is int16, but every value this module RETURNS must be
int64 -- a previous round of this work shipped a crash where a compact
sidecar dtype reached an `nn.Embedding` lookup un-upcast
("Expected tensor for argument #1 'indices' to have ... Long, Int; but got
torch.ShortTensor"). `test_read_result_feeds_an_nn_embedding_lookup` recreates
that exact failure mode directly (a real `nn.Embedding` call, no full model)
so a regression here fails loudly instead of merely differing in dtype.

Also covers the per-column code-range validation (a value that fits int16 but
is outside a real code's legal range must still be refused, on both write and
read) and the audio-file identity check (`num_samples`/`content_hash`) that
lets `matches()` detect a stale/foreign sidecar next to the wrong audio file.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3.defaults import FALLBACK_AUDIO_VOCAB_SIZE, SEMANTIC_VOCAB_SIZE
from core.models.minimax_music3.frame_codes import (
    FRAME_CODES_FORMAT_VERSION,
    RESIDUAL_CODE_MAX,
    SEMANTIC_CODE_MAX,
    MiniMaxMusic3FrameCodes,
    read_frame_codes_sidecar,
    read_frame_codes_sidecar_for_audio,
    sidecar_path_for_audio,
    write_frame_codes_sidecar,
)


def _sample_codes(num_frames=37, num_codebooks=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    # Column 0 (semantic) in [0, SEMANTIC_VOCAB_SIZE); columns 1.. (residual)
    # in [0, FALLBACK_AUDIO_VOCAB_SIZE) -- the ACTUAL legal per-column ranges
    # (design doc), not a single flat range across every column.
    semantic = torch.randint(0, SEMANTIC_VOCAB_SIZE, (num_frames, 1), generator=generator, dtype=torch.int64)
    residual = torch.randint(
        0, FALLBACK_AUDIO_VOCAB_SIZE, (num_frames, num_codebooks - 1), generator=generator, dtype=torch.int64
    )
    frame_codes = torch.cat([semantic, residual], dim=-1)

    prefix_semantic = torch.randint(0, SEMANTIC_VOCAB_SIZE, (1, 1), generator=generator, dtype=torch.int64)
    prefix_residual = torch.randint(
        0, FALLBACK_AUDIO_VOCAB_SIZE, (1, num_codebooks - 1), generator=generator, dtype=torch.int64
    )
    prefix_codes = torch.cat([prefix_semantic, prefix_residual], dim=-1)
    return frame_codes, prefix_codes


def test_round_trip_preserves_values_shape_and_dtype(tmp_path):
    frame_codes, prefix_codes = _sample_codes()
    audio_path = str(tmp_path / "song.flac")

    written_path = write_frame_codes_sidecar(
        audio_path, frame_codes, prefix_codes,
        sample_rate=44100, frame_rate=25.0, prompt="ambient synth", lyrics="[verse]\nhello",
        seed=1234, num_samples=441000, content_hash="abc123", model_hash="deadbeef",
    )
    assert written_path == sidecar_path_for_audio(audio_path)
    assert os.path.isfile(written_path)

    result = read_frame_codes_sidecar(written_path)
    assert isinstance(result, MiniMaxMusic3FrameCodes)
    assert result.format_version == FRAME_CODES_FORMAT_VERSION
    assert result.frame_codes.dtype == torch.int64
    assert result.prefix_codes.dtype == torch.int64
    assert torch.equal(result.frame_codes, frame_codes)
    assert torch.equal(result.prefix_codes, prefix_codes)
    assert result.sample_rate == 44100
    assert result.frame_rate == 25.0
    assert result.prompt == "ambient synth"
    assert result.lyrics == "[verse]\nhello"
    assert result.seed == 1234
    assert result.model_hash == "deadbeef"
    assert result.num_samples == 441000
    assert result.content_hash == "abc123"
    assert result.num_frames == frame_codes.shape[0]
    assert result.num_codebooks == frame_codes.shape[1]


def test_round_trip_from_an_int16_source_tensor_is_lossless(tmp_path):
    # A caller that already narrowed to int16 before calling (the module
    # docstring says this is accepted) must round-trip identically to one
    # that passed int64.
    frame_codes64, prefix_codes64 = _sample_codes(num_frames=12, seed=1)
    frame_codes16 = frame_codes64.to(torch.int16)
    prefix_codes16 = prefix_codes64.to(torch.int16)
    audio_path = str(tmp_path / "song2.flac")

    write_frame_codes_sidecar(
        audio_path, frame_codes16, prefix_codes16,
        sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="[verse]\nl", seed=1, num_samples=1000,
    )
    result = read_frame_codes_sidecar_for_audio(audio_path)
    assert result is not None
    assert torch.equal(result.frame_codes, frame_codes64)
    assert torch.equal(result.prefix_codes, prefix_codes64)


def test_read_result_feeds_an_nn_embedding_lookup(tmp_path):
    """Reproduces the exact failure mode a bare int16 sidecar hits against
    `nn.Embedding` -- see module docstring. Must NOT raise."""
    frame_codes, prefix_codes = _sample_codes(num_frames=5, num_codebooks=8)
    audio_path = str(tmp_path / "song3.flac")
    write_frame_codes_sidecar(
        audio_path, frame_codes, prefix_codes,
        sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="[verse]\nl", seed=1, num_samples=1000,
    )
    result = read_frame_codes_sidecar_for_audio(audio_path)

    embedding = torch.nn.Embedding(SEMANTIC_VOCAB_SIZE, 4)
    embedded = embedding(result.frame_codes)  # would raise on a ShortTensor
    assert embedded.shape == (result.num_frames, result.num_codebooks, 4)


def test_missing_sidecar_returns_none_not_an_exception(tmp_path):
    audio_path = str(tmp_path / "no_sidecar.flac")
    assert read_frame_codes_sidecar_for_audio(audio_path) is None


def test_locate_next_to_audio_strips_the_audio_extension(tmp_path):
    audio_path = str(tmp_path / "base_name.flac")
    expected = str(tmp_path / "base_name.mm3frames.json")
    assert sidecar_path_for_audio(audio_path) == expected


def test_write_refuses_a_shape_mismatch(tmp_path):
    frame_codes, _ = _sample_codes(num_frames=4, num_codebooks=8)
    bad_prefix = torch.zeros(1, 7, dtype=torch.int64)  # wrong num_codebooks
    with pytest.raises(ValueError):
        write_frame_codes_sidecar(
            str(tmp_path / "x.flac"), frame_codes, bad_prefix,
            sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
        )


def test_write_refuses_a_value_outside_the_int16_range(tmp_path):
    frame_codes = torch.zeros(2, 8, dtype=torch.int64)
    frame_codes[0, 0] = 40000  # over int16 ceiling
    prefix_codes = torch.zeros(1, 8, dtype=torch.int64)
    with pytest.raises(ValueError):
        write_frame_codes_sidecar(
            str(tmp_path / "y.flac"), frame_codes, prefix_codes,
            sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
        )


def test_write_refuses_a_negative_value(tmp_path):
    frame_codes = torch.zeros(2, 8, dtype=torch.int64)
    frame_codes[1, 3] = -1
    prefix_codes = torch.zeros(1, 8, dtype=torch.int64)
    with pytest.raises(ValueError):
        write_frame_codes_sidecar(
            str(tmp_path / "z.flac"), frame_codes, prefix_codes,
            sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
        )


def test_write_refuses_a_semantic_value_that_fits_int16_but_exceeds_its_column_range(tmp_path):
    # F1: the exact case the audit demonstrated -- 20000 fits comfortably in
    # int16 (max 32767) but is not a legal semantic code (max SEMANTIC_CODE_MAX
    # = 16383); AUDIO_CODE_OFFSET + 20000 is still a VALID text-token index, so
    # this must be caught by the per-column check, not the int16-overflow one.
    assert SEMANTIC_CODE_MAX < 20000 < 32767
    frame_codes = torch.zeros(2, 8, dtype=torch.int64)
    frame_codes[0, 0] = 20000
    prefix_codes = torch.zeros(1, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="semantic code"):
        write_frame_codes_sidecar(
            str(tmp_path / "semantic_oob.flac"), frame_codes, prefix_codes,
            sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
        )


def test_write_refuses_a_residual_value_that_fits_int16_but_exceeds_its_column_range(tmp_path):
    assert RESIDUAL_CODE_MAX < 5000 < 32767
    frame_codes = torch.zeros(2, 8, dtype=torch.int64)
    frame_codes[1, 3] = 5000  # legal for a semantic code, not for a residual one
    prefix_codes = torch.zeros(1, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="residual code"):
        write_frame_codes_sidecar(
            str(tmp_path / "residual_oob.flac"), frame_codes, prefix_codes,
            sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
        )


def test_write_refuses_a_float_tensor(tmp_path):
    frame_codes = torch.zeros(2, 8, dtype=torch.float32)
    prefix_codes = torch.zeros(1, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="integer-dtype"):
        write_frame_codes_sidecar(
            str(tmp_path / "float.flac"), frame_codes, prefix_codes,
            sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
        )


def test_read_refuses_an_out_of_range_value_written_by_hand(tmp_path):
    # A sidecar that passed write-time validation, then had its JSON edited
    # by hand to introduce an illegal value, must still be refused on READ --
    # this is the half of F1 that "validate on write only" cannot catch.
    frame_codes, prefix_codes = _sample_codes(num_frames=3)
    audio_path = str(tmp_path / "hand_edited.flac")
    path = write_frame_codes_sidecar(
        audio_path, frame_codes, prefix_codes,
        sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
    )
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    import base64
    import numpy as np

    corrupted = frame_codes.clone()
    corrupted[0, 0] = 20000  # legal int16, illegal semantic code
    payload["frame_codes_b64"] = base64.b64encode(corrupted.numpy().astype(np.int16).tobytes()).decode("ascii")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    with pytest.raises(ValueError, match="semantic code"):
        read_frame_codes_sidecar(path)


def test_read_refuses_an_unrecognized_format_version(tmp_path):
    frame_codes, prefix_codes = _sample_codes(num_frames=3)
    audio_path = str(tmp_path / "w.flac")
    path = write_frame_codes_sidecar(
        audio_path, frame_codes, prefix_codes,
        sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
    )
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["format_version"] = 999
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    with pytest.raises(ValueError):
        read_frame_codes_sidecar(path)


def test_read_raises_valueerror_not_keyerror_on_a_missing_field(tmp_path):
    # F7: the docstring promises ValueError for every recognized failure
    # mode; a missing field must not leak a raw KeyError.
    frame_codes, prefix_codes = _sample_codes(num_frames=3)
    audio_path = str(tmp_path / "missing_field.flac")
    path = write_frame_codes_sidecar(
        audio_path, frame_codes, prefix_codes,
        sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=100,
    )
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    del payload["frame_codes_b64"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    with pytest.raises(ValueError):
        read_frame_codes_sidecar(path)


def test_read_raises_valueerror_not_jsondecodeerror_on_truncated_json(tmp_path):
    path = str(tmp_path / "truncated.mm3frames.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"format_version": 1, "frame_codes_b64": "AA')  # truncated, invalid JSON

    with pytest.raises(ValueError):
        read_frame_codes_sidecar(path)


def test_matches_predicate():
    frame_codes, prefix_codes = _sample_codes(num_frames=6, num_codebooks=8)
    result = MiniMaxMusic3FrameCodes(
        frame_codes=frame_codes, prefix_codes=prefix_codes,
        sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0,
        num_samples=1000, content_hash="hash1", model_hash="abc",
    )
    assert result.matches(
        sample_rate=44100, frame_rate=25.0, num_codebooks=8, model_hash="abc",
        num_samples=1000, content_hash="hash1",
    )
    assert not result.matches(sample_rate=48000)
    assert not result.matches(frame_rate=30.0)
    assert not result.matches(num_codebooks=4)
    assert not result.matches(model_hash="different")
    # An empty stored model_hash never blocks a match (older sidecar / no model loaded at write time).
    result_no_hash = MiniMaxMusic3FrameCodes(
        frame_codes=frame_codes, prefix_codes=prefix_codes,
        sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0, num_samples=1000,
    )
    assert result_no_hash.matches(model_hash="anything")


def test_matches_rejects_a_stale_or_foreign_sidecar_by_num_samples(tmp_path):
    # F4: sample_rate/frame_rate/model_hash are all checkpoint-level
    # constants -- identical for every song from the same model -- so none of
    # them can tell "this sidecar belongs to a DIFFERENT audio file at this
    # same path" apart from "this sidecar belongs to the file it was written
    # for". num_samples is per-song and must catch it.
    frame_codes, prefix_codes = _sample_codes(num_frames=4)
    original = MiniMaxMusic3FrameCodes(
        frame_codes=frame_codes, prefix_codes=prefix_codes,
        sample_rate=44100, frame_rate=25.0, prompt="p", lyrics="l", seed=0,
        num_samples=441000, content_hash="original-hash",
    )
    # Same checkpoint-level properties, but the CURRENT audio file at this
    # path (recomputed by a caller, as `matches()`'s docstring specifies) has
    # a different sample count -- e.g. the path was regenerated with a
    # shorter song and the caller forgot (or a bug skipped) rewriting the
    # sidecar to match.
    assert not original.matches(sample_rate=44100, frame_rate=25.0, num_samples=200000)
    # And by content_hash, even if num_samples happened to coincide.
    assert not original.matches(sample_rate=44100, frame_rate=25.0, content_hash="different-hash")
    # The genuinely-matching case still passes both checks.
    assert original.matches(sample_rate=44100, frame_rate=25.0, num_samples=441000, content_hash="original-hash")
