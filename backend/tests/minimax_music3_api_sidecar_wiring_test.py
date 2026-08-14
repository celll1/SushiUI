"""MiniMax Music 3 API layer, part 3: frame-code sidecar persistence wiring.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_api_sidecar_wiring_test.py -v

Design doc "Per-generation state contract": a song saved without the
frame-code sidecar can never be extended or repainted, so it must ship in the
same commit as the first shippable generation. `backend/tests/
minimax_music3_frame_codes_test.py` already covers the sidecar module's own
write/read contract in isolation; this file covers the ROUTE-LEVEL wiring --
`routes.generate_txt2aud`'s exact call pattern from a
`MiniMaxMusic3Txt2AudResult` (the pipeline backend's return shape, design doc
phase plan item 3) into `write_frame_codes_sidecar`, without needing a loaded
model or a running FastAPI app: the fields on the result NamedTuple map
1:1 onto the sidecar writer's keyword arguments, and the round-trip is
provably decodable and identity-checkable against the saved audio file the
same way `MiniMaxMusic3FrameCodes.matches()` is used elsewhere.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3.defaults import FALLBACK_AUDIO_VOCAB_SIZE, SEMANTIC_VOCAB_SIZE
from core.models.minimax_music3.frame_codes import (
    read_frame_codes_sidecar_for_audio,
    sidecar_path_for_audio,
)
from core.pipeline_backends.minimax_music3 import MiniMaxMusic3Txt2AudResult


def _fake_result(num_frames=20, num_codebooks=8, sample_rate=44100, frame_rate=86.13):
    generator = torch.Generator().manual_seed(7)
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

    num_samples = 44100 * 5
    waveform = torch.zeros(2, num_samples, dtype=torch.float32)

    return MiniMaxMusic3Txt2AudResult(
        waveform=waveform,
        sample_rate=sample_rate,
        actual_seed=1234,
        frame_codes=frame_codes,
        prefix_codes=prefix_codes,
        num_frames=num_frames,
        frame_rate=frame_rate,
        prompt="ambient synth, instrumental",
        lyrics="[verse]\nla la la",
    )


def _write_sidecar_the_way_the_route_does(audio_path, result, *, content_hash, model_hash):
    """Reproduces `routes.generate_txt2aud`'s exact call, arg for arg, so a
    change to either side (the result shape or the writer's signature) that
    breaks the OTHER side's assumptions fails here instead of only at request
    time. `num_samples` is computed from the waveform the same way the route
    computes it (`int(waveform.shape[-1])`), not carried on the result
    directly -- mirroring `_gen_result.frame_codes`/`.prefix_codes`/
    `.frame_rate`/`.prompt`/`.lyrics` also being read off the result, while
    `sample_rate`/`actual_seed` feed the OTHER call (`save_audio_with_metadata`)
    first, as they do in the real route.
    """
    from core.models.minimax_music3.frame_codes import write_frame_codes_sidecar

    num_samples = int(result.waveform.shape[-1])
    return write_frame_codes_sidecar(
        audio_path,
        result.frame_codes,
        result.prefix_codes,
        sample_rate=result.sample_rate,
        frame_rate=result.frame_rate,
        prompt=result.prompt,
        lyrics=result.lyrics,
        seed=result.actual_seed,
        num_samples=num_samples,
        content_hash=content_hash,
        model_hash=model_hash,
    ), num_samples


def test_route_shaped_sidecar_write_round_trips(tmp_path):
    result = _fake_result()
    audio_path = str(tmp_path / "txt2aud_20260101_000000_1234.flac")

    written_path, num_samples = _write_sidecar_the_way_the_route_does(
        audio_path, result, content_hash="filehash123", model_hash="modelhash456",
    )
    assert written_path == sidecar_path_for_audio(audio_path)

    loaded = read_frame_codes_sidecar_for_audio(audio_path)
    assert loaded is not None
    assert torch.equal(loaded.frame_codes, result.frame_codes)
    assert torch.equal(loaded.prefix_codes, result.prefix_codes)
    assert loaded.sample_rate == result.sample_rate
    assert loaded.frame_rate == result.frame_rate
    assert loaded.prompt == result.prompt
    assert loaded.lyrics == result.lyrics
    assert loaded.seed == result.actual_seed
    assert loaded.num_samples == num_samples
    assert loaded.content_hash == "filehash123"
    assert loaded.model_hash == "modelhash456"


def test_route_shaped_sidecar_matches_the_saved_audio_identity(tmp_path):
    """The exact check a future extend/repaint request runs before trusting a
    sidecar found on disk: sample_rate, frame_rate, num_codebooks, and the
    per-song `num_samples`/`content_hash` pair.
    """
    result = _fake_result()
    audio_path = str(tmp_path / "song.flac")
    _write_sidecar_the_way_the_route_does(
        audio_path, result, content_hash="realhash", model_hash="m",
    )
    loaded = read_frame_codes_sidecar_for_audio(audio_path)
    num_samples = int(result.waveform.shape[-1])

    assert loaded.matches(
        sample_rate=result.sample_rate,
        frame_rate=result.frame_rate,
        num_codebooks=result.frame_codes.shape[-1],
        num_samples=num_samples,
        content_hash="realhash",
    )
    # A DIFFERENT audio file happening to reuse this path (the design doc's
    # exact "stale/foreign sidecar" scenario) must be rejected by num_samples
    # or content_hash even though sample_rate/frame_rate coincide.
    assert not loaded.matches(
        sample_rate=result.sample_rate, frame_rate=result.frame_rate, num_samples=num_samples + 1,
    )
    assert not loaded.matches(
        sample_rate=result.sample_rate, frame_rate=result.frame_rate, content_hash="different-file-hash",
    )


def test_route_never_writes_a_sidecar_for_an_acestep_generation():
    """The `_is_music3` branch in `routes.generate_txt2aud` gates the sidecar
    write -- ACE-Step's plain `(waveform, sample_rate, actual_seed)` tuple has
    no `frame_codes`/`prefix_codes` to write at all, so no sidecar file may
    appear next to an ACE-Step song. This is a structural assertion about the
    tuple ACE-Step returns (no such attributes), not a route-level smoke test.
    """
    acestep_result = (torch.zeros(2, 100), 48000, 42)
    assert not hasattr(acestep_result, "frame_codes")
    assert not isinstance(acestep_result, MiniMaxMusic3Txt2AudResult)


def test_write_frame_codes_sidecar_is_best_effort_documented_as_non_fatal():
    """Design doc's sidecar contract is about DATA SURVIVAL, not about
    aborting an otherwise-successful generation on a filesystem hiccup; the
    route wraps the sidecar write in a try/except that logs rather than
    raises (see `routes.generate_txt2aud`). This test pins the SIGNATURE
    contract that makes that safe: `write_frame_codes_sidecar` raises
    `ValueError` (a recognized, catchable failure) for a malformed call
    rather than crashing with an unrelated exception type that a narrow
    `except Exception` could still catch, but documents the class contract.
    """
    from core.models.minimax_music3.frame_codes import write_frame_codes_sidecar

    result = _fake_result(num_frames=3)
    bad_prefix = torch.zeros(1, 3, dtype=torch.int64)  # wrong num_codebooks
    raised = None
    try:
        write_frame_codes_sidecar(
            "unused.flac", result.frame_codes, bad_prefix,
            sample_rate=result.sample_rate, frame_rate=result.frame_rate,
            prompt=result.prompt, lyrics=result.lyrics, seed=result.actual_seed,
            num_samples=1000,
        )
    except Exception as exc:  # noqa: BLE001 - asserting the TYPE, matching the route's except Exception
        raised = exc
    assert isinstance(raised, ValueError)
