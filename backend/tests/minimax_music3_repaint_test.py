"""MiniMax Music 3 aud2aud repaint -- design doc phase plan item 8's BACKEND mechanism.

`MiniMaxMusic3Mixin._generate_aud2aud_minimax_music3` (`core.pipeline_backends.minimax_music3`) dispatches
`mode="repaint"` to one of two sub-mechanisms:

  * `_minimax_music3_repaint_regenerate` -- AR-resume with a NEW tail from a chunk-window-snapped point `T`
    onward, discarding the song's own codes after `T`.
  * `_minimax_music3_repaint_rerender` -- keeps the codes, recovers `frame_hiddens` for a chunk-window-snapped
    range via `MiniMaxMusic3Pipeline.recover_frame_hiddens` (teacher-forced, no sampling), and redraws only that
    range's flow-stage rendering with a new seed.

This file pins:

  * both sub-modes' preservation property is SAMPLE-EXACT outside the region each one changes -- computed via
    `compute_cumulative_samples`, the pure geometry this module also exports, and checked against a REAL
    (tiny, synthetic) pipeline's actual `denoise_chunks`/`decode` output, not merely asserted analytically;
  * `mode="cover"` is refused for MiniMax Music 3 with the RVQ-tokenizer-encoder capability reason (not the
    ACE-Step-only mechanism, and not a generic "not implemented" message);
  * `music3_repaint_mode="infill"`/an unsupported value is refused, `"infill"` specifically with the causal-LM
    reason (mid-span infill with a preserved tail is not offered by either sub-mode);
  * a missing/foreign sidecar is refused for both sub-modes (same identity-validation mechanism extend already
    proved in `minimax_music3_extend_test.py`, exercised again here through the two NEW call paths);
  * the result's sidecar (written by a caller, e.g. `routes.py`) round-trips the CORRECT code sequence for each
    sub-mode -- `frame_codes[:T]` + new tail for "regenerate", UNCHANGED codes for "rerender";
  * the budget guard (`check_ar_resume_budget`) fires BEFORE either GPU-staging move for "regenerate" mode.

Uses the SAME real geometry constants (`CHUNK_FRAMES=200`, `CHUNK_HOP=100`, `CROP_LEFT_LATENT=86`,
`CROP_RIGHT_LATENT=258`) production uses -- not shrunk for test speed -- because repaint's whole preservation
property depends on that EXACT arithmetic; a shrunk-constant fixture would prove nothing about the real crop
ratios. To keep this fast without a real checkpoint, the "original song" is built directly from RANDOM
`frame_hiddens` fed through the REAL (tiny) `denoise_chunks`/`decode` (fast: no sequential per-frame loop, matches
`minimax_music3_chunk_geometry_test.py`'s approach) with HANDCRAFTED (all-zero, valid-range) frame codes written to
the sidecar; recovering/resuming from those codes still exercises the REAL `recover_frame_hiddens`/`generate_ar`
resume machinery (with its own batched-replay chunking) against a REAL tiny `Qwen3ForCausalLM`, just decoupled from
"do these codes match this audio" -- which is `minimax_music3_ar_resume_test.py`/`minimax_music3_extend_test.py`'s
job, not this file's. This file is about GEOMETRY (does the splice land on the right sample), not AR fidelity.
"""

import os
import sys

import pytest
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3Config, Qwen3ForCausalLM

from core.models.minimax_music3.defaults import (
    AUDIO_CODE_OFFSET, CHUNK_FRAMES, CHUNK_HOP, CROP_LEFT_LATENT, CROP_RIGHT_LATENT, SEMANTIC_VOCAB_SIZE,
)
from core.models.minimax_music3.frame_codes import write_frame_codes_sidecar
from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline
from core.models.minimax_music3.vendor import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)
from core.pipeline_backends.minimax_music3 import (
    MiniMaxMusic3Mixin,
    MiniMaxMusic3RepaintResult,
    compute_cumulative_samples,
    prepare_chunk_starts,
)

_HIDDEN = 16


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}


def _build_tiny_pipeline(max_position_embeddings: int = 8192) -> MiniMaxMusic3Pipeline:
    torch.manual_seed(1234)
    vocab = AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE + 1

    lm_config = Qwen3Config(
        vocab_size=vocab, hidden_size=_HIDDEN, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=max_position_embeddings, tie_word_embeddings=False,
    )
    language_model = Qwen3ForCausalLM(lm_config).eval()

    rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder(
        hidden_size=_HIDDEN, num_layers=2, num_attention_heads=2, intermediate_size=32,
        audio_vocab_size=17, num_codebooks=8, max_position_embeddings=16,
    ).eval()
    # Real production ratio (44100/24000 * 960/512 ~= 3.4453125) and real hop_length (512) -- the crop constants
    # (86/258 latents) are only meaningful against this ratio; see module docstring.
    condition_encoder = MiniMaxMusic3ConditionEncoder(
        condition_hidden_dim=_HIDDEN, num_condition_layers=8, out_dim=8,
        input_sampling_rate=24000, input_hop_length=960, output_sampling_rate=44100, output_hop_length=512,
    ).eval()
    transformer = MiniMaxMusic3Transformer1DModel(
        in_channels=4, condition_dim=8, num_layers=1, num_attention_heads=1,
        attention_head_dim=4, ff_inner_dim=8, rotary_dim=4, fourier_embedding_dim=8,
    ).eval()
    # upsampling_ratios must multiply to condition_encoder's output_hop_length (512) above -- see
    # minimax_music3_chunk_geometry_test.py's identical invariant comment. Using the REAL (8, 8, 4, 2) product
    # (not a shrunk one) keeps the crop constants' (86/258 latent frames) real MEANING intact -- module docstring.
    # decoder_hidden_dim halves once per upsampling stage (4 stages here), so it must be >= 16 to stay >= 1
    # channel throughout.
    vocoder = MiniMaxMusic3Vocoder(
        latent_channels=4, decoder_input_dim=4, decoder_hidden_dim=32, upsampling_ratios=(8, 8, 4, 2),
        sampling_rate=44100,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(invert_sigmas=True)

    return MiniMaxMusic3Pipeline(
        tokenizer=_FakeTokenizer(), language_model=language_model, rvq_depth_decoder=rvq_depth_decoder,
        condition_encoder=condition_encoder, transformer=transformer, scheduler=scheduler, vocoder=vocoder,
        execution_device=torch.device("cpu"),
    )


class _Manager(MiniMaxMusic3Mixin):
    """Minimal `DiffusionPipelineManager` stand-in -- same pattern as `minimax_music3_extend_test.py`."""

    def __init__(self, pipeline: MiniMaxMusic3Pipeline, device="cpu"):
        self.minimax_music3_components = {
            "tokenizer": pipeline.tokenizer,
            "language_model": pipeline.language_model,
            "rvq_depth_decoder": pipeline.rvq_depth_decoder,
            "condition_encoder": pipeline.condition_encoder,
            "transformer": pipeline.transformer,
            "scheduler": pipeline.scheduler,
            "vocoder": pipeline.vocoder,
        }
        self.is_minimax_music3_model = True
        self.device = device
        self.current_model_info = {}


def _write_float_wav(path: str, wave: torch.Tensor, sample_rate: int) -> None:
    data = wave.detach().cpu().numpy().T
    sf.write(path, data, sample_rate, subtype="FLOAT")


def _make_synthetic_multi_chunk_song(tmp_path, pipeline: MiniMaxMusic3Pipeline, num_frames: int, seed: int,
                                     prompt="a caption", lyrics="[verse]\nhello world"):
    """Builds a real multi-flow-chunk "song" WITHOUT running the (slow, sequential) autoregressive stage:
    `frame_hiddens` is random, matching production shape/dtype, fed through the REAL `denoise_chunks`/`decode`
    (fast -- no per-frame Python loop). `frame_codes`/`prefix_codes` are handcrafted (all-zero, valid range) --
    see module docstring for why this is sufficient for a GEOMETRY test.

    Returns `(wav_path, frame_codes, prefix_codes, waveform)`.
    """
    torch.manual_seed(seed)
    num_codebooks = pipeline.num_codebooks
    frame_hiddens = torch.randn(1, num_frames, num_codebooks * _HIDDEN)

    generator = torch.Generator().manual_seed(seed)
    latent_chunks = pipeline.denoise_chunks(
        frame_hiddens, num_inference_steps=2, flow_guidance_scale=1.7, generator=generator,
    )
    waveform = pipeline.decode(latent_chunks, output_type="pt")[0]  # [2, samples]

    frame_codes = torch.zeros(num_frames, num_codebooks, dtype=torch.long)
    prefix_codes = torch.zeros(1, num_codebooks, dtype=torch.long)

    wav_path = str(tmp_path / "song.wav")
    _write_float_wav(wav_path, waveform, int(pipeline.sampling_rate))
    from utils.image_utils import calculate_file_hash
    content_hash = calculate_file_hash(wav_path)
    write_frame_codes_sidecar(
        wav_path, frame_codes, prefix_codes,
        sample_rate=int(pipeline.sampling_rate), frame_rate=float(pipeline.frame_rate),
        prompt=prompt, lyrics=lyrics, seed=seed, num_samples=int(waveform.shape[-1]),
        content_hash=content_hash,
    )
    return wav_path, frame_codes, prefix_codes, waveform


def _geometry_hop_length(pipeline: MiniMaxMusic3Pipeline) -> int:
    return int(pipeline.latent_hop_length)


def _cumulative_for(pipeline: MiniMaxMusic3Pipeline, num_frames: int):
    ce = pipeline.condition_encoder.config
    return compute_cumulative_samples(
        num_frames, _geometry_hop_length(pipeline), CHUNK_FRAMES, CHUNK_HOP, CROP_LEFT_LATENT, CROP_RIGHT_LATENT,
        ce.input_sampling_rate, ce.input_hop_length, ce.output_sampling_rate, ce.output_hop_length,
    )


# ---------------------------------------------------------------------------
# Pure geometry self-check: the cumulative table's final entry must equal the
# actual decoded sample count for the SAME song (proves the geometry
# functions genuinely mirror decode(), not just an independent guess).
# ---------------------------------------------------------------------------
def test_cumulative_samples_matches_a_real_multi_chunk_decode(tmp_path):
    pipeline = _build_tiny_pipeline()
    num_frames = 350  # 3 chunks: starts = [0, 100, 200]
    assert prepare_chunk_starts(num_frames, CHUNK_FRAMES, CHUNK_HOP) == [0, 100, 200]

    _wav_path, _codes, _prefix, waveform = _make_synthetic_multi_chunk_song(tmp_path, pipeline, num_frames, seed=0)
    cumulative = _cumulative_for(pipeline, num_frames)
    assert len(cumulative) == 4  # cumulative[0..3] for 3 chunks
    assert cumulative[-1] == int(waveform.shape[-1])
    assert cumulative[0] == 0
    assert cumulative[1] < cumulative[2] < cumulative[3]


# ---------------------------------------------------------------------------
# "rerender" mode: an INTERNAL window's boundaries are sample-exact on BOTH
# sides, and the codes never change.
# ---------------------------------------------------------------------------
def test_rerender_preserves_both_boundaries_sample_exact(tmp_path):
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)
    num_frames = 350  # chunk_starts = [0, 100, 200] -> chunk 1 (frames [100, 300)) is fully INTERNAL
    wav_path, sidecar_codes, sidecar_prefix, original_wave = _make_synthetic_multi_chunk_song(
        tmp_path, pipeline, num_frames, seed=0,
    )
    cumulative = _cumulative_for(pipeline, num_frames)
    start_sample, end_sample = cumulative[1], cumulative[2]
    assert 0 < start_sample < end_sample < int(original_wave.shape[-1])

    result = manager._generate_aud2aud_minimax_music3(
        {
            "mode": "repaint", "music3_repaint_mode": "rerender",
            "repaint_start": 100 / pipeline.frame_rate, "repaint_end": 200 / pipeline.frame_rate,
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 999,
        },
        wav_path,
    )
    assert isinstance(result, MiniMaxMusic3RepaintResult)
    assert result.repaint_mode == "rerender"

    # Codes are UNCHANGED -- the whole point of this mode.
    assert torch.equal(result.frame_codes, sidecar_codes)
    assert torch.equal(result.prefix_codes, sidecar_prefix)
    assert result.num_frames == num_frames

    # Both boundaries preserved sample-exact; only the middle span may differ.
    assert torch.equal(result.waveform[..., :start_sample], original_wave[..., :start_sample])
    tail_len = int(original_wave.shape[-1]) - end_sample
    assert torch.equal(result.waveform[..., result.waveform.shape[-1] - tail_len:], original_wave[..., end_sample:])
    assert result.waveform.shape[-1] == original_wave.shape[-1]
    assert torch.isfinite(result.waveform).all()


def test_rerender_at_the_true_song_edges_has_no_declick_ramp_applied():
    """Pure splice-helper check: a range touching sample 0 has no LEFT boundary to declick against, and a range
    touching the file's end has no RIGHT boundary -- mirrors `_minimax_music3_apply_extend_waveform_splice`'s
    identical "no reference on the far side" rule for extend's own single boundary. Each case below leaves the
    OTHER boundary also at a true edge (both `start_sample == 0` and `end_sample == total`), so NEITHER ramp
    applies and the whole new-middle segment must survive completely unmodified -- the cleanest possible check.
    """
    original = torch.linspace(-0.5, 0.5, 800).reshape(2, 400)
    new_middle = torch.full((2, 50), 0.9)

    spliced = MiniMaxMusic3Mixin._minimax_music3_apply_rerender_waveform_splice(
        original, new_middle, start_sample=0, end_sample=400, sample_rate=1000, crossfade_ms=10.0,
    )
    assert spliced.shape[-1] == 50
    assert torch.equal(spliced, new_middle)  # no ramp applied at all -- both sides are true song edges


def test_rerender_declick_ramp_only_touches_the_boundary_that_actually_has_preserved_audio():
    """An INTERNAL left boundary (start_sample > 0) ramps the leading edge; a true right edge (end_sample ==
    total) leaves the trailing edge untouched -- the two behaviors independently, in one call."""
    original = torch.linspace(-0.5, 0.5, 800).reshape(2, 400)
    new_middle = torch.full((2, 50), 0.9)

    spliced = MiniMaxMusic3Mixin._minimax_music3_apply_rerender_waveform_splice(
        original, new_middle, start_sample=350, end_sample=400, sample_rate=1000, crossfade_ms=10.0,
    )
    kept = spliced[..., 350:400]
    assert not torch.equal(kept[..., :10], new_middle[..., :10])  # leading 10 samples ramped (10ms @ 1000Hz)
    assert torch.equal(kept[..., 10:], new_middle[..., 10:])  # everything past the ramp, and the trailing edge, untouched


def test_rerender_waveform_splice_never_modifies_the_original_samples_outside_the_range():
    original = torch.linspace(-0.5, 0.5, 800).reshape(2, 400)
    new_middle = torch.full((2, 50), 0.9)
    spliced = MiniMaxMusic3Mixin._minimax_music3_apply_rerender_waveform_splice(
        original, new_middle, start_sample=80, end_sample=130, sample_rate=1000, crossfade_ms=10.0,
    )
    assert spliced.shape[-1] == 400
    assert torch.equal(spliced[..., :80], original[..., :80])
    assert torch.equal(spliced[..., 130:], original[..., 130:])


# ---------------------------------------------------------------------------
# "regenerate" mode: everything before the (chunk-window-snapped) cut point
# is preserved sample-exact; the tail is discarded and replaced.
# ---------------------------------------------------------------------------
def test_regenerate_preserves_the_prefix_sample_exact_and_discards_the_tail(tmp_path):
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)
    num_frames = 350  # chunk_starts = [0, 100, 200]
    wav_path, sidecar_codes, sidecar_prefix, original_wave = _make_synthetic_multi_chunk_song(
        tmp_path, pipeline, num_frames, seed=0,
    )
    cumulative = _cumulative_for(pipeline, num_frames)
    T = 100  # chunk_starts[1] -- an INTERNAL chunk start (min_index=1 requirement)
    preserved_samples = cumulative[1]

    result = manager._generate_aud2aud_minimax_music3(
        {
            "mode": "repaint", "music3_repaint_mode": "regenerate",
            "repaint_start": T / pipeline.frame_rate, "repaint_end": (T + 3) / pipeline.frame_rate,
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 321,
        },
        wav_path,
    )
    assert isinstance(result, MiniMaxMusic3RepaintResult)
    assert result.repaint_mode == "regenerate"

    # Preserved prefix codes are EXACTLY the original codes up to T.
    assert torch.equal(result.frame_codes[:T], sidecar_codes[:T])
    assert torch.equal(result.prefix_codes, sidecar_prefix)
    # The tail is NEW -- shorter than the original remainder (T=100 onward in the 350-frame original had 250
    # frames; the new tail only asked for up to 3).
    assert result.num_frames < num_frames
    assert result.num_frames > T

    # Preserved waveform prefix is sample-exact.
    assert torch.equal(result.waveform[..., :preserved_samples], original_wave[..., :preserved_samples])
    assert torch.isfinite(result.waveform).all()


def test_regenerate_of_a_regenerate_result_is_itself_repaintable(tmp_path):
    """The result's sidecar (written by a caller, e.g. routes.py) round-trips the CORRECT code sequence: writing
    one for the FIRST regenerate's own output and continuing it AGAIN (via extend) must work, mirroring
    `minimax_music3_extend_test.py`'s "extend of an extend" coverage, now composed with "regenerate".

    Extend, not "rerender", for the second hop -- see `test_rerender_after_a_short_regenerate_result_is_refused_
    not_mis_spliced` below for why: a "regenerate" result's tail was decoded with CONTINUITY-preserving crop
    treatment (deliberately NOT the standard "chunk 0 of a fresh decode" rule -- see `_minimax_music3_repaint_
    regenerate`'s "Splice alignment" docstring), so if the result's OWN total frame count later collapses to fewer
    flow chunks than the geometry that actually produced it, a "rerender"/second "regenerate" recomputing STANDARD
    chunk geometry from that (smaller) total cannot correctly re-derive the true sample boundaries -- and is
    REFUSED by this module's geometry self-check rather than silently mis-splicing (proven by the test below).
    Extend has no such problem: it never recomputes ANY internal geometry of the existing file at all, only
    appends new codes/audio after its end, so it is unaffected and is the correct choice to prove the sidecar
    itself round-trips correctly.
    """
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)
    num_frames = 350
    wav_path, _codes, _prefix, _wave = _make_synthetic_multi_chunk_song(tmp_path, pipeline, num_frames, seed=0)

    first = manager._generate_aud2aud_minimax_music3(
        {
            "mode": "repaint", "music3_repaint_mode": "regenerate",
            "repaint_start": 100 / pipeline.frame_rate, "repaint_end": 103 / pipeline.frame_rate,
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 1,
        },
        wav_path,
    )

    repainted_path = str(tmp_path / "song_repainted.wav")
    _write_float_wav(repainted_path, first.waveform, first.sample_rate)
    from utils.image_utils import calculate_file_hash
    write_frame_codes_sidecar(
        repainted_path, first.frame_codes, first.prefix_codes,
        sample_rate=first.sample_rate, frame_rate=first.frame_rate,
        prompt=first.prompt, lyrics=first.lyrics, seed=first.actual_seed,
        num_samples=int(first.waveform.shape[-1]),
        content_hash=calculate_file_hash(repainted_path),
    )

    second = manager._generate_audoutpaint_minimax_music3(
        {
            "placement": "extend_forward", "extend_duration_sec": 3 / pipeline.frame_rate,
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 2,
        },
        repainted_path,
    )
    assert torch.equal(second.frame_codes[: first.num_frames], first.frame_codes)
    assert second.num_frames > first.num_frames
    assert torch.equal(second.waveform[..., : first.waveform.shape[-1]], first.waveform)


def test_rerender_after_a_short_regenerate_result_is_refused_not_mis_spliced(tmp_path):
    """Known, TESTED limitation (see the previous test's docstring): a "regenerate" result's tail is decoded with
    continuity-preserving crop treatment, not the standard "fresh decode" rule. If the result's own frame count
    later collapses to fewer flow chunks (here: to a single chunk) than the geometry that actually produced it,
    this module's geometry self-check refuses a later "rerender"/"regenerate" request against that file rather
    than silently computing a wrong splice boundary -- as a `ValidationError` (400, USER-reachable), not a
    `RuntimeError` (500): a caller can reach this without there being a bug."""
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)
    num_frames = 350
    wav_path, _codes, _prefix, _wave = _make_synthetic_multi_chunk_song(tmp_path, pipeline, num_frames, seed=0)

    first = manager._generate_aud2aud_minimax_music3(
        {
            "mode": "repaint", "music3_repaint_mode": "regenerate",
            "repaint_start": 100 / pipeline.frame_rate, "repaint_end": 103 / pipeline.frame_rate,
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 1,
        },
        wav_path,
    )
    assert first.num_frames <= CHUNK_FRAMES, "this test needs the collapsed-to-one-chunk case"

    repainted_path = str(tmp_path / "song_repainted_short.wav")
    _write_float_wav(repainted_path, first.waveform, first.sample_rate)
    from utils.image_utils import calculate_file_hash
    write_frame_codes_sidecar(
        repainted_path, first.frame_codes, first.prefix_codes,
        sample_rate=first.sample_rate, frame_rate=first.frame_rate,
        prompt=first.prompt, lyrics=first.lyrics, seed=first.actual_seed,
        num_samples=int(first.waveform.shape[-1]),
        content_hash=calculate_file_hash(repainted_path),
    )

    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError, match="geometry"):
        manager._generate_aud2aud_minimax_music3(
            {
                "mode": "repaint", "music3_repaint_mode": "rerender",
                "repaint_start": 0.0, "repaint_end": first.num_frames / pipeline.frame_rate,
                "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 2,
            },
            repainted_path,
        )


def test_regenerate_requires_at_least_two_chunks():
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    wav_path_holder = {"path": None}

    # A short (<= CHUNK_FRAMES) sidecar has only one chunk -- no internal boundary exists to cut at.
    def _short_sidecar(tmp_path):
        wav_path = str(tmp_path / "short_song.wav")
        _write_float_wav(wav_path, torch.zeros(2, 1000), 44100)
        write_frame_codes_sidecar(
            wav_path, torch.zeros(50, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
            sample_rate=44100, frame_rate=25.0, prompt="a caption", lyrics="[verse]\nhello world",
            seed=0, num_samples=1000,
        )
        return wav_path

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        wav_path = _short_sidecar(pathlib.Path(tmp))
        with pytest.raises(ValidationError, match="longer source song"):
            manager._generate_aud2aud_minimax_music3(
                {
                    "mode": "repaint", "music3_repaint_mode": "regenerate",
                    "repaint_start": 0.5, "repaint_end": 1.0,
                    "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 0,
                },
                wav_path,
            )


# ---------------------------------------------------------------------------
# Refusals that need no real model at all.
# ---------------------------------------------------------------------------
class _FakeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _placeholder_manager():
    manager = _Manager.__new__(_Manager)
    manager.minimax_music3_components = {
        "tokenizer": _FakeTokenizer(), "language_model": object(),
        "rvq_depth_decoder": _FakeConfig(config=_FakeConfig(num_codebooks=8)),
        "condition_encoder": _FakeConfig(config=_FakeConfig(
            input_sampling_rate=24000, input_hop_length=960, output_sampling_rate=44100, output_hop_length=512,
        )),
        "transformer": object(), "scheduler": object(),
        "vocoder": _FakeConfig(config=_FakeConfig(sampling_rate=44100)),
    }
    manager.is_minimax_music3_model = True
    manager.device = "cpu"
    manager.current_model_info = {}
    return manager


def test_mode_cover_is_refused_with_the_capability_reason():
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError) as excinfo:
        manager._generate_aud2aud_minimax_music3({"mode": "cover"}, "does_not_matter.wav")
    detail = str(getattr(excinfo.value, "detail", excinfo.value))
    assert "rvq" in detail.lower() or "tokenizer" in detail.lower()


def test_mode_other_than_repaint_or_cover_is_refused():
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError):
        manager._generate_aud2aud_minimax_music3({"mode": "something_else"}, "does_not_matter.wav")


def test_music3_repaint_mode_infill_is_refused_with_the_causal_lm_reason():
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError) as excinfo:
        manager._generate_aud2aud_minimax_music3(
            {"mode": "repaint", "music3_repaint_mode": "infill"}, "does_not_matter.wav",
        )
    detail = str(getattr(excinfo.value, "detail", excinfo.value))
    assert "causal" in detail.lower()


@pytest.mark.parametrize("bad_mode", [None, "", "bogus", "cover_but_not_really"])
def test_music3_repaint_mode_invalid_values_are_refused(bad_mode):
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError):
        manager._generate_aud2aud_minimax_music3(
            {"mode": "repaint", "music3_repaint_mode": bad_mode}, "does_not_matter.wav",
        )


@pytest.mark.parametrize("repaint_mode", ["regenerate", "rerender"])
def test_missing_sidecar_is_refused(tmp_path, repaint_mode):
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    wav_path = str(tmp_path / "no_sidecar.wav")
    _write_float_wav(wav_path, torch.zeros(2, 100), 44100)

    with pytest.raises(ValidationError, match="sidecar"):
        manager._generate_aud2aud_minimax_music3(
            {"mode": "repaint", "music3_repaint_mode": repaint_mode, "repaint_start": 0.0, "repaint_end": 1.0},
            wav_path,
        )


@pytest.mark.parametrize("repaint_mode", ["regenerate", "rerender"])
def test_foreign_sidecar_is_refused(tmp_path, repaint_mode):
    """A sidecar written for a DIFFERENT audio file (different num_samples) must not be trusted just because it
    happens to sit next to this one -- same identity-validation mechanism `minimax_music3_extend_test.py` already
    proves for extend, exercised here through both repaint sub-mode call paths."""
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    wav_path = str(tmp_path / "song.wav")
    _write_float_wav(wav_path, torch.zeros(2, 100), 44100)
    write_frame_codes_sidecar(
        wav_path, torch.zeros(300, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
        sample_rate=44100, frame_rate=25.0, prompt="a caption", lyrics="[verse]\nhello world",
        seed=0, num_samples=99999,  # deliberately wrong -- does not match the 100-sample file above
    )

    with pytest.raises(ValidationError, match="sidecar"):
        manager._generate_aud2aud_minimax_music3(
            {
                "mode": "repaint", "music3_repaint_mode": repaint_mode,
                "repaint_start": 4.0, "repaint_end": 8.0,
                "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 0,
            },
            wav_path,
        )


def test_reference_audio_must_be_a_server_side_path_not_bytes():
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError):
        manager._generate_aud2aud_minimax_music3(
            {"mode": "repaint", "music3_repaint_mode": "regenerate"}, b"\x00\x01\x02\x03",
        )


# ---------------------------------------------------------------------------
# Budget guard ("regenerate" mode) fires BEFORE GPU staging.
# ---------------------------------------------------------------------------
class _RestingCpuParam:
    device = torch.device("cpu")


class _StagingForbiddenLanguageModel:
    """Raises if the mixin ever tries to STAGE (`.to()`) it -- proves the budget guard raised BEFORE any staging
    call. Mirrors `minimax_music3_extend_test.py`'s identical fixture."""

    def __init__(self, max_position_embeddings):
        self.config = _FakeConfig(max_position_embeddings=max_position_embeddings)

    def to(self, device):
        raise AssertionError("staging must not happen before the pre-flight budget check")

    def parameters(self):
        return iter([_RestingCpuParam()])


def test_regenerate_budget_guard_fires_before_gpu_staging(tmp_path):
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    # Tiny position budget: prompt(5, from _FakeTokenizer) + warm-up(1) + previous(300, at T) + new(up to lots) --
    # comfortably over a small max_position_embeddings.
    manager.minimax_music3_components["language_model"] = _StagingForbiddenLanguageModel(max_position_embeddings=20)
    manager.minimax_music3_components["rvq_depth_decoder"] = _FakeConfig(config=_FakeConfig(num_codebooks=8))

    # The wav's sample count must be geometrically CONSISTENT with a 300-frame song under the real production
    # constants (`_placeholder_manager()`'s condition_encoder/vocoder configs are the REAL ones, not a tiny
    # fixture's) -- this method's own geometry self-check (module docstring) would otherwise refuse BEFORE ever
    # reaching the budget guard this test is actually about. Computed the same way `_minimax_music3_repaint_
    # regenerate` computes it, from the SAME pure function this test imports.
    cumulative = compute_cumulative_samples(
        300, 512, CHUNK_FRAMES, CHUNK_HOP, CROP_LEFT_LATENT, CROP_RIGHT_LATENT, 24000, 960, 44100, 512,
    )
    num_samples = cumulative[-1]

    wav_path = str(tmp_path / "budget_song.wav")
    _write_float_wav(wav_path, torch.zeros(2, num_samples), 44100)
    write_frame_codes_sidecar(
        wav_path, torch.zeros(300, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
        sample_rate=44100, frame_rate=25.0, prompt="a caption", lyrics="[verse]\nhello world",
        seed=0, num_samples=num_samples,
    )

    with pytest.raises(ValidationError, match="limit"):
        manager._generate_aud2aud_minimax_music3(
            {
                "mode": "repaint", "music3_repaint_mode": "regenerate",
                "repaint_start": 100 / 25.0, "repaint_end": 5000 / 25.0,
                "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 0,
            },
            wav_path,
        )
