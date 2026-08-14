"""MiniMax Music 3 audio extend (forward continuation) -- design doc phase plan item 7's BACKEND half.

`MiniMaxMusic3Mixin._generate_audoutpaint_minimax_music3` (`core.pipeline_backends.minimax_music3`) resumes the
autoregressive stage from a song's frame-code sidecar, restricts the flow stage to the newly generated tail, and
splices that tail onto the ORIGINAL (unmodified) waveform read back from disk. This file covers:

  * resume-from-sidecar, run through the mixin end to end, reproduces the SAME new-tail codes a direct
    `MiniMaxMusic3Pipeline.generate_ar(..., resume_frame_codes=..., resume_prefix_codes=...)` call would (the
    mixin's sidecar-to-generate_ar wiring, not `generate_ar`'s own resume correctness -- that is
    `minimax_music3_ar_resume_test.py`'s job);
  * the preserved-span property this implementation actually delivers: the ORIGINAL waveform, read verbatim from
    the source file, comes back byte-for-byte in the result (`_minimax_music3_apply_extend_waveform_splice`'s pure
    unit tests, plus an end-to-end check through the mixin);
  * extend-of-an-extend (a second extend request against the FIRST extend's own output + sidecar);
  * refusals: backward/unsupported placement, a missing sidecar, a foreign/stale sidecar (both a `num_samples`
    mismatch AND -- audit finding F4 -- the harder case of an entirely DIFFERENT source file whose sample count
    happens to coincide, which only the server-computed content hash catches), a mono source file (audit finding
    F3), a non-path `reference_audio`;
  * the budget guard (`check_ar_resume_budget`) firing BEFORE either GPU-staging move, not after.

Two REAL tiny `transformers.Qwen3ForCausalLM` + vendored-module pipelines are used for the equivalence/preservation
tests (mirrors `minimax_music3_ar_resume_test.py`'s approach -- small enough to run on CPU in seconds, no GPU, no
real checkpoint); the refusal/budget tests use lightweight fakes with no torch model at all.

KNOWN COVERAGE GAP (recorded explicitly rather than silently, per the phase-7a audit): every end-to-end test in
this file uses <= 4 original frames and <= 3 new frames, both far below `CHUNK_FRAMES` (200). `prepare_chunks`
therefore always returns `[0]` in every test here -- the multi-chunk overlap-blend/crop-stitch arithmetic inside
`denoise_chunks`/`decode` (exercised for the general, non-extend case by `minimax_music3_chunk_geometry_test.py`)
is never exercised THROUGH THE EXTEND PATH specifically (i.e. the interaction between an extend's resume boundary
and a new region that itself spans multiple flow chunks, which would require a real `extend_duration_sec` past
one ~8s chunk). A real-checkpoint smoke test is the natural place to close this, not a CPU unit test with a tiny
model -- forcing >200 sequential frames through even a tiny `Qwen3ForCausalLM` would make this file materially
slower for a case the general chunk-geometry arithmetic is already independently covered on. Left as a named gap
rather than built out here.
"""

import os
import sys

import pytest
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3Config, Qwen3ForCausalLM

from core.models.minimax_music3.defaults import AUDIO_CODE_OFFSET, SEMANTIC_VOCAB_SIZE
from core.models.minimax_music3.frame_codes import (
    read_frame_codes_sidecar_for_audio,
    sidecar_path_for_audio,
    write_frame_codes_sidecar,
)
from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline
from core.models.minimax_music3.vendor import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)
from core.pipeline_backends.minimax_music3 import MiniMaxMusic3ExtendResult, MiniMaxMusic3Mixin

_HIDDEN = 16


class _FakeTokenizer:
    """`encode_text` needs a tokenizer; the AR loop itself never calls it, so any deterministic id sequence works
    (matches `minimax_music3_ar_resume_test.py`'s fake)."""

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}


def _build_tiny_pipeline(max_position_embeddings: int = 4096) -> MiniMaxMusic3Pipeline:
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
    condition_encoder = MiniMaxMusic3ConditionEncoder(
        condition_hidden_dim=_HIDDEN, num_condition_layers=8, out_dim=8,
        input_sampling_rate=24000, input_hop_length=960, output_sampling_rate=44100, output_hop_length=512,
    ).eval()
    transformer = MiniMaxMusic3Transformer1DModel(
        in_channels=4, condition_dim=8, num_layers=1, num_attention_heads=1,
        attention_head_dim=4, ff_inner_dim=8, rotary_dim=4, fourier_embedding_dim=8,
    ).eval()
    vocoder = MiniMaxMusic3Vocoder(
        latent_channels=4, decoder_input_dim=4, decoder_hidden_dim=4, upsampling_ratios=(2, 2), sampling_rate=44100,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(invert_sigmas=True)

    return MiniMaxMusic3Pipeline(
        tokenizer=_FakeTokenizer(), language_model=language_model, rvq_depth_decoder=rvq_depth_decoder,
        condition_encoder=condition_encoder, transformer=transformer, scheduler=scheduler, vocoder=vocoder,
        execution_device=torch.device("cpu"),
    )


def _frames_worth(pipeline: MiniMaxMusic3Pipeline, num_frames: int) -> float:
    return num_frames / pipeline.frame_rate


class _Manager(MiniMaxMusic3Mixin):
    """Minimal `DiffusionPipelineManager` stand-in -- same pattern as `minimax_music3_staged_offload_test.py`."""

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
    """`subtype='FLOAT'` (32-bit float PCM) keeps the write/read round trip lossless -- required for the
    "preserved span is sample-exact" assertions below to be meaningful (a 16-bit PCM write would itself
    quantize, making an exact-equality check meaningless)."""
    data = wave.detach().cpu().numpy().T  # [samples, channels]
    sf.write(path, data, sample_rate, subtype="FLOAT")


def _make_original_song(
    tmp_path, pipeline: MiniMaxMusic3Pipeline, num_frames: int, seed: int,
    prompt="a caption", lyrics="[verse]\nhello world", write_content_hash: bool = True,
):
    """Runs a fresh (non-resumed) generation through all three stages, writes the wav + sidecar, and returns
    (wav_path, ar_result, waveform). `write_content_hash` defaults True to mirror production
    (`routes.py`'s frame-code-sidecar-write block always writes one) -- most tests here therefore exercise the
    content-hash half of `matches()`, not just `num_samples`."""
    text_ids = pipeline.encode_text(prompt, lyrics)
    generator = torch.Generator().manual_seed(seed)
    ar_result = pipeline.generate_ar(text_ids, _frames_worth(pipeline, num_frames), generator=generator)
    latent_chunks = pipeline.denoise_chunks(
        ar_result.frame_hiddens, num_inference_steps=2, flow_guidance_scale=1.7, generator=generator,
    )
    waveform = pipeline.decode(latent_chunks, output_type="pt")[0]  # [2, samples]

    wav_path = str(tmp_path / "song.wav")
    _write_float_wav(wav_path, waveform, int(pipeline.sampling_rate))
    from utils.image_utils import calculate_file_hash
    content_hash = calculate_file_hash(wav_path) if write_content_hash else ""
    write_frame_codes_sidecar(
        wav_path, ar_result.frame_codes, ar_result.prefix_codes,
        sample_rate=int(pipeline.sampling_rate), frame_rate=float(pipeline.frame_rate),
        prompt=prompt, lyrics=lyrics, seed=seed, num_samples=int(waveform.shape[-1]),
        content_hash=content_hash,
    )
    return wav_path, ar_result, waveform


# ---------------------------------------------------------------------------
# Resume-from-sidecar matches a direct generate_ar resume call, for the shared prefix.
# ---------------------------------------------------------------------------
def test_extend_resume_matches_direct_generate_ar_resume(tmp_path):
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)

    wav_path, ar_orig, original_wave = _make_original_song(tmp_path, pipeline, num_frames=4, seed=0)

    result = manager._generate_audoutpaint_minimax_music3(
        {
            "placement": "extend_forward", "extend_duration_sec": _frames_worth(pipeline, 3),
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 123,
        },
        reference_audio=wav_path,
    )
    assert isinstance(result, MiniMaxMusic3ExtendResult)

    # The preserved (original) prefix of the returned code sequence is exactly what was written to the sidecar.
    assert torch.equal(result.frame_codes[:4], ar_orig.frame_codes)

    # The NEW tail must match an independent, direct generate_ar resume call with the same seed.
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")
    direct = pipeline.generate_ar(
        text_ids, _frames_worth(pipeline, 3), generator=torch.Generator(device="cpu").manual_seed(123),
        resume_frame_codes=ar_orig.frame_codes, resume_prefix_codes=ar_orig.prefix_codes,
    )
    assert torch.equal(result.frame_codes[4:], direct.frame_codes)
    assert result.appended_num_frames == direct.frame_codes.shape[0]
    assert result.num_frames == 4 + direct.frame_codes.shape[0]
    assert torch.equal(result.prefix_codes, ar_orig.prefix_codes)


# ---------------------------------------------------------------------------
# Preserved-span property: sample-exact, both at the pure-function level and end to end.
# ---------------------------------------------------------------------------
def test_waveform_splice_never_modifies_the_original_samples():
    original = torch.linspace(-0.5, 0.5, 200).reshape(2, 100)
    new = torch.full((2, 50), 0.9)
    spliced = MiniMaxMusic3Mixin._minimax_music3_apply_extend_waveform_splice(
        original, new, sample_rate=1000, crossfade_ms=10.0,
    )
    assert spliced.shape[-1] == 150
    # Every original sample survives untouched, at the exact same position.
    assert torch.equal(spliced[..., :100], original)
    # The declick ramp only touches the NEW side, and only its leading edge.
    assert not torch.equal(spliced[..., 100:], new)  # some of the new head was ramped
    assert torch.equal(spliced[..., 110:], new[..., 10:])  # past the 10ms/1000Hz = 10-sample ramp, untouched


def test_waveform_splice_degenerate_empty_sides():
    original = torch.zeros(2, 0)
    new = torch.ones(2, 10)
    assert torch.equal(
        MiniMaxMusic3Mixin._minimax_music3_apply_extend_waveform_splice(original, new, sample_rate=1000), new,
    )
    original2 = torch.ones(2, 10)
    new2 = torch.zeros(2, 0)
    assert torch.equal(
        MiniMaxMusic3Mixin._minimax_music3_apply_extend_waveform_splice(original2, new2, sample_rate=1000), original2,
    )


def test_extend_preserves_the_original_waveform_sample_exact_end_to_end(tmp_path):
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)
    wav_path, _ar_orig, original_wave = _make_original_song(tmp_path, pipeline, num_frames=4, seed=0)

    result = manager._generate_audoutpaint_minimax_music3(
        {
            "placement": "extend_forward", "extend_duration_sec": _frames_worth(pipeline, 3),
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 123,
        },
        reference_audio=wav_path,
    )

    assert result.waveform.shape[-1] > original_wave.shape[-1]
    assert torch.equal(result.waveform[..., :original_wave.shape[-1]], original_wave)


# ---------------------------------------------------------------------------
# Extend-of-an-extend.
# ---------------------------------------------------------------------------
def test_extend_of_an_extend_works(tmp_path):
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)
    wav_path, ar_orig, original_wave = _make_original_song(tmp_path, pipeline, num_frames=4, seed=0)

    first = manager._generate_audoutpaint_minimax_music3(
        {
            "placement": "extend_forward", "extend_duration_sec": _frames_worth(pipeline, 3),
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 123,
        },
        reference_audio=wav_path,
    )

    # Persist the first extend's OWN output as a new "song on disk", exactly what a route committing this
    # result would do -- the wav plus a fresh sidecar carrying the FULL concatenated code sequence.
    extended_path = str(tmp_path / "song_extended.wav")
    _write_float_wav(extended_path, first.waveform, first.sample_rate)
    from utils.image_utils import calculate_file_hash
    write_frame_codes_sidecar(
        extended_path, first.frame_codes, first.prefix_codes,
        sample_rate=first.sample_rate, frame_rate=first.frame_rate,
        prompt=first.prompt, lyrics=first.lyrics, seed=first.actual_seed,
        num_samples=int(first.waveform.shape[-1]),
        content_hash=calculate_file_hash(extended_path),
    )

    second = manager._generate_audoutpaint_minimax_music3(
        {
            "placement": "extend_forward", "extend_duration_sec": _frames_worth(pipeline, 2),
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 456,
        },
        reference_audio=extended_path,
    )

    assert second.num_frames == first.num_frames + second.appended_num_frames
    assert torch.equal(second.frame_codes[: first.num_frames], first.frame_codes)
    # The ENTIRE first-extend waveform (original span + first new tail) is preserved sample-exact across the
    # second hop too -- this is the property that makes repeated extension viable at all.
    assert torch.equal(second.waveform[..., : first.waveform.shape[-1]], first.waveform)
    assert second.waveform.shape[-1] > first.waveform.shape[-1]


# ---------------------------------------------------------------------------
# Refusals that need no real model at all.
# ---------------------------------------------------------------------------
def _placeholder_manager():
    manager = _Manager.__new__(_Manager)
    manager.minimax_music3_components = {
        "tokenizer": object(), "language_model": object(), "rvq_depth_decoder": object(),
        "condition_encoder": object(), "transformer": object(), "scheduler": object(), "vocoder": object(),
    }
    manager.is_minimax_music3_model = True
    manager.device = "cpu"
    manager.current_model_info = {}
    return manager


def test_missing_placement_is_refused():
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError, match="placement"):
        manager._generate_audoutpaint_minimax_music3({}, reference_audio="does_not_matter.wav")


@pytest.mark.parametrize("placement", ["extend_backward", "before", "infill", "bridge"])
def test_unsupported_placement_is_refused_with_a_causal_lm_reason(placement):
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError) as excinfo:
        manager._generate_audoutpaint_minimax_music3(
            {"placement": placement}, reference_audio="does_not_matter.wav",
        )
    detail = str(excinfo.value.detail if hasattr(excinfo.value, "detail") else excinfo.value)
    assert "causal" in detail.lower()


def test_reference_audio_must_be_a_server_side_path_not_bytes():
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError):
        manager._generate_audoutpaint_minimax_music3(
            {"placement": "extend_forward"}, reference_audio=b"\x00\x01\x02\x03",
        )


def test_nonexistent_reference_audio_is_refused():
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    with pytest.raises(ValidationError):
        manager._generate_audoutpaint_minimax_music3(
            {"placement": "extend_forward"}, reference_audio="/no/such/file.wav",
        )


def test_missing_sidecar_is_refused(tmp_path):
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    wav_path = str(tmp_path / "no_sidecar.wav")
    _write_float_wav(wav_path, torch.zeros(2, 100), 44100)

    with pytest.raises(ValidationError, match="sidecar"):
        manager._generate_audoutpaint_minimax_music3(
            {"placement": "extend_forward"}, reference_audio=wav_path,
        )


def test_foreign_sidecar_is_refused(tmp_path):
    """A sidecar written for a DIFFERENT audio file (different num_samples) must not be trusted just because it
    happens to sit next to this one."""
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)

    wav_path, ar_orig, _wave = _make_original_song(tmp_path, pipeline, num_frames=4, seed=0)

    # Overwrite the audio file with DIFFERENT content (different sample count) while leaving the sidecar as-is --
    # simulates a stale/foreign sidecar sitting next to an unrelated file.
    _write_float_wav(wav_path, torch.zeros(2, 12345), int(pipeline.sampling_rate))

    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError, match="sidecar"):
        manager._generate_audoutpaint_minimax_music3(
            {
                "placement": "extend_forward", "extend_duration_sec": _frames_worth(pipeline, 3),
                "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 1,
            },
            reference_audio=wav_path,
        )


# ---------------------------------------------------------------------------
# F4: content-hash identity check is server-computed, not caller-suppliable.
# ---------------------------------------------------------------------------
def test_same_sample_count_different_audio_is_refused_via_content_hash(tmp_path):
    """The auditor's exact demonstrated failure mode: a file with the SAME sample count as the one the sidecar
    was written for, but with ENTIRELY DIFFERENT audio content, must still be refused -- same-sample-count is not
    a coincidence (two songs generated for the same requested duration that both reach the same stop condition
    have identical sample counts BY CONSTRUCTION), so `num_samples` alone is not a strong enough identity check.
    `matches()`'s content-hash comparison is what catches this; it only works because this call computes the
    hash itself rather than trusting one from the caller."""
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)

    wav_path, ar_orig, original_wave = _make_original_song(tmp_path, pipeline, num_frames=4, seed=0)

    # Overwrite with DIFFERENT audio content but the EXACT SAME sample count/shape/dtype -- same "shape" as the
    # auditor's demonstrated failure, not a contrived shape mismatch.
    different_audio = torch.full_like(original_wave, 0.42)
    assert not torch.equal(different_audio, original_wave)
    _write_float_wav(wav_path, different_audio, int(pipeline.sampling_rate))

    from api.error_handlers import ValidationError

    with pytest.raises(ValidationError, match="sidecar"):
        manager._generate_audoutpaint_minimax_music3(
            {
                "placement": "extend_forward", "extend_duration_sec": _frames_worth(pipeline, 3),
                "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 1,
            },
            reference_audio=wav_path,
        )


def test_caller_supplied_content_hash_is_ignored_not_trusted(tmp_path):
    """A caller passing `content_hash` in `params` must have NO effect either way -- it is neither an opt-out
    (a bogus value must not weaken the real, server-computed check) nor a required input (a legitimate request
    that omits it must still succeed, since the value is always computed here regardless)."""
    pipeline = _build_tiny_pipeline()
    manager = _Manager(pipeline)
    wav_path, ar_orig, original_wave = _make_original_song(tmp_path, pipeline, num_frames=4, seed=0)

    result = manager._generate_audoutpaint_minimax_music3(
        {
            "placement": "extend_forward", "extend_duration_sec": _frames_worth(pipeline, 3),
            "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 123,
            "content_hash": "this-is-not-a-real-hash-and-must-be-ignored",
        },
        reference_audio=wav_path,
    )
    assert torch.equal(result.waveform[..., : original_wave.shape[-1]], original_wave)


# ---------------------------------------------------------------------------
# F3: channel-count mismatch is refused right after the source file is read, before any staging.
# ---------------------------------------------------------------------------
def test_mono_source_file_is_refused_before_staging(tmp_path):
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    wav_path = str(tmp_path / "mono_song.wav")
    _write_float_wav(wav_path, torch.zeros(1, 1000), 44100)  # mono, not stereo
    write_frame_codes_sidecar(
        wav_path, torch.zeros(4, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
        sample_rate=44100, frame_rate=25.0, prompt="a caption", lyrics="[verse]\nhello world",
        seed=0, num_samples=1000,
    )
    manager.minimax_music3_components["vocoder"] = _FakeConfig(config=_FakeConfig(sampling_rate=44100))
    manager.minimax_music3_components["condition_encoder"] = _FakeConfig(
        config=_FakeConfig(input_sampling_rate=25 * 512, input_hop_length=512)
    )
    manager.minimax_music3_components["rvq_depth_decoder"] = _FakeConfig(config=_FakeConfig(num_codebooks=8))
    manager.minimax_music3_components["tokenizer"] = _FakeTokenizer()

    with pytest.raises(ValidationError, match="stereo"):
        manager._generate_audoutpaint_minimax_music3(
            {
                "placement": "extend_forward", "extend_duration_sec": 3 / 25.0,
                "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 0,
            },
            reference_audio=wav_path,
        )


# ---------------------------------------------------------------------------
# Budget guard fires BEFORE staging.
# ---------------------------------------------------------------------------
class _FakeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _RestingCpuParam:
    device = torch.device("cpu")


class _StagingForbiddenLanguageModel:
    """Raises if the mixin ever tries to STAGE (`.to()`) it -- proving the budget guard raised BEFORE any
    staging call, not after. `.parameters()` stays harmless (it only reports a resting device, mirroring how
    `MiniMaxMusic3Pipeline.execution_device`'s own fallback reads a parameter's resting device without moving
    anything -- that introspection is not "staging" in the sense this test cares about)."""

    def __init__(self, max_position_embeddings):
        self.config = _FakeConfig(max_position_embeddings=max_position_embeddings)

    def to(self, device):
        raise AssertionError("staging must not happen before the pre-flight budget check")

    def parameters(self):
        return iter([_RestingCpuParam()])


def test_budget_guard_fires_before_gpu_staging(tmp_path):
    from api.error_handlers import ValidationError

    manager = _placeholder_manager()
    # A tiny position budget: prompt(5, from _FakeTokenizer) + warm-up(1) + previous(10) + new(up to 10) == up to
    # 26, comfortably over 20 -- mirrors minimax_music3_ar_resume_test.py's analogous budget test.
    manager.minimax_music3_components["language_model"] = _StagingForbiddenLanguageModel(max_position_embeddings=20)
    manager.minimax_music3_components["tokenizer"] = _FakeTokenizer()
    manager.minimax_music3_components["vocoder"] = _FakeConfig(config=_FakeConfig(sampling_rate=44100))
    manager.minimax_music3_components["condition_encoder"] = _FakeConfig(
        config=_FakeConfig(input_sampling_rate=25 * 512, input_hop_length=512)
    )
    manager.minimax_music3_components["rvq_depth_decoder"] = _FakeConfig(config=_FakeConfig(num_codebooks=8))

    wav_path = str(tmp_path / "budget_song.wav")
    _write_float_wav(wav_path, torch.zeros(2, 1000), 44100)
    write_frame_codes_sidecar(
        wav_path, torch.zeros(10, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
        sample_rate=44100, frame_rate=25.0, prompt="a caption", lyrics="[verse]\nhello world",
        seed=0, num_samples=1000,
    )

    with pytest.raises(ValidationError, match="limit"):
        manager._generate_audoutpaint_minimax_music3(
            {
                "placement": "extend_forward", "extend_duration_sec": 10 / 25.0,
                "num_inference_steps": 2, "flow_guidance_scale": 1.7, "seed": 0,
            },
            reference_audio=wav_path,
        )
