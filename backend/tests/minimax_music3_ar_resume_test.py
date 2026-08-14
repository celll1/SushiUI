"""AR-resume tests: teacher-forced batched replay, budget guards, and the sidecar dtype contract.

``MiniMaxMusic3Pipeline.generate_ar``'s `resume_frame_codes` / `resume_prefix_codes` path (SushiUI addition, not in
upstream diffusers PR #14456) reconstructs the language model's KV cache from stored codes with ONE (or, for a long
history, a few CHUNKED) forward call rather than replaying the sequential sampling loop. This is the highest-risk
surface in the phase-1 port: a mistake here would only show up as bad audio deep into a long extend, with the ~22GB
model already resident.

These tests use a REAL `transformers.Qwen3ForCausalLM` at a tiny config (not a hand-rolled fake) so the
`.model(inputs_embeds=..., past_key_values=..., use_cache=True)` / `.lm_head` / `.config.vocab_size` /
`.config.max_position_embeddings` contract this pipeline actually depends on is exercised for real, while staying
small enough to run on CPU in a couple of seconds -- no GPU, no real checkpoint.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3Config, Qwen3ForCausalLM

import core.models.minimax_music3.pipeline as mm3_pipeline_module
from core.models.minimax_music3.defaults import AUDIO_CODE_OFFSET, MAX_AUDIO_FRAMES, SEMANTIC_VOCAB_SIZE
from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline
from core.models.minimax_music3.vendor import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)

_HIDDEN = 16


class _FakeTokenizer:
    """encode_text needs a tokenizer; the AR loop itself never calls it, so any deterministic id sequence works."""

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}


def _build_tiny_pipeline(
    max_position_embeddings: int = 4096,
    lm_dtype: torch.dtype = torch.float32,
    depth_dtype: torch.dtype = torch.float32,
) -> MiniMaxMusic3Pipeline:
    """`lm_dtype`/`depth_dtype` default to float32 (every EXISTING test in this file relies on that default and is
    unaffected). Passing them apart -- something the loader never currently does, since it drives both from one
    shared `torch_dtype`, but a future per-component quantization pass (design doc phase plan item 9) could -- is
    what the dtype-matrix tests below use to exercise the three LM<->depth-decoder dtype crossings in
    `_generate_depth_codes`/`generate_ar` (`pipeline.py`'s casts at the `rvq_depth_decoder.projection` calls and the
    `frame_hiddens` concatenation)."""
    torch.manual_seed(1234)  # fixed init: both pipelines built from this function are numerically identical

    vocab = AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE + 1  # covers AUDIO_END_TOKEN_ID and every audio code id used
    lm_config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=_HIDDEN,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=False,
    )
    language_model = Qwen3ForCausalLM(lm_config).eval().to(lm_dtype)

    rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder(
        hidden_size=_HIDDEN, num_layers=2, num_attention_heads=2, intermediate_size=32,
        audio_vocab_size=17, num_codebooks=8, max_position_embeddings=16,
    ).eval().to(depth_dtype)
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
        tokenizer=_FakeTokenizer(),
        language_model=language_model,
        rvq_depth_decoder=rvq_depth_decoder,
        condition_encoder=condition_encoder,
        transformer=transformer,
        scheduler=scheduler,
        vocoder=vocoder,
        execution_device=torch.device("cpu"),
    )


def _frames_worth(pipeline: MiniMaxMusic3Pipeline, num_frames: int) -> float:
    return num_frames / pipeline.frame_rate


# ---------------------------------------------------------------------------
# Equivalence: split generation (fresh + resume) must reproduce the straight-through run bit-exactly.
# ---------------------------------------------------------------------------
def test_resume_reproduces_straight_through_generation():
    pipeline = _build_tiny_pipeline()
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")

    gen_full = torch.Generator().manual_seed(0)
    full = pipeline.generate_ar(text_ids, audio_duration=_frames_worth(pipeline, 7), generator=gen_full)

    gen_split = torch.Generator().manual_seed(0)
    part1 = pipeline.generate_ar(text_ids, audio_duration=_frames_worth(pipeline, 4), generator=gen_split)
    part2 = pipeline.generate_ar(
        text_ids,
        audio_duration=_frames_worth(pipeline, 3),
        generator=gen_split,
        resume_frame_codes=part1.frame_codes,
        resume_prefix_codes=part1.prefix_codes,
    )

    assert torch.equal(full.prefix_codes, part1.prefix_codes)
    assert torch.equal(full.frame_codes[:4], part1.frame_codes)
    assert torch.equal(full.frame_codes[4:], part2.frame_codes)
    combined = torch.cat([part1.frame_codes, part2.frame_codes], dim=0)
    assert torch.equal(combined, full.frame_codes)
    # frame_hiddens for the resumed tail must also match the corresponding slice of the straight-through run: this
    # is what the flow stage actually consumes, not just the codes.
    assert torch.allclose(full.frame_hiddens[:, 4:], part2.frame_hiddens, atol=1e-5)


def test_resume_replay_is_equivalent_when_chunked_smaller_than_the_history(monkeypatch):
    # Force the chunked-replay loop (item 4's fix) to actually take more than one iteration by shrinking the chunk
    # size below the replay history length, and confirm the result is unchanged from the single-shot-replay case
    # covered by the test above.
    monkeypatch.setattr(mm3_pipeline_module, "AR_RESUME_REPLAY_CHUNK_FRAMES", 2)

    pipeline = _build_tiny_pipeline()
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")

    gen_full = torch.Generator().manual_seed(0)
    full = pipeline.generate_ar(text_ids, audio_duration=_frames_worth(pipeline, 7), generator=gen_full)

    gen_split = torch.Generator().manual_seed(0)
    part1 = pipeline.generate_ar(text_ids, audio_duration=_frames_worth(pipeline, 4), generator=gen_split)
    # Replay history here is prefix (1) + 4 frames == 5 rows, well above the chunk size of 2, forcing >= 3 chunks.
    part2 = pipeline.generate_ar(
        text_ids,
        audio_duration=_frames_worth(pipeline, 3),
        generator=gen_split,
        resume_frame_codes=part1.frame_codes,
        resume_prefix_codes=part1.prefix_codes,
    )

    assert torch.equal(full.frame_codes[4:], part2.frame_codes)


# ---------------------------------------------------------------------------
# Item 3: sidecar dtype contract + shape validation.
# ---------------------------------------------------------------------------
def test_resume_accepts_a_compact_int16_sidecar_dtype():
    pipeline = _build_tiny_pipeline()
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")

    gen1 = torch.Generator().manual_seed(0)
    part1 = pipeline.generate_ar(text_ids, audio_duration=_frames_worth(pipeline, 4), generator=gen1)

    # The design doc's per-generation sidecar stores codes as int16; generate_ar must accept that directly rather
    # than requiring the caller to pre-cast (nn.Embedding rejects int16 with no useful context otherwise).
    sidecar_frame_codes = part1.frame_codes.to(torch.int16)
    sidecar_prefix_codes = part1.prefix_codes.to(torch.int16)

    # A fresh generator: this test only proves the int16 sidecar is ACCEPTED and produces well-formed output, not
    # bit-exact reproduction (that is covered by test_resume_reproduces_straight_through_generation, which passes
    # torch.long throughout). Passing the int16 codes through `_embed_audio_frames` -> `nn.Embedding` without a
    # crash IS the assertion of interest here.
    gen2 = torch.Generator().manual_seed(0)
    part2 = pipeline.generate_ar(
        text_ids,
        audio_duration=_frames_worth(pipeline, 3),
        generator=gen2,
        resume_frame_codes=sidecar_frame_codes,
        resume_prefix_codes=sidecar_prefix_codes,
    )
    assert part2.frame_codes.dtype == torch.long
    assert part2.frame_codes.shape[1] == pipeline.num_codebooks
    assert torch.isfinite(part2.frame_hiddens).all()


def test_resume_rejects_mismatched_last_dimension():
    pipeline = _build_tiny_pipeline()
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")

    wrong_codebooks = torch.zeros(3, pipeline.num_codebooks + 1, dtype=torch.long)
    prefix = torch.zeros(1, pipeline.num_codebooks, dtype=torch.long)
    with pytest.raises(ValueError, match="num_codebooks"):
        pipeline.generate_ar(
            text_ids, audio_duration=_frames_worth(pipeline, 2),
            resume_frame_codes=wrong_codebooks, resume_prefix_codes=prefix,
        )

    frame_codes = torch.zeros(3, pipeline.num_codebooks, dtype=torch.long)
    wrong_prefix = torch.zeros(1, pipeline.num_codebooks + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="num_codebooks"):
        pipeline.generate_ar(
            text_ids, audio_duration=_frames_worth(pipeline, 2),
            resume_frame_codes=frame_codes, resume_prefix_codes=wrong_prefix,
        )


def test_resume_requires_both_frame_and_prefix_codes_together():
    pipeline = _build_tiny_pipeline()
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")
    frame_codes = torch.zeros(3, pipeline.num_codebooks, dtype=torch.long)

    with pytest.raises(ValueError):
        pipeline.generate_ar(text_ids, audio_duration=_frames_worth(pipeline, 2), resume_frame_codes=frame_codes)
    with pytest.raises(ValueError):
        pipeline.generate_ar(
            text_ids, audio_duration=_frames_worth(pipeline, 2),
            resume_prefix_codes=torch.zeros(1, pipeline.num_codebooks, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Item 2: budget guards.
# ---------------------------------------------------------------------------
def test_resume_is_rejected_when_it_would_exceed_the_frame_cap():
    pipeline = _build_tiny_pipeline(max_position_embeddings=1_000_000)  # do not trip the position guard first
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")

    # A history already at the frame cap: any further request must be rejected outright, not silently clamped.
    resume_frame_codes = torch.zeros(MAX_AUDIO_FRAMES, pipeline.num_codebooks, dtype=torch.long)
    resume_prefix_codes = torch.zeros(1, pipeline.num_codebooks, dtype=torch.long)

    with pytest.raises(ValueError, match="frame"):
        pipeline.generate_ar(
            text_ids,
            audio_duration=_frames_worth(pipeline, 1),
            resume_frame_codes=resume_frame_codes,
            resume_prefix_codes=resume_prefix_codes,
        )


def test_resume_is_rejected_when_it_would_exceed_the_language_model_position_budget():
    # A tiny position budget makes this cheap to trigger without a large resume history.
    pipeline = _build_tiny_pipeline(max_position_embeddings=20)
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")  # 5 prompt tokens (see _FakeTokenizer)

    resume_frame_codes = torch.zeros(10, pipeline.num_codebooks, dtype=torch.long)
    resume_prefix_codes = torch.zeros(1, pipeline.num_codebooks, dtype=torch.long)

    # prompt(5) + warm-up(1) + previous(10) + new(up to 10) == up to 26 > 20.
    with pytest.raises(ValueError, match="position"):
        pipeline.generate_ar(
            text_ids,
            audio_duration=_frames_worth(pipeline, 10),
            resume_frame_codes=resume_frame_codes,
            resume_prefix_codes=resume_prefix_codes,
        )


def test_fresh_generation_within_budget_is_not_rejected():
    pipeline = _build_tiny_pipeline(max_position_embeddings=4096)
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")
    generator = torch.Generator().manual_seed(0)
    result = pipeline.generate_ar(text_ids, audio_duration=_frames_worth(pipeline, 3), generator=generator)
    assert result.frame_codes.shape[0] >= 1


# ---------------------------------------------------------------------------
# Dtype matrix (F2/F3): the language model and RVQ depth decoder cross
# dtypes at three points in `_generate_depth_codes`/`generate_ar`
# (`rvq_depth_decoder.projection(last_hidden)`, `...projection(code_embed)`,
# and the `frame_hiddens` concatenation). The loader currently always hands
# both the SAME `torch_dtype`, so these three sites are unexercised by every
# OTHER test in this file (all float32) and would stay unexercised by a
# regression that reintroduced the crossing -- exactly the gap a real-weight
# run (not this suite) caught once already. bf16/fp16 matmul has no
# optimized CPU kernel, so this stays to the smallest matrix that still
# exercises every crossing: same-dtype (the loader's actual configuration,
# at two different dtypes) and two mismatched pairs (the case the loader
# does not build today but a future per-component quantization pass could).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lm_dtype,depth_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float16),  # mismatched -- the case F2 fixes
        (torch.float32, torch.bfloat16),  # mismatched
    ],
)
def test_generate_ar_runs_under_every_lm_depth_dtype_combination(lm_dtype, depth_dtype):
    pipeline = _build_tiny_pipeline(lm_dtype=lm_dtype, depth_dtype=depth_dtype)
    text_ids = pipeline.encode_text("a caption", "[verse]\nhello world")
    generator = torch.Generator().manual_seed(0)
    result = pipeline.generate_ar(text_ids, audio_duration=_frames_worth(pipeline, 2), generator=generator)
    assert result.frame_codes.shape[0] >= 1
    assert torch.isfinite(result.frame_hiddens.float()).all()
