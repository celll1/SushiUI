"""Checkpoint-contract tests for MiniMax Music 3's flow-stage chunk/crop geometry.

``MiniMaxMusic3Pipeline.prepare_chunks`` (200-frame windows / 100-frame hop)
and ``.decode``'s crop arithmetic (86 leading / 258 trailing latent frames,
see ``defaults.CROP_LEFT_LATENT`` / ``CROP_RIGHT_LATENT``) are ported verbatim
from upstream ``before_denoise.py`` / ``decoders.py`` (diffusers PR #14456).
These tests pin the arithmetic directly (no model needed for
``prepare_chunks``) and exercise the full chunk -> denoise -> decode path on a
tiny synthetic pipeline (real vendored classes, tiny random weights, CPU) to
prove the multi-chunk overlap/crop stitching produces the analytically
expected output length -- the thing most likely to silently break under a
refactor of the loop in ``denoise_chunks``/``decode``.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusers import FlowMatchEulerDiscreteScheduler

from core.models.minimax_music3.defaults import (
    CHUNK_FRAMES,
    CHUNK_HOP,
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
)
from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline
from core.models.minimax_music3.vendor import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)


# ---------------------------------------------------------------------------
# prepare_chunks: pure arithmetic, no model needed.
# ---------------------------------------------------------------------------
def _pipeline_for_chunk_math():
    return MiniMaxMusic3Pipeline(
        tokenizer=None,
        language_model=None,
        rvq_depth_decoder=None,
        condition_encoder=None,
        transformer=None,
        scheduler=None,
        vocoder=None,
    )


def test_prepare_chunks_short_song_is_a_single_chunk():
    pipeline = _pipeline_for_chunk_math()
    frame_hiddens = torch.zeros(1, CHUNK_FRAMES, 1)
    assert pipeline.prepare_chunks(frame_hiddens) == [0]

    frame_hiddens_shorter = torch.zeros(1, CHUNK_FRAMES - 1, 1)
    assert pipeline.prepare_chunks(frame_hiddens_shorter) == [0]


def test_prepare_chunks_hop_arithmetic_pinned():
    pipeline = _pipeline_for_chunk_math()
    # 201 frames: one frame over the single-chunk threshold -> starts at 0, then every CHUNK_HOP frames until
    # `num_frames - CHUNK_HOP`.
    frame_hiddens = torch.zeros(1, CHUNK_FRAMES + 1, 1)
    starts = pipeline.prepare_chunks(frame_hiddens)
    assert starts == list(range(0, (CHUNK_FRAMES + 1) - CHUNK_HOP, CHUNK_HOP))
    assert starts[0] == 0

    # A song several windows long: pin the exact start list for a concrete frame count.
    num_frames = 550
    starts = pipeline.prepare_chunks(torch.zeros(1, num_frames, 1))
    assert starts == [0, 100, 200, 300, 400]  # range(0, 450, 100)


# ---------------------------------------------------------------------------
# Full chunk -> denoise -> decode stitching on a tiny synthetic pipeline.
# ---------------------------------------------------------------------------
def _build_tiny_flow_pipeline():
    condition_hidden_dim = 8
    num_condition_layers = 8
    out_dim = 4
    in_channels = 2

    # `output_hop_length` (condition encoder) must equal the vocoder's total upsample ratio (product of
    # `upsampling_ratios`) on the real checkpoint (512 == 8*8*4*2) -- the crop arithmetic below assumes it. Keep
    # that invariant here with a tiny vocoder (`upsampling_ratios=(2, 2)` -> 4) rather than the real 512.
    condition_encoder = MiniMaxMusic3ConditionEncoder(
        condition_hidden_dim=condition_hidden_dim,
        num_condition_layers=num_condition_layers,
        out_dim=out_dim,
        input_sampling_rate=24000,
        input_hop_length=960,
        output_sampling_rate=44100,
        output_hop_length=4,
    ).eval()
    transformer = MiniMaxMusic3Transformer1DModel(
        in_channels=in_channels,
        condition_dim=out_dim,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=4,
        ff_inner_dim=8,
        rotary_dim=4,
        fourier_embedding_dim=8,
    ).eval()
    vocoder = MiniMaxMusic3Vocoder(
        latent_channels=in_channels,
        decoder_input_dim=4,
        decoder_hidden_dim=4,
        upsampling_ratios=(2, 2),
        sampling_rate=44100,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(invert_sigmas=True)

    pipeline = MiniMaxMusic3Pipeline(
        tokenizer=None,
        language_model=None,
        rvq_depth_decoder=None,
        condition_encoder=condition_encoder,
        transformer=transformer,
        scheduler=scheduler,
        vocoder=vocoder,
        execution_device=torch.device("cpu"),
    )
    return pipeline, condition_hidden_dim, num_condition_layers


def test_multi_chunk_denoise_and_decode_stitches_to_the_expected_sample_count():
    torch.manual_seed(0)
    pipeline, condition_hidden_dim, num_condition_layers = _build_tiny_flow_pipeline()

    # A song long enough to force multiple flow-matching windows (> CHUNK_FRAMES AR frames).
    num_ar_frames = CHUNK_FRAMES + CHUNK_HOP + 5
    frame_hiddens = torch.randn(1, num_ar_frames, num_condition_layers * condition_hidden_dim)

    chunk_starts = pipeline.prepare_chunks(frame_hiddens)
    assert len(chunk_starts) >= 2, "this test requires the multi-window path to actually engage"

    generator = torch.Generator().manual_seed(0)
    latent_chunks = pipeline.denoise_chunks(
        frame_hiddens, num_inference_steps=2, flow_guidance_scale=1.7, generator=generator
    )
    assert len(latent_chunks) == len(chunk_starts)

    audio = pipeline.decode(latent_chunks, output_type="pt")
    assert torch.isfinite(audio).all()
    assert audio.shape[0] == 1 and audio.shape[1] == 2
    assert bool((audio.abs() <= 1.0001).all())

    # Analytic expected sample count: each chunk's latent length comes from the condition encoder's frame -> latent
    # resample (see MiniMaxMusic3ConditionEncoder.forward), and the decode crop keeps 86 leading latent frames off
    # every window but the first and 258 trailing latent frames off every window but the last (defaults.py).
    hop_length = pipeline.latent_hop_length
    upsample = 1
    for ratio in pipeline.vocoder.config.upsampling_ratios:
        upsample *= ratio
    assert hop_length == upsample  # sanity: latent_hop_length IS the vocoder's total upsample ratio

    expected_samples = 0
    num_chunks = len(latent_chunks)
    for chunk_index, latents in enumerate(latent_chunks):
        latent_len = latents.shape[-1]
        left = 0 if chunk_index == 0 else CROP_LEFT_LATENT
        right = 0 if chunk_index == num_chunks - 1 else CROP_RIGHT_LATENT
        expected_samples += (latent_len - left - right) * hop_length

    assert audio.shape[-1] == expected_samples
