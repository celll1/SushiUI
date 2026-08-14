"""Structural constants for MiniMax Music 3.

These are fixed properties of the released checkpoint and the reference
inference recipe -- NOT user-facing generation defaults. User-facing defaults
(``prompt``, ``lyrics``, ``audio_duration``, ``num_inference_steps``, ``seed``,
``flow_guidance_scale``) belong in ``backend/api/param_defaults.py`` and are
added in a later commit; see ``docs/guides/MINIMAX_MUSIC3_DESIGN.md``,
"Generation parameter contract".

Component ``config.json`` kwargs are NOT duplicated here: they are read
straight from each component's own ``config.json`` under the model root (or
the equivalent tree) by the loader (a later commit). This module holds only
the checkpoint-contract constants the pipeline needs and that have nowhere
else to live -- the prompt special tokens, the fixed AR sampling recipe, the
chunking geometry, and the (frozen, checkpoint-derived) component properties
that the upstream ``MiniMaxMusic3ModularPipeline`` exposed as computed
properties (``core.models.minimax_music3.pipeline`` reads the same component
configs the same way; these are the documented fallback defaults for when a
config is absent, matching upstream's own fallback values).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Prompt-assembly special tokens and their ids. Checkpoint contract: even
# whitespace-level changes to the assembled prompt change the generated audio.
# See docs/guides/MINIMAX_MUSIC3_DESIGN.md, "Dependency gate" /
# encoders.py's module docstring in the upstream PR.
# ---------------------------------------------------------------------------
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
CAPTION_START = "<|caption_start|>"
CAPTION_END = "<|caption_end|>"
LYRICS_START = "<|lyrics_start|>"
LYRICS_END = "<|lyrics_end|>"
AUDIO_START = "<|audio_start|>"

AUDIO_END_TOKEN_ID = 151670
AUDIO_CFG_TOKEN_ID = 151654
AUDIO_CODE_OFFSET = 151675
SEMANTIC_VOCAB_SIZE = 16384

MAX_PROMPT_TOKENS = 5_000
MAX_AUDIO_FRAMES = 9_000

# ---------------------------------------------------------------------------
# Autoregressive-stage sampling recipe. Fixed by the reference inference
# recipe; not exposed as a generation parameter (see the design doc's
# generation-parameter table).
# ---------------------------------------------------------------------------
AR_CFG_SCALE = 1.5
AR_CFG_TOP_K = 50
AR_SAMPLING_TOP_K = 50

# ---------------------------------------------------------------------------
# Flow-matching chunking geometry (200-frame windows, 100-frame hop; ~344
# latent frames of overlap between neighboring windows at ~3.445 latents per
# AR frame). See docs/guides/MINIMAX_MUSIC3_DESIGN.md, "Architecture, as
# verified".
# ---------------------------------------------------------------------------
CHUNK_FRAMES = 200
CHUNK_HOP = 100
OVERLAP_LATENT_LENGTH = 172
CROP_LEFT_LATENT = 86
CROP_RIGHT_LATENT = 344 - 86

# NOTE: there is deliberately no ``FLOW_GUIDANCE_SCALE_DEFAULT`` (nor an
# ``audio_duration`` / ``num_inference_steps`` default) constant in this
# module. Those three are user-facing rows in the design doc's
# generation-parameter table (``flow_guidance_scale`` default 1.7,
# ``audio_duration`` default 60.0, ``num_inference_steps`` default 30) and
# belong ONLY in ``backend/api/param_defaults.py`` -- see this module's
# docstring above. ``MiniMaxMusic3Pipeline.generate`` /
# ``.denoise_chunks`` take them as required arguments with no default so a
# caller is forced to supply them explicitly (from ``param_defaults.py`` once
# that commit lands) rather than this module quietly duplicating the value.

# ---------------------------------------------------------------------------
# AR-resume replay chunking (SushiUI addition, not an upstream constant): the
# teacher-forced replay that rebuilds the KV cache from stored frame codes
# (``MiniMaxMusic3Pipeline.generate_ar``'s ``resume_*`` path) runs in windows
# of this many frames per forward call rather than one `[2, F+1, hidden]` call
# for the whole history. At the documented 9000-frame cap that would be an
# 18,001-token forward through the 8B language model on top of an already
# ~22GB-resident model; chunking bounds the extra activation memory to one
# window's worth while leaving the result identical (the equivalence argument
# for teacher-forced replay under an extending KV cache holds for any chunk
# size, including 1 -- the sequential generation loop already IS the
# chunk-size-1 case).
# ---------------------------------------------------------------------------
AR_RESUME_REPLAY_CHUNK_FRAMES = 512

# ---------------------------------------------------------------------------
# Component-property fallback values, matching
# ``MiniMaxMusic3ModularPipeline``'s hardcoded fallbacks (used only if a
# component config is unexpectedly absent; normally these are read from the
# loaded components' own configs -- see ``pipeline.py``).
# ---------------------------------------------------------------------------
FALLBACK_SAMPLING_RATE = 44100
FALLBACK_FRAME_RATE = 25.0
FALLBACK_LATENT_HOP_LENGTH = 512
FALLBACK_NUM_CODEBOOKS = 8
FALLBACK_AUDIO_VOCAB_SIZE = 1024
FALLBACK_NUM_CHANNELS_LATENTS = 128

# The language model's rope_theta must load-time-assert to this value; a
# silent rope fallback degrades output without erroring (dependency-gate note
# in the design doc). Checked by the loader, not by this pipeline module.
EXPECTED_LANGUAGE_MODEL_ROPE_THETA = 1_000_000.0
