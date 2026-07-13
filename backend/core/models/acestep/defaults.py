"""Known-good component configs for ACE-Step 1.5.

These are the confirmed `config.json` contents shipped alongside the ACE-Step
1.5 HuggingFace repos (``acestep-v15-turbo`` for the DiT / lyric-condition
model, the bundled Oobleck VAE, and ``Qwen/Qwen3-Embedding-0.6B`` for the text
encoder). The local ComfyUI-style checkpoint tree at the model root
(``diffusion_models/`` + ``vae/`` + ``text_encoders/``) ships bare
``.safetensors`` files with no accompanying ``config.json``, so these dicts
are the loader's only source of architecture parameters.

DiT variants (base / sft / turbo) share an IDENTICAL 677-tensor state_dict
(verified 2026-07-13: `is_turbo` / `model_version` are metadata-only fields,
never read by the vendored modeling code), so a single config works for all
three; the safetensors filename alone selects the checkpoint.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# AceStepConfig kwargs (2B DiT: base / sft / turbo — architecture-identical).
# ---------------------------------------------------------------------------
ACESTEP_DIT_CONFIG: dict = dict(
    vocab_size=64003,
    fsq_dim=2048,
    fsq_input_levels=[8, 8, 8, 5, 5, 5],
    fsq_input_num_quantizers=1,
    hidden_size=2048,
    intermediate_size=6144,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    tie_word_embeddings=True,
    rope_theta=1000000,
    rope_scaling=None,
    attention_bias=False,
    use_sliding_window=True,
    sliding_window=128,
    layer_types=None,  # AceStepConfig fills the sliding/full alternation itself
    attention_dropout=0.0,
    num_lyric_encoder_hidden_layers=8,
    audio_acoustic_hidden_dim=64,
    pool_window_size=5,
    text_hidden_dim=1024,
    in_channels=192,
    data_proportion=0.5,
    timestep_mu=-0.4,
    timestep_sigma=1.0,
    timbre_hidden_dim=64,
    num_timbre_encoder_hidden_layers=4,
    timbre_fix_frame=750,
    patch_size=2,
    num_attention_pooler_hidden_layers=2,
    num_audio_decoder_hidden_layers=24,
    model_version="turbo",
)

# ---------------------------------------------------------------------------
# AutoencoderOobleck kwargs (ACE-Step 1.5 VAE: 48kHz stereo, 64-dim latent,
# 25Hz latent frame rate == 48000 / prod(downsampling_ratios) / patch_size(2)
# note: the *VAE* hop_length itself is prod(downsampling_ratios) = 1920 ->
# 25Hz raw-latent rate; the DiT's patch_size=2 halves that again to 12.5Hz
# token rate inside the transformer, not exposed at the VAE boundary).
# ---------------------------------------------------------------------------
ACESTEP_VAE_CONFIG: dict = dict(
    encoder_hidden_size=128,
    downsampling_ratios=[2, 4, 4, 6, 10],
    channel_multiples=[1, 2, 4, 8, 16],
    decoder_channels=128,
    decoder_input_channels=64,
    audio_channels=2,
    sampling_rate=48000,
)

# ---------------------------------------------------------------------------
# Qwen3Config kwargs (text encoder: Qwen3-Embedding-0.6B, ACE-Step-tuned
# weights at text_encoders/qwen_0.6b_ace15.safetensors). Loaded as the bare
# transformers.Qwen3Model (encoder-only; no lm_head in the checkpoint).
# ---------------------------------------------------------------------------
ACESTEP_TEXT_ENCODER_CONFIG: dict = dict(
    vocab_size=151669,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=True,
    tie_word_embeddings=True,
    rope_theta=1000000,
    rope_scaling=None,
    attention_bias=False,
    use_sliding_window=False,
    sliding_window=None,
    bos_token_id=151643,
    eos_token_id=151643,
)

# Public HF hub id whose tokenizer is byte-identical (vocab_size matches
# exactly: 151669) to the ACE-Step-tuned Qwen3 text encoder. Used as the
# fallback source for tokenizer files when no local sibling is found.
QWEN3_EMBEDDING_TOKENIZER_HUB_ID = "Qwen/Qwen3-Embedding-0.6B"

SAMPLE_RATE = 48000
# VAE hop_length: prod(downsampling_ratios) = 2*4*4*6*10 = 1920 -> 48000/1920 = 25Hz
LATENT_FRAME_RATE = 25
LATENT_CHANNELS = 64

# ---------------------------------------------------------------------------
# Prompt-assembly constants (Phase 2: txt2aud). Mirrored verbatim from the
# official ACE-Step 1.5 repo's ``acestep/constants.py`` (DEFAULT_DIT_INSTRUCTION,
# SFT_GEN_PROMPT) so the DiT's text-conditioning branch sees the same prompt
# shape it was trained on.
# ---------------------------------------------------------------------------
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

# Reference-audio timbre latent length (30s @ 25Hz) used for the silence
# timbre condition in text2music (no reference audio). Matches
# ACESTEP_DIT_CONFIG["timbre_fix_frame"] and the official
# ``infer_refer_latent``'s ``self.silence_latent[:, :750, :]`` slice.
SILENCE_LATENT_FRAMES = 750
