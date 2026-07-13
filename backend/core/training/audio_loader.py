"""
Audio clip loader for temporal (ACE-Step) training (Phase 8a).

Decodes an audio file to a normalized ``[C, samples]`` waveform tensor (stereo,
48kHz, ``[-1, 1]``) and hands it to an arch-owned VAE-encode callable, mirroring
``video_loader.py``'s ``load_clip`` / ``encode_and_cache_clip`` seam.

Design constraints (mirrors video_loader.py's docstring):
  - Decoder is ``soundfile`` first (libsndfile; fast, no full-container decode),
    ``torchaudio.load()`` fallback -- the SAME resolution order the ACE-Step
    inference backend uses (see
    ``core.pipeline_backends.acestep.AceStepMixin._acestep_load_reference_audio``),
    reused here directly rather than reimplemented.
  - Stereo/48kHz normalization reuses
    ``AceStepMixin._acestep_normalize_stereo_48k`` (mono->stereo duplication,
    ``[:2]`` channel take, resample iff ``sr != 48000``, clamp to ``[-1, 1]``,
    no loudness normalization) -- the same static helper the inference backend
    uses, so training and inference see byte-identical waveform preprocessing.
  - Optional fixed-length windowing (``clip_seconds``): the clip is taken from
    the START of the file (no random-crop windowing, unlike video's
    ``sample_clip_window`` -- ACE-Step LoRA datasets are expected to already be
    pre-trimmed to a consistent per-item duration; random-window cropping is a
    documented follow-up, not implemented here). Short files are NOT loop-padded
    (unlike video's loop-last-frame): the file's own (shorter) length is used
    as-is, so batches must be pre-trimmed to a uniform duration by the dataset
    author for now (no audio bucket-manager yet -- see ``base_trainer.py``'s
    ``acestep_audio_batches`` grouping, which batches by encoded latent frame
    count instead of enforcing a fixed duration up front).
"""

from typing import Optional

import torch


def load_audio(
    audio_path: str,
    clip_seconds: Optional[float] = None,
    sample_rate: int = 48000,
) -> torch.Tensor:
    """Decode an audio file to a ``[2, samples]`` float32 CPU tensor in ``[-1, 1]``.

    Args:
        audio_path: Path to the source audio file.
        clip_seconds: If given, truncate to the first ``clip_seconds`` seconds
            (at ``sample_rate``). Files shorter than this are returned as-is
            (not loop-padded).
        sample_rate: Target sample rate (resampled from source if it differs).

    Returns:
        Float tensor ``[2, samples]`` (stereo), ``[-1, 1]``.

    Raises:
        RuntimeError: if the file cannot be decoded by either backend.
    """
    from core.pipeline_backends.acestep import AceStepMixin

    wav, sr = AceStepMixin._acestep_load_reference_audio(audio_path)  # [C, samples]
    wav = AceStepMixin._acestep_normalize_stereo_48k(wav, sr)  # -> [2, samples] @ sample_rate(48k)

    if sample_rate != 48000:
        # AceStepMixin's helper is hardcoded to 48k (the ACE-Step VAE's native
        # rate); resample again only in the (currently unused) case a caller
        # requests something else.
        import torchaudio
        resampler = torchaudio.transforms.Resample(48000, sample_rate)
        wav = resampler(wav)

    if clip_seconds is not None:
        max_samples = int(round(float(clip_seconds) * sample_rate))
        if wav.shape[1] > max_samples:
            wav = wav[:, :max_samples]

    return wav.contiguous()


def encode_and_cache_audio(
    *,
    cache,
    audio_path: str,
    clip_seconds: Optional[float],
    vae_encode_audio,
    sample_rate: int = 48000,
    device: str = "cuda",
    skip_existing: bool = True,
):
    """Encode-integration SEAM for ACE-Step training (mirrors
    ``video_loader.encode_and_cache_clip``).

    This is the single entry point the base trainer's latent-cache pass calls
    to turn an audio file into a cached 3D latent. It deliberately does NOT
    load or reference the ACE-Step VAE itself -- the caller passes
    ``vae_encode_audio``, a callable it owns that runs the loaded ACE-Step
    Oobleck VAE (``arch.vae_encode_audio``).

    Flow:
      1. Cache hit -> return the cached 3D latent ``[1, T, 64]``.
      2. Miss -> ``load_audio(...) -> [2, samples]`` (stereo, 48kHz, [-1, 1]).
      3. ``vae_encode_audio(waveform)`` -> 3D latent ``[1, T, 64]``.
      4. ``cache.save_audio_latent(...)`` and return the latent.

    Expected ``vae_encode_audio`` signature (arch/acestep.py implements):
        ``vae_encode_audio(waveform: torch.Tensor[2, samples]) -> torch.Tensor[1, T, 64]``

    Args:
        cache: A ``LatentCache`` instance.
        audio_path: Source audio file path.
        clip_seconds: Target clip duration in seconds, or None for full-length.
        vae_encode_audio: Arch-owned callable performing the ACE-Step VAE encode.
        sample_rate: Target sample rate (part of the cache key).
        device: Device to load a cache-hit latent onto.
        skip_existing: Passed through to ``save_audio_latent``.

    Returns:
        3D latent tensor ``[1, T, 64]``.
    """
    cached = cache.load_audio_latent(audio_path, clip_seconds, sample_rate, device=device)
    if cached is not None:
        return cached

    waveform = load_audio(audio_path, clip_seconds=clip_seconds, sample_rate=sample_rate)  # [2, samples]

    latents = vae_encode_audio(waveform)  # arch-owned; -> [1, T, 64]

    if latents.dim() != 3:
        raise ValueError(
            f"[AudioLoader] vae_encode_audio must return a 3D latent "
            f"[1, T, 64], got {latents.dim()}D {tuple(latents.shape)}"
        )

    cache.save_audio_latent(
        audio_path, clip_seconds, sample_rate, latents, skip_existing=skip_existing,
    )
    return latents
