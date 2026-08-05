"""MiniMax-H3 (joint video + audio DiT) support package.

Phase 1 (the load path): ``loader.py`` builds the four components -- the pruned
AdaLN-curve DiT, the 24-channel video VAE, the 32-channel audio VAE and the
50-layer Qwen3-VL text encoder -- from the ComfyUI-style flat tree, and
``core.model_loader`` detects and dispatches to it. The sampler
(``h3_pipeline_ops``), the pipeline-backend mixin, the block-loop wrapper and the
quantization registries are Phase 2/4.

The vendored diffusers ``minimax-h3`` model classes and scheduler live under
``vendor/``.
"""

from .loader import (
    MINIMAX_H3_AUDIO_LATENT_CHANNELS,
    MINIMAX_H3_AUDIO_LATENT_RATE,
    MINIMAX_H3_AUDIO_SAMPLE_RATE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_LATENT_CHANNELS,
    MINIMAX_H3_PIPELINE_CLASS,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_VAE_SPATIAL_COMPRESSION,
    MINIMAX_H3_VAE_TEMPORAL_COMPRESSION,
    MINIMAX_H3_VAE_TILING_POLICY,
    MINIMAX_H3_VIDEO_VAE_DTYPE,
    detect_minimax_h3_layout,
    is_minimax_h3_safetensors,
    keys_look_minimax_h3,
    load_minimax_h3_from_path,
    minimax_h3_latent_frames,
)

__all__ = [
    "MINIMAX_H3_AUDIO_LATENT_CHANNELS",
    "MINIMAX_H3_AUDIO_LATENT_RATE",
    "MINIMAX_H3_AUDIO_SAMPLE_RATE",
    "MINIMAX_H3_FPS",
    "MINIMAX_H3_LATENT_CHANNELS",
    "MINIMAX_H3_PIPELINE_CLASS",
    "MINIMAX_H3_PIXEL_MEAN",
    "MINIMAX_H3_PIXEL_STD",
    "MINIMAX_H3_VAE_SPATIAL_COMPRESSION",
    "MINIMAX_H3_VAE_TEMPORAL_COMPRESSION",
    "MINIMAX_H3_VAE_TILING_POLICY",
    "MINIMAX_H3_VIDEO_VAE_DTYPE",
    "detect_minimax_h3_layout",
    "is_minimax_h3_safetensors",
    "keys_look_minimax_h3",
    "load_minimax_h3_from_path",
    "minimax_h3_latent_frames",
]
