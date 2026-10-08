"""Bound Qwen 2.1 image VAE work with scoped tiling policies."""
from contextlib import contextmanager

import torch

from api.param_defaults import QWEN21_TRAINING_ENCODE_DEFAULTS, QWEN21_DEBUG_DECODE_DEFAULTS


@contextmanager
def bounded_debug_decode(vae):
    from core.models.components.latent_adapter import NormalizedFrameVAE
    from core.models.qwen_image_21.vendor.autoencoder import AutoencoderKLQwenImage21
    codec = vae.inner if isinstance(vae, NormalizedFrameVAE) else vae
    if not isinstance(codec, AutoencoderKLQwenImage21):
        yield
        return
    settings = QWEN21_DEBUG_DECODE_DEFAULTS
    previous = {name: getattr(codec, name) for name in ("use_tiling", *settings)}
    try:
        codec.enable_tiling(**settings)
        yield
    finally:
        codec.clear_cache()
        for name, value in previous.items():
            setattr(codec, name, value)


@contextmanager
def bounded_image_encode(owner, vae, pixels):
    from core.models.components.latent_adapter import NormalizedFrameVAE
    from core.models.qwen_image_21.vendor.autoencoder import AutoencoderKLQwenImage21
    codec = vae.inner if isinstance(vae, NormalizedFrameVAE) else vae
    if not isinstance(codec, AutoencoderKLQwenImage21):
        yield
        return
    if pixels.ndim not in (4, 5) or (pixels.ndim == 5 and pixels.shape[2] != 1):
        yield
        return
    shape = tuple(pixels.shape)
    if pixels.device.type == "cuda" and getattr(owner, "_qwen21_encode_shape", shape) != shape:
        torch.cuda.empty_cache()
    owner._qwen21_encode_shape = shape
    settings = QWEN21_TRAINING_ENCODE_DEFAULTS
    previous = {name: getattr(codec, name) for name in ("use_tiling", *settings)}
    try:
        codec.enable_tiling(**settings)
        if not getattr(owner, "_qwen21_encode_tiling_logged", False):
            print(f"{getattr(owner, 'log_prefix', '[Training]')} Qwen 2.1 image encode: "
                  f"bounded tiling {settings['tile_sample_min_width']}x{settings['tile_sample_min_height']} "
                  "in patched encoder coordinates; decode policy preserved")
            owner._qwen21_encode_tiling_logged = True
        yield
    finally:
        for name, value in previous.items():
            setattr(codec, name, value)
        codec.clear_cache()
