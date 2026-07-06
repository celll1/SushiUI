"""VAEComponentRegistry — training-side re-export (plan A.4, P2).

The canonical implementation lives in ``core.models.components.vae_registry`` (the
shared arch-independent layer). This module re-exports it so the training-side API
surface named in the plan (``core.training.components.vae_registry``) is preserved.

Surface: ``load_vae(spec_or_type, ...)`` / ``normalize(latent, vae, spec)`` /
``is_latent_vae`` (pixel-space = ``latent_channels == 0``), plus the frozen
minit2i functions (``load_minit2i_vae`` / ``normalize_latent`` /
``denormalize_latent`` / ``VAE_REGISTRY`` / ``VAE_SCALE_FACTOR``).
"""

from __future__ import annotations

from core.models.components.vae_registry import (  # noqa: F401
    VAE_REGISTRY,
    VAE_SCALE_FACTOR,
    is_latent_vae,
    vae_latent_channels,
    preview_decoder_for,
    load_minit2i_vae,
    normalize_latent,
    denormalize_latent,
    load_vae,
    normalize,
    resolve_vae_dir,
    vae_identity,
)

__all__ = [
    "load_vae",
    "normalize",
    "is_latent_vae",
    "VAE_REGISTRY",
    "VAE_SCALE_FACTOR",
    "vae_latent_channels",
    "preview_decoder_for",
    "load_minit2i_vae",
    "normalize_latent",
    "denormalize_latent",
    "resolve_vae_dir",
    "vae_identity",
]
