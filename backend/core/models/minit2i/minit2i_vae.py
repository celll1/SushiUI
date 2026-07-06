"""VAE registry + helpers for latent-space MiniT2I — RE-EXPORT SHIM (P2).

The implementation now lives in ``core.models.components.vae_registry`` (the
arch-independent component layer, plan A.4). This module re-exports the SAME
objects so every existing import — including the intra-package relative imports
``from .minit2i_vae import ...`` used by minit2i_loader / minit2i_pipeline_ops and
the absolute ``core.models.minit2i.minit2i_vae`` imports in base_trainer /
sdxl_custom_arch — keeps resolving to identical functions (identity preserved).

MiniT2I can train/infer either in pixel space (vae_type="none") or in a VAE latent
space (sdxl 4ch / flux1 16ch). See the component module for the full documentation.
"""

from __future__ import annotations

from core.models.components.vae_registry import (  # noqa: F401
    VAE_REGISTRY,
    VAE_SCALE_FACTOR,
    is_latent_vae,
    vae_latent_channels,
    preview_decoder_for,
    _local_candidates,
    load_minit2i_vae,
    _scale_shift,
    normalize_latent,
    denormalize_latent,
    load_vae,
    normalize,
)

__all__ = [
    "VAE_REGISTRY",
    "VAE_SCALE_FACTOR",
    "is_latent_vae",
    "vae_latent_channels",
    "preview_decoder_for",
    "load_minit2i_vae",
    "normalize_latent",
    "denormalize_latent",
]
