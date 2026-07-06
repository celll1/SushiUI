"""
VAEComponentRegistry — arch-independent VAE component layer.  [SKELETON — P2]

Generalizes minit2i's ``VAE_REGISTRY`` (+ the ``"none"`` pixel-space sentinel /
``is_latent_vae``) and ``models/common/vae_store.py``, driven by a
``ComponentWiringSpec``. Pixel-space is expressed as ``latent_channels == 0`` in
the spec (branch on the flag, not on ``is_minit2i``).

Planned P2 surface:
    load_vae(spec, path) -> vae | None
    encode(vae, px, spec) -> latent
    normalize(latent, vae, spec) -> latent

P0/P1: skeleton only. Do NOT move minit2i code yet.
"""

from __future__ import annotations


def is_latent_vae(spec) -> bool:
    """True when the arch uses a latent VAE (spec.latent_channels > 0)."""
    return getattr(spec, "latent_channels", 0) > 0


def load_vae(spec, path):  # pragma: no cover - P2
    raise NotImplementedError("vae_registry.load_vae is implemented in phase P2")


def encode(vae, px, spec):  # pragma: no cover - P2
    raise NotImplementedError("vae_registry.encode is implemented in phase P2")


def normalize(latent, vae, spec):  # pragma: no cover - P2
    raise NotImplementedError("vae_registry.normalize is implemented in phase P2")
