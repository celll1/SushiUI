"""
Unified attention conduit for SushiUI.

ONE backend-agnostic entry point (:func:`dispatch_attention`) routes attention
across all model architectures to a selectable kernel (native SDPA /
FlashAttention / SageAttention), with capability guards, layout adaptation, and
native fallback. Adding a future backend (e.g. TQ) is a one-branch change in
``registry.py`` + ``backends.py`` -- no conduit edits.

Public API:
    * ``dispatch_attention``   -- the conduit.
    * ``AttentionMode``        -- INFERENCE / TRAINING.
    * ``normalize_backend``    -- string -> canonical backend key.
    * ``resolve_backend``      -- capability/MODE guards -> effective backend.
    * ``to_diffusers_backend`` -- canonical string -> diffusers registry string.
    * ``AttentionBackend`` / ``BACKENDS`` -- registry descriptors (advanced).
"""

from .config import (
    normalize_backend,
    resolve_backend,
    to_diffusers_backend,
)
from .dispatch import AttentionMode, dispatch_attention
from .registry import BACKENDS, AttentionBackend

__all__ = [
    "dispatch_attention",
    "AttentionMode",
    "normalize_backend",
    "resolve_backend",
    "to_diffusers_backend",
    "AttentionBackend",
    "BACKENDS",
]
