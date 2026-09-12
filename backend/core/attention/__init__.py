"""
Unified attention conduit for SushiUI.

The dense conduit routes Q/K/V to selectable kernels. Semantic mechanisms such
as an H3 video window are a separate contract and may choose a structured
kernel without changing what ``attention_backend`` means.

Public API:
    * ``dispatch_attention``   -- the conduit.
    * ``AttentionMode``        -- INFERENCE / TRAINING.
    * ``normalize_backend``    -- string -> canonical backend key.
    * ``known_backends`` / ``is_known_backend`` / ``validate_backend``
                               -- the accepted vocabulary, derived from the
                                  registry, for API-side validation.
    * ``observed_backends``    -- which backend(s) a generation actually ran.
    * ``resolve_backend``      -- capability/MODE guards -> effective backend.
    * ``to_diffusers_backend`` -- canonical string -> diffusers registry string.
    * ``AttentionBackend`` / ``BACKENDS`` -- registry descriptors (advanced).
"""

from .config import (
    is_known_backend,
    known_backends,
    normalize_backend,
    resolve_backend,
    to_diffusers_backend,
    validate_backend,
)
from .contracts import AttentionFallbackPolicy, AttentionMode
from .dispatch import dispatch_attention, dispatch_attention_varlen
from .mechanisms import AttentionMechanism, known_mechanisms, validate_mechanism
from .observed import begin_generation, observed_backends
from .planned import dispatch_planned_attention
from .registry import BACKENDS, AttentionBackend
from .sol import sol_attention_available

__all__ = [
    "dispatch_attention",
    "dispatch_attention_varlen",
    "dispatch_planned_attention",
    "AttentionMode",
    "AttentionFallbackPolicy",
    "AttentionMechanism",
    "normalize_backend",
    "resolve_backend",
    "to_diffusers_backend",
    "known_backends",
    "is_known_backend",
    "validate_backend",
    "known_mechanisms",
    "validate_mechanism",
    "begin_generation",
    "observed_backends",
    "AttentionBackend",
    "BACKENDS",
    "sol_attention_available",
]
