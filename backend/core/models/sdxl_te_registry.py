"""Text-encoder registry for custom-TE SDXL — RE-EXPORT SHIM (P2).

The implementation now lives in ``core.models.components.te_registry`` (the
arch-independent component layer, plan A.4). This module re-exports the SAME
objects so every existing import (``from core.models.sdxl_te_registry import
load_sdxl_te`` / ``encode_text`` / ``TE_REGISTRY`` / ``is_custom_te``) keeps
resolving to the identical functions — zero caller churn, identity preserved.
"""

from __future__ import annotations

from core.models.components.te_registry import (  # noqa: F401
    TE_REGISTRY,
    _DEFAULT_MAX_LEN,
    is_custom_te,
    _find_position_embedding,
    _extend_position_embeddings,
    load_sdxl_te,
    encode_text,
    load_te,
)

__all__ = [
    "TE_REGISTRY",
    "is_custom_te",
    "load_sdxl_te",
    "encode_text",
    "load_te",
]
