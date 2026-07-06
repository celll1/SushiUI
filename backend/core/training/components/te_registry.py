"""TEComponentRegistry — training-side re-export (plan A.4, P2).

The canonical implementation lives in ``core.models.components.te_registry`` (the
shared arch-independent layer, used by both generation and training). This module
re-exports it so the training-side API surface named in the plan
(``core.training.components.te_registry``) is preserved.

Surface: ``load_te(spec_or_type, ...)`` and ``encode_text(...) -> (hidden, pooled)``
(plan A.4), plus the frozen ``load_sdxl_te`` / ``TE_REGISTRY`` / ``is_custom_te``.
"""

from __future__ import annotations

from core.models.components.te_registry import (  # noqa: F401
    TE_REGISTRY,
    is_custom_te,
    load_sdxl_te,
    encode_text,
    load_te,
)

__all__ = ["load_te", "encode_text", "load_sdxl_te", "TE_REGISTRY", "is_custom_te"]
