"""ComponentWiringSpec — training-side re-export (plan A.4, P2).

The canonical ComponentWiringSpec + per-arch specs now live in the shared
``core.models.components.wiring`` module (arch-independent layer, used by both
generation and training). This module re-exports them so the training-side API
surface named in the plan (``core.training.components.wiring``) is preserved.
"""

from __future__ import annotations

from core.models.components.wiring import (  # noqa: F401
    ComponentWiringSpec,
    SD15_WIRING,
    SDXL_WIRING,
    ZIMAGE_WIRING,
    ANIMA_WIRING,
    LENS_WIRING,
    IDEOGRAM4_WIRING,
    MINIT2I_WIRING,
    KREA2_WIRING,
    FLUX2_WIRING,
    LTX2_WIRING,
    ACESTEP_WIRING,
    MINIMAX_H3_WIRING,
    TemporalSpec,
    LTX2_TEMPORAL,
    MINIMAX_H3_TEMPORAL,
    TEMPORAL_SPECS,
    temporal_spec_for_arch,
)

__all__ = [
    "ComponentWiringSpec",
    "TemporalSpec",
    "LTX2_TEMPORAL",
    "MINIMAX_H3_TEMPORAL",
    "TEMPORAL_SPECS",
    "temporal_spec_for_arch",
    "SD15_WIRING",
    "SDXL_WIRING",
    "ZIMAGE_WIRING",
    "ANIMA_WIRING",
    "LENS_WIRING",
    "IDEOGRAM4_WIRING",
    "MINIT2I_WIRING",
    "KREA2_WIRING",
    "FLUX2_WIRING",
    "LTX2_WIRING",
    "ACESTEP_WIRING",
    "MINIMAX_H3_WIRING",
]
