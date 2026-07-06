"""Trainable text-encoder adapters for custom-TE SDXL — RE-EXPORT SHIM (P2).

The implementation now lives in ``core.models.components.bridge_adapter`` (the
arch-independent component layer, plan A.4). This module re-exports the SAME
``SDXLTEAdapters`` object so existing imports keep resolving identically and the
``sushi.te_adapter.*`` checkpoint key layout is byte-stable (plan F).
"""

from __future__ import annotations

from core.models.components.bridge_adapter import (  # noqa: F401
    _MLP,
    SDXLTEAdapters,
    BridgeAdapter,
)

__all__ = ["SDXLTEAdapters", "BridgeAdapter"]
