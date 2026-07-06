"""BridgeAdapter — training-side re-export (plan A.4, P2).

The canonical implementation lives in ``core.models.components.bridge_adapter``
(the shared arch-independent layer). This module re-exports it so the training-side
API surface named in the plan (``core.training.components.bridge_adapter``) is
preserved.

Surface: ``BridgeAdapter(in_dim, out_dim, in_pooled, out_pooled)`` (the arch-
independent generalization), plus the frozen ``SDXLTEAdapters`` (whose submodule
names back the ``sushi.te_adapter.*`` checkpoint keys, plan F).
"""

from __future__ import annotations

from core.models.components.bridge_adapter import (  # noqa: F401
    BridgeAdapter,
    SDXLTEAdapters,
)

__all__ = ["BridgeAdapter", "SDXLTEAdapters"]
