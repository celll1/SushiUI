"""
Arch-independent component layer (plan A.4, requirement #3).

One module per component KIND:

  * ``wiring.py``         — ComponentWiringSpec dataclass + per-arch specs.
  * ``te_registry.py``    — TEComponentRegistry (generalizes sdxl_te_registry).  [P2]
  * ``vae_registry.py``   — VAEComponentRegistry (generalizes minit2i VAE_REGISTRY
                            + models/common/vae_store).                          [P2]
  * ``bridge_adapter.py`` — dim-bridge adapters (generalizes SDXLTEAdapters).    [P2]

P0/P1: ``wiring`` is real; the registries/bridge are documented skeletons filled
in P2. No sdxl/minit2i code is moved yet.
"""

from core.training.components.wiring import (  # noqa: F401
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
)

__all__ = [
    "ComponentWiringSpec",
    "SD15_WIRING",
    "SDXL_WIRING",
    "ZIMAGE_WIRING",
    "ANIMA_WIRING",
    "LENS_WIRING",
    "IDEOGRAM4_WIRING",
    "MINIT2I_WIRING",
    "KREA2_WIRING",
    "FLUX2_WIRING",
]
