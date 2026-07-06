"""Arch-independent component layer (plan A.4, requirement #3).

Shared by BOTH generation (model_loader, pipeline) and training, so it lives under
``core/models/`` (alongside ``core/models/common/``), NOT under ``core/training/``.
``core/training/components/`` re-exports these to provide the training-side surface.

One module per component KIND:

  * ``wiring.py``         — ComponentWiringSpec dataclass + per-arch specs.
  * ``te_registry.py``    — TE registry (canonical home of sdxl_te_registry).
  * ``vae_registry.py``   — AutoencoderKL registry (canonical home of minit2i_vae);
                            re-exports the shared resolver from common/vae_store.
  * ``bridge_adapter.py`` — dim-bridge adapters (canonical home of SDXLTEAdapters).
  * ``vae_wrapper.py``    — SDXLVAEWrapper (canonical home of sdxl_vae_wrapper).

The old ``models/sdxl_*`` and ``models/minit2i/minit2i_vae`` modules are thin
re-export shims of these, so every existing import keeps working verbatim.
"""

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
