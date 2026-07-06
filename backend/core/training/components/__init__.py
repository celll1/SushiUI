"""
Arch-independent component layer (plan A.4, requirement #3).

One module per component KIND:

  * ``wiring.py``         — ComponentWiringSpec dataclass + per-arch specs.
  * ``te_registry.py``    — TE registry (re-exports models.components.te_registry).
  * ``vae_registry.py``   — VAE registry (re-exports models.components.vae_registry
                            + the shared common/vae_store resolver).
  * ``bridge_adapter.py`` — dim-bridge adapters (re-exports models.components).

P2 (this phase): the canonical implementations were generalized into the shared
``core/models/components/`` package (used by both generation and training); the
modules here re-export them to preserve the training-side API surface, and the old
``models/sdxl_*`` / ``models/minit2i/minit2i_vae`` modules became re-export shims.
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
from core.training.components.te_registry import (  # noqa: F401
    load_te,
    encode_text,
    load_sdxl_te,
)
from core.training.components.vae_registry import (  # noqa: F401
    load_vae,
    is_latent_vae,
)
from core.training.components.bridge_adapter import (  # noqa: F401
    BridgeAdapter,
    SDXLTEAdapters,
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
    "load_te",
    "encode_text",
    "load_sdxl_te",
    "load_vae",
    "is_latent_vae",
    "BridgeAdapter",
    "SDXLTEAdapters",
]
