"""SDXL VAE Wrapper — RE-EXPORT SHIM (P2).

The implementation now lives in ``core.models.components.vae_wrapper`` (the
arch-independent component layer, plan A.4). This module re-exports the SAME
``SDXLVAEWrapper`` / ``get_sdxl_vae`` objects so every existing import keeps
resolving identically (zero caller churn, identity preserved).
"""

from core.models.components.vae_wrapper import (  # noqa: F401
    SDXLVAEWrapper,
    get_sdxl_vae,
)

__all__ = ["SDXLVAEWrapper", "get_sdxl_vae"]
