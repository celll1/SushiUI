"""
ComponentWiringSpec — arch-independent component-wiring layer (plan A.4).

Principle: one module per component KIND (te_registry / vae_registry /
bridge_adapter); each architecture WIRES components by its connection format via
a ComponentWiringSpec, it does not own component code.

Placement (P2 decision): this lives under ``core/models/components/`` — the
arch-independent component layer is shared by BOTH generation (model_loader,
pipeline) and training, so it MUST NOT live under ``core/training/``. It sits
alongside ``core/models/common/`` (the existing shared-helper precedent) and the
per-arch ``core/models/<arch>/`` packages. ``core/training/components/`` re-exports
these symbols to preserve the training-side API surface named in the plan.

Fields are frozen from the real archs (verified against base_trainer.py):

  * anima:   5D / 16ch VAE (encode_image branch ~5491), LLM (Qwen3) TE.
  * flux2:   16ch latents, flux packing, batchnorm-style VAE normalization.
  * krea2:   packed latents (krea_norm), 16ch.
  * minit2i: pixel-space => latent_channels=0 (no VAE).
  * sdxl:    dual TE (2048 hidden / 1280 pooled), sdxl_time_ids micro-cond.

Values here are best-effort scaffolding constants; the arch handlers (P3+) resolve
the real per-arch numbers at load time from ``pipeline._sushi_arch`` and fold
``sdxl_vae_type``/``sdxl_te_type`` in (plan A.4, R5). This module carries NO
behavior — it is pure spec data, so it does not affect cache namespaces (R5/R6).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class ComponentWiringSpec:
    # --- text-encoder side ---
    te_out_dim: Optional[int]        # encoder_hidden_states dim into the backbone
    te_pooled_dim: Optional[int]     # pooled/added-cond dim (SDXL 1280; None if unused)
    te_seq_packing: str              # "clip77" | "raw" | "llm"
    added_cond: Optional[str]        # "sdxl_time_ids" | None (micro-conditioning)
    # --- vae side ---
    latent_channels: int             # 4 (SD), 16 (anima/flux2/krea2), 0 => pixel-space (minit2i)
    latent_ndim: int                 # 4 or 5 (anima 5D)
    latent_packing: str              # "none" | "flux_pack" | "krea_norm"
    vae_scale_factor: int
    vae_norm: str                    # "shift_scale" | "batchnorm" | "identity"

    def replace(self, **changes) -> "ComponentWiringSpec":
        """Return a copy with fields overridden (the graft-expression helper,
        plan A.4 design test)."""
        return replace(self, **changes)


# --- Per-arch wiring specs (scaffolding constants; refined by arch handlers) ---

SD15_WIRING = ComponentWiringSpec(
    te_out_dim=768, te_pooled_dim=None, te_seq_packing="clip77", added_cond=None,
    latent_channels=4, latent_ndim=4, latent_packing="none",
    vae_scale_factor=8, vae_norm="shift_scale",
)

SDXL_WIRING = ComponentWiringSpec(
    te_out_dim=2048, te_pooled_dim=1280, te_seq_packing="clip77", added_cond="sdxl_time_ids",
    latent_channels=4, latent_ndim=4, latent_packing="none",
    vae_scale_factor=8, vae_norm="shift_scale",
)

ZIMAGE_WIRING = ComponentWiringSpec(
    te_out_dim=None, te_pooled_dim=None, te_seq_packing="raw", added_cond=None,
    latent_channels=16, latent_ndim=4, latent_packing="none",
    vae_scale_factor=8, vae_norm="shift_scale",
)

ANIMA_WIRING = ComponentWiringSpec(
    te_out_dim=None, te_pooled_dim=None, te_seq_packing="llm", added_cond=None,
    latent_channels=16, latent_ndim=5, latent_packing="none",
    vae_scale_factor=8, vae_norm="shift_scale",
)

LENS_WIRING = ComponentWiringSpec(
    te_out_dim=None, te_pooled_dim=None, te_seq_packing="raw", added_cond=None,
    latent_channels=16, latent_ndim=4, latent_packing="none",
    vae_scale_factor=8, vae_norm="shift_scale",
)

IDEOGRAM4_WIRING = ComponentWiringSpec(
    te_out_dim=4096, te_pooled_dim=None, te_seq_packing="raw", added_cond=None,
    latent_channels=16, latent_ndim=4, latent_packing="none",
    vae_scale_factor=8, vae_norm="shift_scale",
)

MINIT2I_WIRING = ComponentWiringSpec(
    te_out_dim=1024, te_pooled_dim=None, te_seq_packing="raw", added_cond=None,
    latent_channels=0, latent_ndim=4, latent_packing="none",   # pixel-space, no VAE
    vae_scale_factor=1, vae_norm="identity",
)

KREA2_WIRING = ComponentWiringSpec(
    te_out_dim=2560, te_pooled_dim=None, te_seq_packing="raw", added_cond=None,
    latent_channels=16, latent_ndim=4, latent_packing="krea_norm",
    vae_scale_factor=8, vae_norm="shift_scale",
)

FLUX2_WIRING = ComponentWiringSpec(
    te_out_dim=None, te_pooled_dim=None, te_seq_packing="raw", added_cond=None,
    latent_channels=16, latent_ndim=4, latent_packing="flux_pack",
    vae_scale_factor=8, vae_norm="batchnorm",
)

# LTX-2.3 video: 128ch 5D latents (spatial /32, temporal /8), Gemma3 TE (3840)
# projected by LTX2TextConnectors to caption_channels=3840, audio VAE present.
LTX2_WIRING = ComponentWiringSpec(
    te_out_dim=3840, te_pooled_dim=None, te_seq_packing="llm", added_cond=None,
    latent_channels=128, latent_ndim=5, latent_packing="none",
    vae_scale_factor=32, vae_norm="identity",
)
