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
    latent_channels=32, latent_ndim=4, latent_packing="none",  # AutoencoderKLFlux2 (verified vae/config.json)
    vae_scale_factor=8, vae_norm="shift_scale",
)

IDEOGRAM4_WIRING = ComponentWiringSpec(
    te_out_dim=4096, te_pooled_dim=None, te_seq_packing="raw", added_cond=None,
    latent_channels=32, latent_ndim=4, latent_packing="none",  # AutoencoderKLFlux2 (verified vae/config.json)
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
    latent_channels=32, latent_ndim=4, latent_packing="flux_pack",  # AutoencoderKLFlux2 (vae_store flux2=32)
    vae_scale_factor=8, vae_norm="batchnorm",
)

# LTX-2.3 video: 128ch 5D latents (spatial /32, temporal /8), Gemma3 TE (3840)
# projected by LTX2TextConnectors to caption_channels=3840, audio VAE present.
LTX2_WIRING = ComponentWiringSpec(
    te_out_dim=3840, te_pooled_dim=None, te_seq_packing="llm", added_cond=None,
    latent_channels=128, latent_ndim=5, latent_packing="none",
    vae_scale_factor=32, vae_norm="identity",
)

# ACE-Step 1.5 (turbo): 64ch TEMPORAL-ONLY latents [B, T, 64] (Oobleck VAE,
# 48kHz stereo, hop=1920 -> 25Hz raw latent rate; the DiT additionally
# patch_size=2 halves that to 12.5Hz inside the transformer, not exposed at
# the VAE boundary). Qwen3-Embedding-0.6B TE (1024-dim) feeds BOTH the
# "# Caption" text conditioning and the lyric embedding table
# (embed_tokens only, no lyric transformer forward). latent_ndim=3 reflects
# [B, T, C] (no spatial H/W axis, unlike every image/video arch above).
ACESTEP_WIRING = ComponentWiringSpec(
    te_out_dim=1024, te_pooled_dim=None, te_seq_packing="raw", added_cond=None,
    latent_channels=64, latent_ndim=3, latent_packing="none",
    vae_scale_factor=1920, vae_norm="identity",
)

# MiniMax-H3 (joint video + audio): 24ch 5-D video latents from a causal VAE that
# compresses 16x spatially and 4x temporally, plus a SEPARATE 32ch audio VAE that
# this single-latent spec cannot express (the audio geometry travels in the
# component dict and in `minimax_h3.loader`'s constants).
#
# `te_out_dim` 5120 is the Qwen3-VL-32B hidden size, read raw after decoder layer
# 50 -- there is no pooling and no per-modality connector.
#
# `vae_scale_factor` is the VAE's own SPATIAL compression, 16, matching what the
# field means for every other arch here (LTX-2.3's 32 is that model's VAE ratio).
# It is deliberately NOT 32: the extra factor of 2 the DiT sees comes from the
# transformer's own 2x2 patchify, so the canvas alignment constraint (/32) is a
# `pixel_align` property, not a VAE scale factor. The design document said 32;
# the file's own embedded source_config says `"vae_ratio": 16, "vae_ratio_t": 4`
# and the loaded module agrees.
#
# `vae_norm` is "shift_scale": 24 per-channel mean/std vectors, `(x - mean) / std`
# on encode. NOT "identity" like LTX-2.3 -- and note the pixel convention on the
# other side of the VAE is ImageNet-normalised RGB over [0, 1], not [-1, 1],
# which no field of this spec expresses (see `minimax_h3.loader`).
MINIMAX_H3_WIRING = ComponentWiringSpec(
    te_out_dim=5120, te_pooled_dim=None, te_seq_packing="llm", added_cond=None,
    latent_channels=24, latent_ndim=5, latent_packing="none",
    vae_scale_factor=16, vae_norm="shift_scale",
)
