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
from typing import Callable, Dict, List, Optional, Tuple


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


# ---------------------------------------------------------------------------
# TemporalSpec — the per-arch clip-length / frame-rate / canvas contract of a
# VIDEO architecture. Declarative, so bucketing, the video loader, route
# validation and the frontend all read one table instead of growing their own
# `if arch == ...`.
#
# SCOPE OF THIS REVISION (Phase 2 of the MiniMax-H3 integration): the
# GENERATION side consumes this — route validation and the `video_constraints`
# block of `GET /schema/arch-capabilities`. Threading it through the TRAINING
# call chain (`VideoBucketManager`, `video_loader.load_clip` /
# `encode_and_cache_clip`, the trainer's clip-encode sites, the clip cache key
# and 24 fps resampling) is a separate, larger refactor of shared LTX-serving
# code and is deliberately NOT started here; until it lands those functions keep
# their current hardcoded LTX-2.3 rule.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalSpec:
    """Valid clip lengths, frame rate and canvas envelope of a video arch.

    Valid clip lengths are ``k * frame_multiple + frame_offset``. Two different
    floors exist on purpose and must not be merged:

    * ``min_decodable_frames`` is a HARD VAE floor — below it the decoder
      cannot produce frames at all, so nothing (not training, not a smoke test)
      may go under it;
    * ``min_frames`` / ``max_frames`` are the PRODUCTION generation bounds, i.e.
      the range the released model was trained and documented for. Training
      clips and previews are validated against the grid and the decodable floor
      instead, so a short training bucket is not a violation of these.
    """

    frame_multiple: int
    frame_offset: int
    min_frames: int
    max_frames: Optional[int]
    min_decodable_frames: int
    # Pixel frames -> latent frames. Closed form; measured per arch.
    latent_frames: Callable[[int], int]
    # The arch's own frame rate when it has one (MiniMax-H3 is a fixed 24 fps
    # model); None means the source/native rate is preserved (LTX-2.3).
    fps_fixed: Optional[float]
    # Clip lengths the training bucketer offers by default. Exempt from
    # min/max_frames by construction (see the class docstring).
    default_clip_lengths: Tuple[int, ...]
    # Spatial: the multiple both axes round to, and an orientation-agnostic
    # envelope (`(short, long)`) when the arch has one.
    pixel_align: int = 32
    max_pixel_hw: Optional[Tuple[int, int]] = None
    # What route validation does with an off-grid or out-of-range clip length:
    # snap it (and warn), or refuse the request. This is a per-arch API contract,
    # not a preference, so it is declared here rather than branched on at the
    # route: LTX-2.3 has answered an invalid `num_frames` with a 400 since it
    # shipped and its openapi documents that, while MiniMax-H3's own reference
    # implementation rounds up to the next encodable length with a warning.
    snap_invalid_length: bool = False
    # Environment gate that lowers the PRODUCTION floor to the decodable floor.
    # Grid points below `min_frames` (MiniMax-H3: 22 ... 107) are valid for the
    # VAE and are what a smoke test or a training clip uses; they are simply not
    # what the released model was trained to generate, so they must not be
    # reachable through ordinary API validation. Set the variable in a shell that
    # is deliberately running a short clip; the request still warns.
    smoke_override_env: str = "SUSHI_TEMPORAL_SMOKE"

    def floor(self, smoke: bool = False) -> int:
        """The effective minimum clip length -- production, or the VAE floor."""
        return self.min_decodable_frames if smoke else max(self.min_frames, self.min_decodable_frames)

    def is_valid_length(self, num_frames: int) -> bool:
        """True when ``num_frames`` is on the grid and decodable."""
        return (
            num_frames >= self.min_decodable_frames
            and (num_frames - self.frame_offset) % self.frame_multiple == 0
            and num_frames >= self.frame_offset
        )

    def snap_length(self, num_frames: int, smoke: bool = False) -> int:
        """The nearest valid length inside the production bounds.

        Ties go DOWN (the shorter clip), so a snap never silently costs more
        compute than the caller asked for.
        """
        lo = self.floor(smoke)
        hi = self.max_frames if self.max_frames is not None else max(lo, num_frames)
        # Round to the grid, then clamp into [lo, hi] on the grid.
        k = round((num_frames - self.frame_offset) / self.frame_multiple)
        candidates = {k - 1, k, k + 1}
        lengths = sorted(
            length
            for length in (c * self.frame_multiple + self.frame_offset for c in candidates)
            if lo <= length <= hi
        )
        if not lengths:
            # The request is outside the bounds entirely: clamp to the nearest
            # in-range grid point.
            k_lo = -(-(lo - self.frame_offset) // self.frame_multiple)
            k_hi = (hi - self.frame_offset) // self.frame_multiple
            k = min(max(k, k_lo), k_hi)
            return k * self.frame_multiple + self.frame_offset
        return min(lengths, key=lambda length: (abs(length - num_frames), length))

    def suggested_lengths(self, count: int = 8) -> List[int]:
        """In-range valid lengths, for a client building a clip-length list."""
        lo = max(self.min_frames, self.min_decodable_frames)
        k = -(-(lo - self.frame_offset) // self.frame_multiple)
        out: List[int] = []
        while len(out) < count:
            length = k * self.frame_multiple + self.frame_offset
            if self.max_frames is not None and length > self.max_frames:
                break
            out.append(length)
            k += 1
        return out


# LTX-2.3: `(L - 1) % 8 == 0`, native fps preserved, no canvas cap. These are
# the values the existing hardcoded checks already enforce, restated
# declaratively; nothing about LTX-2.3's behaviour changes by their presence.
LTX2_TEMPORAL = TemporalSpec(
    frame_multiple=8, frame_offset=1, min_frames=1, max_frames=None,
    min_decodable_frames=1, latent_frames=lambda t: (t - 1) // 8 + 1,
    fps_fixed=None, default_clip_lengths=(9, 17, 25, 33, 49),
    pixel_align=32, max_pixel_hw=None, snap_invalid_length=False,
)

# MiniMax-H3. Everything here is MEASURED (Phase 0):
#   * valid T = 17n + 5 for n >= 1 -- T = 5 is on the grid but its 2 latent
#     frames cannot be decoded (`num_chunks` = 0), so 22 frames / 0.917 s is the
#     hard decodable floor;
#   * latent_frames(T) = ceil(T/17)*5 - 3 (ComfyUI's own formula agrees only ON
#     the grid and disagrees off it, so this form is the one to use);
#   * production bounds 124-345: ComfyUI's node pins the trained range at
#     ~124-362 and the official README states 4-15 s output, and 345 = 17*20+5
#     = 14.375 s is the largest grid point <= 15 s. The README's 4 s figure
#     (107 frames) describes the hosted product; the API floor follows the
#     trained-range floor instead, and the discrepancy is recorded here rather
#     than left to look like an oversight.
MINIMAX_H3_TEMPORAL = TemporalSpec(
    frame_multiple=17, frame_offset=5, min_frames=124, max_frames=345,
    min_decodable_frames=22,
    latent_frames=lambda t: 1 if t <= 1 else -(-t // 17) * 5 - 3,
    fps_fixed=24.0, default_clip_lengths=(22, 39),
    pixel_align=32, max_pixel_hw=(768, 1344), snap_invalid_length=True,
)

TEMPORAL_SPECS: Dict[str, TemporalSpec] = {
    "ltx2": LTX2_TEMPORAL,
    "minimax_h3": MINIMAX_H3_TEMPORAL,
}


def temporal_spec_for_arch(arch: Optional[str]) -> Optional[TemporalSpec]:
    """The arch's ``TemporalSpec``, or None for an image/audio architecture."""
    return TEMPORAL_SPECS.get(arch or "")
