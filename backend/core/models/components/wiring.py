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
# CONSUMED BY BOTH SIDES as of Phase 6a:
#   * generation — route validation and the `video_constraints` block of
#     `GET /schema/arch-capabilities`;
#   * training — `bucketing`'s temporal section, `video_loader` (window
#     sampling, `load_clip`, `encode_and_cache_clip`), the clip cache key
#     (`LatentCache.compute_clip_hash`) and the `base_trainer` clip-encode
#     sites, all of which take an explicit `spec` parameter and fall back to
#     the LTX-2.3 rule when it is absent. `ArchHandler.temporal` is where a
#     trainer reads its arch's spec (the temporal analogue of `pixel_align`).
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
    # `snap_length` rounds UP for the same reason.
    snap_invalid_length: bool = False
    # Step-count contract of the arch's SCHEDULER. Two different facts:
    #
    # * `min_inference_steps` is the smallest `num_inference_steps` the
    #   scheduler accepts. Route validation refuses anything below it with a 400
    #   BEFORE the text encoder runs, because the alternative is a 500 raised
    #   from inside the sampler after a full (tens-of-seconds) text encode.
    # * `steps_are_sigma_grid_points` says how the count maps to model
    #   evaluations. LTX-2.3's FlowMatchEulerDiscreteScheduler builds N
    #   timesteps and appends the terminal sigma separately, so N steps run N
    #   evaluations and N=1 is legal. MiniMax-H3's scheduler builds a
    #   `linspace(1, 0, N)` sigma grid with the terminal 0 INCLUDED and sets
    #   `timesteps = 1 - sigmas[:-1]`, so N grid points run N-1 evaluations and
    #   N=1 would run none -- which is exactly why its minimum is 2.
    #
    # They are declared separately rather than derived from each other: the
    # minimum is a validation bound and could have another cause on a future
    # arch, while the mapping is client-visible semantics either way.
    min_inference_steps: int = 1
    steps_are_sigma_grid_points: bool = False
    # Where `POST /generate/outpaint/video` may place the input clip on this
    # architecture. LTX-2.3's `LTX2VideoCondition.index` addresses an arbitrary
    # latent frame and carries a whole clip's frames, so the input goes anywhere
    # ("free"); MiniMax-H3's outpaint path hands the model the first and/or last
    # frame of the span it generates and has no denoising-strength v2v, so the
    # preserved clip must abut a boundary ("extend_forward" /
    # "extend_backward") or sit at both ends of a generated gap ("bridge", two
    # uploads). An offset that is neither is refused rather than approximated by
    # a nearby placement.
    #
    # NOT because MiniMax-H3 lacks index-addressable conditioning -- it has it,
    # measured, and /generate/img2vid places keyframes with it (an anchor's
    # rotary time is `num_text_tokens + (5/3)*f` for any pixel frame f). What is
    # unmeasured is the OUTPAINT shape: a preserved clip anchored mid-span with
    # exact preservation around it. This list is that scope, not an
    # architectural limit; `generation_utils.plan_video_outpaint_placement`
    # states the same thing in the refusal a client sees.
    outpaint_placements: Tuple[str, ...] = ("free",)
    # How many PIXEL frames each LATENT frame covers, cycling. MiniMax-H3's
    # video VAE chunks time as (1, 4, 4, 4, 4) repeating, so latent frame 0
    # carries pixel frame 0 alone, latent 1 carries 1..4, and so on.
    #
    # This is the addressable unit of `POST /generate/inpaint/video`: a
    # requested pixel range is expanded outward to these boundaries, because a
    # latent frame is pinned or generated as a whole. Empty means the arch has
    # not declared it and nothing may claim finer-than-clip temporal addressing
    # for it -- LTX-2.3 leaves it empty because no route reads it there.
    latent_chunk_pattern: Tuple[int, ...] = ()
    # The shortest length worth OFFERING to a client (None = the production
    # floor). Validity and suggestion are different questions: LTX-2.3's grid
    # starts at 1, and a 1-frame "video" is a valid request but not a clip
    # length any UI should list.
    suggested_min_frames: Optional[int] = None
    # Environment gate that lowers the PRODUCTION floor to the decodable floor.
    # Grid points below `min_frames` (MiniMax-H3: 22 ... 107) are valid for the
    # VAE and are what a smoke test or a training clip uses; they are simply not
    # what the released model was trained to generate, so they must not be
    # reachable through ordinary API validation. Set the variable in a shell that
    # is deliberately running a short clip; the request still warns.
    smoke_override_env: str = "SUSHI_TEMPORAL_SMOKE"
    # Environment gate that RAISES the PRODUCTION ceiling past `max_frames`,
    # the mirror image of `smoke_override_env` on the other end of the range.
    # `max_frames` is a DOCUMENTED bound (see `MINIMAX_H3_TEMPORAL`'s comment),
    # not a hard decoder limit the way `min_decodable_frames` is, so a caller
    # who deliberately wants to probe past it may -- but every request that
    # lands past `max_frames` while this is set is warned as untested, because
    # nothing here has measured that quality holds beyond the documented top.
    # None means the arch has not opted in (only MiniMax-H3 sets it; LTX-2.3's
    # `max_frames` is already None, i.e. already uncapped).
    max_override_env: Optional[str] = None

    @property
    def resample_policy(self) -> str:
        """How training turns a source video into this arch's frame rate.

        ``"index"`` — the legacy LTX-2.3 rule: sampled source frames are
        ``start_frame + i*stride`` and the clip simply INHERITS the source fps
        (which is then carried per-sample into the RoPE coords).

        ``"timestamp_nearest"`` — the arch has a fixed frame rate, so target
        frame ``i`` is the source frame nearest ``start_time + i*stride /
        fps_fixed``. Without this, a 30 fps window labelled "24 fps" plays 25 %
        fast and its audio desynchronises.

        This string is part of the clip cache key, so a cache built under one
        policy can never be read as if it had been built under the other.
        """
        return "index" if self.fps_fixed is None else "timestamp_nearest"

    def clip_duration(self, clip_length: int, stride: int = 1) -> Optional[float]:
        """Seconds of OUTPUT timeline a clip occupies, or None when the arch has
        no fixed rate (LTX-2.3 clips are as long as their source says)."""
        if self.fps_fixed is None:
            return None
        return (max(1, int(clip_length)) * max(1, int(stride))) / float(self.fps_fixed)

    def floor(self, smoke: bool = False) -> int:
        """The effective minimum clip length -- production, or the VAE floor."""
        return self.min_decodable_frames if smoke else max(self.min_frames, self.min_decodable_frames)

    def ceiling(self, uncapped: bool = False) -> Optional[int]:
        """The effective maximum clip length -- production, or lifted by
        `max_override_env`. ``uncapped=True`` removes the grid's TOP bound
        only; `is_valid_length` and the grid arithmetic are unaffected, so an
        uncapped request still has to land on ``k * frame_multiple +
        frame_offset``, it is just no longer clamped to `max_frames`."""
        return None if uncapped else self.max_frames

    def is_valid_length(self, num_frames: int) -> bool:
        """True when ``num_frames`` is on the grid and decodable."""
        return (
            num_frames >= self.min_decodable_frames
            and (num_frames - self.frame_offset) % self.frame_multiple == 0
            and num_frames >= self.frame_offset
        )

    def snap_length(self, num_frames: int, smoke: bool = False, uncapped: bool = False) -> int:
        """The next valid length AT OR ABOVE ``num_frames``, inside the bounds.

        Rounds UP, which is what MiniMax-H3's own reference implementation does
        (`align_num_frames` rounds a requested length up to the next encodable
        one): a snap therefore never drops content the caller asked for. It is
        clamped into ``[floor, ceiling(uncapped)]`` on the grid, so an
        over-long request still lands on the largest length the model can
        generate -- unless ``uncapped`` lifts that top, in which case the
        request only rounds up onto the grid and is not clamped at all. Pass
        ``uncapped`` from the SAME environment check the caller used to decide
        whether to accept the request in the first place: this method has no
        access to `max_override_env` on its own, so a caller that decides
        "uncapped" but calls this with the default `False` silently re-imposes
        the cap it meant to lift.
        """
        lo = self.floor(smoke)
        hi = self.ceiling(uncapped)
        # Ceiling division onto the grid, then clamp into [lo, hi] on the grid.
        k = -(-(num_frames - self.frame_offset) // self.frame_multiple)
        k_lo = -(-(lo - self.frame_offset) // self.frame_multiple)
        k = max(k, k_lo)
        if hi is not None:
            k = min(k, (hi - self.frame_offset) // self.frame_multiple)
        return k * self.frame_multiple + self.frame_offset

    def suggested_lengths(self, count: int = 8) -> List[int]:
        """In-range valid lengths, for a client building a clip-length list.

        Starts at ``suggested_min_frames`` where the arch sets one — a length
        can be VALID without being worth offering in a clip-length dropdown
        (LTX-2.3 accepts a 1-frame clip, which is a still image).
        """
        lo = max(self.min_frames, self.min_decodable_frames,
                 self.suggested_min_frames or 0)
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
    # A 1-frame clip stays VALID (nothing about LTX-2.3's validation changes);
    # it is simply not suggested. 9 = 8*1 + 1 is the shortest length that is a
    # clip rather than a still, and is where `default_clip_lengths` starts too.
    suggested_min_frames=9,
    # The step-count fields keep their defaults, MEASURED against the scheduler
    # this arch actually loads (diffusers FlowMatchEulerDiscreteScheduler):
    # `set_timesteps(1)` yields timesteps [1000.] with sigmas [1., 0.], i.e. one
    # model evaluation. N steps = N evaluations, minimum 1.
)

# MiniMax-H3. Everything here is MEASURED (Phase 0), except the top of the
# production range, which is DOCUMENTED (not independently measured here):
#   * valid T = 17n + 5 for n >= 1 -- T = 5 is on the grid but its 2 latent
#     frames cannot be decoded (`num_chunks` = 0), so 22 frames / 0.917 s is the
#     hard decodable floor;
#   * latent_frames(T) = ceil(T/17)*5 - 3 (ComfyUI's own formula agrees only ON
#     the grid and disagrees off it, so this form is the one to use);
#   * production bounds 124-362: ComfyUI's node states the trained range as
#     "~124-362, longer is untested", and 362 = 17*21+5 = 15.083 s is the grid
#     point AT that stated top. An earlier version of this spec used 345 =
#     17*20+5 = 14.375 s instead -- the largest grid point BELOW 15 s -- read
#     off the official README's "output 4-15 s" prose as a strict ceiling. That
#     reading was a rounding mistake, not a second, more conservative source:
#     345 undersold ComfyUI's own stated top of the trained range by one grid
#     step, so this is a correction, not a relaxation. The 17n+5 grid itself is
#     structural (VAE clip_length=17, temporal_compression=4, token_drop=3, all
#     measured), so 362 is a valid length by construction; nothing above has
#     been generated and inspected to confirm quality holds all the way to it,
#     which is exactly what ComfyUI's "untested" qualifier is flagging. The
#     README's 4 s figure (107 frames) describes the hosted product; the API
#     floor follows the trained-range floor (124) instead, and that
#     floor-side discrepancy is unaffected by this correction.
MINIMAX_H3_TEMPORAL = TemporalSpec(
    frame_multiple=17, frame_offset=5, min_frames=124, max_frames=362,
    min_decodable_frames=22,
    latent_frames=lambda t: 1 if t <= 1 else -(-t // 17) * 5 - 3,
    fps_fixed=24.0, default_clip_lengths=(22, 39),
    pixel_align=32, max_pixel_hw=(768, 1344), snap_invalid_length=True,
    # `num_inference_steps` counts sigma grid points (terminal 0 included), so
    # it drives N-1 model evaluations; N=1 gives zero and the vendored
    # scheduler's `set_timesteps` refuses it.
    min_inference_steps=2, steps_are_sigma_grid_points=True,
    # Boundary conditioning only -- see the field's comment. The generated span
    # is anchored on the preserved clip's last frame (extend-forward), its first
    # frame (extend-backward), or both ends of a two-clip bridge.
    outpaint_placements=("extend_forward", "extend_backward", "bridge"),
    # The video VAE's temporal chunking, MEASURED and already relied on by
    # `h3_pipeline_ops._clip_pixel_frames` / the rotary time grid.
    latent_chunk_pattern=(1, 4, 4, 4, 4),
    # The ceiling-side analogue of `smoke_override_env`: lifts `max_frames`
    # (362, DOCUMENTED, not a decoder limit) for a deliberate probe past the
    # trained range. Every request this actually raises above 362 is warned as
    # untested (`generation_utils.validate_video_geometry`) -- there is
    # no silent path past the documented top.
    max_override_env="SUSHI_TEMPORAL_UNCAPPED",
)

TEMPORAL_SPECS: Dict[str, TemporalSpec] = {
    "ltx2": LTX2_TEMPORAL,
    "minimax_h3": MINIMAX_H3_TEMPORAL,
}


def temporal_spec_for_arch(arch: Optional[str]) -> Optional[TemporalSpec]:
    """The arch's ``TemporalSpec``, or None for an image/audio architecture."""
    return TEMPORAL_SPECS.get(arch or "")
