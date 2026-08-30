"""
ArchHandler ABC — the architecture axis of the training refactor.

Training has TWO orthogonal axes (see tmp/TRAINING_REFACTOR_PLAN.md §0):

  * MODE      — subclass hierarchy (BaseTrainer -> FullParameterTrainer /
                LoRATrainer -> ReLoRATrainer, ControlNetTrainer) + the
                ``adapters/`` layer (trainable-param / checkpoint logic).
  * ARCHITECTURE — modeled here by COMPOSITION: an ArchHandler object held by
                the trainer as ``trainer.arch``, replacing the 111 ``if
                self.is_<arch>`` branches with single-dispatch on a registry.

Ownership boundary (plan R3 — MUST be respected):

  * ``adapters/`` OWN trainable-parameter injection and checkpoint save/load
    (mode x arch). They hold ``self.trainer`` and reach into trainer state.
  * ``ArchHandler`` OWNS load / encode / train-step / sample (arch only). It is
    READ-mostly on the trainer: it reads ``trainer.unet / vae / is_sdxl / ...``
    and calls back into the shared spine, but it does NOT duplicate adapter
    responsibilities (no LoRA injection, no checkpoint writing).

  Both objects hold a back-reference to the same trainer (the SAME contract the
  adapters layer already uses, base_adapter.py:23). The handler stays stateless
  w.r.t. trainer mutation: canonical methods take ``trainer`` explicitly so the
  eventual body-move (plan A.3) is a rename+repack, not a rewrite.

Math lives in ``ops/`` (and the existing ``core/models/<arch>/*_pipeline_ops``);
handlers are THIN orchestrators (plan A.4, pain point #4).

This module is P0 scaffolding: signatures + context dataclasses only. No bodies
are moved here yet (that is P3-P7). Nothing calls these methods until later
phases flip the base_trainer dispatchers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from api.param_defaults import TRAINING_DEFAULTS as _TRAINING_DEFAULTS


# ---------------------------------------------------------------------------
# Context dataclasses — freeze the CURRENT keyword bundles so the P6/P7 move is
# a mechanical rename+repack (plan A.3). Field sets are the UNION across every
# arch, extracted verbatim from the real call sites (line refs below).
# ---------------------------------------------------------------------------


@dataclass
class TrainStepContext:
    """Bundles every kwarg passed to ``train_step_<arch>`` from
    ``_execute_forward_backward`` (base_trainer.py:5957-6149).

    The union of arguments across all arch branches (5989-6123). Per-arch
    train_step methods read only the subset they need today; the dispatcher
    populates the rest as ``None`` / defaults. ``loss_scale`` is applied by the
    shared spine AFTER train_step returns (base_trainer.py:6133), so it rides in
    the context but is not consumed by the arch's forward.
    """

    # --- primary tensors (mnt_latents / the text tensor) ---
    latents: torch.Tensor                                   # mnt_latents (5959); flux2 uses packed
    text_embeddings: Optional[torch.Tensor] = None          # SD/SDXL/controlnet/flux2 (prompt_embeds)
    encoder_features: Optional[torch.Tensor] = None         # lens/ideogram4/krea2 (5023/6035/6048)
    text_embeds: Optional[torch.Tensor] = None              # minit2i (6060)
    sensenova_prefix: Optional[Any] = None                  # immutable prompt DynamicCache (B1)

    # --- masks / pooled / micro-conditioning ---
    attention_mask: Optional[Any] = None                    # mnt_attention_mask (5961) - tensor OR anima dict
    encoder_mask: Optional[torch.Tensor] = None             # lens/ideogram4/krea2 (encoder_mask)
    pooled_embeddings: Optional[torch.Tensor] = None        # SD/SDXL/controlnet (mnt_pooled_embeddings)
    time_ids: Optional[torch.Tensor] = None                 # SD/SDXL/controlnet (mnt_time_ids)
    anima_aux: Dict[str, Any] = field(default_factory=dict)  # anima (6009)

    # --- timesteps + schedule cache ---
    timesteps: Optional[torch.Tensor] = None
    alphas_cumprod_cached: Optional[torch.Tensor] = None

    # --- latent geometry (lens/ideogram4/krea2 from lens_latent_shape 5971) ---
    latent_h: Optional[int] = None
    latent_w: Optional[int] = None

    # --- minit2i / REPA ---
    repa_pixels: Optional[torch.Tensor] = None              # mnt_repa_pixels (5972)

    # --- flux2 packing payload (base_trainer.py:6071-6090) ---
    img_ids: Optional[torch.Tensor] = None
    txt_ids: Optional[torch.Tensor] = None
    guidance: Optional[torch.Tensor] = None
    reference_latents_nested: Optional[list] = None

    # --- controlnet mode (base_trainer.py:6097-6109) ---
    use_condition_images: bool = False
    condition_images: Optional[torch.Tensor] = None

    # --- aligned CFG null condition (strategy §5) ---
    # CPU boolean [B], sampled ONCE per assembled optimization batch before any
    # MNT repetition, so every MNT transform of an item carries the same label,
    # and sliced alongside the batch by the OOM micro-batching path. None when
    # the run does not train an aligned null.
    cfg_drop_mask: Optional[torch.Tensor] = None

    # --- debug / profiling (common) ---
    debug_save_path: Optional[Path] = None
    debug_captions: Optional[List[str]] = None
    debug_reference_image_paths: Optional[List[Optional[str]]] = None
    profile_vram: bool = False

    # --- loss scaling (applied by spine after return; base_trainer.py:6133) ---
    loss_scale: float = 1.0


@dataclass
class SampleContext:
    """Bundles the ``_dispatch_sample`` arguments (base_trainer.py:9252-9320).

    Every per-arch ``_generate_sample_<arch>`` / ``generate_sample`` reads a
    subset of these (see the branch table at 9277-9320). Frozen verbatim so the
    P7 move is a repack.
    """

    prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    sampler: str = _TRAINING_DEFAULTS["sample_sampler"]
    negative_prompt: str = ""
    reference_image_path: Optional[str] = None
    condition_image_path: Optional[str] = None
    current_step: int = 0
    schedule_type: str = _TRAINING_DEFAULTS["sample_schedule_type"]
    cfg_schedule_type: str = _TRAINING_DEFAULTS["sample_cfg_schedule_type"]
    cfg_schedule_min: float = _TRAINING_DEFAULTS["sample_cfg_schedule_min"]
    cfg_schedule_max: Optional[float] = _TRAINING_DEFAULTS["sample_cfg_schedule_max"]
    cfg_schedule_power: float = _TRAINING_DEFAULTS["sample_cfg_schedule_power"]
    cfg_rescale_snr_alpha: float = _TRAINING_DEFAULTS["sample_cfg_rescale_snr_alpha"]
    dynamic_threshold_percentile: float = _TRAINING_DEFAULTS["sample_dynamic_threshold_percentile"]
    dynamic_threshold_mimic_scale: float = _TRAINING_DEFAULTS["sample_dynamic_threshold_mimic_scale"]
    nag_enable: bool = _TRAINING_DEFAULTS["sample_nag_enable"]
    nag_scale: float = _TRAINING_DEFAULTS["sample_nag_scale"]
    nag_tau: float = _TRAINING_DEFAULTS["sample_nag_tau"]
    nag_alpha: float = _TRAINING_DEFAULTS["sample_nag_alpha"]
    nag_sigma_end: float = _TRAINING_DEFAULTS["sample_nag_sigma_end"]
    nag_negative_prompt: str = _TRAINING_DEFAULTS["sample_nag_negative_prompt"]
    sensenova_timestep_shift: float = _TRAINING_DEFAULTS["sensenova_sample_timestep_shift"]
    sensenova_img_cfg_scale: float = _TRAINING_DEFAULTS["sensenova_sample_img_cfg_scale"]
    sensenova_cfg_norm: str = _TRAINING_DEFAULTS["sensenova_sample_cfg_norm"]


# ---------------------------------------------------------------------------
# ArchHandler ABC — canonical method set (plan A.3)
# ---------------------------------------------------------------------------

# Forward-ref only for typing; avoids a hard import cycle with wiring.py.
try:  # pragma: no cover - typing convenience
    from core.training.components.wiring import ComponentWiringSpec, TemporalSpec
except Exception:  # pragma: no cover
    ComponentWiringSpec = Any  # type: ignore
    TemporalSpec = Any  # type: ignore


class ArchHandler(ABC):
    """Per-architecture handler bound to a trainer via composition.

    Subclasses set ``name`` (== the ``_build_cache_namespace`` arch string, plan
    R6) and ``wiring`` (a ComponentWiringSpec, plan A.4). All canonical methods
    take ``trainer`` explicitly (handler stays stateless w.r.t. trainer state).

    P0/P1: every method is a stub raising NotImplementedError. Bodies are moved
    from base_trainer.py in phases P3-P7; nothing calls them until then.
    """

    #: Registry key — MUST equal the arch string from _build_cache_namespace
    #: (base_trainer.py:9821-9838). Cache-stability invariant (plan R6).
    name: str = ""

    #: Component-wiring spec (components/wiring.py). Placeholder in P0/P1.
    wiring: "ComponentWiringSpec" = None  # type: ignore

    #: Minimum PIXEL-dimension alignment this arch requires, i.e. every training
    #: image's width/height must be a multiple of this. Default 8 = the VAE
    #: downscale factor (SD/SDXL, latent = pixel/8). Patchified DiTs additionally
    #: patchify the latent by patch_spatial (=2), so their pixel dims must be a
    #: multiple of vae_scale(8) * patch_spatial(2) = 16, and their patchify asserts
    #: on non-conforming dims (e.g. Anima). Used by the no-bucketing area-fit and
    #: the resolution-curriculum refit to align dims to the ARCH's requirement,
    #: not just to /8. Overridden to 16 by every patchified DiT handler below.
    pixel_align: int = 8

    #: Sequence axis of ONE item's text embedding, i.e. of the ``[1, ...]``
    #: tensor ``encode_caption`` returns, as consumed by the batch-assembly
    #: collation in ``BaseTrainer._collate_text_embeddings``.
    #:
    #: 1 is right for ``[1, L, D]`` (SD/SDXL/Z-Image/MiniT2I/FLUX.2/...) AND for
    #: Krea 2's ``[1, L, num_layers, D]``. Lens and Ideogram 4 stack layers
    #: FIRST (``[1, num_layers, L, D]``), so for them axis 1 is num_layers --
    #: reading it as the length makes every item in a batch report the same
    #: value, skips the padding branch, and lets ``torch.cat`` raise on any
    #: batch whose captions tokenise to different L. They override this to 2.
    text_seq_axis: int = 1

    #: TEMPORAL analogue of ``pixel_align`` (Phase 6a): the arch's clip-length
    #: grid, latent-frame closed form, fixed frame rate (if any), default
    #: training clip lengths and canvas envelope, declared once in
    #: ``core.models.components.wiring.TemporalSpec`` and shared with the
    #: GENERATION side (route validation, ``/schema/arch-capabilities``).
    #:
    #: ``None`` for every image / audio architecture — that is also what makes
    #: it the video predicate the trainer branches on
    #: (``BaseTrainer._temporal_spec()``), instead of an ``is_<arch>`` flag that
    #: has to be extended for each new video arch.
    temporal: Optional["TemporalSpec"] = None

    #: Token identifying the SPATIAL TILING configuration this arch's video VAE
    #: uses to encode a training clip, folded into the clip cache key.
    #:
    #: Tiling is not a determinism nicety: on MiniMax-H3, flipping the shipped
    #: tiling flags with everything else held fixed moved the latents by
    #: rel-RMS 0.355 (384x384) / 0.0952 (640x384). A cached latent produced
    #: under one policy is NOT interchangeable with generation under the other,
    #: so the policy has to be part of the key.
    #:
    #: ``None`` means "this arch does not configure tiling for clip encode",
    #: which is true of LTX-2.3 (``ltx2_ops.vae_encode_clip`` calls
    #: ``vae.encode`` with whatever the loader set) and keeps its keys
    #: unchanged. An arch that pins a policy sets a stable string here.
    clip_vae_tiling_policy: Optional[str] = None

    #: Token identifying the AUDIO preprocessing chain of a window-level clip
    #: record (Phase 6b). ``None`` for an arch whose clip record holds only a
    #: video latent — which keeps LTX-2.3's keys unchanged — and a stable string
    #: for one whose record also holds the window's audio latent
    #: (``vae_encode_clip_audio`` below). Part of the clip cache key for the same
    #: reason the tiling policy is: two records built by different audio chains
    #: are different data and must not share an address.
    clip_audio_prep_version: Optional[str] = None

    #: Timestep convention this architecture's ``train_step`` noise-mixing
    #: formula uses: which end of the trainer's ``[0,1]`` ``timestep_sampler``
    #: output is a CLEAN (noise-free) latent.
    #:
    #:   ``"t0"`` -- sampler ``t=0`` is clean, ``t=1`` is pure noise. The
    #:   SD3/FLUX/Z-Image-style convention (``noisy = (1-t)*latents + t*noise``)
    #:   used by every architecture in this codebase except the two below.
    #:   ``"t1"`` -- sampler ``t=1`` is clean, ``t=0`` is pure noise
    #:   (``noisy = t*latents + (1-t)*noise``). SenseNova and MiniT2I.
    #:
    #: This is NOT a distribution choice -- it fixes what a
    #: ``timestep_sampling.mean`` biased toward 0 or 1 actually MEANS for a
    #: given architecture: the same ``mean=-0.8`` concentrates sampling near
    #: the CLEAN side under ``"t0"`` and near the NOISE side under ``"t1"``.
    #: Declared here as the architecture's fixed default. SD15ArchHandler /
    #: SDXLArchHandler override ``resolve_timestep_convention()`` instead,
    #: because that ONE handler flips between the two conventions depending on
    #: the run's ``noise_process`` (the ddpm/flow branches in
    #: ``ops/sd_sdxl_ops.py`` use opposite ends of the sampler range).
    timestep_convention: str = "t0"

    def resolve_timestep_convention(self, trainer: Any = None) -> str:
        """Returns ``"t0"`` or ``"t1"`` -- see ``timestep_convention``.

        Default: the class attribute, independent of trainer state. Overridden
        only by handlers whose convention depends on trainer runtime config
        (SD15/SDXL: ``trainer.noise_process``).
        """
        return self.timestep_convention

    #: At which stage this architecture can build the SAME null condition its
    #: inference CFG uncond branch uses, or ``None`` when it cannot build one at
    #: all. Three admitted values:
    #:
    #:   ``None``        no aligned null exists here. An explicitly supplied
    #:                   ``cfg_uncond_drop_rate`` -- INCLUDING ``0.0`` -- is
    #:                   refused before the model loads, never accepted and
    #:                   ignored. Mirrored for the API process by
    #:                   ``api/arch_capabilities.CFG_NULL_STAGE_BY_ARCH``, which
    #:                   `cfg_null_resolver_test.py` pins against this value.
    #:   ``"collated"``  the null is a rewrite of already-encoded, batched
    #:                   conditioning (``apply_cfg_null_collated``).
    #:   ``"encode"``    the null has to be built while encoding the item,
    #:                   because the inference baseline differs in the token
    #:                   sequence itself (``encode_prompt_cfg_null``).
    #:
    #: This is a capability declaration, not the implementation: a handler that
    #: sets it must also override the hook its stage names.
    cfg_null_stage: Optional[str] = None

    def __init__(self, trainer: Any = None):
        # Optional back-reference (same contract as BaseLoRAAdapter). Canonical
        # methods still take ``trainer`` explicitly; this is a convenience only.
        self.trainer = trainer

    # ---- loading / setup ----
    @abstractmethod
    def load_components(self, trainer) -> None:
        """Was ``_load_<arch>_components`` (base_trainer.py:1122+)."""
        raise NotImplementedError

    @abstractmethod
    def setup_block_swap(self, trainer) -> None:
        """Was ``setup_<arch>_block_swap`` — wrapper construction only; the
        optimizer/fused validation stays central (plan R2)."""
        raise NotImplementedError

    @abstractmethod
    def setup_attention_backend(self, trainer) -> None:
        """Was ``_setup_attention_backend_<arch>`` (per-arch body)."""
        raise NotImplementedError

    # ---- text ----
    @abstractmethod
    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        """Was the per-arch ``_encode_prompt_*`` body. Returns embeds, or
        ``(embeds, pooled)`` for pooled archs (SDXL)."""
        raise NotImplementedError

    def encode_prompt_cfg_null(self, trainer, prompt, *,
                               requires_grad: bool = False, **kwargs):
        """The ``cfg_null=True`` branch of ``encode_prompt``: build the prompt
        encoding the architecture's INFERENCE uncond branch would build, rather
        than encoding ``prompt``.

        Spelled as its own method instead of an ``encode_prompt(..., cfg_null)``
        flag so a handler that has not implemented the encode stage REFUSES
        rather than swallowing an unrecognised keyword. Only a handler with
        ``cfg_null_stage == "encode"`` may override it.

        ``kwargs`` carries whatever the arch's own ``encode_prompt`` accepts
        beyond the prompt (SenseNova: ``reference_image_paths``). They are
        forwarded rather than dropped so the override can REFUSE a conditioning
        its null does not represent, instead of quietly building a different
        one.
        """
        self._reject_cfg_null("encode_prompt_cfg_null", "encode")

    def apply_cfg_null_collated(self, trainer, conditioning, auxiliary,
                                drop_mask):
        """Rewrite the rows of an ALREADY-ENCODED, already-collated batch that
        ``drop_mask`` selects into the architecture's inference null condition,
        and return the rewritten ``(conditioning, auxiliary)``.

        ``drop_mask`` is the one CPU boolean mask sampled per assembled
        optimization batch, before any MNT repetition, so every MNT transform of
        an item carries the same label. Only a handler with
        ``cfg_null_stage == "collated"`` may override it.
        """
        self._reject_cfg_null("apply_cfg_null_collated", "collated")

    def apply_cfg_null_step(self, trainer, ctx, conditioning, auxiliary):
        """THE call site of the collated hook: a handler's own ``train_step``.

        Every ``cfg_null_stage == "collated"`` handler calls this first and
        passes the result on to its ops body, so the rewrite happens at one
        level for all of them -- before the ops body's device/dtype moves, so
        the clone is a host-side copy rather than a device one, and before any
        arch-specific reshaping of the conditioning.
        """
        if getattr(ctx, "cfg_drop_mask", None) is None:
            return conditioning, auxiliary
        return self.apply_cfg_null_collated(trainer, conditioning, auxiliary,
                                            ctx.cfg_drop_mask)

    def _reject_cfg_null(self, hook: str, stage: str):
        raise NotImplementedError(
            f"{type(self).__name__} (arch '{self.name}') declares "
            f"cfg_null_stage={self.cfg_null_stage!r} and does not implement "
            f"{hook}(), which belongs to the '{stage}' stage. An aligned CFG "
            f"null condition is not available for this architecture."
        )

    def collate_aux(self, trainer, batch) -> dict:
        """Was ``_collate_<arch>_aux`` (e.g. ``_collate_anima_aux``).

        No-op default: most archs carry no auxiliary batch payload. Anima
        overrides to assemble the LLM-adapter side payload (source_mask, t5 ids).
        """
        return {}

    # ---- image / vae ----
    @abstractmethod
    def vae_encode(self, trainer, image_tensor, *, image=None, width: int = None,
                   height: int = None, vae_device=None, debug_preprocessing: bool = False):
        """The per-arch branch of ``BaseTrainer.encode_image`` (P5).

        Called by the ``encode_image`` wrapper, which keeps the shared pre-amble
        (resize/crop/micro-cond stash, numpy->tensor) and shared post-amble
        (final ``latents.to(training_dtype, cpu)``) VERBATIM. For the 7 VAE archs
        the wrapper runs this INSIDE ``with torch.no_grad()`` and passes
        ``image_tensor`` already staged on ``vae_device``/``vae_dtype``; the body
        returns raw ``latents`` (still on the VAE device). Pixel-space / minit2i
        (dispatched BEFORE the shared VAE staging, since it may have no VAE) is
        fully self-contained and returns the final CPU tensor directly.
        ``image`` (PIL) is used by the Lens/Ideogram4/Krea2 branches."""
        raise NotImplementedError

    def vae_encode_clip_audio(self, trainer, video_path: str, start_time: float,
                              duration: float):
        """The AUDIO latent of the same clip window ``vae_encode_clip`` encodes.

        Default ``None``: an arch whose clip record is video-only (LTX-2.3 —
        which trains video-only with a grad-free dummy audio branch) never
        produces one, and the cache record then carries exactly the fields it
        always had. MiniMax-H3 overrides it, because its audio rows are part of
        the SAME packed sequence its video rows are and its LoRA targets are
        shared by both modalities.

        Cut by the SAME ``[start_time, start_time + duration)`` window as the
        frames, so A/V alignment is a construction property. ``None`` also means
        "this source has no audio", which the arch's train step must handle.
        """
        return None

    @abstractmethod
    def vae_decode(self, trainer, latents, *, latent_h: int, latent_w: int) -> torch.Tensor:
        """Latent->pixel decode (reused by sampling)."""
        raise NotImplementedError

    # ---- training step: forward + loss (single canonical entry) ----
    @abstractmethod
    def train_step(self, trainer, ctx: TrainStepContext) -> Tuple[float, float, float]:
        """Was ``train_step_<arch>`` (base_trainer.py:6155-8071).
        Returns (loss, pred_loss, recon_loss) — identical contract to today."""
        raise NotImplementedError

    # ---- sampling ----
    @abstractmethod
    def sample(self, trainer, sample_ctx: SampleContext):
        """Was ``_generate_sample_<arch>`` / ``generate_sample``
        (base_trainer.py:8216-9777). Returns a PIL image or ``None``."""
        raise NotImplementedError
