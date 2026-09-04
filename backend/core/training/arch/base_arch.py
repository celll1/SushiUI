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
from types import MappingProxyType
from typing import (Any, Callable, Dict, FrozenSet, List, Mapping, Optional,
                    Tuple)

import torch
from api.param_defaults import TRAINING_DEFAULTS as _TRAINING_DEFAULTS
# The phase-reason prose is re-exported: every arch module imports it from here.
from core.adapters.capability import (ADAPTER_PAIRS as _ADAPTER_PAIRS,
                                      AXIS_GENERATION, AXIS_TRAINING,
                                      PHASE2_PENDING, PHASE3_PENDING,
                                      PHASE3_PENDING_DENSE_ONLY,
                                      QUANTIZED_ADDITIVE_PENDING,
                                      QUANTIZED_ADDITIVE_SHIPPED,
                                      declared_pairs, training_refusal_reason)
from core.adapters.spec import ALGORITHM_LORA, FAMILY_NAMES
from core.adapters.spec import ALGORITHMS as ADAPTER_ALGORITHMS


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
    # Optional. If set, an arch's sample() calls it as
    # step_progress_callback(completed_steps, total_steps) once per completed
    # denoising step; archs that don't wire it simply never call it.
    step_progress_callback: Optional[Callable[[int, int], None]] = None


# ---------------------------------------------------------------------------
# Adapter selection (LyCORIS design boundary 3: the registry, not the trainer,
# says which adapter an architecture uses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRAAdapterPlan:
    """The adapter class an architecture uses plus the arguments it takes beyond
    the four every adapter takes (``trainer, rank, alpha, dtype``).

    Class and kwargs stay separable so a caller can compare a plan without
    constructing an adapter, which is what lets the registry gate run on CPU
    with no model.
    """

    adapter_cls: type
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self, trainer, lora_rank: int, lora_alpha: int, lora_dtype):
        return self.adapter_cls(trainer, lora_rank, lora_alpha, lora_dtype,
                                **self.kwargs)

    @property
    def log_detail(self) -> str:
        if not self.kwargs:
            return ""
        return " (" + ", ".join(f"{k}={v}" for k, v in self.kwargs.items()) + ")"


#: "Initial DoRA" column of the design doc's architecture-feasibility table
#: (``docs/guides/LYCORIS_ADAPTER_DESIGN.md``), verbatim.
DORA_VERDICTS = ("dense", "dense_only", "deferred", "refused")


@dataclass(frozen=True)
class AdapterCapability:
    """Which ``(algorithm, weight_decompose)`` pairs ONE architecture supports.

    A declaration WITH GATES, deliberately in two halves: ``additive_family``
    and ``initial_dora`` record the design-doc verdict, while ``supported`` and
    ``trainable`` come from the two tables in
    ``core.adapters.capability``.

    TWO AXES. ``supported`` is what GENERATES here; ``trainable`` is what a
    trainer may construct, save and resume here. ``require()`` takes the axis as
    a mandatory argument so no caller can enable training by reading the
    generation row.
    """

    additive_family: bool
    initial_dora: str
    supported: FrozenSet[Tuple[str, bool]]
    refusals: Mapping[Tuple[str, bool], str]
    #: LoHa/LoKr/DoRA over a weight-only quantized base. Says nothing about
    #: ordinary LoRA, which IS allowed over one in both generation and training
    #: (``core.adapters.is_lora_wrappable_linear``; ``reject_quantized_base``
    #: gates full fine-tuning and exempts LoRA on purpose). True only where the
    #: architecture has no dense configuration, so enabling the family at all
    #: IS the quantized-base case; ``quantized_base_reason`` carries the scope
    #: either way.
    quantized_base_additive_family: bool
    quantized_base_reason: str = ""
    #: "Yes, later gate", read on the TRAINING axis: the family generates here
    #: but a trainer may not construct it, because the gate is this
    #: architecture's own rather than the general Phase 2 step. The sentence
    #: saying which gate is ``capability.TRAINING_REFUSAL_REASONS``.
    additive_gated: bool = False
    #: The TRAINING axis. Defaults to nothing trainable so a hand-built matrix
    #: cannot open training by omission; ``declare_adapter_capability`` fills it
    #: from ``TRAINABLE_ADAPTER_PAIRS``.
    trainable: FrozenSet[Tuple[str, bool]] = frozenset()
    trainable_refusals: Mapping[Tuple[str, bool], str] = field(
        default_factory=dict)

    def __post_init__(self) -> None:
        if self.initial_dora not in DORA_VERDICTS:
            raise ValueError(
                f"initial_dora={self.initial_dora!r} is not one of {DORA_VERDICTS}")
        object.__setattr__(self, "supported", frozenset(self.supported))
        object.__setattr__(self, "refusals",
                           MappingProxyType(dict(self.refusals)))
        object.__setattr__(self, "trainable", frozenset(self.trainable))
        trainable_refusals = dict(self.trainable_refusals)
        for pair in _ADAPTER_PAIRS:
            if pair not in self.trainable and not trainable_refusals.get(pair):
                trainable_refusals[pair] = (
                    f"{FAMILY_NAMES.get(pair, pair)} adapters cannot be trained "
                    f"on this architecture")
        object.__setattr__(self, "trainable_refusals",
                           MappingProxyType(trainable_refusals))
        untrainable = sorted(self.trainable - self.supported)
        if untrainable:
            raise ValueError(
                f"adapter pairs {untrainable} are declared trainable but do not "
                f"generate here; a checkpoint no loader accepts is not a feature")
        missing = [p for p in _ADAPTER_PAIRS
                   if p not in self.supported and not self.refusals.get(p)]
        if missing:
            raise ValueError(
                f"adapter pairs {missing} are neither supported nor given a "
                f"refusal reason")
        overlap = sorted(self.supported & set(self.refusals))
        if overlap:
            raise ValueError(f"adapter pairs {overlap} are both supported and refused")
        if not self.quantized_base_additive_family and not self.quantized_base_reason:
            raise ValueError("quantized_base_additive_family=False needs a reason")

    def _axis(self, axis: str) -> Tuple[FrozenSet[Tuple[str, bool]],
                                        Mapping[Tuple[str, bool], str]]:
        if axis == AXIS_GENERATION:
            return self.supported, self.refusals
        if axis == AXIS_TRAINING:
            return self.trainable, self.trainable_refusals
        raise ValueError(
            f"axis {axis!r} is not one of "
            f"({AXIS_GENERATION!r}, {AXIS_TRAINING!r})")

    def supports(self, algorithm: str, weight_decompose: bool = False,
                 axis: str = AXIS_GENERATION) -> bool:
        pairs, _refusals = self._axis(axis)
        return (algorithm, bool(weight_decompose)) in pairs

    def refusal_reason(self, algorithm: str, weight_decompose: bool = False,
                       axis: str = AXIS_GENERATION) -> Optional[str]:
        """Why the pair is refused on ``axis``, or ``None`` when it is allowed."""
        pairs, refusals = self._axis(axis)
        pair = (algorithm, bool(weight_decompose))
        if pair in pairs:
            return None
        return refusals.get(
            pair, f"adapter algorithm {algorithm!r} is not recognized")

    def require(self, algorithm: str, weight_decompose: bool, axis: str) -> None:
        """Raise unless the pair is allowed on ``axis``.

        ``axis`` is mandatory: the generation and training rows open
        separately, and defaulting to either is how a flip on one silently
        becomes a flip on the other.
        """
        reason = self.refusal_reason(algorithm, weight_decompose, axis=axis)
        if reason is not None:
            raise ValueError(reason)


def declare_adapter_capability(
    arch: str,
    *,
    additive_family: bool,
    initial_dora: str,
    additive_reason: str,
    dora_reason: str,
    quantized_base_reason: str,
    quantized_base_additive_family: bool = False,
    additive_gated: bool = False,
) -> AdapterCapability:
    """Build one architecture's matrix from its design-doc verdict.

    What is ENABLED is not decided here: it is read from the two tables in
    ``core.adapters.capability``, which generation reads too (it may not import
    this package -- see that module). A flip is one edit to a table; this
    refuses exactly what the table does not enable, so a refusal cannot be
    dropped without the pair being enabled in the same edit.
    """
    supported = declared_pairs(arch, AXIS_GENERATION)
    trainable = declared_pairs(arch, AXIS_TRAINING)
    refusals: Dict[Tuple[str, bool], str] = {}
    for algorithm, decompose in _ADAPTER_PAIRS:
        if (algorithm, decompose) in supported:
            continue
        # DoHa/DoKr are blocked twice over -- by the decomposition AND by the
        # additive algebra underneath it -- unless that algebra is enabled here,
        # in which case saying so would be false.
        if not decompose:
            reason = additive_reason
        elif algorithm != ALGORITHM_LORA and (algorithm, False) not in supported:
            reason = f"{dora_reason}; and {additive_reason}"
        else:
            reason = dora_reason
        refusals[(algorithm, decompose)] = f"{arch}: {reason}"

    trainable_refusals: Dict[Tuple[str, bool], str] = {}
    for pair in _ADAPTER_PAIRS:
        if pair in trainable:
            continue
        # A pair that does not generate here cannot be trained here either, and
        # the generation reason is the one that actually blocks it.
        if pair in supported:
            reason = dora_reason if pair[1] else training_refusal_reason(arch)
            trainable_refusals[pair] = (
                f"{arch}: {FAMILY_NAMES[pair]} adapters cannot be trained -- "
                f"{reason}")
        else:
            trainable_refusals[pair] = refusals[pair]

    return AdapterCapability(
        additive_family=additive_family,
        initial_dora=initial_dora,
        supported=supported,
        refusals=refusals,
        quantized_base_additive_family=quantized_base_additive_family,
        quantized_base_reason=f"{arch}: {quantized_base_reason}",
        additive_gated=additive_gated,
        trainable=trainable,
        trainable_refusals=trainable_refusals,
    )


#: ``ArchHandler``'s default: an architecture that declares no matrix supports
#: nothing, matching ``lora_adapter_class()``, which also refuses by default.
NO_ADAPTER_CAPABILITY = AdapterCapability(
    additive_family=False,
    initial_dora="refused",
    supported=frozenset(),
    refusals={pair: "this architecture declares no adapter capability matrix"
              for pair in _ADAPTER_PAIRS},
    quantized_base_additive_family=False,
    quantized_base_reason="this architecture declares no adapter capability matrix",
    trainable=frozenset(),
    trainable_refusals={
        pair: "this architecture declares no adapter capability matrix"
        for pair in _ADAPTER_PAIRS},
)


def resolve_scope_csv(trainer, key: str, default: str) -> str:
    """An arch's LoRA scope string: trainer attribute, then run config, then the
    architecture's own default. Empty string means "unset" at every tier."""
    return (getattr(trainer, key, "")
            or trainer.config.get(key, "")
            or default)


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

    #: Whether this handler's ``generate_sample`` actually forwards
    #: ``SampleContext.step_progress_callback`` into the arch's sample loop.
    #: The capability table (``api.arch_capabilities``) only says whether
    #: ``sample()`` runs at all, not whether it drives this callback -- an
    #: arch can pass the capability check and still never move the bar past
    #: its start position. Trainer-side ``emit_start()`` gates on BOTH.
    #: Flip this to ``True`` on a handler only once its ``generate_sample``
    #: actually calls ``step_progress_callback``.
    wires_sample_step_progress: bool = False

    def __init__(self, trainer: Any = None):
        # Optional back-reference (same contract as BaseLoRAAdapter). Canonical
        # methods still take ``trainer`` explicitly; this is a convenience only.
        self.trainer = trainer

    #: Which ``(algorithm, weight_decompose)`` pairs this architecture supports.
    #: Every registered handler overrides it; no caller yet (Phase 2/3).
    adapter_capability: AdapterCapability = NO_ADAPTER_CAPABILITY

    # ---- adapter selection (mode x arch) ----
    def lora_adapter_class(self) -> type:
        """This architecture's ``BaseLoRAAdapter`` subclass.

        Overrides import it inside the method, not at module scope: this
        package is imported from ``base_trainer``, i.e. while
        ``core.training`` is still initialising.
        """
        raise NotImplementedError(
            f"{type(self).__name__} (arch '{self.name}') declares no LoRA "
            f"adapter class, so LoRA training cannot be dispatched for it."
        )

    def lora_adapter_kwargs(self, trainer) -> Dict[str, Any]:
        """Constructor arguments beyond ``(trainer, rank, alpha, dtype)``.

        Where an architecture resolves its scope strings from the run config.
        Insertion order is also the order the trainer logs them in.
        """
        return {}

    def lora_adapter_plan(self, trainer) -> LoRAAdapterPlan:
        return LoRAAdapterPlan(self.lora_adapter_class(),
                               self.lora_adapter_kwargs(trainer))

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

        ``drop_mask`` is a CPU boolean mask: the one draw sampled per assembled
        optimization batch on the first MNT transform, and (when
        ``cfg_uncond_drop_per_mnt`` is on) independently redrawn per later
        transform (``BaseTrainer.cfg_drop_mask_for_mnt``) -- always THIS
        forward's own label, never assumed shared with any other MNT
        transform of the same batch. Only a handler with
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
