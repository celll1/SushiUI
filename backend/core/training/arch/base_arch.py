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
    negative_prompt: str = ""
    reference_image_path: Optional[str] = None
    condition_image_path: Optional[str] = None
    current_step: int = 0
    schedule_type: str = "uniform"


# ---------------------------------------------------------------------------
# ArchHandler ABC — canonical method set (plan A.3)
# ---------------------------------------------------------------------------

# Forward-ref only for typing; avoids a hard import cycle with wiring.py.
try:  # pragma: no cover - typing convenience
    from core.training.components.wiring import ComponentWiringSpec
except Exception:  # pragma: no cover
    ComponentWiringSpec = Any  # type: ignore


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

    def collate_aux(self, trainer, batch) -> dict:
        """Was ``_collate_<arch>_aux`` (e.g. ``_collate_anima_aux``).

        No-op default: most archs carry no auxiliary batch payload. Anima
        overrides to assemble the LLM-adapter side payload (source_mask, t5 ids).
        """
        return {}

    # ---- image / vae ----
    @abstractmethod
    def vae_encode(self, trainer, image_tensor, *, width: int, height: int) -> torch.Tensor:
        """The per-arch branch of ``encode_image`` (base_trainer.py:5275-5585).
        Pixel-space archs (minit2i, latent_channels==0) pass pixels through."""
        raise NotImplementedError

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
