"""Training-only SenseNova decoder operations.

The trainer supplies one prompt prefix per step -- a single prompt, or one
packed prefix carrying a segment per item above batch 1. It is immutable
either way, but not always detached: with ``train_text_encoder`` the prefix is
built by a differentiable understanding-branch pass so the gradient reaches the
understanding LoRA (Phase U). Both modes share one structural contract and
differ only in which grad-mode assertion applies -- see
``_assert_immutable_prefix_cache``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from api.param_defaults import TRAINING_DEFAULTS as _TRAINING_DEFAULTS
from ..training_events import emit_training_event, emit_training_warning
from .training_method import (
    _FULL_FINETUNE_METHOD,
    is_full_finetune,
    resolve_training_method,
)


@dataclass(frozen=True)
class SenseNovaTrainingPrefix:
    """Detached prompt K/V reused by every flow step for one sample."""

    cache: Any
    # The NEXT t index, not a token count: image tokens spread over h/w, so with
    # references the prefix's t extent is shorter than its token count. Vendor
    # `_build_t2i_image_indexes` uses this as the image tokens' t coordinate.
    text_length: int
    # Batched (packed) form: ``packed`` is the vendor ``PackedSegments`` of the
    # prompts laid end to end, ``text_lengths`` the per-item t extents. ``None``
    # is the single-prompt form, where ``text_length`` alone is read.
    packed: Any = None
    text_lengths: Optional[List[int]] = None

    @property
    def batch_size(self) -> int:
        return 1 if self.packed is None else int(self.packed.count)


_SENSENOVA_QUANT_LINEAR_COUNT = 588


def resolve_full_finetune_branch(trainer: Any) -> str:
    """Which MoT half a full fine-tune trains: ``"gen"``, ``"und"`` or ``"both"``.

    Reuses the two shipped switches rather than adding a SenseNova-only key, on
    the mapping Phase 1 LoRA already uses: the generation half IS this arch's
    denoiser (``train_unet``) and the understanding half IS its text encoder
    (``train_text_encoder`` -- ``sensenova_adapter`` injects understanding LoRA
    on exactly that flag). The ``getattr`` fallbacks match the call sites those
    two flags already have (``base_trainer`` for ``train_unet``,
    ``sensenova_adapter`` for ``train_text_encoder``).
    """
    train_gen = bool(getattr(trainer, "train_unet", True))
    train_und = bool(getattr(trainer, "train_text_encoder", False))
    if train_gen and train_und:
        return "both"
    if train_gen:
        return "gen"
    if train_und:
        return "und"
    raise ValueError(
        "SenseNova full fine-tuning has nothing to train: train_unet=False and "
        "train_text_encoder=False. For this architecture train_unet selects the "
        "generation half of the MoT decoder (294 Linears) and train_text_encoder "
        "the understanding half (294 more, the same LLM's prompt-encoding path). "
        "Set at least one of them, or use training_method='lora'."
    )


# The optimizers this route runs under. Two conditions, both required: a
# per-parameter fused-backward seam, and state that fits beside the materialized
# half. Measured B/param from SENSENOVA_TRAINING_DESIGN.md 6.5's table, scaled
# over the gen half's 8,103,395,328 parameters and over both halves:
#
#   adamw8bit             2.031250 B/param -> 16.5 GB / 32.9 GB. Has the seam
#                         (FUSED_BACKWARD_OPTIMIZERS, patched in
#                         _setup_fused_backward_pass); excluded on state size,
#                         and it has no host-resident mode to escape into.
#   adamw8bit_ringbuffer  2.031250 B/param on the GPU -- the same 16.5 / 32.9 GB
#                         -- or 0.031250 GPU plus 2.0 pinned on the host with
#                         optimizer_state_host_resident. ADMITTED ONLY IN THAT
#                         MODE (assert_ringbuffer_host_state).
#   lion8bit_ringbuffer   half of the AdamW pair, one moment instead of two:
#                         1.015625 GPU, or 0.015625 GPU / 1.0 host. Same
#                         condition.
#   adafactor             0.002991 B/param (shape-dependent), no condition.
#
# G-RB1's transfer-hiding threshold does not exclude either ring-buffer optimizer
# at this route's resolution. The token grid is /32 in pixel space, so 2048px
# is 4096 image tokens, against thresholds of 2038 (AdamW) / 1019 (Lion); under
# MoT with both halves trainable only the gen half computes over image tokens
# while both halves' state transfers, which roughly doubles the effective
# threshold -- AdamW then clears by ~1%, Lion by 2x. No speed claim follows: this
# route's step wall under either has not been measured (6.5 / 13.4 U-2-4).
SENSENOVA_FULL_FINETUNE_OPTIMIZERS = (
    "adafactor", "adamw8bit_ringbuffer", "lion8bit_ringbuffer",
)

# Measured GPU B/param when host-resident state is OFF (6.5's G-RB2 table), and
# the parameter count of both MoT halves (U-2-1, off the real checkpoint header).
_RINGBUFFER_GPU_STATE_BYTES_PER_PARAM = {
    "adamw8bit_ringbuffer": 2.031250,
    "lion8bit_ringbuffer": 1.015625,
}
_SENSENOVA_BOTH_HALVES_PARAMS = 16_206_790_656


def assert_ringbuffer_host_state(name: str, host_resident: bool) -> None:
    """Refuse a ring-buffer optimizer whose 8-bit state would land on the GPU.

    The optimizer NAME is an API field while ``optimizer_state_host_resident`` is
    a config key, so the two channels can disagree; without this a run started
    from the product would allocate the GPU-state figure below on top of the
    materialized halves and OOM inside step 1.
    """
    per_param = _RINGBUFFER_GPU_STATE_BYTES_PER_PARAM.get(name)
    if per_param is None or host_resident:
        return
    gpu_gb = per_param * _SENSENOVA_BOTH_HALVES_PARAMS / 1e9
    raise ValueError(
        f"SenseNova full fine-tuning accepts optimizer='{name}' only with "
        f"optimizer_state_host_resident=true. Left on the GPU its 8-bit state is "
        f"a measured {per_param} B/param, i.e. {gpu_gb:.1f} GB over both MoT "
        f"halves ({gpu_gb / 2:.1f} GB over one), on top of the materialized bf16 "
        f"weights -- which does not fit the 48 GB budget this route is designed "
        f"against (SENSENOVA_TRAINING_DESIGN.md 6.5). With the flag set the same "
        f"state is pinned host memory and the GPU keeps only absmax. Set "
        f"optimizer_state_host_resident=true, or use optimizer=adafactor."
    )


def assert_full_finetune_contract(trainer: Any, optimizer_type: Any = None) -> None:
    """Refuse the full-fine-tune configurations this route does not implement.

    Called twice per run: from ``load_components`` before the 17.6 GiB load,
    reading ``config['optimizer']``, and from ``setup_optimizer`` with the name
    the run was actually started with, which is an argument rather than a config
    read and can disagree. Options that exist on both channels are checked on
    both, for the same reason.

    Every clause is a condition of the memory budget in
    SENSENOVA_TRAINING_DESIGN.md 6.2/6.3/6.5, not a preference.
    """
    settings = getattr(trainer, "config", None) or {}

    weight_dtype = getattr(trainer, "weight_dtype", None)
    training_dtype = getattr(trainer, "training_dtype", None)
    wrong_dtypes = [
        f"{name}={dtype}"
        for name, dtype in (("weight_dtype", weight_dtype),
                            ("training_dtype", training_dtype))
        if dtype is not None and dtype is not torch.bfloat16
    ]
    if wrong_dtypes:
        raise ValueError(
            "SenseNova full fine-tuning requires bf16 ("
            f"{', '.join(wrong_dtypes)}). The trainable half is dequantized from "
            "int8 into weight_dtype at load, so this setting decides what the "
            "base itself becomes: fp16 would materialize an fp16 base whose "
            "updates cannot be carried by stochastic rounding, and fp32 a "
            "30.2 GiB one. Set weight_dtype=bf16 and training_dtype=bf16."
        )
    if getattr(trainer, "use_grad_scaler", False):
        raise ValueError(
            "SenseNova full fine-tuning does not support FP16 gradient scaling. "
            "Its updates are applied by per-parameter post-accumulate-grad hooks, "
            "which free each gradient as they apply it, so GradScaler never runs "
            "its inf/NaN check and never updates its scale. Set training_dtype=bf16."
        )

    if bool(settings.get("use_ema", False)) or bool(getattr(trainer, "use_ema", False)):
        raise ValueError(
            "SenseNova full fine-tuning does not support use_ema. The EMA update "
            "is attached to the single optimizer.step() call site, and this route "
            "updates each parameter from its own post-accumulate-grad hook, which "
            "bypasses that call site: the shadow would silently never update. "
            "Set use_ema=false."
        )

    groups = max(
        int(settings.get("num_optimizer_groups", 0) or 0),
        int(getattr(trainer, "num_optimizer_groups", 0) or 0),
    )
    if groups:
        raise ValueError(
            f"SenseNova full fine-tuning requires num_optimizer_groups=0, got "
            f"{groups}. Fused optimizer groups call a batched optimizer.step() "
            "instead of the per-parameter hooks this route's memory budget "
            "depends on -- and they are only set up under Block Swap, which this "
            "architecture does not implement, so a non-zero value here would "
            "leave the run with no fused path at all."
        )

    accumulation = int(settings.get("gradient_accumulation_steps", 1) or 1)
    if accumulation != 1:
        raise ValueError(
            f"SenseNova full fine-tuning requires gradient_accumulation_steps=1, "
            f"got {accumulation}. Its updates are applied per parameter during "
            "backward and each gradient is freed as it is applied, so no gradient "
            "survives to be summed across backward passes: every backward would "
            "become its own optimizer step, and the run would move further per "
            "reported step than the effective batch implies (measured 3.88x at "
            "accum=4 with AdamW). A larger effective batch comes from batch_size "
            "with enable_bucketing, not from accumulation. LoRA training on this "
            "architecture does support gradient_accumulation_steps."
        )

    if bool(settings.get("sensenova_four_phase_eviction", False)) or bool(
        getattr(trainer, "sensenova_four_phase_eviction", False)
    ):
        assert_four_phase_contract(trainer)

    if optimizer_type is None and settings.get("optimizer") is None:
        # Absence carries no information on the config channel; the call from
        # setup_optimizer always names one, and that is the authoritative check.
        return
    name = str(
        optimizer_type if optimizer_type is not None else settings["optimizer"]
    ).strip().lower()
    if name in SENSENOVA_FULL_FINETUNE_OPTIMIZERS:
        assert_ringbuffer_host_state(
            name,
            bool(settings.get("optimizer_state_host_resident", False))
            or bool(getattr(trainer, "optimizer_state_host_resident", False)),
        )
        return
    extra = ""
    if name == "adamw":
        extra = (
            " torch.optim.AdamW updates every parameter inside one step() with no "
            "per-parameter seam, so stochastic rounding cannot be attached to it; "
            "measured under round-to-nearest, 84.5% of a bf16 tensor's elements "
            "never move at any step count, while the loss falls normally."
        )
    raise ValueError(
        f"SenseNova full fine-tuning does not support optimizer='{name}'. "
        f"Supported: {', '.join(SENSENOVA_FULL_FINETUNE_OPTIMIZERS)} (the two "
        f"ring-buffer optimizers additionally require "
        f"optimizer_state_host_resident, which moves their 8-bit state to pinned "
        f"host memory). This route "
        f"applies each update from that parameter's own post-accumulate-grad hook, "
        f"so the optimizer needs a per-parameter seam AND state small enough to sit "
        f"beside the materialized bf16 half: adamw8bit has the seam but its measured "
        f"2.031250 B/param is 16.5 GB of state over the generation half's 8.10 G "
        f"parameters (32.9 GB over both halves), against Adafactor's factored second "
        f"moment.{extra} Set optimizer=adafactor, or use training_method='lora', "
        f"which accepts every optimizer this product offers."
    )


def assert_four_phase_contract(trainer: Any) -> None:
    """Refuse four-phase eviction where the split cannot be closed (8.3.2).

    Three clauses, none of them preferences:

    * it needs a trained understanding half. With the understanding half frozen
      the three-state evictor already suffices and the split buys nothing while
      paying an extra understanding forward per step.
    * it needs the eviction it exists to enable. Splitting the backward without
      evicting leaves both halves resident, which is the single-backward path
      with an extra forward bolted on.
    * it needs the fused-backward route. Phase 3 ends with the generation half on
      CPU, so a subsequent ``optimizer.step()`` would meet CUDA gradients on CPU
      parameters. Under fused backward each half is stepped by its own hooks
      while it is resident and there is no such call.
    """
    settings = getattr(trainer, "config", None) or {}

    def either(key: str) -> bool:
        return bool(settings.get(key, False)) or bool(getattr(trainer, key, False))

    if not either("train_text_encoder"):
        raise ValueError(
            "SenseNova sensenova_four_phase_eviction requires train_text_encoder: "
            "the split exists so a TRAINED understanding half can still be "
            "evicted. With it frozen, sensenova_mot_phase_eviction alone already "
            "does this, without the extra understanding forward per step."
        )
    if not either("sensenova_mot_phase_eviction"):
        raise ValueError(
            "SenseNova sensenova_four_phase_eviction requires "
            "sensenova_mot_phase_eviction: on its own the split only adds a "
            "second backward and a recomputed forward, and both halves stay "
            "resident exactly as they do without it."
        )
    if not is_full_finetune(trainer):
        raise ValueError(
            "SenseNova sensenova_four_phase_eviction is implemented for "
            "training_method='full_finetune' only. It leaves the generation half "
            "on CPU at the end of the step, which is safe only on the fused "
            "backward route, where each half is updated by its own "
            "post-accumulate-grad hooks while it is resident; LoRA training calls "
            "optimizer.step() there instead, and would meet CUDA gradients on CPU "
            "parameters."
        )
    reduction = str(
        settings.get("sensenova_four_phase_grad_reduction", None)
        or getattr(trainer, "sensenova_four_phase_grad_reduction", "sum")
    ).strip().lower()
    if reduction not in ("sum", "mean"):
        raise ValueError(
            f"SenseNova sensenova_four_phase_grad_reduction must be 'sum' or "
            f"'mean', got {reduction!r}."
        )
    warn_four_phase_mnt_cost(trainer)


def assert_shared_prefix_contract(trainer: Any) -> None:
    """The shared window needs the split it shares (8.3.5).

    Restated here for a trainer built directly; ``train_runner`` refuses it
    before the load.
    """
    settings = getattr(trainer, "config", None) or {}

    def either(key: str) -> bool:
        return bool(settings.get(key, False)) or bool(getattr(trainer, key, False))

    if not either("sensenova_four_phase_shared_prefix"):
        return
    if not either("sensenova_four_phase_eviction"):
        raise ValueError(
            "SenseNova sensenova_four_phase_shared_prefix requires "
            "sensenova_four_phase_eviction: without the split there is no "
            "boundary cut to share across the MNT window."
        )
    # train_runner refuses this earlier and in config terms (MoT eviction under a
    # full fine-tune needs the both-halves branch). Restated here because a
    # trainer built directly would otherwise meet it as a census-internal
    # complaint: with only the understanding half trained, the deferred group IS
    # the whole expectation set and `set_deferred` refuses it.
    train_gen = settings.get("train_unet", getattr(trainer, "train_unet", True))
    if not bool(train_gen):
        raise ValueError(
            "SenseNova sensenova_four_phase_shared_prefix requires train_unet: "
            "with only the understanding half trained there is no half left "
            "taking a per-iteration update, so deferring the whole trainable set "
            "to the end of the window would leave nothing checked in between."
        )


def warn_four_phase_mnt_cost(trainer: Any) -> bool:
    """Say what MNT > 1 costs under the split, rather than refusing it.

    ``multi_noise_timesteps`` is NOT covered by this route's other clauses --
    ``assert_full_finetune_contract`` refuses gradient accumulation, which is a
    different mechanism, and ``BaseTrainer`` only requires MNT >= 1.

    Two messages, because the shared-prefix route removes the cost the
    per-iteration one announces and replaces it with a change to what is
    trained. Announcing the per-iteration cost under a shared window would name
    a price the run is no longer paying.
    """
    settings = getattr(trainer, "config", None) or {}
    mnt = max(
        int(settings.get("multi_noise_timesteps", 1) or 1),
        int(getattr(trainer, "multi_noise_timesteps", 1) or 1),
    )
    if mnt <= 1:
        return False
    prefix = getattr(trainer, "log_prefix", "[SenseNova]")
    shared = bool(settings.get("sensenova_four_phase_shared_prefix", False)) or bool(
        getattr(trainer, "sensenova_four_phase_shared_prefix", False)
    )
    if shared:
        reduction = str(
            settings.get("sensenova_four_phase_grad_reduction", None)
            or getattr(trainer, "sensenova_four_phase_grad_reduction", "sum")
        )
        emit_training_warning(
            f"SenseNova four-phase eviction with multi_noise_timesteps={mnt} and "
            f"a shared prefix: one understanding forward and one phase-3 backward "
            f"per window, and two weight round trips per batch rather than "
            f"{2 * mnt}. What this changes about training: the understanding half "
            f"takes ONE update per window, computed from the '{reduction}' of the "
            f"window's boundary gradient at the weights the window STARTED with, "
            f"while the generation half takes {mnt}; its Adafactor step counter "
            f"advances once per window, so its beta2 schedule moves {mnt}x slower; "
            f"and that single update uses the scheduler's learning rate after all "
            f"{mnt} iterations. Off (the default) each iteration runs its own "
            f"complete cycle instead.",
            code="sensenova_four_phase_shared_prefix",
            prefix=prefix,
        )
        return True
    emit_training_warning(
        f"SenseNova four-phase eviction with multi_noise_timesteps={mnt}: the "
        f"backward split runs once per MNT iteration, so its two weight "
        f"round trips are paid {mnt} times per step rather than once. This is "
        f"correct, not degraded -- each iteration recomputes the understanding "
        f"forward against the same weights its own forward used. "
        f"sensenova_four_phase_shared_prefix shares one cut across the window "
        f"instead, which changes what the understanding half trains on.",
        code="sensenova_four_phase_mnt_cost",
        prefix=prefix,
    )
    return True


def assert_four_phase_fused_backward(trainer: Any) -> None:
    """Backstop for the clause only the trainer can decide.

    The contract above refuses everything knowable before the load; whether the
    fused backward pass was actually INSTALLED is knowable only afterwards, and
    it is the clause the whole ordering depends on.
    """
    if not getattr(trainer, "sensenova_four_phase_eviction", False):
        return
    if not getattr(trainer, "use_fused_backward", False):
        raise RuntimeError(
            "SenseNova four-phase eviction requires the fused backward pass, which "
            "was not installed for this run. Phase 3 leaves the generation half on "
            "CPU, so the optimizer.step() that would follow cannot update it."
        )


def enforce_full_finetune_stochastic_rounding(trainer: Any) -> bool:
    """Turn stochastic rounding on for this route, and say so.

    The transport is tri-state (unset/True/explicit False); this is the
    trainer-side backstop for an unset value or hand-authored YAML, since
    ``train_runner`` already refuses an explicit False before the load.

    So it is a route requirement, listed per architecture in
    ``param_defaults.FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH``, applied
    here and announced through ``core.training.training_events``, which puts it
    on the ``training_log`` WebSocket channel and on the run row -- so a user
    who unticked the box sees the override in the Training Monitor
    (SENSENOVA_TRAINING_DESIGN.md 13.4). Returns True when it changed the
    setting.
    """
    from api.param_defaults import full_finetune_forces_stochastic_rounding

    if not full_finetune_forces_stochastic_rounding("sensenova"):
        return False
    if getattr(trainer, "optimizer_stochastic_rounding", False):
        return False
    trainer.optimizer_stochastic_rounding = True
    prefix = getattr(trainer, "log_prefix", "[SenseNova]")
    emit_training_warning(
        f"SenseNova full fine-tuning: optimizer_stochastic_rounding was "
        f"off and has been turned on for this run. It is not optional here. The "
        f"trainable half is bf16 with no fp32 master, and under round-to-nearest "
        f"84.5% of a bf16 tensor's elements never move at any step count while "
        f"the loss falls normally (measured, SENSENOVA_TRAINING_DESIGN.md 6.3). "
        f"This route cannot be run with it off; LoRA training on this "
        f"architecture honours the setting.",
        code="sensenova_stochastic_rounding_forced",
        prefix=prefix,
    )
    return True


def assert_full_finetune_stochastic_rounding_attached(
    trainer: Any, optimizer_type: Any = None
) -> None:
    """Fail if the update seam is not actually carrying stochastic rounding.

    Checks the mechanism, not the flag: ``_attach_stochastic_rounding`` reports
    coverage by wrapping ``step_param`` (the entry point this route's
    post-accumulate-grad hooks call), and the hooks resolve
    ``self.optimizer.step_param`` at call time, so the wrapper installed after
    hook registration is the one that runs. A route whose flag is set but whose
    seam is unwrapped writes round-to-nearest while every log line says
    otherwise, which is the failure this whole contract exists to prevent.

    The ring-buffer optimizers are a second, equally valid seam: neither defines
    ``step_param`` (they register their own hooks), yet both round stochastically
    inside their own update, which is why ``_attach_stochastic_rounding`` skips
    them. Checked for ``step_param`` they would fail on a correct configuration,
    so membership in ``_NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS`` counts as
    coverage here (U-2-6).
    """
    from api.param_defaults import full_finetune_forces_stochastic_rounding
    from core.training.base_trainer import BaseTrainer
    from core.training.optimizers.stochastic_rounding import NATIVE_ATTR, WRAPPED_ATTR

    # The one list, read off BaseTrainer, so this cannot disagree with
    # _attach_stochastic_rounding about which optimizers it skips.
    _NATIVE_SR = BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS

    if not full_finetune_forces_stochastic_rounding("sensenova"):
        # Same table the enforcement reads, so the two cannot disagree about
        # whether this route is supposed to be covered.
        return

    if optimizer_type is not None and str(optimizer_type).strip().lower() in _NATIVE_SR:
        # No step_param exists to inspect, and none should: the rounding is in
        # the optimizer's own update. Verified separately by the update-nonzero
        # census (bf16_stochastic_rounding_test), which measures the effect
        # rather than the attachment.
        return

    groups = getattr(trainer, "fused_optimizer_groups", None)
    optimizers = (
        list(groups.optimizers) if groups is not None
        else [getattr(trainer, "optimizer", None)]
    )
    for optimizer in optimizers:
        step_param = getattr(optimizer, "step_param", None)
        if not callable(step_param):
            raise RuntimeError(
                f"SenseNova full fine-tuning found no per-parameter step_param on "
                f"{type(optimizer).__name__} (optimizer={optimizer_type}). Its "
                f"post-accumulate-grad hooks call optimizer.step_param, and "
                f"stochastic rounding is interposed on it; without it the run "
                f"would neither update per parameter nor carry sub-ULP updates."
            )
        covered = getattr(step_param, WRAPPED_ATTR, False) or getattr(
            step_param, NATIVE_ATTR, False
        )
        if not covered:
            raise RuntimeError(
                f"SenseNova full fine-tuning has optimizer_stochastic_rounding set "
                f"but nothing is interposed on {type(optimizer).__name__}."
                f"step_param (optimizer={optimizer_type}). The bf16 updates this "
                f"route applies would be rounded to nearest, which discards every "
                f"update below half a ULP. This is an internal inconsistency, not "
                f"a setting."
            )


def _quantized_linear_flavours() -> "dict[str, type]":
    """The EXACT quantized-Linear classes this guard knows how to census.

    ``ConvRotInt8Linear`` subclasses ``Int8Linear``, so the census below keys on
    ``type(m) is cls``, never ``isinstance``: an isinstance census would fold a
    ConvRot base into the plain-int8 count and accept it silently.
    """
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.common.w4a8_linear import W4A8Linear
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    return {
        "Int8Linear": Int8Linear,
        "ConvRotInt8Linear": ConvRotInt8Linear,
        "Fp8Linear": Fp8Linear,
        "W4A8Linear": W4A8Linear,
    }


def _census_quantized_linears(
    transformer: nn.Module,
) -> "tuple[dict[str, int], dict[str, int]]":
    """Count the decoder's quantized Linears by EXACT class. Returns (known, unknown)."""
    flavours = _quantized_linear_flavours()
    known = tuple(flavours.values())
    counts = {label: 0 for label in flavours}
    unknown: "dict[str, int]" = {}
    for module in transformer.modules():
        if not isinstance(module, known):
            continue
        for label, cls in flavours.items():
            if type(module) is cls:
                counts[label] += 1
                break
        else:
            # A quantized class added later must refuse loudly here rather than
            # be counted as whichever known class it happens to subclass.
            name = type(module).__name__
            unknown[name] = unknown.get(name, 0) + 1
    return counts, unknown


def _own_save_format_remedy(source_metadata: Any) -> str:
    """The extra sentence a checkpoint THIS REPO wrote earns on refusal.

    A full fine-tune's own output is the natural thing to resume from, and at
    the shipped defaults it is exactly what a restart selects:
    ``TRAINING_DEFAULTS["resume_from_checkpoint"] = "latest"`` and
    ``_build_train_section`` always writes it. But only the ``int8`` format
    keeps all 588 decoder Linears in one quantized flavour, so the default
    ``mixed`` and the ``bf16`` option are both refused here -- correctly, and
    with a message that otherwise asks the user to "select the plain-int8
    checkpoint" without saying that the file they are pointing at was shaped by
    a setting they chose when they created the run. The file says which one, so
    the message can too.
    """
    metadata = source_metadata or {}
    effective = str(metadata.get("sensenova_save_format") or "").strip()
    if not effective:
        return ""
    requested = str(metadata.get("sensenova_save_format_requested") or "").strip()
    branch = str(metadata.get("sensenova_trained_branch") or "").strip()
    requested_note = (
        f" (requested '{requested}', written as '{effective}')"
        if requested and requested != effective else ""
    )
    return (
        f" This checkpoint was written by this repo's own full fine-tune as "
        f"sensenova_full_finetune_save_format='{effective}'{requested_note}"
        + (f", branch '{branch}'" if branch else "")
        + ". Of the three formats only 'int8' keeps all 588 decoder Linears in "
        "one quantized flavour, so only 'int8' can be handed to a NEW run as "
        "model_path -- 'mixed' and 'bf16' are outputs, not distributable bases, "
        "and 'int8' is lossy on save (see its API description). RESUMING the run "
        "that wrote this file is a different question and does not go through "
        "this gate: see accept_resume_shaped_base, which takes 'mixed' on a "
        "single-half run and 'bf16' on a both-halves run losslessly, but only "
        "for a checkpoint the resume path selected out of that run's own "
        "output_dir."
    )


# The one on-disk format whose resident layout equals what a fresh int8 load
# plus ``materialize_int8_decoder_linears(branch)`` produces, per branch.
# ``mixed`` keeps the untrained half's int8 codes untouched; with both halves
# trained there is no int8 half left and the writer degenerates it to ``bf16``.
_SENSENOVA_RESUME_FORMAT_FOR_BRANCH = {"gen": "mixed", "und": "mixed", "both": "bf16"}

_RESUME_ENTRY_SUFFIXES = (".safetensors.index.json", ".safetensors")
_RESUME_STEP_RE = re.compile(r"_step_(\d+)\Z")


def _resume_entry_stem(name: str) -> Optional[str]:
    """``name`` with a resume-entry suffix removed, or ``None`` if it has none."""
    for suffix in _RESUME_ENTRY_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _half_linear_layout(transformer: nn.Module) -> "Dict[str, Dict[str, Any]]":
    """Per MoT half, how its decoder Linears are currently shaped.

    ``float`` is a materialized/never-quantized ``nn.Linear``; ``int8`` is a
    PLAIN ``Int8Linear`` (``type is``, so a ``ConvRotInt8Linear`` lands in
    ``other`` instead of being folded in by ``isinstance``).
    """
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets

    layout: "Dict[str, Dict[str, Any]]" = {}
    for half in ("gen", "und"):
        counts = {"float": 0, "int8": 0, "other": 0}
        first_other = None
        for path, _parent, _attr, module in iter_sensenova_lora_targets(
            transformer, branch=half
        ):
            weight = getattr(module, "weight", None)
            if type(module) is Int8Linear:
                counts["int8"] += 1
            elif isinstance(weight, nn.Parameter) and weight.dtype.is_floating_point:
                counts["float"] += 1
            else:
                counts["other"] += 1
                if first_other is None:
                    first_other = f"{path} ({type(module).__name__})"
        layout[half] = {
            "counts": counts,
            "total": sum(counts.values()),
            "first_other": first_other,
        }
    return layout


def _resume_selected_checkpoint(trainer: Any) -> Optional[Path]:
    """The file this trainer is RESUMING from, or ``None`` if it is not resuming.

    Not "a file that says it is ours". WHAT IS ACTUALLY CHECKED is the request
    plus the path's SHAPE, not the identity of the caller: ``resume_from_checkpoint``
    is set, ``model_path`` names an existing file, that file's parent resolves to
    this run's own ``output_dir``, and its name is ``{run_name}_step_<digits>``
    with a checkpoint suffix. That is a proxy for "the resume machinery
    substituted this path" -- a sound one, because ``_load_checkpoint_as_base``
    is the only thing that writes a path of that shape into ``model_path``, but a
    proxy. Its residual is an operator deliberately copying a file into this
    run's output directory under this run's checkpoint name, which is the same
    class of act as pointing ``model_path`` anywhere.

    THAT RESIDUAL HAS A SHARP EDGE. Weights from a DIFFERENT run so placed would
    pass every check here and every check in ``accept_resume_shaped_base`` -- the
    layout and the stamp describe a branch and a format, not an identity -- and
    would then be resumed against THIS run's ``_optimizer.pt`` and
    ``_state.json``. Nothing warns, because the sidecar warning fires on their
    ABSENCE, and they are present. There is no defence here that does not amount
    to trusting a claim; recorded so the next reader does not have to rediscover
    it (SENSENOVA_TRAINING_DESIGN.md 6.4).
    """
    if not str(getattr(trainer, "resume_from_checkpoint", "") or "").strip():
        return None
    model_path = getattr(trainer, "model_path", None)
    output_dir = getattr(trainer, "output_dir", None)
    run_name = str(getattr(trainer, "run_name", "") or "").strip()
    if not model_path or not output_dir or not run_name:
        return None
    path = Path(str(model_path))
    if not path.is_file():
        return None
    try:
        if path.parent.resolve() != Path(str(output_dir)).resolve():
            return None
    except OSError:
        return None
    stem = _resume_entry_stem(path.name)
    if stem is None:
        return None
    if not stem.startswith(f"{run_name}_step_") or not _RESUME_STEP_RE.search(stem):
        return None
    return path


def accept_resume_shaped_base(
    trainer: Any,
    transformer: nn.Module,
    metadata: Any,
    *,
    branch: str,
) -> Optional[str]:
    """Accept a full fine-tune's OWN checkpoint as a resume base, losslessly.

    Returns the accepted format label, or ``None`` to leave the decision to
    ``_assert_supported_quantized_training_base`` unchanged.

    WHY THIS IS A SEPARATE QUESTION from that gate. That gate answers "is this a
    distributable base a new run may be pointed at", and its answer for anything
    but plain int8 is no. Resume asks something narrower: is the tree in front of
    us the layout this run was already training in. For a single-half run that
    layout is 294 float Linears on the trained half and 294 plain ``Int8Linear``
    on the frozen one -- exactly what the ``mixed`` writer emits, and exactly
    what a fresh int8 load plus ``materialize_int8_decoder_linears`` produces.
    For a both-halves run it is 588 float Linears, which is the ``bf16`` writer.
    Both are bit-exact round trips (bf16 in, bf16 out, ``assign=True``), so this
    is the only lossless resume the ``both`` branch has: its ``int8`` alternative
    requantizes every trained weight on every save. MEASURED SEPARATELY, and not
    equally: the write/read round trip is byte-identical for both layouts
    (SENSENOVA_TRAINING_DESIGN.md 13.4 U-2-5, 294/294 and 588/588 SHA-256), but a
    real RESUME has been run only for ``gen``/``mixed`` (8.3.4). The ``both``
    case is the same write/read pair feeding the same acceptance, which is an
    inference, not a measurement.

    WHAT IS TRUSTED. The class census on the CONSTRUCTED TREE decides what is
    accepted; metadata never widens that. Metadata is required and required to
    agree -- a file with no ``sensenova_trained_branch`` / ``sensenova_save_format``
    is refused, and one whose claim contradicts the tree or the run's branch is
    refused by name. So the claim is a necessary condition that can only narrow
    acceptance, and the sentence "a file claiming to be ours is not proof that it
    is" is answered by not relying on the claim for anything load-bearing.

    Materialization is SKIPPED for an accepted base: its halves are already in
    the shape it produces, and running it would refuse (it requires every target
    to be a plain ``Int8Linear``).

    THE bf16 SINGLE-HALF FALLBACK: a single-half ``bf16`` save is float on
    both halves, indistinguishable by class census from a genuine ``both``
    resume, so it is accepted only when the metadata names THIS branch (not
    ``'both'``), and even then the frozen half is not trusted from the
    checkpoint -- it is restored from the run's own base model and verified
    bit-identical first (``restore_sensenova_frozen_half_from_base``).
    """
    layout = _half_linear_layout(transformer)
    if sum(layout[half]["counts"]["float"] for half in ("gen", "und")) == 0:
        # The distributed int8 layout, or something the shipped gate refuses.
        return None
    checkpoint = _resume_selected_checkpoint(trainer)
    if checkpoint is None:
        # A float-carrying tree handed over as model_path: not this path's
        # question, and the shipped gate already refuses it by name.
        return None

    census = "; ".join(
        f"{half} half: " + ", ".join(f"{k}={v}" for k, v in layout[half]["counts"].items())
        + (f", first unexpected {layout[half]['first_other']}"
           if layout[half]["first_other"] else "")
        for half in ("gen", "und")
    )
    # Named WITHOUT its suffix: sensenova_full_finetune_resume_base_test.py
    # asserts the accepted-branch info message below stays free of
    # ".safetensors" (test_the_acceptance_is_announced_on_the_channel_not_only_stdout),
    # so the refusal messages below match that convention rather than
    # switching case by case.
    entry = _resume_entry_stem(checkpoint.name) or checkpoint.stem
    step = int(_RESUME_STEP_RE.search(entry).group(1))
    trained_halves = ("gen", "und") if branch == "both" else (branch,)
    frozen_halves = tuple(h for h in ("gen", "und") if h not in trained_halves)
    HALF = _SENSENOVA_QUANT_LINEAR_COUNT // 2
    expected_format = _SENSENOVA_RESUME_FORMAT_FOR_BRANCH[branch]

    for half in trained_halves:
        if layout[half]["counts"]["float"] != HALF:
            raise RuntimeError(
                f"SenseNova cannot resume the {branch!r} branch from "
                f"{entry}: the {half} half of its decoder is not the "
                f"shape this run trains in. Expected all {HALF} of its "
                f"Linears to be floating-point nn.Linear; got {census}. A "
                f"resume of this branch is only lossless from a checkpoint "
                f"written as sensenova_full_finetune_save_format="
                f"'{expected_format}'; the other formats leave the decoder "
                f"in a different layout and are refused rather than "
                f"reshaped."
            )

    # A single-half branch also accepts a fully-float frozen half (the bf16
    # writer's output): restorable from the run's own base model, verified
    # below. 'int8' requantizes the TRAINED half on save, so it stays refused.
    using_bf16_fallback = False
    for half in frozen_halves:
        if layout[half]["counts"]["int8"] == HALF:
            continue
        if layout[half]["counts"]["float"] == HALF:
            using_bf16_fallback = True
            expected_format = "bf16"
            continue
        raise RuntimeError(
            f"SenseNova cannot resume the {branch!r} branch from "
            f"{entry}: the {half} half of its decoder is not the shape "
            f"this run trains in. Expected all {HALF} of its Linears to be "
            f"plain Int8Linear (the 'mixed' layout), or floating-point "
            f"nn.Linear (the 'bf16' layout, restorable from this run's own "
            f"base model); got {census}. A resume of this branch is only "
            f"lossless from a checkpoint written as "
            f"sensenova_full_finetune_save_format='mixed' or 'bf16'; "
            f"'int8' requantizes the trained half on every save and is "
            f"refused unconditionally."
        )

    claimed = metadata or {}
    claimed_branch = str(claimed.get("sensenova_trained_branch") or "").strip()
    claimed_format = str(claimed.get("sensenova_save_format") or "").strip()
    if not claimed_branch or not claimed_format:
        raise RuntimeError(
            f"SenseNova refuses to resume from {entry}: it carries the "
            f"decoder layout a {branch!r}-branch resume needs, but not this "
            f"repo's own save stamp (sensenova_trained_branch / "
            f"sensenova_save_format). Both the structure and the stamp are "
            f"required -- the structure decides what is loaded, the stamp is what "
            f"rules out an unrelated file that happens to have the same layout. "
            f"Every checkpoint this repo's full fine-tune writes carries both."
        )
    if claimed_branch != branch or claimed_format != expected_format:
        raise RuntimeError(
            f"SenseNova refuses to resume from {entry}: its own metadata "
            f"disagrees with what it is being resumed as. The file says "
            f"sensenova_trained_branch={claimed_branch!r}, "
            f"sensenova_save_format={claimed_format!r}; this run trains the "
            f"{branch!r} branch, whose resumable format is {expected_format!r}. "
            f"The tree loaded as: {census}. A file whose stamp and whose tensors "
            f"tell different stories is refused rather than believed on either."
        )

    if using_bf16_fallback:
        frozen_half = frozen_halves[0]
        base_path = _sensenova_resume_base_model_path(trainer, claimed)
        if not base_path:
            raise RuntimeError(
                f"SenseNova refuses to resume the {branch!r} branch from "
                f"{entry}: it was written as "
                f"sensenova_full_finetune_save_format='bf16', which drops "
                f"the frozen {frozen_half} half's int8 weight_scale, and "
                f"this run's base model path is unknown (no "
                f"sensenova_base_model_path in the checkpoint's metadata, "
                f"and no configured base model path on this run), so the "
                f"frozen half cannot be restored."
            )
        _sensenova_check_base_identity_hint(claimed, base_path, entry=entry)
        from core.models.sensenova.loader import restore_sensenova_frozen_half_from_base

        try:
            restored = restore_sensenova_frozen_half_from_base(
                transformer,
                frozen_half=frozen_half,
                base_path=base_path,
                compute_dtype=trainer.weight_dtype,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"SenseNova refuses to resume the {branch!r} branch from "
                f"{entry}: its frozen {frozen_half} half needs the run's "
                f"base model to restore int8 weights, but {exc}"
            ) from exc
        if restored != HALF:
            raise RuntimeError(
                f"SenseNova bf16 resume fallback restored {restored} of "
                f"{HALF} {frozen_half}-half decoder Linear(s) from "
                f"{base_path!r} for {entry}; expected all of them. Refusing "
                f"rather than resuming with a partially restored frozen "
                f"half."
            )

    aux_base = f"{getattr(trainer, 'run_name', '')}_step_{step:06d}"
    missing = [
        name
        for name in (f"{aux_base}_optimizer.pt", f"{aux_base}_state.json")
        if not (checkpoint.parent / name).is_file()
    ]
    if missing:
        # The weights resume losslessly either way; these two carry the Adafactor
        # state and the epoch/batch position. Losing them silently is what the
        # audit this path answers objected to.
        emit_training_warning(
            f"SenseNova is resuming from {entry} with "
            f"{' and '.join(missing)} absent. The decoder weights resume exactly, "
            f"but the optimizer state and/or the epoch/batch position for step "
            f"{step} cannot be restored from this output directory; the run "
            f"continues from global step {step} with whatever of the two is "
            f"present.",
            code="sensenova_resume_state_incomplete",
            prefix=getattr(trainer, "log_prefix", "[SenseNova]"),
        )

    # On the channel, not just stdout: relaxing a safety gate is at least as
    # worth telling the user about as the degraded case above, which warns.
    detail = (
        f"its frozen {frozen_halves[0]} half was restored from this run's own "
        f"base model, verified tensor-for-tensor before the swap"
        if using_bf16_fallback else
        "the trained half is already floating point, so it is loaded as "
        "saved and not re-materialized from int8"
    )
    emit_training_event(
        "info",
        f"SenseNova is resuming the {branch!r} branch from its own checkpoint "
        f"{entry} at step {step}, accepted losslessly as "
        f"sensenova_full_finetune_save_format='{claimed_format}' ({census}); "
        f"{detail}.",
        code="sensenova_resume_base_accepted",
        prefix=getattr(trainer, "log_prefix", "[SenseNova]"),
    )
    return claimed_format


def _sensenova_resume_base_model_path(trainer: Any, claimed_metadata: Dict[str, Any]) -> Optional[str]:
    """The base model this run trained against, for the bf16 resume fallback.

    Prefers the checkpoint's OWN stamp (``sensenova_base_model_path``, written
    by ``sensenova_adapter.save_checkpoint`` going forward, self-describing
    even if this trainer is somehow reconstructed with a different config).
    Falls back to this run's configured base model path -- what every
    checkpoint written before that stamp existed (e.g. run 125) has instead.
    """
    stamped = str((claimed_metadata or {}).get("sensenova_base_model_path") or "").strip()
    if stamped:
        return stamped
    configured = str(getattr(trainer, "configured_model_path", "") or "").strip()
    return configured or None


def _sensenova_check_base_identity_hint(
    claimed_metadata: Dict[str, Any], base_path: str, *, entry: str
) -> None:
    """Fast pre-check: refuse early if the checkpoint's stamped base size disagrees.

    Only fires when the checkpoint carries ``sensenova_base_model_identity``
    (checkpoints written before that stamp existed have nothing to compare).
    This is a cheap (stat-only) hint, not the proof -- the per-Linear dequant
    compare in ``restore_sensenova_frozen_half_from_base`` is what actually
    verifies the content, and runs regardless of what this finds.
    """
    import os

    stamped_size = str((claimed_metadata or {}).get("sensenova_base_model_identity") or "").strip()
    if not stamped_size:
        return
    try:
        actual_size = str(os.path.getsize(base_path))
    except OSError as exc:
        raise RuntimeError(
            f"SenseNova refuses to resume from {entry}: its stamped base "
            f"model {base_path!r} could not be read ({exc})."
        ) from exc
    if actual_size != stamped_size:
        raise RuntimeError(
            f"SenseNova refuses to resume from {entry}: the base model at "
            f"{base_path!r} is {actual_size} byte(s), but the checkpoint "
            f"was stamped as trained against a base of {stamped_size} "
            f"byte(s). This is not the base this run trained against."
        )


def _assert_supported_quantized_training_base(
    transformer: nn.Module,
    *,
    training_method: str = "lora",
    source_metadata: Any = None,
) -> None:
    """Require all 588 decoder Linears to be ONE supported quantized flavour.

    LoRA (the default, and every method that is not full fine-tuning) accepts
    the plain-int8 and the ConvRot-int8 checkpoints: the adapters wrap the
    quantized module and never differentiate its weight.

    Full fine-tuning accepts the plain-int8 base ONLY, because the trainable
    half is dequantized to real Parameters at load
    (``materialize_int8_decoder_linears``, SENSENOVA_TRAINING_DESIGN.md 6.4
    route (a)) and a ConvRot base cannot be dequantized without inverting its
    rotation.

    Both refuse a mixed base, an off-count base, an unrecognized subclass of a
    known quantized Linear, and an unquantized bf16 base. Fp8/W4A8 are censused
    for the diagnostic but not accepted: no such SenseNova base exists, so
    accepting one would ship an untested path.
    """
    counts, unknown = _census_quantized_linears(transformer)
    present = [label for label, n in counts.items() if n]
    pure = (
        not unknown
        and len(present) == 1
        and counts[present[0]] == _SENSENOVA_QUANT_LINEAR_COUNT
    )
    census = ", ".join(
        f"{label}={n}" for label, n in list(counts.items()) + list(unknown.items())
    )

    if training_method == _FULL_FINETUNE_METHOD:
        if not pure or present[0] != "Int8Linear":
            raise RuntimeError(
                "SenseNova full fine-tuning (training_method='full_finetune') materializes the "
                f"trainable decoder half to floating point at load, so it requires a base "
                f"whose {_SENSENOVA_QUANT_LINEAR_COUNT} decoder Linears are all plain "
                f"Int8Linear; got {census}. A ConvRot-int8 base is refused because "
                "dequantizing it would require inverting its Hadamard rotation; an "
                "unquantized bf16 base is refused because none exists for this repo and "
                "the other two supply routes are undecided (see "
                "docs/guides/SENSENOVA_TRAINING_DESIGN.md 6.4). Remedy: select the "
                "plain-int8 checkpoint, or set training_method='lora', which trains on "
                "either quantized base."
                + _own_save_format_remedy(source_metadata)
            )
        return

    accepted = ("Int8Linear", "ConvRotInt8Linear")
    if not pure or present[0] not in accepted:
        raise RuntimeError(
            "SenseNova training requires a base whose "
            f"{_SENSENOVA_QUANT_LINEAR_COUNT} decoder Linears are all ONE supported "
            f"quantized flavour (all Int8Linear, or all ConvRotInt8Linear); got {census}. "
            "A mixed or partially quantized base is refused, and so is an unquantized "
            "bf16 base -- no bf16 SenseNova checkpoint exists for this repo to train on yet."
            + _own_save_format_remedy(source_metadata)
        )


def _assert_pixel_head_fm_decoder(transformer: nn.Module) -> None:
    """Require the vendor ``use_pixel_head`` fm-head branch.

    ``train_step`` inlines only that branch of ``_t2i_predict_v``: it feeds the
    fm_head a ``b c h w`` map and un-patchifies the ``b 3 H W`` result. The other
    two vendor branches take token-shaped input -- ``use_deep_fm_head`` also
    takes a second ``t`` argument -- and neither is implemented here, so refuse
    rather than reshape into a head that cannot accept it. A missing attribute
    means an unknown tree and is refused for the same reason.
    """
    missing = [
        name
        for name in ("use_pixel_head", "use_deep_fm_head")
        if not hasattr(transformer, name)
    ]
    if missing:
        raise RuntimeError(
            "SenseNova training requires a vendor transformer exposing "
            f"use_pixel_head and use_deep_fm_head; this tree is missing "
            f"{', '.join(missing)}, so the fm-head layout it was built with is "
            "unknown and cannot be assumed to be the pixel-head (ConvDecoder) one "
            "that train_step implements."
        )
    if transformer.use_deep_fm_head:
        raise RuntimeError(
            "SenseNova training does not implement the vendor _t2i_predict_v "
            "use_deep_fm_head branch (FlowMatchingHead called as fm_head(x, t) on "
            "token-shaped input); this checkpoint has fm_head_layers > 2. Only the "
            "use_pixel_head (ConvDecoder) branch is implemented."
        )
    if not transformer.use_pixel_head:
        raise RuntimeError(
            "SenseNova training does not implement the vendor _t2i_predict_v plain "
            "fm_head branch (nn.Sequential called on token-shaped input); this "
            f"checkpoint has use_pixel_head={transformer.use_pixel_head!r}. Only the "
            "use_pixel_head (ConvDecoder) branch is implemented."
        )


def setup_attention_backend(trainer: Any, backend: str) -> None:
    from core.attention import AttentionMode
    from core.models.sensenova.sensenova_pipeline_ops import set_attention_backend

    resolved = trainer._resolve_training_backend(backend)
    count = set_attention_backend(trainer.transformer, resolved, AttentionMode.TRAINING)
    expected = len(trainer.transformer.language_model.model.layers)
    if count != expected:
        raise RuntimeError(
            f"SenseNova configured {count} attention module(s), expected {expected}"
        )


def load_components(trainer: Any) -> None:
    """Load the SenseNova graph, method-aware.

    LoRA leaves the int8 base exactly as the loader produced it -- the adapters
    wrap the quantized modules. Full fine-tuning materializes the half it will
    train to real ``nn.Parameter`` weights here, on the CPU, before the model is
    staged to the GPU (SENSENOVA_TRAINING_DESIGN.md 6.4 route (a)).
    """
    if getattr(trainer, "blocks_to_swap", 0) != 0:
        raise ValueError("SenseNova training does not implement blocks_to_swap; set it to 0")
    from core.models.sensenova.loader import load_sensenova_from_path

    training_method = resolve_training_method(trainer)
    # Resolved BEFORE the 17.6 GiB load so a contradictory train_unet /
    # train_text_encoder pair -- or a configuration the fused backward pass
    # cannot serve -- is refused without paying for it.
    branch = None
    if training_method == _FULL_FINETUNE_METHOD:
        branch = resolve_full_finetune_branch(trainer)
        assert_full_finetune_contract(trainer)

    components = load_sensenova_from_path(trainer.model_path, torch_dtype=trainer.weight_dtype)
    trainer.transformer = components["transformer"]
    # A full fine-tune resuming from its OWN checkpoint is a different question
    # from which base a new run may be pointed at; only the resume path can
    # widen, and only to the layout it was already training in.
    resumed_format = None
    if branch is not None:
        resumed_format = accept_resume_shaped_base(
            trainer, trainer.transformer, components.get("metadata"), branch=branch
        )
    trainer.sensenova_resumed_save_format = resumed_format
    if resumed_format is None:
        _assert_supported_quantized_training_base(
            trainer.transformer,
            training_method=training_method,
            source_metadata=components.get("metadata"),
        )
    _assert_pixel_head_fm_decoder(trainer.transformer)
    if branch is not None and resumed_format is None:
        from core.models.sensenova.loader import materialize_int8_decoder_linears

        materialize_int8_decoder_linears(
            trainer.transformer, branch=branch, dtype=trainer.weight_dtype
        )
    trainer.transformer_original = trainer.transformer
    trainer.transformer_uncond = None
    trainer.tokenizer = components["tokenizer"]
    trainer.sensenova_model_config = components.get("config")
    # The geometry block THIS load accepted, for an export to re-embed verbatim.
    trainer.sensenova_config_dict = components.get("config_dict")
    trainer.text_encoder = None
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    # None for the pixel model; a real module when the base declares a swapped
    # VAE, which `apply_latent_space` below folds into the run's wiring.
    trainer.vae = components.get("vae")
    trainer.unet = None
    trainer.scheduler = None
    trainer.noise_scheduler = None
    trainer.layer_offload_conductor = None
    # The base's declared latent space, then this run's own `vae_swap_source`.
    # BEFORE the freeze: a swap rebinds two Parameters, and fresh ones default
    # to requires_grad=True, so the freeze has to see them.
    from core.training.vae_swap import apply_latent_space

    apply_latent_space(trainer, components.get("declared_vae"))
    if getattr(trainer, "vae", None) is not None:
        trainer.vae.requires_grad_(False)
        trainer.vae.eval()
        trainer.vae.to(trainer.device)
    # Materialized full-FT weights are frozen here too; unfreezing them is the
    # adapter's job, as it is for every other architecture.
    trainer.transformer.requires_grad_(False)
    trainer.transformer.train()
    trainer.transformer.to(trainer.device)
    # Training mode must be stamped even when the selected backend is native.
    setup_attention_backend(trainer, trainer.attention_backend)

    # CANDIDATE fused frozen-base forward, opt-in via SUSHI_CONVROT_TRAIN_FUSED=1.
    # Placed after requires_grad_(False) and after any full-FT materialization,
    # so a trainable half is already real nn.Linear and cannot match; the helper
    # raises if it ever does.
    from core.models.common.quantized_frozen_training import (
        enable_frozen_training_fused,
        frozen_training_fused_requested,
    )

    if frozen_training_fused_requested():
        enable_frozen_training_fused(
            trainer.transformer, label="sensenova training transformer"
        )


def _load_reference_images(reference_image_paths: Optional[List[str]]) -> list:
    """Open reference PILs HERE, never through the trainer's image pipeline.

    That pipeline resizes to the target's bucket and normalizes to [-1,1]; the
    understanding tower wants a per-reference smart-resize with ImageNet stats.
    The two are shape-compatible at patchify, so mixing them would silently
    train mis-normalized conditioning instead of raising. Conversion (incl. the
    RGBA contrasting-background flatten) belongs to vendor `load_image_native`,
    so the PIL is handed over untouched.
    """
    from PIL import Image

    from core.pipeline_backends.sensenova import SENSENOVA_MAX_REFERENCE_IMAGES

    paths = [path for path in (reference_image_paths or []) if path]
    if not paths:
        return []
    if len(paths) > SENSENOVA_MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"SenseNova training accepts at most {SENSENOVA_MAX_REFERENCE_IMAGES} "
            f"reference image(s) per item (the inference cap); got {len(paths)}"
        )
    return [Image.open(path) for path in paths]


def _decoder_attention_dropout(transformer: nn.Module, context: str) -> float:
    """The vendor decoder stack's configured attention dropout."""
    language_model = getattr(transformer, "language_model", None)
    llm = getattr(language_model, "model", None) if language_model is not None else None
    if llm is None or getattr(llm, "layers", None) is None:
        raise RuntimeError(
            f"SenseNova {context} requires the vendor language_model.model decoder "
            f"stack; this tree does not expose it"
        )
    return float(getattr(getattr(llm, "config", None), "attention_dropout", 0.0) or 0.0)


def assert_understanding_training_supported(transformer: nn.Module) -> None:
    """Refuse configurations the differentiable prefix pass cannot serve.

    Fail-closed on a non-zero ``attention_dropout``: the vendor attention keeps
    ``dropout=0.0 if not self.training else self.attention_dropout``, and the
    training path stamps ``transformer.train()``, so a future non-zero config
    would make the checkpointed prefix RECOMPUTE stochastic -- the recomputed
    K/V would silently differ from the ones the forward produced. Upstream's
    default is 0.0, so this refuses nothing that exists today.
    """
    dropout = _decoder_attention_dropout(transformer, "understanding-branch training")
    if dropout != 0.0:
        raise RuntimeError(
            "SenseNova understanding-branch training requires attention_dropout=0.0, "
            f"got {dropout}. The vendor attention applies dropout whenever the module "
            "is in train() mode, which the training path stamps, so a checkpointed "
            "prefix recompute would not reproduce the K/V of its own forward."
        )


def assert_full_finetune_dropout_free(transformer: nn.Module) -> None:
    """Refuse a non-zero ``attention_dropout`` for a full fine-tune of any half.

    The understanding-branch guard above covers the halves it names, but the
    default branch for this route is generation-only, and that configuration
    stamps ``train()`` on the WHOLE MoT decoder -- both halves live in one
    ``language_model``. The prompt prefix is built by the understanding half on
    every step (``encode_prompt``, under ``no_grad`` but not under ``eval()``),
    so a non-zero dropout would randomly zero attention weights in the
    conditioning the loss is computed against, differently on every step and
    differently from inference, with nothing raising. Upstream's default is 0.0,
    so this refuses nothing that exists today.
    """
    dropout = _decoder_attention_dropout(transformer, "full fine-tuning")
    if dropout != 0.0:
        raise RuntimeError(
            "SenseNova full fine-tuning requires attention_dropout=0.0, got "
            f"{dropout}. The training path stamps train() on the whole MoT decoder, "
            "and the vendor attention applies dropout whenever the module is in "
            "that mode, so the prompt prefix -- which the understanding half builds "
            "on every step, including when only the generation half is trained -- "
            "would be a different random projection each step and would not match "
            "the one inference builds."
        )


class _TrainingPrefixLayer:
    """One prefix cache layer built from checkpoint OUTPUTS, not cache writes."""

    flash_k_cache = None
    flash_v_cache = None
    flash_prefix_len = None

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class _TrainingPrefixCache:
    """The ``past_key_values`` surface the generation forward actually reads."""

    _kv_cache_streamer = None
    _kv_cache_streamer_branch = None
    # Vendor ``PackedSegments`` when the K/V hold several prompts end to end.
    packed = None

    def __init__(self, layers: "list[_TrainingPrefixLayer]", packed=None):
        self.layers = layers
        self.packed = packed

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.layers[layer_idx].keys.shape[-2])


def forward_und_prefix_layers(
    model: Any,
    input_ids: Optional[torch.Tensor],
    indexes: torch.Tensor,
    attention_mask: Any,
    *,
    inputs_embeds: Optional[torch.Tensor] = None,
    checkpoint_layers: bool = True,
    packed: Any = None,
) -> _TrainingPrefixCache:
    """Run the understanding decoder stack and return a differentiable prefix.

    ``packed`` (vendor ``PackedSegments``) marks a batch of prompts laid end to
    end along the sequence axis with batch dim 1: ``attention_mask`` must then
    be the packed block-causal mask, and the causal fast path goes through the
    varlen conduit entry so no segment attends into another.

    K/V leave each layer as explicit checkpoint OUTPUTS (the vendor ``return_kv``
    seam) rather than through ``past_key_values.update()``: that write is a
    checkpoint-segment side effect, so a non-reentrant recompute would append a
    second time, and a side-effected tensor is not an output autograd can route
    a gradient through.

    U-0's bitwise K/V parity against vendor ``_t2i_prefix_forward`` (42/42
    layers, checkpointed and not) was measured on ``probes/sensenova_und_prefix
    .training_prefix_forward``, a probe-local twin of this loop rather than this
    function -- so the claim transfers by the two being the same construction,
    not by this function having been run under it. Noted rather than repaired
    because it is also why re-running that gate for U-3 would have proved
    nothing: the twin has no ``inputs_embeds`` parameter to exercise.

    ``inputs_embeds`` is the reference-conditioned entry (Phase U-3): vendor
    ``_build_it2i_inputs`` splices the understanding ViT's rows into the token
    embeddings and hands back EMBEDS, so a loop that only ever calls
    ``embed_tokens`` cannot consume them. Same exclusive-or contract as vendor
    ``Qwen3Model.forward``; the decoder stack below is byte-identical either way,
    which is the whole of what reference conditioning needs from this loop.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError(
            "SenseNova prefix loop takes exactly one of input_ids or inputs_embeds"
        )
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("SenseNova understanding prefix model has no decoder layers")
    config = getattr(model, "config", None)
    depth = int(getattr(config, "num_hidden_layers", len(layers)) or len(layers))
    layers = list(layers)[:depth]
    for layer in layers:
        if getattr(layer, "attention_type", None) not in attention_mask:
            raise ValueError(
                f"SenseNova prefix has no mask for attention type "
                f"{getattr(layer, 'attention_type', None)!r}"
            )
    # Vendor Qwen3Model.forward sets this on the pre-built-mask path.
    model.current_index = indexes[0].max()

    # Equivalence proof: see is_plain_causal_thw_index in modeling_qwen3.py.
    # Computed ONCE for the whole stack (one host sync), not per layer.
    from core.models.sensenova.vendor.modeling_qwen3 import (
        is_plain_causal_thw_index,
        is_plain_causal_thw_index_packed,
    )

    if packed is not None:
        if indexes.shape[-1] != packed.total:
            raise ValueError(
                f"SenseNova packed prefix: indexes cover {indexes.shape[-1]} tokens, "
                f"cu_seqlens describe {packed.total}"
            )
        causal_fastpath = is_plain_causal_thw_index_packed(indexes[0], packed)
    else:
        causal_fastpath = is_plain_causal_thw_index(indexes[0])

    hidden_states = (
        model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    )
    cache_layers: "list[_TrainingPrefixLayer]" = []
    for layer in layers:
        mask = attention_mask[layer.attention_type]

        def layer_forward(states: torch.Tensor, _layer=layer, _mask=mask):
            # Skip only Transformers' cache-dropping wrapper; keep Module hooks.
            return nn.Module.__call__(
                _layer,
                states,
                image_gen_indicators=None,
                exist_non_image_gen_tokens=True,
                exist_image_gen_tokens=False,
                indexes=indexes,
                attention_mask=_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                return_kv=True,
                causal_fastpath=causal_fastpath,
                packed=packed,
            )

        if checkpoint_layers:
            hidden_states, keys, values = checkpoint(
                layer_forward, hidden_states, use_reentrant=False
            )
        else:
            hidden_states, keys, values = layer_forward(hidden_states)
        cache_layers.append(_TrainingPrefixLayer(keys, values))
    return _TrainingPrefixCache(cache_layers, packed=packed)


class _PrefixInputs(NamedTuple):
    """The three vendor prefix arguments, plus which entry ``tokens`` is.

    Positionally compatible with the plain ``(ids, indexes, mask)`` triple both
    vendor builders return, because the four-phase split stores this opaquely
    and replays it, and ``encode_prompt`` reads ``[1]`` for the t extent.
    SENSENOVA_TRAINING_DESIGN.md 13.7.
    """

    tokens: torch.Tensor
    indexes: torch.Tensor
    attention_mask: Any
    embeds: bool = False
    # Vendor ``PackedSegments`` for a batch of prompts packed along the
    # sequence axis (``pack_prefix_inputs``); ``None`` for one prompt.
    packed: Any = None


def pack_prefix_inputs(items: "List[_PrefixInputs]") -> _PrefixInputs:
    """Lay several prompts' prefix inputs end to end along the sequence axis.

    Each item keeps its own t/h/w ``indexes`` (RoPE is applied from these, so a
    token's position is unchanged by where it lands in the packed sequence) and
    its own block-causal region; ``create_packed_block_causal_mask`` closes the
    cross-item cells. Token ids and embeds cannot be mixed: with references some
    items carry embeds and some ids, so everything is embedded by the caller
    beforehand or all items are text-only.
    """
    from core.models.sensenova.vendor.modeling_qwen3 import (
        PackedSegments,
        create_packed_block_causal_mask,
    )

    if not items:
        raise ValueError("pack_prefix_inputs needs at least one prefix")
    if len({bool(item.embeds) for item in items}) != 1:
        raise ValueError(
            "SenseNova packed prefix cannot mix token-id and embedding items; "
            "reference-conditioned and text-only prompts must not share a batch"
        )
    if any(item.packed is not None for item in items):
        raise ValueError("pack_prefix_inputs takes single-prompt inputs")
    lengths = [int(item.tokens.shape[1]) for item in items]
    device = items[0].tokens.device
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    packed = PackedSegments(torch.tensor(offsets, device=device, dtype=torch.int32))
    tokens = torch.cat([item.tokens for item in items], dim=1)
    indexes = torch.cat([item.indexes for item in items], dim=1)
    mask = {"full_attention": create_packed_block_causal_mask(indexes[0], packed)}
    return _PrefixInputs(tokens, indexes, mask, bool(items[0].embeds), packed)


def assert_reference_tower_frozen(transformer: Any) -> None:
    """The understanding ViT is not a training target, asserted rather than assumed.

    ``iter_sensenova_lora_targets`` walks ``language_model.model.layers`` only, so
    ``vision_model`` is outside the 294 by construction -- but "outside the
    enumeration" is a property of one function, and a reference item is the first
    thing that runs the tower under a trainable configuration at all. Why it is
    not a target is SENSENOVA_TRAINING_DESIGN.md 5.2 ground 3 and 13.7, not
    restated here.

    Returns silently on a tree with no ``vision_model``, the same convention the
    target enumerator uses for an unexpected tree shape. Reachable only from the
    reference path, which cannot run at all without the tower.
    """
    tower = getattr(transformer, "vision_model", None)
    if tower is None:
        return
    trainable = [
        name for name, parameter in tower.named_parameters() if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(
            "SenseNova reference conditioning requires a frozen understanding "
            f"vision tower, but {len(trainable)} of its parameters require grad "
            f"(first: {trainable[0]}). The tower is outside the 294 decoder "
            "targets every writer emits, so any gradient it received would train "
            "weights no checkpoint carries and no inference path could load."
        )


def _build_prefix_inputs(
    trainer: Any, transformer: Any, prompt: str, ref_images: list,
    cfg_null: bool = False,
) -> _PrefixInputs:
    """Build the vendor prefix inputs for one item, reference-conditioned or not.

    Always under ``no_grad``: the token embeddings and -- with references -- the
    understanding ViT are frozen on every route (13.7), and the differentiable
    part of the understanding branch starts at the decoder stack that consumes
    this. One function for all three callers, so the reference and text-only
    prefixes cannot drift apart.

    ``cfg_null`` builds the ALIGNED NULL instead of encoding ``prompt``: the
    query inference's text-only uncond branch builds, character for character
    (``sensenova_pipeline_ops.encode_prompt``: ``_build_t2i_query(
    negative_prompt, append_text="<img>")`` with ``negative_prompt`` stripped to
    ""). No ``system_message`` -- the neo1_0 template's own message is empty and
    its MPT formatter then emits no system block at all -- and no think suffix.
    Its token count is shorter than the conditional's, which is why it must be
    built here: ``text_length`` derives from these ``indexes`` and lands in every
    image token's t coordinate in ``train_step``.
    """
    from core.models.sensenova.vendor.utils import SYSTEM_MESSAGE_FOR_GEN

    with torch.no_grad():
        if cfg_null:
            if ref_images:
                raise ValueError(
                    "SenseNova's aligned CFG null is the text-only uncond "
                    "prefix; a reference-conditioned item has no representation "
                    "in it (see api/cfg_null_resolver.py, which refuses this "
                    "combination before the model loads)"
                )
            query = transformer._build_t2i_query("", append_text="<img>")
            return _PrefixInputs(
                *transformer._build_t2i_text_inputs(trainer.tokenizer, query),
                embeds=False,
            )
        if not ref_images:
            query = transformer._build_t2i_query(
                prompt,
                system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text="<think>\n\n</think>\n\n<img>",
            )
            return _PrefixInputs(
                *transformer._build_t2i_text_inputs(trainer.tokenizer, query),
                embeds=False,
            )

        from core.models.sensenova.sensenova_pipeline_ops import (
            _IMG_CONTEXT_TOKEN,
            _embed_reference_images,
            _splice_reference_image_tokens,
        )

        pixel_values, grid_hw = _embed_reference_images(transformer, ref_images)
        # Never set on the text-only path; _build_it2i_inputs asserts on it
        # matching at least one token.
        transformer.img_context_token_id = trainer.tokenizer.convert_tokens_to_ids(
            _IMG_CONTEXT_TOKEN
        )
        query = transformer._build_t2i_query(
            _splice_reference_image_tokens(
                prompt, len(ref_images), grid_hw, transformer.downsample_ratio
            ),
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text="<think>\n\n</think>\n\n<img>",
        )
        return _PrefixInputs(
            *transformer._build_it2i_inputs(
                trainer.tokenizer, query, pixel_values, grid_hw
            ),
            embeds=True,
        )


def _build_trainable_prefix(trainer: Any, transformer: Any, inputs) -> Any:
    """Run ``forward_und_prefix_layers`` under the autocast the adapters need.

    ``LoRALinearLayer`` keeps fp32 adapter weights and relies on an AMBIENT
    autocast to meet the bf16 base activation -- ``train_step`` provides one for
    the generation pass, and without the same wrap here the very first und-LoRA
    prefix pass raises a dtype mismatch at layer 0 (found by running it, U-0;
    re-running the K/V parity gate with autocast on costs nothing numerically).
    """
    tokens, indexes, attention_mask = inputs[0], inputs[1], inputs[2]
    embeds = bool(inputs[3]) if len(inputs) > 3 else False
    packed = inputs[4] if len(inputs) > 4 else None
    dtype = getattr(trainer, "training_dtype", torch.float32)
    device_type = torch.device(getattr(trainer, "device", "cpu")).type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
        return forward_und_prefix_layers(
            transformer.language_model.model,
            None if embeds else tokens,
            indexes,
            attention_mask,
            inputs_embeds=tokens if embeds else None,
            checkpoint_layers=bool(getattr(trainer, "gradient_checkpointing", True)),
            packed=packed,
        )


def _prefix_text_lengths(inputs: _PrefixInputs) -> List[int]:
    """Per-item NEXT t index of a (possibly packed) prefix."""
    if inputs.packed is None:
        return [int(inputs.indexes[0].max()) + 1]
    return [
        int(inputs.indexes[0, start:end].max()) + 1
        for start, end in inputs.packed.bounds()
    ]


def encode_prompts(
    trainer: Any,
    prompts: List[str],
    *,
    requires_grad: bool = False,
    reference_image_paths: Optional[List[Optional[List[str]]]] = None,
    cfg_null: Optional[List[bool]] = None,
) -> SenseNovaTrainingPrefix:
    """``encode_prompt`` for a physical batch: one PACKED prefix for all items.

    The prompts are built exactly as ``encode_prompt`` builds each of them
    (the same query template, the same aligned null per item) and then laid end
    to end with ``pack_prefix_inputs``; the understanding stack runs once over
    the packed sequence, so the K/V of item ``i`` are bitwise what a single
    encode would produce (per-token computation is position-free, and the
    packed mask / varlen attention keep the segments apart). The three routes
    (four-phase cut, differentiable, frozen) are the same as ``encode_prompt``'s;
    the frozen one uses the training prefix loop under ``no_grad`` rather than
    the vendor prefix forward, which has no packed-mask entry.
    """
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise TypeError("SenseNova encode_prompts takes a non-empty list of prompts")
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError("SenseNova encode_prompts takes str prompts")
    batch = len(prompts)
    refs = list(reference_image_paths) if reference_image_paths else [None] * batch
    nulls = [bool(flag) for flag in cfg_null] if cfg_null is not None else [False] * batch
    if len(refs) != batch or len(nulls) != batch:
        raise ValueError(
            "SenseNova encode_prompts: reference_image_paths and cfg_null must "
            "have one entry per prompt"
        )
    for paths, is_null in zip(refs, nulls):
        if is_null and paths:
            raise ValueError(
                "SenseNova's aligned CFG null is the text-only uncond prefix and "
                "has no reference-conditioned form; the run should have been "
                "refused before the model loaded (api/cfg_null_resolver.py)"
            )

    transformer = trainer.transformer
    phase_evictor = getattr(trainer, "sensenova_phase_evictor", None)
    four_phase = getattr(trainer, "sensenova_four_phase", None)
    if requires_grad:
        assert_understanding_training_supported(transformer)
    if requires_grad and four_phase is None and phase_evictor is not None:
        raise RuntimeError(
            "SenseNova understanding-branch training cannot run with MoT phase "
            "eviction: the understanding half must stay resident until backward, "
            "but the evictor moves it to CPU for the denoise phase"
        )
    if phase_evictor is not None:
        phase_evictor.enter_prefix()
        phase_evictor.assert_understanding_resident()

    items = []
    for prompt, paths, is_null in zip(prompts, refs, nulls):
        ref_images = _load_reference_images(paths)
        if requires_grad and ref_images:
            assert_reference_tower_frozen(transformer)
        items.append(_build_prefix_inputs(trainer, transformer, prompt, ref_images, is_null))
    inputs = pack_prefix_inputs(items)
    text_lengths = _prefix_text_lengths(inputs)
    expected_layers = len(transformer.language_model.model.layers)

    if requires_grad and four_phase is not None:
        with torch.no_grad():
            cache = _build_trainable_prefix(trainer, transformer, inputs)
        leaf_cache = four_phase.cut(cache, inputs)
        del cache
        _assert_immutable_prefix_cache(leaf_cache, expected_layers, boundary_leaf=True)
        return SenseNovaTrainingPrefix(
            cache=leaf_cache, text_length=text_lengths[0],
            packed=inputs.packed, text_lengths=text_lengths,
        )
    if requires_grad:
        cache = _build_trainable_prefix(trainer, transformer, inputs)
        _assert_immutable_prefix_cache(cache, expected_layers, trainable=True)
        return SenseNovaTrainingPrefix(
            cache=cache, text_length=text_lengths[0],
            packed=inputs.packed, text_lengths=text_lengths,
        )

    with torch.no_grad():
        cache = forward_und_prefix_layers(
            transformer.language_model.model,
            None if inputs.embeds else inputs.tokens,
            inputs.indexes,
            inputs.attention_mask,
            inputs_embeds=inputs.tokens if inputs.embeds else None,
            checkpoint_layers=False,
            packed=inputs.packed,
        )
    _assert_immutable_prefix_cache(cache, expected_layers)
    if phase_evictor is not None:
        phase_evictor.enter_denoise()
        phase_evictor.assert_generation_resident()
    return SenseNovaTrainingPrefix(
        cache=cache, text_length=text_lengths[0],
        packed=inputs.packed, text_lengths=text_lengths,
    )


def encode_prompt(
    trainer: Any,
    prompt: str,
    *,
    requires_grad: bool = False,
    reference_image_paths: Optional[List[str]] = None,
    cfg_null: bool = False,
) -> SenseNovaTrainingPrefix:
    """Build a prefix without inference streamers or flash buffers.

    ``requires_grad=False`` (the default, and the whole of Phase 1) builds it
    under ``no_grad`` through the vendor prefix forward; ``True`` builds it
    through the differentiable understanding loop above so understanding LoRA
    receives gradient.

    With ``reference_image_paths`` this runs inference's cond branch verbatim
    (understanding-tower ViT embeds spliced into the text prefix); img_cond and
    uncond are CFG-only and carry no loss. Reference conditioning composes with
    a trainable prefix: the spliced rows traverse the same decoder layers in the
    same pass (SENSENOVA_TRAINING_DESIGN.md 13.7).

    ``cfg_null`` REPLACES the item's prompt with inference's own text-only
    uncond query. It changes nothing else: the same route, the same phase
    transitions, the same K/V forward and the same ``text_length`` derivation,
    so the shorter null prefix reaches ``train_step`` and its image indexes
    through the one channel that already carries the length. Nothing is cached
    between items -- the understanding half may be trainable, and a cache handed
    across items would then be stale (strategy §6.3).
    """
    if not isinstance(prompt, str):
        raise TypeError("SenseNova training encodes one prompt at a time")

    transformer = trainer.transformer
    if cfg_null and reference_image_paths:
        # Refused on the PATHS, before they are read: the null has no
        # reference-conditioned form, so loading them would be work done for a
        # condition that cannot be built.
        raise ValueError(
            "SenseNova's aligned CFG null is the text-only uncond prefix and "
            "has no reference-conditioned form; the run should have been "
            "refused before the model loaded (api/cfg_null_resolver.py)"
        )
    ref_images = _load_reference_images(reference_image_paths)
    phase_evictor = getattr(trainer, "sensenova_phase_evictor", None)
    four_phase = getattr(trainer, "sensenova_four_phase", None)
    if requires_grad:
        assert_understanding_training_supported(transformer)
        if ref_images:
            assert_reference_tower_frozen(transformer)
    if requires_grad and four_phase is not None:
        # Phase 1 of the four-phase split (8.3.2). Built under no_grad because
        # phase 3 recomputes it; only the boundary K/V survives this phase.
        if phase_evictor is not None:
            phase_evictor.enter_prefix()
            phase_evictor.assert_understanding_resident()
        inputs = _build_prefix_inputs(trainer, transformer, prompt, ref_images,
                                      cfg_null)
        with torch.no_grad():
            cache = _build_trainable_prefix(trainer, transformer, inputs)
        leaf_cache = four_phase.cut(cache, inputs)
        del cache
        _assert_immutable_prefix_cache(
            leaf_cache,
            len(transformer.language_model.model.layers),
            boundary_leaf=True,
        )
        return SenseNovaTrainingPrefix(
            cache=leaf_cache, text_length=int(inputs[1][0].max()) + 1
        )
    if requires_grad:
        if phase_evictor is not None:
            raise RuntimeError(
                "SenseNova understanding-branch training cannot run with MoT phase "
                "eviction: the understanding half must stay resident until backward, "
                "but the evictor moves it to CPU for the denoise phase"
            )
        inputs = _build_prefix_inputs(trainer, transformer, prompt, ref_images,
                                      cfg_null)
        cache = _build_trainable_prefix(trainer, transformer, inputs)
        _assert_immutable_prefix_cache(
            cache,
            len(transformer.language_model.model.layers),
            trainable=True,
        )
        return SenseNovaTrainingPrefix(
            cache=cache, text_length=int(inputs[1][0].max()) + 1
        )

    if phase_evictor is not None:
        phase_evictor.enter_prefix()
    inputs = _build_prefix_inputs(trainer, transformer, prompt, ref_images,
                                  cfg_null)
    with torch.no_grad():
        forward = (
            transformer._it2i_prefix_forward
            if inputs.embeds
            else transformer._t2i_prefix_forward
        )
        cache, _ = forward(inputs.tokens, inputs.indexes, inputs.attention_mask)
        indexes = inputs.indexes
    expected_layers = len(transformer.language_model.model.layers)
    _assert_immutable_prefix_cache(cache, expected_layers)
    if phase_evictor is not None:
        phase_evictor.enter_denoise()
        phase_evictor.assert_generation_resident()
    # Equals input_ids.shape[1] on the text-only path (t is arange there).
    return SenseNovaTrainingPrefix(
        cache=cache, text_length=int(indexes[0].max()) + 1
    )


def vae_encode(trainer: Any, image_tensor: torch.Tensor, **_: Any) -> torch.Tensor:
    """Normalized RGB straight through, or one VAE encode after a swap (§10.5).

    Natively there is no VAE and the "latent" IS the [-1,1] RGB image. After a
    swap the same call returns a normalised latent; both are what train_step's
    ``images`` argument means.
    """
    from core.models.sensenova.latent_space import token_pixel_width

    if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
        raise ValueError("SenseNova expects BCHW RGB training images")
    align = token_pixel_width(trainer.transformer)
    if image_tensor.shape[-2] % align or image_tensor.shape[-1] % align:
        raise ValueError(
            f"SenseNova image height and width must be divisible by {align}")
    vae = getattr(trainer, "vae", None)
    if vae is None:
        return image_tensor.detach().to(dtype=trainer.training_dtype, device="cpu")

    from core.models.sensenova.latent_space import encode

    latents = encode(vae, image_tensor.to(dtype=next(vae.parameters()).dtype,
                                          device=next(vae.parameters()).device),
                     spec=getattr(trainer, "wiring", None))
    return latents.detach().to(dtype=trainer.training_dtype, device="cpu")


def vae_decode(trainer: Any, latents: torch.Tensor) -> torch.Tensor:
    """Latent -> [-1,1] RGB. Pixel-space runs are already RGB and pass through."""
    vae = getattr(trainer, "vae", None)
    if vae is None:
        return latents
    from core.models.sensenova.latent_space import decode

    return decode(vae, latents.to(dtype=next(vae.parameters()).dtype,
                                  device=next(vae.parameters()).device),
                  spec=getattr(trainer, "wiring", None))


def _save_pixel_debug(
    transformer: Any,
    debug_save_path: Path,
    *,
    t_val: float,
    noise_scale: float,
    images: torch.Tensor,
    z_image: torch.Tensor,
    x0_pred_tokens: torch.Tensor,
    patch: int,
    height: int,
    width: int,
    loss_value: float,
    recon_loss_value: float,
    captions: Optional[List[str]],
    reference_image_paths: Optional[List[Optional[str]]],
    batch_size: int = 1,
    vae: Any = None,
) -> None:
    """Dump this step's pixel tensors, the pixel-space analogue of the latent
    archs' debug latents: ``target`` is their ``latents`` (the clean sample),
    ``noisy`` their ``noisy_latents``, ``pred_x0`` their ``predicted_latent``.

    SenseNova's "latent" already IS [-1,1] RGB, so the previews are written
    directly and the ``.pt`` stays scalar-only (the visualize endpoint prefers
    the webp over false-colouring a tensor, and a full-res pixel tensor per
    dump would be tens of MB).
    """
    from core.models.sensenova.sensenova_pipeline_ops import tensor_to_image

    debug_save_path.mkdir(parents=True, exist_ok=True)
    debug_data: dict = {
        "timestep": t_val,
        "noise_scale": noise_scale,
        "model_type": "sensenova",
        "is_latent": vae is not None,
        "loss": loss_value,
        "recon_loss": recon_loss_value,
        "batch_size": int(batch_size),
    }
    if captions:
        debug_data["caption"] = captions[0]
    if reference_image_paths:
        first_ref = next((p for p in reference_image_paths if p is not None), None)
        if first_ref:
            debug_data["reference_image_path"] = first_ref
    torch.save(debug_data, debug_save_path / f"latents_t{t_val:.4f}.pt")

    x0_pred_image = transformer.unpatchify(x0_pred_tokens.detach(), patch, height, width)
    for name, tensor in (
        ("noisy", z_image),
        ("target", images),
        ("pred_x0", x0_pred_image),
    ):
        # tensor_to_image clamps to [-1,1]: the noised map saturates at low t,
        # which is the same convention the VAE archs' decoded previews use. A
        # swapped run decodes first, so the three previews stay comparable.
        preview = tensor.detach()
        if vae is not None:
            from core.models.sensenova.latent_space import decode as _decode

            preview = _decode(vae, preview)
        tensor_to_image(preview.float()).save(
            debug_save_path / f"decode_t{t_val:.4f}_{name}.webp",
            "WEBP",
            quality=80,
            method=4,
        )


def train_step(
    trainer: Any,
    *,
    images: torch.Tensor,
    prefix: SenseNovaTrainingPrefix,
    timesteps: Optional[torch.Tensor] = None,
    profile_vram: bool = False,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[Optional[str]]] = None,
) -> tuple[torch.Tensor, float, float]:
    """Run one B1 pixel-space flow-matching forward pass."""
    del profile_vram  # Central profiling owns peak-memory reporting.
    if not isinstance(prefix, SenseNovaTrainingPrefix):
        raise TypeError("SenseNova train_step requires SenseNovaTrainingPrefix")
    phase_evictor = getattr(trainer, "sensenova_phase_evictor", None)
    if phase_evictor is not None:
        phase_evictor.enter_denoise()
        phase_evictor.assert_generation_resident()
    from core.models.sensenova.latent_space import gen_geometry

    geometry = gen_geometry(trainer.transformer)
    if images.ndim != 4 or images.shape[1] != geometry.channels:
        raise ValueError(
            f"SenseNova training requires BCHW samples with "
            f"{geometry.channels} channel(s)"
            + ("" if geometry.is_latent else " (RGB)"))
    batch = int(images.shape[0])
    if batch != prefix.batch_size:
        raise ValueError(
            f"SenseNova train_step: {batch} image(s) but the prefix holds "
            f"{prefix.batch_size} prompt(s)"
        )
    height, width = images.shape[-2:]
    # Divisibility is asked of the grid the SAMPLE is on: pixels natively, the
    # latent grid after a swap (where the pixel canvas was already checked by
    # vae_encode against 4*scale).
    if height % geometry.patch or width % geometry.patch:
        raise ValueError(
            f"SenseNova sample height and width must be divisible by "
            f"{geometry.patch}")

    transformer = trainer.transformer
    _assert_pixel_head_fm_decoder(transformer)
    trainable_prefix = bool(getattr(trainer, "train_text_encoder", False))
    # Under the four-phase split the prefix arrives CUT: grad-requiring leaves
    # rather than a live graph, so this backward stops at the boundary.
    boundary_leaf = trainable_prefix and getattr(
        trainer, "sensenova_four_phase", None
    ) is not None
    device = trainer.device
    dtype = trainer.training_dtype
    x0 = images.to(device=device, dtype=dtype)
    if timesteps is None:
        t = trainer.timestep_sampler.sample(batch, device=device)
        if isinstance(t, tuple):
            t = t[0]
    else:
        t = timesteps
    # Keep t in fp32, the dtype the sampler produces and the dtype inference's
    # `ts` carries (linspace, sensenova_pipeline_ops.py:1068): timestep_embedder
    # embeds t's VALUE, and bf16 would quantize it to ~2e-3 in training only.
    t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
    if t.numel() == 1 and batch > 1:
        t = t.expand(batch).contiguous()
    if t.numel() != batch:
        raise ValueError(
            f"SenseNova training requires one timestep per image (got {t.numel()} for {batch})"
        )

    from core.models.sensenova.sensenova_pipeline_ops import (
        _build_step_context,
        compute_noise_scale,
    )

    merge_size = int(1 / transformer.downsample_ratio)
    # The gen ViT's patch, not the understanding tower's: the two are the same
    # 16 natively and differ after a swap (design §10.2).
    grid_h = height // geometry.vit_patch
    grid_w = width // geometry.vit_patch
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("SenseNova image does not align to the merged token grid")
    token_h, token_w = grid_h // merge_size, grid_w // merge_size
    noise_scale = compute_noise_scale(transformer, grid_h, grid_w, merge_size)
    # Inference noises in the image dtype, its fp32 t demoted by 0-dim promotion
    # (sensenova_pipeline_ops.py:1122); cast explicitly so z_image stays
    # training_dtype -- _build_step_context's ViT runs outside the autocast below.
    t_img = t.to(dtype).view(batch, 1, 1, 1)
    z_image = t_img * x0 + (1 - t_img) * (torch.randn_like(x0) * noise_scale)
    shape = SimpleNamespace(
        batch_size=batch,
        merge_size=merge_size,
        grid_h=grid_h,
        grid_w=grid_w,
        token_h=token_h,
        token_w=token_w,
    )
    # _build_step_context is inference's, so it is no-grad by default; it calls
    # the gen ViT and both embedders, i.e. 12 of the 16 fm_modules tensors.
    # Ask the parameters rather than the flag: sensenova_train_fm_modules on an
    # understanding-only run collects no fm parameter (the adapter warns), and a
    # graph there would cost activation memory for nothing.
    # is_grad_enabled keeps this strictly weaker than the decorator it replaced:
    # set_grad_enabled(True) would otherwise re-enable grad inside a caller's
    # no_grad, which the decorator could never do.
    fm_modules = getattr(transformer, "fm_modules", None)
    fm_trainable = (
        torch.is_grad_enabled()
        and fm_modules is not None
        and any(p.requires_grad for p in fm_modules.parameters())
    )
    z, image_embeds, _ = _build_step_context(
        transformer, shape, z_image, t if batch > 1 else t[0], noise_scale,
        enable_grad=fm_trainable,
    )
    packed_gen = None
    if prefix.packed is None:
        indexes = transformer._build_t2i_image_indexes(
            token_h, token_w, prefix.text_length, device=device
        )
    else:
        from core.models.sensenova.vendor.modeling_qwen3 import PackedGenPlan

        # Item i's image tokens sit at t = its own prefix's next index, exactly
        # as the single-prompt form; packing only concatenates them.
        indexes = torch.cat(
            [
                transformer._build_t2i_image_indexes(token_h, token_w, length, device=device)
                for length in prefix.text_lengths
            ],
            dim=1,
        )
        image_embeds = image_embeds.reshape(1, batch * token_h * token_w, -1)
        packed_gen = PackedGenPlan(prefix.packed, batch, token_h * token_w, device)
    _assert_immutable_prefix_cache(
        prefix.cache,
        len(transformer.language_model.model.layers),
        trainable=trainable_prefix,
        boundary_leaf=boundary_leaf,
    )

    device_type = torch.device(device).type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
        hidden = forward_gen_decoder_layers(
            transformer.language_model.model,
            image_embeds,
            indexes=indexes,
            prefix_cache=prefix.cache,
            checkpoint_layers=bool(trainer.gradient_checkpointing),
            trainable_prefix=trainable_prefix and not boundary_leaf,
            boundary_leaf_prefix=boundary_leaf,
            packed_gen=packed_gen,
        )
        decoded = transformer.fm_modules["fm_head"](
            hidden.view(batch, token_h, token_w, -1).permute(0, 3, 1, 2).contiguous()
        )
        patch, channels = geometry.patch, geometry.channels
        x0_pred = (
            decoded.view(batch, channels, token_h, patch, token_w, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(batch, token_h * token_w, patch * patch * channels)
        )
        x0_tokens = transformer.patchify(x0, patch)
        # fp32 t here lifts v into fp32, which the MSE below wanted anyway --
        # the .float() calls become no-ops rather than extra copies.
        denominator = (1 - t).view(batch, 1, 1).clamp_min(transformer.config.t_eps)
        v_pred = (x0_pred - z) / denominator
        v_target = (x0_tokens - z) / denominator
        loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float())
        recon_loss = torch.nn.functional.mse_loss(x0_pred.float(), x0_tokens.float())

    value = float(loss.detach())
    recon_value = float(recon_loss.detach())

    if debug_save_path is not None:
        try:
            # First item only: the dump is one preview triple per step.
            _save_pixel_debug(
                transformer,
                debug_save_path,
                t_val=float(t[0].item()),
                noise_scale=noise_scale,
                images=x0[:1],
                z_image=z_image[:1],
                x0_pred_tokens=x0_pred[:1],
                patch=patch,
                height=height,
                width=width,
                loss_value=value,
                recon_loss_value=recon_value,
                captions=debug_captions,
                reference_image_paths=debug_reference_image_paths,
                batch_size=batch,
                vae=getattr(trainer, "vae", None),
            )
        except Exception as debug_error:
            print(f"{trainer.log_prefix} [debug_latents] save failed: {debug_error}")

    return loss, value, recon_value


def _maybe_install_sample_kv_streaming(trainer: Any, transformer: Any):
    """Install the 2-slot flash-KV prefix streamer for a training-time sample
    (no applicability to ``train_step`` -- see module docstring). A failed
    install is announced on ``training_log``, not just printed, since
    ``add_warning`` is a no-op outside a live generation request."""
    if not bool(getattr(trainer, "sensenova_sample_kv_cache_streaming", False)):
        return None
    from core.models.sensenova import kv_cache_streaming

    streamer = kv_cache_streaming.install(transformer, trainer.device)
    if streamer is None:
        emit_training_warning(
            "SenseNova training-time sample requested KV cache streaming, but "
            "the mechanism could not be installed; this sample runs with the "
            "full per-layer, per-branch KV cache resident instead.",
            code="sensenova_sample_kv_cache_streaming_unavailable",
            prefix=trainer.log_prefix,
        )
    return streamer


def _teardown_sample_kv_streaming(transformer: Any, streamer: Any) -> None:
    if streamer is None and getattr(transformer, "_kv_cache_streamer", None) is None:
        return
    from core.models.sensenova import kv_cache_streaming

    kv_cache_streaming.uninstall(transformer, streamer)


def _log_sample_guidance(trainer: Any, steps: Any) -> None:
    """Turn a sample's per-step CFG guidance into two conditioning-strength series.

    ``cfg_guidance_rel`` is ``||v_cond - v_uncond|| / ||v_uncond||`` averaged
    over the denoise trajectory: the size of the direction CFG applies. It is
    what conditioning collapse destroys -- if the conditional and unconditional
    predictions converge, this goes to zero and the served cfg_scale stops
    doing anything, regardless of what the training loss shows.

    ``cfg_guidance_cos`` is the cosine between the two branches. Rising toward
    1 is the same collapse seen from the direction side rather than the
    magnitude side, and it separates "the branches agree" from "both shrank".

    Also emitted for the first third of the trajectory alone
    (``cfg_guidance_rel_early``), which is where the noise level is highest and
    where the conditioning is measurably most load-bearing: the caption-drop
    ablation moves the understanding branch's gradient 45% at the noisy end and
    only 8% at the clean end.

    Best-effort. A sample that failed, or a run at cfg_scale <= 1 where no
    unconditional branch is built, simply logs nothing.
    """
    try:
        rows = list(steps or [])
        if not rows:
            return
        rel = [r["relative_diff"] for r in rows]
        cos = [r["cosine_similarity"] for r in rows]
        trainer.log_extra_metric("cfg_guidance_rel", sum(rel) / len(rel))
        trainer.log_extra_metric("cfg_guidance_cos", sum(cos) / len(cos))
        early = rel[: max(1, len(rel) // 3)]
        trainer.log_extra_metric("cfg_guidance_rel_early", sum(early) / len(early))
    except Exception:
        return


def generate_sample(
    trainer: Any,
    *,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    timestep_shift: float = _TRAINING_DEFAULTS["sensenova_sample_timestep_shift"],
    img_cfg_scale: float = _TRAINING_DEFAULTS["sensenova_sample_img_cfg_scale"],
    cfg_norm: str = _TRAINING_DEFAULTS["sensenova_sample_cfg_norm"],
    negative_prompt: str = "",
    reference_image_path: Optional[str] = None,
    condition_image_path: Optional[str] = None,
    step_progress_callback: Optional[Callable[[int, int], None]] = None,
):
    """Run one inference txt2img/it2i generation from inside the training loop.

    Drives the SAME ``sensenova_pipeline_ops`` prefix + Euler loop generation
    uses; nothing about the denoise is reimplemented here. The LoRA under
    training is applied automatically because its ``LoRALinearLayer`` wrappers
    ARE the live modules the generation forward calls.

    ``reference_image_path`` runs inference's own reference path: the image goes
    to ``ops.encode_prompt(..., ref_images=...)`` exactly as the generation
    backend passes it, so the sample is built by the SAME cond/img_cond/uncond
    branch logic a real it2i request gets. This is a different function from the
    training-side ``encode_prompt`` above, which builds the cond branch only.

    Returns a PIL image, or ``None`` if the generation failed -- the training
    loop's sample block has no exception guard of its own, so a failed sample
    must never take the run down.
    """
    from core.attention import AttentionMode
    from core.models.sensenova import sensenova_pipeline_ops as ops

    transformer = trainer.transformer
    if condition_image_path:
        # ControlNet-style conditioning, which SenseNova has no entry for
        # (refused in arch_capabilities); unrelated to reference images, which
        # enter as understanding-tower tokens in the prompt prefix.
        print(
            f"{trainer.log_prefix} SenseNova sampling ignores the condition image "
            f"(ControlNet is not supported for SenseNova U1.5)"
        )

    align = ops.token_pixel_width(transformer)
    snapped_width, snapped_height = ops.normalize_resolution(width, height, align)
    if (snapped_width, snapped_height) != (width, height):
        print(
            f"{trainer.log_prefix} SenseNova sample resolution snapped to the "
            f"{align}px token grid: {width}x{height} -> "
            f"{snapped_width}x{snapped_height}"
        )

    backend = trainer._resolve_training_backend(trainer.attention_backend)
    was_training = transformer.training
    evictor = getattr(trainer, "sensenova_phase_evictor", None)
    prefix = None
    kv_streamer = None
    try:
        ref_images = _load_reference_images(
            [reference_image_path] if reference_image_path else None
        )
        if ref_images:
            print(
                f"{trainer.log_prefix} SenseNova sample is reference-conditioned "
                f"(img_cfg_scale={img_cfg_scale}): {reference_image_path}"
            )
        # No-op while the phase evictor owns weight placement.
        trainer.move_main_model_to_gpu()
        transformer.eval()
        # Pass the mode EXPLICITLY: set_attention_backend infers it from
        # torch.is_grad_enabled() otherwise, and this call happens before the
        # no_grad block below.
        ops.set_attention_backend(transformer, backend, AttentionMode.INFERENCE)
        # Independent of the phase evictor (disjoint tensors/hooks, its own
        # pinned pool and CUDA stream); must be installed before encode_prompt
        # so its own KV-cache finalization sees ``transformer._kv_cache_streamer``.
        kv_streamer = _maybe_install_sample_kv_streaming(trainer, transformer)
        with torch.no_grad():
            # The evictor's full/prefix/denoise machine is driven here exactly as
            # a training step drives it, so generation's own prefix->denoise
            # phase change stays the SAME transition pair and the two halves
            # never co-reside.
            if evictor is not None:
                evictor.enter_prefix()
            prefix = ops.encode_prompt(
                transformer,
                trainer.tokenizer,
                prompt,
                snapped_height,
                snapped_width,
                guidance_scale,
                negative_prompt=negative_prompt,
                ref_images=ref_images,
                img_cfg_scale=img_cfg_scale,
            )
            if evictor is not None:
                evictor.enter_denoise()
                evictor.assert_generation_resident()
            # Both CFG branches are computed at every step of this generation
            # anyway, so the guidance strength comes out for the price of a norm
            # -- and it is the direct read on whether the fine-tune's text
            # conditioning is still alive. Nothing here costs a training step.
            with ops.collect_guidance_metrics() as guidance_steps:
                image_tensor = ops.denoise_loop(
                    transformer,
                    prefix,
                    cfg_scale=guidance_scale,
                    timestep_shift=timestep_shift,
                    num_inference_steps=num_inference_steps,
                    seed=seed if seed is not None and seed >= 0 else None,
                    cfg_norm=cfg_norm,
                    progress_callback=step_progress_callback,
                )
            _log_sample_guidance(trainer, guidance_steps)
        image = ops.from_gen_space(
            image_tensor.float(), vae=getattr(trainer, "vae", None),
            spec=getattr(trainer, "wiring", None))
        del image_tensor
        return image
    except Exception as sample_error:
        import traceback

        print(
            f"{trainer.log_prefix} SenseNova sample generation failed "
            f"({type(sample_error).__name__}: {sample_error}); training continues"
        )
        traceback.print_exc()
        return None
    finally:
        if prefix is not None:
            try:
                ops.clear_prefix_caches(prefix)
            except Exception as clear_error:
                print(
                    f"{trainer.log_prefix} SenseNova sample prefix cleanup failed: {clear_error}"
                )
        # Idempotent with clear_prefix_caches's own defence-in-depth teardown
        # (mirrors pipeline_backends.sensenova's own generation finally).
        _teardown_sample_kv_streaming(transformer, kv_streamer)
        # Restore TRAINING mode before the next forward: nothing re-stamps the
        # attention modules after load, so an INFERENCE stamp left here would
        # persist for the rest of the run.
        ops.set_attention_backend(transformer, backend, AttentionMode.TRAINING)
        if was_training:
            transformer.train()
        # The evictor is deliberately left wherever this call ended: both
        # "prefix" (sample raised mid-way) and "denoise" are states the next
        # step's encode_prompt transitions out of legally.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _assert_prefix_cache_structure(prefix_cache: Any, expected_layers: int) -> None:
    """Structural half: layer count, non-empty K/V, no inference-only buffers.

    UNCONDITIONAL -- it holds identically for a detached Phase 1 prefix and a
    differentiable understanding-training one.
    """
    if prefix_cache is None:
        raise ValueError("SenseNova generation training requires a prefix KV cache")
    layers = getattr(prefix_cache, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("SenseNova training requires non-empty prefix KV cache layers")
    if len(layers) != expected_layers:
        raise ValueError(
            f"SenseNova prefix KV cache has {len(layers)} layer(s), expected {expected_layers}"
        )
    if any(
        getattr(prefix_cache, name, None) is not None
        for name in ("_kv_cache_streamer", "_kv_cache_streamer_branch")
    ):
        raise ValueError("SenseNova training cannot use the inference KV cache streamer")

    for layer in layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                raise ValueError(f"SenseNova prefix KV cache layer is missing non-empty {name}")
        if any(
            getattr(layer, name, None) is not None
            for name in ("flash_k_cache", "flash_v_cache")
        ):
            raise ValueError("SenseNova training cannot use prepared inference flash KV buffers")


def _assert_prefix_cache_detached(prefix_cache: Any) -> None:
    """Grad-mode half for a frozen understanding branch."""
    for layer in prefix_cache.layers:
        for name in ("keys", "values"):
            if getattr(layer, name).requires_grad:
                raise ValueError(f"SenseNova prefix KV cache {name} tensors must be detached")


def _assert_prefix_cache_differentiable(prefix_cache: Any) -> None:
    """Grad-mode half for a trainable understanding branch, stated POSITIVELY.

    Dropping the detachment refusal without asserting its opposite reproduces
    the silent failure this whole path exists to avoid: a prefix accidentally
    built under ``no_grad`` still yields a perfectly healthy, falling loss while
    the understanding LoRA never moves a millimetre. Every layer's K/V must
    carry a ``grad_fn``.
    """
    missing = [
        index
        for index, layer in enumerate(prefix_cache.layers)
        if layer.keys.grad_fn is None or layer.values.grad_fn is None
    ]
    if missing:
        raise ValueError(
            "SenseNova understanding-branch training requires a differentiable prefix, "
            f"but {len(missing)} of {len(prefix_cache.layers)} KV cache layer(s) carry no "
            f"grad_fn (first: {missing[:3]}). The prefix was built under no_grad; the loss "
            "would fall normally and the understanding LoRA would never be trained."
        )


def _assert_prefix_cache_boundary_leaf(prefix_cache: Any) -> None:
    """Grad-mode half for the four-phase split's CUT prefix (§8.3.2).

    The split hands the generation forward a cache whose K/V are leaves that
    require grad, precisely so the generation backward terminates in their
    ``.grad`` instead of running on into the understanding half. Such a tensor
    has ``grad_fn is None``, so the ``_differentiable`` check above rejects it --
    and rejecting it is right for the single-backward path, where a leaf means
    the understanding half silently receives nothing.

    Asserted positively for the same reason its sibling is: a cache that is
    merely ``requires_grad=False`` would train nothing and fall normally, and a
    cache that still carries a ``grad_fn`` means the cut did not happen, so the
    generation backward would walk the understanding half after all -- which is
    the residency the phase split exists to avoid.
    """
    broken = [
        index
        for index, layer in enumerate(prefix_cache.layers)
        for tensor in (layer.keys, layer.values)
        if not (tensor.requires_grad and tensor.is_leaf and tensor.grad_fn is None)
    ]
    if broken:
        raise ValueError(
            "SenseNova four-phase eviction requires the boundary prefix K/V to be "
            f"grad-requiring LEAVES, but {len(set(broken))} of {len(prefix_cache.layers)} "
            f"KV cache layer(s) are not (first: {sorted(set(broken))[:3]}). Either the "
            "prefix was built under no_grad (nothing would train) or it was not cut "
            "(the generation backward would run on into the understanding half)."
        )


def _assert_immutable_prefix_cache(
    prefix_cache: Any,
    expected_layers: int,
    *,
    trainable: bool = False,
    boundary_leaf: bool = False,
) -> None:
    _assert_prefix_cache_structure(prefix_cache, expected_layers)
    if boundary_leaf:
        _assert_prefix_cache_boundary_leaf(prefix_cache)
    elif trainable:
        _assert_prefix_cache_differentiable(prefix_cache)
    else:
        _assert_prefix_cache_detached(prefix_cache)


def forward_gen_decoder_layers(
    model: Any,
    hidden_states: torch.Tensor,
    *,
    indexes: torch.Tensor,
    prefix_cache: Any,
    attention_mask: Optional[torch.Tensor] = None,
    checkpoint_layers: bool = False,
    trainable_prefix: bool = False,
    boundary_leaf_prefix: bool = False,
    packed_gen: Any = None,
) -> torch.Tensor:
    """Run the all-generation-token Qwen3 decoder against immutable prefix K/V.

    ``packed_gen`` (vendor ``PackedGenPlan``) is the batched form: the image
    tokens of every item packed along the sequence axis against a packed
    prefix, each item confined to its own prefix by the varlen attention.

    Calling PyTorch's base ``Module.__call__`` bypasses Transformers'
    cache-dropping checkpoint wrapper while preserving module hooks. The cache
    is read through the differentiable ``update_cache=False`` concat path.

    ``boundary_leaf_prefix`` is the four-phase split's cache shape (§8.3.2):
    grad-requiring leaves rather than a live graph.
    """

    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("SenseNova generation training model has no decoder layers")
    _assert_immutable_prefix_cache(
        prefix_cache,
        len(layers),
        trainable=trainable_prefix,
        boundary_leaf=boundary_leaf_prefix,
    )
    cache_packed = getattr(prefix_cache, "packed", None)
    if (packed_gen is None) != (cache_packed is None):
        raise ValueError(
            "SenseNova generation training: a packed prefix needs a PackedGenPlan "
            "and a single-prompt prefix must not get one"
        )
    if packed_gen is not None and packed_gen.prefix is not cache_packed:
        raise ValueError("SenseNova generation training: plan and prefix cache disagree")
    image_gen_indicators = torch.ones(
        hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
    )

    for layer in layers:
        def layer_forward(states: torch.Tensor, _layer=layer) -> torch.Tensor:
            # Skip only Transformers' cache-dropping wrapper; keep Module hooks.
            return nn.Module.__call__(
                _layer,
                states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=False,
                exist_image_gen_tokens=True,
                indexes=indexes,
                attention_mask=attention_mask,
                past_key_values=prefix_cache,
                use_cache=False,
                update_cache=False,
                packed_gen=packed_gen,
            )

        if checkpoint_layers:
            hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
        else:
            hidden_states = layer_forward(hidden_states)

    return model.norm_mot_gen(hidden_states)
