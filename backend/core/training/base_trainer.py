"""
Base Trainer for SushiUI Training Framework

This module contains shared training logic that is common across different
training methods (LoRA, Full Parameter, etc.).

Architecture:
- Supports SD1.5, SDXL, and Z-Image models
- Component-based approach (individual UNet, VAE, TextEncoder loading)
- Gradient checkpointing for memory efficiency
- Mixed precision training support
- SNR-weighted loss calculation
- Latent caching for faster training
- Bucketing support for variable resolutions
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple, Union, Sequence
from io import BytesIO
from PIL import Image, PngImagePlugin
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter
import json
import re
from datetime import datetime
import numpy as np
import gc
import math
from abc import ABC, abstractmethod

from core.attention import (
    AttentionMode,
    normalize_backend,
    resolve_backend,
    to_diffusers_backend,
)
from core.training.lr_utils import reassert_config_lr
from core.training.training_events import emit_training_warning
from core.training.image_preprocessing import flatten_to_rgb


# Marks an optimizer file holding one state per fused optimizer group. Absent in
# files written before fused groups were saved at all (and in every
# single-optimizer run, whose format is unchanged).
FUSED_GROUP_STATES_KEY = "_sushi_fused_group_states"


# safetensors'/torch's OWN reader messages, for the loaders that stringify the
# cause into a plain RuntimeError instead of chaining it. Deliberately does NOT
# contain "safetensor"/"corrupted"/"truncated": those match a structural refusal
# that merely names a checkpoint file, which then reloads every older checkpoint
# (17-25 GiB apiece on SenseNova) to be refused for the same reason.
_CORRUPTION_TEXT_MARKERS = (
    "error while deserializing header",
    "metadataincompletebuffer",
    "header too large",
    "header too small",
    "invalid header length",
    "file not fully covered by metadata",
    "pytorchstreamreader failed",
)


def _corruption_exception_types() -> Tuple[type, ...]:
    from json import JSONDecodeError
    from pickle import UnpicklingError
    from zipfile import BadZipFile

    types: List[type] = [EOFError, UnpicklingError, BadZipFile, JSONDecodeError]
    try:
        from safetensors import SafetensorError
        types.append(SafetensorError)
    except Exception:
        pass
    return tuple(types)


def is_checkpoint_corruption_error(exc: BaseException) -> bool:
    """True only when the checkpoint FILE could not be READ.

    Keyed on the reader's exception type (walking ``__cause__``/``__context__``,
    since loaders re-raise as RuntimeError), not on the prose of the message: a
    structural refusal -- wrong branch, wrong save format, unsupported option --
    is not a corruption signal even when it names a ``.safetensors`` path, and
    treating it as one makes the fallback reload every older checkpoint.
    """
    types = _corruption_exception_types()
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, types):
            return True
        text = str(current).lower()
        if any(marker in text for marker in _CORRUPTION_TEXT_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def all_optimizers(trainer) -> List[Any]:
    """Every optimizer that owns trainable parameters, in parameter order.

    ``trainer.optimizer`` alone is only the first entry of this list.
    """
    fused = getattr(trainer, "fused_optimizer_groups", None)
    optimizers = list(getattr(fused, "optimizers", None) or []) if fused is not None else []
    if optimizers:
        return optimizers
    optimizer = getattr(trainer, "optimizer", None)
    return [optimizer] if optimizer is not None else []


def all_lr_schedulers(trainer) -> List[Any]:
    """The scheduler driving each entry of :func:`all_optimizers`."""
    fused = getattr(trainer, "fused_optimizer_groups", None)
    if fused is not None:
        schedulers = list(getattr(trainer, "lr_schedulers", None) or [])
        if schedulers:
            return schedulers
    return [getattr(trainer, "lr_scheduler", None)]


def setup_fused_grad_norm(trainer, optimizers):
    """Give every fused-backward hook a place to record gradient norms.

    The hooks clear ``param.grad`` as soon as they have applied the update, so
    ``_calculate_grad_norms`` finds nothing and reports 0.0 for every component
    unless they record it first.
    """
    from core.training.optimizers.fused_grad_norm import (
        FusedGradNormAccumulator,
        attach_grad_norm_accumulator,
    )
    accumulator = getattr(trainer, "_fused_grad_norm", None)
    if accumulator is None:
        accumulator = FusedGradNormAccumulator()
        trainer._fused_grad_norm = accumulator
    for optimizer in optimizers:
        attach_grad_norm_accumulator(optimizer, accumulator)
    return accumulator


def setup_update_census(trainer, optimizers):
    """Arm the per-step updated-parameter census (G-RB3), if it is switched on.

    Under the fused backward pass a parameter whose hook never fires, or returns
    early, is updated by nothing for the whole run while the loss falls
    normally. The census counts the updates that were actually applied and
    compares against the parameters the optimizers own. Off by default: it costs
    a set insertion per parameter per step and a set difference per step (47.8
    us/step over 588 parameters), which is small but not free, and it is a
    diagnostic rather than a route requirement.

    Armed by ``optimizer_update_census`` in the run's train_config -- the same
    channel ``use_ema`` / ``gradient_checkpointing`` come through, so an exit
    smoke arms it by writing one key into the config it already builds rather
    than by constructing a trainer and setting an attribute. No API/UI surface;
    see the BaseTrainer.__init__ comment for why. Setting the attribute directly
    still works, and is still what a probe holding a live trainer does.
    See optimizers/update_census.py.
    """
    if not getattr(trainer, "optimizer_update_census", False):
        return None
    from core.training.optimizers.update_census import (
        UpdateCensus,
        attach_update_census,
        trainable_params_of,
    )
    census = getattr(trainer, "_update_census", None)
    if census is None:
        census = UpdateCensus()
        trainer._update_census = census
    # Names only, and a diagnostic must not be able to abort setup: the target
    # module raises when neither self.transformer nor self.unet is loaded.
    names = {}
    try:
        module = trainer._fused_backward_target_module()
    except Exception:
        module = None
    if module is not None:
        names = {id(p): n for n, p in module.named_parameters()}
    expected = []
    for optimizer in optimizers:
        attach_update_census(optimizer, census)
        expected.extend(trainable_params_of(optimizer))
    count = census.expect(expected, names, exempt=census_exempt_names(trainer))
    # Deferred, not exempt: the group stays in the expectation set and is
    # required in full on the backward that closes each window. Raises rather
    # than falling back, unlike the exemption above -- a deferral the census does
    # not know about makes every non-final step of a CORRECT run fail.
    deferred = census_deferred_parameters(trainer)
    deferred_count = census.set_deferred(deferred) if deferred else 0
    print(f"{getattr(trainer, 'log_prefix', '[Trainer]')} Updated-parameter census armed "
          f"for {count} trainable parameter(s)"
          + (f", {len(census.exempt)} path(s) exempt as gradient-unreachable"
             if census.exempt else "")
          + (f", {deferred_count} deferred to the end of each window"
             if deferred_count else ""))
    return census


def grad_norm_bucket(component):
    """``LORA_COMPONENT_*`` -> the grad-norm slot it is reported under.

    One mapping for both halves of ``_calculate_grad_norms``: the LoRA branch
    reads it from the injecting adapter's ``lora_components``, the full-FT branch
    from ``BaseFullParameterAdapter.grad_norm_components``. An unknown or missing
    component is the main trainable model, which is what the LoRA branch already
    did for a layer no adapter registered.
    """
    from core.training.adapters.base_adapter import (
        LORA_COMPONENT_TEXT_ENCODER,
        LORA_COMPONENT_TEXT_ENCODER_1,
        LORA_COMPONENT_TEXT_ENCODER_2,
        LORA_COMPONENT_VISION_ENCODER,
    )
    return {
        LORA_COMPONENT_TEXT_ENCODER_1: 'te1',
        LORA_COMPONENT_TEXT_ENCODER_2: 'te2',
        LORA_COMPONENT_TEXT_ENCODER: 'te',
        LORA_COMPONENT_VISION_ENCODER: 've',
    }.get(component, 'unet')


def census_exempt_names(trainer):
    """Parameter paths no gradient can reach by construction, for this arch.

    Not a tolerance: these are ``requires_grad`` parameters the optimizer owns
    that the loss structurally cannot reach, so a census that demanded them
    would raise on every step of a CORRECT run. SenseNova's understanding branch
    has five (SENSENOVA_TRAINING_DESIGN.md 13.4, U-2-5); the list is taken from
    the function that already predicts them by name, so the two cannot drift.
    """
    if not getattr(trainer, "is_sensenova", False):
        return ()
    try:
        from core.models.sensenova.sensenova_lora import und_gradient_unreachable_paths
        return sorted(und_gradient_unreachable_paths())
    except Exception:
        # A diagnostic must not abort setup. Reported rather than swallowed:
        # without the exemption the census would raise on a correct run.
        print(f"{getattr(trainer, 'log_prefix', '[Trainer]')} WARNING: could not "
              f"resolve the gradient-unreachable exemption list; the updated-"
              f"parameter census may report the structurally dead paths as missing")
        return ()


def census_deferred_parameters(trainer):
    """Parameters whose update lands once per window rather than once per step.

    Only SenseNova's shared-prefix four-phase window has one; empty everywhere
    else, so every other architecture keeps the per-step requirement.
    """
    if not getattr(trainer, "is_sensenova", False):
        return ()
    from core.training.sensenova_four_phase import understanding_deferred_parameters

    return understanding_deferred_parameters(trainer)


# Optimizers for which _setup_fused_backward_pass installs per-parameter hooks
# (the num_optimizer_groups == 0 branch of the Block Swap setup).
FUSED_BACKWARD_OPTIMIZERS = (
    "adafactor", "adamw8bit", "adamw8bit_ringbuffer", "lion8bit_ringbuffer",
)


def refuse_grad_scaler_under_fused_path(trainer, optimizer_type: str, mode: str) -> None:
    """FP16 GradScaler is not honoured by per-parameter fused updates.

    ``_execute_forward_backward`` scales the loss, so the gradients reaching the
    post-accumulate-grad hooks still carry the scale factor. The MAGNITUDE
    survives that: every optimizer this product offers is Adam-family, Adafactor
    or sign-based, where the scale cancels between numerator and denominator
    (measured over 10 CPU steps at scale 2**20, grads 1e-4: adamw
    scaled/unscaled = 1.0001, adafactor exactly 1.0). What does not survive is
    the rest of GradScaler's contract: the hooks apply and free each gradient, so
    the inf/NaN check never runs and ``update()`` never does either. An
    overflowing step is applied instead of skipped, which leaves NaN in the
    optimizer state permanently (measured: still NaN five finite steps later),
    and the scale stays pinned at its initial value, so nothing backs off.

    A fused-aware scaler is implementable -- unscale by the public
    ``get_scale()``, skip the individual non-finite parameter, hold the scale
    fixed -- but is not implemented. What per-parameter hooks cannot reproduce is
    GradScaler's whole-step, all-or-nothing skip.
    """
    if not getattr(trainer, "use_grad_scaler", False):
        return
    raise ValueError(
        f"FP16 mixed precision is unsupported with the {mode} that Block Swap "
        f"requires (training_dtype=fp16, mixed_precision=True, "
        f"blocks_to_swap={getattr(trainer, 'blocks_to_swap', 0)}, "
        f"num_optimizer_groups={getattr(trainer, 'num_optimizer_groups', 0)}, "
        f"optimizer={optimizer_type}). The per-parameter post-accumulate-grad "
        f"hooks apply and free each gradient during the backward pass, so "
        f"GradScaler's inf/NaN check never runs and its scale is never updated: "
        f"an overflowing step is applied instead of skipped, which leaves NaN in "
        f"the optimizer state permanently, and the scale stays at its initial "
        f"value so nothing backs off. "
        f"Options: (1) set training_dtype=bf16, which needs no gradient scaling, "
        f"(2) disable Block Swap (blocks_to_swap=0), which runs the normal "
        f"unscale_()/step()/update() path."
    )


def fused_backward_active(trainer) -> bool:
    """True when the per-parameter hooks, not optimizer.step(), apply the updates."""
    return bool(getattr(trainer, "use_fused_backward", False)) or \
        getattr(trainer, "fused_optimizer_groups", None) is not None


class FatalCudaError(RuntimeError):
    """Raised when a CUDA error is classified as "fatal" (context presumed
    dead: e.g. cudaErrorLaunchFailure / illegal memory access). Subclasses
    RuntimeError so pre-existing ``except RuntimeError`` sites (including
    third-party/library code) still catch it, but callers that specifically
    want to distinguish it from a recoverable OOM can ``except FatalCudaError``.
    """
    pass


class NothingTrainedError(RuntimeError):
    """Raised when a run has completed no backward pass, so its weights are the
    base model's. The emergency handler deliberately writes no checkpoint for
    it: saving an untrained model is the expensive thing this refusal exists to
    avoid, and the base weights are already on disk.
    """
    pass


class BucketsExhaustedError(RuntimeError):
    """Raised when a run that HAS trained loses its last fittable bucket.

    Deliberately not a ``NothingTrainedError``: those weights are worth saving,
    so this falls through to the emergency-save path. ``_unfittable_buckets``
    grows during training, so this is reachable after thousands of successful
    steps -- and ``save_every=0`` is a supported configuration, which would make
    the emergency checkpoint the only copy of the run's work.
    """
    pass


class PartialOptimizerStepError(RuntimeError):
    """Raised when a fused backward died after applying some of its updates.

    There is no snapshot to roll back to (tens of GiB of weights) and the batch
    cannot be retried without applying those updates twice, so the run stops.
    Deliberately neither a ``NothingTrainedError`` (whose weights are the base
    model's) nor a ``BucketsExhaustedError`` (whose weights are worth an
    ordinary emergency checkpoint); this one writes no ORDINARY checkpoint --
    see ``base_trainer._refuse_save_after_partial_step`` for the quarantined,
    weights-only salvage that runs instead when the CUDA context is alive.
    """
    pass


_MEMORY_BUDGET_ADVICE = (
    "The OOM was raised against this process's memory budget, which "
    "torch.cuda.set_per_process_memory_fraction and other resident allocations "
    "can put well below the installed VRAM -- free VRAM on the card does not "
    "mean the budget was not reached. Lower the training resolution, raise "
    "blocks_to_swap / enable activation offload, or raise the process memory "
    "budget, then restart the run."
)


def _vramdiag(tag: str):
    """Compact CUDA-memory snapshot print (used behind the debug_vram flag)."""
    try:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            resv = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[VRAMDIAG] {tag}: allocated={alloc:.2f}GB reserved={resv:.2f}GB peak_alloc={peak:.2f}GB", flush=True)
    except Exception:
        pass


# ============================================================
# Checkpoint entry helpers (single-file + sushiUI shard-index aware)
# ============================================================

# A sharded save writes members named "<stem>-00001-of-000NN.safetensors";
# these belong to their "<stem>.safetensors.index.json" and are never a
# checkpoint entry on their own.
_SHARD_MEMBER_RE = re.compile(r"-\d{5}-of-\d{5}\.safetensors$")
_INDEX_SUFFIX = ".safetensors.index.json"
_SAFETENSORS_SUFFIX = ".safetensors"


def _is_shard_member(name: str) -> bool:
    """True for a sharded weight member file (not a standalone checkpoint)."""
    return bool(_SHARD_MEMBER_RE.search(name))


def _checkpoint_step_from_name(name: str) -> Optional[int]:
    """Parse the training step from a checkpoint entry filename.

    Tolerates both the single-file form (``..._step_500.safetensors``), the
    shard-index form (``..._step_500.safetensors.index.json``) and the training
    state form (``..._step_500_state.json``). Returns ``None`` when no valid
    step is present (never raises, never ``int()``s a shard-member suffix).
    """
    base = name
    if base.endswith(_INDEX_SUFFIX):
        base = base[: -len(_INDEX_SUFFIX)]
    elif base.endswith(_SAFETENSORS_SUFFIX):
        base = base[: -len(_SAFETENSORS_SUFFIX)]
    elif base.endswith("_state.json"):
        base = base[: -len("_state.json")]
    elif base.endswith(".json"):
        base = base[: -len(".json")]
    m = re.search(r"_step_(\d+)", base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _checkpoint_aux_base(entry_path: Path) -> str:
    """Base name used for sibling ``_optimizer.pt`` / ``_state.json`` files.

    ``<run>_step_<n>`` for both the single-file and shard-index entry forms.
    """
    name = entry_path.name
    if name.endswith(_INDEX_SUFFIX):
        return name[: -len(_INDEX_SUFFIX)]
    if name.endswith(_SAFETENSORS_SUFFIX):
        return name[: -len(_SAFETENSORS_SUFFIX)]
    return entry_path.stem


# Marker inserted into the run_name for weight-EMA checkpoint saves
# (base_trainer._save_ema_checkpoint temporarily sets self.run_name to
# f"{run_name}{EMA_RUN_NAME_SUFFIX}" before calling the normal
# save_checkpoint()). Every EMA checkpoint entry therefore contains the
# literal substring "_ema_step_" (suffix + the "_step_" checkpoint-naming
# separator), which _list_checkpoint_entries() callers exclude so EMA
# checkpoints are never mistaken for live-weight checkpoints by resume
# detection or counted against the live-checkpoint rotation limit.
EMA_RUN_NAME_SUFFIX = "_ema"
EMA_ENTRY_MARKER = "_ema_step_"

# Same trick, for weights salvaged from a half-applied fused optimizer step
# (see _save_quarantined_partial_step_checkpoint). Every quarantined entry
# contains "_quarantined_partial_step_step_", which every _list_checkpoint_entries()
# caller excludes -- resume can never silently pick a quarantined save up as
# the live checkpoint. Loading one requires naming its exact filename via
# resume_from_checkpoint, which IS the explicit override: nothing scans for it.
QUARANTINE_RUN_NAME_SUFFIX = "_quarantined_partial_step"
QUARANTINE_ENTRY_MARKER = "_quarantined_partial_step_step_"


def _list_checkpoint_entries(
    output_dir: Path,
    exclude_substr: Optional[Union[str, Sequence[str]]] = None,
) -> List[Path]:
    """Return the checkpoint *entry* files under ``output_dir``.

    An entry is either a single-file ``*_step_*.safetensors`` save or a sharded
    ``*_step_*.safetensors.index.json`` save. Shard MEMBER files
    (``-NNNNN-of-NNNNN.safetensors``) are excluded — they belong to their index.
    ``exclude_substr`` drops entries whose name contains it (e.g.
    ``vision_encoder``); a string or a sequence of strings may be passed.
    """
    if exclude_substr is None:
        excludes: Tuple[str, ...] = ()
    elif isinstance(exclude_substr, str):
        excludes = (exclude_substr,)
    else:
        excludes = tuple(exclude_substr)

    def _is_excluded(name: str) -> bool:
        return any(sub in name for sub in excludes)

    entries: List[Path] = []
    for p in output_dir.glob("*_step_*.safetensors.index.json"):
        if _is_excluded(p.name):
            continue
        entries.append(p)
    for p in output_dir.glob("*_step_*.safetensors"):
        if _is_shard_member(p.name):
            continue
        if _is_excluded(p.name):
            continue
        entries.append(p)
    return entries


def _checkpoint_member_files(entry_path: Path) -> List[Path]:
    """All on-disk files that make up a checkpoint entry (delete as a unit).

    Single-file entry -> just itself. Shard-index entry -> the index plus every
    distinct shard listed in its ``weight_map`` (read from the index), falling
    back to the ``<stem>-NNNNN-of-NNNNN.safetensors`` glob for orphan tolerance.
    """
    name = entry_path.name
    if not name.endswith(_INDEX_SUFFIX):
        return [entry_path]

    files: List[Path] = [entry_path]
    directory = entry_path.parent
    seen: set = set()
    try:
        with open(entry_path, encoding="utf-8") as f:
            index = json.load(f)
        for shard in (index.get("weight_map", {}) or {}).values():
            if shard not in seen:
                seen.add(shard)
                files.append(directory / shard)
    except Exception:
        pass
    # Orphan tolerance: also pick up any matching shard members on disk.
    stem = name[: -len(_INDEX_SUFFIX)]
    for p in directory.glob(f"{stem}-*-of-*.safetensors"):
        if p.name not in seen and _is_shard_member(p.name):
            seen.add(p.name)
            files.append(p)
    return files


# ============================================================
# Training Logger Helper
# ============================================================

def log_verbose(message: str):
    """
    Log verbose messages only to file (not to console).
    Uses global logger from train_runner.py if available.

    Args:
        message: Message to log
    """
    # Import logger from train_runner (circular import avoided by late import)
    try:
        from core.training.train_runner import logger
        if logger is not None:
            logger.log_only(message)
        # If logger not initialized, silently ignore (avoid spamming console during tests)
    except (ImportError, AttributeError):
        # Logger not available (e.g., during unit tests), silently ignore
        pass


# ============================================================
# Utility Functions
# ============================================================

def print_vram_usage(label: str = ""):
    """
    Print detailed VRAM usage statistics.

    Args:
        label: Optional label to identify the checkpoint
    """
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

    print(f"[VRAM] {label if label else 'Current'}")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Peak:      {max_allocated:.2f} GB")


def get_tensor_memory_mb(tensor: torch.Tensor) -> float:
    """Get memory usage of a tensor in MB."""
    return tensor.element_size() * tensor.nelement() / 1024**2


def format_param_count(n: int) -> str:
    """Format parameter count as B (>=1B) or M (>=1M) or K."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    elif n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert dtype string to torch.dtype.

    Args:
        dtype_str: String like "fp16", "fp32", "bf16", "fp8_e4m3fn", "fp8_e5m2"

    Returns:
        torch.dtype
    """
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }

    if dtype_str not in dtype_map:
        print(f"[Trainer] WARNING: Unknown dtype '{dtype_str}', defaulting to fp16")
        return torch.float16

    return dtype_map[dtype_str]


def compute_snr(noise_scheduler, timesteps, alphas_cumprod_cached=None):
    """
    Computes SNR (Signal-to-Noise Ratio) from diffusion timesteps.

    SNR = alpha_bar / (1 - alpha_bar)

    Args:
        noise_scheduler: DDPMScheduler instance
        timesteps: Tensor of timesteps [batch_size]
        alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (optional, for performance)

    Returns:
        SNR values [batch_size]
    """
    # Get alpha_bar for each timestep
    # Use cached version if available (avoids repeated .to(device) calls)
    if alphas_cumprod_cached is not None:
        alphas_cumprod = alphas_cumprod_cached
    else:
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)
    alpha_bar = alphas_cumprod[timesteps].float()

    # SNR = alpha / (1 - alpha)
    snr = alpha_bar / (1.0 - alpha_bar)

    return snr


def apply_snr_weight(loss, timesteps, noise_scheduler, min_snr_gamma=5.0, return_weights=False, alphas_cumprod_cached=None):
    """
    Apply Min-SNR gamma weighting to loss.

    Reference: "Efficient Diffusion Training via Min-SNR Weighting Strategy"
    https://arxiv.org/abs/2303.09556

    This reweights the loss to ensure all timesteps contribute equally to training,
    preventing the model from overfitting to high-noise timesteps.

    Args:
        loss: Unreduced loss tensor [batch_size, ...]
        timesteps: Tensor of timesteps [batch_size]
        noise_scheduler: DDPMScheduler instance
        min_snr_gamma: Minimum SNR gamma value (default: 5.0, standard for SD/SDXL)
        return_weights: If True, also return the weight values [batch_size]
        alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (optional, for performance)

    Returns:
        Weighted loss (same shape as input)
        If return_weights=True: (weighted_loss, weights [batch_size])
    """
    snr = compute_snr(noise_scheduler, timesteps, alphas_cumprod_cached)

    # Min-SNR gamma weighting: min(SNR, gamma) / SNR
    # This clamps the weight for low-noise (high SNR) timesteps
    mse_loss_weights = torch.clamp(snr, max=min_snr_gamma) / snr

    # Keep original 1D weights for return
    weights_1d = mse_loss_weights.clone()

    # Reshape to match loss dimensions [batch_size, 1, 1, 1]
    while mse_loss_weights.dim() < loss.dim():
        mse_loss_weights = mse_loss_weights.unsqueeze(-1)

    # Apply weighting
    weighted_loss = loss * mse_loss_weights

    if return_weights:
        return weighted_loss, weights_1d
    return weighted_loss


def _per_sample_masked_mean(per_element: torch.Tensor, mask: torch.Tensor, channels: int) -> torch.Tensor:
    """Per-sample mean of `per_element` [B,C,H,W] over cells where `mask` [B,1,H,W]
    (float or bool) is truthy. Returns a [B] tensor. Samples whose mask is entirely
    empty (e.g. no seam ring exists for that crop) return NaN rather than a spurious
    0/eps -- callers must convert NaN -> null when serializing so an empty-mask
    sample doesn't silently bias a bin's mean toward 0.

    Monitoring-only helper (outpaint loss-vs-timestep instrumentation); never used
    inside the gradient path.
    """
    m = mask.float()
    cell_count = m.sum(dim=[1, 2, 3])  # [B], H*W cells (mask channel dim is 1)
    denom = (cell_count * float(channels)).clamp_min(1e-8)
    num = (per_element * m).sum(dim=[1, 2, 3])
    result = num / denom
    return torch.where(cell_count > 0, result, torch.full_like(result, float("nan")))


def _per_sample_snr(noise_process: str, timesteps: torch.Tensor, noise_scheduler, alphas_cumprod_cached=None) -> torch.Tensor:
    """Per-sample SNR = alpha_bar/(1-alpha_bar) (DDPM, integer `timesteps`) or
    ((1-t)/t)^2 (flow, continuous t in [0,1]). Monitoring-only (loss-vs-timestep
    instrumentation); independent of Min-SNR gamma clamping used for the loss.
    """
    if noise_process == "flow":
        t = timesteps.float().clamp_min(1e-6)
        return ((1.0 - t) / t) ** 2
    return compute_snr(noise_scheduler, timesteps, alphas_cumprod_cached)


def get_target_from_prediction_type(
    noise_scheduler,
    prediction_type: str,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Get the target tensor based on prediction type (LEGACY - DDPM only).

    DEPRECATED: Use add_noise_unified() and get_target_unified() instead.

    Args:
        noise_scheduler: DDPMScheduler instance
        prediction_type: "epsilon" (noise), "v_prediction", or "sample"
        latents: Original latents [B, C, H, W]
        noise: Sampled noise [B, C, H, W]
        timesteps: Timesteps [B]

    Returns:
        Target tensor for loss calculation
    """
    if prediction_type == "epsilon":
        # Predict noise (most common for SD/SDXL)
        return noise

    elif prediction_type == "v_prediction":
        # Predict velocity (v = alpha_bar * noise - sqrt(1 - alpha_bar) * latents)
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device)
        alpha_bar = alphas_cumprod[timesteps].float()

        # Reshape alpha_bar to [B, 1, 1, 1]
        while alpha_bar.dim() < latents.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        velocity = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * latents
        return velocity

    elif prediction_type == "sample":
        # Predict original sample (less common)
        return latents

    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")


def add_noise_unified(
    noise_process: str,
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Add noise to latents using specified noise process (Unified Framework).

    Args:
        noise_process: "ddpm" or "flow"
        noise_scheduler: Noise scheduler instance (DDPMScheduler or FlowMatchEulerDiscreteScheduler)
        latents: Original latents [B, C, H, W]
        noise: Sampled noise [B, C, H, W]
        timesteps: Timesteps (discrete for DDPM, continuous [0,1] for Flow)

    Returns:
        Noisy latents
    """
    if noise_process == "ddpm":
        # DDPM: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # timesteps are discrete [0, num_train_timesteps)
        return noise_scheduler.add_noise(latents, noise, timesteps)

    elif noise_process == "flow":
        # Flow Matching: x_t = (1 - t) * x_0 + t * noise
        # At t=0: x_t = x_0 (clean latents)
        # At t=1: x_t = noise (pure noise)
        # timesteps are continuous [0, 1]
        t = timesteps.float()
        while t.dim() < latents.dim():
            t = t.unsqueeze(-1)

        noisy_latents = (1.0 - t) * latents + t * noise
        return noisy_latents

    else:
        raise ValueError(f"Unknown noise_process: {noise_process}")


def get_target_unified(
    noise_process: str,
    prediction_target: str,
    noise_scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Get the training target based on noise process and prediction target (Unified Framework).

    Args:
        noise_process: "ddpm" or "flow"
        prediction_target: "epsilon", "velocity", or "sample"
        noise_scheduler: Noise scheduler instance
        latents: Original latents [B, C, H, W]
        noise: Sampled noise [B, C, H, W]
        timesteps: Timesteps (discrete for DDPM, continuous [0,1] for Flow)

    Returns:
        Target tensor for loss calculation
    """
    if noise_process == "ddpm":
        # DDPM noise process with discrete timesteps
        if prediction_target == "epsilon":
            # Predict noise
            return noise

        elif prediction_target == "velocity":
            # Predict velocity: v = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x_0
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device)
            alpha_bar = alphas_cumprod[timesteps].float()

            while alpha_bar.dim() < latents.dim():
                alpha_bar = alpha_bar.unsqueeze(-1)

            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

            velocity = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * latents
            return velocity

        elif prediction_target == "sample":
            # Predict original sample
            return latents

        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    elif noise_process == "flow":
        # Flow Matching with continuous timesteps [0, 1]
        if prediction_target == "epsilon":
            # Predict noise (Flow + epsilon is unusual but supported)
            return noise

        elif prediction_target == "velocity":
            # Predict velocity: v = noise - x_0 (direction from x_0 to noise)
            # This matches diffusers: target = noise - model_input
            return noise - latents

        elif prediction_target == "sample":
            # Predict original sample
            return latents

        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    else:
        raise ValueError(f"Unknown noise_process: {noise_process}")


def predict_original_latent_unified(
    noise_process: str,
    prediction_target: str,
    noise_scheduler,
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Predict original latent from model prediction (Unified Framework).

    Used for regularization losses (SNR, Energy) and reconstruction loss monitoring.

    Args:
        noise_process: "ddpm" or "flow"
        prediction_target: "epsilon", "velocity", or "sample"
        noise_scheduler: Noise scheduler instance
        noisy_latents: Noisy latents [B, C, H, W]
        model_pred: Model prediction [B, C, H, W]
        timesteps: Timesteps (discrete for DDPM, continuous [0,1] for Flow)

    Returns:
        Predicted original latent [B, C, H, W]
    """
    if noise_process == "ddpm":
        # DDPM: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        alpha_bar = alphas_cumprod[timesteps]

        while alpha_bar.dim() < noisy_latents.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        if prediction_target == "epsilon":
            # model_pred = noise, solve for x_0: x_0 = (x_t - sqrt(1 - alpha_bar) * noise) / sqrt(alpha_bar)
            predicted_latent = (noisy_latents - sqrt_one_minus_alpha_bar * model_pred) / sqrt_alpha_bar
        elif prediction_target == "velocity":
            # model_pred = v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_0
            # Solve for x_0: x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
            predicted_latent = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * model_pred
        elif prediction_target == "sample":
            # model_pred = x_0 directly
            predicted_latent = model_pred
        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    elif noise_process == "flow":
        # Flow Matching: x_t = (1 - t) * x_0 + t * noise
        # At t=0: x_t = x_0, At t=1: x_t = noise
        t = timesteps.float()
        while t.dim() < noisy_latents.dim():
            t = t.unsqueeze(-1)

        if prediction_target == "epsilon":
            # model_pred = noise, solve for x_0: x_0 = (x_t - t * noise) / (1 - t)
            # Avoid division by zero at t=1
            epsilon = 1e-8
            predicted_latent = (noisy_latents - t * model_pred) / (1.0 - t + epsilon)
        elif prediction_target == "velocity":
            # model_pred = v = noise - x_0
            # From diffusers: x_0 = x_t - t * v (line 459: x0 = sample - current_sigma * model_output)
            predicted_latent = noisy_latents - t * model_pred
        elif prediction_target == "sample":
            # model_pred = x_0 directly
            predicted_latent = model_pred
        else:
            raise ValueError(f"Unknown prediction_target: {prediction_target}")

    else:
        raise ValueError(f"Unknown noise_process: {noise_process}")

    return predicted_latent


# ============================================================
# Parameter Change Tracker
# ============================================================

# Split to its own module (plan P8). Re-exported here so existing importers of
# ``base_trainer.ParameterChangeTracker`` keep working (zero caller churn).
from core.training.parameter_change_tracker import ParameterChangeTracker
from core.training.periodic_intervals import due as interval_due, normalize_interval


# ============================================================
# Base Trainer Class
# ============================================================

class BaseTrainer(ABC):
    """
    Abstract base trainer class with shared logic for all training methods.

    Subclasses must implement:
    - setup_trainable_parameters()
    - save_checkpoint()
    - load_checkpoint()
    - find_latest_checkpoint() (optional)
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        run_name: str = None,
        run_id: Optional[int] = None,  # Database run ID for metrics logging
        learning_rate: float = 1e-4,
        device: str = "cuda",
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        debug_vram: bool = False,
        use_flash_attention: bool = False,
        attention_backend: Optional[str] = None,
        attention_impl: Optional[str] = None,
        min_snr_gamma: float = 5.0,
        reconstruction_loss_weight: float = 0.0,
        # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
        prompt_chunking_mode: str = "a1111",  # "a1111", "sd_scripts", "nobos"
        max_prompt_chunks: int = 0,  # 0 = unlimited
        # Component-specific learning rates
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
        image_encoder_lr: Optional[float] = None,  # Image Encoder (future T2I support)
        # Block Swap settings (training VRAM optimization)
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = False,
        # Per-bucket activation offload dispatcher (proactive, OOM-detection-free)
        activation_dispatch_enable: bool = False,
        activation_dispatch_margin_gb: float = 1.0,
        activation_dispatch_seed_coef: float = 24.0e-6,
        activation_dispatch_residual_frac: float = 0.85,
        activation_dispatch_threshold_mb: int = 4,
        # Fused optimizer groups (for any optimizer with Block Swap)
        num_optimizer_groups: int = 0,
        # Optimizer options and hyperparameters.
        # (No optimizer_is_paged: paging is selected by the optimizer TYPE
        # NAME -- paged_adamw / paged_adamw8bit / paged_lion8bit -- which is
        # what OptimizerFactory dispatches on.)
        optimizer_cautious: bool = False,
        optimizer_beta1: Optional[float] = None,
        optimizer_beta2: Optional[float] = None,
        optimizer_epsilon: Optional[float] = None,
        optimizer_weight_decay: Optional[float] = None,
        # Schedule-Free optimizer options (RingBuffer optimizers only)
        optimizer_schedule_free: bool = False,
        optimizer_warmup_steps: int = 0,
        optimizer_schedule_free_r: float = 0.0,
        optimizer_schedule_free_weight_lr_power: float = 2.0,
        optimizer_use_radam: bool = False,
        # Stochastic rounding for BF16 parameter updates (RingBuffer optimizers only)
        optimizer_stochastic_rounding: bool = False,
        # Resume training
        resume_from_checkpoint: Optional[str] = None,
        # The full train_config dict from the YAML. Stored as self.config so
        # arch-specific setup (Anima FP8 base, cpu_offload checkpointing,
        # LoRA scope, LR factors, ...) can read its own keys via
        # self.config.get(...). Without this, _load_*_components and
        # _apply_lora run with an empty dict and silently ignore user input.
        train_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base trainer.

        Args:
            model_path: Path to base Stable Diffusion model (or checkpoint for resume)
            resume_from_checkpoint: "latest" to auto-detect, or path to specific checkpoint
            output_dir: Directory to save checkpoints
            run_name: Training run name (for checkpoint filename generation)
            learning_rate: Learning rate
            device: Device to use (cuda/cpu)
            weight_dtype: Model weight dtype (fp16, fp32, bf16, fp8_e4m3fn, fp8_e5m2)
            training_dtype: Training/activation dtype (fp16, bf16, fp8_e4m3fn, fp8_e5m2)
            output_dtype: Output dtype for safetensors (fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2)
            vae_dtype: VAE-specific dtype (fp16, fp32, bf16) - SDXL VAE works fine with fp16
            mixed_precision: Enable mixed precision training (autocast)
            debug_vram: Enable detailed VRAM profiling (default: False)
            use_flash_attention: DEPRECATED compat boolean; when attention_backend is
                None it selects flash (True) vs native (False). Superseded by
                attention_backend and re-derived as (attention_backend != 'native').
            attention_backend: Attention backend string selector for training
                ("native"|"flash"; "sage" refused for training -> native). Overrides
                use_flash_attention when set.
            min_snr_gamma: Min-SNR gamma value for loss weighting (default: 5.0, 0 to disable)
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.run_name = run_name or Path(output_dir).name
        self.run_id = run_id  # Database run ID (for dual logging: TensorBoard + DB)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics buffering for DB performance optimization
        # Batch DB commits every N steps instead of every step
        self._metrics_buffer = []
        self._metrics_flush_interval = 10  # Flush every 10 steps (configurable)
        # Bespoke per-step scalar metrics (arch/method-specific) accumulated by
        # log_extra_metric() and captured into each metrics-buffer row as a
        # {name: float} dict (TrainingMetrics.extra_metrics JSON). Cleared after
        # each capture so a metric emitted only every N steps never goes stale.
        self._extra_metrics = {}
        # Epoch / resume-session tags recorded with each metric (for the UI's
        # epoch-boundary lines and resume markers). resume_seq is recomputed at
        # run start; _current_epoch is updated in the epoch loop.
        self._current_epoch = 0
        self.resume_seq = 0

        # Async DB logging with ThreadPoolExecutor
        # DB writes happen in background thread, not blocking training loop
        from concurrent.futures import ThreadPoolExecutor
        self._db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_logger")
        self._db_futures = []  # Track pending futures for cleanup

        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Component-specific learning rates
        self.unet_lr = unet_lr if unet_lr is not None else learning_rate
        self.text_encoder_lr = text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_1_lr = text_encoder_1_lr if text_encoder_1_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.text_encoder_2_lr = text_encoder_2_lr if text_encoder_2_lr is not None else text_encoder_lr if text_encoder_lr is not None else learning_rate
        self.image_encoder_lr = image_encoder_lr if image_encoder_lr is not None else learning_rate

        # Block Swap settings (training VRAM optimization)
        self.blocks_to_swap = blocks_to_swap
        self.use_pinned_memory = use_pinned_memory
        # H2D-only block swap settings (FLUX.2 LoRA / frozen-base training).
        # Read from the same train_config source that populates blocks_to_swap /
        # use_pinned_memory (train_runner passes those via constructor from
        # train_config.get(...)). Defaults: h2d_only=False, ring_size=2.
        _tc = train_config if train_config else {}
        from api.param_defaults import TRAINING_DEFAULTS as _TD_PHASE_EVICTION
        self.sensenova_mot_phase_eviction = bool(_tc.get(
            "sensenova_mot_phase_eviction",
            _TD_PHASE_EVICTION["sensenova_mot_phase_eviction"],
        ))
        self.sensenova_phase_evictor = None
        # Four-phase eviction (SENSENOVA_TRAINING_DESIGN.md 8.3.2): splits the
        # single backward at the prefix KV cache so a TRAINED understanding half
        # can still be evicted. Same config channel as the flag above.
        self.sensenova_four_phase_eviction = bool(_tc.get(
            "sensenova_four_phase_eviction",
            _TD_PHASE_EVICTION["sensenova_four_phase_eviction"],
        ))
        self.sensenova_four_phase = None
        # Share one boundary cut across an MNT window (8.3.5). Opt-in and OFF by
        # default: it changes what the understanding half trains on, so it is not
        # folded into the split above.
        self.sensenova_four_phase_shared_prefix = bool(_tc.get(
            "sensenova_four_phase_shared_prefix",
            _TD_PHASE_EVICTION["sensenova_four_phase_shared_prefix"],
        ))
        self.sensenova_four_phase_grad_reduction = str(_tc.get(
            "sensenova_four_phase_grad_reduction",
            _TD_PHASE_EVICTION["sensenova_four_phase_grad_reduction"],
        ))
        # Training-time sample only; see the KV cache streaming module and
        # ops/sensenova_ops.py::_maybe_install_sample_kv_streaming.
        self.sensenova_sample_kv_cache_streaming = bool(_tc.get(
            "sensenova_sample_kv_cache_streaming",
            _TD_PHASE_EVICTION["sensenova_sample_kv_cache_streaming"],
        ))
        # Validated by SenseNovaFullParameterAdapter, its only reader.
        self.sensenova_full_finetune_save_format = str(_tc.get(
            "sensenova_full_finetune_save_format",
            _TD_PHASE_EVICTION["sensenova_full_finetune_save_format"],
        ))
        self.block_swap_h2d_only = bool(_tc.get("block_swap_h2d_only", False))
        self.block_swap_ring_size = int(_tc.get("block_swap_ring_size", 2))

        # TREAD token routing (arXiv 2501.04765) — training-only, currently wired
        # for the Anima DiT (other archs ignore self.tread_config). When enabled,
        # the arch ops attach this dict to the transformer for each training
        # forward (cleared for sampling). Default OFF (self.tread_config is None).
        self.tread_config = None
        if bool(_tc.get("tread_enable", False)):
            self.tread_config = {
                "drop_ratio": float(_tc.get("tread_drop_ratio", 0.5)),
                "start_block": int(_tc.get("tread_start_block", 2)),
                "end_block": int(_tc.get("tread_end_block", 26)),
            }
            print(f"[TREAD] Token routing ENABLED: {self.tread_config} "
                  f"(training-only; sampling runs the full network)")
            # torch.compile + routing: the routed span runs a second, smaller token
            # count, so compile sees two static shapes per bucket (recompiles once
            # per shape, not per step). Warn so the user expects the extra compile.
            if str(_tc.get("torch_compile", "off")).lower() not in ("off", "", "none"):
                print("[TREAD] WARNING: torch_compile is on with routing - expect an "
                      "extra one-time recompile for the routed (reduced-token) shape.")

        # Low-rate stochastic depth (per-batch block dropout) — training-only,
        # currently wired for the Anima DiT (other archs ignore self.block_skip_config).
        # Each step, eligible blocks (front/back, outside the protected middle span)
        # are independently dropped with prob block_skip_rate; executed eligible
        # blocks rescale their residual by 1/(1-rate). Sampling runs every block.
        # Default OFF (self.block_skip_config is None).
        self.block_skip_config = None
        _skip_rate = float(_tc.get("block_skip_rate", 0.0))
        if _skip_rate > 0.0:
            # Cap the rate: high dropout on a pretrained DiT degrades quality fast.
            if _skip_rate > 0.35:
                print(f"[BlockSkip] WARNING: block_skip_rate={_skip_rate} exceeds "
                      f"0.35; clamping to 0.35 (high block dropout on a pretrained "
                      f"DiT degrades quality).")
                _skip_rate = 0.35
            self.block_skip_config = {
                "skip_rate": _skip_rate,
                "protect_start": int(_tc.get("block_skip_protect_start", 6)),
                "protect_end": int(_tc.get("block_skip_protect_end", 22)),
            }
            print(f"[BlockSkip] Stochastic depth ENABLED: {self.block_skip_config} "
                  f"(training-only; sampling runs every block)")
            if str(_tc.get("torch_compile", "off")).lower() not in ("off", "", "none"):
                print("[BlockSkip] WARNING: torch_compile is on with per-batch block "
                      "dropout - the skip pattern varies each step (dynamic control "
                      "flow), causing recompiles; prefer torch_compile=off.")

        # DiT-BlockSkip (arXiv 2603.20755) — training-only MEMORY-REDUCTION for
        # LoRA fine-tuning. Skips the first `front` and last `back` transformer
        # blocks; LoRA lives ONLY in the unskipped middle blocks. Each training
        # step the arch ops attach this dict to the transformer, whose forward
        # then (a) runs a no_grad full pass to capture the skipped spans' residual
        # features Delta and (b) runs a gradient pass over ONLY the middle blocks,
        # re-adding Delta at the span boundaries. Backprop flows only through the
        # middle blocks, so the skipped blocks retain NO backward activations and
        # hold NO optimizer state (frozen, no LoRA). Wired for the Anima DiT
        # (anima_ops.py) and for LTX-2.3 video (ltx2_ops.py + the DUAL-stream
        # fold in Ltx2BlockLoopWrapper._blockskip_forward, which folds BOTH the
        # video and audio streams); other archs ignore self.blockskip_config.
        # Default OFF.
        self.blockskip_config = None
        if bool(_tc.get("blockskip_enable", False)):
            trainer_cls = type(self).__name__
            # The skipped front/back blocks run ONLY inside the per-step no_grad
            # precompute pass, so they retain NO backward activations (the memory
            # saving) and receive NO gradient; the middle blocks run under grad and
            # carry the training signal (LoRA adapters or full parameters). NOTE:
            # the skipped blocks are NOT requires_grad_(False) / optimizer-excluded
            # here — any adapter/params they own simply stay gradient-starved (they
            # keep optimizer slots but never update). The fold is exact because both
            # passes use identical (LoRA-active) weights, so the precomputed span
            # residual equals what the full network would produce.
            # ReLoRATrainer (subclass of LoRATrainer, exact name "ReLoRATrainer")
            # and ControlNet are NOT supported.
            if trainer_cls not in ("LoRATrainer", "FullParameterTrainer"):
                raise ValueError(
                    "blockskip_enable is supported only for LoRA and full "
                    f"fine-tune training (trainer is {trainer_cls}). BlockSkip "
                    "freezes the first/last blocks and trains only the middle "
                    "blocks; ReLoRA and ControlNet are unsupported. Disable "
                    "blockskip_enable or switch to LoRA / full-parameter training."
                )
            _bs_front = int(_tc.get("blockskip_front", 4))
            _bs_back = int(_tc.get("blockskip_back", 4))
            if _bs_front < 0 or _bs_back < 0:
                raise ValueError(
                    f"blockskip_front/back must be >= 0 (got front={_bs_front}, "
                    f"back={_bs_back})."
                )
            # Mutual exclusion: these all rewrite the block loop / stream and cannot
            # compose with BlockSkip's precompute+reconstruct.
            if self.tread_config is not None:
                raise ValueError(
                    "blockskip_enable is mutually exclusive with tread_enable "
                    "(both restructure the transformer block loop). Enable only one."
                )
            if self.block_skip_config is not None:
                raise ValueError(
                    "blockskip_enable is mutually exclusive with block_skip_rate "
                    "(stochastic depth). Enable only one."
                )
            if self.blocks_to_swap > 0:
                raise ValueError(
                    "blockskip_enable requires blocks_to_swap=0. BlockSkip already "
                    "removes the skipped blocks from the backward graph; the "
                    "block-swap conductor also cannot manage blocks that BlockSkip "
                    "keeps resident/frozen. Set blocks_to_swap=0."
                )
            self.blockskip_config = {
                "front": _bs_front,
                "back": _bs_back,
            }
            print(f"[BlockSkip] DiT-BlockSkip ENABLED (activation-memory reduction, "
                  f"trainer={trainer_cls}): skip first {_bs_front} + last {_bs_back} "
                  f"blocks; only the middle blocks train under grad (LoRA or full "
                  f"parameters); skipped blocks run no_grad only (no backward "
                  f"activations, gradient-starved - not optimizer-excluded); span "
                  f"residuals kept in memory per step (training-only; sampling runs "
                  f"the full network).")
            if str(_tc.get("torch_compile", "off")).lower() not in ("off", "", "none"):
                print("[BlockSkip] WARNING: torch_compile is on - BlockSkip runs a "
                      "two-pass (no_grad full + grad middle) forward with dynamic "
                      "control flow; prefer torch_compile=off.")

        # Per-bucket activation offload dispatcher settings. The dispatcher is
        # created lazily on the first executed step (once static VRAM is known).
        self.activation_dispatch_enable = activation_dispatch_enable
        self.activation_dispatch_margin_gb = activation_dispatch_margin_gb
        self.activation_dispatch_seed_coef = activation_dispatch_seed_coef
        self.activation_dispatch_residual_frac = activation_dispatch_residual_frac
        self.activation_dispatch_threshold_mb = activation_dispatch_threshold_mb
        self.activation_dispatcher = None
        # Resolution buckets (image w, h) that OOM even at micro-batch=1 -> they
        # don't fit even one sample. Populated by the OOM recovery, consumed at the
        # next epoch's re-bucketing to drop those buckets (no point retrying every
        # occurrence). _batch_was_unfittable is the per-step signal from recovery.
        self._unfittable_buckets = set()
        self._batch_was_unfittable = False

        # Fused optimizer settings (for Block Swap compatibility)
        self.num_optimizer_groups = num_optimizer_groups
        self.use_fused_backward = False  # Adafactor per-parameter updates
        self.fused_optimizer_groups = None  # FusedOptimizerGroups instance (for any optimizer)
        # setup_fused_grad_norm(): where the fused hooks record squared gradient
        # norms before clearing param.grad. None when no fused path is configured.
        self._fused_grad_norm = None

        # Optimizer options and hyperparameters (defaults will be used if None)
        self.optimizer_cautious = optimizer_cautious
        self.optimizer_beta1 = optimizer_beta1
        self.optimizer_beta2 = optimizer_beta2
        self.optimizer_epsilon = optimizer_epsilon
        self.optimizer_weight_decay = optimizer_weight_decay

        # Schedule-Free optimizer options (RingBuffer optimizers only)
        self.optimizer_schedule_free = optimizer_schedule_free
        self.optimizer_warmup_steps = optimizer_warmup_steps
        self.optimizer_schedule_free_r = optimizer_schedule_free_r
        self.optimizer_schedule_free_weight_lr_power = optimizer_schedule_free_weight_lr_power
        self.optimizer_use_radam = optimizer_use_radam

        # Stochastic rounding for BF16 parameter updates. BF16 round-to-nearest
        # drops every update below half a ULP, so with a BF16 weight_dtype most
        # elements never move; stochastic rounding keeps those updates alive in
        # expectation. Honoured by the RingBuffer optimizers only.
        self.optimizer_stochastic_rounding = optimizer_stochastic_rounding

        # Two diagnostics/mode switches with NO API surface, deliberately: they
        # are config-channel only (this dict), so the UI and the OpenAPI schema
        # do not offer them and only a hand-written YAML, a probe or an exit
        # smoke that owns its config can arm them. Neither is a quality setting
        # and neither has a result the UI can display, which is why the
        # param_defaults -> routes -> openapi -> panel chain is not spent on
        # them; the config channel is what makes them armable without
        # constructing a trainer by hand. SENSENOVA_TRAINING_DESIGN.md 6.5.
        #
        # optimizer_state_host_resident: ring-buffer optimizer state on pinned
        # host memory instead of the GPU (the mode those optimizers are named
        # for). Measured at 0.031250 B/param on the GPU against 2.031250 (G-RB2,
        # e6bdcc38). ALSO an API/UI parameter, unlike the census below: SenseNova
        # full fine-tuning admits the two ring-buffer optimizers only with it,
        # and a config-only switch guarding an API-selectable optimizer name is
        # how a run ends up with 32.9 GB of 8-bit state on the GPU.
        # See _ringbuffer_optimizer_kwargs, host_state_allocator.
        self.optimizer_state_host_resident = bool(
            _tc.get("optimizer_state_host_resident", False))
        self._host_state_allocator = None

        # optimizer_update_census (G-RB3): per-step census of which parameters an
        # update actually reached. FUSED-BACKWARD ONLY: setup_update_census() is
        # called from _setup_fused_backward_pass, because the failure it detects
        # (a hook that never fires) exists only where hooks apply the updates.
        # Setting it on a non-fused run is reported, not silently ignored (see
        # the note in setup_optimizer). It RAISES on a shortfall, which is the
        # other reason it is not a checkbox: on a false positive it would take
        # down a run that is training correctly. See optimizers/update_census.py.
        self.optimizer_update_census = bool(
            _tc.get("optimizer_update_census", False))
        self._update_census = None

        # Resume training
        self.resume_from_checkpoint = resume_from_checkpoint
        self._loaded_checkpoint_path = None  # Actual checkpoint path loaded (may differ from requested if fallback occurred)

        # Convert dtype strings to torch.dtype
        self.weight_dtype = get_torch_dtype(weight_dtype)
        self.training_dtype = get_torch_dtype(training_dtype)
        self.output_dtype = get_torch_dtype(output_dtype)
        self.vae_dtype = get_torch_dtype(vae_dtype)
        self.mixed_precision = mixed_precision
        self.debug_vram = debug_vram
        # Attention backend (single source of truth for training). The string
        # selector is authoritative; fall back to the legacy boolean for old
        # presets/YAML that only set use_flash_attention. normalize_backend maps
        # aliases/None -> canonical keys and preserves passthrough backends.
        # R5: self.use_flash_attention becomes a DERIVED compat mirror so all
        # existing `if self.use_flash_attention:` guard sites keep firing AND the
        # string selector (e.g. 'flash'/'sage') triggers them. sage is refused
        # per-hook by resolve_backend(mode=TRAINING) (R4 defense-in-depth).
        self.attention_backend = normalize_backend(
            attention_backend or ('flash' if use_flash_attention else 'native')
        )
        self.use_flash_attention = (self.attention_backend != 'native')
        # Attention implementation registry ("conduit" | "diffusers"). Selects WHICH
        # registry executes the training attention kernel; ORTHOGONAL to
        # attention_backend (WHICH kernel). "conduit" -> unified backend/core/attention
        # dispatch; "diffusers" -> the pre-migration set_attention_backend path.
        #
        # Backward-compat resolution: a FRESH run defaults to "conduit" (the new
        # default). A RESUME whose saved config LACKS the key (attention_impl is None
        # here because train_runner read a pre-migration config) defaults to
        # "diffusers" so in-flight runs reproduce old numerics. The resolved value is
        # persisted back into the run config so every subsequent resume is stable.
        #
        # NOTE: _setup_attention_backend_sd_sdxl and _setup_attention_backend_flux2 both
        # consume self.attention_impl (conduit branch installs the conduit processors with
        # mode=TRAINING; diffusers branch keeps set_attention_backend). Ideogram4 remains on
        # diffusers dispatch (head_dim=256 rules out conduit-only backends) and ignores the flag.
        if attention_impl is None:
            self.attention_impl = "diffusers" if self.resume_from_checkpoint else "conduit"
        else:
            self.attention_impl = attention_impl
        self._persist_attention_impl()
        self.min_snr_gamma = min_snr_gamma
        self.reconstruction_loss_weight = reconstruction_loss_weight

        # Initialize GradScaler for mixed precision training
        # GradScaler is needed when:
        # - training_dtype is fp16 (autocast is used)
        # - This includes cases where LoRA weights (fp32) are autocast to training dtype
        # GradScaler prevents gradient underflow during fp16 backward pass
        #
        # NOTE: BFloat16 does NOT need GradScaler because:
        # - BF16 has the same exponent range as FP32 (8 bits), so it doesn't suffer from
        #   the same overflow/underflow issues as FP16 (5 bit exponent)
        # - PyTorch's _amp_foreach_non_finite_check_and_unscale_cuda is not implemented for BF16
        # - Most modern training (FLUX.2, etc.) uses BF16 without GradScaler
        self.use_grad_scaler = (
            self.mixed_precision and
            self.training_dtype == torch.float16  # Only FP16, not BF16
        )
        # Reject the unsupported FP16 full fine-tune + GradScaler combination.
        # torch's GradScaler.unscale_() requires FP32 master parameters: when the
        # trainable params are themselves FP16 it raises
        # "Attempting to unscale FP16 gradients" at the first optimizer step.
        # For LoRA/ReLoRA/ControlNet the trainable adapter params are lora_dtype
        # (FP32 by default) while the FP16 base stays frozen, so unscale_() only
        # ever touches FP32 grads and is safe. Full-parameter FT trains the base
        # weights directly, so FP16 weight_dtype + FP16 mixed precision is broken.
        # Fail loudly at setup (before any GPU work) with an actionable remedy
        # rather than silently changing numerics.
        from core.training.ops.training_method import is_full_finetune
        if (
            is_full_finetune(self)
            and self.use_grad_scaler
            and self.weight_dtype == torch.float16
        ):
            raise ValueError(
                "FP16 full fine-tune is unsupported with mixed precision: "
                "weight_dtype=fp16 + training_dtype=fp16 + mixed_precision=True "
                "trains FP16 base weights, but torch's GradScaler requires FP32 "
                "master parameters and raises 'Attempting to unscale FP16 "
                "gradients' at step 1. Remedy: set weight_dtype=bf16 and "
                "training_dtype=bf16 (bf16 needs no gradient scaling and is the "
                "default for full fine-tune). LoRA is unaffected because its "
                "trainable adapters are FP32."
            )
        if self.use_grad_scaler:
            from torch.cuda.amp import GradScaler

            # Use higher init_scale for FP16 to prevent gradient underflow
            # Problem: Initial gradients in LoRA training are very small (1e-7 ~ 1e-8)
            # FP16 smallest normal: ~6e-5, so gradients < 6e-5 underflow to 0
            # Solution: Use higher init_scale for FP16 (2^20 = 1048576)
            # - 1e-7 × 2^20 = 0.105 (representable in FP16)
            # - 1e-8 × 2^20 = 0.01 (representable in FP16)
            init_scale = 2**20  # 1048576 (higher scale for SD/SDXL)

            self.grad_scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
            print(f"[Trainer] GradScaler enabled for {training_dtype} training")
            print(f"[Trainer]   Init scale: {init_scale} (2^{init_scale.bit_length()-1})")
            print(f"[Trainer]   Weight dtype: {weight_dtype}")
            print(f"[Trainer]   Training dtype: {training_dtype}")
            if hasattr(self, 'lora_dtype'):
                print(f"[Trainer]   LoRA dtype: {self.lora_dtype}")
        else:
            self.grad_scaler = None
            if self.training_dtype == torch.bfloat16:
                print(f"[Trainer] GradScaler disabled (BF16 has FP32-equivalent exponent range, no scaling needed)")
            else:
                print(f"[Trainer] GradScaler disabled (training_dtype={training_dtype})")

        # Prompt chunking settings (SD/SDXL only)
        self.prompt_chunking_mode = prompt_chunking_mode
        self.max_prompt_chunks = max_prompt_chunks

        # Regularization losses (to prevent overbaking)
        self.snr_regularization_loss = None
        self.energy_regularization_loss = None
        # train_config must be assigned BEFORE _load_*_components / _apply_lora
        # because Anima reads cpu_offload_checkpointing / fp8_base_dtype /
        # anima_lora_scope / etc. during those calls.
        self.config = dict(train_config) if train_config else {}

        # FP16 + a Block Swap fused path: refuse here, before the model load and
        # the latent/text caching that setup_optimizer sits behind. The calls in
        # the two _setup_fused_* methods stay as the backstop.
        if self.use_grad_scaler and self.blocks_to_swap > 0:
            _optimizer_type = str(self.config.get("optimizer", "adamw8bit")).lower()
            if self.num_optimizer_groups > 0:
                refuse_grad_scaler_under_fused_path(
                    self, _optimizer_type, "fused optimizer groups")
            elif _optimizer_type in FUSED_BACKWARD_OPTIMIZERS:
                refuse_grad_scaler_under_fused_path(
                    self, _optimizer_type, "fused backward pass")

        # Full-parameter save: embed the trained VAE into the single-file checkpoint.
        # Kept as Optional[bool]: None = per-arch default. Each full-FT save adapter
        # resolves it via api.param_defaults.resolve_bundle_vae(value, arch)
        # (BUNDLE_VAE_DEFAULTS_BY_ARCH: sd15/sdxl/deus True, others False).
        self.bundle_vae = self.config.get("bundle_vae", None)
        if self.bundle_vae is not None:
            self.bundle_vae = bool(self.bundle_vae)

        # MiniMax-H3's joint video+audio objective weight (SSoT:
        # api/param_defaults.TRAINING_DEFAULTS). Read from the run config rather
        # than added as a positional trainer argument because it is an
        # arch-specific knob, the same way fp8_base_dtype / *_lora_scope are;
        # every other architecture leaves it unread.
        from api.param_defaults import TRAINING_DEFAULTS as _TD_AUDIO
        self.audio_loss_weight = float(
            self.config.get("audio_loss_weight", _TD_AUDIO["audio_loss_weight"]))

        # Per-run gradient checkpointing toggle. Default True preserves the prior
        # unconditional behavior; set False to trade VRAM for speed. Gated at every
        # enable_gradient_checkpointing / gradient_checkpointing_enable call site via
        # the self.gradient_checkpointing guard.
        self.gradient_checkpointing = bool(self.config.get("gradient_checkpointing", True))

        # Opt-in torch.compile for DiT training. "off" (default) disables it;
        # any other value is a torch.compile mode string. Resolved/gated later
        # by _maybe_compile_transformer() (called once before the training loop).
        _tc = self.config.get("torch_compile", "off")
        self.torch_compile = str(_tc) if _tc not in (None, False) else "off"
        self.torch_compile_dynamic = self.config.get("torch_compile_dynamic", None)
        self._transformer_compiled = False

        # Legacy dtype for compatibility
        self.dtype = self.weight_dtype

        # Log prefix for subclass override
        self.log_prefix = "[Trainer]"

        # Log component learning rates
        print(f"{self.log_prefix} ===== Component Learning Rates =====")
        print(f"{self.log_prefix}   Base LR: {self.learning_rate}")
        print(f"{self.log_prefix}   U-Net LR: {self.unet_lr}")
        print(f"{self.log_prefix}   Text Encoder LR: {self.text_encoder_lr}")
        if hasattr(self, 'text_encoder_1_lr'):
            print(f"{self.log_prefix}   Text Encoder 1 LR: {self.text_encoder_1_lr}")
        if hasattr(self, 'text_encoder_2_lr'):
            print(f"{self.log_prefix}   Text Encoder 2 LR: {self.text_encoder_2_lr}")
        # Note: Vision Encoder LR is logged in train() when VE is actually loaded
        print(f"{self.log_prefix} ====================================")

        print(f"[Trainer] Precision settings:")
        print(f"  Weight dtype: {weight_dtype} ({self.weight_dtype})")
        print(f"  Training dtype: {training_dtype} ({self.training_dtype})")
        print(f"  Output dtype: {output_dtype} ({self.output_dtype})")
        print(f"  VAE dtype: {vae_dtype} ({self.vae_dtype})")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Loss calculation: Always FP32 for numerical stability")
        print(f"  Min-SNR gamma: {min_snr_gamma} ({'enabled' if min_snr_gamma > 0 else 'disabled'})")

        # Warn about FP32 training VRAM usage
        if self.training_dtype == torch.float32:
            print(f"[Trainer] WARNING: training_dtype=fp32 uses ~2x VRAM compared to fp16/bf16")
            print(f"[Trainer] WARNING: Consider using training_dtype=fp16 or bf16 for large models")

        # Optimize: disable autocast if training_dtype is fp32 (no benefit, only overhead)
        # autocast with dtype=fp32 does nothing but adds context manager overhead
        if self.training_dtype == torch.float32:
            self.mixed_precision = False
            if mixed_precision:
                print(f"[Trainer] Note: mixed_precision disabled (training_dtype=fp32, autocast has no effect)")

        # Initialize tensorboard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = self.output_dir / "tensorboard" / timestamp
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))

        print(f"{self.log_prefix} Initializing on {self.device}")
        print(f"{self.log_prefix} Tensorboard logs: {tensorboard_dir}")

        # Check if resuming from checkpoint
        checkpoint_to_load = None
        # Trainers that manage their own checkpoint format (ControlNet: directory
        # or lllite-adapter saves) load weights themselves after this __init__ and
        # set _loaded_checkpoint_path directly; skip the file-based base-model
        # resume detection here so it can't mis-load an adapter as a base model.
        if self.resume_from_checkpoint and not getattr(self, '_manages_own_resume', False):
            if self.resume_from_checkpoint.lower() == "latest":
                # Find latest checkpoint in output directory (single-file OR
                # sharded index; shard members are excluded as entries).
                # EMA snapshot checkpoints (a full, separately loadable
                # save under a "_ema"-suffixed run_name) are excluded so
                # resume can never silently pick up EMA-averaged weights
                # in place of the live training weights. Quarantined
                # partial-step saves are excluded for the same reason.
                # Vision Encoder sibling files (saved alongside EMA/quarantine
                # checkpoints as "..._vision_encoder_step_N.safetensors") do not
                # contain EMA_ENTRY_MARKER or QUARANTINE_ENTRY_MARKER -- those
                # markers require "_step_" to immediately follow the suffix,
                # but the VE filename inserts "vision_encoder_" first -- so they
                # are excluded by name here too.
                checkpoint_files = _list_checkpoint_entries(
                    self.output_dir,
                    exclude_substr=("vision_encoder", EMA_ENTRY_MARKER, QUARANTINE_ENTRY_MARKER),
                )
                if checkpoint_files:
                    # Get latest checkpoint by step number
                    def get_step(path):
                        return _checkpoint_step_from_name(path.name) or 0

                    latest_checkpoint = max(checkpoint_files, key=get_step)
                    checkpoint_to_load = str(latest_checkpoint)
                    print(f"{self.log_prefix} Found checkpoint to resume from: {checkpoint_to_load}")
            else:
                # Specific checkpoint path provided (treat as relative to output_dir first, then absolute)
                checkpoint_path_obj = self.output_dir / self.resume_from_checkpoint
                if checkpoint_path_obj.exists():
                    checkpoint_to_load = str(checkpoint_path_obj)
                    print(f"{self.log_prefix} Using specified checkpoint: {checkpoint_to_load}")
                else:
                    # Try as absolute path
                    checkpoint_path_obj = Path(self.resume_from_checkpoint)
                    if checkpoint_path_obj.exists():
                        checkpoint_to_load = str(checkpoint_path_obj)
                        print(f"{self.log_prefix} Using specified checkpoint (absolute path): {checkpoint_to_load}")

        if checkpoint_to_load and QUARANTINE_ENTRY_MARKER in Path(checkpoint_to_load).name:
            # Resume scanning ("latest") never selects this name (see
            # QUARANTINE_ENTRY_MARKER); reaching here means resume_from_checkpoint
            # named it explicitly. That is the only sanctioned way to load a
            # quarantined save, but it is otherwise silent -- warn so the choice
            # is visible in the run's warnings, not just inferred from the taint
            # never having a paired training-state/optimizer file.
            emit_training_warning(
                f"resume_from_checkpoint names a QUARANTINED checkpoint "
                f"({Path(checkpoint_to_load).name}): its weights are half-applied "
                f"from an interrupted fused optimizer step and it has no paired "
                f"optimizer/EMA state, so the optimizer restarts fresh. Verify its "
                f"loss/output before relying on this resume.",
                code="partial_step_quarantined_checkpoint_loaded",
                prefix=self.log_prefix,
            )

        if checkpoint_to_load:
            # Load checkpoint directly as base model (resume training)
            # Use fallback mechanism to handle corrupted checkpoints
            print(f"{self.log_prefix} Loading checkpoint as base model: {checkpoint_to_load}")
            try:
                self._load_checkpoint_as_base(checkpoint_to_load)
                print(f"{self.log_prefix} Successfully loaded checkpoint as base model")
                self._loaded_checkpoint_path = checkpoint_to_load
            except Exception as e:
                if is_checkpoint_corruption_error(e):
                    print(f"{self.log_prefix} WARNING: Checkpoint appears corrupted: {e}")
                    print(f"{self.log_prefix} Attempting to fall back to previous checkpoint...")

                    # Try fallback mechanism
                    success, loaded_path = self._try_load_checkpoint_with_fallback(checkpoint_to_load)

                    if success and loaded_path:
                        print(f"{self.log_prefix} Successfully loaded fallback checkpoint: {loaded_path}")
                        self._loaded_checkpoint_path = loaded_path
                    else:
                        print(f"{self.log_prefix} ERROR: All checkpoints failed to load")
                        print(f"{self.log_prefix} Checkpoint loading failed, but resume_from_checkpoint was specified.")
                        print(f"{self.log_prefix} Aborting training to prevent unintended behavior.")
                        raise RuntimeError(
                            f"Failed to load checkpoint '{checkpoint_to_load}' and all fallback checkpoints. "
                            f"Training aborted to prevent starting from base model when resume was requested. "
                            f"Error: {e}"
                        )
                else:
                    # Non-corruption error, don't fallback
                    print(f"{self.log_prefix} ERROR: Failed to load checkpoint: {e}")
                    print(f"{self.log_prefix} Checkpoint loading failed, but resume_from_checkpoint was specified.")
                    print(f"{self.log_prefix} Aborting training to prevent unintended behavior.")
                    raise RuntimeError(
                        f"Failed to load checkpoint '{checkpoint_to_load}'. "
                        f"Training aborted to prevent starting from base model when resume was requested. "
                        f"Error: {e}"
                    )
        else:
            # Load base model (new training)
            print(f"{self.log_prefix} Loading model from {model_path}")
            self._load_model_components()

        # Bind the per-architecture handler (composition, plan A.2). Constructed
        # HERE — at the end of __init__ — rather than earlier, because is_sdxl is
        # only finalized inside _load_sd_sdxl_components (:2877/:3048), which runs
        # during the load calls above; the other is_<arch> flags are set in
        # _load_model_components / _load_checkpoint_as_base (:1111-1120/:2439-2448).
        # By this point every flag is final, so get_arch_handler resolves the
        # correct handler. P1: the handler is a stub; the if-chains below still
        # drive all behavior and nothing calls handler methods yet.
        from core.training.arch import get_arch_handler
        self.arch = get_arch_handler(self)

    def _load_model_components(self):
        """Load model components (dispatcher for different model types)."""
        # Detect model type
        from core.model_loader import ModelLoader
        model_type = ModelLoader.detect_model_type(self.model_path)
        self.is_zimage = (model_type == "zimage")
        # DEUS support removed - architecture no longer maintained
        self.is_deus = False  # (model_type == "deus")
        self.is_flux2 = (model_type == "flux2")
        self.is_anima = (model_type == "anima")
        self.is_lens  = (model_type == "lens")
        self.is_ideogram4 = (model_type == "ideogram4")
        self.is_minit2i = (model_type == "minit2i")
        self.is_krea2 = (model_type == "krea2")
        self.is_ltx2 = (model_type == "ltx2")
        self.is_minimax_h3 = (model_type == "minimax_h3")
        self.is_acestep = (model_type == "acestep")
        self.is_sensenova = (model_type == "sensenova")
        self.is_sdxl = False

        # P3a: zimage + sd/sdxl loader BODIES moved to ops/ free functions. They
        # SET is_sdxl (sd/sdxl) / call attention setup during load — which runs
        # BEFORE self.arch is bound (:1115) — so the dispatcher CANNOT route via
        # self.arch here; it calls the shared ops functions directly. The arch
        # handlers (arch/sd15.py, arch/sdxl.py, arch/zimage.py) call the SAME
        # functions, so the body is defined exactly once. (lazy import: keeps the
        # ops module loading AFTER base_trainer is fully defined — ops imports
        # base_trainer._vramdiag at its top.)
        from core.training.ops import (
            sd_sdxl_ops, zimage_ops, anima_ops, lens_ops, ideogram4_ops,
            minit2i_ops, krea2_ops, flux2_ops, ltx2_ops, acestep_ops,
            minimax_h3_ops, sensenova_ops,
        )
        if self.is_sensenova:
            if self.blocks_to_swap != 0:
                raise ValueError("SenseNova training does not implement blocks_to_swap; set it to 0")
            sensenova_ops.load_components(self)
        elif self.is_ltx2:
            ltx2_ops.load_components(self)
        elif self.is_minimax_h3:
            minimax_h3_ops.load_components(self)
        elif self.is_acestep:
            acestep_ops.load_components(self)
        elif self.is_zimage:
            zimage_ops.load_components(self)
        # DEUS support removed
        # elif self.is_deus:
        #     self._load_deus_components()
        elif self.is_flux2:
            flux2_ops.load_components(self)
        elif self.is_anima:
            anima_ops.load_components(self)
        elif self.is_lens:
            lens_ops.load_components(self)
        elif self.is_ideogram4:
            ideogram4_ops.load_components(self)
        elif self.is_minit2i:
            minit2i_ops.load_components(self)
        elif self.is_krea2:
            krea2_ops.load_components(self)
        else:
            sd_sdxl_ops.load_components(self)

    # ============================================================
    # Anima (Cosmos-Predict2 DiT) component loading and training
    # ============================================================

    def setup_anima_block_swap(self):
        """Delegator (plan P3b): body lives in ``ops/anima_ops.setup_block_swap``.
        Kept on the trainer because mode subclasses (full_parameter_trainer /
        lora_trainer) call it LATE via ``hasattr(self, "setup_anima_block_swap")``
        after adapter setup; ``arch/anima.py`` calls the same ops function so the
        body is defined exactly once.
        """
        from core.training.ops import anima_ops
        return anima_ops.setup_block_swap(self)

    def setup_lens_block_swap(self):
        """Delegator (plan P3b): body lives in ``ops/lens_ops.setup_block_swap``.
        Kept on the trainer because mode subclasses call it LATE via
        ``hasattr(self, "setup_lens_block_swap")`` after adapter setup;
        ``arch/lens.py`` calls the same ops function (body defined once).
        """
        from core.training.ops import lens_ops
        return lens_ops.setup_block_swap(self)

    def setup_ideogram4_block_swap(self):
        """Delegator (plan P3b): body lives in
        ``ops/ideogram4_ops.setup_block_swap``. Kept on the trainer because mode
        subclasses call it LATE via ``hasattr(self, "setup_ideogram4_block_swap")``
        after adapter setup; ``arch/ideogram4.py`` calls the same ops function
        (body defined once).
        """
        from core.training.ops import ideogram4_ops
        return ideogram4_ops.setup_block_swap(self)

    def setup_krea2_block_swap(self):
        """Delegator (plan P3c): body lives in ``ops/krea2_ops.setup_block_swap``.
        Kept on the trainer because mode subclasses (full_parameter_trainer /
        lora_trainer) call it LATE via ``hasattr(self, "setup_krea2_block_swap")``
        after adapter setup; ``arch/krea2.py`` calls the same ops function so the
        body is defined exactly once.
        """
        from core.training.ops import krea2_ops
        return krea2_ops.setup_block_swap(self)

    def setup_ltx2_wrapper(self):
        """Delegator (plan AP3): body lives in ``ops/ltx2_ops.setup_wrapper``.
        Kept on the trainer because mode subclasses (full_parameter_trainer /
        lora_trainer) call it LATE via ``hasattr(self, "setup_ltx2_wrapper")``
        after adapter setup and BEFORE ``setup_ltx2_block_swap`` -- installs
        ``Ltx2BlockLoopWrapper`` (re-owned block loop) ONLY when an AP3 training
        feature (currently: TREAD) is enabled; a no-op otherwise (byte-identical
        default training path).
        """
        from core.training.ops import ltx2_ops
        return ltx2_ops.setup_wrapper(self)

    def setup_ltx2_block_swap(self):
        """Delegator (plan P5): body lives in ``ops/ltx2_ops.setup_block_swap``.
        Kept on the trainer because mode subclasses (full_parameter_trainer /
        lora_trainer) call it LATE via ``hasattr(self, "setup_ltx2_block_swap")``
        after adapter setup (and after ``setup_ltx2_wrapper``, so the conductor
        registers on the same ``transformer_blocks`` the wrapper's loop uses);
        ``arch/ltx2.py`` calls the same ops function so the body is defined
        exactly once.
        """
        from core.training.ops import ltx2_ops
        return ltx2_ops.setup_block_swap(self)

    def setup_acestep_block_swap(self):
        """Delegator (Phase 8a): body lives in ``ops/acestep_ops.setup_block_swap``.
        Kept on the trainer because mode subclasses (full_parameter_trainer /
        lora_trainer) call it LATE via ``hasattr(self, "setup_acestep_block_swap")``
        after adapter setup (mirrors ``setup_ltx2_block_swap``); ``arch/acestep.py``
        calls the same ops function so the body is defined exactly once.
        """
        from core.training.ops import acestep_ops
        return acestep_ops.setup_block_swap(self)

    def _setup_attention_backend_krea2(self, backend: str):
        """Delegator (plan P3c): body lives in
        ``ops/krea2_ops.setup_attention_backend``. Kept on the trainer because the
        (moved) krea2 loader body calls ``trainer._setup_attention_backend_krea2``;
        ``arch/krea2.py`` calls the same ops function (body defined once).
        """
        from core.training.ops import krea2_ops
        return krea2_ops.setup_attention_backend(self, backend)

    def _discover_default_tagger_dir(self) -> str:
        """Pick a usable tagger model dir under <repo>/tagger_models (newest checkpoint)."""
        from pathlib import Path as _P
        root = _P(__file__).resolve().parents[3] / "tagger_models"
        if not root.is_dir():
            raise FileNotFoundError(f"[REPA] tagger_models dir not found: {root}")
        cands = []
        for d in root.iterdir():
            if d.is_dir() and (d / "base_model_metadata.json").is_file():
                sts = list(d.glob("*.safetensors"))
                if sts:
                    cands.append((max(p.stat().st_mtime for p in sts), str(d)))
        if not cands:
            raise FileNotFoundError(f"[REPA] no usable tagger model under {root}")
        cands.sort()
        return cands[-1][1]

    def _setup_repa(self):
        """Set up REPA for MiniT2I when enabled (frozen encoder + trainable projector).

        The projector must exist before optimizer construction (the adapter adds its
        params to a group). The DiT tap is armed at the aligned block depth so the
        forward stashes the grad-connected image hidden state for the alignment loss.
        """
        self.repa_enable = bool(self.config.get("repa_enable", False))
        self._repa_moved = False
        if not self.repa_enable:
            return

        from core.training.repa import load_repa_encoder, RepaProjector

        source = str(self.config.get("repa_encoder_source", "tagger") or "tagger").strip().lower()
        tagger_dir = str(self.config.get("repa_tagger_model_dir", "") or "").strip()
        siglip2_repo = str(self.config.get("repa_siglip2_repo", "") or "").strip()
        if source == "tagger" and not tagger_dir:
            tagger_dir = self._discover_default_tagger_dir()
            print(f"{self.log_prefix} [REPA] auto-selected tagger dir: {tagger_dir}")

        repa_dtype = getattr(self, "training_dtype", None) or torch.bfloat16
        encoder, enc_dim, native = load_repa_encoder(
            source,
            tagger_model_dir=tagger_dir,
            siglip2_repo=siglip2_repo,
            dtype=repa_dtype,
            device=self.device,
        )
        self.repa_encoder = encoder
        self.repa_enc_dim = enc_dim

        res_override = int(self.config.get("repa_encoder_resolution", 0) or 0)
        self.repa_size = res_override if res_override > 0 else (native or 384)

        cfg = self.transformer.mmjit_config
        hidden = int(cfg.hidden_size)
        depth = int(cfg.depth_double)
        align = int(self.config.get("repa_align_depth", -1))
        if align < 0:
            align = max(0, depth // 3)
        align = max(0, min(align, depth - 1))
        self.repa_align_depth = align
        self.repa_weight = float(self.config.get("repa_weight", 0.5))
        self.repa_proj_lr_factor = float(self.config.get("repa_proj_lr_factor", 1.0))

        self.repa_projector = RepaProjector(hidden, enc_dim).to(device=self.device, dtype=repa_dtype)
        self.repa_projector.train()
        self.transformer.model.net._repa_tap_depth = align
        self._repa_moved = True

        # Resume: load a sibling projector saved next to the base checkpoint, if present
        # (dims must match the current encoder/variant; otherwise keep the fresh head).
        try:
            mp = str(getattr(self, "model_path", "") or "")
            if mp.endswith(".safetensors"):
                from core.training.adapters.minit2i_adapter import _repa_sidecar_path
                sib = _repa_sidecar_path(mp)
                if os.path.isfile(sib):
                    from safetensors.torch import load_file as _load_file
                    self.repa_projector.load_state_dict(_load_file(sib))
                    print(f"{self.log_prefix} [REPA] resumed projector from {sib}")
        except Exception as _e:
            print(f"{self.log_prefix} [REPA] projector resume skipped (using fresh head): {_e}")

        print(f"{self.log_prefix} [REPA] enabled: source={source}, enc_dim={enc_dim}, "
              f"size={self.repa_size}, align_depth={align}/{depth}, weight={self.repa_weight}, "
              f"proj_lr_factor={self.repa_proj_lr_factor}")

    def _ensure_repa_on_device(self):
        """Idempotently ensure the REPA encoder + projector live on the training device."""
        if getattr(self, "_repa_moved", False):
            return
        repa_dtype = getattr(self, "training_dtype", None) or torch.bfloat16
        if getattr(self, "repa_encoder", None) is not None:
            self.repa_encoder = self.repa_encoder.to(self.device)
        if getattr(self, "repa_projector", None) is not None:
            self.repa_projector = self.repa_projector.to(device=self.device, dtype=repa_dtype)
        self._repa_moved = True

    def _get_repa_pixels_for_item(self, item) -> Optional[torch.Tensor]:
        """Load + cache an S x S clean-image tensor [1,3,S,S] in [-1,1] for REPA.

        SigLIP2 normalization is mean=std=0.5 (i.e. [-1,1]); the encoder squishes to
        a fixed square (aspect handled by interpolating its features to the DiT grid).
        Returns None on load failure (the affected batch then skips REPA). A bounded
        in-memory LRU amortizes re-decoding the same images across the swap window.
        """
        try:
            S = int(getattr(self, "repa_size", 384) or 384)
            key = item.get("image_path")
            cache = getattr(self, "_repa_pix_cache", None)
            if cache is None:
                from collections import OrderedDict
                cache = self._repa_pix_cache = OrderedDict()
            if key and key in cache:
                cache.move_to_end(key)
                return cache[key]

            _b = item.get("_danbooru_image_bytes")
            if _b is not None:
                img = Image.open(BytesIO(_b))
            elif key:
                img = Image.open(key)
            else:
                return None
            img = flatten_to_rgb(img).resize((S, S), Image.BICUBIC)

            import numpy as _np
            arr = _np.asarray(img, dtype=_np.float32) / 255.0  # [S,S,3] in [0,1]
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,S,S]
            t = t * 2.0 - 1.0  # -> [-1,1]

            if key:
                cache[key] = t
                cache.move_to_end(key)
                while len(cache) > 4096:
                    cache.popitem(last=False)
            return t
        except Exception as _e:
            if not getattr(self, "_repa_pix_warned", False):
                print(f"{self.log_prefix} [REPA] clean-image load failed "
                      f"(REPA skipped for affected batches): {_e}")
                self._repa_pix_warned = True
            return None

    def _get_original_size_for_item(self, item) -> Tuple[int, int]:
        """Return the real source image (width, height) for SDXL micro-conditioning.

        Resolution order (cheapest first):
          1. Persistent per-path map (populated for free from DB dims at bucketing setup,
             see _seed_orig_size_from_db). Keyed by image_path so it survives per-epoch
             dataset reloads and bucket-dim overwrites.
          2. Image header read (no full decode), then memoized into the map.

        The map is NOT capped: crop augmentation reads every item's size every epoch
        (per-epoch re-bucketing), so an LRU cap would thrash and re-open headers on large
        datasets. Each entry is a tiny (w,h) tuple keyed by path.
        """
        m = getattr(self, "_orig_size_map", None)
        if m is None:
            m = self._orig_size_map = {}
        key = item.get("image_path")
        if key and key in m:
            return m[key]
        _b = item.get("_danbooru_image_bytes")
        if _b is not None:
            with Image.open(BytesIO(_b)) as im:
                wh = im.size  # (w, h) from header
        elif key:
            with Image.open(key) as im:
                wh = im.size
        else:
            raise ValueError("no image_path/bytes for original size")
        if key:
            m[key] = wh
        return wh

    def _seed_orig_size_from_db(self, item) -> None:
        """Populate the original-size map from the DB dims carried on the item, before
        bucketing overwrites item['width']/['height'] with the bucket size. Free (no header
        read) when the DB has dimensions; items without dims fall back to a lazy header read.
        """
        key = item.get("image_path")
        if not key:
            return
        ow, oh = item.get("width"), item.get("height")
        if ow and oh and ow > 0 and oh > 0:
            m = getattr(self, "_orig_size_map", None)
            if m is None:
                m = self._orig_size_map = {}
            if key not in m:
                m[key] = (int(ow), int(oh))

    # ---- Resolution curriculum helpers (opt-in low-res warmup, arch-agnostic) ----
    def _rc_scaled_resolutions(self, base_resolutions, scale):
        """Scale base resolutions for the warmup phase, snapping each to the /64 grid the
        bucket table (RESOLUTIONS_1024) is defined on. This reuses the existing bucket-fit
        logic (get_bucket_sizes) — a warmup base snapped to /64 produces the same
        well-formed (VAE-/8, DiT-patch-safe) buckets a normally-configured base resolution
        would, so anima's /16-pixel patch constraint and other archs' grids are respected
        exactly as they are for a hand-set base resolution. Min 64; distinct values kept."""
        out = []
        for r in base_resolutions:
            s = max(64, int(round(int(r) * float(scale) / 64.0)) * 64)
            if s not in out:
                out.append(s)
        return out

    def _rc_apply_bucketing_grid(self, bucket_manager, active_base_resolutions):
        """Point a BucketManager at a new base-resolution grid (regenerates bucket_lists).
        Does NOT touch bucket_manager.buckets — callers re-assign items separately."""
        from core.training.bucketing import get_bucket_sizes
        bucket_manager.base_resolutions = sorted(active_base_resolutions)
        bucket_manager.bucket_lists = {
            res: get_bucket_sizes(res, bucket_manager.divisibility)
            for res in bucket_manager.base_resolutions
        }

    def _rc_rebucket_items(self, all_items, bucket_manager, active_base_resolutions):
        """Rebuild bucket_manager.buckets for a new resolution phase, reading each item's
        ORIGINAL source size (via the persistent orig-size map, seeded before the first
        assignment) so a warmup->normal switch can grow dims back — assign_image_to_bucket
        overwrites item['width']/['height'] with the bucket size each time, so re-selecting
        from the current (already-bucketed) dims would be lossy."""
        self._rc_apply_bucketing_grid(bucket_manager, active_base_resolutions)
        bucket_manager.buckets = {}
        for item, dataset in all_items:
            try:
                ow, oh = self._get_original_size_for_item(item)
            except Exception:
                ow, oh = item.get("width", 1024), item.get("height", 1024)
            reference_images = item.get("reference_images", [])
            has_reference = len(reference_images) > 0
            _, image_info = bucket_manager.assign_image_to_bucket(
                image_path=item["image_path"],
                width=ow, height=oh,
                caption=item.get("caption", ""),
                dataset_unique_id=getattr(dataset, "unique_id", None),
                has_reference=has_reference,
                reference_images=reference_images if reference_images else None,
            )
            if item.get("_ve_reconstruction_mode"):
                image_info["_ve_reconstruction_mode"] = True
            item["width"] = image_info["bucket_width"]
            item["height"] = image_info["bucket_height"]

    def _rc_count_batches(self, all_items, live_manager, base_res, batch_size):
        """Count the batches ONE epoch would produce at `base_res`, without mutating the
        live BucketManager, item dims, or the global RNG stream (throwaway manager +
        pure select_bucket; originals from the persistent orig-size map). Used by the
        curriculum-aware total_steps correction: the warmup and normal partitions can
        have different batch counts (multi-res "max" fit thresholds shift under scaling;
        divisibility flooring can merge buckets at low res)."""
        from core.training.bucketing import BucketManager
        import random as _random
        tmp = BucketManager(
            base_resolutions=list(base_res),
            divisibility=live_manager.divisibility,
            strategy=live_manager.strategy,
            multi_resolution_mode=live_manager.multi_resolution_mode,
            separate_by_reference=live_manager.separate_by_reference,
        )
        _rng = _random.Random(0)  # isolated: never touches the global shuffle RNG
        counts = {}
        for item, dataset in all_items:
            try:
                ow, oh = self._get_original_size_for_item(item)
            except Exception:
                ow, oh = item.get("width", 1024), item.get("height", 1024)
            b = tmp.select_bucket(ow, oh, rng=_rng)
            key = (b, bool(item.get("reference_images"))) if tmp.separate_by_reference else b
            counts[key] = counts.get(key, 0) + 1
        return sum((n + batch_size - 1) // batch_size for n in counts.values())

    def _rc_refit_items(self, all_items, active_base_resolutions):
        """No-bucketing phase apply: fit each item into the active base-resolution AREA from
        its ORIGINAL size (mirrors the one-time no-bucketing fit, but re-runnable per phase
        and non-destructive since it reads the orig-size map). Within-area items keep their
        aspect, snapped to the arch pixel alignment (parity with the one-time fit)."""
        import math as _math
        # Align to the ARCH's pixel requirement, not just the VAE /8: patchified
        # DiTs (anima/lens/krea2/flux2/zimage/minit2i/ideogram4) require /16 and
        # assert on non-conforming dims (see ArchHandler.pixel_align). SD/SDXL = 8.
        align = self._arch_pixel_align()
        nb_base = max(int(r) for r in active_base_resolutions)
        nb_max_area = nb_base * nb_base
        for item, dataset in all_items:
            try:
                ow, oh = self._get_original_size_for_item(item)
            except Exception:
                ow, oh = int(item.get("width") or 0), int(item.get("height") or 0)
            if ow <= 0 or oh <= 0:
                item["width"], item["height"] = nb_base, nb_base
                continue
            if ow * oh > nb_max_area:
                sc = _math.sqrt(nb_max_area / float(ow * oh))
                item["width"] = max(align, int(ow * sc) // align * align)
                item["height"] = max(align, int(oh * sc) // align * align)
            else:
                # Within-area items keep their aspect but still snap to the arch
                # alignment (no-op for already-aligned / pre-resized datasets;
                # prevents a non-/16 original from tripping the DiT patchify assert).
                item["width"] = max(align, int(ow) // align * align)
                item["height"] = max(align, int(oh) // align * align)

    def _recompute_sdxl_micro_cond(self, item, bucket_w: int, bucket_h: int, strategy: str):
        """Deterministically recompute SDXL time_ids for an item from its real original
        size + bucket + strategy (used when encode_image did not run for this item, e.g.
        swap/cache paths). Returns (orig_h, orig_w, crop_top, crop_left, target_h, target_w).

        - resize: crop=(0,0). - crop: center-crop offset in resized space. - random_crop:
          offset is non-deterministic at consume time -> (0,0) approximation. On any
          failure, fall back to original=target=bucket (legacy-equivalent but correct target).
        """
        try:
            ow, oh = self._get_original_size_for_item(item)
            ct = cl = 0
            if strategy == "crop" and ow > 0 and oh > 0:
                scale = max(bucket_w / ow, bucket_h / oh)
                nw, nh = int(ow * scale), int(oh * scale)
                cl = max(0, (nw - bucket_w) // 2)
                ct = max(0, (nh - bucket_h) // 2)
            return (oh, ow, ct, cl, bucket_h, bucket_w)
        except Exception:
            return (bucket_h, bucket_w, 0, 0, bucket_h, bucket_w)

    def setup_minit2i_block_swap(self):
        """Delegator (plan P3c): body lives in
        ``ops/minit2i_ops.setup_block_swap``. Kept on the trainer because mode
        subclasses (full_parameter_trainer / lora_trainer) call it LATE via
        ``hasattr(self, "setup_minit2i_block_swap")`` after adapter setup;
        ``arch/minit2i.py`` calls the same ops function (body defined once).
        """
        from core.training.ops import minit2i_ops
        return minit2i_ops.setup_block_swap(self)

    # DEUS support removed - architecture no longer maintained
    # def _load_deus_components(self):
    #     """Load DEUS model components.
    #
    #     DEUS architecture:
    #     - SigLIP-2 text encoder (1152d output, variable sequence length)
    #     - U-Net with Transformer2DModel blocks
    #     - SDXL VAE (same scaling factor 0.13025)
    #     - DDPM epsilon prediction
    #
    #     Key differences from SDXL:
    #     - Single text encoder (SigLIP-2) vs dual CLIP
    #     - No pooled_embeddings
    #     - No time_ids / added_cond_kwargs
    #     """
    #     print(f"{self.log_prefix} Detected DEUS model")
    #     print(f"{self.log_prefix} Loading DEUS components from {self.model_path}")
    #
    #     from core.model_loader import ModelLoader
    #     from diffusers import DDPMScheduler
    #
    #     components = ModelLoader.load_deus_from_safetensors(
    #         file_path=self.model_path,
    #         device="cpu",
    #         torch_dtype=self.weight_dtype
    #     )
    #
    #     # Store components
    #     self.unet = components["unet"]
    #     self.vae = components["vae"]
    #     self.text_encoder = components["text_encoder"]
    #     self.tokenizer = components.get("tokenizer")
    #     self.processor = components.get("processor")
    #     self.scheduler = components["scheduler"]
    #     self.pipeline = components.get("pipeline")  # Keep reference for encode_prompt
    #
    #     # DEUS specific: no text_encoder_2, no transformer
    #     self.text_encoder_2 = None
    #     self.tokenizer_2 = None
    #     self.transformer = None
    #     self.transformer_original = None
    #
    #     # Create DDPM scheduler for training
    #     self.noise_scheduler = DDPMScheduler.from_config(self.scheduler.config)
    #
    #     # Save original scheduler for inference (sample generation)
    #     self.original_scheduler = self.scheduler
    #
    #     # Convert VAE to vae_dtype
    #     self.vae = self.vae.to(dtype=self.vae_dtype)
    #
    #     # Enable gradient checkpointing for U-Net (CRITICAL for VRAM reduction)
    #     if hasattr(self.unet, 'enable_gradient_checkpointing'):
    #         self.unet.enable_gradient_checkpointing()
    #         print(f"{self.log_prefix} Gradient checkpointing enabled for DEUS U-Net")
    #     else:
    #         print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for DEUS U-Net")
    #
    #     # Enable gradient checkpointing for Text Encoder
    #     if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
    #         self.text_encoder.gradient_checkpointing_enable()
    #         print(f"{self.log_prefix} Gradient checkpointing enabled for SigLIP-2 Text Encoder")
    #
    #     # Move VAE to device (always frozen during training)
    #     print(f"{self.log_prefix} Moving VAE to {self.device}...")
    #     self.vae.to(self.device)
    #
    #     # Move U-Net to device
    #     print(f"{self.log_prefix} Moving U-Net to {self.device}...")
    #     self.unet.to(self.device)
    #
    #     # Move Text Encoder to device
    #     print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
    #     self.text_encoder.to(self.device)
    #
    #     print(f"{self.log_prefix} DEUS model loaded successfully")
    #     print(f"{self.log_prefix} U-Net: {self.unet.__class__.__name__}")
    #     print(f"{self.log_prefix} Text Encoder: {self.text_encoder.__class__.__name__}")
    #     print(f"{self.log_prefix} Scheduler type: {self.scheduler.__class__.__name__}")
    #
    #     # Debug: Check for inf/nan in U-Net parameters
    #     unet_has_inf = False
    #     unet_has_nan = False
    #     for name, param in self.unet.named_parameters():
    #         if torch.isinf(param).any():
    #             print(f"{self.log_prefix} WARNING: U-Net param '{name}' contains inf!")
    #             unet_has_inf = True
    #         if torch.isnan(param).any():
    #             print(f"{self.log_prefix} WARNING: U-Net param '{name}' contains nan!")
    #             unet_has_nan = True
    #     if not unet_has_inf and not unet_has_nan:
    #         print(f"{self.log_prefix} U-Net parameters: No inf/nan detected")

    def _flux2_block_swap_h2d_args(self):
        """Delegator (plan P3c): body lives in
        ``ops/flux2_ops.block_swap_h2d_args``. Kept on the trainer because it has
        call sites in BOTH the (moved) flux2 loader body AND
        ``_load_checkpoint_as_base`` (which stays in the spine); ``arch/flux2.py``
        routes through the loader, so the body is defined exactly once.
        """
        from core.training.ops import flux2_ops
        return flux2_ops.block_swap_h2d_args(self)

    def _wire_flux2_block_swap_driver(self):
        """Delegator (plan P3c): body lives in
        ``ops/flux2_ops.wire_block_swap_driver``. Kept on the trainer because it
        has call sites in BOTH the (moved) flux2 loader body AND
        ``_load_checkpoint_as_base`` (which stays in the spine); body defined once.
        """
        from core.training.ops import flux2_ops
        return flux2_ops.wire_block_swap_driver(self)

    def _load_checkpoint_as_base(self, checkpoint_path: str):
        """
        Load checkpoint directly as base model (for resume training).

        Uses same VRAM-optimized loading pattern as _load_model_components():
        - Load to CPU first
        - Move to GPU in controlled manner
        - Enable gradient checkpointing

        This avoids loading base model + checkpoint (VRAM duplication).

        Args:
            checkpoint_path: Path to checkpoint file (.safetensors)
        """
        from core.model_loader import ModelLoader
        from diffusers import DDPMScheduler, EulerAncestralDiscreteScheduler

        # Detect model type from checkpoint
        model_type = ModelLoader.detect_model_type(checkpoint_path)
        self.is_zimage = (model_type == "zimage")
        # DEUS support removed - architecture no longer maintained
        self.is_deus = False  # (model_type == "deus")
        self.is_flux2 = (model_type == "flux2")
        self.is_anima = (model_type == "anima")
        self.is_lens  = (model_type == "lens")
        self.is_ideogram4 = (model_type == "ideogram4")
        self.is_minit2i = (model_type == "minit2i")
        self.is_krea2 = (model_type == "krea2")
        self.is_ltx2 = (model_type == "ltx2")
        self.is_minimax_h3 = (model_type == "minimax_h3")
        self.is_acestep = (model_type == "acestep")
        self.is_sensenova = (model_type == "sensenova")
        self.is_sdxl = False

        # DEUS support removed
        # if self.is_deus:
        #     print(f"{self.log_prefix} Loading DEUS checkpoint as base model")
        #     ...
        #     return

        if self.is_sensenova:
            if self.blocks_to_swap != 0:
                raise ValueError("SenseNova training does not implement blocks_to_swap; set it to 0")
            self.model_path = checkpoint_path
            from core.training.ops import sensenova_ops
            sensenova_ops.load_components(self)
            print(f"{self.log_prefix} SenseNova checkpoint loaded successfully as base model")
            return

        if self.is_flux2:
            print(f"{self.log_prefix} Loading FLUX.2 checkpoint as base model")

            # FLUX.2 checkpoints from training are loaded via ModelLoader
            from core.model_loader import ModelLoader

            components = ModelLoader.load_flux2_from_safetensors(
                file_path=checkpoint_path,
                device="cpu",
                torch_dtype=self.weight_dtype
            )

            # Store components
            self.transformer = components["transformer"]
            self.transformer_original = self.transformer  # FLUX.2 doesn't need wrapper
            self.vae = components["vae"]
            self.text_encoder = components["text_encoder"]
            self.tokenizer = components["tokenizer"]
            self.scheduler = components["scheduler"]

            # FLUX.2 specific: no text_encoder_2, no unet
            self.text_encoder_2 = None
            self.tokenizer_2 = None
            self.unet = None
            self.noise_scheduler = self.scheduler

            # Convert VAE to vae_dtype
            self.vae = self.vae.to(dtype=self.vae_dtype)

            # Enable gradient checkpointing for Transformer (CRITICAL for VRAM reduction)
            if not self.gradient_checkpointing:
                print(f"{self.log_prefix} Gradient checkpointing disabled by config (FLUX.2)")
            elif hasattr(self.transformer, 'enable_gradient_checkpointing'):
                self.transformer.enable_gradient_checkpointing()
                print(f"{self.log_prefix} Gradient checkpointing enabled for FLUX.2 Transformer")
            else:
                print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for FLUX.2 Transformer")

            # Enable gradient checkpointing for Text Encoder (Qwen3)
            if self.gradient_checkpointing and hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Qwen3 Text Encoder")

            # Setup attention backend if non-native (FLUX.2 checkpoint resume)
            if self.use_flash_attention:
                self._setup_attention_backend_flux2(self.attention_backend)

            # Freeze all base weights (full parameter training will unfreeze specific layers later)
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.transformer.requires_grad_(False)

            # Setup Block Swap if enabled (before moving to GPU)
            self.flux2_block_offloader = None  # FLUX.2 specific offloader
            self.flux2_transformer_wrapper = None  # Drives the offloader during forward

            if self.blocks_to_swap > 0:
                print(f"{self.log_prefix} Block Swap enabled for FLUX.2 training: {self.blocks_to_swap} blocks")
                print(f"{self.log_prefix} Using FluxBlockOffloader (dual-list architecture)")
                print(f"{self.log_prefix} Pinned memory: {self.use_pinned_memory}")

                # Policy gate: FLUX.2 training block swap requires H2D-only + frozen base
                # (LoRA) + gradient checkpointing. Raises on any unsupported combination.
                _h2d_args = self._flux2_block_swap_h2d_args()

                # Import FLUX.2 specific block offloader
                from core.memory_management import create_flux_block_offloader

                # Check if transformer has required attributes
                if not hasattr(self.transformer, 'transformer_blocks') or not hasattr(self.transformer, 'single_transformer_blocks'):
                    raise ValueError(
                        f"FLUX.2 Transformer must have 'transformer_blocks' and 'single_transformer_blocks' attributes for Block Swap. "
                        f"Found: {type(self.transformer)}"
                    )

                # Initialize FLUX.2 Block Offloader
                self.flux2_block_offloader = create_flux_block_offloader(
                    transformer=self.transformer,
                    blocks_to_swap=self.blocks_to_swap,
                    device=self.device,
                    target_dtype=self.training_dtype,
                    use_pinned_memory=self.use_pinned_memory,
                    supports_backward=True,  # Training mode
                    **_h2d_args,
                )

                # Prepare block devices (keep some on GPU, offload rest to CPU)
                self.flux2_block_offloader.prepare_block_devices_before_forward()

                # Wire the offloader into the forward (wrapper) and backward (hooks).
                self._wire_flux2_block_swap_driver()

                num_dual = len(self.transformer.transformer_blocks)
                num_single = len(self.transformer.single_transformer_blocks)
                print(f"{self.log_prefix}   FLUX.2 Block Swap initialized:")
                print(f"{self.log_prefix}   Dual stream blocks: {num_dual}")
                print(f"{self.log_prefix}   Single stream blocks: {num_single}")
                print(f"{self.log_prefix}   Total blocks: {num_dual + num_single}")
                print(f"{self.log_prefix}   Blocks to swap: {self.blocks_to_swap}")

                # Move VAE and Text Encoder to device (Transformer managed by block offloader)
                print(f"{self.log_prefix} Moving VAE to {self.device}...")
                self.vae.to(self.device)
                print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
                self.text_encoder.to(self.device)
            else:
                # No Block Swap: move everything to GPU
                print(f"{self.log_prefix} Moving VAE to {self.device}...")
                self.vae.to(self.device)

                print(f"{self.log_prefix} Moving Transformer to {self.device}...")
                self.transformer.to(self.device)

                print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
                self.text_encoder.to(self.device)

            print(f"{self.log_prefix} FLUX.2 checkpoint loaded successfully as base model")
            return

        elif self.is_zimage:
            print(f"{self.log_prefix} Loading Z-Image checkpoint as base model")

            # Z-Image checkpoints from training are saved with all components
            # We can load them as a complete model checkpoint
            from core.model_loader import ModelLoader

            # Detect format (ComfyUI or diffusers). Probe key NAMES only — for a
            # sharded save read them from the index (no tensor load); for a plain
            # single file use safe_open. ComfyUI format has keys like
            # "model.diffusion_model.x_embedder.proj.weight"; diffusers format has
            # keys like "transformer.x_embedder.proj.weight".
            from core.models.common.single_file_format import is_index_path
            if is_index_path(checkpoint_path):
                with open(checkpoint_path, encoding='utf-8') as _idxf:
                    _idx = json.load(_idxf)
                keys = list((_idx.get("weight_map") or {}).keys())
            else:
                from safetensors import safe_open
                with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
                    keys = list(f.keys())
            is_comfy_format = any(k.startswith("model.diffusion_model.") for k in keys)

            if is_comfy_format:
                # ComfyUI format checkpoint
                print(f"{self.log_prefix} Detected ComfyUI format Z-Image checkpoint")
                components = ModelLoader.load_zimage_from_comfy_safetensors(
                    file_path=checkpoint_path,
                    device="cpu",
                    torch_dtype=self.weight_dtype,
                    base_model_repo="Tongyi-MAI/Z-Image-Turbo"
                )
            else:
                # Diffusers format checkpoint (training checkpoint)
                # Extract checkpoint directory (assumes checkpoint is in training output dir with other components)
                checkpoint_dir = Path(checkpoint_path).parent

                # Check if other components exist in the same directory
                # Training saves: model_step_xxx.safetensors, vae/, text_encoder/, tokenizer/, scheduler/
                if (checkpoint_dir / "vae").exists():
                    # Load from directory structure
                    print(f"{self.log_prefix} Loading Z-Image from checkpoint directory: {checkpoint_dir}")
                    components = ModelLoader.load_zimage_from_diffusers(
                        model_path=str(checkpoint_dir),
                        device="cpu",
                        torch_dtype=self.weight_dtype
                    )

                    # Load transformer weights from checkpoint file. Read via the
                    # shared reader so a sharded index path loads transparently.
                    from core.models.common.single_file_format import read_state_dict
                    print(f"{self.log_prefix} Loading transformer weights from: {checkpoint_path}")
                    transformer_state_dict, _ = read_state_dict(checkpoint_path)
                    components["transformer"].load_state_dict(transformer_state_dict, strict=False)
                else:
                    # Single-file checkpoint with all components (full model save)
                    # This requires special handling - for now, raise error
                    raise RuntimeError(
                        f"Z-Image checkpoint resume from single-file format not yet supported. "
                        f"Please ensure checkpoint directory contains vae/, text_encoder/, tokenizer/, scheduler/ subdirectories. "
                        f"Checkpoint: {checkpoint_path}"
                    )

            # Store components
            self.transformer_original = components["transformer"]
            self.vae = components["vae"]
            self.text_encoder = components["text_encoder"]
            self.tokenizer = components["tokenizer"]
            self.scheduler = components["scheduler"]

            # Z-Image specific: no text_encoder_2, no unet
            self.text_encoder_2 = None
            self.tokenizer_2 = None
            self.unet = None
            self.noise_scheduler = self.scheduler

            # Save original scheduler for inference (sample generation)
            self.original_scheduler = self.scheduler

            # Convert VAE to vae_dtype
            self.vae = self.vae.to(dtype=self.vae_dtype)

            # Wrap transformer with BatchedZImageWrapperOptimized
            from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
            print(f"{self.log_prefix} Wrapping Z-Image Transformer with BatchedZImageWrapperOptimized")
            self.transformer = BatchedZImageWrapperOptimized(self.transformer_original)
            print(f"{self.log_prefix} Phase 2 optimization: Complete batched processing")

            # Setup attention backend if non-native (Z-Image checkpoint resume)
            if self.use_flash_attention:
                self._setup_attention_backend_zimage(self.attention_backend)

            # Enable gradient checkpointing for Transformer (CRITICAL for VRAM reduction)
            if not self.gradient_checkpointing:
                print(f"{self.log_prefix} Gradient checkpointing disabled by config (Z-Image)")
            elif hasattr(self.transformer, 'enable_gradient_checkpointing'):
                self.transformer.enable_gradient_checkpointing()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Z-Image Transformer")
            else:
                print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for Z-Image Transformer")

            # Enable gradient checkpointing for Text Encoder
            if self.gradient_checkpointing and hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder (Qwen3)")

            # Freeze all base weights (full parameter training will unfreeze specific layers later)
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.transformer.requires_grad_(False)

            # Move models to GPU (controlled, VRAM-optimized)
            # Text Encoder first (smallest)
            print(f"{self.log_prefix} Moving Text Encoder to {self.device}...")
            self.text_encoder.to(self.device)

            # Transformer second (Block Swap will be set up separately if enabled)
            if self.blocks_to_swap > 0:
                print(f"{self.log_prefix} Block Swap enabled: Transformer will stay on CPU during setup")
                # Block Swap setup will happen later in training setup
            else:
                print(f"{self.log_prefix} Moving Transformer to {self.device}...")
                self.transformer_original.to(self.device)

            # VAE stays on CPU (moved to GPU only during sample generation)
            print(f"{self.log_prefix} VAE remains on CPU (will move to GPU during sample generation)")

            print(f"{self.log_prefix} Z-Image checkpoint loaded successfully as base model")

        elif self.is_minit2i:
            # MiniT2I checkpoint resume: the saved single-file carries the full
            # transformer (+ optional FLAN-T5) and its mmjit_config/vae_type in
            # metadata. Reuse the normal MiniT2I loader by pointing model_path at
            # the checkpoint — it reconstructs the transformer, resolves the VAE
            # by vae_type, loads FLAN-T5, sets the scheduler, enables gradient
            # checkpointing, and moves to device. (Do NOT rebuild from the
            # scratch:minit2i sentinel — that would discard trained weights.)
            print(f"{self.log_prefix} Loading MiniT2I checkpoint as base model: {checkpoint_path}")
            self.model_path = checkpoint_path
            from core.training.ops import minit2i_ops
            minit2i_ops.load_components(self)
            print(f"{self.log_prefix} MiniT2I checkpoint loaded successfully as base model")

        elif self.is_krea2:
            # Krea 2 checkpoint resume: the sushiUI single-file (or any supported
            # layout) carries the transformer + variant/config metadata. Reuse the
            # normal loader by pointing model_path at the checkpoint — it rebuilds
            # the transformer, resolves the Qwen3-VL TE / Qwen-Image VAE, sets the
            # scheduler, enables gradient checkpointing, and moves to device.
            print(f"{self.log_prefix} Loading Krea 2 checkpoint as base model: {checkpoint_path}")
            self.model_path = checkpoint_path
            from core.training.ops import krea2_ops
            krea2_ops.load_components(self)
            print(f"{self.log_prefix} Krea 2 checkpoint loaded successfully as base model")

        else:
            # SD/SDXL checkpoint resume
            print(f"{self.log_prefix} Loading SD/SDXL checkpoint as base model")

            from safetensors import safe_open
            from core.model_loader import ModelLoader

            # Peek at keys + metadata only (reads header, not tensors)
            with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
                checkpoint_keys = list(f.keys())
                checkpoint_metadata = f.metadata() or {}

            # Detect if SDXL or SD1.5 based on state dict keys
            # SDXL has text_encoder_2 keys
            is_sdxl_model = any("text_model_2" in k or "conditioner.embedders.1" in k for k in checkpoint_keys)

            # Detect a SushiUI custom architecture (non-standard latent VAE / swapped
            # text encoder). Plain from_single_file cannot reconstruct those (conv
            # channel mismatch / missing TE), so route them through the same
            # sushi.*-aware reconstruction the inference loader uses.
            _cvt = (checkpoint_metadata.get("sushi.vae_type") or "").strip().lower()
            _ctt = (checkpoint_metadata.get("sushi.te_type") or "").strip().lower()
            is_custom_arch = (
                (_cvt and _cvt not in ("none", "sdxl"))
                or (_ctt and _ctt not in ("none", "clip"))
            )

            # Build the training noise scheduler honoring the checkpoint's prediction
            # config (fixes v-pred resume previously hardcoded to epsilon).
            pred_cfg = ModelLoader.detect_prediction_config(
                checkpoint_path, "sdxl" if is_sdxl_model else "sd15"
            )
            _pt_map = {"epsilon": "epsilon", "velocity": "v_prediction", "sample": "sample"}
            _sched_pred_type = _pt_map.get(pred_cfg.get("prediction_target", "epsilon"), "epsilon")
            print(f"{self.log_prefix} Resume prediction config: "
                  f"{pred_cfg.get('noise_process')} / {pred_cfg.get('prediction_target')} "
                  f"(scheduler prediction_type={_sched_pred_type}, source={pred_cfg.get('source')})")
            self.noise_scheduler = DDPMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                clip_sample=False,
                prediction_type=_sched_pred_type
            )

            # Defaults (overwritten below when a custom arch is reconstructed)
            self.sdxl_vae_type = "sdxl"
            self.sdxl_te_type = "none"

            if is_custom_arch:
                print(f"{self.log_prefix} Detected SushiUI custom-arch SDXL checkpoint "
                      f"(vae_type={_cvt or 'sdxl'}, te_type={_ctt or 'clip'}); "
                      f"reconstructing via inference loader")
                temp_pipeline = ModelLoader.reconstruct_sd_sdxl_pipeline(
                    checkpoint_path,
                    "sdxl" if is_sdxl_model else "sd15",
                    self.weight_dtype,
                    self.device,
                )
                arch = getattr(temp_pipeline, "_sushi_arch", {}) or {}

                # Extract raw components
                self.vae = temp_pipeline.vae
                self.text_encoder = temp_pipeline.text_encoder
                self.tokenizer = temp_pipeline.tokenizer
                self.unet = temp_pipeline.unet
                self.original_scheduler = temp_pipeline.scheduler
                if is_sdxl_model:
                    self.text_encoder_2 = temp_pipeline.text_encoder_2
                    self.tokenizer_2 = temp_pipeline.tokenizer_2
                else:
                    self.text_encoder_2 = None
                    self.tokenizer_2 = None

                # Rebuild custom-VAE state
                if arch.get("vae_type"):
                    self.sdxl_vae_type = arch["vae_type"]
                try:
                    self.vae_latent_channels = int(self.vae.config.latent_channels)
                except Exception:
                    self.vae_latent_channels = 4

                # Rebuild custom text-encoder state (swapped encoder + bridge adapters)
                if arch.get("te_type"):
                    self.sdxl_te_type = arch["te_type"]
                    self.te_custom = getattr(temp_pipeline, "_sushi_te", None)
                    self.te_tokenizer = getattr(temp_pipeline, "_sushi_te_tokenizer", None)
                    self.te_adapters = getattr(temp_pipeline, "_sushi_te_adapters", None)
                    self.te_dim = arch.get("te_dim") or 0
                    self.te_max_len = arch.get("te_max_len") or 256
                    self.te_hidden_layer = arch.get("te_hidden_layer")
                    if self.te_hidden_layer is None:
                        self.te_hidden_layer = -2
                    self.sdxl_te_train_encoder = bool(
                        self.config.get("sdxl_te_train_encoder", arch.get("te_embedded", False))
                    )
                    # Restore train/eval + requires_grad for continued training
                    if self.te_adapters is not None:
                        self.te_adapters.requires_grad_(True); self.te_adapters.train()
                    if self.te_custom is not None:
                        if self.sdxl_te_train_encoder:
                            self.te_custom.requires_grad_(True); self.te_custom.train()
                        else:
                            self.te_custom.requires_grad_(False); self.te_custom.eval()
                    print(f"{self.log_prefix} [SDXL custom] Reconstructed text encoder "
                          f"'{self.sdxl_te_type}' (dim={self.te_dim}, max_len={self.te_max_len}, "
                          f"layer={self.te_hidden_layer}, train_encoder={self.sdxl_te_train_encoder})")
            else:
                # Standard SD1.5 / SDXL: plain from_single_file (unchanged path)
                if is_sdxl_model:
                    print(f"{self.log_prefix} Detected SDXL checkpoint")
                    from diffusers import StableDiffusionXLPipeline

                    temp_pipeline = StableDiffusionXLPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=self.weight_dtype,
                        use_safetensors=True,
                        device_map=None,  # Load to CPU first
                    )
                else:
                    print(f"{self.log_prefix} Detected SD1.5 checkpoint")
                    from diffusers import StableDiffusionPipeline

                    temp_pipeline = StableDiffusionPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=self.weight_dtype,
                        use_safetensors=True,
                        device_map=None,  # Load to CPU first
                    )

                # Extract components
                self.vae = temp_pipeline.vae
                self.text_encoder = temp_pipeline.text_encoder
                self.tokenizer = temp_pipeline.tokenizer
                self.unet = temp_pipeline.unet

                # Save original scheduler for inference (sample generation)
                self.original_scheduler = temp_pipeline.scheduler

                # SDXL-specific components
                if is_sdxl_model:
                    self.text_encoder_2 = temp_pipeline.text_encoder_2
                    self.tokenizer_2 = temp_pipeline.tokenizer_2
                else:
                    self.text_encoder_2 = None
                    self.tokenizer_2 = None

            # Store SDXL flag
            self.is_sdxl = is_sdxl_model

            # No transformer for SD/SDXL
            self.transformer = None
            self.transformer_original = None

            # Clean up temporary pipeline
            del temp_pipeline
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Convert VAE to vae_dtype
            self.vae = self.vae.to(dtype=self.vae_dtype)

            # Setup attention backend if non-native (use_flash_attention is derived from it)
            if self.use_flash_attention:
                self._setup_attention_backend_sd_sdxl(self.attention_backend)

            # Enable gradient checkpointing for U-Net (CRITICAL for VRAM reduction)
            if not self.gradient_checkpointing:
                print(f"{self.log_prefix} Gradient checkpointing disabled by config (SD/SDXL)")
            elif hasattr(self.unet, 'enable_gradient_checkpointing'):
                self.unet.enable_gradient_checkpointing()
                print(f"{self.log_prefix} Gradient checkpointing enabled for U-Net")
            else:
                print(f"{self.log_prefix} WARNING: Gradient checkpointing not available for U-Net")

            # Enable gradient checkpointing for Text Encoders
            if self.gradient_checkpointing and hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
                print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder 1")

            if self.gradient_checkpointing and self.text_encoder_2 is not None:
                if hasattr(self.text_encoder_2, 'gradient_checkpointing_enable'):
                    self.text_encoder_2.gradient_checkpointing_enable()
                    print(f"{self.log_prefix} Gradient checkpointing enabled for Text Encoder 2")

            # Freeze VAE
            self.vae.requires_grad_(False)

            # Move models to GPU (controlled, VRAM-optimized)
            # Text Encoder 1 first (smallest)
            print(f"{self.log_prefix} Moving Text Encoder 1 to {self.device}...")
            self.text_encoder.to(self.device)

            # Text Encoder 2 (if SDXL)
            if self.text_encoder_2 is not None:
                print(f"{self.log_prefix} Moving Text Encoder 2 to {self.device}...")
                self.text_encoder_2.to(self.device)

            # U-Net second
            print(f"{self.log_prefix} Moving U-Net to {self.device}...")
            self.unet.to(self.device)

            # VAE stays on CPU (moved to GPU only during sample generation)
            print(f"{self.log_prefix} VAE remains on CPU (will move to GPU during sample generation)")

            print(f"{self.log_prefix} {'SDXL' if is_sdxl_model else 'SD1.5'} checkpoint loaded successfully as base model")

    def _persist_attention_impl(self):
        """Write the resolved ``attention_impl`` back into the run config YAML.

        Makes the choice reproducible across resumes: once a run has resolved to
        "conduit" (fresh) or "diffusers" (resume of a pre-migration config), the
        value is pinned in ``{output_dir}/{run_name}_config.yaml`` so subsequent
        resumes read it explicitly instead of re-deriving from the resume flag.

        Best-effort and non-fatal: silently returns if the config file is absent
        (e.g. programmatic trainer construction) or the expected structure is
        missing. Only rewrites when the stored value actually differs, to avoid
        churning the file on every start.
        """
        try:
            config_path = self.output_dir / f"{self.run_name}_config.yaml"
            if not config_path.exists():
                return
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            train = cfg["config"]["process"][0]["train"]
            if train.get("attention_impl") == self.attention_impl:
                return
            train["attention_impl"] = self.attention_impl
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            # log_prefix is assigned later in __init__; this method runs before that,
            # so use a safe fallback to avoid an AttributeError masking the real work.
            _lp = getattr(self, "log_prefix", "[Trainer]")
            print(f"{_lp} Persisted attention_impl='{self.attention_impl}' to run config")
        except Exception as e:
            _lp = getattr(self, "log_prefix", "[Trainer]")
            print(f"{_lp} [WARN] Could not persist attention_impl to run config: {e}")

    def _resolve_training_backend(self, backend: str) -> str:
        """Apply the TRAINING-mode capability guard to a backend string (R4).

        Runs ``resolve_backend(..., mode=TRAINING)`` so inference-only backends
        (sage — no backward kernel) are stripped to native regardless of the UI
        restriction. This is defense-in-depth: a hand-edited YAML/preset with
        ``attention_backend='sage'`` must never reach a diffusers
        ``set_attention_backend`` / per-module attr in training.

        A neutral probe tensor (head_dim=64, equal q/kv heads, no mask) is passed
        so ONLY the trainability guard is decisive here; the per-call head_dim /
        GQA / mask guards still apply at actual dispatch time inside the model
        forward. Returns the sage-stripped canonical backend key.
        """
        probe = torch.empty(1, 1, 1, 64)
        return resolve_backend(backend, AttentionMode.TRAINING, probe, probe)

    def _setup_attention_backend_zimage(self, backend: str):
        """Thin delegator: the body moved VERBATIM to
        ``ops/zimage_ops.setup_attention_backend`` (plan P3a). Kept on the trainer
        because it is called from multiple load sites — the moved
        ``zimage_ops.load_components`` body AND ``_load_checkpoint_as_base``
        (:2676) — all of which run BEFORE ``self.arch`` is bound (:1115), so they
        cannot route via the handler. ``arch/zimage.py`` calls the same free
        function, so the body is defined exactly once."""
        from core.training.ops import zimage_ops
        return zimage_ops.setup_attention_backend(self, backend)

    def _setup_attention_backend_sd_sdxl(self, backend: str):
        """Thin delegator: the body moved VERBATIM to
        ``ops/sd_sdxl_ops.setup_attention_backend`` (plan P3a). Kept on the trainer
        because it is called from multiple load sites — the moved
        ``sd_sdxl_ops.load_components`` body AND ``_load_checkpoint_as_base``
        (:2905) — all of which run BEFORE ``self.arch`` is bound (:1115), so they
        cannot route via the handler. ``arch/sd15.py`` / ``arch/sdxl.py`` call the
        same free function, so the body is defined exactly once."""
        from core.training.ops import sd_sdxl_ops
        return sd_sdxl_ops.setup_attention_backend(self, backend)

    def _setup_attention_backend_flux2(self, backend: str):
        """Delegator (plan P3c): body lives in
        ``ops/flux2_ops.setup_attention_backend``. Kept on the trainer because it
        has call sites in BOTH the (moved) flux2 loader body AND
        ``_load_checkpoint_as_base`` (P3b audit); ``arch/flux2.py`` calls the same
        ops function, so the body is defined exactly once.
        """
        from core.training.ops import flux2_ops
        return flux2_ops.setup_attention_backend(self, backend)

    def _setup_attention_backend_anima(self, backend: str):
        """Delegator (plan P3b): body lives in
        ``ops/anima_ops.setup_attention_backend``. Kept on the trainer because the
        (moved) anima loader body calls ``trainer._setup_attention_backend_anima``;
        ``arch/anima.py`` calls the same ops function (body defined once).
        """
        from core.training.ops import anima_ops
        return anima_ops.setup_attention_backend(self, backend)

    def _setup_attention_backend_lens(self, backend: str):
        """Delegator (plan P3b): body lives in
        ``ops/lens_ops.setup_attention_backend``. Kept on the trainer because the
        (moved) lens loader body calls ``trainer._setup_attention_backend_lens``;
        ``arch/lens.py`` calls the same ops function (body defined once).
        """
        from core.training.ops import lens_ops
        return lens_ops.setup_attention_backend(self, backend)

    def _setup_attention_backend_ideogram4(self, backend: str):
        """Delegator (plan P3b): body lives in
        ``ops/ideogram4_ops.setup_attention_backend``. Kept on the trainer because
        the (moved) ideogram4 loader body calls
        ``trainer._setup_attention_backend_ideogram4``; ``arch/ideogram4.py`` calls
        the same ops function (body defined once).
        """
        from core.training.ops import ideogram4_ops
        return ideogram4_ops.setup_attention_backend(self, backend)

    def _setup_attention_backend_minit2i(self, backend: str):
        """Delegator (plan P3c): body lives in
        ``ops/minit2i_ops.setup_attention_backend``. Kept on the trainer because
        the (moved) minit2i loader body calls
        ``trainer._setup_attention_backend_minit2i``; ``arch/minit2i.py`` calls the
        same ops function (body defined once).
        """
        from core.training.ops import minit2i_ops
        return minit2i_ops.setup_attention_backend(self, backend)

    # ============================================================
    # Abstract Methods (must be implemented by subclasses)
    # ============================================================

    @abstractmethod
    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        """
        Setup trainable parameters for the model.

        Returns:
            List of parameter groups for optimizer (each with 'params' and 'lr')
        """
        pass

    @abstractmethod
    def save_checkpoint(self, step: int, epoch: int):
        """
        Save training checkpoint.

        Args:
            step: Current training step
            epoch: Current epoch
        """
        pass

    def _save_vision_encoder_checkpoint(self, step: int, epoch: int):
        """
        Save Vision Encoder checkpoint as a separate safetensors file (if loaded and trained).

        The VE checkpoint is saved alongside the main checkpoint with the suffix
        '_vision_encoder_step_XXXXXX.safetensors', independent of the main model format.
        """
        ve_obj = getattr(self, 'vision_encoder', None)
        if ve_obj is None:
            return

        try:
            from safetensors.torch import save_file
            ve_path = self.output_dir / f"{self.run_name}_vision_encoder_step_{step:06d}.safetensors"
            ve_sd = ve_obj.state_dict_for_save()
            metadata = {
                "step": str(step),
                "epoch": str(epoch),
                "model_type": "siglip2_vision_encoder",
            }
            save_file(ve_sd, ve_path, metadata=metadata)
            print(f"{self.log_prefix} Saved Vision Encoder checkpoint: {ve_path}")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to save Vision Encoder checkpoint: {e}")

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint (must be implemented by subclass).

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Step number from checkpoint
        """
        raise NotImplementedError("load_checkpoint() must be implemented by subclass")

    def _compute_dataset_fingerprint(self, datasets: List[Any]) -> dict:
        """
        Compute a fingerprint of the dataset structure for change detection on resume.

        This fingerprint is used to detect if the dataset has changed between training sessions.
        If the dataset changes, the saved random_state (shuffle order) becomes invalid.

        IMPORTANT: Only image file information is included in the fingerprint.
        Caption changes do NOT invalidate the shuffle state because:
        - Captions don't affect the order of images in batches
        - Users may want to edit captions without losing training progress

        Args:
            datasets: List of dataset objects

        Returns:
            Dict containing:
                - dataset_ids: List of dataset unique_ids
                - total_item_count: Total number of items across all datasets
                - image_paths_hash: Hash of sorted image paths (to detect additions/removals)
        """
        import hashlib

        dataset_ids = []
        all_image_paths = []

        for dataset in datasets:
            dataset_ids.append(dataset.unique_id)
            for item in dataset.items:
                # Only include image_path - captions are intentionally excluded
                all_image_paths.append(item.get("image_path", ""))

        # Sort paths for consistent hashing (order within dataset matters, but we hash sorted for detection)
        # Actually, we want to detect if the SET of images changed, not their order
        sorted_paths = sorted(all_image_paths)
        paths_str = "\n".join(sorted_paths)
        paths_hash = hashlib.md5(paths_str.encode('utf-8')).hexdigest()

        return {
            "dataset_ids": dataset_ids,
            "total_item_count": len(all_image_paths),
            "image_paths_hash": paths_hash,
        }

    def _check_dataset_fingerprint_changed(self, saved_fingerprint: Optional[dict], current_fingerprint: dict) -> bool:
        """
        Check if the dataset fingerprint has changed since the checkpoint was saved.

        Args:
            saved_fingerprint: Fingerprint from saved training state (may be None for old checkpoints)
            current_fingerprint: Current dataset fingerprint

        Returns:
            True if dataset has changed (shuffle state should be invalidated)
        """
        if saved_fingerprint is None:
            # Old checkpoint without fingerprint - assume unchanged for backward compatibility
            print(f"{self.log_prefix} No dataset fingerprint in saved state (old checkpoint format)")
            return False

        # Check if any key component changed
        if saved_fingerprint.get("total_item_count") != current_fingerprint.get("total_item_count"):
            print(f"{self.log_prefix} Dataset item count changed: {saved_fingerprint.get('total_item_count')} -> {current_fingerprint.get('total_item_count')}")
            return True

        if saved_fingerprint.get("image_paths_hash") != current_fingerprint.get("image_paths_hash"):
            print(f"{self.log_prefix} Dataset image paths changed (hash mismatch)")
            return True

        if saved_fingerprint.get("dataset_ids") != current_fingerprint.get("dataset_ids"):
            print(f"{self.log_prefix} Dataset IDs changed: {saved_fingerprint.get('dataset_ids')} -> {current_fingerprint.get('dataset_ids')}")
            return True

        return False

    def _resolve_start_epoch(
        self,
        resume_training_state: Optional[dict],
        global_step: int,
        steps_per_epoch: int,
        multi_noise_timesteps: int = 1,
    ) -> int:
        """Epoch index the loop resumes at.

        Prefers the epoch recorded in the training state. Without one, derives it
        from ``global_step`` -- via the crop planner when crop augmentation makes
        the per-epoch step count variable, else by division.
        """
        if resume_training_state is not None and resume_training_state.get('epoch') is not None:
            return int(resume_training_state['epoch'])
        crop_planner = getattr(self, 'crop_planner', None)
        if getattr(self, '_crop_step_offsets', None) is not None and crop_planner is not None:
            return int(crop_planner.epoch_for_step(global_step, multi_noise_timesteps))
        return int(global_step) // max(1, int(steps_per_epoch))

    def _epoch_batch_position(self, batch_idx: int) -> int:
        """Epoch-absolute index of the NEXT batch, for the training state.

        ``batch_idx`` enumerates the batch list as sliced for this session, so on
        a mid-epoch resume the skipped prefix must be added back. Saving the raw
        index made every resume record a position relative to its own resume
        point, so the epoch's data cursor rewound on each restart; an epoch
        longer than one session then never completed and ``epoch`` never advanced
        (run 112 stayed at epoch 0 for ~70k steps with 954,880 batches/epoch).
        """
        return int(batch_idx) + 1 + int(getattr(self, '_epoch_batch_offset', 0) or 0)

    def save_training_state(self, step: int, epoch: int, batch_idx: int, multi_noise_timesteps: int = 1):
        """
        Save training state (epoch progress, batch index, random state) to JSON file.

        This is saved separately from the model checkpoint to keep checkpoint files lightweight.
        Enables mid-epoch resume without re-processing already trained batches.

        Args:
            step: Current global step
            epoch: Current epoch (0-indexed)
            batch_idx: Current batch index within epoch (next batch to process)
            multi_noise_timesteps: MNT value at checkpoint time (for MNT-change detection on resume)
        """
        import json
        import random

        # Use full run_name with zero-padded step (consistent with model checkpoint naming)
        state_file = self.output_dir / f"{self.run_name}_step_{step:06d}_state.json"

        state = {
            "global_step": step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "multi_noise_timesteps": multi_noise_timesteps,  # Save MNT for resume calculation
            "random_state": random.getstate(),  # Save Python random state for batch shuffle reproducibility
            # Dataset fingerprint for change detection on resume
            "dataset_fingerprint": getattr(self, '_dataset_fingerprint', None),
            # Batches-per-epoch: captures a batch_size-only change (not covered by the
            # dataset fingerprint) so the resume structure-change guard can detect it.
            # Read-side treats a missing key as "unchanged" for backward compatibility.
            "batches_per_epoch": getattr(self, '_batches_per_epoch', None),
            # Crop-plan fingerprint: a change in crop augmentation params (or num_epochs)
            # invalidates the saved shuffle/crop reproducibility -> fresh fallback on resume.
            "crop_plan_fingerprint": getattr(self, '_crop_plan_fingerprint', None),
        }

        with open(state_file, 'w') as f:
            # Convert random_state tuple to list for JSON serialization
            state_serializable = state.copy()
            random_state = state["random_state"]
            state_serializable["random_state"] = {
                "version": random_state[0],
                "state": list(random_state[1]),  # Convert tuple to list
                "gauss_next": random_state[2],
            }
            json.dump(state_serializable, f, indent=2)

        print(f"{self.log_prefix} Saved training state to {state_file.name}")

    def load_training_state(self, step: int) -> Optional[dict]:
        """
        Load training state from JSON file.

        Args:
            step: Step number to load state for

        Returns:
            Dict with keys: global_step, epoch, batch_idx, random_state, dataset_fingerprint
            None if state file not found
        """
        import json
        import random
        import re

        # Try new naming format first (consistent with model checkpoint)
        state_file = self.output_dir / f"{self.run_name}_step_{step:06d}_state.json"

        # Fallback to old naming format (short name, no leading zeros) for backward compatibility
        if not state_file.exists():
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)
                state_file_legacy = self.output_dir / f"{short_name}_step_{step}_state.json"
                if state_file_legacy.exists():
                    state_file = state_file_legacy
                    print(f"{self.log_prefix} Using legacy training state file: {state_file.name}")

        if not state_file.exists():
            print(f"{self.log_prefix} No training state file found: {state_file.name}")
            return None

        with open(state_file, 'r') as f:
            state = json.load(f)

        # Restore random_state from serialized format
        random_state_dict = state["random_state"]
        state["random_state"] = (
            random_state_dict["version"],
            tuple(random_state_dict["state"]),  # Convert list back to tuple
            random_state_dict["gauss_next"],
        )

        print(f"{self.log_prefix} Loaded training state: epoch={state['epoch']}, batch_idx={state['batch_idx']}")
        return state

    @staticmethod
    def _optimizer_state_param_count(state_dict: Dict[str, Any]) -> int:
        return sum(len(g.get("params", [])) for g in state_dict.get("param_groups", []) or [])

    def _fast_forward_lr_schedulers(self, global_step: int):
        """Advance EVERY scheduler to the resumed step.

        Under fused optimizer groups each optimizer has its own scheduler and
        the training loop steps them all; advancing only ``self.lr_scheduler``
        would resume groups 1..N-1 at schedule position 0.
        """
        for scheduler in all_lr_schedulers(self):
            if scheduler is None:
                continue
            self._fast_forward_one_lr_scheduler(scheduler, global_step)

    @staticmethod
    def _fast_forward_one_lr_scheduler(scheduler, global_step: int) -> None:
        """Move a fresh scheduler to ``global_step`` without needless replay."""
        from torch.optim.lr_scheduler import LambdaLR

        if isinstance(scheduler, LambdaLR):
            step = int(global_step)
            values = [
                base_lr * lr_lambda(step)
                for base_lr, lr_lambda in zip(
                    scheduler.base_lrs, scheduler.lr_lambdas
                )
            ]
            scheduler.last_epoch = step
            scheduler._step_count = step + 1
            scheduler._last_lr = values
            for group, value in zip(scheduler.optimizer.param_groups, values):
                group["lr"] = value
            return

        # ReLoRA's scheduler has restart history and cannot be positioned from
        # the final step alone.
        for _ in range(global_step):
            scheduler.step()

    def save_optimizer_state(self, step: int):
        """
        Save optimizer state dict to .pt file.

        Under fused optimizer groups all N optimizers are written, under
        ``_sushi_fused_group_states``. A single-optimizer run writes the plain
        ``state_dict()`` it always wrote, so its files are unchanged.

        A fused file read by a build that predates this key fails safely and
        quietly: ``optimizer.load_state_dict()`` raises ``KeyError`` on the
        wrapper dict, the pre-existing salvage path finds no usable prefix and
        prints "Partial optimizer load not applicable", and the run continues
        with fresh optimizer state rather than crashing.

        Args:
            step: Current global step
        """
        optimizers = all_optimizers(self)
        if not optimizers:
            return

        # Use full run_name with zero-padded step (consistent with model checkpoint naming)
        optimizer_file = self.output_dir / f"{self.run_name}_step_{step:06d}_optimizer.pt"

        # Tag the optimizer class so a resume that switches optimizer (e.g. bnb
        # AdamW8bit -> AdamW8bit_RingBuffer) can detect the source format and
        # convert the state instead of crashing.
        states = []
        for optimizer in optimizers:
            state = optimizer.state_dict()
            try:
                state["_sushi_opt_class"] = type(optimizer).__name__
            except Exception:
                pass
            states.append(state)

        if len(states) == 1:
            payload: Dict[str, Any] = states[0]
        else:
            payload = {
                FUSED_GROUP_STATES_KEY: states,
                "_sushi_opt_class": states[0].get("_sushi_opt_class"),
            }
        torch.save(payload, optimizer_file)
        suffix = "" if len(states) == 1 else f" ({len(states)} fused optimizer groups)"
        print(f"{self.log_prefix} Saved optimizer state to {optimizer_file.name}{suffix}")

    def load_optimizer_state(self, step: int) -> bool:
        """
        Load optimizer state dict from .pt file.

        Restores every fused optimizer group, re-slicing the saved states by
        global parameter order when ``num_optimizer_groups`` changed since the
        checkpoint.

        Args:
            step: Step number to load optimizer state for

        Returns:
            True if successfully loaded, False otherwise
        """
        import re

        optimizers = all_optimizers(self)
        if not optimizers:
            print(f"{self.log_prefix} WARNING: Cannot load optimizer state (optimizer not initialized)")
            return False

        # Try new naming format first (consistent with model checkpoint)
        optimizer_file = self.output_dir / f"{self.run_name}_step_{step:06d}_optimizer.pt"

        # Fallback to old naming format (short name, no leading zeros) for backward compatibility
        if not optimizer_file.exists():
            match = re.match(r'\d{8}_\d{6}_([a-f0-9]+)', self.run_name)
            if match:
                short_name = match.group(1)
                optimizer_file_legacy = self.output_dir / f"{short_name}_step_{step}_optimizer.pt"
                if optimizer_file_legacy.exists():
                    optimizer_file = optimizer_file_legacy
                    print(f"{self.log_prefix} Using legacy optimizer state file: {optimizer_file.name}")

        if not optimizer_file.exists():
            print(f"{self.log_prefix} No optimizer state file found: {optimizer_file.name}")
            print(f"{self.log_prefix} Starting with fresh optimizer state")
            return False

        try:
            payload = torch.load(optimizer_file, map_location='cpu')
        except Exception as e:
            print(f"{self.log_prefix} ERROR: Failed to load optimizer file: {e}")
            print(f"{self.log_prefix} Continuing with fresh optimizer state")
            return False

        saved_states, fused_save = self._split_saved_optimizer_states(payload)

        if len(optimizers) == 1 and len(saved_states) == 1:
            return self._load_one_optimizer_state(
                optimizers[0], saved_states[0], optimizer_file.name)

        live_counts = [sum(len(g["params"]) for g in optimizer.param_groups)
                       for optimizer in optimizers]
        saved_counts = [self._optimizer_state_param_count(state) for state in saved_states]

        if live_counts == saved_counts:
            results = [
                self._load_one_optimizer_state(
                    optimizer, state, f"{optimizer_file.name} [group {i}]")
                for i, (optimizer, state) in enumerate(zip(optimizers, saved_states))
            ]
            return all(results)

        is_pre_fix_partial = (
            not fused_save and len(saved_states) == 1
            and len(optimizers) > 1 and sum(saved_counts) < sum(live_counts)
        )
        if is_pre_fix_partial and saved_counts[0] == live_counts[0]:
            # Written before fused groups were saved at all: only optimizer 0's
            # moments exist on disk, and no remap can invent the rest. The
            # ``sum(saved) < sum(live)`` check keeps this from matching a
            # genuinely different, smaller optimizer file whose total happens
            # to equal one live group's size by coincidence.
            emit_training_warning(
                f"{optimizer_file.name} predates fused-optimizer-group state saving: it "
                f"holds only optimizer group 0 of {len(optimizers)}. That group's moments "
                f"resume; the other {len(optimizers) - 1} start fresh, because their state "
                f"was never written.",
                code="optimizer_state_partial_fused_resume",
                prefix=self.log_prefix,
            )
            return self._load_one_optimizer_state(
                optimizers[0], saved_states[0], optimizer_file.name)

        if sum(live_counts) == sum(saved_counts):
            # num_optimizer_groups changed (either direction, 0 included). Both
            # layouts partition the SAME flat parameter list in the same order,
            # so the saved per-group states re-slice onto the live ones exactly.
            print(f"{self.log_prefix} Optimizer state was saved as {len(saved_states)} "
                  f"group(s) and this run has {len(optimizers)}; re-slicing "
                  f"{sum(saved_counts)} parameters' state by global order")
            remapped = self._repartition_optimizer_states(saved_states, optimizers)
            results = [
                self._load_one_optimizer_state(
                    optimizer, state, f"{optimizer_file.name} [group {i}]")
                for i, (optimizer, state) in enumerate(zip(optimizers, remapped))
            ]
            return all(results)

        if is_pre_fix_partial:
            # saved_counts[0] != live_counts[0] here, or the branch above would
            # have matched: still a pre-fix file, just also resumed under a
            # different num_optimizer_groups than it was saved with.
            cause = ("that file may predate fused-optimizer-group state saving "
                      "(it holds one group of a run that had more).")
        elif len(optimizers) > 1 or len(saved_states) > 1:
            # A genuine size mismatch under grouped optimizers. Unlike a
            # single-optimizer resume (which salvages a common prefix via
            # _load_one_optimizer_state), this path cannot: the fused groups
            # are arbitrary chunks of the flat parameter list, not per-component
            # boundaries, so there is no structural check that a size change is
            # a trailing addition/removal rather than an unrelated parameter
            # set. Resetting everything is the safe choice, not evidence that
            # the parameter set changed.
            cause = ("the trainable parameter set may not have changed at all: "
                      "grouped-optimizer resumes reset every group on any size "
                      "change instead of keeping the common prefix a "
                      "single-optimizer resume would.")
        else:
            cause = "the trainable parameter set changed since that checkpoint."

        emit_training_warning(
            f"the optimizer state in {optimizer_file.name} covers {sum(saved_counts)} "
            f"parameter tensor(s) in {len(saved_states)} group(s) and this run has "
            f"{sum(live_counts)} in {len(optimizers)}; nothing was restored and every "
            f"moment starts fresh, because {cause}",
            code="optimizer_state_not_restored",
            prefix=self.log_prefix,
        )
        return False

    @staticmethod
    def _split_saved_optimizer_states(payload) -> Tuple[List[Dict[str, Any]], bool]:
        """``(per-group states, was written by the fused path)``."""
        if isinstance(payload, dict) and isinstance(payload.get(FUSED_GROUP_STATES_KEY), list):
            return list(payload[FUSED_GROUP_STATES_KEY]), True
        return [payload], False

    def _repartition_optimizer_states(
        self, saved_states: List[Dict[str, Any]], optimizers: List[Any]
    ) -> List[Dict[str, Any]]:
        """Re-slice saved per-group state onto a different number of optimizers.

        Each ``state`` is keyed by an index into ITS OWN optimizer's flat
        parameter order, and the fused split is a contiguous partition of the
        adapter's flat list, so concatenating by group order recovers a global
        index that both layouts agree on.
        """
        flat: Dict[int, Any] = {}
        offset = 0
        for state in saved_states:
            for key, value in (state.get("state") or {}).items():
                flat[offset + int(key)] = value
            offset += self._optimizer_state_param_count(state)

        tag = saved_states[0].get("_sushi_opt_class") if saved_states else None
        remapped = []
        cursor = 0
        for optimizer in optimizers:
            live = optimizer.state_dict()
            count = self._optimizer_state_param_count(live)
            state = {i: flat[cursor + i] for i in range(count) if (cursor + i) in flat}
            entry = {"state": state, "param_groups": live.get("param_groups", [])}
            if tag is not None:
                entry["_sushi_opt_class"] = tag
            remapped.append(entry)
            cursor += count
        return remapped

    def _load_one_optimizer_state(self, optimizer, optimizer_state, label: str) -> bool:
        """Load one saved state dict into one optimizer."""

        def move_tensors_to_device(obj, device):
            """Recursively move all tensors in nested dict/list to target device."""
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: move_tensors_to_device(v, device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_tensors_to_device(v, device) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(move_tensors_to_device(v, device) for v in obj)
            else:
                return obj

        try:
            # If the checkpoint was written by a compatible-but-different 8-bit
            # optimizer (e.g. bnb AdamW8bit -> AdamW8bit_RingBuffer), convert the
            # state (key remap + absmax copy; the quantization scheme is identical)
            # so momentum/variance carry over instead of crashing or resetting.
            _carry_step = 0
            try:
                from core.training.optimizers.optimizer_state_convert import (
                    maybe_convert_optimizer_state,
                )
                _converted, _carry_step = maybe_convert_optimizer_state(
                    optimizer_state, optimizer, log_prefix=self.log_prefix
                )
                if _converted is not None:
                    optimizer_state = _converted
            except Exception as _conv_err:
                print(f"{self.log_prefix} [OptConvert] conversion skipped: {_conv_err}")

            # Recursively move all optimizer state tensors to GPU
            # This is necessary for 8-bit optimizers that have CUDA-only buffers
            # (absmax_z, absmax1, absmax2, etc.) which must be on CUDA device
            print(f"{self.log_prefix} Moving optimizer state tensors to {self.device}...")
            optimizer_state = move_tensors_to_device(optimizer_state, self.device)

            # Attempt to load state dict with error handling
            try:
                optimizer.load_state_dict(optimizer_state)

                # IMPORTANT: After load_state_dict(), move all tensors in optimizer.state to GPU
                # load_state_dict() may create new tensor references, so we need to move again
                moved_count = 0
                for param_state in optimizer.state.values():
                    for key, value in param_state.items():
                        if isinstance(value, torch.Tensor) and not value.is_cuda:
                            param_state[key] = value.to(self.device)
                            moved_count += 1
                if moved_count > 0:
                    print(f"{self.log_prefix} Moved {moved_count} optimizer state tensors to {self.device}")

                # Carry the step counter across a cross-implementation conversion
                # so AdamW bias correction continues from the right step (bnb keeps
                # step per-param; Ring Buffer uses a global step_count).
                if _carry_step > 0 and hasattr(optimizer, "step_count"):
                    optimizer.step_count = _carry_step
                    print(f"{self.log_prefix} [OptConvert] carried step_count={_carry_step}")

                print(f"{self.log_prefix} Successfully loaded optimizer state from {label}")
                return True
            except Exception as e:
                # The usual cause here is a param-GROUP count change between runs —
                # specifically adding/removing the REPA projector group (which is
                # always appended LAST). Rather than reset the WHOLE optimizer
                # (losing transformer/TE momentum+variance), try a prefix-preserving
                # partial load: keep the overlapping leading groups' state and only
                # drop/skip the projector group's state. This makes turning REPA
                # off (or on) mid-training non-destructive for the model's optimizer.
                print(f"{self.log_prefix} WARNING: Failed to load optimizer state directly: {e}")
                try:
                    cur_sd = optimizer.state_dict()
                    cur_groups = cur_sd.get("param_groups", [])
                    saved_groups = optimizer_state.get("param_groups", [])
                    saved_state = optimizer_state.get("state", {})
                    n = min(len(cur_groups), len(saved_groups))
                    # Only safe when the overlapping leading groups have identical
                    # param counts (i.e. the sole difference is a trailing group
                    # added/removed). Otherwise this is a genuine incompatibility.
                    prefix_ok = n > 0 and all(
                        len(cur_groups[i]["params"]) == len(saved_groups[i]["params"])
                        for i in range(n)
                    )
                    if not prefix_ok:
                        raise RuntimeError("optimizer param groups are not prefix-compatible")
                    # Number of params in the overlapping prefix (saved 'state' is keyed
                    # by a flat param index in param_groups order; the trailing group's
                    # params have the highest indices).
                    overlap_params = sum(len(cur_groups[i]["params"]) for i in range(n))
                    filtered_state = {k: v for k, v in saved_state.items() if int(k) < overlap_params}
                    partial = {"state": filtered_state, "param_groups": cur_groups}
                    optimizer.load_state_dict(partial)
                    for param_state in optimizer.state.values():
                        for key, value in param_state.items():
                            if isinstance(value, torch.Tensor) and not value.is_cuda:
                                param_state[key] = value.to(self.device)
                    kept = "model groups preserved; REPA projector group reset" if len(cur_groups) < len(saved_groups) \
                        else "model groups preserved; new (REPA projector) group starts fresh"
                    print(f"{self.log_prefix} Partial optimizer state load OK ({kept})")
                    return True
                except Exception as e2:
                    print(f"{self.log_prefix} Partial optimizer load not applicable: {e2}")
                    print(f"{self.log_prefix} This can also happen if the optimizer type or trainable")
                    print(f"{self.log_prefix} parameters changed. Continuing with fresh optimizer state")
                    print(f"{self.log_prefix} (momentum/variance will be reset)")
                    return False
        except Exception as e:
            print(f"{self.log_prefix} ERROR: Failed to restore optimizer state from {label}: {e}")
            print(f"{self.log_prefix} Continuing with fresh optimizer state")
            return False

    def find_latest_checkpoint(self) -> Optional[Tuple[str, int]]:
        """
        Find the latest checkpoint in output directory.

        Returns:
            Tuple of (checkpoint_path, step) or None if no checkpoints exist
        """
        # Search for checkpoint entries (single-file or sharded index; shard
        # members excluded) with pattern: {run_name}_step_{step}.safetensors[.index.json]
        # EMA snapshot checkpoints and quarantined partial-step saves are
        # excluded (see EMA_ENTRY_MARKER / QUARANTINE_ENTRY_MARKER). Their
        # Vision Encoder siblings carry neither marker (see the "latest"
        # resume probe above for why) so "vision_encoder" is excluded too.
        checkpoint_files = _list_checkpoint_entries(
            self.output_dir,
            exclude_substr=("vision_encoder", EMA_ENTRY_MARKER, QUARANTINE_ENTRY_MARKER),
        )

        # Search for training state files with pattern: {run_name}_step_{step}_state.json
        state_files = list(self.output_dir.glob("*_step_*_state.json"))

        if not checkpoint_files and not state_files:
            print(f"{self.log_prefix} No checkpoints found in {self.output_dir}")
            return None

        # Helper to extract step number from filename (tolerates both forms)
        def get_step(path):
            return _checkpoint_step_from_name(path.name) or 0

        # Find latest step from both sources
        latest_checkpoint_step = 0
        latest_checkpoint_path = None
        latest_state_step = 0

        if checkpoint_files:
            latest_checkpoint_path = max(checkpoint_files, key=get_step)
            latest_checkpoint_step = get_step(latest_checkpoint_path)

        if state_files:
            latest_state_path = max(state_files, key=get_step)
            latest_state_step = get_step(latest_state_path)

        # Debug: Print all checkpoints
        print(f"{self.log_prefix} Found checkpoint files:")
        for ckpt in sorted(checkpoint_files, key=get_step):
            step_num = get_step(ckpt)
            print(f"{self.log_prefix}   - {ckpt.name} → step {step_num}")

        print(f"{self.log_prefix} Found training state files:")
        for state in sorted(state_files, key=get_step):
            step_num = get_step(state)
            print(f"{self.log_prefix}   - {state.name} → step {step_num}")

        # Use the latest step (state.json takes priority as it represents interrupted training)
        if latest_state_step > latest_checkpoint_step:
            print(f"{self.log_prefix} Latest state.json: step {latest_state_step}")
            print(f"{self.log_prefix} Latest safetensors: step {latest_checkpoint_step}")
            print(f"{self.log_prefix} WARNING: State file is newer than checkpoint - this should not happen")
            print(f"{self.log_prefix} Using checkpoint step {latest_checkpoint_step}")
            step = latest_checkpoint_step
        else:
            step = max(latest_checkpoint_step, latest_state_step)

        if latest_checkpoint_path is None:
            print(f"{self.log_prefix} ERROR: No safetensors checkpoint found")
            return None

        print(f"{self.log_prefix} Selected latest checkpoint: {latest_checkpoint_path.name} (step {step})")
        return (str(latest_checkpoint_path), step)

    def _get_sorted_checkpoints(self) -> List[Tuple[Path, int]]:
        """
        Get all checkpoints sorted by step number (descending, newest first).

        Returns:
            List of (checkpoint_path, step_number) tuples, sorted newest first.
            Empty list if no checkpoints exist.
        """
        # EMA snapshot checkpoints and quarantined partial-step saves are
        # excluded (see EMA_ENTRY_MARKER / QUARANTINE_ENTRY_MARKER) so they are
        # never offered as fallback/resume candidates -- a quarantined entry
        # being unreadable-by-scan must never be misread as corruption and
        # walked past (see is_checkpoint_corruption_error). Their Vision
        # Encoder siblings carry neither marker, so "vision_encoder" is
        # excluded as well (see the "latest" resume probe for why).
        checkpoint_files = _list_checkpoint_entries(
            self.output_dir,
            exclude_substr=("vision_encoder", EMA_ENTRY_MARKER, QUARANTINE_ENTRY_MARKER),
        )

        if not checkpoint_files:
            return []

        def get_step(path):
            return _checkpoint_step_from_name(path.name) or 0

        # Sort by step number descending (newest first)
        sorted_checkpoints = sorted(checkpoint_files, key=get_step, reverse=True)
        return [(ckpt, get_step(ckpt)) for ckpt in sorted_checkpoints]

    def _try_load_checkpoint_with_fallback(self, checkpoint_path: str) -> Tuple[bool, Optional[str]]:
        """
        Try to load a checkpoint, with fallback to previous checkpoints if corrupted.

        Args:
            checkpoint_path: Path to the checkpoint to load (or "latest" for auto-detection)

        Returns:
            Tuple of (success, loaded_checkpoint_path).
            If success is False, loaded_checkpoint_path is None.
        """
        # Get sorted list of all checkpoints
        sorted_checkpoints = self._get_sorted_checkpoints()

        if not sorted_checkpoints:
            print(f"{self.log_prefix} No checkpoints found for fallback")
            return (False, None)

        # If specific checkpoint was requested, find its index
        if checkpoint_path and checkpoint_path.lower() != "latest":
            checkpoint_path_obj = Path(checkpoint_path)
            start_idx = 0
            for i, (ckpt, step) in enumerate(sorted_checkpoints):
                if ckpt.name == checkpoint_path_obj.name or str(ckpt) == checkpoint_path:
                    start_idx = i
                    break
        else:
            # Start from the newest checkpoint
            start_idx = 0

        # Try loading checkpoints starting from the requested one
        for i in range(start_idx, len(sorted_checkpoints)):
            ckpt_path, ckpt_step = sorted_checkpoints[i]
            ckpt_path_str = str(ckpt_path)

            if i > start_idx:
                print(f"{self.log_prefix} Attempting fallback to previous checkpoint: {ckpt_path.name} (step {ckpt_step})")

            try:
                self._load_checkpoint_as_base(ckpt_path_str)
                if i > start_idx:
                    print(f"{self.log_prefix} Successfully loaded fallback checkpoint: {ckpt_path.name}")
                return (True, ckpt_path_str)
            except Exception as e:
                if is_checkpoint_corruption_error(e):
                    print(f"{self.log_prefix} WARNING: Checkpoint corrupted: {ckpt_path.name}")
                    print(f"{self.log_prefix}   Error: {e}")
                    if i + 1 < len(sorted_checkpoints):
                        print(f"{self.log_prefix}   Will try previous checkpoint...")
                        continue
                    else:
                        print(f"{self.log_prefix} ERROR: No more checkpoints to try")
                        return (False, None)
                else:
                    # Non-corruption error, don't fallback
                    print(f"{self.log_prefix} ERROR: Failed to load checkpoint (non-corruption): {e}")
                    raise

        print(f"{self.log_prefix} ERROR: All checkpoints failed to load")
        return (False, None)

    def _safe_unlink(self, path, attempts: int = 5, delay: float = 0.6) -> bool:
        """Delete a file, tolerating transient Windows locks (antivirus / indexer
        holding a just-written file → PermissionError [Errno 13]). Retries a few
        times with backoff; a final failure is logged and swallowed (a leftover old
        checkpoint is harmless — cleanup retries next time). Never raises."""
        import time as _time
        for i in range(attempts):
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return True
            except (PermissionError, OSError) as e:
                if i < attempts - 1:
                    _time.sleep(delay * (i + 1))
                    continue
                print(f"{self.log_prefix} WARNING: could not delete {path.name} ({e}); leaving it (non-fatal)")
                return False
        return False

    def _cleanup_old_checkpoints(self, max_step_saves_to_keep: int):
        """
        Delete old checkpoints, keeping only the most recent N checkpoints.

        Args:
            max_step_saves_to_keep: Maximum number of checkpoints to keep (0 = keep all)
        """
        if max_step_saves_to_keep <= 0:
            return

        # Find main checkpoint entries only (single-file or sharded index; shard
        # members are grouped under their index, VE checkpoints excluded to avoid
        # double-counting; EMA snapshot checkpoints excluded -- they are rotated
        # separately below, paired 1:1 with their live-weight counterpart).
        # Quarantined partial-step saves are excluded from rotation entirely: a
        # failure artefact that a human has not yet looked at must not be
        # deleted by the same policy that prunes ordinary successful saves.
        checkpoint_files = _list_checkpoint_entries(
            self.output_dir, exclude_substr=("vision_encoder", EMA_ENTRY_MARKER, QUARANTINE_ENTRY_MARKER)
        )
        if len(checkpoint_files) <= max_step_saves_to_keep:
            return

        # Sort by step number
        def get_step(path):
            return _checkpoint_step_from_name(path.name) or 0

        checkpoint_files.sort(key=get_step, reverse=True)

        # Delete old checkpoints
        checkpoints_to_delete = checkpoint_files[max_step_saves_to_keep:]
        for checkpoint_path in checkpoints_to_delete:
            step_num = get_step(checkpoint_path)
            # Also delete associated _optimizer.pt file and _state.json file.
            # Aux base is {short_name}_step_{step} for both single-file and
            # sharded-index entries.
            aux_base = _checkpoint_aux_base(checkpoint_path)
            optimizer_pt_path = checkpoint_path.parent / f"{aux_base}_optimizer.pt"
            state_json_path = checkpoint_path.parent / f"{aux_base}_state.json"

            # A sharded save is deleted as a unit: index.json + every shard file.
            member_files = _checkpoint_member_files(checkpoint_path)
            print(f"{self.log_prefix} Deleting old checkpoint: {checkpoint_path.name}"
                  + (f" (+{len(member_files) - 1} shard file(s))" if len(member_files) > 1 else ""))
            for member in member_files:
                self._safe_unlink(member)

            if optimizer_pt_path.exists():
                print(f"{self.log_prefix} Deleting old optimizer state: {optimizer_pt_path.name}")
                self._safe_unlink(optimizer_pt_path)

            if state_json_path.exists():
                print(f"{self.log_prefix} Deleting old training state: {state_json_path.name}")
                self._safe_unlink(state_json_path)

            # Also delete VE checkpoint for this step if it exists
            ve_pattern = f"*_vision_encoder_step_{step_num:06d}.safetensors"
            for ve_file in checkpoint_path.parent.glob(ve_pattern):
                print(f"{self.log_prefix} Deleting old VE checkpoint: {ve_file.name}")
                self._safe_unlink(ve_file)

            # Also delete the paired EMA snapshot checkpoint for this step, if
            # any (use_ema writes it under run_name + EMA_RUN_NAME_SUFFIX, so
            # it is never itself picked up by the entry listing above -- prune
            # it here in lockstep with its live counterpart instead of letting
            # it accumulate indefinitely).
            ema_run_name = f"{self.run_name}{EMA_RUN_NAME_SUFFIX}"
            ema_index_path = checkpoint_path.parent / f"{ema_run_name}_step_{step_num:06d}.safetensors.index.json"
            if ema_index_path.exists():
                for member in _checkpoint_member_files(ema_index_path):
                    print(f"{self.log_prefix} Deleting old EMA checkpoint: {member.name}")
                    self._safe_unlink(member)
            else:
                for ema_file in checkpoint_path.parent.glob(f"{ema_run_name}_step_{step_num:06d}*.safetensors"):
                    print(f"{self.log_prefix} Deleting old EMA checkpoint: {ema_file.name}")
                    self._safe_unlink(ema_file)

    # ============================================================
    # Optimizer Setup
    # ============================================================

    def _build_component_lr_list(self):
        """
        Build a (component_lrs, component_names) pair matching the actual optimizer
        param group order created by setup_trainable_parameters() + VE append.

        Re-derived from trainer attributes, so it only describes the U-Net-based
        architectures plus the SenseNova / ControlNet / VE special cases: it is
        EMPTY on every DiT architecture, and can be non-empty yet misaligned
        (train_text_encoder on Flux2/MiniT2I/Z-Image yields just ``TE1`` while
        group 0 is the transformer). The resume path prefers the snapshot
        ``_record_configured_group_lrs`` takes off the adapter's own groups and
        uses this only when its length matches the live groups.

        Group ordering:
          - UNet (if train_unet)
          - TE1 (if train_text_encoder and text_encoder is not None)
          - TE2 (if train_text_encoder and is_sdxl and text_encoder_2 is not None)
          - VE  (if _train_vision_encoder and vision_encoder is not None)

        Returns:
            Tuple[List[float], List[str]]: (lrs, names) matching optimizer group indices
        """
        from core.training.adapters.base_adapter import resolve_component_lr

        lrs = []
        names = []

        if getattr(self, 'train_unet', True) and getattr(self, 'unet', None) is not None:
            lrs.append(resolve_component_lr(self, 'unet_lr', label="U-Net"))
            names.append("U-Net")

        # SenseNova's two groups both live inside `transformer` (`unet` is None
        # and `text_encoder` is None), so neither the U-Net entry above nor the
        # TE1 entry below fires and a resume would reset both to learning_rate.
        # Order mirrors SenseNovaLoRAAdapter.setup_trainable_parameters.
        if getattr(self, 'is_sensenova', False):
            if getattr(self, 'train_unet', True):
                lrs.append(resolve_component_lr(
                    self, 'unet_lr', label="SenseNova generation branch"))
                names.append("MoT-Generation")
            if getattr(self, 'train_text_encoder', False):
                # Same resolver the adapter's group uses, so this list (which the
                # resume re-assert writes back) cannot drift from what trains.
                lrs.append(resolve_component_lr(
                    self, 'text_encoder_1_lr', 'text_encoder_lr', 'unet_lr',
                    label="SenseNova understanding branch"))
                names.append("MoT-Understanding")

        # ControlNetTrainer sets train_unet=False (it never trains the base UNet)
        # but still creates a single optimizer group over ITS OWN module at
        # unet_lr (see controlnet_sd15_adapter.py / controlnet_sdxl_adapter.py
        # setup_trainable_parameters -> {"params": ..., "lr": self.trainer.unet_lr}).
        # Without this entry the list stays empty for ControlNet runs and the
        # resume LR-remap in train() falls through to its `else: new_lr =
        # self.learning_rate` branch, silently overwriting the intended unet_lr.
        if getattr(self, 'controlnet', None) is not None:
            lrs.append(resolve_component_lr(self, 'unet_lr', label="ControlNet"))
            names.append("ControlNet")

        if getattr(self, 'train_text_encoder', False):
            if getattr(self, 'text_encoder', None) is not None:
                lrs.append(resolve_component_lr(
                    self, 'text_encoder_1_lr', 'text_encoder_lr', label="TE1"))
                names.append("TE1")
            if getattr(self, 'is_sdxl', False) and getattr(self, 'text_encoder_2', None) is not None:
                lrs.append(resolve_component_lr(self, 'text_encoder_2_lr', label="TE2"))
                names.append("TE2")

        if getattr(self, '_train_vision_encoder', False) and getattr(self, 'vision_encoder', None) is not None:
            ve_lr = resolve_component_lr(self, '_vision_encoder_lr', 'text_encoder_lr',
                                         label="vision encoder")
            lrs.append(ve_lr)
            names.append("VisionEncoder")

        return lrs, names

    # Recorded at the end of setup_optimizer; None means "never recorded".
    _configured_group_lrs = None
    _configured_group_names = None

    def _record_configured_group_lrs(self, requested_group_lrs=None):
        """Snapshot the BASE learning rate of every optimizer param group.

        The adapter owns both the group order and the per-component factors
        (``anima_*_lr_factor``, ``lens_*_lr_factor``, ``minit2i_lr_factor``, the
        REPA projector, the SDXL custom-TE bridge), so each group's own LR is
        the only description of the run's rates that cannot drift from what
        trains. ``_build_component_lr_list`` re-derives one from trainer
        attributes instead, and is EMPTY on every DiT architecture
        (``self.unet is None``), which made a resume broadcast
        ``learning_rate`` over every group.

        Base, not current: the scheduler already exists by the time this runs
        and a warmup lambda has scaled ``group['lr']`` (to 0.0 at step 0).
        """
        self._configured_group_lrs = None
        self._configured_group_names = None

        # Every fused group's groups, not just optimizers[0]'s: the resume writes
        # back by index over the same flattened list.
        groups = [g for optimizer in all_optimizers(self)
                  for g in list(getattr(optimizer, "param_groups", []) or [])]
        if not groups:
            return

        fused = getattr(self, "fused_optimizer_groups", None)
        flattened = bool(fused and getattr(fused, "optimizers", None))
        requested = list(requested_group_lrs or [])

        if (not flattened and len(requested) == len(groups)
                and all(v is not None for v in requested)):
            lrs = [float(v) for v in requested]
        else:
            # A fused path rebuilt the optimizer: read each live group's base.
            lrs = []
            for group in groups:
                base = group.get("initial_lr", group.get("lr"))
                if base is None:
                    return
                lrs.append(float(base))

        self._configured_group_lrs = lrs
        self._configured_group_names = self._name_configured_groups(groups, lrs)

    def _name_configured_groups(self, groups, lrs):
        """Per-group labels for the log lines, best available source first."""
        names = [g.get("name") for g in groups]
        if all(names):
            return [str(n) for n in names]
        try:
            legacy_lrs, legacy_names = self._build_component_lr_list()
        except Exception:
            legacy_lrs, legacy_names = [], []
        if (len(legacy_lrs) == len(groups)
                and all(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
                        for a, b in zip(legacy_lrs, lrs))):
            return list(legacy_names)
        return [str(g.get("name") or f"group{i}") for i, g in enumerate(groups)]

    def _configured_component_lr_description(self, n_groups):
        """``(lrs, names, source)`` for ``n_groups`` groups, or ``([], [], reason)``.

        Never returns a description that does not correspond index-for-index to
        the live param groups: both consumers write BY INDEX, so a list of the
        wrong length -- or a scalar broadcast over the groups -- assigns some
        component another component's rate.
        """
        snapshot = getattr(self, "_configured_group_lrs", None)
        if snapshot and len(snapshot) == n_groups:
            names = list(getattr(self, "_configured_group_names", None) or [])
            return list(snapshot), names, "the optimizer's own param groups"

        try:
            lrs, names = self._build_component_lr_list()
        except Exception:
            lrs, names = [], []
        if lrs and len(lrs) == n_groups:
            return list(lrs), list(names), "_build_component_lr_list"

        if n_groups == 1:
            # Unambiguous: one group, one configured rate, no index to get wrong.
            return [float(self.learning_rate)], ["group0"], "the run's learning_rate"

        return [], [], (
            f"no per-group description is available: the optimizer snapshot holds "
            f"{len(snapshot or [])} rate(s) and _build_component_lr_list describes "
            f"{len(lrs)} ({names}), against {n_groups} live param group(s)"
        )

    def _report_effective_component_lrs(self, requested_group_lrs=None):
        """The rate each optimizer group ends up training at, and what changed it.

        Call this LAST in ``setup_optimizer``: ``_setup_fused_optimizer_groups``
        flattens every param group into one list and rebuilds N optimizers at
        ``self.learning_rate``, so anything checked before it describes an
        optimizer that is about to be discarded.

        ``requested_group_lrs`` is the pre-fused ``[group['lr'], ...]`` the
        adapter built; passing it is what makes that flattening visible.

        The index-wise comparison against ``_build_component_lr_list`` is a
        consistency check between two producers of the same description, not an
        independent measurement -- both read the same attributes with the same
        precedence. It cannot run at all where that list is empty (every DiT
        architecture: ``self.unet`` is None there and only the SenseNova /
        ControlNet / VE branches fill it), so it says so rather than looking
        like a check that passed.
        """
        optimizer = getattr(self, 'optimizer', None)
        if optimizer is None:
            return
        fused = getattr(self, 'fused_optimizer_groups', None)
        fused_optimizers = list(getattr(fused, 'optimizers', []) or []) if fused else []
        groups = [g for o in (fused_optimizers or [optimizer]) for g in o.param_groups]
        # BASE rates, not the schedule-scaled current ones: the scheduler is
        # built before this runs, and a warmup lambda makes group['lr'] 0.0 at
        # step 0 -- which read as both a mismatch and a dead group.
        effective = [g.get('initial_lr', g.get('lr')) for g in groups]

        if fused_optimizers and requested_group_lrs and len(set(requested_group_lrs)) > 1:
            emit_training_warning(
                f"num_optimizer_groups={self.num_optimizer_groups} rebuilt the optimizer "
                f"from a flat parameter list at the run's base learning rate, so the "
                f"per-component rates the adapter set ({requested_group_lrs}) are not "
                f"what trains: every one of the {len(groups)} group(s) now runs at "
                f"{sorted(set(effective))}. Set num_optimizer_groups=0 to keep "
                f"per-component learning rates.",
                code="component_lr_flattened",
                prefix=self.log_prefix,
            )

        component_names: List[str] = []
        aligned = False
        try:
            component_lrs, component_names = self._build_component_lr_list()
            aligned = len(component_lrs) == len(groups)
            for i, group in enumerate(groups):
                if not aligned or effective[i] is None:
                    continue
                if not math.isclose(float(effective[i]), float(component_lrs[i]),
                                    rel_tol=1e-9, abs_tol=0.0):
                    emit_training_warning(
                        f"{component_names[i]} trains at lr={effective[i]!r}, but the "
                        f"configured learning rate for it is {component_lrs[i]!r}. The "
                        f"optimizer group wins until the next resume, which rewrites it "
                        f"to the configured value.",
                        code="component_lr_mismatch",
                        prefix=self.log_prefix,
                    )
            if not aligned:
                print(f"{self.log_prefix} NOTE: per-component LR verification did not run: "
                      f"_build_component_lr_list describes {len(component_lrs)} group(s) "
                      f"{component_names} and the optimizer has {len(groups)}. The group "
                      f"LRs printed above are the effective rates; they were not checked "
                      f"against the configured per-component ones.")
        except Exception as e:
            print(f"{self.log_prefix} NOTE: per-component LR verification did not run: {e}")

        for i, group in enumerate(groups):
            if effective[i] is not None and float(effective[i]) == 0.0 and group.get('params'):
                label = component_names[i] if aligned else f"Optimizer group {i}"
                emit_training_warning(
                    f"{label} is in the optimizer with lr=0: its "
                    f"{len(group['params'])} parameter tensor(s) will not change.",
                    code="component_lr_zero",
                    prefix=self.log_prefix,
                )

    def _reassert_config_lr_on_resume(self):
        """Make the YAML config's per-component LRs win over the resumed ones.

        Called from ``train()``'s two resume branches, AFTER
        ``load_optimizer_state()`` (which imports the checkpoint's ``lr`` into
        every param group) and AFTER the scheduler fast-forward loop.

        The LR written to each group is ``configured_base_lr *
        schedule_multiplier(last_epoch)``, not the bare base LR: the training
        loop calls ``optimizer.step()`` BEFORE ``lr_scheduler.step()``, and a
        mid-epoch resume slices the batch list rather than iterating it, so a
        flat write would make the first post-resume step run at the
        un-multiplied base LR (unbounded error mid-warmup, 1/floor_ratio in a
        plateau_cosine_floor decay tail). See core/training/lr_utils.py.

        The description comes from ``_configured_component_lr_description``,
        which only ever returns one that corresponds index-for-index to the live
        param groups. Passing a SCALAR here instead broadcasts it over every
        group (``lr_utils.resolve_group_lrs``), which is why an unusable
        description refuses to write rather than falling back to one: on a
        3-group Anima full FT it turned [2e-5, 4e-5, 1e-5] into
        [1e-4, 1e-4, 1e-4].
        """
        optimizers = all_optimizers(self)
        if not optimizers:
            return

        schedulers = all_lr_schedulers(self)
        groups = [g for optimizer in optimizers
                  for g in list(getattr(optimizer, 'param_groups', []) or [])]
        if not groups:
            return

        component_lrs, component_names, source = \
            self._configured_component_lr_description(len(groups))

        if not component_lrs:
            emit_training_warning(
                f"the configured learning rates were NOT re-asserted on this resume: "
                f"{source}. Writing by index with a description that does not match the "
                f"groups would give components each other's rates, so every group keeps "
                f"the rate its checkpoint carried -- which is correct unless the config's "
                f"lr/unet_lr/text_encoder_lr was edited since that checkpoint, in which "
                f"case the edit has not taken effect.",
                code="component_lr_resume_unavailable",
                prefix=self.log_prefix,
            )
            return

        print(f"{self.log_prefix} LR re-assertion: {len(component_lrs)} configured rate(s) "
              f"for {len(groups)} param group(s), described by {source}")

        offset = 0
        for i, optimizer in enumerate(optimizers):
            count = len(list(getattr(optimizer, 'param_groups', []) or []))
            reassert_config_lr(
                optimizer,
                schedulers[i] if i < len(schedulers) else None,
                component_lrs[offset:offset + count],
                log_prefix=self.log_prefix,
                component_names=component_names[offset:offset + count],
                fallback_lr=self.learning_rate,
            )
            offset += count

    def _resolved_optimizer_hyperparameters(self) -> Dict[str, Any]:
        """weight_decay / betas / eps exactly as the user configured them.

        One place, because the fused-optimizer-groups path used to re-create
        every optimizer with hardcoded ``weight_decay=0.01``,
        ``betas=(0.9, 0.999)``, ``eps=1e-8``: a Block-Swap run with
        ``num_optimizer_groups > 0`` silently discarded the configured values
        while the YAML and the UI still said otherwise.

        ``None`` means "not configured" for all four, and the fallbacks below
        are the ones this method has always applied.
        """
        return {
            "weight_decay": self.optimizer_weight_decay if self.optimizer_weight_decay is not None else 0.01,
            "beta1": self.optimizer_beta1 if self.optimizer_beta1 is not None else 0.9,
            "beta2": self.optimizer_beta2 if self.optimizer_beta2 is not None else 0.999,
            "eps": self.optimizer_epsilon if self.optimizer_epsilon is not None else 1e-8,
        }

    # Options only the RingBuffer optimizers implement, with the value that
    # means "not requested". Used to warn instead of silently dropping them when
    # another optimizer is selected (the shipped full-FT default, adamw8bit, is
    # one of those). Keep in sync with _ringbuffer_optimizer_kwargs().
    #
    # optimizer_stochastic_rounding is deliberately NOT in this list: it is not
    # a ring-buffer-only option any more. _attach_stochastic_rounding() applies
    # it to any optimizer that exposes a per-parameter update seam, and reports
    # by name the ones that do not.
    _RINGBUFFER_ONLY_OPTIONS = (
        ("optimizer_cautious", False),
        ("optimizer_schedule_free", False),
        ("optimizer_schedule_free_r", 0.0),
        ("optimizer_schedule_free_weight_lr_power", 2.0),
        ("optimizer_use_radam", False),
    )

    def _ringbuffer_optimizer_kwargs(self) -> Dict[str, Any]:
        """Options only the RingBuffer optimizers accept.

        One place so that every option the user can set reaches the optimizer.
        ``stochastic_rounding`` used to be missing here, which meant the flag was
        accepted by the API, written into the YAML and then dropped: the
        optimizers resolved ``kwargs.get("stochastic_rounding", False)`` and
        rounded BF16 updates to nearest regardless of the user's choice.

        ``get_state_buffer`` used to be in that same state -- both optimizers
        resolved ``kwargs.get("get_state_buffer", None)``, nothing supplied one,
        so their CPU-resident state never activated and they allocated 8-bit
        state on the GPU. It is supplied here now, from
        ``HostOptimizerStateAllocator``, when ``optimizer_state_host_resident``
        is set. It is NOT a user option (it is an allocator, not a config flag);
        the trainer owns the allocator so it outlives optimizer construction and
        can be read back for the host-RAM accounting (G-RB2). See
        host_state_allocator.py and docs/guides/SENSENOVA_TRAINING_DESIGN.md 6.5.
        """
        kwargs = {
            "cautious": self.optimizer_cautious,
            "schedule_free": self.optimizer_schedule_free,
            "warmup_steps": self.optimizer_warmup_steps,
            "r": self.optimizer_schedule_free_r,
            "weight_lr_power": self.optimizer_schedule_free_weight_lr_power,
            "use_radam": self.optimizer_use_radam,
            "stochastic_rounding": self.optimizer_stochastic_rounding,
        }
        if getattr(self, "optimizer_state_host_resident", False):
            from .optimizers.host_state_allocator import HostOptimizerStateAllocator
            if getattr(self, "_host_state_allocator", None) is None:
                self._host_state_allocator = HostOptimizerStateAllocator()
            kwargs["get_state_buffer"] = self._host_state_allocator
        return kwargs

    # Measured HOST bytes per parameter of host-resident 8-bit ring-buffer state
    # (SENSENOVA_TRAINING_DESIGN.md 6.5's G-RB2 table): AdamW keeps two moments,
    # Lion one. absmax stays on the GPU and is not counted here.
    _RINGBUFFER_HOST_STATE_BYTES_PER_PARAM = {
        "adamw8bit_ringbuffer": 2.0,
        "lion8bit_ringbuffer": 1.0,
    }

    def _announce_host_state_budget(self, optimizer_type: str, param_groups) -> None:
        """Say what the pinned host allocation costs BEFORE it is taken.

        The allocation is unpageable and lives for the whole run, so the number
        has to be visible before the machine is committed to it, not inferred
        afterwards from a swap storm.
        """
        if not getattr(self, "optimizer_state_host_resident", False):
            return
        per_param = self._RINGBUFFER_HOST_STATE_BYTES_PER_PARAM.get(
            optimizer_type.lower())
        if per_param is None:
            return
        params = sum(p.numel() for group in param_groups for p in group["params"])
        line = (
            f"{self.log_prefix} HOST RAM announce: {optimizer_type} with "
            f"optimizer_state_host_resident will pin "
            f"{format_param_count(params)} params x {per_param:g} B/param = "
            f"{params * per_param / 1024 ** 3:.2f} GiB of host memory "
            f"(unpageable, held for the whole run)"
        )
        try:
            import psutil
            info = psutil.Process().memory_info()
            peak = getattr(info, "peak_wset", info.rss)
            line += (f"; process working set now {info.rss / 1024 ** 3:.2f} GiB "
                     f"(peak {peak / 1024 ** 3:.2f} GiB)")
        except Exception as exc:
            line += f"; current working set unavailable ({exc})"
        print(line)

    # Optimizers that apply stochastic rounding inside their own update, so
    # _attach_stochastic_rounding() must leave them alone.
    _NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS = ("adamw8bit_ringbuffer", "lion8bit_ringbuffer")

    # bitsandbytes optimizers with NO per-parameter fused-backward implementation.
    # Their step() runs after Block Swap has moved parameters to the CPU, and every
    # bitsandbytes optimizer raises on a CPU-resident parameter, so Block Swap +
    # any of these is a guaranteed crash on the first step. ``adamw8bit`` is
    # deliberately absent: _setup_fused_backward_pass patches a per-parameter
    # step_param onto it. ``adamw`` (torch) is absent because torch's own AdamW
    # updates CPU parameters correctly.
    _BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS = (
        "lion8bit", "paged_adamw", "paged_adamw8bit", "paged_lion8bit",
    )

    def _attach_stochastic_rounding(self, optimizer_type: str):
        """Make ``optimizer_stochastic_rounding`` reach the optimizer that was chosen.

        Full fine-tuning writes optimizer updates straight into BF16 storage
        with no FP32 master, so round-to-nearest deterministically discards every
        update below half a ULP and those weights never move again. Only the two
        ring-buffer optimizers implemented the repair; the shipped full-FT
        default is ``adamw8bit``, so a user who changed nothing got the defect.

        Third-party optimizers are covered without modifying them, by making the
        parameter FP32 for the duration of one per-parameter update call and
        rounding the result back stochastically -- see
        ``stochastic_rounding.attach_stochastic_rounding``. Optimizers with no
        per-parameter entry point cannot be covered that way and are named here
        rather than left to look covered.

        Must run AFTER the Block Swap setup: ``_setup_fused_backward_pass``
        installs ``step_param`` (and ``_setup_fused_optimizer_groups`` replaces
        ``self.optimizer`` outright), so attaching earlier would be discarded.
        """
        if not self.optimizer_stochastic_rounding:
            return

        name = optimizer_type.lower()
        if name in self._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS:
            return  # applied inside the optimizer; already logged above

        from .optimizers.stochastic_rounding import attach_stochastic_rounding

        groups = getattr(self, "fused_optimizer_groups", None)
        optimizers = list(groups.optimizers) if groups is not None else [self.optimizer]

        # transformers' Adafactor updates every parameter inside one step() with
        # no per-parameter entry point, but the fused variant this repo already
        # ships for Block Swap is the same algorithm exposed as step_param(). Use
        # it when stochastic rounding is asked for, so Adafactor is covered
        # whether or not Block Swap is on. (When Block Swap is on the patch has
        # already been applied and this is a no-op.)
        #
        # Stated plainly because it is a real consequence, not a side effect:
        # ticking stochastic rounding on an Adafactor run REPLACES
        # transformers.Adafactor.step with adafactor_fused's port of it (from
        # sd-scripts), which then dispatches through the interposed step_param.
        # Same algorithm, different implementation; without it step() would keep
        # writing BF16 with round-to-nearest while the log claimed otherwise.
        if name == "adafactor":
            from .optimizers.adafactor_fused import patch_adafactor_fused
            for optimizer in optimizers:
                if not hasattr(optimizer, "step_param"):
                    patch_adafactor_fused(optimizer)
                    print(f"{self.log_prefix} Adafactor: step() replaced by the fused "
                          f"per-parameter implementation (same algorithm) so stochastic "
                          f"rounding can be applied to each parameter update")

        covered = []
        for optimizer in optimizers:
            covered.extend(attach_stochastic_rounding(optimizer))

        if covered:
            print(f"{self.log_prefix} Stochastic rounding attached to '{optimizer_type}' "
                  f"({', '.join(sorted(set(covered)))}); BF16 parameter updates below "
                  f"half a ULP are now carried in expectation instead of discarded")
        else:
            print(f"{self.log_prefix} WARNING: optimizer_stochastic_rounding is NOT applied: "
                  f"'{optimizer_type}' updates all of its parameters inside one call and "
                  f"exposes no per-parameter seam to interpose on. Its BF16 updates below "
                  f"half a ULP are discarded and those weights never move. "
                  f"Choose adamw8bit, lion8bit, adafactor, adamw8bit_ringbuffer or "
                  f"lion8bit_ringbuffer for a covered optimizer.")

        if self.weight_dtype != torch.bfloat16:
            print(f"{self.log_prefix} NOTE: stochastic rounding applies to BF16 parameters "
                  f"only; weight dtype is {self.weight_dtype}, so parameters in that dtype "
                  f"are updated unchanged")

    def setup_optimizer(
        self,
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "constant",
        total_steps: int = 1000,
    ):
        """
        Setup optimizer and learning rate scheduler.

        Args:
            optimizer_type: Optimizer type (adamw, adamw8bit, adafactor, etc.)
            lr_scheduler_type: LR scheduler type (constant, cosine, etc.)
            total_steps: Total training steps
        """
        from core.training.adapters.base_adapter import resolve_component_lr

        # Get trainable parameters from subclass
        param_groups = self.setup_trainable_parameters()

        # Add Vision Encoder parameters if training is enabled
        if getattr(self, '_train_vision_encoder', False) and getattr(self, 'vision_encoder', None) is not None:
            ve_lr = resolve_component_lr(self, '_vision_encoder_lr', 'text_encoder_lr',
                                         label="vision encoder")
            ve_params = list(self.vision_encoder.parameters())
            if ve_params:
                param_groups.append({"params": ve_params, "lr": ve_lr})
                ve_total = sum(p.numel() for p in ve_params)
                print(f"{self.log_prefix} Vision Encoder: Added {len(ve_params)} param tensors ({ve_total/1e6:.1f}M params, lr={ve_lr}) to optimizer")
                # Set requires_grad on VE model
                for p in ve_params:
                    p.requires_grad_(True)

        print(f"{self.log_prefix} Setting up optimizer: {optimizer_type}")
        print(f"{self.log_prefix} LR scheduler: {lr_scheduler_type}")

        # The authoritative optimizer name: an argument, not the config value
        # load_components checked before the 17.6 GiB load.
        from core.training.ops.training_method import is_full_finetune
        if getattr(self, "is_sensenova", False) and is_full_finetune(self):
            from core.training.ops.sensenova_ops import (
                assert_full_finetune_contract,
                enforce_full_finetune_stochastic_rounding,
            )
            assert_full_finetune_contract(self, optimizer_type)
            # Before the optimizer is built: the adamw8bit patch and the
            # ring-buffer kwargs below both read the flag as they construct.
            enforce_full_finetune_stochastic_rounding(self)

        # Lion's Schedule-Free kernel writes the wrong sequence into the
        # parameter. Schedule-Free keeps a POSITION sequence z and derives the
        # weights from it; lion8bit_schedulefree_kernel.cu instead uses z for
        # Lion's momentum EMA (z = beta2*z + (1-beta2)*g) and then stores
        # x = (1-ckp1)*z + ckp1*y into the parameter. ckp1 is ~1/k, so within a
        # few steps the parameter IS the momentum buffer: measured with random
        # gradients, corr(p, z) = 0.994 by step 5 and 0.9996 by step 20, and
        # mean|p| left its initial 1.6e-2 for the momentum's own scale. The
        # original weights are gone either way -- upward here, and down to
        # 2.5e-5 under a constant gradient.
        #
        # This is refused rather than patched: a correct Lion Schedule-Free needs
        # BOTH a position sequence and a momentum EMA, and _init_param_state
        # allocates exactly one 8-bit state for this mode, so a fix changes the
        # state layout, the kernel signature and the checkpoint format. Refusing
        # is checked here, before the factory call, because the factory's errors
        # are caught below and turned into a silent fall back to AdamW.
        if optimizer_type.lower() == "lion8bit_ringbuffer" and self.optimizer_schedule_free:
            raise ValueError(
                "optimizer_schedule_free is not supported with 'lion8bit_ringbuffer'. Its "
                "Schedule-Free path stores Lion's momentum EMA into the parameter instead of "
                "the Schedule-Free position sequence, which replaces the weights with the "
                "momentum buffer within a few steps. "
                "Options: (1) use 'adamw8bit_ringbuffer', whose Schedule-Free path keeps the "
                "position sequence, (2) set optimizer_schedule_free=false to use plain Lion."
            )

        # Create optimizer using factory
        from .optimizer_factory import OptimizerFactory
        try:
            # Use hyperparameters from config, or fall back to defaults
            hyper = self._resolved_optimizer_hyperparameters()
            weight_decay = hyper["weight_decay"]
            beta1 = hyper["beta1"]
            beta2 = hyper["beta2"]
            eps = hyper["eps"]

            # Lion optimizers use 'lion_betas' kwarg instead of 'betas', and don't have epsilon
            optimizer_kwargs = {
                "weight_decay": weight_decay,
            }
            if "lion" in optimizer_type.lower():
                optimizer_kwargs["lion_betas"] = (beta1, beta2)
                # Lion doesn't use epsilon
            else:
                optimizer_kwargs["betas"] = (beta1, beta2)
                optimizer_kwargs["eps"] = eps

            # Pass cautious and Schedule-Free options to RingBuffer optimizers
            if "ringbuffer" in optimizer_type.lower():
                self._announce_host_state_budget(optimizer_type, param_groups)
                optimizer_kwargs.update(self._ringbuffer_optimizer_kwargs())
                if self.optimizer_stochastic_rounding:
                    print(f"{self.log_prefix} Stochastic rounding enabled for BF16 parameter updates")
                    if self.weight_dtype != torch.bfloat16:
                        print(f"{self.log_prefix} NOTE: stochastic rounding only applies to BF16 "
                              f"parameters; weight dtype is {self.weight_dtype}")
                    if self.optimizer_schedule_free:
                        # The Schedule-Free 'z' sequence is covered too: 8-bit z is
                        # re-quantized stochastically inside the CUDA kernel, and a
                        # parameter-dtype z is updated through an FP32 image and
                        # rounded back. exp_avg_sq is deliberately not covered.
                        print(f"{self.log_prefix} NOTE: with schedule_free, stochastic rounding also "
                              f"covers the z sequence (8-bit z is re-quantized stochastically); "
                              f"the exp_avg_sq second moment is not covered")
            else:
                # Never accept these options silently: only the RingBuffer
                # optimizers implement them, and the shipped full-FT default
                # (adamw8bit) is not one of them.
                ignored = [name for name, unset in self._RINGBUFFER_ONLY_OPTIONS
                           if getattr(self, name) != unset]
                if ignored:
                    print(f"{self.log_prefix} WARNING: {', '.join(ignored)} "
                          f"{'is' if len(ignored) == 1 else 'are'} not supported by "
                          f"'{optimizer_type}' and will not be applied "
                          f"(supported: adamw8bit_ringbuffer, lion8bit_ringbuffer)")

            self.optimizer = OptimizerFactory.create_optimizer(
                optimizer_type=optimizer_type,
                params=param_groups,
                learning_rate=self.learning_rate,
                **optimizer_kwargs,
            )
        except (ValueError, ImportError) as e:
            print(f"{self.log_prefix} ERROR: {e}")
            print(f"{self.log_prefix} Falling back to AdamW")
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                eps=1e-8,
            )

        # Set optimizer to train mode (required for RingBuffer optimizers)
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
            print(f"{self.log_prefix} Optimizer set to train mode")

        # Log actual LR values for each parameter group
        print(f"{self.log_prefix} ===== Optimizer Parameter Group LRs =====")
        for i, group in enumerate(self.optimizer.param_groups):
            group_lr = group.get('lr', 'N/A')
            num_tensors = len(group['params'])
            num_scalars = sum(p.numel() for p in group['params'])
            print(f"{self.log_prefix}   Group {i}: lr={group_lr}, tensors={num_tensors}, params={format_param_count(num_scalars)}")
        print(f"{self.log_prefix} ==========================================")
        # What the adapter asked for, kept for the report below: the fused paths
        # can replace self.optimizer entirely before this method returns.
        requested_group_lrs = [g.get('lr') for g in self.optimizer.param_groups]

        # Setup LR scheduler
        if str(lr_scheduler_type).lower() == "plateau_cosine_floor":
            self.lr_scheduler = self._build_plateau_cosine_floor_scheduler(
                self.optimizer, total_steps
            )
        else:
            from diffusers.optimization import get_scheduler as get_diffusers_scheduler
            self.lr_scheduler = get_diffusers_scheduler(
                lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.optimizer_warmup_steps,
                num_training_steps=total_steps,
            )

        # Initialize weight EMA (opt-in, default off). Must run after the
        # optimizer (and therefore the trainable param groups) exists.
        self._setup_ema()

        # Setup fused backward/optimizer groups if Block Swap is enabled
        if self.blocks_to_swap > 0:
            if self.num_optimizer_groups > 0:
                # Validate compatibility: Block Swap + Fused Optimizer Groups + 8bit optimizer.
                # 8-bit optimizers (bitsandbytes and our ring-buffer variants) require the
                # parameter to be on CUDA at update time; Fused Optimizer Groups call a batched
                # optimizer.step() after Block Swap may have moved some params to CPU. The
                # ring-buffer optimizers must use num_optimizer_groups=0 (they register their
                # own per-parameter fused-backward hooks instead).
                # The paged_* names are here for the same reason as their
                # un-prefixed siblings: bitsandbytes' PagedAdamW8bit /
                # PagedLion8bit are 8-bit optimizers, and paging their STATE to
                # host memory does nothing about the PARAMETER that Block Swap
                # moved to the CPU. Omitting them let exactly the same crash
                # through under a different name.
                if optimizer_type.lower() in [
                    "adamw8bit", "lion8bit", "adafactor8bit",
                    "paged_adamw8bit", "paged_lion8bit",
                    "adamw8bit_ringbuffer", "lion8bit_ringbuffer",
                ]:
                    raise ValueError(
                        f"Block Swap + Fused Optimizer Groups (num_optimizer_groups>0) is incompatible "
                        f"with 8-bit optimizers ({optimizer_type}). 8-bit optimizers cannot update "
                        f"CPU-resident parameters that Block Swap creates. "
                        f"Options: (1) set num_optimizer_groups=0 (ring-buffer/8bit optimizers register "
                        f"their own per-parameter fused-backward hooks), "
                        f"(2) use a non-8bit optimizer (AdamW, Lion, etc.) with num_optimizer_groups, "
                        f"(3) disable Block Swap (blocks_to_swap=0)"
                    )

                if getattr(self, "use_ema", False):
                    raise NotImplementedError(
                        "use_ema is not yet supported together with Block Swap + Fused "
                        "Optimizer Groups (num_optimizer_groups>0). The EMA update is "
                        "attached to the single self.optimizer.step() call site in the "
                        "main training loop; Fused Optimizer Groups instead update "
                        "parameters via per-parameter post-accumulate-grad hooks that "
                        "bypass that call site entirely, so EMA would silently never "
                        "update. Disable use_ema, or set num_optimizer_groups=0."
                    )

                # Fused optimizer groups: works with non-8bit optimizers only
                self._setup_fused_optimizer_groups(optimizer_type, total_steps, lr_scheduler_type)
            elif optimizer_type.lower() in FUSED_BACKWARD_OPTIMIZERS:
                if getattr(self, "use_ema", False):
                    raise NotImplementedError(
                        "use_ema is not yet supported together with the fused backward "
                        "pass (Block Swap + Adafactor/AdamW8bit/ring-buffer optimizers). "
                        "The EMA update is attached to the single self.optimizer.step() "
                        "call site in the main training loop; the fused backward pass "
                        "instead updates parameters via per-parameter "
                        "post-accumulate-grad hooks that bypass that call site entirely, "
                        "so EMA would silently never update. Disable use_ema, disable "
                        "Block Swap, or use a non-fused optimizer configuration."
                    )
                # Fused backward pass: Adafactor / AdamW8bit / ring-buffer optimizers.
                # The ring-buffer optimizers register their per-parameter post-accumulate-grad
                # hooks inside _setup_fused_backward_pass, so their updates run before Block Swap
                # moves each block to CPU (otherwise CPU-resident params are silently skipped).
                self._setup_fused_backward_pass(optimizer_type)
            elif optimizer_type.lower() in self._BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS:
                # Every bitsandbytes optimizer -- 8-bit AND the 32-bit paged one --
                # refuses a CPU-resident parameter: Optimizer.step() reaches
                # bitsandbytes.functional.is_on_gpu(), which raises rather than
                # updating on the host. Block Swap leaves every swapped block on
                # the CPU when its backward hook fires, so by the time
                # optimizer.step() runs those parameters are exactly that.
                # Measured on the installed build (bitsandbytes 0.49.1): a CPU
                # BF16 parameter with a CPU gradient makes Lion8bit, AdamW8bit,
                # PagedAdamW, PagedAdamW8bit and PagedLion8bit all raise inside
                # is_on_gpu's own error formatting ("AttributeError: 'NoneType'
                # object has no attribute 'shape'"), which says nothing about the
                # actual problem.
                #
                # adamw8bit escapes this list only because
                # _setup_fused_backward_pass installs a per-parameter step_param
                # for it. These names have no such implementation, so refuse the
                # configuration here -- where the message can name the remedies --
                # instead of failing mid-backward with that AttributeError.
                raise ValueError(
                    f"Block Swap (blocks_to_swap={self.blocks_to_swap}) is incompatible with "
                    f"the '{optimizer_type}' optimizer. bitsandbytes optimizers cannot update "
                    f"the CPU-resident parameters Block Swap creates, and no fused backward "
                    f"pass is implemented for this optimizer, so optimizer.step() raises on "
                    f"the first step. "
                    f"Options: (1) use 'lion8bit_ringbuffer' (Lion) or 'adamw8bit_ringbuffer' "
                    f"(AdamW), which keep 8-bit state and register their own per-parameter "
                    f"fused-backward hooks that run while the parameter is still on the GPU, "
                    f"(2) use 'adamw8bit' or 'adafactor', which have a fused backward pass, "
                    f"(3) disable Block Swap (blocks_to_swap=0)."
                )
        elif (getattr(self, "is_sensenova", False)
              and is_full_finetune(self)
              and self.num_optimizer_groups == 0
              and optimizer_type.lower() in FUSED_BACKWARD_OPTIMIZERS):
            # The hooks have no block-swap dependency; the setup above sits
            # inside `blocks_to_swap > 0` only because that is the one place
            # every other architecture needs them. SenseNova refuses a non-zero
            # blocks_to_swap and would otherwise hold every gradient of the half
            # it trains resident until optimizer.step().
            self._setup_fused_backward_pass(optimizer_type)

        if (getattr(self, "is_sensenova", False) and is_full_finetune(self)
                and not fused_backward_active(self)):
            # Loud, because unfused is not a degraded mode here: it is 15.1 GiB
            # of resident gradients on a route budgeted for none. Reachable by
            # widening the optimizer contract without adding the name to
            # FUSED_BACKWARD_OPTIMIZERS, or by a PyTorch too old for
            # register_post_accumulate_grad_hook (which _setup_fused_backward_pass
            # warns about and returns from).
            raise RuntimeError(
                f"SenseNova full fine-tuning did not install its fused backward "
                f"pass (optimizer={optimizer_type}, blocks_to_swap="
                f"{self.blocks_to_swap}, num_optimizer_groups="
                f"{self.num_optimizer_groups}). Without it every gradient of the "
                f"materialized half stays resident until optimizer.step(), which "
                f"this route's memory budget assumes never happens. This is an "
                f"internal inconsistency, not a setting: the optimizer contract "
                f"and FUSED_BACKWARD_OPTIMIZERS disagree, or this PyTorch build "
                f"has no register_post_accumulate_grad_hook."
            )

        # The census only exists where hooks apply the updates; say so rather
        # than leaving the switch looking active.
        if (getattr(self, "optimizer_update_census", False)
                and getattr(self, "_update_census", None) is None):
            print(f"{self.log_prefix} NOTE: optimizer_update_census is set but this "
                  f"run has no fused backward pass (optimizer={optimizer_type}, "
                  f"blocks_to_swap={self.blocks_to_swap}, num_optimizer_groups="
                  f"{self.num_optimizer_groups}), so no census is taken. It detects "
                  f"a per-parameter hook that never fires, which is a failure only "
                  f"the fused path can have; optimizer.step() updates every "
                  f"parameter it owns in one call.")

        # Last, because the Block Swap setup above installs step_param and can
        # replace self.optimizer: stochastic rounding has to wrap whatever
        # actually ends up performing the update.
        self._attach_stochastic_rounding(optimizer_type)

        # After every path that can replace the optimizer or its LRs.
        self._record_configured_group_lrs(requested_group_lrs)
        self._report_effective_component_lrs(requested_group_lrs)

        if getattr(self, "is_sensenova", False) and is_full_finetune(self):
            from core.training.ops.sensenova_ops import (
                assert_four_phase_fused_backward,
                assert_full_finetune_stochastic_rounding_attached,
            )
            assert_full_finetune_stochastic_rounding_attached(self, optimizer_type)
            assert_four_phase_fused_backward(self)
            self._assert_ringbuffer_state_host_resident(optimizer_type)

    def _assert_ringbuffer_state_host_resident(self, optimizer_type: str) -> None:
        """Prove the 8-bit state is where the budget says, not that a flag is set.

        A ``get_state_buffer`` that handed back CUDA tensors leaves the flag true
        and the bytes on the GPU, which is the misbudget this route cannot
        absorb. The state is allocated lazily by the first backward, so it is
        forced here: the failure then lands at optimizer setup, next to the
        announce, instead of inside step 1's autograd engine.
        """
        if optimizer_type.lower() not in self._RINGBUFFER_HOST_STATE_BYTES_PER_PARAM:
            return
        from .optimizers.host_state_allocator import assert_state_host_resident

        if not hasattr(self.optimizer, "_init_param_state"):
            raise RuntimeError(
                f"optimizer={optimizer_type} was requested but this run holds a "
                f"{type(self.optimizer).__name__}, which has no ring-buffer "
                f"state: the factory raised and setup_optimizer fell back to "
                f"torch AdamW, whose fp32 state does not fit this route."
            )
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if len(self.optimizer.state[param]) == 0:
                    self.optimizer._init_param_state(param)
        census = assert_state_host_resident(self.optimizer)
        host = sum(b["cpu"] for b in census.values())
        cuda = sum(b["cuda"] for b in census.values())
        print(f"{self.log_prefix} Ring-buffer optimizer state census: "
              f"{host / 1024 ** 3:.2f} GiB host (all pinned), "
              f"{cuda / 1024 ** 3:.2f} GiB on the GPU (absmax)")

    def _fused_backward_target_module(self):
        """Return the main trainable module the ring-buffer optimizers register their
        post-accumulate-grad hooks on.

        Arch-dependent: U-Net archs (SD/SDXL) keep the trainable model on
        ``self.unet`` and set ``self.transformer = None``; transformer/DiT archs
        (LTX-2.3, Anima, FLUX.2, Z-Image, ...) set ``self.unet = None`` and keep the
        DiT on ``self.transformer``. (Previously hardcoded ``self.unet``, which is
        None for DiT archs -> AttributeError under block-swap + ring-buffer.)

        The hooks themselves are registered from ``optimizer.param_groups``, which
        also covers the text-encoder / vision-encoder groups ``setup_optimizer``
        appends and this module does not contain; the module is passed for
        parameter names and for the check that none of ITS trainable parameters is
        missing from the optimizer.
        """
        module = getattr(self, "transformer", None)
        if module is None:
            module = getattr(self, "unet", None)
        if module is None:
            raise RuntimeError(
                "Fused-backward ring-buffer setup found neither self.transformer nor "
                "self.unet; the trainable model must be loaded before optimizer setup."
            )
        return module

    def _warn_grad_clipping_ignored_under_fused(self, max_grad_norm: float) -> None:
        """Say once that ``max_grad_norm`` does nothing under fused backward.

        Global-norm clipping needs the whole gradient vector before it can pick a
        scale, and the fused backward pass updates each parameter the moment its
        own gradient exists -- by the time the last one arrives the first is
        already applied. The two cannot coexist; the setting is ignored, and
        substituting a per-parameter clip under the same name would be a
        different algorithm.
        """
        if max_grad_norm is None or max_grad_norm <= 0:
            return
        if not fused_backward_active(self):
            return
        if getattr(self, "_fused_clipping_warned", False):
            return
        self._fused_clipping_warned = True
        mode = ("fused optimizer groups" if self.fused_optimizer_groups is not None
                else "the fused backward pass")
        emit_training_warning(
            f"max_grad_norm={max_grad_norm} is IGNORED under "
            f"{mode}. Gradient clipping by global norm has to see every gradient "
            f"before it can scale them, and this mode applies each parameter's update "
            f"as soon as that parameter's gradient exists, so no global norm is ever "
            f"available. No clipping of any kind is applied. To clip, disable the fused "
            f"path (blocks_to_swap=0 and num_optimizer_groups=0); to silence this, set "
            f"max_grad_norm=0.",
            code="fused_grad_clipping_ignored",
            prefix=self.log_prefix,
        )

    def _warn_gradient_accumulation_ignored_under_fused(
        self,
        gradient_accumulation_steps: int,
        batch_size: int,
        multi_noise_timesteps: int = 1,
    ) -> None:
        """Say once, before the run starts, that accumulation does not happen here.

        Accumulating means holding every parameter's summed gradient until the end
        of the window; the fused paths exist precisely to never have all gradients
        resident at once, and free each one as its update is applied. So the window
        cannot span backward passes: each backward becomes its own optimizer step.
        The setting is left as the user wrote it and nothing is refused -- the
        effective batch is just reported as what it will actually be.
        """
        accum = int(gradient_accumulation_steps or 1)
        if accum <= 1:
            return
        if not fused_backward_active(self):
            return
        if getattr(self, "_fused_accum_warned", False):
            return
        self._fused_accum_warned = True
        mode = ("fused optimizer groups" if self.fused_optimizer_groups is not None
                else "the fused backward pass")
        mnt = max(1, int(multi_noise_timesteps or 1))
        intended = batch_size * accum // mnt
        emit_training_warning(
            f"gradient_accumulation_steps={accum} is IGNORED under "
            f"{mode}. Its hooks apply each parameter's update and free that gradient as soon "
            f"as it exists, so no gradient survives to be summed across backward passes. "
            f"Every backward pass becomes its own optimizer step: the optimizer steps {accum}x "
            f"more often than the reported step count, each step seeing ONE batch of "
            f"{batch_size} -- not the effective batch of {intended} that "
            f"batch_size x gradient_accumulation_steps"
            f"{' / multi_noise_timesteps' if mnt > 1 else ''} implies. "
            f"Each of those steps also sees the loss divided by {accum}, which for an "
            f"adaptive optimizer (AdamW/Lion/Adafactor) barely shrinks the update, so the run "
            f"moves further per reported step than a non-fused run would, on noisier gradients. "
            f"The LR schedule still advances once per {accum} backward passes. "
            f"To actually accumulate, disable the fused path (blocks_to_swap=0 and "
            f"num_optimizer_groups=0); to silence this, set gradient_accumulation_steps=1, "
            f"which is what this run is doing.",
            code="fused_gradient_accumulation_ignored",
            prefix=self.log_prefix,
        )

    def _setup_fused_backward_pass(self, optimizer_type: str):
        """
        Setup fused backward pass for Block Swap compatibility.

        Registers post-accumulate-grad hooks that update parameters immediately
        after gradients are computed, before Block Swap moves them to CPU.

        Works with Adafactor or AdamW8bit optimizers (PyTorch 2.1+).

        Args:
            optimizer_type: Optimizer type ("adafactor" or "adamw8bit")
        """
        refuse_grad_scaler_under_fused_path(self, optimizer_type, "fused backward pass")

        print(f"{self.log_prefix} Setting up fused backward pass for {optimizer_type}...")

        # Check PyTorch version
        import torch
        if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
            print(f"{self.log_prefix} WARNING: PyTorch 2.1+ required for fused backward pass")
            print(f"{self.log_prefix} Current version: {torch.__version__}")
            print(f"{self.log_prefix} Fused backward pass disabled")
            return

        # Schedule-Free has no fused-backward implementation. The ring-buffer
        # hooks below apply the plain (non-Schedule-Free) 8-bit update and read
        # state['exp_avg'] + state['exp_avg_sq'] / state['absmax1'] +
        # state['absmax2'] (AdamW) or state['exp_avg'] / state['absmax']
        # (Lion), none of which _init_param_state allocates in Schedule-Free
        # mode -- it allocates z / absmax_z (plus exp_avg_sq / absmax2 for
        # AdamW) instead, so the combination raises KeyError inside the first
        # backward pass. Refuse
        # it here, where the message can say what to change, rather than
        # accepting the option and failing (or worse, silently running a
        # different algorithm than the one that was asked for).
        if (optimizer_type.lower() in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer")
                and self.optimizer_schedule_free):
            raise ValueError(
                f"optimizer_schedule_free is not supported with the fused backward pass "
                f"that Block Swap requires for '{optimizer_type}'. The per-parameter "
                f"hooks implement the standard update only. "
                f"Options: (1) set optimizer_schedule_free=false, "
                f"(2) disable Block Swap (blocks_to_swap=0), which runs the "
                f"Schedule-Free path inside optimizer.step()."
            )

        setup_fused_grad_norm(self, [self.optimizer])
        # Before the branch below, which returns early for the ring-buffer
        # optimizers. param_groups are final by now, so the expectation set is.
        setup_update_census(self, [self.optimizer])

        # Patch optimizer with step_param method
        if optimizer_type.lower() == "adafactor":
            from .optimizers.adafactor_fused import patch_adafactor_fused
            patch_adafactor_fused(self.optimizer)
        elif optimizer_type.lower() == "adamw8bit":
            from .optimizers.adamw8bit_fused import patch_adamw8bit_fused
            # step_param delegates to bitsandbytes' own per-parameter seam, so the
            # state stays 8-bit and stays the same format step() writes. It applies
            # stochastic rounding itself rather than being wrapped, to hand the
            # kernel an FP32 view without turning the state FP32 too.
            patch_adamw8bit_fused(self.optimizer, self.optimizer_stochastic_rounding)
        elif optimizer_type.lower() == "adamw8bit_ringbuffer":
            # AdamW8bit_RingBuffer has built-in hook support via patch_adamw8bit_ringbuffer
            from .optimizers.adamw8bit_ringbuffer import patch_adamw8bit_ringbuffer
            # Note: patch_adamw8bit_ringbuffer registers hooks itself, so we don't need the loop below.
            # The main trainable module is arch-dependent: U-Net archs (SD/SDXL) use
            # self.unet; transformer archs (LTX-2.3, Anima, FLUX.2, Z-Image, ...) set
            # self.unet=None and keep the DiT on self.transformer. It is passed for
            # names and for the orphan check; the hooks come from param_groups, so
            # text-encoder / vision-encoder groups are covered too.
            patch_adamw8bit_ringbuffer(self._fused_backward_target_module(), self.optimizer)
            self.use_fused_backward = True
            print(f"{self.log_prefix} AdamW8bit_RingBuffer hooks registered via patch_adamw8bit_ringbuffer")
            return  # Skip the hook registration loop below
        elif optimizer_type.lower() == "lion8bit_ringbuffer":
            # Lion8bit_RingBuffer has built-in hook support via register_lion8bit_fused_backward
            from .optimizers.lion8bit_ringbuffer import register_lion8bit_fused_backward
            # Note: register_lion8bit_fused_backward registers hooks itself, so we don't need the loop below.
            # See the adamw8bit_ringbuffer branch: target module is arch-dependent
            # (self.unet for U-Net archs, self.transformer for DiT archs).
            register_lion8bit_fused_backward(self.optimizer, self._fused_backward_target_module())
            self.use_fused_backward = True
            print(f"{self.log_prefix} Lion8bit_RingBuffer hooks registered via register_lion8bit_fused_backward")
            return  # Skip the hook registration loop below

        # Register hooks for all trainable parameters
        from .optimizers.fused_grad_norm import record_fused_grad_norm

        hooks_registered = 0
        for param_group in self.optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, pg=param_group):
                        """Hook called when gradient is ready for this parameter"""
                        # No clipping here: a global-norm clip cannot be applied
                        # per parameter (see _warn_grad_clipping_ignored_under_fused).

                        # Before the update, which is free to scale the gradient
                        # in place, and before the clear below.
                        record_fused_grad_norm(self.optimizer, tensor)

                        # Update THIS parameter immediately (while on GPU)
                        self.optimizer.step_param(tensor, pg)

                        # Clear gradient to save memory
                        tensor.grad = None

                    # Register hook: called when gradient for THIS parameter is ready
                    parameter.register_post_accumulate_grad_hook(__grad_hook)
                    hooks_registered += 1

        self.use_fused_backward = True
        print(f"{self.log_prefix} Registered {hooks_registered} fused backward hooks")
        print(f"{self.log_prefix} Optimizer.step() and zero_grad() will be called by hooks automatically")

    def _setup_fused_optimizer_groups(self, optimizer_type: str, total_steps: int, lr_scheduler_type: str):
        """
        Setup fused optimizer groups for Block Swap compatibility.

        Works with ANY optimizer (AdamW, AdamW8bit, Lion8bit, etc.) by dividing
        parameters into groups and updating each group when all its gradients are ready.

        Args:
            optimizer_type: Optimizer type (adamw, adamw8bit, etc.)
            total_steps: Total training steps
            lr_scheduler_type: LR scheduler type
        """
        refuse_grad_scaler_under_fused_path(self, optimizer_type, "fused optimizer groups")

        print(f"{self.log_prefix} Setting up fused optimizer groups...")

        # Check PyTorch version
        import torch
        if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
            print(f"{self.log_prefix} WARNING: PyTorch 2.1+ required for fused optimizer groups")
            print(f"{self.log_prefix} Current version: {torch.__version__}")
            print(f"{self.log_prefix} Fused optimizer groups disabled")
            return

        # Get trainable parameters from current optimizer
        trainable_params = []
        for param_group in self.optimizer.param_groups:
            trainable_params.extend(param_group["params"])

        # Create multiple optimizers by dividing parameters into groups
        from .optimizers.fused_optimizer_groups import create_optimizer_groups, FusedOptimizerGroups

        # Configured hyperparameters, not hardcoded ones: this path re-creates
        # every optimizer from scratch, so passing literals here discarded the
        # user's optimizer_weight_decay / betas / epsilon for exactly the runs
        # that need Block Swap. Lion takes its betas under a different keyword,
        # the same split setup_optimizer() makes.
        hyper = self._resolved_optimizer_hyperparameters()
        group_kwargs: Dict[str, Any] = {}
        if "lion" in optimizer_type.lower():
            group_kwargs["lion_betas"] = (hyper["beta1"], hyper["beta2"])
        optimizers = create_optimizer_groups(
            params=trainable_params,
            optimizer_type=optimizer_type,
            num_groups=self.num_optimizer_groups,
            learning_rate=self.learning_rate,
            weight_decay=hyper["weight_decay"],
            betas=(hyper["beta1"], hyper["beta2"]),
            eps=hyper["eps"],
            **group_kwargs,
        )

        # Replace self.optimizer with first optimizer (for compatibility)
        self.optimizer = optimizers[0]

        # Set all optimizers to train mode (required for RingBuffer optimizers)
        for optimizer in optimizers:
            if hasattr(optimizer, 'train'):
                optimizer.train()
        print(f"{self.log_prefix} All {len(optimizers)} optimizers set to train mode")

        # Create LR schedulers for all optimizers
        lr_schedulers = []
        if str(lr_scheduler_type).lower() == "plateau_cosine_floor":
            # Same lambda applied independently to each optimizer group so all
            # groups stay in lockstep (matches the main-path behavior).
            for optimizer in optimizers:
                lr_schedulers.append(
                    self._build_plateau_cosine_floor_scheduler(optimizer, total_steps)
                )
        else:
            from diffusers.optimization import get_scheduler as get_diffusers_scheduler
            for optimizer in optimizers:
                lr_scheduler = get_diffusers_scheduler(
                    lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.optimizer_warmup_steps,
                    num_training_steps=total_steps,
                )
                lr_schedulers.append(lr_scheduler)

        # Replace self.lr_scheduler with first scheduler (for compatibility)
        self.lr_scheduler = lr_schedulers[0]

        # Store all schedulers for stepping
        self.lr_schedulers = lr_schedulers

        setup_fused_grad_norm(self, optimizers)

        # Create FusedOptimizerGroups instance
        self.fused_optimizer_groups = FusedOptimizerGroups(
            optimizers=optimizers,
            # No clipping: the hook's clip is per parameter, which is not the
            # global-norm clip max_grad_norm names. _warn_grad_clipping_ignored_
            # under_fused says so once when max_grad_norm > 0.
            max_grad_norm=0.0,
        )

        # Register hooks
        self.fused_optimizer_groups.register_hooks()

        print(f"{self.log_prefix} Fused optimizer groups setup complete")
        print(f"{self.log_prefix} Optimizer.step() and zero_grad() will be called by hooks automatically")

    def _build_plateau_cosine_floor_scheduler(self, optimizer, total_steps: int):
        """Build a warmup -> plateau -> cosine-decay-to-floor LambdaLR.

        multiplier(step):
          - step < W (warmup): linear ramp step/W, 0 -> 1 (skipped if W == 0)
          - W <= step < D (plateau): 1.0
          - D <= step < T (cosine decay): F + 0.5*(1-F)*(1 + cos(pi*(step-D)/(T-D)))
          - step >= T: F (hold floor forever, never decays to 0)

        W = self.optimizer_warmup_steps, D = round(lr_decay_start_ratio * T),
        T = total_steps (the same value passed as num_training_steps to
        diffusers' get_scheduler() at this call site), F = lr_floor_ratio.

        Built as a plain torch.optim.lr_scheduler.LambdaLR (not a diffusers
        scheduler) so the resume fast-forward (`_fast_forward_lr_schedulers`)
        advances it correctly via last_epoch, exactly like diffusers' own
        LambdaLR-based schedulers.
        """
        from torch.optim.lr_scheduler import LambdaLR

        W = max(0, int(self.optimizer_warmup_steps))
        T = max(1, int(total_steps))
        decay_start_ratio = float(self.config.get("lr_decay_start_ratio", 0.85))
        floor_ratio = float(self.config.get("lr_floor_ratio", 0.25))
        D = round(decay_start_ratio * T)
        # Keep the three segments well-formed even at extreme ratio values.
        D = max(W, min(D, T))

        def lr_lambda(step: int) -> float:
            if W > 0 and step < W:
                return step / float(W)
            if step < D:
                return 1.0
            if step < T:
                span = max(1, T - D)
                progress = (step - D) / float(span)
                return floor_ratio + 0.5 * (1.0 - floor_ratio) * (1.0 + math.cos(math.pi * progress))
            return floor_ratio

        print(f"{self.log_prefix} LR scheduler: plateau_cosine_floor "
              f"(warmup={W}, plateau_end={D}, total={T}, floor_ratio={floor_ratio})")
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ============================================================
    # Weight EMA (opt-in, default off)
    # ============================================================

    def _build_ema_param_name_map(self) -> Dict[int, str]:
        """Map ``id(param) -> real dotted parameter name`` for the EMA shadow.

        Scans this trainer instance's own attributes for ``nn.Module``
        values (e.g. ``self.unet``, ``self.text_encoder``, ``self.controlnet``)
        and dict-of-``nn.Module`` containers (e.g. ``self.lora_layers:
        Dict[str, nn.Module]``), and builds names from each container's own
        ``named_parameters()``. This is deliberately generic (no per-arch/
        per-adapter special-casing) so it works uniformly across every
        trainer subclass without maintenance as new adapters are added, and
        the names it returns are genuine dotted parameter names inside the
        real model objects -- not synthetic group/index placeholders.
        """
        name_map: Dict[int, str] = {}
        for attr_name, attr_val in vars(self).items():
            if isinstance(attr_val, torch.nn.Module):
                for pname, p in attr_val.named_parameters():
                    name_map.setdefault(id(p), f"{attr_name}.{pname}")
            elif isinstance(attr_val, dict):
                for key, sub_val in attr_val.items():
                    if isinstance(sub_val, torch.nn.Module):
                        for pname, p in sub_val.named_parameters():
                            name_map.setdefault(id(p), f"{attr_name}.{key}.{pname}")
        return name_map

    def _setup_ema(self):
        """Initialize the weight-EMA shadow (opt-in via config `use_ema`).

        Must be called after self.optimizer exists (i.e. from setup_optimizer,
        after the optimizer/param groups are built), so the trainable
        parameter set (LoRA adapter params / full-FT params / ControlNet
        params -- whatever setup_trainable_parameters() registered into the
        optimizer) is known.

        Each trainable tensor is keyed by its REAL dotted parameter name
        (via _build_ema_param_name_map(), which walks the actual model
        object(s) this trainer owns), not a synthetic "group{g}.param{i}"
        placeholder -- this is what lets _save_ema_checkpoint() swap EMA
        values directly into the live model objects and reuse the normal,
        arch-specific save_checkpoint() to produce a real, loadable
        checkpoint (see _save_ema_checkpoint).

        Config knobs:
        - ema_decay (float, default 0.9999): per-*applied-update* decay.
        - ema_update_every (int, default 1): only apply the EMA update every
          N optimizer steps (see _update_ema for the decay-power correction
          this implies).
        - ema_device ("cpu" default | "cuda"): where the shadow tensors live.
          "cpu" avoids extra VRAM (adds one GPU->CPU sync per applied
          update); "cuda" keeps the shadow on the parameter's own device
          (no sync, costs ~one extra copy of the trainable params in VRAM).
        """
        self.use_ema = bool(self.config.get("use_ema", False))
        self.ema_decay = float(self.config.get("ema_decay", 0.9999))
        self.ema_update_every = max(1, int(self.config.get("ema_update_every", 1)))
        ema_device = str(self.config.get("ema_device", "cpu")).lower()
        self.ema_device = ema_device if ema_device in ("cpu", "cuda") else "cpu"
        self.ema_shadow: Dict[str, torch.Tensor] = {}
        self._ema_param_order: List[Tuple[str, torch.Tensor]] = []
        self._ema_step_counter = 0

        if not self.use_ema:
            return

        if self.optimizer is None:
            print(f"{self.log_prefix} WARNING: use_ema requested but optimizer is not set up; EMA disabled")
            self.use_ema = False
            return

        name_map = self._build_ema_param_name_map()
        unnamed = 0
        seen_names: set = set()
        for g_idx, group in enumerate(self.optimizer.param_groups):
            for p_idx, param in enumerate(group["params"]):
                if not param.requires_grad:
                    continue
                name = name_map.get(id(param))
                if name is None:
                    unnamed += 1
                    name = f"_unnamed.group{g_idx}.param{p_idx}"
                if name in seen_names:
                    # Extremely unlikely (two distinct containers reusing the
                    # same dotted name) -- disambiguate rather than silently
                    # overwrite a shadow slot.
                    name = f"{name}#g{g_idx}p{p_idx}"
                seen_names.add(name)
                self._ema_param_order.append((name, param))
                shadow_device = param.device if self.ema_device == "cuda" else torch.device("cpu")
                self.ema_shadow[name] = param.detach().to(dtype=torch.float32, device=shadow_device).clone()

        if unnamed:
            print(f"{self.log_prefix} WARNING: {unnamed} trainable EMA tensor(s) could not be matched "
                  f"to a real parameter name; using fallback synthetic names for those only")

        n_params = sum(t.numel() for _, t in self._ema_param_order)
        print(f"{self.log_prefix} Weight EMA enabled: decay={self.ema_decay}, "
              f"update_every={self.ema_update_every}, shadow_device={self.ema_device}, "
              f"{len(self._ema_param_order)} tensors ({format_param_count(n_params)} params)")

    def _update_ema(self):
        """Update the EMA shadow in-place after an optimizer.step().

        No-op unless use_ema is enabled. No-grad, in-place, fp32
        accumulation. Only actually applies every `ema_update_every`
        optimizer steps (default 1 = every step); when skipping steps, the
        decay used for the applied update is raised to the power of
        `ema_update_every` so the EMA's effective averaging horizon
        (~1/(1-decay) steps) stays approximately constant regardless of how
        often the update runs -- e.g. decay=0.9999 with update_every=1 has
        the same ~10,000-step horizon as decay=0.9999**10=0.999 applied
        every 10th step.
        """
        if not getattr(self, "use_ema", False):
            return
        self._ema_step_counter += 1
        if self._ema_step_counter % self.ema_update_every != 0:
            return
        decay = self.ema_decay ** self.ema_update_every
        cuda_shadow = (self.ema_device == "cuda")
        with torch.no_grad():
            for name, param in self._ema_param_order:
                shadow = self.ema_shadow[name]
                if cuda_shadow:
                    new_val = param.detach().to(dtype=torch.float32, non_blocking=True)
                else:
                    new_val = param.detach().to(dtype=torch.float32, device="cpu", non_blocking=True)
                shadow.mul_(decay).add_(new_val, alpha=1.0 - decay)

    def save_ema_state(self, step: int):
        """Save the EMA shadow (for resume), alongside the optimizer state.

        No-op unless use_ema is enabled.
        """
        if not getattr(self, "use_ema", False):
            return
        ema_state_file = self.output_dir / f"{self.run_name}_step_{step:06d}_ema_state.pt"
        torch.save({"decay": self.ema_decay, "shadow": self.ema_shadow}, ema_state_file)
        print(f"{self.log_prefix} Saved EMA shadow state to {ema_state_file.name}")

    def load_ema_state(self, step: int) -> bool:
        """Restore the EMA shadow on resume.

        Design choice: the EMA shadow is resumed from its own dedicated
        state file (`*_ema_state.pt`, written by save_ema_state next to the
        optimizer state), keyed by the same real parameter names
        _setup_ema() just built -- NOT by re-parsing the `_ema`-suffixed
        checkpoint written by _save_ema_checkpoint(). This is the simpler,
        more robust option: the shadow-state file is a flat name->tensor
        dict that trivially round-trips through _ema_param_order, whereas
        reloading a full arch-specific checkpoint would require re-running
        each adapter's load path a second time (once for live weights, once
        for the shadow) purely to recover tensors this dedicated file
        already stores directly.

        If use_ema is on but no saved shadow is found (e.g. EMA was just
        enabled on a run that didn't have it before), the shadow stays at
        its _setup_ema()-time initialization (a fresh copy of the current,
        already-resumed live weights) instead of crashing.
        """
        if not getattr(self, "use_ema", False):
            return False
        ema_state_file = self.output_dir / f"{self.run_name}_step_{step:06d}_ema_state.pt"
        if not ema_state_file.exists():
            print(f"{self.log_prefix} No saved EMA state found at step {step}; "
                  f"re-initializing EMA shadow from current weights")
            return False
        try:
            saved = torch.load(ema_state_file, map_location="cpu")
            shadow = saved.get("shadow", {}) if isinstance(saved, dict) else {}
            restored = 0
            for name, param in self._ema_param_order:
                if name in shadow:
                    shadow_device = param.device if self.ema_device == "cuda" else torch.device("cpu")
                    self.ema_shadow[name] = shadow[name].to(device=shadow_device)
                    restored += 1
            print(f"{self.log_prefix} Restored EMA shadow ({restored}/{len(self._ema_param_order)} "
                  f"tensors) from {ema_state_file.name}")
            return True
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to load EMA state: {e} -- "
                  f"re-initializing EMA shadow from current weights")
            return False

    def _save_ema_checkpoint(self, step: int, epoch: int):
        """Save a REAL, loadable checkpoint of the EMA-averaged weights.

        Reuses the trainer's own save_checkpoint() (the exact same
        arch/method-specific format used for the live checkpoint -- LoRA
        safetensors merge, ControlNet layout, full-FT state dict, sharding,
        etc.) instead of reimplementing any of the 20+ per-arch/per-method
        save formats. Mechanism:

          1. Stash each live trainable parameter's `.data` (clone).
          2. Copy the EMA shadow value into that SAME nn.Parameter object's
             `.data`, in place. These are the exact objects registered with
             the optimizer (see _setup_ema / _build_ema_param_name_map) and
             owned by whatever module save_checkpoint() actually reads from
             (self.lora_layers / self.controlnet / self.unet / ...), so the
             swap is visible to save_checkpoint() with no extra plumbing.
          3. Temporarily set self.run_name to f"{run_name}{EMA_RUN_NAME_SUFFIX}"
             and call self.save_checkpoint(step, epoch) -- every trainer's
             save_checkpoint() builds its output path from self.run_name, so
             this produces a fully separate, fully loadable checkpoint file
             (see EMA_ENTRY_MARKER / EMA_RUN_NAME_SUFFIX for how resume
             detection and rotation cleanup avoid ever picking it up as a
             live checkpoint).
          4. Restore both self.run_name and the live parameter data in a
             `finally`, so a failure anywhere in the save (including inside
             save_checkpoint()) can never leave live training weights
             EMA-corrupted or the run_name permanently altered.
        """
        if not getattr(self, "use_ema", False):
            return
        if not self._ema_param_order:
            return

        stash: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in self._ema_param_order:
                stash[name] = param.data.detach().clone()

        original_run_name = self.run_name
        try:
            with torch.no_grad():
                for name, param in self._ema_param_order:
                    ema_val = self.ema_shadow[name].to(dtype=param.dtype, device=param.device, non_blocking=True)
                    param.data.copy_(ema_val)
            self.run_name = f"{original_run_name}{EMA_RUN_NAME_SUFFIX}"
            self.save_checkpoint(step, epoch)
            print(f"{self.log_prefix} Saved EMA-averaged checkpoint (loadable, run_name={self.run_name})")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to save EMA checkpoint: {e}")
        finally:
            self.run_name = original_run_name
            with torch.no_grad():
                for name, param in self._ema_param_order:
                    param.data.copy_(stash[name])

    # ============================================================
    # Prompt Encoding
    # ============================================================

    def _has_fp8_text_encoder(self) -> bool:
        """
        Check if text encoder has FP8 quantized weights.

        Returns:
            True if any text encoder has FP8 weights
        """
        # Check text_encoder
        if self.text_encoder is not None:
            for module in self.text_encoder.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        return True

        # Check text_encoder_2 (SDXL)
        if self.text_encoder_2 is not None:
            for module in self.text_encoder_2.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        return True

        return False

    def encode_prompt(self, prompt: str, requires_grad: bool = False):
        """
        Encode text prompt to embeddings with chunking support for long prompts (>75 tokens).

        Args:
            prompt: Text prompt to encode
            requires_grad: Whether to enable gradient computation for text encoders

        Returns:
            For SD1.5: text_embeddings tensor
            For SDXL: tuple of (text_embeddings, pooled_embeddings)
        """
        # Safeguard: text encoding on CPU while training on GPU is slow (and usually about
        # to fail with a device mismatch, since inputs go to self.device). Warn once.
        # cpu_prefetch intentionally runs the TE on CPU, so skip it there.
        if (self.text_encoder is not None
                and str(getattr(self, "device", "cpu")) != "cpu"
                and getattr(self, "_text_encoding_mode", None) != "cpu_prefetch"
                and not getattr(self, "_warned_te_cpu", False)):
            try:
                if next(self.text_encoder.parameters()).device.type == "cpu":
                    self._warned_te_cpu = True
                    print(f"{self.log_prefix} WARNING: text encoder is on CPU while the "
                          f"trainer device is {self.device}. Text encoding will be slow or "
                          f"fail with a device mismatch; the text encoder should be on GPU "
                          f"during encoding. (logged once)")
            except StopIteration:
                pass

        # Custom SDXL Text Encoder: bypass CLIP and use the swapped encoder + bridge
        # adapters (returns the SDXL (embeddings[B,L,2048], pooled[B,1280]) contract).
        if self.is_sdxl and getattr(self, "sdxl_te_type", "none") not in ("none", "clip", "", None):
            return self._encode_prompt_custom_te(prompt, requires_grad)

        # DEUS support removed
        # if self.is_deus:
        #     return self._encode_prompt_deus(prompt, requires_grad)

        # Check prompt length - use tokenizer_2 for SDXL as it determines chunking
        tokenizer = self.tokenizer_2 if self.is_sdxl else self.tokenizer
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]

        # If prompt is short (<=75 tokens), use simple encoding
        if len(tokens) <= 75:
            return self._encode_prompt_simple(prompt, requires_grad)

        # Long prompt - use chunking
        return self._encode_prompt_chunked(prompt, requires_grad)

    # DEUS support removed - architecture no longer maintained
    # def _encode_prompt_deus(self, prompt: str, requires_grad: bool = False):
    #     """
    #     Encode prompt using DEUS's SigLIP-2 text encoder.
    #     ...
    #     """
    #     pass

    def _encode_prompt_custom_te(self, prompt: str, requires_grad: bool = False):
        """Encode a prompt with the swapped SDXL text encoder + bridge adapters.

        Returns (embeddings[1,L,2048], pooled[1,1280]). The encoder body is frozen by
        default (run under no_grad); the adapters carry the trainable gradient. When
        sdxl_te_train_encoder is set, the body is also run with grad.

        P4: verbatim body moved to ``ops/sd_sdxl_ops.encode_prompt_custom_te``;
        this stays a thin delegator (called by the ``encode_prompt`` dispatcher).
        """
        from core.training.ops import sd_sdxl_ops
        return sd_sdxl_ops.encode_prompt_custom_te(self, prompt, requires_grad)

    def _encode_prompt_simple(self, prompt: str, requires_grad: bool = False):
        """
        Encode short prompt (<=75 tokens) using standard method.

        P4: verbatim body moved to ``ops/sd_sdxl_ops.encode_prompt_simple``.
        """
        from core.training.ops import sd_sdxl_ops
        return sd_sdxl_ops.encode_prompt_simple(self, prompt, requires_grad)

    def _encode_prompt_chunked(self, prompt: str, requires_grad: bool = False):
        """
        Encode long prompt (>75 tokens) using chunking.
        Splits prompt into 75-token chunks and concatenates embeddings.

        P4: verbatim body moved to ``ops/sd_sdxl_ops.encode_prompt_chunked``.
        """
        from core.training.ops import sd_sdxl_ops
        return sd_sdxl_ops.encode_prompt_chunked(self, prompt, requires_grad)

    def encode_prompt_zimage(
        self,
        prompt: str,
        max_sequence_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt using Qwen3 text encoder with chat template (Z-Image).

        Args:
            prompt: Text prompt
            max_sequence_length: Maximum sequence length

        Returns:
            Tuple of (prompt_embeds, attention_mask)
        """
        # P4: verbatim body moved to ``ops/zimage_ops.encode_prompt``. This stays
        # a thin delegator (called by encode_caption AND the sampling path).
        from core.training.ops import zimage_ops
        return zimage_ops.encode_prompt(self, prompt, max_sequence_length)

    def encode_prompt_anima(self, prompt: str, qwen3_max_length: int = 512,
                             t5_max_length: int = 512):
        """Encode prompt for Anima using the Phase A/B inference pipeline.

        Returns the same dict-style auxiliary payload that the Anima DiT
        forward expects: prompt_embeds (Qwen3 hidden states), source_mask
        (Qwen3 attention mask), t5_input_ids, t5_attn_mask. Caching is
        handled upstream — this method always re-encodes.
        """
        # P4: verbatim body moved to ``ops/anima_ops.encode_prompt``.
        from core.training.ops import anima_ops
        return anima_ops.encode_prompt(self, prompt, qwen3_max_length, t5_max_length)

    def encode_prompt_lens(self, prompt: str, max_length: int = 512):
        """Encode prompt for Lens using the inference encode_prompt function.

        Returns (stacked_features, encoder_mask) where stacked_features is a
        tensor of shape [num_layers, L, enc_hidden_dim] and encoder_mask is
        [L] bool. Each is detached and stored per-sample in the latent cache.
        """
        # P4: verbatim body moved to ``ops/lens_ops.encode_prompt``.
        from core.training.ops import lens_ops
        return lens_ops.encode_prompt(self, prompt, max_length)

    def encode_prompt_ideogram4(self, prompt: str, max_length: int = 512):
        """Encode prompt for Ideogram 4: 13-layer Qwen3-VL hidden states.

        Returns (stacked [1, 13, L, 4096], mask [L]) — same contract as
        encode_prompt_lens so the caching/batching path produces
        [B, 13, L, 4096] / [B, L]. The 13 layers are concatenated to the
        53248-dim conditioning inside train_step_ideogram4.
        """
        # P4: verbatim body moved to ``ops/ideogram4_ops.encode_prompt``.
        from core.training.ops import ideogram4_ops
        return ideogram4_ops.encode_prompt(self, prompt, max_length)

    def encode_prompt_krea2(self, prompt: str, max_length: int = 512):
        """Encode prompt for Krea 2: 12-layer Qwen3-VL hidden-state stack.

        Returns (embeds [1, seq, 12, 2560], mask [seq]) so the cache/batch path
        produces [B, seq, 12, 2560] / [B, seq]. The DiT fuses the layer axis
        internally (text_fusion) inside train_step_krea2 / the forward pass.
        """
        # P4: verbatim body moved to ``ops/krea2_ops.encode_prompt``.
        from core.training.ops import krea2_ops
        return krea2_ops.encode_prompt(self, prompt, max_length)

    def encode_prompt_minit2i(self, prompt: str, requires_grad: bool = False):
        """Encode prompt for MiniT2I: FLAN-T5-Large last_hidden_state + attention mask.

        Returns (embeds [1, L, 1024], mask [L]) so the cache/batch path produces
        [B, L, 1024] / [B, L]. The mask drives mask_token uncond inside the model.

        requires_grad=False (frozen TE): no_grad encode, detached + moved to CPU for
        caching. requires_grad=True (TE training): grad-enabled encode kept on the TE
        device so gradients flow back into FLAN-T5.
        """
        # P4: verbatim body moved to ``ops/minit2i_ops.encode_prompt``.
        from core.training.ops import minit2i_ops
        return minit2i_ops.encode_prompt(self, prompt, requires_grad=requires_grad)

    def encode_caption(self, caption: str, requires_grad: bool = False, lyrics: str = "",
                       reference_image_paths=None):
        """
        Unified caption encoding for all architectures.

        Args:
            caption: The item's caption text.
            requires_grad: Whether to keep a gradient-carrying graph (trainable TE).
            reference_image_paths: SenseNova ONLY -- the item's reference image
                paths, spliced into the prompt prefix as understanding-tower
                tokens (``ops/sensenova_ops.encode_prompt``). Every other arch
                ignores this argument; FLUX.2 conditions on references through
                VAE latents in the train loop instead.
            lyrics: ACE-Step ONLY -- the item's per-item lyrics text ("" for
                instrumental / no-lyrics, the default; every other arch ignores
                this argument). See ``ops/acestep_ops.py``'s module docstring.

        Returns:
            Tuple of (embeddings, auxiliary_data):
            - Z-Image: (prompt_embeds, attention_mask)
            - SD1.5: (text_embeddings, None)
            - SDXL: (text_embeddings, pooled_embeddings)
            - FLUX.2: (prompt_embeds, None) - text_ids computed in train_step
            - Anima: (prompt_embeds, anima_aux_dict) where aux dict has
              {source_mask, t5_input_ids, t5_attn_mask}
            - Lens: (stacked_features [num_layers, L, D], encoder_mask [L])
            - ACE-Step: (text_hidden_states, aux_dict) where aux dict has
              {text_attention_mask, lyric_hidden_states, lyric_attention_mask}
        """
        if self.is_zimage:
            return self.encode_prompt_zimage(caption)
        elif self.is_sensenova:
            return self.arch.encode_prompt(
                self,
                caption,
                requires_grad=requires_grad,
                reference_image_paths=reference_image_paths,
            ), None
        elif self.is_lens:
            return self.encode_prompt_lens(caption)
        elif self.is_ideogram4:
            return self.encode_prompt_ideogram4(caption)
        elif self.is_minit2i:
            return self.encode_prompt_minit2i(caption, requires_grad=requires_grad)
        elif self.is_krea2:
            return self.encode_prompt_krea2(caption)
        elif self.is_anima:
            payload = self.encode_prompt_anima(caption)
            # Return the Qwen3 hidden states as the primary embedding plus the
            # rest as a dict so callers can hand them to train_step_anima as
            # a single bundle.
            return payload["prompt_embeds"], {
                "source_mask": payload["source_mask"],
                "t5_input_ids": payload["t5_input_ids"],
                "t5_attn_mask": payload["t5_attn_mask"],
            }
        elif self.is_ltx2:
            # LTX-2.3: post-connector video text embedding + aux dict
            # {audio_text_embedding, mask} handed to train_step_ltx2 as a bundle
            # (mirrors anima's payload contract).
            from core.training.ops import ltx2_ops
            video_emb, aux = ltx2_ops.encode_prompt(self, caption)
            return video_emb, aux
        elif self.is_minimax_h3:
            # MiniMax-H3: Qwen3-VL layer-50 hidden states + a one-key aux dict
            # {num_text_tokens}. There is no text mask and no per-modality
            # connector -- the caption's own rows are packed into the attended
            # sequence, so its TOKEN COUNT is what train_step needs (and what the
            # batch assembly's zero-padding would otherwise destroy).
            from core.training.ops import minimax_h3_ops
            text_emb, aux = minimax_h3_ops.encode_prompt(self, caption)
            return text_emb, aux
        elif self.is_acestep:
            # ACE-Step: Qwen3 "# Caption" hidden states + aux dict
            # {text_attention_mask, lyric_hidden_states, lyric_attention_mask}
            # handed to train_step_acestep as a bundle (mirrors ltx2's payload
            # contract). `lyrics` is per-item (see docstring above).
            from core.training.ops import acestep_ops
            text_emb, aux = acestep_ops.encode_prompt(self, caption, lyrics=lyrics)
            return text_emb, aux
        elif self.is_flux2:
            # FLUX.2: Use Qwen3 text encoder with hidden state extraction
            # Note: text_ids are generated dynamically in train_step_flux2, not cached
            prompt_embeds, _ = self._flux2_encode_prompt(caption)
            return prompt_embeds, None  # text_ids are computed in train_step
        elif self.is_sdxl:
            text_emb, pooled_emb = self.encode_prompt(caption, requires_grad=requires_grad)
            return text_emb, pooled_emb
        else:
            text_emb = self.encode_prompt(caption, requires_grad=requires_grad)
            return text_emb, None

    @staticmethod
    def _aux_to_cpu(auxiliary_data):
        """Move caption auxiliary data to CPU, tolerating every arch's shape.

        auxiliary_data is a tensor (SDXL pooled embeds, Z-Image attention mask),
        a DICT of tensors (Anima: source_mask/t5_input_ids/t5_attn_mask), or
        None (SD1.5). The swap/prefetch buffers previously assumed a tensor and
        crashed on Anima with 'dict' object has no attribute 'cpu'.
        """
        if auxiliary_data is None:
            return None
        if isinstance(auxiliary_data, dict):
            return {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in auxiliary_data.items()}
        return auxiliary_data.cpu()

    def _aux_to_device(self, auxiliary_data, non_blocking: bool = True):
        """Inverse of _aux_to_cpu: move auxiliary data to self.device."""
        if auxiliary_data is None:
            return None
        if isinstance(auxiliary_data, dict):
            return {k: (v.to(self.device, non_blocking=non_blocking)
                        if isinstance(v, torch.Tensor) else v)
                    for k, v in auxiliary_data.items()}
        return auxiliary_data.to(self.device, non_blocking=non_blocking)

    # ``_collate_anima_aux`` moved to ``ops/anima_ops.collate_aux`` (plan P4).
    # Call sites in the train loop now dispatch via ``self.arch.collate_aux``;
    # only the anima handler overrides the base_arch no-op default.

    def encode_captions_batched(self, captions, requires_grad: bool = False, lyrics=None):
        """Encode a list of captions in one forward pass when possible.

        Returns a list of (embedding, auxiliary_data) tuples in the same
        order as the input — drop-in compatible with calling encode_caption
        per sample. The CPU prefetch worker (Phase F) uses this to amortise
        per-call overhead so the frozen TE on CPU keeps up with GPU iters.

        Anima has a true batched path (single Qwen3 forward); the other
        architectures currently fall back to per-sample encode_caption.
        Override in subclasses that can add a real batched implementation.

        Args:
            lyrics: ACE-Step ONLY -- an optional list of per-item lyrics
                strings, parallel to ``captions`` (same length/order). ``None``
                (default) is treated as all-``""`` (every other arch ignores
                this argument entirely).
        """
        if not captions:
            return []

        if self.is_anima:
            from core.models.anima.anima_pipeline_ops import encode_prompts_batched
            results = encode_prompts_batched(
                text_encoder=self.text_encoder,
                qwen3_tokenizer=self.tokenizer,
                t5_tokenizer=self.t5_tokenizer,
                prompts=list(captions),
                device=str(self.text_encoder.device) if hasattr(self.text_encoder, "device") else "cpu",
                dtype=self.training_dtype,
                qwen3_max_length=512,
                t5_max_length=512,
            )
            out = []
            for r in results:
                out.append((
                    r["prompt_embeds"].detach(),
                    {
                        "source_mask": r["source_mask"].detach(),
                        "t5_input_ids": r["t5_input_ids"].detach(),
                        "t5_attn_mask": r["t5_attn_mask"].detach(),
                    },
                ))
            return out

        # Fallback: per-sample. No speedup but correct for any arch.
        if lyrics is not None:
            if len(lyrics) != len(captions):
                raise ValueError(
                    f"[encode_captions_batched] lyrics list length ({len(lyrics)}) "
                    f"!= captions list length ({len(captions)})"
                )
            return [
                self.encode_caption(c, requires_grad=requires_grad, lyrics=(lyr or ""))
                for c, lyr in zip(captions, lyrics)
            ]
        return [self.encode_caption(c, requires_grad=requires_grad) for c in captions]

    # ============================================================
    # VRAM Management (swap mode for all architectures)
    # ============================================================

    def move_text_encoder_to_gpu(self):
        """Move Text Encoder(s) to GPU for encoding."""
        if self.is_minimax_h3:
            # MiniMax-H3's Qwen3-VL conditioner is NEVER moved, in either
            # direction. `.to()` detaches all 902 of its tensors from the file
            # mapping and turns a memory-mapped 48 GiB module into an anonymous
            # resident copy (73.08 GB peak RSS against 49.82 GB, MEASURED) -- and
            # it does not fit in VRAM in any case. Its GPU work is done one
            # decoder layer at a time by `h3_pipeline_ops.encode_prompt` via
            # `torch.func.functional_call`, which never writes back.
            return
        if self.is_lens:
            # Lens mxfp4 TE: .to('cpu') cannot free the kernels CUDA buffers, so
            # move_text_encoder_to_cpu() deletes the object entirely.  Reload here
            # if it was previously freed.
            if self.text_encoder is None:
                from core.models.lens.lens_loader import reload_lens_text_encoder
                transformer = getattr(self, "transformer", None) or getattr(self, "transformer_original", None)
                selected_layers = (
                    tuple(transformer.config.selected_layer_index)
                    if transformer is not None and hasattr(transformer, "config")
                    and hasattr(transformer.config, "selected_layer_index") else None
                )
                print(f"[Lens TE] Reloading text encoder for swap (mxfp4, ~4 s)...")
                self.text_encoder = reload_lens_text_encoder(
                    self.model_path,
                    torch_dtype=getattr(self, "weight_dtype", torch.bfloat16),
                    selected_layers=selected_layers,
                )
            # The mxfp4 kernels library allocates CUDA memory during from_pretrained;
            # the non-quantised params live on CPU and are moved here for encoding.
            self.text_encoder.to(self.device)
            return

        if self.text_encoder is not None:
            # SD/SDXL: CLIPTextModel
            self.text_encoder.to(self.device)
            # Ensure embedding layer stays on GPU (critical for gradient checkpointing)
            if hasattr(self.text_encoder, 'text_model') and hasattr(self.text_encoder.text_model, 'embeddings'):
                self.text_encoder.text_model.embeddings.to(self.device)
        if self.is_sdxl and self.text_encoder_2 is not None:
            self.text_encoder_2.to(self.device)
            # Ensure embedding layer stays on GPU (critical for gradient checkpointing)
            if hasattr(self.text_encoder_2, 'text_model') and hasattr(self.text_encoder_2.text_model, 'embeddings'):
                self.text_encoder_2.text_model.embeddings.to(self.device)

    def move_text_encoder_to_cpu(self):
        """Move Text Encoder(s) to CPU to free VRAM."""
        if self.is_minimax_h3:
            # Already CPU/memory-mapped and must stay that way -- see
            # move_text_encoder_to_gpu.
            return
        if self.is_lens:
            # .to('cpu') only moves the non-quantised PyTorch params; the kernels
            # library FP4 CUDA buffers (~9.7 GB) remain allocated.  The only way
            # to release them is to delete the object and GC.
            import gc as _gc
            self.text_encoder = None
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        if self.text_encoder is not None:
            self.text_encoder.to("cpu")
        if self.is_sdxl and self.text_encoder_2 is not None:
            self.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    def move_main_model_to_gpu(self):
        """Move main model (U-Net or Transformer) to GPU for training."""
        if (
            self.is_sensenova
            and getattr(self, "sensenova_phase_evictor", None) is not None
        ):
            return
        if self.is_zimage or self.is_anima or self.is_lens or self.is_ideogram4 or self.is_minit2i or self.is_krea2 or self.is_ltx2 or self.is_acestep or self.is_minimax_h3 or self.is_sensenova:
            if self.transformer_original is not None:
                self.transformer_original.to(self.device)
        else:
            if self.unet is not None:
                self.unet.to(self.device)

    def move_main_model_to_cpu(self):
        """Move main model (U-Net or Transformer) to CPU to free VRAM."""
        if (
            self.is_sensenova
            and getattr(self, "sensenova_phase_evictor", None) is not None
        ):
            return
        if self.is_zimage or self.is_anima or self.is_lens or self.is_ideogram4 or self.is_minit2i or self.is_krea2 or self.is_ltx2 or self.is_acestep or self.is_minimax_h3 or self.is_sensenova:
            if self.transformer_original is not None:
                self.transformer_original.to("cpu")
        else:
            if self.unet is not None:
                self.unet.to("cpu")
        torch.cuda.empty_cache()

    def move_vae_to_gpu(self):
        """Move VAE to GPU for encoding/decoding."""
        if self.vae is not None:
            self.vae.to(device=self.device, dtype=self.vae_dtype)

    def move_vae_to_cpu(self):
        """Move VAE to CPU to free VRAM."""
        if self.vae is not None:
            self.vae.to(device="cpu", dtype=self.vae_dtype)
        torch.cuda.empty_cache()

    def _ve_set_device(self, device):
        """Move the Vision Encoder model AND its AdamW optimizer state to `device` together,
        so optimizer steps stay device-consistent.

        Used to offload the trained VE (params + state) during reference-free batches and
        reload it before a reference batch's step. The optimizer-state move is a no-op until
        the VE has actually been stepped (AdamW allocates exp_avg/exp_avg_sq lazily on the
        first gradient), so a run that never uses the VE pays nothing — and for it the state
        offload below frees the ~743MB (92.9M params x 2 x fp32) it would otherwise hold once
        the VE is exercised.
        """
        ve = getattr(self, "vision_encoder", None)
        if ve is None:
            return
        ve.to(device if isinstance(device, str) else str(device))
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return
        try:
            ve_params = set(ve.parameters())
            for p, st in opt.state.items():
                if p in ve_params:
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(device)
        except Exception:
            pass

    def _main_model_module(self):
        """Return the trainable main-model module (Transformer for DiT archs, else U-Net).

        Mirrors the arch dispatch in move_main_model_to_cpu/gpu so the three stay
        consistent. Returns None if the module is not present.
        """
        if self.is_zimage or self.is_anima or self.is_lens or self.is_ideogram4 or self.is_minit2i or self.is_krea2 or self.is_ltx2 or self.is_acestep or self.is_minimax_h3 or self.is_sensenova:
            return getattr(self, "transformer_original", None)
        return getattr(self, "unet", None)

    def _relocate_main_model_optimizer_state(self, device):
        """Move the optimizer's state tensors for main-model params to `device`.

        Mirrors _ve_set_device's optimizer-state handling so optimizer steps stay
        device-consistent when the main model is offloaded for a VAE encode phase.
        Matches by parameter identity, so it covers both full-finetune (all main-model
        params trained) and LoRA (only the injected adapter params are in optimizer.state,
        and those are still parameters of the main model module). It is a no-op until the
        optimizer exists and has actually been stepped (Adam-family state is allocated
        lazily on the first gradient), so a fresh run's pre-training encode pays nothing;
        it matters for resumed runs whose loaded state is GPU-resident.
        """
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return
        try:
            model = self._main_model_module()
            if model is None:
                return
            main_params = set(model.parameters())
            if not main_params:
                return
            for p, st in opt.state.items():
                if p in main_params:
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(device)
        except Exception:
            pass

    def _relocate_text_encoder_optimizer_state(self, device):
        """Move the optimizer's state tensors for text-encoder params to `device`.

        Companion to _relocate_main_model_optimizer_state for the text encoders
        (self.text_encoder / self.text_encoder_2 — the same modules
        move_text_encoder_to_cpu/gpu relocate). Used when the TEs are parked on CPU
        for a VAE-only encode phase so their Adam/optimizer state (fp32 m/v — as large
        as the trained params) does not stay pinned on the GPU beside the VAE. Matches
        by parameter identity, so it is scoped to whatever TE params are actually in the
        optimizer (full-finetune with train_text_encoder; empty otherwise) and is a
        no-op until the optimizer exists and has been stepped (Adam-family state is
        allocated lazily on the first gradient). No-op for frozen / cached-TE setups.
        """
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return
        try:
            te_params = set()
            if getattr(self, "text_encoder", None) is not None:
                te_params.update(self.text_encoder.parameters())
            if getattr(self, "is_sdxl", False) and getattr(self, "text_encoder_2", None) is not None:
                te_params.update(self.text_encoder_2.parameters())
            if not te_params:
                return
            for p, st in opt.state.items():
                if p in te_params:
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(device)
        except Exception:
            pass

    # ============================================================
    # Image Encoding
    # ============================================================

    def encode_image(
        self,
        image: Image.Image,
        target_size: int = 512,
        target_width: int = None,
        target_height: int = None,
        bucket_strategy: str = "crop",
        crop_box: Optional[Tuple[int, int, int, int]] = None,
        time_ids_override: Optional[Tuple[int, int, int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Encode image to latents.

        Args:
            image: PIL Image
            target_size: Square target size (deprecated, use target_width/height)
            target_width: Target width (for bucketing)
            target_height: Target height (for bucketing)
            bucket_strategy: Strategy for fitting image to target size
                - "resize": Direct resize (may distort aspect ratio, fastest)
                - "crop": Aspect ratio preserving resize + center crop (default)
                - "random_crop": Random crop at original resolution (no downscale, for tiled inference training)
            crop_box: Optional (cx, cy, cw, ch) in original pixels. When provided (used by
                the epoch-dynamic CropPlanner), the exact region is cropped from the
                original then resized to (target_width, target_height), bypassing
                bucket_strategy. time_ids_override is used verbatim for micro-conditioning.
            time_ids_override: Optional kohya-style SDXL time_ids
                (orig_h, orig_w, crop_top, crop_left, target_h, target_w) to record for
                this encode (paired with crop_box).

        Returns:
            Latent tensor
        """
        image = flatten_to_rgb(image)

        # Determine target dimensions
        if target_width is not None and target_height is not None:
            width, height = target_width, target_height
        else:
            width, height = target_size, target_size

        img_width, img_height = image.size

        # SDXL micro-conditioning capture: the real source size + the crop top-left
        # (in resized-image space). Set once after the bucket branch below, so it is
        # available regardless of which model-type return path runs. time_ids order is
        # [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
        orig_w, orig_h = img_width, img_height
        crop_left, crop_top = 0, 0

        # Planner-provided time_ids for the crop_box path (None for the strategy path).
        _microcond_override = None

        if crop_box is not None:
            # Epoch-dynamic crop path (CropPlanner): crop the exact region from the
            # original and resize to the target bucket, bypassing bucket_strategy. The
            # micro-conditioning time_ids come from the planner (time_ids_override).
            cx, cy, cw, ch = crop_box
            cx = max(0, min(cx, img_width - 1))
            cy = max(0, min(cy, img_height - 1))
            cw = max(1, min(cw, img_width - cx))
            ch = max(1, min(ch, img_height - cy))
            region = image.crop((cx, cy, cx + cw, cy + ch))
            if region.size != (width, height):
                region = region.resize((width, height), Image.LANCZOS)
            image = region
            crop_left, crop_top = cx, cy
            _microcond_override = (
                tuple(time_ids_override) if time_ids_override is not None
                else (orig_h, orig_w, cy, cx, height, width)
            )
        else:
            if img_width * img_height > 5000 * 5000:
                print(f"[encode_image] Resizing large image {img_width}x{img_height} -> {width}x{height}")

            # Apply bucketing strategy
            if bucket_strategy == "resize":
                # Direct resize (may distort aspect ratio)
                image = image.resize((width, height), Image.LANCZOS)

            elif bucket_strategy == "crop":
                # Aspect ratio preserving resize + center crop (default)
                scale = max(width / img_width, height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                image = image.resize((new_width, new_height), Image.LANCZOS)

                # Center crop
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                crop_left, crop_top = left, top
                image = image.crop((left, top, left + width, top + height))

            elif bucket_strategy == "random_crop":
                # Random crop at original resolution (no resize)
                # Enables model to learn inference on partial regions of large images (for tiled inference)
                import random

                # If image is smaller than target, resize it first
                if img_width < width or img_height < height:
                    scale = max(width / img_width, height / img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    img_width, img_height = new_width, new_height

                # Random crop from original (or upscaled) resolution
                max_left = img_width - width
                max_top = img_height - height
                left = random.randint(0, max_left) if max_left > 0 else 0
                top = random.randint(0, max_top) if max_top > 0 else 0
                crop_left, crop_top = left, top
                image = image.crop((left, top, left + width, top + height))

            else:
                raise ValueError(f"Unknown bucket_strategy: {bucket_strategy}. Must be 'resize', 'crop', or 'random_crop'")

        if image.size != (width, height):
            print(f"[encode_image] ERROR: Final image size {image.size} != target {(width, height)}")

        # SDXL micro-conditioning for this encode: (orig_h, orig_w, crop_top, crop_left,
        # target_h, target_w). Read by the batch loop right after encode_image() and
        # carried per-item into the SDXL time_ids (replacing the hardcoded values).
        # crop_box path uses the planner-provided override.
        self._last_micro_cond = (
            _microcond_override if _microcond_override is not None
            else (orig_h, orig_w, crop_top, crop_left, height, width)
        )

        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = (image_array - 0.5) * 2.0

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

        # minit2i is dispatched BEFORE the shared VAE staging below: pixel-space has
        # no VAE, so ``next(self.vae.parameters())`` must not run. Its handler is
        # fully self-contained (pixel no-VAE + latent early-return paths) and returns
        # the final CPU/training-dtype tensor directly (no shared post-amble). The two
        # sub-branch bodies moved VERBATIM to ops/minit2i_ops.vae_encode (P5).
        if self.is_minit2i or self.is_sensenova:
            return self.arch.vae_encode(self, image_tensor, image=image, width=width, height=height)

        vae_device = next(self.vae.parameters()).device
        # Safeguard: VAE encoding on CPU while training on GPU is a silent, catastrophic
        # slowdown (GPU idle for minutes per image). encode_image follows the VAE's device,
        # so a VAE left on CPU (e.g. after sample generation) goes unnoticed. Warn once.
        if (vae_device.type == "cpu" and str(getattr(self, "device", "cpu")) != "cpu"
                and not getattr(self, "_warned_vae_cpu", False)):
            self._warned_vae_cpu = True
            print(f"{self.log_prefix} WARNING: VAE latent encoding is running on CPU while "
                  f"the trainer device is {self.device}. This is extremely slow (GPU idle). "
                  f"The VAE should be on GPU during encoding. (logged once)")
        # Match VAE dtype to prevent type mismatch errors
        image_tensor = image_tensor.to(device=vae_device, dtype=self.vae_dtype)

        # DEBUG: Log preprocessing
        debug_preprocessing = False  # Set to True to debug latent encoding
        if debug_preprocessing:
            print(f"[encode_image DEBUG] Image tensor before VAE:")
            print(f"  Shape: {image_tensor.shape}, dtype: {image_tensor.dtype}, device: {image_tensor.device}")
            print(f"  Mean: {image_tensor.mean():.6f}, Std: {image_tensor.std():.6f}")
            print(f"  Min: {image_tensor.min():.6f}, Max: {image_tensor.max():.6f}")

        # Encode to latents. The 7 VAE archs' per-arch branch bodies moved VERBATIM
        # to ops/<arch>_ops.vae_encode (P5), dispatched through the arch handler. The
        # branch runs under this no_grad and returns raw latents (still on vae_device);
        # the shared post-amble below performs the final dtype/CPU move. All
        # encode_image call sites (cache pre-encode / train loop / sampling) run
        # post-__init__, so self.arch is always bound here.
        with torch.no_grad():
            latents = self.arch.vae_encode(
                self, image_tensor, image=image, width=width, height=height,
                vae_device=vae_device, debug_preprocessing=debug_preprocessing,
            )

        # Clean up image_tensor before moving latents to CPU
        del image_tensor

        # Convert to training dtype and move to CPU immediately to free VRAM
        latents = latents.to(dtype=self.training_dtype, device='cpu')

        # DEBUG: Log final latents after dtype conversion
        if debug_preprocessing:
            print(f"[encode_image DEBUG] Final latents (after dtype={self.training_dtype}, device=cpu):")
            print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")
            print(f"  Min: {latents.min():.6f}, Max: {latents.max():.6f}")

        return latents

    # ============================================================
    # OOM Recovery: Batch Splitting
    # ============================================================

    # Substrings that unambiguously indicate a recoverable allocation failure
    # (retrying with a smaller batch / offload can succeed).
    _CUDA_OOM_MARKERS = (
        "out of memory", "cannot allocate memory", "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed", "cufft_alloc_failed",
    )
    # Substrings that indicate the CUDA context itself is corrupted (sticky --
    # every subsequent CUDA call on this process will keep failing). Retrying
    # is futile; only CPU-side/host state can still be salvaged.
    _CUDA_FATAL_MARKERS = (
        "unspecified launch failure", "illegal memory access", "device-side assert",
        "misaligned address", "uncorrectable ecc", "ecc error", "launch timed out",
        "invalid device context", "driver shutting down",
    )

    def _cuda_is_available(self) -> bool:
        """Instance-method seam around ``torch.cuda.is_available()``.

        Callers that need "is there a CUDA context to be alive/dead at all"
        (as opposed to the CUDA-context-alive canary itself) go through this
        instead of the bare module call, so a test can override it on a stub
        independently of whether the machine running the test actually has a
        GPU (see fused_partial_step_oom_test.py).
        """
        return torch.cuda.is_available()

    @staticmethod
    def _cuda_context_alive() -> bool:
        """Cheap canary probe: can we still issue CUDA ops on this process?

        Used to resolve AMBIGUOUS CUDA error strings (bare "cuda error",
        "cublas_status_execution_failed", or bare cudnn/cusparse/cufft mentions
        without an alloc-failure marker) that could be either a transient OOM
        or a dead context. Side-effect-free beyond a tiny allocation.
        """
        try:
            if not torch.cuda.is_available():
                return False
            return bool((torch.zeros(8, device="cuda") + 1).sum().item() == 8)
        except Exception:
            return False

    @classmethod
    def _classify_cuda_error(cls, e: Exception) -> str:
        """Classify an exception as "oom" (recoverable), "fatal" (CUDA context
        presumed dead, do not retry), or "not_cuda" (unrelated to CUDA).

        IMPORTANT: this is the single source of truth for CUDA-error triage --
        every call site (proactive/reactive OOM recovery, the outer safety
        net, the emergency-save handler) must go through this classifier so
        "oom" behavior stays byte-for-byte identical to before this change.
        """
        try:
            if isinstance(e, torch.OutOfMemoryError):
                return "oom"
        except Exception:
            pass
        s = str(e).lower()
        if any(m in s for m in cls._CUDA_OOM_MARKERS):
            return "oom"
        if any(m in s for m in cls._CUDA_FATAL_MARKERS):
            return "fatal"
        # Ambiguous residue: a CUDA-flavored message with neither an explicit
        # alloc-failure nor a known-fatal marker (e.g. bare "cuda error",
        # "cublas_status_execution_failed", or bare cudnn/cusparse/cufft).
        is_ambiguous_cuda = (
            "cuda error" in s or "cublas" in s or "cudnn" in s or "cusparse" in s or "cufft" in s
        )
        if not is_ambiguous_cuda:
            return "not_cuda"
        return "oom" if cls._cuda_context_alive() else "fatal"

    @classmethod
    def _is_cuda_oom(cls, e: Exception) -> bool:
        """Back-compat wrapper: True only for the recoverable "oom" class."""
        return cls._classify_cuda_error(e) == "oom"

    def _cleanup_incomplete_step_checkpoint_dir(self, step: int) -> None:
        """Best-effort: remove a DIRECTORY-style checkpoint for ``step`` if it
        exists but has no weights file (e.g. diffusers ``save_pretrained``
        mkdir'd the directory but a dead/erroring CUDA context aborted the
        write before any weights were serialized -- see
        ``controlnet_sdxl_adapter._save_standard_checkpoint``).

        Only directories are considered; single-file checkpoints
        (``*.safetensors``) are effectively atomic writes and are left alone.
        Silently no-ops if nothing needs cleaning or on any error -- this is
        a hygiene step for resume, never allowed to mask the original error.
        """
        try:
            pattern = f"{self.run_name}*_step_{step:06d}"
            for p in self.output_dir.glob(pattern):
                if not p.is_dir():
                    continue
                has_weights = (
                    (p / "diffusion_pytorch_model.safetensors").exists()
                    or (p / "diffusion_pytorch_model.safetensors.index.json").exists()
                )
                if not has_weights:
                    import shutil
                    shutil.rmtree(p, ignore_errors=True)
                    print(f"{self.log_prefix} [EMERGENCY] Removed incomplete checkpoint dir: {p.name}")
        except Exception:
            pass

    def _oom_recovery_cleanup(self) -> None:
        """Release tensors/cache after a CUDA OOM so a retry starts clean."""
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        cond = getattr(self, "layer_offload_conductor", None)
        if cond is not None:
            try:
                cond.clear_activations()
            except Exception:
                pass
        flx = getattr(self, "flux2_block_offloader", None)
        if flx is not None:
            try:
                flx.clear_activations()
            except Exception:
                pass
        gc.collect()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _ltx2_batch_fps_tensor(batch):
        """Build the per-sample LTX-2.3 clip fps tensor ``[B]`` from the batch
        ITEMS (aligned to batch/latent order).

        fps is a property of the VIDEO CLIP, not the caption, so it is threaded
        from the dataset item -> batch -> collated aux -> TrainStepContext here
        (NOT via the per-caption ``_ltx2aux.pt`` text cache). VideoBucketManager
        groups by (spatial_bucket, clip_length) — NOT by fps — so a batch may
        mix fps; this yields the real per-sample value. Stills / items without a
        recorded fps fall back to the LTX default (24.0), which is irrelevant to
        a T=1 clip's single temporal RoPE position. Carried as a torch.Tensor so
        the OOM micro-batch splitter (``_slice_aux``) slices it by [lo:hi].
        """
        from core.training.ops.ltx2_ops import _DEFAULT_FPS
        vals = []
        for item, _dataset in batch:
            # A clip RESAMPLED to a fixed rate (Phase 6a) plays at target_fps,
            # not at the source rate — the model must be told the rate of the
            # frames it is actually given. LTX-2.3 items never carry
            # ``target_fps``, so this reads ``fps`` exactly as before.
            v = item.get("target_fps") or item.get("fps")
            vals.append(float(v) if v else _DEFAULT_FPS)
        return torch.tensor(vals, dtype=torch.float32)

    @staticmethod
    def _minimax_h3_batch_audio(batch):
        """Build MiniMax-H3's per-sample AUDIO payload from the batch ITEMS.

        The window's audio latent belongs to the sampled CLIP, not to the caption
        (design section 10 / audit 5.2), so it never travels in the per-caption
        text aux. It is stashed on the item by ``_encode_video_clip`` /
        ``_load_or_encode_video_clip`` -- the same call that produced that
        window's VIDEO latent, from the SAME timestamps -- and collected here in
        batch order.

        Returns ``{"audio_latents": [B, 2*T_aud, 32] or absent,
        "audio_present": [B] bool}``. A silent or audio-less source contributes
        zero rows and a False flag; ``train_step`` then feeds that sample noise
        audio rows and excludes it from the audio loss, rather than training the
        audio head on silence that was never in the file.
        """
        # EVERYTHING is brought to the CPU before stacking. The two producers
        # disagree about device by construction: a cache HIT comes back on
        # `self.device` (`load_clip_record` -> `torch.load(map_location=device)`)
        # while a MISS returns `.cpu()` (the audio VAE encode), so a batch that
        # mixes the two -- or mixes a source that has audio with one that does
        # not, whose filler this function allocates -- would raise inside
        # `torch.stack` mid-run. `train_step` moves the result to the training
        # device anyway, so the CPU is both the safe and the free choice.
        mats, present = [], []
        for item, _dataset in batch:
            lat = item.get("_clip_audio_latent")
            ok = isinstance(lat, torch.Tensor)
            present.append(ok)
            mats.append(lat.detach().cpu() if ok else None)
        out = {"audio_present": torch.tensor(present, dtype=torch.bool)}
        real = [m for m in mats if m is not None]
        if not real:
            return out
        rows = max(m.shape[0] for m in real)
        cols = real[0].shape[1]
        stacked = []
        for m in mats:
            if m is None:
                stacked.append(torch.zeros(rows, cols, dtype=real[0].dtype))
            elif m.shape[0] != rows:
                # Reachable when a window at the very end of a source yields a
                # short audio read (the clip span and the clip duration differ by
                # one frame); the sample's rows are zero-padded to the batch shape
                # and its `audio_present` flag still says the audio is real, which
                # is what the loss reads.
                pad = torch.zeros(rows - m.shape[0], cols, dtype=m.dtype)
                stacked.append(torch.cat([m, pad], dim=0))
            else:
                stacked.append(m)
        out["audio_latents"] = torch.stack(stacked, dim=0)
        return out

    @staticmethod
    def _slice_aux(aux, lo, hi):
        """Slice a per-batch auxiliary payload by [lo:hi] for micro-batching.

        The auxiliary payload (carried in mnt_attention_mask) can be:
          - None                -> None
          - a torch.Tensor      -> aux[lo:hi]
          - a dict of tensors   -> {k: v[lo:hi] if tensor else v}  (anima)
        Non-tensor dict values are passed through unchanged.
        """
        if aux is None:
            return None
        if isinstance(aux, dict):
            return {
                k: (v[lo:hi] if isinstance(v, torch.Tensor) else v)
                for k, v in aux.items()
            }
        if isinstance(aux, torch.Tensor):
            return aux[lo:hi]
        return aux

    @staticmethod
    def _collate_sensenova_b1_prefix(prefixes: List[Any]) -> Any:
        """Return the one opaque prefix allowed in a physical SenseNova batch."""
        if len(prefixes) != 1:
            raise ValueError("SenseNova B1 collation requires exactly one prompt prefix")
        return prefixes[0]

    def _sensenova_mnt_conditioning(
        self,
        prefix: Any,
        *,
        captions: Optional[List[str]] = None,
        mnt_index: int = 0,
    ):
        """Build the opaque conditioning payload for one MNT iteration.

        A frozen understanding branch reuses the same detached prefix every
        iteration. A TRAINABLE one cannot: the MNT loop steps the optimizer once
        per iteration, so reusing the graph (``retain_graph``) would either trip
        the version counter or backpropagate against stale parameters. The
        prefix is therefore recomputed per iteration, the same resolution
        ``need_recompute_text_embeddings`` applies to the other architectures'
        trainable text encoders.

        The SHARED-WINDOW four-phase route is the exception, and reuses the
        prefix for the same reason the frozen branch does. What arrives there is
        not a graph but the boundary LEAVES, which no optimizer step can stale
        because the understanding half is bit-identically invariant until phase 3
        runs at the window's end; every iteration's backward accumulates into
        those same leaves rather than building a second set.
        """
        four_phase = getattr(self, "sensenova_four_phase", None)
        if four_phase is not None and four_phase.shared_window:
            return None, None, None, prefix
        if (
            mnt_index > 0
            and bool(getattr(self, "train_text_encoder", False))
            and captions
        ):
            prefix, _ = self.encode_caption(captions[0], requires_grad=True)
        return None, None, None, prefix

    def _microbatch_two_stage(self, micro_bs: int, eff_bs: int, b: dict):
        """Run a batch (the _execute_forward_backward args in dict ``b``) as
        micro-chunks of size ``micro_bs`` with gradient accumulation, returning
        (loss, pred, recon) full-batch weighted means.

        Per-chunk loss is scaled by chunk/eff_bs so the accumulated gradient equals
        the full-batch mean gradient. Inputs that carry a live encoder graph
        (on-the-fly TE/VE training) are run on DETACHED LEAF copies per chunk and
        back-propagated through the encoder ONCE at the end (two-stage), so the
        shared graph isn't freed mid-loop. Used by both the proactive escalate
        decision and the reactive OOM retry.
        """
        batch_size = b["mnt_latents"].shape[0]
        graph_names = ("mnt_latents", "mnt_text_embeddings",
                       "mnt_pooled_embeddings", "mnt_repa_pixels")
        graph_inputs = {n: b[n] for n in graph_names
                        if isinstance(b.get(n), torch.Tensor) and b[n].grad_fn is not None}
        grad_acc = {n: torch.zeros_like(t) for n, t in graph_inputs.items()}
        loss_acc = pred_acc = recon_acc = 0.0
        for lo in range(0, batch_size, micro_bs):
            hi = min(lo + micro_bs, batch_size)
            w = hi - lo

            def _sl(name):
                full = b.get(name)
                if full is None:
                    return None
                if name in graph_inputs:
                    return full[lo:hi].detach().requires_grad_(True)
                return full[lo:hi]

            leaves = {n: _sl(n) for n in graph_names}
            l, p, r = self._execute_forward_backward(
                mnt_latents=leaves["mnt_latents"],
                mnt_text_embeddings=leaves["mnt_text_embeddings"],
                mnt_attention_mask=self._slice_aux(b["mnt_attention_mask"], lo, hi),
                mnt_pooled_embeddings=leaves["mnt_pooled_embeddings"],
                timesteps=b["timesteps"][lo:hi],
                debug_save_path=b["debug_save_path"] if lo == 0 else None,
                batch_captions=b["batch_captions"][lo:hi] if b["batch_captions"] else None,
                batch_reference_paths=b["batch_reference_paths"][lo:hi] if b["batch_reference_paths"] else None,
                alphas_cumprod_cached=b["alphas_cumprod_cached"],
                use_condition_images=b["use_condition_images"],
                condition_images_batch=b["condition_images_batch"][lo:hi] if b["condition_images_batch"] is not None else None,
                reference_latents_nested=b["reference_latents_nested"][lo:hi] if b["reference_latents_nested"] is not None else None,
                lens_latent_shape=b["lens_latent_shape"],
                mnt_repa_pixels=leaves["mnt_repa_pixels"],
                mnt_time_ids=b["mnt_time_ids"][lo:hi] if b["mnt_time_ids"] is not None else None,
                loss_weight_maps_batch=b["loss_weight_maps_batch"][lo:hi] if b.get("loss_weight_maps_batch") is not None else None,
                sensenova_prefix=b.get("sensenova_prefix"),
                loss_scale=w / eff_bs,
            )
            for n, leaf in leaves.items():
                if n in graph_inputs and leaf is not None and leaf.grad is not None:
                    grad_acc[n][lo:hi] = leaf.grad
            loss_acc += l * w
            pred_acc += p * w
            recon_acc += r * w
        if graph_inputs:
            # Its own backward, so its own counter window (the chunk backwards
            # above each took one); this one reaches the encoder parameters.
            self._reset_fused_group_counters()
            torch.autograd.backward(tensors=list(graph_inputs.values()),
                                    grad_tensors=[grad_acc[n] for n in graph_inputs])
            self._flush_fused_group_partials()
        return loss_acc / batch_size, pred_acc / batch_size, recon_acc / batch_size

    def _forward_backward_with_oom_recovery(
        self,
        mnt_latents: torch.Tensor,
        mnt_text_embeddings: torch.Tensor,
        mnt_attention_mask: Optional[torch.Tensor],
        mnt_pooled_embeddings: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        debug_save_path: Optional[Path],
        batch_captions: Optional[List[str]],
        batch_reference_paths: Optional[List[Optional[str]]],
        alphas_cumprod_cached: Optional[torch.Tensor],
        use_condition_images: bool,
        condition_images_batch: Optional[torch.Tensor],
        reference_latents_nested: Optional[list],
        min_split_batch_size: int = 1,
        lens_latent_shape: Optional[Tuple[int, int]] = None,
        mnt_repa_pixels: Optional[torch.Tensor] = None,
        mnt_time_ids: Optional[torch.Tensor] = None,
        effective_batch_size: Optional[int] = None,
        loss_weight_maps_batch: Optional[torch.Tensor] = None,
        sensenova_prefix: Optional[Any] = None,
    ) -> Tuple[float, float, float, bool]:
        """
        Execute forward + backward pass with OOM recovery via batch splitting.

        When OOM occurs, the batch is split in half and processed sequentially.
        Gradients are accumulated across splits, achieving the same result as
        processing the full batch (except for BatchNorm, which this model doesn't use).

        Args:
            mnt_latents: Latents for this MNT iteration [B, C, H, W]
            mnt_text_embeddings: Text embeddings [B, seq_len, dim]
            mnt_attention_mask: Attention mask (Z-Image only)
            mnt_pooled_embeddings: Pooled embeddings (SDXL only)
            timesteps: Timesteps for diffusion [B]
            debug_save_path: Path to save debug latents
            batch_captions: Captions for debug output
            alphas_cumprod_cached: Cached alphas_cumprod tensor
            use_condition_images: Whether ControlNet conditioning is used
            condition_images_batch: ControlNet condition images
            reference_latents_nested: Reference latents for FLUX.2
            min_split_batch_size: Minimum batch size (stop splitting below this)
            loss_weight_maps_batch: Outpaint-mode per-item latent-space loss weight maps
                [B, 1, H/8, W/8] (None outside outpaint conditioning_mode)

        Returns:
            Tuple of (loss_value, pred_loss_value, recon_loss_value, cuda_error_skip) as Python floats
            cuda_error_skip is True if batch was skipped due to unrecoverable CUDA error
        """
        batch_size = mnt_latents.shape[0]
        # Original full-batch size, preserved across recursive splits so every
        # leaf chunk scales its loss by chunk/B_eff. Without this, accumulating
        # per-half MEAN losses overcounts the gradient (sum of means != full mean
        # -> ~Nx too large). effective_batch_size is None only at the top level.
        eff_bs = effective_batch_size if effective_batch_size is not None else batch_size

        # All _execute_forward_backward args in one dict, so the micro-batch helper
        # (used by both the proactive escalate decision and the reactive OOM retry)
        # can slice them uniformly.
        _batch = dict(
            mnt_latents=mnt_latents, mnt_text_embeddings=mnt_text_embeddings,
            mnt_attention_mask=mnt_attention_mask, mnt_pooled_embeddings=mnt_pooled_embeddings,
            timesteps=timesteps, debug_save_path=debug_save_path,
            batch_captions=batch_captions, batch_reference_paths=batch_reference_paths,
            alphas_cumprod_cached=alphas_cumprod_cached, use_condition_images=use_condition_images,
            condition_images_batch=condition_images_batch, reference_latents_nested=reference_latents_nested,
            lens_latent_shape=lens_latent_shape, mnt_repa_pixels=mnt_repa_pixels,
            mnt_time_ids=mnt_time_ids, loss_weight_maps_batch=loss_weight_maps_batch,
            sensenova_prefix=sensenova_prefix,
        )

        # The applied-update window is this WHOLE call, not each backward inside
        # it. Narrower (per backward) would clear the count between micro-chunks,
        # hiding a chunk that applied its updates before a later one died; wider
        # (per batch) would carry a COMPLETED iteration's count into the next
        # one's forward, where an ordinary OOM would then be misread as a
        # half-applied step -- MNT > 1 calls this once per iteration.
        from .optimizers.update_census import reset_applied_updates
        reset_applied_updates()

        _disp_cm, _disp_info = self._activation_dispatch_begin(mnt_latents)
        _micro_bs = _disp_info[4] if _disp_info else None
        try:
            try:
                # Proactive path: escalate -> micro-batch (two-stage); else full batch.
                if _micro_bs is not None and _micro_bs < batch_size:
                    loss, pred_loss, recon_loss = self._microbatch_two_stage(_micro_bs, eff_bs, _batch)
                else:
                    loss, pred_loss, recon_loss = self._execute_forward_backward(
                        loss_scale=batch_size / eff_bs, **_batch)
                return loss, pred_loss, recon_loss, False  # success

            except RuntimeError as e:
                _cls = self._classify_cuda_error(e)
                if _cls == "not_cuda":
                    raise
                if _cls == "fatal":
                    # Sticky CUDA-context corruption -- retrying (offload / micro-batch)
                    # is futile and would just burn time before the same error recurs.
                    # Do NOT set _actdispatch_oom / _batch_was_unfittable: those are
                    # OOM-only signals that must not be poisoned by a dying context.
                    raise FatalCudaError(str(e)) from e
                # Before any recovery decision: if this backward already applied
                # updates, neither skipping nor retrying is sound.
                self._refuse_partial_fused_step(e)
                self._actdispatch_oom = True  # tell dispatch_end to flag this bucket
                # REACTIVE recovery. With the memory-fraction cap an over-budget
                # allocation RAISES here instead of silently spilling to shared host
                # memory (WDDM) -- so we get here fast (no thrashing, stop signals
                # still responsive) and retry THIS batch micro-batched to shrink the
                # per-forward footprint. The two-stage helper handles on-the-fly
                # encoder graphs, so we no longer have to skip those batches.
                self._oom_recovery_cleanup()
                if batch_size <= min_split_batch_size:
                    # One sample already doesn't fit -> this bucket is un-fittable.
                    self._batch_was_unfittable = True
                    print(f"{self.log_prefix} [OOM] batch_size={batch_size} already minimal, SKIPPING BATCH "
                          f"(bucket won't fit one sample) ({str(e)[:120]})")
                    return 0.0, 0.0, 0.0, True
                # OFFLOAD-FIRST rung (mirrors the proactive ordering): before
                # shrinking the batch, retry the SAME full batch with activation
                # offload. If the failed attempt already offloaded, retry with a
                # LOWERED threshold to widen the offloadable set. Micro-batch halving
                # serializes the batch and lowers per-image throughput, so pure
                # offload (value-exact) is preferred whenever it can rescue the step.
                if _disp_info is not None and self.activation_dispatcher is not None:
                    _new_cm = self._actdispatch_offload_retry_ctx(_disp_cm, _disp_info)
                    if _new_cm is not None:
                        _disp_cm = _new_cm  # finally exits the swapped-in context
                        print(f"{self.log_prefix} [OOM] retrying batch {batch_size} with activation "
                              f"offload (no split) after: {str(e)[:80]}")
                        try:
                            loss, pred_loss, recon_loss = self._execute_forward_backward(
                                loss_scale=batch_size / eff_bs, **_batch)
                            # Success: dispatch_end will record the measured offloaded
                            # volume so the proactive path picks 'offload' next time.
                            # (info[6]/info[7] already track the swapped-in context.)
                            self._actdispatch_oom = False
                            _disp_info[3] = "offload"
                            return loss, pred_loss, recon_loss, False
                        except RuntimeError as e_off:
                            _cls_off = self._classify_cuda_error(e_off)
                            if _cls_off == "not_cuda":
                                raise
                            if _cls_off == "fatal":
                                raise FatalCudaError(str(e_off)) from e_off
                            self._refuse_partial_fused_step(e_off)
                            e = e_off
                            self._oom_recovery_cleanup()
                if fused_backward_active(self):
                    # Micro-splitting under a fused path is not gradient
                    # accumulation: each chunk's hooks apply their own optimizer
                    # step, so a chunk that OOMs after an earlier one succeeded
                    # leaves the weights mid-step. The PROACTIVE splitter already
                    # refuses for this reason (see _activation_dispatch_begin);
                    # this is the same rule on the reactive ladder, which had
                    # kept it. Offload was tried above and is the last rung.
                    self._batch_was_unfittable = True
                    print(f"{self.log_prefix} [OOM] batch_size={batch_size} still out of memory "
                          f"under the fused backward pass, which cannot micro-split; SKIPPING BATCH "
                          f"(bucket excluded) ({str(e)[:120]})")
                    return 0.0, 0.0, 0.0, True
                for _retry_micro in (max(1, batch_size // 2), 1):
                    if _retry_micro >= batch_size:
                        continue
                    print(f"{self.log_prefix} [OOM] retrying batch {batch_size} micro-batched "
                          f"(micro={_retry_micro}) after: {str(e)[:80]}")
                    try:
                        loss, pred_loss, recon_loss = self._microbatch_two_stage(_retry_micro, eff_bs, _batch)
                        return loss, pred_loss, recon_loss, False
                    except RuntimeError as e2:
                        _cls2 = self._classify_cuda_error(e2)
                        if _cls2 == "not_cuda":
                            raise
                        if _cls2 == "fatal":
                            raise FatalCudaError(str(e2)) from e2
                        self._refuse_partial_fused_step(e2)
                        e = e2
                        self._oom_recovery_cleanup()
                # Even micro-batch=1 OOMs -> the bucket can't fit a single sample.
                self._batch_was_unfittable = True
                print(f"{self.log_prefix} [OOM] still out of memory at micro-batch=1, SKIPPING BATCH "
                      f"(bucket won't fit one sample) ({str(e)[:120]})")
                return 0.0, 0.0, 0.0, True
        finally:
            self._activation_dispatch_end(_disp_cm, _disp_info)


    @staticmethod
    def _actdispatch_latent_key(mnt_latents: torch.Tensor):
        """``(latent_h, latent_w, latent_t, batch)`` for the activation dispatcher.

        Video architectures (LTX-2.3, MiniMax-H3) hand this method a 5-D latent
        ``[B, C, T, H', W']``; the clip length ``T`` moves the transformer's packed
        sequence length -- and therefore the activation footprint -- directly
        (measured 2.36 GB at T_lat=7 vs 8.90 GB at T_lat=37 for MiniMax-H3 at
        384x640), so it MUST be part of the bucket key. Image architectures hand it
        a 4-D latent ``[B, C, H', W']`` with no temporal axis at all and get
        ``latent_t = 1``, which leaves their key and their predicted volume exactly
        what they were.

        The 5-D guard is deliberate: ``shape[-3]`` of a 4-D latent is the CHANNEL
        count (4 for SDXL, 16 for the flow-matching DiTs), and folding that into the
        key would be a silent, large miskey of every image bucket.
        """
        shape = mnt_latents.shape
        lh = int(shape[-2])
        lw = int(shape[-1])
        bs = int(shape[0])
        lt = int(shape[-3]) if len(shape) >= 5 else 1
        return lh, lw, max(1, lt), bs

    def _activation_dispatch_begin(self, mnt_latents: torch.Tensor):
        """Decide the per-bucket activation-offload mode and enter the offload
        context. Returns (entered_context_or_None, info_or_None).

        Proactive: the decision is made from a memory PREDICTION (never a caught
        OOM), which is required on Windows WDDM where overruns spill silently
        instead of raising. Disabled / non-CUDA paths return (None, None).
        """
        if not self.activation_dispatch_enable or not torch.cuda.is_available():
            return None, None
        try:
            lh, lw, lt, bs = self._actdispatch_latent_key(mnt_latents)
        except Exception:
            return None, None

        # Live VRAM state. Headroom = how much THIS process can still allocate now
        # = driver-free + its own reusable reserved cache (reserved - allocated).
        # Computed EVERY step so it adapts to co-located processes (e.g. the backend
        # serving inference/UI during training), unlike a one-shot startup budget
        # (which could latch a transiently-low value and choke the whole run).
        GB = 1024 ** 3
        resident_gb = torch.cuda.memory_allocated() / GB
        free_gb = torch.cuda.mem_get_info()[0] / GB
        reserved_gb = torch.cuda.memory_reserved() / GB
        total_gb = torch.cuda.get_device_properties(0).total_memory / GB
        headroom_gb = (free_gb + (reserved_gb - resident_gb)) - self.activation_dispatch_margin_gb

        if self.activation_dispatcher is None:
            from core.memory_management import ActivationDispatcher
            self.activation_dispatcher = ActivationDispatcher(
                budget_gb=total_gb,
                margin_gb=self.activation_dispatch_margin_gb,
                seed_coef=self.activation_dispatch_seed_coef,
                residual_frac=self.activation_dispatch_residual_frac,
                threshold_bytes=self.activation_dispatch_threshold_mb * 1024 * 1024,
            )
            self._actdispatch_logged = set()
            # Cap the caching allocator near the FULL dedicated VRAM (not a startup
            # snapshot) so an over-budget allocation RAISES OutOfMemoryError instead
            # of silently spilling to shared host memory (WDDM). A high, fixed cap
            # never chokes normal use; the proactive headroom check above (which
            # adapts to co-located VRAM use) is what actually prevents spills.
            try:
                frac = max(0.80, min(0.985, (total_gb - 1.0) / total_gb))
                torch.cuda.set_per_process_memory_fraction(frac)
                print(f"{self.log_prefix} [ActDispatch] enabled (total~{total_gb:.1f}GB, "
                      f"resident~{resident_gb:.1f}GB, headroom~{headroom_gb:.1f}GB, "
                      f"margin={self.activation_dispatch_margin_gb}GB, "
                      f"alloc_cap={frac:.3f}={frac*total_gb:.1f}GB -> OOM-not-spill)")
            except Exception as _e:
                print(f"{self.log_prefix} [ActDispatch] enabled (total~{total_gb:.1f}GB, "
                      f"headroom~{headroom_gb:.1f}GB; alloc cap failed: {_e})")

        disp = self.activation_dispatcher
        self._actdispatch_oom = False  # set by the reactive handler if this step OOMs
        mode = disp.decide(lh, lw, bs, headroom_gb, lt=lt)
        _headroom_gb = headroom_gb
        _act_pred_gb = disp.base_act(lh, lw, bs, lt=lt)

        # Block-swap activation offload (LayerOffloadConductor) already moves
        # activations; suppress the dispatcher offload to avoid double offload.
        conductor = getattr(self, "layer_offload_conductor", None)
        if conductor is not None and getattr(conductor, "enable_activation_offload", False):
            mode = "fast"

        # Log label for the bucket. The temporal extent is shown only when there
        # is one, so image-arch log lines are byte-identical to before.
        _bkt = f"{lw}x{lh}" + (f"x{lt}t" if lt > 1 else "")

        # Throttle decision logging to once per (bucket, decision) so aspect
        # bucketing (hundreds of distinct shapes) doesn't flood the log.
        def _log_once(key, msg):
            if key not in self._actdispatch_logged:
                self._actdispatch_logged.add(key)
                print(msg)

        # Escalate: even offload won't fit the full batch -> plan a micro-batch
        # split (gradient accumulation keeps the effective batch). Only escalate
        # splits; fast/offload buckets are never split, so batches that already
        # fit are never slowed down by accumulation.
        micro_bs = None
        step_threshold = disp.threshold_bytes
        if mode == "escalate":
            if fused_backward_active(self):
                # Fused backward cannot micro-split (per-param updates fire during
                # backward -- fused optimizer GROUPS step from their hooks too, so
                # they are covered by the same rule), so the escalate ladder
                # becomes: offload -> offload with a
                # LOWERED threshold_bytes to widen the offloadable set -> un-fittable.
                # Offload is value-exact (no gradient error), so pushing more saved
                # tensors to CPU is the correct lever before declaring the bucket
                # un-fittable rather than silently spilling.
                step_threshold = max(256 * 1024, disp.threshold_bytes // 16)
                _log_once((lh, lw, lt, bs, "fused"),
                          f"{self.log_prefix} [ActDispatch] bucket {_bkt} bs{bs} won't fit; "
                          f"micro-batch split disabled under fused backward (Block Swap); "
                          f"offload with lowered threshold={step_threshold // 1024}KB")
            else:
                planned = disp.plan_micro_bs(lh, lw, bs, headroom_gb, lt=lt)
                if planned < bs:
                    micro_bs = planned
                    _log_once((lh, lw, lt, bs, "split", micro_bs),
                              f"{self.log_prefix} [ActDispatch] bucket {_bkt} bs{bs} -> "
                              f"micro-batch={micro_bs} (act~{_act_pred_gb:.1f}GB, "
                              f"headroom~{_headroom_gb:.1f}GB, resident~{resident_gb:.1f}GB)")
                else:
                    _log_once((lh, lw, lt, bs, "tight"),
                              f"{self.log_prefix} [ActDispatch] bucket {_bkt} bs{bs} tight at "
                              f"micro-batch=1 (act~{_act_pred_gb:.1f}GB, headroom~{_headroom_gb:.1f}GB, "
                              f"resident~{resident_gb:.1f}GB); offload only")

        use_offload = mode in ("offload", "escalate")
        from core.memory_management import offload_activations
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        # Measured offloadable volume for this step: offload_activations increments
        # stats["bytes"] by the byte volume it packs to CPU; _activation_dispatch_end
        # feeds it back to record() so the per-bucket offloadable fit is calibrated.
        stats = {"bytes": 0}
        cm = offload_activations(use_offload, threshold_bytes=step_threshold, stats=stats)
        cm.__enter__()
        # Mutable list so the reactive OOM ladder can swap in an offload retry
        # context (new mode/stats) and have dispatch_end record it correctly.
        # `lt` is appended LAST so the existing positional indices (3=mode, 6=stats,
        # 7=threshold) that the OOM ladder mutates in place keep their meaning.
        return cm, [lh, lw, bs, mode, micro_bs, resident_gb, stats, step_threshold, lt]

    def _activation_dispatch_end(self, cm, info) -> None:
        """Exit the offload context and self-calibrate from the measured peak."""
        if cm is None:
            return
        cm.__exit__(None, None, None)
        if info is None or self.activation_dispatcher is None:
            return
        lh, lw, bs, mode, micro_bs, resident_gb = info[0], info[1], info[2], info[3], info[4], info[5]
        stats = info[6] if len(info) > 6 else None
        lt = info[8] if len(info) > 8 else 1
        try:
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            if getattr(self, "_actdispatch_oom", False):
                # This step raised OOM under the cap -> its true activation exceeds
                # the headroom (the measured peak is only the capped lower bound).
                # Flag the bucket to escalate next time instead of re-OOMing.
                self.activation_dispatcher.mark_overflow(lh, lw, bs, lt=lt)
            else:
                # Record every executed step. For a micro-split, the peak reflects
                # micro_bs samples; record() scales it back to the full bucket so the
                # bucket can learn it actually fits and stop splitting next time.
                record_mode = "base" if mode == "fast" else "offload"
                # A measured 0 (offload ran but nothing exceeded the threshold) is a
                # valid measurement, distinct from None (no stats available ->
                # residual_frac fallback in record()).
                offloaded_gb = None
                if record_mode == "offload" and stats is not None:
                    offloaded_gb = stats.get("bytes", 0) / (1024 ** 3)
                self.activation_dispatcher.record(
                    lh, lw, bs, record_mode, peak_gb, resident_gb,
                    executed_bs=(micro_bs if micro_bs is not None else bs),
                    offloaded_gb=offloaded_gb,
                    measured_threshold_bytes=(info[7] if len(info) > 7 else None),
                    lt=lt)
            if self.debug_vram:
                extra = f" micro_bs={micro_bs}" if micro_bs is not None else ""
                bkt = f"{lw}x{lh}" + (f"x{lt}t" if lt > 1 else "")
                cached = self.activation_dispatcher.base_act(lh, lw, bs, lt=lt)
                print(f"{self.log_prefix} [ActDispatch] bucket {bkt} bs{bs} "
                      f"mode={mode}{extra} peak={peak_gb:.2f}GB cached_act={cached:.2f}GB")
        except Exception:
            pass

    def _actdispatch_offload_retry_ctx(self, old_cm, info):
        """Swap the active offload context for the reactive OOM offload-first rung.

        Exits ``old_cm`` and enters a fresh ``offload_activations`` context for the
        same batch with offload enabled. If the failed step already offloaded, the
        threshold is lowered to widen the offloadable set; otherwise the dispatcher's
        default threshold is used. On swap, ``info[6]`` (stats) and ``info[7]``
        (threshold) are updated IN PLACE so _activation_dispatch_end always records
        the volume packed by the ACTIVE context -- even when the retry itself fails
        and the step falls through to micro-batching inside the swapped-in context.
        Returns the new context, or None when a retry cannot help (no way to widen
        an already-minimal offload set).
        """
        disp = self.activation_dispatcher
        if disp is None or info is None:
            return None
        prev_mode = info[3]
        prev_threshold = info[7] if len(info) > 7 else disp.threshold_bytes
        already_offloading = prev_mode in ("offload", "escalate")
        if already_offloading:
            new_threshold = max(64 * 1024, prev_threshold // 16)
            if new_threshold >= prev_threshold:
                # Already at the minimum offloadable set -> retry would be identical.
                return None
        else:
            new_threshold = disp.threshold_bytes
        try:
            old_cm.__exit__(None, None, None)
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        from core.memory_management import offload_activations
        new_stats = {"bytes": 0}
        cm = offload_activations(True, threshold_bytes=new_threshold, stats=new_stats)
        cm.__enter__()
        info[6] = new_stats
        info[7] = new_threshold
        return cm

    def _reset_fused_group_counters(self):
        """Arm the fused optimizer groups for ONE backward pass.

        Their hooks count gradients, which arrive per backward -- so this belongs
        immediately before every ``backward()``, not once per batch. A batch runs
        several backwards whenever MNT > 1, or it is micro-split, or an OOM retry
        splits it; counting those together pushes the count past the group size
        and the ``== group size`` step condition never holds again, silently
        dropping every step after the first (and leaving its gradient live).
        """
        groups = getattr(self, "fused_optimizer_groups", None)
        if groups is not None:
            groups.reset_counters()

    def _flush_fused_group_partials(self):
        """Apply the gradients this backward produced for INCOMPLETE groups.

        Belongs immediately after every ``backward()``: the end of the backward
        is the only point at which "this parameter got no gradient this time" is
        decided, and the hooks only step a group whose parameters all got one.
        Without this, a group holding any parameter that a given backward does
        not reach -- the Vision Encoder on a reference-free batch (see the epoch
        VE-offload note in ``train()``), a block that stochastic depth dropped
        this step -- never steps at all, freezing the parameters that DID get a
        gradient and merely share the group by index order.
        """
        groups = getattr(self, "fused_optimizer_groups", None)
        if groups is not None:
            groups.step_incomplete_groups()

    def _refuse_partial_fused_step(self, exc: Exception) -> None:
        """Stop the run if the OOM being recovered from left a half-applied step.

        Called from every OOM handler in ``_forward_backward_with_oom_recovery``,
        which otherwise skips the batch or retries it -- and the retry re-applies
        the updates already written, then reports success.

        Conservative in one direction only: an OOM raised after the LAST hook
        fired counts as partial too, stopping a run whose weights were in fact
        consistent. That is the safe side of a rare case.
        """
        from .optimizers.update_census import applied_updates
        if not fused_backward_active(self):
            return
        applied = applied_updates()
        if applied <= 0:
            return
        # Same taint record _note_partial_step_taint keeps for the non-OOM
        # escapes, so the emergency/interrupt handler's quarantine-or-refuse
        # decision (_refuse_save_after_partial_step) covers this route too --
        # an OOM mid-fused-backward is not less half-applied than a KeyError
        # mid-fused-backward, and the CUDA context is alive by definition here
        # (this exception classified as recoverable OOM, not fatal).
        self._partial_step_taint = {
            "applied": applied,
            "kind": type(exc).__name__,
            "detail": str(exc)[:200],
        }
        raise PartialOptimizerStepError(
            self._partial_fused_step_message(applied, exc)
        ) from exc

    def _resume_point_sentence(self) -> str:
        """Where the last CONSISTENT weights are, in this run's actual terms."""
        last = getattr(self, "_last_periodic_checkpoint_step", None)
        if last is not None:
            return f"Resume from the last periodic checkpoint (step {last})."
        resumed = getattr(self, "_resume_checkpoint_label", None)
        if resumed:
            return (f"This invocation wrote no periodic checkpoint of its own, so the "
                    f"last consistent weights are the ones it resumed from ({resumed}).")
        every = getattr(self, "_periodic_save_every", None)
        if every:
            why = (f"the run did not reach its first checkpoint interval "
                   f"(save_every={every})")
        elif every == 0:
            why = "save_every=0 disables periodic checkpointing"
        else:
            why = "no periodic checkpoint was written"
        return (f"There is no checkpoint to resume from ({why}), so the run must be "
                f"started again.")

    def _partial_fused_step_message(self, applied: int, exc: Exception) -> str:
        resume = self._resume_point_sentence()
        return (
            f"Training stopped: an out-of-memory error interrupted a fused "
            f"backward pass after {applied} parameter update(s) had already been "
            f"applied. Each parameter is updated from its own post-accumulate-grad "
            f"hook the moment its gradient exists, so those parameters carry this "
            f"step's update and the rest do not: the in-memory weights are a "
            f"mixture of two steps. They cannot be repaired -- there is no "
            f"snapshot to roll back to, and re-running the batch would apply "
            f"those updates twice -- so the batch is NOT skipped, and no ordinary "
            f"checkpoint, training state, optimizer, or EMA file is written for "
            f"this step (the tainted weights may instead be salvaged to a "
            f"separate, manually-loaded quarantined checkpoint -- see the "
            f"training log for whether that happened). {resume} "
            + _MEMORY_BUDGET_ADVICE
            + f" The out-of-memory error was: {str(exc)[:200]}"
        )

    def _execute_forward_backward(
        self,
        mnt_latents: torch.Tensor,
        mnt_text_embeddings: torch.Tensor,
        mnt_attention_mask: Optional[torch.Tensor],
        mnt_pooled_embeddings: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        debug_save_path: Optional[Path],
        batch_captions: Optional[List[str]],
        batch_reference_paths: Optional[List[Optional[str]]],
        alphas_cumprod_cached: Optional[torch.Tensor],
        use_condition_images: bool,
        condition_images_batch: Optional[torch.Tensor],
        reference_latents_nested: Optional[list],
        lens_latent_shape: Optional[Tuple[int, int]] = None,
        mnt_repa_pixels: Optional[torch.Tensor] = None,
        mnt_time_ids: Optional[torch.Tensor] = None,
        loss_weight_maps_batch: Optional[torch.Tensor] = None,
        sensenova_prefix: Optional[Any] = None,
        loss_scale: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Execute forward pass (train_step_xxx) and backward pass for a batch.

        Returns loss values as Python floats (not tensors).
        Gradients are accumulated in model parameters.

        loss_scale multiplies the loss BEFORE backward. Used by proactive
        micro-batching: when a bucket is split into chunks of size m out of a
        batch B, each chunk is scaled by m/B so the accumulated gradient equals
        the full-batch mean gradient (sum_i (m_i/B) * grad(mean_loss_i)).
        The returned loss VALUE stays unscaled (per-chunk mean) for reporting.
        """
        # Forward pass (architecture-specific)
        if self.is_sensenova:
            from core.training.arch.base_arch import TrainStepContext
            ctx = TrainStepContext(
                latents=mnt_latents,
                sensenova_prefix=sensenova_prefix,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_zimage:
            # P6b: route via the arch handler (registry dispatch). The kwargs
            # bundle is frozen into TrainStepContext; the handler unpacks it into
            # ops/zimage_ops.train_step (verbatim body).
            from core.training.arch.base_arch import TrainStepContext
            ctx = TrainStepContext(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                attention_mask=mnt_attention_mask,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_anima:
            # Anima carries the LLM-Adapter side payload (source_mask, t5 ids)
            # in mnt_attention_mask, which here holds a dict produced by
            # encode_caption() rather than a single tensor.
            from core.training.arch.base_arch import TrainStepContext
            anima_aux = mnt_attention_mask if isinstance(mnt_attention_mask, dict) else {}
            ctx = TrainStepContext(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                anima_aux=anima_aux,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_ltx2:
            # LTX-2.3 carries the audio-text embedding + mask (+ fps) in
            # mnt_attention_mask as a dict (produced by collate_aux), same
            # pattern as anima. latents are 5D [B, 128, T_lat, H', W'].
            from core.training.arch.base_arch import TrainStepContext
            ltx2_aux = mnt_attention_mask if isinstance(mnt_attention_mask, dict) else {}
            ctx = TrainStepContext(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                anima_aux=ltx2_aux,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_minimax_h3:
            # MiniMax-H3 carries {num_text_tokens, audio_latents, audio_present}
            # in mnt_attention_mask as a dict (collate_aux + the per-clip audio
            # injection below), same pattern as ltx2/anima. latents are 5D
            # [B, 24, T_lat, H', W'].
            from core.training.arch.base_arch import TrainStepContext
            h3_aux = mnt_attention_mask if isinstance(mnt_attention_mask, dict) else {}
            ctx = TrainStepContext(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                anima_aux=h3_aux,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_acestep:
            # ACE-Step carries the caption's text_attention_mask in
            # mnt_attention_mask as a dict (produced by collate_aux), same
            # pattern as ltx2/anima. latents are 3D [B, T_lat, 64].
            from core.training.arch.base_arch import TrainStepContext
            acestep_aux = mnt_attention_mask if isinstance(mnt_attention_mask, dict) else {}
            ctx = TrainStepContext(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                anima_aux=acestep_aux,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_lens:
            # mnt_text_embeddings: [B, num_layers, L, D]
            # mnt_attention_mask:  [B, L] encoder mask
            from core.training.arch.base_arch import TrainStepContext
            _lh, _lw = lens_latent_shape if lens_latent_shape else (None, None)
            ctx = TrainStepContext(
                latents=mnt_latents,
                encoder_features=mnt_text_embeddings,
                encoder_mask=mnt_attention_mask,
                timesteps=timesteps,
                profile_vram=self.debug_vram,
                latent_h=_lh,
                latent_w=_lw,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_ideogram4:
            # mnt_text_embeddings: [B, 13, L, 4096]; mnt_attention_mask: [B, L]
            from core.training.arch.base_arch import TrainStepContext
            _lh, _lw = lens_latent_shape if lens_latent_shape else (None, None)
            ctx = TrainStepContext(
                latents=mnt_latents,
                encoder_features=mnt_text_embeddings,
                encoder_mask=mnt_attention_mask,
                timesteps=timesteps,
                profile_vram=self.debug_vram,
                latent_h=_lh,
                latent_w=_lw,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_krea2:
            # mnt_latents: packed [B, N, 64]; mnt_text_embeddings: [B, seq, 12, 2560];
            # mnt_attention_mask: [B, seq]
            # P6c: route via the arch handler; ops/krea2_ops.train_step (verbatim).
            from core.training.arch.base_arch import TrainStepContext
            _lh, _lw = lens_latent_shape if lens_latent_shape else (None, None)
            ctx = TrainStepContext(
                latents=mnt_latents,
                encoder_features=mnt_text_embeddings,
                encoder_mask=mnt_attention_mask,
                timesteps=timesteps,
                profile_vram=self.debug_vram,
                latent_h=_lh,
                latent_w=_lw,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_minit2i:
            # mnt_latents: [B, 3, H, W] in [-1,1] (pixel-space, no VAE)
            # mnt_text_embeddings: [B, L, 1024]; mnt_attention_mask: [B, L]
            # P6c: route via the arch handler; ops/minit2i_ops.train_step (verbatim).
            from core.training.arch.base_arch import TrainStepContext
            ctx = TrainStepContext(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                attention_mask=mnt_attention_mask,
                timesteps=timesteps,
                profile_vram=self.debug_vram,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                repa_pixels=mnt_repa_pixels,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif self.is_flux2:
            # FLUX.2 training with position IDs. The packing / position-ID prep
            # stays in the spine (shared packing helpers) and is frozen into ctx;
            # P6c routes the forward+loss via the arch handler
            # (ops/flux2_ops.train_step, verbatim).
            from core.training.arch.base_arch import TrainStepContext
            img_ids = self._flux2_prepare_latent_ids(mnt_latents).to(self.device)
            packed_latents = self._flux2_pack_latents(mnt_latents)
            txt_ids = self._flux2_prepare_text_ids(mnt_text_embeddings).to(self.device)

            # Prepare reference latents
            mnt_reference_latents_nested = None
            if reference_latents_nested is not None:
                mnt_reference_latents_nested = [
                    [lat.detach() for lat in item_lats]
                    for item_lats in reference_latents_nested
                ]

            ctx = TrainStepContext(
                latents=packed_latents,
                text_embeddings=mnt_text_embeddings,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=timesteps,
                guidance=None,
                reference_latents_nested=mnt_reference_latents_nested,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)
        elif use_condition_images and condition_images_batch is not None:
            # ControlNet training
            mnt_condition_images = condition_images_batch.detach()
            loss, pred_loss, recon_loss = self.train_step_controlnet(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                condition_images=mnt_condition_images,
                pooled_embeddings=mnt_pooled_embeddings,
                time_ids=mnt_time_ids,
                timesteps=timesteps,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
                loss_weight_map=loss_weight_maps_batch,
            )
        else:
            # SD1.5/SDXL — P6a: route via the arch handler (registry dispatch).
            # The kwargs bundle is frozen into TrainStepContext; the handler
            # unpacks it into ops/sd_sdxl_ops.train_step (verbatim body).
            from core.training.arch.base_arch import TrainStepContext
            ctx = TrainStepContext(
                latents=mnt_latents,
                text_embeddings=mnt_text_embeddings,
                pooled_embeddings=mnt_pooled_embeddings,
                time_ids=mnt_time_ids,
                timesteps=timesteps,
                debug_save_path=debug_save_path,
                debug_captions=batch_captions if debug_save_path else None,
                debug_reference_image_paths=batch_reference_paths if debug_save_path else None,
                profile_vram=self.debug_vram,
                alphas_cumprod_cached=alphas_cumprod_cached,
            )
            loss, pred_loss, recon_loss = self.arch.train_step(self, ctx)

        # Backward pass. With gradient accumulation (optimizer steps every
        # `_grad_accum_steps` backward passes), scale the loss so the accumulated
        # gradient is the AVERAGE of the window, not the sum. Without this the
        # summed gradient grows with the accumulation count and is clipped harder
        # by max_grad_norm, negating the variance-reduction benefit. accum=1 is a
        # no-op, so existing (non-accumulating) runs are unchanged. Report the
        # unscaled loss value below.
        accum = getattr(self, "_grad_accum_steps", 1) or 1
        loss_for_backward = loss * loss_scale if loss_scale != 1.0 else loss
        if accum > 1:
            loss_for_backward = loss_for_backward / accum
        self._reset_fused_group_counters()
        _applied_before = self._applied_updates_now()
        try:
            if self.use_grad_scaler:
                self.grad_scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            four_phase = getattr(self, "sensenova_four_phase", None)
            if four_phase is not None:
                # The backward above stopped at the boundary K/V leaves. Phase 3
                # runs HERE, not at the optimizer-step seam, on both routes: the
                # update census asserts completeness inside the MNT loop, which
                # is upstream of that seam, and the fused grad norms are read
                # there too, so phase 3 landing after it would report the
                # understanding half as never updated AND drop its grad norm on
                # a correct run. Under a shared window this is a no-op until the
                # window's final backward, which does the one capture+flush.
                four_phase.after_generation_backward()
            self._flush_fused_group_partials()
        except BaseException as _exc:
            # Scoped to the backward: an exception raised before or after it
            # cannot have interrupted the hooks, and keeps its ordinary
            # emergency save.
            self._note_partial_step_taint(_applied_before, _exc)
            raise

        # Extract values before deleting tensors
        loss_value = loss.item()
        pred_loss_value = pred_loss.item() if isinstance(pred_loss, torch.Tensor) else pred_loss
        recon_loss_value = recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss

        # Free computation graph
        del loss, loss_for_backward, pred_loss, recon_loss

        return loss_value, pred_loss_value, recon_loss_value

    def _assert_sensenova_step_seam_residency(self, four_phase) -> None:
        """Which MoT half must be GPU-resident at the optimizer-step seam.

        Named so it can be driven without the 1000-line train loop around it.
        Which half is resident is decided by whether phase 3 ran with the
        backward that just finished, NOT by whether the split is armed: on a
        shared window's non-final iterations phase 3 has not run, the generation
        half is still resident, and asserting the understanding half there raises
        outside any try (the seam has no handler above it).
        """
        phase_evictor = getattr(self, "sensenova_phase_evictor", None)
        if phase_evictor is None:
            return
        if four_phase is not None and four_phase.phase_three_ran:
            phase_evictor.assert_understanding_resident()
        else:
            phase_evictor.assert_generation_resident()

    @staticmethod
    def _applied_updates_now() -> int:
        from .optimizers.update_census import applied_updates
        return applied_updates()

    def _note_partial_step_taint(self, applied_before: int, exc: BaseException) -> None:
        """Record that an exception left this backward's updates half-applied.

        The OOM route refuses with ``PartialOptimizerStepError`` before any save.
        Every other way out of a backward -- a non-CUDA RuntimeError, a
        non-RuntimeError (the schedule-free KeyError this file documents at
        ``_setup_fused_backward_pass``), a fatal CUDA error with a live context,
        Ctrl-C -- lands in the interrupt or emergency handler, which would
        otherwise write those weights plus a paired state.json and optimizer
        state that resume would trust. This is what tells them not to.
        """
        if not fused_backward_active(self):
            return
        applied = self._applied_updates_now() - applied_before
        if applied <= 0:
            return
        self._partial_step_taint = {
            "applied": applied,
            "kind": type(exc).__name__,
            "detail": str(exc)[:200],
        }

    def _save_quarantined_partial_step_checkpoint(self, step: int, epoch: int) -> bool:
        """Salvage the tainted weights under a name resume scanning never selects.

        Reuses save_checkpoint() through the same run_name-swap
        ``_save_ema_checkpoint`` uses: every trainer builds its output path from
        ``self.run_name``, so swapping it in produces a fully separate,
        normally-formatted checkpoint that QUARANTINE_ENTRY_MARKER then hides
        from every scanner (find_latest_checkpoint, _get_sorted_checkpoints,
        the "latest" resume probe in __init__, and rotation cleanup). Loading
        it back requires naming its exact filename via resume_from_checkpoint
        -- that IS the explicit override; nothing scans for it.

        Weights only. No training-state.json (it would pair a batch position
        with weights that are not that batch's consistent result, and a
        state.json existing next to a live-looking checkpoint is exactly the
        auto-resumable shape this exists to avoid). No optimizer/EMA save
        either: that state is equally half-applied, and shipping it would
        invite treating this as a normal resume point instead of the manual,
        human-verified decision it is meant to force.

        Best-effort: called from an interrupt/emergency handler that must
        finish its own cleanup regardless, so a failure here is reported, not
        raised.
        """
        original_run_name = self.run_name
        try:
            self.run_name = f"{original_run_name}{QUARANTINE_RUN_NAME_SUFFIX}"
            self.save_checkpoint(step=step, epoch=epoch)
            return True
        except Exception as e:
            print(f"{self.log_prefix} [FAILED] Could not write a quarantined checkpoint either: {e}")
            return False
        finally:
            self.run_name = original_run_name

    def _refuse_save_after_partial_step(self, when: str, global_step: int, epoch: int) -> bool:
        """True (having said why) if the ORDINARY save path must not run.

        A half-applied fused step's weights are still salvaged into a
        quarantined artefact when the CUDA context is provably alive (the
        same in-process canary the OOM/fatal classifier uses) -- writing
        nothing throws away real training progress on top of the
        interruption. Under a dead context nothing GPU-side can be read back
        at all, so this still writes nothing there, exactly as before. Either
        way, no ordinary checkpoint/state/optimizer/EMA file is written for
        this step: the last periodic checkpoint remains the resume point.
        """
        taint = getattr(self, "_partial_step_taint", None)
        if not taint:
            return False
        print(f"\n{self.log_prefix} [FAILED] {when} inside a fused backward pass that had "
              f"already applied {taint['applied']} parameter update(s) "
              f"({taint['kind']}: {taint['detail']}).")
        print(f"{self.log_prefix} [FAILED] No ordinary checkpoint, training state, optimizer "
              f"state or EMA file is written for step {global_step}: the in-memory weights "
              f"carry part of that step and the optimizer state carries part of its own, "
              f"and nothing on disk should be resumable from them.")
        # Same aliveness probe order as the ordinary emergency path (13584-13588):
        # probe BEFORE touching CUDA further. _cuda_context_alive()'s own
        # allocation is 8 floats and wrapped in try/except, so it cannot itself
        # raise, but under genuine VRAM exhaustion it can still (rarely) return
        # False when a slightly larger, real salvage-copy would have succeeded.
        # That is the same risk the ordinary path accepts at the same point;
        # this mirrors it rather than inventing a different order.
        ctx_alive = self._cuda_context_alive() if self._cuda_is_available() else True
        if ctx_alive:
            # Mirror the ordinary emergency path's CPU-move + cache-clear
            # (13620-13636) before attempting the salvage save: the dominant
            # trigger for this taint is an OOM, i.e. peak VRAM pressure, so
            # freeing what we can BEFORE the salvage copy (rather than after)
            # is what gives it room to succeed.
            try:
                print(f"{self.log_prefix} [FAILED] Moving model to CPU to free GPU memory "
                      f"before the salvage attempt...")
                self.move_main_model_to_cpu()
                self.move_text_encoder_to_cpu()
                self.move_vae_to_cpu()
            except Exception as move_error:
                print(f"{self.log_prefix} [FAILED] Failed to move model to CPU: {move_error}")
            try:
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        if ctx_alive and self._save_quarantined_partial_step_checkpoint(global_step, epoch):
            salvage_msg = (
                f"The tainted WEIGHTS ONLY were salvaged to a QUARANTINED checkpoint "
                f"(run_name suffix '{QUARANTINE_RUN_NAME_SUFFIX}', step {global_step}). "
                f"The weights are half-applied (part of one step's update, part of "
                f"another's) and no optimizer state accompanies them. Resume scanning "
                f"ignores it and it has no paired training-state or optimizer file, so "
                f"it cannot be auto-resumed: using it requires passing its exact "
                f"filename as resume_from_checkpoint, and its loss/output should be "
                f"checked before doing so."
            )
            emit_training_warning(
                salvage_msg, code="partial_step_quarantined_checkpoint", prefix=self.log_prefix,
            )
            print(f"{self.log_prefix} [FAILED] {salvage_msg}")
        elif not ctx_alive:
            print(f"{self.log_prefix} [FAILED] The CUDA context is dead, so even the tainted "
                  f"weights cannot be read back -- no quarantined checkpoint was attempted.")
        print(f"{self.log_prefix} [FAILED] {self._resume_point_sentence()}")
        return True

    # ============================================================
    # Training Step
    # ============================================================

    def train_step(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        pooled_embeddings: torch.Tensor = None,
        time_ids: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        debug_save_path: Optional[Path] = None,
        debug_captions: Optional[List[str]] = None,
        debug_reference_image_paths: Optional[List[str]] = None,
        profile_vram: bool = False,
        alphas_cumprod_cached: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """SD1.5/SDXL training step — thin delegator (P6a).

        The VERBATIM body lives in ``ops/sd_sdxl_ops.train_step`` (mechanical
        ``self.`` -> ``trainer.`` receiver rename only). Kept as a public
        delegator so any direct ``self.train_step(...)`` caller keeps working;
        ``_execute_forward_backward`` routes via ``self.arch.train_step`` which
        reaches the same ops function.
        """
        from core.training.ops import sd_sdxl_ops
        return sd_sdxl_ops.train_step(
            self,
            latents=latents,
            text_embeddings=text_embeddings,
            pooled_embeddings=pooled_embeddings,
            time_ids=time_ids,
            timesteps=timesteps,
            debug_save_path=debug_save_path,
            debug_captions=debug_captions,
            debug_reference_image_paths=debug_reference_image_paths,
            profile_vram=profile_vram,
            alphas_cumprod_cached=alphas_cumprod_cached,
        )
    def train_step_controlnet(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        condition_images: torch.Tensor,
        pooled_embeddings: torch.Tensor = None,
        time_ids: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        profile_vram: bool = False,
        alphas_cumprod_cached: Optional[torch.Tensor] = None,
        loss_weight_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Perform single ControlNet training step (SD1.5/SDXL).

        Standard ControlNet:
        1. ControlNet forward: condition_images + noisy_latents -> residuals
        2. UNet forward with residuals injected -> model_pred
        3. Loss = MSE(model_pred, target)

        UNet is frozen but runs with gradients enabled so that gradient
        flows back through the residual additions to the ControlNet.

        Args:
            latents: Image latents [B, C, H, W]
            text_embeddings: Text prompt embeddings
            condition_images: Condition image tensor [B, 3, H, W] in [0, 1] range
            pooled_embeddings: Pooled text embeddings (SDXL only)
            timesteps: Optional timesteps tensor
            profile_vram: If True, print VRAM usage
            alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU
            loss_weight_map: Optional per-sample latent-space loss weight
                [B, 1, H, W] (outpaint conditioning_mode only). None (default)
                reproduces the unweighted loss exactly.

        Returns:
            (loss_tensor, pred_loss_value, recon_loss_value)
        """
        if profile_vram:
            print_vram_usage("[train_step_controlnet] Start")

        # Move tensors to GPU
        latents = latents.to(device=self.device, dtype=self.training_dtype, non_blocking=True)
        condition_images = condition_images.to(device=self.device, dtype=self.training_dtype, non_blocking=True)

        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Sample timesteps (DDPM)
        noise_process = getattr(self, 'noise_process', 'ddpm')

        if timesteps is None:
            if noise_process == "ddpm":
                if self.timestep_sampler is not None:
                    timesteps_continuous = self.timestep_sampler.sample(batch_size, self.device)
                    timesteps = ((1.0 - timesteps_continuous) * self.noise_scheduler.config.num_train_timesteps).long()
                    timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
                else:
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (batch_size,), device=self.device,
                    ).long()
            elif noise_process == "flow":
                if self.timestep_sampler is not None:
                    timesteps = self.timestep_sampler.sample(batch_size, self.device)
                else:
                    timesteps = torch.rand((batch_size,), device=self.device)
        else:
            if noise_process == "ddpm":
                timesteps = ((1.0 - timesteps) * self.noise_scheduler.config.num_train_timesteps).long()
                timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)

        # Add noise to latents
        noisy_latents = add_noise_unified(
            noise_process=noise_process,
            noise_scheduler=self.noise_scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )

        # Prepare added_cond_kwargs for SDXL
        added_cond_kwargs = None
        if self.is_sdxl and pooled_embeddings is not None:
            if time_ids is not None:
                add_time_ids = time_ids.to(device=self.device, dtype=pooled_embeddings.dtype)
            else:
                latent_height, latent_width = latents.shape[2], latents.shape[3]
                image_height, image_width = latent_height * 8, latent_width * 8
                add_time_ids = torch.tensor([[
                    image_height, image_width, 0, 0, image_height, image_width
                ]], dtype=pooled_embeddings.dtype, device=self.device).repeat(batch_size, 1)

            added_cond_kwargs = {
                "text_embeds": pooled_embeddings,
                "time_ids": add_time_ids,
            }

        if profile_vram:
            print_vram_usage("[train_step_controlnet] Before ControlNet forward")

        # Enable gradients for gradient checkpointing (ControlNet needs grad flow)
        noisy_latents.requires_grad_(True)
        text_embeddings.requires_grad_(True)
        if pooled_embeddings is not None:
            pooled_embeddings.requires_grad_(True)

        # ControlNet forward pass (trainable)
        # Get adapter from ControlNetTrainer
        controlnet_adapter = self.adapter
        controlnet_module = self.controlnet
        is_lllite = getattr(self, 'controlnet_type', 'standard') == 'lllite'

        # The base device-management (onthefly_gpu / swap_onthefly latent+text
        # encoding, post-load VRAM offload) only tracks the UNet/TE/VAE. The
        # trainable ControlNet is a SEPARATE model, created at the base UNet's
        # device -- which is CPU when the UNet has been offloaded after load --
        # and is never moved back, so its first forward hits a cuda-vs-cpu addmm
        # mismatch. Ensure it is resident on the compute device before the
        # forward (a no-op once it is already there; cheap first-parameter
        # device check keeps per-step overhead negligible). This applies to BOTH
        # standard ControlNet (forward below) AND LLLite (apply_patches below) --
        # the LLLite module's Conv2d/Linear weights also inherit the UNet's
        # device at creation time and are never moved by move_main_model_to_gpu/cpu
        # (which only tracks unet/text_encoder/vae), so gating this on
        # `not is_lllite` left LLLite's first apply_patches() on a possibly-CPU
        # module while inputs are on self.device -> device-mismatch crash.
        try:
            _cn_first_param = next(controlnet_module.parameters())
            if _cn_first_param.device != self.device:
                controlnet_module.to(self.device)
        except StopIteration:
            pass

        if is_lllite:
            # LLLite mode: apply patches to UNet attention layers before forward
            controlnet_module.apply_patches(self.unet, condition_images)
            controlnet_output = None
        else:
            # Standard ControlNet: get residuals from ControlNet forward
            if self.mixed_precision:
                with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                    controlnet_output = controlnet_adapter.controlnet_forward(
                        controlnet=controlnet_module,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        text_embeddings=text_embeddings,
                        condition_images=condition_images,
                        added_cond_kwargs=added_cond_kwargs,
                    )
            else:
                controlnet_output = controlnet_adapter.controlnet_forward(
                    controlnet=controlnet_module,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    text_embeddings=text_embeddings,
                    condition_images=condition_images,
                    added_cond_kwargs=added_cond_kwargs,
                )

        if profile_vram:
            print_vram_usage("[train_step_controlnet] After ControlNet forward")

        # UNet forward pass
        try:
            if controlnet_output is not None:
                # Standard ControlNet: inject residuals into UNet
                down_block_res_samples, mid_block_res_sample = controlnet_output

                # UNet is frozen but we need gradients to flow through residual additions
                if self.mixed_precision:
                    with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                        if self.is_sdxl and added_cond_kwargs is not None:
                            model_pred = self.unet(
                                noisy_latents,
                                timesteps,
                                text_embeddings,
                                added_cond_kwargs=added_cond_kwargs,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            ).sample
                        else:
                            model_pred = self.unet(
                                noisy_latents,
                                timesteps,
                                text_embeddings,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            ).sample
                else:
                    if self.is_sdxl and added_cond_kwargs is not None:
                        model_pred = self.unet(
                            noisy_latents,
                            timesteps,
                            text_embeddings,
                            added_cond_kwargs=added_cond_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        ).sample
                    else:
                        model_pred = self.unet(
                            noisy_latents,
                            timesteps,
                            text_embeddings,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        ).sample
            else:
                # LLLite mode: patches already applied, normal UNet forward
                if self.mixed_precision:
                    with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                        if self.is_sdxl and added_cond_kwargs is not None:
                            model_pred = self.unet(
                                noisy_latents, timesteps, text_embeddings,
                                added_cond_kwargs=added_cond_kwargs,
                            ).sample
                        else:
                            model_pred = self.unet(
                                noisy_latents, timesteps, text_embeddings,
                            ).sample
                else:
                    if self.is_sdxl and added_cond_kwargs is not None:
                        model_pred = self.unet(
                            noisy_latents, timesteps, text_embeddings,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                    else:
                        model_pred = self.unet(
                            noisy_latents, timesteps, text_embeddings,
                        ).sample
        finally:
            # Remove LLLite patches after UNet forward (must always run)
            if is_lllite:
                controlnet_module.remove_patches(self.unet)

        if profile_vram:
            print_vram_usage("[train_step_controlnet] After UNet forward")

        # Get prediction target
        prediction_target = getattr(self, 'prediction_target', 'epsilon')
        target = get_target_unified(
            noise_process=noise_process,
            prediction_target=prediction_target,
            noise_scheduler=self.noise_scheduler,
            latents=latents,
            noise=noise,
            timesteps=timesteps,
        )

        # Calculate loss (always in fp32)
        loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        if loss_weight_map is not None:
            # Outpaint conditioning_mode: down-weight the KNOWN region and keep the
            # GENERATE region at full weight (see condition-load branch that builds
            # this map). Broadcasts [B,1,H,W] over [B,C,H,W]. None (default,
            # non-outpaint) skips this entirely -> unweighted loss unchanged.
            _wm = loss_weight_map.to(loss_per_element)
            # GENERATE-region-only MSE (monitoring, no grad): mean raw MSE over ONLY
            # the generate cells (known region excluded). The weighted `loss` scalar
            # is diluted ~21% by the trivially-predictable known region and buried in
            # per-timestep noise; this metric isolates the region the ControlNet must
            # actually outpaint, so genuine learning becomes visible above the noise
            # floor. Generate cells = weight >= ~1.0 (known cells are down-weighted to
            # outpaint_known_loss_weight, e.g. 0.3), so a >0.5 threshold splits them.
            with torch.no_grad():
                _gen_mask = (_wm > 0.5).float()  # [B,1,H,W]
                _denom = _gen_mask.sum() * float(loss_per_element.shape[1]) + 1e-8
                self._last_gen_region_loss = float((loss_per_element * _gen_mask).sum() / _denom)
            # seam_loss (monitoring, no grad, ALWAYS computed in outpaint mode --
            # behavior-neutral instrument): raw (unweighted) MSE over ONLY the
            # 1-cell generate-side ring immediately adjacent to the known region
            # (the same ring outpaint_seam_loss_boost targets, re-derived from _wm
            # so it stays correct regardless of the boost/ring-width settings).
            # gen_loss cannot see this -- the ring is only ~2-3% of generate
            # cells, invisible under per-timestep noise averaged over the whole
            # generate region.
            _known_mask = (_wm < 0.5)  # [B,1,H,W] bool, robust to any generate-side boost
            with torch.no_grad():
                _dil = F.max_pool2d(_known_mask.float(), kernel_size=3, stride=1, padding=1)
                _ring_mask = (_dil > 0.5) & (~_known_mask)  # [B,1,H,W] bool
                _ring_count = _ring_mask.float().sum()
                if _ring_count.item() > 0:
                    _ring_denom = _ring_count * float(loss_per_element.shape[1]) + 1e-8
                    self._last_seam_ring_loss = float((loss_per_element * _ring_mask.float()).sum() / _ring_denom)
                else:
                    # Degenerate rect (e.g. full-bleed / no known region) -> no ring exists.
                    self._last_seam_ring_loss = None
            # Loss-vs-timestep instrumentation (monitoring, no grad, ALWAYS on in
            # outpaint mode -- see scratchpad "Outpaint ControlNet: loss-vs-timestep
            # instrumentation" design doc). Per-SAMPLE raw eps-space region MSE
            # (known/generate/seam-ring) + per-sample SNR, stashed for the JSONL
            # sidecar written at the logging site. Per-sample, NOT batch-mean:
            # with B=2, binning batch-mean-t against batch-mean loss convolves the
            # loss curve with the triangular mean-t distribution and destroys the
            # high-noise-tail curvature the diagnostic exists to read (design doc
            # G2). Computed on loss_per_element BEFORE the _wm multiply below and
            # BEFORE Min-SNR -- raw region MSE, not what is actually trained on
            # (design doc G1); x0-space companions are added in the recon block.
            with torch.no_grad():
                _lvt_c = float(loss_per_element.shape[1])
                _eps_known_ps = _per_sample_masked_mean(loss_per_element, _known_mask, _lvt_c)
                _eps_gen_ps = _per_sample_masked_mean(loss_per_element, _gen_mask, _lvt_c)
                _eps_seam_ps = _per_sample_masked_mean(loss_per_element, _ring_mask.float(), _lvt_c)
                _snr_ps = _per_sample_snr(noise_process, timesteps, self.noise_scheduler, alphas_cumprod_cached)

                def _nan_to_none(_vals):
                    return [None if v != v else v for v in _vals]

                self._last_loss_vs_t = {
                    "t": timesteps.detach().float().cpu().tolist(),
                    "snr": _snr_ps.detach().float().cpu().tolist(),
                    "eps_known": _nan_to_none(_eps_known_ps.detach().cpu().tolist()),
                    "eps_gen": _nan_to_none(_eps_gen_ps.detach().cpu().tolist()),
                    "eps_seam": _nan_to_none(_eps_seam_ps.detach().cpu().tolist()),
                }
            # Cross-seam error-continuity aux term (grad-carrying), opt-in via
            # outpaint_seam_grad_lambda (default 0.0 = off, term not computed,
            # loss byte-identical to today). Computed DIRECTLY on the native
            # prediction-space error e = model_pred - target (no x0
            # reconstruction needed -- per-sample-scalar timestep factor makes
            # any spatial finite difference of the x0-space error proportional
            # to the same finite difference in native space; see
            # scratchpad/outpaint_seam_auxloss.md SS2.2).
            _seam_lambda = float(getattr(self, "outpaint_seam_grad_lambda", 0.0) or 0.0)
            if _seam_lambda > 0.0:
                _e = model_pred.float() - target.float()
                _mx = (_known_mask[..., :, 1:] != _known_mask[..., :, :-1]).float()
                _my = (_known_mask[..., 1:, :] != _known_mask[..., :-1, :]).float()
                _num = ((_e[..., :, 1:] - _e[..., :, :-1]).pow(2) * _mx).sum([1, 2, 3]) \
                    + ((_e[..., 1:, :] - _e[..., :-1, :]).pow(2) * _my).sum([1, 2, 3])
                _den = (_mx.sum([1, 2, 3]) + _my.sum([1, 2, 3])) * float(_e.shape[1]) + 1e-8
                seam_grad_per_sample = _num / _den  # [B], per-cross-seam-pair normalized
            else:
                seam_grad_per_sample = None
            loss_per_element = loss_per_element * _wm
            # Opt-in (outpaint_loss_normalize, default False = unchanged behavior):
            # loss_per_sample below is a plain .mean([1,2,3]) over ALL C*H*W
            # elements, so the weighted sum gets diluted by the total pixel count
            # regardless of how much of it is full-weight (generate) vs
            # down-weighted (known) -- a larger generate-region rect (more
            # full-weight pixels) yields a larger per-sample loss purely from
            # area, not learning signal. When enabled, divide each sample's
            # weighted loss by that sample's own mean weight so the reduction is
            # effectively per-weighted-pixel and decoupled from rect size.
            if getattr(self, 'outpaint_loss_normalize', False):
                _sample_mean_w = _wm.mean(dim=[1, 2, 3], keepdim=True).clamp_min(1e-8)
                loss_per_element = loss_per_element / _sample_mean_w
        else:
            self._last_gen_region_loss = None
            self._last_seam_ring_loss = None
            self._last_loss_vs_t = None
            seam_grad_per_sample = None
        loss_per_sample = loss_per_element.mean([1, 2, 3])
        if seam_grad_per_sample is not None:
            # Added BEFORE Min-SNR so the aux term inherits the same per-sample
            # timestep balancing as the main loss (constraint 4 in the design doc).
            loss_per_sample = loss_per_sample + _seam_lambda * seam_grad_per_sample

        # Apply Min-SNR gamma weighting
        if self.min_snr_gamma > 0 and prediction_target == "epsilon":
            loss_per_sample_weighted = apply_snr_weight(
                loss_per_sample, timesteps, self.noise_scheduler, self.min_snr_gamma,
                alphas_cumprod_cached=alphas_cumprod_cached
            )
        else:
            loss_per_sample_weighted = loss_per_sample

        mse_loss = loss_per_sample_weighted.mean()
        loss = mse_loss

        # Reconstruction loss (monitoring only, no gradients for ControlNet training)
        with torch.no_grad():
            predicted_latent = predict_original_latent_unified(
                noise_process=noise_process,
                prediction_target=prediction_target,
                noise_scheduler=self.noise_scheduler,
                noisy_latents=noisy_latents,
                model_pred=model_pred,
                timesteps=timesteps,
            )
            recon_loss_per_element = F.mse_loss(predicted_latent.float(), latents.float(), reduction="none")
            recon_loss = recon_loss_per_element.mean()

            # Loss-vs-timestep instrumentation, x0-space half (see eps-space block
            # above for rationale). This is where the diagnostic reads cleanest
            # (design doc G3): unlike eps-space MSE, x0-space error genuinely rises
            # with noise level for unanchored (generate) pixels, so a known-region
            # curve that stays flat/low at high t is real evidence of anchoring.
            # recon_loss_per_element already exists unconditionally above (used for
            # the always-on `recon_loss` monitor) -- region-masking it here is free,
            # no extra forward/backward and no new tensor materialized beyond the
            # per-sample reductions themselves.
            if self._last_loss_vs_t is not None:
                _lvt_c2 = float(recon_loss_per_element.shape[1])
                _x0_known_ps = _per_sample_masked_mean(recon_loss_per_element, _known_mask, _lvt_c2)
                _x0_gen_ps = _per_sample_masked_mean(recon_loss_per_element, _gen_mask, _lvt_c2)
                self._last_loss_vs_t["x0_known"] = _nan_to_none(_x0_known_ps.detach().cpu().tolist())
                self._last_loss_vs_t["x0_gen"] = _nan_to_none(_x0_gen_ps.detach().cpu().tolist())

        if profile_vram:
            print_vram_usage("[train_step_controlnet] After loss calculation")

        pred_loss_value = mse_loss.item()
        recon_loss_value = recon_loss.item()

        # Cleanup
        del noise, noisy_latents, model_pred, target, recon_loss, predicted_latent
        if controlnet_output is not None:
            del down_block_res_samples, mid_block_res_sample
        if added_cond_kwargs is not None:
            del added_cond_kwargs

        return loss, pred_loss_value, recon_loss_value

    # ============================================================
    # FLUX.2 Position ID Helpers
    # ============================================================

    def _flux2_prepare_text_ids(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        Prepare 4D position IDs for FLUX.2 text embeddings.

        FLUX.2 uses 4D position coordinates: (T, H, W, L)
        - T: Time coordinate (0 for text)
        - H: Height coordinate (0 for text - dummy dimension)
        - W: Width coordinate (0 for text - dummy dimension)
        - L: Sequence position (0 to seq_len-1)

        Args:
            prompt_embeds: Text embeddings [B, seq_len, hidden_dim]

        Returns:
            text_ids: Position IDs [B, seq_len, 4]
        """
        batch_size, seq_len, _ = prompt_embeds.shape
        out_ids = []

        for _ in range(batch_size):
            t = torch.arange(1)  # Time: 0
            h = torch.arange(1)  # Height: 0 (dummy)
            w = torch.arange(1)  # Width: 0 (dummy)
            l = torch.arange(seq_len)  # Sequence position
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def _flux2_prepare_latent_ids(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Prepare 4D position IDs for FLUX.2 image latents.

        FLUX.2 uses 4D position coordinates: (T, H, W, L)
        - T: Time coordinate (0 for single image)
        - H: Height coordinate (0 to height-1)
        - W: Width coordinate (0 to width-1)
        - L: Channel/patch coordinate (0 for unpatchified)

        Args:
            latents: Image latents [B, C, H, W]

        Returns:
            img_ids: Position IDs [B, H*W, 4]
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # Time: 0
        h = torch.arange(height)  # Height positions
        w = torch.arange(width)  # Width positions
        l = torch.arange(1)  # Patch/channel: 0

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    def _flux2_pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents from (B, C, H, W) to (B, H*W, C) for FLUX.2 transformer.

        Args:
            latents: Image latents [B, C, H, W]

        Returns:
            packed_latents: [B, H*W, C]
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    def _flux2_encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt for FLUX.2 using Qwen3 text encoder.

        FLUX.2 Klein uses Qwen3 with hidden states from layers 9, 18, 27.
        Output is concatenated: (B, seq_len, 3 * hidden_dim)

        IMPORTANT: This must match pipeline.py _flux2_encode_prompt() exactly,
        including chat template application, attention_mask, and use_cache settings.

        Args:
            prompt: Text prompt
            max_sequence_length: Maximum sequence length

        Returns:
            Tuple of (prompt_embeds, text_ids)
        """
        # Apply chat template (must match inference exactly)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        # Forward through text encoder (must match inference exactly)
        with torch.no_grad():
            output = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Extract hidden states from specified layers (9, 18, 27 for Klein 4B)
        # FLUX.2 Klein uses layers 9, 18, 27 (1-indexed), which are indices 9, 18, 27 in hidden_states array
        # This must match inference code in pipeline.py:_flux2_encode_prompt()
        hidden_states_layers = (9, 18, 27)  # Same as inference

        # Stack hidden states
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=self.training_dtype, device=self.device)

        # Reshape: (B, num_layers, seq_len, hidden_dim) -> (B, seq_len, num_layers * hidden_dim)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        # Prepare text IDs
        text_ids = self._flux2_prepare_text_ids(prompt_embeds).to(self.device)

        return prompt_embeds, text_ids

    # ============================================================
    # Sample Generation
    # ============================================================

    def generate_sample(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
        current_step: int = 0,
        schedule_type: str = "uniform",
        condition_image_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
    ) -> "Image.Image":
        """
        Generate sample image during training (SD/SDXL).
        Uses custom_sampling_loop() - EXACTLY the same method as normal txt2img generation.

        Args:
            prompt: Text prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            seed: Random seed (-1 for random)
            current_step: Current training step (for logging)
            schedule_type: Timestep schedule type (uniform, karras, exponential)

        Returns:
            PIL Image
        """
        from core.training.ops import sd_sdxl_ops
        return sd_sdxl_ops.generate_sample(
            self,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            current_step=current_step,
            schedule_type=schedule_type,
            condition_image_path=condition_image_path,
            reference_image_path=reference_image_path,
        )

    # ============================================================
    # Unified sample dispatch (shared by step-0 + periodic sampling)
    # ============================================================

    def _dispatch_sample(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        negative_prompt: str = "",
        reference_image_path: Optional[str] = None,
        condition_image_path: Optional[str] = None,
        current_step: int = 0,
        schedule_type: str = "uniform",
    ) -> Optional[Image.Image]:
        """Route a sample request to the correct per-architecture helper.

        Both the step-0 verification block and the periodic sampling block call
        this single method, so their architecture coverage can never drift apart
        (the class of bug where an arch was wired into one block but not the
        other and crashed via the SD/SDXL ``generate_sample`` fallback).

        Returns a PIL image, or ``None`` when the architecture cannot sample yet
        (ideogram4) — callers must skip saving in that case rather than crash.
        """
        # P7: SampleContext (frozen in P0 from this signature) is unpacked by
        # each arch handler's sample(). Single dispatch point keeps step-0 and
        # periodic sampling coverage identical across archs.
        from core.training.arch.base_arch import SampleContext
        sample_ctx = SampleContext(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt,
            reference_image_path=reference_image_path,
            condition_image_path=condition_image_path,
            current_step=current_step,
            schedule_type=schedule_type,
        )
        return self.arch.sample(self, sample_ctx)

    def _step0_marker_path(self) -> Path:
        return self.output_dir / "samples" / ".step0_done"

    def _step0_sample_done_for_this_run(self, step0_sample_path: Path) -> bool:
        """True only when the marker names THIS run's own DB row, not merely
        when the PNG exists -- path existence alone matches a different run
        that reused the same run_name (routes.py's ``mkdir(exist_ok=True)``
        inherits the previous run's output_dir and everything in it)."""
        if self.run_id is None or not step0_sample_path.exists():
            return False
        marker_path = self._step0_marker_path()
        if not marker_path.exists():
            return False
        try:
            return marker_path.read_text().strip() == str(self.run_id)
        except OSError:
            return False

    def _mark_step0_sample_done(self) -> None:
        if self.run_id is None:
            return
        marker_path = self._step0_marker_path()
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(str(self.run_id))
        except OSError:
            pass

    def _run_step0_sample_if_due(
        self,
        *,
        sample_every_n_steps: int,
        sample_width: int,
        sample_height: int,
        sample_guidance_scale: float,
        sample_steps: int,
        sample_seed: int,
        sample_schedule_type: str,
        global_step: int,
    ) -> None:
        """Step-0 base-model verification sample.

        Covers a relaunch after a crash between this sample's own save and the
        first checkpoint (both would otherwise dispatch a second "step 0"
        generation); a crash DURING the sample leaves no PNG and is not
        covered, since the guard has nothing to key off yet.
        """
        if not (sample_every_n_steps > 0 and global_step == 0):
            return
        step0_sample_path = self.output_dir / "samples" / f"step_{0:06d}_sample_0.png"
        if self._step0_sample_done_for_this_run(step0_sample_path):
            print(f"{self.log_prefix} [Step 0] Skipping sample: {step0_sample_path.name} "
                  f"was already produced by this run (run_id={self.run_id}; marker matches).")
            return
        if step0_sample_path.exists():
            print(f"{self.log_prefix} [Step 0] {step0_sample_path.name} exists but its marker "
                  f"does not name this run (run_id={self.run_id}); regenerating so the base-model "
                  f"check reflects this run, not whatever produced the existing file.")
        step0_prompt = self._sample_prompts[0].get('positive', 'a beautiful landscape') if self._sample_prompts else 'a beautiful landscape'
        print(f"{self.log_prefix} [Step 0] Generating sample to verify base model...")
        print(f"{self.log_prefix} [Step 0] Sample params: width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, steps={sample_steps}, seed={sample_seed}")
        sample = self._dispatch_sample(
            step0_prompt,
            width=sample_width,
            height=sample_height,
            num_inference_steps=sample_steps,
            guidance_scale=sample_guidance_scale,
            seed=sample_seed,
            current_step=0,
            schedule_type=sample_schedule_type,
        )
        # None => architecture can't sample yet; skip saving.
        if sample is not None:
            step0_sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample.save(step0_sample_path)
            self._mark_step0_sample_done()
            print(f"{self.log_prefix} [Step 0] Saved sample to {step0_sample_path.relative_to(self.output_dir)}")

    def _flux2_unpack_latents_with_ids(self, x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        """Unpack latents using position IDs: (B, H*W, C) -> (B, C, H, W)"""
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def _flux2_patchify_latents_for_training(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify latents for 2x2 patches: (B, 32, H, W) -> (B, 128, H/2, W/2)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    def _flux2_unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Unpatchify latents from 2x2 patches: (B, 128, H/2, W/2) -> (B, 32, H, W)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    # ============================================================
    # Latent Cache Management (to be added in continuation)
    # ============================================================

    def _setup_latent_caches(self, datasets: List[Any]) -> Dict[str, Any]:
        """
        Setup per-dataset latent caches.

        Args:
            datasets: List of dataset objects

        Returns:
            Dictionary mapping dataset_unique_id to LatentCache instance
        """
        from core.training.latent_cache import LatentCache, get_cache_base_dir

        latent_caches = {}
        # Use global cache directory (shared across all training runs)
        # This allows cache reuse when training the same dataset multiple times
        base_cache_dir = get_cache_base_dir()
        namespace = self._build_cache_namespace()
        print(f"{self.log_prefix} Using global latent cache directory: {base_cache_dir}")
        print(f"{self.log_prefix} Latent cache namespace (arch/VAE identity): {namespace}")

        for dataset in datasets:
            latent_caches[dataset.unique_id] = LatentCache(
                dataset_unique_id=dataset.unique_id,
                base_cache_dir=str(base_cache_dir),
                namespace=namespace,
            )
            cache_dir = Path(base_cache_dir) / dataset.unique_id / namespace
            print(f"{self.log_prefix} Setup latent cache for dataset '{dataset.unique_id}': {cache_dir}")

        return latent_caches

    def _build_cache_namespace(self) -> str:
        """
        Build the architecture/VAE-identity namespace for this run's disk caches.

        Latents (and text embeddings) are stored under
        ``{base}/{dataset_id}/{namespace}/`` so that caches encoded for one
        model family / VAE are never read back for another that shares the
        dataset. Reuses the trainer's own architecture flags (no parallel
        naming) — the same names ``ModelLoader.detect_model_type`` produces.
        """
        from core.training.latent_cache import build_cache_namespace

        # Arch string derives from the bound handler (plan P8). The handler's
        # ``name`` is the registry key, which ``resolve_arch_name`` computes from
        # the SAME ``is_<arch>`` flag-priority chain this method previously
        # inlined (arch/__init__.py) — and registry keys are asserted byte-equal
        # to these namespace strings (plan R6). So the emitted namespace is
        # byte-identical to the pre-P8 chain for every arch/config.
        arch = self.arch.name

        # VAE / TE identity only vary within SDXL (custom-arch swaps); other
        # architectures have an arch-determined VAE and TE.
        vae_type = getattr(self, "sdxl_vae_type", None) if arch == "sdxl" else None
        te_type = getattr(self, "sdxl_te_type", None) if arch == "sdxl" else None

        # Latent channel count directly encodes the shape that triggered the
        # channel-mismatch crash; derive from the loaded VAE when available.
        latent_channels = getattr(self, "vae_latent_channels", None)
        if latent_channels is None:
            try:
                latent_channels = int(self.vae.config.latent_channels)
            except Exception:
                latent_channels = None

        latent_dtype = None
        vdt = getattr(self, "vae_dtype", None)
        if vdt is not None:
            latent_dtype = str(vdt)

        return build_cache_namespace(
            arch=arch,
            vae_type=vae_type,
            te_type=te_type,
            latent_channels=latent_channels,
            latent_dtype=latent_dtype,
        )

    def _check_stop_requested(self):
        """Abort a long pre-run/encode phase promptly on a user stop request.

        Mirrors the in-loop stop-flag check in the training batch loop (see the
        ``.stop_training`` handling in train()): the API's
        ``TrainingProcess.stop()`` touches ``<output_dir>/.stop_training``. When
        present we remove the flag and raise ``KeyboardInterrupt`` so the user
        stop propagates through the exact same path a mid-training stop uses —
        non-zero process exit, which the monitor reports as a user-requested stop
        (run status 'stopped'), with the checkpoint/cleanup handled by the same
        ``except KeyboardInterrupt`` block. Already-written cache entries are
        valid per-file and are intentionally left in place.

        The check is a cheap ``Path.is_file()``; it is safe to call per item.
        """
        stop_flag_file = self.output_dir / ".stop_training"
        if stop_flag_file.is_file():
            print(f"\n{self.log_prefix} Stop flag detected during pre-encode, aborting...")
            try:
                stop_flag_file.unlink()
            except OSError:
                pass
            raise KeyboardInterrupt("Training stopped by user")

    def _validate_and_generate_latent_caches(
        self,
        datasets: List[Any],
        latent_caches: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        force_recache: bool = False,
    ):
        """
        Check latent caches and generate missing ones.

        Args:
            datasets: List of dataset objects
            latent_caches: Dictionary of latent caches
            progress_callback: Progress callback function
            force_recache: Force regenerate all caches even if they exist
        """
        if force_recache:
            print(f"{self.log_prefix} Force recache enabled: regenerating all latent caches...")
        else:
            print(f"{self.log_prefix} Checking and generating missing latent caches...")

        # Generate missing latents (this will skip already cached items unless force_recache=True)
        self._generate_missing_latents_with_model_offload(
            datasets=datasets,
            latent_caches=latent_caches,
            progress_callback=progress_callback,
            force_recache=force_recache,
        )

    def _generate_missing_latents_with_model_offload(
        self,
        datasets: List[Any],
        latent_caches: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        force_recache: bool = False,
    ):
        """
        Generate missing latents with model offloading for memory efficiency.

        Args:
            datasets: List of dataset objects
            latent_caches: Dictionary of latent caches
            progress_callback: Progress callback function
        """
        log_verbose(f"[Latent Cache] Generating latent cache with model offloading...")

        # Count total items
        total_items = sum(len(dataset.items) for dataset in datasets)
        processed_items = 0

        # This pre-training VAE encode phase does not need the training stack. If the
        # main model (U-Net/Transformer) + text encoders are left GPU-resident here they
        # co-reside with the VAE plus (on resume) the optimizer state — a batch-size-
        # independent ~47GB VRAM pin observed on SDXL full_finetune at step 0 that spills
        # into Windows shared memory. Offload them to CPU for the encode, restore in the
        # finally so an encode failure can never strand the model on CPU. Guards make this
        # a no-op when a component is already on CPU (e.g. cached-TE / block-swap setups).
        main_model = self._main_model_module()
        main_on_gpu = (
            main_model is not None
            and next(main_model.parameters()).device.type != "cpu"
        )
        te_on_gpu = (
            self.text_encoder is not None
            and next(self.text_encoder.parameters()).device.type != "cpu"
        )
        te2_on_gpu = (
            getattr(self, "is_sdxl", False)
            and getattr(self, "text_encoder_2", None) is not None
            and next(self.text_encoder_2.parameters()).device.type != "cpu"
        )

        try:
            if main_on_gpu:
                print(f"{self.log_prefix} Offloading main model to CPU for VAE encode phase...")
                self.move_main_model_to_cpu()
                self._relocate_main_model_optimizer_state("cpu")
            if te_on_gpu or te2_on_gpu:
                print(f"{self.log_prefix} Offloading text encoder(s) to CPU for VAE encode phase...")
                self.move_text_encoder_to_cpu()
            if (main_on_gpu or te_on_gpu or te2_on_gpu) and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Move VAE to GPU (only if not already there)
            vae_current_device = next(self.vae.parameters()).device
            if vae_current_device != self.device:
                print(f"{self.log_prefix} Moving VAE from {vae_current_device} to {self.device}...")
                self.vae.to(device=self.device, dtype=self.vae_dtype)
            else:
                print(f"{self.log_prefix} VAE already on {self.device}, skipping move")

            iteration_count = 0
            for dataset in datasets:
                cache = latent_caches[dataset.unique_id]

                # Log to file only (no console spam)
                log_verbose(f"[Latent Cache] Caching dataset {dataset.unique_id} ({len(dataset.items)} items)...")

                for item in tqdm(dataset.items, desc=f"Caching {dataset.unique_id}", disable=True):
                    # Abort promptly on user stop (raises KeyboardInterrupt; the
                    # finally below restores the training stack to its devices).
                    self._check_stop_requested()

                    # Video-clip item (P4/P5): item_type=="video" carries a
                    # video_path + clip window; encode a 5D clip latent via the
                    # video VAE (encode_and_cache_clip seam). item_type=="single"
                    # (stills) fall through to the still encode below, which for
                    # a 5D arch also yields a 5D T=1 latent (same train_step).
                    if self._temporal_spec() is not None and item.get("item_type") == "video":
                        try:
                            from core.training.video_loader import encode_and_cache_clip
                            _spec = self._temporal_spec()
                            # DETERMINISTIC (centered) window, not a random one.
                            # A cache entry is addressed by its window, so a
                            # randomly-sampled write can never be read back --
                            # the reader would compute a different key every
                            # time. `_video_clip_window(training=False)` is the
                            # same call the pre_encoded_cache READ makes, which
                            # is what makes the two agree by construction.
                            (v_path, v_w, v_h, clip_length, stride, fps,
                             window) = self._video_clip_window(item, training=False)
                            encode_and_cache_clip(
                                cache=cache,
                                video_path=v_path,
                                width=v_w, height=v_h,
                                clip_start=window.start_frame,
                                clip_length=clip_length,
                                stride=stride,
                                vae_encode_clip=lambda clip: self.arch.vae_encode_clip(self, clip),
                                fps=fps,
                                device=str(self.device),
                                spec=_spec,
                                start_time=window.start_time,
                                source_fps=fps,
                                tiling_policy=self._clip_vae_tiling_policy(),
                                audio_prep_version=self._clip_audio_prep_version(),
                                audio_encode_window=self._clip_audio_seam(v_path),
                            )
                            iteration_count += 1
                            processed_items += 1
                        except Exception as e:  # noqa: BLE001
                            print(f"{self.log_prefix} WARNING: video clip encode failed "
                                  f"({os.path.basename(str(item.get('video_path', '')))}): {e}")
                        continue

                    # Audio-clip item (Phase 8a, ACE-Step): item_type=="audio"
                    # carries an audio_path (no still-image concept for this
                    # arch); encode a 3D [1, T, 64] latent via the ACE-Step
                    # (Oobleck) VAE (encode_and_cache_audio seam). Mirrors the
                    # LTX-2.3 video-clip branch above; audio has no random
                    # clip-window sampling (see audio_loader.py's docstring —
                    # dataset clips are expected pre-trimmed to a consistent
                    # duration per training run).
                    if self.is_acestep and item.get("item_type") == "audio":
                        try:
                            from core.training.audio_loader import encode_and_cache_audio
                            a_path = item.get("audio_path") or item["image_path"]
                            clip_seconds = item.get("clip_seconds")
                            sample_rate = int(getattr(self, "acestep_sample_rate", 48000))
                            encode_and_cache_audio(
                                cache=cache,
                                audio_path=a_path,
                                clip_seconds=(float(clip_seconds) if clip_seconds else None),
                                vae_encode_audio=lambda wav: self.arch.vae_encode_audio(self, wav),
                                sample_rate=sample_rate,
                                device=str(self.device),
                            )
                            iteration_count += 1
                            processed_items += 1
                        except Exception as e:  # noqa: BLE001
                            print(f"{self.log_prefix} WARNING: audio clip encode failed "
                                  f"({os.path.basename(str(item.get('audio_path', '')))}): {e}")
                        continue

                    # Check if already cached (skip if force_recache is False)
                    image_path = item["image_path"]
                    width = item["width"]
                    height = item["height"]

                    if not force_recache and cache.has_latent(image_path, width, height):
                        processed_items += 1
                        continue

                    # Load and encode image
                    try:
                        image = Image.open(image_path)

                        latent = self.encode_image(
                            image=image,
                            target_width=width,
                            target_height=height,
                        )

                        # Save to cache
                        cache.save_latent(
                            image_path=image_path,
                            width=width,
                            height=height,
                            latents=latent,
                        )

                        iteration_count += 1

                    except Exception as e:
                        # Use repr() to avoid UnicodeEncodeError on Windows (cp932)
                        safe_path = os.path.basename(image_path)
                        try:
                            print(f"{self.log_prefix} ERROR encoding {safe_path}: {e}")
                        except UnicodeEncodeError:
                            # Fallback: encode-safe output
                            print(f"{self.log_prefix} ERROR encoding image (path contains non-ASCII chars): {e}")
                    finally:
                        # Clean up to prevent VRAM accumulation
                        if 'image' in locals():
                            image.close()
                            del image
                        if 'latent' in locals():
                            del latent
                        # Clear CUDA cache periodically (every 50 images)
                        if iteration_count % 50 == 0:
                            torch.cuda.empty_cache()

                    processed_items += 1

                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            phase="latent_cache",
                            step=processed_items,
                            total=total_items,
                        )

            # VAE stays on CPU (already there)
            log_verbose(f"[Latent Cache] Generation complete ({iteration_count} images encoded)")
        finally:
            # Restore the training stack to its pre-encode devices. In finally so a
            # failure during encode cannot leave the model/TEs stranded on CPU.
            if main_on_gpu:
                self.move_main_model_to_gpu()
                self._relocate_main_model_optimizer_state(self.device)
            if te_on_gpu or te2_on_gpu:
                self.move_text_encoder_to_gpu()
            if (main_on_gpu or te_on_gpu or te2_on_gpu) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _temporal_spec(self):
        """This trainer's ``TemporalSpec``, or None when the arch is not a video
        architecture (Phase 6a).

        This is the VIDEO predicate the clip paths branch on. It is declared on
        the arch handler (``ArchHandler.temporal``, the temporal analogue of
        ``pixel_align``) so a second video architecture is one table entry, not
        another ``is_<arch>`` flag threaded through six call sites. Today
        exactly one handler declares it — ltx2 — so every guard below is
        equivalent to the ``self.is_ltx2`` check it replaces.
        """
        return getattr(getattr(self, "arch", None), "temporal", None)

    def _clip_vae_tiling_policy(self) -> Optional[str]:
        """Token for the arch's clip-encode VAE tiling policy (cache key)."""
        return getattr(getattr(self, "arch", None), "clip_vae_tiling_policy", None)

    def _refuse_unsupported_audio_only_items(self, datasets) -> None:
        """Video archs (LTX-2.3, MiniMax-H3): a standalone ``item_type=="audio"``
        item has no encode path anywhere in this trainer -- these archs derive
        their audio latent from a PAIRED video item's own audio track (see e.g.
        ``minimax_h3_ops.vae_encode_audio_window``), not a separate dataset
        item. Left undetected, every latent-encoding mode's default branch
        falls through to ``Image.open(item["image_path"])`` on the audio file,
        which fails with PIL's "cannot identify image file" deep inside the
        training loop instead of here at setup.
        """
        if self.is_acestep or self._temporal_spec() is None:
            return
        if not any(item.get("item_type") == "audio"
                   for dataset in datasets for item in dataset.items):
            return
        raise ValueError(
            f"This dataset contains item_type=='audio' items, which "
            f"{getattr(self.arch, 'name', 'this architecture')} does not support as a "
            f"standalone training item -- its audio comes from a paired video item's own "
            f"audio track, not a separate dataset item. Remove the audio-only item(s) or "
            f"attach the audio as part of a video item instead.")

    def _annotate_video_items(self, datasets, base_resolutions) -> int:
        """Route video items through VideoBucketManager (P5 video wiring).

        For every ``item_type=="video"`` item, assigns a (÷pixel_align spatial
        bucket, clip_length) via ``VideoBucketManager.assign_video_to_bucket``
        and copies the resulting fields (``bucket_width``/``bucket_height``/
        ``clip_length``/``stride``/``num_frames``/``fps``/``target_fps``/
        ``item_type``/``video_path``) onto the item dict in place. These are
        exactly the keys the 5 video encode-site guards + ``_encode_video_clip``
        read. Image items are untouched (this only runs for a video arch and
        only visits item_type=="video" items), so image bucketing is
        byte-for-byte unchanged.

        The arch's ``TemporalSpec`` drives which clip lengths are legal and
        whether the clip is resampled to a fixed frame rate; ``None`` (a
        non-video arch) is an immediate no-op.

        Returns the number of video items annotated.
        """
        spec = self._temporal_spec()
        if spec is None:
            return 0

        from core.training.bucketing import VideoBucketManager

        base_res = base_resolutions or [1024]
        allowed = (
            self.config.get("ltx2_clip_lengths")
            or self.config.get("allowed_clip_lengths")
            or list(spec.default_clip_lengths)
        )
        stride = int(
            self.config.get("ltx2_clip_stride",
                            self.config.get("clip_stride", 1)) or 1
        )

        vbm = VideoBucketManager(
            base_resolutions=list(base_res),
            allowed_clip_lengths=list(allowed),
            stride=stride,
            temporal_spec=spec,
        )

        count = 0
        for dataset in datasets:
            for item in dataset.items:
                if item.get("item_type") != "video":
                    continue
                v_path = item.get("video_path") or item.get("image_path")
                width = int(item.get("width") or 0) or 1024
                height = int(item.get("height") or 0) or 1024
                num_frames = int(item.get("num_frames") or 0)
                _, video_info = vbm.assign_video_to_bucket(
                    video_path=v_path,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    caption=item.get("caption", ""),
                    fps=item.get("fps"),
                    dataset_unique_id=getattr(dataset, "unique_id", None),
                )
                # Copy the bucket-derived fields onto the training item dict.
                item["item_type"] = "video"
                item["video_path"] = video_info["video_path"]
                item["bucket_width"] = video_info["bucket_width"]
                item["bucket_height"] = video_info["bucket_height"]
                item["clip_length"] = video_info["clip_length"]
                item["stride"] = video_info["stride"]
                item["num_frames"] = video_info["num_frames"]
                if video_info.get("fps") is not None:
                    item["fps"] = video_info["fps"]
                # Fixed-fps archs only (MiniMax-H3): the rate the RESAMPLED clip
                # plays at, distinct from item["fps"] (still the source rate,
                # which the resampler and the cache key need). LTX-2.3 items
                # never gain this key.
                if video_info.get("target_fps") is not None:
                    item["target_fps"] = video_info["target_fps"]
                # Keep width/height consistent with the chosen ÷32 spatial bucket so
                # any code reading item["width"]/["height"] agrees with the encode.
                item["width"] = video_info["bucket_width"]
                item["height"] = video_info["bucket_height"]
                count += 1

        if count:
            print(f"{self.log_prefix} [{self.arch.name} video] Assigned {count} video "
                  f"item(s) to (spatial÷{vbm.divisibility}, clip_length) buckets: "
                  f"{vbm.get_bucket_counts()}")
        return count

    def _encode_video_clip(self, item: Dict[str, Any]) -> torch.Tensor:
        """Encode a video-clip item to a 5D latent ``[1, C, T_lat, H', W']``.

        Mirrors the video-clip branch in ``_generate_latent_cache_with_offloading``
        but returns the latent directly (no cache write) so the swap /
        on-the-fly latent paths can route ``item_type=="video"`` items through the
        video VAE instead of ``PIL.Image.open`` (which cannot read ``.webm``).

        Uses ``video_loader.sample_clip_window`` + ``load_clip`` (clip window from
        the item's VideoBucketManager params, with the arch's ``TemporalSpec`` so
        a fixed-fps arch resamples instead of relabelling) and
        ``arch.vae_encode_clip``. The VAE is assumed already GPU-resident
        (callers move it before the encode loop, same as the still path).
        """
        from core.training.video_loader import load_clip

        spec = self._temporal_spec()
        v_path, v_w, v_h, clip_length, stride, source_fps, window = self._video_clip_window(
            item, training=True)
        clip = load_clip(
            v_path, clip_length, window.start_frame, stride,
            target_w=v_w, target_h=v_h,
            spec=spec, start_time=window.start_time, source_fps=source_fps,
        )  # [T, C, H, W]
        # arch.vae_encode_clip(trainer, clip) -> [1, C, T_lat, H', W'] (normalised).
        latents = self.arch.vae_encode_clip(self, clip)
        # The AUDIO half of the SAME window, for an arch whose packed sequence
        # carries audio rows (MiniMax-H3). Stashed on the item because that is
        # where the batch assembly collects per-CLIP payloads from
        # (_minimax_h3_batch_audio); no-op for a video-only arch, whose handler
        # inherits the base-class seam and returns None.
        self._stash_clip_audio(item, spec, v_path, window, clip_length, stride)
        return latents

    def _video_clip_window(self, item: Dict[str, Any], *, training: bool):
        """Resolve one video item's clip window + geometry (shared by every video
        encode site, so the window a cache WRITE used and the window a cache READ
        recomputes can never drift apart).

        ``training=True`` samples a RANDOM window (a fresh crop of the timeline
        every step, which is the point of the swap / on-the-fly modes);
        ``training=False`` takes the CENTERED window, which is deterministic and
        is therefore the only one a disk cache can address.
        """
        from core.training.video_loader import sample_clip_window

        spec = self._temporal_spec()
        v_path = item.get("video_path") or item["image_path"]
        v_w = int(item.get("bucket_width", item.get("width")))
        v_h = int(item.get("bucket_height", item.get("height")))
        clip_length = int(item["clip_length"])
        stride = int(item.get("stride", 1))
        source_fps = item.get("fps")
        window = sample_clip_window(
            int(item.get("num_frames", clip_length)),
            clip_length, stride, training=training,
            spec=spec, source_fps=source_fps,
        )
        return v_path, v_w, v_h, clip_length, stride, source_fps, window

    def _stash_clip_audio(self, item, spec, video_path, window, clip_length, stride):
        """Encode this window's AUDIO latent and put it on the item for the batch
        assembly to collect.

        A source without an audio track yields ``None``, which the train step
        reads as an explicit SILENT window rather than as missing data.
        """
        arch = getattr(self, "arch", None)
        if arch is None or spec is None:
            return
        duration = spec.clip_duration(clip_length, stride)
        if duration is None or getattr(arch, "clip_audio_prep_version", None) is None:
            item["_clip_audio_latent"] = None
            return
        try:
            item["_clip_audio_latent"] = arch.vae_encode_clip_audio(
                self, video_path, float(window.start_time), float(duration))
        except Exception as e:  # noqa: BLE001
            print(f"{self.log_prefix} WARNING: clip audio encode failed for "
                  f"{os.path.basename(str(video_path))}: {e}")
            item["_clip_audio_latent"] = None

    def _load_or_encode_video_clip(self, item: Dict[str, Any], cache) -> torch.Tensor:
        """Video-clip latent for the ``pre_encoded_cache`` mode, THROUGH the clip
        cache.

        The generic ``cache.load_latent(image_path, w, h)`` path is keyed by
        ``compute_image_hash`` and cannot address a clip record at all (clip
        records are keyed by WINDOW), so a video item that reached it was
        guaranteed to miss and fall into ``_regenerate_single_latent``, which
        opens the path with PIL. This routes video items to the window-level
        record instead -- the same ``encode_and_cache_clip`` seam the pre-encode
        pass writes with, on the deterministic (centered) window, so the write and
        the read address the same key.
        """
        from core.training.video_loader import encode_and_cache_clip

        spec = self._temporal_spec()
        v_path, v_w, v_h, clip_length, stride, source_fps, window = self._video_clip_window(
            item, training=False)
        result = encode_and_cache_clip(
            cache=cache,
            video_path=v_path, width=v_w, height=v_h,
            clip_start=window.start_frame, clip_length=clip_length, stride=stride,
            vae_encode_clip=lambda clip: self.arch.vae_encode_clip(self, clip),
            fps=source_fps, device=str(self.device), spec=spec,
            start_time=window.start_time, source_fps=source_fps,
            tiling_policy=self._clip_vae_tiling_policy(),
            audio_prep_version=self._clip_audio_prep_version(),
            audio_encode_window=self._clip_audio_seam(v_path),
            return_record=True,
        )
        item["_clip_audio_latent"] = result.get("audio_latents")
        return result["latents"]

    def _clip_audio_seam(self, video_path: str):
        """The ``(start_sec, duration_sec) -> audio latent`` callable for this
        arch, or None when its clip record is video-only."""
        arch = getattr(self, "arch", None)
        if arch is None or getattr(arch, "clip_audio_prep_version", None) is None:
            return None
        return lambda start_sec, duration: arch.vae_encode_clip_audio(
            self, video_path, float(start_sec), float(duration))

    def _clip_audio_prep_version(self) -> Optional[str]:
        """Token for the arch's clip-record audio preprocessing chain (cache key)."""
        return getattr(getattr(self, "arch", None), "clip_audio_prep_version", None)

    def _arch_pixel_align(self) -> int:
        """The multiple every STILL canvas dimension must be a multiple of.

        One reader for the three places that need it: the still `BucketManager`,
        the one-time base-area fit and the per-epoch re-fit. They were three
        separate expressions and the bucket one was a hardcoded `8`, which is the
        only value that is wrong for a patchified DiT: at base 640, 37 of the 42
        generated buckets are not /32 and 29 are not /16, and a 1120x360 image
        gives MiniMax-H3 an odd latent height that `patchify_video_latents`
        refuses -- mid-run, after the whole caching pass.

        SD/SDXL declare 8, so their bucket lists are unchanged; 512 and 1024
        generate an identical list under any of 8/16/32 in any case.
        """
        return int(getattr(getattr(self, "arch", None), "pixel_align", 8))

    @staticmethod
    def _still_latent_5d_is_valid(latent, width: int, height: int) -> bool:
        """Is this cached 5D STILL latent ``[1, C, 1, H/vsf, W/vsf]`` this bucket's?

        A video arch routes a still through the same 5D ``train_step`` as a clip
        (``ltx2_ops.vae_encode`` / ``minimax_h3_ops.vae_encode``, T=1), so its
        cached record carries a TEMPORAL axis at index 2. The generic 4D check
        compares that axis against ``height // 8`` and mismatches on EVERY still
        (1 != 48 at 384), so the loop needs a 5D-aware predicate rather than a
        blanket skip.

        The spatial factor is per-arch (MiniMax-H3 /16, LTX-2.3 /32) and is not
        declared anywhere this loop can read, so what is checked is what is
        knowable: exactly one latent frame, and both axes reduced from THIS
        bucket's canvas by the SAME integer factor. That still catches the case
        the branch exists for -- a record cached at a different bucket -- because
        a stale ``(lh, lw)`` does not generally divide the new canvas evenly by
        one common factor (e.g. a 512x768 H3 record against a 384x640 bucket:
        384 % 32 == 0 but 640 % 48 != 0).
        """
        if getattr(latent, "ndim", 0) != 5 or int(latent.shape[2]) != 1:
            return False
        lh, lw = int(latent.shape[3]), int(latent.shape[4])
        if lh <= 0 or lw <= 0:
            return False
        if int(height) % lh or int(width) % lw:
            return False
        return (int(height) // lh) == (int(width) // lw)

    def _regenerate_single_latent(
        self,
        image_path: str,
        width: int,
        height: int,
        cache: Any,
        latent_caches: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Regenerate a single latent on-the-fly during training with model offloading.

        This is called when latent cache is corrupted or has shape mismatch.
        Offloads training components temporarily and loads VAE to GPU.

        Args:
            image_path: Path to source image
            width: Target width
            height: Target height
            cache: LatentCache object
            latent_caches: Dictionary of all latent caches (unused, for future use)

        Returns:
            Regenerated latent tensor
        """
        # MiniT2I: pixel-space, no VAE. The "latent" is just the [-1,1] RGB image,
        # so encode_image is a cheap CPU op — no model offloading needed.
        if self.is_minit2i:
            print(f"{self.log_prefix} [Latent Regeneration] MiniT2I pixel-latent for: {image_path}")
            image = Image.open(image_path)
            latent = self.encode_image(image=image, target_width=width, target_height=height)
            image.close()
            cache.save_latent(image_path=image_path, width=width, height=height, latents=latent)
            return latent

        print(f"{self.log_prefix} [Latent Regeneration] Offloading models...")

        # Save current device states
        if self.is_zimage:
            transformer_device = next(self.transformer.parameters()).device
            text_encoder_device = next(self.text_encoder.parameters()).device
        else:
            # NOT `self.unet`: every DiT arch leaves it None (see e.g.
            # minimax_h3_ops.load_components), so the bare attribute access turns
            # this FALLBACK -- reached from a cache warning, i.e. a path that is
            # meant to recover -- into an uncaught
            # `AttributeError: 'NoneType' object has no attribute 'parameters'`.
            # `_main_model_module()` is the existing shared resolver
            # (Transformer for DiT archs, U-Net otherwise) and mirrors
            # move_main_model_to_cpu/gpu, so the offload below moves the module
            # that is actually resident.
            main_module = self._main_model_module()
            unet_device = (next(main_module.parameters()).device
                           if main_module is not None else None)
            if self.text_encoder:
                text_encoder_device = next(self.text_encoder.parameters()).device
            if self.is_sdxl and self.text_encoder_2:
                text_encoder_2_device = next(self.text_encoder_2.parameters()).device

        vae_device = next(self.vae.parameters()).device

        try:
            # Offload training components to CPU
            if self.is_zimage:
                if transformer_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving Transformer to CPU...")
                    self.transformer.to('cpu')
                if text_encoder_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving Text Encoder to CPU...")
                    self.text_encoder.to('cpu')
            else:
                if main_module is not None and unet_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving main model to CPU...")
                    main_module.to('cpu')
                if self.text_encoder and text_encoder_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving Text Encoder to CPU...")
                    self.text_encoder.to('cpu')
                if self.is_sdxl and self.text_encoder_2 and text_encoder_2_device != torch.device('cpu'):
                    print(f"{self.log_prefix} [Latent Regeneration] Moving Text Encoder 2 to CPU...")
                    self.text_encoder_2.to('cpu')

            torch.cuda.empty_cache()

            # Move VAE to GPU
            if vae_device != self.device:
                print(f"{self.log_prefix} [Latent Regeneration] Moving VAE to GPU...")
                self.vae.to(device=self.device, dtype=self.vae_dtype)

            # Load and encode image
            print(f"{self.log_prefix} [Latent Regeneration] Encoding image: {image_path}")
            image = Image.open(image_path)
            latent = self.encode_image(
                image=image,
                target_width=width,
                target_height=height,
            )
            image.close()

            # Save to cache
            cache.save_latent(
                image_path=image_path,
                width=width,
                height=height,
                latents=latent,
            )
            print(f"{self.log_prefix} [Latent Regeneration] Latent regenerated and saved to cache")

        finally:
            # Restore original device states
            print(f"{self.log_prefix} [Latent Regeneration] Restoring models...")
            if self.is_zimage:
                if transformer_device != torch.device('cpu'):
                    self.transformer.to(transformer_device)
                if text_encoder_device != torch.device('cpu'):
                    self.text_encoder.to(text_encoder_device)
            else:
                if main_module is not None and unet_device != torch.device('cpu'):
                    main_module.to(unet_device)
                if self.text_encoder and text_encoder_device != torch.device('cpu'):
                    self.text_encoder.to(text_encoder_device)
                if self.is_sdxl and self.text_encoder_2 and text_encoder_2_device != torch.device('cpu'):
                    self.text_encoder_2.to(text_encoder_2_device)

            if vae_device != self.device:
                self.vae.to(device=vae_device, dtype=self.vae_dtype)

            torch.cuda.empty_cache()
            print(f"{self.log_prefix} [Latent Regeneration] Models restored")

        return latent

    def _load_or_regenerate_acestep_audio_latent(self, item: Dict[str, Any], cache: Any) -> torch.Tensor:
        """Load (or regenerate-and-cache) the ACE-Step audio-clip latent for one
        dataset item, mirroring ``_regenerate_single_latent``'s role for still
        images/``_regenerate_single_latent``'s device-offload dance -- but keyed
        by ``(audio_path, clip_seconds, sample_rate)`` via
        ``LatentCache.load_audio_latent``/``save_audio_latent``
        (``compute_audio_hash``), NOT the image ``(width, height)`` scheme
        ``_regenerate_single_latent`` uses (audio latents have no spatial axis
        at all -- see ``audio_loader.py`` / this module's ACE-Step docstrings).

        Cache hit: returns the cached ``[1, T, 64]`` latent directly (moved to
        ``self.device``), no model movement needed. Cache miss: offloads the
        DiT (and text encoder, if resident) to CPU, moves the VAE to GPU,
        VAE-encodes the audio file fresh via
        ``audio_loader.encode_and_cache_audio`` (which also persists the
        result, so subsequent items / epochs / resumed runs hit the cache),
        then restores the pre-call device layout -- exactly the same
        offload/restore contract as ``_regenerate_single_latent``, adapted for
        ACE-Step's component set (``.transformer``/``.text_encoder``, no
        U-Net / text_encoder_2).
        """
        from core.training.audio_loader import encode_and_cache_audio

        audio_path = item.get("audio_path") or item["image_path"]
        _clip_seconds = item.get("clip_seconds")
        clip_seconds = float(_clip_seconds) if _clip_seconds else None
        sample_rate = int(getattr(self, "acestep_sample_rate", 48000))

        cached = cache.load_audio_latent(audio_path, clip_seconds, sample_rate, device=str(self.device))
        if cached is not None:
            return cached

        print(f"{self.log_prefix} [Latent Regeneration] Audio cache miss for "
              f"{os.path.basename(str(audio_path))}, regenerating...")

        transformer_device = next(self.transformer.parameters()).device
        text_encoder_device = (
            next(self.text_encoder.parameters()).device if self.text_encoder is not None else None
        )
        vae_device = next(self.vae.parameters()).device

        try:
            if transformer_device != torch.device('cpu'):
                print(f"{self.log_prefix} [Latent Regeneration] Moving ACE-Step DiT to CPU...")
                self.transformer.to('cpu')
            if text_encoder_device is not None and text_encoder_device != torch.device('cpu'):
                print(f"{self.log_prefix} [Latent Regeneration] Moving Text Encoder to CPU...")
                self.text_encoder.to('cpu')

            torch.cuda.empty_cache()

            if vae_device != self.device:
                print(f"{self.log_prefix} [Latent Regeneration] Moving VAE to GPU...")
                self.vae.to(device=self.device, dtype=self.vae_dtype)

            latent = encode_and_cache_audio(
                cache=cache,
                audio_path=audio_path,
                clip_seconds=clip_seconds,
                vae_encode_audio=lambda wav: self.arch.vae_encode_audio(self, wav),
                sample_rate=sample_rate,
                device=str(self.device),
            )
            print(f"{self.log_prefix} [Latent Regeneration] Audio latent regenerated and saved to cache")
        finally:
            print(f"{self.log_prefix} [Latent Regeneration] Restoring models...")
            if transformer_device != torch.device('cpu'):
                self.transformer.to(transformer_device)
            if text_encoder_device is not None and text_encoder_device != torch.device('cpu'):
                self.text_encoder.to(text_encoder_device)
            if vae_device != self.device:
                self.vae.to(device=vae_device, dtype=self.vae_dtype)
            torch.cuda.empty_cache()

        return latent.to(self.device)

    def _setup_text_encoder_caches(self, datasets: List[Any]) -> Dict[str, Path]:
        """
        Setup per-dataset text encoder cache directories for all architectures.
        Similar to _setup_latent_caches(), this only creates directories.

        Args:
            datasets: List of dataset objects

        Returns:
            Dictionary mapping dataset_unique_id to cache directory path
        """
        from pathlib import Path
        from core.training.latent_cache import get_cache_base_dir

        base_dir = Path(get_cache_base_dir())
        text_encoder_caches = {}
        namespace = self._build_cache_namespace()

        arch_name = ("Z-Image" if self.is_zimage else
                     "Lens" if self.is_lens else
                     ("SDXL" if self.is_sdxl else "SD1.5"))
        print(f"{self.log_prefix} Setting up text encoder cache directories ({arch_name})...")
        print(f"{self.log_prefix} Using global cache directory: {base_dir}")
        print(f"{self.log_prefix} Text embedding cache namespace (arch identity): {namespace}")

        for dataset in datasets:
            cache_dir = base_dir / dataset.unique_id / namespace / "text_embeddings"
            cache_dir.mkdir(parents=True, exist_ok=True)
            text_encoder_caches[dataset.unique_id] = cache_dir
            print(f"{self.log_prefix} Setup text encoder cache for dataset '{dataset.unique_id}': {cache_dir}")

        return text_encoder_caches

    def _text_cache_key(self, caption: str, lyrics: str = "") -> str:
        """Compute the on-disk text-embedding cache hash for one (caption,
        lyrics) pair.

        ``lyrics`` is ACE-Step ONLY (see ``ops/acestep_ops.py``'s module
        docstring); every other arch always calls this with ``lyrics=""``.
        When lyrics is empty this returns EXACTLY the historical
        ``md5(caption)`` key (so every existing cache entry, and every
        non-ACE-Step arch, is byte-identical / fully backward compatible).
        A non-empty lyrics string folds into the hash so two items sharing a
        caption but differing in lyrics never collide on the same cache file
        (the critical correctness point for per-item lyrics support).
        """
        import hashlib
        if lyrics:
            return hashlib.md5(f"{caption}\x1elyrics:{lyrics}".encode()).hexdigest()
        return hashlib.md5(caption.encode()).hexdigest()

    def _validate_and_generate_text_encoder_caches(
        self,
        datasets: List[Any],
        text_encoder_caches: Dict[str, Path],
        progress_callback: Optional[Callable] = None,
        epoch_num: Optional[int] = None,
    ):
        """
        Check text encoder caches and encode missing captions.
        Similar to _validate_and_generate_latent_caches(), this generates missing embeddings.

        Args:
            datasets: List of dataset objects
            text_encoder_caches: Dictionary mapping dataset_unique_id to cache directory
            progress_callback: Progress callback function
            epoch_num: Current epoch number (for logging)
        """
        arch_name = ("Z-Image" if self.is_zimage else
                     "Lens" if self.is_lens else
                     ("SDXL" if self.is_sdxl else "SD1.5"))
        epoch_info = f" (Epoch {epoch_num + 1})" if epoch_num is not None else ""
        print(f"{self.log_prefix} Validating and generating text encoder caches ({arch_name}){epoch_info}...")

        # Collect captions per dataset
        dataset_captions = {}
        total_captions = 0

        for dataset in datasets:
            unique_pairs = set()
            caption_samples = []
            for item in dataset.items:
                caption = item.get("caption", "")
                # ACE-Step ONLY: lyrics is a SEPARATE per-item conditioning
                # signal (see ops/acestep_ops.py's module docstring); every
                # other arch's item dicts never carry a "lyrics" key, so this
                # is always "" for them (identical dedup/hash behavior as
                # before this field existed).
                lyrics = item.get("lyrics", "") if self.is_acestep else ""
                # Cache EVERY (caption, lyrics) pair, including the empty
                # string(s). An empty caption/lyrics is a legitimate
                # (unconditional / instrumental) conditioning; skipping it here
                # would leave those items with no disk cache entry, so at
                # train time they fall into the pre_encoded on-the-fly fallback
                # while the Text Encoder is offloaded to CPU -> device-mismatch
                # crash. md5("") is a valid key and encode_caption("") is the
                # same unconditional encode the swap_onthefly path already runs.
                unique_pairs.add((caption, lyrics))
                if caption and len(caption_samples) < 3:
                    caption_samples.append(caption)
            dataset_captions[dataset.unique_id] = unique_pairs
            total_captions += len(unique_pairs)
            _pair_note = " (caption+lyrics pairs)" if self.is_acestep else ""
            print(f"{self.log_prefix} Dataset '{dataset.unique_id}': {len(unique_pairs)} unique captions{_pair_note}")
            if caption_samples and epoch_num is not None:
                print(f"{self.log_prefix}   Sample captions (epoch {epoch_num + 1}):")
                for i, sample in enumerate(caption_samples[:3], 1):
                    print(f"{self.log_prefix}     [{i}] {sample[:80]}...")

        print(f"{self.log_prefix} Total unique captions across all datasets: {total_captions}")

        # Encode missing captions for each dataset
        total_encoded = 0
        total_cached = 0

        # Move text encoder(s) to GPU for encoding
        self.move_text_encoder_to_gpu()

        try:
            for dataset in datasets:
                cache_dir = text_encoder_caches[dataset.unique_id]
                pairs = dataset_captions[dataset.unique_id]

                # Check which (caption, lyrics) pairs are missing
                captions_to_encode = []
                for caption, lyrics in pairs:
                    caption_hash = self._text_cache_key(caption, lyrics)
                    embeds_path = cache_dir / f"{caption_hash}_embeds.pt"

                    # Check auxiliary data file (architecture-specific)
                    if self.is_zimage or self.is_lens or self.is_ideogram4 or self.is_minit2i or self.is_krea2:
                        auxiliary_path = cache_dir / f"{caption_hash}_mask.pt"
                    elif self.is_ltx2:
                        # LTX-2.3 aux is a dict {audio_text_embedding, mask, fps};
                        # persisted (cannot be cheaply reconstructed like anima).
                        auxiliary_path = cache_dir / f"{caption_hash}_ltx2aux.pt"
                    elif self.is_minimax_h3:
                        # MiniMax-H3 aux is a dict {num_text_tokens}; persisted
                        # alongside the embedding because the packed layout's row
                        # count depends on it (see encode_caption).
                        auxiliary_path = cache_dir / f"{caption_hash}_h3aux.pt"
                    elif self.is_acestep:
                        # ACE-Step aux is a dict {text_attention_mask,
                        # lyric_hidden_states, lyric_attention_mask}; persisted
                        # (mirrors ltx2's aux-dict pattern). "v2" suffix: the aux
                        # schema gained the two lyric_* keys (per-item lyrics
                        # follow-up) -- old "_acestepaux.pt" files predate them
                        # and would fail collate_aux's key check, so this bumps
                        # the on-disk filename to force a clean regeneration
                        # instead of silently loading a stale/incompatible dict.
                        auxiliary_path = cache_dir / f"{caption_hash}_acestepauxv2.pt"
                    elif self.is_sdxl:
                        auxiliary_path = cache_dir / f"{caption_hash}_pooled.pt"
                    else:
                        auxiliary_path = None  # SD1.5 / anima have no persisted auxiliary data

                    # Check if all required files exist
                    if auxiliary_path is not None:
                        if not (embeds_path.exists() and auxiliary_path.exists()):
                            captions_to_encode.append((caption, lyrics))
                        else:
                            total_cached += 1
                    else:
                        if not embeds_path.exists():
                            captions_to_encode.append((caption, lyrics))
                        else:
                            total_cached += 1

                if len(captions_to_encode) == 0:
                    print(f"{self.log_prefix} Dataset '{dataset.unique_id}': All {len(pairs)} captions already cached")
                else:
                    print(f"{self.log_prefix} Dataset '{dataset.unique_id}': Encoding {len(captions_to_encode)}/{len(pairs)} captions...")

                    for idx, (caption, lyrics) in enumerate(tqdm(captions_to_encode, desc=f"Encoding captions [{dataset.unique_id}]")):
                        # Abort promptly on user stop (raises KeyboardInterrupt; the
                        # finally below moves the text encoder(s) back to CPU).
                        self._check_stop_requested()

                        # Encode caption (unified method)
                        embeddings, auxiliary_data = self.encode_caption(caption, requires_grad=False, lyrics=lyrics)
                        embeds_cpu = embeddings.cpu()
                        auxiliary_cpu = self._aux_to_cpu(auxiliary_data)

                        # Save immediately to disk to avoid memory accumulation
                        caption_hash = self._text_cache_key(caption, lyrics)
                        embeds_path = cache_dir / f"{caption_hash}_embeds.pt"

                        try:
                            # Save main embeddings
                            torch.save(embeds_cpu, embeds_path)

                            # Save auxiliary data (architecture-specific)
                            if (self.is_zimage or self.is_lens or self.is_ideogram4 or self.is_minit2i or self.is_krea2) and auxiliary_cpu is not None:
                                mask_path = cache_dir / f"{caption_hash}_mask.pt"
                                torch.save(auxiliary_cpu, mask_path)
                            elif self.is_ltx2 and auxiliary_cpu is not None:
                                ltx2aux_path = cache_dir / f"{caption_hash}_ltx2aux.pt"
                                torch.save(auxiliary_cpu, ltx2aux_path)
                            elif self.is_minimax_h3 and auxiliary_cpu is not None:
                                h3aux_path = cache_dir / f"{caption_hash}_h3aux.pt"
                                torch.save(auxiliary_cpu, h3aux_path)
                            elif self.is_acestep and auxiliary_cpu is not None:
                                acestepaux_path = cache_dir / f"{caption_hash}_acestepauxv2.pt"
                                torch.save(auxiliary_cpu, acestepaux_path)
                            elif self.is_sdxl and auxiliary_cpu is not None:
                                pooled_path = cache_dir / f"{caption_hash}_pooled.pt"
                                torch.save(auxiliary_cpu, pooled_path)
                            # SD1.5: no auxiliary data to save

                            total_encoded += 1

                            # Free memory immediately after saving
                            del embeds_cpu, embeddings
                            if auxiliary_cpu is not None:
                                del auxiliary_cpu, auxiliary_data
                        except Exception as e:
                            print(f"{self.log_prefix} WARNING: Failed to save cache for caption '{caption[:30]}...': {e}")

                        # Progress callback
                        if progress_callback:
                            progress_callback(
                                phase="text_encoder_cache",
                                step=total_cached + total_encoded,
                                total=total_captions,
                            )

        finally:
            # Move text encoder(s) back to CPU
            self.move_text_encoder_to_cpu()

        print(f"{self.log_prefix} Text encoder cache validation complete:")
        print(f"{self.log_prefix}   - Cached: {total_cached}")
        print(f"{self.log_prefix}   - Newly encoded: {total_encoded}")
        print(f"{self.log_prefix}   - Total: {total_cached + total_encoded}")

    def _load_caption_embedding_from_disk(
        self,
        caption: str,
        dataset_unique_id: str,
        text_encoder_caches: Dict[str, Path],
        lyrics: str = "",
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Load caption embedding from disk cache for all architectures.

        Args:
            caption: Caption text
            dataset_unique_id: Dataset unique ID
            text_encoder_caches: Dictionary mapping dataset_unique_id to cache directory
            lyrics: ACE-Step ONLY -- the item's per-item lyrics text ("" default;
                every other arch ignores this). Must match the value passed to
                encode_caption()/the cache-generation pass for the SAME item, or
                the lookup misses (see _text_cache_key).

        Returns:
            Tuple of (embeddings, auxiliary_data) if cached, None otherwise:
            - Z-Image: (prompt_embeds, attention_mask)
            - SD1.5: (text_embeddings, None)
            - SDXL: (text_embeddings, pooled_embeddings)
        """
        cache_dir = text_encoder_caches.get(dataset_unique_id)
        if cache_dir is None:
            return None

        caption_hash = self._text_cache_key(caption, lyrics)
        embeds_path = cache_dir / f"{caption_hash}_embeds.pt"

        # Check architecture-specific auxiliary file
        if self.is_zimage or self.is_lens or self.is_ideogram4 or self.is_minit2i or self.is_krea2:
            auxiliary_path = cache_dir / f"{caption_hash}_mask.pt"
        elif self.is_ltx2:
            auxiliary_path = cache_dir / f"{caption_hash}_ltx2aux.pt"
        elif self.is_minimax_h3:
            auxiliary_path = cache_dir / f"{caption_hash}_h3aux.pt"
        elif self.is_acestep:
            # "v2" suffix -- see _validate_and_generate_text_encoder_caches's
            # matching comment (aux schema gained lyric_* keys).
            auxiliary_path = cache_dir / f"{caption_hash}_acestepauxv2.pt"
        elif self.is_sdxl:
            auxiliary_path = cache_dir / f"{caption_hash}_pooled.pt"
        else:
            auxiliary_path = None  # SD1.5 / anima have no persisted auxiliary data

        # Check if required files exist
        if auxiliary_path is not None:
            if not (embeds_path.exists() and auxiliary_path.exists()):
                return None
        else:
            if not embeds_path.exists():
                return None

        # Load embeddings
        try:
            embeddings = torch.load(embeds_path, map_location='cpu')
            if auxiliary_path is not None:
                auxiliary_data = torch.load(auxiliary_path, map_location='cpu')
            elif self.is_anima:
                # Anima's auxiliary payload (source_mask / t5_input_ids /
                # t5_attn_mask) is NOT persisted alongside the cached Qwen3
                # hidden states: every element is a pure tokenizer product
                # (source_mask == qwen3 attention_mask; the T5 tensors are the
                # T5 tokenizer's input_ids / attention_mask) with a FIXED
                # max_length=512 padding, so it is deterministically and cheaply
                # reconstructable from the caption alone. Reconstructing here
                # (rather than persisting) keeps the on-disk cache format
                # unchanged, so pre-existing anima embeds-only cache entries
                # keep working with no regeneration. The reconstructed tensors
                # match encode_prompt_anima's per-sample shape (batch dim
                # dropped) so collate_aux batches them identically to the
                # swap_onthefly / onthefly_gpu live-encode paths.
                # IMPORTANT: the cached prompt_embeds were produced from the
                # emphasis-STRIPPED prompt (anima_pipeline_ops.encode_prompt runs
                # _build_emphasis before tokenizing), so the aux must be rebuilt
                # from the same clean prompt or source_mask/t5 ids diverge for
                # any caption containing ( ) [ ] emphasis syntax.
                from core.models.anima.anima_pipeline_ops import (
                    _build_emphasis,
                    tokenize_for_anima,
                )
                clean_prompt, _weights = _build_emphasis(caption or "", self.tokenizer, 512)
                toks = tokenize_for_anima(self.tokenizer, self.t5_tokenizer, clean_prompt)
                auxiliary_data = {
                    "source_mask": toks["qwen3_attn_mask"][0],
                    "t5_input_ids": toks["t5_input_ids"][0],
                    "t5_attn_mask": toks["t5_attn_mask"][0],
                }
            else:
                auxiliary_data = None
            return (embeddings, auxiliary_data)
        except Exception as e:
            print(f"{self.log_prefix} WARNING: Failed to load cached embedding for caption '{caption[:30]}...': {e}")
            return None

    # ============================================================
    # Training Loop Infrastructure
    # ============================================================

    def _maybe_compile_transformer(self):
        """Opt-in torch.compile for DiT training (config key ``torch_compile``).

        Compiles the DiT transformer's ``forward`` IN PLACE (replaces the
        instance ``forward`` attribute with the compiled callable) rather than
        wrapping the module in an ``OptimizedModule``. This is deliberate:

          * ``self.transformer`` stays the SAME nn.Module object, so
            ``state_dict()`` keys remain UNPREFIXED (no ``_orig_mod.``) and every
            checkpoint save path (e.g. AnimaFullParameterAdapter.save_checkpoint
            reads ``trainer.transformer.state_dict()``) is byte-for-byte
            unaffected.
          * Optimizer parameter references, block-swap attribute access, and the
            sampling helpers that call ``self.transformer(...)`` all keep working
            unchanged (nn.Module.__call__ dispatches to the compiled forward).

        Gating (each skip is logged, never raises):
          * DiT only — ``self.transformer`` must be set (U-Net archs skip).
          * Full-parameter FT only — LoRA skips (compile over freshly-inserted
            LoRA wrappers is recompile-heavy and not the measurement target).
          * Incompatible with block swap (``blocks_to_swap > 0``) — the
            LayerOffloadConductor's CPU<->GPU hooks and Dynamo conflict; skip.

        Any Inductor / compile failure at call time is caught and the module is
        left in eager mode (safe fallback). Must be called once, AFTER the model
        is on device/dtype, AFTER gradient-checkpointing + adapter/optimizer
        setup, and BEFORE the training loop.
        """
        mode = getattr(self, "torch_compile", "off") or "off"
        if mode == "off":
            return
        if self.transformer is None:
            print(f"{self.log_prefix} torch_compile={mode!r} requested but this "
                  f"architecture has no DiT transformer (U-Net archs are not "
                  f"supported yet); skipping compile.")
            return
        # Gate to full-parameter FT: adapter paths (LoRA/ReLoRA/ControlNet) freeze
        # the base, and compiling over freshly-inserted adapter wrappers is
        # recompile-heavy and off-target.
        from core.training.ops.training_method import is_full_finetune
        trainer_cls = type(self).__name__
        if not is_full_finetune(self):
            print(f"{self.log_prefix} torch_compile={mode!r} requested but trainer "
                  f"is {trainer_cls} (not full-parameter FT); compile is gated to "
                  f"full-parameter FT - skipping (adapter paths run eager).")
            return
        if getattr(self, "blocks_to_swap", 0) > 0:
            print(f"{self.log_prefix} torch_compile={mode!r} requested but block "
                  f"swap is active (blocks_to_swap={self.blocks_to_swap}); "
                  f"incompatible - skipping compile.")
            return
        if getattr(self, "_transformer_compiled", False):
            return

        dynamic = getattr(self, "torch_compile_dynamic", None)
        try:
            # Inductor/Triton compilation is LAZY: it runs on the first forward
            # per input shape, NOT at the torch.compile() call below. Two layers
            # of fallback so a compile failure never crashes training:
            #   1. Dynamo suppress_errors: falls back to eager for any graph it
            #      cannot trace/lower (covers most inductor-lowering failures and
            #      per-shape recompiles under bucketing).
            #   2. A guarded forward wrapper: some Triton codegen errors (e.g.
            #      "CantSplit" on an awkward bucket shape) are raised at kernel
            #      launch and are NOT intercepted by suppress_errors. The wrapper
            #      catches ANY exception from the compiled FORWARD path and
            #      permanently reverts self.transformer.forward to the original
            #      eager forward, then re-runs the forward eagerly (side-effect-
            #      free, so a re-run is safe).
            #      LIMITATION: only the forward is guarded. The AOTAutograd
            #      backward graph executes inside loss.backward(), outside this
            #      wrapper — a Triton kernel-launch failure there still crashes
            #      the run. The feature is opt-in (default "off") precisely for
            #      this reason.
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            orig_forward = self.transformer.forward
            compiled_forward = torch.compile(orig_forward, mode=mode, dynamic=dynamic)
            _fb = {"eager": False}

            def _guarded_compiled_forward(*args, **kwargs):
                if _fb["eager"]:
                    return orig_forward(*args, **kwargs)
                try:
                    return compiled_forward(*args, **kwargs)
                except Exception as ce:  # Triton/Inductor runtime codegen failure
                    print(f"{self.log_prefix} WARNING: compiled DiT forward failed "
                          f"({type(ce).__name__}: {str(ce)[:200]}); permanently "
                          f"falling back to EAGER for the rest of training.")
                    _fb["eager"] = True
                    self._transformer_compiled = False
                    # Bypass the wrapper on all subsequent calls.
                    self.transformer.forward = orig_forward
                    return orig_forward(*args, **kwargs)

            self.transformer.forward = _guarded_compiled_forward
            self._transformer_compiled = True
            print(f"{self.log_prefix} torch.compile ENABLED for DiT transformer "
                  f"(mode={mode!r}, dynamic={dynamic}). First training step will "
                  f"pay a one-time compilation cost; steady-state should speed up. "
                  f"(Guarded: any compile failure reverts to eager.)")
        except Exception as e:
            print(f"{self.log_prefix} WARNING: torch.compile(mode={mode!r}) failed "
                  f"to set up ({type(e).__name__}: {e}); continuing in eager mode.")
            self._transformer_compiled = False

    def train(
        self,
        datasets: List[Any],
        num_epochs: int = 10,
        total_steps: Optional[int] = None,  # If specified, overrides num_epochs
        batch_size: int = 1,
        save_every_n_steps: int = 500,
        sample_every_n_steps: int = 500,
        sample_prompts: Optional[List[Dict[str, str]]] = None,
        sample_guidance_scale: float = 3.5,
        sample_steps: int = 28,
        sample_width: int = 1024,
        sample_height: int = 1024,
        sample_seed: int = -1,
        sample_schedule_type: str = "uniform",
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "constant",
        enable_bucketing: bool = True,
        base_resolutions: Optional[List[int]] = None,
        bucket_strategy: str = "resize",
        multi_resolution_mode: str = "max",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        multi_noise_timesteps: int = 1,
        multi_noise_mode: str = "independent",  # Unused (MNT disabled), kept for compatibility
        trajectory_blend_alpha: float = 0.7,  # Unused (MNT disabled), kept for compatibility
        timestep_sampling_config: Optional[Dict[str, Any]] = None,
        debug_latents: bool = False,
        debug_latents_every: int = 50,
        progress_callback: Optional[Callable] = None,
        update_total_steps_callback: Optional[Callable[[int], None]] = None,
        run_id: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        force_recache: bool = False,
        max_step_saves_to_keep: int = 3,
        text_encoding_mode: str = "swap_onthefly",
        text_encoding_swap_interval: int = 256,
        text_encoding_prefetch_depth: int = 4,
        latent_encoding_mode: str = "swap_onthefly",
        latent_encoding_swap_interval: int = 256,
        use_reference_images: bool = False,
        train_vision_encoder: bool = False,
        vision_encoder_path: Optional[str] = None,
        vision_encoder_lr: Optional[float] = None,
        gradient_routing_ve: bool = False,
        param_tracking: bool = False,
        param_tracking_interval: int = 100,
        priority_training: Optional[Dict] = None,
    ):
        """
        Main training loop.

        Args:
            datasets: List of dataset objects
            num_epochs: Number of training epochs
            batch_size: Batch size per step
            save_every_n_steps: Save checkpoint every N steps
            sample_every_n_steps: Generate sample every N steps
            sample_prompts: List of sample prompt dicts [{positive, negative, condition_image_path?}, ...]
            optimizer_type: Optimizer type
            lr_scheduler_type: LR scheduler type
            enable_bucketing: Enable resolution bucketing
            base_resolutions: List of base resolutions (e.g., [512, 768, 1024])
            bucket_strategy: Bucketing strategy ("resize", "crop", "random_crop")
            multi_resolution_mode: Multi-resolution mode ("max", "random")
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Max gradient norm for clipping
            debug_latents: Enable debug latent saving
            debug_latents_every: Save debug latents every N steps
            progress_callback: Progress callback function
            text_encoding_mode: Text encoding mode for Z-Image
                - "swap_onthefly": Swap TE/Transformer, encode on-the-fly (recommended for large datasets)
                - "pre_encoded_cache": Use pre-encoded disk cache (NOT recommended for large datasets)
                - "onthefly_gpu": Encode on-the-fly on GPU without cache (NOT recommended for Z-Image)
            text_encoding_swap_interval: Swap interval for swap_onthefly mode (default: 256 steps)
            use_reference_images: Arm per-item reference conditioning. SD/SDXL
                also arm it implicitly when vision_encoder_path is selected.
        """
        # 0 means "never" for every optional periodic action; see
        # periodic_intervals. gradient_accumulation_steps is not optional, so 0
        # folds to 1 rather than disabling the optimizer step.
        save_every_n_steps = normalize_interval(save_every_n_steps)
        sample_every_n_steps = normalize_interval(sample_every_n_steps)
        debug_latents_every = normalize_interval(debug_latents_every)
        gradient_accumulation_steps = normalize_interval(gradient_accumulation_steps, minimum=1)

        if self.is_sensenova:
            from core.training.ops.training_method import is_full_finetune
            _sensenova_full_ft = is_full_finetune(self)
            if batch_size != 1:
                raise ValueError(
                    "SenseNova training requires batch_size=1"
                    + ("" if _sensenova_full_ft else
                       "; use gradient_accumulation_steps for a larger effective batch")
                )
            if self.blocks_to_swap != 0:
                raise ValueError("SenseNova training does not implement blocks_to_swap; set it to 0")
            if _sensenova_full_ft and int(gradient_accumulation_steps or 1) != 1:
                # The argument, not the config value assert_full_finetune_contract
                # read: train() is called with its own. Full fine-tuning here runs
                # under the fused backward pass, which applies and frees each
                # gradient as it appears, so nothing survives to be accumulated.
                raise ValueError(
                    f"SenseNova full fine-tuning requires gradient_accumulation_steps=1, "
                    f"got {gradient_accumulation_steps}. Its updates are applied per "
                    f"parameter during backward and each gradient is freed as it is "
                    f"applied, so every backward would become its own optimizer step "
                    f"rather than one step over the effective batch. LoRA training on "
                    f"this architecture does support gradient_accumulation_steps."
                )
            text_encoding_mode = "onthefly_gpu"
            latent_encoding_mode = "onthefly_gpu"

        # Store references for subclass access
        self._training_datasets = datasets
        self._sample_prompts = sample_prompts or [{"positive": "a beautiful landscape", "negative": ""}]

        print(f"{self.log_prefix} Starting training...")
        print(f"{self.log_prefix} Datasets: {len(datasets)}")
        print(f"{self.log_prefix} Epochs: {num_epochs}")
        print(f"{self.log_prefix} Batch size: {batch_size}")
        print(f"{self.log_prefix} Gradient accumulation: {gradient_accumulation_steps}")
        print(f"{self.log_prefix} Debug latents: {debug_latents} (every {debug_latents_every} steps)")
        if save_every_n_steps == 0:
            print(f"{self.log_prefix} Periodic checkpointing: DISABLED (save_every=0); "
                  f"only interrupt/emergency saves will write a checkpoint")

        # Compute dataset fingerprint for change detection on resume
        # This is stored in training state and compared when resuming
        # IMPORTANT: Only image paths are included - caption changes do NOT invalidate shuffle state
        self._dataset_fingerprint = self._compute_dataset_fingerprint(datasets)
        print(f"{self.log_prefix} Dataset fingerprint: {self._dataset_fingerprint['total_item_count']} items, hash={self._dataset_fingerprint['image_paths_hash'][:8]}...")

        _arch_name = getattr(getattr(self, "arch", None), "name", "")
        _sd_ve_arch = _arch_name in ("sd15", "sdxl")
        if vision_encoder_path and not _sd_ve_arch:
            raise ValueError(
                "vision_encoder_path is supported only for SD1.5/SDXL training; "
                f"selected architecture is {_arch_name or 'unknown'}"
            )
        if train_vision_encoder and not vision_encoder_path:
            raise ValueError("train_vision_encoder=True requires vision_encoder_path")
        if _sd_ve_arch and vision_encoder_path and not use_reference_images:
            print(
                f"{self.log_prefix} Reference images: enabling use_reference_images "
                "because a SigLIP2 vision encoder is selected"
            )
            use_reference_images = True

        if use_reference_images:
            print(f"{self.log_prefix} Reference images: ENABLED (conditioning will be applied)")
            if _sd_ve_arch and not vision_encoder_path:
                raise ValueError(
                    "SD1.5/SDXL use_reference_images=True requires "
                    "vision_encoder_path"
                )
            elif not (self.is_flux2 or self.is_sensenova or _sd_ve_arch):
                print(
                    f"{self.log_prefix} WARNING: use_reference_images is supported "
                    "only for FLUX.2, SenseNova, and SD1.5/SDXL with a SigLIP2 "
                    "vision encoder; it will be ignored"
                )

        # Load Vision Encoder if specified (SigLIP2 for SDXL/SD1.5)
        if vision_encoder_path:
            print(f"{self.log_prefix} Vision Encoder: Loading from {vision_encoder_path}")
            try:
                from core.vision_encoder import SigLIP2VisionEncoderWrapper
                self.vision_encoder = SigLIP2VisionEncoderWrapper(vision_encoder_path, device="cpu")
                self._train_vision_encoder = train_vision_encoder
                self._gradient_routing_ve = gradient_routing_ve
                self._vision_encoder_lr = vision_encoder_lr
                if train_vision_encoder:
                    # Move to GPU immediately and keep it there for the duration of training.
                    # Per-batch CPU offloading is skipped when training VE (92.9M params ≈ 186MB
                    # is negligible vs UNet, and PCIe round-trips per batch hurt throughput).
                    self.vision_encoder.to(self.device)
                    # Gradient checkpointing on the trained VE: the reference images are
                    # encoded one forward each (their graphs all held until the batch
                    # backward), so activation memory adds up. GC trades a little recompute
                    # for much lower activation VRAM with identical gradients (settings-neutral).
                    # use_reentrant=False: the pixel inputs don't require grad.
                    try:
                        _ve_model = getattr(self.vision_encoder, "model", None)
                        if _ve_model is not None and self.gradient_checkpointing and hasattr(_ve_model, "gradient_checkpointing_enable"):
                            try:
                                _ve_model.gradient_checkpointing_enable(
                                    gradient_checkpointing_kwargs={"use_reentrant": False}
                                )
                            except TypeError:
                                # Older transformers without the kwargs argument.
                                _ve_model.gradient_checkpointing_enable()
                            print(f"{self.log_prefix} Gradient checkpointing enabled for Vision Encoder")
                    except Exception as _ve_gc_err:
                        print(f"{self.log_prefix} WARNING: could not enable VE gradient checkpointing: {_ve_gc_err}")
                    print(f"{self.log_prefix} Vision Encoder: Will be trained (lr={vision_encoder_lr or 'inherit'}), kept on GPU")
                else:
                    print(f"{self.log_prefix} Vision Encoder: Frozen (inference only, CPU offloaded between batches)")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to load Vision Encoder: {e}")
                self.vision_encoder = None
                self._train_vision_encoder = False
                self._vision_encoder_lr = None
        else:
            if not hasattr(self, 'vision_encoder'):
                self.vision_encoder = None
            self._train_vision_encoder = False
            self._vision_encoder_lr = None

        # Validate text_encoding_mode when Text Encoder is trainable
        # Check if any Text Encoder has trainable parameters (works for both LoRA and full fine-tune)
        text_encoder_trainable = False
        te1_trainable_tensors = 0
        te1_trainable_scalars = 0
        te2_trainable_tensors = 0
        te2_trainable_scalars = 0

        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            te1_trainable_tensors = sum(1 for p in self.text_encoder.parameters() if p.requires_grad)
            te1_trainable_scalars = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
            text_encoder_trainable = te1_trainable_tensors > 0

        if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
            te2_trainable_tensors = sum(1 for p in self.text_encoder_2.parameters() if p.requires_grad)
            te2_trainable_scalars = sum(p.numel() for p in self.text_encoder_2.parameters() if p.requires_grad)
            text_encoder_trainable = text_encoder_trainable or (te2_trainable_tensors > 0)

        # Custom SDXL TE: CLIP is unused; the trainable bridge is the TE adapters (and
        # optionally the encoder body). Treat encoding as trainable so embeddings are
        # recomputed each step (with grad through the adapters) instead of cached.
        if getattr(self, "sdxl_te_type", "none") not in ("none", "clip", "", None) \
                and getattr(self, "te_adapters", None) is not None:
            if any(p.requires_grad for p in self.te_adapters.parameters()) \
                    or bool(getattr(self, "sdxl_te_train_encoder", False)):
                text_encoder_trainable = True
                print(f"{self.log_prefix}   Custom TE adapters trainable -> recompute embeddings each step")

        # Log trainable parameter counts (U-Net + Text Encoders)
        unet_obj = getattr(self, 'unet', None) or getattr(self, 'transformer', None)
        if unet_obj is not None:
            unet_trainable_tensors = sum(1 for p in unet_obj.parameters() if p.requires_grad)
            unet_trainable_scalars = sum(p.numel() for p in unet_obj.parameters() if p.requires_grad)
            print(f"{self.log_prefix} Trainable parameters:")
            print(f"{self.log_prefix}   U-Net/Transformer: tensors={unet_trainable_tensors}, params={format_param_count(unet_trainable_scalars)}")
        else:
            print(f"{self.log_prefix} Trainable parameters:")
        if text_encoder_trainable:
            if te1_trainable_tensors > 0:
                print(f"{self.log_prefix}   Text Encoder 1:    tensors={te1_trainable_tensors}, params={format_param_count(te1_trainable_scalars)}")
            if te2_trainable_tensors > 0:
                print(f"{self.log_prefix}   Text Encoder 2:    tensors={te2_trainable_tensors}, params={format_param_count(te2_trainable_scalars)}")
        if getattr(self, '_train_vision_encoder', False) and getattr(self, 'vision_encoder', None) is not None:
            ve_trainable_tensors = sum(1 for p in self.vision_encoder.parameters() if p.requires_grad)
            ve_trainable_scalars = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
            print(f"{self.log_prefix}   Vision Encoder:    tensors={ve_trainable_tensors}, params={format_param_count(ve_trainable_scalars)}")

        # A weight-only quantized base costs MORE memory than the bf16 base it
        # replaced when gradient checkpointing is off, because every quantized
        # Linear's dequantized weight is retained by autograd until backward and
        # none of them are freed in between. Reported once here, where both facts
        # are known; see adapters/base_adapter.py and
        # core/training/INT8_W8A8_TRAINING_GATE.md (G4).
        try:
            from core.training.adapters.base_adapter import (
                warn_quantized_base_without_checkpointing,
            )

            warn_quantized_base_without_checkpointing(
                unet_obj,
                gradient_checkpointing=self.gradient_checkpointing,
                log_prefix=self.log_prefix,
            )
        except Exception as _qb_warn_err:
            print(f"{self.log_prefix} WARNING: quantized-base memory check skipped: {_qb_warn_err}")

        # If Text Encoder is trainable, embeddings must be recomputed each step
        if text_encoder_trainable and text_encoding_mode in ['swap_onthefly', 'pre_encoded_cache', 'cpu_prefetch']:
            print(f"{self.log_prefix} WARNING: Text Encoder is trainable but text_encoding_mode='{text_encoding_mode}'")
            print(f"{self.log_prefix} Text embeddings would be cached and NOT updated during training!")
            print(f"{self.log_prefix} Overriding to 'onthefly_gpu' - embeddings must be recomputed each step")
            text_encoding_mode = 'onthefly_gpu'

        # cpu_prefetch mode pins the (frozen) TE to CPU and lets a worker
        # thread encode upcoming batches in parallel with GPU train steps.
        # Reject the mode when the architecture's TE can't safely run on
        # CPU (currently only FP8-quantised TEs; standard transformers can).
        if text_encoding_mode == 'cpu_prefetch':
            try:
                if hasattr(self, '_has_fp8_text_encoder') and self._has_fp8_text_encoder():
                    print(f"{self.log_prefix} WARNING: cpu_prefetch is incompatible with "
                          f"FP8-quantised text encoders (CPU lacks _scaled_mm support). "
                          f"Falling back to swap_onthefly.")
                    text_encoding_mode = 'swap_onthefly'
            except Exception:
                pass

        # Log final text encoding mode
        print(f"{self.log_prefix} Text encoding mode: {text_encoding_mode}")
        if text_encoding_mode == 'cpu_prefetch':
            print(f"{self.log_prefix}   prefetch_depth: {text_encoding_prefetch_depth}")
        # Record the mode so encode_prompt can skip the "TE on CPU" warning for
        # cpu_prefetch, where running the TE on CPU is intentional.
        self._text_encoding_mode = text_encoding_mode

        # Setup debug directory
        debug_dir = None
        if debug_latents:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            print(f"{self.log_prefix} Debug latents will be saved to: {debug_dir}")

        # Setup bucketing
        if enable_bucketing:
            from core.training.bucketing import BucketManager

            # Default to [1024] if not specified
            if base_resolutions is None:
                base_resolutions = [1024]

            # Sampler tidiness only for SenseNova: at its forced batch_size=1 the
            # prefix is per-item anyway, so this never underwrites prefix shape.
            separate_by_reference = use_reference_images and (self.is_flux2 or self.is_sensenova)

            # Align to the ARCH's pixel requirement, not a hardcoded /8 -- the
            # same read the two NO-bucketing fit paths already do. See
            # `_arch_pixel_align` for the measured reason.
            _bucket_align = self._arch_pixel_align()
            bucket_manager = BucketManager(
                base_resolutions=base_resolutions,
                divisibility=_bucket_align,
                strategy=bucket_strategy,
                multi_resolution_mode=multi_resolution_mode,
                separate_by_reference=separate_by_reference
            )
            print(f"{self.log_prefix} Bucketing enabled: base_resolutions={base_resolutions}, strategy={bucket_strategy}, mode={multi_resolution_mode}, divisibility={_bucket_align}")
            if separate_by_reference:
                print(f"{self.log_prefix} Reference separation enabled: batches will be separated by reference image availability")
        else:
            bucket_manager = None
            print(f"{self.log_prefix} Bucketing disabled")

        # VIDEO items: route through VideoBucketManager to attach
        # clip_length/stride/bucket dims/fps BEFORE the image bucketing loop below
        # (which skips item_type=="video"). Runs regardless of enable_bucketing so
        # video items always gain the keys _encode_video_clip reads. No-op for a
        # non-video arch (no TemporalSpec) and for datasets without video items.
        if self._temporal_spec() is not None:
            self._annotate_video_items(datasets, base_resolutions)

        # Epoch-dynamic crop planner (SDXL only). Re-buckets each item per epoch from a
        # constrained random crop (scale/crop extrapolation). Requires bucketing + SDXL.
        # When disabled (default), the code path below is unchanged.
        self.crop_planner = None
        if bool(self.config.get("crop_augment_enable", False)):
            if not self.is_sdxl:
                print(f"{self.log_prefix} crop_augment_enable ignored: only supported for SDXL")
            elif bucket_manager is None:
                print(f"{self.log_prefix} crop_augment_enable ignored: requires enable_bucketing")
            else:
                from core.training.crop_planner import CropPlanner
                _cfg = dict(self.config)
                if int(_cfg.get("crop_plan_seed", 0) or 0) == 0:
                    _cfg["crop_plan_seed"] = int(self.config.get("seed", 0) or 0)
                self.crop_planner = CropPlanner(
                    config=_cfg,
                    base_resolutions=bucket_manager.base_resolutions,
                    multi_resolution_mode=bucket_manager.multi_resolution_mode,
                    divisibility=8,
                )
                print(f"{self.log_prefix} Epoch-dynamic crop augmentation ENABLED "
                      f"(full_prob={self.crop_planner.full_image_prob}, "
                      f"max_bucket_prob={self.crop_planner.max_bucket_prob}, "
                      f"min_area={self.crop_planner.min_area_ratio}, "
                      f"min_short={self.crop_planner.min_short_side_px}, "
                      f"aspect={self.crop_planner.aspect_mode}, "
                      f"smaller_mode={self.crop_planner.smaller_bucket_mode}, "
                      f"seed={self.crop_planner.seed})")

        # Validate MNT parameters
        if multi_noise_timesteps < 1:
            raise ValueError(f"multi_noise_timesteps must be >= 1, got {multi_noise_timesteps}")

        # Setup timestep sampler
        from .timestep_sampler import TimestepSampler

        if timestep_sampling_config is None:
            # No explicit config: resolve the per-architecture default (SSOT in
            # param_defaults). Only MiniT2I differs from uniform; all others use
            # uniform [0,1]. This keeps non-UI/API callers consistent with the UI.
            from api.param_defaults import TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH
            arch = (
                "minit2i" if self.is_minit2i else
                "zimage" if self.is_zimage else
                "flux2" if self.is_flux2 else
                "anima" if self.is_anima else
                "ltx2" if self.is_ltx2 else
                "minimax_h3" if self.is_minimax_h3 else
                "sensenova" if self.is_sensenova else
                "lens" if self.is_lens else
                "ideogram4" if self.is_ideogram4 else
                "krea2" if self.is_krea2 else
                "_default"
            )
            timestep_sampling_config = dict(
                TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH.get(
                    arch, TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["_default"]
                )
            )
            print(f"{self.log_prefix} No timestep_sampling supplied; using per-arch "
                  f"default for '{arch}': {timestep_sampling_config}")

        timestep_sampler = TimestepSampler.from_config(timestep_sampling_config)
        print(f"{self.log_prefix} Timestep sampler: {timestep_sampler.__class__.__name__}")
        print(f"{self.log_prefix} Timestep range: [{timestep_sampler.min_timestep:.3f}, {timestep_sampler.max_timestep:.3f}]")
        # Log distribution-specific parameters
        if hasattr(timestep_sampler, 'mean') and hasattr(timestep_sampler, 'std'):
            print(f"{self.log_prefix} Timestep params: mean={timestep_sampler.mean:.2f}, std={timestep_sampler.std:.2f}")
        elif hasattr(timestep_sampler, 'alpha') and hasattr(timestep_sampler, 'beta'):
            print(f"{self.log_prefix} Timestep params: alpha={timestep_sampler.alpha:.2f}, beta={timestep_sampler.beta:.2f}")
        print(f"{self.log_prefix} Multi Noise-Timesteps (MNT): {multi_noise_timesteps}")

        # Cache alphas_cumprod on GPU to avoid repeated .to(device) calls in compute_snr()
        # This is called thousands of times during training, so caching saves significant overhead
        # Note: Flow Matching schedulers (FLUX.2) don't have alphas_cumprod
        if hasattr(self.noise_scheduler, 'alphas_cumprod'):
            alphas_cumprod_cached = self.noise_scheduler.alphas_cumprod.to(device=self.device)
            print(f"{self.log_prefix} Cached alphas_cumprod on GPU ({alphas_cumprod_cached.shape[0]} steps)")
        else:
            # FLUX.2 uses Flow Matching (no alphas_cumprod, SNR weighting not applicable)
            alphas_cumprod_cached = None
            print(f"{self.log_prefix} Flow Matching scheduler detected (no alphas_cumprod)")

        if multi_noise_timesteps > 1:
            print(f"{self.log_prefix} MNT enabled: Each batch will be processed {multi_noise_timesteps} times with different timesteps")

        # Gradient accumulation: the optimizer steps every `gradient_accumulation_steps`
        # backward passes (global_step counts MNT iterations, so the accumulation
        # window is measured in MNT-iterations — see should_step_optimizer below).
        # Each backward divides its loss by this so the accumulated gradient is the
        # window average (used in _execute_forward_backward).
        self._grad_accum_steps = gradient_accumulation_steps
        print(f"{self.log_prefix} Gradient accumulation steps: {gradient_accumulation_steps} "
              f"(optimizer steps every {gradient_accumulation_steps} backward pass(es); "
              f"with MNT={multi_noise_timesteps}, that is "
              f"{gradient_accumulation_steps / multi_noise_timesteps:g} batch(es) per step)")

        # Calculate total steps and epochs
        total_items = sum(len(dataset.items) for dataset in datasets)
        batches_per_epoch = (total_items + batch_size - 1) // batch_size
        steps_per_epoch = batches_per_epoch * multi_noise_timesteps  # MNT multiplier
        # Persist for save_training_state() so a batch_size-only change (which the dataset
        # fingerprint does NOT capture) is detectable by the resume structure-change guard.
        self._batches_per_epoch = batches_per_epoch

        # If total_steps is specified, calculate num_epochs; otherwise use num_epochs
        if total_steps is not None:
            # Step-based training: calculate epochs needed
            num_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch
            actual_total_steps = total_steps
            print(f"{self.log_prefix} Training mode: Step-based ({total_steps} steps)")
            print(f"{self.log_prefix} Calculated epochs needed: {num_epochs}")
        else:
            # Epoch-based training: calculate total steps
            actual_total_steps = steps_per_epoch * num_epochs
            print(f"{self.log_prefix} Training mode: Epoch-based ({num_epochs} epochs)")

        print(f"{self.log_prefix} Total items: {total_items}")
        print(f"{self.log_prefix} Batches per epoch: {batches_per_epoch}")
        print(f"{self.log_prefix} Steps per epoch (with MNT): {steps_per_epoch}")
        print(f"{self.log_prefix} Total training steps: {actual_total_steps}")

        # ---- Resolution curriculum (opt-in): warm up at a lower resolution, then switch
        # to the target resolution at an epoch boundary. Arch-agnostic (pure data-pipeline
        # feature): fewer latent tokens during warmup -> much cheaper attention/step.
        #
        # Switch semantics: warmup covers whole epochs [0, switch_epoch) where
        #   switch_epoch = ceil(warmup_steps / steps_per_epoch), clamped to num_epochs.
        # So the requested warmup_steps ROUNDS UP to the end of the epoch that contains it
        # (batches are planned per epoch; a mid-epoch resolution swap would strand the
        # remaining batches at the wrong dims). This keeps per-epoch batch planning intact.
        #
        # Accounting: the aspect-ratio bucket PARTITION is scale-invariant (scaling a base
        # resolution changes each bucket's pixel dims but not which items share a bucket),
        # so steps_per_epoch is identical in both phases and total_steps / per-epoch offsets
        # are unaffected. (Step-based runs skip the per-epoch recalc entirely.)
        _rc_normal_res = sorted(base_resolutions) if base_resolutions else [1024]
        self._rc_active = False
        self._rc_normal_res = _rc_normal_res
        self._rc_warmup_res = None
        self._rc_switch_epoch = 0
        self._rc_current_phase = None  # "warmup" | "normal"
        _rc_enable = bool(self.config.get("res_curriculum_enable", False))
        _rc_warmup_steps = int(self.config.get("res_curriculum_warmup_steps", 0) or 0)
        _rc_scale = float(self.config.get("res_curriculum_warmup_scale", 0.5) or 0.5)
        if _rc_enable and self.crop_planner is not None:
            # Crop augmentation owns per-epoch re-bucketing (SDXL); the curriculum's
            # phase re-bucket would fight it. Disable explicitly instead of logging
            # ENABLED and silently being overridden.
            print(f"{self.log_prefix} [ResCurriculum] DISABLED: crop augmentation is "
                  f"active and owns per-epoch bucketing - the two features do not combine.")
            _rc_enable = False
        if _rc_enable and _rc_warmup_steps > 0 and 0.0 < _rc_scale < 1.0:
            _rc_warmup_res = self._rc_scaled_resolutions(_rc_normal_res, _rc_scale)
            _rc_switch_epoch = min(
                num_epochs,
                (_rc_warmup_steps + steps_per_epoch - 1) // steps_per_epoch,
            )
            if _rc_warmup_res == _rc_normal_res:
                print(f"{self.log_prefix} [ResCurriculum] scale={_rc_scale} leaves the "
                      f"warmup resolution equal to the target ({_rc_normal_res}) after /64 "
                      f"snapping - curriculum has no effect, running normally.")
            elif _rc_switch_epoch <= 0:
                print(f"{self.log_prefix} [ResCurriculum] warmup_steps={_rc_warmup_steps} "
                      f"< one epoch worth of steps; no warmup epochs, running normally.")
            else:
                self._rc_active = True
                self._rc_warmup_res = _rc_warmup_res
                self._rc_switch_epoch = _rc_switch_epoch
                _rc_effective_warmup_steps = _rc_switch_epoch * steps_per_epoch
                print(f"{self.log_prefix} [ResCurriculum] ENABLED: warmup {_rc_warmup_res} "
                      f"(target {_rc_normal_res}), scale={_rc_scale}. Switch at epoch "
                      f"{_rc_switch_epoch + 1} (warmup_steps={_rc_warmup_steps} rounds up to "
                      f"{_rc_effective_warmup_steps} steps = epoch end).")
                if _rc_switch_epoch >= num_epochs:
                    print(f"{self.log_prefix} [ResCurriculum] WARNING: switch epoch "
                          f"({_rc_switch_epoch}) >= num_epochs ({num_epochs}) - the entire "
                          f"run stays in warmup (never reaches the target resolution).")
                if str(self.config.get("torch_compile", "off")).lower() not in ("off", "", "none"):
                    print(f"{self.log_prefix} [ResCurriculum] WARNING: torch_compile is on - "
                          f"the resolution switch changes token shapes and forces a one-time "
                          f"recompile at the switch epoch.")
                if latent_encoding_mode == "pre_encoded_cache":
                    print(f"{self.log_prefix} [ResCurriculum] NOTE: pre_encoded_cache mode "
                          f"caches BOTH warmup and target latents (keyed per-file by w_h, so "
                          f"no cache poisoning) - extra disk for the warmup entries.")

        # Crop augmentation: batch count varies per epoch (per-epoch re-bucketing), so
        # compute exact per-epoch step offsets up front for accurate total_steps and
        # resume epoch lookup. Header sizes are read once (cached for the encode pass).
        self._crop_step_offsets = None
        self._crop_plan_fingerprint = None
        if self.crop_planner is not None:
            print(f"{self.log_prefix} [crop] Precomputing step plan: reading {total_items} "
                  f"image sizes + sampling per-epoch bucket distribution...")
            if progress_callback:
                progress_callback(phase="crop_precompute", step=0, total=total_items)
            _plan_items = []
            _cp_done = 0
            for dataset in datasets:
                for item in dataset.items:
                    # At this point item['width']/['height'] are still the DB original
                    # dims (the bucket-dim overwrite happens later) -> seed the size map
                    # for free, avoiding image-header reads. Items without DB dims fall
                    # back to a lazy header read inside _get_original_size_for_item.
                    self._seed_orig_size_from_db(item)
                    try:
                        ow, oh = self._get_original_size_for_item(item)
                    except Exception:
                        ow, oh = item.get("width", 1024), item.get("height", 1024)
                    _plan_items.append((item["image_path"], ow, oh))
                    _cp_done += 1
                    # Throttle UI updates (every ~2000 items) to avoid flooding.
                    if progress_callback and _cp_done % 2000 == 0:
                        progress_callback(phase="crop_precompute", step=_cp_done, total=total_items)
            if progress_callback:
                progress_callback(phase="crop_precompute", step=total_items, total=total_items)
            self.crop_planner.precompute(_plan_items, num_epochs, batch_size)
            self._crop_step_offsets = self.crop_planner.step_offsets(multi_noise_timesteps)
            self._crop_plan_fingerprint = self.crop_planner.fingerprint(
                dataset_fingerprint=getattr(self, "_dataset_fingerprint", None),
                num_epochs=num_epochs,
            )
            _crop_total = self._crop_step_offsets[-1]
            print(f"{self.log_prefix} [crop] Per-epoch step accounting (sampled): "
                  f"total_steps {actual_total_steps} -> {_crop_total} "
                  f"(batches/epoch: {[self.crop_planner.batches_per_epoch(e) for e in range(min(num_epochs, 8))]}"
                  f"{'...' if num_epochs > 8 else ''})")
            actual_total_steps = _crop_total

        # Update DB with calculated total_steps (for resume correctness)
        if update_total_steps_callback is not None:
            update_total_steps_callback(actual_total_steps)

        # Setup optimizer
        self.setup_optimizer(
            optimizer_type=optimizer_type,
            lr_scheduler_type=lr_scheduler_type,
            total_steps=actual_total_steps,
        )

        # Whether a fused path is active is only known once the optimizer exists;
        # say what it does to the accumulation window here, before any batch runs,
        # rather than at the first step of a run that may go for hours.
        self._warn_gradient_accumulation_ignored_under_fused(
            gradient_accumulation_steps, batch_size, multi_noise_timesteps
        )

        # Resolution curriculum phase-0 setup: seed the original-size map (so a later
        # warmup->target rebucket can grow dims back), and point the initial bucketing at
        # the WARMUP grid so the whole up-front assignment/cache pass runs at low res.
        if self._rc_active:
            for dataset in datasets:
                for item in dataset.items:
                    self._seed_orig_size_from_db(item)
            self._rc_current_phase = "warmup"
            if bucket_manager:
                self._rc_apply_bucketing_grid(bucket_manager, self._rc_warmup_res)
            print(f"{self.log_prefix} [ResCurriculum] Phase 0 = WARMUP at {self._rc_warmup_res}")

        # Apply bucketing to datasets
        if bucket_manager:
            # Bucket assignment is O(N) over every item; for large datasets (millions)
            # this takes a while with no output, so report progress to console (tqdm)
            # and the UI (progress_callback).
            total_bucket_items = sum(len(dataset.items) for dataset in datasets)
            print(f"{self.log_prefix} Assigning {total_bucket_items} images to buckets...")
            if progress_callback:
                progress_callback(phase="bucketing", step=0, total=total_bucket_items)
            _bucket_pbar = tqdm(total=total_bucket_items, desc="Bucketing", unit="img")
            _bucket_done = 0
            for dataset in datasets:
                for item in dataset.items:
                    # Video items already bucketed by VideoBucketManager
                    # (÷32 spatial + clip_length); never run them through the image
                    # BucketManager (would overwrite bucket dims / drop clip fields).
                    # ACE-Step audio items have no spatial dims at all (no
                    # image_path-shaped width/height concept) — also skipped;
                    # batched separately below (acestep_audio_batches).
                    if (self._temporal_spec() is not None and item.get("item_type") == "video") or \
                       (self.is_acestep and item.get("item_type") == "audio"):
                        continue
                    # For ve_reconstruction_mode items: inject reference_images BEFORE bucketing
                    # so bucket_manager records has_reference=True and includes reference_images
                    # in image_info. This must happen here, not in the epoch loop, because
                    # bucket_manager creates new image_info dicts that would lose the flag.
                    if item.get("_ve_reconstruction_mode") and not item.get("reference_images"):
                        item["reference_images"] = [item["image_path"]]

                    width = item.get("width", 1024)
                    height = item.get("height", 1024)
                    # Check if item has reference images
                    reference_images = item.get("reference_images", [])
                    has_reference = len(reference_images) > 0

                    bucket_key, image_info = bucket_manager.assign_image_to_bucket(
                        image_path=item["image_path"],
                        width=width,
                        height=height,
                        caption=item.get("caption", ""),
                        dataset_unique_id=dataset.unique_id,
                        has_reference=has_reference,
                        reference_images=reference_images if reference_images else None,
                    )
                    # Propagate _ve_reconstruction_mode into image_info so training step
                    # can zero text embeddings for these items.
                    if item.get("_ve_reconstruction_mode"):
                        image_info["_ve_reconstruction_mode"] = True
                    # Update item with bucket dimensions
                    item["width"] = image_info["bucket_width"]
                    item["height"] = image_info["bucket_height"]

                    _bucket_done += 1
                    _bucket_pbar.update(1)
                    # Throttle UI updates (every ~2000 items) to avoid flooding.
                    if progress_callback and _bucket_done % 2000 == 0:
                        progress_callback(phase="bucketing", step=_bucket_done, total=total_bucket_items)
                        # This loop runs before train()'s own KeyboardInterrupt
                        # handler is in scope (see _check_stop_requested's
                        # docstring), so a stop here propagates all the way up
                        # to train_runner.py's main() except KeyboardInterrupt.
                        self._check_stop_requested()
            _bucket_pbar.close()
            if progress_callback:
                progress_callback(phase="bucketing", step=total_bucket_items, total=total_bucket_items)

            # Print bucket statistics
            bucket_counts = bucket_manager.get_bucket_counts()
            print(f"{self.log_prefix} Bucket distribution:")
            for bucket_size, count in sorted(bucket_counts.items()):
                print(f"  {bucket_size}: {count} images")

            # Print reference image statistics if separation is enabled
            if bucket_manager.separate_by_reference:
                ref_stats = bucket_manager.get_reference_statistics()
                print(f"{self.log_prefix} Reference image distribution:")
                print(f"  With reference: {ref_stats['with_reference']} images")
                print(f"  Without reference: {ref_stats['without_reference']} images")
        else:
            # No-bucketing path: item width/height come straight from the dataset DB,
            # i.e. the ORIGINAL image dimensions — base_resolutions previously fed only
            # the BucketManager, so without bucketing it was silently ignored and every
            # VAE encode (swap prefill / disk cache / on-the-fly) plus every training
            # step ran at the original resolution. Live-measured on a dataset with
            # 3.76MP-avg / 37MP-max images: a single original-resolution VAE encode
            # transiently allocated >20GB (torch peak) and pinned ~46.6GB at step 0,
            # independent of architecture, batch size, and the requested
            # base_resolutions. Fit oversized items into the base-resolution AREA
            # (aspect-preserving, /8-aligned) so base_resolutions bounds memory here
            # exactly as it does in the bucketed path. Items already within the area
            # are left untouched, so pre-resized datasets keep identical behavior.
            # Resolution curriculum: phase-0 (warmup) fits into the scaled base area.
            _nb_res = self._rc_warmup_res if (self._rc_active and self._rc_current_phase == "warmup") \
                else (base_resolutions or [1024])
            # Align to the ARCH's pixel requirement, not just the VAE /8: patchified
            # DiTs (anima/lens/krea2/flux2/zimage/minit2i/ideogram4) require /16 and
            # assert on non-/16 dims (see ArchHandler.pixel_align). SD/SDXL = 8.
            _nb_align = self._arch_pixel_align()
            _nb_base = max(int(r) for r in _nb_res)
            _nb_max_area = _nb_base * _nb_base
            _nb_clamped = 0
            for dataset in datasets:
                for item in dataset.items:
                    # Video items keep their VideoBucketManager ÷32 dims
                    # (do not re-fit into the still base-area path). ACE-Step
                    # audio items have no width/height concept — also skipped.
                    if (self._temporal_spec() is not None and item.get("item_type") == "video") or \
                       (self.is_acestep and item.get("item_type") == "audio"):
                        continue
                    w = int(item.get("width") or 0)
                    h = int(item.get("height") or 0)
                    if w <= 0 or h <= 0:
                        item["width"], item["height"] = _nb_base, _nb_base
                        continue
                    if w * h > _nb_max_area:
                        _scale = math.sqrt(_nb_max_area / float(w * h))
                        item["width"] = max(_nb_align, int(w * _scale) // _nb_align * _nb_align)
                        item["height"] = max(_nb_align, int(h * _scale) // _nb_align * _nb_align)
                        _nb_clamped += 1
                    else:
                        # Within-area items snap to the arch alignment (no-op for
                        # already-aligned / pre-resized datasets; prevents a non-/16
                        # original from tripping the DiT patchify assert).
                        item["width"] = max(_nb_align, w // _nb_align * _nb_align)
                        item["height"] = max(_nb_align, h // _nb_align * _nb_align)
            if _nb_clamped:
                print(f"{self.log_prefix} Bucketing disabled: fitted {_nb_clamped} item(s) "
                      f"exceeding the base-resolution area into {_nb_base}x{_nb_base} "
                      f"(aspect-preserving, /{_nb_align}-aligned) to bound VAE-encode/training memory")

        # MiniT2I is pixel-space (no VAE): the "latent" is just the resized [-1,1]
        # RGB image, so a disk latent cache would store full-resolution RGB tensors
        # (~48x a VAE latent) while saving only a trivial resize/normalise. Force
        # on-the-fly GPU encoding to skip the upfront caching pass and disk usage.
        if self.is_minit2i and not getattr(self, "minit2i_latent", False) \
                and latent_encoding_mode != "onthefly_gpu":
            print(f"{self.log_prefix} MiniT2I (pixel-space) detected: forcing "
                  f"latent_encoding_mode='onthefly_gpu' (was '{latent_encoding_mode}') - "
                  f"no VAE, so disk latent caching is wasteful")
            latent_encoding_mode = "onthefly_gpu"
        # Latent-space MiniT2I uses small VAE latents -> the normal disk cache is fine
        # (no override; behaves like SD/Z-Image).

        # Crop augmentation requires per-epoch re-encoding with a per-(item,epoch) crop,
        # which the disk/swap latent caches (keyed by image_path + bucket size) cannot
        # represent. Force on-the-fly GPU encoding.
        if self.crop_planner is not None and latent_encoding_mode != "onthefly_gpu":
            print(f"{self.log_prefix} Crop augmentation: forcing latent_encoding_mode="
                  f"'onthefly_gpu' (was '{latent_encoding_mode}') - disk/swap latent caches "
                  f"cannot represent per-epoch crops")
            latent_encoding_mode = "onthefly_gpu"

        # ACE-Step audio items: the only latent-cache pass that knows how to
        # encode audio clips is _validate_and_generate_latent_caches's
        # item_type=="audio" branch (encode_and_cache_audio), which only runs
        # under pre_encoded_cache mode (see the mode-dependent setup just
        # below). swap_onthefly's in-memory buffer and onthefly_gpu's
        # per-batch encode were built only for still-image / LTX-2.3-video
        # items and have no audio-clip branch -- selecting them for an audio
        # dataset would crash deep inside the training loop instead of at
        # setup. Force pre_encoded_cache (mirrors the minit2i / crop_planner
        # force-overrides above) so an audio dataset always gets a
        # working (and cache-hit-on-resume) latent path.
        if self.is_acestep and latent_encoding_mode != "pre_encoded_cache" and any(
            item.get("item_type") == "audio" for dataset in datasets for item in dataset.items
        ):
            print(f"{self.log_prefix} ACE-Step audio dataset detected: forcing "
                  f"latent_encoding_mode='pre_encoded_cache' (was '{latent_encoding_mode}') - "
                  f"swap_onthefly/onthefly_gpu have no audio-clip encode path yet")
            latent_encoding_mode = "pre_encoded_cache"

        self._refuse_unsupported_audio_only_items(datasets)

        # Setup latent caches (mode-dependent)
        latent_caches = None
        print(f"{self.log_prefix} Latent encoding mode: {latent_encoding_mode}")
        if latent_encoding_mode == "swap_onthefly":
            print(f"{self.log_prefix} Latent swap interval: {latent_encoding_swap_interval} steps")
            print(f"{self.log_prefix} VAE will swap with main model during training")
            # No cache setup needed for swap mode
        elif latent_encoding_mode == "pre_encoded_cache":
            print(f"{self.log_prefix} Using pre-encoded latent disk cache mode")
            latent_caches = self._setup_latent_caches(datasets)
            self._validate_and_generate_latent_caches(datasets, latent_caches, progress_callback, force_recache=force_recache)
            # Resolution curriculum: the up-front pass above cached the WARMUP-dim latents
            # (items currently hold warmup dims). Also pre-generate the TARGET-dim latents
            # now (VAE still resident) so the mid-run switch never re-encodes: rebucket to
            # the target grid, generate the missing (target-keyed) entries, then restore
            # warmup dims for epoch 0. Cache keys embed w_h, so the two sets coexist.
            if self._rc_active and self._rc_switch_epoch < num_epochs:
                _rc_pairs = [(item, dataset) for dataset in datasets for item in dataset.items]
                if bucket_manager:
                    self._rc_rebucket_items(_rc_pairs, bucket_manager, self._rc_normal_res)
                else:
                    self._rc_refit_items(_rc_pairs, self._rc_normal_res)
                print(f"{self.log_prefix} [ResCurriculum] Pre-generating TARGET-resolution "
                      f"latent cache ({self._rc_normal_res}) up front...")
                self._validate_and_generate_latent_caches(datasets, latent_caches, progress_callback, force_recache=False)
                # Restore warmup dims + warmup bucket assignment for epoch 0.
                if bucket_manager:
                    self._rc_rebucket_items(_rc_pairs, bucket_manager, self._rc_warmup_res)
                else:
                    self._rc_refit_items(_rc_pairs, self._rc_warmup_res)
        elif latent_encoding_mode == "onthefly_gpu":
            print(f"{self.log_prefix} Using on-the-fly GPU latent encoding (no cache)")
            # No cache setup needed
        else:
            raise ValueError(f"Invalid latent_encoding_mode: {latent_encoding_mode}")

        # Setup text encoder caches (all architectures)
        text_encoder_caches = None
        print(f"{self.log_prefix} Text encoding mode: {text_encoding_mode}")
        if text_encoding_mode == "swap_onthefly":
            print(f"{self.log_prefix} Swap interval: {text_encoding_swap_interval} steps")
            if self.is_zimage:
                print(f"{self.log_prefix} Text encoder will swap with transformer during training")
            else:
                print(f"{self.log_prefix} Text encoder will swap with U-Net during training")
            # No cache setup needed for swap mode
        elif text_encoding_mode == "pre_encoded_cache":
            print(f"{self.log_prefix} Using pre-encoded disk cache mode")
            text_encoder_caches = self._setup_text_encoder_caches(datasets)
        elif text_encoding_mode == "onthefly_gpu":
            print(f"{self.log_prefix} Using on-the-fly GPU encoding (no cache)")
            # No cache setup needed
        elif text_encoding_mode == "cpu_prefetch":
            print(f"{self.log_prefix} Using CPU-parallel prefetch encoding "
                  f"(TE pinned to CPU, worker prefetches up to "
                  f"{text_encoding_prefetch_depth} batches ahead)")
            # No cache setup; the worker is created per-epoch alongside the
            # main loop (uses the pre-built `batches` list as its work
            # queue). Treated as a peer of swap_onthefly downstream.
        else:
            raise ValueError(f"Invalid text_encoding_mode: {text_encoding_mode}")

        # Clean up stop flag from previous run (if any)
        stop_flag_file = self.output_dir / ".stop_training"
        if stop_flag_file.exists():
            print(f"{self.log_prefix} Removing stale stop flag from previous run")
            stop_flag_file.unlink()

        # Opt-in torch.compile for DiT training. Runs here — after model
        # device/dtype + gradient-checkpointing + adapter/optimizer setup, and
        # after the stop-flag cleanup — but before the first step, so the
        # one-time compilation cost is paid inside the loop's first iteration.
        self._maybe_compile_transformer()

        # Training loop
        global_step = 0
        start_epoch = 0
        resume_batch_idx = 0  # Batch index to resume from within epoch
        # Batches of the current epoch already completed before this session's
        # batch list starts (see _epoch_batch_position).
        self._epoch_batch_offset = 0
        # Bind batch_idx up-front so the emergency-save handler (which references
        # batch_idx) never raises UnboundLocalError when a crash occurs BEFORE the
        # first batch iteration (dataloader / bucket assembly / model setup), which
        # would otherwise mask the real error and abort the emergency checkpoint.
        batch_idx = 0
        resume_training_state = None  # Training state for mid-epoch resume

        # Resume from checkpoint if requested
        # NOTE: Checkpoint weights were already loaded in __init__() if resume_from_checkpoint was set
        # Here we only need to extract step number and load training state (epoch/batch_idx)
        if resume_from_checkpoint:
            if resume_from_checkpoint.lower() == "latest":
                # Use the checkpoint that was actually loaded in __init__ (may differ from "latest" if fallback occurred)
                if self._loaded_checkpoint_path:
                    checkpoint_path = self._loaded_checkpoint_path
                    # Extract step number from filename
                    import re
                    match = re.search(r'_step_(\d+)', Path(checkpoint_path).stem)
                    if match:
                        checkpoint_step = int(match.group(1))
                    else:
                        print(f"{self.log_prefix} WARNING: Could not extract step number from loaded checkpoint: {checkpoint_path}")
                        checkpoint_step = 0
                    checkpoint_result = (checkpoint_path, checkpoint_step)
                elif getattr(self, '_manages_own_resume', False):
                    # Subclasses that manage their own checkpoint format (currently
                    # ControlNetTrainer: directory saves for standard CN, adapter
                    # .safetensors for LLLite) already ran their own resume-detection
                    # in __init__ and left _loaded_checkpoint_path unset because
                    # NOTHING was found there (a legitimate fresh run). Falling back
                    # to find_latest_checkpoint() here would independently rescan
                    # generic `*_step_*_state.json` sidecars and could restore
                    # global_step/optimizer state onto the freshly-initialized
                    # ControlNet weights (an orphaned state.json from a deleted/
                    # differently-named checkpoint) -- start fresh instead.
                    checkpoint_result = None
                else:
                    # Fallback to find_latest_checkpoint (should not normally happen)
                    checkpoint_result = self.find_latest_checkpoint()

                if checkpoint_result is not None:
                    checkpoint_path, checkpoint_step = checkpoint_result
                    print(f"{self.log_prefix} Resuming from checkpoint (weights already loaded in __init__): {checkpoint_path}")
                    # NOTE: Model weights were already loaded in __init__()
                    # We only need the step number here
                    global_step = checkpoint_step

                    # Try to load training state for mid-epoch resume
                    resume_training_state = self.load_training_state(checkpoint_step)
                    if resume_training_state:
                        start_epoch = self._resolve_start_epoch(
                            resume_training_state, global_step, steps_per_epoch, multi_noise_timesteps)
                        resume_batch_idx = resume_training_state['batch_idx']

                        # Use global_step from state.json (most accurate, saved at same time as batch_idx)
                        if 'global_step' in resume_training_state:
                            global_step = resume_training_state['global_step']
                            print(f"{self.log_prefix} Loaded training state: epoch={start_epoch}, batch_idx={resume_batch_idx}, global_step={global_step}")
                        else:
                            # Fallback: use global_step from checkpoint filename
                            print(f"{self.log_prefix} WARNING: No global_step in training state, using checkpoint filename: {global_step}")
                            print(f"{self.log_prefix} Loaded training state: epoch={start_epoch}, batch_idx={resume_batch_idx}")

                        print(f"{self.log_prefix} Mid-epoch resume: epoch {start_epoch + 1}, batch {resume_batch_idx}, step {global_step}")

                        # Restore ReLoRA-specific state (merge_count, etc.)
                        if hasattr(self, '_restore_relora_state'):
                            self._restore_relora_state(resume_training_state)
                    else:
                        # No training state file, fall back to epoch-level resume.
                        start_epoch = self._resolve_start_epoch(
                            None, global_step, steps_per_epoch, multi_noise_timesteps)
                        print(f"{self.log_prefix} Resuming from step {global_step}, epoch {start_epoch + 1}")

                    # Fast-forward every lr_scheduler to match the checkpoint
                    self._fast_forward_lr_schedulers(global_step)

                    # Load optimizer state (momentum, variance, etc.) BEFORE the
                    # LR re-assertion below: torch's Optimizer.load_state_dict
                    # takes only 'params' from the live group and every other key
                    # -- 'lr' included -- from the SAVED group, so loading after
                    # it would silently reinstate the checkpoint's LR for the
                    # first step after a resume.
                    self.load_optimizer_state(checkpoint_step)

                    # Re-assert the YAML config's LR over whatever the resume
                    # restored (needed when the user edits LR before resuming),
                    # AT the schedule's current position -- see lr_utils. The
                    # per-group rates come from the snapshot setup_optimizer
                    # recorded off the adapter's own param groups.
                    self._reassert_config_lr_on_resume()

                    # Restore EMA shadow (no-op unless use_ema; re-inits from
                    # current weights if no saved shadow is found)
                    self.load_ema_state(checkpoint_step)
                else:
                    print(f"{self.log_prefix} No checkpoint found for auto-resume, starting from scratch")
            else:
                # User specified a specific checkpoint file
                checkpoint_path = self.output_dir / resume_from_checkpoint
                if checkpoint_path.exists():
                    print(f"{self.log_prefix} Resuming from specified checkpoint (weights already loaded in __init__): {checkpoint_path}")

                    # NOTE: Model weights were already loaded in __init__()
                    # Extract step number from filename
                    import re
                    match = re.search(r'_step_(\d+)', checkpoint_path.stem)
                    if match:
                        checkpoint_step = int(match.group(1))
                        global_step = checkpoint_step
                    else:
                        print(f"{self.log_prefix} WARNING: Could not extract step number from filename: {checkpoint_path.name}")
                        global_step = 0

                    # Try to load training state for mid-epoch resume
                    resume_training_state = self.load_training_state(checkpoint_step)
                    if resume_training_state:
                        start_epoch = self._resolve_start_epoch(
                            resume_training_state, global_step, steps_per_epoch, multi_noise_timesteps)
                        resume_batch_idx = resume_training_state['batch_idx']

                        # Use global_step from state.json (most accurate, saved at same time as batch_idx)
                        if 'global_step' in resume_training_state:
                            global_step = resume_training_state['global_step']
                            print(f"{self.log_prefix} Loaded training state: epoch={start_epoch}, batch_idx={resume_batch_idx}, global_step={global_step}")
                        else:
                            # Fallback: use global_step from checkpoint filename
                            print(f"{self.log_prefix} WARNING: No global_step in training state, using checkpoint filename: {global_step}")
                            print(f"{self.log_prefix} Loaded training state: epoch={start_epoch}, batch_idx={resume_batch_idx}")

                        print(f"{self.log_prefix} Mid-epoch resume: epoch {start_epoch + 1}, batch {resume_batch_idx}, step {global_step}")

                        # Restore ReLoRA-specific state (merge_count, etc.)
                        if hasattr(self, '_restore_relora_state'):
                            self._restore_relora_state(resume_training_state)
                    else:
                        # No training state file, fall back to epoch-level resume.
                        start_epoch = self._resolve_start_epoch(
                            None, global_step, steps_per_epoch, multi_noise_timesteps)
                        print(f"{self.log_prefix} Resuming from step {global_step}, epoch {start_epoch + 1}")

                    # Fast-forward every lr_scheduler to match the checkpoint
                    self._fast_forward_lr_schedulers(global_step)

                    # Load optimizer state (momentum, variance, etc.) BEFORE the
                    # LR re-assertion below -- see the note in the "latest"
                    # branch: Optimizer.load_state_dict restores the checkpoint's
                    # 'lr' into every param group, so it must not run after it.
                    self.load_optimizer_state(checkpoint_step)

                    # Re-assert the YAML config's LR over whatever the resume
                    # restored, at the schedule's current position (see the
                    # "latest" branch and lr_utils).
                    self._reassert_config_lr_on_resume()

                    # Restore EMA shadow (no-op unless use_ema; re-inits from
                    # current weights if no saved shadow is found)
                    self.load_ema_state(checkpoint_step)
                else:
                    print(f"{self.log_prefix} WARNING: Checkpoint not found: {checkpoint_path}")
                    print(f"{self.log_prefix} Starting from scratch")

        # ============================================================
        # Resume structure-change guard (dataset composition or batch structure)
        # ============================================================
        # The in-loop fingerprint check (~"Mid-epoch resume: restore random state")
        # only fires INSIDE the epoch loop. When the stored epoch/batches_per_epoch
        # combination makes ``range(start_epoch, num_epochs)`` empty (e.g. the dataset
        # grew so the stored epoch index exceeds the new epoch count), that check never
        # runs and the MNT-recompute below can produce a negative total_steps while the
        # loop trains 0 batches. Hoist an equivalent guard to BEFORE the MNT-recompute so
        # it fires even when the (broken) epoch range would be empty.
        #
        # Policy A': keep global_step + optimizer state + the configured ``steps`` as the
        # global_step stop target; discard the non-portable epoch/batch bookkeeping and
        # restart from a fresh epoch boundary. Fires ONLY when the dataset fingerprint or
        # batches_per_epoch changed, so a normal same-structure resume is byte-identical.
        self._resume_structure_changed = False
        if resume_training_state is not None:
            saved_fp = resume_training_state.get('dataset_fingerprint')
            fp_changed = self._check_dataset_fingerprint_changed(saved_fp, self._dataset_fingerprint)
            saved_bpe = resume_training_state.get('batches_per_epoch')
            bpe_changed = (saved_bpe is not None and saved_bpe != batches_per_epoch)
            if fp_changed or bpe_changed:
                self._resume_structure_changed = True
                print(f"{self.log_prefix} Dataset/batch structure changed since checkpoint "
                      f"(fingerprint_changed={fp_changed}, batches_per_epoch: {saved_bpe} -> {batches_per_epoch})")
                print(f"{self.log_prefix} Stored epoch/batch position is not portable; keeping "
                      f"global_step={global_step} and optimizer state, restarting epoch bookkeeping from a fresh boundary.")
                print(f"{self.log_prefix} Configured total_steps={actual_total_steps} is the global_step stop target "
                      f"(remaining {actual_total_steps - global_step} steps ~= "
                      f"{max(0, actual_total_steps - global_step) // max(1, multi_noise_timesteps)} batches at MNT={multi_noise_timesteps})")
                start_epoch = 0
                resume_batch_idx = 0
                resume_training_state = None  # disarms MNT-recompute, in-loop restore, and batch truncation
                if actual_total_steps <= global_step:
                    print(f"{self.log_prefix} WARNING: global_step ({global_step}) already >= total_steps "
                          f"({actual_total_steps}); increase `steps` in config to continue training.")
                if str(lr_scheduler_type).lower() != "constant":
                    print(f"{self.log_prefix} WARNING: non-constant LR scheduler across a dataset-structure change; "
                          f"LR position was fast-forwarded by old global_step and may not match intent.")
                if getattr(self, '_rc_active', False):
                    print(f"{self.log_prefix} WARNING: res-curriculum epoch counter resets to 0; warmup would re-run. "
                          f"Curriculum + dataset swap is unsupported.")

        # ============================================================
        # MNT Change Detection and total_steps Recalculation
        # ============================================================
        # When MNT changes between runs, we need to recalculate total_steps:
        # - global_step (from checkpoint) = already completed steps
        # - remaining_steps = (remaining batches) * new_mnt
        # - new_total_steps = global_step + remaining_steps
        #
        # This ensures training continues for the correct duration regardless
        # of MNT changes during resume.
        if resume_training_state is not None and global_step > 0:
            checkpoint_mnt = resume_training_state.get('multi_noise_timesteps', 1)

            if checkpoint_mnt != multi_noise_timesteps:
                print(f"{self.log_prefix} MNT changed: {checkpoint_mnt} -> {multi_noise_timesteps}")

                # Calculate remaining batches from current position
                remaining_batches_in_epoch = batches_per_epoch - resume_batch_idx
                remaining_full_epochs = num_epochs - start_epoch - 1
                remaining_full_epoch_batches = remaining_full_epochs * batches_per_epoch
                total_remaining_batches = remaining_batches_in_epoch + remaining_full_epoch_batches

                # Calculate remaining steps with NEW MNT value
                remaining_steps = total_remaining_batches * multi_noise_timesteps

                # New total_steps = already completed + remaining
                new_actual_total_steps = global_step + remaining_steps

                print(f"{self.log_prefix} Recalculating total_steps due to MNT change:")
                print(f"{self.log_prefix}   Completed steps (from checkpoint): {global_step}")
                print(f"{self.log_prefix}   Remaining batches: {total_remaining_batches}")
                print(f"{self.log_prefix}   Remaining steps (with new MNT={multi_noise_timesteps}): {remaining_steps}")
                print(f"{self.log_prefix}   Old total_steps: {actual_total_steps}")
                print(f"{self.log_prefix}   New total_steps: {new_actual_total_steps}")

                actual_total_steps = new_actual_total_steps

                # Update DB with corrected total_steps
                if update_total_steps_callback is not None:
                    update_total_steps_callback(actual_total_steps)

                # Note: LR scheduler was already fast-forwarded to global_step
                # It will continue from there with the remaining steps
                # No need to reinitialize optimizer/scheduler since global_step is preserved
                #
                # Warning: For non-constant LR schedulers (cosine, etc.), the scheduler's
                # total_steps was set to the old value. This may cause incorrect LR decay.
                # For constant scheduler, this is not an issue.
                if lr_scheduler_type.lower() != "constant":
                    print(f"{self.log_prefix} WARNING: MNT change with {lr_scheduler_type} LR scheduler")
                    print(f"{self.log_prefix} WARNING: LR scheduler was initialized with old total_steps")
                    print(f"{self.log_prefix} WARNING: LR decay curve may be affected. Consider using 'constant' scheduler for MNT experiments.")

        # Clean up future steps in database (old data from previous interrupted training)
        # This prevents duplicate metrics when training resumes from an earlier step
        if self.run_id is not None:
            self._cleanup_future_metrics(global_step)

        # ============================================================
        # Parameter Change Tracker initialization
        # ============================================================
        self._param_tracker: Optional[ParameterChangeTracker] = None
        if param_tracking:
            tracked_components: Dict[str, torch.nn.Module] = {}
            if getattr(self, 'unet', None) is not None:
                tracked_components['unet'] = self.unet
            elif getattr(self, 'transformer', None) is not None:
                tracked_components['unet'] = self.transformer  # flux2 / zimage
            if getattr(self, 'text_encoder', None) is not None:
                tracked_components['te1'] = self.text_encoder
            if getattr(self, 'text_encoder_2', None) is not None:
                tracked_components['te2'] = self.text_encoder_2
            if (getattr(self, '_train_vision_encoder', False)
                    and getattr(self, 'vision_encoder', None) is not None):
                tracked_components['ve'] = self.vision_encoder
            if tracked_components:
                print(f"{self.log_prefix} [ParamTracker] Initializing "
                      f"(interval={param_tracking_interval} steps, "
                      f"components={list(tracked_components.keys())})...")
                self._param_tracker = ParameterChangeTracker(
                    tracked_components, interval=param_tracking_interval
                )
            else:
                print(f"{self.log_prefix} [ParamTracker] No trainable components found, disabled")

        # Generate step 0 sample to verify base model output.
        self._run_step0_sample_if_due(
            sample_every_n_steps=sample_every_n_steps,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_guidance_scale=sample_guidance_scale,
            sample_steps=sample_steps,
            sample_seed=sample_seed,
            sample_schedule_type=sample_schedule_type,
            global_step=global_step,
        )

        # ------------------------------------------------------------------
        # Online Danbooru augmentation (image-generation) setup
        # ------------------------------------------------------------------
        # Background collector + interrupt-batch injection.  Unlike the tagger
        # there is NO vocabulary expansion (diffusion text conditioning is
        # open-vocab) — we only fetch extra Danbooru images (static user queries
        # + auto/manual under-represented tags) and inject them as ordinary
        # training samples.  Injection encodes via the existing swap-buffer
        # refill cycle, so it is only enabled for the on-the-fly-capable latent
        # modes (pre_encoded_cache / disk caches offload the VAE differently).
        self._danbooru_collector = None
        self._danbooru_inj_batch_size = 0
        self._danbooru_inj_interval = 0
        self._danbooru_metrics_path = None
        self._danbooru_caption_config = None
        try:
            if bool(self.config.get("danbooru_aug_enable", False)):
                if latent_encoding_mode not in ("swap_onthefly", "onthefly_gpu"):
                    print(f"{self.log_prefix} [DanbooruAug] Disabled: requires "
                          f"latent_encoding_mode swap_onthefly/onthefly_gpu "
                          f"(got {latent_encoding_mode}).")
                elif getattr(self, "use_condition_images", False) and not getattr(self, "_is_outpaint_mode", False):
                    # ControlNet condition-image training needs a paired condition
                    # image per sample, which Danbooru-injected samples don't have.
                    # Outpaint conditioning_mode is exempt: its condition is built
                    # from each item's OWN image (no paired dataset), so a
                    # Danbooru-injected sample is just as usable as any other item.
                    print(f"{self.log_prefix} [DanbooruAug] Disabled: not supported "
                          f"with ControlNet condition-image training.")
                else:
                    from core.training.danbooru_image_augment import (
                        DanbooruImageCollector, DatasetTagFrequencyAnalyzer,
                    )
                    # Coarse aspect-ratio bucket set centred on the base-resolution
                    # area.  Using a small set (not the ~41 full training buckets)
                    # so a full SAME-resolution injection batch can actually be
                    # drained from a bounded buffer; also avoids the extreme
                    # 8192x128-style buckets.
                    _R = (base_resolutions[0] if base_resolutions else 1024)
                    _area = float(_R * _R)
                    _bucket_res = []
                    for _aw, _ah in ((1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16)):
                        _w = int((_area * _aw / _ah) ** 0.5)
                        _h = int((_area * _ah / _aw) ** 0.5)
                        _w -= _w % 8
                        _h -= _h % 8
                        _bucket_res.append((max(64, _w), max(64, _h)))

                    # Auto deficiency: rarest tags in the training dataset captions.
                    # Cap the scan so startup stays bounded on multi-million-item
                    # datasets (a sample is enough to surface the rarest tags).
                    _defic_queries = []
                    if bool(self.config.get("danbooru_aug_deficiency_enable", True)):
                        _an = DatasetTagFrequencyAnalyzer()
                        _MAX_SCAN = 100000
                        _scanned = 0
                        _scan_done = False
                        for _ds in datasets:
                            if _scan_done:
                                break
                            for _it in getattr(_ds, "items", []):
                                if _scanned >= _MAX_SCAN:
                                    _scan_done = True
                                    break
                                _scanned += 1
                                _cap = _it.get("caption") or ""
                                if _cap:
                                    _an.add_caption_tags(
                                        [t.strip() for t in _cap.replace("\n", ",").split(",") if t.strip()]
                                    )
                        _defic_queries = _an.deficient_queries(
                            min_count=int(self.config.get("danbooru_aug_deficiency_min_count", 20)),
                            top_k=int(self.config.get("danbooru_aug_deficiency_top_k", 200)),
                        )
                        print(f"{self.log_prefix} [DanbooruAug] Scanned {_scanned} captions "
                              f"({_an.total_unique_tags} unique tags) -> {len(_defic_queries)} deficiency queries")
                    # Manual deficiency: explicit user top-up tags.
                    _manual = self.config.get("danbooru_aug_deficiency_manual", "") or ""
                    _manual_q = [
                        q.strip().replace(" ", "_")
                        for q in _manual.replace("\n", ",").split(",") if q.strip()
                    ]
                    _defic_queries = list(dict.fromkeys(_defic_queries + _manual_q))

                    _static = [
                        q.strip()
                        for q in (self.config.get("danbooru_aug_queries", "") or "").splitlines()
                        if q.strip()
                    ]
                    # Default buffer scales with batch size so a same-bucket
                    # injection batch can be filled even when images spread across
                    # several aspect-ratio buckets.
                    _bs = self.config.get("danbooru_aug_buffer_size") or max(32, 16 * batch_size)
                    self._danbooru_inj_batch_size = max(
                        1, round(float(self.config.get("danbooru_aug_injection_ratio", 1.0)) * batch_size)
                    )
                    self._danbooru_inj_interval = max(
                        1, int(self.config.get("danbooru_aug_injection_interval", 4))
                    )
                    self._danbooru_metrics_path = os.path.join(
                        str(self.output_dir), "danbooru_metrics.json"
                    )
                    # Dedicated caption-processing config for injected samples
                    # (separate from per-dataset caption_processing). Consumed by
                    # process_caption_with_tag_data() at splice time, per-epoch.
                    self._danbooru_caption_config = {
                        "caption_dropout_rate": float(self.config.get("danbooru_aug_caption_dropout_rate", 0.0)),
                        "keep_tokens": int(self.config.get("danbooru_aug_keep_tokens", 0)),
                        "shuffle_tokens": bool(self.config.get("danbooru_aug_shuffle_tags", False)),
                        "shuffle_per_epoch": True,
                        "shuffle_keep_first_n": int(self.config.get("danbooru_aug_shuffle_keep_first_n", 0)),
                        "shuffle_tag_groups": ["General", "Character", "Copyright", "Artist", "Meta"],
                        "shuffle_groups_together": False,
                        "exclude_person_count_from_shuffle": True,
                        "tag_dropout_rate": float(self.config.get("danbooru_aug_tag_dropout_rate", 0.0)),
                        "tag_dropout_per_epoch": True,
                        "tag_dropout_keep_first_n": int(self.config.get("danbooru_aug_tag_dropout_keep_first_n", 0)),
                        "tag_dropout_category_rates": {},
                        "tag_dropout_exclude_person_count": True,
                        "category_order": ["Rating", "General", "Character", "Copyright", "Artist", "Meta"],
                    }
                    self._danbooru_collector = DanbooruImageCollector(
                        static_queries=_static,
                        deficiency_queries=_defic_queries,
                        bucket_resolutions=_bucket_res,
                        weight_static=float(self.config.get("danbooru_aug_weight_static", 1.0)),
                        weight_deficiency=float(self.config.get("danbooru_aug_weight_deficiency", 1.0)),
                        min_score=int(self.config.get("danbooru_aug_min_score", 0)),
                        max_posts_per_query=int(self.config.get("danbooru_aug_max_posts_per_query", 200)),
                        api_interval=float(self.config.get("danbooru_aug_api_interval", 1.4)),
                        dl_speed_kbps=int(self.config.get("danbooru_aug_dl_speed_kbps", 500)),
                        buffer_size=int(_bs),
                        include_rating_tag=bool(self.config.get("danbooru_aug_include_rating_tag", False)),
                        max_caption_tags=int(self.config.get("danbooru_aug_max_caption_tags", 0)),
                        quality_tag_enable=bool(self.config.get("danbooru_quality_tag_enable", False)),
                        quality_tag_thresholds=str(self.config.get("danbooru_quality_tag_thresholds", "") or ""),
                        quality_tag_attach_negative=bool(self.config.get("danbooru_quality_tag_attach_negative", False)),
                        control_dir=str(self.output_dir),
                    )
                    # Configure the download-speed safety monitor (throttle/ban guard).
                    from core.tagger.download_speed_monitor import get_speed_monitor
                    get_speed_monitor().configure(
                        enabled=bool(self.config.get("danbooru_speed_check_enable", True)),
                        degraded_kbps=int(self.config.get("danbooru_speed_degraded_kbps", 250)),
                        min_slow_streak=int(self.config.get("danbooru_speed_min_slow_streak", 8)),
                        min_slow_seconds=float(self.config.get("danbooru_speed_min_slow_seconds", 90)),
                        cooldown_seconds=float(self.config.get("danbooru_speed_cooldown_seconds", 3600)),
                    )
                    self._danbooru_collector.start()
                    print(f"{self.log_prefix} [DanbooruAug] Enabled: "
                          f"{len(_static)} static + {len(_defic_queries)} deficiency queries, "
                          f"inject {self._danbooru_inj_batch_size} img every "
                          f"{self._danbooru_inj_interval} batches, buffer={_bs}")
        except Exception as _dae:  # noqa: BLE001 — never block training on aug setup
            print(f"{self.log_prefix} [DanbooruAug] Setup failed (continuing without): {_dae}")
            self._danbooru_collector = None

        # "Did this invocation train anything?" -- asserted before reporting
        # success, so a run whose every batch was dropped or skipped cannot
        # finish green. This is the only product-default detector for that: the
        # optimizer update census is opt-in AND is deliberately not asserted for
        # a skipped batch, which is exactly the case that produces a no-op run.
        # Reset per call, not per resume.
        self._epochs_entered = 0
        self._backwards_completed = 0
        self._batches_skipped = 0
        # What "resume from the last periodic checkpoint" refers to, or None
        # when none has been written yet -- in which case _resume_point_sentence
        # falls back to what this invocation resumed from, and says which of
        # "save_every=0" and "not reached yet" is the actual reason.
        self._last_periodic_checkpoint_step = None
        self._periodic_save_every = save_every_n_steps
        self._resume_checkpoint_label = (
            getattr(self, "_loaded_checkpoint_path", None)
            or resume_from_checkpoint
            or self.resume_from_checkpoint
        )
        self._partial_step_taint = None

        try:
            # resume_seq: 0 for a fresh run, one past the highest recorded seq when
            # resuming (this run already has metric rows from a prior session). New
            # steps continue the global step counter, so (run_id, step) stays unique;
            # resume_seq only labels which session each row came from.
            if self.run_id is not None:
                try:
                    from database.models import TrainingMetrics
                    from database import get_training_db
                    from sqlalchemy import func as _sqlfunc
                    _db = next(get_training_db())
                    _max_seq = _db.query(_sqlfunc.max(TrainingMetrics.resume_seq)).filter(
                        TrainingMetrics.run_id == self.run_id
                    ).scalar()
                    _db.close()
                    self.resume_seq = (int(_max_seq) + 1) if _max_seq is not None else 0
                except Exception as _e:
                    print(f"{self.log_prefix} resume_seq detection failed ({_e}); defaulting to 0")
                    self.resume_seq = 0
                print(f"{self.log_prefix} Metrics resume_seq = {self.resume_seq}")

            for epoch in range(start_epoch, num_epochs):
                # Recorded with each metric (for epoch-boundary markers in the UI).
                self._current_epoch = epoch
                self._epochs_entered += 1
                self._epoch_batch_offset = 0  # only the resumed epoch is truncated
                print(f"\n{self.log_prefix} Epoch {epoch + 1}/{num_epochs}")

                # Reload datasets for per-epoch shuffle/dropout
                # (This regenerates captions with different shuffle/dropout based on epoch_num)
                for dataset in datasets:
                    if hasattr(dataset, 'reload_for_epoch'):
                        new_items = dataset.reload_for_epoch(epoch_num=epoch, run_id=run_id)
                        if new_items is not None:
                            # Dataset was reloaded with new items
                            dataset.items = new_items
                            print(f"{self.log_prefix} Reloaded dataset {dataset.unique_id} for epoch {epoch + 1} ({len(dataset.items)} items)")
                        else:
                            # Dataset reload skipped (same epoch as initial load, items already loaded)
                            print(f"{self.log_prefix} Using pre-loaded dataset {dataset.unique_id} for epoch {epoch + 1} ({len(dataset.items)} items)")

                # Validate and generate text encoder cache for new captions (all architectures)
                # Only for pre_encoded_cache mode
                if text_encoding_mode == "pre_encoded_cache":
                    self._validate_and_generate_text_encoder_caches(datasets, text_encoder_caches, progress_callback, epoch_num=epoch)

                # Create all_items list (needed for swap buffer and batching)
                all_items = []
                for dataset in datasets:
                    all_items.extend([(item, dataset) for item in dataset.items])

                # Resolution curriculum: apply this epoch's phase resolution.
                #   warmup while epoch < switch_epoch, else target.
                # Bucketing: rebuild bucket assignments only when the phase changes (the
                #   bucket dicts persist across epochs). No-bucketing: reload reintroduces
                #   original dims every epoch, so re-fit each epoch to the active area.
                # Skipped when crop augmentation owns per-epoch re-bucketing (SDXL-only;
                # crop_planner is fixed at the target grid — the two features don't combine).
                if self._rc_active and self.crop_planner is None:
                    _rc_desired = "warmup" if epoch < self._rc_switch_epoch else "normal"
                    _rc_desired_res = self._rc_warmup_res if _rc_desired == "warmup" else self._rc_normal_res
                    _rc_changed = (_rc_desired != self._rc_current_phase)
                    if bucket_manager is not None:
                        if _rc_changed:
                            self._rc_rebucket_items(all_items, bucket_manager, _rc_desired_res)
                    else:
                        # No-bucketing: dims come straight from all_items each epoch.
                        self._rc_refit_items(all_items, _rc_desired_res)
                    if _rc_changed:
                        print(f"{self.log_prefix} [ResCurriculum] Epoch {epoch + 1}: phase "
                              f"-> {_rc_desired.upper()} at {_rc_desired_res}")
                        self._rc_current_phase = _rc_desired

                # Mid-epoch resume: restore random state BEFORE building batches
                # This ensures batches are shuffled in the same order as the interrupted run
                if epoch == start_epoch and resume_training_state is not None:
                    import random

                    # Check if dataset has changed since checkpoint was saved
                    # If changed, the saved random_state is invalid and should NOT be restored
                    saved_fingerprint = resume_training_state.get('dataset_fingerprint')
                    dataset_changed = self._check_dataset_fingerprint_changed(saved_fingerprint, self._dataset_fingerprint)

                    # Crop-plan change also invalidates the saved shuffle/crop reproducibility.
                    saved_crop_fp = resume_training_state.get('crop_plan_fingerprint')
                    crop_changed = (saved_crop_fp != getattr(self, '_crop_plan_fingerprint', None))
                    if crop_changed and not dataset_changed:
                        print(f"{self.log_prefix} WARNING: Crop augmentation params changed since checkpoint!")
                    dataset_changed = dataset_changed or crop_changed

                    if dataset_changed:
                        print(f"{self.log_prefix} WARNING: Dataset has changed since checkpoint was saved!")
                        print(f"{self.log_prefix} Saved shuffle state is invalid - using fresh random state")
                        print(f"{self.log_prefix} Restarting current epoch from batch 0 (global_step={global_step} preserved)")
                        # Do NOT restore random state - let it use current random state.
                        # Also clear resume_training_state so the batch-truncation at
                        # ``batches = batches[resume_batch_idx:]`` below does NOT run —
                        # otherwise we'd skip the first resume_batch_idx batches of an
                        # entirely different sample order, which means arbitrary
                        # samples get skipped rather than the ones already trained on.
                        resume_training_state = None
                        resume_batch_idx     = 0
                    else:
                        print(f"{self.log_prefix} Dataset unchanged - restoring random state for mid-epoch resume...")
                        random.setstate(resume_training_state['random_state'])

                # Load priority training config (if specified)
                priority_config = None
                if priority_training:
                    try:
                        from core.training.priority_training import (
                            PriorityTrainingConfig, classify_items, build_priority_batches
                        )
                        if isinstance(priority_training, dict) and "_legacy_path" in priority_training:
                            priority_config = PriorityTrainingConfig.load(priority_training["_legacy_path"])
                        else:
                            priority_config = PriorityTrainingConfig.from_dict(priority_training)
                    except Exception as e:
                        print(f"{self.log_prefix} WARNING: Failed to load priority training config: {e}")
                        print(f"{self.log_prefix} Continuing with normal training")

                # Epoch-dynamic crop re-bucketing (SDXL crop augmentation). Re-populate
                # bucket_manager.buckets for THIS epoch from per-(item,epoch) crop specs,
                # and attach item["_crop_spec"] (read by the encode path). Skipped for
                # priority training (the priority path builds its own bucket managers);
                # in that case crop augmentation degrades to full-image bucketing.
                if self.crop_planner is not None and bucket_manager is not None:
                    if priority_config and priority_config.entries:
                        if epoch == start_epoch:
                            print(f"{self.log_prefix} WARNING: crop augmentation is not applied "
                                  f"together with priority training (using full-image bucketing)")
                    else:
                        from core.training.bucketing import BucketResolution
                        bucket_manager.buckets = {}
                        _crop_count = 0
                        _excluded_unfit = 0
                        for item, dataset in all_items:
                            image_path = item["image_path"]
                            try:
                                ow, oh = self._get_original_size_for_item(item)
                            except Exception:
                                ow, oh = item.get("width", 1024), item.get("height", 1024)
                            spec = self.crop_planner.spec_for(epoch, image_path, ow, oh)
                            # Drop items whose chosen bucket previously OOM'd at even
                            # one sample -- it cannot fit on this hardware/config.
                            if (spec.bucket_w, spec.bucket_h) in self._unfittable_buckets:
                                _excluded_unfit += 1
                                continue
                            reference_images = item.get("reference_images", [])
                            has_reference = len(reference_images) > 0
                            _, image_info = bucket_manager.assign_image_to_bucket(
                                image_path=image_path,
                                width=spec.bucket_w,
                                height=spec.bucket_h,
                                caption=item.get("caption", ""),
                                dataset_unique_id=getattr(dataset, "unique_id", None),
                                has_reference=has_reference,
                                reference_images=reference_images if reference_images else None,
                                forced_bucket=BucketResolution(spec.bucket_w, spec.bucket_h),
                            )
                            if item.get("_ve_reconstruction_mode"):
                                image_info["_ve_reconstruction_mode"] = True
                            image_info["_crop_spec"] = spec
                            image_info["width"] = spec.bucket_w
                            image_info["height"] = spec.bucket_h
                            if not spec.is_full:
                                _crop_count += 1
                        _unfit_note = (f", {_excluded_unfit} excluded (un-fittable buckets: "
                                       f"{len(self._unfittable_buckets)})") if _excluded_unfit else ""
                        print(f"{self.log_prefix} [crop] Epoch {epoch + 1}: re-bucketed "
                              f"{len(all_items)} items ({_crop_count} cropped, "
                              f"{len(all_items) - _crop_count} full); "
                              f"{len(bucket_manager.get_bucket_counts())} buckets{_unfit_note}")

                # LTX-2.3 VIDEO batching (P6): the image bucket_manager SKIPS
                # item_type=="video" items (P5 skip-guards), so for a video dataset it
                # yields ZERO batches -> "Buffer pre-filled with 0 latents" -> 0 steps.
                # Build video batches directly from the annotated video item dicts (fields
                # set by _annotate_video_items), grouping by
                # (bucket_width, bucket_height, clip_length) so each batch is UNIFORM in
                # (spatial, frame-count) -- required for the 5D latents to stack (P4c).
                # Each emitted batch has the SAME shape as the image path: [(item, dataset), ...].
                # Grouped from the annotated item dicts directly (not VBM.build_batch_indices)
                # so item["image_path"] stays the real training key and there is no coupling
                # to VideoBucketManager internals. No-op (empty) for a non-video arch
                # and for image-only datasets, so the image path stays byte-for-byte
                # unchanged.
                ltx2_video_batches = []
                if self._temporal_spec() is not None:
                    from collections import OrderedDict as _OD
                    _vgroups = _OD()
                    for _item, _dataset in all_items:
                        if _item.get("item_type") != "video":
                            continue
                        _vkey = (_item.get("bucket_width"),
                                 _item.get("bucket_height"),
                                 _item.get("clip_length"))
                        _vgroups.setdefault(_vkey, []).append((_item, _dataset))
                    for _vkey, _members in _vgroups.items():
                        for _i in range(0, len(_members), batch_size):
                            ltx2_video_batches.append(_members[_i:_i + batch_size])
                _has_ltx2_video = bool(ltx2_video_batches)

                # ACE-Step AUDIO batching (Phase 8a): mirrors the LTX-2.3 video
                # batching above exactly -- the image bucket_manager skips
                # item_type=="audio" items (skip-guards added above), so a
                # pure-audio dataset would otherwise yield zero batches. Group
                # by the item's DECLARED clip duration (clip_seconds, falling
                # back to the probed audio_meta duration, rounded to avoid
                # float-probe jitter splitting an otherwise-uniform dataset
                # into spurious groups) rather than the post-encode latent
                # frame count, since encoding happens later and a shared
                # declared duration deterministically yields a shared encoded
                # T (same VAE, same input length). No-op (empty) for non-ACE-Step
                # and for image/video-only datasets.
                acestep_audio_batches = []
                if self.is_acestep:
                    from collections import OrderedDict as _OD2
                    _agroups = _OD2()
                    for _item, _dataset in all_items:
                        if _item.get("item_type") != "audio":
                            continue
                        _araw = _item.get("clip_seconds") or _item.get("duration")
                        _akey = round(float(_araw), 2) if _araw else None
                        _agroups.setdefault(_akey, []).append((_item, _dataset))
                    for _akey, _members in _agroups.items():
                        for _i in range(0, len(_members), batch_size):
                            acestep_audio_batches.append(_members[_i:_i + batch_size])
                _has_acestep_audio = bool(acestep_audio_batches)

                # When video/audio items are present, exclude them from the IMAGE-side
                # batching (priority classify + simple sequential chunking) so they are
                # not double-counted or placed into ÷8 image buckets.
                _image_all_items = (
                    [x for x in all_items
                     if x[0].get("item_type") not in ("video", "audio")]
                    if (_has_ltx2_video or _has_acestep_audio) else all_items
                )

                # Create batches
                if bucket_manager:
                    # BucketManager only manages items, we need to pair with datasets
                    # Build mapping from image_path to dataset
                    path_to_dataset = {}
                    for dataset in datasets:
                        for item in dataset.items:
                            path_to_dataset[item["image_path"]] = dataset

                    if priority_config and priority_config.entries:
                        # Priority training: split items, build priority batches first.
                        # LTX-2.3 video items are excluded here (batched separately below)
                        # so they are not routed through the ÷8 image bucket manager.
                        priority_items, normal_items = classify_items(_image_all_items, priority_config)

                        # Build priority batches (sorted by entry index, bucketed by resolution)
                        priority_batches = build_priority_batches(
                            priority_items, batch_size, bucket_manager
                        )

                        # Build normal batches from remaining items using bucket manager
                        # Temporarily replace bucket contents with normal items only
                        from core.training.bucketing import BucketManager
                        normal_bucket_manager = BucketManager(
                            base_resolutions=bucket_manager.base_resolutions,
                            divisibility=8,
                            strategy=bucket_manager.strategy,
                            multi_resolution_mode=bucket_manager.multi_resolution_mode,
                        )
                        for item, dataset in normal_items:
                            normal_bucket_manager.assign_image_to_bucket(
                                image_path=item["image_path"],
                                width=item.get("width", 1024),
                                height=item.get("height", 1024),
                                caption=item.get("caption", ""),
                                dataset_unique_id=getattr(dataset, 'unique_id', None),
                            )
                        normal_item_batches = normal_bucket_manager.build_batch_indices(batch_size)
                        normal_batches = []
                        for item_batch in normal_item_batches:
                            batch_with_dataset = [
                                (item, path_to_dataset[item["image_path"]])
                                for item in item_batch
                            ]
                            normal_batches.append(batch_with_dataset)

                        # Combine: priority x multiplier + normal + LTX-2.3 video + ACE-Step audio
                        batches = (priority_batches * priority_config.multiplier + normal_batches
                                   + ltx2_video_batches + acestep_audio_batches)
                        print(f"{self.log_prefix} [PriorityTraining] Epoch batch structure: "
                              f"{len(priority_batches)} priority batches x {priority_config.multiplier} "
                              f"+ {len(normal_batches)} normal batches "
                              f"+ {len(ltx2_video_batches)} video batches "
                              f"+ {len(acestep_audio_batches)} audio batches = {len(batches)} total")
                    else:
                        # Standard bucketed batching (no priority)
                        item_batches = bucket_manager.build_batch_indices(batch_size)
                        batches = []
                        for item_batch in item_batches:
                            batch_with_dataset = [
                                (item, path_to_dataset[item["image_path"]])
                                for item in item_batch
                            ]
                            batches.append(batch_with_dataset)
                        # Append LTX-2.3 video / ACE-Step audio batches (empty for
                        # image-only datasets).
                        batches = batches + ltx2_video_batches + acestep_audio_batches
                else:
                    # Simple sequential batching. LTX-2.3 video / ACE-Step audio items
                    # are batched separately (grouped by (spatial, clip_length) /
                    # clip duration) and appended, so video/audio-only datasets still
                    # get non-empty, uniform batches here.
                    if priority_config and priority_config.entries:
                        priority_items, normal_items = classify_items(_image_all_items, priority_config)
                        p_items = [(item, dataset) for item, dataset, _ in priority_items]
                        priority_batches = [p_items[i:i+batch_size] for i in range(0, len(p_items), batch_size)]
                        normal_batches = [normal_items[i:i+batch_size] for i in range(0, len(normal_items), batch_size)]
                        batches = (priority_batches * priority_config.multiplier + normal_batches
                                   + ltx2_video_batches + acestep_audio_batches)
                        print(f"{self.log_prefix} [PriorityTraining] Epoch batch structure: "
                              f"{len(priority_batches)} priority x {priority_config.multiplier} "
                              f"+ {len(normal_batches)} normal "
                              f"+ {len(ltx2_video_batches)} video "
                              f"+ {len(acestep_audio_batches)} audio = {len(batches)} total")
                    else:
                        batches = [_image_all_items[i:i+batch_size] for i in range(0, len(_image_all_items), batch_size)]
                        batches = batches + ltx2_video_batches + acestep_audio_batches

                batches = self._drop_unfittable_batches(batches)

                # Mid-epoch resume: skip completed batches
                # (random state was already restored before batch building)
                if epoch == start_epoch and resume_training_state is not None:
                    print(f"{self.log_prefix} Skipping {resume_batch_idx} completed batches...")
                    batches = batches[resume_batch_idx:]
                    self._epoch_batch_offset = resume_batch_idx

                    # Clear resume state so we don't skip batches in subsequent epochs
                    resume_training_state = None

                # Inject reference_images for ve_reconstruction_mode items (use own image as reference).
                # Must happen BEFORE the batch splitting below so these items go into "ref" sub-batches.
                if getattr(self, 'vision_encoder', None) is not None:
                    for _b in batches:
                        for _item, _ in _b:
                            if _item.get("_ve_reconstruction_mode") and not _item.get("reference_images"):
                                _item["reference_images"] = [_item["image_path"]]

                # When VE is configured, split any mixed batch (ref + no-ref) into pure sub-batches.
                # Ref-image batches and no-ref batches have different embedding shapes so they cannot
                # be collated together.
                if getattr(self, 'vision_encoder', None) is not None:
                    import random as _random_ve
                    clean_batches = []
                    for _b in batches:
                        _ref_items   = [(_i, _ds) for _i, _ds in _b if _i.get("reference_images")]
                        _noref_items = [(_i, _ds) for _i, _ds in _b if not _i.get("reference_images")]
                        if _ref_items and _noref_items:
                            # Mixed batch: split into two pure sub-batches
                            clean_batches.append(_ref_items)
                            clean_batches.append(_noref_items)
                            if self.debug_vram:
                                print(f"{self.log_prefix} [VE] Split mixed batch → "
                                      f"{len(_ref_items)} ref + {len(_noref_items)} no-ref sub-batches")
                        else:
                            clean_batches.append(_b)
                    _random_ve.shuffle(clean_batches)
                    batches = clean_batches

                # Interrupt-batch injection of online Danbooru samples (image-gen
                # augmentation).  Drained from the bounded collector buffer and
                # spliced every N base batches; their latents/embeddings are
                # encoded by the swap-refill cycle below (no per-step encoder
                # swap).  Injected batches are first-class — counted in the step
                # totals computed further down.
                if getattr(self, "_danbooru_collector", None) is not None and self._danbooru_inj_interval > 0:
                    try:
                        self._danbooru_collector.reset_download_cycle()
                    except Exception:
                        pass
                    from core.training.caption_processor import process_caption_with_tag_data
                    _inj_n = self._danbooru_inj_batch_size
                    _cap_cfg = self._danbooru_caption_config or {}
                    _pseudo_ds = datasets[0] if datasets else None
                    _spliced = []
                    _injected = 0
                    for _bi, _b in enumerate(batches):
                        _spliced.append(_b)
                        if (_bi + 1) % self._danbooru_inj_interval == 0:
                            _items = self._danbooru_collector.drain_batch(_inj_n)
                            if _items:
                                _danb_batch = []
                                for _ri in _items:
                                    _ipath = f"danbooru://{_ri.post_id}"
                                    # Build the caption per-epoch with the dedicated
                                    # shuffle/dropout config (seeded by path+epoch).
                                    try:
                                        _cap = process_caption_with_tag_data(
                                            _ri.tag_data, epoch, _ipath, _cap_cfg
                                        )
                                    except Exception:
                                        _cap = ", ".join(t["tag"] for t in _ri.tag_data)
                                    _danb_batch.append(({
                                        "image_path": _ipath,
                                        "caption": _cap,
                                        "width": _ri.bucket_w,
                                        "height": _ri.bucket_h,
                                        "_danbooru_image_bytes": _ri.image_bytes,
                                        "_danbooru": True,
                                    }, _pseudo_ds))
                                _spliced.append(_danb_batch)
                                _injected += 1
                    if _injected:
                        batches = _spliced
                        print(f"{self.log_prefix} [DanbooruAug] Injected {_injected} batch(es) "
                              f"x{_inj_n} img into epoch {epoch + 1} (total batches now {len(batches)})")

                # Initialize swap mode buffer if needed (all architectures)
                # Use dict keyed by image_path for robust lookup (immune to index misalignment)
                use_swap_buffer = text_encoding_mode in ("swap_onthefly", "cpu_prefetch")
                swap_buffer = {} if use_swap_buffer else None
                next_swap_at_step = 0 if swap_buffer is not None else -1

                # cpu_prefetch sets up its own background worker for the
                # frozen TE; the per-batch refill in this loop is replaced
                # by a queue.get() against that worker. The pre-fill below
                # only runs for swap_onthefly.
                te_prefetcher = None
                if text_encoding_mode == "cpu_prefetch":
                    from core.training.cpu_te_prefetch import CpuTextEncoderPrefetcher
                    # Make sure the TE lives on CPU before the worker reads it.
                    if self.text_encoder is not None:
                        self.text_encoder.to("cpu").eval().requires_grad_(False)
                    te_prefetcher = CpuTextEncoderPrefetcher(
                        encode_batch_fn=lambda caps, lyr=None: self.encode_captions_batched(
                            caps, requires_grad=False, lyrics=lyr
                        ),
                        batches=list(batches),
                        prefetch_depth=int(text_encoding_prefetch_depth or 4),
                        log_prefix=f"{self.log_prefix} [cpu_prefetch]",
                    )
                    te_prefetcher.start()
                    print(f"{self.log_prefix} cpu_prefetch worker engaged "
                          f"(TE pinned on CPU; main model on GPU)")

                # Pre-fill swap buffer for first interval (swap_onthefly only —
                # cpu_prefetch's worker drains lazily via the queue).
                if swap_buffer is not None and text_encoding_mode == "swap_onthefly":
                    print(f"{self.log_prefix} Pre-filling swap buffer for first {text_encoding_swap_interval} steps...")
                    if progress_callback:
                        progress_callback(
                            phase="text_encoder_cache",
                            step=0,
                            total=text_encoding_swap_interval
                        )

                    # Move Text Encoder to GPU for encoding
                    self.move_text_encoder_to_gpu()
                    # Move main model to CPU to free VRAM
                    self.move_main_model_to_cpu()

                    # Encode captions for first interval
                    # Use batches (which have bucket info) instead of all_items
                    buffer_items = []
                    for batch in batches[:text_encoding_swap_interval]:
                        buffer_items.extend(batch)
                    for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding captions")):
                        # Abort promptly on user stop (KeyboardInterrupt is caught by
                        # train()'s handler; the process exits, so device restoration
                        # of the swapped-out main model is moot).
                        self._check_stop_requested()

                        caption = item.get("caption", "")
                        image_path = item["image_path"]
                        embeddings, auxiliary_data = self.encode_caption(
                            caption, requires_grad=False, lyrics=item.get("lyrics", "")
                        )
                        # Store on CPU to save GPU VRAM, keyed by image_path
                        # auxiliary_data: attention_mask (Z-Image), pooled_embeddings (SDXL), None (SD1.5)
                        swap_buffer[image_path] = (
                            embeddings.cpu(),
                            self._aux_to_cpu(auxiliary_data),
                            caption,  # String (CPU memory, minimal overhead)
                        )

                        # Send progress update
                        if progress_callback and idx % 10 == 0:
                            progress_callback(
                                phase="text_encoder_cache",
                                step=idx,
                                total=len(buffer_items)
                            )

                    # Move Text Encoder back to CPU
                    self.move_text_encoder_to_cpu()
                    # Move main model to GPU for training
                    self.move_main_model_to_gpu()

                    next_swap_at_step = text_encoding_swap_interval
                    print(f"{self.log_prefix} Buffer pre-filled with {len(swap_buffer)} embeddings")

                # Initialize latent swap mode buffer if needed
                # Use dict keyed by image_path for robust lookup (immune to index misalignment)
                latent_swap_buffer = {} if latent_encoding_mode == "swap_onthefly" else None
                next_latent_swap_at_step = 0 if latent_swap_buffer is not None else -1

                # Pre-fill latent swap buffer for first interval
                if latent_swap_buffer is not None:
                    print(f"{self.log_prefix} Pre-filling latent swap buffer for first {latent_encoding_swap_interval} steps...")
                    if self.debug_vram:
                        _vramdiag("prefill_start")
                    if progress_callback:
                        progress_callback(
                            phase="latent_cache",
                            step=0,
                            total=latent_encoding_swap_interval
                        )

                    # This prefill is a VAE-only encode: nothing but the VAE needs to be
                    # GPU-resident. The main model (offloaded below) is not the whole story
                    # — when text_encoding_mode is NOT swap_onthefly the TE pre-fill block
                    # above is skipped, so the text encoder(s) are still on the GPU here and
                    # co-reside with the VAE. On epoch>=2 / resume their optimizer state
                    # (fp32 m/v, as large as the trained TE params) is GPU-resident too. The
                    # pre_encoded_cache path already offloads the TEs for its encode; mirror
                    # that here so the two latent paths are symmetric and neither pins the
                    # training stack beside the VAE. Guarded by pre-encode device so it is a
                    # no-op for swap-TE / cached-TE / frozen-TE setups, and restores only
                    # what was on the GPU to begin with.
                    te_on_gpu = (
                        self.text_encoder is not None
                        and next(self.text_encoder.parameters()).device.type != "cpu"
                    )
                    te2_on_gpu = (
                        getattr(self, "is_sdxl", False)
                        and getattr(self, "text_encoder_2", None) is not None
                        and next(self.text_encoder_2.parameters()).device.type != "cpu"
                    )

                    # Move VAE to GPU for encoding
                    self.move_vae_to_gpu()
                    # Move main model to CPU to free VRAM. move_main_model_to_cpu moves only
                    # the weights; relocate the optimizer's GPU-resident state here too so it
                    # does not co-reside with the VAE during this prefill encode (mirrors
                    # _ve_set_device). Done at the call site rather than inside
                    # move_main_model_to_cpu because the mid-training latent/TE swaps that
                    # also call it must keep optimizer state on the GPU for optimizer.step.
                    # No-op on fresh runs (optimizer state allocated lazily on first step).
                    self.move_main_model_to_cpu()
                    self._relocate_main_model_optimizer_state("cpu")
                    if self.debug_vram:
                        _vramdiag("after_move_main_model_to_cpu")
                    # Offload the text encoder(s) + their optimizer state too (no-op when
                    # already on CPU, e.g. swap/cached-TE modes).
                    if te_on_gpu or te2_on_gpu:
                        self.move_text_encoder_to_cpu()
                        self._relocate_text_encoder_optimizer_state("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if self.debug_vram:
                        _vramdiag("after_te_offload_and_empty_cache")

                    try:
                        # Encode images for first interval
                        # Use batches (which have bucket info) instead of all_items
                        buffer_items = []
                        for batch in batches[:latent_encoding_swap_interval]:
                            buffer_items.extend(batch)
                        for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding latents")):
                            # Abort promptly on user stop (raises KeyboardInterrupt; the
                            # finally below restores the training stack to the GPU).
                            self._check_stop_requested()

                            image_path = item["image_path"]
                            caption = item.get("caption", "")
                            width = item.get("width") or item.get("bucket_width")
                            height = item.get("height") or item.get("bucket_height")

                            # LTX-2.3 video clip: item_type=="video" carries a .webm
                            # video_path (never a still image); encode a 5D clip
                            # latent via the LTX video VAE instead of Image.open.
                            if self._temporal_spec() is not None and item.get("item_type") == "video":
                                latent = self._encode_video_clip(item)
                                latent_swap_buffer[image_path] = (latent.cpu(), caption)
                                if self.debug_vram and idx % 50 == 0:
                                    _vramdiag(f"prefill_item_{idx}")
                                if progress_callback and idx % 10 == 0:
                                    progress_callback(
                                        phase="latent_cache",
                                        step=idx,
                                        total=len(buffer_items)
                                    )
                                continue

                            # Load and encode image
                            image = Image.open(image_path)
                            latent = self.encode_image(
                                image=image,
                                target_width=width,
                                target_height=height,
                                bucket_strategy=bucket_strategy
                            )
                            # Store on CPU to save GPU VRAM, keyed by image_path
                            # This eliminates index-based lookup issues with variable batch sizes
                            latent_swap_buffer[image_path] = (
                                latent.cpu(),
                                caption,  # String (CPU memory, minimal overhead)
                            )

                            if self.debug_vram and idx % 50 == 0:
                                _vramdiag(f"prefill_item_{idx}")

                            # Send progress update
                            if progress_callback and idx % 10 == 0:
                                progress_callback(
                                    phase="latent_cache",
                                    step=idx,
                                    total=len(buffer_items)
                                )
                    finally:
                        # Restore the training stack in a finally so an encode failure can
                        # never strand the model / TEs on CPU. Main model always returns to
                        # GPU (training follows immediately); TEs return only if they were
                        # on the GPU before this prefill (swap-TE mode wants them on CPU).
                        self.move_vae_to_cpu()
                        self.move_main_model_to_gpu()
                        self._relocate_main_model_optimizer_state(self.device)
                        if te_on_gpu or te2_on_gpu:
                            self.move_text_encoder_to_gpu()
                            self._relocate_text_encoder_optimizer_state(self.device)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    if self.debug_vram:
                        _vramdiag("after_prefill_restore")
                    next_latent_swap_at_step = latent_encoding_swap_interval
                    print(f"{self.log_prefix} Latent buffer pre-filled with {len(latent_swap_buffer)} latents")

                # onthefly_gpu mode: Ensure text encoders and main model are on GPU for entire epoch
                if text_encoding_mode == "onthefly_gpu":
                    print(f"{self.log_prefix} Moving text encoders to GPU for onthefly_gpu mode...")
                    self.move_text_encoder_to_gpu()
                    # Ensure U-Net is on GPU (critical for mid-epoch resume)
                    self.move_main_model_to_gpu()

                # Ensure the main model is GPU-resident before the first training step
                # of this epoch. The swap_onthefly latent/TE prefills (above) and the
                # onthefly_gpu block each leave it on the GPU, but a pure
                # pre_encoded_cache config runs NONE of those prefills and relies solely
                # on the pre-training VAE-encode phase's *conditional* restore
                # (`if main_on_gpu`). If any prior phase left the main model on CPU — the
                # pre-encode offload when the model entered it already on CPU
                # (main_on_gpu=False, so its finally skips the GPU restore), a step-0
                # verification sample, or a preview request — the epoch loop would
                # otherwise start step 0 with the U-Net/Transformer on CPU while latents
                # are on cuda, crashing in the time-embedding Linear ("mat1 and mat2 must
                # be on the same device: cuda vs cpu"). Guarded to a no-op when the model
                # is already on GPU, and skipped under block swap (which intentionally
                # keeps blocks on CPU and stages them per-forward).
                if self.blocks_to_swap == 0:
                    main_model = self._main_model_module()
                    if main_model is not None and \
                            next(main_model.parameters()).device.type == "cpu":
                        print(f"{self.log_prefix} Staging main model to GPU for training "
                              f"(pre_encoded_cache path had no prefill to stage it)...")
                        self.move_main_model_to_gpu()
                        self._relocate_main_model_optimizer_state(self.device)

                # Training loop
                # Calculate expected steps for this epoch (accounting for MNT and mid-epoch resume)
                epoch_batches = len(batches)  # After mid-epoch resume slicing
                epoch_steps = epoch_batches * multi_noise_timesteps
                epoch_start_step = global_step

                # Update total_steps with actual batch count (first epoch only)
                # This corrects for bucketing overhead (each bucket rounds up batch count)
                # Works for both new training and resumed training.
                # Skip when crop augmentation is active: batch count varies per epoch, so
                # the exact total comes from CropPlanner.step_offsets (set above), not from
                # the first epoch's count.
                # Skip in step-based mode (total_steps is not None): the user requested an
                # exact global-step bound, so actual_total_steps must stay = total_steps.
                # Recomputing it here from epoch*steps_per_epoch would override the requested
                # bound and let training run to an epoch-derived count instead of stopping at
                # total_steps (the loop stop condition at global_step >= actual_total_steps).
                if epoch == start_epoch and self._crop_step_offsets is None and total_steps is None:
                    # Calculate actual steps per epoch (before mid-epoch slicing)
                    if bucket_manager:
                        # For bucketing: use the full batch count before resume slicing.
                        # Priority training path may not define `item_batches`, so use
                        # the pre-sliced `batches` list which is always available here.
                        full_batch_count = len(batches)
                    else:
                        # For simple batching: calculate from total items
                        full_batch_count = (len(all_items) + batch_size - 1) // batch_size

                    if self._rc_active:
                        # Curriculum-aware two-phase accounting: this epoch's len(batches)
                        # counts only the CURRENT phase's partition, and the warmup and
                        # normal partitions can differ (multi-res "max" fit thresholds
                        # shift under scaling; divisibility flooring can merge buckets at
                        # low res). Extrapolating one phase across all epochs would let an
                        # epoch-derived run stop early (or late) mid-normal-phase.
                        _warm_epochs = min(self._rc_switch_epoch, num_epochs)
                        if bucket_manager:
                            if epoch < self._rc_switch_epoch:
                                _warm_count = full_batch_count
                                _norm_count = self._rc_count_batches(
                                    all_items, bucket_manager, self._rc_normal_res, batch_size)
                            else:  # correcting during the normal phase (resume case)
                                _norm_count = full_batch_count
                                _warm_count = self._rc_count_batches(
                                    all_items, bucket_manager, self._rc_warmup_res, batch_size)
                        else:
                            # No-bucketing: batch count is item-count based, phase-invariant.
                            _warm_count = _norm_count = full_batch_count
                        actual_total_steps = (
                            _warm_epochs * _warm_count
                            + (num_epochs - _warm_epochs) * _norm_count
                        ) * multi_noise_timesteps
                        if _warm_count != _norm_count:
                            print(f"{self.log_prefix} [ResCurriculum] Per-phase batch counts: "
                                  f"warmup={_warm_count}, normal={_norm_count} "
                                  f"({_warm_epochs} warmup epoch(s) of {num_epochs})")
                    else:
                        actual_steps_per_epoch = full_batch_count * multi_noise_timesteps
                        actual_total_steps = actual_steps_per_epoch * num_epochs

                    # Update DB if actual differs from initial estimate
                    if actual_total_steps != steps_per_epoch * num_epochs:
                        print(f"{self.log_prefix} Correcting total_steps: {steps_per_epoch * num_epochs} → {actual_total_steps} (bucketing overhead)")
                        if update_total_steps_callback is not None:
                            update_total_steps_callback(actual_total_steps)

                # Vision Encoder VRAM management: if NO batch this epoch uses a reference
                # image (e.g. no VE-reconstruction data in the run), the trained VE would
                # just waste ~186MB on GPU. Offload it to CPU for the epoch; the per-batch
                # encode block reloads it just-in-time should a reference batch appear. This
                # is safe for the optimizer because reference-free batches produce no VE grad
                # (set_to_none), so the step skips the VE params even while on CPU.
                self._ve_idle_batches = 0
                if getattr(self, '_train_vision_encoder', False) and self.vision_encoder is not None:
                    _epoch_has_ref = any(
                        any((_it.get("reference_images") for _it, _ds in _b)) for _b in batches
                    )
                    if not _epoch_has_ref:
                        try:
                            if next(self.vision_encoder.model.parameters()).device.type != "cpu":
                                self._ve_set_device("cpu")
                                torch.cuda.empty_cache()
                                print(f"{self.log_prefix} [VE] Epoch {epoch + 1}: no reference-image "
                                      f"batches - Vision Encoder offloaded to CPU (~186MB freed)")
                        except Exception:
                            pass

                for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs} ({epoch_steps} steps)")):
                    # Drop any partial count a batch that never finished its
                    # backward left behind. The counters that matter are armed
                    # per backward, in _reset_fused_group_counters.
                    self._reset_fused_group_counters()

                    # Vision Encoder VRAM management (mixed epochs): offload the trained VE
                    # after a sustained run of reference-free batches; the encode block
                    # reloads it just-in-time when a reference batch arrives. Hysteresis
                    # (offload only after _VE_OFFLOAD_AFTER idle batches, fire once) avoids
                    # thrashing on interspersed reference batches — important for iter speed.
                    if getattr(self, '_train_vision_encoder', False) and self.vision_encoder is not None:
                        _b_has_ref = any((_it.get("reference_images") for _it, _ds in batch))
                        if _b_has_ref:
                            self._ve_idle_batches = 0
                        else:
                            self._ve_idle_batches = getattr(self, "_ve_idle_batches", 0) + 1
                            if self._ve_idle_batches == 64:  # fire exactly once at the threshold
                                try:
                                    if next(self.vision_encoder.model.parameters()).device.type != "cpu":
                                        self._ve_set_device("cpu")
                                        torch.cuda.empty_cache()
                                        print(f"{self.log_prefix} [VE] 64 reference-free batches - "
                                              f"Vision Encoder offloaded to CPU (~186MB freed; reloads "
                                              f"on the next reference batch)")
                                except Exception:
                                    pass

                    # Aspect-ratio bucketing produces many distinct tensor shapes
                    # (this dataset has ~140 buckets). Each new shape reserves fresh
                    # CUDA blocks the allocator can't reuse for other shapes, so
                    # reserved VRAM grows monotonically across buckets and on Windows
                    # spills into shared memory (catastrophic slowdown). Release cached
                    # blocks when the bucket changes so reserved memory tracks the
                    # current shape rather than the union of every shape seen.
                    # This batch's resolution bucket (image w, h), used both for the
                    # per-shape empty_cache below and to record an un-fittable bucket
                    # if the OOM recovery can't fit even one sample.
                    _cur_bucket_wh = None
                    self._batch_was_unfittable = False
                    try:
                        _bfirst = batch[0][0] if (batch and isinstance(batch[0], tuple)) else (batch[0] if batch else None)
                        if isinstance(_bfirst, dict):
                            _bhw = (
                                _bfirst.get("bucket_width") or _bfirst.get("width"),
                                _bfirst.get("bucket_height") or _bfirst.get("height"),
                            )
                            _cur_bucket_wh = _bhw
                            if _bhw != getattr(self, "_prev_bucket_hw", None):
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                self._prev_bucket_hw = _bhw
                    except Exception:
                        pass

                    # Check for stop flag (user-requested stop from frontend)
                    stop_flag_file = self.output_dir / ".stop_training"
                    if stop_flag_file.exists():
                        print(f"\n{self.log_prefix} Stop flag detected, stopping training...")
                        stop_flag_file.unlink()  # Clean up flag file
                        raise KeyboardInterrupt("Training stopped by user")

                    # Periodically publish Danbooru augmentation metrics for the
                    # UI (read by the /training/runs/{id}/danbooru-metrics endpoint).
                    if getattr(self, "_danbooru_collector", None) is not None and \
                            self._danbooru_metrics_path and batch_idx % 25 == 0:
                        try:
                            import json as _dj
                            with open(self._danbooru_metrics_path, "w", encoding="utf-8") as _mf:
                                _dj.dump(self._danbooru_collector.get_metrics(), _mf, ensure_ascii=False)
                        except Exception:
                            pass

                    # Check for on-demand preview requests from the API
                    # (file-based RPC, see core/training/training_preview_rpc.py).
                    # Each request is processed in-place using the current
                    # in-training model state.  Failures are isolated per
                    # request and reported via the result file — training
                    # never crashes because of a bad request.
                    _preview_ran = False
                    try:
                        from core.training.training_preview_rpc import (
                            list_pending_requests, read_request, cleanup_stale,
                        )
                        _pending = list_pending_requests(self.output_dir)
                        if _pending:
                            _preview_ran = True
                            from core.training.training_inference import TrainingPreviewGenerator
                            if not hasattr(self, "_preview_gen"):
                                self._preview_gen = TrainingPreviewGenerator(self)
                            for _req_path in _pending:
                                _req = read_request(_req_path)
                                # Always delete the request file first so a
                                # malformed / re-emitted request isn't picked
                                # up twice.
                                try: _req_path.unlink()
                                except OSError: pass
                                if _req is None:
                                    continue
                                _rid = _req.get("request_id", "?")
                                _params = _req.get("params", {})
                                print(f"\n{self.log_prefix} Preview request {_rid} - processing...")
                                self._preview_gen.process_request(_rid, _params)
                                print(f"{self.log_prefix} Preview request {_rid} - done")
                            cleanup_stale(str(self.output_dir))
                    except Exception as _pe:   # noqa: BLE001
                        # Never let preview handling kill training
                        print(f"{self.log_prefix} WARNING: preview poll failed: {_pe}")

                    # A preview generation moves the model's components around and offloads
                    # them to CPU when done, breaking the device layout the training loop
                    # relies on (next encode/forward would hit a CPU/GPU mismatch — e.g.
                    # text encoder left on CPU). Restore it before continuing this step.
                    if _preview_ran:
                        try:
                            self.move_main_model_to_gpu()
                            if text_encoding_mode == "onthefly_gpu":
                                self.move_text_encoder_to_gpu()
                            # VAE is re-homed by the per-item onthefly encode guard.
                            print(f"{self.log_prefix} Restored training device layout after preview generation")
                        except Exception as _re:
                            print(f"{self.log_prefix} WARNING: failed to restore device layout after preview: {_re}")

                    # cpu_prefetch path: drain the background worker for this
                    # batch (and let it run ahead while we train). We pull
                    # exactly one batch's worth of embeddings per outer
                    # iteration so the dict size stays bounded.
                    if te_prefetcher is not None and swap_buffer is not None:
                        # Evict the previous batch's entries — they were fully
                        # consumed by the train step we just finished. Without
                        # this, swap_buffer grows monotonically across the epoch
                        # and pins ~1 MB of CPU RAM per image (Anima Qwen3
                        # embeddings) for the entire run.
                        if batch_idx > 0:
                            for prev_entry in batches[batch_idx - 1]:
                                prev_item, _ = prev_entry if isinstance(prev_entry, tuple) else (prev_entry, None)
                                if isinstance(prev_item, dict):
                                    ip = prev_item.get("image_path")
                                    if ip is not None:
                                        swap_buffer.pop(ip, None)
                        # Only top up when the entries for the *current* batch
                        # aren't already in the buffer (the worker writes
                        # ahead in batch order). We pull until we've covered
                        # this batch_idx; usually one pull suffices.
                        while batch_idx >= next_swap_at_step:
                            try:
                                pulled_idx, payload = te_prefetcher.next(timeout=120.0)
                            except Exception as pe:
                                print(f"{self.log_prefix} cpu_prefetch worker timeout / error: {pe}")
                                break
                            if pulled_idx < 0:
                                # Sentinel — worker is done; nothing more to drain.
                                break
                            # Merge into swap_buffer keyed by image_path
                            swap_buffer.update(payload)
                            next_swap_at_step = pulled_idx + 1

                    # Check if we need to refill swap buffer (swap_onthefly path)
                    if swap_buffer is not None and text_encoding_mode == "swap_onthefly" \
                            and batch_idx >= next_swap_at_step:
                        # Calculate next batch range
                        start_idx = next_swap_at_step
                        end_idx = min(start_idx + text_encoding_swap_interval, len(batches))
                        # Use batches (which have bucket info) instead of all_items
                        buffer_items = []
                        for batch in batches[start_idx:end_idx]:
                            buffer_items.extend(batch)

                        print(f"\n{self.log_prefix} Refilling swap buffer (steps {start_idx}-{end_idx})...")
                        if progress_callback:
                            progress_callback(
                                phase="text_encoder_cache",
                                step=0,
                                total=len(buffer_items)
                            )

                        # Move Text Encoder to GPU
                        self.move_text_encoder_to_gpu()
                        # Move main model to CPU
                        self.move_main_model_to_cpu()

                        # Clear old buffer and encode new captions (dict keyed by image_path)
                        swap_buffer.clear()
                        for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding captions", leave=False)):
                            caption = item.get("caption", "")
                            image_path = item["image_path"]
                            embeddings, auxiliary_data = self.encode_caption(
                                caption, requires_grad=False, lyrics=item.get("lyrics", "")
                            )
                            # Store on CPU to save GPU VRAM, keyed by image_path
                            swap_buffer[image_path] = (
                                embeddings.cpu(),
                                self._aux_to_cpu(auxiliary_data),
                                caption,  # String (CPU memory, minimal overhead)
                            )

                            # Send progress update
                            if progress_callback and idx % 10 == 0:
                                progress_callback(
                                    phase="text_encoder_cache",
                                    step=idx,
                                    total=len(buffer_items)
                                )

                        # Move Text Encoder back to CPU
                        self.move_text_encoder_to_cpu()
                        # Move main model to GPU
                        self.move_main_model_to_gpu()

                        # Clear CUDA cache after model movement to free fragmented memory
                        torch.cuda.empty_cache()

                        next_swap_at_step += text_encoding_swap_interval
                        print(f"{self.log_prefix} Buffer refilled with {len(swap_buffer)} embeddings")

                    # Check if we need to refill latent swap buffer
                    if latent_swap_buffer is not None and batch_idx >= next_latent_swap_at_step:
                        # Calculate next batch range
                        start_idx = next_latent_swap_at_step
                        end_idx = min(start_idx + latent_encoding_swap_interval, len(batches))
                        # Use batches (which have bucket info) instead of all_items
                        buffer_items = []
                        for batch in batches[start_idx:end_idx]:
                            buffer_items.extend(batch)

                        print(f"\n{self.log_prefix} Refilling latent swap buffer (steps {start_idx}-{end_idx})...")
                        if progress_callback:
                            progress_callback(
                                phase="latent_cache",
                                step=0,
                                total=len(buffer_items)
                            )

                        # Move VAE to GPU
                        self.move_vae_to_gpu()
                        # Move main model to CPU
                        self.move_main_model_to_cpu()

                        # Clear old buffer and encode new latents (dict keyed by image_path)
                        latent_swap_buffer.clear()
                        corrupted_images = []  # Track corrupted images for logging
                        for idx, (item, dataset) in enumerate(tqdm(buffer_items, desc="Encoding latents", leave=False)):
                            image_path = item["image_path"]
                            caption = item.get("caption", "")
                            width = item.get("width") or item.get("bucket_width")
                            height = item.get("height") or item.get("bucket_height")

                            # Load and encode image with corruption handling
                            try:
                                # LTX-2.3 video clip: encode a 5D clip latent via the
                                # LTX video VAE (Image.open cannot read .webm).
                                if self._temporal_spec() is not None and item.get("item_type") == "video":
                                    latent = self._encode_video_clip(item)
                                    latent_swap_buffer[image_path] = (latent.cpu(), caption)
                                    if progress_callback and idx % 10 == 0:
                                        progress_callback(
                                            phase="latent_cache",
                                            step=idx,
                                            total=len(buffer_items)
                                        )
                                    continue
                                _danb_b = item.get("_danbooru_image_bytes")
                                if _danb_b is not None:
                                    # Online Danbooru sample — decode the in-memory
                                    # bytes lazily here (one at a time, no disk path).
                                    image = Image.open(BytesIO(_danb_b))
                                else:
                                    image = Image.open(image_path)
                                # Force load to detect truncated images early
                                image.load()
                                latent = self.encode_image(
                                    image=image,
                                    target_width=width,
                                    target_height=height,
                                    bucket_strategy=bucket_strategy
                                )
                                # Store on CPU to save GPU VRAM, keyed by image_path
                                latent_swap_buffer[image_path] = (
                                    latent.cpu(),
                                    caption,  # String (CPU memory, minimal overhead)
                                )
                                if _danb_b is not None:
                                    # Free the in-memory Danbooru bytes immediately
                                    # once its latent is buffered (incremental
                                    # cleanup — keeps CPU RAM bounded).
                                    item["_danbooru_image_bytes"] = None
                            except Exception as img_error:
                                # Log corrupted image and skip it
                                corrupted_images.append(image_path)
                                print(f"{self.log_prefix} [CORRUPTED IMAGE] Skipping: {image_path}")
                                print(f"{self.log_prefix} [CORRUPTED IMAGE] Error: {str(img_error)[:200]}")
                                continue

                            # Send progress update
                            if progress_callback and idx % 10 == 0:
                                progress_callback(
                                    phase="latent_cache",
                                    step=idx,
                                    total=len(buffer_items)
                                )

                        # Log summary of corrupted images
                        if corrupted_images:
                            print(f"{self.log_prefix} [CORRUPTED IMAGES] Total skipped: {len(corrupted_images)}")
                            for path in corrupted_images:
                                print(f"{self.log_prefix} [CORRUPTED IMAGES]   - {path}")

                        # Move VAE back to CPU
                        self.move_vae_to_cpu()
                        # Move main model to GPU
                        self.move_main_model_to_gpu()

                        # Clear CUDA cache after model movement to free fragmented memory
                        torch.cuda.empty_cache()

                        next_latent_swap_at_step += latent_encoding_swap_interval
                        print(f"{self.log_prefix} Latent buffer refilled with {len(latent_swap_buffer)} latents")

                    # ============================================================
                    # Batch data preparation (ONCE per batch, OUTSIDE MNT loop)
                    # ============================================================
                    # IMPORTANT: Prepare batch tensors once and reuse across MNT iterations
                    # This prevents redundant CPU->GPU transfers and reduces VRAM fragmentation

                    latents_list = []
                    text_embeddings_list = []
                    sensenova_prefixes = []
                    auxiliary_data_list = []  # Unified: attention_mask (Z-Image), pooled_embeddings (SDXL), or None (SD1.5)
                    reference_latents_list = []  # FLUX.2 reference image conditioning
                    condition_images_list = []  # ControlNet condition images [B, 3, H, W]
                    loss_weight_maps_list = []  # Outpaint-mode per-item latent-space loss weight [1,1,H/8,W/8] or None (parallel to condition_images_list)
                    repa_pixels_list = []  # REPA clean-image S x S [-1,1] tensors (MiniT2I, parallel to latents_list)
                    _repa_active = bool(getattr(self, "repa_enable", False)) and self.is_minit2i
                    # SDXL micro-conditioning: per-item (orig_h,orig_w,crop_top,crop_left,
                    # target_h,target_w) for time_ids, parallel to latents_list.
                    micro_cond_list = []
                    _sdxl_microcond_active = self.is_sdxl and bool(self.config.get("sdxl_micro_conditioning", True))
                    self._last_micro_cond = None

                    # Flag to track if batch should be skipped due to corrupted image
                    batch_has_corrupted_image = False
                    corrupted_image_path = None

                    for item, dataset in batch:
                        # BucketManager stores bucket_width/bucket_height, not width/height
                        width = item.get("width") or item.get("bucket_width")
                        height = item.get("height") or item.get("bucket_height")
                        image_path = item["image_path"]

                        # Load latent (mode-specific)
                        if latent_encoding_mode == "swap_onthefly":
                            # Get from swap buffer using image_path as key (dict lookup)
                            # This eliminates index-based alignment issues
                            if image_path in latent_swap_buffer:
                                latent_cpu, buffer_caption = latent_swap_buffer[image_path]
                                # Transfer to GPU
                                latent = latent_cpu.to(self.device, non_blocking=True)
                                latents_list.append(latent)
                                # Update caption from buffer (ensures correct pairing)
                                item["caption"] = buffer_caption
                            else:
                                # Fallback to on-the-fly encoding (image not in buffer)
                                # This happens when buffer hasn't been refilled yet for this batch
                                # or when image was skipped during buffer refill (corrupted)
                                print(f"{self.log_prefix} WARNING: Image not in latent swap buffer, encoding on-the-fly: {image_path}")
                                try:
                                    self.move_vae_to_gpu()
                                    # LTX-2.3 video clip: encode a 5D clip latent via
                                    # the LTX video VAE (Image.open cannot read .webm).
                                    if self._temporal_spec() is not None and item.get("item_type") == "video":
                                        latent = self._encode_video_clip(item)
                                        latent = latent.to(self.device)
                                        latents_list.append(latent)
                                        self.move_vae_to_cpu()
                                    else:
                                        _danb_b = item.get("_danbooru_image_bytes")
                                        if _danb_b is not None:
                                            image = Image.open(BytesIO(_danb_b))
                                        else:
                                            image = Image.open(image_path)
                                        image.load()  # Force load to detect truncated images
                                        latent = self.encode_image(
                                            image=image,
                                            target_width=width,
                                            target_height=height,
                                            bucket_strategy=bucket_strategy
                                        )
                                        # Ensure latent is on training device
                                        latent = latent.to(self.device)
                                        latents_list.append(latent)
                                        self.move_vae_to_cpu()
                                        if _danb_b is not None:
                                            item["_danbooru_image_bytes"] = None
                                except Exception as img_error:
                                    # Corrupted image - log and skip entire batch
                                    print(f"{self.log_prefix} [CORRUPTED IMAGE] Batch skipped due to: {image_path}")
                                    print(f"{self.log_prefix} [CORRUPTED IMAGE] Error: {str(img_error)[:200]}")
                                    # Set flag to skip this batch
                                    batch_has_corrupted_image = True
                                    corrupted_image_path = image_path
                                    break

                        elif latent_encoding_mode == "pre_encoded_cache":
                            cache = latent_caches[dataset.unique_id]

                            # ACE-Step audio-clip item: [1, T, 64] temporal-only latent,
                            # no width/height/bucket concept at all (see module docstring
                            # / audio_loader.py) -- keyed by (audio_path, clip_seconds,
                            # sample_rate) via compute_audio_hash, NOT compute_image_hash,
                            # so it must go through load_audio_latent/save_audio_latent,
                            # never the generic load_latent(width, height) below (which
                            # would silently miss on a different hash scheme and fall
                            # into _regenerate_single_latent's Image.open/encode_image --
                            # NotImplementedError for this audio-only arch, and the
                            # width/height-keyed shape-validation chain below would also
                            # crash on None//int). Mirrors the LTX-2.3 video-clip
                            # special-case elsewhere in this same loop
                            # (_encode_video_clip), except audio clips ARE
                            # disk-cached (no random per-step re-crop -- audio_loader.py's
                            # docstring: clips are taken from a fixed START offset, so a
                            # cache hit is always valid, unlike video's intentional
                            # random window). Skips straight to latents_list.append() --
                            # the caption/text-embedding encode below this if/elif chain
                            # still runs normally for this item (no `continue`).
                            if self.is_acestep and item.get("item_type") == "audio":
                                latent = self._load_or_regenerate_acestep_audio_latent(item, cache)
                                latents_list.append(latent)
                            elif self._temporal_spec() is not None and item.get("item_type") == "video":
                                # Video clip: keyed by WINDOW (compute_clip_hash),
                                # not by (path, w, h). The generic load_latent
                                # below cannot address it and would miss into
                                # _regenerate_single_latent -> PIL.Image.open on a
                                # .webm/.mp4. Same seam the pre-encode pass wrote
                                # with, so this is a hit.
                                latents_list.append(self._load_or_encode_video_clip(item, cache))
                            else:
                                # Load from disk cache
                                latent = cache.load_latent(item["image_path"], width, height)

                                # On-the-fly regeneration if cache is corrupted or incompatible
                                if latent is None:
                                    print(f"{self.log_prefix} WARNING: Latent cache miss or corrupted for {item['image_path']}, regenerating...")
                                    latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)

                                # Validate latent shape.
                                # Lens / Ideogram4 latents are [1, N, 128] (3D flat sequence); skip the 4D check.
                                if self.is_minit2i and getattr(self, "minit2i_latent", False):
                                    # Latent-space: VAE latent [1, C, H/vsf, W/vsf].
                                    vsf = getattr(self, "minit2i_vae_scale_factor", 8)
                                    eh, ew = height // vsf, width // vsf
                                    if latent.ndim != 4 or latent.shape[2] != eh or latent.shape[3] != ew:
                                        print(f"{self.log_prefix} WARNING: MiniT2I latent shape mismatch for {item['image_path']}")
                                        print(f"{self.log_prefix}   Expected: [1, C, {eh}, {ew}]  Got: {list(latent.shape)}")
                                        print(f"{self.log_prefix}   Regenerating latent...")
                                        latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)
                                elif self.is_minit2i:
                                    # Pixel-space: "latent" is the [-1,1] RGB image [1, 3, H, W] (full res, no VAE downscale).
                                    if (latent.ndim != 4 or latent.shape[1] != 3
                                            or latent.shape[2] != height or latent.shape[3] != width):
                                        print(f"{self.log_prefix} WARNING: MiniT2I pixel-latent shape mismatch for {item['image_path']}")
                                        print(f"{self.log_prefix}   Expected: [1, 3, {height}, {width}]  Got: {list(latent.shape)}")
                                        print(f"{self.log_prefix}   Regenerating latent...")
                                        latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)
                                elif self.is_krea2:
                                    # Krea 2: packed latent [1, (H//16)*(W//16), 64].
                                    expected_seq_len = (height // 16) * (width // 16)
                                    if latent.ndim != 3 or latent.shape[1] != expected_seq_len or latent.shape[2] != 64:
                                        print(f"{self.log_prefix} WARNING: Krea 2 latent shape mismatch for {item['image_path']}")
                                        print(f"{self.log_prefix}   Expected: [1, {expected_seq_len}, 64]  Got: {list(latent.shape)}")
                                        print(f"{self.log_prefix}   Regenerating latent...")
                                        latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)
                                elif latent.ndim == 5:
                                    # A VIDEO arch's STILL: [1, C, 1, H/vsf, W/vsf].
                                    # LTX-2.3 and MiniMax-H3 route a still through the
                                    # SAME 5D train_step as a clip (T=1), so its cached
                                    # latent has a TEMPORAL axis at index 2 -- which the
                                    # 4D check below compares against `height // 8`,
                                    # mismatching every single time (1 != 48 at 384) and
                                    # sending a perfectly good latent into
                                    # `_regenerate_single_latent` on the first batch.
                                    if not self._still_latent_5d_is_valid(latent, width, height):
                                        print(f"{self.log_prefix} WARNING: 5D still-latent shape mismatch for {item['image_path']}")
                                        print(f"{self.log_prefix}   Expected: [1, C, 1, {height}/vsf, {width}/vsf]  Got: {list(latent.shape)}")
                                        print(f"{self.log_prefix}   Regenerating latent...")
                                        latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)
                                elif not (self.is_lens or self.is_ideogram4):
                                    expected_latent_height = height // 8
                                    expected_latent_width = width // 8
                                    if latent.shape[2] != expected_latent_height or latent.shape[3] != expected_latent_width:
                                        print(f"{self.log_prefix} WARNING: Latent shape mismatch for {item['image_path']}")
                                        print(f"{self.log_prefix}   Expected: [1, {self.vae_latent_channels}, {expected_latent_height}, {expected_latent_width}]")
                                        print(f"{self.log_prefix}   Got: {list(latent.shape)}")
                                        print(f"{self.log_prefix}   Regenerating latent...")
                                        latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)
                                else:
                                    # Lens: [1, latent_h*latent_w, 128]
                                    expected_seq_len = (height // 16) * (width // 16)
                                    if latent.ndim != 3 or latent.shape[1] != expected_seq_len or latent.shape[2] != 128:
                                        print(f"{self.log_prefix} WARNING: Lens latent shape mismatch for {item['image_path']}")
                                        print(f"{self.log_prefix}   Expected: [1, {expected_seq_len}, 128]  Got: {list(latent.shape)}")
                                        print(f"{self.log_prefix}   Regenerating latent...")
                                        latent = self._regenerate_single_latent(item["image_path"], width, height, cache, latent_caches)

                                latents_list.append(latent)

                        elif latent_encoding_mode == "onthefly_gpu":
                            # Encode on GPU without cache. Ensure the VAE is on GPU first:
                            # encode_image runs the VAE on its current device and does NOT
                            # move it, so after a (step-0 or mid-epoch) sample generation
                            # moved the VAE to CPU, the encode would silently run on CPU
                            # (GPU idle, minutes-long stall). .to(cuda) is a no-op when the
                            # VAE is already on GPU, so the per-item guard is cheap.
                            self.move_vae_to_gpu()
                            try:
                                # LTX-2.3 video clip: encode a 5D clip latent via the
                                # LTX video VAE (Image.open cannot read .webm).
                                if self._temporal_spec() is not None and item.get("item_type") == "video":
                                    latents_list.append(self._encode_video_clip(item))
                                else:
                                    _danb_b = item.get("_danbooru_image_bytes")
                                    if _danb_b is not None:
                                        image = Image.open(BytesIO(_danb_b))
                                    else:
                                        image = Image.open(item["image_path"])
                                    image.load()  # Force load to detect truncated images
                                    # Crop augmentation: use the per-(item,epoch) crop_box +
                                    # kohya time_ids from the planner (pixel <-> time_ids
                                    # consistency for both full and cropped cases).
                                    _spec = item.get("_crop_spec")
                                    latent = self.encode_image(
                                        image=image,
                                        target_width=width,
                                        target_height=height,
                                        bucket_strategy=bucket_strategy,
                                        crop_box=_spec.crop_box if _spec is not None else None,
                                        time_ids_override=_spec.time_ids if _spec is not None else None,
                                    )
                                    latents_list.append(latent)
                                    if _danb_b is not None:
                                        item["_danbooru_image_bytes"] = None
                            except Exception as img_error:
                                # Corrupted image - log and skip entire batch
                                print(f"{self.log_prefix} [CORRUPTED IMAGE] Batch skipped due to: {item['image_path']}")
                                print(f"{self.log_prefix} [CORRUPTED IMAGE] Error: {str(img_error)[:200]}")
                                batch_has_corrupted_image = True
                                corrupted_image_path = item["image_path"]
                                break

                        # REPA: clean-image pixels for this item (parallel to latents_list).
                        # A latent was appended above for this item (corrupted items break
                        # earlier), so this keeps 1:1 alignment. None -> REPA skipped for batch.
                        if _repa_active:
                            repa_pixels_list.append(self._get_repa_pixels_for_item(item))

                        # SDXL micro-conditioning per item: prefer the exact values
                        # captured by encode_image (onthefly path; exact even for
                        # random_crop), else recompute deterministically from the real
                        # original size + bucket + strategy (swap/cache paths). Reset the
                        # capture so a non-encoding item never reuses a prior item's value.
                        if _sdxl_microcond_active:
                            cap = self._last_micro_cond
                            self._last_micro_cond = None
                            if cap is None:
                                cap = self._recompute_sdxl_micro_cond(item, width, height, bucket_strategy)
                            micro_cond_list.append(cap)

                        # Encode caption (mode-specific, architecture-unified)
                        caption = item.get("caption", "")

                        if self.is_sensenova:
                            # References enter through the PROMPT PREFIX here, not
                            # through encode_image: sensenova_ops owns their loading
                            # and ImageNet normalization so the trainer's bucket /
                            # [-1,1] pipeline can never touch them.
                            # requires_grad follows train_text_encoder: SenseNova's
                            # prompt encoder IS the understanding branch of the same
                            # LLM that denoises, so a trainable "text encoder" means
                            # a differentiable prefix pass.
                            prefix, _ = self.encode_caption(
                                caption,
                                requires_grad=bool(getattr(self, "train_text_encoder", False)),
                                reference_image_paths=(
                                    item.get("reference_images") or []
                                ) if use_reference_images else None,
                            )
                            sensenova_prefixes.append(prefix)
                        elif text_encoding_mode in ("swap_onthefly", "cpu_prefetch"):
                            # Get from swap buffer using image_path as key (dict lookup).
                            # Both swap_onthefly and cpu_prefetch share this consumer:
                            # cpu_prefetch's daemon worker fills swap_buffer ahead of
                            # time via te_prefetcher.next(); swap_onthefly refills it
                            # synchronously in a separate branch above.
                            if image_path in swap_buffer:
                                embeddings_cpu, auxiliary_cpu, buffer_caption = swap_buffer[image_path]
                                # Transfer to GPU
                                embeddings = embeddings_cpu.to(self.device, non_blocking=True)
                                auxiliary = self._aux_to_device(auxiliary_cpu)
                                text_embeddings_list.append(embeddings)
                                auxiliary_data_list.append(auxiliary)
                                # Override caption from buffer (correct pairing)
                                caption = buffer_caption
                            else:
                                # Fallback to on-the-fly encoding (image not in buffer)
                                print(f"{self.log_prefix} WARNING: Image not in text swap buffer, encoding on-the-fly: {image_path}")
                                embeddings, auxiliary = self.encode_caption(
                                    caption, requires_grad=True, lyrics=item.get("lyrics", "")
                                )
                                text_embeddings_list.append(embeddings)
                                auxiliary_data_list.append(auxiliary)

                        elif text_encoding_mode == "pre_encoded_cache":
                            # Load from disk cache (per-dataset)
                            cached_result = self._load_caption_embedding_from_disk(
                                caption=caption,
                                dataset_unique_id=dataset.unique_id,
                                text_encoder_caches=text_encoder_caches,
                                lyrics=item.get("lyrics", ""),
                            )
                            if cached_result is not None:
                                embeddings_cpu, auxiliary_cpu = cached_result
                                embeddings = embeddings_cpu.to(self.device, non_blocking=True)
                                auxiliary = self._aux_to_device(auxiliary_cpu)
                                text_embeddings_list.append(embeddings)
                                auxiliary_data_list.append(auxiliary)
                            else:
                                # Cache miss in pre_encoded_cache mode. This should
                                # NOT happen: the cache-generation phase caches every
                                # caption (including the empty string). A miss here
                                # therefore signals an incomplete/stale cache -> warn
                                # loudly. Critically, pre_encoded mode offloads the
                                # Text Encoder to CPU after the cache phase, so a bare
                                # encode_caption() would crash with a device mismatch
                                # (params on CPU, inputs on GPU). Stage the TE to GPU
                                # on demand, encode, then move it back — a slow but
                                # correct recovery instead of a hard crash. requires_grad
                                # is False to match the cache-hit branch (pre_encoded
                                # mode keeps the TE frozen / grad-free).
                                print(f"{self.log_prefix} WARNING: Caption not in pre-encoded cache "
                                      f"(incomplete cache); staging Text Encoder to GPU for a "
                                      f"one-off on-the-fly encode: '{caption[:30]}...'")
                                self.move_text_encoder_to_gpu()
                                try:
                                    embeddings, auxiliary = self.encode_caption(
                                        caption, requires_grad=False, lyrics=item.get("lyrics", "")
                                    )
                                finally:
                                    self.move_text_encoder_to_cpu()
                                text_embeddings_list.append(embeddings)
                                auxiliary_data_list.append(auxiliary)

                        elif text_encoding_mode == "onthefly_gpu":
                            # Encode on GPU without cache
                            embeddings, auxiliary = self.encode_caption(
                                caption, requires_grad=True, lyrics=item.get("lyrics", "")
                            )
                            text_embeddings_list.append(embeddings)
                            auxiliary_data_list.append(auxiliary)

                        # ============================================================
                        # Reference Image Latent Encoding (FLUX.2 only)
                        # ============================================================
                        # Note: Only items WITH reference images are conditioned.
                        # If an item has no reference images, we append None to maintain list alignment.
                        # Later, if ANY item in batch has no reference, we skip conditioning for entire batch.
                        #
                        # Multiple reference images per item:
                        # - Each item can have multiple reference images (up to 10)
                        # - reference_latents_list contains List[List[Tensor]] or List[None]
                        # - Each inner list has latents for that item's reference images
                        # - train_step applies T=10, 20, 30... to each reference image
                        #
                        # is_flux2, NOT "any arch with references": this branch is
                        # VAE-latent conditioning at the target's bucket size.
                        # SenseNova's references are ViT tokens in the prompt prefix
                        # (built in its encode_caption branch above) and must never
                        # reach encode_image -- see ops/sensenova_ops.
                        if use_reference_images and self.is_flux2:
                            reference_images = item.get("reference_images", [])
                            if reference_images:
                                # Encode all reference images for this item (max 10)
                                item_ref_latents = []
                                for ref_idx, ref_image_path in enumerate(reference_images[:10]):
                                    try:
                                        ref_image = Image.open(ref_image_path)
                                        # Use same bucket dimensions as target image
                                        ref_latent = self.encode_image(
                                            image=ref_image,
                                            target_width=width,
                                            target_height=height,
                                            bucket_strategy=bucket_strategy
                                        )
                                        item_ref_latents.append(ref_latent.to(self.device))
                                    except Exception as e:
                                        print(f"{self.log_prefix} WARNING: Failed to encode reference image {ref_image_path}: {e}")
                                        # Skip this reference image, continue with others
                                        continue

                                if item_ref_latents:
                                    # Successfully encoded at least one reference image
                                    reference_latents_list.append(item_ref_latents)
                                else:
                                    # All reference images failed - mark as None
                                    reference_latents_list.append(None)
                            else:
                                # No reference images for this item - mark as None
                                reference_latents_list.append(None)

                        # ControlNet: Load condition images from reference_images[0]
                        # Condition images stay in pixel space [0, 1] (not VAE-encoded)
                        use_condition_images = getattr(self, 'use_condition_images', False)
                        if use_condition_images:
                            if getattr(self, '_is_outpaint_mode', False):
                                # Outpaint-native conditioning (PART B): self-supervised
                                # crop->full. Build the conditioning from the item's OWN
                                # image (no paired reference_images dataset) -- reference
                                # images are ignored entirely in this mode.
                                try:
                                    from core.utils.crop_mask_condition import build_crop_mask_condition
                                    # Danbooru-injected items (fake "danbooru://" image_path,
                                    # no real file on disk) carry their pixels in
                                    # _danbooru_image_bytes instead -- same convention as the
                                    # latent-load branch above. NOTE: onthefly_gpu latent
                                    # loading nulls this after its own use (same item, earlier
                                    # in this per-item loop), so it is only available here for
                                    # modes that don't consume it first.
                                    _danb_b = item.get("_danbooru_image_bytes")
                                    if _danb_b is not None:
                                        full_image = flatten_to_rgb(Image.open(BytesIO(_danb_b))).resize(
                                            (width, height), Image.LANCZOS
                                        )
                                    else:
                                        full_image = flatten_to_rgb(Image.open(image_path)).resize(
                                            (width, height), Image.LANCZOS
                                        )
                                    full_np = np.array(full_image)
                                    _op_planner = self.get_outpaint_planner()
                                    rect = _op_planner.rect_for(epoch, image_path, width, height)
                                    # R1 (scratchpad/outpaint_boundary_structure_fix.md D3-R1):
                                    # per-sample randomized edge softness, drawn from an RNG
                                    # stream independent of rect_for's (see feather_for's
                                    # docstring). 0.0/0.0 range (default) -> always 0.0 -> the
                                    # razor-sharp default path, byte-identical to before R1.
                                    _edge_feather_px = _op_planner.feather_for(epoch, image_path)
                                    cond_np, gate_np = build_crop_mask_condition(
                                        full_np, rect, (width, height), edge_feather_px=_edge_feather_px
                                    )
                                    if not self.outpaint_mask_channel:
                                        cond_np = cond_np[:, :, :3]
                                    cond_tensor = torch.from_numpy(cond_np).permute(2, 0, 1).unsqueeze(0).float()  # [1, C, H, W]

                                    # Loss weight map (latent space [1,1,H/8,W/8]): the
                                    # KNOWN region (inside rect, gate==0) is down-weighted
                                    # to outpaint_known_loss_weight; the GENERATE region
                                    # (gate==1) stays at full weight 1.0. Derived from the
                                    # same gate the conditioning was built from (pixel-exact
                                    # agreement), then downsampled with the codebase's
                                    # standard mask->latent convention (nearest, /8) --
                                    # see custom_sampling.py's inpaint mask_latent build.
                                    # Computed BEFORE the two list appends so an exception
                                    # here can never desync condition_images_list from
                                    # loss_weight_maps_list (both append together, below).
                                    _F = torch.nn.functional
                                    known_w = float(self.outpaint_known_loss_weight)
                                    gate_t = torch.from_numpy(gate_np).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
                                    weight_t = known_w + (1.0 - known_w) * gate_t
                                    weight_latent = _F.interpolate(
                                        weight_t, size=(height // 8, width // 8), mode="nearest"
                                    )
                                    # Optional seam-ring boost: add extra loss weight on the
                                    # GENERATE-side latent cells immediately adjacent to the
                                    # known region (the boundary the model must render a
                                    # coherent continuation across). 0 (default) = no boost.
                                    seam_boost = float(self.outpaint_seam_loss_boost)
                                    if seam_boost > 0.0:
                                        known_latent = _F.interpolate(
                                            (1.0 - gate_t), size=(height // 8, width // 8), mode="nearest"
                                        )  # 1 on known, 0 on generate (latent res, binary)
                                        dil1 = _F.max_pool2d(known_latent, kernel_size=3, stride=1, padding=1)
                                        ring1 = (dil1 > 0.5).float() * (known_latent <= 0.5).float()  # generate cells touching known
                                        weight_latent = weight_latent + seam_boost * ring1
                                        # Optional 2nd ring (outpaint_seam_ring_width=2, default 1 =
                                        # byte-identical to the single-ring behavior above): one more
                                        # dilation step outward (k=5), weighted at HALF the boost
                                        # increment -- the seam physics (visible discontinuity +
                                        # VAE encode-bleed floor) spans roughly the first 1-2 latent
                                        # cells, not a hard cliff at cell 1.
                                        if int(getattr(self, "outpaint_seam_ring_width", 1)) >= 2:
                                            dil2 = _F.max_pool2d(known_latent, kernel_size=5, stride=1, padding=2)
                                            ring2 = (dil2 > 0.5).float() * (known_latent <= 0.5).float() * (1.0 - ring1)
                                            weight_latent = weight_latent + (seam_boost * 0.5) * ring2

                                    # Both appends adjacent -> lists stay length-aligned.
                                    condition_images_list.append(cond_tensor)
                                    loss_weight_maps_list.append(weight_latent)
                                except Exception as e:
                                    print(f"{self.log_prefix} WARNING: Failed to build outpaint condition for {image_path}: {e}")
                                    condition_images_list.append(None)
                                    loss_weight_maps_list.append(None)
                            else:
                                reference_images = item.get("reference_images", [])
                                if reference_images:
                                    try:
                                        # Use first reference image only
                                        cond_image = flatten_to_rgb(Image.open(reference_images[0]))
                                        # Resize to match target dimensions
                                        cond_image = cond_image.resize((width, height), Image.LANCZOS)
                                        # Convert to tensor [0, 1] range: [1, 3, H, W]
                                        import torchvision.transforms.functional as TF
                                        cond_tensor = TF.to_tensor(cond_image).unsqueeze(0)  # [1, 3, H, W]
                                        condition_images_list.append(cond_tensor)
                                        loss_weight_maps_list.append(None)
                                    except Exception as e:
                                        print(f"{self.log_prefix} WARNING: Failed to load condition image {reference_images[0]}: {e}")
                                        condition_images_list.append(None)
                                        loss_weight_maps_list.append(None)
                                else:
                                    # No reference image - mark as None (will skip this item)
                                    condition_images_list.append(None)
                                    loss_weight_maps_list.append(None)

                    # Skip batch if corrupted image was detected
                    if batch_has_corrupted_image:
                        print(f"{self.log_prefix} [CORRUPTED IMAGE] Skipping batch due to corrupted image: {corrupted_image_path}")
                        # Cleanup partial lists
                        del latents_list, text_embeddings_list, auxiliary_data_list
                        if reference_latents_list:
                            del reference_latents_list
                        if condition_images_list:
                            del condition_images_list
                        if loss_weight_maps_list:
                            del loss_weight_maps_list
                        # Update global_step for skipped batch (to maintain step counting)
                        # Each batch would have processed multi_noise_timesteps steps
                        global_step += multi_noise_timesteps
                        self._batches_skipped += 1
                        continue

                    # Stack batch with size validation
                    # Filter out latents with mismatched spatial dimensions (rare edge case)
                    if len(latents_list) > 1:
                        # Get expected shape from first latent
                        expected_shape = latents_list[0].shape[2:]  # (H, W)
                        valid_indices = []
                        for idx, lat in enumerate(latents_list):
                            if lat.shape[2:] == expected_shape:
                                valid_indices.append(idx)
                            else:
                                print(f"{self.log_prefix} WARNING: Latent size mismatch in batch - expected {expected_shape}, got {lat.shape[2:]}, skipping item")

                        if len(valid_indices) < len(latents_list):
                            # Filter lists to keep only valid items
                            latents_list = [latents_list[i] for i in valid_indices]
                            text_embeddings_list = [text_embeddings_list[i] for i in valid_indices]
                            auxiliary_data_list = [auxiliary_data_list[i] for i in valid_indices]
                            if reference_latents_list:
                                reference_latents_list = [reference_latents_list[i] for i in valid_indices]
                            if condition_images_list:
                                condition_images_list = [condition_images_list[i] for i in valid_indices]
                            if loss_weight_maps_list:
                                loss_weight_maps_list = [loss_weight_maps_list[i] for i in valid_indices]
                            if repa_pixels_list:
                                repa_pixels_list = [repa_pixels_list[i] for i in valid_indices]
                            if micro_cond_list:
                                micro_cond_list = [micro_cond_list[i] for i in valid_indices]

                    # Skip batch if no valid latents remain
                    if len(latents_list) == 0:
                        print(f"{self.log_prefix} WARNING: No valid latents in batch, skipping")
                        self._batches_skipped += 1
                        continue

                    # Create batch tensors (ONCE, reused across MNT iterations)
                    latents = torch.cat(latents_list, dim=0)

                    sensenova_prefix = None
                    if self.is_sensenova:
                        sensenova_prefix = self._collate_sensenova_b1_prefix(
                            sensenova_prefixes
                        )

                    # onthefly_gpu latent encoding keeps the VAE on GPU per batch (see the
                    # encode branch). The VAE is not needed during the forward/backward, so
                    # offload it to CPU now to free its VRAM during the peak; the next batch's
                    # encode moves it back. No empty_cache(): the freed block stays in the
                    # allocator cache and is reused by the train step (calling empty_cache
                    # every batch would hurt throughput).
                    if latent_encoding_mode == "onthefly_gpu" and self.vae is not None:
                        self.vae.to(device="cpu", dtype=self.vae_dtype)

                    # REPA clean-image batch [B,3,S,S] (CPU). Requires a pixel for every
                    # surviving item; if any failed to load, skip REPA for this batch.
                    repa_pixels_batch = None
                    if _repa_active and repa_pixels_list and len(repa_pixels_list) == len(latents_list) \
                            and all(rp is not None for rp in repa_pixels_list):
                        repa_pixels_batch = torch.cat(repa_pixels_list, dim=0)

                    # SDXL micro-conditioning batch [B,6] (CPU float). Per-item time_ids
                    # (orig_h,orig_w,crop_top,crop_left,target_h,target_w). None disables
                    # it for the batch -> train_step falls back to the legacy values.
                    time_ids_batch = None
                    if _sdxl_microcond_active and micro_cond_list and len(micro_cond_list) == len(latents_list) \
                            and all(mc is not None for mc in micro_cond_list):
                        time_ids_batch = torch.tensor(micro_cond_list, dtype=torch.float32)

                    # Text embeddings are [1, seq_len, dim], use cat to get [batch_size, seq_len, dim]
                    # IMPORTANT: Pad embeddings to same sequence length if chunking is used
                    if text_embeddings_list:
                        # Per-item encoders (Anima/Z-Image) drop the batch dim and
                        # return 2D [L, D]; the collation below assumes 3D [1, L, D]
                        # (it reads shape[1] as seq and cats on dim 0). Without this
                        # normalization, cat(dim=0) at batch_size>=2 collapses the
                        # batch into the sequence axis, yielding a 2D context that the
                        # Anima LLM-Adapter mis-reshapes (head_dim ends up 16 vs rope
                        # 64 -> RuntimeError at _adapter_apply_rotary_pos_emb).
                        text_embeddings_list = [
                            emb.unsqueeze(0) if emb.dim() == 2 else emb
                            for emb in text_embeddings_list
                        ]
                        # Check if all embeddings have same sequence length
                        seq_lengths = [emb.shape[1] for emb in text_embeddings_list]
                        max_seq_len = max(seq_lengths)

                        if len(set(seq_lengths)) > 1:
                            # Different sequence lengths - need padding
                            padded_embeddings = []
                            for emb in text_embeddings_list:
                                if emb.shape[1] < max_seq_len:
                                    # Pad to max_seq_len with zeros
                                    pad_length = max_seq_len - emb.shape[1]
                                    padding = torch.zeros(
                                        (emb.shape[0], pad_length, emb.shape[2]),
                                        dtype=emb.dtype,
                                        device=emb.device
                                    )
                                    emb = torch.cat([emb, padding], dim=1)
                                padded_embeddings.append(emb)
                            text_embeddings = torch.cat(padded_embeddings, dim=0)  # [batch, seq_len, dim]
                        else:
                            # All same length - direct concatenation
                            text_embeddings = torch.cat(text_embeddings_list, dim=0)  # [batch, seq_len, dim]
                    else:
                        text_embeddings = None

                    # Prepare auxiliary data (attention_mask for Z-Image, pooled_embeddings for SDXL)
                    # These are also reused across MNT iterations
                    attention_mask = None
                    pooled_embeddings = None
                    if self.is_zimage or self.is_lens or self.is_ideogram4 or self.is_minit2i or self.is_krea2:
                        attention_mask = torch.stack([aux for aux in auxiliary_data_list if aux is not None], dim=0)
                    elif self.is_anima:
                        # Anima aux is a per-item dict {source_mask, t5_input_ids,
                        # t5_attn_mask}; collate into one dict of batched [B, L]
                        # tensors carried through attention_mask (the anima
                        # train-step path reads the dict from mnt_attention_mask).
                        attention_mask = self.arch.collate_aux(self, auxiliary_data_list)
                    elif self.is_ltx2:
                        # LTX-2.3 aux is a per-item dict {audio_text_embedding,
                        # mask}; collate into one dict carried through
                        # attention_mask (train_step_ltx2 reads it). fps is a
                        # per-CLIP property (not per-caption), so inject a
                        # per-sample fps tensor [B] from the batch items here.
                        attention_mask = self.arch.collate_aux(self, auxiliary_data_list)
                        attention_mask["fps"] = self._ltx2_batch_fps_tensor(batch)
                    elif self.is_minimax_h3:
                        # MiniMax-H3 aux is a per-item dict {num_text_tokens};
                        # the WINDOW's audio latent is a property of the sampled
                        # CLIP, not of the caption, so it is injected here from
                        # the batch items (mirrors ltx2's per-clip fps).
                        attention_mask = self.arch.collate_aux(self, auxiliary_data_list)
                        attention_mask.update(self._minimax_h3_batch_audio(batch))
                    elif self.is_acestep:
                        # ACE-Step aux is a per-item dict {text_attention_mask};
                        # collate into one dict carried through attention_mask
                        # (train_step_acestep reads it).
                        attention_mask = self.arch.collate_aux(self, auxiliary_data_list)
                    elif self.is_sdxl and any(aux is not None for aux in auxiliary_data_list):
                        pooled_embeddings = torch.cat([aux for aux in auxiliary_data_list if aux is not None], dim=0)

                    # Prepare reference latents for FLUX.2 conditioning
                    # Only apply conditioning if ALL items in batch have valid reference latents
                    # reference_latents_list is now List[List[Tensor]] or List[None]
                    # We pass the nested structure to train_step which handles T coordinates
                    # SenseNova has no entry here: its references are already inside
                    # each item's prompt prefix.
                    reference_latents_nested = None
                    if use_reference_images and self.is_flux2 and reference_latents_list:
                        # Check if any item is missing reference latent (None)
                        if all(lat is not None for lat in reference_latents_list):
                            # Pass nested list structure to train_step
                            # train_step will apply T=10, 20, 30... per reference image
                            reference_latents_nested = reference_latents_list
                        else:
                            # Mixed batch (some with, some without reference) - skip conditioning
                            # This ensures consistent training behavior
                            pass

                    # Prepare condition images batch for ControlNet training
                    condition_images_batch = None
                    loss_weight_maps_batch = None
                    use_condition_images = getattr(self, 'use_condition_images', False)
                    if use_condition_images and condition_images_list:
                        # Only use batch if ALL items have valid condition images
                        if all(ci is not None for ci in condition_images_list):
                            condition_images_batch = torch.cat(condition_images_list, dim=0)  # [B, 3, H, W]
                            # Outpaint mode: parallel per-item latent-space loss weight
                            # maps (built in lockstep with condition_images_list in the
                            # condition-load branch above), so a successful condition
                            # always has a matching weight map here. Non-outpaint modes
                            # never populate loss_weight_maps_list with non-None entries,
                            # so this stays None for them (unchanged behavior).
                            if getattr(self, '_is_outpaint_mode', False) and loss_weight_maps_list \
                                    and all(w is not None for w in loss_weight_maps_list):
                                loss_weight_maps_batch = torch.cat(loss_weight_maps_list, dim=0)  # [B, 1, H/8, W/8]
                        else:
                            # Mixed batch (some without condition images) - skip this batch
                            print(f"{self.log_prefix} WARNING: Some items in batch missing condition images, skipping batch")
                            del latents_list, text_embeddings_list, auxiliary_data_list
                            if reference_latents_list:
                                del reference_latents_list
                            del condition_images_list
                            if loss_weight_maps_list:
                                del loss_weight_maps_list
                            # Update global_step for skipped batch (to maintain step counting),
                            # consistent with the corrupted-image skip above -- both are
                            # per-batch skips of what would have been multi_noise_timesteps
                            # MNT iterations, so leaving this one un-advanced drifts
                            # progress/resume totals relative to the other skip path.
                            global_step += multi_noise_timesteps
                            self._batches_skipped += 1
                            continue

                    # Free individual item lists (no longer needed, batch tensors are created)
                    del latents_list, text_embeddings_list, auxiliary_data_list
                    del sensenova_prefixes
                    if reference_latents_list:
                        del reference_latents_list
                    if condition_images_list:
                        del condition_images_list
                    if loss_weight_maps_list:
                        del loss_weight_maps_list

                    # Collect batch captions for debug (done once, outside MNT loop)
                    batch_captions = [item.get("caption", "") for item, dataset in batch]

                    # Collect first reference image path per item for debug visualization
                    _ref_paths = [
                        (item.get("reference_images") or [None])[0]
                        for item, dataset in batch
                    ]
                    batch_reference_paths = _ref_paths if any(p is not None for p in _ref_paths) else None

                    batch_size = latents.shape[0]

                    # ============================================================
                    # MNT loop: Process same batch with different noise-timesteps
                    # ============================================================
                    # Sequential MNT Implementation (VRAM optimized):
                    # Each MNT iteration: forward → backward → optimizer.step() → zero_grad()
                    # This prevents gradient accumulation across MNT iterations, keeping
                    # VRAM usage at MNT=1 level regardless of actual MNT value.
                    #
                    # For gradient accumulation across batches, we track accumulated_steps
                    # and only run optimizer.step() when accumulation is complete.
                    #
                    # IMPORTANT: When Text Encoder is trainable AND MNT > 1, we need to
                    # re-encode text embeddings for each MNT iteration to maintain gradient flow.
                    # Otherwise, detach() would cut the gradient to Text Encoder.
                    need_recompute_text_embeddings = (
                        text_encoder_trainable and
                        multi_noise_timesteps > 1 and
                        text_encoding_mode == "onthefly_gpu"
                    )

                    for mnt_idx in range(multi_noise_timesteps):
                        # A skipped batch discarded this window's shared boundary
                        # cut. The remaining iterations have no cut to accumulate
                        # into (the shared route does not re-encode) and no
                        # begin_window to close, so they would train the
                        # generation half against a dead window. End the batch;
                        # the next one cuts afresh.
                        _fp_abort = getattr(self, "sensenova_four_phase", None)
                        if _fp_abort is not None and _fp_abort.window_aborted:
                            break

                        # Sample timesteps for this MNT iteration
                        timesteps = timestep_sampler.sample(batch_size, self.device)

                        # Determine if we should save debug latents (only on first MNT iteration)
                        # With MNT > 1, global_step increments multiple times per batch.
                        # We check if any step within this batch's MNT range hits the debug interval.
                        # Example: MNT=32, debug_every=200
                        #   - Batch 6: steps 192-223, includes step 200 → save at mnt_idx=0
                        #   - Old logic: mnt_idx=0, global_step=192 → 192 % 200 != 0 → NO save (BUG)
                        #   - New logic: mnt_idx=0, check if 192..223 contains a multiple of 200 → YES → save
                        debug_save_path = None
                        if mnt_idx == 0 and debug_dir is not None and debug_latents_every > 0:
                            # batch_start_step = global_step (current step before MNT loop increments)
                            # batch_end_step = global_step + multi_noise_timesteps - 1 (inclusive)
                            batch_start_step = global_step
                            batch_end_step = global_step + multi_noise_timesteps - 1
                            # Check if any multiple of debug_latents_every falls within [batch_start, batch_end]
                            # This happens when floor(batch_end / every) > floor((batch_start - 1) / every)
                            # Or more simply: batch_start <= k*every <= batch_end for some integer k
                            next_debug_step = ((batch_start_step // debug_latents_every) + 1) * debug_latents_every
                            if batch_start_step % debug_latents_every == 0 or next_debug_step <= batch_end_step:
                                # Determine which step to use for the filename
                                if batch_start_step % debug_latents_every == 0:
                                    save_step = batch_start_step
                                else:
                                    save_step = next_debug_step
                                debug_save_path = debug_dir / f"step_{save_step:06d}"

                        # Detach latents to create fresh computation graph for this MNT iteration
                        # This is necessary because backward() frees the graph
                        mnt_latents = latents.detach()
                        # REPA clean-image pixels are timestep-independent -> same across MNT.
                        mnt_repa_pixels = repa_pixels_batch
                        # SDXL time_ids are per-item (size/crop), timestep-independent.
                        mnt_time_ids = time_ids_batch

                        # Handle text embeddings based on training mode
                        mnt_sensenova_prefix = None
                        if self.is_sensenova:
                            (
                                mnt_text_embeddings,
                                mnt_attention_mask,
                                mnt_pooled_embeddings,
                                mnt_sensenova_prefix,
                            ) = self._sensenova_mnt_conditioning(
                                sensenova_prefix,
                                captions=batch_captions,
                                mnt_index=mnt_idx,
                            )
                            # Declare the window the shared boundary cut serves.
                            # No-op unless the shared route is armed.
                            if mnt_idx == 0:
                                _fp_window = getattr(self, "sensenova_four_phase", None)
                                if _fp_window is not None:
                                    _fp_window.begin_window(multi_noise_timesteps)
                        elif need_recompute_text_embeddings:
                            # Text Encoder trainable + MNT > 1: Re-encode text for each iteration
                            # This creates a fresh computation graph with gradient flow to Text Encoder.
                            # NOTE: unreachable for ACE-Step (text_encoder is unconditionally frozen in
                            # acestep_ops.load_components, so text_encoder_trainable is always False) --
                            # `caption`-only (no per-item lyrics) here is therefore never a lyrics gap.
                            mnt_text_embeddings_list = []
                            mnt_auxiliary_data_list = []
                            for caption in batch_captions:
                                embeddings, auxiliary = self.encode_caption(caption, requires_grad=True)
                                mnt_text_embeddings_list.append(embeddings)
                                mnt_auxiliary_data_list.append(auxiliary)

                            # Stack embeddings (handle variable sequence lengths).
                            # Normalize 2D [L, D] per-item embeds (Anima/Z-Image drop
                            # the batch dim) to 3D [1, L, D] so cat(dim=0) batches
                            # correctly instead of collapsing batch into the seq axis.
                            mnt_text_embeddings_list = [
                                emb.unsqueeze(0) if emb.dim() == 2 else emb
                                for emb in mnt_text_embeddings_list
                            ]
                            seq_lengths = [emb.shape[1] for emb in mnt_text_embeddings_list]
                            max_seq_len = max(seq_lengths)
                            if len(set(seq_lengths)) > 1:
                                padded_embeddings = []
                                for emb in mnt_text_embeddings_list:
                                    if emb.shape[1] < max_seq_len:
                                        pad_length = max_seq_len - emb.shape[1]
                                        padding = torch.zeros(
                                            (emb.shape[0], pad_length, emb.shape[2]),
                                            dtype=emb.dtype, device=emb.device
                                        )
                                        emb = torch.cat([emb, padding], dim=1)
                                    padded_embeddings.append(emb)
                                mnt_text_embeddings = torch.cat(padded_embeddings, dim=0)
                            else:
                                mnt_text_embeddings = torch.cat(mnt_text_embeddings_list, dim=0)

                            # Prepare auxiliary data
                            if self.is_zimage:
                                mnt_attention_mask = torch.stack([aux for aux in mnt_auxiliary_data_list if aux is not None], dim=0)
                                mnt_pooled_embeddings = None
                            elif self.is_lens or self.is_ideogram4 or self.is_minit2i or self.is_krea2:
                                # encoder_mask per sample: [L] → stacked to [B, L]
                                mnt_attention_mask = torch.stack([aux for aux in mnt_auxiliary_data_list if aux is not None], dim=0)
                                mnt_pooled_embeddings = None
                            elif self.is_anima:
                                # Anima: per-item dict → one dict of batched [B, L] tensors.
                                mnt_attention_mask = self.arch.collate_aux(self, mnt_auxiliary_data_list)
                                mnt_pooled_embeddings = None
                            elif self.is_ltx2:
                                # LTX-2.3: per-item dict → one collated aux dict.
                                # fps is per-CLIP: inject per-sample fps [B] from
                                # the batch items (not the per-caption text aux).
                                mnt_attention_mask = self.arch.collate_aux(self, mnt_auxiliary_data_list)
                                mnt_attention_mask["fps"] = self._ltx2_batch_fps_tensor(batch)
                                mnt_pooled_embeddings = None
                            elif self.is_minimax_h3:
                                # MiniMax-H3: per-item dict + the per-clip audio
                                # latent injected from the batch items.
                                mnt_attention_mask = self.arch.collate_aux(self, mnt_auxiliary_data_list)
                                mnt_attention_mask.update(self._minimax_h3_batch_audio(batch))
                                mnt_pooled_embeddings = None
                            elif self.is_acestep:
                                # ACE-Step: per-item dict → one collated aux dict.
                                mnt_attention_mask = self.arch.collate_aux(self, mnt_auxiliary_data_list)
                                mnt_pooled_embeddings = None
                            elif self.is_sdxl and any(aux is not None for aux in mnt_auxiliary_data_list):
                                mnt_pooled_embeddings = torch.cat([aux for aux in mnt_auxiliary_data_list if aux is not None], dim=0)
                                mnt_attention_mask = None
                            else:
                                mnt_attention_mask = None
                                mnt_pooled_embeddings = None

                            del mnt_text_embeddings_list, mnt_auxiliary_data_list
                        else:
                            # MNT == 1 or Text Encoder frozen: reuse the pre-computed embeddings.
                            # When the TE is trainable at MNT==1 the batch-assembly encode (done
                            # with requires_grad=True for onthefly_gpu) carries a graph we must
                            # keep — there is a single backward, so gradients can flow back into
                            # the text encoder. Otherwise detach to allow safe reuse / avoid
                            # backward-through-graph-twice when MNT>1 with a frozen TE.
                            keep_te_grad = (text_encoder_trainable and multi_noise_timesteps == 1
                                            and text_encoding_mode == "onthefly_gpu")
                            if keep_te_grad:
                                mnt_text_embeddings = text_embeddings
                                mnt_attention_mask = attention_mask
                                mnt_pooled_embeddings = pooled_embeddings
                            else:
                                mnt_text_embeddings = text_embeddings.detach() if text_embeddings is not None else None
                                if isinstance(attention_mask, dict):
                                    # Anima: dict of batched tensors, detach each entry.
                                    mnt_attention_mask = {
                                        k: (v.detach() if isinstance(v, torch.Tensor) else v)
                                        for k, v in attention_mask.items()
                                    }
                                else:
                                    mnt_attention_mask = attention_mask.detach() if attention_mask is not None else None
                                mnt_pooled_embeddings = pooled_embeddings.detach() if pooled_embeddings is not None else None

                        # === Vision Encoder: per-item encoding (SD1.5/SDXL only) ===
                        # Each batch item is conditioned on its own reference image only.
                        # Batches without any reference images skip VE entirely.
                        # When train_vision_encoder=True, VE is already on GPU (moved at training start)
                        # and stays there for the entire training — no per-batch offloading.
                        # When train_vision_encoder=False, VE is moved to GPU for encoding and back to CPU after.
                        ve_obj = getattr(self, 'vision_encoder', None)
                        if ve_obj is not None and mnt_text_embeddings is not None and not self.is_flux2 and not self.is_zimage:
                            train_ve = getattr(self, '_train_vision_encoder', False)
                            ref_paths = [_item.get("reference_images", [None])[0] for _item, _ in batch]
                            batch_has_ref = any(p is not None for p in ref_paths)
                            # Gradient Routing: block gradient flow to TE when batch has reference images,
                            # allowing U-net cross-attention K,V projections to learn VE's feature subspace.
                            if getattr(self, '_gradient_routing_ve', False) and batch_has_ref:
                                mnt_text_embeddings = mnt_text_embeddings.detach()
                                if mnt_pooled_embeddings is not None:
                                    mnt_pooled_embeddings = mnt_pooled_embeddings.detach()
                            # VE Reconstruction Mode: zero text embeddings for items that use their own
                            # image as reference. Mask broadcasts over sequence dim (handles chunking).
                            _ve_recon_mask = [bool(_item.get("_ve_reconstruction_mode")) for _item, _ in batch]
                            if any(_ve_recon_mask) and mnt_text_embeddings is not None:
                                _mask = torch.tensor(
                                    _ve_recon_mask,
                                    dtype=mnt_text_embeddings.dtype,
                                    device=mnt_text_embeddings.device,
                                ).view(-1, 1, 1)  # [B, 1, 1] broadcasts over [B, seq_len, dim]
                                mnt_text_embeddings = mnt_text_embeddings * (1.0 - _mask)
                                if mnt_pooled_embeddings is not None:
                                    _mask_p = _mask.view(-1, 1)  # [B, 1] for pooled embedding
                                    mnt_pooled_embeddings = mnt_pooled_embeddings * (1.0 - _mask_p)
                            if batch_has_ref:
                                try:
                                    # Reload VE params AND its optimizer state to GPU before
                                    # the step (they may have been offloaded during a
                                    # reference-free run); keeps optimizer.step() consistent.
                                    self._ve_set_device(self.device)
                                    ve_obj.train(train_ve)
                                    target_dim = mnt_text_embeddings.shape[-1]
                                    ve_pos_list = []
                                    for _ref_path in ref_paths:
                                        if _ref_path is not None:
                                            _pil = flatten_to_rgb(Image.open(_ref_path))
                                            # with_grad=True keeps gradients flowing through VE for training;
                                            # with_grad=False (default) wraps in torch.no_grad() for inference.
                                            _ve_pos_i, _ = ve_obj.encode(
                                                [_pil],
                                                target_dim=target_dim,
                                                dtype=self.training_dtype,
                                                with_grad=train_ve,
                                            )
                                            ve_pos_list.append(_ve_pos_i.to(self.device))  # [1, 257, dim]
                                    if ve_pos_list:
                                        # Stack per-item embeddings: [B, 257, dim]
                                        ve_pos_batch = torch.cat(ve_pos_list, dim=0)
                                        mnt_text_embeddings = torch.cat([mnt_text_embeddings, ve_pos_batch], dim=1)
                                    if not train_ve:
                                        # No gradients needed — offload immediately
                                        ve_obj.to("cpu")
                                        torch.cuda.empty_cache()
                                except Exception as _ve_err:
                                    print(f"{self.log_prefix} WARNING: VE encoding failed: {_ve_err}, skipping VE conditioning")
                                    try:
                                        ve_obj.to("cpu")
                                    except Exception:
                                        pass

                        # Training step with OOM recovery (forward + backward)
                        # If OOM occurs, the batch is automatically split and processed sequentially
                        # Wrap in try-except as final safety net - if all recovery fails, skip batch
                        cuda_error_skip = False  # Flag to skip optimizer step when CUDA is in bad state

                        # The fused hooks clear param.grad; arm them to record its
                        # squared norm first, but only on the steps whose norms are
                        # reported (the same condition should_step_optimizer uses
                        # below, for the step this backward is about to become).
                        if self._fused_grad_norm is not None:
                            self._fused_grad_norm.begin_step(
                                (global_step + 1) % gradient_accumulation_steps == 0
                            )
                        # G-RB3: start this step's updated-parameter census. Armed
                        # on every backward, not only on optimizer steps: under
                        # the fused path each backward IS an optimizer step.
                        # `expect_deferred` is the exception a deferral window
                        # needs: on a shared four-phase window's non-final
                        # backwards the understanding half is correctly not
                        # updated, and is required in full again on the backward
                        # that closes the window. Every other route answers True
                        # here, so the census is unchanged for them.
                        if self._update_census is not None:
                            _fp_census = getattr(self, "sensenova_four_phase", None)
                            self._update_census.begin_step(
                                True,
                                expect_deferred=(
                                    _fp_census is None
                                    or _fp_census.is_final_iteration()
                                ),
                            )

                        # Lens: pass latent spatial dims so train_step_lens can build img_shapes
                        # correctly for non-square resolutions.  width/height from batch loop.
                        batch_lens_latent_shape = None
                        if (self.is_lens or self.is_ideogram4 or self.is_krea2) and width and height:
                            batch_lens_latent_shape = (height // 16, width // 16)

                        try:
                            mnt_loss_value, mnt_pred_loss_value, mnt_recon_loss_value, cuda_error_skip = self._forward_backward_with_oom_recovery(
                                mnt_latents=mnt_latents,
                                mnt_text_embeddings=mnt_text_embeddings,
                                mnt_attention_mask=mnt_attention_mask,
                                mnt_pooled_embeddings=mnt_pooled_embeddings,
                                timesteps=timesteps,
                                debug_save_path=debug_save_path,
                                batch_captions=batch_captions,
                                batch_reference_paths=batch_reference_paths,
                                alphas_cumprod_cached=alphas_cumprod_cached,
                                use_condition_images=use_condition_images,
                                condition_images_batch=condition_images_batch,
                                reference_latents_nested=reference_latents_nested,
                                min_split_batch_size=1,
                                lens_latent_shape=batch_lens_latent_shape,
                                mnt_repa_pixels=mnt_repa_pixels,
                                mnt_time_ids=mnt_time_ids,
                                loss_weight_maps_batch=loss_weight_maps_batch,
                                sensenova_prefix=mnt_sensenova_prefix,
                            )
                        except PartialOptimizerStepError:
                            # Half-applied step: the safety net below would swallow
                            # it as a recoverable OOM (it is raised FROM one) and
                            # skip the batch, which is the behaviour it exists to
                            # prevent.
                            raise
                        except FatalCudaError:
                            # Sticky CUDA-context corruption, already classified by the
                            # inner recovery path -- do not swallow it here. Re-raise so
                            # the outer emergency-save handler gets a chance to save
                            # whatever CPU-side state is still salvageable and exit.
                            raise
                        except Exception as batch_error:
                            # Final safety net: if all OOM recovery attempts failed,
                            # skip this batch and continue training. Use the single
                            # classifier so this stays in lockstep with the inner
                            # recovery path -- only a genuinely recoverable OOM is
                            # swallowed here; a fatal CUDA error is re-raised.
                            _cls_outer = self._classify_cuda_error(batch_error)
                            if _cls_outer == "fatal":
                                raise FatalCudaError(str(batch_error)) from batch_error
                            is_cuda_error = _cls_outer == "oom"
                            if is_cuda_error:
                                print(f"{self.log_prefix} [FATAL CUDA Error] All recovery attempts failed, SKIPPING BATCH")
                                print(f"{self.log_prefix} [FATAL CUDA Error] {str(batch_error)[:200]}")
                                # Set flag to skip optimizer step - CUDA is in bad state
                                cuda_error_skip = True
                                # Aggressive cleanup
                                try:
                                    self.optimizer.zero_grad(set_to_none=True)
                                except Exception as e:
                                    print(f"{self.log_prefix} [FATAL CUDA Error] zero_grad failed: {e}")
                                # The abandoned batch's prefix was already CUT during
                                # prep, so its boundary leaves outlive the skip. Without
                                # this the NEXT batch's cut() raises "never captured",
                                # which _classify_cuda_error cannot see as an OOM and so
                                # kills a run this path exists to keep alive.
                                _four_phase = getattr(self, "sensenova_four_phase", None)
                                if _four_phase is not None:
                                    _dropped = _four_phase.discard()
                                    if _dropped:
                                        # A shared window's earlier iterations
                                        # already applied their GENERATION
                                        # updates; their understanding gradient
                                        # dies with the cut. Announced and
                                        # charted rather than left to widen the
                                        # asymmetry silently.
                                        from core.training.training_events import (
                                            emit_training_warning,
                                        )
                                        emit_training_warning(
                                            f"SenseNova four-phase: a skipped batch "
                                            f"discarded the shared boundary cut after "
                                            f"{_dropped} of its generation backward(s) "
                                            f"had already applied their updates, so "
                                            f"that window contributes nothing to the "
                                            f"understanding half "
                                            f"({_four_phase.dropped_backwards} "
                                            f"backward(s) dropped this run).",
                                            code="sensenova_four_phase_window_dropped",
                                            prefix=self.log_prefix,
                                        )
                                        # Gated with the warning, not outside it:
                                        # the per-iteration route never drops a
                                        # window, so charting it there would add
                                        # a permanently-zero series.
                                        self.log_extra_metric(
                                            "sn_und_grad_dropped",
                                            float(_four_phase.dropped_backwards),
                                        )
                                gc.collect()
                                try:
                                    torch.cuda.synchronize()
                                except Exception:
                                    pass
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                # Skip this batch with zero loss
                                mnt_loss_value, mnt_pred_loss_value, mnt_recon_loss_value = 0.0, 0.0, 0.0
                            else:
                                # Non-CUDA error - re-raise
                                raise

                        # If the recovery couldn't fit even one sample, record this
                        # resolution bucket so the next epoch's re-bucketing drops it
                        # (no point full-attempting + OOM-skipping it every occurrence).
                        if getattr(self, "_batch_was_unfittable", False) and _cur_bucket_wh and all(_cur_bucket_wh):
                            if _cur_bucket_wh not in self._unfittable_buckets:
                                self._unfittable_buckets.add(_cur_bucket_wh)
                                print(f"{self.log_prefix} [OOM] bucket {_cur_bucket_wh[0]}x{_cur_bucket_wh[1]} "
                                      f"won't fit one sample -> excluding it from subsequent epochs "
                                      f"({len(self._unfittable_buckets)} bucket(s) excluded so far)")

                        if not cuda_error_skip:
                            self._backwards_completed += 1
                        else:
                            self._batches_skipped += 1

                        # G-RB3: every trainable parameter must have received an
                        # update during that backward. Skipped when the batch was
                        # abandoned, since then no backward completed.
                        if self._update_census is not None and not cuda_error_skip:
                            self._update_census.assert_complete(
                                f"global_step={global_step}"
                            )

                        # Clear MNT iteration tensors (backward already done in helper)
                        del mnt_latents, mnt_text_embeddings
                        if mnt_attention_mask is not None:
                            del mnt_attention_mask
                        if mnt_pooled_embeddings is not None:
                            del mnt_pooled_embeddings

                        # Clear saved activations immediately after backward to prevent VRAM leaks
                        if hasattr(self, 'layer_offload_conductor') and self.layer_offload_conductor is not None:
                            self.layer_offload_conductor.clear_activations()

                        # FLUX.2: Clear block swap activations
                        if hasattr(self, 'flux2_block_offloader') and self.flux2_block_offloader is not None:
                            self.flux2_block_offloader.clear_activations()

                        # Increment global step for each MNT iteration
                        global_step += 1
                        if self.debug_vram and global_step in (1, 5, 10):
                            _vramdiag(f"train_step_{global_step}")

                        # ============================================================
                        # Per-MNT-iteration logging (for real-time frontend updates)
                        # ============================================================
                        # Log loss immediately for each MNT iteration so frontend
                        # updates every step, not just every MNT*grad_accum steps.
                        # Grad norm will be updated after optimizer step.
                        # Note: mnt_loss_value, mnt_pred_loss_value, mnt_recon_loss_value
                        # are already extracted as floats by _forward_backward_with_oom_recovery()
                        mnt_current_lr = self.lr_scheduler.get_last_lr()[0]

                        # Record the ACTUALLY-APPLIED per-step LR (as opposed to
                        # mnt_current_lr above, which is the scheduler's own view via
                        # get_last_lr() -- under the resume LR-group remap the group
                        # actually stepped can diverge from what the scheduler thinks
                        # it set). optimizer.param_groups[0]['lr'] is what the just-
                        # completed step actually used, so it's the more truthful
                        # value to chart, especially now that schedules can be
                        # non-constant (plateau_cosine_floor). Routed through the
                        # generic extra-metrics channel (same mechanism as gen_loss/
                        # seam_loss/known_loss) so it needs no DB column and is
                        # captured by the _log_metrics_to_db call below, same step.
                        # This is the single per-step point shared by LoRA, full-FT,
                        # and ControlNet training (all subclass BaseTrainer.train()),
                        # so every method gets this series with no extra call sites.
                        # Defensive: never let a missing/malformed optimizer raise in
                        # the hot training loop.
                        try:
                            if self.optimizer is not None and self.optimizer.param_groups:
                                self.log_extra_metric("lr", self.optimizer.param_groups[0]["lr"])

                                # Also log each param group's ACTUAL LR labeled by
                                # component, when the run trains more than one
                                # component (e.g. UNet + TE1/TE2, or +VE) at
                                # potentially-different LRs. Single-group runs
                                # (e.g. ControlNet) keep only the "lr" series
                                # above -- do not duplicate it here.
                                # _build_component_lr_list() is only called
                                # once per step (cached in a local) and is
                                # itself best-effort: any mismatch/exception
                                # just falls back to the single "lr" series.
                                if len(self.optimizer.param_groups) > 1:
                                    _component_lrs, _component_names = self._build_component_lr_list()
                                    for _i, _pg in enumerate(self.optimizer.param_groups):
                                        _name = _component_names[_i] if _i < len(_component_names) else f"g{_i}"
                                        _key = "lr_" + re.sub(r'[^a-z0-9]+', '', _name.lower())
                                        self.log_extra_metric(_key, float(_pg["lr"]))
                        except Exception:
                            pass

                        # SenseNova MoT phase eviction: this step's half swaps.
                        # Drained ONCE here because log_extra_metric overwrites,
                        # and a four-phase step performs two swaps -- per-swap
                        # calls would silently report only the last one.
                        _sn_evictor = getattr(self, "sensenova_phase_evictor", None)
                        if _sn_evictor is not None:
                            try:
                                _sn = _sn_evictor.drain_transfer_stats()
                            except Exception:
                                _sn = None
                            if _sn is not None:
                                self.log_extra_metric("sn_d2h_s", _sn["d2h_seconds"])
                                self.log_extra_metric("sn_h2d_s", _sn["h2d_seconds"])
                                self.log_extra_metric("sn_d2h_gib", _sn["d2h_bytes"] / 2 ** 30)
                                self.log_extra_metric("sn_h2d_gib", _sn["h2d_bytes"] / 2 ** 30)
                                # Which unit the two seconds series are in --
                                # see metric_registry's note on sn_swap_overlap.
                                # Taken from the drained dict, not off the
                                # evictor: the drain resets it.
                                self.log_extra_metric(
                                    "sn_swap_overlap",
                                    1.0 if _sn.get("overlap_active") else 0.0,
                                )
                                if torch.cuda.is_available():
                                    self.log_extra_metric(
                                        "sn_peak_alloc_gib",
                                        torch.cuda.max_memory_allocated() / 2 ** 30,
                                    )
                                    self.log_extra_metric(
                                        "sn_peak_resv_gib",
                                        torch.cuda.max_memory_reserved() / 2 ** 30,
                                    )

                        # TensorBoard logging (per-iteration for loss only)
                        self.writer.add_scalar("train/loss", mnt_loss_value, global_step)
                        self.writer.add_scalar("train/pred_loss", mnt_pred_loss_value, global_step)
                        self.writer.add_scalar("train/recon_loss", mnt_recon_loss_value, global_step)
                        self.writer.add_scalar("train/lr", mnt_current_lr, global_step)

                        # Outpaint conditioning: generate-region-only MSE (monitoring).
                        # Set by train_step_controlnet only in outpaint mode; None
                        # otherwise -> skipped (no effect on any other training path).
                        # Logged to TensorBoard AND a plain JSONL sidecar so the
                        # generate-region learning trend can be read back cheaply
                        # without a DB schema change.
                        _gen_loss = getattr(self, "_last_gen_region_loss", None)
                        if _gen_loss is not None:
                            self.writer.add_scalar("train/gen_loss", _gen_loss, global_step)
                            # Route into the generic extra-metrics channel so it
                            # is persisted to DB + charted (captured by the
                            # _log_metrics_to_db call just below, same step).
                            self.log_extra_metric("gen_loss", _gen_loss)
                            try:
                                import json as _json
                                with open(self.output_dir / "gen_region_loss.jsonl", "a") as _gf:
                                    _gf.write(_json.dumps({
                                        "step": global_step,
                                        "gen_loss": _gen_loss,
                                        "loss": mnt_loss_value,
                                    }) + "\n")
                            except Exception:
                                pass

                        # Outpaint conditioning: seam-ring-only MSE (monitoring, always
                        # on in outpaint mode -- unlike gen_loss, this isolates the
                        # ~2-3% of generate cells immediately adjacent to the known
                        # region, which gen_loss averages away. Behavior-neutral
                        # instrument, independent of outpaint_seam_loss_boost.
                        _seam_loss = getattr(self, "_last_seam_ring_loss", None)
                        if _seam_loss is not None:
                            self.writer.add_scalar("train/seam_loss", _seam_loss, global_step)
                            self.log_extra_metric("seam_loss", _seam_loss)

                        # Outpaint conditioning: loss-vs-timestep instrumentation
                        # (monitoring, always on in outpaint mode -- see scratchpad
                        # "Outpaint ControlNet: loss-vs-timestep instrumentation"
                        # design doc). Per-sample (t, snr, region-loss) tuples for
                        # the last MNT micro-batch, written to a JSONL sidecar for
                        # offline re-binning (batch-mean-t would destroy the
                        # high-noise-tail curvature -- per-sample arrays are
                        # mandatory, see design doc G2). `known_loss` (the
                        # batch-mean of eps_known, ignoring empty-mask samples) is
                        # also routed into the generic extra-metrics channel for a
                        # live known-vs-gen glance on the existing loss chart; t/snr
                        # /x0 stay JSONL-only (not per-step scalars, would need
                        # per-sample keys and pollute the chart registry).
                        _lvt = getattr(self, "_last_loss_vs_t", None)
                        if _lvt is not None:
                            _known_vals = [v for v in _lvt.get("eps_known", []) if v is not None]
                            if _known_vals:
                                _known_loss_mean = sum(_known_vals) / len(_known_vals)
                                self.log_extra_metric("known_loss", _known_loss_mean)
                            try:
                                import json as _json
                                with open(self.output_dir / "loss_vs_t.jsonl", "a") as _lf:
                                    _lf.write(_json.dumps({
                                        "step": global_step,
                                        "t": _lvt.get("t", []),
                                        "snr": _lvt.get("snr", []),
                                        "eps_known": _lvt.get("eps_known", []),
                                        "eps_gen": _lvt.get("eps_gen", []),
                                        "eps_seam": _lvt.get("eps_seam", []),
                                        "x0_known": _lvt.get("x0_known", []),
                                        "x0_gen": _lvt.get("x0_gen", []),
                                    }) + "\n")
                            except Exception:
                                pass

                        # Database logging (per-iteration, loss only - grad_norm logged at optimizer step)
                        # Grad norm is only available after optimizer step, so we don't log it here.
                        # This prevents grad_norm=0 from corrupting smoothed grad norm charts.
                        if self.run_id is not None:
                            self._log_metrics_to_db(
                                step=global_step,
                                loss=mnt_pred_loss_value,
                                recon_loss=mnt_recon_loss_value,
                                learning_rate=mnt_current_lr,
                                grad_norm=None,  # Don't set - will be updated after optimizer step
                                grad_norm_text_encoder=None,
                                grad_norm_unet=None
                            )

                        # Progress callback (per-iteration for real-time UI updates)
                        if progress_callback:
                            progress_callback(
                                phase="training",
                                step=global_step,
                                total=actual_total_steps,
                                epoch=epoch,
                                loss=mnt_loss_value,
                            )

                        # ============================================================
                        # Sequential MNT: Optimizer step after each MNT iteration
                        # ============================================================
                        # This prevents gradient accumulation across MNT iterations,
                        # keeping VRAM at MNT=1 level.
                        #
                        # Key insight: Each MNT iteration is treated as an independent
                        # training step. Gradient accumulation (if configured) happens
                        # across these MNT steps, not across batches.
                        #
                        # global_step = (batch_idx * multi_noise_timesteps) + (mnt_idx + 1)
                        # We step optimizer when global_step is divisible by gradient_accumulation_steps
                        #
                        # IMPORTANT: Skip optimizer step if CUDA error occurred and batch was skipped.
                        # When CUDA is in bad state, grad_scaler.unscale_() will fail.
                        should_step_optimizer = (global_step % gradient_accumulation_steps == 0)

                        if cuda_error_skip:
                            # CUDA error occurred - skip optimizer step entirely
                            # The batch was skipped, so there are no valid gradients to step with
                            print(f"{self.log_prefix} [CUDA Recovery] Skipping optimizer step (batch was skipped)")
                            grad_norm_total, grad_norm_te, grad_norm_unet, grad_norm_ve = 0.0, 0.0, 0.0, 0.0
                            # Still step LR scheduler to keep it in sync with global_step
                            if should_step_optimizer:
                                try:
                                    if self.fused_optimizer_groups is not None:
                                        for lr_scheduler in self.lr_schedulers:
                                            lr_scheduler.step()
                                    else:
                                        self.lr_scheduler.step()
                                except Exception as lr_err:
                                    print(f"{self.log_prefix} [CUDA Recovery] LR scheduler step failed: {lr_err}")
                        elif should_step_optimizer:
                            four_phase = getattr(self, "sensenova_four_phase", None)
                            if four_phase is not None:
                                # Normally a no-op: phase 3 already ran with the
                                # backward. This is the accumulation path's seam,
                                # kept so a future route that defers it lands
                                # before the grad norms rather than after.
                                four_phase.flush()
                            self._assert_sensenova_step_seam_residency(four_phase)
                            if not self.use_fused_backward and self.fused_optimizer_groups is None:
                                # Normal flow: optimizer.step() and zero_grad() here
                                if self.use_grad_scaler:
                                    # GradScaler flow
                                    self.grad_scaler.unscale_(self.optimizer)
                                    grad_norm_total, grad_norm_te, grad_norm_te1, grad_norm_te2, grad_norm_unet, grad_norm_ve = self._calculate_grad_norms()
                                    if max_grad_norm > 0:
                                        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_grad_norm)
                                    self.grad_scaler.step(self.optimizer)
                                    self.grad_scaler.update()
                                    self.optimizer.zero_grad()
                                    self._update_ema()
                                else:
                                    # Normal flow without GradScaler
                                    grad_norm_total, grad_norm_te, grad_norm_te1, grad_norm_te2, grad_norm_unet, grad_norm_ve = self._calculate_grad_norms()
                                    if max_grad_norm > 0:
                                        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_grad_norm)
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()
                                    self._update_ema()
                            else:
                                # Fused backward/groups flow: the hooks have already
                                # stepped and cleared the grads, so the norms come
                                # from what they recorded first. No clipping is
                                # possible here - see the warning.
                                self._warn_grad_clipping_ignored_under_fused(max_grad_norm)
                                grad_norm_total, grad_norm_te, grad_norm_te1, grad_norm_te2, grad_norm_unet, grad_norm_ve = self._calculate_grad_norms()

                            # LR scheduler step
                            if self.fused_optimizer_groups is not None:
                                for lr_scheduler in self.lr_schedulers:
                                    lr_scheduler.step()
                            else:
                                self.lr_scheduler.step()

                            # Log grad_norm to TensorBoard
                            self.writer.add_scalar("train/grad_norm", grad_norm_total, global_step)
                            if grad_norm_te > 0.0:
                                self.writer.add_scalar("train/grad_norm_text_encoder", grad_norm_te, global_step)
                            self.writer.add_scalar("train/grad_norm_unet", grad_norm_unet, global_step)
                            if grad_norm_te1 > 0.0:
                                self.writer.add_scalar("train/grad_norm_text_encoder_1", grad_norm_te1, global_step)
                            if grad_norm_te2 > 0.0:
                                self.writer.add_scalar("train/grad_norm_text_encoder_2", grad_norm_te2, global_step)
                            if grad_norm_ve > 0.0:
                                self.writer.add_scalar("train/grad_norm_vision_encoder", grad_norm_ve, global_step)

                            # Update grad_norm in database
                            if self.run_id is not None:
                                self._log_metrics_to_db(
                                    step=global_step,
                                    loss=None,
                                    recon_loss=None,
                                    learning_rate=None,
                                    grad_norm=grad_norm_total,
                                    grad_norm_text_encoder=grad_norm_te if grad_norm_te > 0.0 else None,
                                    grad_norm_text_encoder_1=grad_norm_te1 if grad_norm_te1 > 0.0 else None,
                                    grad_norm_text_encoder_2=grad_norm_te2 if grad_norm_te2 > 0.0 else None,
                                    grad_norm_unet=grad_norm_unet,
                                    grad_norm_vision_encoder=grad_norm_ve if grad_norm_ve > 0.0 else None,
                                )

                            # Parameter change tracking (B: update norm, C: cumulative drift)
                            if self._param_tracker is not None:
                                pt = self._param_tracker.compute(global_step)
                                if pt is not None:
                                    un = pt['update_norm']
                                    cd = pt['cumulative_drift']
                                    for name, val in un.items():
                                        self.writer.add_scalar(f"param/update_norm_{name}", val, global_step)
                                    for name, val in cd.items():
                                        self.writer.add_scalar(f"param/cumulative_drift_{name}", val, global_step)
                                    if self.run_id is not None:
                                        self._log_metrics_to_db(
                                            step=global_step,
                                            param_update_norm_unet=un.get('unet'),
                                            param_update_norm_te1=un.get('te1'),
                                            param_update_norm_te2=un.get('te2'),
                                            param_update_norm_ve=un.get('ve'),
                                            param_cumulative_drift_unet=cd.get('unet'),
                                            param_cumulative_drift_te1=cd.get('te1'),
                                            param_cumulative_drift_te2=cd.get('te2'),
                                            param_cumulative_drift_ve=cd.get('ve'),
                                        )

                            # ReLoRA merge-reinit cycle hook
                            # Only active for ReLoRATrainer (has should_merge method)
                            if hasattr(self, 'should_merge'):
                                is_first_batch = (batch_idx == 0 and mnt_idx == 0)
                                if self.should_merge(global_step, epoch, is_first_batch):
                                    self.perform_merge_reinit_cycle(global_step, epoch)

                        # Force CUDA memory cleanup between MNT iterations to prevent
                        # VRAM fragmentation and accumulation. Skip on last iteration
                        # since batch cleanup follows immediately.
                        if multi_noise_timesteps > 1 and mnt_idx < multi_noise_timesteps - 1:
                            torch.cuda.empty_cache()

                    # Free batch tensors AFTER all MNT iterations complete
                    del latents, text_embeddings
                    if attention_mask is not None:
                        del attention_mask
                    if pooled_embeddings is not None:
                        del pooled_embeddings
                    if reference_latents_nested is not None:
                        del reference_latents_nested

                    # ============================================================
                    # Post-batch processing (Sequential MNT: optimizer step done in loop)
                    # ============================================================
                    # With Sequential MNT, optimizer.step() is called inside the MNT loop
                    # after each MNT iteration. Here we only handle:
                    # - TensorBoard flushing
                    # - Checkpoint saving
                    # - Sample generation

                    # Flush TensorBoard writer periodically to prevent DRAM accumulation
                    # (TensorBoard buffers events internally, can accumulate GBs over long training)
                    if global_step % 100 == 0:
                        self.writer.flush()
                        # Also clear CUDA cache to prevent fragmented memory accumulation
                        torch.cuda.empty_cache()

                    # Save checkpoint (check against global_step which increments per MNT iteration)
                    if interval_due(global_step, save_every_n_steps):
                        # Transient Windows file locks (antivirus / indexer) can raise
                        # PermissionError mid-save; a single such failure must NOT kill a
                        # multi-hour run. Treat the periodic save as best-effort: log and
                        # continue, the next interval will save again. (Disk-full or real
                        # errors still surface in the log.)
                        try:
                            # Flush metrics buffer before checkpoint to ensure consistency
                            if self.run_id is not None:
                                self._log_metrics_to_db(step=global_step, force_flush=True)
                            self.save_checkpoint(step=global_step, epoch=epoch)
                            self._last_periodic_checkpoint_step = global_step
                            # Save training state (epoch progress) for mid-epoch resume
                            self.save_training_state(step=global_step, epoch=epoch, batch_idx=self._epoch_batch_position(batch_idx), multi_noise_timesteps=multi_noise_timesteps)
                            # Save optimizer state (momentum, variance, etc.)
                            self.save_optimizer_state(step=global_step)
                            # Save EMA shadow state + weight snapshot (no-op unless use_ema)
                            self.save_ema_state(step=global_step)
                            self._save_ema_checkpoint(step=global_step, epoch=epoch)
                            # Cleanup old checkpoints (LoRA uses 3-arg version, Full FT uses 1-arg version)
                            if hasattr(self, '_cleanup_old_checkpoints'):
                                import inspect
                                sig = inspect.signature(self._cleanup_old_checkpoints)
                                if len(sig.parameters) == 3:
                                    # LoRATrainer version: (current_step, max_to_keep, save_every)
                                    self._cleanup_old_checkpoints(global_step, max_step_saves_to_keep, save_every_n_steps)
                                else:
                                    # BaseTrainer/FullParameterTrainer version: (max_step_saves_to_keep)
                                    self._cleanup_old_checkpoints(max_step_saves_to_keep)
                        except (PermissionError, OSError) as _save_err:
                            print(f"{self.log_prefix} WARNING: checkpoint save at step {global_step} "
                                  f"failed ({type(_save_err).__name__}: {_save_err}); continuing, "
                                  f"will retry at next interval")
                        # Clear CUDA cache after checkpoint save to free temporary buffers
                        torch.cuda.empty_cache()

                    # Generate sample
                    # Also generate at step 0 to verify base model output
                    # With MNT > 1, check if any step in the batch's MNT range contains a sample interval
                    # batch range: [global_step - multi_noise_timesteps + 1, global_step] (inclusive)
                    should_generate_sample = False
                    sample_step = global_step  # Default: use current global_step for filename

                    if sample_every_n_steps > 0:
                        batch_start_step = global_step - multi_noise_timesteps + 1
                        batch_end_step = global_step

                        # batch_start_step <= 0 (not just == 0): an MNT window
                        # that never completes (e.g. an OOM-discard mid-window
                        # on the run's first batch) can leave global_step at
                        # multi_noise_timesteps - 1 without global_step ever
                        # having been exactly 0, which still yields <= 0 here.
                        # Routed through the same run-identifying guard as the
                        # pre-loop call so this can't re-overwrite THIS run's
                        # own already-saved step-0 sample.
                        if batch_start_step <= 0:
                            step0_sample_path = self.output_dir / "samples" / f"step_{0:06d}_sample_0.png"
                            if not self._step0_sample_done_for_this_run(step0_sample_path):
                                should_generate_sample = True
                                sample_step = 0
                        else:
                            # Check if any multiple of sample_every_n_steps falls within [batch_start, batch_end]
                            next_sample_step = ((batch_start_step // sample_every_n_steps) + 1) * sample_every_n_steps
                            if batch_start_step % sample_every_n_steps == 0:
                                should_generate_sample = True
                                sample_step = batch_start_step
                            elif next_sample_step <= batch_end_step:
                                should_generate_sample = True
                                sample_step = next_sample_step

                    if should_generate_sample:
                        import torchvision

                        for sample_idx, prompt_config in enumerate(self._sample_prompts):
                            positive = prompt_config.get('positive', 'a beautiful landscape')
                            condition_image_path = prompt_config.get('condition_image_path') or None
                            reference_image_path = prompt_config.get('reference_image_path') or None

                            print(f"{self.log_prefix} Generating sample {sample_idx} with prompt='{positive[:50]}...', width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, steps={sample_steps}, seed={sample_seed}")
                            sample = self._dispatch_sample(
                                positive,
                                width=sample_width,
                                height=sample_height,
                                num_inference_steps=sample_steps,
                                guidance_scale=sample_guidance_scale,
                                seed=sample_seed,
                                negative_prompt=prompt_config.get('negative', ''),
                                reference_image_path=reference_image_path,
                                condition_image_path=condition_image_path,
                                current_step=global_step,
                                schedule_type=sample_schedule_type,
                            )
                            # None => architecture can't sample yet; skip this prompt.
                            if sample is None:
                                continue

                            # Save sample with format matching API expectations: step_{step:06d}_sample_{i}.png
                            # Use sample_step (which accounts for MNT batch range) for consistent naming
                            sample_path = self.output_dir / "samples" / f"step_{sample_step:06d}_sample_{sample_idx}.png"
                            sample_path.parent.mkdir(parents=True, exist_ok=True)

                            # Embed generation metadata in PNG for display in Training Monitor
                            png_metadata = PngImagePlugin.PngInfo()
                            png_metadata.add_text("prompt", positive)
                            png_metadata.add_text("negative_prompt", prompt_config.get('negative', ''))
                            png_metadata.add_text("steps", str(sample_steps))
                            png_metadata.add_text("cfg_scale", str(sample_guidance_scale))
                            png_metadata.add_text("seed", str(sample_seed))
                            png_metadata.add_text("width", str(sample_width))
                            png_metadata.add_text("height", str(sample_height))
                            png_metadata.add_text("schedule_type", sample_schedule_type)
                            if condition_image_path:
                                png_metadata.add_text("condition_image_path", condition_image_path)
                            if reference_image_path:
                                png_metadata.add_text("reference_image_path", reference_image_path)
                            sample.save(sample_path, pnginfo=png_metadata)
                            if sample_step == 0 and sample_idx == 0:
                                self._mark_step0_sample_done()
                            print(f"{self.log_prefix} Saved sample to {sample_path}")

                            # Log to TensorBoard
                            image_tensor = torchvision.transforms.ToTensor()(sample)
                            self.writer.add_image(f"samples/sample_{sample_idx}", image_tensor, global_step=sample_step)

                            # Free sample-related tensors
                            del sample, image_tensor

                        torch.cuda.empty_cache()

                        # onthefly_gpu mode: Restore text encoders to GPU after sample generation
                        if text_encoding_mode == "onthefly_gpu":
                            self.move_text_encoder_to_gpu()

                    # Note: Progress callback is now called per-MNT-iteration (above)
                    # for real-time frontend updates during MNT training.

                    # Check if total_steps reached
                    # Use actual_total_steps (which may be recalculated on MNT change during resume)
                    if global_step >= actual_total_steps:
                        print(f"\n{self.log_prefix} Reached target steps ({actual_total_steps}), stopping training")
                        # Skipped batches advance global_step, so a run whose every
                        # batch was skipped reaches the target having trained nothing
                        # and would exit here rather than through the epoch-exhaustion
                        # path below.
                        self._assert_trained_something()
                        return  # Exit training loop

                    # Note: With Sequential MNT, optimizer.step() and loss deletion
                    # are handled inside the MNT loop. No else clause needed here.

                # End of per-batch loop. Stop the cpu_prefetch worker (if any)
                # so the daemon thread doesn't hold onto the TE / batch list
                # past the epoch boundary. The next epoch creates a fresh
                # prefetcher because the `batches` list is rebuilt.
                if te_prefetcher is not None:
                    te_prefetcher.stop()
                    te_prefetcher = None

            # All epochs complete — stop the Danbooru collector thread.
            try:
                if getattr(self, "_danbooru_collector", None) is not None:
                    self._danbooru_collector.stop()
            except Exception:
                pass

            self._assert_trained_something()

        except KeyboardInterrupt:
            # Stop cpu_prefetch worker if it was running (no-op otherwise)
            try:
                if 'te_prefetcher' in locals() and te_prefetcher is not None:
                    te_prefetcher.stop()
            except Exception:
                pass
            try:
                if getattr(self, "_danbooru_collector", None) is not None:
                    self._danbooru_collector.stop()
            except Exception:
                pass
            print(f"\n{self.log_prefix} Training interrupted by user")
            if self._refuse_save_after_partial_step("The interrupt landed", global_step, epoch):
                self.writer.close()
                raise
            print(f"{self.log_prefix} Saving checkpoint at step {global_step}, epoch {epoch}...")

            # Try to save checkpoint (even if it fails, continue to save state)
            checkpoint_saved = False
            try:
                self.save_checkpoint(step=global_step, epoch=epoch)
                checkpoint_saved = True
                print(f"{self.log_prefix} Checkpoint saved successfully")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to save checkpoint: {e}")
                import traceback
                traceback.print_exc()

            # Try to save training state (independent of checkpoint save)
            # Note: If stopped mid-MNT, skip the current batch and resume from next batch
            # This is acceptable as MNT iterations are gradient accumulation (can skip partial progress)
            state_saved = False
            try:
                self.save_training_state(step=global_step, epoch=epoch, batch_idx=self._epoch_batch_position(batch_idx), multi_noise_timesteps=multi_noise_timesteps)
                state_saved = True
                print(f"{self.log_prefix} Training state saved successfully")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to save training state: {e}")

            # Try to save optimizer state (independent of checkpoint/state save)
            optimizer_saved = False
            try:
                self.save_optimizer_state(step=global_step)
                self.save_ema_state(step=global_step)
                self._save_ema_checkpoint(step=global_step, epoch=epoch)
                optimizer_saved = True
                print(f"{self.log_prefix} Optimizer state saved successfully")
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to save optimizer state: {e}")
                import traceback
                traceback.print_exc()

            # Try to cleanup old checkpoints (even if above failed)
            try:
                self._cleanup_old_checkpoints(max_step_saves_to_keep)
            except Exception as e:
                print(f"{self.log_prefix} ERROR: Failed to cleanup old checkpoints: {e}")
                import traceback
                traceback.print_exc()

            if checkpoint_saved and state_saved:
                print(f"{self.log_prefix} Checkpoint and state saved successfully, exiting...")
            elif checkpoint_saved:
                print(f"{self.log_prefix} Checkpoint saved (but state save failed), exiting...")
            elif state_saved:
                print(f"{self.log_prefix} State saved (but checkpoint save failed), exiting...")
            else:
                print(f"{self.log_prefix} WARNING: Both checkpoint and state save failed, exiting...")

            self.writer.close()
            raise

        except Exception as e:
            # Stop cpu_prefetch worker on failure too
            try:
                if 'te_prefetcher' in locals() and te_prefetcher is not None:
                    te_prefetcher.stop()
            except Exception:
                pass
            try:
                if getattr(self, "_danbooru_collector", None) is not None:
                    self._danbooru_collector.stop()
            except Exception:
                pass
            if isinstance(e, PartialOptimizerStepError):
                # The ordinary emergency checkpoint would be the half-applied
                # weights this refusal exists to keep off disk; the quarantine
                # decision (salvage vs. write nothing) is made below, keyed off
                # the same _partial_step_taint this exception's raise site set.
                print(f"\n{self.log_prefix} [FAILED] {e}")
                self._refuse_save_after_partial_step(
                    f"{type(e).__name__} was raised", global_step, epoch)
                self.writer.close()
                raise

            if self._refuse_save_after_partial_step(
                    f"{type(e).__name__} was raised", global_step, epoch):
                self.writer.close()
                raise

            if isinstance(e, NothingTrainedError):
                # No backward completed, so the weights are the base model's.
                # Writing them back out is the expensive no-op this refusal
                # exists to prevent.
                print(f"\n{self.log_prefix} [FAILED] {e}")
                print(f"{self.log_prefix} [FAILED] No checkpoint written: nothing was trained")
                self.writer.close()
                raise

            # Emergency checkpoint save on any unhandled exception (CUDA errors, etc.)
            print(f"\n{self.log_prefix} [EMERGENCY] Training failed with error: {type(e).__name__}: {str(e)[:200]}")
            print(f"{self.log_prefix} [EMERGENCY] Attempting to save emergency checkpoint at step {global_step}, epoch {epoch}...")

            # Probe FIRST: a "fatal" CUDA error (FatalCudaError) means the context is
            # presumed dead, but even a plain Exception could be a fatal CUDA error
            # that unwound through non-RuntimeError frames, so always probe rather
            # than trusting the exception type alone.
            _ctx_alive = self._cuda_context_alive() if self._cuda_is_available() else True

            if not _ctx_alive:
                print(f"{self.log_prefix} [EMERGENCY] CUDA context corrupted; weights at step "
                      f"{global_step} are unrecoverable -- resume from the last periodic checkpoint")
                # Every CUDA-touching step below is skipped: moving modules to CPU,
                # cuda.synchronize/empty_cache, save_checkpoint (model weights),
                # save_optimizer_state / save_ema_state (optimizer/EMA tensors are on
                # CUDA), all require a working CUDA context and would themselves raise
                # (this is exactly how run112's emergency save cascade-failed).
                checkpoint_saved = False
                optimizer_saved = False
                # Do NOT write a training-state JSON here either: it would record
                # global_step/epoch/batch_idx with NO matching model checkpoint
                # (checkpoint_saved=False above), and a mid-epoch resume pairs
                # state.json's batch_idx with the model weights of the SAME step --
                # writing it would actively mislead resume into either failing to
                # find weights at that step or (worse) silently resuming from an
                # older checkpoint's weights with a newer/mismatched batch_idx.
                # Skipping it here means resume cleanly falls back to the last
                # periodic checkpoint + its own (paired, valid) state.json.
                state_saved = False
                print(f"{self.log_prefix} [EMERGENCY] Skipping model/optimizer/state save "
                      f"(dead CUDA context) -- only closing CPU-side resources")
                # Best-effort: remove a directory-style checkpoint for THIS step if
                # some earlier stage (e.g. a periodic save racing the same failure)
                # left an incomplete one (mkdir'd but never populated with weights --
                # see controlnet_sdxl_adapter._save_standard_checkpoint). File-style
                # checkpoints (.safetensors) are not touched; they aren't at risk of
                # this partial-mkdir failure mode.
                self._cleanup_incomplete_step_checkpoint_dir(global_step)
            else:
                # For CUDA errors, first try to move model to CPU to free GPU memory
                try:
                    print(f"{self.log_prefix} [EMERGENCY] Moving model to CPU to free GPU memory...")
                    self.move_main_model_to_cpu()
                    self.move_text_encoder_to_cpu()
                    self.move_vae_to_cpu()
                except Exception as move_error:
                    print(f"{self.log_prefix} [EMERGENCY] Failed to move model to CPU: {move_error}")

                # Try to clear CUDA cache (may fail if context is corrupted)
                try:
                    import gc
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # Ignore - CUDA may be in bad state

                # Try to save checkpoint
                checkpoint_saved = False
                try:
                    self.save_checkpoint(step=global_step, epoch=epoch)
                    checkpoint_saved = True
                    print(f"{self.log_prefix} [EMERGENCY] Checkpoint saved successfully")
                except Exception as save_error:
                    print(f"{self.log_prefix} [EMERGENCY] Failed to save checkpoint: {save_error}")
                    import traceback
                    traceback.print_exc()
                    # The save may have mkdir'd a directory-style checkpoint before
                    # failing mid-write; don't leave it for resume to trip over.
                    self._cleanup_incomplete_step_checkpoint_dir(global_step)

                # Try to save training state
                state_saved = False
                try:
                    self.save_training_state(step=global_step, epoch=epoch, batch_idx=self._epoch_batch_position(batch_idx), multi_noise_timesteps=multi_noise_timesteps)
                    state_saved = True
                    print(f"{self.log_prefix} [EMERGENCY] Training state saved successfully")
                except Exception as state_error:
                    print(f"{self.log_prefix} [EMERGENCY] Failed to save training state: {state_error}")

                # Try to save optimizer state
                optimizer_saved = False
                try:
                    self.save_optimizer_state(step=global_step)
                    self.save_ema_state(step=global_step)
                    self._save_ema_checkpoint(step=global_step, epoch=epoch)
                    optimizer_saved = True
                    print(f"{self.log_prefix} [EMERGENCY] Optimizer state saved successfully")
                except Exception as opt_error:
                    print(f"{self.log_prefix} [EMERGENCY] Failed to save optimizer state: {opt_error}")

                # Try to cleanup old checkpoints (only if this emergency save actually
                # wrote a new checkpoint -- mirrors the KeyboardInterrupt handler above).
                # Wrapped so a cleanup failure can never mask the original emergency.
                if checkpoint_saved:
                    try:
                        self._cleanup_old_checkpoints(max_step_saves_to_keep)
                    except Exception as cleanup_error:
                        print(f"{self.log_prefix} [EMERGENCY] Failed to cleanup old checkpoints: {cleanup_error}")
                        import traceback
                        traceback.print_exc()

            # Summary
            if checkpoint_saved or state_saved or optimizer_saved:
                saved_items = []
                if checkpoint_saved:
                    saved_items.append("checkpoint")
                if state_saved:
                    saved_items.append("state")
                if optimizer_saved:
                    saved_items.append("optimizer")
                print(f"{self.log_prefix} [EMERGENCY] Saved: {', '.join(saved_items)}")
                print(f"{self.log_prefix} [EMERGENCY] Training can be resumed from step {global_step}")
            else:
                print(f"{self.log_prefix} [EMERGENCY] WARNING: All save attempts failed!")
                print(f"{self.log_prefix} [EMERGENCY] Training progress may be lost")

            self.writer.close()
            raise  # Re-raise the original exception

        print(f"{self.log_prefix} Training complete!")

        # Cleanup resources
        self.cleanup()

    def _drop_unfittable_batches(self, batches):
        """Drop batches in buckets that OOM'd at batch size 1, refusing if none survive.

        Covers the non-crop path and backs up the crop re-bucketing, which
        already excludes such items. Idempotent. Emptying the epoch is raised,
        not logged: the loop below would otherwise iterate over nothing and the
        run would report success having trained nothing. Which exception depends
        on whether anything HAS been trained -- see BucketsExhaustedError.
        """
        if not self._unfittable_buckets or not batches:
            return batches

        def _bucket_of(_b):
            try:
                it = _b[0][0] if isinstance(_b[0], tuple) else _b[0]
                return (it.get("bucket_width") or it.get("width"),
                        it.get("bucket_height") or it.get("height"))
            except Exception:
                return None

        n0 = len(batches)
        kept = [b for b in batches if _bucket_of(b) not in self._unfittable_buckets]
        if len(kept) < n0:
            print(f"{self.log_prefix} [OOM] dropped {n0 - len(kept)} batch(es) in "
                  f"un-fittable buckets ({len(self._unfittable_buckets)} excluded)")
        if not kept:
            if getattr(self, "_backwards_completed", 0) > 0:
                raise BucketsExhaustedError(self._buckets_exhausted_message(n0))
            raise NothingTrainedError(self._nothing_trainable_message(n0))
        return kept

    def _assert_trained_something(self):
        """Refuse to report success for a run that completed no backward pass.

        Independent of the optimizer update census, which is opt-in and skips
        abandoned batches by design. Covers every whole-batch skip -- OOM,
        corrupted image, no valid latents, missing condition images -- not only
        the OOM bucket exclusion. An empty epoch range (a resume at or past the
        last epoch) is a legitimate no-op and is not caught.
        """
        if getattr(self, "_epochs_entered", 0) > 0 and getattr(self, "_backwards_completed", 0) == 0:
            raise NothingTrainedError(self._nothing_trainable_message(0))

    def _nothing_trainable_message(self, batches_before_drop: int = 0) -> str:
        """Message for a run that has nothing left to train on.

        Deliberately does not claim the card is exhausted: the excluding OOM is
        raised against whatever budget the process runs under, which
        ``set_per_process_memory_fraction`` can put well below the installed
        VRAM.
        """
        buckets = ", ".join(f"{w}x{h}" for w, h in sorted(self._unfittable_buckets))
        if buckets:
            head = (f"Training has nothing left to run: every resolution bucket that "
                    f"remained OOM'd at batch size 1 and was excluded ({buckets}).")
        else:
            head = ("Training has nothing left to run: no batch completed a backward "
                    "pass (every batch was skipped or dropped, or the dataset "
                    "produced none).")
        if batches_before_drop:
            head += f" All {batches_before_drop} batch(es) of this epoch were dropped."
        skipped = getattr(self, "_batches_skipped", 0)
        if skipped:
            head += (f" {skipped} batch(es) were skipped before their backward pass "
                     f"(OOM, corrupted image, no valid latents, or missing condition "
                     f"images -- see the WARNING lines above).")
        return (
            head + " No parameter was updated, so this run is failed rather than "
            "reported complete. " + _MEMORY_BUDGET_ADVICE
        )

    def _buckets_exhausted_message(self, batches_before_drop: int = 0) -> str:
        """Message for a run that trained, then lost its last fittable bucket.

        Must not repeat _nothing_trainable_message's "no parameter was updated":
        here thousands may have been, and the emergency checkpoint about to be
        written is what carries them.
        """
        buckets = ", ".join(f"{w}x{h}" for w, h in sorted(self._unfittable_buckets))
        done = getattr(self, "_backwards_completed", 0)
        head = (f"Training cannot continue: every remaining resolution bucket has "
                f"OOM'd at batch size 1 and been excluded ({buckets}).")
        if batches_before_drop:
            head += f" All {batches_before_drop} batch(es) of this epoch were dropped."
        return (
            head + f" {done} backward pass(es) completed before this, so the weights "
            f"are NOT the base model's -- an emergency checkpoint is being written "
            f"to preserve that work, and this run is failed rather than reported "
            f"complete. " + _MEMORY_BUDGET_ADVICE
        )

    def _full_parameter_grad_components(self):
        """``id(param)`` -> component, from the full-parameter adapter. Cached.

        The trainable set of a full fine-tune is fixed once
        ``prepare_models_for_training`` has run, so this is built once per run
        rather than per step. A diagnostic must not be able to abort training:
        an adapter that raises while classifying (SenseNova's re-resolves its
        scope, which asserts the materialized Linear count) is reported once and
        then treated as having no opinion, which is the pre-existing
        module-derived bucketing.
        """
        cached = getattr(self, '_full_param_grad_components', None)
        if cached is not None:
            return cached
        components = {}
        resolve = getattr(getattr(self, 'adapter', None), 'grad_norm_components', None)
        if callable(resolve):
            try:
                components = resolve() or {}
            except Exception as exc:
                emit_training_warning(
                    f"{type(self.adapter).__name__}.grad_norm_components() failed "
                    f"({type(exc).__name__}: {exc}); per-component gradient norms "
                    f"fall back to one bucket per module for this run",
                    code="grad_norm_components_failed",
                    prefix=getattr(self, 'log_prefix', '[Trainer]'),
                )
                components = {}
        self._full_param_grad_components = components
        return components

    def _calculate_grad_norms(self):
        """
        Calculate gradient norms for different parameter groups.

        Returns:
            Tuple of (total_grad_norm, text_encoder_grad_norm, text_encoder_1_grad_norm,
                      text_encoder_2_grad_norm, unet_grad_norm, vision_encoder_grad_norm)
            text_encoder_1/2 are non-zero only where the two encoders are
            distinguishable (SDXL LoRA + Full FT, SD1.5 which reports its single
            CLIP as TE1, SenseNova whose understanding MoT half is its prompt
            encoder and reports as TE1 under both training methods); other
            architectures' TE LoRA lands in the combined text_encoder bucket.

        Under fused backward the squared norms come from the accumulator the
        hooks filled before clearing each gradient; otherwise they are measured
        from the live gradients. Either way this method decides the components,
        and the squares are read in one device->host sync rather than one per
        parameter.
        """
        # id(param) -> ||grad||^2, or None when it has to be measured below.
        recorded = None
        if fused_backward_active(self):
            accumulator = getattr(self, "_fused_grad_norm", None)
            recorded = accumulator.squared_norms() if accumulator is not None else {}

        def _has_grad(param):
            if recorded is not None:
                return id(param) in recorded
            return param.grad is not None

        entries: List[Tuple[Any, str]] = []  # (param, 'unet'|'te'|'te1'|'te2'|'ve')

        # For LoRA training, iterate through lora_layers dict
        if hasattr(self, 'lora_layers'):
            grad_count = 0
            # Components come from the adapter that injected each layer. Inferring
            # them from substrings of the LoRA key ('unet'/'transformer'/'te1_')
            # mis-binned every architecture whose keys are plain module paths
            # (SenseNova) or use another prefix (FLUX.2/MiniT2I text encoders):
            # they landed in the total only, leaving grad_norm_unet at 0.0.
            # Local import: core.training.adapters pulls every arch's model
            # modules, which base_trainer must not require at module load.
            from core.training.adapters.base_adapter import LORA_COMPONENT_UNET
            components = getattr(getattr(self, 'adapter', None), 'lora_components', None) or {}
            unclassified = []
            for lora_name, lora_layer in self.lora_layers.items():
                component = components.get(lora_name)
                if component is None:
                    unclassified.append(lora_name)
                    component = LORA_COMPONENT_UNET  # main trainable model
                for param in lora_layer.parameters():
                    if _has_grad(param):
                        grad_count += 1

                        entries.append((param, grad_norm_bucket(component)))

            if unclassified and not hasattr(self, '_grad_norm_unclassified_warned'):
                print(f"{self.log_prefix} [GradNorm] WARNING: {len(unclassified)} LoRA layer(s) "
                      f"were injected without a registered component and are being reported "
                      f"under grad_norm_unet; the adapter should call register_lora_layer(). "
                      f"Examples: {unclassified[:3]}")
                self._grad_norm_unclassified_warned = True

            # Debug: Print first calculation only
            if grad_count > 0 and not hasattr(self, '_grad_norm_debug_printed'):
                print(f"{self.log_prefix} [GradNorm] Calculated from {grad_count} parameters with gradients")
                print(f"{self.log_prefix} [GradNorm] Sample LoRA layer names (first 3):")
                for i, name in enumerate(list(self.lora_layers.keys())[:3]):
                    print(f"{self.log_prefix}   {name}")
                self._grad_norm_debug_printed = True

        # For Full Fine-Tuning, iterate through base model parameters
        else:
            # SD1.5/SDXL: Direct text_encoder access — treat as TE1
            if hasattr(self, 'text_encoder') and self.text_encoder is not None:
                for name, param in self.text_encoder.named_parameters():
                    if _has_grad(param):
                        entries.append((param, 'te1'))

            # Iterate through text encoder 2 parameters (if trainable, SDXL) — TE2
            if hasattr(self, 'text_encoder_2') and self.text_encoder_2 is not None:
                for name, param in self.text_encoder_2.named_parameters():
                    if _has_grad(param):
                        entries.append((param, 'te2'))

            # Iterate through U-Net parameters (if trainable, SD1.5/SDXL)
            if hasattr(self, 'unet') and self.unet is not None:
                for name, param in self.unet.named_parameters():
                    if _has_grad(param):
                        entries.append((param, 'unet'))

            # Iterate through Transformer parameters (if trainable, Z-Image)
            if hasattr(self, 'transformer_original') and self.transformer_original is not None:
                for name, param in self.transformer_original.named_parameters():
                    if _has_grad(param):
                        entries.append((param, 'unet'))

            # Iterate through Vision Encoder parameters (if training VE, SD1.5/SDXL only)
            if getattr(self, '_train_vision_encoder', False) and getattr(self, 'vision_encoder', None) is not None:
                for param in self.vision_encoder.parameters():
                    if _has_grad(param):
                        entries.append((param, 've'))

            # Iterate through ControlNet parameters (ControlNet training freezes
            # UNet/TE/VAE and trains self.controlnet, so none of the branches above
            # catch its grads -> total_grad_norm was 0.0 for every CN step. Report
            # the CN grad norm under both the total and the "unet" (main trainable
            # model) slot so the convergence signal is usable.
            if getattr(self, 'controlnet', None) is not None:
                for param in self.controlnet.parameters():
                    if _has_grad(param):
                        entries.append((param, 'unet'))

            # The loops above bucket by the MODULE a parameter was found on,
            # which is right only where one module is one component. SenseNova
            # keeps both MoT halves inside transformer_original, so the
            # understanding half was reported as U-Net and no separate
            # MoT-Understanding norm existed for a `und` or `both` run. The
            # adapter that built the optimizer groups classifies its own
            # parameters; every adapter that does not override it returns {} and
            # nothing below changes.
            overrides = self._full_parameter_grad_components()
            if overrides:
                entries = [
                    (param, grad_norm_bucket(overrides[id(param)]))
                    if id(param) in overrides else (param, bucket)
                    for param, bucket in entries
                ]

        if recorded is None:
            from .optimizers.fused_grad_norm import squared_norms_from_grads
            recorded = squared_norms_from_grads(param for param, _ in entries)

        total_grad_norm = 0.0
        text_encoder_grad_norm = 0.0
        text_encoder_1_grad_norm = 0.0
        text_encoder_2_grad_norm = 0.0
        unet_grad_norm = 0.0
        vision_encoder_grad_norm = 0.0
        for param, bucket in entries:
            square = recorded.get(id(param), 0.0)
            total_grad_norm += square
            if bucket == 'te1':
                text_encoder_grad_norm += square
                text_encoder_1_grad_norm += square
            elif bucket == 'te2':
                text_encoder_grad_norm += square
                text_encoder_2_grad_norm += square
            elif bucket == 'te':
                text_encoder_grad_norm += square
            elif bucket == 've':
                vision_encoder_grad_norm += square
            else:
                unet_grad_norm += square

        # Take square root to get L2 norm
        total_grad_norm = total_grad_norm ** 0.5
        text_encoder_grad_norm = text_encoder_grad_norm ** 0.5
        text_encoder_1_grad_norm = text_encoder_1_grad_norm ** 0.5
        text_encoder_2_grad_norm = text_encoder_2_grad_norm ** 0.5
        unet_grad_norm = unet_grad_norm ** 0.5
        vision_encoder_grad_norm = vision_encoder_grad_norm ** 0.5

        # Debug: Print values once
        if not hasattr(self, '_grad_norm_values_printed'):
            print(f"{self.log_prefix} [GradNorm] Total: {total_grad_norm:.6f}, TE: {text_encoder_grad_norm:.6f}, TE1: {text_encoder_1_grad_norm:.6f}, TE2: {text_encoder_2_grad_norm:.6f}, UNet: {unet_grad_norm:.6f}, VE: {vision_encoder_grad_norm:.6f}")
            self._grad_norm_values_printed = True

        return total_grad_norm, text_encoder_grad_norm, text_encoder_1_grad_norm, text_encoder_2_grad_norm, unet_grad_norm, vision_encoder_grad_norm

    def log_extra_metric(self, name: str, value):
        """Record a bespoke, arch/method-specific per-step scalar metric.

        This is the SINGLE producer hook for optional metrics that are not
        universal across trainers (e.g. REPA alignment for MiniT2I, the
        generate-region-only MSE for outpaint ControlNet). The value is routed
        generically into TrainingMetrics.extra_metrics (a {name: float} JSON
        dict) and surfaced on the loss chart via
        core.training.metric_registry.EXTRA_METRIC_DEFS — so a new metric needs
        no DB column, no API/param threading, and no chart change.

        Non-finite values (NaN/inf) are dropped: SQLite's JSON1 functions reject
        the non-standard ``NaN`` token json.dumps would emit.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(v):
            return
        self._extra_metrics[name] = v

    def _log_metrics_to_db(
        self,
        step: int,
        loss: float = None,
        recon_loss: float = None,
        learning_rate: float = None,
        grad_norm: float = None,
        grad_norm_text_encoder: float = None,
        grad_norm_text_encoder_1: float = None,
        grad_norm_text_encoder_2: float = None,
        grad_norm_unet: float = None,
        grad_norm_vision_encoder: float = None,
        param_update_norm_unet: float = None,
        param_update_norm_te1: float = None,
        param_update_norm_te2: float = None,
        param_update_norm_ve: float = None,
        param_cumulative_drift_unet: float = None,
        param_cumulative_drift_te1: float = None,
        param_cumulative_drift_te2: float = None,
        param_cumulative_drift_ve: float = None,
        force_flush: bool = False
    ):
        """
        Log training metrics to database with buffering (dual logging: TensorBoard + DB).

        OPTIMIZED: Buffers metrics and batch commits every N steps to reduce I/O overhead.
        This reduces DB operations from every step to every _metrics_flush_interval steps.

        Features:
        - UPSERT behavior: Same (run_id, step) will overwrite existing values
        - Allows training restart from checkpoint without duplicating metrics
        - Fast queries: indexed by (run_id, step) for incremental fetching
        - Partial update: If a parameter is None, existing value is preserved
        - Buffered commits: Batches DB writes for performance

        Args:
            step: Global training step
            loss: Prediction loss value (MSE with Min-SNR weighting), None to keep existing
            recon_loss: Reconstruction loss value, None to keep existing
            learning_rate: Current learning rate, None to keep existing
            grad_norm: Total gradient norm, None to keep existing
            grad_norm_text_encoder: Text encoder gradient norm, None to keep existing
            grad_norm_unet: U-Net/Transformer gradient norm, None to keep existing
            grad_norm_vision_encoder: Vision Encoder gradient norm, None to keep existing
            force_flush: If True, flush buffer immediately (for checkpoints, end of training)

        Note:
            The 'loss' parameter stores prediction loss (not combined loss).
            This allows monitoring pred_loss and recon_loss separately in DB.
            Combined loss can be calculated as: (1-β)*loss + β*recon_loss
        """
        # Buffer the metrics (merge if same step already exists in buffer)
        # This handles the case where loss and grad_norm are logged separately for the same step
        existing_entry = None
        for entry in self._metrics_buffer:
            if entry['step'] == step:
                existing_entry = entry
                break

        if existing_entry is not None:
            # Merge: update existing entry with new non-None values
            if loss is not None:
                existing_entry['loss'] = loss
            if recon_loss is not None:
                existing_entry['recon_loss'] = recon_loss
            if learning_rate is not None:
                existing_entry['learning_rate'] = learning_rate
            if grad_norm is not None:
                existing_entry['grad_norm'] = grad_norm
            if grad_norm_text_encoder is not None:
                existing_entry['grad_norm_text_encoder'] = grad_norm_text_encoder
            if grad_norm_text_encoder_1 is not None:
                existing_entry['grad_norm_text_encoder_1'] = grad_norm_text_encoder_1
            if grad_norm_text_encoder_2 is not None:
                existing_entry['grad_norm_text_encoder_2'] = grad_norm_text_encoder_2
            if grad_norm_unet is not None:
                existing_entry['grad_norm_unet'] = grad_norm_unet
            if grad_norm_vision_encoder is not None:
                existing_entry['grad_norm_vision_encoder'] = grad_norm_vision_encoder
            if param_update_norm_unet is not None:
                existing_entry['param_update_norm_unet'] = param_update_norm_unet
            if param_update_norm_te1 is not None:
                existing_entry['param_update_norm_te1'] = param_update_norm_te1
            if param_update_norm_te2 is not None:
                existing_entry['param_update_norm_te2'] = param_update_norm_te2
            if param_update_norm_ve is not None:
                existing_entry['param_update_norm_ve'] = param_update_norm_ve
            if param_cumulative_drift_unet is not None:
                existing_entry['param_cumulative_drift_unet'] = param_cumulative_drift_unet
            if param_cumulative_drift_te1 is not None:
                existing_entry['param_cumulative_drift_te1'] = param_cumulative_drift_te1
            if param_cumulative_drift_te2 is not None:
                existing_entry['param_cumulative_drift_te2'] = param_cumulative_drift_te2
            if param_cumulative_drift_ve is not None:
                existing_entry['param_cumulative_drift_ve'] = param_cumulative_drift_ve
            # Merge any bespoke metrics accumulated for this same step (per key),
            # then clear the accumulator so it isn't re-applied on a later call.
            if self._extra_metrics:
                existing_entry['extra'] = {**(existing_entry.get('extra') or {}), **self._extra_metrics}
                self._extra_metrics = {}
        else:
            # New step: add to buffer. epoch/resume_seq are run-context attributes
            # (set in the epoch loop / at run start) rather than per-call args, so
            # the many existing call sites stay unchanged.
            self._metrics_buffer.append({
                'step': step,
                'epoch': getattr(self, '_current_epoch', None),
                'resume_seq': getattr(self, 'resume_seq', 0),
                'loss': loss,
                'recon_loss': recon_loss,
                # Bespoke arch/method-specific scalars (REPA, outpaint gen_loss, …)
                # accumulated via log_extra_metric() this step. Captured by value;
                # the accumulator is cleared below so metrics emitted only some
                # steps never carry stale values forward.
                'extra': dict(self._extra_metrics) if self._extra_metrics else None,
                'learning_rate': learning_rate,
                'grad_norm': grad_norm,
                'grad_norm_text_encoder': grad_norm_text_encoder,
                'grad_norm_text_encoder_1': grad_norm_text_encoder_1,
                'grad_norm_text_encoder_2': grad_norm_text_encoder_2,
                'grad_norm_unet': grad_norm_unet,
                'grad_norm_vision_encoder': grad_norm_vision_encoder,
                'param_update_norm_unet': param_update_norm_unet,
                'param_update_norm_te1': param_update_norm_te1,
                'param_update_norm_te2': param_update_norm_te2,
                'param_update_norm_ve': param_update_norm_ve,
                'param_cumulative_drift_unet': param_cumulative_drift_unet,
                'param_cumulative_drift_te1': param_cumulative_drift_te1,
                'param_cumulative_drift_te2': param_cumulative_drift_te2,
                'param_cumulative_drift_ve': param_cumulative_drift_ve,
            })
            # Reset the per-step extra-metric accumulator now that it is captured.
            if self._extra_metrics:
                self._extra_metrics = {}

        # Only flush when buffer is full or force_flush is requested
        should_flush = force_flush or len(self._metrics_buffer) >= self._metrics_flush_interval
        if not should_flush:
            return

        # Copy buffer and clear immediately (so training can continue adding to new buffer)
        buffer_to_flush = self._metrics_buffer.copy()
        self._metrics_buffer = []

        if force_flush:
            # Synchronous flush for checkpoints/end of training (ensure data is written)
            self._flush_metrics_to_db(buffer_to_flush)
        else:
            # Async flush: submit to background thread, don't block training
            # Clean up completed futures first
            self._db_futures = [f for f in self._db_futures if not f.done()]
            future = self._db_executor.submit(self._flush_metrics_to_db, buffer_to_flush)
            self._db_futures.append(future)

    def _flush_metrics_to_db(self, buffer: list):
        """
        Actually flush metrics buffer to database (runs in background thread).

        Args:
            buffer: List of metrics dicts to flush
        """
        if not buffer:
            return

        try:
            from database.models import TrainingMetrics
            from database import get_training_db

            # Get database session
            db = next(get_training_db())

            for metrics in buffer:
                m_step = metrics['step']
                m_loss = metrics['loss']
                m_recon_loss = metrics['recon_loss']
                m_extra = metrics.get('extra')
                m_learning_rate = metrics['learning_rate']
                m_grad_norm = metrics['grad_norm']
                m_grad_norm_te = metrics['grad_norm_text_encoder']
                m_grad_norm_te1 = metrics.get('grad_norm_text_encoder_1')
                m_grad_norm_te2 = metrics.get('grad_norm_text_encoder_2')
                m_grad_norm_unet = metrics['grad_norm_unet']
                m_grad_norm_ve = metrics.get('grad_norm_vision_encoder')
                m_param_upd_unet = metrics.get('param_update_norm_unet')
                m_param_upd_te1  = metrics.get('param_update_norm_te1')
                m_param_upd_te2  = metrics.get('param_update_norm_te2')
                m_param_upd_ve   = metrics.get('param_update_norm_ve')
                m_param_dft_unet = metrics.get('param_cumulative_drift_unet')
                m_param_dft_te1  = metrics.get('param_cumulative_drift_te1')
                m_param_dft_te2  = metrics.get('param_cumulative_drift_te2')
                m_param_dft_ve   = metrics.get('param_cumulative_drift_ve')

                # UPSERT: Check if metric exists for this (run_id, step)
                existing = db.query(TrainingMetrics).filter(
                    TrainingMetrics.run_id == self.run_id,
                    TrainingMetrics.step == m_step
                ).first()

                if existing:
                    # Update existing metric (training restarted from checkpoint)
                    if m_loss is not None:
                        existing.loss = m_loss
                    if m_recon_loss is not None:
                        existing.recon_loss = m_recon_loss
                    if m_extra:
                        # Reassign a NEW dict so SQLAlchemy sees the mutation
                        # (in-place edits of a JSON column are not tracked).
                        existing.extra_metrics = {**(existing.extra_metrics or {}), **m_extra}
                    if m_learning_rate is not None:
                        existing.learning_rate = m_learning_rate
                    if m_grad_norm is not None:
                        existing.grad_norm = m_grad_norm
                    if m_grad_norm_te is not None:
                        existing.grad_norm_text_encoder = m_grad_norm_te
                    if m_grad_norm_te1 is not None:
                        existing.grad_norm_text_encoder_1 = m_grad_norm_te1
                    if m_grad_norm_te2 is not None:
                        existing.grad_norm_text_encoder_2 = m_grad_norm_te2
                    if m_grad_norm_unet is not None:
                        existing.grad_norm_unet = m_grad_norm_unet
                    if m_grad_norm_ve is not None:
                        existing.grad_norm_vision_encoder = m_grad_norm_ve
                    if m_param_upd_unet is not None:
                        existing.param_update_norm_unet = m_param_upd_unet
                    if m_param_upd_te1 is not None:
                        existing.param_update_norm_te1 = m_param_upd_te1
                    if m_param_upd_te2 is not None:
                        existing.param_update_norm_te2 = m_param_upd_te2
                    if m_param_upd_ve is not None:
                        existing.param_update_norm_ve = m_param_upd_ve
                    if m_param_dft_unet is not None:
                        existing.param_cumulative_drift_unet = m_param_dft_unet
                    if m_param_dft_te1 is not None:
                        existing.param_cumulative_drift_te1 = m_param_dft_te1
                    if m_param_dft_te2 is not None:
                        existing.param_cumulative_drift_te2 = m_param_dft_te2
                    if m_param_dft_ve is not None:
                        existing.param_cumulative_drift_ve = m_param_dft_ve
                    existing.timestamp = datetime.now()
                else:
                    # Insert new metric
                    metric = TrainingMetrics(
                        run_id=self.run_id,
                        step=m_step,
                        epoch=metrics.get('epoch'),
                        resume_seq=metrics.get('resume_seq', 0),
                        loss=m_loss if m_loss is not None else 0.0,
                        recon_loss=m_recon_loss if m_recon_loss is not None else 0.0,
                        extra_metrics=(m_extra or None),
                        learning_rate=m_learning_rate if m_learning_rate is not None else 0.0,
                        grad_norm=m_grad_norm,
                        grad_norm_text_encoder=m_grad_norm_te,
                        grad_norm_text_encoder_1=m_grad_norm_te1,
                        grad_norm_text_encoder_2=m_grad_norm_te2,
                        grad_norm_unet=m_grad_norm_unet,
                        grad_norm_vision_encoder=m_grad_norm_ve,
                        param_update_norm_unet=m_param_upd_unet,
                        param_update_norm_te1=m_param_upd_te1,
                        param_update_norm_te2=m_param_upd_te2,
                        param_update_norm_ve=m_param_upd_ve,
                        param_cumulative_drift_unet=m_param_dft_unet,
                        param_cumulative_drift_te1=m_param_dft_te1,
                        param_cumulative_drift_te2=m_param_dft_te2,
                        param_cumulative_drift_ve=m_param_dft_ve,
                    )
                    db.add(metric)

            # Single commit for entire buffer
            db.commit()
            db.close()

            # Broadcast latest metrics to WebSocket clients
            # Only send the most recent entry to avoid flooding
            if buffer:
                latest = buffer[-1]
                try:
                    from api.websocket import manager as ws_manager
                    ws_manager.send_training_metrics(
                        run_id=self.run_id,
                        step=latest['step'],
                        loss=latest['loss'],
                        recon_loss=latest['recon_loss'],
                        extra=latest.get('extra'),
                        learning_rate=latest['learning_rate'],
                        grad_norm=latest['grad_norm'],
                        grad_norm_text_encoder=latest['grad_norm_text_encoder'],
                        grad_norm_text_encoder_1=latest.get('grad_norm_text_encoder_1'),
                        grad_norm_text_encoder_2=latest.get('grad_norm_text_encoder_2'),
                        grad_norm_unet=latest['grad_norm_unet'],
                        grad_norm_vision_encoder=latest.get('grad_norm_vision_encoder'),
                        epoch=latest.get('epoch'),
                        resume_seq=latest.get('resume_seq', 0),
                    )
                except Exception:
                    pass  # Non-critical

        except Exception as e:
            # Non-critical: Continue training even if DB logging fails
            print(f"{self.log_prefix} WARNING: Failed to log metrics to DB: {e}")

    def _shutdown_db_executor(self):
        """Shutdown the DB executor and wait for pending writes to complete."""
        if hasattr(self, '_db_executor') and self._db_executor is not None:
            # Wait for all pending futures
            from concurrent.futures import wait
            if self._db_futures:
                wait(self._db_futures, timeout=30)  # Wait up to 30 seconds
            self._db_executor.shutdown(wait=True)
            self._db_executor = None

    def _cleanup_future_metrics(self, current_step: int):
        """
        Clean up future metrics in database (old data from previous interrupted training).

        When training resumes from an earlier step (e.g., resume from step 100 when previous
        run reached step 500), the UPSERT logic will overwrite steps 1-100, but steps 101-500
        from the old run will remain in the database, causing duplicate/stale data.

        This method removes all metrics with step > current_step to prevent this issue.

        Args:
            current_step: Current global step (resume point)
        """
        try:
            from database.models import TrainingMetrics
            from database import get_training_db

            # Get database session
            db = next(get_training_db())

            # Find future metrics (step > current_step)
            future_metrics = db.query(TrainingMetrics).filter(
                TrainingMetrics.run_id == self.run_id,
                TrainingMetrics.step > current_step
            ).all()

            if future_metrics:
                # Get range for logging
                future_steps = [m.step for m in future_metrics]
                min_future_step = min(future_steps)
                max_future_step = max(future_steps)

                print(f"{self.log_prefix} Found {len(future_metrics)} old metrics (steps {min_future_step}-{max_future_step}) beyond current step {current_step}")
                print(f"{self.log_prefix} Cleaning up old metrics to prevent duplicates...")

                # Delete future metrics
                for metric in future_metrics:
                    db.delete(metric)

                db.commit()
                print(f"{self.log_prefix} Deleted {len(future_metrics)} old metrics")
            else:
                print(f"{self.log_prefix} No old metrics beyond current step {current_step} (clean start)")

            db.close()

        except Exception as e:
            # Non-critical: Log warning but continue training
            print(f"{self.log_prefix} WARNING: Failed to cleanup future metrics: {e}")

    def cleanup(self):
        """
        Cleanup training resources.

        - Flush metrics buffer to database
        - Remove Layer Offload Conductor hooks
        - Restore layers to GPU
        - Close TensorBoard writer
        """
        print(f"{self.log_prefix} Cleaning up training resources...")

        if getattr(self, "sensenova_phase_evictor", None) is not None:
            try:
                self.sensenova_phase_evictor.teardown()
            except Exception as exc:
                print(f"{self.log_prefix} WARNING: SenseNova eviction teardown failed: {exc}")
            finally:
                self.sensenova_phase_evictor = None

        # Flush any remaining metrics to database
        if hasattr(self, '_metrics_buffer') and self._metrics_buffer and self.run_id is not None:
            print(f"{self.log_prefix} Flushing {len(self._metrics_buffer)} remaining metrics to database...")
            self._log_metrics_to_db(step=0, force_flush=True)  # step ignored, force_flush processes buffer

        # Shutdown DB executor (wait for async writes to complete)
        self._shutdown_db_executor()

        # Cleanup Layer Offload Conductor
        if hasattr(self, 'layer_offload_conductor') and self.layer_offload_conductor is not None:
            print(f"{self.log_prefix} Cleaning up LayerOffloadConductor...")
            self.layer_offload_conductor.cleanup()
            self.layer_offload_conductor = None

        # Cleanup FLUX.2 block offloader + its forward/backward driver.
        # Drop the wrapper and remove backward hooks to avoid leaking hooks across runs.
        if getattr(self, 'flux2_block_offloader', None) is not None:
            print(f"{self.log_prefix} Cleaning up FLUX.2 block offloader...")
            _flx = self.flux2_block_offloader
            if hasattr(_flx, 'remove_backward_hooks'):
                try:
                    _flx.remove_backward_hooks()
                except Exception as e:
                    print(f"{self.log_prefix} WARNING: remove_backward_hooks failed: {e}")
            self.flux2_block_offloader = None
        if getattr(self, 'flux2_transformer_wrapper', None) is not None:
            self.flux2_transformer_wrapper = None

        # Close TensorBoard writer
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            print(f"{self.log_prefix} TensorBoard writer closed")

        print(f"{self.log_prefix} Cleanup complete")
