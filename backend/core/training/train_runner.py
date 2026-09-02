"""
Training Runner for SushiUI

Entry point for training processes. Reads YAML config and executes training.
Can be run as: python -m core.train_runner config.yaml run_id
"""

import sys
import yaml
import os
import signal
import time
import re
import json

# Reduce CUDA caching-allocator fragmentation across the many aspect-ratio bucket
# shapes (the allocator otherwise reserves a non-reusable block per distinct shape,
# so `reserved` grows far beyond `allocated` and the real "available VRAM" becomes
# history-dependent). MUST be set before torch initialises the CUDA allocator.
# Respect an existing user setting (escape hatch: set PYTORCH_CUDA_ALLOC_CONF).
#   - expandable_segments: best fix, but LINUX-ONLY (PyTorch logs "not supported on
#     this platform" on Windows), so only enable it off-Windows.
#   - garbage_collection_threshold: cross-platform; releases cached blocks once
#     reserved exceeds the fraction, so reserved tracks the live shape instead of
#     the union of every shape seen.
import platform as _platform
_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
_opts = []
if "expandable_segments" not in _alloc_conf and _platform.system() != "Windows":
    _opts.append("expandable_segments:True")
if "garbage_collection_threshold" not in _alloc_conf:
    _opts.append("garbage_collection_threshold:0.8")
if _opts:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        (_alloc_conf + "," if _alloc_conf else "") + ",".join(_opts))
    print(f"[TrainRunner] PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

# Belt-and-braces on top of the per-module disable_scaled_mm() calls every
# trainer-side loader already makes (see krea2_ops.py / ideogram4_ops.py): force
# the whole training process onto the dequant-only FP8 path regardless of which
# loader ran, so a future training entry point that obtains an Fp8Linear module
# without going through those loaders cannot silently regress into W8A8.
#
# UNCONDITIONAL, not setdefault(): training_process.py hands this subprocess an
# os.environ.copy() of the backend's environment, so an operator who launched the
# backend with SUSHI_FP8_SCALED_MM=1 (for inference) had that "1" INHERITED here
# and setdefault() left it in place -- exactly the case this line exists to
# prevent. Training has no legitimate use for W8A8: a LoRA fitted against W8A8
# conditioning (~2.7e-02 rel RMS noisier) is fitted against a base function
# nobody runs at inference.
#
# MUST be set before `core.models.ideogram4.vendor.fp8_linear` is first
# imported: `_SCALED_MM_ENABLED` is initialized from this variable at that
# module's import time, and nothing above this point (stdlib only, plus the
# PYTORCH_CUDA_ALLOC_CONF logic) imports it or anything that transitively
# imports it. The runtime toggle (set_scaled_mm_enabled / the
# /system/fp8-scaled-mm endpoint) lives in the API process and cannot reach this
# one, so the import-time value is the whole story here.
os.environ["SUSHI_FP8_SCALED_MM"] = "0"
# Identical treatment for the INT8 W8A8 path (torch._int_mm). Same inheritance
# problem, same reasoning, same ordering requirement: `int8_linear`'s
# `_INT8_MM_ENABLED` is initialized from this variable at ITS import time, and
# nothing above this line imports it.
os.environ["SUSHI_INT8_MM"] = "0"

import torch
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add backend directory to path for imports (extensions, database, etc.)
backend_dir = Path(__file__).parent.parent.parent  # backend/
sys.path.insert(0, str(backend_dir))

from database import get_training_db, get_datasets_db
from database.models import TrainingRun, Dataset, DatasetItem, DatasetCaption
from sqlalchemy.orm import Session
from core.training.caption_processor import (
    apply_caption_dropout,
    get_default_caption_processing_config,
    process_caption,
)
from core.training.base_trainer import DEFAULT_MAX_OPTIMIZER_SAVES_TO_KEEP
from api.param_defaults import TRAINING_DEFAULTS, TRAINING_SAMPLE_DEFAULTS_BY_ARCH

# Second half of the FP8 hard-off above. The env write only works while nothing
# has imported fp8_linear yet, which holds for the shipped launch path
# (training_process.py runs this file as a SCRIPT, so `core.training/__init__` --
# which pulls base_trainer and, transitively, fp8_linear -- is not executed
# before line 57). It does NOT hold for `import core.training.train_runner`,
# where the package __init__ runs first and `_SCALED_MM_ENABLED` is already
# initialized from the inherited environment by then. This call is order-
# independent: it flips the module global whenever fp8_linear is reachable, and
# it also clears any probe cache that a "1" had populated.
try:
    from core.models.ideogram4.vendor.fp8_linear import set_scaled_mm_enabled as _fp8_hard_off
    _fp8_hard_off(False, origin="default")
except Exception as _e:  # pragma: no cover - fp8 support is optional at import time
    print(f"[TrainRunner] Could not force the FP8 dequant path: {_e}")

# Same second half for INT8. Separate try block on purpose: if the fp8 import
# fails for any reason, the int8 hard-off must still run.
try:
    from core.models.ideogram4.vendor.int8_linear import set_int8_mm_enabled as _int8_hard_off
    _int8_hard_off(False, origin="default")
except Exception as _e:  # pragma: no cover - int8 support is optional at import time
    print(f"[TrainRunner] Could not force the INT8 dequant path: {_e}")


def _is_krea2_base_model(base_model_path: str) -> bool:
    """Krea 2 base-model check for the bf16 force.

    Path-name match first (cheap, also covers not-yet-downloaded/scratch specs),
    then key/config-based detection so renamed sushiUI checkpoints are caught.
    """
    lowered = base_model_path.lower()
    if 'krea2' in lowered or 'krea-2' in lowered:
        return True
    try:
        from core.model_loader import ModelLoader
        return ModelLoader.detect_model_type(base_model_path) == "krea2"
    except Exception:
        return False


def _is_bf16_native_base_model(base_model_path: str) -> bool:
    """Large bf16-native DiT / audio architectures not handled by name guards.
    caught by the per-name checks above. bf16 is their correct training dtype and
    is REQUIRED for Full fine-tune (fp16 Full-FT is rejected by the trainer --
    GradScaler.unscale_ needs fp32 master params; bf16 needs no scaler). Path-name
    match first, then key/config detection for renamed checkpoints.

    MiniMax-H3 is here for a sharper reason than "native precision": its released
    DiT is weight-only FP8 and `weight_dtype` is the dtype those codes DEQUANTIZE
    INTO inside every forward. Left at the non-bf16-native default (fp32), the
    50-block stack runs fp32 and the per-block dequantized-weight transient
    roughly doubles -- no error, just a run that is a different function from the
    measured one. `minimax_h3_ops.load_components` normalizes it as a second line
    of defence; this keeps the config that reaches the trainer honest in the first
    place."""
    lowered = (base_model_path or "").lower()
    if any(s in lowered for s in ("lens", "ltx", "ace-step", "acestep",
                                  "minimax", "minimax_h3", "minimax-h3",
                                  "sensenova", "sense-nova")):
        return True
    try:
        from core.model_loader import ModelLoader
        return ModelLoader.detect_model_type(base_model_path) in (
            "lens", "ltx2", "acestep", "minimax_h3", "sensenova")
    except Exception:
        return False


def _apply_sensenova_training_contract(
    base_model_path: str,
    network_type: str,
    train_config: Dict[str, Any],
    process_config: Dict[str, Any],
) -> bool:
    """Validate the SenseNova B1 training envelope, per training method.

    ``lora`` and ``full_finetune`` are the two accepted methods. Everything else
    is refused BY NAME rather than by falling through: this check is the only
    thing that ever refused a SenseNova ControlNet run (that method has a
    ``TRAINING_UNSUPPORTED`` entry but no trainer-side guard reading it, unlike
    ReLoRA's ``_refuse_unsupported_relora``), so accepting whatever is not LoRA
    would open it.

    Full fine-tuning adds the clauses of its memory budget that are knowable
    from the config alone (SENSENOVA_TRAINING_DESIGN.md 6.2/6.4); the rest are
    checked by ``ops.sensenova_ops.assert_full_finetune_contract``, still before
    the 17.6 GiB load.
    """
    if network_type == "vae_decoder":
        return False
    try:
        from core.model_loader import ModelLoader
        is_sensenova = ModelLoader.detect_model_type(base_model_path) == "sensenova"
    except Exception:
        lowered = (base_model_path or "").lower()
        is_sensenova = "sensenova" in lowered or "sense-nova" in lowered
    if not is_sensenova:
        return False
    if network_type not in ("lora", "full_finetune"):
        raise ValueError(
            f"SenseNova training supports network.type='lora' and "
            f"network.type='full_finetune', not '{network_type}'"
        )
    is_full_finetune = network_type == "full_finetune"
    batch_size = _normalize_sensenova_integer(train_config, "batch_size", 1)
    if batch_size > 1 and not _normalize_sensenova_bool(train_config, "enable_bucketing", False):
        # A physical batch is one pixel tensor at one resolution (packed
        # prompts, SENSENOVA_TRAINING_DESIGN.md "Packed batches"); only the
        # bucket manager guarantees that.
        raise ValueError(
            f"SenseNova training with batch_size={batch_size} requires "
            "enable_bucketing so every item in a batch has the same resolution; "
            "batch_size=1 works without bucketing"
        )
    if is_full_finetune:
        _apply_sensenova_full_finetune_contract(train_config)
    elif not _normalize_sensenova_bool(train_config, "train_unet", True):
        # LoRA only. Under full fine-tuning the understanding half alone is a
        # branch resolve_full_finetune_branch names ("und"); under LoRA it is
        # not an artefact -- SenseNovaLoRAAdapter.save_checkpoint refuses a
        # generation-free file, so the run would train to its first save (100
        # steps by default) and die there.
        raise ValueError(
            "SenseNova LoRA requires train_unet=True: inference applies both MoT "
            "branches from one file, so an understanding-only LoRA has no "
            "consumer and is refused when it is saved. Set train_text_encoder to "
            "add the understanding half alongside the generation one."
        )
    blocks_to_swap = _normalize_sensenova_integer(train_config, "blocks_to_swap", 0)
    if blocks_to_swap != 0:
        raise ValueError("SenseNova training does not implement blocks_to_swap; set it to 0")
    # Normalized, not gated: reference conditioning is armed run-globally here and
    # applied per item (Phase 3), and composes with a trainable understanding
    # branch (Phase U-3) rather than being refused against it. Strict typing
    # still applies.
    _normalize_sensenova_bool(train_config, "use_reference_images", False)
    from api.param_defaults import TRAINING_DEFAULTS

    phase_eviction = _normalize_sensenova_bool(
        train_config,
        "sensenova_mot_phase_eviction",
        TRAINING_DEFAULTS["sensenova_mot_phase_eviction"],
    )
    # Reused, never duplicated: SenseNova's prompt encoder IS the understanding
    # branch of the same LLM that denoises, so `train_text_encoder` arms
    # understanding-branch LoRA.
    train_understanding = _normalize_sensenova_bool(
        train_config, "train_text_encoder", False
    )
    four_phase = _normalize_sensenova_bool(
        train_config,
        "sensenova_four_phase_eviction",
        TRAINING_DEFAULTS["sensenova_four_phase_eviction"],
    )
    shared_prefix = _normalize_sensenova_bool(
        train_config,
        "sensenova_four_phase_shared_prefix",
        TRAINING_DEFAULTS["sensenova_four_phase_shared_prefix"],
    )
    reduction = str(
        train_config.get(
            "sensenova_four_phase_grad_reduction",
            TRAINING_DEFAULTS["sensenova_four_phase_grad_reduction"],
        )
        or TRAINING_DEFAULTS["sensenova_four_phase_grad_reduction"]
    ).strip().lower()
    if reduction not in ("sum", "mean"):
        raise ValueError(
            f"SenseNova sensenova_four_phase_grad_reduction must be 'sum' or "
            f"'mean', got {reduction!r}."
        )
    train_config["sensenova_four_phase_grad_reduction"] = reduction
    if shared_prefix and not four_phase:
        raise ValueError(
            "SenseNova sensenova_four_phase_shared_prefix requires "
            "sensenova_four_phase_eviction: it shares ONE boundary cut across an "
            "MNT window, and without the split there is no boundary cut to share."
        )
    if four_phase:
        # The lift of the refusal below, and its preconditions. Every clause is
        # restated by ops.sensenova_ops.assert_four_phase_contract inside the
        # trainer; what this adds is that the message arrives before the load.
        if not is_full_finetune:
            raise ValueError(
                "SenseNova sensenova_four_phase_eviction requires "
                "network.type='full_finetune': the split leaves the generation "
                "half on CPU at the step boundary, which is only safe on the "
                "fused backward route where each half is updated by its own "
                "per-parameter hooks while it is resident."
            )
        if not train_understanding:
            raise ValueError(
                "SenseNova sensenova_four_phase_eviction requires "
                "train_text_encoder: it exists so a TRAINED understanding half "
                "can still be evicted. With that half frozen, "
                "sensenova_mot_phase_eviction alone already does this."
            )
        if not phase_eviction:
            raise ValueError(
                "SenseNova sensenova_four_phase_eviction requires "
                "sensenova_mot_phase_eviction: on its own the split only adds a "
                "second backward and a recomputed understanding forward, with "
                "both halves resident exactly as they are without it."
            )
    elif train_understanding and phase_eviction:
        # An explicit error, not a silent auto-disable: both flags are opt-in,
        # and quietly dropping either one breaks a contract the user set (a VRAM
        # budget, or which weights get trained).
        raise ValueError(
            "SenseNova train_text_encoder cannot be combined with "
            "sensenova_mot_phase_eviction: the understanding half must stay "
            "GPU-resident until backward, while the evictor moves it to CPU for "
            "the denoise phase. Set sensenova_four_phase_eviction to split the "
            "backward at the prefix KV cache instead, which keeps the "
            "understanding half through its own backward; that route is "
            "full-finetune only. Otherwise disable one of the two."
        )
    if phase_eviction and is_full_finetune:
        # The evictor requires the two MoT halves to be per-layer symmetric
        # (`select_mot_weight_modules(require_exact_symmetry=True)`), and full
        # fine-tuning materializes only the branch it trains: `gen` or `und`
        # leaves one half bf16 and the other int8, so the halves differ in dtype
        # at every layer. Refused here rather than by the selector, which raises
        # AFTER the 17.6 GiB load and the materialize, in a message about
        # layer-0 mlp tensor shapes.
        train_gen = _normalize_sensenova_bool(train_config, "train_unet", True)
        if not (train_gen and train_understanding):
            raise ValueError(
                "SenseNova MoT phase eviction under full fine-tuning requires "
                "both train_unet and train_text_encoder (the 'both' branch): the "
                "evictor moves whole halves and requires them to hold the same "
                "kind of weight, but a single-branch full fine-tune materializes "
                "only the half it trains and leaves the other quantized. Train "
                "both halves, or disable sensenova_mot_phase_eviction."
            )
    if phase_eviction:
        groups = _normalize_sensenova_integer(
            train_config, "num_optimizer_groups", 0
        )
        if groups != 0:
            raise ValueError(
                "SenseNova MoT phase eviction requires num_optimizer_groups=0"
            )
        if _normalize_sensenova_bool(
            train_config, "block_swap_h2d_only", False
        ):
            raise ValueError(
                "SenseNova MoT phase eviction is independent of block swap; "
                "set block_swap_h2d_only=false"
            )
    pageable_staging = _normalize_sensenova_bool(
        train_config,
        "sensenova_mot_pageable_staging",
        TRAINING_DEFAULTS["sensenova_mot_pageable_staging"],
    )
    if pageable_staging and not phase_eviction:
        raise ValueError(
            "SenseNova sensenova_mot_pageable_staging requires "
            "sensenova_mot_phase_eviction: it selects how the evictor stages "
            "a half to CPU, and with the evictor off nothing is ever staged."
        )
    overlap_transfer = _normalize_sensenova_bool(
        train_config,
        "sensenova_mot_overlap_transfer",
        TRAINING_DEFAULTS["sensenova_mot_overlap_transfer"],
    )
    if overlap_transfer and not phase_eviction:
        raise ValueError(
            "SenseNova sensenova_mot_overlap_transfer requires "
            "sensenova_mot_phase_eviction: it selects how the evictor moves a "
            "half, and with the evictor off nothing is ever moved."
        )
    if overlap_transfer and pageable_staging:
        # Refused rather than silently degraded: see sensenova_phase_eviction's
        # OVERLAPPED TRANSFER note.
        raise ValueError(
            "SenseNova sensenova_mot_overlap_transfer cannot be combined with "
            "sensenova_mot_pageable_staging: an async copy against pageable "
            "host memory is staged through a driver bounce buffer and is "
            "effectively host-synchronous, so the overlap would pay its "
            "correctness cost for no concurrency. Enable one of the two."
        )
    _warn_on_sensenova_timestep_sampling(base_model_path, train_config)
    train_config["text_encoding_mode"] = "onthefly_gpu"
    train_config["latent_encoding_mode"] = "onthefly_gpu"
    return True


def _apply_sensenova_full_finetune_contract(train_config: Dict[str, Any]) -> None:
    """The full-fine-tune clauses that are decidable from the config alone.

    Duplicating nothing: each clause below is checked again inside the trainer
    (``assert_full_finetune_contract`` before the load,
    ``BaseTrainer.train`` from its own arguments). What this adds is the point
    at which it is checked -- before the checkpoint load, not minutes in.

    ``weight_dtype``/``training_dtype`` are NOT checked here: the full-finetune
    dispatch below forces both to bf16 for this architecture via
    ``_is_bf16_native_base_model``, so a config value is not what the trainer
    will see.
    """
    accumulation = _normalize_sensenova_integer(
        train_config, "gradient_accumulation_steps", 1
    )
    if accumulation != 1:
        raise ValueError(
            f"SenseNova full fine-tuning requires gradient_accumulation_steps=1, "
            f"got {accumulation}. Its updates are applied per parameter during "
            f"backward and each gradient is freed as it is applied, so no "
            f"gradient survives to be summed across backward passes. Physical "
            f"batch 1 with no accumulation is what this route trains; LoRA "
            f"training on this architecture does support accumulation."
        )
    groups = _normalize_sensenova_integer(train_config, "num_optimizer_groups", 0)
    if groups != 0:
        raise ValueError(
            f"SenseNova full fine-tuning requires num_optimizer_groups=0, got "
            f"{groups}. Fused optimizer groups replace the per-parameter hooks "
            f"this route's memory budget depends on with a batched "
            f"optimizer.step(), and they are only ever set up under Block Swap, "
            f"which this architecture does not implement."
        )
    if _normalize_sensenova_bool(train_config, "use_ema", False):
        raise ValueError(
            "SenseNova full fine-tuning does not support use_ema: the EMA update "
            "is attached to the single optimizer.step() call site, which this "
            "route never reaches, so the shadow would silently never update."
        )
    if train_config.get("optimizer_stochastic_rounding") is False:
        # Only reachable for an explicit False; a None (unset) is forced on
        # silently by enforce_full_finetune_stochastic_rounding instead.
        raise ValueError(
            "SenseNova full fine-tuning requires optimizer_stochastic_rounding "
            "and cannot run with it explicitly set to False: the trainable "
            "half is bf16 with no fp32 master, and under round-to-nearest "
            "84.5% of its elements never move at any step count while the "
            "loss falls normally (measured, SENSENOVA_TRAINING_DESIGN.md 6.3). "
            "Leave it unset to let this route enable it, or set it to True."
        )
    from api.param_defaults import (
        SENSENOVA_FULL_FINETUNE_SAVE_FORMATS, TRAINING_DEFAULTS,
    )
    from core.training.ops.sensenova_ops import (
        SENSENOVA_FULL_FINETUNE_OPTIMIZERS, assert_ringbuffer_host_state,
    )

    optimizer = train_config.get("optimizer")
    if optimizer is not None:
        name = str(optimizer).strip().lower()
        if name not in SENSENOVA_FULL_FINETUNE_OPTIMIZERS:
            raise ValueError(
                f"SenseNova full fine-tuning does not support optimizer='{name}'. "
                f"Supported: {', '.join(SENSENOVA_FULL_FINETUNE_OPTIMIZERS)}. Each "
                f"update is applied from that parameter's own "
                f"post-accumulate-grad hook, so the optimizer needs a "
                f"per-parameter seam and state small enough to sit beside the "
                f"dequantized bf16 half."
            )
        # The optimizer name and the residency flag arrive on different channels
        # and can disagree; checked on both, like every other clause here.
        assert_ringbuffer_host_state(name, _normalize_sensenova_bool(
            train_config, "optimizer_state_host_resident",
            TRAINING_DEFAULTS["optimizer_state_host_resident"],
        ))
    # Refused here rather than at the first save: the adapter resolves this
    # value only when it writes, and save_every defaults to 100 steps, so an
    # unknown format authored in a hand-written YAML would take the run down
    # after it had already trained (SENSENOVA_TRAINING_DESIGN.md 6.4). The API
    # constrains the field to a Literal, so only that path can reach it.
    save_format = str(train_config.get(
        "sensenova_full_finetune_save_format",
        TRAINING_DEFAULTS["sensenova_full_finetune_save_format"],
    )).strip().lower()
    if save_format not in SENSENOVA_FULL_FINETUNE_SAVE_FORMATS:
        raise ValueError(
            f"Unknown sensenova_full_finetune_save_format {save_format!r}. "
            f"Supported: {', '.join(SENSENOVA_FULL_FINETUNE_SAVE_FORMATS)}."
        )
    train_config["sensenova_full_finetune_save_format"] = save_format
    _warn_on_unresumable_sensenova_save_format(train_config, save_format)


def _warn_on_unresumable_sensenova_save_format(
    train_config: Dict[str, Any], save_format: str
) -> None:
    """Say at run creation which resume path a full fine-tune's save format leaves it.

    Which format resumes without any extra requirement depends on the branch
    (``ops.sensenova_ops`` ``_SENSENOVA_RESUME_FORMAT_FOR_BRANCH``), and the
    answer used to only be discovered when a run was restarted -- possibly
    hours in, and the file already written by then. Every format still
    produces a checkpoint this repo can load for INFERENCE; not every format
    resumes THIS run the same way:

    * the branch's own lossless format (``mixed`` for a single half, ``bf16``
      for ``both``) resumes with nothing else required;
    * ``bf16`` on a single-half branch ALSO resumes, but only via
      ``accept_resume_shaped_base``'s base-model fallback: it restores the
      frozen half's int8 weights from this run's configured base model,
      verified tensor-for-tensor before use, so it fails if that base is
      later moved, deleted, or replaced with a different file;
    * ``int8`` resumes but is lossy (requantizes the trained half on every
      save);
    * any other case is refused at resume rather than silently reshaped.
    """
    from types import SimpleNamespace

    from core.training.ops.sensenova_ops import (
        _SENSENOVA_RESUME_FORMAT_FOR_BRANCH, resolve_full_finetune_branch,
    )
    from core.training.training_events import emit_training_warning

    train_gen = _normalize_sensenova_bool(train_config, "train_unet", True)
    train_und = _normalize_sensenova_bool(train_config, "train_text_encoder", False)
    try:
        # The one resolver, so a run with nothing to train cannot be warned about
        # a branch it does not have; that configuration is refused in the trainer
        # and this advisory has no business naming "und" for it.
        branch = resolve_full_finetune_branch(
            SimpleNamespace(train_unet=train_gen, train_text_encoder=train_und)
        )
    except ValueError:
        return
    lossless = _SENSENOVA_RESUME_FORMAT_FOR_BRANCH[branch]
    # What the writer will actually emit: 'mixed' has no int8 half to keep when
    # both halves are trained, so it degenerates to 'bf16' -- which IS resumable.
    effective = "bf16" if (save_format == "mixed" and branch == "both") else save_format
    if effective == lossless:
        return
    if effective == "bf16" and branch != "both":
        # Resumable via accept_resume_shaped_base's base-model fallback, not
        # refused -- but conditionally, unlike 'lossless' above.
        emit_training_warning(
            f"SenseNova full fine-tuning on the {branch!r} branch is set to "
            f"sensenova_full_finetune_save_format='bf16'; a restart resumes by "
            f"restoring the frozen half from this run's configured base model "
            f"(verified tensor-for-tensor against the saved checkpoint) rather "
            f"than from '{lossless}', which keeps both halves inside the "
            f"checkpoint file itself. This fails if that base model is later "
            f"moved, deleted, or replaced with a different file.",
            code="sensenova_save_format_resume_needs_base",
        )
        return
    if effective == "int8":
        detail = (
            "'int8' resumes, but it requantizes the trained half on every save, "
            "so each restart re-rounds the weights onto the int8 grid"
        )
    else:
        detail = (
            f"'{save_format}' leaves the decoder in a layout a {branch!r}-branch "
            f"resume does not accept, so a restart of this run would be refused"
        )
    emit_training_warning(
        f"SenseNova full fine-tuning on the {branch!r} branch resumes losslessly "
        f"only from sensenova_full_finetune_save_format='{lossless}'; this run is "
        f"set to '{save_format}'. {detail}. The format cannot be changed after the "
        f"fact -- the weights are already written in the shape it chose.",
        code="sensenova_save_format_not_resumable",
    )


def _normalize_sensenova_integer(
    train_config: Dict[str, Any], key: str, default: int
) -> int:
    """Normalize one SenseNova integer field without accepting bools or floats."""
    value = train_config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"SenseNova {key} must be an integer, got boolean {value!r}")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?[0-9]+", text):
            raise ValueError(f"SenseNova {key} must be an integer, got {value!r}")
        normalized = int(text, 10)
    else:
        raise ValueError(
            f"SenseNova {key} must be an integer, got {value!r} "
            f"({type(value).__name__})"
        )
    train_config[key] = normalized
    return normalized


def _normalize_sensenova_bool(
    train_config: Dict[str, Any], key: str, default: bool
) -> bool:
    """Normalize one SenseNova boolean field without Python truthiness."""
    value = train_config.get(key, default)
    if isinstance(value, bool):
        normalized = value
    elif isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        normalized = bool(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1"):
            normalized = True
        elif text in ("false", "0"):
            normalized = False
        else:
            raise ValueError(f"SenseNova {key} must be a boolean, got {value!r}")
    else:
        raise ValueError(
            f"SenseNova {key} must be a boolean, got {value!r} "
            f"({type(value).__name__})"
        )
    train_config[key] = normalized
    return normalized


_SENSENOVA_TAIL_T = 0.9
_SENSENOVA_TAIL_MASS_LIMIT = 0.01
_SENSENOVA_WEIGHT_RATIO_LIMIT = 2.0
_SENSENOVA_PROBE_SAMPLES = 200000
_SENSENOVA_PROBE_SEED = 0


def _sensenova_config_t_eps(base_model_path: str):
    """Read t_eps from the checkpoint's config.json (no model weights loaded)."""
    try:
        path = Path(base_model_path or "")
        candidates = [path / "config.json", path.parent / "config.json"]
    except (TypeError, ValueError, OSError):
        return None
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        value = data.get("t_eps")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if 0.0 < value < 1.0:
            return value
    return None


def _sensenova_timestep_probe(sampling_config: Dict[str, Any], t_eps):
    """Estimate P(t > 0.9) and, when t_eps is known, E[1/(1-t)^2] for one config."""
    from .timestep_sampler import TimestepSampler

    sampler = TimestepSampler.from_config(dict(sampling_config))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_SENSENOVA_PROBE_SEED)
        samples = sampler.sample(_SENSENOVA_PROBE_SAMPLES, torch.device("cpu"))
    samples = samples.detach().double().clamp(0.0, 1.0)
    tail = float((samples > _SENSENOVA_TAIL_T).double().mean())
    if t_eps is None:
        return tail, None
    weights = 1.0 / (1.0 - samples).clamp_min(t_eps) ** 2
    return tail, float(weights.mean())


def _warn_on_sensenova_timestep_sampling(
    base_model_path: str, train_config: Dict[str, Any]
) -> bool:
    """Warn once when timestep_sampling puts weight on SenseNova's t->1 side.

    Advisory only: nothing is clamped and no value is rewritten.
    """
    configured = train_config.get("timestep_sampling")
    if configured is None or not isinstance(configured, dict):
        return False
    from api.param_defaults import TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH

    default = TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["sensenova"]
    if configured == default:
        return False
    t_eps = _sensenova_config_t_eps(base_model_path)
    try:
        tail, weight = _sensenova_timestep_probe(configured, t_eps)
        default_tail, default_weight = _sensenova_timestep_probe(default, t_eps)
    except Exception as exc:
        print(
            "[TrainRunner] SenseNova timestep_sampling check skipped "
            f"({type(exc).__name__}: {exc})"
        )
        return False
    triggered = tail >= _SENSENOVA_TAIL_MASS_LIMIT
    if weight is not None and default_weight is not None and default_weight > 0:
        if weight >= _SENSENOVA_WEIGHT_RATIO_LIMIT * default_weight:
            triggered = True
    if not triggered:
        return False

    if t_eps is None:
        clamp_text = "clamped only by the model's t_eps at 1/t_eps^2"
    else:
        clamp_text = (
            f"clamped only by the model's t_eps at 1/t_eps^2 = {1.0 / (t_eps ** 2):.1f} "
            f"(t_eps={t_eps:g})"
        )
    print("[TrainRunner] WARNING: SenseNova timestep_sampling departs from the "
          "architecture default toward t=1 (the clean side).")
    print(f"[TrainRunner]   configured:   {configured}")
    print(f"[TrainRunner]   arch default: {default}")
    print("[TrainRunner]   SenseNova's velocity-space MSE reduces to "
          "mse(x0_pred, x0) / (1-t)^2 (the noised sample z cancels on both sides), "
          "and SenseNova uses t=0 = noise / t=1 = clean, so the effective per-sample "
          f"loss weight grows without bound as t -> 1 and is {clamp_text}.")
    print(f"[TrainRunner]   P(t > {_SENSENOVA_TAIL_T:g}): configured {tail * 100:.2f}% vs "
          f"arch default {default_tail * 100:.4f}% "
          f"({_SENSENOVA_PROBE_SAMPLES}-sample estimate, seed {_SENSENOVA_PROBE_SEED})")
    if weight is not None and default_weight is not None:
        print(f"[TrainRunner]   E[1/(1-t)^2] (t_eps-clamped): configured {weight:.1f} vs "
              f"arch default {default_weight:.1f}")
    print("[TrainRunner]   SenseNova's train_step returns this loss unchanged; the shared "
          "min-SNR weighting applies only to epsilon-prediction losses and does not "
          "offset it.")
    print("[TrainRunner]   Recommended: the architecture default "
          f"{default['distribution']}(mean={default['mean']}, std={default['std']}). "
          "The configured setting is kept as-is.")
    return True


# train_config keys that each turn ON one trainable component. `train_unet`
# defaults True; the rest default False.
_TRAINING_SCOPE_FLAGS = (
    "train_unet", "train_text_encoder", "train_image_encoder",
)


def _normalize_scope_flag(
    train_config: Dict[str, Any], key: str, default: bool
) -> bool:
    """Read one scope flag without Python truthiness, and write the bool back.

    A hand-written YAML ``train_unet: "false"`` is a non-empty string: it would
    pass this guard AND read as trainable in the trainer. The API types these as
    booleans, so only that path can produce one. Normalizing in place means the
    trainer receives what the config meant, not what it happened to be truthy as.
    """
    value = train_config.get(key, default)
    if value is None:
        normalized = default          # an explicit YAML `null` means "unset"
    elif isinstance(value, bool):
        normalized = value
    elif isinstance(value, int) and value in (0, 1):
        normalized = bool(value)
    elif isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0"):
        normalized = value.strip().lower() in ("true", "1")
    else:
        raise ValueError(
            f"Training config {key} must be a boolean, got {value!r} "
            f"({type(value).__name__})"
        )
    train_config[key] = normalized
    return normalized


def _assert_training_scope_is_nonempty(
    network_type: str, train_config: Dict[str, Any]
) -> None:
    """Refuse a run that would train nothing, from the config and before the load.

    Architecture-independent. SenseNova names this case itself
    (``resolve_full_finetune_branch``), but sd15/sdxl/krea2 collect no parameters
    and hand the optimizer an empty list minutes later, with the checkpoint
    already resident.

    Only the three methods that read the flags are checked: ControlNet trains its
    own module with ``train_unet=False`` by construction, and ``vae_decoder`` has
    no such flags at all.
    """
    if network_type not in ("lora", "relora", "full_finetune"):
        return
    on = [name for name in _TRAINING_SCOPE_FLAGS
          if _normalize_scope_flag(train_config, name, name == "train_unet")]
    # A trained vision encoder is a fourth component, and the only one that is
    # not one of the flags above (it needs its weights named too).
    if _normalize_scope_flag(train_config, "train_vision_encoder", False) \
            and train_config.get("vision_encoder_path"):
        on.append("train_vision_encoder")
    if on:
        return
    raise ValueError(
        f"This {network_type} run has nothing to train: "
        + ", ".join(f"{name}=false" for name in _TRAINING_SCOPE_FLAGS)
        + ". Set at least one of them (or train_vision_encoder together with "
        "vision_encoder_path)."
    )


def _apply_reference_training_contract(
    base_model_path: str, train_config: Dict[str, Any]
) -> None:
    """Normalize and reject reference settings before model loading."""
    from core.model_loader import ModelLoader

    model_type = ModelLoader.detect_model_type(base_model_path)
    use_references = _normalize_scope_flag(
        train_config, "use_reference_images", False
    )
    train_ve = _normalize_scope_flag(train_config, "train_vision_encoder", False)
    ve_path = train_config.get("vision_encoder_path")
    is_sd_ve = model_type in ("sd15", "sdxl")

    if ve_path and not is_sd_ve:
        raise ValueError(
            "vision_encoder_path is supported only for SD1.5/SDXL training; "
            f"selected architecture is {model_type}"
        )
    if train_ve and not ve_path:
        raise ValueError("train_vision_encoder=true requires vision_encoder_path")
    if is_sd_ve and ve_path:
        train_config["use_reference_images"] = True
    elif is_sd_ve and use_references:
        raise ValueError(
            "SD1.5/SDXL use_reference_images=true requires vision_encoder_path"
        )
    elif use_references and model_type not in ("flux2", "sensenova"):
        raise ValueError(
            "use_reference_images is supported only for FLUX.2, SenseNova, "
            "and SD1.5/SDXL with a SigLIP2 vision encoder"
        )


def _prepare_training_process_config(
    config: Dict[str, Any], base_model_path: str
):
    """Extract and preflight the process block before dataset work begins."""
    process_config = config['config']['process'][0]
    train_config = process_config['train']
    network_config = process_config.get('network', {})
    network_type = network_config.get('type', 'lora')
    _assert_training_scope_is_nonempty(network_type, train_config)
    _apply_reference_training_contract(base_model_path, train_config)
    _apply_sensenova_training_contract(
        base_model_path, network_type, train_config, process_config
    )
    return process_config, train_config, network_config, network_type


def _preflight_cfg_null_caption_conflict(
    train_config: Dict[str, Any], base_model_path: str,
    dataset_configs: List[Dict[str, Any]], datasets_db,
) -> None:
    """Refuse an unusable aligned-CFG-null combination, here.

    The route runs the same check on POST/PUT /training/runs, which a
    hand-authored YAML never touches. A run that pairs an explicit
    ``cfg_uncond_drop_rate`` with a dataset ``caption_dropout_rate`` (or an
    enabled Danbooru caption dropout) trains two different empty-condition
    representations at an uncontrolled combined rate -- the outcome the feature
    exists to prevent -- so it has to be refused where training actually starts.
    ``resolve_and_check`` also refuses a nonzero rate on a reference-conditioned
    run, whose inference CFG baseline is a different branch; that half reads
    ``use_reference_images``, already normalised by
    ``_apply_reference_training_contract`` when this runs.

    Run before dataset scanning and model loading. The dataset half of the
    check comes from the datasets DB, which is why it lives here and not in the
    trainer: caption processing is never written to the YAML. The resolved
    pairs are parked on ``train_config`` under the resolver's own key, so
    ``BaseTrainer.cfg_null_drop_rate()`` re-checks against the same inputs.

    Warnings are left to the trainer, which emits them on the training-events
    channel from this same data.
    """
    from api.cfg_null_resolver import DATASET_CAPTION_CONFIGS_KEY, resolve_and_check
    from api.error_handlers import ValidationError
    from core.training.training_config import _detect_arch

    caption_configs = []
    for ds_config in dataset_configs:
        dataset = datasets_db.query(Dataset).filter(
            Dataset.id == ds_config.get("dataset_id")).first()
        if dataset is None:
            continue
        caption_configs.append(
            (dataset.name or dataset.path, dataset.caption_processing))
    train_config[DATASET_CAPTION_CONFIGS_KEY] = caption_configs
    try:
        resolve_and_check(train_config, arch=_detect_arch(base_model_path),
                          dataset_caption_configs=caption_configs)
    except ValidationError as exc:
        raise ValueError(f"{exc.message}: {exc.detail}" if exc.detail
                         else exc.message)


def _update_phase_progress(run_id: int, phase: str, progress: float, detail: str = None):
    """
    Update training run phase progress in database.

    Args:
        run_id: Training run ID
        phase: Current phase name
        progress: Progress percentage (0-100)
        detail: Optional detail string
    """
    if run_id is None:
        return
    try:
        training_db = next(get_training_db())
        try:
            run = training_db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                run.phase = phase
                run.phase_progress = progress
                if detail:
                    run.phase_detail = detail
                training_db.commit()
        finally:
            training_db.close()
    except Exception as e:
        print(f"[TrainRunner] Warning: Failed to update phase progress: {e}")


def _check_init_stop(output_dir):
    """Abort a long-running INIT phase (dataset scan / caption processing /
    bucketing) promptly on a user stop request.

    Mirrors ``BaseTrainer._check_stop_requested()`` (used for the pre-encode
    phases once training has started), but runs during train_runner.py's
    dataset-loading stage, which happens BEFORE BaseTrainer.train() is ever
    called and can take many minutes on multi-million-item datasets. Without
    this check, ``TrainingProcess.stop()``'s ``.stop_training`` flag file is
    written but never observed until the scan finishes, so the API's /stop
    call hangs.

    Uses KeyboardInterrupt deliberately: it is the same stop token
    BaseTrainer and the SIGINT handler above use, and — being a
    BaseException, not an Exception — it bypasses main()'s
    ``except Exception`` handler so an intentional stop is never misreported
    as a training failure.

    A cheap ``Path.is_file()`` stat; safe to call frequently.
    """
    if output_dir is None:
        return
    flag = Path(output_dir) / ".stop_training"
    if flag.is_file():
        print(f"[TrainRunner] Stop flag detected during initialization, aborting...")
        try:
            flag.unlink()
        except OSError:
            pass
        raise KeyboardInterrupt("Training stopped by user during initialization")


class TeeOutput:
    """
    Redirects output to both console and file (like Unix tee command).
    """
    def __init__(self, console, file):
        self.console = console
        self.file = file

    def write(self, message):
        # Console first (primary output to the parent process). Guard it too so a
        # broken pipe can't crash training.
        try:
            self.console.write(message)
            self.console.flush()
        except Exception:
            pass
        # The log file is secondary: a transient lock (antivirus/indexer, or the
        # file being opened elsewhere on Windows) must NEVER kill a multi-hour run.
        # On failure, drop the file handle so we don't keep raising every print().
        if self.file:
            try:
                self.file.write(message)
                self.file.flush()
            except Exception:
                self.file = None

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        if self.file:
            try:
                self.file.flush()
            except Exception:
                self.file = None

    def isatty(self):
        # Teed output is captured/logged, never an interactive terminal — return
        # False so libraries (e.g. transformers' loading report) don't emit ANSI
        # colour codes into the log file.
        return False

    def __getattr__(self, name):
        # Delegate any other stdout attribute probes (fileno, encoding, buffer,
        # ...) to the underlying console so TeeOutput is a drop-in stdout.
        # __getattr__ only fires for attributes not set on the instance, so
        # write/flush/isatty are unaffected. Guard the own attrs to avoid
        # infinite recursion before __init__ sets them.
        if name in ("console", "file"):
            raise AttributeError(name)
        return getattr(self.console, name)


class TrainingLogger:
    """
    Logger for training that supports both console+file and file-only output.

    Usage:
        logger.info("This goes to both console and file")
        logger.log_only("This goes only to file (verbose logs)")
    """
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.original_stdout = sys.stdout

    def info(self, message):
        """Print to both console and log file."""
        print(message)

    def log_only(self, message):
        """Print only to log file, not to console (for verbose logs)."""
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()
        # If no log file, silently ignore (don't spam console)


# Global logger instance (initialized in main)
logger: TrainingLogger = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def detect_start_epoch_from_checkpoint(output_dir: str, resume_from_checkpoint: str) -> int:
    """
    Detect the start epoch from checkpoint state file.

    This is used to load dataset items with the correct epoch_num at initialization,
    avoiding redundant dataset scanning when resuming training.

    Args:
        output_dir: Training output directory containing checkpoints
        resume_from_checkpoint: Checkpoint setting ("latest", specific path, or None)

    Returns:
        Start epoch (0 for new training, actual epoch for resume)
    """
    import json
    import re

    if not resume_from_checkpoint:
        return 0

    output_path = Path(output_dir)
    if not output_path.exists():
        return 0

    # Find the latest checkpoint step
    if resume_from_checkpoint == "latest":
        # Find all state files
        state_files = list(output_path.glob("*_step_*_state.json"))
        if not state_files:
            return 0

        # Extract step numbers and find the latest
        def get_step(path):
            try:
                step_str = path.stem.split("_step_")[-1].replace("_state", "")
                return int(step_str)
            except:
                return 0

        state_files_with_steps = [(f, get_step(f)) for f in state_files]
        state_files_with_steps.sort(key=lambda x: x[1], reverse=True)

        if not state_files_with_steps:
            return 0

        latest_state_file = state_files_with_steps[0][0]
    else:
        # Specific checkpoint path - find corresponding state file
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            # Try relative to output_dir
            checkpoint_path = output_path / resume_from_checkpoint

        if not checkpoint_path.exists():
            return 0

        # Extract step from checkpoint filename
        try:
            step_str = checkpoint_path.stem.split("_step_")[-1]
            step = int(step_str)
        except:
            return 0

        # Find corresponding state file
        state_file_pattern = f"*_step_{step:06d}_state.json"
        state_files = list(output_path.glob(state_file_pattern))

        # Try without leading zeros for legacy format
        if not state_files:
            state_file_pattern = f"*_step_{step}_state.json"
            state_files = list(output_path.glob(state_file_pattern))

        if not state_files:
            return 0

        latest_state_file = state_files[0]

    # Read epoch from state file
    try:
        with open(latest_state_file, 'r') as f:
            state = json.load(f)
        epoch = state.get('epoch', 0)
        print(f"[TrainRunner] Detected start_epoch={epoch} from checkpoint: {latest_state_file.name}")
        return epoch
    except Exception as e:
        print(f"[TrainRunner] Warning: Failed to read state file {latest_state_file}: {e}")
        return 0


# =============================================================================
# Dataset Cache System
# =============================================================================
# Caches dataset items to avoid repeated DB queries on resume.
# Cache is invalidated when dataset is modified (item count or updated_at changes).

import hashlib
import json
import pickle

def _compute_dataset_cache_key(db: Session, dataset_ids: list, caption_types: list = None) -> str:
    """
    Compute cache key based on dataset state.

    The key includes:
    - Dataset IDs
    - Item counts per dataset
    - Latest updated_at timestamp per dataset
    - Caption types configuration

    If any of these change, the cache is invalidated.
    """
    key_parts = []

    for dataset_id in sorted(dataset_ids):
        # Get dataset info
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            continue

        # Get item count and latest update
        item_count = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).count()

        # Get latest updated_at from items
        from sqlalchemy import func
        latest_update = db.query(func.max(DatasetItem.updated_at)).filter(
            DatasetItem.dataset_id == dataset_id
        ).scalar()

        # Get latest caption update
        latest_caption_update = db.query(func.max(DatasetCaption.updated_at)).join(
            DatasetItem, DatasetCaption.item_id == DatasetItem.id
        ).filter(DatasetItem.dataset_id == dataset_id).scalar()

        key_parts.append(f"{dataset_id}:{item_count}:{latest_update}:{latest_caption_update}")

    # Include caption types in key
    if caption_types:
        key_parts.append(f"caption_types:{','.join(sorted(caption_types))}")

    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def _get_dataset_cache_path(output_dir: Path, cache_key: str) -> Path:
    """Get path to dataset cache file."""
    cache_dir = output_dir / ".dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"dataset_{cache_key}.pkl"


def _prune_captioned_dataset_caches(output_dir: Path) -> None:
    """Delete caption-bearing dataset pickles from a PIXELS-ONLY run's cache dir.

    A ``vae_decoder`` run reads only ``dataset_<key>_nocap.pkl``. Any other
    ``dataset_*.pkl`` in its own ``.dataset_cache/`` was written by an earlier
    load of the SAME run that still built captions (e.g. the same config run
    before pixels-only loading existed) — it can never be read again, and it is
    not small: run 113's 22 datasets left 8.4 GB behind. The cache dir belongs
    to exactly one training run (``output_dir/.dataset_cache``), so nothing else
    can be relying on these files.

    Best-effort: any failure is logged and ignored, since this is disk hygiene,
    not correctness.
    """
    cache_dir = Path(output_dir) / ".dataset_cache"
    if not cache_dir.is_dir():
        return
    freed, removed = 0, 0
    try:
        entries = list(cache_dir.glob("dataset_*.pkl"))
    except OSError:
        return
    for entry in entries:
        if entry.name.endswith("_nocap.pkl"):
            continue
        try:
            size = entry.stat().st_size
            entry.unlink()
            freed += size
            removed += 1
        except OSError as e:
            print(f"[TrainRunner] Could not remove stale dataset cache {entry.name}: {e}")
    if removed:
        # ASCII only: train_runner runs as a subprocess whose stdout is a PIPE,
        # so on Windows it encodes with the locale codepage (cp932 here) and a
        # non-ASCII character in a print raises UnicodeEncodeError mid-run.
        print(f"[TrainRunner] Removed {removed} stale caption-bearing dataset cache "
              f"file(s) ({freed / (1024 ** 3):.2f} GB); this run loads image paths only")


def _load_dataset_cache(cache_path: Path) -> dict:
    """Load dataset cache from disk."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache
    except Exception as e:
        print(f"[TrainRunner] Warning: Failed to load dataset cache: {e}")
        return None


def _save_dataset_cache(cache_path: Path, cache_data: dict):
    """Save dataset cache to disk.

    Writes to a ``.tmp`` sibling then ``os.replace``s it into place, so a
    process killed mid-write (e.g. a stop request arriving during the pickle
    dump) can never leave a truncated/corrupt cache file at ``cache_path`` --
    ``os.replace`` is atomic on both POSIX and Windows.
    """
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        print(f"[TrainRunner] Dataset cache saved: {cache_path}")
    except Exception as e:
        print(f"[TrainRunner] Warning: Failed to save dataset cache: {e}")
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _apply_video_metadata(item_dict: dict, item_type, exif_data, image_path: str) -> None:
    """Propagate LTX-2.3 video / ACE-Step audio metadata onto an item dict (in place).

    For item_type=="video" (LTX-2.3 VIDEO datasets), spreads the probed metadata
    stored in DatasetItem.exif_data ({video_path, fps, num_frames, duration,
    width, height, codec} per P4a) so the trainer's video-clip encode guards
    (item_type=="video") and VideoBucketManager see it.

    For item_type=="audio" (ACE-Step datasets, Phase 8a), spreads the probed
    metadata stored in DatasetItem.exif_data ({audio_path, sample_rate,
    duration, channels} per ``dataset_scanner.probe_audio_metadata``) so the
    trainer's audio-clip encode guard (item_type=="audio") and the
    acestep_audio_batches grouping (keyed by duration) see it.

    Image items get item_type="single" and are otherwise untouched.
    """
    if item_type == "video":
        item_dict["item_type"] = "video"
        meta = exif_data if isinstance(exif_data, dict) else {}
        item_dict["video_path"] = meta.get("video_path") or image_path
        if meta.get("fps") is not None:
            item_dict["fps"] = meta.get("fps")
        if meta.get("num_frames") is not None:
            item_dict["num_frames"] = meta.get("num_frames")
        if meta.get("duration") is not None:
            item_dict["duration"] = meta.get("duration")
        # Prefer probed spatial dims when the DB row lacks them.
        if not item_dict.get("width") and meta.get("width"):
            item_dict["width"] = meta.get("width")
        if not item_dict.get("height") and meta.get("height"):
            item_dict["height"] = meta.get("height")
    elif item_type == "audio":
        item_dict["item_type"] = "audio"
        meta = exif_data if isinstance(exif_data, dict) else {}
        item_dict["audio_path"] = meta.get("audio_path") or image_path
        if meta.get("sample_rate") is not None:
            item_dict["sample_rate"] = meta.get("sample_rate")
        if meta.get("duration") is not None:
            item_dict["duration"] = meta.get("duration")
        if meta.get("channels") is not None:
            item_dict["channels"] = meta.get("channels")
    else:
        item_dict["item_type"] = item_type or "single"


def get_dataset_items_fast(db: Session, dataset_id: int, caption_types: list = None,
                           run_id: int = None, output_dir=None,
                           skip_captions: bool = False) -> list:
    """
    Get all items from dataset using optimized JOIN query.

    This replaces N+1 queries with a single JOIN query.
    Returns raw data without caption processing (for caching).

    Args:
        db: Database session
        dataset_id: Dataset ID
        caption_types: List of caption types to use
        run_id: Optional training run id — when given, reports phase progress to
            the DB (phase_detail/phase_progress) so the frontend bar updates
            during the (slow, first-epoch) bulk read of large datasets.
        output_dir: Optional training output dir — when given, checked for a
            ``.stop_training`` flag so a user stop during this (potentially
            many-minutes-long) DB read/scan aborts promptly instead of
            blocking until it finishes.
        skip_captions: Read PIXELS ONLY — do not join the caption table and do
            not select a primary caption per item. Set only by the VAE
            fine-tune path (``network.type == "vae_decoder"``), whose dataset
            consumes ``image_path`` and nothing else (see
            ``vae/vae_dataset.py``: ``VaeRawImageDataset`` /
            ``make_validation_batch``). Every OTHER training method (the four
            diffusion methods and the tagger) is text-conditioned and must keep
            the default False. ``item.captions`` is not touched at ANY of its
            three access sites in this mode, so dropping the eager join cannot
            degrade into an N+1 lazy load. Consequences for the returned dicts:
            ``raw_caption`` is "" and ``tag_data`` is None (same keys, empty
            values), and an ``item_type == "audio"`` item has NO ``lyrics`` key
            at all — reading it would mean touching ``item.captions``. Every
            non-caption field (image_path, width/height, related_images,
            item_type and the video/audio metadata from
            ``_apply_video_metadata``) is produced exactly as in the default
            mode.

    Returns:
        List of dicts with item data and caption info
    """
    from sqlalchemy.orm import joinedload

    # The JOIN materialization itself can't be subdivided; flag it so the UI
    # doesn't look stalled while a multi-million-row dataset is read.
    _update_phase_progress(run_id, "initializing", 0.0,
                           f"Reading dataset {dataset_id} from DB...")

    # Last chance to abort before the blocking .all() materialization below,
    # which cannot be interrupted mid-flight once started.
    _check_init_stop(output_dir)

    # Single query with JOIN to get all items with their captions.
    # ORDER BY id gives a deterministic base order across DB backends/versions
    # so the per-epoch shuffle (base_trainer.py) has a stable, reproducible
    # starting point instead of whatever order the DB happens to return.
    _query = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).order_by(DatasetItem.id)
    if not skip_captions:
        _query = _query.options(joinedload(DatasetItem.captions))
    items = _query.all()

    dataset_items = []
    skipped_missing = 0
    _n_items = len(items)
    _last_emit = 0.0
    for _idx, item in enumerate(items):
        # Throttled progress (~2x/sec) over the post-fetch processing loop.
        if run_id is not None:
            _now = time.time()
            if _now - _last_emit >= 0.5:
                _last_emit = _now
                _pct = (_idx / _n_items * 100.0) if _n_items else 0.0
                _update_phase_progress(
                    run_id, "initializing", _pct,
                    f"Reading dataset {dataset_id}: {_idx:,}/{_n_items:,} items",
                )
                _check_init_stop(output_dir)
        # Skip items whose image file no longer exists on disk
        if not os.path.exists(item.image_path):
            skipped_missing += 1
            continue

        # Find primary caption
        primary_caption = None
        if skip_captions:
            pass  # pixels-only (VAE fine-tune): item.captions is never accessed
        elif caption_types:
            for caption_type in caption_types:
                for caption in item.captions:
                    if caption.caption_type == caption_type:
                        primary_caption = caption
                        break
                if primary_caption:
                    break
        else:
            # Auto-select: priority order
            for caption_type in ["tags", "natural_language"]:
                for caption in item.captions:
                    if caption.caption_type == caption_type:
                        primary_caption = caption
                        break
                if primary_caption:
                    break
            # Fallback to any caption
            if not primary_caption and item.captions:
                primary_caption = item.captions[0]

        item_dict = {
            "image_path": item.image_path,
            "raw_caption": primary_caption.content if primary_caption else "",
            "tag_data": primary_caption.tag_data if primary_caption else None,
            "is_tags_format": getattr(primary_caption, 'is_tags_format', True) if primary_caption else True,
            "width": item.width,
            "height": item.height,
            "related_images": item.related_images,
        }
        # ACE-Step audio items: source LYRICS from a SEPARATE, dedicated
        # caption_type=="lyrics" DatasetCaption row -- independent of whichever
        # caption_type was selected above as the primary "caption" (tags /
        # natural_language). Lyrics is a second, parallel conditioning signal
        # (not a substitute for the caption), so it is looked up on its own
        # rather than folded into the caption_types priority search. Missing
        # ("" default) preserves the pre-existing instrumental-only behavior
        # for every item/dataset that has never had a lyrics caption added.
        if item.item_type == "audio" and not skip_captions:
            lyrics_caption = None
            for caption in item.captions:
                if caption.caption_type == "lyrics":
                    lyrics_caption = caption
                    break
            item_dict["lyrics"] = lyrics_caption.content if lyrics_caption else ""
        # LTX-2.3 video items: carry item_type + probed video metadata so the
        # trainer's video-clip encode path (item_type=="video") and
        # VideoBucketManager see it. Image items are unchanged (item_type="single").
        _apply_video_metadata(item_dict, item.item_type, item.exif_data, item.image_path)
        dataset_items.append(item_dict)

    if skipped_missing > 0:
        print(f"[get_dataset_items_fast] WARNING: Skipped {skipped_missing} items whose image files no longer exist on disk. "
              f"Re-scan the dataset to clean up stale records.")

    return dataset_items


def get_dataset_items_cached(
    db: Session,
    dataset_id: int,
    output_dir: Path,
    epoch_num: int = 0,
    run_id: int = None,
    caption_types: list = None,
    use_cache: bool = True,
    force_reload: bool = False,
    skip_captions: bool = False,
) -> list:
    """
    Get dataset items with caching support.

    On first load (or cache miss), fetches from DB with optimized JOIN query
    and saves raw data to cache. On subsequent loads (resume), loads from cache
    and applies caption processing.

    Args:
        db: Database session
        dataset_id: Dataset ID
        output_dir: Training output directory (for cache storage)
        epoch_num: Current epoch number (for per-epoch shuffle/dropout)
        run_id: Training run ID (for phase progress updates)
        caption_types: List of caption types to use
        use_cache: Whether to use caching (default: True)
        force_reload: Force reload from DB even if cache exists
        skip_captions: Pixels-only mode for the VAE fine-tune path — forwarded
            to ``get_dataset_items_fast`` and, additionally, skips the
            per-epoch caption-processing pass (``_process_cached_items``)
            entirely, since there is no caption to shuffle/drop out. The
            on-disk cache key is suffixed so a caption-free cache can never be
            mistaken for a full one (or vice versa) if the same output_dir is
            ever reused.

    Returns:
        List of dataset items with processed captions
    """
    import time
    start_time = time.time()

    # Entry check: abort immediately if a stop was requested before this
    # dataset's load even began (e.g. between datasets in a multi-dataset run).
    _check_init_stop(output_dir)

    # Get dataset info for caption config
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    caption_config = dataset.caption_processing or get_default_caption_processing_config()

    # Honor the dataset's configured caption_types (from Dataset Management ->
    # caption_processing) when the caller passes none. The YAML datasets section does not
    # carry caption_types, so ds_config["caption_types"] arrives empty; without this fallback
    # get_dataset_items_fast() would hit its hardcoded ["tags", "natural_language"] default
    # and silently train on tags even when the dataset is set to captions.
    if not caption_types:
        _cfg_types = caption_config.get("caption_types")
        if _cfg_types:
            caption_types = list(_cfg_types)
            print(f"[TrainRunner] caption_types not supplied; using dataset.caption_processing: {caption_types}")

    # Compute cache key (includes caption_types, so a corrected type invalidates stale cache)
    cache_key = _compute_dataset_cache_key(db, [dataset_id], caption_types)
    if skip_captions:
        # A pixels-only cache holds no captions; keep it in a separate slot so it
        # can never be picked up by (or overwrite) a text-conditioned run's cache.
        cache_key = f"{cache_key}_nocap"
    cache_path = _get_dataset_cache_path(output_dir, cache_key)

    raw_items = None

    # Try to load from cache
    if use_cache and not force_reload:
        cache_data = _load_dataset_cache(cache_path)
        if cache_data and cache_data.get("dataset_id") == dataset_id:
            raw_items = cache_data.get("items", [])
            print(f"[TrainRunner] Loaded {len(raw_items)} items from cache ({cache_path.name})")

    # After the (potentially slow, pickle-deserializing) cache load / cache-miss
    # decision above, and before we either fetch from DB or process captions.
    _check_init_stop(output_dir)

    # If no cache, fetch from DB with optimized query
    if raw_items is None:
        print(f"[TrainRunner] Fetching dataset {dataset_id} from DB (optimized JOIN query)...")
        raw_items = get_dataset_items_fast(db, dataset_id, caption_types, run_id=run_id,
                                           output_dir=output_dir, skip_captions=skip_captions)
        print(f"[TrainRunner] Fetched {len(raw_items)} items in {time.time() - start_time:.2f}s")

        # Save to cache. Checked immediately before, so a stop request doesn't
        # kick off a multi-minute pickle dump of a multi-million-item dataset.
        if use_cache:
            _check_init_stop(output_dir)
            cache_data = {
                "dataset_id": dataset_id,
                "cache_key": cache_key,
                "items": raw_items,
                "created_at": datetime.now().isoformat(),
            }
            _save_dataset_cache(cache_path, cache_data)

    if skip_captions:
        # VAE fine-tune: the raw items ALREADY carry everything the consumer
        # reads (image_path, plus width/height/item_type/related_images for the
        # shared wrappers). Running the caption pass here would process millions
        # of empty captions to produce a field nothing reads.
        print(f"[TrainRunner] Caption processing skipped (pixels-only training)")
        _update_phase_progress(run_id, "initializing", 100.0,
                               f"Loaded {len(raw_items)} items (captions not used)")
        processed_items = raw_items
    else:
        # Apply caption processing (must be done every time for shuffle/dropout)
        print(f"[TrainRunner] Processing captions for epoch {epoch_num}...")
        processed_items = _process_cached_items(
            raw_items=raw_items,
            epoch_num=epoch_num,
            caption_config=caption_config,
            run_id=run_id,
            output_dir=output_dir,
        )

    elapsed = time.time() - start_time
    print(f"[TrainRunner] Dataset loading complete: {len(processed_items)} items in {elapsed:.2f}s")

    return processed_items


def _process_cached_items(
    raw_items: list,
    epoch_num: int,
    caption_config: dict,
    run_id: int = None,
    output_dir=None,
) -> list:
    """
    Apply caption processing to cached raw items.

    Args:
        raw_items: List of raw item dicts from cache
        epoch_num: Current epoch number
        caption_config: Caption processing configuration
        run_id: Training run ID (for progress updates)
        output_dir: Optional training output dir — when given, checked for a
            ``.stop_training`` flag so a user stop during caption processing
            aborts promptly instead of blocking until it finishes.

    Returns:
        List of processed items
    """
    total_items = len(raw_items)
    processed_items = []

    # Initial progress update
    _update_phase_progress(run_id, "initializing", 0.0, f"Processing captions: 0/{total_items}")

    for idx, item in enumerate(raw_items):
        raw_caption = item.get("raw_caption", "")
        tag_data_str = item.get("tag_data")
        is_tags_format = item.get("is_tags_format", True)

        if is_tags_format:
            # Tags format: Apply tag processing
            if tag_data_str:
                # Fast path: Use pre-categorized tag_data
                try:
                    tag_data = json.loads(tag_data_str)
                except:
                    tag_data = None

                if tag_data:
                    from core.training.caption_processor import process_caption_with_tag_data
                    processed_caption = process_caption_with_tag_data(
                        tag_data=tag_data,
                        epoch_num=epoch_num,
                        item_path=item["image_path"],
                        caption_config=caption_config,
                    )
                else:
                    # Fallback to legacy path
                    processed_caption = process_caption(
                        caption=raw_caption,
                        epoch_num=epoch_num,
                        item_path=item["image_path"],
                        normalize_tags=caption_config.get("normalize_tags", True),
                        category_order=caption_config.get("category_order", None),
                        caption_dropout_rate=caption_config.get("caption_dropout_rate", 0.0),
                        token_dropout_rate=caption_config.get("token_dropout_rate", 0.0),
                        keep_tokens=caption_config.get("keep_tokens", 0),
                        shuffle_tokens=caption_config.get("shuffle_tokens", False),
                        shuffle_per_epoch=caption_config.get("shuffle_per_epoch", False),
                        shuffle_keep_first_n=caption_config.get("shuffle_keep_first_n", 0),
                        shuffle_tag_groups=caption_config.get("shuffle_tag_groups", None),
                        shuffle_groups_together=caption_config.get("shuffle_groups_together", False),
                        tag_group_dir=caption_config.get("tag_group_dir", "taglist"),
                        exclude_person_count_from_shuffle=caption_config.get("exclude_person_count_from_shuffle", False),
                        tag_dropout_rate=caption_config.get("tag_dropout_rate", 0.0),
                        tag_dropout_per_epoch=caption_config.get("tag_dropout_per_epoch", False),
                        tag_dropout_keep_first_n=caption_config.get("tag_dropout_keep_first_n", 0),
                        tag_dropout_category_rates=caption_config.get("tag_dropout_category_rates", {}),
                        tag_dropout_exclude_person_count=caption_config.get("tag_dropout_exclude_person_count", False),
                    )
            else:
                # No tag_data, use legacy path
                processed_caption = process_caption(
                    caption=raw_caption,
                    epoch_num=epoch_num,
                    item_path=item["image_path"],
                    normalize_tags=caption_config.get("normalize_tags", True),
                    category_order=caption_config.get("category_order", None),
                    caption_dropout_rate=caption_config.get("caption_dropout_rate", 0.0),
                    token_dropout_rate=caption_config.get("token_dropout_rate", 0.0),
                    keep_tokens=caption_config.get("keep_tokens", 0),
                    shuffle_tokens=caption_config.get("shuffle_tokens", False),
                    shuffle_per_epoch=caption_config.get("shuffle_per_epoch", False),
                    shuffle_keep_first_n=caption_config.get("shuffle_keep_first_n", 0),
                    shuffle_tag_groups=caption_config.get("shuffle_tag_groups", None),
                    shuffle_groups_together=caption_config.get("shuffle_groups_together", False),
                    tag_group_dir=caption_config.get("tag_group_dir", "taglist"),
                    exclude_person_count_from_shuffle=caption_config.get("exclude_person_count_from_shuffle", False),
                    tag_dropout_rate=caption_config.get("tag_dropout_rate", 0.0),
                    tag_dropout_per_epoch=caption_config.get("tag_dropout_per_epoch", False),
                    tag_dropout_keep_first_n=caption_config.get("tag_dropout_keep_first_n", 0),
                    tag_dropout_category_rates=caption_config.get("tag_dropout_category_rates", {}),
                    tag_dropout_exclude_person_count=caption_config.get("tag_dropout_exclude_person_count", False),
                )
        else:
            # Whole-caption dropout is format-agnostic; token operations stay tag-only.
            processed_caption = apply_caption_dropout(
                raw_caption, caption_config.get("caption_dropout_rate", 0.0)
            )

        # Build processed item dict
        processed_item = {
            "image_path": item["image_path"],
            "caption": processed_caption,
            "raw_caption": raw_caption,
            "tag_data": tag_data_str,
            "is_tags_format": is_tags_format,
            "width": item.get("width"),
            "height": item.get("height"),
        }

        # Carry LTX-2.3 video / ACE-Step audio fields through this re-copy (item
        # is a dict from the fast/DB load above, so item_type + media metadata
        # already live on it).
        if item.get("item_type") == "video":
            processed_item["item_type"] = "video"
            for _vk in ("video_path", "fps", "num_frames", "duration"):
                if _vk in item:
                    processed_item[_vk] = item[_vk]
        elif item.get("item_type") == "audio":
            processed_item["item_type"] = "audio"
            for _ak in ("audio_path", "sample_rate", "duration", "channels", "lyrics"):
                if _ak in item:
                    processed_item[_ak] = item[_ak]
        else:
            processed_item["item_type"] = item.get("item_type", "single")

        # Add reference images if available
        if item.get("related_images") and "reference" in item.get("related_images", {}):
            processed_item["reference_images"] = item["related_images"]["reference"]

        processed_items.append(processed_item)

        # Progress update every 1000 items (for UI responsiveness)
        if (idx + 1) % 1000 == 0:
            progress_pct = ((idx + 1) / total_items) * 100.0
            _update_phase_progress(run_id, "initializing", progress_pct, f"Processing captions: {idx + 1}/{total_items}")
            _check_init_stop(output_dir)

        # Console logging every 10000 items (to reduce log spam)
        if (idx + 1) % 10000 == 0:
            print(f"[TrainRunner] Processed {idx + 1}/{total_items} captions ({(idx + 1) / total_items * 100:.1f}%)")

    # Final progress update
    _update_phase_progress(run_id, "initializing", 100.0, f"Processing captions: {total_items}/{total_items}")

    return processed_items


def get_dataset_items(db: Session, dataset_id: int, epoch_num: int = 0, run_id: int = None, caption_types: list = None) -> list:
    """
    Get all items from dataset with caption processing applied.

    NOTE: This is the legacy function that queries DB for each item.
    For better performance, use get_dataset_items_cached() instead.

    Args:
        db: Database session
        dataset_id: Dataset ID
        epoch_num: Current epoch number (for per-epoch shuffle/dropout)
        run_id: Training run ID (for phase progress updates)
        caption_types: List of caption types to use (e.g., ["tags", "natural_language"]). If None/empty, auto-select.

    Returns:
        List of dataset items with processed captions
    """
    # Get dataset and its caption processing config
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    # Get caption processing config (or defaults)
    caption_config = dataset.caption_processing or get_default_caption_processing_config()

    # Debug: Log caption config for first item only
    if epoch_num == 0:
        print(f"[TrainRunner] Caption config for dataset {dataset_id}:")
        print(f"  category_order: {caption_config.get('category_order', None)}")
        print(f"  normalize_tags: {caption_config.get('normalize_tags', True)}")
        print(f"  shuffle_tokens: {caption_config.get('shuffle_tokens', False)}")

    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()
    total_items = len(items)
    print(f"[TrainRunner] Processing {total_items} items from dataset {dataset_id}...")

    dataset_items = []
    # Check if category_order is enabled
    has_category_order = caption_config.get("category_order") and len(caption_config.get("category_order", [])) > 0

    # Determine which caption types to use
    # Priority: 1) caption_types parameter (from dataset_configs, legacy)
    #           2) caption_config.caption_types (from dataset.caption_processing, new standard)
    #           3) Auto-select (priority: tags > natural_language > others)
    if caption_types:
        # Legacy: from dataset_configs (Training Config page)
        selected_caption_types = caption_types
        print(f"[TrainRunner] Using selected caption types (from dataset_configs): {selected_caption_types}")
    elif caption_config.get("caption_types"):
        # New standard: from caption_processing (Dataset Management page)
        selected_caption_types = caption_config.get("caption_types")
        print(f"[TrainRunner] Using selected caption types (from caption_processing): {selected_caption_types}")
    else:
        # Auto-select: priority order: tags > natural_language > others
        selected_caption_types = None  # Will auto-select per item
        print(f"[TrainRunner] No caption types specified - will auto-select per item (priority: tags > natural_language)")

    # Update phase to "initializing" for dataset loading
    _update_phase_progress(run_id, "initializing", 0.0, f"Loading dataset: 0/{total_items} items")

    for idx, item in enumerate(items):
        # Phase update every 1000 items (for UI responsiveness)
        if (idx + 1) % 1000 == 0:
            progress_pct = ((idx + 1) / total_items) * 100.0
            _update_phase_progress(run_id, "initializing", progress_pct, f"Loading dataset: {idx + 1}/{total_items} items")

        # Console log every 10000 items (to reduce log spam)
        if (idx + 1) % 10000 == 0:
            progress_pct = ((idx + 1) / total_items) * 100.0
            print(f"[TrainRunner] Processed {idx + 1}/{total_items} items ({progress_pct:.1f}%)")

        # Get caption based on selected caption_types
        primary_caption = None
        if selected_caption_types:
            # Try each selected caption type in order
            for caption_type in selected_caption_types:
                primary_caption = db.query(DatasetCaption).filter(
                    DatasetCaption.item_id == item.id,
                    DatasetCaption.caption_type == caption_type
                ).first()
                if primary_caption:
                    break
        else:
            # Auto-select: try "tags" first, then "natural_language", then any other
            for caption_type in ["tags", "natural_language"]:
                primary_caption = db.query(DatasetCaption).filter(
                    DatasetCaption.item_id == item.id,
                    DatasetCaption.caption_type == caption_type
                ).first()
                if primary_caption:
                    break

            # If still not found, use any caption type
            if not primary_caption:
                primary_caption = db.query(DatasetCaption).filter(
                    DatasetCaption.item_id == item.id
                ).first()

        raw_caption = primary_caption.content if primary_caption else ""

        # Check if caption is tags format (Danbooru tags) or natural language
        is_tags_format = primary_caption.is_tags_format if primary_caption and hasattr(primary_caption, 'is_tags_format') else True  # Default to True for backward compatibility

        if is_tags_format:
            # Tags format: Apply tag processing (normalization, shuffle, dropout, etc.)
            # Check if tag_data is available (pre-categorized tags for fast processing)
            tag_data_available = primary_caption and primary_caption.tag_data

            if tag_data_available:
                # Fast path: Use pre-categorized tag_data
                import json
                try:
                    tag_data = json.loads(primary_caption.tag_data)
                except:
                    tag_data = None
                    tag_data_available = False

            if tag_data_available and tag_data:
                # Fast per-epoch shuffle/dropout using pre-categorized tags
                from core.training.caption_processor import process_caption_with_tag_data
                processed_caption = process_caption_with_tag_data(
                    tag_data=tag_data,
                    epoch_num=epoch_num,
                    item_path=item.image_path,
                    caption_config=caption_config,
                )
            else:
                # Legacy path: Use process_caption with category lookup
                processed_caption = process_caption(
                    caption=raw_caption,
                    epoch_num=epoch_num,
                    item_path=item.image_path,
                    normalize_tags=caption_config.get("normalize_tags", True),
                    category_order=caption_config.get("category_order", None),
                    caption_dropout_rate=caption_config.get("caption_dropout_rate", 0.0),
                    token_dropout_rate=caption_config.get("token_dropout_rate", 0.0),
                    keep_tokens=caption_config.get("keep_tokens", 0),
                    shuffle_tokens=caption_config.get("shuffle_tokens", False),
                    shuffle_per_epoch=caption_config.get("shuffle_per_epoch", False),
                    shuffle_keep_first_n=caption_config.get("shuffle_keep_first_n", 0),
                    shuffle_tag_groups=caption_config.get("shuffle_tag_groups", None),
                    shuffle_groups_together=caption_config.get("shuffle_groups_together", False),
                    tag_group_dir=caption_config.get("tag_group_dir", "taglist"),
                    exclude_person_count_from_shuffle=caption_config.get("exclude_person_count_from_shuffle", False),
                    tag_dropout_rate=caption_config.get("tag_dropout_rate", 0.0),
                    tag_dropout_per_epoch=caption_config.get("tag_dropout_per_epoch", False),
                    tag_dropout_keep_first_n=caption_config.get("tag_dropout_keep_first_n", 0),
                    tag_dropout_category_rates=caption_config.get("tag_dropout_category_rates", {}),
                    tag_dropout_exclude_person_count=caption_config.get("tag_dropout_exclude_person_count", False),
                )
        else:
            # Whole-caption dropout is format-agnostic; token operations stay tag-only.
            processed_caption = apply_caption_dropout(
                raw_caption, caption_config.get("caption_dropout_rate", 0.0)
            )
            print(f"[TrainRunner] Natural language caption (skipping tag processing): {raw_caption[:50]}...")

        # Build dataset item dict
        item_dict = {
            "image_path": item.image_path,
            "caption": processed_caption,
            "width": item.width,
            "height": item.height,
        }
        # LTX-2.3 video items: carry item_type + probed video metadata (see fast path).
        _apply_video_metadata(item_dict, item.item_type, item.exif_data, item.image_path)

        # Add reference images if available
        if item.related_images and "reference" in item.related_images:
            item_dict["reference_images"] = item.related_images["reference"]

        dataset_items.append(item_dict)

    # Mark dataset loading as complete
    _update_phase_progress(run_id, "initializing", 100.0, f"Loaded {total_items}/{total_items} items")
    print(f"[TrainRunner] Completed processing {total_items} items from dataset {dataset_id}")
    return dataset_items


def update_training_progress(
    db: Session,
    run_id: int,
    phase: str,
    step: int,
    total: int,
    epoch: int = 0,
    loss: float = None,
    lr: float = None,
    detail: str = None,
):
    """
    Update training run progress in database with phase-based progress.

    Args:
        db: Database session
        run_id: Training run ID
        phase: Current phase ("initializing", "latent_cache", "text_encoder_cache", "training", "sampling")
        step: Current step within phase
        total: Total steps in phase
        epoch: Current epoch (training phase only)
        loss: Current loss (training phase only)
        lr: Learning rate (training phase only)
        detail: Pre-built phase_detail override (used by "sampling" to name the
            prompt being rendered); ignored for phases with a fixed template.
    """
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run:
        # Update phase
        run.phase = phase

        # Calculate phase progress (cap at 100% to prevent exceeding due to mid-epoch resume)
        phase_progress = (step / total * 100.0) if total > 0 else 0.0
        phase_progress = min(phase_progress, 100.0)  # Cap at 100%
        run.phase_progress = phase_progress

        # Update phase detail
        if phase == "initializing":
            run.phase_detail = f"Loading dataset: {step}/{total} items"
        elif phase == "bucketing":
            run.phase_detail = f"Assigning buckets: {step}/{total} images"
        elif phase == "crop_precompute":
            run.phase_detail = f"Planning crop schedule: {step}/{total} images"
        elif phase == "latent_cache":
            run.phase_detail = f"Generating latent cache: {step}/{total} items"
        elif phase == "text_encoder_cache":
            run.phase_detail = f"Encoding captions: {step}/{total} captions"
        elif phase == "sampling":
            run.phase_detail = detail or f"Generating sample: {step}/{total}"
        elif phase == "training":
            run.phase_detail = f"Epoch {epoch}, Step {step}/{total}"
            run.current_step = step
            if loss is not None:
                run.loss = loss
            if lr is not None:
                run.learning_rate = lr
            # Overall progress = phase_progress during training (capped at 100%)
            run.progress = phase_progress

        db.commit()


def _resolve_save_every_n_steps(save_every_unit: str, save_every: int,
                                 dataset_item_count: int, batch_size: int) -> int:
    """
    Convert a `save_every`/`save_every_unit` pair to save_every_n_steps.

    This is an approximation used only to seed the trainer's save cadence
    before bucketing/dataloader construction; the trainer recalculates the
    real steps-per-epoch once the actual batch count is known. It does not
    account for gradient_accumulation_steps or dataset repeats/multiplicity
    (neither did the four call sites this replaces, which computed this
    identically). Ceil division so a trailing partial batch counts as a step,
    matching how the trainer's own dataloader yields a final short batch.
    """
    if save_every_unit == 'epochs':
        steps_per_epoch = (dataset_item_count + batch_size - 1) // batch_size
        return save_every * steps_per_epoch
    return save_every


def _resolve_training_sample_config(
    process_config: Dict[str, Any], arch: str = "_default"
) -> Dict[str, Any]:
    """Resolve the generated YAML sample section against the API defaults."""
    section = process_config.get("sample", {})
    arch_defaults = TRAINING_SAMPLE_DEFAULTS_BY_ARCH.get(
        arch, TRAINING_SAMPLE_DEFAULTS_BY_ARCH["_default"]
    )
    prompts = section.get("prompts", section.get("sample_prompts"))
    if not prompts:
        prompts = [dict(prompt) for prompt in TRAINING_DEFAULTS["sample_prompts"]]
    return {
        "sample_every": section.get("sample_every", TRAINING_DEFAULTS["sample_every"]),
        "prompts": prompts,
        "width": section.get("width", TRAINING_DEFAULTS["sample_width"]),
        "height": section.get("height", TRAINING_DEFAULTS["sample_height"]),
        "sample_steps": section.get("sample_steps", arch_defaults["sample_steps"]),
        "guidance_scale": section.get("guidance_scale", arch_defaults["sample_cfg_scale"]),
        "sampler": section.get("sampler", TRAINING_DEFAULTS["sample_sampler"]),
        "schedule_type": section.get("schedule_type", TRAINING_DEFAULTS["sample_schedule_type"]),
        "cfg_schedule_type": section.get("cfg_schedule_type", TRAINING_DEFAULTS["sample_cfg_schedule_type"]),
        "cfg_schedule_min": section.get("cfg_schedule_min", TRAINING_DEFAULTS["sample_cfg_schedule_min"]),
        "cfg_schedule_max": section.get("cfg_schedule_max", TRAINING_DEFAULTS["sample_cfg_schedule_max"]),
        "cfg_schedule_power": section.get("cfg_schedule_power", TRAINING_DEFAULTS["sample_cfg_schedule_power"]),
        "cfg_rescale_snr_alpha": section.get("cfg_rescale_snr_alpha", TRAINING_DEFAULTS["sample_cfg_rescale_snr_alpha"]),
        "dynamic_threshold_percentile": section.get("dynamic_threshold_percentile", TRAINING_DEFAULTS["sample_dynamic_threshold_percentile"]),
        "dynamic_threshold_mimic_scale": section.get("dynamic_threshold_mimic_scale", TRAINING_DEFAULTS["sample_dynamic_threshold_mimic_scale"]),
        "nag_enable": section.get("nag_enable", TRAINING_DEFAULTS["sample_nag_enable"]),
        "nag_scale": section.get("nag_scale", TRAINING_DEFAULTS["sample_nag_scale"]),
        "nag_tau": section.get("nag_tau", TRAINING_DEFAULTS["sample_nag_tau"]),
        "nag_alpha": section.get("nag_alpha", TRAINING_DEFAULTS["sample_nag_alpha"]),
        "nag_sigma_end": section.get("nag_sigma_end", TRAINING_DEFAULTS["sample_nag_sigma_end"]),
        "nag_negative_prompt": section.get("nag_negative_prompt", TRAINING_DEFAULTS["sample_nag_negative_prompt"]),
        "seed": section.get("seed", TRAINING_DEFAULTS["sample_seed"]),
        "sensenova_timestep_shift": section.get("sensenova_timestep_shift", TRAINING_DEFAULTS["sensenova_sample_timestep_shift"]),
        "sensenova_img_cfg_scale": section.get("sensenova_img_cfg_scale", TRAINING_DEFAULTS["sensenova_sample_img_cfg_scale"]),
        "sensenova_cfg_norm": section.get("sensenova_cfg_norm", TRAINING_DEFAULTS["sensenova_sample_cfg_norm"]),
    }


def main():
    """Main training entry point."""
    # Fix Windows cp932 encoding issue: force UTF-8 for stdout/stderr
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) < 3:
        print("Usage: python -m core.train_runner <config_path> <run_id>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_id = int(sys.argv[2])

    print(f"[TrainRunner] Starting training")
    print(f"[TrainRunner] Config: {config_path}")
    print(f"[TrainRunner] Run ID: {run_id}")

    # Set up training log file (will be created after we load config and get output_dir)
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Declare global logger
    global logger

    # Set up signal handlers to convert SIGTERM to KeyboardInterrupt
    # This allows graceful shutdown with checkpoint saving when user stops training
    def signal_handler(signum, frame):
        print(f"\n[TrainRunner] Received signal {signum}, converting to KeyboardInterrupt for graceful shutdown...")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)  # Also handle Ctrl+C explicitly
    print(f"[TrainRunner] Signal handlers registered (SIGTERM, SIGINT)")

    # Load config
    config = load_config(config_path)
    print(f"[TrainRunner] Loaded config: {config['job']}")

    # ============================================================
    # Set Up Training Log File
    # ============================================================
    try:
        # Get training folder from config
        training_folder = config['config']['process'][0].get('training_folder')
        if training_folder:
            training_folder_path = Path(training_folder)

            # Create logs directory
            logs_dir = training_folder_path / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"training_{timestamp}.log"
            log_file_path = logs_dir / log_filename

            # Open log file
            log_file = open(log_file_path, 'w', encoding='utf-8')

            # Redirect stdout and stderr to both console and file
            sys.stdout = TeeOutput(original_stdout, log_file)
            sys.stderr = TeeOutput(original_stderr, log_file)

            # Initialize global logger
            logger = TrainingLogger(log_file=log_file)

            print(f"[TrainRunner] Training log will be saved to: {log_file_path}")
        else:
            print(f"[TrainRunner] Warning: training_folder not found in config, log file not created")
            # Initialize logger without file
            logger = TrainingLogger(log_file=None)
    except Exception as e:
        print(f"[TrainRunner] Warning: Failed to set up log file: {e}")
        # Initialize logger without file
        logger = TrainingLogger(log_file=None)

    # The generation model's VRAM belongs to the BACKEND process. Releasing it
    # from here is impossible: importing core.pipeline in this child builds a
    # fresh, empty DiffusionPipelineManager, so the unload block that used to sit
    # here freed nothing and printed success while doing it. The real release now
    # runs in the backend, in start_training_run, before this process is spawned.

    # Get database sessions (separate DBs for training and datasets)
    training_db_gen = get_training_db()
    training_db = next(training_db_gen)

    datasets_db_gen = get_datasets_db()
    datasets_db = next(datasets_db_gen)

    try:
        # Get training run info (from training.db)
        run = training_db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if not run:
            print(f"[TrainRunner] ERROR: Training run {run_id} not found")
            sys.exit(1)

        # Extract the process/train block before dataset discovery, scanning,
        # cache access, wrapper construction, or model loading. SenseNova's
        # contract must fail before those side effects, and the network type
        # must come from this same process block used by dispatch below.
        process_config, train_config, network_config, network_type = (
            _prepare_training_process_config(config, run.base_model_path)
        )

        # Get dataset configs from YAML (priority) or database (fallback)
        # This ensures YAML edits are reflected in training
        process_config_for_datasets = process_config
        yaml_datasets = process_config_for_datasets.get('datasets', [])

        dataset_configs = []
        if yaml_datasets:
            # Build dataset_configs from YAML datasets section
            for yaml_ds in yaml_datasets:
                from core.training.dataset_params import read_dataset_params
                ds_config = {
                    "dataset_id": yaml_ds.get("dataset_id"),
                    "filters": {},
                    **read_dataset_params(yaml_ds),
                }
                # If dataset_id is missing in YAML, try to resolve from folder_path
                if not ds_config["dataset_id"] and yaml_ds.get("folder_path"):
                    folder_path = yaml_ds.get("folder_path")
                    dataset_by_path = datasets_db.query(Dataset).filter(Dataset.path == folder_path).first()
                    if dataset_by_path:
                        ds_config["dataset_id"] = dataset_by_path.id
                        print(f"[TrainRunner] Resolved dataset_id={dataset_by_path.id} from folder_path={folder_path}")
                    else:
                        print(f"[TrainRunner] WARNING: Could not resolve dataset from folder_path={folder_path}")
                        continue
                if ds_config["dataset_id"]:
                    dataset_configs.append(ds_config)
            print(f"[TrainRunner] Loaded {len(dataset_configs)} dataset(s) from YAML")

        # Fallback to database if YAML has no datasets
        if not dataset_configs:
            dataset_configs = run.dataset_configs or []
            if dataset_configs:
                print(f"[TrainRunner] Loaded {len(dataset_configs)} dataset(s) from database (fallback)")

        # Fallback to legacy single dataset
        if not dataset_configs and run.dataset_id:
            dataset_configs = [{"dataset_id": run.dataset_id, "caption_types": [], "filters": {}}]
            print(f"[TrainRunner] Using legacy single dataset_id={run.dataset_id}")

        if not dataset_configs:
            print("[TrainRunner] ERROR: No datasets configured")
            sys.exit(1)

        # Before the scan and before any model load: the aligned CFG null
        # cannot share a run with whole-caption dropout.
        _preflight_cfg_null_caption_conflict(
            train_config, run.base_model_path, dataset_configs, datasets_db)

        # ============================================================
        # Detect Start Epoch for Resume Training (before dataset loading)
        # ============================================================
        resume_from_checkpoint = train_config.get('resume_from_checkpoint')

        # Detect start_epoch from checkpoint to load dataset with correct epoch_num
        # This avoids redundant dataset scanning when resuming (initial load + epoch start)
        start_epoch = detect_start_epoch_from_checkpoint(run.output_dir, resume_from_checkpoint)
        if start_epoch > 0:
            print(f"[TrainRunner] Resume training detected: loading dataset for epoch {start_epoch}")
        else:
            print(f"[TrainRunner] New training: loading dataset for epoch 0")

        # Pixels-only training methods: no caption is ever read, so the caption
        # join + per-item caption processing (minutes on multi-million-item
        # datasets) is pure waste. Resolved from the SAME process config block
        # the network_type dispatch below uses, so the two can't disagree.
        # `vae_decoder` is the only such method: the four diffusion methods
        # (lora / relora / full_finetune / controlnet) and the tagger are all
        # text-conditioned and keep the full caption pipeline.
        skip_captions = str(
            (process_config_for_datasets.get('network') or {}).get('type', 'lora')
        ).lower() == 'vae_decoder'
        if skip_captions:
            print("[TrainRunner] VAE fine-tune: loading image paths only "
                  "(captions are not read by this training method)")
            # Reclaim any caption-bearing pickles this run wrote before it
            # switched to the pixels-only cache slot; they are unreadable from
            # here on and can be tens of GB.
            _prune_captioned_dataset_caches(Path(run.output_dir))

        print(f"[TrainRunner] Loading {len(dataset_configs)} dataset(s)...")

        # Load all datasets and combine items
        all_dataset_items = []
        dataset_unique_ids = []  # Collect unique IDs for cache management
        for i, ds_config in enumerate(dataset_configs):
            # Check for a user stop request before starting each dataset's
            # (potentially many-minutes-long) scan/load, so a stop between
            # datasets in a multi-dataset run doesn't wait for the next one.
            _check_init_stop(run.output_dir)

            dataset_id = ds_config["dataset_id"]
            dataset = datasets_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                print(f"[TrainRunner] ERROR: Dataset {dataset_id} not found")
                sys.exit(1)

            print(f"[TrainRunner] Dataset {i+1}: {dataset.name} ({dataset.path})")
            dataset_unique_ids.append(dataset.unique_id)

            # Get dataset items with caching support
            # On resume: loads from cache (fast), on first run: fetches from DB and caches
            from core.training.dataset_params import read_dataset_params
            ds_params = read_dataset_params(ds_config)
            caption_types = ds_params["caption_types"]
            ve_reconstruction_mode = ds_params["ve_reconstruction_mode"]
            output_dir = Path(run.output_dir)
            is_resume = start_epoch > 0
            dataset_items = get_dataset_items_cached(
                db=datasets_db,
                dataset_id=dataset_id,
                output_dir=output_dir,
                epoch_num=start_epoch,
                run_id=run_id,
                caption_types=caption_types,
                use_cache=True,
                force_reload=not is_resume,  # Force reload on new training to ensure fresh data
                skip_captions=skip_captions,
            )
            print(f"[TrainRunner]   Items: {len(dataset_items)}")

            # Add dataset_unique_id to each item for cache management
            for item in dataset_items:
                item["dataset_unique_id"] = dataset.unique_id
                if ve_reconstruction_mode:
                    item["_ve_reconstruction_mode"] = True

            all_dataset_items.extend(dataset_items)

        print(f"[TrainRunner] Total dataset items: {len(all_dataset_items)}")

        if len(all_dataset_items) == 0:
            print("[TrainRunner] ERROR: All datasets are empty")
            sys.exit(1)

        # Use combined dataset items
        dataset_items = all_dataset_items

        # Extract remaining config sections (process_config and train_config already extracted above)
        network_config = process_config.get('network', {})
        model_config = process_config.get('model', {})

        # ============================================================
        # Dataset Wrapper Class for New Interface
        # ============================================================
        class TrainRunnerDataset:
            """
            Dataset wrapper for train_runner.py to use new BaseTrainer.train() interface.

            This wrapper converts the old dataset_items format (list of dicts) to
            the new Dataset object format expected by BaseTrainer.train().
            """
            def __init__(self, unique_id: str, items: List[Dict], dataset_config: Dict, output_dir: Path, initial_epoch: int = 0):
                self.unique_id = unique_id
                self.items = items
                self.dataset_config = dataset_config
                self.output_dir = output_dir  # For dataset cache storage
                self.cache_dir = Path(f"./latent_cache/{unique_id}")
                # Track which epoch the initial items were loaded for (to avoid redundant reload)
                self._initial_load_epoch = initial_epoch
                self._has_been_reloaded = False

                # Extract caption configuration from first item (all items share same config)
                if items:
                    self.caption_config = {
                        "normalize_tags": items[0].get("normalize_tags", True),
                        "shuffle_tokens": items[0].get("shuffle_tokens", True),
                        "category_order": items[0].get("category_order", []),
                    }
                else:
                    self.caption_config = {
                        "normalize_tags": True,
                        "shuffle_tokens": True,
                        "category_order": [],
                    }

            def reload_for_epoch(self, epoch_num: int, run_id: int) -> List[Dict] | None:
                """
                Reload dataset items with caption processing for the current epoch.

                This method is called by the trainer at the start of each epoch to
                get freshly processed captions (with shuffling, etc.).

                Returns:
                    List of items if reload was performed, None if skipped (same epoch as initial load)
                """
                # Skip reload if this is the same epoch as initial load and hasn't been reloaded yet
                # This avoids redundant dataset scanning at training start
                if not self._has_been_reloaded and epoch_num == self._initial_load_epoch:
                    self._has_been_reloaded = True  # Mark as "processed" so next epoch will reload
                    return None  # Signal to caller that reload was skipped (use existing items)

                self._has_been_reloaded = True
                from core.training.dataset_params import read_dataset_params
                dataset_id = self.dataset_config["dataset_id"]
                ds_params = read_dataset_params(self.dataset_config)
                caption_types = ds_params["caption_types"]

                # Use cached loading - caption processing is applied per-epoch
                items = get_dataset_items_cached(
                    db=datasets_db,
                    dataset_id=dataset_id,
                    output_dir=self.output_dir,
                    epoch_num=epoch_num,
                    run_id=run_id,
                    caption_types=caption_types,
                    use_cache=True,
                    force_reload=False,  # Use cache for epoch reloads
                    skip_captions=skip_captions,
                )

                # Add dataset_unique_id for cache management
                for item in items:
                    item["dataset_unique_id"] = self.unique_id

                return items

        # ============================================================
        # Prepare Datasets for New Interface
        # ============================================================
        print(f"[TrainRunner] Preparing {len(dataset_configs)} dataset(s) for training...")

        # Convert dataset_items to Dataset objects, grouped by unique_id
        from collections import defaultdict
        items_by_dataset = defaultdict(lambda: {"items": [], "config": None})

        for item in dataset_items:
            unique_id = item.get("dataset_unique_id", "default")
            items_by_dataset[unique_id]["items"].append(item)

        # Match dataset configs to items
        for ds_config in dataset_configs:
            dataset_id = ds_config["dataset_id"]
            dataset = datasets_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset and dataset.unique_id in items_by_dataset:
                items_by_dataset[dataset.unique_id]["config"] = ds_config

        # Create Dataset wrapper objects with initial_epoch for skip-reload optimization
        training_output_dir = Path(run.output_dir)
        training_datasets = [
            TrainRunnerDataset(unique_id, data["items"], data["config"], output_dir=training_output_dir, initial_epoch=start_epoch)
            for unique_id, data in items_by_dataset.items()
            if data["config"] is not None
        ]

        print(f"[TrainRunner] Created {len(training_datasets)} dataset wrapper(s) (initial_epoch={start_epoch})")
        for ds in training_datasets:
            print(f"  Dataset {ds.unique_id}: {len(ds.items)} items")

        # ============================================================
        # Determine Training Method
        # ============================================================

        if network_type == 'lora':
            print("[TrainRunner] Training method: LoRA")
            from core.training.lora_trainer import LoRATrainer

            # Get dtype settings from unified dtype section
            dtype_config = process_config.get('dtype', {})
            weight_dtype = dtype_config.get('weight', 'fp16')
            training_dtype = dtype_config.get('training', 'fp16')
            output_dtype = dtype_config.get('save', 'fp32')
            vae_dtype = dtype_config.get('vae', 'fp16')

            # Z-Image requires BFloat16 for numerical stability (trained with bf16)
            if 'z-image' in run.base_model_path.lower() or 'zimage' in run.base_model_path.lower():
                print("[TrainRunner] Z-Image model detected: forcing training_dtype=bf16 for numerical stability")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Anima (Cosmos-Predict2 DiT) is also trained in bf16 — force it
            # whenever the model path or arch metadata says so.
            if 'anima' in run.base_model_path.lower():
                print("[TrainRunner] Anima model detected: forcing training_dtype=bf16 for numerical stability")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Ideogram 4 (flow-matching DiT, fp8 base) is trained in bf16.
            if 'ideogram4' in run.base_model_path.lower() or 'ideogram-4' in run.base_model_path.lower():
                print("[TrainRunner] Ideogram 4 model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # MiniT2I (pixel-space MM-JiT, flow matching, x0 prediction) is trained in bf16.
            if 'minit2i' in run.base_model_path.lower():
                print("[TrainRunner] MiniT2I model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Krea 2 (single-stream flow-matching MMDiT) is trained in bf16.
            if _is_krea2_base_model(run.base_model_path):
                print("[TrainRunner] Krea 2 model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Lens / LTX-2.3 / ACE-Step (bf16-native DiT / audio) are trained in bf16 too
            # -- backend enforcement so API-created runs (not just the frontend preset)
            # get bf16, avoiding the fp16 Full-FT rejection.
            if _is_bf16_native_base_model(run.base_model_path):
                print("[TrainRunner] bf16-native model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'

            mixed_precision = train_config.get('mixed_precision', True)
            debug_vram = train_config.get('debug_vram', False)  # Debug VRAM profiling (default: False)
            use_flash_attention = train_config.get('use_flash_attention', False)  # Flash Attention (default: False)
            # Attention backend string selector (native|flash); back-compat maps the
            # legacy boolean to flash when the string key is absent.
            attention_backend = train_config.get('attention_backend') or ('flash' if use_flash_attention else 'native')
            # Attention implementation registry (conduit|diffusers). None when the
            # saved config lacks the key -> base_trainer applies resume backward-compat
            # (fresh -> conduit, resume w/o key -> diffusers) and persists the choice.
            attention_impl = train_config.get('attention_impl')
            min_snr_gamma = train_config.get('min_snr_gamma', 5.0)  # Min-SNR gamma weighting (default: 5.0)
            reconstruction_loss_weight = train_config.get('reconstruction_loss_weight', 0.0)  # Dual loss weight (default: 0.0, pred loss only)

            # Get component-specific learning rates from train_config
            unet_lr = train_config.get('unet_lr')
            text_encoder_lr = train_config.get('text_encoder_lr')
            text_encoder_1_lr = train_config.get('text_encoder_1_lr')
            text_encoder_2_lr = train_config.get('text_encoder_2_lr')

            # Get optimizer options and hyperparameters from train_config
            optimizer_cautious = train_config.get('optimizer_cautious', False)
            optimizer_beta1 = train_config.get('optimizer_beta1')
            optimizer_beta2 = train_config.get('optimizer_beta2')
            optimizer_epsilon = train_config.get('optimizer_epsilon')
            optimizer_weight_decay = train_config.get('optimizer_weight_decay')

            # Schedule-Free optimizer options (RingBuffer optimizers only)
            optimizer_schedule_free = train_config.get('optimizer_schedule_free', False)
            optimizer_warmup_steps = train_config.get('optimizer_warmup_steps', 0)
            optimizer_schedule_free_r = train_config.get('optimizer_schedule_free_r', 0.0)
            optimizer_schedule_free_weight_lr_power = train_config.get('optimizer_schedule_free_weight_lr_power', 2.0)
            optimizer_use_radam = train_config.get('optimizer_use_radam', False)
            # Stochastic rounding for BF16 parameter updates (RingBuffer optimizers)
            optimizer_stochastic_rounding = train_config.get('optimizer_stochastic_rounding', False)

            # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
            prompt_chunking_mode = train_config.get('prompt_chunking_mode', 'a1111')
            max_prompt_chunks = train_config.get('max_prompt_chunks', 0)

            # Training scope control. `train_unet` reaches the trainer here for
            # the same reason it does on the full-FT path: without it the
            # constructor default (True) always won and the UI checkbox was
            # inert. LoRATrainer._apply_lora gates injection on it for every
            # architecture, so a text-encoder-only LoRA is what False means.
            train_unet = train_config.get('train_unet', True)
            train_text_encoder = train_config.get('train_text_encoder', False)

            # Initialize trainer
            trainer = LoRATrainer(
                model_path=run.base_model_path,
                output_dir=run.output_dir,
                run_name=run.run_name,  # Pass run_name for checkpoint filename generation
                run_id=run_id,  # Pass run_id for DB metrics logging
                lora_rank=network_config.get('linear', 16),
                lora_alpha=network_config.get('linear_alpha', 16),
                lora_dtype=network_config.get('lora_dtype', 'fp32'),
                learning_rate=train_config.get('lr', 1e-4),
                weight_dtype=weight_dtype,
                training_dtype=training_dtype,
                output_dtype=output_dtype,
                vae_dtype=vae_dtype,
                mixed_precision=mixed_precision,
                debug_vram=debug_vram,
                use_flash_attention=use_flash_attention,
                attention_backend=attention_backend,
                attention_impl=attention_impl,
                min_snr_gamma=min_snr_gamma,
                reconstruction_loss_weight=reconstruction_loss_weight,
                # Component-specific learning rates
                unet_lr=unet_lr,
                text_encoder_lr=text_encoder_lr,
                text_encoder_1_lr=text_encoder_1_lr,
                text_encoder_2_lr=text_encoder_2_lr,
                # Optimizer options and hyperparameters
                optimizer_cautious=optimizer_cautious,
                optimizer_beta1=optimizer_beta1,
                optimizer_beta2=optimizer_beta2,
                optimizer_epsilon=optimizer_epsilon,
                optimizer_weight_decay=optimizer_weight_decay,
                # Schedule-Free optimizer options
                optimizer_schedule_free=optimizer_schedule_free,
                optimizer_warmup_steps=optimizer_warmup_steps,
                optimizer_schedule_free_r=optimizer_schedule_free_r,
                optimizer_schedule_free_weight_lr_power=optimizer_schedule_free_weight_lr_power,
                optimizer_use_radam=optimizer_use_radam,
                optimizer_stochastic_rounding=optimizer_stochastic_rounding,
                # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
                prompt_chunking_mode=prompt_chunking_mode,
                max_prompt_chunks=max_prompt_chunks,
                # Training scope control
                train_unet=train_unet,
                train_text_encoder=train_text_encoder,
                # Full YAML train_config — must reach BaseTrainer.__init__
                # BEFORE _load_*_components runs, so arch-specific setup
                # (Anima FP8 base / checkpointing / LoRA scope / LR factors)
                # can read its own keys via self.config.get(...).
                train_config=train_config,
            )

            # Note: setup_optimizer() is now called inside train() method
            # This avoids double initialization and provides clearer separation of concerns

            # Get optimizer settings (passed to train() method)
            optimizer_type = train_config.get('optimizer', 'adamw8bit')
            lr_scheduler_type = train_config.get('lr_scheduler', 'constant')

            # ============================================================
            # Validate Prediction Configuration (Unified Framework)
            # ============================================================
            from core.model_loader import ModelLoader

            # Detect model's prediction configuration
            model_type = ModelLoader.detect_model_type(run.base_model_path)
            model_pred_config = ModelLoader.detect_prediction_config(run.base_model_path, model_type)

            print(f"[TrainRunner] Model prediction configuration detected:")
            print(f"  Noise Process: {model_pred_config['noise_process']}")
            print(f"  Prediction Target: {model_pred_config['prediction_target']}")
            print(f"  Detection Source: {model_pred_config['source']}")

            # Get training configuration (with "auto" support)
            training_noise_process = train_config.get('noise_process', 'auto')
            training_prediction_target = train_config.get('prediction_target', 'auto')
            strict_validation = train_config.get('strict_validation', False)

            # Auto-detect: use model's configuration
            if training_noise_process == 'auto':
                training_noise_process = model_pred_config['noise_process']
                print(f"[TrainRunner] noise_process='auto' → using model's config: {training_noise_process}")

            if training_prediction_target == 'auto':
                training_prediction_target = model_pred_config['prediction_target']
                print(f"[TrainRunner] prediction_target='auto' → using model's config: {training_prediction_target}")

            # Validate compatibility
            mismatch_warnings = []
            if training_noise_process != model_pred_config['noise_process']:
                mismatch_warnings.append(
                    f"noise_process mismatch: model={model_pred_config['noise_process']}, training={training_noise_process}"
                )
            if training_prediction_target != model_pred_config['prediction_target']:
                mismatch_warnings.append(
                    f"prediction_target mismatch: model={model_pred_config['prediction_target']}, training={training_prediction_target}"
                )

            if mismatch_warnings:
                print(f"\n{'='*60}")
                print(f"[TrainRunner] WARNING: PREDICTION CONFIG MISMATCH DETECTED")
                print(f"{'='*60}")
                for warning in mismatch_warnings:
                    print(f"  - {warning}")
                print(f"\nThis may cause training instability or poor convergence.")
                print(f"Model was trained with: {model_pred_config['noise_process']} + {model_pred_config['prediction_target']}")
                print(f"You are training with: {training_noise_process} + {training_prediction_target}")

                if strict_validation:
                    print(f"\nERROR: strict_validation=True: Aborting training due to mismatch.")
                    print(f"{'='*60}\n")
                    sys.exit(1)
                else:
                    print(f"\nWARNING: strict_validation=False: Continuing with warning.")
                    print(f"Set strict_validation=true in training config to abort on mismatch.")
                    print(f"{'='*60}\n")
            else:
                print(f"[TrainRunner] OK Prediction configuration validated successfully")

            # Store final training config for trainer
            trainer.noise_process = training_noise_process
            trainer.prediction_target = training_prediction_target

            # ============================================================
            # Setup Regularization Loss (SNR or Energy)
            # ============================================================
            regularization_type = train_config.get('regularization_type', None)
            if regularization_type:
                print(f"[TrainRunner] Initializing {regularization_type.upper()} regularization...")
                trainer.config = train_config  # Pass config for factory function

                if regularization_type.lower() == 'snr':
                    from core.training.losses.snr_regularization import create_snr_regularization_loss
                    trainer.snr_regularization_loss = create_snr_regularization_loss(train_config)
                    print(f"[TrainRunner] SNR Regularization enabled:")
                    print(f"  Weight: {train_config.get('snr_regularization_weight', 0.1)}")
                    print(f"  Timestep adaptive: {train_config.get('snr_timestep_adaptive', True)}")
                    print(f"  Penalty mode: {train_config.get('snr_penalty_mode', 'relu')}")
                elif regularization_type.lower() == 'energy':
                    from core.training.losses.energy_regularization import create_energy_regularization_loss
                    trainer.energy_regularization_loss = create_energy_regularization_loss(train_config)
                    print(f"[TrainRunner] Energy Regularization enabled:")
                    print(f"  Weight: {train_config.get('energy_regularization_weight', 0.05)}")
                    print(f"  Timestep adaptive: {train_config.get('energy_timestep_adaptive', True)}")
                    print(f"  Penalty mode: {train_config.get('energy_penalty_mode', 'abs')}")
                    print(f"  Normalize by pixels: {train_config.get('energy_normalize_by_pixels', True)}")
                else:
                    print(f"[TrainRunner] WARNING: Unknown regularization type '{regularization_type}', skipping")
            else:
                print(f"[TrainRunner] Regularization disabled (regularization_type not set)")

            # Determine epochs or steps
            num_epochs = train_config.get('epochs', None)
            total_steps_config = train_config.get('steps', None)

            if num_epochs:
                print(f"[TrainRunner] Training for {num_epochs} epochs")
            elif total_steps_config:
                # Pass total_steps_config to trainer; it will calculate epochs based on actual batch count
                # (batch count depends on bucketing, which is only known after dataset processing)
                num_epochs = None  # Will be calculated by trainer
                print(f"[TrainRunner] Training for {total_steps_config} steps (epochs will be calculated by trainer)")
            else:
                num_epochs = 1

            # Progress callback (update DB only, no print to avoid cluttering tqdm output)
            def progress_callback(phase: str, step: int, total: int, epoch: int = 0, loss: float = None, detail: str = None):
                # Get current learning rate from optimizer (if available)
                lr = None
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    lr = trainer.optimizer.param_groups[0]['lr']
                    # Debug: Log LR retrieval
                    if phase == "training" and step % 100 == 0:
                        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                        print(f"[ProgressCallback] Step {step}: LR={lr:.2e}, Loss={loss_str}")
                update_training_progress(training_db, run_id, phase, step, total, epoch, loss, lr, detail)

            # Total steps callback (called once when actual total_steps is determined)
            def update_total_steps_callback(total_steps: int):
                print(f"[TrainRunner] Updating total_steps in DB: {total_steps}")
                run.total_steps = total_steps
                training_db.commit()

            # Update status to running
            run.status = "running"
            training_db.commit()
            print("[TrainRunner] Status updated to 'running'")

            sample_config = _resolve_training_sample_config(process_config, model_type)
            sample_prompts = sample_config["prompts"]

            # Debug: Log sample generation settings
            print(f"[TrainRunner] Sample generation settings:")
            print(f"  sample_every: {sample_config['sample_every']}")
            print(f"  sample_prompts: {len(sample_prompts) if sample_prompts else 0} prompts")
            if sample_prompts:
                for i, prompt in enumerate(sample_prompts):
                    print(f"    Prompt {i}: positive={prompt.get('positive', '')[:50]}..., negative={prompt.get('negative', '')[:50]}...")
            print(f"  sample_config: {sample_config}")

            # Get debug parameters from config
            debug_latents = train_config.get('debug_latents', False)
            debug_latents_every = train_config.get('debug_latents_every', 50)

            # Get bucketing parameters from config
            enable_bucketing = train_config.get('enable_bucketing', False)
            base_resolutions = train_config.get('base_resolutions', [1024])
            bucket_strategy = train_config.get('bucket_strategy', 'resize')
            multi_resolution_mode = train_config.get('multi_resolution_mode', 'max')

            # Get latent caching parameters
            # Check datasets config first, then fall back to train config
            cache_latents_to_disk = True  # Default
            force_recache = False  # Default
            if 'datasets' in process_config and len(process_config['datasets']) > 0:
                cache_latents_to_disk = process_config['datasets'][0].get('cache_latents_to_disk', True)
                force_recache = process_config['datasets'][0].get('force_recache', False)

            # Convert save_every parameters to new interface (save_every_n_steps)
            save_every_unit = process_config['save'].get('save_every_unit', 'steps')
            save_every = process_config['save'].get('save_every', 100)
            max_step_saves_to_keep = process_config['save'].get('max_step_saves_to_keep', 3)
            max_optimizer_saves_to_keep = process_config['save'].get(
                'max_optimizer_saves_to_keep', DEFAULT_MAX_OPTIMIZER_SAVES_TO_KEEP)

            save_every_n_steps = _resolve_save_every_n_steps(
                save_every_unit, save_every, len(dataset_items), train_config.get('batch_size', 1))
            if save_every_unit == 'epochs':
                print(f"[TrainRunner] Converted save_every={save_every} epochs to save_every_n_steps={save_every_n_steps}")

            print(f"[TrainRunner] Max step saves to keep: {max_step_saves_to_keep} "
                  f"(optimizer states: {max_optimizer_saves_to_keep})")

            # Get sample generation settings
            sample_guidance_scale = sample_config["guidance_scale"]
            sample_steps = sample_config["sample_steps"]
            sample_width = sample_config["width"]
            sample_height = sample_config["height"]
            sample_seed = sample_config["seed"]
            sample_sampler = sample_config["sampler"]
            sample_schedule_type = sample_config["schedule_type"]
            sensenova_sample_timestep_shift = sample_config["sensenova_timestep_shift"]
            sensenova_sample_img_cfg_scale = sample_config["sensenova_img_cfg_scale"]
            sensenova_sample_cfg_norm = sample_config["sensenova_cfg_norm"]
            print(f"[TrainRunner] Sample generation config: width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, sample_steps={sample_steps}, sampler={sample_sampler}, schedule_type={sample_schedule_type}, seed={sample_seed}")

            # Get resume from checkpoint setting
            resume_from_checkpoint = train_config.get('resume_from_checkpoint')
            if resume_from_checkpoint:
                print(f"[TrainRunner] Resume from checkpoint: {resume_from_checkpoint}")

            # Log force_recache setting
            if force_recache:
                print(f"[TrainRunner] Force recache enabled: all latent caches will be regenerated")

            # Get text encoding mode
            text_encoding_mode = train_config.get('text_encoding_mode', 'swap_onthefly')
            text_encoding_swap_interval = train_config.get('text_encoding_swap_interval', 256)
            text_encoding_prefetch_depth = train_config.get('text_encoding_prefetch_depth', 4)

            # Get latent encoding mode
            latent_encoding_mode = train_config.get('latent_encoding_mode', 'swap_onthefly')
            latent_encoding_swap_interval = train_config.get('latent_encoding_swap_interval', 256)

            # Get Multi Noise-Timestep (MNT) settings
            multi_noise_timesteps = train_config.get('multi_noise_timesteps', 1)
            multi_noise_mode = train_config.get('multi_noise_mode', 'independent')
            trajectory_blend_alpha = train_config.get('trajectory_blend_alpha', 0.7)
            timestep_sampling_config = train_config.get('timestep_sampling', None)

            # Get reference image settings
            use_reference_images = train_config.get('use_reference_images', False)
            vision_encoder_path = train_config.get('vision_encoder_path', None)
            train_vision_encoder = train_config.get('train_vision_encoder', False)
            vision_encoder_lr = train_config.get('vision_encoder_lr', None)
            gradient_routing_ve = train_config.get('gradient_routing_ve', False)
            param_tracking = train_config.get('param_tracking', False)
            param_tracking_interval = train_config.get('param_tracking_interval', 100)

            # Get priority training settings (inline dict or legacy file path)
            priority_training = train_config.get('priority_training', None)
            # Legacy support: if it's a string path, load from file
            if isinstance(priority_training, str):
                from core.training.priority_training import PriorityTrainingConfig
                priority_training = {"_legacy_path": priority_training}

            # Start training with new interface
            trainer.train(
                datasets=training_datasets,
                num_epochs=num_epochs if num_epochs else 1,
                total_steps=total_steps_config,  # Pass total_steps from YAML
                batch_size=train_config.get('batch_size', 1),
                save_every_n_steps=save_every_n_steps,
                sample_every_n_steps=sample_config["sample_every"],
                sample_prompts=sample_prompts,
                sample_guidance_scale=sample_guidance_scale,
                sample_steps=sample_steps,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_seed=sample_seed,
                sample_sampler=sample_sampler,
                sample_schedule_type=sample_schedule_type,
                sample_cfg_schedule_type=sample_config["cfg_schedule_type"],
                sample_cfg_schedule_min=sample_config["cfg_schedule_min"],
                sample_cfg_schedule_max=sample_config["cfg_schedule_max"],
                sample_cfg_schedule_power=sample_config["cfg_schedule_power"],
                sample_cfg_rescale_snr_alpha=sample_config["cfg_rescale_snr_alpha"],
                sample_dynamic_threshold_percentile=sample_config["dynamic_threshold_percentile"],
                sample_dynamic_threshold_mimic_scale=sample_config["dynamic_threshold_mimic_scale"],
                sample_nag_enable=sample_config["nag_enable"],
                sample_nag_scale=sample_config["nag_scale"],
                sample_nag_tau=sample_config["nag_tau"],
                sample_nag_alpha=sample_config["nag_alpha"],
                sample_nag_sigma_end=sample_config["nag_sigma_end"],
                sample_nag_negative_prompt=sample_config["nag_negative_prompt"],
                sensenova_sample_timestep_shift=sensenova_sample_timestep_shift,
                sensenova_sample_img_cfg_scale=sensenova_sample_img_cfg_scale,
                sensenova_sample_cfg_norm=sensenova_sample_cfg_norm,
                optimizer_type=optimizer_type,
                lr_scheduler_type=lr_scheduler_type,
                enable_bucketing=enable_bucketing,
                base_resolutions=base_resolutions,
                bucket_strategy="resize",
                multi_resolution_mode="max",
                gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
                max_grad_norm=train_config.get('max_grad_norm', 1.0),
                debug_latents=debug_latents,
                debug_latents_every=debug_latents_every,
                progress_callback=progress_callback,
                update_total_steps_callback=update_total_steps_callback,
                run_id=run_id,
                resume_from_checkpoint=resume_from_checkpoint,
                force_recache=force_recache,
                max_step_saves_to_keep=max_step_saves_to_keep,
                max_optimizer_saves_to_keep=max_optimizer_saves_to_keep,
                text_encoding_mode=text_encoding_mode,
                text_encoding_swap_interval=text_encoding_swap_interval,
                text_encoding_prefetch_depth=text_encoding_prefetch_depth,
                latent_encoding_mode=latent_encoding_mode,
                latent_encoding_swap_interval=latent_encoding_swap_interval,
                multi_noise_timesteps=multi_noise_timesteps,
                multi_noise_mode=multi_noise_mode,
                trajectory_blend_alpha=trajectory_blend_alpha,
                timestep_sampling_config=timestep_sampling_config,
                use_reference_images=use_reference_images,
                vision_encoder_path=vision_encoder_path,
                train_vision_encoder=train_vision_encoder,
                vision_encoder_lr=vision_encoder_lr,
                gradient_routing_ve=gradient_routing_ve,
                param_tracking=param_tracking,
                param_tracking_interval=param_tracking_interval,
                priority_training=priority_training,
            )

            print("[TrainRunner] Training completed successfully!")

            # Update run status
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            training_db.commit()

        elif network_type == 'relora':
            print("[TrainRunner] Training method: ReLoRA (Reinitialized Low-Rank Adaptation)")
            from core.training.relora_trainer import ReLoRATrainer

            # Get dtype settings from unified dtype section
            dtype_config = process_config.get('dtype', {})
            weight_dtype = dtype_config.get('weight', 'fp16')
            training_dtype = dtype_config.get('training', 'fp16')
            output_dtype = dtype_config.get('save', 'fp32')
            vae_dtype = dtype_config.get('vae', 'fp16')

            # Z-Image requires BFloat16 for numerical stability (trained with bf16)
            if 'z-image' in run.base_model_path.lower() or 'zimage' in run.base_model_path.lower():
                print("[TrainRunner] Z-Image model detected: forcing training_dtype=bf16 for numerical stability")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Anima (Cosmos-Predict2 DiT) is also trained in bf16 — force it
            # whenever the model path or arch metadata says so.
            if 'anima' in run.base_model_path.lower():
                print("[TrainRunner] Anima model detected: forcing training_dtype=bf16 for numerical stability")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Ideogram 4 (flow-matching DiT, fp8 base) is trained in bf16.
            if 'ideogram4' in run.base_model_path.lower() or 'ideogram-4' in run.base_model_path.lower():
                print("[TrainRunner] Ideogram 4 model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # MiniT2I (pixel-space MM-JiT, flow matching, x0 prediction) is trained in bf16.
            if 'minit2i' in run.base_model_path.lower():
                print("[TrainRunner] MiniT2I model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Krea 2 (single-stream flow-matching MMDiT) is trained in bf16.
            if _is_krea2_base_model(run.base_model_path):
                print("[TrainRunner] Krea 2 model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Lens / LTX-2.3 / ACE-Step (bf16-native DiT / audio) are trained in bf16 too
            # -- backend enforcement so API-created runs (not just the frontend preset)
            # get bf16, avoiding the fp16 Full-FT rejection.
            if _is_bf16_native_base_model(run.base_model_path):
                print("[TrainRunner] bf16-native model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'

            mixed_precision = train_config.get('mixed_precision', True)
            debug_vram = train_config.get('debug_vram', False)
            use_flash_attention = train_config.get('use_flash_attention', False)
            # Attention backend string selector (native|flash); back-compat maps the
            # legacy boolean to flash when the string key is absent.
            attention_backend = train_config.get('attention_backend') or ('flash' if use_flash_attention else 'native')
            # Attention implementation registry (conduit|diffusers). None when the
            # saved config lacks the key -> base_trainer applies resume backward-compat
            # (fresh -> conduit, resume w/o key -> diffusers) and persists the choice.
            attention_impl = train_config.get('attention_impl')
            min_snr_gamma = train_config.get('min_snr_gamma', 5.0)
            reconstruction_loss_weight = train_config.get('reconstruction_loss_weight', 0.0)

            # Component-specific learning rates
            unet_lr = train_config.get('unet_lr')
            text_encoder_lr = train_config.get('text_encoder_lr')
            text_encoder_1_lr = train_config.get('text_encoder_1_lr')
            text_encoder_2_lr = train_config.get('text_encoder_2_lr')

            # Optimizer options and hyperparameters
            optimizer_cautious = train_config.get('optimizer_cautious', False)
            optimizer_beta1 = train_config.get('optimizer_beta1')
            optimizer_beta2 = train_config.get('optimizer_beta2')
            optimizer_epsilon = train_config.get('optimizer_epsilon')
            optimizer_weight_decay = train_config.get('optimizer_weight_decay')

            # Schedule-Free optimizer options (RingBuffer optimizers only)
            optimizer_schedule_free = train_config.get('optimizer_schedule_free', False)
            optimizer_warmup_steps = train_config.get('optimizer_warmup_steps', 0)
            optimizer_schedule_free_r = train_config.get('optimizer_schedule_free_r', 0.0)
            optimizer_schedule_free_weight_lr_power = train_config.get('optimizer_schedule_free_weight_lr_power', 2.0)
            optimizer_use_radam = train_config.get('optimizer_use_radam', False)
            # Stochastic rounding for BF16 parameter updates (RingBuffer optimizers)
            optimizer_stochastic_rounding = train_config.get('optimizer_stochastic_rounding', False)

            # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
            prompt_chunking_mode = train_config.get('prompt_chunking_mode', 'a1111')
            max_prompt_chunks = train_config.get('max_prompt_chunks', 0)

            # Training scope control (ReLoRATrainer subclasses LoRATrainer, so
            # the same flag governs the same injection).
            train_unet = train_config.get('train_unet', True)
            train_text_encoder = train_config.get('train_text_encoder', False)

            # ReLoRA-specific settings
            relora_config = network_config.get('relora', {})
            relora_merge_every = relora_config.get('merge_every', 500)
            relora_merge_unit = relora_config.get('merge_unit', 'steps')
            restart_warmup_steps = relora_config.get('restart_warmup_steps', 100)
            optimizer_reset_strategy = relora_config.get('optimizer_reset_strategy', 'full_reset')
            optimizer_pruning_ratio = relora_config.get('optimizer_pruning_ratio', 0.9)

            print(f"[TrainRunner] ReLoRA config: merge_every={relora_merge_every} {relora_merge_unit}, "
                  f"restart_warmup={restart_warmup_steps}, reset_strategy={optimizer_reset_strategy}")

            # Initialize ReLoRA trainer
            trainer = ReLoRATrainer(
                # Base model settings
                model_path=run.base_model_path,
                output_dir=run.output_dir,
                run_name=run.run_name,
                run_id=run_id,
                # LoRA settings (inherited)
                lora_rank=network_config.get('linear', 16),
                lora_alpha=network_config.get('linear_alpha', 16),
                lora_dtype=network_config.get('lora_dtype', 'fp32'),
                learning_rate=train_config.get('lr', 1e-4),
                # Dtype settings
                weight_dtype=weight_dtype,
                training_dtype=training_dtype,
                output_dtype=output_dtype,
                vae_dtype=vae_dtype,
                mixed_precision=mixed_precision,
                debug_vram=debug_vram,
                use_flash_attention=use_flash_attention,
                attention_backend=attention_backend,
                attention_impl=attention_impl,
                min_snr_gamma=min_snr_gamma,
                reconstruction_loss_weight=reconstruction_loss_weight,
                # Component-specific learning rates
                unet_lr=unet_lr,
                text_encoder_lr=text_encoder_lr,
                text_encoder_1_lr=text_encoder_1_lr,
                text_encoder_2_lr=text_encoder_2_lr,
                # Optimizer options and hyperparameters
                optimizer_cautious=optimizer_cautious,
                optimizer_beta1=optimizer_beta1,
                optimizer_beta2=optimizer_beta2,
                optimizer_epsilon=optimizer_epsilon,
                optimizer_weight_decay=optimizer_weight_decay,
                # Schedule-Free optimizer options
                optimizer_schedule_free=optimizer_schedule_free,
                optimizer_warmup_steps=optimizer_warmup_steps,
                optimizer_schedule_free_r=optimizer_schedule_free_r,
                optimizer_schedule_free_weight_lr_power=optimizer_schedule_free_weight_lr_power,
                optimizer_use_radam=optimizer_use_radam,
                optimizer_stochastic_rounding=optimizer_stochastic_rounding,
                # Prompt chunking settings
                prompt_chunking_mode=prompt_chunking_mode,
                max_prompt_chunks=max_prompt_chunks,
                # Training scope control
                train_unet=train_unet,
                train_text_encoder=train_text_encoder,
                # ReLoRA-specific settings
                relora_merge_every=relora_merge_every,
                relora_merge_unit=relora_merge_unit,
                restart_warmup_steps=restart_warmup_steps,
                optimizer_reset_strategy=optimizer_reset_strategy,
                optimizer_pruning_ratio=optimizer_pruning_ratio,
                # See LoRATrainer construction above for why this is needed.
                train_config=train_config,
            )

            # Get optimizer settings
            optimizer_type = train_config.get('optimizer', 'adamw8bit')
            lr_scheduler_type = train_config.get('lr_scheduler', 'constant')

            # ============================================================
            # Validate Prediction Configuration (same as LoRA)
            # ============================================================
            from core.model_loader import ModelLoader

            model_type = ModelLoader.detect_model_type(run.base_model_path)
            model_pred_config = ModelLoader.detect_prediction_config(run.base_model_path, model_type)

            print(f"[TrainRunner] Model prediction configuration detected:")
            print(f"  Noise Process: {model_pred_config['noise_process']}")
            print(f"  Prediction Target: {model_pred_config['prediction_target']}")
            print(f"  Detection Source: {model_pred_config['source']}")

            training_noise_process = train_config.get('noise_process', 'auto')
            training_prediction_target = train_config.get('prediction_target', 'auto')
            strict_validation = train_config.get('strict_validation', False)

            if training_noise_process == 'auto':
                training_noise_process = model_pred_config['noise_process']
                print(f"[TrainRunner] noise_process='auto' -> using model's config: {training_noise_process}")

            if training_prediction_target == 'auto':
                training_prediction_target = model_pred_config['prediction_target']
                print(f"[TrainRunner] prediction_target='auto' -> using model's config: {training_prediction_target}")

            mismatch_warnings = []
            if training_noise_process != model_pred_config['noise_process']:
                mismatch_warnings.append(
                    f"noise_process mismatch: model={model_pred_config['noise_process']}, training={training_noise_process}"
                )
            if training_prediction_target != model_pred_config['prediction_target']:
                mismatch_warnings.append(
                    f"prediction_target mismatch: model={model_pred_config['prediction_target']}, training={training_prediction_target}"
                )

            if mismatch_warnings:
                print(f"\n{'='*60}")
                print(f"[TrainRunner] PREDICTION CONFIG MISMATCH DETECTED")
                print(f"{'='*60}")
                for warning in mismatch_warnings:
                    print(f"  - {warning}")
                print(f"\nThis may cause training instability or poor convergence.")
                if strict_validation:
                    print(f"\nstrict_validation=True: Aborting training due to mismatch.")
                    print(f"{'='*60}\n")
                    sys.exit(1)
                else:
                    print(f"\nstrict_validation=False: Continuing with warning.")
                    print(f"{'='*60}\n")
            else:
                print(f"[TrainRunner] Prediction configuration validated successfully")

            trainer.noise_process = training_noise_process
            trainer.prediction_target = training_prediction_target

            # ============================================================
            # Setup Regularization Loss (same as LoRA)
            # ============================================================
            regularization_type = train_config.get('regularization_type', None)
            if regularization_type:
                print(f"[TrainRunner] Initializing {regularization_type.upper()} regularization...")
                trainer.config = train_config

                if regularization_type.lower() == 'snr':
                    from core.training.losses.snr_regularization import create_snr_regularization_loss
                    trainer.snr_regularization_loss = create_snr_regularization_loss(train_config)
                elif regularization_type.lower() == 'energy':
                    from core.training.losses.energy_regularization import create_energy_regularization_loss
                    trainer.energy_regularization_loss = create_energy_regularization_loss(train_config)
                else:
                    print(f"[TrainRunner] WARNING: Unknown regularization type '{regularization_type}', skipping")
            else:
                print(f"[TrainRunner] Regularization disabled (regularization_type not set)")

            # Determine epochs or steps
            num_epochs = train_config.get('epochs', None)
            total_steps_config = train_config.get('steps', None)

            if num_epochs:
                print(f"[TrainRunner] Training for {num_epochs} epochs")
            elif total_steps_config:
                num_epochs = None
                print(f"[TrainRunner] Training for {total_steps_config} steps (epochs will be calculated by trainer)")
            else:
                num_epochs = 1

            # Progress callback
            def progress_callback(phase: str, step: int, total: int, epoch: int = 0, loss: float = None, detail: str = None):
                lr = None
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    lr = trainer.optimizer.param_groups[0]['lr']
                    if phase == "training" and step % 100 == 0:
                        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                        print(f"[ProgressCallback] Step {step}: LR={lr:.2e}, Loss={loss_str}")
                update_training_progress(training_db, run_id, phase, step, total, epoch, loss, lr, detail)

            def update_total_steps_callback(total_steps: int):
                print(f"[TrainRunner] Updating total_steps in DB: {total_steps}")
                run.total_steps = total_steps
                training_db.commit()

            # Update status to running
            run.status = "running"
            training_db.commit()
            print("[TrainRunner] Status updated to 'running'")

            sample_config = _resolve_training_sample_config(process_config, model_type)
            sample_prompts = sample_config["prompts"]

            # Get debug parameters
            debug_latents = train_config.get('debug_latents', False)
            debug_latents_every = train_config.get('debug_latents_every', 50)

            # Get bucketing parameters
            enable_bucketing = train_config.get('enable_bucketing', False)
            base_resolutions = train_config.get('base_resolutions', [1024])

            # Get latent caching parameters
            cache_latents_to_disk = True
            force_recache = False
            if 'datasets' in process_config and len(process_config['datasets']) > 0:
                cache_latents_to_disk = process_config['datasets'][0].get('cache_latents_to_disk', True)
                force_recache = process_config['datasets'][0].get('force_recache', False)

            # Save settings
            save_every_unit = process_config['save'].get('save_every_unit', 'steps')
            save_every = process_config['save'].get('save_every', 100)
            max_step_saves_to_keep = process_config['save'].get('max_step_saves_to_keep', 3)
            max_optimizer_saves_to_keep = process_config['save'].get(
                'max_optimizer_saves_to_keep', DEFAULT_MAX_OPTIMIZER_SAVES_TO_KEEP)

            save_every_n_steps = _resolve_save_every_n_steps(
                save_every_unit, save_every, len(dataset_items), train_config.get('batch_size', 1))
            if save_every_unit == 'epochs':
                print(f"[TrainRunner] Converted save_every={save_every} epochs to save_every_n_steps={save_every_n_steps}")

            print(f"[TrainRunner] Max step saves to keep: {max_step_saves_to_keep} "
                  f"(optimizer states: {max_optimizer_saves_to_keep})")

            sample_guidance_scale = sample_config["guidance_scale"]
            sample_steps = sample_config["sample_steps"]
            sample_width = sample_config["width"]
            sample_height = sample_config["height"]
            sample_seed = sample_config["seed"]
            sample_sampler = sample_config["sampler"]
            sample_schedule_type = sample_config["schedule_type"]
            sensenova_sample_timestep_shift = sample_config["sensenova_timestep_shift"]
            sensenova_sample_img_cfg_scale = sample_config["sensenova_img_cfg_scale"]
            sensenova_sample_cfg_norm = sample_config["sensenova_cfg_norm"]

            resume_from_checkpoint = train_config.get('resume_from_checkpoint')
            if resume_from_checkpoint:
                print(f"[TrainRunner] Resume from checkpoint: {resume_from_checkpoint}")

            if force_recache:
                print(f"[TrainRunner] Force recache enabled: all latent caches will be regenerated")

            # Text/Latent encoding modes
            text_encoding_mode = train_config.get('text_encoding_mode', 'swap_onthefly')
            text_encoding_swap_interval = train_config.get('text_encoding_swap_interval', 256)
            text_encoding_prefetch_depth = train_config.get('text_encoding_prefetch_depth', 4)
            latent_encoding_mode = train_config.get('latent_encoding_mode', 'swap_onthefly')
            latent_encoding_swap_interval = train_config.get('latent_encoding_swap_interval', 256)

            # Multi Noise Timestep (MNT) settings
            multi_noise_timesteps = train_config.get('multi_noise_timesteps', 1)
            multi_noise_mode = train_config.get('multi_noise_mode', 'independent')
            trajectory_blend_alpha = train_config.get('trajectory_blend_alpha', 0.7)
            timestep_sampling_config = train_config.get('timestep_sampling', None)

            # Reference image settings
            use_reference_images = train_config.get('use_reference_images', False)
            vision_encoder_path = train_config.get('vision_encoder_path', None)
            train_vision_encoder = train_config.get('train_vision_encoder', False)
            vision_encoder_lr = train_config.get('vision_encoder_lr', None)
            gradient_routing_ve = train_config.get('gradient_routing_ve', False)
            param_tracking = train_config.get('param_tracking', False)
            param_tracking_interval = train_config.get('param_tracking_interval', 100)

            # Start ReLoRA training
            trainer.train(
                datasets=training_datasets,
                num_epochs=num_epochs if num_epochs else 1,
                total_steps=total_steps_config,
                batch_size=train_config.get('batch_size', 1),
                save_every_n_steps=save_every_n_steps,
                sample_every_n_steps=sample_config["sample_every"],
                sample_prompts=sample_prompts,
                sample_guidance_scale=sample_guidance_scale,
                sample_steps=sample_steps,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_seed=sample_seed,
                sample_sampler=sample_sampler,
                sample_schedule_type=sample_schedule_type,
                sample_cfg_schedule_type=sample_config["cfg_schedule_type"],
                sample_cfg_schedule_min=sample_config["cfg_schedule_min"],
                sample_cfg_schedule_max=sample_config["cfg_schedule_max"],
                sample_cfg_schedule_power=sample_config["cfg_schedule_power"],
                sample_cfg_rescale_snr_alpha=sample_config["cfg_rescale_snr_alpha"],
                sample_dynamic_threshold_percentile=sample_config["dynamic_threshold_percentile"],
                sample_dynamic_threshold_mimic_scale=sample_config["dynamic_threshold_mimic_scale"],
                sample_nag_enable=sample_config["nag_enable"],
                sample_nag_scale=sample_config["nag_scale"],
                sample_nag_tau=sample_config["nag_tau"],
                sample_nag_alpha=sample_config["nag_alpha"],
                sample_nag_sigma_end=sample_config["nag_sigma_end"],
                sample_nag_negative_prompt=sample_config["nag_negative_prompt"],
                sensenova_sample_timestep_shift=sensenova_sample_timestep_shift,
                sensenova_sample_img_cfg_scale=sensenova_sample_img_cfg_scale,
                sensenova_sample_cfg_norm=sensenova_sample_cfg_norm,
                optimizer_type=optimizer_type,
                lr_scheduler_type=lr_scheduler_type,
                enable_bucketing=enable_bucketing,
                base_resolutions=base_resolutions,
                bucket_strategy="resize",
                multi_resolution_mode="max",
                gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
                max_grad_norm=train_config.get('max_grad_norm', 1.0),
                debug_latents=debug_latents,
                debug_latents_every=debug_latents_every,
                progress_callback=progress_callback,
                update_total_steps_callback=update_total_steps_callback,
                run_id=run_id,
                resume_from_checkpoint=resume_from_checkpoint,
                force_recache=force_recache,
                max_step_saves_to_keep=max_step_saves_to_keep,
                max_optimizer_saves_to_keep=max_optimizer_saves_to_keep,
                text_encoding_mode=text_encoding_mode,
                text_encoding_swap_interval=text_encoding_swap_interval,
                text_encoding_prefetch_depth=text_encoding_prefetch_depth,
                latent_encoding_mode=latent_encoding_mode,
                latent_encoding_swap_interval=latent_encoding_swap_interval,
                multi_noise_timesteps=multi_noise_timesteps,
                multi_noise_mode=multi_noise_mode,
                trajectory_blend_alpha=trajectory_blend_alpha,
                timestep_sampling_config=timestep_sampling_config,
                use_reference_images=use_reference_images,
                vision_encoder_path=vision_encoder_path,
                train_vision_encoder=train_vision_encoder,
                vision_encoder_lr=vision_encoder_lr,
                gradient_routing_ve=gradient_routing_ve,
                param_tracking=param_tracking,
                param_tracking_interval=param_tracking_interval,
                priority_training=priority_training,
            )

            print("[TrainRunner] ReLoRA training completed successfully!")

            # Update run status
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            training_db.commit()

        elif network_type == 'full_finetune':
            print("[TrainRunner] Training method: Full Parameter Fine-Tuning")
            from core.training.full_parameter_trainer import FullParameterTrainer

            # Get resume from checkpoint setting
            resume_from_checkpoint = train_config.get('resume_from_checkpoint')
            if resume_from_checkpoint:
                print(f"[TrainRunner] Resume from checkpoint: {resume_from_checkpoint}")

            # Get dtype settings from unified dtype section
            dtype_config = process_config.get('dtype', {})
            weight_dtype = dtype_config.get('weight', 'fp16')
            training_dtype = dtype_config.get('training', 'fp16')
            output_dtype = dtype_config.get('save', 'fp32')
            vae_dtype = dtype_config.get('vae', 'fp16')

            # Z-Image requires BFloat16 for numerical stability (trained with bf16)
            if 'z-image' in run.base_model_path.lower() or 'zimage' in run.base_model_path.lower():
                print("[TrainRunner] Z-Image model detected: forcing training_dtype=bf16 for numerical stability")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Anima (Cosmos-Predict2 DiT) is also trained in bf16 — force it
            # whenever the model path or arch metadata says so.
            if 'anima' in run.base_model_path.lower():
                print("[TrainRunner] Anima model detected: forcing training_dtype=bf16 for numerical stability")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Ideogram 4 (flow-matching DiT, fp8 base) is trained in bf16.
            if 'ideogram4' in run.base_model_path.lower() or 'ideogram-4' in run.base_model_path.lower():
                print("[TrainRunner] Ideogram 4 model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # MiniT2I (pixel-space MM-JiT, flow matching, x0 prediction) is trained in bf16.
            if 'minit2i' in run.base_model_path.lower():
                print("[TrainRunner] MiniT2I model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Krea 2 (single-stream flow-matching MMDiT) is trained in bf16.
            if _is_krea2_base_model(run.base_model_path):
                print("[TrainRunner] Krea 2 model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'
            # Lens / LTX-2.3 / ACE-Step (bf16-native DiT / audio) are trained in bf16 too
            # -- backend enforcement so API-created runs (not just the frontend preset)
            # get bf16, avoiding the fp16 Full-FT rejection.
            if _is_bf16_native_base_model(run.base_model_path):
                print("[TrainRunner] bf16-native model detected: forcing training_dtype=bf16")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'

            mixed_precision = train_config.get('mixed_precision', True)
            debug_vram = train_config.get('debug_vram', False)  # Debug VRAM profiling (default: False)
            use_flash_attention = train_config.get('use_flash_attention', False)  # Flash Attention (default: False)
            # Attention backend string selector (native|flash); back-compat maps the
            # legacy boolean to flash when the string key is absent.
            attention_backend = train_config.get('attention_backend') or ('flash' if use_flash_attention else 'native')
            # Attention implementation registry (conduit|diffusers). None when the
            # saved config lacks the key -> base_trainer applies resume backward-compat
            # (fresh -> conduit, resume w/o key -> diffusers) and persists the choice.
            attention_impl = train_config.get('attention_impl')
            min_snr_gamma = train_config.get('min_snr_gamma', 5.0)  # Min-SNR gamma weighting (default: 5.0)
            reconstruction_loss_weight = train_config.get('reconstruction_loss_weight', 0.0)  # Dual loss weight (default: 0.0, pred loss only)

            # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
            prompt_chunking_mode = train_config.get('prompt_chunking_mode', 'a1111')
            max_prompt_chunks = train_config.get('max_prompt_chunks', 0)

            # Get component-specific learning rates from train_config
            unet_lr = train_config.get('unet_lr')
            text_encoder_lr = train_config.get('text_encoder_lr')
            text_encoder_1_lr = train_config.get('text_encoder_1_lr')
            text_encoder_2_lr = train_config.get('text_encoder_2_lr')
            image_encoder_lr = train_config.get('image_encoder_lr')

            # train_unet was NOT read here, so the constructor default (True)
            # won every run even though `_build_train_section` always emits the
            # key. On SenseNova the two flags ARE the two MoT halves, so
            # train_unet=False + train_text_encoder=True asked for one half and
            # dequantized both.
            train_unet = train_config.get('train_unet', True)
            train_text_encoder = train_config.get('train_text_encoder', False)
            train_image_encoder = train_config.get('train_image_encoder', False)

            # Get optimizer options and hyperparameters from train_config.
            # Same keys, same defaults and the same read sites as the LoRA /
            # ReLoRA / ControlNet branches. They were absent here, so a full
            # fine-tune silently ran with the BaseTrainer fallbacks no matter
            # what the user set: betas (0.9, 0.999), eps 1e-8, weight_decay 0.01,
            # no cautious masking, no Schedule-Free -- and, because
            # optimizer_warmup_steps also feeds the LR scheduler's
            # num_warmup_steps, no LR warmup at all.
            optimizer_cautious = train_config.get('optimizer_cautious', False)
            optimizer_beta1 = train_config.get('optimizer_beta1')
            optimizer_beta2 = train_config.get('optimizer_beta2')
            optimizer_epsilon = train_config.get('optimizer_epsilon')
            optimizer_weight_decay = train_config.get('optimizer_weight_decay')

            # Schedule-Free optimizer options (RingBuffer optimizers only)
            optimizer_schedule_free = train_config.get('optimizer_schedule_free', False)
            optimizer_warmup_steps = train_config.get('optimizer_warmup_steps', 0)
            optimizer_schedule_free_r = train_config.get('optimizer_schedule_free_r', 0.0)
            optimizer_schedule_free_weight_lr_power = train_config.get('optimizer_schedule_free_weight_lr_power', 2.0)
            optimizer_use_radam = train_config.get('optimizer_use_radam', False)

            # Stochastic rounding for BF16 parameter updates (RingBuffer optimizers).
            # Full fine-tuning is where this matters: the block above forces
            # weight_dtype=bf16 for Z-Image, Anima, Ideogram 4, MiniT2I, Krea 2
            # and the bf16-native models (Lens / LTX-2.3 / ACE-Step), and BF16
            # round-to-nearest silently drops every optimizer update below half
            # a ULP (an element only moves when |w| <= 512*lr).
            # NOT covered: Flux2, SD1.5 and SDXL are absent from that block and
            # keep the configured weight dtype (fp16 by default). fp16 has the
            # same failure with a threshold of |w| <= 2048*lr, but stochastic
            # rounding as implemented applies to bf16 parameters only.
            optimizer_stochastic_rounding = train_config.get('optimizer_stochastic_rounding', False)

            # Initialize trainer
            trainer = FullParameterTrainer(
                model_path=run.base_model_path,
                output_dir=run.output_dir,
                run_name=run.run_name,  # Pass run_name for checkpoint filename generation
                run_id=run_id,  # Pass run_id for DB metrics logging
                learning_rate=train_config.get('lr', 1e-4),
                weight_dtype=weight_dtype,
                training_dtype=training_dtype,
                output_dtype=output_dtype,
                vae_dtype=vae_dtype,
                mixed_precision=mixed_precision,
                debug_vram=debug_vram,
                use_flash_attention=use_flash_attention,
                attention_backend=attention_backend,
                attention_impl=attention_impl,
                min_snr_gamma=min_snr_gamma,
                reconstruction_loss_weight=reconstruction_loss_weight,
                blocks_to_swap=train_config.get('blocks_to_swap', 0),
                use_pinned_memory=train_config.get('use_pinned_memory', False),
                activation_dispatch_enable=train_config.get('activation_dispatch_enable', False),
                activation_dispatch_margin_gb=train_config.get('activation_dispatch_margin_gb', 1.0),
                activation_dispatch_seed_coef=train_config.get('activation_dispatch_seed_coef', 24.0e-6),
                activation_dispatch_residual_frac=train_config.get('activation_dispatch_residual_frac', 0.85),
                activation_dispatch_threshold_mb=train_config.get('activation_dispatch_threshold_mb', 4),
                num_optimizer_groups=train_config.get('num_optimizer_groups', 0),
                # Optimizer options and hyperparameters
                optimizer_cautious=optimizer_cautious,
                optimizer_beta1=optimizer_beta1,
                optimizer_beta2=optimizer_beta2,
                optimizer_epsilon=optimizer_epsilon,
                optimizer_weight_decay=optimizer_weight_decay,
                # Schedule-Free optimizer options
                optimizer_schedule_free=optimizer_schedule_free,
                optimizer_warmup_steps=optimizer_warmup_steps,
                optimizer_schedule_free_r=optimizer_schedule_free_r,
                optimizer_schedule_free_weight_lr_power=optimizer_schedule_free_weight_lr_power,
                optimizer_use_radam=optimizer_use_radam,
                # Stochastic rounding for BF16 parameter updates
                optimizer_stochastic_rounding=optimizer_stochastic_rounding,
                # Component-specific learning rates
                unet_lr=unet_lr,
                text_encoder_lr=text_encoder_lr,
                text_encoder_1_lr=text_encoder_1_lr,
                text_encoder_2_lr=text_encoder_2_lr,
                image_encoder_lr=image_encoder_lr,
                # Training scope control
                train_unet=train_unet,
                train_text_encoder=train_text_encoder,
                train_image_encoder=train_image_encoder,
                # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
                prompt_chunking_mode=prompt_chunking_mode,
                max_prompt_chunks=max_prompt_chunks,
                # Resume training
                resume_from_checkpoint=resume_from_checkpoint,
                # See LoRATrainer construction above for why this is needed.
                train_config=train_config,
            )

            # Note: setup_optimizer() is now called inside train() method
            # This avoids double initialization and provides clearer separation of concerns

            # Get optimizer settings (passed to train() method)
            optimizer_type = train_config.get('optimizer', 'adamw8bit')
            lr_scheduler_type = train_config.get('lr_scheduler', 'constant')

            # ============================================================
            # Validate Prediction Configuration (Unified Framework)
            # ============================================================
            from core.model_loader import ModelLoader

            # Detect model's prediction configuration
            model_type = ModelLoader.detect_model_type(run.base_model_path)
            model_pred_config = ModelLoader.detect_prediction_config(run.base_model_path, model_type)

            print(f"[TrainRunner] Model prediction configuration detected:")
            print(f"  Noise Process: {model_pred_config['noise_process']}")
            print(f"  Prediction Target: {model_pred_config['prediction_target']}")
            print(f"  Detection Source: {model_pred_config['source']}")

            # Get training configuration (with "auto" support)
            training_noise_process = train_config.get('noise_process', 'auto')
            training_prediction_target = train_config.get('prediction_target', 'auto')
            strict_validation = train_config.get('strict_validation', False)

            # Auto-detect: use model's configuration
            if training_noise_process == 'auto':
                training_noise_process = model_pred_config['noise_process']
                print(f"[TrainRunner] noise_process='auto' → using model's config: {training_noise_process}")

            if training_prediction_target == 'auto':
                training_prediction_target = model_pred_config['prediction_target']
                print(f"[TrainRunner] prediction_target='auto' → using model's config: {training_prediction_target}")

            # Validate compatibility
            mismatch_warnings = []
            if training_noise_process != model_pred_config['noise_process']:
                mismatch_warnings.append(
                    f"noise_process mismatch: model={model_pred_config['noise_process']}, training={training_noise_process}"
                )
            if training_prediction_target != model_pred_config['prediction_target']:
                mismatch_warnings.append(
                    f"prediction_target mismatch: model={model_pred_config['prediction_target']}, training={training_prediction_target}"
                )

            if mismatch_warnings:
                print(f"\n{'='*60}")
                print(f"[TrainRunner] WARNING: PREDICTION CONFIG MISMATCH DETECTED")
                print(f"{'='*60}")
                for warning in mismatch_warnings:
                    print(f"  - {warning}")
                print(f"\nThis may cause training instability or poor convergence.")
                print(f"Model was trained with: {model_pred_config['noise_process']} + {model_pred_config['prediction_target']}")
                print(f"You are training with: {training_noise_process} + {training_prediction_target}")

                if strict_validation:
                    print(f"\nERROR: strict_validation=True: Aborting training due to mismatch.")
                    print(f"{'='*60}\n")
                    sys.exit(1)
                else:
                    print(f"\nWARNING: strict_validation=False: Continuing with warning.")
                    print(f"Set strict_validation=true in training config to abort on mismatch.")
                    print(f"{'='*60}\n")
            else:
                print(f"[TrainRunner] OK Prediction configuration validated successfully")

            # Store final training config for trainer
            trainer.noise_process = training_noise_process
            trainer.prediction_target = training_prediction_target

            # ============================================================
            # Setup Regularization Loss (SNR or Energy)
            # ============================================================
            regularization_type = train_config.get('regularization_type', None)
            if regularization_type:
                print(f"[TrainRunner] Initializing {regularization_type.upper()} regularization...")
                trainer.config = train_config  # Pass config for factory function

                if regularization_type.lower() == 'snr':
                    from core.training.losses.snr_regularization import create_snr_regularization_loss
                    trainer.snr_regularization_loss = create_snr_regularization_loss(train_config)
                    print(f"[TrainRunner] SNR Regularization enabled:")
                    print(f"  Weight: {train_config.get('snr_regularization_weight', 0.1)}")
                    print(f"  Timestep adaptive: {train_config.get('snr_timestep_adaptive', True)}")
                    print(f"  Penalty mode: {train_config.get('snr_penalty_mode', 'relu')}")
                elif regularization_type.lower() == 'energy':
                    from core.training.losses.energy_regularization import create_energy_regularization_loss
                    trainer.energy_regularization_loss = create_energy_regularization_loss(train_config)
                    print(f"[TrainRunner] Energy Regularization enabled:")
                    print(f"  Weight: {train_config.get('energy_regularization_weight', 0.05)}")
                    print(f"  Timestep adaptive: {train_config.get('energy_timestep_adaptive', True)}")
                    print(f"  Penalty mode: {train_config.get('energy_penalty_mode', 'abs')}")
                    print(f"  Normalize by pixels: {train_config.get('energy_normalize_by_pixels', True)}")
                else:
                    print(f"[TrainRunner] WARNING: Unknown regularization type '{regularization_type}', skipping")
            else:
                print(f"[TrainRunner] Regularization disabled (regularization_type not set)")

            # Determine epochs or steps
            num_epochs = train_config.get('epochs', None)
            total_steps_config = train_config.get('steps', None)

            if num_epochs:
                print(f"[TrainRunner] Training for {num_epochs} epochs")
            elif total_steps_config:
                num_epochs = None  # Will be calculated by trainer
                print(f"[TrainRunner] Training for {total_steps_config} steps (epochs will be calculated by trainer)")
            else:
                num_epochs = 1

            # Progress callback
            def progress_callback(phase: str, step: int, total: int, epoch: int = 0, loss: float = None, detail: str = None):
                # Get current learning rate from optimizer (if available)
                lr = None
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    lr = trainer.optimizer.param_groups[0]['lr']
                    # Debug: Log LR retrieval
                    if phase == "training" and step % 100 == 0:
                        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                        print(f"[ProgressCallback] Step {step}: LR={lr:.2e}, Loss={loss_str}")
                update_training_progress(training_db, run_id, phase, step, total, epoch, loss, lr, detail)

            # Total steps callback
            def update_total_steps_callback(total_steps: int):
                print(f"[TrainRunner] Updating total_steps in DB: {total_steps}")
                run.total_steps = total_steps
                training_db.commit()

            # Update status to running
            run.status = "running"
            training_db.commit()
            print("[TrainRunner] Status updated to 'running'")

            sample_config = _resolve_training_sample_config(process_config, model_type)
            sample_prompts = sample_config["prompts"]

            # Debug: Log sample generation settings
            print(f"[TrainRunner] Sample generation settings:")
            print(f"  sample_every: {sample_config['sample_every']}")
            print(f"  sample_prompts: {len(sample_prompts) if sample_prompts else 0} prompts")
            if sample_prompts:
                for i, prompt in enumerate(sample_prompts):
                    print(f"    Prompt {i}: positive={prompt.get('positive', '')[:50]}..., negative={prompt.get('negative', '')[:50]}...")
            print(f"  sample_config: {sample_config}")

            # Get debug parameters from config
            debug_latents = train_config.get('debug_latents', False)
            debug_latents_every = train_config.get('debug_latents_every', 50)

            # Get bucketing parameters from config
            enable_bucketing = train_config.get('enable_bucketing', False)
            base_resolutions = train_config.get('base_resolutions', [1024])
            bucket_strategy = train_config.get('bucket_strategy', 'resize')
            multi_resolution_mode = train_config.get('multi_resolution_mode', 'max')

            # Get latent caching parameters
            cache_latents_to_disk = True  # Default
            force_recache = False  # Default
            if 'datasets' in process_config and len(process_config['datasets']) > 0:
                cache_latents_to_disk = process_config['datasets'][0].get('cache_latents_to_disk', True)
                force_recache = process_config['datasets'][0].get('force_recache', False)

            # Convert save_every parameters to new interface (save_every_n_steps)
            save_every_unit = process_config['save'].get('save_every_unit', 'steps')
            save_every = process_config['save'].get('save_every', 100)
            max_step_saves_to_keep = process_config['save'].get('max_step_saves_to_keep', 3)
            max_optimizer_saves_to_keep = process_config['save'].get(
                'max_optimizer_saves_to_keep', DEFAULT_MAX_OPTIMIZER_SAVES_TO_KEEP)

            save_every_n_steps = _resolve_save_every_n_steps(
                save_every_unit, save_every, len(dataset_items), train_config.get('batch_size', 1))
            if save_every_unit == 'epochs':
                print(f"[TrainRunner] Converted save_every={save_every} epochs to save_every_n_steps={save_every_n_steps}")

            print(f"[TrainRunner] Max step saves to keep: {max_step_saves_to_keep} "
                  f"(optimizer states: {max_optimizer_saves_to_keep})")

            # Get sample generation settings
            sample_guidance_scale = sample_config["guidance_scale"]
            sample_steps = sample_config["sample_steps"]
            sample_width = sample_config["width"]
            sample_height = sample_config["height"]
            sample_seed = sample_config["seed"]
            sample_sampler = sample_config["sampler"]
            sample_schedule_type = sample_config["schedule_type"]
            sensenova_sample_timestep_shift = sample_config["sensenova_timestep_shift"]
            sensenova_sample_img_cfg_scale = sample_config["sensenova_img_cfg_scale"]
            sensenova_sample_cfg_norm = sample_config["sensenova_cfg_norm"]
            print(f"[TrainRunner] Sample generation config: width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, sample_steps={sample_steps}, sampler={sample_sampler}, schedule_type={sample_schedule_type}, seed={sample_seed}")

            # Get resume from checkpoint setting
            resume_from_checkpoint = train_config.get('resume_from_checkpoint')
            if resume_from_checkpoint:
                print(f"[TrainRunner] Resume from checkpoint: {resume_from_checkpoint}")

            # Log force_recache setting
            if force_recache:
                print(f"[TrainRunner] Force recache enabled: all latent caches will be regenerated")

            # Get text encoding mode
            text_encoding_mode = train_config.get('text_encoding_mode', 'swap_onthefly')
            text_encoding_swap_interval = train_config.get('text_encoding_swap_interval', 256)
            text_encoding_prefetch_depth = train_config.get('text_encoding_prefetch_depth', 4)

            # Get latent encoding mode
            latent_encoding_mode = train_config.get('latent_encoding_mode', 'swap_onthefly')
            latent_encoding_swap_interval = train_config.get('latent_encoding_swap_interval', 256)

            # Get Multi Noise-Timestep (MNT) settings
            multi_noise_timesteps = train_config.get('multi_noise_timesteps', 1)
            multi_noise_mode = train_config.get('multi_noise_mode', 'independent')
            trajectory_blend_alpha = train_config.get('trajectory_blend_alpha', 0.7)
            timestep_sampling_config = train_config.get('timestep_sampling', None)

            # Get reference image settings
            use_reference_images = train_config.get('use_reference_images', False)
            vision_encoder_path = train_config.get('vision_encoder_path', None)
            train_vision_encoder = train_config.get('train_vision_encoder', False)
            vision_encoder_lr = train_config.get('vision_encoder_lr', None)
            gradient_routing_ve = train_config.get('gradient_routing_ve', False)
            param_tracking = train_config.get('param_tracking', False)
            param_tracking_interval = train_config.get('param_tracking_interval', 100)

            # Get priority training settings (inline dict or legacy file path)
            priority_training = train_config.get('priority_training', None)
            # Legacy support: if it's a string path, load from file
            if isinstance(priority_training, str):
                from core.training.priority_training import PriorityTrainingConfig
                priority_training = {"_legacy_path": priority_training}

            # Start training with new interface
            trainer.train(
                datasets=training_datasets,
                num_epochs=num_epochs if num_epochs else 1,
                total_steps=total_steps_config,  # Pass total_steps from YAML
                batch_size=train_config.get('batch_size', 1),
                save_every_n_steps=save_every_n_steps,
                sample_every_n_steps=sample_config["sample_every"],
                sample_prompts=sample_prompts,
                sample_guidance_scale=sample_guidance_scale,
                sample_steps=sample_steps,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_seed=sample_seed,
                sample_sampler=sample_sampler,
                sample_schedule_type=sample_schedule_type,
                sample_cfg_schedule_type=sample_config["cfg_schedule_type"],
                sample_cfg_schedule_min=sample_config["cfg_schedule_min"],
                sample_cfg_schedule_max=sample_config["cfg_schedule_max"],
                sample_cfg_schedule_power=sample_config["cfg_schedule_power"],
                sample_cfg_rescale_snr_alpha=sample_config["cfg_rescale_snr_alpha"],
                sample_dynamic_threshold_percentile=sample_config["dynamic_threshold_percentile"],
                sample_dynamic_threshold_mimic_scale=sample_config["dynamic_threshold_mimic_scale"],
                sample_nag_enable=sample_config["nag_enable"],
                sample_nag_scale=sample_config["nag_scale"],
                sample_nag_tau=sample_config["nag_tau"],
                sample_nag_alpha=sample_config["nag_alpha"],
                sample_nag_sigma_end=sample_config["nag_sigma_end"],
                sample_nag_negative_prompt=sample_config["nag_negative_prompt"],
                sensenova_sample_timestep_shift=sensenova_sample_timestep_shift,
                sensenova_sample_img_cfg_scale=sensenova_sample_img_cfg_scale,
                sensenova_sample_cfg_norm=sensenova_sample_cfg_norm,
                optimizer_type=optimizer_type,
                lr_scheduler_type=lr_scheduler_type,
                enable_bucketing=enable_bucketing,
                base_resolutions=base_resolutions,
                bucket_strategy="resize",
                multi_resolution_mode="max",
                gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
                max_grad_norm=train_config.get('max_grad_norm', 1.0),
                debug_latents=debug_latents,
                debug_latents_every=debug_latents_every,
                progress_callback=progress_callback,
                update_total_steps_callback=update_total_steps_callback,
                run_id=run_id,
                resume_from_checkpoint=resume_from_checkpoint,
                force_recache=force_recache,
                max_step_saves_to_keep=max_step_saves_to_keep,
                max_optimizer_saves_to_keep=max_optimizer_saves_to_keep,
                text_encoding_mode=text_encoding_mode,
                text_encoding_swap_interval=text_encoding_swap_interval,
                text_encoding_prefetch_depth=text_encoding_prefetch_depth,
                latent_encoding_mode=latent_encoding_mode,
                latent_encoding_swap_interval=latent_encoding_swap_interval,
                multi_noise_timesteps=multi_noise_timesteps,
                multi_noise_mode=multi_noise_mode,
                trajectory_blend_alpha=trajectory_blend_alpha,
                timestep_sampling_config=timestep_sampling_config,
                use_reference_images=use_reference_images,
                vision_encoder_path=vision_encoder_path,
                train_vision_encoder=train_vision_encoder,
                vision_encoder_lr=vision_encoder_lr,
                gradient_routing_ve=gradient_routing_ve,
                param_tracking=param_tracking,
                param_tracking_interval=param_tracking_interval,
                priority_training=priority_training,
            )

            print("[TrainRunner] Training completed successfully!")

            # Update run status
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            training_db.commit()

        elif network_type == 'controlnet':
            print("[TrainRunner] Training method: ControlNet")
            from core.training.controlnet_trainer import ControlNetTrainer

            # Get dtype settings from unified dtype section
            dtype_config = process_config.get('dtype', {})
            weight_dtype = dtype_config.get('weight', 'fp16')
            training_dtype = dtype_config.get('training', 'fp16')
            output_dtype = dtype_config.get('save', 'fp32')
            vae_dtype = dtype_config.get('vae', 'fp16')

            mixed_precision = train_config.get('mixed_precision', True)
            debug_vram = train_config.get('debug_vram', False)
            use_flash_attention = train_config.get('use_flash_attention', False)
            # Attention backend string selector (native|flash); back-compat maps the
            # legacy boolean to flash when the string key is absent.
            attention_backend = train_config.get('attention_backend') or ('flash' if use_flash_attention else 'native')
            # Attention implementation registry (conduit|diffusers). None when the
            # saved config lacks the key -> base_trainer applies resume backward-compat
            # (fresh -> conduit, resume w/o key -> diffusers) and persists the choice.
            attention_impl = train_config.get('attention_impl')
            min_snr_gamma = train_config.get('min_snr_gamma', 5.0)

            # ControlNet-specific parameters
            controlnet_config = network_config.get('controlnet', {})
            controlnet_type = controlnet_config.get('type', 'standard')
            controlnet_pretrained_path = controlnet_config.get('pretrained_path')
            init_from_unet = controlnet_config.get('init_from_unet', True)
            lllite_conditioning_channels = controlnet_config.get('lllite_conditioning_channels', 32)
            lllite_rank = controlnet_config.get('lllite_rank', 64)

            # Condition generation parameters
            condition_preprocessors = controlnet_config.get('condition_preprocessors')
            condition_cache_mode = controlnet_config.get('condition_cache_mode', 'on_the_fly')

            # Outpaint-native conditioning (PART B)
            conditioning_mode = controlnet_config.get('conditioning_mode', 'preprocessor')
            outpaint_crop_min_area = controlnet_config.get('outpaint_crop_min_area', 0.15)
            outpaint_crop_max_area = controlnet_config.get('outpaint_crop_max_area', 0.8)
            outpaint_edge_anchor_prob = controlnet_config.get('outpaint_edge_anchor_prob', 0.34)
            outpaint_corner_anchor_prob = controlnet_config.get('outpaint_corner_anchor_prob', 0.33)
            outpaint_mask_channel = controlnet_config.get('outpaint_mask_channel', True)
            outpaint_known_loss_weight = controlnet_config.get('outpaint_known_loss_weight', 0.3)
            outpaint_seam_loss_boost = controlnet_config.get('outpaint_seam_loss_boost', 0.0)
            outpaint_seam_ring_width = controlnet_config.get('outpaint_seam_ring_width', 1)
            outpaint_seam_grad_lambda = controlnet_config.get('outpaint_seam_grad_lambda', 0.0)
            outpaint_loss_normalize = controlnet_config.get('outpaint_loss_normalize', False)
            # R1 (scratchpad/outpaint_boundary_structure_fix.md D3-R1)
            outpaint_edge_feather_min_px = controlnet_config.get('outpaint_edge_feather_min_px', 0.0)
            outpaint_edge_feather_max_px = controlnet_config.get('outpaint_edge_feather_max_px', 0.0)

            # Prompt chunking settings
            prompt_chunking_mode = train_config.get('prompt_chunking_mode', 'a1111')
            max_prompt_chunks = train_config.get('max_prompt_chunks', 0)

            # Get component-specific learning rates
            unet_lr = train_config.get('unet_lr')

            # Get optimizer options and hyperparameters
            optimizer_cautious = train_config.get('optimizer_cautious', False)
            optimizer_beta1 = train_config.get('optimizer_beta1')
            optimizer_beta2 = train_config.get('optimizer_beta2')
            optimizer_epsilon = train_config.get('optimizer_epsilon')
            optimizer_weight_decay = train_config.get('optimizer_weight_decay')

            # Schedule-Free optimizer options
            optimizer_schedule_free = train_config.get('optimizer_schedule_free', False)
            optimizer_warmup_steps = train_config.get('optimizer_warmup_steps', 0)
            optimizer_schedule_free_r = train_config.get('optimizer_schedule_free_r', 0.0)
            optimizer_schedule_free_weight_lr_power = train_config.get('optimizer_schedule_free_weight_lr_power', 2.0)
            optimizer_use_radam = train_config.get('optimizer_use_radam', False)
            # Stochastic rounding for BF16 parameter updates (RingBuffer optimizers)
            optimizer_stochastic_rounding = train_config.get('optimizer_stochastic_rounding', False)

            # Initialize ControlNet trainer
            trainer = ControlNetTrainer(
                model_path=run.base_model_path,
                output_dir=run.output_dir,
                run_name=run.run_name,
                run_id=run_id,
                controlnet_type=controlnet_type,
                controlnet_pretrained_path=controlnet_pretrained_path,
                init_from_unet=init_from_unet,
                # ControlNet loads its own (directory/lllite) checkpoint in __init__
                # on resume, so the constructor needs this (base_trainer only used it
                # via .train() before, which is why CN resume silently restarted).
                resume_from_checkpoint=train_config.get('resume_from_checkpoint'),
                lllite_conditioning_channels=lllite_conditioning_channels,
                lllite_rank=lllite_rank,
                condition_preprocessors=condition_preprocessors,
                condition_cache_mode=condition_cache_mode,
                conditioning_mode=conditioning_mode,
                outpaint_crop_min_area=outpaint_crop_min_area,
                outpaint_crop_max_area=outpaint_crop_max_area,
                outpaint_edge_anchor_prob=outpaint_edge_anchor_prob,
                outpaint_corner_anchor_prob=outpaint_corner_anchor_prob,
                outpaint_mask_channel=outpaint_mask_channel,
                outpaint_known_loss_weight=outpaint_known_loss_weight,
                outpaint_seam_loss_boost=outpaint_seam_loss_boost,
                outpaint_seam_ring_width=outpaint_seam_ring_width,
                outpaint_seam_grad_lambda=outpaint_seam_grad_lambda,
                outpaint_loss_normalize=outpaint_loss_normalize,
                outpaint_edge_feather_min_px=outpaint_edge_feather_min_px,
                outpaint_edge_feather_max_px=outpaint_edge_feather_max_px,
                learning_rate=train_config.get('lr', 1e-4),
                weight_dtype=weight_dtype,
                training_dtype=training_dtype,
                output_dtype=output_dtype,
                vae_dtype=vae_dtype,
                mixed_precision=mixed_precision,
                debug_vram=debug_vram,
                use_flash_attention=use_flash_attention,
                attention_backend=attention_backend,
                attention_impl=attention_impl,
                min_snr_gamma=min_snr_gamma,
                # Component-specific learning rates
                unet_lr=unet_lr,
                # Optimizer options and hyperparameters
                optimizer_cautious=optimizer_cautious,
                optimizer_beta1=optimizer_beta1,
                optimizer_beta2=optimizer_beta2,
                optimizer_epsilon=optimizer_epsilon,
                optimizer_weight_decay=optimizer_weight_decay,
                # Schedule-Free optimizer options
                optimizer_schedule_free=optimizer_schedule_free,
                optimizer_warmup_steps=optimizer_warmup_steps,
                optimizer_schedule_free_r=optimizer_schedule_free_r,
                optimizer_schedule_free_weight_lr_power=optimizer_schedule_free_weight_lr_power,
                optimizer_use_radam=optimizer_use_radam,
                optimizer_stochastic_rounding=optimizer_stochastic_rounding,
                # Prompt chunking settings
                prompt_chunking_mode=prompt_chunking_mode,
                max_prompt_chunks=max_prompt_chunks,
                # See LoRATrainer construction above for why this is needed.
                train_config=train_config,
            )

            # Get optimizer settings
            optimizer_type = train_config.get('optimizer', 'adamw8bit')
            lr_scheduler_type = train_config.get('lr_scheduler', 'constant')

            # Validate Prediction Configuration
            from core.model_loader import ModelLoader
            model_type = ModelLoader.detect_model_type(run.base_model_path)
            model_pred_config = ModelLoader.detect_prediction_config(run.base_model_path, model_type)

            print(f"[TrainRunner] Model prediction configuration:")
            print(f"  Noise Process: {model_pred_config['noise_process']}")
            print(f"  Prediction Target: {model_pred_config['prediction_target']}")

            training_noise_process = train_config.get('noise_process', 'auto')
            training_prediction_target = train_config.get('prediction_target', 'auto')

            if training_noise_process == 'auto':
                training_noise_process = model_pred_config['noise_process']
            if training_prediction_target == 'auto':
                training_prediction_target = model_pred_config['prediction_target']

            trainer.noise_process = training_noise_process
            trainer.prediction_target = training_prediction_target

            # Determine epochs or steps
            num_epochs = train_config.get('epochs', None)
            total_steps_config = train_config.get('steps', None)

            if num_epochs:
                print(f"[TrainRunner] Training for {num_epochs} epochs")
            elif total_steps_config:
                num_epochs = None  # Will be calculated by trainer
                print(f"[TrainRunner] Training for {total_steps_config} steps (epochs will be calculated by trainer)")
            else:
                num_epochs = 1

            # Progress callback (update DB only)
            def progress_callback(phase: str, step: int, total: int, epoch: int = 0, loss: float = None, detail: str = None):
                lr = None
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    lr = trainer.optimizer.param_groups[0]['lr']
                    if phase == "training" and step % 100 == 0:
                        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                        print(f"[ProgressCallback] Step {step}: LR={lr:.2e}, Loss={loss_str}")
                update_training_progress(training_db, run_id, phase, step, total, epoch, loss, lr, detail)

            # Total steps callback
            def update_total_steps_callback(total_steps: int):
                print(f"[TrainRunner] Updating total_steps in DB: {total_steps}")
                run.total_steps = total_steps
                training_db.commit()

            # Update status to running
            run.status = "running"
            training_db.commit()
            print("[TrainRunner] Status updated to 'running'")

            sample_config = _resolve_training_sample_config(process_config, model_type)
            sample_prompts = sample_config["prompts"]

            # Legacy migration: if old-style sample_condition_image_path exists at sample level,
            # apply it to all prompts that don't have their own condition_image_path
            legacy_condition_path = process_config['sample'].get('sample_condition_image_path')
            if legacy_condition_path:
                print(f"[TrainRunner] Migrating legacy sample_condition_image_path to per-prompt format: {legacy_condition_path}")
                for prompt in sample_prompts:
                    if isinstance(prompt, dict) and not prompt.get('condition_image_path'):
                        prompt['condition_image_path'] = legacy_condition_path

            # Get sample generation settings
            sample_guidance_scale = sample_config["guidance_scale"]
            sample_steps = sample_config["sample_steps"]
            sample_width = sample_config["width"]
            sample_height = sample_config["height"]
            sample_seed = sample_config["seed"]
            sample_sampler = sample_config["sampler"]
            sample_schedule_type = sample_config["schedule_type"]
            sensenova_sample_timestep_shift = sample_config["sensenova_timestep_shift"]
            sensenova_sample_img_cfg_scale = sample_config["sensenova_img_cfg_scale"]
            sensenova_sample_cfg_norm = sample_config["sensenova_cfg_norm"]
            print(f"[TrainRunner] Sample generation config: width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, sample_steps={sample_steps}, sampler={sample_sampler}, schedule_type={sample_schedule_type}, seed={sample_seed}")

            # Get debug parameters from config
            debug_latents = train_config.get('debug_latents', False)
            debug_latents_every = train_config.get('debug_latents_every', 50)

            # Get bucketing parameters from config
            enable_bucketing = train_config.get('enable_bucketing', False)
            base_resolutions = train_config.get('base_resolutions', [1024])

            # Get latent caching parameters
            cache_latents_to_disk = True  # Default
            force_recache = False  # Default
            if 'datasets' in process_config and len(process_config['datasets']) > 0:
                cache_latents_to_disk = process_config['datasets'][0].get('cache_latents_to_disk', True)
                force_recache = process_config['datasets'][0].get('force_recache', False)

            # Convert save_every parameters to new interface (save_every_n_steps)
            save_every_unit = process_config['save'].get('save_every_unit', 'steps')
            save_every = process_config['save'].get('save_every', 100)
            max_step_saves_to_keep = process_config['save'].get('max_step_saves_to_keep', 3)
            max_optimizer_saves_to_keep = process_config['save'].get(
                'max_optimizer_saves_to_keep', DEFAULT_MAX_OPTIMIZER_SAVES_TO_KEEP)

            save_every_n_steps = _resolve_save_every_n_steps(
                save_every_unit, save_every, len(dataset_items), train_config.get('batch_size', 1))
            if save_every_unit == 'epochs':
                print(f"[TrainRunner] Converted save_every={save_every} epochs to save_every_n_steps={save_every_n_steps}")

            print(f"[TrainRunner] Max step saves to keep: {max_step_saves_to_keep} "
                  f"(optimizer states: {max_optimizer_saves_to_keep})")

            # Get resume from checkpoint setting
            resume_from_checkpoint = train_config.get('resume_from_checkpoint')
            if resume_from_checkpoint:
                print(f"[TrainRunner] Resume from checkpoint: {resume_from_checkpoint}")

            # Log force_recache setting
            if force_recache:
                print(f"[TrainRunner] Force recache enabled: all latent caches will be regenerated")

            # Get text encoding mode
            text_encoding_mode = train_config.get('text_encoding_mode', 'swap_onthefly')
            text_encoding_swap_interval = train_config.get('text_encoding_swap_interval', 256)
            text_encoding_prefetch_depth = train_config.get('text_encoding_prefetch_depth', 4)

            # Get latent encoding mode
            latent_encoding_mode = train_config.get('latent_encoding_mode', 'swap_onthefly')
            latent_encoding_swap_interval = train_config.get('latent_encoding_swap_interval', 256)

            # Get Multi Noise-Timestep (MNT) settings
            multi_noise_timesteps = train_config.get('multi_noise_timesteps', 1)
            multi_noise_mode = train_config.get('multi_noise_mode', 'independent')
            trajectory_blend_alpha = train_config.get('trajectory_blend_alpha', 0.7)
            timestep_sampling_config = train_config.get('timestep_sampling', None)

            # Start training
            trainer.train(
                datasets=training_datasets,
                num_epochs=num_epochs if num_epochs else 1,
                total_steps=total_steps_config,
                batch_size=train_config.get('batch_size', 1),
                save_every_n_steps=save_every_n_steps,
                sample_every_n_steps=sample_config["sample_every"],
                sample_prompts=sample_prompts,
                sample_guidance_scale=sample_guidance_scale,
                sample_steps=sample_steps,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_seed=sample_seed,
                sample_sampler=sample_sampler,
                sample_schedule_type=sample_schedule_type,
                sample_cfg_schedule_type=sample_config["cfg_schedule_type"],
                sample_cfg_schedule_min=sample_config["cfg_schedule_min"],
                sample_cfg_schedule_max=sample_config["cfg_schedule_max"],
                sample_cfg_schedule_power=sample_config["cfg_schedule_power"],
                sample_cfg_rescale_snr_alpha=sample_config["cfg_rescale_snr_alpha"],
                sample_dynamic_threshold_percentile=sample_config["dynamic_threshold_percentile"],
                sample_dynamic_threshold_mimic_scale=sample_config["dynamic_threshold_mimic_scale"],
                sample_nag_enable=sample_config["nag_enable"],
                sample_nag_scale=sample_config["nag_scale"],
                sample_nag_tau=sample_config["nag_tau"],
                sample_nag_alpha=sample_config["nag_alpha"],
                sample_nag_sigma_end=sample_config["nag_sigma_end"],
                sample_nag_negative_prompt=sample_config["nag_negative_prompt"],
                sensenova_sample_timestep_shift=sensenova_sample_timestep_shift,
                sensenova_sample_img_cfg_scale=sensenova_sample_img_cfg_scale,
                sensenova_sample_cfg_norm=sensenova_sample_cfg_norm,
                optimizer_type=optimizer_type,
                lr_scheduler_type=lr_scheduler_type,
                enable_bucketing=enable_bucketing,
                base_resolutions=base_resolutions,
                bucket_strategy="resize",
                multi_resolution_mode="max",
                gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
                max_grad_norm=train_config.get('max_grad_norm', 1.0),
                debug_latents=debug_latents,
                debug_latents_every=debug_latents_every,
                progress_callback=progress_callback,
                update_total_steps_callback=update_total_steps_callback,
                run_id=run_id,
                resume_from_checkpoint=resume_from_checkpoint,
                force_recache=force_recache,
                max_step_saves_to_keep=max_step_saves_to_keep,
                max_optimizer_saves_to_keep=max_optimizer_saves_to_keep,
                text_encoding_mode=text_encoding_mode,
                text_encoding_swap_interval=text_encoding_swap_interval,
                text_encoding_prefetch_depth=text_encoding_prefetch_depth,
                latent_encoding_mode=latent_encoding_mode,
                latent_encoding_swap_interval=latent_encoding_swap_interval,
                multi_noise_timesteps=multi_noise_timesteps,
                multi_noise_mode=multi_noise_mode,
                trajectory_blend_alpha=trajectory_blend_alpha,
                timestep_sampling_config=timestep_sampling_config,
            )

            print("[TrainRunner] Training completed successfully!")

            # Update run status
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            training_db.commit()

        elif network_type == 'vae_decoder':
            # Decoder-only VAE fine-tune (design.md Phase 1).
            #
            # This branch deliberately does NOT touch any of the diffusion
            # plumbing above it (dtype forcing, noise process/prediction target
            # detection, samplers, latent/TE caches): a VAE trainer has no
            # denoiser, no scheduler and no text encoder. It reuses only the
            # pieces that hang off the TrainingRun row -- the dataset items
            # already loaded above, the .stop_training sentinel, the checkpoint
            # routes and the TrainingMetrics chart channel.
            print("[TrainRunner] Training method: VAE decoder fine-tune")
            from core.training.vae.vae_config import (
                VaeConfigError, resolve_vae_training_config,
            )
            from core.training.vae.vae_trainer import VaeTrainer

            try:
                vae_cfg = resolve_vae_training_config(
                    process_config, base_model_path=run.base_model_path or "")
            except VaeConfigError as e:
                # A refused configuration is a user error, not a crash: report it
                # verbatim on the run so the UI shows the actionable message.
                print(f"[TrainRunner] VAE training config REFUSED: {e}")
                run.status = "failed"
                run.error_message = str(e)
                training_db.commit()
                sys.exit(1)

            print(f"[TrainRunner] VAE config: {vae_cfg}")

            # Flatten the dataset wrappers back into a plain item list; the VAE
            # dataset only needs image_path (train_runner.py:549-556).
            vae_items = [item for ds in training_datasets for item in ds.items]
            print(f"[TrainRunner] VAE training items: {len(vae_items)}")

            run.total_steps = int(vae_cfg["total_steps"])
            run.status = "running"
            training_db.commit()
            print("[TrainRunner] Status updated to 'running'")

            def vae_progress_callback(phase: str, step: int, total: int,
                                      epoch: int = 0, loss: float = None,
                                      lr: float = None):
                update_training_progress(training_db, run_id, phase, step, total,
                                         epoch, loss, lr)

            trainer = VaeTrainer(
                vae_cfg,
                output_dir=run.output_dir,
                run_name=run.run_name,
                run_id=run_id,
                progress_callback=vae_progress_callback,
            )
            try:
                stopped = trainer.train(vae_items)
            finally:
                trainer.cleanup()

            if stopped:
                print("[TrainRunner] VAE training stopped by user")
                run.status = "stopped"
                run.phase_detail = "Stopped by user"
                run.completed_at = datetime.utcnow()
                training_db.commit()
            else:
                print("[TrainRunner] Training completed successfully!")
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                training_db.commit()

        else:
            print(f"[TrainRunner] ERROR: Unsupported network type: {network_type}")
            sys.exit(1)

    except KeyboardInterrupt:
        # Distinguish an intentional user stop (SIGTERM/SIGINT from
        # TrainingProcess.stop(), or a .stop_training flag caught by
        # _check_init_stop() during dataset scan/caption processing/
        # bucketing) from an actual failure. KeyboardInterrupt is a
        # BaseException, not an Exception, so it is never caught by the
        # `except Exception` below -- an aborted initialization is always
        # correctly reported as "stopped", never "failed".
        #
        # No model/checkpoint exists yet during init, and the dataset scan
        # is in-memory (plus a tolerant pickle cache) with no incremental DB
        # writes, so aborting anywhere here is inherently DB-consistent.
        print(f"[TrainRunner] Training stopped by user during initialization")

        # Reuse the existing long-lived training_db session (same as the
        # "failed" path below) rather than opening a new one.
        run = training_db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = "stopped"
            run.phase_detail = "Initialization stopped by user"
            training_db.commit()

        # Non-zero exit so the parent monitor's returncode != 0 / user-stopped
        # path fires (matches the existing convention for the failed path,
        # which exits 1; use 2 here so the two outcomes remain distinguishable
        # in process exit codes if ever inspected).
        sys.exit(2)

    except Exception as e:
        print(f"[TrainRunner] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        # Update run status to failed (in training.db)
        run = training_db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.error_message = str(e)
            training_db.commit()

        sys.exit(1)

    finally:
        training_db.close()
        datasets_db.close()

        # Close log file and restore original stdout/stderr
        if log_file:
            print(f"[TrainRunner] Closing training log file...")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
