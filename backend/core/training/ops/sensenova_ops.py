"""Training-only SenseNova decoder operations.

The trainer supplies one prompt prefix per physical B1 batch. It is immutable
either way, but not always detached: with ``train_text_encoder`` the prefix is
built by a differentiable understanding-branch pass so the gradient reaches the
understanding LoRA (Phase U). Both modes share one structural contract and
differ only in which grad-mode assertion applies -- see
``_assert_immutable_prefix_cache``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..training_events import emit_training_warning
from .training_method import _FULL_FINETUNE_METHOD, resolve_training_method


@dataclass(frozen=True)
class SenseNovaTrainingPrefix:
    """Detached prompt K/V reused by every flow step for one sample."""

    cache: Any
    # The NEXT t index, not a token count: image tokens spread over h/w, so with
    # references the prefix's t extent is shorter than its token count. Vendor
    # `_build_t2i_image_indexes` uses this as the image tokens' t coordinate.
    text_length: int


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


# The optimizer this route runs under. Two conditions, both required: a
# per-parameter fused-backward seam, and state that fits beside the materialized
# half. Each name below is excluded by the condition that actually applies to it
# (measured B/param from SENSENOVA_TRAINING_DESIGN.md 6.5's table, scaled over
# the gen half's 8,103,395,328 parameters and over both halves):
#
#   adamw8bit             2.031250 B/param -> 16.5 GB / 32.9 GB. Has the seam
#                         (FUSED_BACKWARD_OPTIMIZERS, patched in
#                         _setup_fused_backward_pass); excluded on state size.
#   adamw8bit_ringbuffer  2.031250 B/param -> 16.5 GB / 32.9 GB on the GPU.
#                         Excluded on state size. Its host-resident mode is now
#                         reachable (base_trainer._ringbuffer_optimizer_kwargs
#                         supplies get_state_buffer when
#                         optimizer_state_host_resident is set) and measured at
#                         0.031250 B/param on the GPU with 2.0 pinned on the
#                         host -- but that switch is config-channel only (a key
#                         in the run YAML, no API/UI surface), so a run started
#                         from the product gets the 16.5 GB figure. The
#                         exclusion stands on what a user can actually select.
#   lion8bit_ringbuffer   1.015625 B/param -> 8.2 GB / 16.5 GB: HALF the AdamW
#                         pair, one moment instead of two (0.015625 GPU /
#                         1.0 host in host-resident mode). The state-size
#                         argument does not exclude it, and G-RB2 / G-RB3 are
#                         now discharged (U-2-6). What is still missing is the
#                         step wall this route would pay for it: SenseNova sits
#                         BELOW G-RB1's transfer-hiding threshold (1024 image
#                         tokens at 1024^2 against Lion's 1019), and U-2-4 has
#                         not measured a real step. Admitted only when that is.
#   adafactor             0.002991 B/param (shape-dependent) -- the admitted one.
SENSENOVA_FULL_FINETUNE_OPTIMIZERS = ("adafactor",)


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
            "accum=4 with AdamW). Physical batch 1 with no accumulation is what "
            "this route trains. LoRA training on this architecture does support "
            "gradient_accumulation_steps."
        )

    if optimizer_type is None and settings.get("optimizer") is None:
        # Absence carries no information on the config channel; the call from
        # setup_optimizer always names one, and that is the authoritative check.
        return
    name = str(
        optimizer_type if optimizer_type is not None else settings["optimizer"]
    ).strip().lower()
    if name in SENSENOVA_FULL_FINETUNE_OPTIMIZERS:
        return
    extra = ""
    if name == "adamw":
        extra = (
            " torch.optim.AdamW updates every parameter inside one step() with no "
            "per-parameter seam, so stochastic rounding cannot be attached to it; "
            "measured under round-to-nearest, 84.5% of a bf16 tensor's elements "
            "never move at any step count, while the loss falls normally."
        )
    elif name in ("adamw8bit_ringbuffer", "lion8bit_ringbuffer"):
        state = (
            "2.031250 B/param, the same 16.5 GB over the generation half as "
            "adamw8bit" if name == "adamw8bit_ringbuffer"
            else "1.015625 B/param, 8.2 GB over the generation half -- half the "
            "AdamW pair, since Lion keeps one moment"
        )
        extra = (
            f" The ring-buffer optimizers are the intended second option. Their "
            f"host-resident state mode is wired up now but has no setting to turn "
            f"it on, so a run started from the product allocates 8-bit state on "
            f"the GPU at a measured {state}. What is not measured is the step "
            f"time this route would pay: its resolution band sits below the "
            f"threshold at which host state stops costing wall clock."
        )
    raise ValueError(
        f"SenseNova full fine-tuning does not support optimizer='{name}'. "
        f"Supported: {', '.join(SENSENOVA_FULL_FINETUNE_OPTIMIZERS)}. This route "
        f"applies each update from that parameter's own post-accumulate-grad hook, "
        f"so the optimizer needs a per-parameter seam AND state small enough to sit "
        f"beside the materialized bf16 half: adamw8bit has the seam but its measured "
        f"2.031250 B/param is 16.5 GB of state over the generation half's 8.10 G "
        f"parameters (32.9 GB over both halves), against Adafactor's factored second "
        f"moment.{extra} Set optimizer=adafactor, or use training_method='lora', "
        f"which accepts every optimizer this product offers."
    )


def enforce_full_finetune_stochastic_rounding(trainer: Any) -> bool:
    """Turn stochastic rounding on for this route, and say so.

    Not a default and not a refusal, because the transport cannot express the
    difference between the two: ``routes.py`` declares
    ``optimizer_stochastic_rounding`` as a plain ``bool`` and
    ``training_config`` writes the YAML key only when it is true, so an omitted
    key and an explicit false both reach the trainer as ``False``. Refusing on
    ``False`` would refuse every request; accepting it would run this route the
    way the contract already refuses ``optimizer=adamw`` for -- 84.5% of a bf16
    tensor's elements never moving at any step count, with the loss falling
    normally (SENSENOVA_TRAINING_DESIGN.md 6.3).

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
        "one quantized flavour, so only 'int8' can be trained on again -- "
        "'mixed' and 'bf16' are outputs, not bases. Resuming a run therefore "
        "requires that the run was created with "
        "sensenova_full_finetune_save_format='int8'; it cannot be changed after "
        "the fact, because the weights of the other two formats are already "
        "written in a shape this loader does not accept as a training base. "
        "'int8' is lossy on save (see its API description)."
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
    _assert_supported_quantized_training_base(
        trainer.transformer,
        training_method=training_method,
        source_metadata=components.get("metadata"),
    )
    _assert_pixel_head_fm_decoder(trainer.transformer)
    if branch is not None:
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
    trainer.vae = None
    trainer.unet = None
    trainer.scheduler = None
    trainer.noise_scheduler = None
    trainer.layer_offload_conductor = None
    # Materialized full-FT weights are frozen here too; unfreezing them is the
    # adapter's job, as it is for every other architecture.
    trainer.transformer.requires_grad_(False)
    trainer.transformer.train()
    trainer.transformer.to(trainer.device)
    # Training mode must be stamped even when the selected backend is native.
    setup_attention_backend(trainer, trainer.attention_backend)


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

    def __init__(self, layers: "list[_TrainingPrefixLayer]"):
        self.layers = layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.layers[layer_idx].keys.shape[-2])


def forward_und_prefix_layers(
    model: Any,
    input_ids: torch.Tensor,
    indexes: torch.Tensor,
    attention_mask: Any,
    *,
    checkpoint_layers: bool = True,
) -> _TrainingPrefixCache:
    """Run the understanding decoder stack and return a differentiable prefix.

    K/V leave each layer as explicit checkpoint OUTPUTS (the vendor ``return_kv``
    seam) rather than through ``past_key_values.update()``: that write is a
    checkpoint-segment side effect, so a non-reentrant recompute would append a
    second time, and a side-effected tensor is not an output autograd can route
    a gradient through. Bitwise parity with vendor ``_t2i_prefix_forward`` was
    verified on the real checkpoint, checkpointed and not (Phase U-0).
    """
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

    hidden_states = model.embed_tokens(input_ids)
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
            )

        if checkpoint_layers:
            hidden_states, keys, values = checkpoint(
                layer_forward, hidden_states, use_reentrant=False
            )
        else:
            hidden_states, keys, values = layer_forward(hidden_states)
        cache_layers.append(_TrainingPrefixLayer(keys, values))
    return _TrainingPrefixCache(cache_layers)


def _build_trainable_prefix(trainer: Any, transformer: Any, inputs) -> Any:
    """Run ``forward_und_prefix_layers`` under the autocast the adapters need.

    ``LoRALinearLayer`` keeps fp32 adapter weights and relies on an AMBIENT
    autocast to meet the bf16 base activation -- ``train_step`` provides one for
    the generation pass, and without the same wrap here the very first und-LoRA
    prefix pass raises a dtype mismatch at layer 0 (found by running it, U-0;
    re-running the K/V parity gate with autocast on costs nothing numerically).
    """
    input_ids, indexes, attention_mask = inputs
    dtype = getattr(trainer, "training_dtype", torch.float32)
    device_type = torch.device(getattr(trainer, "device", "cpu")).type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
        return forward_und_prefix_layers(
            transformer.language_model.model,
            input_ids,
            indexes,
            attention_mask,
            checkpoint_layers=bool(getattr(trainer, "gradient_checkpointing", True)),
        )


def encode_prompt(
    trainer: Any,
    prompt: str,
    *,
    requires_grad: bool = False,
    reference_image_paths: Optional[List[str]] = None,
) -> SenseNovaTrainingPrefix:
    """Build a prefix without inference streamers or flash buffers.

    ``requires_grad=False`` (the default, and the whole of Phase 1) builds it
    under ``no_grad`` through the vendor prefix forward; ``True`` builds it
    through the differentiable understanding loop above so understanding LoRA
    receives gradient.

    With ``reference_image_paths`` this runs inference's cond branch verbatim
    (understanding-tower ViT embeds spliced into the text prefix); img_cond and
    uncond are CFG-only and carry no loss.
    """
    if not isinstance(prompt, str):
        raise TypeError("SenseNova training encodes one prompt at a time")

    from core.models.sensenova.vendor.utils import SYSTEM_MESSAGE_FOR_GEN

    transformer = trainer.transformer
    ref_images = _load_reference_images(reference_image_paths)
    phase_evictor = getattr(trainer, "sensenova_phase_evictor", None)
    if requires_grad:
        assert_understanding_training_supported(transformer)
        if ref_images:
            raise NotImplementedError(
                "SenseNova understanding-branch training is text-only in this "
                "implementation; reference-conditioned items need the reference "
                "tower's differentiable path, which is not wired yet"
            )
        if phase_evictor is not None:
            raise RuntimeError(
                "SenseNova understanding-branch training cannot run with MoT phase "
                "eviction: the understanding half must stay resident until backward, "
                "but the evictor moves it to CPU for the denoise phase"
            )
        query = transformer._build_t2i_query(
            prompt,
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text="<think>\n\n</think>\n\n<img>",
        )
        with torch.no_grad():
            inputs = transformer._build_t2i_text_inputs(trainer.tokenizer, query)
        cache = _build_trainable_prefix(trainer, transformer, inputs)
        indexes = inputs[1]
        _assert_immutable_prefix_cache(
            cache,
            len(transformer.language_model.model.layers),
            trainable=True,
        )
        return SenseNovaTrainingPrefix(
            cache=cache, text_length=int(indexes[0].max()) + 1
        )

    if phase_evictor is not None:
        phase_evictor.enter_prefix()
    with torch.no_grad():
        if ref_images:
            from core.models.sensenova.sensenova_pipeline_ops import (
                _IMG_CONTEXT_TOKEN,
                _embed_reference_images,
                _splice_reference_image_tokens,
            )

            pixel_values, grid_hw = _embed_reference_images(transformer, ref_images)
            # Never set on the Phase 1 text-only path; _build_it2i_inputs asserts
            # on it matching at least one token.
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
            inputs = transformer._build_it2i_inputs(
                trainer.tokenizer, query, pixel_values, grid_hw
            )
            cache, _ = transformer._it2i_prefix_forward(*inputs)
        else:
            query = transformer._build_t2i_query(
                prompt,
                system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text="<think>\n\n</think>\n\n<img>",
            )
            inputs = transformer._build_t2i_text_inputs(trainer.tokenizer, query)
            cache, _ = transformer._t2i_prefix_forward(*inputs)
        indexes = inputs[1]
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
    """Return normalized RGB directly; SenseNova is a pixel-space model."""
    if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
        raise ValueError("SenseNova expects BCHW RGB training images")
    if image_tensor.shape[-2] % 32 or image_tensor.shape[-1] % 32:
        raise ValueError("SenseNova image height and width must be divisible by 32")
    return image_tensor.detach().to(dtype=trainer.training_dtype, device="cpu")


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
        "is_latent": False,
        "loss": loss_value,
        "recon_loss": recon_loss_value,
        "batch_size": 1,
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
        # which is the same convention the VAE archs' decoded previews use.
        tensor_to_image(tensor.detach().float()).save(
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
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError("SenseNova training currently requires batch_size=1 BCHW RGB")
    height, width = images.shape[-2:]
    if height % 32 or width % 32:
        raise ValueError("SenseNova image height and width must be divisible by 32")

    transformer = trainer.transformer
    _assert_pixel_head_fm_decoder(transformer)
    trainable_prefix = bool(getattr(trainer, "train_text_encoder", False))
    device = trainer.device
    dtype = trainer.training_dtype
    x0 = images.to(device=device, dtype=dtype)
    if timesteps is None:
        t = trainer.timestep_sampler.sample(1, device=device)
        if isinstance(t, tuple):
            t = t[0]
    else:
        t = timesteps
    # Keep t in fp32, the dtype the sampler produces and the dtype inference's
    # `ts` carries (linspace, sensenova_pipeline_ops.py:1068): timestep_embedder
    # embeds t's VALUE, and bf16 would quantize it to ~2e-3 in training only.
    t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
    if t.numel() != 1:
        raise ValueError("SenseNova training requires one timestep for batch_size=1")

    from core.models.sensenova.sensenova_pipeline_ops import (
        _build_step_context,
        compute_noise_scale,
    )

    merge_size = int(1 / transformer.downsample_ratio)
    grid_h, grid_w = height // transformer.patch_size, width // transformer.patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("SenseNova image does not align to the merged token grid")
    token_h, token_w = grid_h // merge_size, grid_w // merge_size
    noise_scale = compute_noise_scale(transformer, grid_h, grid_w, merge_size)
    # Inference noises in the image dtype, its fp32 t demoted by 0-dim promotion
    # (sensenova_pipeline_ops.py:1122); cast explicitly so z_image stays
    # training_dtype -- _build_step_context's ViT runs outside the autocast below.
    z_image = t.to(dtype).view(1, 1, 1, 1) * x0 + (1 - t).to(dtype).view(1, 1, 1, 1) * (
        torch.randn_like(x0) * noise_scale
    )
    shape = SimpleNamespace(
        batch_size=1,
        merge_size=merge_size,
        grid_h=grid_h,
        grid_w=grid_w,
        token_h=token_h,
        token_w=token_w,
    )
    z, image_embeds, _ = _build_step_context(
        transformer, shape, z_image, t[0], noise_scale
    )
    indexes = transformer._build_t2i_image_indexes(
        token_h, token_w, prefix.text_length, device=device
    )
    _assert_immutable_prefix_cache(
        prefix.cache,
        len(transformer.language_model.model.layers),
        trainable=trainable_prefix,
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
            trainable_prefix=trainable_prefix,
        )
        decoded = transformer.fm_modules["fm_head"](
            hidden.view(1, token_h, token_w, -1).permute(0, 3, 1, 2).contiguous()
        )
        patch = transformer.patch_size * merge_size
        x0_pred = (
            decoded.view(1, 3, token_h, patch, token_w, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(1, token_h * token_w, patch * patch * 3)
        )
        x0_tokens = transformer.patchify(x0, patch)
        # fp32 t here lifts v into fp32, which the MSE below wanted anyway --
        # the .float() calls become no-ops rather than extra copies.
        denominator = (1 - t).view(1, 1, 1).clamp_min(transformer.config.t_eps)
        v_pred = (x0_pred - z) / denominator
        v_target = (x0_tokens - z) / denominator
        loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float())
        recon_loss = torch.nn.functional.mse_loss(x0_pred.float(), x0_tokens.float())

    value = float(loss.detach())
    recon_value = float(recon_loss.detach())

    if debug_save_path is not None:
        try:
            _save_pixel_debug(
                transformer,
                debug_save_path,
                t_val=float(t[0].item()),
                noise_scale=noise_scale,
                images=x0,
                z_image=z_image,
                x0_pred_tokens=x0_pred,
                patch=patch,
                height=height,
                width=width,
                loss_value=value,
                recon_loss_value=recon_value,
                captions=debug_captions,
                reference_image_paths=debug_reference_image_paths,
            )
        except Exception as debug_error:
            print(f"{trainer.log_prefix} [debug_latents] save failed: {debug_error}")

    return loss, value, recon_value


def generate_sample(
    trainer: Any,
    *,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: str = "",
    reference_image_path: Optional[str] = None,
    condition_image_path: Optional[str] = None,
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
    from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS
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

    snapped_width, snapped_height = ops.normalize_resolution(width, height)
    if (snapped_width, snapped_height) != (width, height):
        print(
            f"{trainer.log_prefix} SenseNova sample resolution snapped to the "
            f"{ops.TOKEN_GRID_ALIGN}px token grid: {width}x{height} -> "
            f"{snapped_width}x{snapped_height}"
        )

    backend = trainer._resolve_training_backend(trainer.attention_backend)
    was_training = transformer.training
    evictor = getattr(trainer, "sensenova_phase_evictor", None)
    prefix = None
    try:
        ref_images = _load_reference_images(
            [reference_image_path] if reference_image_path else None
        )
        img_cfg_scale = SENSENOVA_GENERATION_DEFAULTS["img_cfg_scale"]
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
            image_tensor = ops.denoise_loop(
                transformer,
                prefix,
                cfg_scale=guidance_scale,
                timestep_shift=SENSENOVA_GENERATION_DEFAULTS["timestep_shift"],
                num_inference_steps=num_inference_steps,
                seed=seed if seed is not None and seed >= 0 else None,
                cfg_norm=SENSENOVA_GENERATION_DEFAULTS["cfg_norm"],
            )
        image = ops.tensor_to_image(image_tensor.float())
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


def _assert_immutable_prefix_cache(
    prefix_cache: Any, expected_layers: int, *, trainable: bool = False
) -> None:
    _assert_prefix_cache_structure(prefix_cache, expected_layers)
    if trainable:
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
) -> torch.Tensor:
    """Run the all-generation-token Qwen3 decoder against immutable prefix K/V.

    Calling PyTorch's base ``Module.__call__`` bypasses Transformers'
    cache-dropping checkpoint wrapper while preserving module hooks. The cache
    is read through the differentiable ``update_cache=False`` concat path.
    """

    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("SenseNova generation training model has no decoder layers")
    _assert_immutable_prefix_cache(prefix_cache, len(layers), trainable=trainable_prefix)
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
            )

        if checkpoint_layers:
            hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
        else:
            hidden_states = layer_forward(hidden_states)

    return model.norm_mot_gen(hidden_states)
