"""VAE-training config resolution + the HARD-REFUSE gate matrix.

The YAML config carries the VAE-specific block under ``process.vae`` (flat
key/value, same key set as ``api.param_defaults.VAE_TRAINING_DEFAULTS``); the
generic run-shape knobs (batch size, steps, LR, optimizer, save cadence,
resume) stay in the existing ``process.train`` / ``process.save`` sections so
the pre-existing routes, resume plumbing and checkpoint-keep field keep working
unchanged.

Every refusal below is deliberate and documented in design.md §4 (component
toggle matrix). Refusing at config-resolution time — i.e. BEFORE a single
training step or a model load — is the whole point: a VAE fine-tune that
silently trains nothing, or silently breaks the latent contract, is far worse
than one that will not start.
"""

from __future__ import annotations

import math
from typing import Any, Dict


class VaeConfigError(ValueError):
    """A VAE training configuration that must not be allowed to run."""


# Allowed values for the enumerated keys.
VALID_SOURCES = ("model", "path", "store")
VALID_DECODER_BLOCKS = ("all", "up_blocks", "mid_block", "conv_out")
VALID_ENCODER_BLOCKS = ("all", "down_blocks", "mid_block", "conv_out")
VALID_DTYPES = ("bf16", "fp32")
VALID_LPIPS_NETS = ("vgg", "alex", "squeeze")
# How much an image is resampled before the square crop is taken. Defined here,
# with the rest of the enums, and imported by vae_dataset (rather than the other
# way round) so that this module stays free of torch/PIL: it is the pure-config
# gate and is exercised by a fast, GPU-free test file.
VALID_CROP_SCALE_POLICIES = ("downscale", "native", "mixed")

# Every optimizer name OptimizerFactory resolves. Anything else raises inside
# the factory, but only AFTER the base VAE is loaded and the trainable set
# selected, which is why the set is checked at config-resolution time instead.
#
# The two ``*_ringbuffer`` entries are included deliberately. They are named for
# a ring-buffer allocator (``get_state_buffer``) that only BaseTrainer builds,
# and this trainer passes none — but that is not a failure: the allocator
# argument defaults to None and both implementations fall back to a GPU
# allocation for their 8-bit state (adamw8bit_ringbuffer.py "Ring Buffer
# disabled: GPU allocation (bitsandbytes-compatible)", lion8bit_ringbuffer.py
# likewise). Verified by construction + one real ``step()`` on CUDA: both build,
# allocate uint8 state on cuda:0 and move the parameters. Refusing them would
# therefore break a configuration that works today. What they do NOT do here is
# the thing their name promises, so build_optimizer says so out loud
# (RINGBUFFER_OPTIMIZERS below).
#
# The frontend panel offers the seven non-ringbuffer names; this set is a
# superset of that list, never a subset, so nothing the UI can produce is
# refused.
VALID_OPTIMIZERS = ("adamw", "adamw8bit", "adafactor", "lion8bit",
                    "paged_adamw", "paged_adamw8bit", "paged_lion8bit",
                    "adamw8bit_ringbuffer", "lion8bit_ringbuffer")

# Names whose ring-buffer behaviour is inactive without an allocator. Read by
# VaeTrainer.build_optimizer, which logs what the run actually gets.
RINGBUFFER_OPTIMIZERS = ("adamw8bit_ringbuffer", "lion8bit_ringbuffer")

# The only two `optimizer*` keys VaeTrainer.build_optimizer consumes. Every
# other one the diffusion generators can write (optimizer_cautious,
# optimizer_schedule_free[_r|_weight_lr_power], optimizer_use_radam,
# optimizer_warmup_steps, optimizer_stochastic_rounding,
# optimizer_beta1/beta2/epsilon) is refused in _validate rather than accepted
# and dropped. Add a key here only together with the code that reads it.
#
# The refusal is by PREFIX, not by list, so it also still covers keys nothing
# writes any more -- notably optimizer_is_paged, removed from the diffusion
# surface but present in every YAML written before that. A VAE run handed one
# is told to delete it, which is the right answer: it never did anything.
_VAE_SUPPORTED_OPTIMIZER_KEYS = frozenset({"optimizer", "optimizer_weight_decay"})

# ``diffusers.optimization.get_scheduler`` names this trainer can run.
# ``piecewise_constant`` is deliberately absent: it requires a ``step_rules``
# argument that ``VaeTrainer.build_optimizer`` never passes, so asking for it
# could only land in that method's except-branch and run at a CONSTANT LR while
# the YAML, the sidecar and /params all still say otherwise. Mirrored by the
# LR_SCHEDULERS list in the frontend panel.
VALID_LR_SCHEDULERS = ("constant", "constant_with_warmup", "linear", "cosine",
                       "cosine_with_restarts", "polynomial")

# Loss keys that participate in the "at least one active term" check.
_LOSS_WEIGHT_KEYS = ("mse_weight", "l1_weight", "lpips_weight",
                     "ycbcr_dc_weight", "pattern_weight", "l_invented_weight")

# Upper bounds for the L_invented keys, mirroring the `maximum:` that
# openapi.yaml declares for each of them (VaeTrainingDefaults). They are here so
# the declared contract is ENFORCED rather than merely documented; the sibling
# pattern_weight declares no maximum, so nothing else in this file gains one.
# If a bound is wrong it is changed in openapi.yaml and here together.
_INVENTED_WEIGHT_MAX = 10.0
_INVENTED_CHANNEL_MAX = 4.0
_INVENTED_THRESHOLD_MAX = 8.0

# Keys where a wrong answer changes WHAT IS TRAINED or WHAT IS WRITTEN, and so
# must never be decided by Python truthiness. See strict_bool().
_STRICT_BOOL_KEYS = ("train_decoder", "train_encoder",
                     "acknowledge_latent_space_break", "export_bare_ldm",
                     "ema_enabled")

_TRUE_STRINGS = frozenset({"true", "yes", "on", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "off", "0"})

# Integer keys, with the bound each one means and WHY a value outside it is
# refused rather than clamped. Every reason below names a failure that is
# otherwise SILENT: the run starts, reports success, and trains something other
# than what was asked for. ``resolution`` / ``validation_resolution`` are not
# here because they carry an additional multiple-of-8 rule and are checked
# together with it.
#   key: (minimum, maximum or None, reason)
_INT_KEY_BOUNDS: Dict[str, Any] = {
    "batch_size": (
        1, None,
        "Fewer than one image per micro-step is not a batch."),
    "total_steps": (
        1, None,
        "A run with no optimizer step trains nothing and still exports a file."),
    "gradient_accumulation_steps": (
        1, None,
        "At least one micro-step has to be accumulated per optimizer step."),
    "lr_warmup_steps": (
        0, None,
        "A negative warmup length is not a schedule; 0 means no warmup."),
    "seed": (
        0, 2 ** 32 - 1,
        "Outside this range the run does not use the seed it records: the "
        "trainer seeds python and torch with the literal value but numpy with "
        "seed % 2**32, so seed=-1 gives numpy 4294967295 and seed=2**32+7 gives "
        "it 7, while train_state.json and the export sidecar record the "
        "original. The three generators then disagree and the recorded seed "
        "does not reproduce the run. Note also that this trainer has no "
        "'-1 means random' convention (unlike the generation seeds in "
        "api/param_defaults.py), so a -1 here runs as a literal seed."),
    "num_workers": (
        0, None,
        "0 loads in the main process. A negative worker count is rejected by "
        "DataLoader itself, but only after the base VAE has been loaded."),
    "save_every": (
        0, None,
        "0 disables periodic checkpoints (the final one is still written). A "
        "negative value would disable them SILENTLY, because the trainer only "
        "tests save_every > 0."),
    "max_step_saves_to_keep": (
        0, None,
        "0 keeps every checkpoint. A negative value would also keep them all, "
        "silently, because pruning is skipped for keep <= 0 - so a run that "
        "asked for pruning would quietly fill the disk instead."),
    "validation_every": (
        0, None,
        "0 disables validation. A negative value disables it silently (the "
        "trainer only tests validation_every > 0), and the held-out PSNR / "
        "blockiness series is the only signal that a fine-tune is going wrong."),
    "validation_num_images": (
        1, None,
        "The held-out split is items[-validation_num_images:]: 0 makes the "
        "TRAINING split empty (items[:-0] is items[:0]) while validating on the "
        "whole dataset, and a negative value trains on the first few images "
        "only (items[:-(-1)] is items[:1]) with no error."),
    "pattern_size": (
        1, None,
        "The pattern term groups the residual by (row % pattern_size, "
        "col % pattern_size), which divides by zero at 0 and is meaningless "
        "below it."),
}


def _as_number(cfg: Dict[str, Any], key: str) -> float:
    """Coerce a config value to a finite float, or refuse it by name.

    Non-finite is refused as hard as non-numeric: a NaN weight makes the total
    loss NaN, and the only thing that notices is the trainer's non-finite-loss
    abort — one full model load and one training step later.
    """
    value = cfg[key]
    if isinstance(value, bool):
        # bool is an int subclass, so float(True) == 1.0 would let a stray
        # `true` become a weight of 1.0.
        raise VaeConfigError(
            f"{key} must be a number, got {value!r} (a boolean)."
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise VaeConfigError(f"{key} must be a number, got {value!r}")
    if math.isnan(number) or math.isinf(number):
        raise VaeConfigError(
            f"{key} must be a finite number, got {number}. A non-finite value "
            f"reaches the loss, and nothing notices until the trainer aborts on "
            f"a non-finite loss - after the model load."
        )
    cfg[key] = number
    return number


def _as_int(cfg: Dict[str, Any], key: str) -> int:
    """Coerce a config value to an int, or refuse it by name.

    A fractional value is refused rather than truncated: ``int(2.7)`` is 2, so a
    mistyped count would change the shape of the run without saying so.
    """
    value = cfg[key]
    if isinstance(value, bool):
        raise VaeConfigError(
            f"{key} must be an integer, got {value!r} (a boolean)."
        )
    if isinstance(value, float):
        if not value.is_integer():
            raise VaeConfigError(
                f"{key} must be a whole number, got {value}. Truncating it "
                f"would change the run's shape without reporting it."
            )
        value = int(value)
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise VaeConfigError(f"{key} must be an integer, got {cfg[key]!r}")
    cfg[key] = number
    return number


def _as_text(cfg: Dict[str, Any], key: str) -> str:
    """Coerce a config value to a string, or refuse it by name.

    ``str()`` accepts anything, so a list or a dict here would become the
    literal text ``"['a', 'b']"`` and be reported as a missing file much later.
    """
    value = cfg[key]
    if value is None:
        cfg[key] = ""
        return ""
    if not isinstance(value, str):
        raise VaeConfigError(
            f"{key} must be a string, got {value!r} ({type(value).__name__})."
        )
    cfg[key] = value
    return value


def strict_bool(value: Any, key: str) -> bool:
    """Parse a boolean the way a config file means it, not the way Python does.

    ``bool("false")`` is ``True``. A YAML that quotes its booleans —
    ``train_encoder: "false"`` — is entirely ordinary (editors, templating and
    hand-quoting all produce it), and under a bare ``bool()`` cast it would
    silently ENABLE encoder training, i.e. open the double gate by accident.
    That is the exact failure this whole gate exists to prevent, so every gate
    key is parsed here instead: real booleans and 0/1 pass through, the explicit
    string spellings are accepted, and ANYTHING else raises rather than being
    guessed at.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    raise VaeConfigError(
        f"{key} must be a boolean, got {value!r} ({type(value).__name__}). "
        f"Accepted: true/false, yes/no, on/off, 1/0. This key is parsed strictly "
        f"because Python would read the string \"false\" as True, which would "
        f"silently change what the run trains or writes."
    )


def _vae_training_defaults() -> Dict[str, Any]:
    """Fetch the SSOT defaults.

    Imported lazily and function-locally on purpose: ``backend/api/__init__.py``
    imports ``api.routes`` (the whole API surface), so a module-level
    ``from api.param_defaults import ...`` inside ``core/training/`` would create
    an import cycle when routes.py imports a training module. This mirrors the
    existing ``from api.param_defaults import resolve_bundle_vae`` call sites in
    ``core/training/adapters/*.py``.
    """
    from api.param_defaults import VAE_TRAINING_DEFAULTS
    return dict(VAE_TRAINING_DEFAULTS)


def resolve_vae_training_config(
    process_config: Dict[str, Any],
    *,
    base_model_path: str = "",
) -> Dict[str, Any]:
    """Merge ``process.vae`` / ``process.train`` / ``process.save`` over the SSOT
    defaults, then run the refusal gate.

    Returns a single flat dict with exactly the ``VAE_TRAINING_DEFAULTS`` key
    set (plus ``resume_from``), which is what ``VaeTrainer`` consumes.

    Raises:
        VaeConfigError: on any refused combination (see design.md §4).
    """
    cfg = _vae_training_defaults()

    train_section = process_config.get("train") or {}
    save_section = process_config.get("save") or {}
    # `or {}` would be wrong here: an empty-but-malformed section (``vae: []``,
    # ``vae: ""``) is falsy, so it would silently become "no options given" and
    # the run would train with defaults the user never wrote. Only an ABSENT or
    # explicitly-null section means "use the defaults"; anything else must be a
    # mapping. (Found by mutation-testing the type guard, 2026-07-29.)
    vae_section = process_config.get("vae")
    if vae_section is None:
        vae_section = {}

    if not isinstance(vae_section, dict):
        raise VaeConfigError(
            "process.vae must be a mapping of VAE training options "
            f"(got {type(vae_section).__name__})"
        )

    # 1) generic run-shape keys, read from the shared sections so the existing
    #    UI/routes/resume plumbing continues to own them.
    _copy(cfg, train_section, "batch_size", "batch_size")
    _copy(cfg, train_section, "steps", "total_steps")
    _copy(cfg, train_section, "gradient_accumulation_steps", "gradient_accumulation_steps")
    _copy(cfg, train_section, "lr", "learning_rate")
    _copy(cfg, train_section, "optimizer", "optimizer")
    _copy(cfg, train_section, "optimizer_weight_decay", "optimizer_weight_decay")
    _copy(cfg, train_section, "max_grad_norm", "max_grad_norm")
    _copy(cfg, train_section, "lr_scheduler", "lr_scheduler")
    _copy(cfg, train_section, "lr_warmup_steps", "lr_warmup_steps")
    # seed / num_workers are written into process.vae by generate_vae_config (a
    # train-section placement did not survive the GET /params readback, since
    # neither is a TrainingRunCreateRequest field). These two reads are kept only
    # so a HAND-WRITTEN yaml that puts them under `train:` still works; the
    # process.vae entry below always wins.
    _copy(cfg, train_section, "seed", "seed")
    _copy(cfg, train_section, "num_workers", "num_workers")
    _copy(cfg, save_section, "save_every", "save_every")
    _copy(cfg, save_section, "max_step_saves_to_keep", "max_step_saves_to_keep")

    # 2) VAE-specific keys override anything above (an explicit process.vae entry
    #    always wins, so a hand-written YAML stays self-describing).
    unknown = sorted(set(vae_section) - set(cfg))
    if unknown:
        raise VaeConfigError(
            f"Unknown key(s) in process.vae: {unknown}. "
            f"Valid keys: {sorted(cfg)}"
        )
    for key, value in vae_section.items():
        if value is not None:
            cfg[key] = value

    # 3) resume: honour the existing train.resume_from_checkpoint field first
    #    (that is what the checkpoint-list UI writes), with process.vae.resume_from
    #    as an explicit override.
    cfg["resume_from"] = (
        vae_section.get("resume_from")
        or train_section.get("resume_from_checkpoint")
        or None
    )

    if cfg.get("vae_source") == "model" and not cfg.get("vae_path"):
        cfg["vae_path"] = base_model_path

    _validate(cfg, train_section)
    if cfg["vae_source"] == "model":
        _refuse_model_source_without_vae(base_model_path)
    return cfg


def _refuse_model_source_without_vae(base_model_path: str) -> None:
    """Reject model-owned VAE selection for architectures with no VAE."""
    if not base_model_path:
        return
    try:
        from core.model_loader import ModelLoader
        arch = ModelLoader.detect_model_type(base_model_path)
    except Exception:
        return
    if arch == "sensenova":
        raise VaeConfigError(
            "vae_source='model' is not available for SenseNova because it is a "
            "pixel-space model with no VAE component. Use vae_source='path' or "
            "vae_source='store' to fine-tune an explicit VAE."
        )


def _copy(cfg: Dict[str, Any], section: Dict[str, Any], src_key: str, dst_key: str):
    value = section.get(src_key)
    if value is not None:
        cfg[dst_key] = value


def _validate(cfg: Dict[str, Any], train_section: Dict[str, Any]) -> None:
    # ---- component toggle matrix (design.md §4) ---------------------------
    # Strict parsing, NOT bool(): see strict_bool() for why a quoted "false"
    # must not be able to open the gate.
    for key in _STRICT_BOOL_KEYS:
        cfg[key] = strict_bool(cfg[key], key)

    train_decoder = cfg["train_decoder"]
    train_encoder = cfg["train_encoder"]
    acknowledged = cfg["acknowledge_latent_space_break"]

    # The DOUBLE GATE. Training the encoder moves the latent distribution, so
    # every latent cache, every LoRA and every diffusion model trained against
    # this VAE stops matching it. Neither key alone is enough, in EITHER
    # direction: a bare train_encoder must not run, and a stale acknowledgement
    # left in a config must not silently authorise a later run that did not ask
    # for encoder training.
    if train_encoder and not acknowledged:
        raise VaeConfigError(
            "train_encoder=true requires acknowledge_latent_space_break=true in "
            "the same config. Training the encoder changes the latent "
            "distribution: existing latent caches, LoRAs and diffusion "
            "checkpoints built against this VAE will no longer match it, and the "
            "result is a new VAE rather than a drop-in replacement. Set both keys "
            "to run it, or train_encoder=false for a decoder-only fine-tune."
        )
    if acknowledged and not train_encoder:
        raise VaeConfigError(
            "acknowledge_latent_space_break=true but train_encoder=false. The "
            "acknowledgement only applies to encoder training; leaving it set "
            "while the encoder is frozen would let it silently authorise a later "
            "run. Set train_encoder=true as well, or remove the acknowledgement."
        )

    if train_encoder and not train_decoder:
        # design.md §4: encoder-only training under a frozen decoder. The
        # reconstruction objective can still be differentiated, but every
        # gradient path to it goes through a decoder that is not allowed to
        # adapt, so the only way to reduce the loss is to deform the latent
        # distribution into whatever the fixed decoder already inverts well.
        raise VaeConfigError(
            "train_encoder=true with train_decoder=false is not supported: the "
            "only way to reduce a reconstruction loss through a frozen decoder "
            "is to deform the latent distribution to suit it. Train both "
            "(train_decoder=true), or train the decoder only."
        )
    if not train_decoder:
        raise VaeConfigError(
            "Nothing to train: train_decoder=false and train_encoder=false. "
            "Set train_decoder=true (decoder-only fine-tune)."
        )

    if cfg["decoder_blocks"] not in VALID_DECODER_BLOCKS:
        raise VaeConfigError(
            f"decoder_blocks must be one of {list(VALID_DECODER_BLOCKS)}, "
            f"got {cfg['decoder_blocks']!r}"
        )
    if cfg["encoder_blocks"] not in VALID_ENCODER_BLOCKS:
        raise VaeConfigError(
            f"encoder_blocks must be one of {list(VALID_ENCODER_BLOCKS)}, "
            f"got {cfg['encoder_blocks']!r}"
        )

    # A bare LDM .safetensors carries no config.json, so whatever loads it
    # inherits scaling_factor / shift_factor from the model it is plugged into.
    # For a decoder-only fine-tune those are still correct (the encoder that
    # defined them is untouched); after an encoder fine-tune they are not, and
    # nothing downstream can detect that. Refused here, before the run, rather
    # than after the training finishes.
    if cfg["export_bare_ldm"] and train_encoder:
        raise VaeConfigError(
            "export_bare_ldm=true is refused when train_encoder=true. A bare LDM "
            ".safetensors has no config.json, so the consumer inherits "
            "scaling_factor / shift_factor from the model it is loaded into, "
            "and an encoder fine-tune is precisely what makes those wrong, with "
            "no way for the consumer to notice. The diffusers directory export "
            "(which carries its own config.json and provenance sidecar) is "
            "always written."
        )

    if cfg["vae_source"] not in VALID_SOURCES:
        raise VaeConfigError(
            f"vae_source must be one of {list(VALID_SOURCES)}, "
            f"got {cfg['vae_source']!r}"
        )
    # Both are fed to str() by the loader, which accepts anything: a list here
    # would become the literal text "['a']" and surface as "no loadable VAE"
    # much later, blaming the path rather than the type.
    _as_text(cfg, "vae_path")
    _as_text(cfg, "vae_arch")
    if cfg["vae_source"] == "store":
        if not cfg.get("vae_arch"):
            raise VaeConfigError(
                "vae_source='store' requires vae_arch (a vae_store key: "
                "sdxl / sd15 / flux1 / flux2 / qwen / ...)."
            )
    elif not cfg.get("vae_path"):
        raise VaeConfigError(
            "No base VAE to train: vae_path is empty "
            f"(vae_source={cfg['vae_source']!r}). vae_source='model' takes the "
            "run's own base_model_path; vae_source='path' needs an explicit "
            "diffusers VAE directory or bare .safetensors in vae_path."
        )

    # ---- dtype ------------------------------------------------------------
    # fp16 is refused OUTRIGHT in Phase 1, for two independent reasons:
    #   - SD1.5/SDXL-family VAEs overflow fp16 in their decoder activations
    #     (the documented reason madebyollin/sdxl-vae-fp16-fix exists), and a
    #     training forward hits it sooner than inference does;
    #   - there is no GradScaler anywhere in this trainer, so on any other VAE
    #     fp16 would silently underflow the gradients instead.
    # bf16 (default) needs neither a scaler nor a fix, and fp32 is exact.
    if cfg["dtype"] == "fp16":
        raise VaeConfigError(
            "dtype='fp16' is not supported for VAE training. SD1.5/SDXL-family "
            "VAEs overflow fp16 in their decoder activations, and for every "
            "other VAE this trainer has no gradient scaler, so fp16 gradients "
            "would silently underflow. Use dtype='bf16' (default) or 'fp32'."
        )
    if cfg["dtype"] not in VALID_DTYPES:
        raise VaeConfigError(
            f"dtype must be one of {list(VALID_DTYPES)}, got {cfg['dtype']!r}"
        )

    # ---- raw pixels only --------------------------------------------------
    # VAE training is DEFINED by a live VAE forward on raw pixels; a
    # pre-encoded latent cache would make the trainable decoder see cached
    # latents produced by a different (or no longer matching) encoder and would
    # skip the encoder forward entirely. Mirrors the existing outpaint-ControlNet
    # refusal of pre_encoded_cache.
    latent_mode = train_section.get("latent_encoding_mode")
    if latent_mode == "pre_encoded_cache":
        raise VaeConfigError(
            "latent_encoding_mode='pre_encoded_cache' is incompatible with VAE "
            "training: the objective is a live encode->decode forward on raw "
            "pixels, so there is no cached latent to consume. Remove the key or "
            "set it to 'swap_onthefly' (it is ignored by the VAE trainer)."
        )

    # ---- optimizer options this trainer does not consume -------------------
    # VaeTrainer.build_optimizer passes exactly optimizer type / params / lr /
    # weight_decay to OptimizerFactory, so every other `optimizer_*` key in the
    # train section would be read by nobody. generate_vae_config never writes
    # one, but the diffusion generators do, so a hand-written or hand-merged
    # YAML can carry them; accepting them would mean a run that reports
    # cautious masking / Schedule-Free / stochastic rounding in its own config
    # while doing none of it. Named refusal instead of silence.
    unsupported = sorted(k for k in train_section
                         if k.startswith("optimizer")
                         and k not in _VAE_SUPPORTED_OPTIMIZER_KEYS)
    if unsupported:
        message = (
            f"Unsupported optimizer option(s) in process.train for a VAE run: "
            f"{unsupported}. This trainer builds its optimizer with "
            f"{sorted(_VAE_SUPPORTED_OPTIMIZER_KEYS)} only, so the listed keys "
            f"would have no effect on the run. Remove them."
        )
        if "optimizer_warmup_steps" in unsupported:
            # The one key with a working equivalent here: build_optimizer passes
            # lr_warmup_steps to get_scheduler, so warmup IS available under
            # that name. Say so rather than only "remove it".
            message += (" For LR warmup use lr_warmup_steps, which this trainer "
                        "passes to the LR scheduler.")
        raise VaeConfigError(message)

    # ---- losses -----------------------------------------------------------
    for key in _LOSS_WEIGHT_KEYS:
        if _as_number(cfg, key) < 0:
            raise VaeConfigError(f"{key} must be >= 0, got {cfg[key]}")

    if not any(cfg[key] > 0 for key in _LOSS_WEIGHT_KEYS):
        # kl_weight is deliberately NOT in this set. It is a regulariser on the
        # posterior, not a reconstruction signal: a run with every
        # reconstruction weight at 0 and only KL active would minimise the loss
        # by collapsing the posterior, whether or not the encoder is trainable.
        raise VaeConfigError(
            "All loss weights are 0: there is no training signal. Set at least "
            f"one of {list(_LOSS_WEIGHT_KEYS)} above 0 (default: mse_weight=1.0)."
        )

    if _as_number(cfg, "kl_weight") < 0:
        raise VaeConfigError(f"kl_weight must be >= 0, got {cfg['kl_weight']}")

    if cfg["lpips_net"] not in VALID_LPIPS_NETS:
        raise VaeConfigError(
            f"lpips_net must be one of {list(VALID_LPIPS_NETS)}, "
            f"got {cfg['lpips_net']!r}"
        )
    if cfg["lpips_weight"] > 0:
        # Fail BEFORE training starts, never mid-run.
        try:
            import lpips  # noqa: F401
        except Exception as e:
            raise VaeConfigError(
                f"lpips_weight={cfg['lpips_weight']} but the 'lpips' package is "
                f"not importable ({type(e).__name__}: {e}). Install it "
                "(pip install lpips) or set lpips_weight=0."
            )

    # ---- YCbCr colour-drift sub-parameters --------------------------------
    # Type and range are checked UNCONDITIONALLY (same reasoning as the
    # L_invented block below: a typo must not sit in a config until the day the
    # term is switched on); the "both channels are 0" consistency refusal fires
    # only when the term is actually active.
    for key in ("ycbcr_dc_y_weight", "ycbcr_dc_chroma_weight"):
        if _as_number(cfg, key) < 0:
            # A negative channel weight does not merely disable the channel: the
            # term is summed over channels, so the run is REWARDED for enlarging
            # that channel's colour error. Nothing downstream reports it -- the
            # total loss simply goes down.
            raise VaeConfigError(
                f"{key} must be >= 0, got {cfg[key]}. The YCbCr term is summed "
                f"over its channels, so a negative channel weight pays the run "
                f"for increasing that channel's colour error instead of "
                f"reducing it, and the total loss still falls. Use 0 to switch "
                f"the channel off."
            )
    if cfg["ycbcr_dc_weight"] > 0 and cfg["ycbcr_dc_y_weight"] <= 0 \
            and cfg["ycbcr_dc_chroma_weight"] <= 0:
        # Both at 0 makes the term identically zero while still paying for a
        # full YCbCr conversion and two Charbonnier passes every step -- and it
        # slips past the "at least one active loss" check, which only sees the
        # top-level ycbcr_dc_weight.
        raise VaeConfigError(
            "ycbcr_dc_y_weight and ycbcr_dc_chroma_weight are both 0, which "
            "makes the YCbCr colour term identically zero while still computing "
            "it every step -- and the 'at least one active loss' check only "
            "looks at ycbcr_dc_weight, so this configuration would otherwise "
            "run with no training signal at all. Set at least one above 0, or "
            "set ycbcr_dc_weight=0 to disable the term."
        )
    if _as_number(cfg, "ycbcr_dc_eps") <= 0:
        # Charbonnier is sqrt(d^2 + eps^2) - eps. At eps=0 that is |d|, whose
        # derivative at d=0 is 0/0 -> NaN, and identical pixels are common in
        # the flat regions this term exists for. A negative eps squares to the
        # same smoothing but then ADDS |eps| instead of subtracting it, so the
        # reported loss carries a constant offset that no chart explains.
        raise VaeConfigError(
            f"ycbcr_dc_eps must be > 0, got {cfg['ycbcr_dc_eps']}. It is the "
            f"Charbonnier smoothing constant in sqrt(d^2 + eps^2) - eps: at 0 "
            f"the term degenerates to |d|, whose gradient at an exactly-zero "
            f"residual is NaN, and a negative value adds a constant offset to "
            f"the reported loss instead of subtracting one."
        )

    # ---- shapes / cadence -------------------------------------------------
    for key in ("resolution", "validation_resolution"):
        value = _as_int(cfg, key)
        if value < 64 or value % 8 != 0:
            raise VaeConfigError(
                f"{key} must be a multiple of 8 and >= 64, got {value}"
            )
        cfg[key] = value

    # ---- crop scale policy ------------------------------------------------
    # Which pixels the decoder is trained on. Getting this wrong changes what
    # every step learns from, silently, so an out-of-enum value must not fall
    # back to a default.
    if cfg["crop_scale_policy"] not in VALID_CROP_SCALE_POLICIES:
        raise VaeConfigError(
            f"crop_scale_policy must be one of "
            f"{list(VALID_CROP_SCALE_POLICIES)}, got {cfg['crop_scale_policy']!r}. "
            f"'downscale' scales the short side to exactly resolution (the "
            f"historical behaviour), 'native' crops out of the full-size pixels, "
            f"'mixed' draws the downscale factor per sample."
        )

    max_downscale = _as_number(cfg, "crop_scale_max_downscale")
    if max_downscale < 0:
        raise VaeConfigError(
            f"crop_scale_max_downscale must be >= 0 (0 = unbounded), "
            f"got {max_downscale}"
        )
    if 0 < max_downscale < 1.0:
        # A "max downscale" below 1 would name an UPSCALE bound, which the knob
        # does not mean; silently clamping it to 1 would train on a distribution
        # the user did not ask for.
        raise VaeConfigError(
            f"crop_scale_max_downscale is a downscale factor, so it must be 0 "
            f"(unbounded) or >= 1.0; got {max_downscale}. 1.0 means 'never "
            f"downscale', which is what crop_scale_policy='native' already says."
        )
    if max_downscale > 0 and cfg["crop_scale_policy"] != "mixed":
        # Refused rather than ignored: the bound is only consulted by the
        # per-sample draw, so under 'downscale'/'native' it would be a knob the
        # user set, the YAML recorded, and nothing read.
        raise VaeConfigError(
            f"crop_scale_max_downscale={max_downscale} is only read when "
            f"crop_scale_policy='mixed' (it bounds the per-sample downscale "
            f"draw), but the policy is {cfg['crop_scale_policy']!r}. Set the "
            f"policy to 'mixed', or leave crop_scale_max_downscale at 0."
        )
    cfg["crop_scale_max_downscale"] = max_downscale

    # ---- integer counts / cadences ----------------------------------------
    # One table, one bound per key, each with the silent failure it prevents:
    # see _INT_KEY_BOUNDS. Every one of these was previously cast with a bare
    # int() and never range-checked, so e.g. validation_num_images=-1 resolved
    # to "train on the first image only" without a word.
    for key, (low, high, reason) in _INT_KEY_BOUNDS.items():
        value = _as_int(cfg, key)
        if value < low:
            raise VaeConfigError(f"{key} must be >= {low}, got {value}. {reason}")
        if high is not None and value > high:
            raise VaeConfigError(f"{key} must be <= {high}, got {value}. {reason}")

    if cfg["pattern_weight"] > 0 and cfg["pattern_size"] > cfg["resolution"]:
        # The pattern term crops to whole cells: with a cell larger than the
        # image it has none, returns a hard zero, and the weight the user set is
        # paid for and never applied.
        raise VaeConfigError(
            f"pattern_size={cfg['pattern_size']} is larger than "
            f"resolution={cfg['resolution']}, so the pattern term has no whole "
            f"cell to group by and returns exactly 0 on every step while "
            f"pattern_weight={cfg['pattern_weight']} says it is active. Reduce "
            f"pattern_size, raise resolution, or set pattern_weight=0."
        )

    if cfg["lr_warmup_steps"] >= cfg["total_steps"]:
        # The whole run would be warmup, so the configured learning rate is
        # never actually reached -- a run that trains at a fraction of the LR
        # its YAML, its sidecar and its chart label all report.
        raise VaeConfigError(
            f"lr_warmup_steps={cfg['lr_warmup_steps']} is not below "
            f"total_steps={cfg['total_steps']}: the entire run would be warmup, "
            f"so learning_rate={cfg['learning_rate']} is never reached even "
            f"though the YAML, the sidecar and the LR chart all report it. Use "
            f"a warmup shorter than the run."
        )

    ema_decay = _as_number(cfg, "ema_decay")
    if not (0.0 < ema_decay < 1.0):
        raise VaeConfigError(f"ema_decay must be in (0, 1), got {ema_decay}")
    # The boolean toggles were already parsed strictly (and written back as real
    # bools) at the top of this function.

    # ---- optimisation ------------------------------------------------------
    if _as_number(cfg, "learning_rate") <= 0:
        # 0 is refused rather than treated as "freeze": every optimizer step
        # becomes a no-op, so the run finishes, reports success and exports a
        # copy of the base VAE. A negative LR ascends the loss.
        raise VaeConfigError(
            f"learning_rate must be > 0, got {cfg['learning_rate']}. At 0 every "
            f"optimizer step is a no-op, so the run would finish, report "
            f"success and export the base VAE unchanged; a negative value "
            f"ascends the loss instead of descending it. To train nothing, do "
            f"not start a run."
        )
    try:
        _clip_probe = float(cfg["max_grad_norm"])
    except (TypeError, ValueError):
        _clip_probe = None
    if _clip_probe is not None and math.isinf(_clip_probe) and _clip_probe > 0:
        # An infinite bound WAS a working spelling of "do not clip"
        # (clip_grad_norm_ clamps the scale factor at 1.0, so inf/total_norm
        # never scales anything). It is refused now only because the generic
        # non-finite guard would otherwise report it as a NaN/inf typo; the
        # message routes it to the one spelling this trainer keeps.
        raise VaeConfigError(
            "max_grad_norm=inf: clipping is disabled with max_grad_norm=0, "
            "which is how the key is spelled in the rest of this repository "
            "(base_trainer, fused_optimizer_groups). An infinite bound has the "
            "same effect but is not accepted, so that 'no clipping' has exactly "
            "one spelling in a config, a chart legend and a sidecar."
        )
    if _as_number(cfg, "max_grad_norm") < 0:
        # clip_grad_norm_ scales by max_norm/total_norm and clamps that ratio
        # only from ABOVE, so a negative bound negates every gradient.
        raise VaeConfigError(
            f"max_grad_norm must be >= 0 (0 disables clipping), got "
            f"{cfg['max_grad_norm']}. torch.nn.utils.clip_grad_norm_ scales the "
            f"gradients by max_grad_norm/total_norm and clamps that factor only "
            f"from above, so a negative bound flips the sign of every gradient "
            f"and the run ascends the loss."
        )
    if _as_number(cfg, "optimizer_weight_decay") < 0:
        raise VaeConfigError(
            f"optimizer_weight_decay must be >= 0, got "
            f"{cfg['optimizer_weight_decay']}. A negative decay multiplies "
            f"every weight by more than 1 on every step, which grows the VAE's "
            f"weights without limit and is not reported anywhere until the loss "
            f"stops being finite."
        )

    optimizer = str(cfg["optimizer"]).strip().lower()
    if optimizer not in VALID_OPTIMIZERS:
        # OptimizerFactory raises for an unknown name, but only after the base
        # VAE has been loaded and the trainable set selected -- i.e. after the
        # slowest part of a run's startup.
        raise VaeConfigError(
            f"optimizer must be one of {list(VALID_OPTIMIZERS)}, got "
            f"{cfg['optimizer']!r}. OptimizerFactory resolves exactly this set; "
            f"anything else raises there, after the base VAE has already been "
            f"loaded."
        )
    cfg["optimizer"] = optimizer

    scheduler = str(cfg["lr_scheduler"]).strip().lower()
    if scheduler not in VALID_LR_SCHEDULERS:
        # build_optimizer catches a get_scheduler failure and CONTINUES at a
        # constant LR, so an unknown name here is not an error at run time --
        # it is a run that silently ignores the schedule it recorded.
        raise VaeConfigError(
            f"lr_scheduler must be one of {list(VALID_LR_SCHEDULERS)}, got "
            f"{cfg['lr_scheduler']!r}. An unrecognised name is not an error at "
            f"run time: the trainer falls back to a constant learning rate and "
            f"keeps going, so the run would silently ignore the schedule its "
            f"config records."
        )
    cfg["lr_scheduler"] = scheduler

    if scheduler == "constant" and cfg["lr_warmup_steps"] > 0:
        # diffusers' get_scheduler does not merely ignore the argument here, it
        # never receives it: the CONSTANT branch is
        # `return schedule_func(optimizer, last_epoch=last_epoch)`, taken before
        # the "all other schedulers require num_warmup_steps" line. So the run
        # trains at the full LR from step 0 while the YAML, the provenance
        # sidecar and the LR chart all record a warmup. Both keys are UI-
        # reachable and `constant` is the default, which makes this the most
        # likely spelling of the mistake, not the least.
        raise VaeConfigError(
            f"lr_scheduler='constant' ignores lr_warmup_steps "
            f"({cfg['lr_warmup_steps']}): diffusers' get_scheduler returns the "
            f"constant schedule without ever passing num_warmup_steps to it, so "
            f"the run would train at the full learning rate from step 0 while "
            f"the YAML, the provenance sidecar and the LR chart all record a "
            f"warmup. Use lr_scheduler='constant_with_warmup' to actually warm "
            f"up, or lr_warmup_steps=0."
        )

    # ---- L_invented sub-parameters ----------------------------------------
    # Type and range are checked UNCONDITIONALLY, so a typo'd value cannot sit
    # in a config until the day someone turns the term on. The two
    # *consistency* refusals below (which reject a combination that is
    # individually legal) fire ONLY when the term is actually on: with
    # l_invented_weight=0 nothing here is read, so refusing would break "off by
    # default is completely inert" — and the remediation those messages name
    # ("set l_invented_weight=0") would already be true and could not clear it.
    invented_on = float(cfg["l_invented_weight"]) > 0
    # Upper bounds are the ones openapi.yaml declares for these properties. The
    # spec is the contract, so it is enforced here rather than documented and
    # ignored (an unenforced maximum: 10 accepted l_invented_weight=1e6).
    for key, upper in (("l_invented_weight", _INVENTED_WEIGHT_MAX),
                       ("l_invented_y_weight", _INVENTED_CHANNEL_MAX),
                       ("l_invented_chroma_weight", _INVENTED_CHANNEL_MAX)):
        _as_number(cfg, key)
        if cfg[key] < 0:
            raise VaeConfigError(f"{key} must be >= 0, got {cfg[key]}")
        if cfg[key] > upper:
            raise VaeConfigError(
                f"{key} must be <= {upper} (the bound openapi.yaml declares for "
                f"it), got {cfg[key]}."
            )
    if invented_on and (cfg["l_invented_y_weight"] <= 0
                        and cfg["l_invented_chroma_weight"] <= 0):
        # Both channel weights at 0 makes the term identically 0 while still
        # costing a full mask + projection pass every step: a weight the user
        # set, the YAML recorded, and nothing acted on.
        raise VaeConfigError(
            "l_invented_y_weight and l_invented_chroma_weight are both 0, which "
            "makes the invented-HF term identically zero while still computing "
            "it every step. Set at least one above 0, or set "
            "l_invented_weight=0 to disable the term."
        )
    for key in ("l_invented_flat_t_y", "l_invented_flat_t_c"):
        _as_number(cfg, key)
        if cfg[key] > _INVENTED_THRESHOLD_MAX:
            raise VaeConfigError(
                f"{key} must be <= {_INVENTED_THRESHOLD_MAX} (the bound "
                f"openapi.yaml declares for it), got {cfg[key]}."
            )
        if cfg[key] < 0:
            raise VaeConfigError(f"{key} must be >= 0, got {cfg[key]}")
        if invented_on and cfg[key] <= 0:
            # 0 is refused rather than clamped: a zero plane-residual threshold
            # selects no window at all on real data, so the term would be
            # silently inert for the whole run.
            raise VaeConfigError(
                f"{key} must be > 0 (it is a plane-fit residual threshold in "
                f"8-bit levels), got {cfg[key]}. A threshold of 0 selects no "
                f"window on real data, so the invented-HF term would train "
                f"nothing."
            )
