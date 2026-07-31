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
    return cfg


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

    # ---- losses -----------------------------------------------------------
    for key in _LOSS_WEIGHT_KEYS:
        try:
            cfg[key] = float(cfg[key])
        except (TypeError, ValueError):
            raise VaeConfigError(f"{key} must be a number, got {cfg[key]!r}")
        if cfg[key] < 0:
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

    try:
        cfg["kl_weight"] = float(cfg["kl_weight"])
    except (TypeError, ValueError):
        raise VaeConfigError(f"kl_weight must be a number, got {cfg['kl_weight']!r}")
    if cfg["kl_weight"] < 0:
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

    # ---- shapes / cadence -------------------------------------------------
    for key in ("resolution", "validation_resolution"):
        value = int(cfg[key])
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

    try:
        max_downscale = float(cfg["crop_scale_max_downscale"])
    except (TypeError, ValueError):
        raise VaeConfigError(
            "crop_scale_max_downscale must be a number, got "
            f"{cfg['crop_scale_max_downscale']!r}"
        )
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

    for key in ("batch_size", "total_steps", "gradient_accumulation_steps",
                "save_every", "validation_every", "validation_num_images",
                "num_workers", "max_step_saves_to_keep", "lr_warmup_steps",
                "pattern_size", "seed"):
        cfg[key] = int(cfg[key])
    if cfg["batch_size"] < 1 or cfg["total_steps"] < 1 or cfg["gradient_accumulation_steps"] < 1:
        raise VaeConfigError(
            "batch_size / total_steps / gradient_accumulation_steps must be >= 1"
        )

    ema_decay = float(cfg["ema_decay"])
    if not (0.0 < ema_decay < 1.0):
        raise VaeConfigError(f"ema_decay must be in (0, 1), got {ema_decay}")
    cfg["ema_decay"] = ema_decay
    # The boolean toggles were already parsed strictly (and written back as real
    # bools) at the top of this function.
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["max_grad_norm"] = float(cfg["max_grad_norm"])
    cfg["optimizer_weight_decay"] = float(cfg["optimizer_weight_decay"])
    cfg["ycbcr_dc_y_weight"] = float(cfg["ycbcr_dc_y_weight"])
    cfg["ycbcr_dc_chroma_weight"] = float(cfg["ycbcr_dc_chroma_weight"])
    cfg["ycbcr_dc_eps"] = float(cfg["ycbcr_dc_eps"])

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
        try:
            cfg[key] = float(cfg[key])
        except (TypeError, ValueError):
            raise VaeConfigError(f"{key} must be a number, got {cfg[key]!r}")
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
        try:
            cfg[key] = float(cfg[key])
        except (TypeError, ValueError):
            raise VaeConfigError(f"{key} must be a number, got {cfg[key]!r}")
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
