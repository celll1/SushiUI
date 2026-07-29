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
VALID_DTYPES = ("bf16", "fp32")
VALID_LPIPS_NETS = ("vgg", "alex", "squeeze")

# Loss keys that participate in the "at least one active term" check.
_LOSS_WEIGHT_KEYS = ("mse_weight", "l1_weight", "lpips_weight",
                     "ycbcr_dc_weight", "pattern_weight")


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
    vae_section = process_config.get("vae") or {}

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
    train_decoder = bool(cfg["train_decoder"])
    train_encoder = bool(cfg["train_encoder"])

    if train_encoder:
        # Phase 2. Training the encoder moves the latent distribution, which
        # invalidates every latent cache, every LoRA and every diffusion model
        # trained against this VAE. v1 does not ship it, and a config asking for
        # it must fail loudly rather than silently train the decoder only.
        raise VaeConfigError(
            "train_encoder=true is not supported in this version (Phase 2). "
            "Encoder training moves the latent distribution and invalidates "
            "every latent cache / LoRA / diffusion model trained against this "
            "VAE. Set train_encoder=false (decoder-only)."
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
        raise VaeConfigError(
            "All loss weights are 0: there is no training signal. Set at least "
            f"one of {list(_LOSS_WEIGHT_KEYS)} above 0 (default: mse_weight=1.0)."
        )

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
    cfg["ema_enabled"] = bool(cfg["ema_enabled"])
    cfg["learning_rate"] = float(cfg["learning_rate"])
    cfg["max_grad_norm"] = float(cfg["max_grad_norm"])
    cfg["optimizer_weight_decay"] = float(cfg["optimizer_weight_decay"])
    cfg["ycbcr_dc_y_weight"] = float(cfg["ycbcr_dc_y_weight"])
    cfg["ycbcr_dc_chroma_weight"] = float(cfg["ycbcr_dc_chroma_weight"])
    cfg["ycbcr_dc_eps"] = float(cfg["ycbcr_dc_eps"])
