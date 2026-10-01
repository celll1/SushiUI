"""Strict Ming conditioner training ownership and named rate resolution."""

import math

COMPONENTS = ("language", "vision", "connector", "queries", "projections")


def validate_groups(value):
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or item not in COMPONENTS for item in value):
        raise ValueError("Ming conditioner groups must be a list of native component names")
    if len(value) != len(set(value)):
        raise ValueError("Ming conditioner groups must be unique")
    return list(value)


def validate_rates(value):
    if value is None:
        return None
    if not isinstance(value, dict) or any(name not in COMPONENTS for name in value):
        raise ValueError("Unknown Ming conditioner rate group")
    if any(not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate) or rate < 0
           for rate in value.values()):
        raise ValueError("Ming conditioner rates must be finite nonnegative numbers")
    return dict(value)


def resolve_ownership(config):
    from api.param_defaults import TRAINING_DEFAULTS
    configured = validate_groups(config.get("ming_conditioner_groups", TRAINING_DEFAULTS["ming_conditioner_groups"]))
    groups = set(COMPONENTS if config.get("train_text_encoder", TRAINING_DEFAULTS["train_text_encoder"]) else ()) if configured is None else set(configured)
    overrides = validate_rates(config.get("ming_conditioner_lrs", TRAINING_DEFAULTS["ming_conditioner_lrs"]))
    fallback = config.get("text_encoder_lr")
    if fallback is None:
        fallback = config.get("learning_rate", config.get("lr", TRAINING_DEFAULTS["learning_rate"]))
    rates = {name: (overrides or {}).get(name, fallback) for name in COMPONENTS if name in groups}
    validate_rates(rates)
    return groups, rates



def preflight_training(config, *, method):
    from api.arch_capabilities import TRAINING_UNSUPPORTED
    from api.param_defaults import TRAINING_DEFAULTS
    from .cache import configured_modes
    reason = TRAINING_UNSUPPORTED.get("ming_image_01", {}).get(method)
    if reason is not None or method not in {"full_finetune", "lora"}:
        raise ValueError(reason or f"Ming does not support training method {method}")
    if "train_text_encoder" in config and config["train_text_encoder"] is None:
        from api.param_defaults import resolve_full_finetune_train_text_encoder
        config["train_text_encoder"] = (resolve_full_finetune_train_text_encoder(None, "ming_image_01")
            if method == "full_finetune" else TRAINING_DEFAULTS["train_text_encoder"])
    choice = config.get("train_adapter")
    if choice is not None and type(choice) is not bool:
        raise ValueError("Ming train_adapter must be null or a boolean")
    rate = config.get("adapter_lr")
    if rate is not None and (not isinstance(rate, (int, float)) or isinstance(rate, bool)
            or not math.isfinite(rate) or rate < 0):
        raise ValueError("Ming caption adapter rate must be finite and nonnegative")
    groups, _ = resolve_ownership(config)
    if method == "lora" and groups:
        raise ValueError("Ming conditioner LoRA is not yet connected; freeze native conditioner groups")
    config["train_text_encoder"] = bool(groups)
    if not config.get("train_unet", TRAINING_DEFAULTS["train_unet"]) and not groups and config.get("train_adapter") is not True:
        raise ValueError("Ming training requires DiT parameters or a native conditioner group")
    if config.get("train_image_encoder") or config.get("train_vision_encoder") or config.get("vision_encoder_path"):
        raise ValueError("Ming native vision training uses ming_conditioner_groups, not an external image encoder")
    modes = configured_modes(config)
    if groups and modes["text_encoding_mode"] == "cpu_prefetch":
        raise ValueError("Ming CPU prefetch requires frozen conditioner groups")
    if config.get("noise_process", TRAINING_DEFAULTS["noise_process"]) not in {"auto", "flow"}:
        raise ValueError("Ming training requires the native flow noise process")
    if config.get("prediction_target", TRAINING_DEFAULTS["prediction_target"]) not in {"auto", "velocity"}:
        raise ValueError("Ming training requires native clean-minus-noise velocity prediction")
    layers = config.get("ming_image_01_sample_layers", TRAINING_DEFAULTS["ming_image_01_sample_layers"])
    if not isinstance(layers, int) or isinstance(layers, bool) or layers < 1:
        raise ValueError("Ming training preview layer count must be an integer >= 1")
    return groups
