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
    from core.training import guidance_loss as _guidance
    from .cache import configured_modes
    from .convrot_training import settings
    settings(config, method=method)
    reason = TRAINING_UNSUPPORTED.get("ming_image_01", {}).get(method)
    if reason is not None or method not in {"full_finetune", "lora"}:
        raise ValueError(reason or f"Ming does not support training method {method}")
    if "train_text_encoder" in config and config["train_text_encoder"] is None:
        from api.param_defaults import resolve_full_finetune_train_text_encoder
        config["train_text_encoder"] = (resolve_full_finetune_train_text_encoder(None, "ming_image_01")
            if method == "full_finetune" else TRAINING_DEFAULTS["train_text_encoder"])
    chunk = config.get("qwen_full_kv_query_chunk_tokens", TRAINING_DEFAULTS["qwen_full_kv_query_chunk_tokens"])
    if type(chunk) is not int or chunk < 0:
        raise ValueError("Ming full-KV query chunk must be a nonnegative integer")
    global_enabled=config.get("dit_partition_global_adapter_enabled",config.get("qwen_partition_global_adapter_enabled",TRAINING_DEFAULTS["dit_partition_global_adapter_enabled"]))
    if global_enabled:
        if not config.get("train_unet",TRAINING_DEFAULTS["train_unet"]):raise ValueError("Ming global adapter requires train_unet")
        if not config.get("dit_partition_training_enabled",config.get("qwen_partition_training_enabled",False)):
            raise ValueError("Ming global adapter requires spatial partition training")
    materialize = config.get("full_finetune_dequantize_int8_base", TRAINING_DEFAULTS["full_finetune_dequantize_int8_base"])
    if type(materialize) is not bool or (materialize and method != "full_finetune"):
        raise ValueError("Ming full dequantization requires a boolean opt-in and full_finetune")
    export = config.get("training_export_format", TRAINING_DEFAULTS["training_export_format"])
    if not isinstance(export, str) or export not in {"none", "int8_convrot"} or (export != "none" and method != "full_finetune"):
        raise ValueError("Ming inference training export requires full_finetune and a supported format")
    choice = config.get("train_adapter")
    if choice is not None and type(choice) is not bool:
        raise ValueError("Ming train_adapter must be null or a boolean")
    rate = config.get("adapter_lr")
    if rate is not None and (not isinstance(rate, (int, float)) or isinstance(rate, bool)
            or not math.isfinite(rate) or rate < 0):
        raise ValueError("Ming caption adapter rate must be finite and nonnegative")
    groups, _ = resolve_ownership(config)
    guidance=_guidance.setting(config,"weight")
    if not isinstance(guidance,(float,int)) or isinstance(guidance,bool) or not math.isfinite(guidance) or not 0<=guidance<=1:
        raise ValueError("Ming guidance weight must be finite and within [0, 1]")
    high=_guidance.setting(config,"high_noise_weight")
    if not isinstance(high,(float,int)) or isinstance(high,bool) or not math.isfinite(high) or not 0<=high<=1:
        raise ValueError("Ming high-noise guidance weight must be finite and within [0, 1]")
    ramp=_guidance.setting(config,"weight_schedule")
    if groups and (guidance or (ramp=="high_noise_smoothstep" and high>0)):
        raise ValueError("Ming guidance loss requires a frozen text encoder/conditioner")

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
