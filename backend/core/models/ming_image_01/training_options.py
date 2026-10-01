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
        fallback = config.get("learning_rate", TRAINING_DEFAULTS["learning_rate"])
    rates = {name: (overrides or {}).get(name, fallback) for name in COMPONENTS if name in groups}
    validate_rates(rates)
    return groups, rates
