"""Training configuration gates shared by requests, YAML and runtime resolution."""
from .spec import validate_spec


def validate_training_rewiring(config, architecture):
    from api.param_defaults import TRAINING_DEFAULTS
    mode = config.get("dit_rewiring_mode", TRAINING_DEFAULTS["dit_rewiring_mode"])
    if mode not in ("inherit", "configure"):
        raise ValueError("dit_rewiring_mode must be inherit or configure")
    spec = config.get("dit_rewiring_spec", TRAINING_DEFAULTS["dit_rewiring_spec"])
    if spec is not None:
        spec = validate_spec(spec)
    if mode == "configure" or spec is not None:
        if architecture != "qwen_image_21":
            raise ValueError(f"Residual rewiring is not yet supported for {architecture}")
        if spec is not None and spec["architecture"] != architecture:
            raise ValueError("Training topology architecture mismatch")
        validate_training_features(config)
    return mode, spec


def validate_training_features(config):
    for key in ("dit_partition_training_enabled", "qwen_partition_training_enabled",
                "dit_partition_global_adapter_enabled", "qwen_partition_global_adapter_enabled",
                "tread_enable", "repa_enable", "blockskip_enable"):
        if config.get(key):
            raise ValueError(f"Residual rewiring is incompatible with {key}")
