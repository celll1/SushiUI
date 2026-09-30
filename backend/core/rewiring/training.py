"""Training configuration gates shared by requests, YAML and runtime resolution."""
from .capability import RUNTIME_ARCHITECTURES
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
        if architecture not in RUNTIME_ARCHITECTURES:
            raise ValueError(f"Residual rewiring is not yet supported for {architecture}")
        if spec is not None and spec["architecture"] != architecture:
            raise ValueError("Training topology architecture mismatch")
        validate_training_features(config)
    return mode, spec


def validate_training_features(config):
    from api.param_defaults import TRAINING_DEFAULTS
    if float(config.get("block_skip_rate", TRAINING_DEFAULTS["block_skip_rate"])) > 0:
        raise ValueError("Residual rewiring is incompatible with block_skip_rate")
    for key in ("dit_partition_training_enabled", "qwen_partition_training_enabled",
                "dit_partition_global_adapter_enabled", "qwen_partition_global_adapter_enabled",
                "tread_enable", "repa_enable", "blockskip_enable"):
        if config.get(key):
            raise ValueError(f"Residual rewiring is incompatible with {key}")


NO_RESUME_SPEC = object()


def resolve_training_topology(model, config, architecture, validate_topology_identity,
                              restore_topology, default_topology=None, *, resuming=False,
                              pending_adapter_resume=False, resume_spec=NO_RESUME_SPEC,
                              install=True):
    mode, requested = validate_training_rewiring(config, architecture)
    existing = getattr(model, "_residual_unet_spec", None)
    if mode == "configure" and requested is None:
        requested = None if default_topology is None else default_topology(dict(model.config))
        if requested is None:
            raise ValueError(f"No rewiring preset for this {architecture} block count; supply an explicit spec")
    if requested is not None:
        requested = validate_topology_identity(requested, dict(model.config))
    if resume_spec is not NO_RESUME_SPEC:
        if resume_spec is not None:
            resume_spec = validate_topology_identity(resume_spec, dict(model.config))
        if existing is not None and existing != resume_spec:
            raise ValueError("Resume checkpoint and base topology disagree")
        inherited = resume_spec
    else:
        inherited = requested if pending_adapter_resume and existing is None else existing
    if mode == "inherit":
        if requested is not None and requested != inherited:
            raise ValueError("Explicit inherited topology does not match checkpoint")
        resolved = inherited
    else:
        resolved = requested
        if (resuming or resume_spec is not NO_RESUME_SPEC) and resolved != inherited:
            raise ValueError("Resume requires the checkpoint topology without edits")
    if resolved is not None:
        validate_training_features(config)
    if install:
        restore_topology(model, resolved)
        config["dit_rewiring_mode"] = mode
        config["dit_rewiring_spec"] = resolved
    return resolved
