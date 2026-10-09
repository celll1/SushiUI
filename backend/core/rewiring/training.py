"""Training configuration gates shared by requests, YAML and runtime resolution."""
from .capability import RUNTIME_ARCHITECTURES
from .spec import validate_spec


def validate_training_rewiring(config, architecture, *, checkpoint_inheritance=False):
    from api.param_defaults import TRAINING_DEFAULTS
    mode = config.get("dit_rewiring_mode", TRAINING_DEFAULTS["dit_rewiring_mode"])
    if mode not in ("inherit", "configure"):
        raise ValueError("dit_rewiring_mode must be inherit or configure")
    spec = config.get("dit_rewiring_spec", TRAINING_DEFAULTS["dit_rewiring_spec"])
    if spec is not None:
        spec = validate_spec(spec)
    if mode == "configure" or spec is not None:
        if architecture not in RUNTIME_ARCHITECTURES and not (mode == "inherit" and checkpoint_inheritance):
            raise ValueError(f"Residual rewiring is not yet supported for {architecture}")
        if spec is not None and spec["architecture"] != architecture:
            raise ValueError("Training topology architecture mismatch")
        validate_training_features(config, architecture)
    return mode, spec


def validate_training_features(config, architecture=None):
    from api.param_defaults import TRAINING_DEFAULTS
    requested = config.get("dit_rewiring_spec")
    requested = None if requested is None else validate_spec(requested)
    hybrid = requested is not None and requested["method"] == "hybrid_unet_ssdit"
    if hybrid:
        from .hybrid_spec import skip_storage
        if (any(skip_storage(item) == "recompute" for item in requested["skips"])
                and int(config.get("blocks_to_swap", TRAINING_DEFAULTS["blocks_to_swap"])) > 0):
            raise ValueError("Hybrid encoder replay is incompatible with Block Swap; use blocks_to_swap=0")
        if config.get("training_method") not in (None, "full", "full_finetune", "lora"):
            raise ValueError("Hybrid U-Net supports full-parameter or LoRA training")
        if architecture != "anima" and int(config.get("blocks_to_swap", TRAINING_DEFAULTS["blocks_to_swap"])) > 0:
            raise ValueError("Hybrid U-Net training Block Swap currently supports Anima only; use blocks_to_swap=0 and activation offload for other architectures")
    if float(config.get("block_skip_rate", TRAINING_DEFAULTS["block_skip_rate"])) > 0:
        raise ValueError("Residual rewiring is incompatible with block_skip_rate")
    for key in ("dit_partition_training_enabled", "qwen_partition_training_enabled",
                "dit_partition_global_adapter_enabled", "qwen_partition_global_adapter_enabled",
                "tread_enable", "repa_enable", "blockskip_enable"):
        if (architecture in {"ming_image_01", "anima"} or hybrid and architecture == "qwen_image_21") and key in {"dit_partition_training_enabled", "qwen_partition_training_enabled",
                "dit_partition_global_adapter_enabled", "qwen_partition_global_adapter_enabled"}:
            continue
        if config.get(key):
            raise ValueError(f"Residual rewiring is incompatible with {key}")


NO_RESUME_SPEC = object()


def validate_loaded_hybrid_training(trainer, model):
    spec = getattr(model, "_residual_unet_spec", None)
    if spec is not None and spec["method"] == "hybrid_unet_ssdit":
        from core.training.ops.training_method import resolve_training_method
        validate_training_features(dict(trainer.config, dit_rewiring_spec=spec,
            training_method=resolve_training_method(trainer)), spec["architecture"])


def prepare_topology_io_rebase(config, architecture, source_config, *, declaration=None):
    """Validate source identity before IO migration; preserve its execution layout."""
    from .spec import rebase_spec, validate_model_identity, digest
    if declaration is not None:
        declaration["rewiring_source_config_sha256"] = digest(source_config)
    _, requested = validate_training_rewiring(config, architecture)
    if requested is None:
        return None
    requested = validate_model_identity(requested, architecture, requested["streams"], source_config)

    def finish(target_config):
        config["dit_rewiring_spec"] = rebase_spec(requested, target_config)

    return finish


def resolve_training_topology(model, config, architecture, validate_topology_identity,
                              restore_topology, default_topology=None, *, resuming=False,
                              pending_adapter_resume=False, resume_spec=NO_RESUME_SPEC,
                              install=True):
    existing = getattr(model, "_residual_unet_spec", None)
    mode, requested = validate_training_rewiring(config, architecture,
        checkpoint_inheritance=existing is not None or
        (resume_spec is not NO_RESUME_SPEC and resume_spec is not None))
    if requested is not None and existing is not None:
        pair = getattr(model,"vae_latent_bridge",None)
        source_identity = (pair.declaration.get("rewiring_source_config_sha256") if pair is not None
                           else getattr(model,"_rewiring_source_config_sha256",None))
        if source_identity is not None and requested["base_config_sha256"] == source_identity:
            rebound = dict(requested,base_config_sha256=existing["base_config_sha256"])
            if rebound == existing:
                requested = existing
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
        validate_training_features(dict(config, dit_rewiring_spec=resolved), architecture)
    if install:
        restore_topology(model, resolved)
        config["dit_rewiring_mode"] = mode
        config["dit_rewiring_spec"] = resolved
    return resolved
