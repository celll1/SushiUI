"""Compatibility checks for metadata-driven adapter topology scopes."""
from .spec import read_checkpoint_metadata


def resolve_adapter_topology(model, files, validate_identity):
    base = getattr(model, "_residual_unet_spec", None)
    if base is not None:
        base = validate_identity(base, dict(model.config))
    declarations = []
    for file in files:
        topology = read_checkpoint_metadata(file.metadata)
        if topology is not None:
            topology = validate_identity(topology, dict(model.config))
        declarations.append(topology)
    required = declarations[0] if declarations else base
    if any(value != required for value in declarations):
        raise ValueError("Selected LoRAs require different residual topologies")
    if base is not None and required != base:
        raise ValueError("LoRA topology does not match the loaded checkpoint")
    return required
