"""Native Ming transformer construction and parameter accounting."""

from __future__ import annotations

import inspect
from typing import Any


def create_transformer(config: dict[str, Any], *, device: str, dtype, topology=None):
    import torch
    from .vendor.diffusion.transformer import DiffusionTransformer

    parameters = inspect.signature(DiffusionTransformer.__init__).parameters
    arguments = {key: value for key, value in config.items() if key in parameters and key != "self"}
    with torch.device(device):
        model = DiffusionTransformer(**arguments)
        if topology is not None:
            from .residual_unet import restore_topology
            restore_topology(model, topology)
    return model.to(dtype=dtype)


def transformer_parameter_census(config: dict[str, Any]) -> dict[str, Any]:
    """Count native parameters on meta; no model weights or CUDA allocation."""
    import torch

    model = create_transformer(config, device="meta", dtype=torch.float32)
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + parameter.numel()
        if parameter.device.type != "meta":
            raise RuntimeError("Parameter census unexpectedly allocated a model tensor")
    return {"total_parameters": sum(groups.values()), "groups": groups,
            "main_blocks": len(model.layers), "noise_refiner_blocks": len(model.noise_refiner),
            "context_refiner_blocks": len(model.context_refiner),
            "accounting": "native_parameters_not_serialized_buffers"}
