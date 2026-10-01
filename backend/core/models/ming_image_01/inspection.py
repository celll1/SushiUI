"""Loaded Ming structure snapshots without CUDA probes or residency changes."""

from core.rewiring.parameters import parameter_counts
from .residual_unet import native_config


def _parameters(modules, seen):
    result = []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def loaded_model_info(components):
    """Count loaded parameter objects; serialized storage remains a separate census."""
    groups = {"dit": (components.transformer,), **components.conditioner.component_groups(),
              "vae": (components.vae,)}
    seen, census = set(), {}
    for name, modules in groups.items():
        parameters = _parameters(modules, seen)
        dtypes, devices = {}, {}
        for parameter in parameters:
            count = parameter.numel()
            dtype = str(parameter.dtype).removeprefix("torch.")
            dtypes[dtype] = dtypes.get(dtype, 0) + count * parameter.element_size()
            device = str(parameter.device)
            devices[device] = devices.get(device, 0) + count * parameter.element_size()
        census[name] = {"parameters": sum(p.numel() for p in parameters),
                        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
                        "logical_parameter_bytes": sum(dtypes.values()),
                        "bytes_by_dtype": dtypes, "logical_bytes_by_device": devices}
    owned = {id(parameter) for module in (components.transformer, components.conditioner, components.vae)
             for parameter in module.parameters()}
    if owned != seen:
        raise ValueError("Ming component census found parameters without explicit ownership")
    config = native_config(components.transformer)
    return {"type": "ming_image_01", "profile": components.artifact.profile,
            "is_v_prediction": False,
            "component_parameter_counts": census,
            "total_loaded_parameters": sum(row["parameters"] for row in census.values()),
            "trainable_parameters": sum(row["trainable_parameters"] for row in census.values()),
            "parameter_accounting": "unique_loaded_parameter_objects_excludes_buffers_and_quantization_payloads",
            "serialized_storage": components.artifact.summary()["components"],
            "conditioner_cacheable": components.conditioner.cacheable,
            "dit_rewiring_parameter_counts": parameter_counts("ming_image_01", config),
            "dit_rewiring_spec": getattr(components.transformer, "_residual_unet_spec", None),
            "dit_rewiring_token_stride": 16,
            "main_blocks": len(components.transformer.layers),
            "noise_refiner_blocks": len(components.transformer.noise_refiner),
            "context_refiner_blocks": len(components.transformer.context_refiner)}
