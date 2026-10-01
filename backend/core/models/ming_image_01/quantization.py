"""Header-validated native DiT weight-only quantized package loading."""

from __future__ import annotations


_FLOATS = {"F16", "BF16", "F32"}
_QUANTIZED = {"I8", "F8_E4M3"}


def _component_files(directory):
    from .artifact import _json, _inside
    indices = list(directory.glob("*.safetensors.index.json"))
    return ([_inside(directory, path) for path in sorted(set(_json(indices[0])["weight_map"].values()))]
            if indices else list(directory.glob("*.safetensors")))


def _convrot_layers(artifact):
    from safetensors import safe_open
    from core.models.common.convrot_marker import supported_int8_convrot_marker
    headers = artifact.tensor_headers["transformer"]
    markers = {key for key in headers if key.endswith(".comfy_quant")}
    if not markers:
        return {}
    for key in markers:
        h = headers[key]
        if h.dtype != "U8" or len(h.shape) != 1 or not 0 < h.elements <= 4096:
            raise ValueError(f"Invalid Ming ConvRot marker header: {key}")
    common_headers = {key: {"shape": list(h.shape), "dtype": h.dtype} for key, h in headers.items()}
    layers = {}
    for path in _component_files(artifact.components["transformer"]):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in markers.intersection(handle.keys()):
                config = supported_int8_convrot_marker(key, handle.get_tensor(key), common_headers, path=str(path))
                if config is None:
                    raise ValueError(f"Unsupported Ming ConvRot marker: {key}")
                layers[key[:-len(".comfy_quant")]] = config
    if len(layers) != len(markers):
        raise ValueError("Ming ConvRot marker payload is missing")
    return layers


def prepare_quantized_transformer(artifact, dtype):
    """Validate before loading other components; construct quantized slots on meta."""
    import torch
    from .model_definition import create_transformer
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear

    headers = artifact.tensor_headers["transformer"]
    for name, component in artifact.tensor_headers.items():
        if name != "transformer" and any(key.endswith((".weight_scale", ".comfy_quant")) or
                (key.endswith(".weight") and h.dtype in _QUANTIZED) for key, h in component.items()):
            raise ValueError(f"Ming quantized {name} loading is not implemented")
    if all(h.dtype in _FLOATS and not key.endswith((".weight_scale", ".comfy_quant"))
           for key, h in headers.items()):
        return None
    convrot = _convrot_layers(artifact)
    model = create_transformer(artifact.configs["transformer"], device="meta", dtype=dtype)
    expected = model.state_dict()
    quantized = {}
    for key, h in headers.items():
        if key.endswith(".weight") and h.dtype in _QUANTIZED:
            path = key[:-len(".weight")]
            try:
                module = model.get_submodule(path)
            except AttributeError as exc:
                raise ValueError(f"Unknown Ming quantized Linear: {path}") from exc
            scale = headers.get(path + ".weight_scale")
            if not isinstance(module, torch.nn.Linear) or h.shape != (module.out_features, module.in_features):
                raise ValueError(f"Ming quantized Linear geometry differs: {path}")
            scale_shapes = {(module.out_features,)}
            if path in convrot:
                scale_shapes.add((module.out_features, 1))
            if scale is None or scale.dtype != "F32" or scale.shape not in scale_shapes:
                raise ValueError(f"Ming quantized Linear requires a per-row F32 scale: {path}")
            quantized[path] = (module, h.dtype)
        elif key.endswith(".comfy_quant") and key[:-len(".comfy_quant")] in convrot:
            continue
        elif h.dtype not in _FLOATS or key.endswith(".comfy_quant"):
            raise ValueError(f"Unsupported Ming quantized tensor: {key} ({h.dtype})")
    allowed = set(expected) | {path + ".weight_scale" for path in quantized} | {path + ".comfy_quant" for path in convrot}
    if set(headers) != allowed:
        raise ValueError(f"Ming quantized state mismatch: missing={sorted(allowed-set(headers))[:8]}, unexpected={sorted(set(headers)-allowed)[:8]}")
    for key, value in expected.items():
        if headers[key].shape != tuple(value.shape):
            raise ValueError(f"Ming quantized state shape differs: {key}")
    if convrot:
        from core.models.common.convrot_int8_linear import ConvRotInt8Linear, require_convrot_int8_runtime
        require_convrot_int8_runtime()
    for path, (linear, format_) in quantized.items():
        parent, _, leaf = path.rpartition(".")
        cls = Int8Linear if format_ == "I8" else Fp8Linear
        with torch.device("meta"):
            if path in convrot:
                replacement = ConvRotInt8Linear(linear.in_features, linear.out_features,
                    linear.bias is not None, dtype, **convrot[path])
            else:
                replacement = cls(linear.in_features, linear.out_features, linear.bias is not None, dtype)
        setattr(model.get_submodule(parent) if parent else model, leaf, replacement)
    return model


def load_quantized_transformer(model, artifact, dtype):
    """Assign validated CPU shards without materializing the whole checkpoint."""
    import torch
    from safetensors.torch import load_file
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    files = _component_files(artifact.components["transformer"])
    for path in files:
        state = load_file(str(path), device="cpu")
        for key, value in state.items():
            if key.endswith(".weight_scale") and (not torch.isfinite(value).all() or (value <= 0).any()):
                raise ValueError(f"Ming quantized Linear scale must be finite and positive: {key}")
            if key.endswith(".weight_scale") and isinstance(model.get_submodule(key[:-len(".weight_scale")]), ConvRotInt8Linear):
                state[key] = value.reshape(-1).contiguous()
            if artifact.tensor_headers["transformer"][key].dtype in _FLOATS and not key.endswith(".weight_scale"):
                state[key] = value.to(dtype=dtype)
        info = model.load_state_dict(state, strict=False, assign=True)
        if info.unexpected_keys:
            raise ValueError(f"Ming quantized shard state mismatch: {info.unexpected_keys[:8]}")
        del state
    if any(value.device.type == "meta" for value in model.state_dict().values()):
        raise ValueError("Ming quantized checkpoint left unloaded tensors")
    return model
