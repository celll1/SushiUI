"""Header-validated native DiT weight-only quantized package loading."""

from __future__ import annotations


_FLOATS = {"F16", "BF16", "F32"}
_QUANTIZED = {"I8", "F8_E4M3"}


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
            if scale is None or scale.dtype != "F32" or scale.shape != (module.out_features,):
                raise ValueError(f"Ming quantized Linear requires a per-row F32 scale: {path}")
            quantized[path] = (module, h.dtype)
        elif h.dtype not in _FLOATS or key.endswith(".comfy_quant"):
            raise ValueError(f"Unsupported Ming quantized tensor: {key} ({h.dtype})")
    allowed = set(expected) | {path + ".weight_scale" for path in quantized}
    if set(headers) != allowed:
        raise ValueError(f"Ming quantized state mismatch: missing={sorted(allowed-set(headers))[:8]}, unexpected={sorted(set(headers)-allowed)[:8]}")
    for key, value in expected.items():
        if headers[key].shape != tuple(value.shape):
            raise ValueError(f"Ming quantized state shape differs: {key}")
    for path, (linear, format_) in quantized.items():
        parent, _, leaf = path.rpartition(".")
        cls = Int8Linear if format_ == "I8" else Fp8Linear
        with torch.device("meta"):
            replacement = cls(linear.in_features, linear.out_features, linear.bias is not None, dtype)
        setattr(model.get_submodule(parent) if parent else model, leaf, replacement)
    return model


def load_quantized_transformer(model, artifact, dtype):
    """Assign validated CPU shards without materializing the whole checkpoint."""
    import torch
    from safetensors.torch import load_file
    from .artifact import _json, _inside

    directory = artifact.components["transformer"]
    indices = list(directory.glob("*.safetensors.index.json"))
    files = ([_inside(directory, path) for path in sorted(set(_json(indices[0])["weight_map"].values()))]
             if indices else list(directory.glob("*.safetensors")))
    for path in files:
        state = load_file(str(path), device="cpu")
        for key, value in state.items():
            if key.endswith(".weight_scale") and (not torch.isfinite(value).all() or (value <= 0).any()):
                raise ValueError(f"Ming quantized Linear scale must be finite and positive: {key}")
            if artifact.tensor_headers["transformer"][key].dtype in _FLOATS and not key.endswith(".weight_scale"):
                state[key] = value.to(dtype=dtype)
        info = model.load_state_dict(state, strict=False, assign=True)
        if info.unexpected_keys:
            raise ValueError(f"Ming quantized shard state mismatch: {info.unexpected_keys[:8]}")
        del state
    if any(value.device.type == "meta" for value in model.state_dict().values()):
        raise ValueError("Ming quantized checkpoint left unloaded tensors")
    return model
