"""Header-validated native DiT weight-only quantized package loading."""

from __future__ import annotations


_FLOATS = {"F16", "BF16", "F32"}
_QUANTIZED = {"I8", "F8_E4M3"}


def _component_files(directory):
    from .artifact import _json, _inside
    indices = list(directory.glob("*.safetensors.index.json"))
    return ([_inside(directory, path) for path in sorted(set(_json(indices[0])["weight_map"].values()))]
            if indices else list(directory.glob("*.safetensors")))


def _convrot_layers(artifact, component="transformer"):
    from safetensors import safe_open
    from core.models.common.convrot_marker import supported_int8_convrot_marker
    headers = artifact.tensor_headers[component]
    markers = {key for key in headers if key.endswith(".comfy_quant")}
    if not markers:
        return {}
    for key in markers:
        h = headers[key]
        if h.dtype != "U8" or len(h.shape) != 1 or not 0 < h.elements <= 4096:
            raise ValueError(f"Invalid Ming ConvRot marker header: {key}")
    common_headers = {key: {"shape": list(h.shape), "dtype": h.dtype} for key, h in headers.items()}
    layers = {}
    for path in _component_files(artifact.components[component]):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in markers.intersection(handle.keys()):
                config = supported_int8_convrot_marker(key, handle.get_tensor(key), common_headers, path=str(path))
                if config is None:
                    raise ValueError(f"Unsupported Ming ConvRot marker: {key}")
                layers[key[:-len(".comfy_quant")]] = config
    if len(layers) != len(markers):
        raise ValueError("Ming ConvRot marker payload is missing")
    return layers


def prepare_quantized_transformer(artifact, dtype, *, topology=None):
    """Validate before loading other components; construct quantized slots on meta."""
    from .model_definition import create_transformer

    headers = artifact.tensor_headers["transformer"]
    for name, component in artifact.tensor_headers.items():
        if name not in {"transformer", "mllm", "connector", "mlp"} and any(key.endswith((".weight_scale", ".comfy_quant")) or
                (key.endswith(".weight") and h.dtype in _QUANTIZED) for key, h in component.items()):
            raise ValueError(f"Ming quantized {name} loading is not implemented")
    if (topology or {}).get("method") != "hybrid_unet_ssdit" and all(
            h.dtype in _FLOATS and not key.endswith((".weight_scale", ".comfy_quant")) for key, h in headers.items()):
        return None
    convrot = _convrot_layers(artifact)
    model = create_transformer(artifact.configs["transformer"], device="meta", dtype=dtype, topology=topology)
    return prepare_component_slots(model,headers,convrot,dtype)


def prepare_component_slots(model, headers, convrot, dtype):
    """Validate a meta component and replace declared packed Linear slots."""
    import torch
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    expected = model.state_dict(keep_vars=True)
    tied = {}
    for key,value in expected.items():
        tied.setdefault(id(value),[]).append(key)
    aliases = [keys for keys in tied.values() if len(keys)>1]
    missing_aliases = set()
    for keys in aliases:
        present = set(keys).intersection(headers)
        if not present:
            continue
        if any(headers[key].dtype not in _FLOATS for key in present):
            raise ValueError("Ming packed tied weights are unsupported; keep embeddings/output heads floating")
        missing_aliases.update(set(keys)-present)
    model._ming_quantized_alias_groups = aliases
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
    allowed = (set(expected)-missing_aliases) | {path + ".weight_scale" for path in quantized} | {path + ".comfy_quant" for path in convrot}
    if set(headers) != allowed:
        raise ValueError(f"Ming quantized state mismatch: missing={sorted(allowed-set(headers))[:8]}, unexpected={sorted(set(headers)-allowed)[:8]}")
    for key, value in expected.items():
        if key in missing_aliases:
            continue
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


def load_quantized_component(model, artifact, component, dtype):
    """Assign validated CPU shards without materializing the whole checkpoint."""
    import torch
    from safetensors.torch import load_file
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    files = _component_files(artifact.components[component])
    for path in files:
        state = load_file(str(path), device="cpu")
        for key, value in state.items():
            if key.startswith("rewiring_modules.") and not torch.isfinite(value).all():
                raise ValueError(f"Ming hybrid checkpoint contains nonfinite weights: {key}")
            if key.endswith(".weight_scale") and (not torch.isfinite(value).all() or (value <= 0).any()):
                raise ValueError(f"Ming quantized Linear scale must be finite and positive: {key}")
            if key.endswith(".weight_scale") and isinstance(model.get_submodule(key[:-len(".weight_scale")]), ConvRotInt8Linear):
                state[key] = value.reshape(-1).contiguous()
            if artifact.tensor_headers[component][key].dtype in _FLOATS and not key.endswith(".weight_scale"):
                state[key] = value.to(dtype=dtype)
        info = model.load_state_dict(state, strict=False, assign=True)
        if info.unexpected_keys:
            raise ValueError(f"Ming quantized shard state mismatch: {info.unexpected_keys[:8]}")
        del state
    for keys in getattr(model,"_ming_quantized_alias_groups",[]):
        loaded = [key for key in keys if key in artifact.tensor_headers[component]]
        if not loaded:
            continue
        def parameter(key):
            parent,_,name=key.rpartition(".")
            return getattr(model.get_submodule(parent) if parent else model,name)
        canonical=parameter(loaded[0])
        if any(not torch.equal(canonical,parameter(key)) for key in loaded[1:]):
            raise ValueError("Ming tied floating checkpoint weights disagree")
        for key in keys:
            parent,_,name=key.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model,name,canonical)
    if any(value.device.type == "meta" for value in model.state_dict().values()):
        raise ValueError("Ming quantized checkpoint left unloaded tensors")
    return model


def load_quantized_transformer(model, artifact, dtype):
    return load_quantized_component(model,artifact,"transformer",dtype)


def configure_frozen_quantized_training(model):
    """Pin frozen bases to differentiable dequantization despite inherited flags."""
    from core.models.ideogram4.vendor.int8_linear import Int8Linear, disable_int8_mm
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear, disable_scaled_mm
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    disable_int8_mm(model)
    disable_scaled_mm(model)
    counts = {"int8":0,"e4m3":0,"convrot_int8":0}
    for module in model.modules():
        if isinstance(module,ConvRotInt8Linear):
            module._force_dequant = True
            counts["convrot_int8"] += 1
        elif isinstance(module,Int8Linear):
            counts["int8"] += 1
        elif isinstance(module,Fp8Linear):
            counts["e4m3"] += 1
    return {"execution":"dequant_autograd", "formats":counts} if sum(counts.values()) else None



def materialize_training_linears(model, dtype):
    """Stream frozen packed CPU rows into trainable floating Linear weights."""
    import torch
    from core.models.ideogram4.vendor.int8_linear import Int8Linear
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    if dtype not in {torch.float16,torch.bfloat16,torch.float32}:
        raise ValueError("Ming full dequantization requires FP16, BF16 or FP32 weights")
    targets = []
    counts = {"int8":0,"e4m3":0,"convrot_int8":0}
    for path, child in model.named_modules():
        if isinstance(child,ConvRotInt8Linear): kind="convrot_int8"
        elif isinstance(child,Int8Linear): kind="int8"
        elif isinstance(child,Fp8Linear): kind="e4m3"
        else: continue
        if child.weight.device.type != "cpu" or child.weight_scale.device.type != "cpu" or (child.bias is not None and child.bias.device.type != "cpu"):
            raise ValueError("Ming full dequantization requires CPU-resident packed weights/scales/bias")
        if child.weight.shape!=(child.out_features,child.in_features) or child.weight_scale.shape!=(child.out_features,) or not torch.isfinite(child.weight_scale).all() or (child.weight_scale<=0).any():
            raise ValueError("Invalid Ming full dequantization row/scale geometry")
        targets.append((path,kind))
    if not targets:
        return None
    if any(kind=="convrot_int8" for _,kind in targets):
        from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    for path, kind in targets:
        child = model.get_submodule(path)
        if kind=="convrot_int8":
            weight = dequantize_int8_convrot_weight(child.weight,child.weight_scale.reshape(-1,1),child.convrot_groupsize).to(dtype=dtype).contiguous()
        elif kind=="int8":
            weight = child.weight * child.weight_scale.to(dtype=dtype).unsqueeze(1)
        else:
            weight = child.weight.to(dtype=dtype) * child.weight_scale.to(dtype=dtype).unsqueeze(1)
        if not torch.isfinite(weight).all():
            raise ValueError(f"Ming materialized floating weight is nonfinite: {path}")
        bias = None if child.bias is None else child.bias.detach().to(dtype=dtype).clone()
        if bias is not None and not torch.isfinite(bias).all():
            raise ValueError(f"Ming materialized floating bias is nonfinite: {path}")
        replacement = torch.nn.Linear(child.in_features,child.out_features,bias=bias is not None,device="meta",dtype=dtype)
        replacement.weight = torch.nn.Parameter(weight.detach().contiguous())
        if bias is not None:
            replacement.bias = torch.nn.Parameter(bias)
        replacement.train(child.training)
        parent_name,_,name = path.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent,name,replacement)
        counts[kind] += 1
    return {"schema_version":1,"execution":"materialized_floating","formats":counts,
            "weight_dtype":str(dtype).removeprefix("torch."),"lossy_source":True}


def saved_full_initialization(artifact, *, component="dit"):
    from .artifact import _json, _inside
    path = _inside(artifact.root,"training.json")
    if not path.is_file():
        return None
    keys = {"dit":"ming_full_initialization", "conditioner":"ming_conditioner_full_initialization"}
    if component not in keys:
        raise ValueError("Unknown Ming full-initialization component")
    value = _json(path).get(keys[component])
    if value is None:
        return None
    if (not isinstance(value,dict) or set(value)!={"schema_version","execution","formats","weight_dtype","lossy_source"}
            or type(value["schema_version"]) is not int or value["schema_version"]!=1 or value["execution"]!="materialized_floating"
            or value["weight_dtype"] not in {"float16","bfloat16","float32"} or value["lossy_source"] is not True
            or not isinstance(value["formats"],dict) or set(value["formats"])!={"int8","e4m3","convrot_int8"}
            or any(type(count) is not int or count<0 for count in value["formats"].values())
            or not sum(value["formats"].values())):
        raise ValueError("Invalid Ming full-training initialization metadata")
    return value
