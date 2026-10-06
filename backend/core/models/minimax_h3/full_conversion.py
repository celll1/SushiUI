"""Bounded CPU expansion of the official scaled-FP8 DiT for Full training."""

import hashlib
import json
import math
from pathlib import Path
import struct

import torch
from safetensors import safe_open

from core.models.common.streaming_safetensors import DTYPES, tensor_bytes, write_stream


def plan_conversion(source, *, official_dir=None):
    from .loader import _guard_component_file, _synthesize_transformer_config, _rename_dit_key, _dit_target_dtype
    source = Path(source)
    header, metadata = _guard_component_file(str(source),label="H3 Full conversion")
    config = _synthesize_transformer_config(header,official_dir)
    curve = config.get("adaln_curve_grid") is not None
    specifications, converted = {}, {}
    with safe_open(str(source),framework="pt",device="cpu") as handle:
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(".comfy_quant"):
                from core.models.common.quantized_checkpoint_guard import decode_comfy_quant_marker
                marker = decode_comfy_quant_marker(handle.get_tensor(key))
                if (not isinstance(marker,dict) or marker.get("format") != "float8_e4m3fn"
                        or set(marker)-{"format","full_precision_matrix_mult"}
                        or ("full_precision_matrix_mult" in marker
                            and type(marker["full_precision_matrix_mult"]) is not bool)):
                    raise ValueError("H3 Full conversion only supports unrotated scaled float8_e4m3fn declarations")
                weight = header.get(key[:-len(".comfy_quant")]+".weight")
                if weight is None or weight["dtype"] != "F8_E4M3":
                    raise ValueError("H3 Full conversion found a marker without an FP8 weight")
                continue
            if key.endswith(".weight_scale"):
                weight = header.get(key[:-len(".weight_scale")]+".weight")
                if weight is None or weight["dtype"] != "F8_E4M3":
                    raise ValueError("H3 Full conversion found a scale without an FP8 weight")
                continue
            if key.endswith(".input_scale"):
                continue
            dtype = spec["dtype"]
            if dtype in ("F8_E5M2","I8","U8") or key.endswith(".pre_quant_scale"):
                raise ValueError("H3 Full conversion refuses unknown packed/quantized tensor semantics")
            if dtype == "F8_E4M3":
                scale_key = key[:-len(".weight")]+".weight_scale" if key.endswith(".weight") else ""
                scale = header.get(scale_key)
                if (len(spec["shape"]) != 2 or min(spec["shape"]) < 1 or scale is None
                        or scale["dtype"] != "F32" or math.prod(scale["shape"]) not in (1,spec["shape"][0])):
                    raise ValueError("H3 Full conversion requires matrix weights and scalar/per-row FP32 scales")
                target = _dit_target_dtype(_rename_dit_key(key),torch.bfloat16,curve)
                dtype = "F32" if target == torch.float32 else "BF16"
                converted[key] = scale_key
            specifications[key] = dict(dtype=dtype,shape=spec["shape"])
            tensor_bytes(specifications[key])
    if not converted:
        raise ValueError("H3 Full conversion source has no supported scaled-FP8 weights")
    signature = dict(bytes=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,
        header_sha256=hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest())
    record = dict(version=1,method="scaled_fp8_dequant",source=str(source.resolve()),source_signature=signature,
                  converted_matrices=len(converted),compute_dtype="bf16_with_native_fp32_heads")
    output_metadata = dict(metadata)
    output_metadata.update(model_type="minimax_h3",**{"modelspec.architecture":"minimax_h3"},
        **{"sushi.h3_dequantization":json.dumps(record,sort_keys=True,allow_nan=False)})
    return header,specifications,converted,output_metadata


def convert_checkpoint(source,output,*,work_bytes=64*1024*1024,progress=None,official_dir=None):
    source, output = Path(source),Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("H3 Full conversion cannot overwrite its source")
    if type(work_bytes) is not int or work_bytes < 1:
        raise ValueError("H3 conversion work byte budget must be positive")
    header,specifications,converted,metadata = plan_conversion(source,official_dir=official_dir)
    signature = (source.stat().st_size,source.stat().st_mtime_ns)
    with source.open("rb") as raw:
        base = 8+struct.unpack("<Q",raw.read(8))[0]
        with safe_open(str(source),framework="pt",device="cpu") as small:
            def chunks():
                for key in sorted(specifications):
                    spec = header[key]
                    shape = spec["shape"]
                    rows, width = (shape[0],math.prod(shape[1:])) if shape else (1,1)
                    if math.prod(shape) == 0:
                        continue
                    original_dtype = DTYPES[spec["dtype"]]
                    item_size = torch.empty((),dtype=original_dtype).element_size()
                    target_dtype = DTYPES[specifications[key]["dtype"]]
                    target_size = torch.empty((),dtype=target_dtype).element_size()
                    cost = width*(item_size+2*target_size+1)
                    if cost > work_bytes:
                        raise ValueError(f"H3 conversion budget cannot hold one row of {key}")
                    per_chunk = max(1,work_bytes//cost)
                    scale = None
                    if key in converted:
                        scale = small.get_tensor(converted[key]).flatten().to(target_dtype)
                        if not torch.isfinite(scale).all() or (scale <= 0).any():
                            raise ValueError("H3 conversion requires finite positive weight scales")
                    for start in range(0,rows,per_chunk):
                        count = min(per_chunk,rows-start)
                        raw.seek(base+spec["data_offsets"][0]+start*width*item_size)
                        buffer = bytearray(count*width*item_size)
                        if raw.readinto(buffer) != len(buffer):
                            raise OSError("H3 conversion source ended before its declared tensor bytes")
                        tensor = torch.frombuffer(buffer,dtype=original_dtype).reshape(count,width)
                        if scale is not None:
                            tensor = tensor.to(target_dtype)
                            tensor.mul_(scale if scale.numel() == 1 else scale[start:start+count,None])
                        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                            raise ValueError(f"H3 conversion produced non-finite tensor {key}")
                        yield key,tensor
                if (source.stat().st_size,source.stat().st_mtime_ns) != signature:
                    raise ValueError("H3 conversion source changed while reading")
            return write_stream(output,specifications,metadata,chunks(),progress=progress)
