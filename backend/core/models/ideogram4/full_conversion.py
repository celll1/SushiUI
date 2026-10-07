"""Bounded expansion of Ideogram's row-scaled FP8 effective weights."""

import hashlib
import json
import math
from pathlib import Path
import struct

import torch
from safetensors import safe_open

from core.models.common.streaming_safetensors import DTYPES, tensor_bytes, write_stream


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_conversion(source):
    source = Path(source)
    with source.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > 16 * 1024**2:
            raise ValueError("Ideogram conversion header exceeds the limit")
        header = json.loads(handle.read(length))
    specifications, converted = {}, {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            if any(marker in key for marker in (".comfy_quant", ".pre_quant_scale", ".quant_state", ".absmax")):
                raise ValueError("Ideogram conversion refuses packed or rotated quantization")
            if key.endswith(".weight_scale"):
                weight = header.get(key[:-len(".weight_scale")] + ".weight")
                if weight is None or weight["dtype"] != "F8_E4M3":
                    raise ValueError("Ideogram conversion found an orphan weight scale")
                continue
            dtype = spec["dtype"]
            if dtype in ("I8", "U8", "F8_E5M2"):
                raise ValueError("Ideogram conversion refuses unsupported quantized tensors")
            if dtype == "F8_E4M3":
                scale_key = key[:-len(".weight")] + ".weight_scale" if key.endswith(".weight") else ""
                scale = header.get(scale_key)
                if (len(spec["shape"]) != 2 or min(spec["shape"]) < 1 or scale is None
                        or scale["dtype"] != "F32" or scale["shape"] != [spec["shape"][0]]):
                    raise ValueError("Ideogram conversion requires FP8 matrix weights with per-row FP32 scales")
                converted[key] = scale_key
                dtype = "BF16"
            specifications[key] = dict(dtype=dtype, shape=spec["shape"])
            tensor_bytes(specifications[key])
    if not converted:
        raise ValueError("Ideogram conversion found no row-scaled FP8 weights")
    record = dict(version=1, method="row_scaled_fp8_effective_bf16", source=str(source.resolve()),
                  source_sha256=_digest(source), converted_matrices=len(converted),
                  compute_dtype="bf16", recovered_unquantized_original=False)
    metadata["sushi.ideogram_fp8_expansion"] = json.dumps(record, sort_keys=True)
    return header, specifications, converted, metadata


def convert_checkpoint(source, output, *, work_bytes=64 * 1024**2, progress=None):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Ideogram conversion cannot overwrite its source")
    if output.exists():
        raise FileExistsError(output)
    if type(work_bytes) is not int or work_bytes < 1:
        raise ValueError("Ideogram conversion work budget must be positive")
    signature = (source.stat().st_size, source.stat().st_mtime_ns)
    header, specs, converted, metadata = plan_conversion(source)
    record = json.loads(metadata["sushi.ideogram_fp8_expansion"])
    with source.open("rb") as raw, safe_open(str(source), framework="pt", device="cpu") as handle:
        base = 8 + struct.unpack("<Q", raw.read(8))[0]

        def chunks():
            for key in sorted(specs):
                spec = header[key]
                shape = spec["shape"]
                if math.prod(shape) == 0:
                    continue
                rows, width = (shape[0], math.prod(shape[1:])) if shape else (1, 1)
                original_dtype, target_dtype = DTYPES[spec["dtype"]], DTYPES[specs[key]["dtype"]]
                item_size = torch.empty((), dtype=original_dtype).element_size()
                target_size = torch.empty((), dtype=target_dtype).element_size()
                cost = width * (item_size + 2 * target_size + 1) + (7 if key in converted else 0)
                if cost > work_bytes:
                    raise ValueError(f"Ideogram work budget cannot hold one row of {key}")
                per_chunk = max(1, work_bytes // cost)
                scale = None
                if key in converted:
                    scale = handle.get_slice(converted[key])
                for start in range(0, rows, per_chunk):
                    count = min(per_chunk, rows - start)
                    raw.seek(base + spec["data_offsets"][0] + start * width * item_size)
                    buffer = bytearray(count * width * item_size)
                    if raw.readinto(buffer) != len(buffer):
                        raise OSError("Ideogram conversion source ended prematurely")
                    tensor = torch.frombuffer(buffer, dtype=original_dtype).reshape(count, width)
                    if scale is not None:
                        rows_scale = scale[start:start + count]
                        if not torch.isfinite(rows_scale).all() or (rows_scale <= 0).any():
                            raise ValueError("Ideogram conversion requires finite positive scales")
                        tensor = tensor.to(target_dtype) * rows_scale.to(target_dtype)[:, None]
                    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                        raise ValueError(f"Ideogram conversion produced nonfinite tensor {key}")
                    yield key, tensor
            if ((source.stat().st_size, source.stat().st_mtime_ns) != signature
                    or _digest(source) != record["source_sha256"]):
                raise ValueError("Ideogram conversion source changed while reading")

        return write_stream(output, specs, metadata, chunks(), progress=progress)
