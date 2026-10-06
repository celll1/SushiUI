"""Atomic safetensors output without retaining a full state dictionary."""

import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import torch

DTYPES = {"BOOL":torch.bool,"U8":torch.uint8,"I8":torch.int8,"I16":torch.int16,
          "I32":torch.int32,"I64":torch.int64,"F16":torch.float16,"BF16":torch.bfloat16,
          "F32":torch.float32,"F64":torch.float64,"F8_E4M3":torch.float8_e4m3fn,
          "F8_E5M2":torch.float8_e5m2}


def tensor_bytes(spec):
    shape, dtype = spec.get("shape"), spec.get("dtype")
    if (dtype not in DTYPES or not isinstance(shape,(list,tuple))
            or any(type(n) is not int or n < 0 for n in shape)):
        raise ValueError("Invalid streaming tensor geometry/dtype")
    return math.prod(shape)*torch.empty((),dtype=DTYPES[dtype]).element_size()


def write_stream(path, specs, metadata, chunks, *, progress=None):
    """Chunks are ordered (key, CPU tensor); a key can span several chunks."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Streaming output already exists: {path}")
    if (not specs or "__metadata__" in specs or any(not isinstance(k,str) or not k for k in specs)
            or any(not isinstance(k,str) or not isinstance(v,str) for k,v in metadata.items())):
        raise ValueError("Invalid streaming tensor names/metadata")
    ordered = sorted(specs)
    header, total = {}, 0
    for key in ordered:
        size = tensor_bytes(specs[key])
        header[key] = dict(dtype=specs[key]["dtype"],shape=list(specs[key]["shape"]),data_offsets=[total,total+size])
        total += size
    header["__metadata__"] = metadata
    encoded = json.dumps(header,separators=(",",":"),allow_nan=False).encode("utf-8")
    encoded += b" "*((-len(encoded))%8)
    path.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path.parent).free < total+len(encoded)+8:
        raise OSError("Insufficient space for streaming safetensors output")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+".",suffix=".partial",delete=False) as out:
            temporary = Path(out.name)
            out.write(struct.pack("<Q",len(encoded)))
            out.write(encoded)
            index, offset, written = 0, 0, 0
            for key,tensor in chunks:
                while index < len(ordered) and tensor_bytes(specs[ordered[index]]) == offset:
                    index += 1
                    offset = 0
                if index >= len(ordered) or key != ordered[index]:
                    raise ValueError("Streaming tensor chunks are missing, repeated or out of order")
                if (not isinstance(tensor,torch.Tensor) or tensor.device.type != "cpu"
                        or tensor.dtype != DTYPES[specs[key]["dtype"]] or not tensor.is_contiguous()):
                    raise ValueError("Streaming chunks require contiguous CPU tensors with the declared dtype")
                raw = memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy())
                count = raw.nbytes
                if count < 1 or offset+count > tensor_bytes(specs[key]):
                    raise ValueError("Streaming tensor chunk byte length differs from its header")
                if out.write(raw) != count:
                    raise OSError("Incomplete streaming tensor write")
                offset += count
                written += count
                if progress:
                    progress(written,total)
            if written != total:
                raise ValueError("Streaming tensor output is incomplete")
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary,path)
        temporary.unlink()
        temporary = None
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
