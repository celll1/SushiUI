"""Verified, bounded-memory BF16 backing for the released FP32 Gemma component."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from filelock import FileLock
from safetensors import safe_open

from core.models.common.streaming_safetensors import DTYPES, tensor_bytes, write_stream


def _digest(path):
    before = path.stat()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Gemma backing input changed while hashing")
    return digest


def _local(directory, name):
    reference = Path(name)
    if reference.is_absolute() or reference.drive or reference.root or ".." in reference.parts:
        raise ValueError("Gemma cache input escapes its component directory")
    return directory / reference


def prepare_bf16_backing(directory, cache_root):
    """Return the source when no conversion is needed; originals are never edited."""
    directory, cache_root = Path(directory), Path(cache_root)
    config_path = directory / "config.json"
    if not config_path.is_file():
        return directory
    if json.loads(config_path.read_text(encoding="utf-8")).get("model_type") != "gemma3":
        return directory
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else None
    if index is not None:
        mapping = index["weight_map"]
        files = sorted(set(mapping.values()))
    elif (directory / "model.safetensors").is_file():
        mapping, files = None, ["model.safetensors"]
    else:
        return directory
    specifications, seen, convert = {}, set(), False
    for filename in files:
        path = _local(directory, filename)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            specs = {}
            for key in handle.keys():
                if key in seen or (mapping is not None and mapping.get(key) != filename):
                    raise ValueError(f"Invalid Gemma cache shard index: {key}")
                seen.add(key)
                view = handle.get_slice(key)
                dtype = view.get_dtype()
                if dtype not in DTYPES:
                    return directory
                floating = DTYPES[dtype].is_floating_point
                if floating and dtype not in ("F32", "F16", "BF16"):
                    return directory
                convert |= floating and dtype != "BF16"
                specs[key] = {"shape": view.get_shape(), "dtype": "BF16" if floating else dtype}
            specifications[filename] = specs
    if mapping is not None and seen != mapping.keys():
        raise ValueError("Gemma cache shard index has missing tensors")
    if not convert:
        return directory

    companion_names = ["config.json"] + (["generation_config.json"]
                                          if (directory / "generation_config.json").is_file() else [])
    inputs = files + companion_names + ([index_path.name] if index is not None else [])
    source_hashes = {name: _digest(_local(directory, name)) for name in inputs}
    identity = hashlib.sha256(json.dumps(
        {"version": 1, "dtype": "BF16", "sources": source_hashes}, sort_keys=True).encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / identity
    with FileLock(str(cache_root / (identity + ".lock"))):
        if target.exists():
            record = json.loads((target / "backing.json").read_text(encoding="utf-8"))
            if record.get("source_hashes") != source_hashes or not record.get("all_equal_to_source_cast"):
                raise ValueError("Gemma BF16 backing provenance mismatch")
            outputs = record.get("output_hashes", {})
            expected = set(files + companion_names + ([index_path.name] if index is not None else []))
            if set(outputs) != expected or any(_digest(_local(target, name)) != digest
                                             for name, digest in outputs.items()):
                raise ValueError("Gemma BF16 backing content changed; remove its cache directory to rebuild")
            return target

        required = sum(tensor_bytes(spec) for specs in specifications.values() for spec in specs.values())
        if shutil.disk_usage(cache_root).free < required + 16 * 1024**2:
            raise OSError(f"Gemma BF16 backing needs {required} bytes of free cache space")
        temporary = Path(tempfile.mkdtemp(prefix=identity + ".partial-", dir=cache_root))
        try:
            total_elements, total_tensors = 0, 0
            for filename, specs in specifications.items():
                source = _local(directory, filename)
                print(f"[LTX2Loader] Preparing verified BF16 text backing: {filename}", flush=True)

                def chunks():
                    with safe_open(str(source), framework="pt", device="cpu") as handle:
                        for key in sorted(specs):
                            tensor = handle.get_tensor(key).reshape(-1)
                            dtype = DTYPES[specs[key]["dtype"]]
                            count = max(1, (64 * 1024**2) // max(tensor.element_size(), dtype.itemsize))
                            for start in range(0, tensor.numel(), count):
                                yield key, tensor[start:start + count].to(dtype).contiguous()

                write_stream(temporary / filename, specs, {"format": "pt"}, chunks())
                with safe_open(str(source), framework="pt", device="cpu") as original, \
                        safe_open(str(temporary / filename), framework="pt", device="cpu") as converted:
                    for key in sorted(specs):
                        raw, value = original.get_tensor(key).reshape(-1), converted.get_tensor(key).reshape(-1)
                        count = max(1, (64 * 1024**2) // max(raw.element_size(), value.element_size()))
                        for start in range(0, raw.numel(), count):
                            if not torch.equal(value[start:start + count], raw[start:start + count].to(value.dtype)):
                                raise ValueError(f"Gemma BF16 backing cast verification failed: {key}")
                        total_tensors += 1
                        total_elements += value.numel()
                if _digest(source) != source_hashes[filename]:
                    raise ValueError("Gemma source changed during backing preparation")
            for name in companion_names:
                if _digest(directory / name) != source_hashes[name]:
                    raise ValueError("Gemma config changed during backing preparation")
                shutil.copyfile(directory / name, temporary / name)
            if index is not None:
                if _digest(index_path) != source_hashes[index_path.name]:
                    raise ValueError("Gemma index changed during backing preparation")
                index["metadata"] = dict(index.get("metadata") or {}, total_size=required)
                (temporary / index_path.name).write_text(json.dumps(index, indent=2), encoding="utf-8")
            outputs = files + companion_names + ([index_path.name] if index is not None else [])
            record = dict(version=1, source_hashes=source_hashes,
                          output_hashes={name: _digest(temporary / name) for name in outputs},
                          all_equal_to_source_cast=True, elements=total_elements, tensors=total_tensors,
                          tensor_bytes=required)
            (temporary / "backing.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            temporary.rename(target)
        finally:
            if temporary.exists():
                if temporary.resolve().parent != cache_root.resolve():
                    raise RuntimeError("Unsafe Gemma temporary cleanup path")
                shutil.rmtree(temporary)
    return target
