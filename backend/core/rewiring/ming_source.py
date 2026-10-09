"""Bounded, deterministic identity for a native Ming transformer and its index."""
import hashlib
import struct

from .fitting import InitializationCancelled


def transformer_files(directory):
    from core.models.ming_image_01.artifact import _json
    indices = list(directory.glob("*.safetensors.index.json"))
    weights = sorted(directory.glob("*.safetensors"))
    if len(indices) > 1 or not weights:
        raise ValueError("Ming transformer requires native weights and at most one index")
    index = indices[0] if indices else None
    if index is None:
        if len(weights) != 1:
            raise ValueError("Multiple Ming transformer shards require an index")
    else:
        mapping = _json(index).get("weight_map")
        if not isinstance(mapping,dict) or not mapping or any(not isinstance(value,str) for value in mapping.values()):
            raise ValueError("Ming transformer index requires a nonempty string-valued weight_map")
        if set(mapping.values()) != {path.name for path in weights}:
            raise ValueError("Ming initialization requires exactly the indexed transformer shards")
    return weights,index


def source_identity(directory, *, cancelled=None, progress=None):
    weights,index = transformer_files(directory)
    paths = sorted(weights+([index] if index is not None else []))
    total = sum(path.stat().st_size for path in paths)
    completed = 0
    hashes = {}
    group = hashlib.sha256()
    for path in paths:
        digest = hashlib.sha256()
        before = path.stat()
        with path.open("rb") as stream:
            while chunk := stream.read(8*1024*1024):
                if cancelled is not None and cancelled():
                    raise InitializationCancelled("Ming source verification cancelled")
                digest.update(chunk)
                completed += len(chunk)
                if progress is not None:
                    progress(completed,total)
        after = path.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError("Ming native transformer changed during verification")
        hashes[path.name] = digest.hexdigest()
        name = path.name.encode("utf-8")
        group.update(struct.pack("<Q",len(name))+name+digest.digest())
    return (hashes[weights[0].name] if index is None else group.hexdigest()),hashes
