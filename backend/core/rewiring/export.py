"""Streaming metadata-only export for generation-only acoustic checkpoints."""
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

from safetensors import safe_open

from .inspect import inspect_model
from .spec import (META_VERSION, META_SPEC, META_HASH, checkpoint_metadata,
                   digest, read_checkpoint_metadata, validate_spec)


def export_acoustic_topology(model_path, output_name, spec):
    source = Path(model_path).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise ValueError("Residual acoustic export requires a standalone safetensors file")
    if (not isinstance(output_name, str) or not output_name.lower().endswith(".safetensors")
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', output_name)
            or output_name[-1:] in (" ", ".")
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", output_name)):
        raise ValueError("Export name must be a new safetensors basename without directory characters")
    output = source.parent / output_name
    if output.exists() or output.is_symlink():
        raise ValueError("Residual export never overwrites an existing file")
    info = inspect_model(source)
    graph = validate_spec(spec)
    if info["architecture"] not in ("yue2", "minimax_music3"):
        raise ValueError("Acoustic export supports complete YuE2 and flat MiniMax Music 3 only")
    if (graph["architecture"] != info["architecture"] or graph["streams"] != info["streams"]
            or graph["base_config_sha256"] != info["base_config_sha256"]
            or graph["block_layout_sha256"] != info["block_layout_sha256"]):
        raise ValueError("Export topology belongs to a different native model configuration")
    if any(region["down_factor"][1] != 1 for region in graph["regions"]):
        raise ValueError("Acoustic export shrinks the time axis only: [factor,1]")
    temporary = None
    try:
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated acoustic safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("Invalid acoustic safetensors header length")
            header = json.loads(stream.read(length))
            metadata = header.get("__metadata__", {})
            if read_checkpoint_metadata(metadata) != info["topology"]:
                raise ValueError("Source topology changed after inspection")
            if (info["architecture"] == "minimax_music3"
                    and "diffusion_transformer.transformer.project_in.weight" not in header):
                raise ValueError("Music 3 export requires a flat complete flow checkpoint")
            for key in (META_VERSION, META_SPEC, META_HASH):
                metadata.pop(key, None)
            metadata.update(checkpoint_metadata(graph))
            header["__metadata__"] = metadata
            encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            encoded += b" " * (-len(encoded) % 8)
            if len(encoded) > 16 * 1024 * 1024:
                raise ValueError("Exported acoustic safetensors header exceeds the native limit")
            with tempfile.NamedTemporaryFile(dir=source.parent, prefix=".sushi-unet-", suffix=".tmp",
                                             delete=False) as target:
                temporary = Path(target.name)
                target.write(struct.pack("<Q", len(encoded)))
                target.write(encoded)
                shutil.copyfileobj(stream, target, length=8 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            after = os.fstat(stream.fileno())
            named = source.stat()
            identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity(before) != identity(after) or identity(before) != identity(named):
                raise ValueError("Source checkpoint changed during export")
        if inspect_model(source) != info:
            raise ValueError("Source model configuration changed during export")
        with safe_open(str(temporary), framework="pt", device="cpu") as handle:
            if read_checkpoint_metadata(handle.metadata() or {}) != graph:
                raise ValueError("Export topology verification failed")
        # Atomic publication fails if another export claimed the same name.
        os.link(temporary, output)
        return {"model_path": str(output), "spec_sha256": digest(graph)}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
