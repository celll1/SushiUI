"""Publish fitted hybrid modules without materializing the native tensor payload."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile

from safetensors import safe_open
import torch

from .fitting import InitializationCancelled, state_digest
from .provenance import checkpoint_metadata as initialization_metadata, validate_record
from .spec import checkpoint_metadata, read_checkpoint_metadata, _unique_object

_DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}


def validate_output_name(output_name):
    if (not isinstance(output_name, str) or not output_name.lower().endswith(".safetensors")
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', output_name)
            or output_name[-1:] in (" ", ".")
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", output_name)):
        raise ValueError("Initialized export requires a new safetensors basename")


def export_initialized_anima(model, source, output_name, *, cancelled=None):
    return export_initialized_transformer(model,source,output_name,architecture="anima",cancelled=cancelled)


def export_initialized_transformer(model, source, output_name, *, architecture, cancelled=None, output_directory=None):
    """Caller owns model exclusivity; source bytes and fitted modules are verified."""
    source = Path(source).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise ValueError("Initialized export requires a standalone native safetensors source")
    validate_output_name(output_name)
    directory = source.parent if output_directory is None else Path(output_directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Initialized export destination must be a directory")
    output = directory / output_name
    if output.exists() or output.is_symlink():
        raise FileExistsError("Initialized export never overwrites an existing file")
    if architecture == "anima":
        from core.models.anima.residual_unet import preflight_artifact, validate_topology_identity
        topology, config, alignment = preflight_artifact(source, include_alignment=True)
        config_key, input_key = "transformer_config", "x_embedder.proj.1.weight"
    elif architecture == "qwen_image_21":
        from core.models.qwen_image_21.residual_unet import validate_topology_identity
        with safe_open(str(source),framework="pt",device="cpu") as handle:
            source_metadata = handle.metadata() or {}
        if source_metadata.get("model_type") != architecture or source_metadata.get("component") != "transformer":
            raise ValueError("Qwen initialized export requires a native transformer component")
        config = json.loads(source_metadata.get("config") or "{}",object_pairs_hook=_unique_object)
        if not isinstance(config,dict) or not config:
            raise ValueError("Qwen initialization source requires transformer config metadata")
        topology = read_checkpoint_metadata(source_metadata)
        alignment = config.get("vae_alignment")
        config_key, input_key = "config", "img_in.weight"
    else:
        raise ValueError("Initialized standalone export supports Anima and Qwen-Image 2.1")
    if topology is not None or alignment is not None:
        raise ValueError("Initialization export requires a native source without U-Net or VAE swap")
    spec = validate_topology_identity(getattr(model, "_residual_unet_spec", None), config)
    record = validate_record(getattr(model, "_hybrid_initialization", None), spec)
    modules = getattr(model, "rewiring_modules", None)
    if modules is None or modules.declaration != spec:
        raise ValueError("Initialized export requires matching fitted hybrid modules")
    state = {name: value.detach().cpu().contiguous().clone()
             for name, value in modules.state_dict().items()}
    if state_digest(state) != record["initialized_weights_sha256"]:
        raise ValueError("Initialized module weights changed after fitting")
    if any(value.dtype not in _DTYPES or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Initialized modules require finite floating weights")

    def check_cancel():
        if cancelled is not None and cancelled():
            raise InitializationCancelled("Initialized model export cancelled")

    temporary = None
    try:
        check_cancel()
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated native safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= 16*1024*1024:
                raise ValueError("Invalid native safetensors header length")
            raw_header = stream.read(length)
            header = json.loads(raw_header, object_pairs_hook=_unique_object)
            with safe_open(str(source), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                keys = set(handle.keys())
            if read_checkpoint_metadata(metadata) is not None:
                raise ValueError("Native source topology changed during initialization export")
            tensor_prefix = "net." if architecture == "anima" and "net."+input_key in keys else ""
            if tensor_prefix+input_key not in keys:
                raise ValueError("Initialization source contains no native input projection")
            if any(key.startswith(tensor_prefix+"rewiring_modules.") for key in keys):
                raise ValueError("Initialization source already contains learned hybrid modules")
            metadata = deepcopy(metadata)
            metadata.update(checkpoint_metadata(spec))
            metadata.update(initialization_metadata(record, spec))
            metadata[config_key] = json.dumps(config)
            header["__metadata__"] = metadata
            body_length = before.st_size-8-length
            offset = body_length
            encoded_tensors = []
            for name, value in sorted(state.items()):
                payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
                key = tensor_prefix+"rewiring_modules."+name
                header[key] = dict(dtype=_DTYPES[value.dtype], shape=list(value.shape),
                                   data_offsets=[offset, offset+len(payload)])
                encoded_tensors.append(payload)
                offset += len(payload)
            encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            encoded += b" "*(-len(encoded)%8)
            if len(encoded) > 16*1024*1024:
                raise ValueError("Initialized safetensors header exceeds size limit")
            source_hash = hashlib.sha256(prefix+raw_header)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".sushi-hybrid-", suffix=".tmp",
                                             delete=False) as target:
                temporary = Path(target.name)
                target.write(struct.pack("<Q", len(encoded)))
                target.write(encoded)
                while chunk := stream.read(8*1024*1024):
                    check_cancel()
                    source_hash.update(chunk)
                    target.write(chunk)
                for payload in encoded_tensors:
                    check_cancel()
                    target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            after = os.fstat(stream.fileno())
            identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity(before) != identity(after) or identity(before) != identity(source.stat()):
                raise ValueError("Native source changed during initialization export")
            if source_hash.hexdigest() != record["source_sha256"]:
                raise ValueError("Native source checksum differs from initialization provenance")
        with safe_open(str(temporary), framework="pt", device="cpu") as handle:
            if read_checkpoint_metadata(handle.metadata() or {}) != spec:
                raise ValueError("Initialized export topology verification failed")
            for name, value in state.items():
                if not torch.equal(handle.get_tensor(tensor_prefix+"rewiring_modules."+name), value):
                    raise ValueError("Initialized export module verification failed")
        check_cancel()
        os.link(temporary, output)
        return {"model_path": str(output), "initialization": deepcopy(record)}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_initialized_qwen_package(model, source_manifest, destination, *, cancelled=None):
    """Publish a loadable split package; unchanged assets remain referenced at their source."""
    from core.models.qwen_image_21.artifact import load_manifest
    from core.models.qwen_image_21.residual_unet import validate_topology_identity
    source_manifest = Path(source_manifest).resolve(strict=True)
    if source_manifest.is_dir():
        source_manifest = source_manifest/"manifest.json"
    manifest = load_manifest(str(source_manifest))
    spec = validate_topology_identity(getattr(model,"_residual_unet_spec",None),manifest.transformer_config)
    validate_record(getattr(model,"_hybrid_initialization",None),spec)
    if manifest.transformer_config.get("vae_alignment") is not None:
        raise ValueError("Initialization package requires a native source codec")
    destination = Path(destination).absolute()
    validate_output_name(destination.name+".safetensors")
    if destination.name in (".","..") or destination.name.endswith((" ",".")):
        raise ValueError("Initialized package requires a new directory basename")
    if os.path.lexists(destination):
        raise FileExistsError("Initialized package never overwrites an existing path")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    references = {key:getattr(manifest,key) for key in
                  ("text_encoder","vae","processor","text_encoder_config","scheduler")}
    for key,path in references.items():
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"Qwen source asset is missing: {key}")
    with source_manifest.open(encoding="utf-8") as stream:
        raw = json.load(stream,object_pairs_hook=_unique_object)
    if (raw.get("transformer_config") != manifest.transformer_config
            or (raw.get("vae_config") or {}) != manifest.vae_config):
        raise ValueError("Qwen source manifest changed during initialization export")

    def check_cancel():
        if cancelled is not None and cancelled():
            raise InitializationCancelled("Initialized package export cancelled")

    with tempfile.TemporaryDirectory(dir=destination.parent,prefix=".sushi-qwen-init-") as temporary:
        staging = Path(temporary)/"package"
        staging.mkdir()
        check_cancel()
        result = export_initialized_transformer(model,manifest.transformer,"transformer.safetensors",
            architecture="qwen_image_21",cancelled=cancelled,output_directory=staging)
        raw["components"] = {**references,"transformer":"transformer.safetensors"}
        raw["transformer_config"] = deepcopy(manifest.transformer_config)
        with (staging/"manifest.json").open("w",encoding="utf-8") as stream:
            json.dump(raw,stream,indent=2,allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        prepared = load_manifest(str(staging))
        if Path(prepared.transformer) != staging/"transformer.safetensors":
            raise ValueError("Initialized package transformer reference verification failed")
        from .inspect import inspect_model
        inspected = inspect_model(staging)
        if inspected["topology"] != spec or inspected["initialization"] != result["initialization"]:
            raise ValueError("Initialized package declarations differ from fitted modules")
        check_cancel()
        if os.path.lexists(destination):
            raise FileExistsError("Initialized package destination was created during export")
        os.rename(staging,destination)
    return {"model_path":str(destination),"initialization":deepcopy(result["initialization"])}
