"""Architecture inspection for a topology editor; reads only config and headers."""
import json
from pathlib import Path

from safetensors import safe_open

from .artifacts import read_artifact_topology, read_directory_topology
from .spec import digest


def inspect_model(model_path):
    from core.models.qwen_image_21.artifact import load_manifest
    from core.models.qwen_image_21.residual_unet import (
        default_topology, native_config, validate_topology_identity)

    path = Path(model_path)
    if path.is_dir():
        if (path / "manifest.json").is_file():
            manifest = load_manifest(str(path))
            config = manifest.transformer_config
            topology = read_artifact_topology(manifest.transformer)
        else:
            config_path = path / "transformer" / "config.json"
            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            if not isinstance(config,dict):
                raise ValueError("Native model config must be an object")
            if config.get("_class_name") != "QwenImage21Transformer2DModel":
                raise ValueError("Residual inspection currently supports Qwen-Image 2.1 only")
            topology = read_directory_topology(path / "transformer")
    else:
        if not path.exists() and Path(str(path)+".index.json").exists():
            path = Path(str(path)+".index.json")
        if str(path).endswith(".safetensors.index.json"):
            with path.open(encoding="utf-8") as handle:
                index = json.load(handle)
                if not isinstance(index,dict):
                    raise ValueError("Checkpoint index must be an object")
                metadata = index.get("metadata") or {}
                if not isinstance(metadata,dict):
                    raise ValueError("Checkpoint metadata must be an object")
        else:
            with safe_open(str(path),framework="pt",device="cpu") as handle:
                metadata = handle.metadata() or {}
        if metadata.get("model_type") != "qwen_image_21" or metadata.get("component") not in {"transformer","training_bundle"}:
            raise ValueError("Residual inspection requires a Qwen-Image 2.1 model artifact")
        config = json.loads(metadata.get("config") or "{}")
        topology = read_artifact_topology(path)
    if not isinstance(config,dict):
        raise ValueError("Native model config must be an object")
    config = native_config(config)
    if topology is not None:
        topology = validate_topology_identity(topology,config)
    streams = [{"name":"main","block_count":config["num_layers"]}]
    return dict(architecture="qwen_image_21", streams=streams,
                base_config_sha256=digest(config), block_layout_sha256=digest(streams),
                topology=topology, topology_sha256=None if topology is None else digest(topology),
                default_spec=default_topology(config),
                latent_scale_factor=16, pixel_alignment=32)
